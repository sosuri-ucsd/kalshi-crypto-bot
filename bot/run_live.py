"""
bot/run_live.py — Live integration: WebSocket → StrategyBrain → JSONL decisions.

This script:
  1. Loads recalibrated OLS coefficients (from recal.py)
  2. Opens a Kalshi WebSocket and subscribes to 'trade' channel for
     all active KXBTC15M / KXETH15M / KXSOL15M markets
  3. Feeds trade messages to StrategyBrain.on_trade()
  4. At open_ts + K for each window, fetches current yes_ask / no_ask
     via REST and calls brain.decide()
  5. Logs every TradeDecision to decisions.jsonl (one JSON object per line)
  6. Prints human-readable summaries to stdout

IMPORTANT — This script NEVER places orders.
It outputs structured TradeDecision records. Your Claude Code session
(or any other integration layer) reads decisions.jsonl and handles execution.

Usage:
    cd /Users/sourishsuri/kalshi
    python3 -m bot.run_live

Requirements:
    pip install websockets requests cryptography --break-system-packages

Environment:
    Reads KEY_PATH and KEY_ID from bot/config.py (same credentials as
    kalshi_live.py — no new credentials needed).
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set

# Third-party (install with pip if missing)
try:
    import websockets
    import requests
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install websockets requests cryptography --break-system-packages")
    sys.exit(1)

from .brain import StrategyBrain, TradeDecision
from .config import (
    KEY_PATH, KEY_ID, REST_BASE, WS_URL,
    SERIES_SPEC, DECISIONS_JSONL,
    MARKET_DISCOVERY_INTERVAL_SEC, REST_ORDERBOOK_INTERVAL_SEC,
)
from .recal import fit_all_series

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot.run_live")


# ---------------------------------------------------------------------------
# Auth helpers (identical pattern to kalshi_live.py)
# ---------------------------------------------------------------------------

def _load_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign(private_key, text: str) -> str:
    sig = private_key.sign(
        text.encode("utf-8"),
        asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def _signed_get(private_key, path: str, params=None):
    ts = str(int(time.time() * 1000))
    sig = _sign(private_key, ts + "GET" + path)
    headers = {
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }
    r = requests.get(REST_BASE + path, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _auth_headers_for_ws(private_key) -> dict:
    """Generate auth headers for WebSocket handshake."""
    ts = str(int(time.time() * 1000))
    path = "/trade-api/ws/v2"
    sig = _sign(private_key, ts + "GET" + path)
    return {
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

def _fetch_open_markets(private_key, series: str) -> list[dict]:
    """Return list of open market dicts for a series."""
    try:
        resp = _signed_get(
            private_key,
            "/trade-api/v2/markets",
            {"series_ticker": series, "status": "open", "limit": 50},
        )
        return resp.get("markets", [])
    except Exception as e:
        log.warning(f"Market discovery error for {series}: {e}")
        return []


def _fetch_market_prices(private_key, ticker: str) -> Optional[dict]:
    """
    Returns {'yes_ask': float, 'no_ask': float} in [0,1] dollars,
    or None if the market is unreachable.

    Kalshi API v2 now returns prices as dollar strings in fields named
    "yes_ask_dollars" / "no_ask_dollars" (e.g. "0.3800").
    Old API used integer-cent fields "yes_ask" / "no_ask" (e.g. 38).
    We try the new format first and fall back to the old one.
    """
    try:
        resp = _signed_get(private_key, f"/trade-api/v2/markets/{ticker}")
        m = resp.get("market", {})

        def _price(dollars_key: str, cents_key: str) -> float:
            val = m.get(dollars_key)
            if val is not None:
                return float(val)           # already in [0,1] dollars
            val = m.get(cents_key, 0)
            return int(val) / 100.0         # integer cents → dollars

        yes_ask = _price("yes_ask_dollars", "yes_ask")
        no_ask  = _price("no_ask_dollars",  "no_ask")
        return {"yes_ask": yes_ask, "no_ask": no_ask}
    except Exception as e:
        log.warning(f"Price fetch error for {ticker}: {e}")
        return None


def _fetch_balance(private_key) -> float:
    """Return available balance in dollars."""
    try:
        resp = _signed_get(private_key, "/trade-api/v2/portfolio/balance")
        # balance is in cents; subtract reserved (payout) to get available
        total_cents = resp.get("balance", 0)
        reserved_cents = resp.get("payout", 0)
        return (total_cents - reserved_cents) / 100.0
    except Exception as e:
        log.warning(f"Balance fetch error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# WebSocket trade price parsing
# ---------------------------------------------------------------------------

def _parse_trade_price(msg: dict) -> float:
    """
    Extract YES price in [0,1] dollars from a Kalshi trade WebSocket message.

    Kalshi has two price formats depending on contract type:
      Old (integer cents):  {"yes_price": 38}         → 0.38
      New (dollar string):  {"yes_price": "0.3800"}   → 0.38
      New (dollar float):   {"yes_price": 0.38}        → 0.38
      Possibly:             {"count_dollars": "0.38"}  or other keys

    We try the most common fields in order and return the first valid result.
    """
    # Try 'yes_price' — may be int cents, float, or string dollars
    raw = msg.get("yes_price")
    if raw is not None:
        try:
            val = float(raw)
            # If val > 1 it's in cents (old format); if <= 1 it's already dollars
            return val / 100.0 if val > 1.0 else val
        except (TypeError, ValueError):
            pass

    # Fallback: 'count_dollars' or 'price_dollars' (speculative new field names)
    for key in ("count_dollars", "yes_price_dollars", "price"):
        raw = msg.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    return 0.0   # unknown format — caller will see AP drift toward 0


# ---------------------------------------------------------------------------
# Decision logger
# ---------------------------------------------------------------------------

def _log_decision(decision: TradeDecision, jsonl_path: str) -> None:
    """Append one TradeDecision to decisions.jsonl."""
    d = dataclasses.asdict(decision)
    d["ts_human"] = datetime.datetime.utcfromtimestamp(decision.ts).isoformat() + "Z"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(d) + "\n")


def _print_decision(decision: TradeDecision) -> None:
    tag = "🟢 BET" if decision.action == "BET" else "⬜ PASS"
    ts_str = datetime.datetime.utcfromtimestamp(decision.ts).strftime("%H:%M:%S")
    if decision.action == "BET":
        print(
            f"{ts_str} {tag} {decision.ticker} "
            f"| side={decision.side.upper()} @ {decision.entry_price:.2f} "
            f"| p_hat={decision.p_hat:.3f} AP={decision.ap:.3f} "
            f"| n_trades={decision.n_trades} "
            f"| contracts={decision.contracts} fee=${decision.fee_dollars:.3f} "
            f"| kelly={decision.kelly_capped:.4f} "
            f"| b0={decision.b0:.4f} b1={decision.b1:.4f}"
        )
    else:
        print(
            f"{ts_str} {tag} {decision.ticker} | {decision.reason} "
            f"| n={decision.n_trades} p_hat={decision.p_hat:.3f}"
        )


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

class LiveRunner:
    """
    Manages WebSocket connection, window scheduling, and brain integration.
    """

    def __init__(self, private_key, brain: StrategyBrain, jsonl_path: str):
        self._key = private_key
        self._brain = brain
        self._jsonl = jsonl_path

        # ticker → scheduled asyncio.Task for decide() at open_ts + K
        self._decision_tasks: Dict[str, asyncio.Task] = {}

        # Currently tracked tickers
        self._tracked: Set[str] = set()

        # Bankroll cache (refreshed every discovery cycle)
        self._bankroll: float = 0.0

        # WebSocket message counter
        self._msg_count: int = 0

        # Queue for immediate WS subscription when new tickers are discovered.
        # _discover_loop puts new tickers here; _ws_sub_drain sends subscribe msgs.
        self._sub_queue: asyncio.Queue = asyncio.Queue()

    # ------------------------------------------------------------------ #
    # Market discovery loop                                               #
    # ------------------------------------------------------------------ #

    async def _discover_loop(self):
        """Periodically poll REST for open markets and register new windows."""
        while True:
            self._bankroll = _fetch_balance(self._key)
            log.info(f"Available balance: ${self._bankroll:.2f}")

            now = time.time()
            new_tickers = []

            for series in SERIES_SPEC:
                markets = _fetch_open_markets(self._key, series)
                for m in markets:
                    ticker = m.get("ticker", "")
                    if not ticker or ticker in self._tracked:
                        continue

                    # Parse open_ts from market open_time field
                    open_time_str = m.get("open_time", "")
                    try:
                        open_ts = datetime.datetime.fromisoformat(
                            open_time_str.replace("Z", "+00:00")
                        ).timestamp()
                    except Exception:
                        log.warning(f"Could not parse open_time for {ticker}: {open_time_str}")
                        continue

                    spec = SERIES_SPEC[series]
                    K = spec["K"]
                    decision_ts = open_ts + K

                    # Skip if the decision window has already passed
                    if decision_ts <= now:
                        log.debug(f"Skipping {ticker}: decision window already passed")
                        continue

                    self._tracked.add(ticker)
                    self._brain.register_window(ticker, series, open_ts)
                    new_tickers.append(ticker)

                    # Immediately subscribe WS to this ticker (don't wait for
                    # the 500-message re-subscribe cycle — that's the old bug)
                    await self._sub_queue.put(ticker)

                    # Schedule the decide() call
                    delay = decision_ts - now
                    task = asyncio.create_task(
                        self._scheduled_decide(ticker, delay)
                    )
                    self._decision_tasks[ticker] = task

                    log.info(
                        f"Registered {ticker} ({series}) "
                        f"K={K}s decide in {delay:.0f}s"
                    )

            if new_tickers:
                # Re-subscribe WebSocket to include new tickers
                # (handled by the WS loop which re-subscribes on new tickers)
                log.info(f"New tickers: {new_tickers}")

            await asyncio.sleep(MARKET_DISCOVERY_INTERVAL_SEC)

    # ------------------------------------------------------------------ #
    # Scheduled decide() callback                                         #
    # ------------------------------------------------------------------ #

    async def _scheduled_decide(self, ticker: str, delay_sec: float):
        """Wait `delay_sec` seconds then fire decide()."""
        await asyncio.sleep(max(0.0, delay_sec))

        # Fetch current prices
        prices = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_market_prices, self._key, ticker
        )
        if prices is None:
            log.warning(f"Could not fetch prices for {ticker} at decision time")
            prices = {"yes_ask": 0.0, "no_ask": 0.0}

        bankroll = self._bankroll

        decision = self._brain.decide(
            ticker=ticker,
            yes_ask=prices["yes_ask"],
            no_ask=prices["no_ask"],
            bankroll=bankroll,
        )

        _log_decision(decision, self._jsonl)
        _print_decision(decision)

        # Clean up
        self._decision_tasks.pop(ticker, None)

    # ------------------------------------------------------------------ #
    # WebSocket loop                                                       #
    # ------------------------------------------------------------------ #

    async def _ws_loop(self):
        """
        Connect to Kalshi WebSocket, subscribe to trade channel,
        and feed messages to brain.on_trade().

        Two concurrent sub-tasks run inside each connection:
          _ws_recv_loop  — receives trade messages, feeds brain.on_trade()
          _ws_sub_drain  — drains self._sub_queue and sends subscribe msgs
                           immediately whenever new tickers are discovered.

        Reconnects automatically on disconnect.
        """
        while True:
            headers = _auth_headers_for_ws(self._key)
            log.info(f"Connecting to WebSocket: {WS_URL}")

            try:
                async with websockets.connect(
                    WS_URL,
                    additional_headers=headers,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    log.info("WebSocket connected")

                    # Subscribe to any tickers already known at connect time
                    tickers = list(self._tracked)
                    if tickers:
                        await ws.send(json.dumps({
                            "id": 1,
                            "cmd": "subscribe",
                            "params": {"channels": ["trade"], "market_tickers": tickers},
                        }))
                        log.info(f"Subscribed to {len(tickers)} existing tickers")

                    # Run receive and subscription-drain concurrently.
                    # When the WS drops, recv raises ConnectionClosed which
                    # cancels the drain task and falls through to reconnect.
                    await asyncio.gather(
                        self._ws_recv_loop(ws),
                        self._ws_sub_drain(ws),
                    )

            except websockets.exceptions.ConnectionClosed as e:
                log.warning(f"WebSocket closed: {e}. Reconnecting in 5s...")
            except Exception as e:
                log.error(f"WebSocket error: {e}. Reconnecting in 5s...")

            await asyncio.sleep(5)

    async def _ws_recv_loop(self, ws):
        """Receive trade messages from WebSocket and feed brain."""
        async for raw_msg in ws:
            self._msg_count += 1
            try:
                data = json.loads(raw_msg)
                await self._handle_ws_message(data)
            except Exception as e:
                log.debug(f"WS parse error: {e}")

    async def _ws_sub_drain(self, ws):
        """
        Drain self._sub_queue and send subscribe messages immediately.
        Called alongside _ws_recv_loop inside each WS connection.
        New tickers discovered by _discover_loop land here within milliseconds.
        """
        while True:
            ticker = await self._sub_queue.get()
            try:
                await ws.send(json.dumps({
                    "id": int(time.time() * 1000),
                    "cmd": "subscribe",
                    "params": {"channels": ["trade"], "market_tickers": [ticker]},
                }))
                log.info(f"WS subscribed → {ticker}")
            except Exception:
                # WS is broken; put ticker back so reconnect picks it up
                await self._sub_queue.put(ticker)
                raise

    async def _handle_ws_message(self, data: dict):
        """
        Parse a Kalshi WebSocket message and feed trades to brain.

        Kalshi trade message format (channel='trade'):
          {
            "type": "trade",
            "msg": {
              "market_ticker": "KXBTC15M-26JUL031445-50000",
              "yes_price": 62,          ← cents [0, 100]
              "created_time": "2026-07-01T14:30:01.123456Z",
              ...
            }
          }
        """
        msg_type = data.get("type", "")

        if msg_type == "trade":
            msg = data.get("msg", {})
            ticker = msg.get("market_ticker", "")
            if not ticker or ticker not in self._tracked:
                return

            # --- DEBUG: print first trade message so we can see the field names ---
            if not getattr(self, "_printed_trade_sample", False):
                log.info(f"TRADE MSG SAMPLE (first received): {json.dumps(msg)}")
                self._printed_trade_sample = True

            # --- Price parsing: try multiple field names/formats ---
            # Old API: yes_price = integer cents (e.g. 38 means $0.38)
            # New API (fractional/tapered_deci_cent): may use different field names
            price_dollars = _parse_trade_price(msg)

            created_time_str = msg.get("created_time", "")
            try:
                trade_ts = datetime.datetime.fromisoformat(
                    created_time_str.replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                # New WS format uses ts_ms (milliseconds) or ts (seconds)
                if "ts_ms" in msg:
                    trade_ts = msg["ts_ms"] / 1000.0
                elif "ts" in msg:
                    trade_ts = float(msg["ts"])
                else:
                    trade_ts = time.time()

            self._brain.on_trade(
                ticker=ticker,
                price_dollars=price_dollars,
                trade_ts=trade_ts,
            )

        # Alternative format: wrapped in {"recv_ts": ..., "raw": {...}}
        # (same as ws_kalshi.py JSONL format — handle if replaying from file)
        elif "recv_ts" in data and "raw" in data:
            await self._handle_ws_message(data["raw"])

    # ------------------------------------------------------------------ #
    # Public: run both loops concurrently                                 #
    # ------------------------------------------------------------------ #

    async def run(self):
        """Start discovery loop and WebSocket loop concurrently."""
        await asyncio.gather(
            self._discover_loop(),
            self._ws_loop(),
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def _main():
    print("=" * 70)
    print("Kalshi Brain — Live Decision Engine")
    print("NO orders will be placed. Decisions logged to decisions.jsonl.")
    print("=" * 70)

    # Load credentials
    key_path = KEY_PATH
    log.info(f"Loading key from {key_path}")
    private_key = _load_key(key_path)

    # Initial balance
    bankroll = _fetch_balance(private_key)
    if bankroll <= 0:
        log.warning("Balance is $0 or unavailable — continuing in observation mode")

    log.info(f"Initial bankroll: ${bankroll:.2f}")

    # Recalibrate OLS coefficients
    log.info("Fitting OLS coefficients from historical data...")
    coefs = fit_all_series(verbose=True)
    for series, (b0, b1) in coefs.items():
        log.info(f"  {series}: b0={b0:.4f}, b1={b1:.4f}")

    # Create brain
    brain = StrategyBrain(coefs=coefs, initial_bankroll=bankroll)

    # Ensure decisions.jsonl exists
    Path(DECISIONS_JSONL).parent.mkdir(parents=True, exist_ok=True)

    # Schedule daily recalibration at 2am UTC
    async def _daily_recal():
        while True:
            now = datetime.datetime.utcnow()
            next_2am = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_2am <= now:
                next_2am += datetime.timedelta(days=1)
            wait_sec = (next_2am - now).total_seconds()
            log.info(f"Next recalibration in {wait_sec/3600:.1f}h (at 02:00 UTC)")
            await asyncio.sleep(wait_sec)
            log.info("Running daily OLS recalibration...")
            new_coefs = fit_all_series(verbose=True)
            brain.update_coefs(new_coefs)
            log.info("Recalibration complete")

    # Run everything
    runner = LiveRunner(
        private_key=private_key,
        brain=brain,
        jsonl_path=DECISIONS_JSONL,
    )

    await asyncio.gather(
        runner.run(),
        _daily_recal(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("Shutting down.")
