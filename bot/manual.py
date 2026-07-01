"""
bot/manual.py — Manual trading mode: bot signals, you place the bet.

Run with: python3 -m bot.manual
"""

from __future__ import annotations

import asyncio
import datetime
import subprocess
import time
from typing import Dict, Optional

from .brain import StrategyBrain
from .config import KEY_PATH, SERIES_SPEC, MARKET_DISCOVERY_INTERVAL_SEC
from .feeds.binance_ws import BinanceFeed
from .recal import fit_all_series
from .run_live import (
    _load_key, _fetch_open_markets, _fetch_market_prices,
    _fetch_balance, LiveRunner,
)

BALANCE_REFRESH_SEC  = 60    # re-fetch Kalshi balance every 60s
SIGNAL_COOLDOWN_SEC  = 300   # 5 min between BET signals on same ticker
PULSE_INTERVAL_SEC   = 4     # live pulse print every 4s


# ── macOS notification ──────────────────────────────────────────────────────

def _notify(title: str, body: str, urgent: bool = False):
    try:
        sound = "Glass" if urgent else "Tink"
        script = (
            f'display notification "{body}" '
            f'with title "{title}" '
            f'sound name "{sound}"'
        )
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=3)
    except Exception:
        pass


# ── Helpers ─────────────────────────────────────────────────────────────────

def _asset_emoji(series: str) -> str:
    if "BTC" in series: return "₿"
    if "ETH" in series: return "Ξ"
    if "SOL" in series: return "◎"
    return "○"

def _confidence_bar(edge: float) -> tuple[str, str]:
    if edge >= 0.30:   return "VERY STRONG", "████████████"
    elif edge >= 0.20: return "STRONG",      "█████████░░░"
    elif edge >= 0.12: return "MODERATE",    "██████░░░░░░"
    else:              return "WEAK",        "███░░░░░░░░░"

def _tier_cap(edge: float) -> float:
    if edge >= 0.30:   return 0.25
    elif edge >= 0.20: return 0.18
    elif edge >= 0.12: return 0.10
    else:              return 0.05

def _contracts_for_balance(balance: float, tier_cap: float, entry_price: float) -> int:
    return max(1, int(balance * tier_cap / entry_price))

def _arrow(current: float, prev: float) -> str:
    if current > prev + 0.005: return "↑"
    if current < prev - 0.005: return "↓"
    return "→"


# ── Live pulse ───────────────────────────────────────────────────────────────

def _print_pulse(
    now_str: str,
    ticker_states: list[dict],  # [{series, asset, price, strike, gbm, gbm_prev, t_rem, status, in_pos}]
):
    """Print a compact live market pulse — one line per market."""
    print(f"\n{'━'*68}")
    print(f"  LIVE  {now_str}")
    print(f"{'━'*68}")
    for s in ticker_states:
        asset    = s["asset"]
        series   = s["series"].replace("KXBTC15M","BTC").replace("KXETH15M","ETH").replace("KXSOL15M","SOL")
        price    = s["price"]
        strike   = s["strike"]
        gbm      = s["gbm"]
        gbm_prev = s["gbm_prev"]
        t_rem    = s["t_rem"]
        status   = s["status"]
        in_pos   = s["in_pos"]

        if strike > 0 and price > 0:
            diff_pct = (price - strike) / strike * 100
            diff_str = f"{diff_pct:+.3f}%"
            pos_str  = "ABOVE" if price > strike else "BELOW"
        else:
            diff_str = "  n/a "
            pos_str  = "  "

        arr      = _arrow(gbm, gbm_prev)
        mins     = int(t_rem / 60)
        secs     = int(t_rem % 60)
        time_str = f"{mins}m{secs:02d}s"
        pos_flag = " [IN]" if in_pos else ""

        print(
            f"  {asset} {series:<4}  "
            f"${price:>9,.2f}  {diff_str} {pos_str} strike  "
            f"GBM {gbm:.2f}{arr}  "
            f"{time_str}  → {status}{pos_flag}"
        )
    print(f"{'━'*68}")


# ── Signal printers ──────────────────────────────────────────────────────────

def _print_entry_signal(decision, close_ts: float, floor_strike: float, balance: float):
    side      = decision.side.upper()
    series    = decision.series
    asset     = _asset_emoji(series)
    secs_left = close_ts - time.time()
    mins_left = int(secs_left / 60)
    secs_rem  = int(secs_left % 60)
    close_str = datetime.datetime.fromtimestamp(close_ts).strftime("%H:%M:%S")
    now_str   = datetime.datetime.now().strftime("%H:%M:%S")

    edge          = abs(decision.p_hat - 0.5)
    conf_label, conf_bar = _confidence_bar(edge)
    tier          = _tier_cap(edge)
    contracts     = _contracts_for_balance(balance, tier, decision.entry_price)
    outlay        = contracts * decision.entry_price
    max_win       = contracts * (1.0 - decision.entry_price)
    blend_pct     = decision.p_hat * 100
    gbm_pct       = decision.gbm_prob * 100
    ap_pct        = (decision.ap - 0.5) * 100 if decision.ap else 0.0
    is_last_min   = "LAST_MIN" in decision.reason

    tag = "⚡ LAST-MINUTE BET" if is_last_min else "◆ BET SIGNAL"

    print()
    print("╔" + "═" * 60 + "╗")
    print(f"║  {asset}  {tag}  ·  {series}  ·  {now_str}".ljust(61) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Ticker   {decision.ticker}".ljust(61) + "║")
    print(f"║  Side     {'YES ▲  price must END above strike' if side == 'YES' else 'NO  ▼  price must END below strike'}".ljust(61) + "║")
    print(f"║  Strike   ${floor_strike:,.4f}".ljust(61) + "║")
    print(f"║  Closes   {close_str} local  ({mins_left}m {secs_rem}s left)".ljust(61) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  SIGNALS".ljust(61) + "║")
    if not is_last_min:
        print(f"║    AP signal  {decision.ap:.4f}  →  {ap_pct:+.1f}% toward {side}".ljust(61) + "║")
    print(f"║    GBM        {decision.gbm_prob:.4f}  →  {gbm_pct:.1f}% chance YES".ljust(61) + "║")
    print(f"║    Blend      {decision.p_hat:.4f}  →  {blend_pct:.1f}% chance YES".ljust(61) + "║")
    print(f"║    Confidence {conf_label}  {conf_bar}".ljust(61) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  YOUR BET  (${balance:.2f} balance · {tier*100:.0f}% tier)".ljust(61) + "║")
    print(f"║    Buy {side:<3}   {contracts} contract(s) @ ${decision.entry_price:.3f}  =  ${outlay:.2f} outlay".ljust(61) + "║")
    print(f"║    Win        +${max_win:.2f}  if price settles {side}".ljust(61) + "║")
    print(f"║    Lose       -${outlay:.2f}  if price settles {'NO' if side == 'YES' else 'YES'}".ljust(61) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  HOW TO BET".ljust(61) + "║")
    print(f"║    1. kalshi.com  →  search {decision.ticker}".ljust(61) + "║")
    print(f"║    2. Click {side}  →  {contracts} contract(s)  →  confirm".ljust(61) + "║")
    print("╚" + "═" * 60 + "╝")
    print()

    _notify(
        title=f"{asset} {'⚡' if is_last_min else '◆'} {side} {series}",
        body=f"{side} @ ${decision.entry_price:.2f} · {contracts} contracts · ${outlay:.2f}\n{conf_label} · {mins_left}m {secs_rem}s left",
        urgent=True,
    )


def _print_exit_signal(ticker: str, side: str, reason: str,
                       exit_price: float, entry_price: float, contracts: int):
    series   = ticker.split("-")[0] if "-" in ticker else ticker
    asset    = _asset_emoji(series)
    now_str  = datetime.datetime.now().strftime("%H:%M:%S")

    if side == "yes":
        pnl_now  = (exit_price - entry_price) * contracts
        pnl_zero = -entry_price * contracts
    else:
        pnl_now  = (entry_price - exit_price) * contracts
        pnl_zero = -entry_price * contracts

    pnl_str  = f"{'+'if pnl_now >= 0 else ''}{pnl_now:.2f}"
    save_str = f"{abs(pnl_now - pnl_zero):.2f}"

    print()
    print("╔" + "═" * 60 + "╗")
    print(f"║  {asset}  ✕ EXIT NOW  ·  {series}  ·  {now_str}".ljust(61) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Ticker    {ticker}".ljust(61) + "║")
    print(f"║  Your bet  {side.upper()}  ({contracts} contracts @ ${entry_price:.3f})".ljust(61) + "║")
    print(f"║  Sell at   ~${exit_price:.3f}  (current {side.upper()} bid)".ljust(61) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  WHY EXIT".ljust(61) + "║")
    print(f"║    {reason}".ljust(61) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  P&L".ljust(61) + "║")
    print(f"║    Sell now   ${pnl_str}  (partial exit)".ljust(61) + "║")
    print(f"║    Hold + lose  ${pnl_zero:.2f}  (if signal correct)".ljust(61) + "║")
    print(f"║    Selling saves  ~${save_str}  vs holding to zero".ljust(61) + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  HOW TO EXIT".ljust(61) + "║")
    print(f"║    kalshi.com  →  Portfolio  →  {ticker}".ljust(61) + "║")
    print(f"║    Sell  →  {contracts} contract(s)  →  confirm".ljust(61) + "║")
    print("╚" + "═" * 60 + "╝")
    print()

    _notify(
        title=f"{asset} ✕ EXIT {side.upper()} {series}",
        body=f"Sell @ ~${exit_price:.2f} · P&L {pnl_str} · {reason[:50]}",
        urgent=True,
    )


# ── Manual trader ─────────────────────────────────────────────────────────────

class ManualTrader(LiveRunner):

    def __init__(self, private_key, brain: StrategyBrain, balance: float):
        super().__init__(private_key=private_key, brain=brain, jsonl_path="/dev/null")
        self._binance          = BinanceFeed()
        self._floor_strikes    : Dict[str, float] = {}
        self._close_ts         : Dict[str, float] = {}
        self._manual_positions : Dict[str, dict]  = {}
        self._last_signal_ts   : Dict[str, float] = {}   # cooldown per ticker
        self._balance          = balance
        self._last_balance_refresh = 0.0

        # For live pulse: track previous GBM per ticker
        self._gbm_prev_pulse   : Dict[str, float] = {}

    async def _refresh_balance(self):
        now = time.time()
        if now - self._last_balance_refresh > BALANCE_REFRESH_SEC:
            loop = asyncio.get_running_loop()
            bal  = await loop.run_in_executor(None, _fetch_balance, self._key)
            if bal > 0:
                self._balance = bal
            self._last_balance_refresh = now

    def _status_for(self, ticker: str, gbm: float, t_rem: float) -> str:
        """One-word action recommendation for the live pulse."""
        w = self._brain.window_state(ticker)
        pos = self._manual_positions.get(ticker)

        if pos:
            side = pos["side"]
            if side == "yes" and gbm < 0.40:
                return "EXIT — GBM reversed"
            if side == "no"  and gbm > 0.60:
                return "EXIT — GBM reversed"
            if t_rem <= 90:
                return "EXIT — pre-expiry"
            dist = abs(gbm - 0.5)
            return f"HOLD {side.upper()}  GBM {gbm:.2f}"

        if t_rem <= 0:
            return "CLOSED"
        if t_rem <= 90 and gbm >= 0.85:
            return f"⚡ LAST-MIN BET {'YES' if gbm > 0.5 else 'NO'}"
        if t_rem <= 90:
            return "WATCH (late)"
        if w and w.decided:
            return "DECIDED (cooldown)"

        elapsed = time.time() - (w.open_ts if w else time.time())
        if elapsed < 30:
            return "HARD BLOCK"

        if gbm >= 0.75:
            return f"STRONG → BET {'YES' if gbm > 0.5 else 'NO'} soon"
        if gbm > 0.60:
            gbm_min = w.gbm_min if w else 1.0
            if gbm_min <= 0.45:
                return f"BOUNCE ↑ → BET YES"
            return f"WAIT for dip (min {gbm_min:.2f})"
        if gbm < 0.40:
            gbm_max = w.gbm_max if w else 0.0
            if gbm_max >= 0.55:
                return f"BOUNCE ↓ → BET NO"
            return f"WAIT for spike (max {gbm_max:.2f})"
        return "WATCHING"

    async def _discover_loop(self):
        while True:
            now = time.time()
            for series in SERIES_SPEC:
                markets = _fetch_open_markets(self._key, series)
                for m in markets:
                    ticker = m.get("ticker", "")
                    if not ticker or ticker in self._tracked:
                        continue
                    open_time_str  = m.get("open_time", "")
                    close_time_str = m.get("close_time", "")
                    try:
                        open_ts  = datetime.datetime.fromisoformat(
                            open_time_str.replace("Z", "+00:00")).timestamp()
                        close_ts = datetime.datetime.fromisoformat(
                            close_time_str.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        continue
                    if now > open_ts + 720:
                        continue

                    floor_strike = float(m.get("floor_strike", 0.0))
                    self._tracked.add(ticker)
                    self._close_ts[ticker]      = close_ts
                    self._floor_strikes[ticker] = floor_strike
                    self._brain.register_window(ticker, series, open_ts,
                                                floor_strike=floor_strike)
                    await self._sub_queue.put(ticker)

                    asset     = _asset_emoji(series)
                    close_str = datetime.datetime.fromtimestamp(close_ts).strftime("%H:%M:%S")
                    mins_left = int((close_ts - now) / 60)
                    print(f"  {asset}  New window: {ticker}  "
                          f"close={close_str}  strike=${floor_strike:,.4f}  ({mins_left}m left)")

            await asyncio.sleep(MARKET_DISCOVERY_INTERVAL_SEC)

    async def _gbm_loop(self):
        """Update GBM every 4s and print the live pulse."""
        while True:
            await asyncio.sleep(PULSE_INTERVAL_SEC)
            now = time.time()

            ticker_states = []
            for ticker in sorted(self._tracked):
                close_ts = self._close_ts.get(ticker)
                if close_ts is None:
                    continue
                t_rem  = close_ts - now
                series = self._brain._series_from_ticker(ticker)
                s_now  = self._binance.get_price(series)
                sigma  = self._binance.get_sigma_per_second(series)

                if s_now > 0 and sigma > 0 and t_rem > 0:
                    self._brain.update_gbm(ticker, s_now, sigma, t_rem)

                w   = self._brain.window_state(ticker)
                gbm = w.gbm_prob if w else 0.5
                gbm_prev = self._gbm_prev_pulse.get(ticker, 0.5)
                self._gbm_prev_pulse[ticker] = gbm

                ticker_states.append({
                    "series":   series,
                    "asset":    _asset_emoji(series),
                    "price":    s_now,
                    "strike":   self._floor_strikes.get(ticker, 0.0),
                    "gbm":      gbm,
                    "gbm_prev": gbm_prev,
                    "t_rem":    max(0, t_rem),
                    "status":   self._status_for(ticker, gbm, t_rem),
                    "in_pos":   ticker in self._manual_positions,
                })

            if ticker_states:
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                _print_pulse(now_str, ticker_states)

            await self._check_exit_signals()

    async def _check_exit_signals(self):
        now = time.time()
        for ticker, pos in list(self._manual_positions.items()):
            if now - pos["entry_ts"] < 90:
                continue
            w        = self._brain.window_state(ticker)
            gbm      = w.gbm_prob if w else 0.5
            close_ts = self._close_ts.get(ticker, now + 999)
            t_rem    = close_ts - now

            exit_reason = None

            if pos["side"] == "yes" and gbm < 0.40:
                exit_reason = f"GBM reversed to {gbm:.2f} — price moving against YES"
            elif pos["side"] == "no" and gbm > 0.60:
                exit_reason = f"GBM reversed to {gbm:.2f} — price moving against NO"

            if exit_reason is None and t_rem <= 120:
                series       = self._brain._series_from_ticker(ticker)
                s_now        = self._binance.get_price(series)
                floor_strike = self._floor_strikes.get(ticker, 0.0)
                if s_now > 0 and floor_strike > 0:
                    dist_pct = abs(s_now - floor_strike) / floor_strike
                    if dist_pct < 0.001:
                        exit_reason = (
                            f"Pin risk — ${s_now:.4f} within {dist_pct*100:.3f}% "
                            f"of strike ${floor_strike:.4f}"
                        )

            if exit_reason is None and t_rem <= 90:
                exit_reason = f"Pre-expiry — {t_rem:.0f}s left, take the bid now"

            if exit_reason:
                loop   = asyncio.get_running_loop()
                prices = await loop.run_in_executor(
                    None, _fetch_market_prices, self._key, ticker
                )
                yes_ask = prices["yes_ask"] if prices else 0.5
                no_ask  = prices["no_ask"]  if prices else 0.5
                exit_price = (1.0 - no_ask) if pos["side"] == "yes" else (1.0 - yes_ask)
                _print_exit_signal(
                    ticker, pos["side"], exit_reason,
                    exit_price, pos["entry_price"],
                    pos.get("contracts", 1),
                )
                self._manual_positions.pop(ticker, None)

    async def _entry_loop(self):
        while True:
            await asyncio.sleep(30)
            await self._refresh_balance()
            now = time.time()

            for ticker in list(self._tracked):
                # Cooldown — don't re-signal same ticker within 5 min
                last_sig = self._last_signal_ts.get(ticker, 0)
                if now - last_sig < SIGNAL_COOLDOWN_SEC:
                    continue
                # Skip if already in a position on this ticker
                if ticker in self._manual_positions:
                    continue

                close_ts = self._close_ts.get(ticker, now + 999)
                t_rem    = close_ts - now
                if t_rem <= 0:
                    continue

                w   = self._brain.window_state(ticker)
                gbm = w.gbm_prob if w else 0.5

                # Skip fetch if GBM is completely flat (save API calls)
                if abs(gbm - 0.5) < 0.08 and t_rem > 90:
                    continue

                loop   = asyncio.get_running_loop()
                prices = await loop.run_in_executor(
                    None, _fetch_market_prices, self._key, ticker
                )
                if prices is None:
                    continue

                decision = self._brain.try_signal(
                    ticker, prices["yes_ask"], prices["no_ask"],
                    bankroll=self._balance,
                    t_remaining=t_rem,
                )

                if decision.action == "BET":
                    floor_strike = self._floor_strikes.get(ticker, 0.0)
                    edge         = abs(decision.p_hat - 0.5)
                    tier         = _tier_cap(edge)
                    contracts    = _contracts_for_balance(
                        self._balance, tier, decision.entry_price
                    )
                    _print_entry_signal(decision, close_ts, floor_strike, self._balance)
                    self._last_signal_ts[ticker] = now
                    self._manual_positions[ticker] = {
                        "side":        decision.side,
                        "entry_price": decision.entry_price,
                        "entry_ts":    now,
                        "contracts":   contracts,
                    }

    async def run(self):
        asyncio.create_task(self._binance.start())
        asyncio.create_task(self._gbm_loop())
        asyncio.create_task(self._entry_loop())
        await asyncio.gather(
            self._discover_loop(),
            self._ws_loop(),
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

async def _main():
    print()
    print("╔" + "═" * 50 + "╗")
    print("║        KALSHI MANUAL TRADING MODE               ║")
    print("║  Live pulse every 4s. Signals when ready.       ║")
    print("║  YOU place bets on Kalshi. Ctrl+C to stop.      ║")
    print("╚" + "═" * 50 + "╝")
    print()

    private_key = _load_key(KEY_PATH)

    print("  Fetching your Kalshi balance...")
    balance = _fetch_balance(private_key)
    print(f"  Balance: ${balance:.2f}")
    print()

    print("  Fitting OLS signal coefficients...")
    coefs = fit_all_series(verbose=False)
    for series, (b0, b1) in coefs.items():
        asset = _asset_emoji(series)
        print(f"    {asset}  {series}  b0={b0:.4f}  b1={b1:.4f}")
    print()
    print("  Live pulse starting in ~10s once markets are discovered...")
    print()

    brain  = StrategyBrain(coefs=coefs, initial_bankroll=balance)
    trader = ManualTrader(private_key=private_key, brain=brain, balance=balance)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(_main())
