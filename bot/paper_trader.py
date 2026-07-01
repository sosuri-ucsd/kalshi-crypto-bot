"""
bot/paper_trader.py

Paper trading engine. Runs the exact same brain as run_live.py but
SIMULATES fills instead of placing real orders. Polls Kalshi REST for
settlement results and reports full risk metrics.

Also detects dual-side arb opportunities (yes_ask + no_ask < 0.97),
logged separately — risk-free guaranteed profit that real execution
can exploit.

Usage:
    cd /Users/sourishsuri/kalshi
    python3 -m bot.paper_trader                  # default $500 paper bankroll
    python3 -m bot.paper_trader --bankroll 20    # test with $20

Press Ctrl+C → prints full risk report + saves bot/paper_report.json
Running: hourly snapshot auto-prints to console.

Output files:
    bot/paper_trades.jsonl     open/settle/dual_arb events (append-only)
    bot/paper_report.json      latest metrics snapshot (overwritten)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .brain import StrategyBrain
from .feeds.binance_ws import BinanceFeed
from .config import (
    KEY_PATH, REPO_ROOT, SERIES_SPEC,
    MARKET_DISCOVERY_INTERVAL_SEC,
)
from .recal import fit_all_series
from .run_live import (
    _load_key, _signed_get, _auth_headers_for_ws,
    _fetch_open_markets, _fetch_market_prices, _fetch_balance,
    _print_decision, LiveRunner,
)
from .sizing import kalshi_fee

log = logging.getLogger("bot.paper_trader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PAPER_TRADES_JSONL = str(Path(REPO_ROOT) / "bot" / "paper_trades.jsonl")
PAPER_REPORT_JSON  = str(Path(REPO_ROOT) / "bot" / "paper_report.json")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DUAL_ARB_THRESHOLD   = 0.97   # yes_ask + no_ask below this → arb signal
SETTLE_POLL_DELAY    = 60     # seconds after close_time before first poll
SETTLE_POLL_RETRIES  = 8      # max retries
SETTLE_POLL_INTERVAL = 30     # seconds between retries
METRICS_PRINT_EVERY  = 3600   # print hourly snapshot

# Dynamic exit constants (from hamad + brandononchain synthesis)
EXIT_GBM_THRESHOLD   = 0.40   # exit YES if GBM < 0.40; exit NO if GBM > 0.60
MIN_HOLD_SECS        = 90     # don't exit in first 90s after entry (noise filter)
PRE_EXPIRY_SECS      = 90     # sell anything within 90s of settlement, take the bid

# Pin risk constants — price hovering at floor_strike near expiry
PIN_ZONE_PCT         = 0.0010  # 0.1% of floor_strike (e.g. $65 on BTC at $65k)
PIN_WINDOW_SECS      = 120     # start watching for pin in last 2 minutes
PIN_FLIP_KELLY_SCALE = 0.4     # counter-bet after pin exit = 40% of normal size


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class SimulatedPosition:
    """One simulated paper bet."""
    ticker:        str
    series:        str
    side:          str            # 'yes' or 'no'
    contracts:     int
    fill_price:    float          # dollars per contract (yes_ask or no_ask)
    total_cost:    float          # fill_price × contracts
    fill_fee:      float          # kalshi_fee(contracts, fill_price)
    total_outlay:  float          # total_cost + fill_fee
    decision_ts:   float          # unix ts of decision
    close_ts:      float          # unix ts of market settlement
    p_hat:         float
    ap:            float
    n_trades:      int
    kelly_capped:  float
    b0:            float
    b1:            float
    gbm:           float = 0.5    # GBM P(YES) at decision time (capped at 0.80)
    # filled after settlement:
    settled:       bool           = False
    win:           Optional[bool] = None
    pnl:           Optional[float] = None   # net dollars after fee
    return_pct:    Optional[float] = None   # pnl / total_outlay


@dataclasses.dataclass
class DualArbRecord:
    """Detected dual-side arb opportunity."""
    ticker:         str
    series:         str
    yes_ask:        float
    no_ask:         float
    combined:       float      # yes_ask + no_ask
    contracts:      int        # how many pairs fit in 5% cap
    total_outlay:   float
    total_fees:     float
    locked_profit:  float      # guaranteed net pnl if both sides bought
    ts:             float


# --------------------------------------------------------------------------- #
# Paper book — position ledger + metrics engine
# --------------------------------------------------------------------------- #

class PaperBook:
    def __init__(self, initial_bankroll: float):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.peak_bankroll    = initial_bankroll
        self.start_ts         = time.time()
        self.open:  Dict[str, SimulatedPosition] = {}
        self.done:  List[SimulatedPosition]      = []
        self.arbs:  List[DualArbRecord]          = []

    # --- position lifecycle ----------------------------------------------- #

    def add(self, pos: SimulatedPosition):
        self.open[pos.ticker] = pos
        self.current_bankroll -= pos.total_outlay
        print(
            f"  📝 PAPER FILL  {pos.ticker[:40]}  "
            f"{pos.side.upper()} ×{pos.contracts} @ ${pos.fill_price:.3f}  "
            f"outlay=${pos.total_outlay:.3f}  "
            f"paper_bankroll=${self.current_bankroll:.2f}"
        )

    def settle(self, ticker: str, result: str) -> Optional[SimulatedPosition]:
        pos = self.open.pop(ticker, None)
        if pos is None:
            return None
        pos.win = (result == pos.side)
        if pos.win:
            pos.pnl = pos.contracts * 1.00 - pos.total_outlay
        else:
            pos.pnl = -pos.total_outlay
        pos.return_pct = pos.pnl / pos.total_outlay if pos.total_outlay > 0 else 0.0
        pos.settled    = True
        self.done.append(pos)
        self.current_bankroll += pos.total_outlay + pos.pnl   # restore then net
        self.peak_bankroll = max(self.peak_bankroll, self.current_bankroll)
        icon = "✅" if pos.win else "❌"
        print(
            f"  {icon} SETTLED    {ticker[:40]}  result={result.upper()}  "
            f"side={pos.side.upper()}  "
            f"pnl=${pos.pnl:+.3f} ({pos.return_pct*100:+.1f}%)  "
            f"paper_bankroll=${self.current_bankroll:.2f}"
        )
        return pos

    def add_arb(self, rec: DualArbRecord):
        self.arbs.append(rec)
        print(
            f"  ⚡ DUAL ARB    {rec.ticker[:40]}  "
            f"yes={rec.yes_ask:.3f} no={rec.no_ask:.3f}  "
            f"combined={rec.combined:.3f}  "
            f"guaranteed_pnl=${rec.locked_profit:+.4f} (×{rec.contracts} pairs)"
        )

    # --- metrics ------------------------------------------------------------ #

    def metrics(self) -> dict:
        n = len(self.done)
        pending = len(self.open)

        if n == 0:
            return {
                "bets_settled": 0,
                "bets_pending": pending,
                "message": "No settled bets yet — waiting for first settlement.",
            }

        wins     = [p for p in self.done if p.win]
        returns  = [p.return_pct for p in self.done]
        pnls     = [p.pnl        for p in self.done]

        mean_r = sum(returns) / n
        var_r  = sum((r - mean_r)**2 for r in returns) / max(n - 1, 1)
        std_r  = math.sqrt(var_r)
        sharpe = (mean_r / std_r) if std_r > 1e-9 else float("nan")

        gross_p = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p < 0))
        pf = (gross_p / gross_l) if gross_l > 1e-9 else float("inf")

        total_pnl  = sum(pnls)
        total_fees = sum(p.fill_fee for p in self.done)

        # Max drawdown in cumulative P&L
        cum_pnl, peak_pnl, max_dd = 0.0, 0.0, 0.0
        for p in sorted(self.done, key=lambda x: x.decision_ts):
            cum_pnl  += p.pnl
            peak_pnl  = max(peak_pnl, cum_pnl)
            max_dd    = max(max_dd, peak_pnl - cum_pnl)

        elapsed_days = max((time.time() - self.start_ts) / 86400, 1 / 24)

        # Per-series breakdown
        by_series: Dict[str, dict] = {}
        for p in self.done:
            s = p.series
            if s not in by_series:
                by_series[s] = {"n": 0, "wins": 0, "pnl": 0.0}
            by_series[s]["n"]    += 1
            by_series[s]["wins"] += int(p.win or False)
            by_series[s]["pnl"]  += p.pnl or 0.0

        # Bankroll CAGR (annualized, if we have > 1 day)
        if elapsed_days >= 1:
            cagr = (self.current_bankroll / self.initial_bankroll) ** (365 / elapsed_days) - 1
        else:
            cagr = None

        return {
            "bets_settled":        n,
            "bets_pending":        pending,
            "win_rate_pct":        len(wins) / n * 100,
            "mean_return_pct":     mean_r * 100,
            "std_return_pct":      std_r * 100,
            "sharpe_per_bet":      sharpe,
            "profit_factor":       pf,
            "total_pnl":           total_pnl,
            "total_fees_paid":     total_fees,
            "max_drawdown_dollars":max_dd,
            "initial_bankroll":    self.initial_bankroll,
            "current_bankroll":    self.current_bankroll,
            "bankroll_return_pct": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100,
            "cagr_pct":            cagr * 100 if cagr is not None else None,
            "elapsed_days":        elapsed_days,
            "bets_per_day":        n / elapsed_days,
            "dual_arbs_detected":  len(self.arbs),
            "by_series":           by_series,
        }

    def print_report(self):
        m = self.metrics()
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print()
        print("=" * 65)
        print(f"  PAPER TRADING REPORT  —  {now_str}")
        print("=" * 65)

        if "message" in m:
            print(f"  {m['message']}")
            print("=" * 65)
            return

        print(f"  Bets settled        : {m['bets_settled']}")
        print(f"  Bets pending        : {m['bets_pending']}")
        print(f"  Win rate            : {m['win_rate_pct']:.1f}%")
        print(f"  Mean return / bet   : {m['mean_return_pct']:+.2f}%")
        print(f"  Std return / bet    : {m['std_return_pct']:.2f}%")
        print(f"  Sharpe (per-bet)    : {m['sharpe_per_bet']:.3f}")
        print(f"    (>0.5 = useful; >1.0 = solid; >2.0 = exceptional)")
        print(f"  Profit factor       : {m['profit_factor']:.2f}x")
        print(f"    (>1.0 = profitable; >1.5 = good)")
        print(f"  Total P&L           : ${m['total_pnl']:+.4f}")
        print(f"  Total fees paid     : ${m['total_fees_paid']:.4f}")
        print(f"  Max drawdown        : ${m['max_drawdown_dollars']:.4f}")
        print(f"  Bankroll            : ${m['initial_bankroll']:.2f} → ${m['current_bankroll']:.2f}  "
              f"({m['bankroll_return_pct']:+.2f}%)")
        if m["cagr_pct"] is not None:
            print(f"  Annualized CAGR     : {m['cagr_pct']:+.1f}%")
        print(f"  Elapsed             : {m['elapsed_days']:.2f} days")
        print(f"  Bets / day          : {m['bets_per_day']:.1f}")
        print(f"  Dual arbs detected  : {m['dual_arbs_detected']}")
        print()
        print("  Per-series breakdown:")
        for series, d in m["by_series"].items():
            wr = d["wins"] / d["n"] * 100 if d["n"] else 0
            print(
                f"    {series:<14} n={d['n']:<4}  "
                f"win_rate={wr:.1f}%  "
                f"pnl=${d['pnl']:+.4f}"
            )
        print("=" * 65)

        try:
            with open(PAPER_REPORT_JSON, "w") as f:
                json.dump(m, f, indent=2, default=str)
            print(f"  Saved → {PAPER_REPORT_JSON}")
        except Exception as e:
            print(f"  (Report save failed: {e})")
        print()


# --------------------------------------------------------------------------- #
# Paper trader — extends LiveRunner
# --------------------------------------------------------------------------- #

class PaperTrader(LiveRunner):
    """
    Extends LiveRunner: intercepts _scheduled_decide() to simulate fills,
    polls for settlement, and tracks all risk metrics via PaperBook.
    """

    def __init__(self, private_key, brain: StrategyBrain, initial_bankroll: float):
        super().__init__(
            private_key  = private_key,
            brain        = brain,
            jsonl_path   = PAPER_TRADES_JSONL,
        )
        self._book      = PaperBook(initial_bankroll)
        self._close_ts: Dict[str, float] = {}    # ticker → settlement unix ts
        self._binance   = BinanceFeed()
        self._floor_strikes: Dict[str, float] = {}  # ticker → floor strike price

    # ------------------------------------------------------------------ #
    # Override discovery to also capture close_ts                         #
    # ------------------------------------------------------------------ #

    async def _discover_loop(self):
        while True:
            self._bankroll = self._book.current_bankroll
            log.info(f"Paper bankroll: ${self._bankroll:.2f}  "
                     f"(settled={len(self._book.done)}  "
                     f"pending={len(self._book.open)})")

            now = time.time()
            for series in SERIES_SPEC:
                markets = _fetch_open_markets(self._key, series)
                for m in markets:
                    ticker = m.get("ticker", "")
                    if not ticker or ticker in self._tracked:
                        continue

                    open_time_str  = m.get("open_time",  "")
                    close_time_str = m.get("close_time", "")

                    try:
                        open_ts = datetime.datetime.fromisoformat(
                            open_time_str.replace("Z", "+00:00")
                        ).timestamp()
                    except Exception:
                        continue

                    try:
                        close_ts = datetime.datetime.fromisoformat(
                            close_time_str.replace("Z", "+00:00")
                        ).timestamp()
                    except Exception:
                        close_ts = open_ts + 900  # 15-min fallback

                    # Skip windows that are already past 12 minutes (too late to enter)
                    if now > open_ts + 720:
                        log.debug(f"Skip {ticker}: past 12-minute entry cutoff")
                        continue

                    floor_strike = float(m.get("floor_strike", 0.0))
                    self._tracked.add(ticker)
                    self._close_ts[ticker] = close_ts
                    self._floor_strikes[ticker] = floor_strike
                    self._brain.register_window(
                        ticker, series, open_ts, floor_strike=floor_strike
                    )

                    # Push to subscription queue so WS subscribes immediately
                    await self._sub_queue.put(ticker)

                    close_str = datetime.datetime.utcfromtimestamp(close_ts).strftime("%H:%M:%S")
                    log.info(
                        f"Registered {ticker} ({series})  "
                        f"entry=confidence-gated  close={close_str}Z"
                    )

            await asyncio.sleep(MARKET_DISCOVERY_INTERVAL_SEC)

    # ------------------------------------------------------------------ #
    # Override decide: add paper fill + dual arb check                    #
    # ------------------------------------------------------------------ #

    async def _scheduled_decide(self, ticker: str, delay_sec: float):
        await asyncio.sleep(max(0.0, delay_sec))

        prices = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_market_prices, self._key, ticker
        )
        if prices is None:
            prices = {"yes_ask": 0.0, "no_ask": 0.0}

        yes_ask  = prices["yes_ask"]
        no_ask   = prices["no_ask"]
        bankroll = self._book.current_bankroll

        decision = self._brain.decide(
            ticker   = ticker,
            yes_ask  = yes_ask,
            no_ask   = no_ask,
            bankroll = bankroll,
        )

        _print_decision(decision)

        # --- Dual-side arb (independent of brain) ---
        self._check_dual_arb(ticker, decision.series, yes_ask, no_ask, bankroll)

        # --- Paper fill if brain says BET ---
        if decision.action == "BET" and decision.contracts > 0:
            fill_price   = yes_ask if decision.side == "yes" else no_ask
            fee          = kalshi_fee(decision.contracts, fill_price)
            total_cost   = fill_price * decision.contracts
            total_outlay = total_cost + fee

            close_ts = self._close_ts.get(ticker, time.time() + 900)

            pos = SimulatedPosition(
                ticker       = ticker,
                series       = decision.series,
                side         = decision.side,
                contracts    = decision.contracts,
                fill_price   = fill_price,
                total_cost   = total_cost,
                fill_fee     = fee,
                total_outlay = total_outlay,
                decision_ts  = time.time(),
                close_ts     = close_ts,
                p_hat        = decision.p_hat,
                ap           = decision.ap,
                n_trades     = decision.n_trades,
                kelly_capped = decision.kelly_capped,
                b0           = decision.b0,
                b1           = decision.b1,
                gbm          = decision.gbm_prob,
            )

            self._book.add(pos)

            # Append open event to JSONL
            _append_jsonl(PAPER_TRADES_JSONL, {
                "event":    "open",
                **dataclasses.asdict(pos),
                "ts_human": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
            })

            # Schedule settlement poll
            asyncio.create_task(self._await_settlement(pos))

        self._decision_tasks.pop(ticker, None)

    # ------------------------------------------------------------------ #
    # Dual-side arb detection                                             #
    # ------------------------------------------------------------------ #

    def _check_dual_arb(
        self, ticker: str, series: str,
        yes_ask: float, no_ask: float, bankroll: float
    ):
        if yes_ask <= 0 or no_ask <= 0:
            return
        combined = yes_ask + no_ask
        if combined >= DUAL_ARB_THRESHOLD:
            return

        # How many pairs fit in 5% of bankroll?
        contracts = max(1, int(0.05 * bankroll / combined))

        fee_yes = kalshi_fee(contracts, yes_ask)
        fee_no  = kalshi_fee(contracts, no_ask)
        total_fees = fee_yes + fee_no
        total_outlay = combined * contracts + total_fees
        locked_profit = contracts * 1.00 - total_outlay

        if locked_profit <= 0:
            return  # fees eat the profit

        rec = DualArbRecord(
            ticker        = ticker,
            series        = series,
            yes_ask       = yes_ask,
            no_ask        = no_ask,
            combined      = combined,
            contracts     = contracts,
            total_outlay  = total_outlay,
            total_fees    = total_fees,
            locked_profit = locked_profit,
            ts            = time.time(),
        )
        self._book.add_arb(rec)
        _append_jsonl(PAPER_TRADES_JSONL, {
            "event": "dual_arb",
            **dataclasses.asdict(rec),
        })

    # ------------------------------------------------------------------ #
    # Settlement polling                                                  #
    # ------------------------------------------------------------------ #

    async def _await_settlement(self, pos: SimulatedPosition):
        """Wait until close_ts + delay, then poll for result."""
        wait = pos.close_ts + SETTLE_POLL_DELAY - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        for attempt in range(SETTLE_POLL_RETRIES):
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._poll_result, pos.ticker
            )
            if result is not None:
                settled = self._book.settle(pos.ticker, result)
                if settled:
                    self._brain.on_settlement(settled.pnl or 0.0)
                    _append_jsonl(PAPER_TRADES_JSONL, {
                        "event":    "settle",
                        **dataclasses.asdict(settled),
                        "ts_human": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
                    })
                return

            log.info(
                f"Settlement pending: {pos.ticker} "
                f"(attempt {attempt+1}/{SETTLE_POLL_RETRIES})"
            )
            await asyncio.sleep(SETTLE_POLL_INTERVAL)

        log.warning(f"Could not settle {pos.ticker} after {SETTLE_POLL_RETRIES} attempts")

    def _poll_result(self, ticker: str) -> Optional[str]:
        """Returns 'yes', 'no', or None if not yet settled."""
        try:
            resp   = _signed_get(self._key, f"/trade-api/v2/markets/{ticker}")
            market = resp.get("market", {})
            result = market.get("result", "")
            if result in ("yes", "no"):
                return result
            # Finalized markets that don't have 'result' key yet
            if market.get("status", "") == "finalized":
                return result or None
            return None
        except Exception as e:
            log.debug(f"Settlement poll error {ticker}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Hourly metrics print + run()                                        #
    # ------------------------------------------------------------------ #

    async def _entry_loop(self):
        """
        Every 30s, check all tracked windows for confidence-gated entry.
        Replaces the fixed one-shot _scheduled_decide approach.
        Brain's try_decide() returns BET only when GBM is confident AND AP confirms.
        """
        while True:
            await asyncio.sleep(30)
            now = time.time()

            for ticker in list(self._tracked):
                w = self._brain.window_state(ticker)
                if w is None or w.decided:
                    continue

                close_ts = self._close_ts.get(ticker, now + 999)
                if close_ts - now < 90:
                    continue  # too close to settlement, skip

                # Only fetch Kalshi prices if GBM is actually confident
                gbm = w.gbm_prob
                if abs(gbm - 0.5) < 0.10:
                    continue  # GBM still ambiguous, don't even bother fetching

                prices = await asyncio.get_event_loop().run_in_executor(
                    None, _fetch_market_prices, self._key, ticker
                )
                if prices is None:
                    continue

                yes_ask  = prices["yes_ask"]
                no_ask   = prices["no_ask"]
                bankroll = self._book.current_bankroll

                decision = self._brain.try_decide(ticker, yes_ask, no_ask, bankroll)
                _print_decision(decision)

                self._check_dual_arb(ticker, decision.series, yes_ask, no_ask, bankroll)

                if decision.action == "BET" and decision.contracts > 0:
                    await self._open_position(ticker, decision, yes_ask, no_ask, close_ts)

    async def _open_position(
        self, ticker: str, decision, yes_ask: float, no_ask: float, close_ts: float
    ):
        """Simulate a paper fill and schedule settlement polling."""
        fill_price   = yes_ask if decision.side == "yes" else no_ask
        fee          = kalshi_fee(decision.contracts, fill_price)
        total_cost   = fill_price * decision.contracts
        total_outlay = total_cost + fee

        pos = SimulatedPosition(
            ticker       = ticker,
            series       = decision.series,
            side         = decision.side,
            contracts    = decision.contracts,
            fill_price   = fill_price,
            total_cost   = total_cost,
            fill_fee     = fee,
            total_outlay = total_outlay,
            decision_ts  = time.time(),
            close_ts     = close_ts,
            p_hat        = decision.p_hat,
            ap           = decision.ap,
            n_trades     = decision.n_trades,
            kelly_capped = decision.kelly_capped,
            b0           = decision.b0,
            b1           = decision.b1,
            gbm          = decision.gbm_prob,
        )

        self._book.add(pos)
        _append_jsonl(PAPER_TRADES_JSONL, {
            "event":    "open",
            **dataclasses.asdict(pos),
            "reason":   decision.reason,
            "ts_human": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        })
        asyncio.create_task(self._await_settlement(pos))

    async def _metrics_loop(self):
        while True:
            await asyncio.sleep(METRICS_PRINT_EVERY)
            self._book.print_report()

    async def _gbm_loop(self):
        """Update GBM probability every 4 seconds. Also checks dynamic exits."""
        while True:
            await asyncio.sleep(4)
            now = time.time()
            for ticker in list(self._tracked):
                close_ts = self._close_ts.get(ticker)
                if close_ts is None:
                    continue
                t_remaining = close_ts - now
                if t_remaining <= 0:
                    continue
                series = self._brain._series_from_ticker(ticker)
                s_now = self._binance.get_price(series)
                sigma = self._binance.get_sigma_per_second(series)
                if s_now <= 0 or sigma <= 0:
                    continue
                prob = self._brain.update_gbm(ticker, s_now, sigma, t_remaining)
                log.debug(
                    f"GBM {ticker}: s={s_now:.2f} σ={sigma:.6f} "
                    f"t={t_remaining:.0f}s P(YES)={prob:.4f}"
                )

            # Check exit conditions for all open positions
            await self._check_position_exits()

    async def _check_position_exits(self):
        """
        Dynamic exit logic. Every 4s, check each open position:
          1. Signal reversal: GBM has flipped hard against the position
          2. Pin risk: price within 0.1% of floor_strike in last 2 minutes — coin flip, exit
          3. Pre-expiry rescue: <90s to settlement, take whatever bid is available
        """
        now = time.time()
        for ticker, pos in list(self._book.open.items()):
            # Don't exit in first MIN_HOLD_SECS after entry (noise filter)
            if now - pos.decision_ts < MIN_HOLD_SECS:
                continue

            w = self._brain.window_state(ticker)
            gbm = w.gbm_prob if w else 0.5
            close_ts = self._close_ts.get(ticker, now + 999)
            t_remaining = close_ts - now

            exit_reason = None

            # --- Signal reversal exit ---
            if pos.side == "yes" and gbm < EXIT_GBM_THRESHOLD:
                exit_reason = f"signal_flip_gbm={gbm:.3f}"
            elif pos.side == "no" and gbm > (1.0 - EXIT_GBM_THRESHOLD):
                exit_reason = f"signal_flip_gbm={gbm:.3f}"

            # --- Pin risk exit (last 2 minutes, price at strike = coin flip) ---
            if exit_reason is None and t_remaining <= PIN_WINDOW_SECS:
                series = self._brain._series_from_ticker(ticker)
                s_now = self._binance.get_price(series)
                floor_strike = self._floor_strikes.get(ticker, 0.0)
                if s_now > 0 and floor_strike > 0:
                    dist_pct = abs(s_now - floor_strike) / floor_strike
                    if dist_pct < PIN_ZONE_PCT:
                        exit_reason = (
                            f"pin_risk_t={t_remaining:.0f}s "
                            f"s={s_now:.2f} k={floor_strike:.2f} "
                            f"dist={dist_pct*100:.3f}%"
                        )

            # --- Pre-expiry rescue exit ---
            if exit_reason is None and t_remaining <= PRE_EXPIRY_SECS:
                exit_reason = f"pre_expiry_t={t_remaining:.0f}s"

            if exit_reason:
                exited_pos = pos
                exited_side = pos.side
                await self._exit_position(pos, exit_reason)

                # After a pin-risk exit: if price is clearly on the wrong side,
                # place a small counter-bet (40% normal Kelly, same window)
                if "pin_risk" in exit_reason and t_remaining > 30:
                    series = self._brain._series_from_ticker(ticker)
                    s_now = self._binance.get_price(series)
                    floor_strike = self._floor_strikes.get(ticker, 0.0)
                    if s_now > 0 and floor_strike > 0:
                        counter_side = "no" if s_now < floor_strike else "yes"
                        if counter_side != exited_side:
                            # Only flip if price actually crossed to other side
                            log.info(
                                f"PIN FLIP candidate {ticker}: "
                                f"s={s_now:.2f} k={floor_strike:.2f} "
                                f"→ counter={counter_side.upper()}"
                            )
                            await self._pin_counter_bet(
                                ticker, series, counter_side, t_remaining
                            )

    async def _pin_counter_bet(
        self, ticker: str, series: str, side: str, t_remaining: float
    ):
        """Small counter-bet after pin exit. Reduced size, last-minute only."""
        prices = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_market_prices, self._key, ticker
        )
        if prices is None:
            return

        yes_ask = prices["yes_ask"]
        no_ask  = prices["no_ask"]
        entry_price = yes_ask if side == "yes" else no_ask

        if entry_price <= 0.02 or entry_price >= 0.98:
            return  # market already almost settled, spread is gone

        # Use brain's Kelly but scale down to PIN_FLIP_KELLY_SCALE
        # p is roughly what GBM says; in pin zone it's near 0.5
        # We're betting because price just crossed the strike, not because model says so
        # Use 50/50 adjusted: if price just went below strike, P(NO) slightly > 0.5
        p_counter = 0.58  # modest edge estimate for a fresh pin cross
        from .sizing import kelly_fraction, kalshi_fee
        kf = kelly_fraction(p_counter, entry_price, side)
        if kf <= 0:
            return

        bankroll = self._book.current_bankroll
        kf_capped = min(kf * PIN_FLIP_KELLY_SCALE, 0.02)  # hard cap 2% for counter
        contracts = max(1, math.floor(bankroll * kf_capped / entry_price))

        fee          = kalshi_fee(contracts, entry_price)
        total_cost   = entry_price * contracts
        total_outlay = total_cost + fee

        close_ts = self._close_ts.get(ticker, time.time() + t_remaining)

        pos = SimulatedPosition(
            ticker       = ticker,
            series       = series,
            side         = side,
            contracts    = contracts,
            fill_price   = entry_price,
            total_cost   = total_cost,
            fill_fee     = fee,
            total_outlay = total_outlay,
            decision_ts  = time.time(),
            close_ts     = close_ts,
            p_hat        = p_counter,
            ap           = 0.0,
            n_trades     = 0,
            kelly_capped = kf_capped,
            b0           = 0.0,
            b1           = 0.0,
            gbm          = p_counter,
        )

        self._book.add(pos)
        _append_jsonl(PAPER_TRADES_JSONL, {
            "event":    "open",
            **dataclasses.asdict(pos),
            "reason":   f"pin_counter_t={t_remaining:.0f}s",
            "ts_human": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        })
        asyncio.create_task(self._await_settlement(pos))

    async def _exit_position(self, pos: "SimulatedPosition", reason: str):
        """Simulate selling the position at current Kalshi bid."""
        # Fetch current market prices
        prices = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_market_prices, self._key, pos.ticker
        )
        if prices is None:
            log.warning(f"Exit aborted for {pos.ticker}: could not fetch prices")
            return

        yes_ask = prices["yes_ask"]
        no_ask  = prices["no_ask"]

        # In a binary, selling YES ≈ receiving YES bid ≈ 1 - no_ask
        # Selling NO ≈ receiving NO bid ≈ 1 - yes_ask
        if pos.side == "yes":
            exit_price = max(0.01, 1.0 - no_ask)
        else:
            exit_price = max(0.01, 1.0 - yes_ask)

        # Simulate fill
        exit_fee      = kalshi_fee(pos.contracts, exit_price)
        gross_proceeds = exit_price * pos.contracts
        net_proceeds   = gross_proceeds - exit_fee
        pnl            = net_proceeds - pos.total_outlay

        # Update position state
        pos.settled    = True
        pos.win        = pnl > 0
        pos.pnl        = round(pnl, 4)
        pos.return_pct = round(pnl / pos.total_outlay, 4) if pos.total_outlay else 0.0

        # Remove from open book, add to done
        self._book.open.pop(pos.ticker, None)
        self._book.done.append(pos)
        self._book.current_bankroll += pos.total_outlay + pnl
        self._book.peak_bankroll = max(self._book.peak_bankroll, self._book.current_bankroll)
        self._brain.on_settlement(pnl)

        icon = "✅" if pos.win else "🔴"
        print(
            f"  {icon} EARLY EXIT  {pos.ticker[:40]}  "
            f"reason={reason}  side={pos.side.upper()}  "
            f"entry=${pos.fill_price:.3f} exit=${exit_price:.3f}  "
            f"pnl=${pnl:+.3f} ({pos.return_pct*100:+.1f}%)  "
            f"paper_bankroll=${self._book.current_bankroll:.2f}"
        )

        _append_jsonl(PAPER_TRADES_JSONL, {
            "event":       "exit",
            **dataclasses.asdict(pos),
            "exit_reason": reason,
            "exit_price":  round(exit_price, 4),
            "ts_human":    datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        })

    async def run(self):
        asyncio.create_task(self._binance.start())
        asyncio.create_task(self._gbm_loop())
        asyncio.create_task(self._entry_loop())
        await asyncio.gather(
            self._discover_loop(),
            self._ws_loop(),
            self._metrics_loop(),
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _append_jsonl(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

async def _main(initial_bankroll: float):
    print("=" * 65)
    print("  KALSHI PAPER TRADER")
    print("  Real WebSocket data. Simulated fills. No real orders.")
    print(f"  Starting paper bankroll: ${initial_bankroll:.2f}")
    print("  Ctrl+C → print full report")
    print("=" * 65)

    private_key = _load_key(KEY_PATH)

    real_balance = _fetch_balance(private_key)
    print(f"  Real Kalshi balance (FYI): ${real_balance:.2f}")

    print("\nFitting OLS coefficients from historical data...")
    coefs = fit_all_series(verbose=True)
    for series, (b0, b1) in coefs.items():
        print(f"  {series}: b0={b0:.4f}  b1={b1:.4f}")

    brain  = StrategyBrain(coefs=coefs, initial_bankroll=initial_bankroll)
    trader = PaperTrader(
        private_key      = private_key,
        brain            = brain,
        initial_bankroll = initial_bankroll,
    )

    # Graceful Ctrl+C → print report
    def _on_sigint(signum, frame):
        print("\n\nInterrupted.")
        trader._book.print_report()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)

    # Daily recalibration at 2am UTC
    async def _daily_recal():
        while True:
            now     = datetime.datetime.now(datetime.timezone.utc)
            next_2am = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_2am <= now:
                next_2am += datetime.timedelta(days=1)
            await asyncio.sleep((next_2am - now).total_seconds())
            log.info("Running daily OLS recalibration...")
            new_coefs = fit_all_series(verbose=True)
            brain.update_coefs(new_coefs)
            log.info("Recalibration complete.")

    await asyncio.gather(trader.run(), _daily_recal())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi paper trader")
    parser.add_argument(
        "--bankroll", type=float, default=500.0,
        help="Starting paper bankroll in dollars (default: 500)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    asyncio.run(_main(args.bankroll))
