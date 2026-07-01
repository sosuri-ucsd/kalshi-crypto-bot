"""
bot/brain.py — StrategyBrain: the complete decision engine.

This module is PURE LOGIC. It:
  - Accepts trade messages via on_trade()
  - Accepts new window registrations via register_window()
  - Produces TradeDecision objects via decide()
  - Tracks session-level drawdown for circuit breakers

It NEVER:
  - Places orders
  - Calls any external API
  - Reads credentials

The caller (run_live.py or Claude Code integration) is responsible for:
  - Subscribing to WebSocket trade messages
  - Discovering active windows and their open_ts
  - Fetching current yes_ask / no_ask at decision time
  - Fetching current bankroll
  - Logging the returned TradeDecision
  - Calling on_settlement() after each bet resolves

Architecture note
-----------------
Each 15-minute window has its own WindowState that accumulates the first K
seconds of trade prices. At exactly open_ts + K, the caller invokes decide().
Only one decision is produced per window (subsequent calls return PASS
with reason='already_decided').
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import (
    SERIES_SPEC,
    MIN_TRADES_LIQUIDITY,
    KELLY_HALF,
    CAP_FRAC,
    DRAWDOWN_HALVE,
    DRAWDOWN_PAUSE,
    PAUSE_DURATION_SEC,
)
from .sizing import kalshi_fee, kelly_fraction


# ---------------------------------------------------------------------------
# GBM helpers
# ---------------------------------------------------------------------------

def _norm_cdf(z: float) -> float:
    """Abramowitz & Stegun rational approximation of Normal CDF. Max error 7.5e-8."""
    sign = 1.0 if z >= 0 else -1.0
    z = abs(z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530
              + t * (-0.356563782
              + t * (1.781477937
              + t * (-1.821255978
              + t * 1.330274429))))
    approx = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    return approx if sign > 0 else 1.0 - approx


def gbm_prob(
    s_now: float,
    floor_strike: float,
    sigma_per_second: float,
    t_remaining_seconds: float,
) -> float:
    """
    Probability that price ends above floor_strike under GBM.
    Returns 0.5 (neutral) if inputs are invalid or sigma is zero.
    """
    if s_now <= 0 or floor_strike <= 0 or sigma_per_second <= 0 or t_remaining_seconds <= 0:
        return 0.5
    denom = sigma_per_second * math.sqrt(t_remaining_seconds)
    if denom < 1e-10:
        return 0.5
    z = (math.log(s_now) - math.log(floor_strike)) / denom
    z = max(-5.0, min(5.0, z))
    return _norm_cdf(z)


# ---------------------------------------------------------------------------
# Output dataclass — everything Claude Code / the integration layer needs
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """
    Structured output from StrategyBrain.decide().

    If action == 'PASS', all sizing fields are 0 / empty and reason explains why.
    If action == 'BET',  submit a limit buy order for `contracts` on `side` at
                          `entry_price` (the ask that was fetched by the caller).
    """
    # Identification
    ticker: str            # e.g. 'KXBTC15M-26JUL031445-50000'
    series: str            # 'KXBTC15M' | 'KXETH15M' | 'KXSOL15M'

    # Signal
    ap: float              # Anchor Price = mean(prices where ts < open_ts + K)
    n_trades: int          # number of trades used to compute AP (liquidity check)
    p_hat: float           # OLS-implied P(YES) = b0 + b1 * ap

    # Sizing
    side: str              # 'yes' | 'no' | '' (PASS)
    entry_price: float     # ask price paid (dollars, [0,1])
    kelly_f: float         # raw half-Kelly fraction
    kelly_capped: float    # after CAP_FRAC and drawdown-scale
    contracts: int         # floor(bankroll * kelly_capped / entry_price)
    fee_dollars: float     # estimated taker fee for this order
    net_edge: float        # gross edge − fee per dollar at risk (informational)

    # Decision
    action: str            # 'BET' | 'PASS'
    reason: str            # empty for BET; explains PASS

    # Metadata
    ts: float              # unix timestamp of decision
    b0: float = 0.0        # OLS intercept used
    b1: float = 0.0        # OLS slope used
    kelly_scale: float = 1.0   # circuit-breaker scale applied (1.0 or 0.5)
    gbm_prob: float = 0.5      # GBM P(YES) at decision time (capped at 0.80)


# ---------------------------------------------------------------------------
# Per-window state
# ---------------------------------------------------------------------------

@dataclass
class WindowState:
    ticker: str
    series: str
    open_ts: float        # unix timestamp of market open
    K: int                # accumulation window in seconds

    # Accumulated trade prices within first K seconds
    prices: List[float] = field(default_factory=list)

    # Set to True after decide() runs once for this window
    decided: bool = False

    # GBM fields — populated by update_gbm() each tick
    floor_strike: float = 0.0
    gbm_prob: float     = 0.5
    gbm_prev: float     = 0.5   # GBM value from previous tick (for direction)
    gbm_min:  float     = 1.0   # lowest GBM seen this window (bounce detection)
    gbm_max:  float     = 0.0   # highest GBM seen this window (bounce detection)


# ---------------------------------------------------------------------------
# Main brain
# ---------------------------------------------------------------------------

class StrategyBrain:
    """
    Stateful decision engine.  Create once per trading session.

    Parameters
    ----------
    coefs : dict
        {series: (b0, b1)} from bot.recal.fit_all_series()
    initial_bankroll : float
        Session-start bankroll in dollars. Used for drawdown tracking.
    cap_frac : float
        Maximum Kelly fraction per bet (default from config: 0.05 = 5%).
    """

    def __init__(
        self,
        coefs: Dict[str, Tuple[float, float]],
        initial_bankroll: float,
        cap_frac: float = CAP_FRAC,
    ):
        self.coefs = coefs                          # {series: (b0, b1)}
        self._session_start = initial_bankroll
        self._current_bankroll = initial_bankroll
        self.cap_frac = cap_frac

        self._windows: Dict[str, WindowState] = {}  # ticker → WindowState
        self._paused_until: float = 0.0             # unix ts
        self._kelly_scale: float = 1.0              # 1.0 or 0.5

    # ------------------------------------------------------------------
    # Public API: window management
    # ------------------------------------------------------------------

    def register_window(
        self, ticker: str, series: str, open_ts: float, floor_strike: float = 0.0
    ) -> None:
        """
        Call this when a new 15-min market opens (or when run_live discovers it).

        Creates a fresh WindowState; safe to call multiple times for the same
        ticker (idempotent if already registered and not yet decided).
        """
        if series not in SERIES_SPEC:
            return
        if ticker in self._windows and not self._windows[ticker].decided:
            return  # already tracking, don't reset mid-window
        self._windows[ticker] = WindowState(
            ticker=ticker,
            series=series,
            open_ts=open_ts,
            K=SERIES_SPEC[series]["K"],
        )
        self._windows[ticker].floor_strike = floor_strike

    def expire_window(self, ticker: str) -> None:
        """Call when a market closes so its state can be garbage-collected."""
        self._windows.pop(ticker, None)

    # ------------------------------------------------------------------
    # Public API: confidence-gated entry (main entry path for paper/live)
    # ------------------------------------------------------------------

    def try_decide(
        self,
        ticker: str,
        yes_ask: float,
        no_ask: float,
        bankroll: float,
    ) -> TradeDecision:
        """
        Confidence-gated entry — call every 30s after market open.

        Enters only when:
          1. Past 30s hard block (floor_strike discovery period)
          2. Before 12-minute cutoff (last 3min = manage existing only)
          3. GBM is confident: |gbm_prob - 0.5| >= 0.25
          4. AP doesn't strongly contradict GBM direction
          5. Blended signal clears the edge threshold
          6. Window not already decided

        Returns TradeDecision with action=BET (enter now) or action=PASS (keep waiting).
        PASS does NOT mark the window as decided — it stays open for next evaluation.
        """
        HARD_BLOCK_SECS  = 30     # floor_strike discovery period
        LATE_CUTOFF_SECS = 720    # 12 minutes — last 3min = manage only
        GBM_CAP          = 0.80   # never trust GBM beyond 80%

        now = time.time()
        w = self._windows.get(ticker)
        series = w.series if w else self._series_from_ticker(ticker)
        spec = SERIES_SPEC.get(series, {})
        T                    = spec.get("T",         0.15)
        GBM_CONFIDENCE_FLOOR = spec.get("GBM_FLOOR", 0.15)  # per-series entry floor
        series_cap           = spec.get("CAP",       self.cap_frac)  # per-series bet cap
        b0, b1 = self.coefs.get(series, (0.0, 1.0))

        def _pass(reason: str, ap: float = 0.0, n: int = 0, p: float = 0.0) -> TradeDecision:
            return TradeDecision(
                ticker=ticker, series=series, ap=ap, n_trades=n, p_hat=p,
                side="", entry_price=0.0, kelly_f=0.0, kelly_capped=0.0,
                contracts=0, fee_dollars=0.0, net_edge=0.0,
                action="PASS", reason=reason, ts=now, b0=b0, b1=b1,
                kelly_scale=self._kelly_scale,
            )

        if w is None:
            return _pass("no_window")
        if w.decided:
            return _pass("already_decided")
        if now < self._paused_until:
            return _pass(f"circuit_breaker_paused_{int(self._paused_until - now)}s")

        # Time gates
        elapsed = now - w.open_ts
        if elapsed < HARD_BLOCK_SECS:
            return _pass(f"hard_block_{elapsed:.0f}s<30s")
        if elapsed > LATE_CUTOFF_SECS:
            return _pass(f"late_window_{elapsed:.0f}s>720s")

        # GBM confidence gate — primary entry filter
        gbm_raw = w.gbm_prob
        if abs(gbm_raw - 0.5) < GBM_CONFIDENCE_FLOOR:
            return _pass(f"gbm_unconvincing_{gbm_raw:.3f}")

        # Bounce entry filter — wait for the dip, enter on recovery
        # For YES: GBM must have dipped to ≤0.45 at some point, now recovering ≥0.55
        # For NO:  GBM must have spiked to ≥0.55 at some point, now falling ≤0.45
        # This gives a better entry price than chasing the signal at peak confidence.
        # Skip this filter if GBM is extremely strong (≥0.75) — those are strong trends,
        # not bounces, and waiting for a dip means missing the trade entirely.
        BOUNCE_DIP_THRESH   = 0.45   # GBM must have touched here for YES entry
        BOUNCE_SPIKE_THRESH = 0.55   # GBM must have touched here for NO entry
        BOUNCE_SKIP_ABOVE   = 0.75   # bypass bounce filter if GBM is this decisive

        if gbm_raw > 0.5 and gbm_raw < BOUNCE_SKIP_ABOVE:
            # Want YES — did GBM dip first?
            if w.gbm_min > BOUNCE_DIP_THRESH:
                return _pass(f"waiting_for_dip_min={w.gbm_min:.3f}_now={gbm_raw:.3f}")
            # Is it actually recovering (rising)?
            if gbm_raw <= w.gbm_prev:
                return _pass(f"not_bouncing_yet_prev={w.gbm_prev:.3f}_now={gbm_raw:.3f}")

        if gbm_raw < 0.5 and gbm_raw > (1 - BOUNCE_SKIP_ABOVE):
            # Want NO — did GBM spike first?
            if w.gbm_max < BOUNCE_SPIKE_THRESH:
                return _pass(f"waiting_for_spike_max={w.gbm_max:.3f}_now={gbm_raw:.3f}")
            # Is it actually falling?
            if gbm_raw >= w.gbm_prev:
                return _pass(f"not_falling_yet_prev={w.gbm_prev:.3f}_now={gbm_raw:.3f}")

        # AP signal
        n_trades = len(w.prices)
        if n_trades < MIN_TRADES_LIQUIDITY:
            return _pass(f"insufficient_liquidity_n={n_trades}", n=n_trades)

        ap = sum(w.prices) / len(w.prices)
        p_hat_raw = b0 + b1 * ap
        p_hat = max(0.05, min(0.95, p_hat_raw))

        # AP must confirm GBM direction — both signals must agree
        # AP p_hat must be on the same side as GBM (not just not-contradicting)
        if gbm_raw > 0.5 and p_hat < 0.50:
            return _pass(f"ap_not_confirming_yes_p={p_hat:.3f}", ap=ap, n=n_trades, p=p_hat)
        if gbm_raw < 0.5 and p_hat > 0.50:
            return _pass(f"ap_not_confirming_no_p={p_hat:.3f}", ap=ap, n=n_trades, p=p_hat)

        # Blend
        gbm_capped = max(1.0 - GBM_CAP, min(GBM_CAP, gbm_raw))
        blended_p = 0.3 * p_hat + 0.7 * gbm_capped
        gbm_p = gbm_capped

        if abs(blended_p - 0.5) < T:
            return _pass(f"below_threshold_{blended_p:.3f}", ap=ap, n=n_trades, p=blended_p)

        # Side
        side = "yes" if blended_p > 0.5 else "no"
        entry_price = yes_ask if side == "yes" else no_ask

        if entry_price <= 0.0 or entry_price >= 1.0:
            return _pass(f"bad_entry_{side}={entry_price:.3f}", ap=ap, n=n_trades, p=blended_p)

        # Kelly
        kf = kelly_fraction(blended_p, entry_price, side)
        if kf <= 0.0:
            return _pass(f"no_edge_{side}", ap=ap, n=n_trades, p=blended_p)

        # Tiered cap — bet size scales with signal strength, not flat gambling
        # Both AP and GBM must agree (enforced above), so confidence is real
        edge_magnitude = abs(blended_p - 0.5)
        if edge_magnitude >= 0.30:        # blend >= 0.80 or <= 0.20 — very strong
            tier_cap = 0.25
        elif edge_magnitude >= 0.20:      # blend 0.70-0.80 — strong
            tier_cap = 0.18
        elif edge_magnitude >= 0.12:      # blend 0.62-0.70 — moderate
            tier_cap = 0.10
        else:                             # weak signal, barely above threshold
            tier_cap = 0.05

        kf_scaled = kf * self._kelly_scale
        kf_capped = min(kf_scaled, tier_cap)

        stake = bankroll * kf_capped
        contracts = max(1, math.floor(stake / entry_price))
        fee = kalshi_fee(contracts, entry_price)

        gross_edge = (blended_p - entry_price) if side == "yes" else ((1.0 - blended_p) - entry_price)
        cost_per_contract = entry_price + fee / max(contracts, 1)
        net_edge = gross_edge - (cost_per_contract - entry_price)

        # Mark decided only on actual BET
        w.decided = True
        self._current_bankroll = bankroll

        return TradeDecision(
            ticker=ticker, series=series,
            ap=round(ap, 4), n_trades=n_trades, p_hat=round(blended_p, 4),
            side=side, entry_price=round(entry_price, 4),
            kelly_f=round(kf, 5), kelly_capped=round(kf_capped, 5),
            contracts=contracts, fee_dollars=round(fee, 4),
            net_edge=round(net_edge, 5),
            action="BET",
            reason=f"t+{elapsed:.0f}s ap={p_hat:.3f} gbm={gbm_p:.3f} blend={blended_p:.3f}",
            ts=now, b0=round(b0, 6), b1=round(b1, 6),
            kelly_scale=self._kelly_scale, gbm_prob=round(gbm_p, 4),
        )

    def try_signal(
        self,
        ticker: str,
        yes_ask: float,
        no_ask: float,
        bankroll: float,
        t_remaining: float,
    ) -> TradeDecision:
        """
        Like try_decide() but for manual mode:
          - Never marks w.decided = True (allows re-signaling with cooldown)
          - Allows late-window entry if GBM >= 0.85 (last 90s high-conviction bet)
          - AP confirmation relaxed in last 90s (GBM dominates near expiry)
        """
        HARD_BLOCK_SECS     = 30
        LATE_CUTOFF_SECS    = 720
        LAST_MIN_GBM_FLOOR  = 0.85   # only this confident = bet in last 90s
        GBM_CAP             = 0.80

        now    = time.time()
        w      = self._windows.get(ticker)
        series = w.series if w else self._series_from_ticker(ticker)
        spec   = SERIES_SPEC.get(series, {})
        T                    = spec.get("T",         0.15)
        GBM_CONFIDENCE_FLOOR = spec.get("GBM_FLOOR", 0.15)
        b0, b1 = self.coefs.get(series, (0.0, 1.0))

        def _pass(reason: str, ap: float = 0.0, n: int = 0, p: float = 0.0) -> TradeDecision:
            return TradeDecision(
                ticker=ticker, series=series, ap=ap, n_trades=n, p_hat=p,
                side="", entry_price=0.0, kelly_f=0.0, kelly_capped=0.0,
                contracts=0, fee_dollars=0.0, net_edge=0.0,
                action="PASS", reason=reason, ts=now, b0=b0, b1=b1,
                kelly_scale=self._kelly_scale,
            )

        if w is None:
            return _pass("no_window")
        if now < self._paused_until:
            return _pass(f"circuit_breaker_paused_{int(self._paused_until - now)}s")

        elapsed = now - w.open_ts
        if elapsed < HARD_BLOCK_SECS:
            return _pass(f"hard_block_{elapsed:.0f}s")

        gbm_raw = w.gbm_prob

        # ── Last-minute high-conviction path ───────────────────────────────
        # If GBM is >= 0.85 and < 90s left, skip all other filters and just bet.
        # "100% chance it's going up — bet right now" case.
        last_min = (t_remaining <= 90)
        if last_min:
            if abs(gbm_raw - 0.5) < LAST_MIN_GBM_FLOOR - 0.5:
                return _pass(f"last_min_gbm_not_strong_enough_{gbm_raw:.3f}")
            side        = "yes" if gbm_raw > 0.5 else "no"
            entry_price = yes_ask if side == "yes" else no_ask
            if entry_price <= 0.0 or entry_price >= 1.0:
                return _pass(f"bad_entry_{entry_price:.3f}")
            # Small flat sizing for last-minute bets — never more than 5%
            stake     = bankroll * 0.05
            contracts = max(1, math.floor(stake / entry_price))
            fee       = kalshi_fee(contracts, entry_price)
            return TradeDecision(
                ticker=ticker, series=series,
                ap=0.0, n_trades=0, p_hat=round(gbm_raw, 4),
                side=side, entry_price=round(entry_price, 4),
                kelly_f=0.05, kelly_capped=0.05,
                contracts=contracts, fee_dollars=round(fee, 4), net_edge=0.0,
                action="BET",
                reason=f"LAST_MIN gbm={gbm_raw:.3f} t={t_remaining:.0f}s",
                ts=now, b0=round(b0, 6), b1=round(b1, 6),
                kelly_scale=self._kelly_scale, gbm_prob=round(gbm_raw, 4),
            )

        # ── Normal window path ─────────────────────────────────────────────
        if elapsed > LATE_CUTOFF_SECS:
            return _pass(f"late_window_{elapsed:.0f}s>720s")

        if abs(gbm_raw - 0.5) < GBM_CONFIDENCE_FLOOR:
            return _pass(f"gbm_unconvincing_{gbm_raw:.3f}")

        # Bounce filter
        BOUNCE_DIP_THRESH   = 0.45
        BOUNCE_SPIKE_THRESH = 0.55
        BOUNCE_SKIP_ABOVE   = 0.75

        if gbm_raw > 0.5 and gbm_raw < BOUNCE_SKIP_ABOVE:
            if w.gbm_min > BOUNCE_DIP_THRESH:
                return _pass(f"waiting_for_dip_min={w.gbm_min:.3f}")
            if gbm_raw <= w.gbm_prev:
                return _pass(f"not_bouncing_yet_prev={w.gbm_prev:.3f}")

        if gbm_raw < 0.5 and gbm_raw > (1 - BOUNCE_SKIP_ABOVE):
            if w.gbm_max < BOUNCE_SPIKE_THRESH:
                return _pass(f"waiting_for_spike_max={w.gbm_max:.3f}")
            if gbm_raw >= w.gbm_prev:
                return _pass(f"not_falling_yet_prev={w.gbm_prev:.3f}")

        # AP signal
        n_trades = len(w.prices)
        if n_trades < MIN_TRADES_LIQUIDITY:
            return _pass(f"insufficient_liquidity_n={n_trades}", n=n_trades)

        ap        = sum(w.prices) / len(w.prices)
        p_hat_raw = b0 + b1 * ap
        p_hat     = max(0.05, min(0.95, p_hat_raw))

        if gbm_raw > 0.5 and p_hat < 0.50:
            return _pass(f"ap_not_confirming_yes_p={p_hat:.3f}", ap=ap, n=n_trades, p=p_hat)
        if gbm_raw < 0.5 and p_hat > 0.50:
            return _pass(f"ap_not_confirming_no_p={p_hat:.3f}", ap=ap, n=n_trades, p=p_hat)

        gbm_capped = max(1.0 - GBM_CAP, min(GBM_CAP, gbm_raw))
        blended_p  = 0.3 * p_hat + 0.7 * gbm_capped

        if abs(blended_p - 0.5) < T:
            return _pass(f"below_threshold_{blended_p:.3f}", ap=ap, n=n_trades, p=blended_p)

        side        = "yes" if blended_p > 0.5 else "no"
        entry_price = yes_ask if side == "yes" else no_ask

        if entry_price <= 0.0 or entry_price >= 1.0:
            return _pass(f"bad_entry_{entry_price:.3f}", ap=ap, n=n_trades, p=blended_p)

        kf = kelly_fraction(blended_p, entry_price, side)
        if kf <= 0.0:
            return _pass(f"no_edge_{side}", ap=ap, n=n_trades, p=blended_p)

        edge_magnitude = abs(blended_p - 0.5)
        if edge_magnitude >= 0.30:   tier_cap = 0.25
        elif edge_magnitude >= 0.20: tier_cap = 0.18
        elif edge_magnitude >= 0.12: tier_cap = 0.10
        else:                        tier_cap = 0.05

        kf_capped = min(kf * self._kelly_scale, tier_cap)
        stake     = bankroll * kf_capped
        contracts = max(1, math.floor(stake / entry_price))
        fee       = kalshi_fee(contracts, entry_price)

        gross_edge       = (blended_p - entry_price) if side == "yes" else ((1.0 - blended_p) - entry_price)
        cost_per_contract= entry_price + fee / max(contracts, 1)
        net_edge         = gross_edge - (cost_per_contract - entry_price)

        # NOTE: does NOT set w.decided = True — manual mode manages its own cooldown
        self._current_bankroll = bankroll

        return TradeDecision(
            ticker=ticker, series=series,
            ap=round(ap, 4), n_trades=n_trades, p_hat=round(blended_p, 4),
            side=side, entry_price=round(entry_price, 4),
            kelly_f=round(kf, 5), kelly_capped=round(kf_capped, 5),
            contracts=contracts, fee_dollars=round(fee, 4),
            net_edge=round(net_edge, 5),
            action="BET",
            reason=f"t+{elapsed:.0f}s ap={p_hat:.3f} gbm={gbm_capped:.3f} blend={blended_p:.3f}",
            ts=now, b0=round(b0, 6), b1=round(b1, 6),
            kelly_scale=self._kelly_scale, gbm_prob=round(gbm_capped, 4),
        )

    def update_gbm(
        self,
        ticker: str,
        s_now: float,
        sigma_per_second: float,
        t_remaining_seconds: float,
    ) -> float:
        """
        Update the GBM probability for a window. Called every ~4 seconds by the runner.
        Returns the new gbm_prob, or 0.5 if window not found or floor_strike not set.
        """
        w = self._windows.get(ticker)
        if w is None or w.floor_strike <= 0:
            return 0.5
        prob = gbm_prob(s_now, w.floor_strike, sigma_per_second, t_remaining_seconds)
        w.gbm_prev = w.gbm_prob          # save previous before overwriting
        w.gbm_prob = prob
        w.gbm_min  = min(w.gbm_min, prob)
        w.gbm_max  = max(w.gbm_max, prob)
        return prob

    # ------------------------------------------------------------------
    # Public API: trade feed
    # ------------------------------------------------------------------

    def on_trade(
        self,
        ticker: str,
        price_dollars: float,
        trade_ts: float,
    ) -> None:
        """
        Feed a single trade into the accumulator.

        Called for every WebSocket trade message for a tracked ticker.

        Args:
            ticker:        market ticker string
            price_dollars: yes_price in dollars [0, 1]
            trade_ts:      unix timestamp from the trade's created_time field
        """
        w = self._windows.get(ticker)
        if w is None or w.decided:
            return
        # Only accumulate trades within the first K seconds of the window
        if trade_ts < w.open_ts + w.K:
            w.prices.append(price_dollars)

    # ------------------------------------------------------------------
    # Public API: decision
    # ------------------------------------------------------------------

    def decide(
        self,
        ticker: str,
        yes_ask: float,
        no_ask: float,
        bankroll: float,
    ) -> TradeDecision:
        """
        Produce a TradeDecision at the K-second decision instant.

        Call this at open_ts + K (or as soon as K seconds have elapsed
        since market open). The caller must provide current Kalshi ask prices
        and the available account balance.

        Args:
            ticker:    market ticker
            yes_ask:   current YES ask in dollars [0, 1] (from Kalshi REST)
            no_ask:    current NO ask in dollars [0, 1]
            bankroll:  available balance in dollars (cash only, not locked)

        Returns:
            TradeDecision with action='BET' or action='PASS'.
            The caller logs this and optionally submits the order.
        """
        now = time.time()
        self._current_bankroll = bankroll

        w = self._windows.get(ticker)
        series = w.series if w else self._series_from_ticker(ticker)
        spec = SERIES_SPEC.get(series, {})
        T = spec.get("T", 0.15)
        b0, b1 = self.coefs.get(series, (0.0, 1.0))

        def _pass(reason: str, ap: float = 0.0, n: int = 0, p: float = 0.0) -> TradeDecision:
            return TradeDecision(
                ticker=ticker, series=series, ap=ap, n_trades=n, p_hat=p,
                side="", entry_price=0.0, kelly_f=0.0, kelly_capped=0.0,
                contracts=0, fee_dollars=0.0, net_edge=0.0,
                action="PASS", reason=reason, ts=now, b0=b0, b1=b1,
                kelly_scale=self._kelly_scale,
            )

        # ── PASS gate 1: no window registered ──────────────────────────
        if w is None:
            return _pass("no_window_registered")

        # ── PASS gate 2: already decided ───────────────────────────────
        if w.decided:
            return _pass("already_decided")

        # ── PASS gate 3: circuit breaker paused ────────────────────────
        if now < self._paused_until:
            remaining = int(self._paused_until - now)
            return _pass(f"circuit_breaker_paused_{remaining}s")

        # ── Mark decided immediately (even if we return PASS below) ────
        # This prevents double-firing if decide() is called twice.
        w.decided = True

        # ── Liquidity filter ───────────────────────────────────────────
        n_trades = len(w.prices)
        if n_trades < MIN_TRADES_LIQUIDITY:
            return _pass(f"insufficient_liquidity_n={n_trades}", n=n_trades)

        # ── Anchor Price ───────────────────────────────────────────────
        ap = sum(w.prices) / len(w.prices)

        # ── OLS-implied probability ────────────────────────────────────
        p_hat_raw = b0 + b1 * ap
        p_hat = max(0.05, min(0.95, p_hat_raw))

        # ── Blend AP signal with GBM ───────────────────────────────────
        # Cap GBM at [0.20, 0.80] — crypto has fat tails, never let GBM say >80%
        GBM_CAP = 0.80
        ap_weight = 0.3
        gbm_weight = 0.7
        if w and w.gbm_prob != 0.5:
            gbm_capped = max(1.0 - GBM_CAP, min(GBM_CAP, w.gbm_prob))
            blended_p = ap_weight * p_hat + gbm_weight * gbm_capped
        else:
            blended_p = p_hat  # fall back to AP only if no GBM signal

        gbm_p = gbm_capped if (w and w.gbm_prob != 0.5) else (w.gbm_prob if w else 0.5)

        # ── Threshold filter ───────────────────────────────────────────
        if abs(blended_p - 0.5) < T:
            return _pass(
                f"below_threshold_{blended_p:.3f}",
                ap=ap, n=n_trades, p=blended_p,
            )

        # ── Determine side ─────────────────────────────────────────────
        if blended_p > 0.5:
            side = "yes"
            entry_price = yes_ask
        else:
            side = "no"
            entry_price = no_ask

        # ── Safety: reject if entry_price is missing / implausible ─────
        if entry_price <= 0.0 or entry_price >= 1.0:
            return _pass(
                f"bad_entry_price_{side}={entry_price:.3f}",
                ap=ap, n=n_trades, p=blended_p,
            )

        # ── Kelly fraction ─────────────────────────────────────────────
        kf = kelly_fraction(blended_p, entry_price, side)
        if kf <= 0.0:
            return _pass(
                f"no_edge_{side}_p={blended_p:.3f}_entry={entry_price:.3f}",
                ap=ap, n=n_trades, p=blended_p,
            )

        # ── Apply circuit-breaker scale and hard cap ───────────────────
        kf_scaled = kf * self._kelly_scale
        kf_capped = min(kf_scaled, self.cap_frac)

        # ── Contracts and fee ──────────────────────────────────────────
        stake = bankroll * kf_capped
        contracts = max(1, math.floor(stake / entry_price))

        fee = kalshi_fee(contracts, entry_price)

        # ── Net edge (informational; not used in sizing) ───────────────
        if side == "yes":
            gross_edge = blended_p - entry_price
        else:
            gross_edge = (1.0 - blended_p) - entry_price
        cost_per_contract = entry_price + fee / max(contracts, 1)
        net_edge = gross_edge - (cost_per_contract - entry_price)

        return TradeDecision(
            ticker=ticker,
            series=series,
            ap=round(ap, 4),
            n_trades=n_trades,
            p_hat=round(blended_p, 4),
            side=side,
            entry_price=round(entry_price, 4),
            kelly_f=round(kf, 5),
            kelly_capped=round(kf_capped, 5),
            contracts=contracts,
            fee_dollars=round(fee, 4),
            net_edge=round(net_edge, 5),
            action="BET",
            reason=f"ap={p_hat:.4f} gbm={gbm_p:.4f} blend={blended_p:.4f}",
            ts=now,
            b0=round(b0, 6),
            b1=round(b1, 6),
            kelly_scale=self._kelly_scale,
            gbm_prob=round(gbm_p, 4),
        )

    # ------------------------------------------------------------------
    # Public API: settlement feedback
    # ------------------------------------------------------------------

    def on_settlement(self, pnl_dollars: float) -> None:
        """
        Call after each bet resolves.

        Updates bankroll and checks circuit-breaker thresholds.
        pnl_dollars is negative for a loss (after fees), positive for a win.

        Circuit breaker logic (from research synthesis):
          - 10% session drawdown → scale Kelly by 0.5
          - 20% session drawdown → pause all trading for 15 minutes
        """
        self._current_bankroll += pnl_dollars

        if self._session_start <= 0:
            return

        drawdown = (self._session_start - self._current_bankroll) / self._session_start
        drawdown = max(0.0, drawdown)

        if drawdown >= DRAWDOWN_PAUSE:
            self._paused_until = time.time() + PAUSE_DURATION_SEC
            self._kelly_scale = 1.0  # reset scale (pause supersedes halve)
        elif drawdown >= DRAWDOWN_HALVE:
            self._kelly_scale = 0.5
        else:
            self._kelly_scale = 1.0

    # ------------------------------------------------------------------
    # Public API: update OLS coefficients (called after daily recal)
    # ------------------------------------------------------------------

    def update_coefs(self, coefs: Dict[str, Tuple[float, float]]) -> None:
        """Replace OLS coefficients with freshly refitted values."""
        self.coefs = coefs

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return time.time() < self._paused_until

    @property
    def kelly_scale(self) -> float:
        return self._kelly_scale

    @property
    def session_drawdown(self) -> float:
        if self._session_start <= 0:
            return 0.0
        return max(0.0, (self._session_start - self._current_bankroll) / self._session_start)

    def window_state(self, ticker: str) -> Optional[WindowState]:
        return self._windows.get(ticker)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _series_from_ticker(ticker: str) -> str:
        """
        Best-effort series extraction from ticker string.
        e.g. 'KXBTC15M-26JUL031445-50000' → 'KXBTC15M'
        """
        for s in SERIES_SPEC:
            if ticker.startswith(s):
                return s
        # Fallback: take everything before first hyphen followed by a date-like segment
        parts = ticker.split("-")
        return parts[0] if parts else ticker
