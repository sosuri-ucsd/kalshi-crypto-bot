"""
bot/config.py — Frozen strategy specification.

All constants are derived from the validated test chain (h5 → h12).
Do NOT change K, T, or KELLY_HALF without re-running significance tests.
"""

import os

# ---------------------------------------------------------------------------
# Series specification (validated in h7 / h12)
# K: seconds to accumulate trades for Anchor Price
# T: |p_hat - 0.5| threshold to trigger a bet
# ---------------------------------------------------------------------------
SERIES_SPEC: dict[str, dict] = {
    # K:         seconds to accumulate Anchor Price trades
    # T:         |blend - 0.5| threshold to trigger a bet
    # GBM_FLOOR: |GBM - 0.5| minimum confidence to even evaluate entry
    #            BTC is less volatile so GBM moves slowly — lower floor = more bets
    # CAP:       max fraction of bankroll per bet (overrides global CAP_FRAC)
    "KXBTC15M": {"K": 300, "T": 0.15, "GBM_FLOOR": 0.10, "CAP": 0.25},  # primary — lower floor, bigger bets
    "KXETH15M": {"K": 120, "T": 0.15, "GBM_FLOOR": 0.15, "CAP": 0.18},  # default
    "KXSOL15M": {"K": 300, "T": 0.15, "GBM_FLOOR": 0.15, "CAP": 0.18},  # default — leave as is
}

# Map series → the column names in h1_comprehensive_features.csv
def ap_col(series: str) -> str:
    K = SERIES_SPEC[series]["K"]
    return f"avg_price_t{K}s"

def n_trades_col(series: str) -> str:
    K = SERIES_SPEC[series]["K"]
    return f"n_trades_t{K}s"

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
KELLY_HALF: float = 0.5          # half-Kelly multiplier (synced with h12)
CAP_FRAC:   float = 0.20         # hard cap: never risk > 20% of bankroll per bet (~$2-5 on $20)

# ---------------------------------------------------------------------------
# Liquidity filter (synced with h12)
# ---------------------------------------------------------------------------
MIN_TRADES_LIQUIDITY: int = 5    # n_trades_tKs must be >= 5

# ---------------------------------------------------------------------------
# Circuit breakers (from research: brandononchain/kalshibot pattern)
# ---------------------------------------------------------------------------
DRAWDOWN_HALVE: float  = 0.10    # at 10% session drawdown: scale Kelly by 0.5
DRAWDOWN_PAUSE: float  = 0.20    # at 20% session drawdown: pause all trading
PAUSE_DURATION_SEC: int = 900    # pause duration: 15 minutes

# ---------------------------------------------------------------------------
# File paths (relative to repo root /Users/sourishsuri/kalshi)
# ---------------------------------------------------------------------------
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV    = os.path.join(REPO_ROOT, "h1_comprehensive_features.csv")
KEY_PATH        = os.path.join(REPO_ROOT, "kalshi_key.key")
KEY_ID          = "YOUR-KALSHI-API-KEY-ID-HERE"  # replace with your Kalshi API key ID
REST_BASE       = "https://api.elections.kalshi.com"
WS_URL          = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# ---------------------------------------------------------------------------
# WebSocket / REST polling
# ---------------------------------------------------------------------------
MARKET_DISCOVERY_INTERVAL_SEC: int = 60    # poll for new open tickers every 60s
REST_ORDERBOOK_INTERVAL_SEC:   int = 30    # refresh yes_ask/no_ask every 30s

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
DECISIONS_JSONL = os.path.join(REPO_ROOT, "bot", "decisions.jsonl")
