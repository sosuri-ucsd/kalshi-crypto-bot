# Kalshi Crypto Arbitrage Bot

A quantitative trading bot exploiting the Binance-to-Kalshi repricing lag on 15-minute binary prediction markets for BTC, ETH, and SOL.

---

## Overview

Kalshi's BTC/ETH/SOL 15-minute markets settle YES if the Binance spot price at close is at or above a floor strike recorded at window open. Kalshi reprices these odds 3-7 seconds after Binance spot moves. During that window, the market is mispriced. This bot connects to Binance at 200ms WebSocket latency, detects the move in real time, and bets on Kalshi before odds catch up. Two signals drive every decision - an Anchor Price (AP) signal derived from early-window Kalshi trade prices, and a GBM signal computed from live Binance spot price and rolling volatility. Both must agree on direction before any bet is placed. The system currently runs in paper trading mode on live Kalshi data.

---

## How It Works

### The Edge

Kalshi reprices BTC/ETH/SOL odds 3-7 seconds after Binance spot moves. During that window the market is mispriced. The bot detects the move on Binance and bets on Kalshi before it catches up.

### Signal 1: Anchor Price (AP)

The Anchor Price is the mean of YES trade prices accumulated during the first K seconds of each 15-minute window. Kalshi's early-window price already encodes the market's partial read on where spot is relative to the floor strike. An OLS regression maps AP to P(YES):

```
p_hat = b0 + b1 * AP
```

Coefficients are refit daily on an expanding window of prior data (`bot/recal.py`). Validated on historical Kalshi trade data through a 12-step research pipeline with chronological train/holdout splits and block-bootstrap confidence intervals.

### Signal 2: GBM (Geometric Brownian Motion)

The GBM signal computes the risk-neutral probability using live Binance spot price and rolling realized volatility:

```
Z = (ln(S_now) - ln(floor_strike)) / (sigma_per_second * sqrt(T_remaining))
P(YES) = Normal_CDF(Z)
```

`sigma_per_second` is the standard deviation of log-returns from the trailing 600 Binance aggTrade ticks (approximately 10 minutes), normalized from per-tick to per-second units via `sqrt(ticks_per_second)`. GBM is capped at 0.80 - crypto fat tails mean the formula overestimates certainty at extremes.

### Entry Logic

1. GBM confidence gate: `|GBM - 0.5|` must exceed a per-series floor before any evaluation begins
2. AP must confirm GBM direction - both signals must be on the same side, not just not-contradicting
3. Bounce filter: for YES bets, GBM must have dipped below 0.45 at some earlier point in the window and now be recovering. For NO bets, GBM must have spiked above 0.55 and now be falling. This avoids buying the top or selling the bottom. The filter is bypassed if GBM exceeds 0.75 (strong trend, not a bounce).
4. Blend and threshold: `blended_p = 0.3 * p_hat + 0.7 * GBM_capped`. No bet if `|blended_p - 0.5| < T`.

### Exit Logic

- Dynamic exit: if GBM reverses past threshold mid-window (below 0.40 for YES, above 0.60 for NO), sell at current bid
- Pin risk: if spot is within 0.1% of floor strike in the last 2 minutes, exit immediately - outcome is coin-flip and Kalshi spread is widest at the pin
- Pre-expiry: close all positions 90 seconds before settlement regardless of GBM value

### Sizing

Kelly criterion with half-Kelly multiplier (0.5x), tiered by signal strength:

| Edge magnitude | Max bet |
|----------------|---------|
| >= 30% | 25% of bankroll |
| >= 20% | 18% of bankroll |
| >= 12% | 10% of bankroll |
| < 12% | 5% of bankroll |

Circuit breakers: 10% session drawdown halves the Kelly fraction; 20% session drawdown pauses all trading for 15 minutes.

---

## Series Configuration

| Series | Asset | AP window (K) | GBM floor | Bet cap |
|--------|-------|---------------|-----------|---------|
| KXBTC15M | Bitcoin | 300s | 0.10 | 25% |
| KXETH15M | Ethereum | 120s | 0.15 | 18% |
| KXSOL15M | Solana | 300s | 0.15 | 18% |

BTC uses a lower GBM floor because it is less volatile - GBM moves slowly on BTC and needs a lower threshold to fire. ETH uses a shorter AP window because it has lower Kalshi trade volume, so fewer trades accumulate in 300s.

---

## Architecture

```
bot/
  brain.py         - pure decision logic (no API calls, no credentials)
  config.py        - all strategy constants
  sizing.py        - Kelly fraction, Kalshi fee calculation
  recal.py         - daily OLS recalibration on historical data
  run_live.py      - Kalshi WebSocket and REST integration
  paper_trader.py  - paper trading on live Kalshi data (simulated fills)
  manual.py        - signal mode: live pulse + alerts, you place bets manually
  executor.py      - order execution stub (not yet connected to live trading)
  feeds/
    binance_ws.py  - Binance WebSocket feed, rolling realized sigma
```

`brain.py` is the core. It is pure logic: no network calls, no file I/O, no credentials. All external I/O lives in `run_live.py` and `paper_trader.py`. This separation makes the signal logic independently testable and replaceable.

---

## Setup

```bash
pip install websockets requests cryptography numpy pandas scikit-learn
```

1. Place your Kalshi RSA private key at `kalshi_key.key` in the repo root.
2. Update `KEY_ID` in `bot/config.py` with your Kalshi API key ID.
3. Prepare the feature dataset: `h1_comprehensive_features.csv` must be present at the repo root for OLS recalibration. See `bot/recal.py` for the expected schema.

---

## Running

### Paper trading (simulated fills on live Kalshi data)

```bash
python3 -m bot.paper_trader --bankroll 20
```

With debug logging to see GBM updates every 4 seconds:

```bash
python3 -m bot.paper_trader --bankroll 20 --log-level DEBUG
```

### Manual signal mode

The bot watches all open markets and prints a live pulse every 4 seconds showing GBM probability, spot vs. strike distance, and a one-line action recommendation. When a signal fires, it prints a full bet card with exact contracts and outlay based on your real Kalshi balance. You place the bet manually on Kalshi.

```bash
python3 -m bot.manual
```

### Dry-run order check

```bash
python3 -c "
from bot.executor import place_order
from bot.run_live import _load_key
key = _load_key('kalshi_key.key')
result = place_order(key, 'KXBTC15M-26JUL031445-50000', 'yes', 1, 55, dry_run=True)
print(result)
"
```

---

## Results

Paper trading results are tracked in `bot/paper_trades.jsonl` (gitignored locally). The paper trader runs on live Kalshi WebSocket data with simulated fills at the ask price. Metrics including win rate, Sharpe per bet, profit factor, and bankroll curve are printed hourly and on Ctrl+C exit.

Results are from paper trading on live Kalshi data and do not guarantee live trading performance.

---

## Disclaimer

This project is for research and educational purposes. Prediction market trading involves financial risk. Paper trading performance does not guarantee future live results.
