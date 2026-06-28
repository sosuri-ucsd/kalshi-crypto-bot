# Kalshi 15-Min Crypto Edge Tests

Statistical research project testing for exploitable mispricing in Kalshi's 15-minute
binary options markets on BTC, ETH, and SOL. The core question: can early-window price
information — or its correlation with Binance spot — generate a profitable betting signal?

The answer so far is no, and the project documents exactly why, in enough methodological
detail to distinguish "tested properly and found nothing" from "didn't look hard enough."

---

## Methodology rules (applied throughout)

- **Chronological 70/30 train/holdout split.** Hyperparameters and spec selection happen on
  the train set only. Results are reported once on the untouched holdout — never re-tested.
- **BH-FDR correction** across all hypothesis batteries. Every grid search and model
  comparison applies Benjamini-Hochberg to control false discovery rate across the batch.
- **Block-bootstrap by day** for confidence intervals. Individual 15-min windows within a
  calendar day are correlated (same spot path), so bootstrapping resamples whole days.
- **Costed backtests.** Profit calculations deduct an assumed 0.01 round-trip transaction
  cost per contract throughout.
- **Holdout reported once.** No iterating on holdout results.

---

## Results

| # | Script | Question | Verdict |
|---|--------|----------|---------|
| 1 | `tests/h1_strategy_backtest.py` | Profit from betting when early price deviates from 0.50? (all series pooled) | **No edge.** Holdout mean profit −0.003/contract, 95% CI [−0.086, +0.085] over 16 days |
| 2 | `tests/h5_per_series_regression.py` | Does early avg price (`sig`) predict outcome, per-series? | **Signal is real, not exploitable.** OOS AUC 0.62–0.65 across BTC/ETH/SOL; mechanically expected (price = market's probability estimate), betting cost equals the signal |
| 3 | `tests/h6_calibration_signal_test.py` | Is the price systematically miscalibrated (slope ≠ 1)? | **Miscalibration real in-sample, doesn't generalize.** Holdout Brier improvement CIs include 0 in all 3 series |
| 4 | `tests/h3_h4_leadlag.py` | Does Kalshi price lead/lag Binance spot? | **Association BH-survives two confound checks; OOS forecasting power is zero.** H4 OOS R² = −0.00005; costed P&L = −0.010/trade |

Full methodology notes, confound checks, and diagnostics: [TESTLOG.md](TESTLOG.md).

---

## Repo layout

```
collection/         data collection layer
  backfill_kalshi.py          REST backfill: Kalshi market metadata + OHLC candles
  fetch_kalshi_trades.py      REST backfill: trade-level tape -> kalshi_trades.json
  fetch_binance_1s.py         Binance 1s kline fetch (recent window) -> binance_1s_*.json
  fetch_binance_history.py    Binance 1m history -> binance_klines_*.json
  fetch_settlements.py        Settlement results -> settlements.json
  ws_kalshi.py                Live Kalshi WebSocket feed (ticks -> data/kalshi_*.jsonl)
  ws_binance.py               Live Binance WebSocket feed (spot prices)
  kalshi_live.py              Live trading skeleton (order placement, not yet active)
  test_kalshi.py              API connectivity smoke test
  start_session.sh / stop_session.sh   Session management for live data collection

features/           feature extraction layer (inputs -> flat CSVs for test scripts)
  build_features_backfill.py  Kalshi candle data -> windows_backfill.csv (15,548 windows)
  build_features.py           Live/incremental variant of above
  bucket_kalshi_trades.py     Trade tape -> 5s buckets -> kalshi_trades_5s.json
  stream_process_trades.py    Streaming variant: trade tape -> 5s buckets + fixedtime features
  h1_grid_extract.py          Trade-level features for 25-spec grid -> h1_comprehensive_features.csv
  h1_avgN_extract.py          First-N-trades features -> h1_avgN_features.csv

tests/              hypothesis tests (reported results in TESTLOG.md)
  h1_strategy_backtest.py     TESTLOG #1: pooled threshold-sweep profit backtest
  h5_per_series_regression.py TESTLOG #2: per-series linear + logistic battery on sig/log_speed
  h6_calibration_signal_test.py TESTLOG #3: calibration slope vs. 1, OOS Brier improvement
  h3_h4_leadlag.py            TESTLOG #4: Kalshi vs Binance spot lead/lag + OOS forecast
  h1_grid_battery.py          Core grid search: 225 boundary×threshold combos, BH-FDR, AUC/Brier
  h1_augmented_model.py       Augmented model (sig + momentum + vol): Clark-West test, block bootstrap
  stress_test.py              Stress tests: leave-one-day-out CV, permutation test on momentum
  power_calc.py               Sample-size / power calculation for per-day effect
  momentum_sigtest.py         Momentum signal significance: correlation + AUC + permutation p-value
  archive/                    Superseded exploratory scripts (see header comment in each)

README.md           this file
TESTLOG.md          full test log with raw numbers, confound checks, open questions
learning_guide.md   self-study guide: every statistical method used and the 4 source papers
```

---

## How to run

All scripts resolve data-file paths relative to the **repo root** — run them from there:

```bash
# Step 1: collect (needs kalshi_key.key at repo root — not checked in)
python collection/backfill_kalshi.py          # -> backfill_markets.json, backfill_candles.json
python collection/fetch_kalshi_trades.py       # -> kalshi_trades.json
python collection/fetch_binance_1s.py          # -> binance_1s_BTC/ETH/SOLUSDT.json
python collection/fetch_binance_history.py     # -> binance_klines_BTC/ETH/SOLUSDT.json

# Step 2: build features
python features/build_features_backfill.py    # -> windows_backfill.csv
python features/bucket_kalshi_trades.py        # -> kalshi_trades_5s.json
python features/h1_grid_extract.py             # -> h1_comprehensive_features.csv
python features/h1_avgN_extract.py             # -> h1_avgN_features.csv

# Step 3: run tests
python tests/h1_strategy_backtest.py
python tests/h5_per_series_regression.py
python tests/h6_calibration_signal_test.py
python tests/h3_h4_leadlag.py
```

Data files (`*.json`, `*.csv`, `data/`) are gitignored. They live at the repo root during a
session and are never committed. The `kalshi_key.key` RSA private key is also gitignored.

---

## Dataset

15,548 windows across BTC, ETH, SOL 15-min contracts on Kalshi (2026-05-02 to 2026-06-27).
Binance 1s spot data covers only 2026-06-17 to 06-26 (~9 of 56 days) — the lead/lag test
(TESTLOG #4) is restricted to this 1,493-window overlap. Closing the coverage gap requires
a new multi-day Binance 1s backfill (not started).

---

## What was archived and why

`tests/archive/` holds 5 scripts that were exploratory dead ends or were superseded by
better versions. Each has a one-line header explaining what replaced it:

- `analyze_h1_h2.py` — 1-min kline feature extraction, replaced by trade-level extraction
  in `features/h1_grid_extract.py` (finer resolution, no stale-price floor hack needed)
- `analyze_h2_subminute.py` — sub-minute CSV extraction step that was never consumed by any
  reported test; the lead/lag analysis (`h3_h4_leadlag.py`) reads 5s buckets directly
- `h1_logistic.py` — single-boundary logistic fit predating the full 225-combo grid battery
- `h1_avgN_analysis.py` — per-N exploratory pass predating `h5_per_series_regression.py`
  (which adds proper chronological split and BH correction)
- `momentum_demo.py` — quick demo without significance tests, replaced by `momentum_sigtest.py`
