# Test log

Running record of every statistical test performed on the Kalshi 15-min BTC/ETH/SOL
dataset. Temp repo -- code + this log only (no data blobs, see .gitignore). Gets folded
into the final project repo once the methodology settles.

Dataset as of this log: 15,548 windows in `windows_backfill.csv` (2026-05-02 to
2026-06-27), trade-level data in `kalshi_trades.json` -> bucketed 5s in
`kalshi_trades_5s.json`, features in `h1_comprehensive_features.csv` (via
`h1_grid_extract.py`). Binance 1s spot data only covers 2026-06-17 to 06-26 (partial,
~9 of 56 days -- see entry below, fix in progress).

---

## 1. `h1_strategy_backtest.py` -- profit backtest, threshold sweep

**Question:** can you profit by betting yes/no when the early-window price deviates
from 0.50 by more than some threshold?

**Method:** 25 boundary specs (time: 5,10,15,20,30,45,60,90,120,180,300,450,600s;
trade-count: 5,10,25,50,75,100,150,200,300,500,750,1000 trades) x 9 thresholds
(0.00-0.40) = 225 combos. Spec+threshold selected on TRAIN (earliest 70%) by costed
mean profit, n>=40 required. Reported once on untouched HOLDOUT (latest 30%),
block-bootstrap-by-day 95% CI.

**Result:** best train candidate = "first 50 trades", threshold=0.25 (train mean
profit costed = +0.0346/contract, n=301, t=2.79). On holdout: n=88,
mean_profit_costed = **-0.0026**, se=0.0367, t=0.20, 95% CI **[-0.0856, 0.0848]**
over 16 holdout days.

**Verdict: NO edge survives.** All 225 combos pooled (BTC+ETH+SOL together, not
split by series) -- none survived holdout. This was the all-series-pooled version;
superseded by the per-series tests below.

---

## 2. `h5_per_series_regression.py` -- per-series linear/logistic battery on `sig`

**Question:** does the early-window average price (`sig`) and early trading speed
(`log_speed`, the volume/intensity idea) predict the outcome, tested separately per
coin (not pooled)?

**Method:** data-derived early window per series so elapsed time is matched
(~60s median): BTC=first 1000 trades, ETH=first 50, SOL=first 10. Per series:
simple linear (y~sig), multiple linear (y~sig+log_speed), simple logistic,
multiple logistic. Chronological 70/30 split. HC1-robust SEs (linear), IRLS
logistic. BH-FDR correction across all 18 coefficient tests (sig + log_speed,
4 models x 3 series - wait, 6 per series x 3 = 18). Out-of-sample: AUC, Brier vs
base-rate, block-bootstrap-by-day CI.

**Result (all 3 series, all 4 model types):**
- BTC (n=5123): sig coef 1.34-1.34, p<.00001 every model. log_speed p=.49-.50 (null).
  OOS: AUC=.647, d(Brier)=.0158, CI [.0101,.0217] (17 days) -> survives.
- ETH (n=4976): sig coef 1.245, p<.00001 every model. log_speed p=.98-.99 (null).
  OOS: AUC=.644, d(Brier)=.0152, CI [.0099,.0207] (17 days) -> survives.
- SOL (n=4988): sig coef 1.19, p<.00001 every model. log_speed p=.26-.28 (null).
  OOS: AUC=.617, d(Brier)=.0103, CI [.0031,.0174] (17 days) -> survives.
- BH-FDR: 12/12 `sig` tests survive, 0/6 `log_speed` tests survive.

**Verdict:** `sig` is a real, statistically robust, out-of-sample-validated predictor
of outcome in all 3 series. NOT an exploitable edge -- mechanically expected
(price = market's probability estimate), and the cost of betting on it equals the
signal itself. `log_speed` (volume/intensity) adds nothing in any series.

---

## 3. `h6_calibration_signal_test.py` -- is the price systematically miscalibrated?

**Question:** is the linear-probability-model slope on `sig` significantly
different from 1 (market under/overconfident), and does correcting for it forecast
better out-of-sample (pure Brier score, no costs/bets)?

**Result:** slope significantly >1 in all 3 series in-sample (BTC 1.34 z=?,
ETH 1.25 z=3.17 p=.0016, SOL 1.19 z=2.36 p=.018) -- all 6 slope/intercept tests
BH-survive. But recalibrated-forecast Brier improvement on HOLDOUT: BTC CI
[-0.00049, 0.00300], ETH [-0.00064, 0.00272], SOL [-0.00221, 0.00086] -- all
include 0.

**Verdict:** in-sample miscalibration is real, does NOT generalize out-of-sample
in any series. Dead end as currently specified.

---

## 4. `h3_h4_leadlag.py` -- Kalshi vs Binance spot lead/lag

**Question:** does Kalshi price move before/after/with Binance spot price?

**Run 1 (before fixes):** n=262,768 pooled 5s obs across only **1,493 of 15,548
windows** (Binance 1s files only cover 2026-06-17 to 06-26, a 9-day slice of the
56-day window range -- ~90% of windows have no matching Binance data; this is a
real coverage gap, fix tracked below).

H3 (pooled OLS, cluster-robust by window): intercept p=.046; t-2,t-1 (Kalshi
lagging Binance) not significant; t+0 (contemporaneous) coef=35.9 t=25.8 p<.00001;
t+1 coef=76.2 t=23.2 p<.00001; t+2 coef=4.0 t=6.0 p<.00001 -- t+0,t+1,t+2 all
BH-survive.

H4 (OOS forecast): **crashed** -- IndexError, h4_train_X ended up empty (the
chronological train/test split was computed over ALL 15,548 windows by date, but
usable (Binance-matched) windows were apparently concentrated outside the train
range given the coverage gap).

**Known issues, fix in progress:**
1. Coverage gap (90% of windows have no Binance match) -- need either a Binance
   1s backfill covering the full 05-02 to 06-27 range, or to restrict/caveat the
   analysis to the 06-17 to 06-26 overlap.
2. Crash: train/test split must be computed over USABLE windows only, not all
   windows, and the code needs to handle an empty train/test set gracefully.
3. Methodology: the t+1/t+2 "Kalshi leads Binance" coefficients are suspect --
   likely confounded by (a) Binance's own short-horizon return autocorrelation
   (5s crypto returns aren't independent) and (b) Kalshi price staleness (thin
   markets forward-fill the last trade price across empty buckets, which can
   mechanically create a spurious lead). Needs an AR-baseline-controlled
   (Granger-style) test restricted to buckets with a genuine new Kalshi trade,
   not the forward-filled price.

*(this entry will be updated in place once the fix is run -- see commit history)*

---
