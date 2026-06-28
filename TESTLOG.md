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

**Run 2 (after fixes):**

Fix 1 (crash): train/test split is now computed over the 1,493 USABLE windows only
(in their own chronological order), not over all 15,548. Result: 1,045 train /
448 test windows, both non-empty. H4 no longer crashes.

Fix 2 (staleness): both H3's outcome (`kdelta_t`) and H4's forecast target
(`kdelta_t+1`) now require `traded=True` on that bucket (a genuine new Kalshi
trade, not a forward-filled stale carry). Excluded 19,767 stale buckets from H3,
11,883 more from H4.

**H3 after fix:** n=243,123 obs, same pattern as before, slightly *stronger*:
t+0 coef=38.1 t=24.7 p<.00001, t+1 coef=82.0 t=23.7 p<.00001, t+2 coef=4.3 t=5.3
p<.00001, all BH-survive; t-1/t-2 still null. **The staleness fix did NOT kill the
t+1 effect** -- ruling out forward-fill staleness as the explanation.

**H3b diagnostic (new) -- is it just Binance's own autocorrelation?**
Regressed `binance_ret_t+1 ~ binance_ret_t-2,t-1,t+0` on the same windows: t-1 coef
0.035 (p<.00001), t+0 coef 0.055 (p<.00001), R^2=0.0045. Binance's own near-term
returns ARE significantly autocorrelated, which is a real, available channel for
the H3 t+1 coefficient to be partly/wholly spurious (collinearity bleed-through
from the true t+0 effect) -- but R^2=0.0045 is small, so it's not obviously enough
*on its own* to fully explain a t+1 coefficient more than 2x the size of t+0.
Honest read: the t+1 "lead" is not fully explained by either confound tested so
far; likely some mix of real microstructure lag (Kalshi order processing/matching
delay) and residual autocorrelation bleed. Not chasing this further -- see verdict.

**H4 after fix (the test that actually matters -- can this be traded):**
n_test=72,596, out-of-sample R^2 = **-0.00005** (i.e. zero -- the model explains
nothing on genuinely future data). Costed backtest: mean P&L/trade = **-0.0099**,
i.e. directionally betting on this "predictive" relationship just bleeds the
assumed 0.01 round-trip cost, consistent with the predictions being indistinguishable
from noise.

**Verdict: the H3 association is statistically real (BH-survives, survives two
separate confound checks) but H4 confirms it carries ZERO out-of-sample forecasting
power.** Whatever is producing the t+1 coefficient (partial Binance autocorrelation
bleed + possibly genuine but tiny microstructure lag), it is not exploitable as a
trading signal. Consistent with every other test in this project: no edge here either.

**Still open, not fixed (lower priority, user flagged skepticism about this whole
avenue up front):** the 90%-missing Binance coverage gap. All numbers above are
restricted to the 1,493-window / 9-day overlap (2026-06-17 to 06-26) where Binance
data exists; the other ~47 days of the 56-day Kalshi range are entirely untested
here. Closing this would require a new Binance 1s backfill (a multi-day-spanning
candle fetch -- a separate, larger task, not started).

---
