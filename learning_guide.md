# Learning Guide: Kalshi 15-Min Crypto Strategy Research

This is a self-study map for everything used in the project so far: the four outside sources, and every statistical method built into the pipeline scripts. It's organized so you learn the *tools* before the *papers* — the papers assume you already know what a calibration slope or a Clark-West test is, so reading them first will feel like jargon soup. Read this guide top to bottom once, then go back and actually study each linked concept at your own pace.

---

## Part 1: Learn the tools first (suggested order)

These are listed in the order it makes sense to learn them, not the order they appear in the scripts. Each entry says what it is, why it's in this project specifically, and what file to go look at once you understand the idea — seeing your own data plugged into the formula is the fastest way to make it click.

### 1. Logistic regression (fit via IRLS)
**What it is:** A model for predicting a probability (between 0 and 1) from one or more input features, instead of predicting a number directly like ordinary regression. "IRLS" (iteratively reweighted least squares) is just the specific numerical recipe used to fit it — it's mathematically equivalent to Newton's method applied to the logistic likelihood.
**Why it's here:** Every signal test in this project — does early price predict the outcome? does momentum add anything? — boils down to fitting a logistic regression and checking if a coefficient is meaningfully different from zero.
**Where to see it:** `h1_grid_battery.py`, `h1_augmented_model.py`, `stress_test.py` all contain a hand-written `fit()` / `logistic_irls()` function. Worth reading line by line once you know the concept — there's no library here, it's the actual matrix algebra.
**Good starting point to learn it properly:** any intro stats/ML resource on "logistic regression" and "maximum likelihood estimation" — you want to understand *why* you can't just use linear regression on a 0/1 outcome, and what the sigmoid function is doing.

### 2. AUC (area under the ROC curve)
**What it is:** A single number, 0.5 to 1.0, measuring how well a model ranks positive outcomes above negative outcomes. 0.5 = no better than random; 1.0 = perfect separation. It's computed here via the Mann-Whitney U / rank-sum formula, not by actually drawing a curve.
**Why it's here:** It's the main "does this signal discriminate at all" metric used throughout the grid search.
**Where to see it:** `auc_score()` function, appears identically in `h1_grid_battery.py`, `stress_test.py`.

### 3. Brier score
**What it is:** Mean squared error applied to probability forecasts — `(predicted_probability - actual_outcome)^2`, averaged. Lower is better. Unlike AUC, it cares about *calibration*, not just ranking — a model that's well-ranked but always overconfident will have a bad Brier score even with a great AUC.
**Why it's here:** It's the metric used for "did adding momentum actually improve the forecast," because that question is about probability quality, not just ranking.
**Where to see it:** `brier()` function in `stress_test.py`, `power_calc.py`.

### 4. McFadden pseudo-R²
**What it is:** A rough analog of R² for logistic regression (ordinary R² doesn't apply to binary outcomes). It's a likelihood-ratio-based measure of how much better the model fits than a constant-probability baseline.
**Why it's here:** Used as a quick "how much explanatory power does this feature have" sanity number alongside AUC.
**Note:** values are typically much smaller than OLS R² even for genuinely useful models — 0.04-0.05 is not "bad" in this context, it's normal for a single weak-to-moderate predictor.

### 5. Cox calibration regression
**What it is:** Take your model's predicted probabilities, convert both predictions and a price/odds into log-odds (logit) space, and regress actual outcome on logit(prediction). The resulting slope tells you about calibration: slope = 1 is perfect, slope > 1 means the original probabilities were too close to 50% (underconfident / compressed), slope < 1 means they were too extreme (overconfident).
**Why it's here:** This is the single most important diagnostic in the whole project — it's how you discovered that Kalshi's quoted price gets *more* underconfident as the window approaches settlement, which is the calibration-slope-inversion finding.
**Where to see it:** embedded in `h1_grid_battery.py`'s per-boundary diagnostics, and explicitly cited against the literature in `h1_augmented_model.py`'s closing printout.
**Read this concept carefully** — it's also exactly Equation 1 in the "Decomposing Crowd Wisdom" paper (Part 2 below), so understanding it here means that paper's entire Section 3 becomes easy to read.

### 6. Benjamini-Hochberg (BH) false discovery rate correction
**What it is:** When you test many hypotheses at once (e.g., 13 different time boundaries in the grid search), some will look "significant" by pure chance even if nothing is real — this is the multiple-testing problem. BH-FDR adjusts your significance threshold across the whole batch of tests so your false-positive rate stays controlled, without being as harsh as a simple Bonferroni correction.
**Why it's here:** The boundary grid search and the augmented-model spec comparison both test many candidate features/cutoffs at once — without this correction, "we found a significant boundary" is nearly meaningless.
**Where to see it:** applied across the grid in `h1_grid_battery.py` and across the 5 candidate specs in `h1_augmented_model.py`.

### 7. Clark-West (2007) test
**What it is:** A statistical test specifically built for comparing out-of-sample forecast accuracy between two models *when one model is nested inside the other* (i.e., the bigger model contains all of the smaller model's inputs plus more). Naively comparing out-of-sample errors between nested models is biased toward the smaller model, because the extra parameters in the bigger model add estimation noise even when they have real predictive value. Clark-West corrects for that bias.
**Why it's here:** This is exactly the situation in `h1_augmented_model.py` — comparing "base price alone" against "base price + momentum," where the second model nests the first.
**Where to see it:** `clark_west()` function in `h1_augmented_model.py`, explicitly borrowed from this paper's methodology, and also the headline OOS test in the Mohanty/Krishnamachari paper (Part 2, source #1).

### 8. Block bootstrap by day
**What it is:** Bootstrapping (resampling with replacement to build a confidence interval) normally assumes each resampled unit is independent. Individual 15-minute windows within the same day are *not* independent of each other — they all share the same underlying spot-price path. Block bootstrap fixes this by resampling whole *days* at a time rather than individual windows, so the resampled data respects the real dependency structure.
**Why it's here:** Used to put a defensible confidence interval on the Brier-score improvement from momentum, given that you only have a handful of distinct calendar days.
**Where to see it:** `N_BOOTSTRAP = 3000` block-bootstrap loop in `h1_augmented_model.py`.

### 9. Leave-one-day-out cross-validation (LODO-CV)
**What it is:** Train the model on all days except one, test on the held-out day, repeat for every day. This is the most honest test of generalization available with limited data — it directly answers "if tomorrow looked like a day the model has never seen, would the signal still work?"
**Why it's here:** Stress test #1 in `stress_test.py` — checks whether the momentum signal's improvement holds up day-by-day rather than being an artifact of one unusually favorable day in the sample.

### 10. Permutation test
**What it is:** To get a p-value with zero distributional assumptions, randomly shuffle the outcome labels thousands of times, recompute your statistic (e.g., correlation) each time, and see how often the shuffled version is as extreme as your real, unshuffled result. If only 0.5% of random shuffles beat your real result, your empirical p-value is 0.005 — no normality assumption needed anywhere.
**Why it's here:** Stress test #2 in `stress_test.py` — a model-free sanity check on the momentum-outcome correlation, useful precisely because it doesn't rely on the same assumptions the logistic regression does.

### 11. Power / sample-size calculation
**What it is:** Given an estimated effect size and its variability, calculate how much data (here, how many calendar days) you'd need before a confidence interval around that effect reliably excludes zero. The formula used here, `n = (z * sd / mean)^2`, is the standard one-sample-mean power calculation applied at the day level.
**Why it's here:** This is the single number driving Task #13 — `power_calc.py` says you need roughly 25-30 days of data before you can trust that the backtested per-bet edge isn't just noise, which is exactly why the dataset expansion is happening right now.

### 12. Kelly criterion / quarter-Kelly position sizing
**What it is:** A formula for the bet size that maximizes long-run growth rate of capital given an edge and odds. Full Kelly is mathematically optimal but extremely sensitive to overestimating your edge (a small estimation error can lead to oversized bets and large drawdowns); quarter-Kelly (betting 1/4 of what full Kelly recommends) trades some growth rate for a lot of robustness to that estimation error.
**Why it's here:** This is the position-sizing piece that's currently missing from the project — it's how a validated statistical edge actually turns into a bet size, rather than staying a number in a backtest.
**Where it comes from:** explicitly recommended in the PolySwarm paper (Part 2, source #3).

### 13. Digital option pricing / lognormal hitting probability
**What it is:** A Kalshi 15-minute binary contract is mathematically a "cash-or-nothing digital option" — it pays $1 if spot price is above the strike at close, $0 otherwise. Under a simple assumption that spot price follows a lognormal random walk over the remaining time, there's a closed-form formula for the true probability of finishing above the strike, using only the current spot price, the strike, time remaining, and a volatility estimate.
**Why it's here:** This is the proposed second, independent signal — a "fair value" probability computed from first principles (physics of the spot price), to compare against Kalshi's quoted price, separate from the momentum/microstructure signal already validated.
**Where it comes from:** the derivatives-pricing framing in the SSRN paper (Part 2, source #4).

### 14. KL / JS divergence for mispricing detection
**What it is:** Kullback-Leibler and Jensen-Shannon divergence are ways to measure how different two probability distributions are. Applied here, they'd measure the gap between a model-implied probability distribution and the market's implied distribution, flagging moments where the two disagree enough to be a tradeable signal.
**Why it's here:** Used in the PolySwarm paper to detect cross-market inefficiencies; conceptually the same tool you'd use to systematically flag "model and market disagree a lot right now" once you have a second signal source.

---

## Part 2: The four outside sources, in reading order

Read these *after* Part 1 — they'll go faster once the vocabulary above is familiar.

**1. "Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets" (arXiv:2602.19520)**
Read this first. It's the most directly relevant paper to what we're doing, and it's the one that defines calibration slope (Part 1, #5) in the way our scripts use it. Focus on Section 2.3 (how calibration is measured), the three "stylised facts" in Section 3 (especially Table 3, the slope-by-time-to-resolution table — this is the table our own within-window finding sits next to), and Section 4.3 on the trade-size/scale effect. You can skim the Bayesian decomposition model in Section 5 — useful context but not essential to use the result.

**2. "Do Prediction Markets Forecast Cryptocurrency Volatility? Evidence from Kalshi Macro Contracts" (arXiv:2604.01431)**
Read this second. It's not about the same markets (daily macro contracts, not 15-minute crypto), but it's the methodological template — Clark-West test, BH-FDR correction, HAC standard errors — that our own augmented-model script directly copies. Focus on Section 3 (Empirical Strategy: this is basically a worked example of everything in Part 1, #7 and #6) and Section 5 (out-of-sample results) to see what a "real but modest" out-of-sample edge looks like in practice (2-4% MSFE improvement) — useful as a calibration of expectations.

**3. SSRN paper (papers.ssrn.com/sol3/papers.cfm?abstract_id=6748186)**
Read this third. It reframes a binary prediction contract as a cash-or-nothing digital option and discusses the variance risk premium angle — this is the conceptual basis for Part 1, #13, the proposed second signal.

**4. "PolySwarm" — multi-agent LLM swarm trading paper (arXiv:2604.03888)**
Read this last. It's the most different in kind — an engineering/systems paper about an LLM-agent swarm trading Polymarket, with Bayesian aggregation, Kelly sizing, and KL/JS divergence-based mispricing detection. Most of the architecture (the LLM swarm itself) isn't directly relevant to what we're building, but the position-sizing and mispricing-detection ideas (Part 1, #12 and #14) are worth extracting.

---

## A note on the fifth source

There was also a Reddit thread referenced earlier in this project that I was never able to recover (it's not search-indexed and was never pasted into the conversation). If you find the actual thread or want to paste the relevant parts, it's worth adding here — the secondhand advice you already shared (about reverse-engineering visible-wallet strategies on Polymarket) doesn't port directly to Kalshi since Kalshi's public trade feed is anonymized — no account-level data the way an on-chain Polymarket wallet gives you — so the most useful related exercise here is closer to "find the footprint of a strategy in the aggregate tape" rather than "follow a specific trader."
