import csv, math, datetime, random

WINDOWS_CSV = "windows_backfill.csv"
GRID_CSV = "h1_comprehensive_features.csv"
TRAIN_FRACTION = 0.70

# Same data-derived early windows as h5_per_series_regression.py (median elapsed ~60s
# across all three series despite very different trade volumes) -- see that file's
# header comment for how these were picked.
SERIES_N = {"KXBTC15M": 1000, "KXETH15M": 50, "KXSOL15M": 10}
SERIES_LABEL = {"KXBTC15M": "BTC", "KXETH15M": "ETH", "KXSOL15M": "SOL"}

# THIS SCRIPT TESTS ONE THING ONLY: is the market's early-window price (`sig`)
# MISCALIBRATED relative to the true outcome -- i.e. is the linear-probability-model
# slope on sig significantly different from 1 (and/or intercept significantly
# different from 0)? If slope > 1, the price understates how extreme the true
# probability actually is (market is underconfident this early). That gap, if real
# and if it generalizes out-of-sample, is the only legitimate "signal" left after
# h5 showed raw sig and early trading speed give you nothing tradeable beyond what
# you already pay for. No costs, no bets, no P&L anywhere in this file -- purely:
# is the miscalibration statistically real and does it predict the outcome BETTER
# than the raw price does, out of sample, in pure forecast-accuracy terms (Brier score).

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def mean(xs):
    return sum(xs) / len(xs)

def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))

def mat_transpose(A):
    return [list(row) for row in zip(*A)]

def mat_mul(A, B):
    ra, ca = len(A), len(A[0])
    cb = len(B[0])
    out = [[0.0] * cb for _ in range(ra)]
    for i in range(ra):
        Ai = A[i]
        for k in range(ca):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            outi = out[i]
            for j in range(cb):
                outi[j] += a * Bk[j]
    return out

def mat_inverse(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise ValueError("singular matrix")
        M[col], M[pivot_row] = M[pivot_row], M[col]
        pivot = M[col][col]
        M[col] = [x / pivot for x in M[col]]
        for r in range(n):
            if r != col:
                factor = M[r][col]
                if factor != 0.0:
                    M[r] = [M[r][k] - factor * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]

def ols_fit(X, y):
    Xt = mat_transpose(X)
    XtX = mat_mul(Xt, X)
    XtX_inv = mat_inverse(XtX)
    Xty = mat_mul(Xt, [[v] for v in y])
    beta_mat = mat_mul(XtX_inv, Xty)
    beta = [row[0] for row in beta_mat]
    return beta, XtX_inv

def ols_robust_se(X, y, beta, XtX_inv):
    n, k = len(X), len(X[0])
    resid = [y[i] - sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]
    meat = [[0.0] * k for _ in range(k)]
    for i in range(n):
        u2 = resid[i] ** 2
        Xi = X[i]
        for a in range(k):
            xa = Xi[a] * u2
            if xa == 0.0:
                continue
            for b in range(k):
                meat[a][b] += xa * Xi[b]
    mid = mat_mul(XtX_inv, mat_mul(meat, XtX_inv))
    dof_corr = n / max(n - k, 1)
    se = [math.sqrt(max(mid[j][j], 0.0) * dof_corr) for j in range(k)]
    return resid, se

def norm_p(z):
    if math.isnan(z):
        return float("nan")
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

def chi2_sf_wilson_hilferty(x, k):
    if x <= 0:
        return 1.0
    h = 2.0 / (9.0 * k)
    z = ((x / k) ** (1.0 / 3.0) - (1 - h)) / math.sqrt(h)
    return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))

def brier(p, y):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)

def bh_correction(pvals, alpha=0.05):
    idx_sorted = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    survives = [False] * m
    max_k = -1
    for rank, i in enumerate(idx_sorted, start=1):
        if pvals[i] <= (rank / m) * alpha:
            max_k = rank
    if max_k >= 0:
        for rank, i in enumerate(idx_sorted, start=1):
            if rank <= max_k:
                survives[i] = True
    return survives

def block_bootstrap_diff(days_per_obs, a_per_obs, b_per_obs, n_boot=3000, seed=42):
    by_day = {}
    for d, a, b in zip(days_per_obs, a_per_obs, b_per_obs):
        by_day.setdefault(d, []).append(a - b)
    days = sorted(by_day.keys())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample_days = [rng.choice(days) for _ in range(len(days))]
        pool = []
        for d in sample_days:
            pool.extend(by_day[d])
        if len(pool) < 5:
            continue
        boots.append(mean(pool))
    boots.sort()
    if len(boots) < 20:
        return None
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return lo, hi, len(days)

def load_series_data(series_name):
    N = SERIES_N[series_name]
    with open(WINDOWS_CSV) as f:
        wrows = {r["ticker"]: r for r in csv.DictReader(f)
                 if r["covered_from_open"] == "True" and r["covered_to_close"] == "True" and r["series"] == series_name}
    with open(GRID_CSV) as f:
        grows = {r["ticker"]: r for r in csv.DictReader(f) if r["series"] == series_name}
    rows = []
    for ticker, wr in wrows.items():
        g = grows.get(ticker)
        if not g:
            continue
        sig_raw = g.get(f"avg_price_n{N}", "")
        y_raw = g.get("y", "")
        if sig_raw == "" or y_raw == "":
            continue
        sig = float(sig_raw)
        if sig <= 0.001 or sig >= 0.999:
            continue
        rows.append({"ticker": ticker, "y": int(y_raw), "sig": sig,
                     "open_ts": parse_time(wr["open_time"]).timestamp(), "day": wr["open_time"][:10]})
    rows.sort(key=lambda r: r["open_ts"])
    return rows

def chrono_split(rows):
    n_train = int(len(rows) * TRAIN_FRACTION)
    return rows[:n_train], rows[n_train:]

def run_series(series_name, global_pvals):
    N = SERIES_N[series_name]
    label = SERIES_LABEL[series_name]
    rows = load_series_data(series_name)
    print(f"\n{'='*90}")
    print(f"=== {label} ({series_name}) -- early window = first {N} trades, n={len(rows)} usable windows ===")
    print(f"{'='*90}")
    if len(rows) < 200:
        print("Not enough rows -- skipping.")
        return

    train, test = chrono_split(rows)
    print(f"train n={len(train)}, holdout n={len(test)}\n")

    y_tr = [r["y"] for r in train]
    X_tr = [[1.0, r["sig"]] for r in train]
    beta, inv = ols_fit(X_tr, y_tr)
    resid, se = ols_robust_se(X_tr, y_tr, beta, inv)
    b0, b1 = beta
    se0, se1 = se

    z_slope = (b1 - 1.0) / se1
    p_slope = norm_p(z_slope)
    z_int = b0 / se0
    p_int = norm_p(z_int)

    # joint test: is sig already perfectly calibrated (a=0, b=1), i.e. phat=sig exactly?
    sse_unrestricted = sum(r ** 2 for r in resid)
    sse_restricted = sum((y - r["sig"]) ** 2 for y, r in zip(y_tr, train))
    n, k = len(train), 2
    F = ((sse_restricted - sse_unrestricted) / 2) / (sse_unrestricted / (n - k))
    p_joint = chi2_sf_wilson_hilferty(F * 2, 2) if F > 0 else 1.0

    print(f"Fitted on TRAIN: y = {b0:.4f} + {b1:.4f} * sig    (perfect calibration would be y = 0 + 1*sig)")
    print(f"  slope vs 1:      z={z_slope:7.2f}  p={p_slope:.5f}   ({'slope > 1 -> market UNDERCONFIDENT' if b1 > 1 else 'slope < 1 -> market OVERCONFIDENT'})")
    print(f"  intercept vs 0:  z={z_int:7.2f}  p={p_int:.5f}")
    print(f"  joint test (a=0,b=1 simultaneously): F~{F:.2f}, p~{p_joint:.5f}  (tests whether sig is ALREADY perfectly calibrated)")

    global_pvals.append((f"{label}: slope!=1", p_slope))
    global_pvals.append((f"{label}: intercept!=0", p_int))

    # ---- out-of-sample: does the TRAIN-fitted recalibration actually forecast better than raw sig, on holdout? ----
    y_te = [r["y"] for r in test]
    sig_te = [r["sig"] for r in test]
    phat_te = [min(max(b0 + b1 * s, 0.001), 0.999) for s in sig_te]
    days_te = [r["day"] for r in test]

    brier_raw = brier(sig_te, y_te)
    brier_recal = brier(phat_te, y_te)
    d_brier = brier_raw - brier_recal  # positive = recalibrated forecast beat raw price OOS

    sq_err_raw = [(s - y) ** 2 for s, y in zip(sig_te, y_te)]
    sq_err_recal = [(p - y) ** 2 for p, y in zip(phat_te, y_te)]
    bb = block_bootstrap_diff(days_te, sq_err_raw, sq_err_recal)

    print(f"\nOut-of-sample forecast accuracy (holdout n={len(test)}, pure Brier score, no costs/bets anywhere):")
    print(f"  Brier(raw price as forecast)          = {brier_raw:.5f}")
    print(f"  Brier(recalibrated forecast)           = {brier_recal:.5f}")
    print(f"  d(Brier) = raw - recalibrated          = {d_brier:.5f}  (positive = recalibration is a BETTER forecast than the raw price, out of sample)")
    if bb:
        lo, hi, ndays = bb
        survives = lo > 0 or hi < 0
        print(f"  Block-bootstrap-by-day 95% CI on d(Brier): [{lo:.5f}, {hi:.5f}] over {ndays} holdout days")
        print(f"  -> Recalibration signal {'IS REAL, survives out-of-sample' if survives else 'DOES NOT clearly survive out-of-sample'} "
              f"(CI {'excludes' if survives else 'includes'} 0)")
    else:
        print("  Not enough holdout days for a stable bootstrap CI.")

def main():
    print("Calibration-miscalibration test: is the market's early-window price slope != 1 relative to true outcome?")
    print("Tested per series (not pooled), no trading strategy, no costs -- pure forecast-accuracy statistics.\n")
    for s, n in SERIES_N.items():
        print(f"  {SERIES_LABEL[s]}: first {n} trades")

    global_pvals = []
    for series_name in ["KXBTC15M", "KXETH15M", "KXSOL15M"]:
        run_series(series_name, global_pvals)

    print(f"\n{'='*90}")
    print("=== FINAL: BH-FDR correction across all slope/intercept tests above ===")
    print(f"{'='*90}")
    labels = [lbl for lbl, _ in global_pvals]
    pvals = [p for _, p in global_pvals]
    survives = bh_correction(pvals)
    print(f"{'test':25s} {'p-value':>10s} {'BH-survives':>12s}")
    for lbl, p, surv in zip(labels, pvals, survives):
        print(f"{lbl:25s} {p:10.5f} {'yes' if surv else 'no':>12s}")
    print(f"\n{sum(survives)}/{len(global_pvals)} survive correction.")
    print("\nRead this together with the per-series out-of-sample Brier results above: a slope significantly != 1 in-sample")
    print("only counts as a REAL signal if the recalibration also beats raw price OUT OF SAMPLE with a CI excluding 0.")
    print("If both hold, the miscalibration is real and durable -- what to DO about it (sizing, thresholds, costs) is a")
    print("separate, later question, deliberately not addressed in this file.")

if __name__ == "__main__":
    main()
