import csv, json, math, datetime, random

WINDOWS_CSV = "windows_backfill.csv"
AVGN_CSV = "h1_avgN_features.csv"
H1H2_CSV = "h1_h2_features.csv"
N_GRID = [10, 25, 50, 100, 250, 500]
PRIMARY_N = 100
TRAIN_FRACTION = 0.7
N_BOOTSTRAP = 2000

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)

def mat_inverse(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        pr = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pr][col]) < 1e-12:
            raise ValueError("singular")
        M[col], M[pr] = M[pr], M[col]
        piv = M[col][col]
        M[col] = [x / piv for x in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                if f != 0:
                    M[r] = [M[r][k] - f * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]

def logistic_irls(X, y, max_iter=100, tol=1e-10):
    n, k = len(X), len(X[0])
    beta = [0.0] * k
    for _ in range(max_iter):
        eta = [sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]
        mu = [sigmoid(e) for e in eta]
        w = [max(m * (1 - m), 1e-10) for m in mu]
        grad = [sum(X[i][j] * (y[i] - mu[i]) for i in range(n)) for j in range(k)]
        XtWX = [[0.0] * k for _ in range(k)]
        for i in range(n):
            wi = w[i]
            Xi = X[i]
            for a in range(k):
                xa = Xi[a] * wi
                if xa == 0:
                    continue
                for b in range(k):
                    XtWX[a][b] += xa * Xi[b]
        inv = mat_inverse(XtWX)
        delta = [sum(inv[a][b] * grad[b] for b in range(k)) for a in range(k)]
        beta = [beta[j] + delta[j] for j in range(k)]
        if max(abs(d) for d in delta) < tol:
            break
    eta = [sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]
    mu = [sigmoid(e) for e in eta]
    w = [max(m * (1 - m), 1e-10) for m in mu]
    XtWX = [[0.0] * k for _ in range(k)]
    for i in range(n):
        wi = w[i]
        Xi = X[i]
        for a in range(k):
            xa = Xi[a] * wi
            for b in range(k):
                XtWX[a][b] += xa * Xi[b]
    cov = mat_inverse(XtWX)
    se = [math.sqrt(max(cov[j][j], 0.0)) for j in range(k)]
    ll = sum(y[i] * math.log(max(mu[i], 1e-15)) + (1 - y[i]) * math.log(max(1 - mu[i], 1e-15)) for i in range(n))
    return beta, se, ll, mu

def auc_score(p, y):
    n = len(p)
    combined = sorted(zip(p, y))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for kx in range(i, j):
            ranks[kx] = avg_rank
        i = j
    n_pos = sum(y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = sum(ranks[i] for i in range(n) if combined[i][1] == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def brier(p, y):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)

def main():
    with open(WINDOWS_CSV) as f:
        wrows = {r["ticker"]: r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"}
    with open(AVGN_CSV) as f:
        avgn = {r["ticker"]: r for r in csv.DictReader(f)}
    with open(H1H2_CSV) as f:
        baseline = {r["ticker"]: r for r in csv.DictReader(f)}

    data = []
    for ticker, wr in wrows.items():
        a = avgn.get(ticker)
        b = baseline.get(ticker)
        if not a or not b:
            continue
        open_ts = parse_time(wr["open_time"]).timestamp()
        day = wr["open_time"][:10]
        rec = {
            "ticker": ticker, "series": wr["series"], "open_ts": open_ts, "day": day,
            "y": int(a["y"]), "n_trades_total": int(a["n_trades_total"]),
            "early_price_baseline": float(b["early_price"]),
            "sigma_annual": float(b["sigma_annual"]),
        }
        for N in N_GRID:
            v = a.get(f"avg_price_{N}", "")
            rec[f"avg_price_{N}"] = float(v) if v != "" else None
        data.append(rec)

    data.sort(key=lambda r: r["open_ts"])
    n_total = len(data)
    print(f"Total usable windows: {n_total}", flush=True)

    is_eth = lambda r: 1.0 if r["series"] == "KXETH15M" else 0.0
    is_sol = lambda r: 1.0 if r["series"] == "KXSOL15M" else 0.0

    print("\n=== Sensitivity curve: simple logistic y ~ avg_price_N, across N ===", flush=True)
    print(f"{'N':>5s} {'n':>6s} {'coef':>9s} {'se':>9s} {'z':>7s} {'p':>9s} {'pseudoR2':>9s} {'AUC':>7s}", flush=True)
    for N in N_GRID:
        sub = [r for r in data if r[f"avg_price_{N}"] is not None]
        n = len(sub)
        p = [r[f"avg_price_{N}"] for r in sub]
        y = [r["y"] for r in sub]
        X = [[1.0, pi] for pi in p]
        beta, se, ll, mu = logistic_irls(X, y)
        ybar = sum(y) / n
        ll_null = n * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))
        pr2 = 1 - ll / ll_null
        a_ = auc_score(p, y)
        z = beta[1] / se[1] if se[1] > 0 else float("nan")
        pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        print(f"{N:5d} {n:6d} {beta[1]:9.4f} {se[1]:9.4f} {z:7.2f} {pval:9.4f} {pr2:9.4f} {a_:7.4f}", flush=True)

    N = PRIMARY_N
    sub = [r for r in data if r[f"avg_price_{N}"] is not None]
    n = len(sub)
    p = [r[f"avg_price_{N}"] for r in sub]
    y = [r["y"] for r in sub]
    sigma = [r["sigma_annual"] for r in sub]
    eth = [is_eth(r) for r in sub]
    sol = [is_sol(r) for r in sub]
    base_p = [r["early_price_baseline"] for r in sub]

    print(f"\n=== PRIMARY: N={N} (n={n}) ===", flush=True)

    X1 = [[1.0, pi] for pi in p]
    beta1, se1, ll1, mu1 = logistic_irls(X1, y)
    ybar = sum(y) / n
    ll_null = n * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))
    pr2_1 = 1 - ll1 / ll_null
    auc1 = auc_score(p, y)
    print(f"Model 1: y~avg_price_{N}: coef={beta1[1]:.4f} se={se1[1]:.4f} z={beta1[1]/se1[1]:.2f} pseudoR2={pr2_1:.4f} AUC={auc1:.4f}", flush=True)

    X2 = [[1.0, p[i], eth[i], sol[i], sigma[i]] for i in range(n)]
    beta2, se2, ll2, mu2 = logistic_irls(X2, y)
    pr2_2 = 1 - ll2 / ll_null
    z_eth = beta2[2] / se2[2] if se2[2] > 0 else float("nan")
    z_sol = beta2[3] / se2[3] if se2[3] > 0 else float("nan")
    p_eth = 2 * (1 - 0.5 * (1 + math.erf(abs(z_eth) / math.sqrt(2))))
    p_sol = 2 * (1 - 0.5 * (1 + math.erf(abs(z_sol) / math.sqrt(2))))
    print(f"Model 2 (+series+sigma): coef_avgN={beta2[1]:.4f} se={se2[1]:.4f} z={beta2[1]/se2[1]:.2f} "
          f"coef_ETH={beta2[2]:.4f}(p={p_eth:.3f}) coef_SOL={beta2[3]:.4f}(p={p_sol:.3f}) pseudoR2={pr2_2:.4f}", flush=True)

    eps = 1e-4
    logit_p = [math.log(min(max(pi, eps), 1 - eps) / (1 - min(max(pi, eps), 1 - eps))) for pi in p]
    Xc = [[1.0, lp] for lp in logit_p]
    betac, sec, llc, muc = logistic_irls(Xc, y)
    print(f"Calibration (Cox): intercept={betac[0]:.4f}(se={sec[0]:.4f}) slope={betac[1]:.4f}(se={sec[1]:.4f}) "
          "[well-calibrated would be intercept~0, slope~1]", flush=True)

    brier_avgN = [(pi - yi) ** 2 for pi, yi in zip(p, y)]
    brier_base = [(pi - yi) ** 2 for pi, yi in zip(base_p, y)]
    diffs = [a_ - b_ for a_, b_ in zip(brier_avgN, brier_base)]
    mean_diff = sum(diffs) / n
    rng = random.Random(42)
    boot_diffs = []
    idxs = list(range(n))
    for _ in range(N_BOOTSTRAP):
        sample = [rng.choice(idxs) for _ in range(n)]
        boot_diffs.append(sum(diffs[i] for i in sample) / n)
    boot_diffs.sort()
    lo = boot_diffs[int(0.025 * N_BOOTSTRAP)]
    hi = boot_diffs[int(0.975 * N_BOOTSTRAP)]
    print(f"\nPaired Brier comparison (avg_price_{N} vs single-first-candle baseline):", flush=True)
    print(f"  mean Brier(avgN)={sum(brier_avgN)/n:.4f}  mean Brier(baseline)={sum(brier_base)/n:.4f}", flush=True)
    print(f"  mean diff (avgN - baseline): {mean_diff:.5f}, 95% bootstrap CI: [{lo:.5f}, {hi:.5f}] "
          "(negative & CI excludes 0 => averaging meaningfully helps)", flush=True)

    days = sorted(set(r["day"] for r in sub))
    by_day = {d: [] for d in days}
    for i, r in enumerate(sub):
        by_day[r["day"]].append(i)
    print(f"\nBlock bootstrap by day ({len(days)} day-blocks: {days}) for avg_price_{N} coefficient CI...", flush=True)
    boot_coefs = []
    for _ in range(500):
        sample_days = [rng.choice(days) for _ in range(len(days))]
        idx_pool = []
        for d in sample_days:
            idx_pool.extend(by_day[d])
        if len(idx_pool) < 10:
            continue
        yb = [y[i] for i in idx_pool]
        if sum(yb) == 0 or sum(yb) == len(yb):
            continue
        Xb = [[1.0, p[i]] for i in idx_pool]
        try:
            bb, _, _, _ = logistic_irls(Xb, yb, max_iter=50)
            boot_coefs.append(bb[1])
        except Exception:
            continue
    boot_coefs.sort()
    if len(boot_coefs) > 20:
        blo = boot_coefs[int(0.025 * len(boot_coefs))]
        bhi = boot_coefs[int(0.975 * len(boot_coefs))]
        print(f"  block-bootstrap 95% CI on avg_price_{N} coefficient: [{blo:.4f}, {bhi:.4f}] "
              f"(n_successful_resamples={len(boot_coefs)}/500) "
              "-- only a handful of day-blocks exist, so this CI is necessarily coarse; "
              "treat as a sanity check on the naive SE above, not a precise estimate.", flush=True)
    else:
        print("  not enough successful resamples to report a block-bootstrap CI.", flush=True)

    n_train = int(n * TRAIN_FRACTION)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n))
    Xtr = [[1.0, p[i]] for i in train_idx]
    ytr = [y[i] for i in train_idx]
    beta_tr, se_tr, ll_tr, _ = logistic_irls(Xtr, ytr)
    p_test_pred = [sigmoid(beta_tr[0] + beta_tr[1] * p[i]) for i in test_idx]
    y_test = [y[i] for i in test_idx]
    auc_test = auc_score(p_test_pred, y_test)
    brier_test = brier(p_test_pred, y_test)
    auc_test_raw = auc_score([p[i] for i in test_idx], y_test)
    print(f"\nChronological holdout: train n={len(train_idx)} (earlier in time), test n={len(test_idx)} (later, untouched during fit)", flush=True)
    print(f"  fitted-on-train model evaluated on test: AUC={auc_test:.4f}, Brier={brier_test:.4f}", flush=True)
    print(f"  raw avg_price_{N} (no fitting) on test: AUC={auc_test_raw:.4f}", flush=True)
    print(f"  (compare to in-sample AUC={auc1:.4f} above - a big drop here would mean the in-sample number was overfit/regime-specific)", flush=True)

if __name__ == "__main__":
    main()
