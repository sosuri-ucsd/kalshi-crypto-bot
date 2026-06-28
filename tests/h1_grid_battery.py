import csv, json, math, datetime, random

WINDOWS_CSV = "windows_backfill.csv"
GRID_CSV = "h1_comprehensive_features.csv"
TRAIN_FRACTION = 0.7

K_GRID = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 450, 600]
N_GRID = [5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
BOUNDARIES = [(f"t{K}s", f"{K}s elapsed") for K in K_GRID] + [(f"n{N}", f"first {N} trades") for N in N_GRID]

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

def fit_eval(sub, col):
    n = len(sub)
    p = [r[col] for r in sub]
    y = [r["y"] for r in sub]
    X = [[1.0, pi] for pi in p]
    beta, se, ll, mu = logistic_irls(X, y)
    ybar = sum(y) / n
    ll_null = n * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))
    pr2 = 1 - ll / ll_null
    a_ = auc_score(p, y)
    z = beta[1] / se[1] if se[1] > 0 else float("nan")
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if not math.isnan(z) else float("nan")
    return {"n": n, "coef": beta[1], "se": se[1], "z": z, "p": pval, "pr2": pr2, "auc": a_}

def cox_calibration(sub, col):
    eps = 1e-4
    p = [r[col] for r in sub]
    y = [r["y"] for r in sub]
    logit_p = [math.log(min(max(pi, eps), 1 - eps) / (1 - min(max(pi, eps), 1 - eps))) for pi in p]
    X = [[1.0, lp] for lp in logit_p]
    beta, se, ll, mu = logistic_irls(X, y)
    return beta[0], se[0], beta[1], se[1]

def chrono_holdout(sub_sorted, col):
    n = len(sub_sorted)
    n_train = int(n * TRAIN_FRACTION)
    train = sub_sorted[:n_train]
    test = sub_sorted[n_train:]
    p_tr = [r[col] for r in train]
    y_tr = [r["y"] for r in train]
    X_tr = [[1.0, pi] for pi in p_tr]
    beta, se, ll, mu = logistic_irls(X_tr, y_tr)
    p_test = [sigmoid(beta[0] + beta[1] * r[col]) for r in test]
    y_test = [r["y"] for r in test]
    auc_t = auc_score(p_test, y_test)
    brier_t = brier(p_test, y_test)
    auc_raw = auc_score([r[col] for r in test], y_test)
    return {"n_train": len(train), "n_test": len(test), "auc_oos": auc_t, "brier_oos": brier_t, "auc_oos_raw": auc_raw}

def main():
    with open(WINDOWS_CSV) as f:
        wrows = {r["ticker"]: r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"}
    with open(GRID_CSV) as f:
        grows = list(csv.DictReader(f))

    data = []
    for r in grows:
        wr = wrows.get(r["ticker"])
        if not wr:
            continue
        rec = {"ticker": r["ticker"], "series": r["series"], "y": int(r["y"]),
               "open_ts": parse_time(wr["open_time"]).timestamp(), "day": wr["open_time"][:10]}
        for suffix, _ in BOUNDARIES:
            col = f"avg_price_{suffix}"
            v = r.get(col, "")
            rec[col] = float(v) if v != "" else None
        data.append(rec)
    data.sort(key=lambda r: r["open_ts"])
    print(f"Total usable windows: {len(data)}\n")

    results = []
    for suffix, label in BOUNDARIES:
        col = f"avg_price_{suffix}"
        sub = [r for r in data if r[col] is not None]
        if len(sub) < 50:
            continue
        fit = fit_eval(sub, col)
        hold = chrono_holdout(sub, col)
        cal_i, cal_i_se, cal_s, cal_s_se = cox_calibration(sub, col)
        results.append({"suffix": suffix, "label": label, "col": col, **fit, **{f"hold_{k}": v for k, v in hold.items()},
                         "cal_intercept": cal_i, "cal_slope": cal_s})

    pvals = [r["p"] for r in results]
    survives = bh_correction(pvals)
    for r, s in zip(results, survives):
        r["bh_survives"] = s

    print("=== Full boundary sweep, ranked by OUT-OF-SAMPLE AUC (the trustworthy number) ===")
    print(f"{'boundary':>10s} {'n':>5s} {'in-AUC':>7s} {'OOS-AUC':>8s} {'OOS-Brier':>9s} {'p(BH)':>6s} {'pseudoR2':>8s} {'cal-slope':>9s} {'n_test':>7s}")
    results_sorted = sorted(results, key=lambda r: -r["hold_auc_oos"])
    for r in results_sorted:
        bh = "yes" if r["bh_survives"] else "no "
        print(f"{r['label']:>10s} {r['n']:5d} {r['auc']:7.4f} {r['hold_auc_oos']:8.4f} {r['hold_brier_oos']:9.4f} "
              f"{bh:>6s} {r['pr2']:8.4f} {r['cal_slope']:9.3f} {r['hold_n_test']:7d}")

    print("\n=== Per-coin breakdown for the TOP 5 boundaries by OOS AUC ===")
    top5 = results_sorted[:5]
    for r in top5:
        col = r["col"]
        print(f"\n-- {r['label']} ({col}) --")
        for coin in ["KXBTC15M", "KXETH15M", "KXSOL15M"]:
            sub_coin = [d for d in data if d["series"] == coin and d[col] is not None]
            if len(sub_coin) < 30:
                continue
            f_ = fit_eval(sub_coin, col)
            print(f"   {coin:10s} n={f_['n']:4d}  in-AUC={f_['auc']:.4f}  pseudoR2={f_['pr2']:.4f}  p={f_['p']:.4f}")

    print("\n=== Best single boundary overall (by OOS AUC): ===")
    best = results_sorted[0]
    print(f"{best['label']}  ({best['col']})  OOS-AUC={best['hold_auc_oos']:.4f}  in-sample AUC={best['auc']:.4f}  "
          f"calibration slope={best['cal_slope']:.3f} (1.0=perfectly calibrated)")

if __name__ == "__main__":
    main()
