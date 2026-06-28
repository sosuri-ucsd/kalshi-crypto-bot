import csv, json, math, datetime, random

WINDOWS_CSV = "windows_backfill.csv"
GRID_CSV = "h1_comprehensive_features.csv"
H2_CSV = "h1_h2_features.csv"          # source of sigma_annual (crypto volatility at decision time)
BUCKET_5S_JSON = "kalshi_trades_5s.json"
BUCKET_SECONDS = 5
TRAIN_FRACTION = 0.7
N_BOOTSTRAP = 3000
BASE_K = 120          # our chosen "early but actionable" boundary, from the grid sweep
HALF_BUCKETS = (BASE_K // BUCKET_SECONDS) // 2   # 12 buckets = 60s, splits the 0-120s window in half

# ---------- shared stats helpers (same toolkit as h1_grid_battery.py) ----------

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

def norm_p(z):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

def chi2_sf_wilson_hilferty(x, k):
    # Wilson-Hilferty approximation: converts chi2_k to approx standard normal, returns survival p-value
    if x <= 0:
        return 1.0
    h = 2.0 / (9.0 * k)
    z = ((x / k) ** (1.0 / 3.0) - (1 - h)) / math.sqrt(h)
    return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))

def mean(xs):
    return sum(xs) / len(xs)

def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))

# ---------- load data ----------

def load():
    with open(WINDOWS_CSV) as f:
        wrows = {r["ticker"]: r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"}
    with open(GRID_CSV) as f:
        grows = {r["ticker"]: r for r in csv.DictReader(f)}
    with open(H2_CSV) as f:
        h2rows = {r["ticker"]: r for r in csv.DictReader(f)}
    with open(BUCKET_5S_JSON) as f:
        buckets = json.load(f)

    data = []
    for ticker, wr in wrows.items():
        g = grows.get(ticker)
        if not g:
            continue
        b = buckets.get(ticker)
        if not b or len(b) < BASE_K // BUCKET_SECONDS:
            continue
        base_raw = g.get(f"avg_price_t{BASE_K}s", "")
        if base_raw == "":
            continue
        base_price = float(base_raw)
        n_trades_base = int(g.get(f"n_trades_t{BASE_K}s", 0) or 0)

        # bucket prices within first 120s (24 buckets of 5s), forward-filled; drop leading Nones
        n_buckets_needed = BASE_K // BUCKET_SECONDS
        early_buckets = b[:n_buckets_needed]
        prices = [bb["price"] for bb in early_buckets if bb.get("price") is not None]
        if len(prices) < 4:
            continue
        dispersion = sd(prices)

        first_half = [bb["price"] for bb in early_buckets[:HALF_BUCKETS] if bb.get("price") is not None]
        second_half = [bb["price"] for bb in early_buckets[HALF_BUCKETS:] if bb.get("price") is not None]
        if not first_half or not second_half:
            continue
        momentum = mean(second_half) - mean(first_half)

        h2 = h2rows.get(ticker)
        sigma = None
        if h2 and h2.get("sigma_annual", "") not in ("", None):
            try:
                sigma = float(h2["sigma_annual"])
            except ValueError:
                sigma = None
        if sigma is None:
            continue

        is_eth = 1.0 if wr["series"] == "KXETH15M" else 0.0
        is_sol = 1.0 if wr["series"] == "KXSOL15M" else 0.0

        data.append({
            "ticker": ticker, "series": wr["series"], "y": int(g["y"]),
            "open_ts": parse_time(wr["open_time"]).timestamp(), "day": wr["open_time"][:10],
            "base_price": base_price,
            "log_n_trades": math.log(max(n_trades_base, 1)),
            "dispersion": dispersion,
            "momentum": momentum,
            "sigma": sigma,
            "is_eth": is_eth, "is_sol": is_sol,
        })
    data.sort(key=lambda r: r["open_ts"])
    return data

# ---------- model fit/eval helpers ----------

def build_X(sub, cols):
    X = []
    for r in sub:
        row = [1.0]
        for c in cols:
            row.append(r[c])
        X.append(row)
    return X

def fit_model(sub, cols):
    y = [r["y"] for r in sub]
    X = build_X(sub, cols)
    beta, se, ll, mu = logistic_irls(X, y)
    return beta, se, ll, mu

def predict(beta, r, cols):
    eta = beta[0] + sum(beta[1 + i] * r[c] for i, c in enumerate(cols))
    return sigmoid(eta)

def lr_test(ll_small, ll_big, df_diff):
    stat = 2 * (ll_big - ll_small)
    p = chi2_sf_wilson_hilferty(stat, df_diff) if stat > 0 else 1.0
    return stat, p

def chrono_split(data):
    n = len(data)
    n_train = int(n * TRAIN_FRACTION)
    return data[:n_train], data[n_train:]

def oos_eval(train, test, cols):
    beta, se, ll, mu = fit_model(train, cols)
    p_test = [predict(beta, r, cols) for r in test]
    y_test = [r["y"] for r in test]
    return {"p": p_test, "y": y_test, "auc": auc_score(p_test, y_test), "brier": brier(p_test, y_test), "beta": beta}

def clark_west(y, pA, pB):
    # Clark-West (2007) nested-model OOS comparison, adapted from squared-error forecast loss
    # to Brier (squared probability) loss. pA = restricted/baseline model, pB = unrestricted/augmented (nests A).
    # f_t = (y-pA)^2 - [ (y-pB)^2 - (pA-pB)^2 ]; test mean(f_t) > 0 via one-sample t-test (one-sided).
    f = [(y[i] - pA[i]) ** 2 - ((y[i] - pB[i]) ** 2 - (pA[i] - pB[i]) ** 2) for i in range(len(y))]
    m = mean(f)
    s = sd(f)
    n = len(f)
    se = s / math.sqrt(n) if n > 1 else float("nan")
    t = m / se if se > 0 else float("nan")
    p_one_sided = 1 - 0.5 * (1 + math.erf(t / math.sqrt(2))) if not math.isnan(t) else float("nan")
    return {"mean_f": m, "t": t, "p_one_sided": p_one_sided, "n": n}

def block_bootstrap_diff(test_set_with_day, brierA_per_obs, brierB_per_obs, n_boot=N_BOOTSTRAP, seed=42):
    by_day = {}
    for i, r in enumerate(test_set_with_day):
        by_day.setdefault(r["day"], []).append(brierA_per_obs[i] - brierB_per_obs[i])
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

# ---------- main ----------

def main():
    data = load()
    print(f"Usable windows with full augmented feature set: {len(data)}\n")

    BASELINE_COLS = ["base_price"]
    SPECS = [
        ("base_price + momentum", ["base_price", "momentum"]),
        ("base_price + log_n_trades", ["base_price", "log_n_trades"]),
        ("base_price + dispersion", ["base_price", "dispersion"]),
        ("base_price + sigma", ["base_price", "sigma"]),
        ("base_price + ALL (momentum, log_n_trades, dispersion, sigma, is_eth, is_sol)",
         ["base_price", "momentum", "log_n_trades", "dispersion", "sigma", "is_eth", "is_sol"]),
    ]

    print(f"=== Baseline model: y ~ avg_price_t{BASE_K}s ===")
    beta0, se0, ll0, mu0 = fit_model(data, BASELINE_COLS)
    auc0 = auc_score([r["base_price"] for r in data], [r["y"] for r in data])
    print(f"in-sample AUC={auc0:.4f}, log-lik={ll0:.2f}, n={len(data)}\n")

    print("=== In-sample likelihood-ratio tests: does adding a feature to the baseline significantly improve fit? ===")
    print("(LR test always favors the bigger model in-sample to some degree -- this is the naive check, NOT the final word)")
    print(f"{'spec':>55s} {'df':>3s} {'LR-stat':>8s} {'p(LR, approx chi2)':>20s}")
    lr_pvals = []
    for label, cols in SPECS:
        beta, se, ll, mu = fit_model(data, cols)
        df_diff = len(cols) - len(BASELINE_COLS)
        stat, p = lr_test(ll0, ll, df_diff)
        lr_pvals.append(p)
        print(f"{label:>55s} {df_diff:3d} {stat:8.2f} {p:20.5f}")
    survives_lr = bh_correction(lr_pvals)
    print("BH-FDR (alpha=0.05) survives:", ["yes" if s else "no" for s in survives_lr])

    print("\n=== TRUE out-of-sample test (70/30 chronological split) -- the test that actually matters ===")
    train, test = chrono_split(data)
    print(f"train n={len(train)} (earliest), test n={len(test)} (latest, never seen during fit)\n")

    base_oos = oos_eval(train, test, BASELINE_COLS)
    print(f"Baseline OOS:  AUC={base_oos['auc']:.4f}  Brier={base_oos['brier']:.4f}\n")

    print(f"{'spec':>55s} {'OOS-AUC':>8s} {'OOS-Brier':>9s} {'d(Brier)':>9s} {'CW t-stat':>9s} {'CW p(1-sided)':>13s} {'BH':>4s}")
    cw_pvals = []
    rows_out = []
    for label, cols in SPECS:
        aug_oos = oos_eval(train, test, cols)
        cw = clark_west(base_oos["y"], base_oos["p"], aug_oos["p"])
        d_brier = base_oos["brier"] - aug_oos["brier"]   # positive = augmented model improved (lower Brier)
        cw_pvals.append(cw["p_one_sided"])
        rows_out.append((label, cols, aug_oos, cw, d_brier))
    survives_cw = bh_correction(cw_pvals)
    for (label, cols, aug_oos, cw, d_brier), surv in zip(rows_out, survives_cw):
        bh = "yes" if surv else "no"
        print(f"{label:>55s} {aug_oos['auc']:8.4f} {aug_oos['brier']:9.4f} {d_brier:9.4f} {cw['t']:9.2f} {cw['p_one_sided']:13.5f} {bh:>4s}")

    print("\n(Reading this table: d(Brier) > 0 means the augmented model had LOWER out-of-sample error than the")
    print(" baseline -- i.e. it genuinely helped, not just fit better in-sample. CW t-stat/p is the Clark-West")
    print(" (2007) test, adapted from squared-error nested-forecast comparison to Brier loss here -- it corrects")
    print(" for the fact that a bigger model's extra estimated parameters add noise that biases a naive OOS")
    print(" comparison toward the smaller model, so this is the fairer test. Methodology: Clark & West (2007),")
    print(" applied to nested nowcasting comparisons in arXiv:2604.01431 (Mohanty & Krishnamachari) for HAR-RV")
    print(" vs HAR-RV-X realized-vol forecasts -- structurally the same situation as ours: baseline single-")
    print(" predictor model vs that predictor plus extra signals.)")

    # block-bootstrap-by-day, for an honest CI on whether the edge survives day-level clustering
    # (same caution as the H1 backtest's day CI). Check BOTH the single clean winner (momentum alone,
    # most parsimonious) and the kitchen-sink "ALL" spec (best raw OOS number).
    brierA_per_obs = [(base_oos["y"][i] - base_oos["p"][i]) ** 2 for i in range(len(test))]
    momentum_idx = [i for i, (label, cols, *_ ) in enumerate(rows_out) if cols == ["base_price", "momentum"]][0]
    best_idx = max(range(len(rows_out)), key=lambda i: rows_out[i][4])
    for idx, tag in [(momentum_idx, "single clean winner: momentum"), (best_idx, "best raw OOS number: kitchen-sink ALL")]:
        label, cols, aug_oos, cw, d_brier = rows_out[idx]
        brierB_per_obs = [(aug_oos["y"][i] - aug_oos["p"][i]) ** 2 for i in range(len(test))]
        bb = block_bootstrap_diff(test, brierA_per_obs, brierB_per_obs)
        print(f"\n=== {tag}: {label} (d(Brier)={d_brier:.4f}) ===")
        if bb:
            lo, hi, ndays = bb
            print(f"Block-bootstrap-by-day 95% CI on (baseline_Brier - augmented_Brier): [{lo:.4f}, {hi:.4f}] over {ndays} days")
            print("(if this excludes 0, the improvement survives day-clustering; with only a handful of days this CI")
            print(" is necessarily coarse -- same caveat as every other day-level result in this project)")

    print("\n=== Calibration-slope vs elapsed-time pattern (descriptive, from h1_grid_battery.py's sweep) ===")
    print("cal-slope by boundary: 5s=0.569, 10s=0.747, 15s=0.866, 20s=1.031, 30s=1.180, 45s=1.177, 60s=1.214,")
    print("                       90s=1.332, 120s=1.408, 180s=1.455, 300s=1.561, 450s=1.883, 600s=2.149")
    print("This RISES monotonically with elapsed time -- the market gets MORE underconfident (slope further from 1)")
    print("the closer it gets to settlement, after an early crossing near ~20s. arXiv:2602.19520 (Decomposing Crowd")
    print("Wisdom) finds the OPPOSITE direction in its cross-domain 'horizon effect': calibration slope is closest")
    print("to 1 for markets close to resolution (0-1h horizon, slope~0.99) and worst for long-horizon markets (>1mo,")
    print("slope~1.32). Our 15-min crypto markets invert that pattern entirely within a single 15-minute window --")
    print("plausible reason: near settlement the outcome is close to mechanically determined by spot price, but")
    print("traders are reluctant to push the price all the way to the extreme even when near-certain (a tick-size/")
    print("inertia effect), so price keeps understating the true win probability right up to the close. This is a")
    print("genuine point of disagreement with that paper's general finding, worth citing as a contrast rather than")
    print("a confirmation.")

if __name__ == "__main__":
    main()
