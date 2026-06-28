import csv, json, math, datetime
from bisect import bisect_right

WINDOWS_CSV = "windows_backfill.csv"
KALSHI_5S_FILE = "kalshi_trades_5s.json"
SYMBOL_MAP = {"KXBTC15M": "BTCUSDT", "KXETH15M": "ETHUSDT", "KXSOL15M": "SOLUSDT"}
BUCKET_SECONDS = 5
MAX_LAG = 2          # +/- 2 buckets = +/- 10s around each Kalshi observation
TRAIN_FRACTION = 0.7  # chronological split for H4 - earliest 70% of windows = train
COST_PER_TRADE = 0.01  # assumed round-trip Kalshi bid/ask spread cost, in price units (cents->dollars)

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def load_binance_1s(sym):
    fname = f"binance_1s_{sym}.json"
    with open(fname) as f:
        raw = json.load(f)
    times = [k[0] // 1000 for k in raw]
    closes = [float(k[4]) for k in raw]
    return times, closes

def price_at_or_before(times, closes, ts):
    i = bisect_right(times, ts) - 1
    if i < 0:
        return None
    return closes[i]

# ---------------- tiny pure-python linear algebra (no numpy dependency) ----------------

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
            raise ValueError("singular matrix in OLS - check for collinear/constant columns")
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

def ols_cluster_robust(X, y, clusters):
    """
    Pooled OLS with cluster-robust (CR1) sandwich standard errors.
    X: list of rows (incl. intercept col), y: list of floats, clusters: list of cluster ids (one per row).
    This matters here because rows within the same 15-min window are NOT independent -
    naive OLS standard errors would be badly overstated in significance otherwise.
    """
    n, k = len(X), len(X[0])
    beta, XtX_inv = ols_fit(X, y)
    resid = [y[i] - sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]

    by_cluster = {}
    for i in range(n):
        by_cluster.setdefault(clusters[i], []).append(i)

    meat = [[0.0] * k for _ in range(k)]
    for idxs in by_cluster.values():
        s = [0.0] * k
        for i in idxs:
            u = resid[i]
            Xi = X[i]
            for j in range(k):
                s[j] += Xi[j] * u
        for a in range(k):
            sa = s[a]
            if sa == 0.0:
                continue
            for b in range(k):
                meat[a][b] += sa * s[b]

    G = len(by_cluster)
    corr = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0

    mid = mat_mul(XtX_inv, mat_mul(meat, XtX_inv))
    se = [math.sqrt(max(mid[j][j], 0.0) * corr) for j in range(k)]
    return beta, se, n, k, G, resid

def t_and_p(beta, se):
    out = []
    for b, s in zip(beta, se):
        if s > 0:
            t = b / s
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        else:
            t, p = float("nan"), float("nan")
        out.append((t, p))
    return out

def bh_correction(pvals, labels, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns list of (label, pval, survives)."""
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
    return [(labels[i], pvals[i], survives[i]) for i in range(m)]

# ---------------- data loading / panel construction ----------------

def load_windows():
    with open(WINDOWS_CSV) as f:
        return [r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"]

def build_aligned_series(ticker, series_name, bucket_series, binance):
    sym = SYMBOL_MAP.get(series_name)
    if not sym or sym not in binance:
        return None
    times, closes = binance[sym]
    kprices = [b["price"] for b in bucket_series]
    if any(p is None for p in kprices):
        return None  # window had zero Kalshi trades the whole time - can't use it
    traded = [bool(b.get("traded", False)) for b in bucket_series]
    bucket_ts_list = [b["bucket_ts"] for b in bucket_series]
    bprices = [price_at_or_before(times, closes, ts) for ts in bucket_ts_list]
    if any(p is None or p <= 0 for p in bprices):
        return None
    return kprices, bprices, bucket_ts_list, traded

def main():
    rows = load_windows()
    with open(KALSHI_5S_FILE) as f:
        kalshi_5s = json.load(f)

    binance = {}
    for sym in set(SYMBOL_MAP.values()):
        try:
            binance[sym] = load_binance_1s(sym)
            print(f"Loaded {len(binance[sym][0])} 1s Binance candles for {sym}", flush=True)
        except FileNotFoundError:
            print(f"WARNING: binance_1s_{sym}.json not found - windows for that symbol will be skipped", flush=True)

    # sort windows chronologically (must NOT be random - we need a genuinely future,
    # non-overlapping holdout, same lesson as the Polymarket same-window contamination case)
    rows_sorted = sorted(rows, key=lambda r: parse_time(r["open_time"]))

    # ---- PASS 1: figure out which windows are actually usable (Kalshi trades AND
    # matching Binance coverage) BEFORE computing the train/test split. The old version
    # split on ALL 15,548 windows by date, but Binance 1s data only covers ~9 of the ~56
    # days in windows_backfill.csv, so the ~1,493 usable windows are NOT evenly spread
    # across that split - they clustered almost entirely in one partition, leaving the
    # other with zero rows and crashing H4 with an empty training set. Fix: compute the
    # 70/30 split over the usable windows ONLY, in their own chronological order. ----
    usable = []  # (ticker, series_name, kprices, bprices, traded) in chronological order
    n_windows_skipped = 0
    for r in rows_sorted:
        ticker = r["ticker"]
        series_name = r["series"]
        bucket_series = kalshi_5s.get(ticker)
        if not bucket_series:
            n_windows_skipped += 1
            continue
        aligned = build_aligned_series(ticker, series_name, bucket_series, binance)
        if aligned is None:
            n_windows_skipped += 1
            continue
        kprices, bprices, _, traded = aligned
        usable.append((ticker, series_name, kprices, bprices, traded))

    n_train_windows = int(len(usable) * TRAIN_FRACTION)
    train_tickers = set(u[0] for u in usable[:n_train_windows])
    test_tickers = set(u[0] for u in usable[n_train_windows:])
    print(f"\nWindows usable: {len(usable)}, skipped (no Kalshi trades / missing Binance data): {n_windows_skipped}", flush=True)
    print(f"Chronological split for H4, computed over USABLE windows only: {len(train_tickers)} train, "
          f"{len(test_tickers)} test (later in time, never touched during fitting)", flush=True)

    offsets = list(range(-MAX_LAG, MAX_LAG + 1))  # e.g. [-2,-1,0,1,2]
    # offset > 0  => binance return AFTER the current Kalshi tick -> tests "Kalshi LEADS Binance"
    # offset < 0  => binance return BEFORE the current Kalshi tick -> tests "Kalshi LAGS Binance"
    h3_labels = ["intercept"] + [f"binance_ret_t{o:+d}" for o in offsets]

    h3_X, h3_y, h3_clusters = [], [], []
    h3b_X, h3b_y, h3b_clusters = [], [], []
    h4_train_X, h4_train_y = [], []
    h4_test_rows = []  # (X_row, y_actual, ticker) for test set, predictors restricted to non-future info

    n_stale_excluded_h3, n_stale_excluded_h4 = 0, 0

    for ticker, series_name, kprices, bprices, traded in usable:
        kdelta = [None] + [kprices[i] - kprices[i - 1] for i in range(1, len(kprices))]
        bret = [None] + [math.log(bprices[i] / bprices[i - 1]) for i in range(1, len(bprices))]

        n_buckets = len(kprices)
        is_train = ticker in train_tickers

        for t in range(MAX_LAG, n_buckets - MAX_LAG):
            if kdelta[t] is None:
                continue
            # FIX for the staleness/non-synchronous-trading confound: Kalshi is thinly
            # traded (especially SOL), so most 5s buckets just forward-fill the last
            # trade price -> kdelta is mechanically 0, not a real "no movement" data
            # point. Using those as the regression OUTCOME lets stretches of stale
            # zeros line up with whatever Binance is doing and manufacture a spurious
            # "lead". Only treat bucket t as a real outcome if Kalshi actually traded in
            # bucket t (traded[t] is True) - i.e. a genuine new price was set.
            if not traded[t]:
                n_stale_excluded_h3 += 1
                continue
            # --- H3 panel: full lead/lag window (uses future Binance info on purpose,
            # this is an EXPLANATORY regression about direction of co-movement, not a forecast) ---
            feats_h3 = [1.0]
            ok = True
            for o in offsets:
                idx = t + o
                if bret[idx] is None:
                    ok = False
                    break
                feats_h3.append(bret[idx])
            if ok:
                h3_X.append(feats_h3)
                h3_y.append(kdelta[t])
                h3_clusters.append(ticker)
                # --- H3b diagnostic: is Binance's own t+1 return predictable from its
                # t-2..t+0 returns (own short-horizon autocorrelation)? If yes, that's a
                # competing explanation for the H3 "t+1 effect": bret_t+1 is collinear
                # with bret_t+0 (which truly co-moves with kdelta_t), so some of bret_t0's
                # true contemporaneous effect can leak into the bret_t+1 coefficient even
                # with no real "Kalshi leads Binance" relationship. ---
                h3b_X.append([1.0, bret[t - 2], bret[t - 1], bret[t]])
                h3b_y.append(bret[t + 1])
                h3b_clusters.append(ticker)

            # --- H4 panel: forecasting model, ONLY past/contemporaneous info allowed
            # (offsets <= 0), predicting next-tick Kalshi delta (kdelta[t+1]) ---
            if t + 1 < n_buckets and kdelta[t + 1] is not None:
                if not traded[t + 1]:
                    n_stale_excluded_h4 += 1
                    continue
                feats_h4 = [1.0]
                ok4 = True
                for o in offsets:
                    if o > 0:
                        continue  # would be future info relative to the forecast origin - not allowed
                    idx = t + o
                    if bret[idx] is None:
                        ok4 = False
                        break
                    feats_h4.append(bret[idx])
                feats_h4.append(kdelta[t])  # own last move as an autoregressive control
                if ok4:
                    if is_train:
                        h4_train_X.append(feats_h4)
                        h4_train_y.append(kdelta[t + 1])
                    else:
                        h4_test_rows.append((feats_h4, kdelta[t + 1], ticker))

    print(f"Stale (non-traded) buckets excluded as regression outcomes: {n_stale_excluded_h3} (H3), "
          f"{n_stale_excluded_h4} (H4 additional)", flush=True)

    # ================= H3: pooled lead-lag regression =================
    print(f"\n=== H3: Kalshi_delta_t ~ Binance_log_return at t-{MAX_LAG}..t+{MAX_LAG} (5s buckets) ===", flush=True)
    print(f"n={len(h3_y)} pooled 5s observations across {len(set(h3_clusters))} windows (cluster-robust SEs by window)", flush=True)
    if len(h3_y) > len(h3_X[0]) + 1 and len(set(h3_clusters)) > 1:
        beta, se, n, k, G, resid = ols_cluster_robust(h3_X, h3_y, h3_clusters)
        tp = t_and_p(beta, se)
        coef_pvals = [p for _, p in tp[1:]]  # exclude intercept from BH correction
        coef_labels = h3_labels[1:]
        bh = bh_correction(coef_pvals, coef_labels)
        bh_map = {lbl: (p, surv) for lbl, p, surv in bh}
        print(f"{'term':22s} {'coef':>10s} {'se':>10s} {'t':>8s} {'p':>8s} {'BH-survives':>12s}", flush=True)
        for label, b, s, (t, p) in zip(h3_labels, beta, se, tp):
            bh_flag = "" if label == "intercept" else ("yes" if bh_map[label][1] else "no")
            print(f"{label:22s} {b:10.6f} {s:10.6f} {t:8.2f} {p:8.4f} {bh_flag:>12s}", flush=True)
        print("\nInterpretation: significant coefficient at t-k (negative offset) => Kalshi LAGS Binance "
              "(reacts to Binance's past). Significant at t+k (positive offset) => Kalshi LEADS Binance "
              "(predicts Binance's future move). BH-survives uses Benjamini-Hochberg FDR correction across "
              f"the {len(coef_labels)} lag/lead coefficients to control for testing multiple lags at once.", flush=True)
    else:
        print("Not enough data yet to fit H3 regression.", flush=True)

    # ================= H3b: Binance's own short-horizon autocorrelation (diagnostic) =================
    print(f"\n=== H3b diagnostic: Binance_ret_t+1 ~ Binance_ret_t-2,t-1,t+0 (own autocorrelation, "
          f"same {len(set(h3b_clusters))} windows) ===", flush=True)
    print("Purpose: if Binance's own near-term returns are autocorrelated, that alone can make bret_t+1\n"
          "collinear with bret_t+0 (which truly co-moves with kdelta_t), producing exactly the spurious\n"
          "'Kalshi leads Binance at t+1' pattern seen in H3 even with zero real lead. This is the\n"
          "competing, non-staleness explanation for that result.", flush=True)
    if len(h3b_y) > 5 and len(set(h3b_clusters)) > 1:
        beta_b, se_b, n_b, k_b, G_b, resid_b = ols_cluster_robust(h3b_X, h3b_y, h3b_clusters)
        tp_b = t_and_p(beta_b, se_b)
        labels_b = ["intercept", "bret_t-2", "bret_t-1", "bret_t+0"]
        mean_y = sum(h3b_y) / len(h3b_y)
        sst_b = sum((y - mean_y) ** 2 for y in h3b_y)
        sse_b = sum(r ** 2 for r in resid_b)
        r2_b = 1 - sse_b / sst_b if sst_b > 0 else float("nan")
        print(f"{'term':12s} {'coef':>10s} {'se':>10s} {'t':>8s} {'p':>8s}", flush=True)
        for label, b, s, (t, p) in zip(labels_b, beta_b, se_b, tp_b):
            print(f"{label:12s} {b:10.6f} {s:10.6f} {t:8.2f} {p:8.4f}", flush=True)
        print(f"R^2 = {r2_b:.5f}  (in-sample fraction of bret_t+1's variance explained by its own t-2..t+0 lags -- "
              f"the higher this is, the more the H3 't+1 leads' coefficient is plausibly just autocorrelation "
              f"bleeding through collinear lags rather than Kalshi genuinely front-running)", flush=True)
    else:
        print("Not enough data for H3b diagnostic.", flush=True)

    # ================= H4: out-of-sample forecast + costed backtest =================
    print(f"\n=== H4: out-of-sample forecast (train on earlier {len(train_tickers)} windows, "
          f"test on later {len(test_tickers)} windows, never touched during fit) ===", flush=True)
    if h4_train_X and len(h4_train_y) > len(h4_train_X[0]) + 1 and h4_test_rows:
        beta4, _ = ols_fit(h4_train_X, h4_train_y)
        preds = [sum(x[j] * beta4[j] for j in range(len(beta4))) for x, _, _ in h4_test_rows]
        actuals = [a for _, a, _ in h4_test_rows]
        mean_actual = sum(actuals) / len(actuals)
        sst = sum((a - mean_actual) ** 2 for a in actuals)
        sse = sum((a - p) ** 2 for a, p in zip(actuals, preds))
        oos_r2 = 1 - sse / sst if sst > 0 else float("nan")
        print(f"n_test={len(actuals)}, out-of-sample R^2 = {oos_r2:.5f} "
              "(this is the honest number - in-sample R^2 on the training data will look better and should be ignored)", flush=True)

        # naive costed backtest: trade in the predicted direction, pay a flat round-trip cost
        pnl = []
        for pred, actual in zip(preds, actuals):
            if abs(pred) < 1e-9:
                continue
            signal = 1.0 if pred > 0 else -1.0
            pnl.append(signal * actual - COST_PER_TRADE)
        if pnl:
            mean_pnl = sum(pnl) / len(pnl)
            var_pnl = sum((x - mean_pnl) ** 2 for x in pnl) / max(len(pnl) - 1, 1)
            se_pnl = math.sqrt(var_pnl / len(pnl)) if len(pnl) > 1 else float("nan")
            t_pnl = mean_pnl / se_pnl if se_pnl > 0 else float("nan")
            print(f"\nCosted backtest on test set: {len(pnl)} trades, assumed round-trip cost={COST_PER_TRADE}", flush=True)
            print(f"  mean P&L per trade: {mean_pnl:.6f}  (naive SE={se_pnl:.6f}, t={t_pnl:.2f})", flush=True)
            print("  CAVEAT: this SE treats trades as independent, which they are NOT (heavy clustering "
                  "within windows) - treat this t-stat as optimistic; a proper block-bootstrap by window "
                  "is the next step before trusting this number.", flush=True)
    else:
        print("Not enough data yet to fit/test H4.", flush=True)

if __name__ == "__main__":
    main()
