import csv, json, math, datetime, random

WINDOWS_CSV = "windows_backfill.csv"
GRID_CSV = "h1_comprehensive_features.csv"
BUCKET_5S_JSON = "kalshi_trades_5s.json"
BUCKET_SECONDS = 5
FLAT_COST = 0.01  # assumed round-trip cost per contract, same assumption as the H4 backtest
# "t" specs = time boundary in seconds since window open ("first 15 seconds" = ("t", 15))
# "n" specs = first-N-trades boundary, regardless of how long that took ("first 100 trades" = ("n", 100))
# Both come straight from the columns h1_grid_extract.py already wrote to h1_comprehensive_features.csv.
SPECS = ([("t", k) for k in [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 450, 600]]
         + [("n", n) for n in [5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]])
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
N_BOOTSTRAP = 2000
TRAIN_FRACTION = 0.70  # chronological split -- same convention as h1_grid_battery.py / h1_augmented_model.py.
                       # spec/threshold is SELECTED on train only; profit/CI is REPORTED on test only.

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def entry_price_at(buckets, elapsed_seconds):
    idx = int(elapsed_seconds) // BUCKET_SECONDS
    if idx >= len(buckets):
        idx = len(buckets) - 1
    if idx < 0:
        return None
    return buckets[idx].get("price")

def spec_label(spec):
    kind, val = spec
    return f"first {val}s" if kind == "t" else f"first {val} trades"

def build_bets(rows, spec, T):
    """For a given subset of windows, a (kind, val) boundary spec, and threshold
    T, return the list of bets that would have been placed.
    kind='t' -> time-based boundary (avg_price_t{val}s, entry at val seconds in)
    kind='n' -> trade-count boundary (avg_price_n{val}, entry at the elapsed time
                implied by frac_window_n{val} -- i.e. however long it actually
                took for {val} trades to happen in that window)."""
    kind, val = spec
    sigcol = f"avg_price_t{val}s" if kind == "t" else f"avg_price_n{val}"
    out = []
    for r in rows:
        sig_raw = r["grow"].get(sigcol, "")
        if sig_raw == "":
            continue
        sig = float(sig_raw)
        if abs(sig - 0.5) < T:
            continue
        if kind == "t":
            elapsed = val
        else:
            frac_raw = r["grow"].get(f"frac_window_n{val}", "")
            if frac_raw == "":
                continue
            elapsed = float(frac_raw) * (r["close_ts"] - r["open_ts"])
        entry = entry_price_at(r["buckets"], elapsed)
        if entry is None or entry <= 0.001 or entry >= 0.999:
            continue
        bet_yes = sig > 0.5
        cost = entry if bet_yes else (1 - entry)
        win = (r["y"] == 1) if bet_yes else (r["y"] == 0)
        profit = (1 - cost) if win else -cost
        profit_costed = profit - FLAT_COST
        out.append({"sig": sig, "profit": profit, "profit_costed": profit_costed, "win": win, "day": r["day"]})
    return out

def summarize(bets):
    n = len(bets)
    if n == 0:
        return None
    profits = [b["profit"] for b in bets]
    profits_c = [b["profit_costed"] for b in bets]
    mean_p = sum(profits) / n
    mean_pc = sum(profits_c) / n
    win_rate = sum(1 for b in bets if b["win"]) / n
    var_p = sum((p - mean_p) ** 2 for p in profits) / max(n - 1, 1)
    se = math.sqrt(var_p / n) if n > 1 else float("nan")
    t = mean_p / se if se > 0 else float("nan")
    return {"n": n, "win_rate": win_rate, "mean_p": mean_p, "mean_pc": mean_pc, "se": se, "t": t, "bets": bets}

def block_bootstrap_ci(bets, n_boot=N_BOOTSTRAP, seed=42):
    days = sorted(set(b["day"] for b in bets))
    by_day = {d: [] for d in days}
    for b in bets:
        by_day[b["day"]].append(b["profit_costed"])
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample_days = [rng.choice(days) for _ in range(len(days))]
        pool = []
        for d in sample_days:
            pool.extend(by_day[d])
        if len(pool) < 5:
            continue
        boot_means.append(sum(pool) / len(pool))
    boot_means.sort()
    if len(boot_means) <= 20:
        return None, None, len(days)
    lo = boot_means[int(0.025 * len(boot_means))]
    hi = boot_means[int(0.975 * len(boot_means))]
    return lo, hi, len(days)

def main():
    with open(WINDOWS_CSV) as f:
        wrows = {r["ticker"]: r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"}
    with open(GRID_CSV) as f:
        grows = {r["ticker"]: r for r in csv.DictReader(f)}
    with open(BUCKET_5S_JSON) as f:
        bucket5s = json.load(f)

    base = []
    for ticker, wr in wrows.items():
        g = grows.get(ticker)
        if not g:
            continue
        b = bucket5s.get(ticker)
        if not b:
            continue
        open_ts = parse_time(wr["open_time"]).timestamp()
        close_ts = parse_time(wr["close_time"]).timestamp()
        day = wr["open_time"][:10]
        base.append({"ticker": ticker, "series": wr["series"], "y": int(g["y"]), "day": day,
                     "open_ts": open_ts, "close_ts": close_ts, "grow": g, "buckets": b})

    base.sort(key=lambda r: r["open_ts"])
    n_train = int(len(base) * TRAIN_FRACTION)
    train, test = base[:n_train], base[n_train:]
    print(f"Base windows usable: {len(base)}  ->  train={len(train)} (earliest {TRAIN_FRACTION*100:.0f}%), "
          f"test/holdout={len(test)} (latest {(1-TRAIN_FRACTION)*100:.0f}%)\n")

    print("=== TRAIN sweep (selection only -- these numbers are NOT the report, they're just used to pick spec/threshold) ===")
    print(f"{'spec':>16s} {'thresh':>7s} {'n_bets':>7s} {'win%':>6s} {'mean_profit':>11s} {'mean_profit_costed':>19s} {'se':>8s} {'t-stat':>7s}")
    train_results = []
    for spec in SPECS:
        for T in THRESHOLDS:
            bets = build_bets(train, spec, T)
            s = summarize(bets)
            if s is None or s["n"] < 15:
                continue
            train_results.append({"spec": spec, "T": T, **s})
            print(f"{spec_label(spec):>16s} {T:7.2f} {s['n']:7d} {s['win_rate']*100:5.1f}% {s['mean_p']:11.4f} {s['mean_pc']:19.4f} {s['se']:8.4f} {s['t']:7.2f}")

    candidates = [r for r in train_results if r["n"] >= 40]
    candidates.sort(key=lambda r: -r["mean_pc"])
    print("\n=== TRAIN candidates with n>=40 bets, ranked by COSTED mean profit per contract (selection step) ===")
    for r in candidates[:15]:
        print(f"  {spec_label(r['spec']):>16s} thresh={r['T']:.2f}  n={r['n']:4d}  win_rate={r['win_rate']*100:5.1f}%  "
              f"mean_profit(no cost)={r['mean_p']:.4f}  mean_profit(costed)={r['mean_pc']:.4f}  t={r['t']:.2f}")

    if not candidates:
        print("\nNo train candidate cleared the n>=40 bar -- nothing to evaluate on holdout.")
        return

    best = candidates[0]
    spec_star, T_star = best["spec"], best["T"]
    print(f"\nSelected on TRAIN only: {spec_label(spec_star)}, threshold={T_star} "
          f"(train mean_profit_costed={best['mean_pc']:.4f}, train n={best['n']})")

    print("\n=== HOLDOUT (untouched 30%, never used for selection) -- the trustworthy numbers ===")
    test_bets = build_bets(test, spec_star, T_star)
    test_summary = summarize(test_bets)
    if test_summary is None or test_summary["n"] < 15:
        print(f"  Too few holdout bets (n={0 if test_summary is None else test_summary['n']}) at this K/threshold "
              "to report a meaningful result. This itself is informative: the train-selected setup doesn't "
              "generate enough signal on fresh data.")
        return

    print(f"  n_bets={test_summary['n']}  win_rate={test_summary['win_rate']*100:.1f}%  "
          f"mean_profit(no cost)={test_summary['mean_p']:.4f}  mean_profit(costed)={test_summary['mean_pc']:.4f}  "
          f"se={test_summary['se']:.4f}  t={test_summary['t']:.2f}")

    lo, hi, ndays = block_bootstrap_ci(test_bets)
    if lo is not None:
        survives = lo > 0 or hi < 0
        print(f"  Block-bootstrap-by-day 95% CI on costed mean profit/contract (holdout only): [{lo:.4f}, {hi:.4f}] "
              f"over {ndays} holdout days")
        print(f"  -> Edge {'SURVIVES' if survives else 'DOES NOT clearly survive'} out-of-sample "
              f"({'CI excludes 0' if survives else 'CI includes 0'}).")
    else:
        print(f"  Not enough holdout days ({ndays}) for a stable block-bootstrap CI -- treat this holdout "
              "result as directional only, not conclusive.")

if __name__ == "__main__":
    main()
