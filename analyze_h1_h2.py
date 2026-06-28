import csv, json, math, statistics, datetime
from bisect import bisect_right

WINDOWS_CSV = "windows_backfill.csv"
MARKETS_FILE = "backfill_markets.json"
CANDLES_FILE = "backfill_candles.json"
SYMBOL_MAP = {"KXBTC15M": "BTCUSDT", "KXETH15M": "ETHUSDT", "KXSOL15M": "SOLUSDT"}
OUT_CSV = "h1_h2_features.csv"
ANNUAL_SECONDS = 365 * 24 * 3600
# Our spot price comes from 1-minute Binance klines, so it can be up to ~60s
# stale. Don't let tau shrink below that resolution - otherwise the fair-value
# formula collapses into a step function near expiry and "confidently" flips
# to 0/1 based on a spot reading that's actually too stale to support that
# confidence. This will tighten up once we have sub-minute spot data.
TAU_FLOOR_SECONDS = 60
TAU_FLOOR = TAU_FLOOR_SECONDS / ANNUAL_SECONDS

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def load_binance(sym):
    fname = f"binance_klines_{sym}.json"
    with open(fname) as f:
        raw = json.load(f)
    # raw kline: [open_time_ms, open, high, low, close, volume, close_time_ms, ...]
    times = [k[0] // 1000 for k in raw]
    closes = [float(k[4]) for k in raw]
    return times, closes

def price_at_or_before(times, closes, ts):
    i = bisect_right(times, ts) - 1
    if i < 0:
        return None
    return closes[i]

def get_price_dollars(candle):
    # mean_dollars is only populated when there was an actual trade in that
    # minute; fall back to close_dollars, then to the yes_bid/yes_ask midpoint.
    p = candle.get("price", {})
    if p.get("mean_dollars") is not None:
        return float(p["mean_dollars"])
    if p.get("close_dollars") is not None:
        return float(p["close_dollars"])
    bid = candle.get("yes_bid", {}).get("close_dollars")
    ask = candle.get("yes_ask", {}).get("close_dollars")
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return None

def find_last_traded_candle(cs_sorted):
    # Walk backward from close looking for a candle that had a real trade
    # (mean_dollars present). The literal last candle is often "dead" -
    # trading halts before settlement and Kalshi fills that minute with a
    # neutral 0.50 placeholder, which is not a real market price.
    for c in reversed(cs_sorted):
        if c.get("price", {}).get("mean_dollars") is not None:
            return c, True
    return cs_sorted[-1], False

def get_strike_info(m):
    strike_type = (m.get("strike_type") or "").lower()
    floor_strike = m.get("floor_strike")
    cap_strike = m.get("cap_strike")
    if floor_strike is not None:
        return float(floor_strike), strike_type
    if cap_strike is not None:
        return float(cap_strike), strike_type
    return None, strike_type

def main():
    with open(MARKETS_FILE) as f:
        markets = json.load(f)
    with open(CANDLES_FILE) as f:
        candles_by_ticker = json.load(f)

    binance = {}
    for sym in set(SYMBOL_MAP.values()):
        binance[sym] = load_binance(sym)
        print(f"Loaded {len(binance[sym][0])} Binance candles for {sym}", flush=True)

    # Debug: show strike fields on a sample market so we can confirm sign convention
    for ticker, m in markets.items():
        if m.get("result"):
            print("=== Sample market (for strike-field verification) ===", flush=True)
            print("ticker:", ticker, flush=True)
            for k in ("strike_type", "floor_strike", "cap_strike", "custom_strike", "result"):
                print(f"  {k}: {m.get(k)}", flush=True)
            print("======================================================", flush=True)
            break

    with open(WINDOWS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"]

    out_rows = []
    skipped = 0

    for r in rows:
        ticker = r["ticker"]
        series = r["series"]
        sym = SYMBOL_MAP.get(series)
        m = markets.get(ticker)
        cs = candles_by_ticker.get(ticker)
        if not m or not cs or not sym:
            skipped += 1
            continue

        strike, strike_type = get_strike_info(m)
        if strike is None:
            skipped += 1
            continue

        result = m.get("result", "")
        y = 1 if result == "yes" else (0 if result == "no" else None)
        if y is None:
            skipped += 1
            continue

        cs_sorted = sorted(cs, key=lambda c: c.get("end_period_ts", 0))
        early = cs_sorted[0]
        late, late_was_traded = find_last_traded_candle(cs_sorted)
        early_price = get_price_dollars(early)
        late_price = get_price_dollars(late)
        if early_price is None or late_price is None:
            skipped += 1
            continue

        open_ts = int(parse_time(r["open_time"]).timestamp())
        close_ts = int(parse_time(r["close_time"]).timestamp())
        late_ts = late.get("end_period_ts", close_ts)

        times, closes = binance[sym]
        spot = price_at_or_before(times, closes, late_ts)
        if spot is None or spot <= 0 or strike <= 0:
            skipped += 1
            continue

        # realized vol from this window's own Binance path (log returns of 1-min closes).
        # IMPORTANT: only use data up to late_ts (the evaluation point), never up to
        # close_ts - using returns between late_ts and close_ts would be look-ahead
        # bias (estimating "what we could have known" using info from the future
        # relative to the moment we're pricing). Same class of bug as the Polymarket
        # same-window contamination issue.
        window_closes = [c for t, c in zip(times, closes) if open_ts <= t <= late_ts]
        log_rets = []
        for i in range(1, len(window_closes)):
            if window_closes[i - 1] > 0:
                log_rets.append(math.log(window_closes[i] / window_closes[i - 1]))
        if len(log_rets) >= 2:
            sigma_per_min = statistics.pstdev(log_rets)
        else:
            sigma_per_min = 0.0005  # fallback assumption if not enough data
        sigma_annual = sigma_per_min * math.sqrt(365 * 24 * 60)

        tau = max((close_ts - late_ts) / ANNUAL_SECONDS, TAU_FLOOR)
        d = math.log(spot / strike) / (sigma_annual * math.sqrt(tau) + 1e-12)
        fair_value = norm_cdf(d)
        if "less" in strike_type:
            fair_value = 1 - fair_value

        gap = late_price - fair_value

        out_rows.append({
            "ticker": ticker, "series": series, "result": result, "y": y,
            "early_price": early_price, "late_price": late_price,
            "late_was_traded": late_was_traded,
            "minutes_before_close": round((close_ts - late_ts) / 60, 2),
            "spot_at_late": spot, "strike": strike, "strike_type": strike_type,
            "sigma_annual": round(sigma_annual, 4), "tau_seconds_remaining": round(tau * ANNUAL_SECONDS, 1),
            "fair_value": round(fair_value, 4), "gap_late_vs_fair": round(gap, 4),
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows to {OUT_CSV} ({skipped} skipped for missing data)", flush=True)

    # --- H1: does early price predict outcome? ---
    early_prices = [r["early_price"] for r in out_rows]
    ys = [r["y"] for r in out_rows]
    n = len(out_rows)
    brier = sum((p - y) ** 2 for p, y in zip(early_prices, ys)) / n
    mean_p, mean_y = statistics.mean(early_prices), statistics.mean(ys)
    cov = sum((p - mean_p) * (y - mean_y) for p, y in zip(early_prices, ys)) / n
    sd_p, sd_y = statistics.pstdev(early_prices), statistics.pstdev(ys)
    corr = cov / (sd_p * sd_y) if sd_p > 0 and sd_y > 0 else float("nan")

    print("\n=== H1: early-window price vs settlement ===", flush=True)
    print(f"n = {n}", flush=True)
    print(f"Brier score (lower=better, 0.25=coinflip baseline): {brier:.4f}", flush=True)
    print(f"Correlation(early_price, outcome): {corr:.4f}", flush=True)

    print("\nCalibration by early_price decile (predicted vs actual yes-rate):", flush=True)
    deciles = sorted(out_rows, key=lambda r: r["early_price"])
    bucket_size = max(n // 10, 1)
    for i in range(0, n, bucket_size):
        bucket = deciles[i:i + bucket_size]
        if not bucket:
            continue
        avg_pred = statistics.mean(b["early_price"] for b in bucket)
        avg_actual = statistics.mean(b["y"] for b in bucket)
        print(f"  predicted ~{avg_pred:.2f} -> actual yes-rate {avg_actual:.2f} (n={len(bucket)})", flush=True)

    # --- H2: late-window mispricing vs fair value ---
    n_dead = sum(1 for r in out_rows if not r["late_was_traded"])
    print(f"\nWindows where the literal last 1-min candle had no real trade "
          f"(had to walk back to find one): {n_dead}/{n} "
          f"({100*n_dead/n:.1f}%)", flush=True)
    if n_dead:
        avg_back = statistics.mean(r["minutes_before_close"] for r in out_rows if not r["late_was_traded"])
        print(f"  avg minutes before close we had to go back, for those: {avg_back:.2f}", flush=True)

    gaps = [r["gap_late_vs_fair"] for r in out_rows]
    print("\n=== H2: late-window quoted price vs fair value (ALL rows) ===", flush=True)
    print(f"mean gap: {statistics.mean(gaps):.4f}", flush=True)
    print(f"stdev gap: {statistics.pstdev(gaps):.4f}", flush=True)
    print(f"mean |gap|: {statistics.mean(abs(g) for g in gaps):.4f}", flush=True)

    traded_rows = [r for r in out_rows if r["late_was_traded"]]
    if traded_rows:
        t_gaps = [r["gap_late_vs_fair"] for r in traded_rows]
        print(f"\n=== H2: same, restricted to rows with a REAL traded late price (n={len(traded_rows)}) ===", flush=True)
        print(f"mean gap: {statistics.mean(t_gaps):.4f}", flush=True)
        print(f"stdev gap: {statistics.pstdev(t_gaps):.4f}", flush=True)
        print(f"mean |gap|: {statistics.mean(abs(g) for g in t_gaps):.4f}", flush=True)

    big_gaps = sorted(traded_rows, key=lambda r: -abs(r["gap_late_vs_fair"]))[:10]
    print("\nLargest 10 mispricings among REAL-traded late prices (|late_price - fair_value|):", flush=True)
    for r in big_gaps:
        print(f"  {r['ticker']}: late_price={r['late_price']:.3f} fair={r['fair_value']:.3f} "
              f"gap={r['gap_late_vs_fair']:.3f} mins_before_close={r['minutes_before_close']:.2f} result={r['result']}", flush=True)

    print("\nGap stats bucketed by how far before close the late price was taken:", flush=True)
    buckets = [(0, 1), (1, 3), (3, 6), (6, 100)]
    for lo, hi in buckets:
        bucket_rows = [r for r in traded_rows if lo <= r["minutes_before_close"] < hi]
        if not bucket_rows:
            continue
        bg = [r["gap_late_vs_fair"] for r in bucket_rows]
        label = f"{lo}-{hi}min" if hi < 100 else f"{lo}+min"
        print(f"  {label} before close (n={len(bucket_rows)}): mean gap={statistics.mean(bg):.4f} "
              f"mean |gap|={statistics.mean(abs(g) for g in bg):.4f}", flush=True)

if __name__ == "__main__":
    main()
