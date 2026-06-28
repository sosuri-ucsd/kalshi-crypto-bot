import csv, json, math, statistics, datetime
from bisect import bisect_right

WINDOWS_CSV = "windows_backfill.csv"
MARKETS_FILE = "backfill_markets.json"
KALSHI_5S_FILE = "kalshi_trades_5s.json"
SYMBOL_MAP = {"KXBTC15M": "BTCUSDT", "KXETH15M": "ETHUSDT", "KXSOL15M": "SOLUSDT"}
OUT_CSV = "h2_subminute_features.csv"
ANNUAL_SECONDS = 365 * 24 * 3600
# Binance 1s data resolution - tau still shouldn't go below this, but it's a
# much smaller floor than the 60s we needed with 1-min data.
TAU_FLOOR_SECONDS = 1
TAU_FLOOR = TAU_FLOOR_SECONDS / ANNUAL_SECONDS

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

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

def find_last_traded_bucket(series):
    for b in reversed(series):
        if b.get("traded") and b.get("price") is not None:
            return b
    return None

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
    with open(KALSHI_5S_FILE) as f:
        kalshi_5s = json.load(f)

    binance = {}
    for sym in set(SYMBOL_MAP.values()):
        binance[sym] = load_binance_1s(sym)
        print(f"Loaded {len(binance[sym][0])} 1s Binance candles for {sym}", flush=True)

    with open(WINDOWS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"]

    out_rows = []
    skipped = 0
    no_late_trade = 0

    for r in rows:
        ticker = r["ticker"]
        series_name = r["series"]
        sym = SYMBOL_MAP.get(series_name)
        m = markets.get(ticker)
        bucket_series = kalshi_5s.get(ticker)
        if not m or not bucket_series or not sym:
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

        late_bucket = find_last_traded_bucket(bucket_series)
        if late_bucket is None:
            no_late_trade += 1
            skipped += 1
            continue
        late_price = late_bucket["price"]
        late_ts = late_bucket["bucket_ts"]

        close_ts = int(parse_time(r["close_time"]).timestamp())
        open_ts = int(parse_time(r["open_time"]).timestamp())

        times, closes = binance[sym]
        spot = price_at_or_before(times, closes, late_ts)
        if spot is None or spot <= 0 or strike <= 0:
            skipped += 1
            continue

        # Only use data up to late_ts (the evaluation point), never up to close_ts -
        # using returns between late_ts and close_ts would be look-ahead bias. Same
        # class of bug as the Polymarket same-window contamination issue.
        window_closes = [c for t, c in zip(times, closes) if open_ts <= t <= late_ts]
        log_rets = []
        for i in range(1, len(window_closes)):
            if window_closes[i - 1] > 0:
                log_rets.append(math.log(window_closes[i] / window_closes[i - 1]))
        if len(log_rets) >= 2:
            sigma_per_sec = statistics.pstdev(log_rets)
        else:
            sigma_per_sec = 0.0005 / math.sqrt(60)
        sigma_annual = sigma_per_sec * math.sqrt(ANNUAL_SECONDS)

        tau = max((close_ts - late_ts) / ANNUAL_SECONDS, TAU_FLOOR)
        d = math.log(spot / strike) / (sigma_annual * math.sqrt(tau) + 1e-12)
        fair_value = norm_cdf(d)
        if "less" in strike_type:
            fair_value = 1 - fair_value

        gap = late_price - fair_value

        out_rows.append({
            "ticker": ticker, "series": series_name, "result": result, "y": y,
            "late_price": late_price, "spot_at_late": spot, "strike": strike,
            "strike_type": strike_type, "sigma_annual": round(sigma_annual, 4),
            "seconds_before_close": round(close_ts - late_ts, 1),
            "fair_value": round(fair_value, 4), "gap_late_vs_fair": round(gap, 4),
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows to {OUT_CSV} "
          f"({skipped} skipped, {no_late_trade} of those had zero trades all window)", flush=True)

    gaps = [r["gap_late_vs_fair"] for r in out_rows]
    print("\n=== H2 (sub-minute): late-window quoted price vs fair value ===", flush=True)
    print(f"mean gap: {statistics.mean(gaps):.4f}", flush=True)
    print(f"stdev gap: {statistics.pstdev(gaps):.4f}", flush=True)
    print(f"mean |gap|: {statistics.mean(abs(g) for g in gaps):.4f}", flush=True)

    print("\nGap stats bucketed by seconds before close of the late trade:", flush=True)
    buckets = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 180), (180, 100000)]
    for lo, hi in buckets:
        bucket_rows = [r for r in out_rows if lo <= r["seconds_before_close"] < hi]
        if not bucket_rows:
            continue
        bg = [r["gap_late_vs_fair"] for r in bucket_rows]
        label = f"{lo}-{hi}s" if hi < 100000 else f"{lo}s+"
        print(f"  {label} before close (n={len(bucket_rows)}): mean gap={statistics.mean(bg):.4f} "
              f"mean |gap|={statistics.mean(abs(g) for g in bg):.4f}", flush=True)

    big_gaps = sorted(out_rows, key=lambda r: -abs(r["gap_late_vs_fair"]))[:10]
    print("\nLargest 10 mispricings:", flush=True)
    for r in big_gaps:
        print(f"  {r['ticker']}: late_price={r['late_price']:.3f} fair={r['fair_value']:.3f} "
              f"gap={r['gap_late_vs_fair']:.3f} secs_before_close={r['seconds_before_close']:.1f} result={r['result']}", flush=True)

if __name__ == "__main__":
    main()
