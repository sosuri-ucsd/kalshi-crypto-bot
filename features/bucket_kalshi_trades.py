import csv, json, datetime
try:
    import ijson
except ImportError:
    raise SystemExit(
        "Missing dependency 'ijson' (needed to stream kalshi_trades.json instead of\n"
        "json.load()'ing the whole multi-GB file into RAM).\n"
        "Install it once with:  pip3 install ijson\nThen re-run this script."
    )

WINDOWS_CSV = "windows_backfill.csv"
TRADES_FILE = "kalshi_trades.json"
OUT_FILE = "kalshi_trades_5s.json"
BUCKET_SECONDS = 5

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def bucket_one(trades, open_ts, close_ts):
    parsed = []
    for t in trades:
        try:
            ts = int(parse_time(t["created_time"]).timestamp())
            price = float(t["yes_price_dollars"])
        except (KeyError, ValueError, TypeError):
            continue
        parsed.append((ts, price))
    parsed.sort()

    n_buckets = (close_ts - open_ts) // BUCKET_SECONDS + 1
    series = []
    last_price = None
    ti = 0
    for b in range(n_buckets):
        bucket_start = open_ts + b * BUCKET_SECONDS
        bucket_end = bucket_start + BUCKET_SECONDS
        traded_this_bucket = False
        while ti < len(parsed) and parsed[ti][0] < bucket_end:
            if parsed[ti][0] >= bucket_start:
                last_price = parsed[ti][1]
                traded_this_bucket = True
            ti += 1
        series.append({"bucket_ts": bucket_start, "price": last_price, "traded": traded_this_bucket})
    return series, len(parsed)

def main():
    with open(WINDOWS_CSV) as f:
        rows = {r["ticker"]: r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"}

    # Output (bucketed series) is orders of magnitude smaller than the raw
    # trades file -- ~180 buckets x ~10k windows x a tiny dict is tens of MB,
    # totally fine to hold in memory. The INPUT is what's huge; that's what
    # we stream instead of json.load()-ing whole.
    out = {}
    n_zero_trade_windows = 0
    total_trades = 0
    seen = set()

    with open(TRADES_FILE, "rb") as f:
        for ticker, trades in ijson.kvitems(f, ""):
            r = rows.get(ticker)
            if r is None:
                continue
            seen.add(ticker)
            open_ts = int(parse_time(r["open_time"]).timestamp())
            close_ts = int(parse_time(r["close_time"]).timestamp())
            if not trades:
                n_zero_trade_windows += 1
                out[ticker] = []
                continue
            series, n_parsed = bucket_one(trades, open_ts, close_ts)
            total_trades += n_parsed
            out[ticker] = series

    # Windows not yet present in kalshi_trades.json (fetch incomplete, or
    # genuinely never fetched) -- flag separately from true zero-trade windows.
    missing = [t for t in rows if t not in seen]
    if missing:
        print(f"WARNING: {len(missing)} windows in {WINDOWS_CSV} have NO entry yet in {TRADES_FILE} "
              f"(fetch_kalshi_trades.py likely still running / incomplete). Treating them as zero-trade "
              f"for now -- re-run this script after the fetch finishes for a complete dataset.")
        for t in missing:
            out[t] = []
            n_zero_trade_windows += 1

    with open(OUT_FILE, "w") as f:
        json.dump(out, f)
    print(f"Wrote bucketed ({BUCKET_SECONDS}s) series for {len(out)} tickers to {OUT_FILE}")
    print(f"Total raw trades bucketed: {total_trades}")
    print(f"Windows with zero trades (incl. {len(missing)} not-yet-fetched): {n_zero_trade_windows}/{len(rows)}")

if __name__ == "__main__":
    main()
