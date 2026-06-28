import csv, json, time, os, datetime
import requests

WINDOWS_CSV = "windows_backfill.csv"
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
SYMBOL_MAP = {"KXBTC15M": "BTCUSDT", "KXETH15M": "ETHUSDT", "KXSOL15M": "SOLUSDT"}
INTERVAL = "1s"
INTERVAL_SECONDS = 1
CHUNK_SECONDS = 1000 * INTERVAL_SECONDS  # 1000 candles per request max
ONLY_COVERED = True

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def load_windows():
    with open(WINDOWS_CSV) as f:
        reader = csv.DictReader(f)
        return list(reader)

def time_range_per_symbol(rows):
    ranges = {}
    for r in rows:
        if ONLY_COVERED and not (r["covered_from_open"] == "True" and r["covered_to_close"] == "True"):
            continue
        sym = SYMBOL_MAP.get(r["series"])
        if not sym:
            continue
        open_ts = int(parse_time(r["open_time"]).timestamp())
        close_ts = int(parse_time(r["close_time"]).timestamp())
        lo, hi = ranges.get(sym, (open_ts, close_ts))
        ranges[sym] = (min(lo, open_ts), max(hi, close_ts))
    return ranges

def fetch_klines(symbol, start_ms, end_ms, attempts=5):
    for i in range(attempts):
        try:
            params = {"symbol": symbol, "interval": INTERVAL, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
            r = requests.get(BINANCE_BASE, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 418):
                wait = int(r.headers.get("Retry-After", 5))
                print(f"  rate limited ({r.status_code}), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"  status {r.status_code}: {r.text[:200]}", flush=True)
        except requests.exceptions.RequestException as e:
            print(f"  request failed ({e})", flush=True)
        time.sleep(min(2 * (i + 1), 10))
    print(f"  giving up on {symbol} {start_ms}-{end_ms}", flush=True)
    return None

def main():
    rows = load_windows()
    ranges = time_range_per_symbol(rows)
    print(f"Fetching {INTERVAL} klines. Time ranges per symbol:", flush=True)
    for sym, (lo, hi) in ranges.items():
        print(f"  {sym}: {datetime.datetime.utcfromtimestamp(lo)} -> {datetime.datetime.utcfromtimestamp(hi)} "
              f"({(hi-lo)/3600:.1f} hours, ~{(hi-lo)//INTERVAL_SECONDS} candles)", flush=True)

    for sym, (lo, hi) in ranges.items():
        out_file = f"binance_{INTERVAL}_{sym}.json"
        all_klines = []
        if os.path.exists(out_file):
            with open(out_file) as f:
                all_klines = json.load(f)
            print(f"\n{sym}: resuming, already have {len(all_klines)} candles in {out_file}", flush=True)

        have_through = lo
        if all_klines:
            have_through = max(have_through, all_klines[-1][0] // 1000 + INTERVAL_SECONDS)

        cur = have_through
        print(f"\nFetching {sym} from {datetime.datetime.utcfromtimestamp(cur)} to {datetime.datetime.utcfromtimestamp(hi)}...", flush=True)
        req_count = 0
        while cur < hi:
            chunk_end = min(cur + CHUNK_SECONDS, hi)
            klines = fetch_klines(sym, cur * 1000, chunk_end * 1000)
            if klines is None:
                break
            if not klines:
                cur = chunk_end
                continue
            all_klines.extend(klines)
            cur = klines[-1][0] // 1000 + INTERVAL_SECONDS
            req_count += 1
            tmp_file = out_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(all_klines, f)
            os.replace(tmp_file, out_file)
            if req_count % 10 == 0:
                print(f"  {sym}: {len(all_klines)} candles so far, up to {datetime.datetime.utcfromtimestamp(cur)}", flush=True)
            time.sleep(0.15)

        tmp_file = out_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(all_klines, f)
        os.replace(tmp_file, out_file)
        print(f"{sym}: done, {len(all_klines)} candles saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
