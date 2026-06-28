import csv, json, datetime, ijson, sys, os, time

WINDOWS_CSV = "windows_backfill.csv"
TRADES_FILE = "kalshi_trades.json"
OUT_5S = "kalshi_trades_5s.json"
OUT_FIXEDTIME_CSV = "h1_fixedtime_features.csv"
BUCKET_SECONDS = 5
K_GRID = [15, 30, 60, 120, 300]  # fixed-time-window horizons, in seconds from window open
TIME_BUDGET = float(os.environ.get("TIME_BUDGET", "35"))  # seconds; checkpoint+exit before tool-call timeout

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def main():
    t0 = time.time()
    with open(WINDOWS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"]
    windows = {}
    for r in rows:
        windows[r["ticker"]] = {
            "series": r["series"],
            "open_ts": int(parse_time(r["open_time"]).timestamp()),
            "close_ts": int(parse_time(r["close_time"]).timestamp()),
        }
    print(f"Loaded {len(windows)} covered windows from {WINDOWS_CSV}", flush=True)

    bucket_out = {}
    fixedtime_rows = []
    done = set()
    if os.path.exists(OUT_5S):
        try:
            with open(OUT_5S) as f:
                bucket_out = json.load(f)
            print(f"Resumed: {len(bucket_out)} tickers already in {OUT_5S} checkpoint", flush=True)
        except Exception as e:
            print(f"Could not load existing {OUT_5S} checkpoint ({e}), starting fresh", flush=True)
            bucket_out = {}
    if os.path.exists(OUT_FIXEDTIME_CSV):
        try:
            with open(OUT_FIXEDTIME_CSV) as f:
                fixedtime_rows = list(csv.DictReader(f))
            print(f"Resumed: {len(fixedtime_rows)} rows already in {OUT_FIXEDTIME_CSV} checkpoint", flush=True)
        except Exception as e:
            print(f"Could not load existing {OUT_FIXEDTIME_CSV} checkpoint ({e}), starting fresh", flush=True)
            fixedtime_rows = []
    done = set(bucket_out.keys())
    n_already_done = len(done)
    n_target = len(windows)
    print(f"Already done: {n_already_done}/{n_target}. Time budget this run: {TIME_BUDGET}s", flush=True)

    n_seen = 0
    n_matched = 0
    n_zero_trade = 0
    total_trades_processed = 0
    hit_budget = False

    print("Streaming kalshi_trades.json via ijson (bounded memory, one ticker at a time)...", flush=True)
    with open(TRADES_FILE, "rb") as f:
        for ticker, trades in ijson.kvitems(f, ""):
            n_seen += 1
            if n_seen % 500 == 0:
                elapsed = time.time() - t0
                print(f"  ...seen {n_seen} tickers, matched(new) {n_matched}, elapsed {elapsed:.1f}s", flush=True)
            w = windows.get(ticker)
            if w is None:
                continue
            if ticker in done:
                continue  # already processed in a prior checkpointed run
            if time.time() - t0 > TIME_BUDGET:
                hit_budget = True
                break
            n_matched += 1
            open_ts, close_ts = w["open_ts"], w["close_ts"]
            window_secs = close_ts - open_ts

            if not trades:
                n_zero_trade += 1
                bucket_out[ticker] = []
                row = {"ticker": ticker, "series": w["series"], "n_trades_total": 0}
                for K in K_GRID:
                    row[f"avg_price_{K}s"] = ""
                    row[f"n_trades_{K}s"] = 0
                fixedtime_rows.append(row)
                continue

            parsed = []
            for t in trades:
                try:
                    ts = int(parse_time(t["created_time"]).timestamp())
                    price = float(t["yes_price_dollars"])
                except (KeyError, ValueError, TypeError):
                    continue
                parsed.append((ts, price))
            parsed.sort()
            total_trades_processed += len(parsed)

            # ---- 5s bucket series (for H2/H3/H4) ----
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
            bucket_out[ticker] = series

            # ---- fixed-time-window early features (for cross-coin confound fix) ----
            row = {"ticker": ticker, "series": w["series"], "n_trades_total": len(parsed)}
            for K in K_GRID:
                cutoff = open_ts + K
                subset = [p for ts_, p in parsed if ts_ < cutoff]
                if subset:
                    row[f"avg_price_{K}s"] = round(sum(subset) / len(subset), 4)
                    row[f"n_trades_{K}s"] = len(subset)
                else:
                    row[f"avg_price_{K}s"] = ""
                    row[f"n_trades_{K}s"] = 0
            fixedtime_rows.append(row)

            # free memory for this ticker before moving on
            del trades, parsed, series

    status = "STOPPED (hit time budget)" if hit_budget else "FULLY DONE (reached end of file)"
    print(f"\n{status}. n_seen_this_run={n_seen} tickers scanned, n_matched_new_this_run={n_matched}, "
          f"n_zero_trade_new={n_zero_trade}, total_trades_processed_new={total_trades_processed}", flush=True)
    print(f"Cumulative progress: {len(bucket_out)}/{n_target} covered windows done", flush=True)

    with open(OUT_5S, "w") as f:
        json.dump(bucket_out, f)
    print(f"Wrote {OUT_5S} ({len(bucket_out)} tickers, checkpoint)", flush=True)

    fieldnames = ["ticker", "series", "n_trades_total"] + \
                 [f"{p}{K}s" for K in K_GRID for p in ("avg_price_", "n_trades_")]
    with open(OUT_FIXEDTIME_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixedtime_rows)
    print(f"Wrote {OUT_FIXEDTIME_CSV} ({len(fixedtime_rows)} rows, checkpoint)", flush=True)
    if not hit_budget:
        for K in K_GRID:
            have = sum(1 for r in fixedtime_rows if r.get(f"avg_price_{K}s") != "")
            print(f"  K={K}s: {have}/{len(fixedtime_rows)} windows had >=1 trade within first {K}s", flush=True)

    print(f"\nThis run's elapsed: {time.time()-t0:.1f}s. RUN_AGAIN={'yes' if hit_budget else 'no'}", flush=True)

if __name__ == "__main__":
    main()
