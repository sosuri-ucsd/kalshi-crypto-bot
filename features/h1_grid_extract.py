import csv, json, datetime, ijson, time

WINDOWS_CSV = "windows_backfill.csv"
MARKETS_FILE = "backfill_markets.json"
TRADES_FILE = "kalshi_trades.json"
OUT_CSV = "h1_comprehensive_features.csv"

# Time boundaries (seconds elapsed since window open)
K_GRID = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 450, 600]
# Trade-count boundaries (first N trades, regardless of how long that took)
N_GRID = [5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

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
    print(f"Loaded {len(windows)} covered windows", flush=True)

    with open(MARKETS_FILE) as f:
        markets = json.load(f)
    print("Loaded markets file", flush=True)

    out_rows = []
    n_matched = 0
    n_seen = 0

    print("Streaming kalshi_trades.json...", flush=True)
    with open(TRADES_FILE, "rb") as f:
        for ticker, trades in ijson.kvitems(f, ""):
            n_seen += 1
            w = windows.get(ticker)
            if w is None:
                continue
            m = markets.get(ticker)
            if not m:
                continue
            result = m.get("result", "")
            y = 1 if result == "yes" else (0 if result == "no" else None)
            if y is None:
                continue
            n_matched += 1
            open_ts = w["open_ts"]

            parsed = []
            for t in trades:
                try:
                    ts = int(parse_time(t["created_time"]).timestamp())
                    price = float(t["yes_price_dollars"])
                except (KeyError, ValueError, TypeError):
                    continue
                parsed.append((ts, price))
            parsed.sort()
            if not parsed:
                continue

            row = {"ticker": ticker, "series": w["series"], "result": result, "y": y,
                   "n_trades_total": len(parsed)}

            # time-based boundaries
            for K in K_GRID:
                cutoff = open_ts + K
                subset = [p for ts_, p in parsed if ts_ < cutoff]
                if subset:
                    row[f"avg_price_t{K}s"] = round(sum(subset) / len(subset), 4)
                    row[f"n_trades_t{K}s"] = len(subset)
                else:
                    row[f"avg_price_t{K}s"] = ""
                    row[f"n_trades_t{K}s"] = 0

            # trade-count-based boundaries
            for N in N_GRID:
                subset = parsed[:N]
                if len(subset) < N:
                    row[f"avg_price_n{N}"] = ""
                    row[f"frac_window_n{N}"] = ""
                    continue
                avg_price = sum(p for _, p in subset) / len(subset)
                last_ts = subset[-1][0]
                window_secs = w["close_ts"] - open_ts
                frac = (last_ts - open_ts) / window_secs if window_secs > 0 else None
                row[f"avg_price_n{N}"] = round(avg_price, 4)
                row[f"frac_window_n{N}"] = round(frac, 4) if frac is not None else ""

            out_rows.append(row)
            del trades, parsed

    print(f"n_seen={n_seen}, n_matched={n_matched}", flush=True)

    fieldnames = ["ticker", "series", "result", "y", "n_trades_total"] + \
                 [f"{p}t{K}s" for K in K_GRID for p in ("avg_price_", "n_trades_")] + \
                 [f"{p}n{N}" for N in N_GRID for p in ("avg_price_", "frac_window_")]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {OUT_CSV}", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
