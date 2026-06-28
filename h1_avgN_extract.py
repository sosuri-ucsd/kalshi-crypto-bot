import csv, json, datetime

WINDOWS_CSV = "windows_backfill.csv"
MARKETS_FILE = "backfill_markets.json"
TRADES_FILE = "kalshi_trades.json"
OUT_CSV = "h1_avgN_features.csv"
N_GRID = [10, 25, 50, 100, 250, 500]  # primary N = 100, rest are a robustness curve

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def main():
    with open(WINDOWS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"]
    with open(MARKETS_FILE) as f:
        markets = json.load(f)

    print("Loading kalshi_trades.json (~1.9GB - this is the slow part, give it a minute)...", flush=True)
    with open(TRADES_FILE) as f:
        trades_by_ticker = json.load(f)
    print("Loaded.", flush=True)

    out_rows = []
    skipped = 0
    for r in rows:
        ticker = r["ticker"]
        m = markets.get(ticker)
        if not m:
            skipped += 1
            continue
        result = m.get("result", "")
        y = 1 if result == "yes" else (0 if result == "no" else None)
        if y is None:
            skipped += 1
            continue

        trades = trades_by_ticker.get(ticker, [])
        if not trades:
            skipped += 1
            continue

        parsed = []
        for t in trades:
            try:
                ts = parse_time(t["created_time"]).timestamp()
                price = float(t["yes_price_dollars"])
            except (KeyError, ValueError, TypeError):
                continue
            parsed.append((ts, price))
        parsed.sort()
        if not parsed:
            skipped += 1
            continue

        open_ts = parse_time(r["open_time"]).timestamp()
        close_ts = parse_time(r["close_time"]).timestamp()
        window_secs = close_ts - open_ts

        row = {"ticker": ticker, "series": r["series"], "result": result, "y": y,
               "n_trades_total": len(parsed)}
        for N in N_GRID:
            subset = parsed[:N]
            if len(subset) < N:
                row[f"avg_price_{N}"] = ""
                row[f"frac_window_{N}"] = ""
                continue
            avg_price = sum(p for _, p in subset) / len(subset)
            last_ts_in_subset = subset[-1][0]
            frac_window = (last_ts_in_subset - open_ts) / window_secs if window_secs > 0 else None
            row[f"avg_price_{N}"] = round(avg_price, 4)
            row[f"frac_window_{N}"] = round(frac_window, 4) if frac_window is not None else ""
        out_rows.append(row)

    fieldnames = ["ticker", "series", "result", "y", "n_trades_total"] + \
                 [f"{p}{N}" for N in N_GRID for p in ("avg_price_", "frac_window_")]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows to {OUT_CSV} ({skipped} skipped: missing market/result/trades)", flush=True)
    for N in N_GRID:
        have = sum(1 for r in out_rows if r.get(f"avg_price_{N}") != "")
        print(f"  N={N}: {have}/{len(out_rows)} windows had >= {N} trades available", flush=True)

if __name__ == "__main__":
    main()
