import json, csv, datetime, os

MARKETS_FILE = "backfill_markets.json"
CANDLES_FILE = "backfill_candles.json"
OUT_CSV = "windows_backfill.csv"
SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M"]
COVERAGE_TOLERANCE_MIN = 2  # candle must exist within this many minutes of open/close to count as "covered"

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def get_candle_ts(candle):
    for key in ("end_period_ts", "end_ts", "ts", "timestamp", "period_start"):
        if key in candle:
            return candle[key]
    return None

def get_strike(m):
    parts = []
    if m.get("floor_strike") is not None:
        parts.append(f"floor={m['floor_strike']}")
    if m.get("cap_strike") is not None:
        parts.append(f"cap={m['cap_strike']}")
    if m.get("custom_strike"):
        parts.append(f"custom={m['custom_strike']}")
    return ";".join(parts) if parts else ""

def main():
    with open(MARKETS_FILE) as f:
        markets = json.load(f)
    with open(CANDLES_FILE) as f:
        candles_by_ticker = json.load(f)

    for ticker, cs in candles_by_ticker.items():
        if cs:
            print("=== Sample candle (for field-name verification) ===", flush=True)
            print("ticker:", ticker, flush=True)
            print(json.dumps(cs[0], indent=2)[:1500], flush=True)
            print("=====================================================", flush=True)
            break

    rows = []
    counts = {"total": 0, "covered_from_open": 0, "covered_to_close": 0, "covered_both": 0, "no_candles": 0}

    for ticker, cs in candles_by_ticker.items():
        m = markets.get(ticker)
        if not m:
            continue
        series = next((s for s in SERIES if ticker.startswith(s)), "UNKNOWN")
        open_time = parse_time(m["open_time"])
        close_time = parse_time(m["close_time"])
        open_ts = int(open_time.timestamp())
        close_ts = int(close_time.timestamp())

        counts["total"] += 1

        if not cs:
            counts["no_candles"] += 1
            rows.append({
                "ticker": ticker, "series": series,
                "open_time": m["open_time"], "close_time": m["close_time"],
                "result": m.get("result", ""), "strike": get_strike(m),
                "n_candles": 0, "first_candle_gap_min": "", "last_candle_gap_min": "",
                "covered_from_open": False, "covered_to_close": False,
            })
            continue

        timestamps = [t for t in (get_candle_ts(c) for c in cs) if t is not None]
        if not timestamps:
            covered_from_open = covered_to_close = False
            first_gap = last_gap = ""
        else:
            first_ts, last_ts = min(timestamps), max(timestamps)
            first_gap = round((first_ts - open_ts) / 60, 2)
            last_gap = round((close_ts - last_ts) / 60, 2)
            covered_from_open = first_gap <= COVERAGE_TOLERANCE_MIN
            covered_to_close = last_gap <= COVERAGE_TOLERANCE_MIN

        if covered_from_open:
            counts["covered_from_open"] += 1
        if covered_to_close:
            counts["covered_to_close"] += 1
        if covered_from_open and covered_to_close:
            counts["covered_both"] += 1

        rows.append({
            "ticker": ticker, "series": series,
            "open_time": m["open_time"], "close_time": m["close_time"],
            "result": m.get("result", ""), "strike": get_strike(m),
            "n_candles": len(cs), "first_candle_gap_min": first_gap, "last_candle_gap_min": last_gap,
            "covered_from_open": covered_from_open, "covered_to_close": covered_to_close,
        })

    tmp_file = OUT_CSV + ".tmp"
    with open(tmp_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_file, OUT_CSV)

    print(f"\nWrote {len(rows)} windows to {OUT_CSV}", flush=True)
    print(f"Total: {counts['total']}", flush=True)
    print(f"No candles at all: {counts['no_candles']}", flush=True)
    print(f"Covered from open (candle within {COVERAGE_TOLERANCE_MIN}min of open): {counts['covered_from_open']}", flush=True)
    print(f"Covered to close (candle within {COVERAGE_TOLERANCE_MIN}min of close): {counts['covered_to_close']}", flush=True)
    print(f"Covered both ends (usable window): {counts['covered_both']}", flush=True)

    by_series = {}
    for r in rows:
        s = r["series"]
        by_series.setdefault(s, {"total": 0, "covered": 0})
        by_series[s]["total"] += 1
        if r["covered_from_open"] and r["covered_to_close"]:
            by_series[s]["covered"] += 1
    print("\nBy series:", flush=True)
    for s, c in by_series.items():
        print(f"  {s}: {c['covered']}/{c['total']} covered", flush=True)

if __name__ == "__main__":
    main()
