import base64, csv, json, os, time, datetime
import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
try:
    import ijson
except ImportError:
    raise SystemExit(
        "Missing dependency 'ijson' (needed to read kalshi_trades.json without loading\n"
        "the whole multi-GB file into RAM -- that's what just caused the OOM kill).\n"
        "Install it once with:  pip3 install ijson\nThen re-run this script."
    )

KEY_ID = "af5c74de-f8a7-47c7-a36f-ff09c57afb45"
KEY_PATH = "kalshi_key.key"
REST_BASE = "https://api.elections.kalshi.com"
WINDOWS_CSV = "windows_backfill.csv"
OUT_FILE = "kalshi_trades.json"

def load_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def sign(private_key, text):
    sig = private_key.sign(
        text.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")

def signed_get(private_key, path, params=None, attempts=8):
    for i in range(attempts):
        try:
            ts = str(int(time.time() * 1000))
            sig = sign(private_key, ts + "GET" + path)
            headers = {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts}
            r = requests.get(REST_BASE + path, headers=headers, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                print(f"  rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"  status {r.status_code} for {path} {params}: {r.text[:200]}", flush=True)
        except requests.exceptions.RequestException as e:
            print(f"  request failed ({type(e).__name__}: {e})", flush=True)
        time.sleep(min(3 * (i + 1), 20))
    print(f"  giving up on {path} {params}", flush=True)
    return None

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def repair_tail_if_needed(path):
    """Byte-level only, never loads the file into memory. If the file was left
    truncated by a crash mid-append, trim back to the last complete entry."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(size - 1)
        last_byte = f.read(1)
    if last_byte == b"}":
        return
    print(f"  {path} looks truncated (crash mid-write) -- auto-repairing...", flush=True)
    chunk = min(size, 200 * 1024 * 1024)
    with open(path, "rb") as f:
        f.seek(size - chunk)
        tail = f.read()
    idx = tail.rfind(b"]")
    if idx == -1:
        raise RuntimeError(f"could not find a safe repair point in {path}; refusing to guess")
    abs_pos = (size - chunk) + idx
    with open(path, "r+b") as f:
        f.truncate(abs_pos + 1)
        f.seek(0, 2)
        f.write(b"}")
    print("  repaired (the one in-progress ticker at crash time was dropped, everything before it is intact)", flush=True)

def existing_ticker_keys(path):
    """Stream-parse the {ticker: [...]} file to collect just the SET of ticker
    keys already present, one entry at a time -- never holds the whole file's
    trade data in memory at once. This replaces the old load_or_repair(), which
    did a full json.load() on the entire file every resume and is exactly what
    OOM-killed the process once the file passed a few GB."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    keys = set()
    with open(path, "rb") as f:
        for key, _value in ijson.kvitems(f, ""):
            keys.add(key)
    return keys

def stream_summary(path):
    """Final stats computed the same streaming way -- one ticker's trades in
    memory at a time, not the whole file."""
    n_tickers = total_trades = zero_trade = 0
    with open(path, "rb") as f:
        for _key, value in ijson.kvitems(f, ""):
            n_tickers += 1
            total_trades += len(value)
            if len(value) == 0:
                zero_trade += 1
    return n_tickers, total_trades, zero_trade

def append_ticker(path, ticker, trades):
    """Append one ticker's trades into the existing {ticker: [...]} JSON file
    in place, without rewriting the whole file. Disk cost is O(this ticker's
    data), not O(file size) -- critical once the file is tens of GB."""
    entry = ("," + json.dumps(ticker) + ":" + json.dumps(trades) + "}").encode()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "wb") as f:
            f.write(b"{" + entry[1:])
        return
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.seek(size - 1)
        last_byte = f.read(1)
        if last_byte != b"}":
            raise ValueError(f"{path} does not end with '}}' (ends with {last_byte!r}) -- refusing to append blindly, run repair_tail_if_needed first")
        f.seek(size - 1)
        f.write(entry)
        f.flush()
        os.fsync(f.fileno())

def fetch_trades_for_ticker(private_key, ticker, min_ts, max_ts, max_pages=50):
    trades = []
    cursor = None
    seen_cursors = set()
    for page in range(max_pages):
        params = {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        body = signed_get(private_key, "/trade-api/v2/markets/trades", params)
        if body is None:
            break
        batch = body.get("trades", [])
        trades.extend(batch)
        cursor = body.get("cursor")
        if page > 0:
            print(f"    {ticker}: page {page+1}, {len(trades)} trades so far...", flush=True)
        if not cursor or not batch:
            break
        if cursor in seen_cursors:
            print(f"    {ticker}: cursor not advancing ({cursor}), stopping to avoid infinite loop", flush=True)
            break
        seen_cursors.add(cursor)
    else:
        print(f"    {ticker}: hit max_pages={max_pages} cap, stopping early", flush=True)
    return trades

def main():
    private_key = load_key(KEY_PATH)

    with open(WINDOWS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["covered_from_open"] == "True" and r["covered_to_close"] == "True"]

    repair_tail_if_needed(OUT_FILE)
    done_tickers = existing_ticker_keys(OUT_FILE)
    print(f"Resuming: already have trades for {len(done_tickers)} tickers", flush=True)

    todo = [r for r in rows if r["ticker"] not in done_tickers]
    print(f"Fetching trades for {len(todo)} remaining tickers (of {len(rows)} covered windows)...", flush=True)

    for i, r in enumerate(todo):
        ticker = r["ticker"]
        open_ts = int(parse_time(r["open_time"]).timestamp())
        close_ts = int(parse_time(r["close_time"]).timestamp())
        print(f"[{i+1}/{len(todo)}] fetching {ticker}...", flush=True)
        trades = fetch_trades_for_ticker(private_key, ticker, open_ts, close_ts)
        print(f"  -> {len(trades)} trades", flush=True)
        time.sleep(1.0)
        append_ticker(OUT_FILE, ticker, trades)
        if (i + 1) % 5 == 0:
            print(f"  ...{i+1}/{len(todo)} done, checkpoint saved", flush=True)

    n_tickers, total_trades, zero_trade_windows = stream_summary(OUT_FILE)
    print(f"\nDone. {n_tickers} tickers, {total_trades} total trades, "
          f"{zero_trade_windows} windows with zero trades.", flush=True)

if __name__ == "__main__":
    main()
