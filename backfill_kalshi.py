import base64, json, time, datetime, os
import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = "af5c74de-f8a7-47c7-a36f-ff09c57afb45"
KEY_PATH = "kalshi_key.key"
REST_BASE = "https://api.elections.kalshi.com"
SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M"]
MAX_PER_SERIES = 5000  # most recent N settled markets per series to pull candles for
OUT_MARKETS = "backfill_markets.json"
OUT_CANDLES = "backfill_candles.json"

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

def signed_get(private_key, path, params=None, attempts=5):
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
                print(f"  rate limited on {path}, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"  status {r.status_code} for {path} {params}: {r.text[:200]}", flush=True)
        except requests.exceptions.RequestException as e:
            print(f"  request failed ({e})", flush=True)
        time.sleep(min(2 * (i + 1), 8))
    print(f"  giving up on {path} {params}", flush=True)
    return None

def fetch_all_markets(private_key, series_ticker):
    markets = []
    cursor = None
    while True:
        params = {"series_ticker": series_ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        body = signed_get(private_key, "/trade-api/v2/markets", params)
        if body is None:
            break
        batch = body.get("markets", [])
        markets.extend(batch)
        cursor = body.get("cursor")
        print(f"  {series_ticker}: {len(markets)} markets so far (cursor={'yes' if cursor else 'none'})...", flush=True)
        if not cursor or not batch:
            break
    return markets

def fetch_candles(private_key, series_ticker, ticker, open_time, close_time):
    start_ts = int(open_time.timestamp())
    end_ts = int(close_time.timestamp()) + 60  # pad a little past close
    path = f"/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks"
    body = signed_get(private_key, path, {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1})
    if body is None:
        return None
    return body.get("candlesticks", [])

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def main():
    private_key = load_key(KEY_PATH)

    print("=== Step 1: enumerate markets per series ===", flush=True)
    all_markets = {}
    if os.path.exists(OUT_MARKETS):
        with open(OUT_MARKETS) as f:
            all_markets = json.load(f)
        print(f"Loaded {len(all_markets)} existing markets from {OUT_MARKETS} (new fetch will merge on top, never wipe)", flush=True)

    for s in SERIES:
        print(f"Fetching markets for {s}...", flush=True)
        ms = fetch_all_markets(private_key, s)
        print(f"  {s}: {len(ms)} total markets returned", flush=True)
        if not ms:
            existing_for_series = sum(1 for t in all_markets if t.startswith(s))
            print(f"  {s}: fetch returned nothing (likely network failure) -- keeping {existing_for_series} previously-saved markets for this series untouched", flush=True)
            continue
        for m in ms:
            all_markets[m["ticker"]] = m

    tmp_file = OUT_MARKETS + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(all_markets, f, indent=2)
    os.replace(tmp_file, OUT_MARKETS)
    print(f"Saved {len(all_markets)} markets total to {OUT_MARKETS}", flush=True)

    settled = {t: m for t, m in all_markets.items() if m.get("result")}
    print(f"\n{len(settled)} of {len(all_markets)} markets have a settlement result", flush=True)

    selected = {}
    for s in SERIES:
        series_settled = [m for t, m in settled.items() if t.startswith(s)]
        series_settled.sort(key=lambda m: m["close_time"], reverse=True)
        chosen = series_settled[:MAX_PER_SERIES]
        print(f"  {s}: {len(series_settled)} settled total, selecting {len(chosen)} most recent", flush=True)
        for m in chosen:
            selected[m["ticker"]] = m

    print(f"\n=== Step 2: fetch 1-min candlesticks for {len(selected)} selected markets ===", flush=True)

    candles = {}
    if os.path.exists(OUT_CANDLES):
        with open(OUT_CANDLES) as f:
            candles = json.load(f)
        print(f"Resuming: already have candles for {len(candles)} tickers", flush=True)

    todo = [t for t in selected if t not in candles]
    print(f"Fetching candles for {len(todo)} remaining tickers...", flush=True)

    for i, ticker in enumerate(todo):
        m = selected[ticker]
        s_match = next((s for s in SERIES if ticker.startswith(s)), None)
        if not s_match:
            continue
        open_time = parse_time(m["open_time"])
        close_time = parse_time(m["close_time"])
        cs = fetch_candles(private_key, s_match, ticker, open_time, close_time)
        if cs is None:
            continue
        candles[ticker] = cs
        time.sleep(0.15)
        tmp_file = OUT_CANDLES + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(candles, f, indent=2)
        os.replace(tmp_file, OUT_CANDLES)
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(todo)} done, checkpoint saved", flush=True)

    tmp_file = OUT_CANDLES + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(candles, f, indent=2)
    os.replace(tmp_file, OUT_CANDLES)
    print(f"\nSaved candles for {len(candles)} tickers to {OUT_CANDLES}", flush=True)

if __name__ == "__main__":
    main()
