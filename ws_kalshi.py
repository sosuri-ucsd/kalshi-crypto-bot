import asyncio, base64, json, time, os, datetime
import requests
import websockets
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = "af5c74de-f8a7-47c7-a36f-ff09c57afb45"
KEY_PATH = "kalshi_key.key"
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
REST_BASE = "https://api.elections.kalshi.com"
SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M"]
LOG_DIR = "data"

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

def get_with_retry(url, headers, params, attempts=3):
    for i in range(attempts):
        try:
            return requests.get(url, headers=headers, params=params, timeout=8)
        except requests.exceptions.RequestException as e:
            wait = min(2 * (i + 1), 5)
            print(f"Request failed ({e}), retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Failed after retries: " + url)

def get_active_tickers(private_key):
    tickers = []
    for s in SERIES:
        ts = str(int(time.time() * 1000))
        path = "/trade-api/v2/markets"
        sig = sign(private_key, ts + "GET" + path)
        headers = {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts}
        r = get_with_retry(REST_BASE + path, headers, {"series_ticker": s, "status": "open"})
        for m in r.json().get("markets", []):
            tickers.append(m["ticker"])
    return tickers

async def get_active_tickers_safe(private_key, timeout):
    return await asyncio.wait_for(asyncio.to_thread(get_active_tickers, private_key), timeout=timeout)

async def rollover_loop(private_key, ws, subscribed, next_id):
    while True:
        now = datetime.datetime.now(datetime.timezone.utc)
        next_min = ((now.minute // 15) + 1) * 15
        if next_min == 60:
            next_boundary = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        else:
            next_boundary = now.replace(minute=next_min, second=0, microsecond=0)

        sleep_s = max((next_boundary - now).total_seconds() - 3, 0)
        await asyncio.sleep(sleep_s)

        deadline = next_boundary + datetime.timedelta(seconds=60)
        caught_series = set()
        while datetime.datetime.now(datetime.timezone.utc) < deadline and len(caught_series) < len(SERIES):
            remaining = (deadline - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            try:
                active = set(await get_active_tickers_safe(private_key, timeout=max(remaining, 1)))
                new = active - subscribed
                if new:
                    sub = {
                        "id": next_id[0],
                        "cmd": "subscribe",
                        "params": {"channels": ["ticker", "trade", "orderbook_delta"], "market_tickers": list(new)},
                    }
                    await ws.send(json.dumps(sub))
                    next_id[0] += 1
                    subscribed |= new
                    for t in new:
                        for s in SERIES:
                            if t.startswith(s):
                                caught_series.add(s)
                    print("Rolled over to new markets:", new, flush=True)
            except asyncio.TimeoutError:
                print(f"Rollover check timed out for boundary {next_boundary} (will retry)", flush=True)
            except Exception as e:
                print("Rollover check failed (will retry):", e, flush=True)
            if len(caught_series) < len(SERIES):
                await asyncio.sleep(1)

        missing = set(SERIES) - caught_series
        if missing:
            print(f"Rollover GAVE UP for boundary {next_boundary} after 60s deadline, missing series: {missing}", flush=True)

async def capture():
    private_key = load_key(KEY_PATH)
    os.makedirs(LOG_DIR, exist_ok=True)
    fname = os.path.join(LOG_DIR, f"kalshi_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.jsonl")
    f = open(fname, "a")
    print("Logging to", fname, flush=True)

    while True:
        try:
            tickers = await get_active_tickers_safe(private_key, timeout=30)
            print("Starting tickers:", tickers, flush=True)

            ts = str(int(time.time() * 1000))
            path = "/trade-api/ws/v2"
            sig = sign(private_key, ts + "GET" + path)
            headers = {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts}

            async with websockets.connect(WS_URL, additional_headers=headers) as ws:
                subscribed = set(tickers)
                next_id = [1]
                if tickers:
                    sub = {
                        "id": next_id[0],
                        "cmd": "subscribe",
                        "params": {"channels": ["ticker", "trade", "orderbook_delta"], "market_tickers": tickers},
                    }
                    await ws.send(json.dumps(sub))
                    next_id[0] += 1
                print("Subscribed. Listening... (Ctrl+C to stop)", flush=True)

                roll_task = asyncio.create_task(rollover_loop(private_key, ws, subscribed, next_id))
                try:
                    async for message in ws:
                        recv_ts = time.time()
                        data = json.loads(message)
                        f.write(json.dumps({"recv_ts": recv_ts, "raw": data}) + "\n")
                        f.flush()
                        print(data.get("type"), data.get("msg", {}).get("market_ticker", ""), flush=True)
                finally:
                    roll_task.cancel()
        except Exception as e:
            print(f"Connection issue ({e}), reconnecting in 3s...", flush=True)
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(capture())
