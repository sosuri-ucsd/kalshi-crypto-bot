import asyncio, websockets, json, time
from datetime import datetime

SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]
URL = "wss://stream.binance.com:9443/stream?streams=" + "/".join(f"{s}@trade" for s in SYMBOLS)

def fname():
    return f"data/binance_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.jsonl"

async def heartbeat(counts, outfile):
    while True:
        await asyncio.sleep(60)
        size_mb = 0
        try:
            import os
            size_mb = os.path.getsize(outfile) / 1e6
        except FileNotFoundError:
            pass
        print(f"[heartbeat] msgs={counts['n']}  file={outfile}  size={size_mb:.2f}MB", flush=True)

async def capture():
    outfile = fname()
    counts = {"n": 0}
    hb_task = asyncio.create_task(heartbeat(counts, outfile))

    while True:
        try:
            async with websockets.connect(URL, ping_interval=20, ping_timeout=20) as ws:
                print(f"Connected. Writing to {outfile}", flush=True)
                with open(outfile, "a") as f:
                    async for message in ws:
                        counts["n"] += 1
                        f.write(json.dumps({"recv_ts": time.time(), "raw": json.loads(message)}) + "\n")
        except Exception as e:
            print(f"Disconnected ({e}), reconnecting in 3s...", flush=True)
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(capture())
