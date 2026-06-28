import base64, json, time, glob, os
import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = "af5c74de-f8a7-47c7-a36f-ff09c57afb45"
KEY_PATH = "kalshi_key.key"
REST_BASE = "https://api.elections.kalshi.com"
SETTLEMENTS_PATH = "settlements.json"

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

def get_market(private_key, ticker, retries=4, backoff=2):
    path = f"/trade-api/v2/markets/{ticker}"
    for attempt in range(retries):
        try:
            ts = str(int(time.time() * 1000))
            sig = sign(private_key, ts + "GET" + path)
            headers = {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts}
            r = requests.get(REST_BASE + path, headers=headers, timeout=10)
            return r.status_code, r.json()
        except requests.exceptions.RequestException as e:
            print(f"  {ticker}: request failed ({e}), retrying in {backoff}s...")
            time.sleep(backoff)
    print(f"  {ticker}: giving up after {retries} attempts (will retry next run)")
    return None, None

def find_all_tickers_seen():
    tickers = set()
    for fname in glob.glob("data/kalshi_*.jsonl"):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ticker = rec.get("raw", {}).get("msg", {}).get("market_ticker")
                if ticker:
                    tickers.add(ticker)
    return tickers

def load_existing_settlements():
    if os.path.exists(SETTLEMENTS_PATH):
        with open(SETTLEMENTS_PATH) as f:
            return json.load(f)
    return {}

def main():
    private_key = load_key(KEY_PATH)

    all_tickers = find_all_tickers_seen()
    existing = load_existing_settlements()
    new_tickers = sorted(all_tickers - set(existing.keys()))

    print(f"Tickers seen total: {len(all_tickers)}")
    print(f"Already have settlements: {len(existing)}")
    print(f"New tickers to fetch: {len(new_tickers)}")

    for t in new_tickers:
        status, body = get_market(private_key, t)
        if status is None:
            continue
        existing[t] = body
        print(t, "| status:", status)
        with open(SETTLEMENTS_PATH, "w") as f:
            json.dump(existing, f, indent=2)

    print(f"\nSaved {len(existing)} total settlements to {SETTLEMENTS_PATH}")

if __name__ == "__main__":
    main()
