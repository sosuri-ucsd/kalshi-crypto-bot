import requests, datetime, base64, json
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

KEY_ID = "af5c74de-f8a7-47c7-a36f-ff09c57afb45"
KEY_PATH = "kalshi_key.key"

def load_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

def sign(private_key, text):
    sig = private_key.sign(
        text.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")

private_key = load_key(KEY_PATH)

# 1. Filter series for the BTC/ETH/SOL 15-min markets specifically
r1 = requests.get("https://api.elections.kalshi.com/trade-api/v2/series?category=Crypto")
all_series = r1.json().get("series", [])
print("=== CRYPTO SERIES MATCHING 15M / BTC / ETH / SOL ===")
for s in all_series:
    t = s.get("ticker", "")
    title = s.get("title", "")
    tags = s.get("tags", [])
    if "15M" in t.upper() or "15m" in title.lower() or any(x in title for x in ["BTC", "ETH", "SOL", "Bitcoin", "Ethereum", "Solana"]):
        print(t, "|", title, "|", tags)

# 2. Signed check against PRODUCTION (your key's actual environment)
ts = str(int(datetime.datetime.now().timestamp() * 1000))
path = "/trade-api/v2/portfolio/balance"
sig = sign(private_key, ts + "GET" + path)
headers = {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts}
r2 = requests.get("https://api.elections.kalshi.com" + path, headers=headers)
print("=== BALANCE (signed, production) ===")
print(r2.status_code, r2.text)

# 3. Find the currently active BTC 15m market + pull its order book
ts = str(int(datetime.datetime.now().timestamp() * 1000))
path = "/trade-api/v2/markets"
sig = sign(private_key, ts + "GET" + path)
headers = {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts}
r3 = requests.get("https://api.elections.kalshi.com" + path, headers=headers,
                   params={"series_ticker": "KXBTC15M", "status": "open"})
markets = r3.json().get("markets", [])
print("=== ACTIVE KXBTC15M MARKET(S) ===")
for m in markets:
    print(m["ticker"], "| open:", m["open_time"], "| close:", m["close_time"], "| yes_bid:", m.get("yes_bid"), "| yes_ask:", m.get("yes_ask"))

if markets:
    ticker = markets[0]["ticker"]
    ts2 = str(int(datetime.datetime.now().timestamp() * 1000))
    book_path = f"/trade-api/v2/markets/{ticker}/orderbook"
    sig2 = sign(private_key, ts2 + "GET" + book_path)
    headers2 = {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-SIGNATURE": sig2, "KALSHI-ACCESS-TIMESTAMP": ts2}
    r4 = requests.get("https://api.elections.kalshi.com" + book_path, headers=headers2)
    print("=== ORDERBOOK for", ticker, "===")
    print(r4.status_code, r4.text[:1500])
