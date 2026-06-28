"""
Live, read-only Kalshi market data CLI -- orderbook, trades, market/series/event
lookups, exchange status. Uses the same signed-request auth as backfill_kalshi.py
and fetch_kalshi_trades.py (kalshi_key.key), so no new credentials or third-party
MCP/skill needed.

This is deliberately READ-ONLY: there is no order-placement / trade-execution
command here, only data lookups for research and signal monitoring.

Usage:
  python3 kalshi_live.py status
  python3 kalshi_live.py markets [--series KXBTC15M] [--status open] [--limit 50]
  python3 kalshi_live.py search "BTC"
  python3 kalshi_live.py market KXBTC15M-26JUN171445-45
  python3 kalshi_live.py orderbook KXBTC15M-26JUN171445-45 [--depth 10]
  python3 kalshi_live.py trades KXBTC15M-26JUN171445-45 [--limit 100]
  python3 kalshi_live.py series KXBTC15M
  python3 kalshi_live.py series_list
  python3 kalshi_live.py events [--series KXBTC15M]

All commands print JSON to stdout.
"""
import argparse, base64, json, sys, time
import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = "af5c74de-f8a7-47c7-a36f-ff09c57afb45"
KEY_PATH = "kalshi_key.key"
REST_BASE = "https://api.elections.kalshi.com"


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
                wait = int(r.headers.get("Retry-After", 3))
                time.sleep(wait)
                continue
            print(f"status {r.status_code} for {path} {params}: {r.text[:300]}", file=sys.stderr)
            return None
        except requests.exceptions.RequestException as e:
            print(f"request failed ({e})", file=sys.stderr)
        time.sleep(min(2 * (i + 1), 8))
    return None


def cmd_status(key, args):
    print(json.dumps(signed_get(key, "/trade-api/v2/exchange/status"), indent=2))


def cmd_markets(key, args):
    params = {"limit": args.limit}
    if args.series:
        params["series_ticker"] = args.series
    if args.status:
        params["status"] = args.status
    print(json.dumps(signed_get(key, "/trade-api/v2/markets", params), indent=2))


def cmd_search(key, args):
    body = signed_get(key, "/trade-api/v2/markets", {"limit": 200})
    if not body:
        print(json.dumps(None))
        return
    kw = args.keyword.lower()
    hits = [m for m in body.get("markets", []) if kw in m.get("ticker", "").lower() or kw in m.get("title", "").lower()]
    print(json.dumps(hits, indent=2))


def cmd_market(key, args):
    print(json.dumps(signed_get(key, f"/trade-api/v2/markets/{args.ticker}"), indent=2))


def cmd_orderbook(key, args):
    params = {"depth": args.depth} if args.depth else None
    print(json.dumps(signed_get(key, f"/trade-api/v2/markets/{args.ticker}/orderbook", params), indent=2))


def cmd_trades(key, args):
    params = {"ticker": args.ticker, "limit": args.limit}
    print(json.dumps(signed_get(key, "/trade-api/v2/markets/trades", params), indent=2))


def cmd_series(key, args):
    print(json.dumps(signed_get(key, f"/trade-api/v2/series/{args.series_ticker}"), indent=2))


def cmd_series_list(key, args):
    print(json.dumps(signed_get(key, "/trade-api/v2/series"), indent=2))


def cmd_events(key, args):
    params = {"series_ticker": args.series} if args.series else None
    print(json.dumps(signed_get(key, "/trade-api/v2/events", params), indent=2))


def main():
    p = argparse.ArgumentParser(description="Read-only live Kalshi market data CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)

    sp = sub.add_parser("markets"); sp.add_argument("--series"); sp.add_argument("--status"); sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(func=cmd_markets)

    sp = sub.add_parser("search"); sp.add_argument("keyword"); sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("market"); sp.add_argument("ticker"); sp.set_defaults(func=cmd_market)

    sp = sub.add_parser("orderbook"); sp.add_argument("ticker"); sp.add_argument("--depth", type=int, default=0)
    sp.set_defaults(func=cmd_orderbook)

    sp = sub.add_parser("trades"); sp.add_argument("ticker"); sp.add_argument("--limit", type=int, default=100)
    sp.set_defaults(func=cmd_trades)

    sp = sub.add_parser("series"); sp.add_argument("series_ticker"); sp.set_defaults(func=cmd_series)

    sub.add_parser("series_list").set_defaults(func=cmd_series_list)

    sp = sub.add_parser("events"); sp.add_argument("--series"); sp.set_defaults(func=cmd_events)

    args = p.parse_args()
    key = load_key(KEY_PATH)
    args.func(key, args)


if __name__ == "__main__":
    main()
