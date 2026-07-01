"""
bot/executor.py — Real order placement on Kalshi.

This module contains ONE function: place_order(). It places a limit buy
order on Kalshi via the REST API. It does NOT make trading decisions —
that is solely brain.py's job.

SECURITY RULES (enforced by design):
  - The brain (brain.py) NEVER imports or calls this module.
  - Only run_live.py calls place_order(), and only after brain.decide()
    returns action='BET'.
  - place_order() does NOT read KEY_PATH — the caller must load the key
    and pass it in. This prevents any code path from accidentally
    re-reading the credentials file.
  - All orders are limit orders (never market orders). This ensures the
    fill price is at most what we decided is acceptable.
  - client_order_id includes a timestamp so duplicate detection works.

Usage (called from run_live.py, NOT by you directly):
    from bot.executor import place_order

    result = place_order(
        private_key  = private_key,        # already-loaded RSA key
        ticker       = "KXBTC15M-26JUL031500-50000",
        side         = "yes",              # 'yes' or 'no'
        contracts    = 3,
        price_cents  = 55,                 # limit price in cents [1, 99]
    )
    # result: {"order_id": "...", "status": "resting"/"filled", ...}
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

from .config import KEY_ID, REST_BASE

# --------------------------------------------------------------------------- #
# Auth (same pattern as kalshi_live.py and run_live.py)
# --------------------------------------------------------------------------- #

def _sign(private_key, text: str) -> str:
    sig = private_key.sign(
        text.encode("utf-8"),
        asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def _auth_headers(private_key, method: str, path: str) -> dict:
    ts  = str(int(time.time() * 1000))
    sig = _sign(private_key, ts + method + path)
    return {
        "KALSHI-ACCESS-KEY":       KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type":            "application/json",
    }


# --------------------------------------------------------------------------- #
# place_order — the only function callers should use
# --------------------------------------------------------------------------- #

def place_order(
    private_key,
    ticker:       str,
    side:         str,   # 'yes' or 'no'
    contracts:    int,
    price_cents:  int,   # limit price in cents [1, 99]
    dry_run:      bool = False,
) -> dict:
    """
    Place a limit buy order on Kalshi.

    Args:
        private_key:  loaded RSA private key (from _load_key())
        ticker:       market ticker, e.g. "KXBTC15M-26JUL031500-50000"
        side:         'yes' or 'no'
        contracts:    number of contracts (int ≥ 1)
        price_cents:  limit price in cents [1, 99]
        dry_run:      if True, builds the payload but does NOT send. Returns
                      {"dry_run": True, "payload": <what would be sent>}.

    Returns:
        dict with keys from Kalshi API:
          - order_id:    str  (Kalshi's order UUID)
          - status:      str  ('resting', 'filled', 'canceled')
          - filled_count: int (contracts actually filled)
          - remaining:   int
          - error:       str  (only present if request failed)

    Raises:
        ValueError for bad arguments.
        requests.HTTPError if Kalshi returns a non-2xx status.
    """
    # --- input validation ---
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got: {side!r}")
    if contracts < 1:
        raise ValueError(f"contracts must be >= 1, got: {contracts}")
    if not (1 <= price_cents <= 99):
        raise ValueError(f"price_cents must be in [1, 99], got: {price_cents}")

    # --- build payload ---
    # Kalshi v2 order format:
    #   action:     'buy'  (we always buy; selling uses a separate cancel/sell flow)
    #   type:       'limit'  (never market — too risky)
    #   side:       'yes' or 'no'
    #   yes_price / no_price: the limit price in cents for the relevant side
    #   count:      number of contracts
    #   client_order_id: unique ID for idempotency (timestamp + uuid prefix)
    client_order_id = f"brain-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    payload: dict = {
        "ticker":           ticker,
        "action":           "buy",
        "type":             "limit",
        "side":             side,
        "count":            contracts,
        "client_order_id":  client_order_id,
    }
    if side == "yes":
        payload["yes_price"] = price_cents
    else:
        payload["no_price"] = price_cents

    if dry_run:
        return {"dry_run": True, "payload": payload}

    # --- sign and send ---
    path    = "/trade-api/v2/portfolio/orders"
    headers = _auth_headers(private_key, "POST", path)

    resp = requests.post(
        REST_BASE + path,
        headers = headers,
        data    = json.dumps(payload),
        timeout = 10,
    )

    if not resp.ok:
        # Parse Kalshi error body if possible
        try:
            err_body = resp.json()
        except Exception:
            err_body = {"raw": resp.text}
        return {
            "error":       f"HTTP {resp.status_code}",
            "error_body":  err_body,
            "ticker":      ticker,
            "side":        side,
            "contracts":   contracts,
            "price_cents": price_cents,
        }

    body  = resp.json()
    order = body.get("order", body)  # Kalshi wraps in {"order": {...}}

    return {
        "order_id":     order.get("order_id", ""),
        "status":       order.get("status", ""),
        "filled_count": order.get("filled_count", 0),
        "remaining":    order.get("remaining_count", contracts),
        "yes_price":    order.get("yes_price", price_cents),
        "no_price":     order.get("no_price",  100 - price_cents),
        "ticker":       ticker,
        "side":         side,
        "contracts":    contracts,
        "client_order_id": client_order_id,
    }


# --------------------------------------------------------------------------- #
# cancel_order — for take-profit exits
# --------------------------------------------------------------------------- #

def cancel_order(private_key, order_id: str) -> dict:
    """
    Cancel an open resting order (e.g. for take-profit exit logic).

    Returns Kalshi API response dict, or {'error': ...} on failure.
    """
    path    = f"/trade-api/v2/portfolio/orders/{order_id}"
    headers = _auth_headers(private_key, "DELETE", path)

    resp = requests.delete(REST_BASE + path, headers=headers, timeout=10)

    if not resp.ok:
        try:
            err_body = resp.json()
        except Exception:
            err_body = {"raw": resp.text}
        return {"error": f"HTTP {resp.status_code}", "error_body": err_body}

    return resp.json()


# --------------------------------------------------------------------------- #
# get_order — check fill status
# --------------------------------------------------------------------------- #

def get_order(private_key, order_id: str) -> dict:
    """Check the current status of an order."""
    path    = f"/trade-api/v2/portfolio/orders/{order_id}"
    ts      = str(int(time.time() * 1000))
    sig     = _sign(private_key, ts + "GET" + path)
    headers = {
        "KALSHI-ACCESS-KEY":       KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }
    resp = requests.get(REST_BASE + path, headers=headers, timeout=10)
    if not resp.ok:
        return {"error": f"HTTP {resp.status_code}"}
    body  = resp.json()
    order = body.get("order", body)
    return {
        "order_id":     order.get("order_id", order_id),
        "status":       order.get("status", ""),
        "filled_count": order.get("filled_count", 0),
        "remaining":    order.get("remaining_count", 0),
    }
