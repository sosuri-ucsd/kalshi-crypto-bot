"""
bot/sizing.py — Fee and Kelly sizing functions, synced exactly with h12.

Both functions are unit-tested via smoke test in run_live.py startup.
"""

import math


def kalshi_fee(contracts: int, price_dollars: float) -> float:
    """
    Real Kalshi taker fee formula.

    fee = ceil(0.07 × C × P × (1 − P) × 100) / 100   [dollars]

    where C = contracts, P = price in dollars [0, 1].
    Fee is highest at P=0.50 ($0.035 × C) and approaches zero at 0 or 1.

    Args:
        contracts:     number of contracts purchased
        price_dollars: entry price in [0, 1] (i.e., cents / 100)

    Returns:
        fee in dollars (rounded up to nearest cent)
    """
    if contracts <= 0:
        return 0.0
    p = max(0.0, min(1.0, price_dollars))
    raw = 0.07 * contracts * p * (1.0 - p) * 100.0
    return math.ceil(raw) / 100.0


def kelly_fraction(p_hat: float, entry: float, side: str) -> float:
    """
    Half-Kelly fraction for a binary Kalshi bet, synced with h12.

    For a YES bet:
        You pay `entry` dollars per contract; receive $1 if YES, $0 if NO.
        Optimal Kelly: f* = (p_hat − entry) / (1 − entry)
        Applied at half-Kelly: f = 0.5 × f*

    For a NO bet:
        You pay `(1 − entry)` ... but Kalshi prices NO separately.
        We receive $1 if NO (i.e., p(NO) = 1 − p_hat).
        Optimal Kelly: f* = ((1 − p_hat) − no_entry) / (1 − no_entry)
        where no_entry = `entry` argument (the NO ask price).
        Applied at half-Kelly: f = 0.5 × f*

    Args:
        p_hat:  model-implied P(YES) in [0, 1]
        entry:  ask price for the chosen side in [0, 1] dollars
        side:   'yes' or 'no'

    Returns:
        Half-Kelly fraction ≥ 0 (0 if no positive edge).
    """
    if side == "yes":
        edge = p_hat - entry
        if edge <= 0.0 or entry >= 1.0:
            return 0.0
        return 0.5 * edge / (1.0 - entry)

    elif side == "no":
        # p(NO winning) = 1 - p_hat; we pay `entry` (the NO ask)
        p_no = 1.0 - p_hat
        edge = p_no - entry
        if edge <= 0.0 or entry >= 1.0:
            return 0.0
        return 0.5 * edge / (1.0 - entry)

    return 0.0


# ---------------------------------------------------------------------------
# Quick self-check (run: python3 -m bot.sizing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # fee at P=0.50, 10 contracts → 0.07*10*0.5*0.5*100 = 17.5 → ceil = 18 → $0.18
    fee = kalshi_fee(10, 0.50)
    assert fee == 0.18, f"fee check failed: {fee}"

    # fee at P=0.30, 5 contracts → 0.07*5*0.3*0.7*100 = 7.35 → ceil = 8 → $0.08
    fee2 = kalshi_fee(5, 0.30)
    assert fee2 == 0.08, f"fee check 2 failed: {fee2}"

    # Kelly YES: p_hat=0.65, entry=0.50 → edge=0.15, denom=0.50 → f*=0.30, half=0.15
    kf = kelly_fraction(0.65, 0.50, "yes")
    assert abs(kf - 0.15) < 1e-9, f"kelly YES check failed: {kf}"

    # Kelly NO: p_hat=0.35, entry=0.52 → p_no=0.65, edge=0.13, denom=0.48 → f*=0.2708, half=0.1354
    kf_no = kelly_fraction(0.35, 0.52, "no")
    expected = 0.5 * (0.65 - 0.52) / (1 - 0.52)
    assert abs(kf_no - expected) < 1e-9, f"kelly NO check failed: {kf_no}"

    print("sizing.py: all checks passed")
