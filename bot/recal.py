"""
bot/recal.py — Daily OLS recalibration of (b0, b1) per series.

Replicates the walk-forward logic from h11/h12: for each series, fit
    E[y] = b0 + b1 * avg_price_tKs
using all historical rows where the window's date < today (expanding window).

Usage
-----
    from bot.recal import fit_all_series
    coefs = fit_all_series()          # {series: (b0, b1)}

    # Or for a specific cutoff date (e.g. run as if it were 2025-03-01):
    coefs = fit_all_series(cutoff_date="2025-03-01")

Returns a dict {series: (b0, b1)}. Falls back to (0.0, 1.0) if a series
has fewer than 30 qualifying rows (same guard as h11).
"""

from __future__ import annotations

import csv
import datetime
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import FEATURES_CSV, SERIES_SPEC, ap_col, n_trades_col, MIN_TRADES_LIQUIDITY

# Minimum qualifying rows required before we trust the OLS fit
MIN_ROWS_FOR_FIT: int = 30

# Fallback coefficients if there is not enough data
_FALLBACK: Tuple[float, float] = (0.0, 1.0)


def _extract_date_from_ticker(ticker: str) -> Optional[datetime.date]:
    """
    Extract the settlement date from a Kalshi ticker.

    Expected formats:
        KXBTC15M-26JUN171445-50000    → 2026-06-17
        KXBTC15M-25DEC311445-50000    → 2025-12-31
    """
    # Match 6-digit DDMMMYY or DDMMMYYYY blocks (e.g. 26JUN17, 25DEC31)
    m = re.search(r'-(\d{2})([A-Z]{3})(\d{2,4})\d{4}-', ticker)
    if not m:
        return None
    day_str, mon_str, year_str = m.group(1), m.group(2), m.group(3)
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    try:
        day = int(day_str)
        month = month_map[mon_str]
        if len(year_str) == 2:
            year = 2000 + int(year_str)
        else:
            year = int(year_str)
        return datetime.date(year, month, day)
    except (KeyError, ValueError):
        return None


def _ols_1d(x: list[float], y: list[float]) -> Tuple[float, float]:
    """
    Ordinary Least Squares: y = b0 + b1 * x.
    Returns (b0, b1). Uses raw arithmetic to avoid numpy dependency.
    """
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return _FALLBACK

    b1 = (n * sxy - sx * sy) / denom
    b0 = (sy - b1 * sx) / n
    return (b0, b1)


def fit_all_series(
    cutoff_date: Optional[str] = None,
    features_csv: str = FEATURES_CSV,
    verbose: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """
    Fit OLS (b0, b1) per series from h1_comprehensive_features.csv.

    Args:
        cutoff_date:  ISO date string 'YYYY-MM-DD'. Rows from this date onward
                      are excluded (strictly-prior-data expanding window).
                      Defaults to today.
        features_csv: path to h1_comprehensive_features.csv.
        verbose:      print per-series fit stats to stdout.

    Returns:
        {series: (b0, b1)} — one entry per series in SERIES_SPEC.
        Falls back to (0.0, 1.0) for series with insufficient data.
    """
    if cutoff_date is None:
        cutoff = datetime.date.today()
    else:
        cutoff = datetime.date.fromisoformat(cutoff_date)

    # ── accumulate x, y per series ─────────────────────────────────────
    data: Dict[str, Tuple[list, list]] = {s: ([], []) for s in SERIES_SPEC}

    path = Path(features_csv)
    if not path.exists():
        print(f"[recal] WARNING: {features_csv} not found. Using fallback coefs.",
              file=sys.stderr)
        return {s: _FALLBACK for s in SERIES_SPEC}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            series = row.get("series", "")
            if series not in SERIES_SPEC:
                continue

            # Date filter: only use rows strictly before cutoff
            ticker = row.get("ticker", "")
            row_date = _extract_date_from_ticker(ticker)
            if row_date is None or row_date >= cutoff:
                continue

            # Liquidity filter (same as h12)
            n_col = n_trades_col(series)
            try:
                n = int(row.get(n_col, 0) or 0)
            except ValueError:
                n = 0
            if n < MIN_TRADES_LIQUIDITY:
                continue

            # AP value
            ap_c = ap_col(series)
            ap_str = row.get(ap_c, "")
            if not ap_str:
                continue
            try:
                ap = float(ap_str)
            except ValueError:
                continue

            # Outcome
            y_str = row.get("y", "")
            try:
                y = int(y_str)
            except ValueError:
                continue
            if y not in (0, 1):
                continue

            data[series][0].append(ap)
            data[series][1].append(float(y))

    # ── fit per series ──────────────────────────────────────────────────
    coefs: Dict[str, Tuple[float, float]] = {}
    for series, (xs, ys) in data.items():
        n = len(xs)
        if n < MIN_ROWS_FOR_FIT:
            if verbose:
                print(f"[recal] {series}: only {n} rows (need {MIN_ROWS_FOR_FIT}), using fallback")
            coefs[series] = _FALLBACK
            continue

        b0, b1 = _ols_1d(xs, ys)

        if verbose:
            y_mean = sum(ys) / len(ys)
            y_hat = [b0 + b1 * x for x in xs]
            ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(ys, y_hat))
            ss_tot = sum((yi - y_mean) ** 2 for yi in ys)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(
                f"[recal] {series}: n={n}, b0={b0:.4f}, b1={b1:.4f}, "
                f"R²={r2:.4f}, cutoff={cutoff}"
            )

        coefs[series] = (b0, b1)

    return coefs


# ---------------------------------------------------------------------------
# CLI: run standalone to print current coefficients
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit OLS coefficients per series")
    parser.add_argument("--cutoff", default=None,
                        help="ISO date YYYY-MM-DD (default: today)")
    parser.add_argument("--csv", default=FEATURES_CSV,
                        help="Path to h1_comprehensive_features.csv")
    args = parser.parse_args()

    coefs = fit_all_series(
        cutoff_date=args.cutoff,
        features_csv=args.csv,
        verbose=True,
    )
    print("\nCoefficients for use in StrategyBrain:")
    for series, (b0, b1) in coefs.items():
        print(f"  {series}: b0={b0:.6f}, b1={b1:.6f}")
