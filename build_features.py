import json, csv
from datetime import datetime

def parse_iso(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()

with open('settlements.json') as f:
    settlements = json.load(f)

data = {t: {'trades': [], 'tickers': []} for t in settlements}

import glob
files = sorted(glob.glob('data/kalshi_*.jsonl'))
for fname in files:
    with open(fname) as f:
        for line in f:
            d = json.loads(line)
            raw = d['raw']
            msg = raw.get('msg', {})
            ticker = msg.get('market_ticker')
            if ticker not in data:
                continue
            if raw.get('type') == 'trade':
                data[ticker]['trades'].append(msg)
            elif raw.get('type') == 'ticker':
                data[ticker]['tickers'].append(msg)

# Measured worst-case lag between window open and our first captured message,
# given the OLD 30s-poll rollover loop. Anything within this is "covered enough"
# for first-30s analysis. Tighten this once ws_kalshi.py uses boundary-aware polling.
COVERAGE_TOLERANCE_S = 35

rows = []
for ticker, market_wrap in settlements.items():
    m = market_wrap['market']
    open_ts = parse_iso(m['open_time'])
    close_ts = parse_iso(m['close_time'])
    result = m.get('result', '')
    asset = 'BTC' if 'BTC' in ticker else ('ETH' if 'ETH' in ticker else 'SOL')

    trades = sorted(data[ticker]['trades'], key=lambda x: x['ts_ms'])
    ticks = sorted(data[ticker]['tickers'], key=lambda x: x['ts_ms'])

    all_ts_ms = [t['ts_ms'] for t in trades] + [t['ts_ms'] for t in ticks]
    first_seen_ms = min(all_ts_ms) if all_ts_ms else None
    lag_after_open_s = (first_seen_ms / 1000 - open_ts) if first_seen_ms is not None else None
    covered_from_open = bool(first_seen_ms is not None and lag_after_open_s <= COVERAGE_TOLERANCE_S)

    first30_cut_ms = (open_ts + 30) * 1000
    last2min_cut_ms = (close_ts - 120) * 1000
    close_ms = close_ts * 1000

    trades_first30 = [t for t in trades if t['ts_ms'] <= first30_cut_ms]
    n_trades_first30 = len(trades_first30) if covered_from_open else None
    vol_first30 = sum(float(t['count_fp']) for t in trades_first30) if covered_from_open else None
    yes_frac_first30 = None
    vwap_yes_first30 = None
    if covered_from_open and vol_first30:
        yes_vol = sum(float(t['count_fp']) for t in trades_first30 if t['taker_side'] == 'yes')
        yes_frac_first30 = yes_vol / vol_first30
        vwap_yes_first30 = sum(float(t['yes_price_dollars']) * float(t['count_fp']) for t in trades_first30) / vol_first30

    ticks_before_close = [t for t in ticks if t['ts_ms'] <= close_ms]
    last_tick = ticks_before_close[-1] if ticks_before_close else None
    last_mid_before_close = ((float(last_tick['yes_bid_dollars']) + float(last_tick['yes_ask_dollars'])) / 2) if last_tick else None

    ticks_2min_before = [t for t in ticks if t['ts_ms'] <= last2min_cut_ms]
    last_tick_2min = ticks_2min_before[-1] if ticks_2min_before else None
    last_mid_2min_before_close = ((float(last_tick_2min['yes_bid_dollars']) + float(last_tick_2min['yes_ask_dollars'])) / 2) if last_tick_2min else None

    rows.append({
        'ticker': ticker,
        'asset': asset,
        'result': result,
        'covered_from_open': covered_from_open,
        'lag_after_open_s': round(lag_after_open_s, 1) if lag_after_open_s is not None else '',
        'floor_strike': m.get('floor_strike'),
        'expiration_value': m.get('expiration_value'),
        'n_trades_first30': n_trades_first30 if n_trades_first30 is not None else '',
        'vol_first30': round(vol_first30, 2) if vol_first30 is not None else '',
        'yes_frac_first30': round(yes_frac_first30, 4) if yes_frac_first30 is not None else '',
        'vwap_yes_first30': round(vwap_yes_first30, 4) if vwap_yes_first30 is not None else '',
        'last_mid_2min_before_close': round(last_mid_2min_before_close, 4) if last_mid_2min_before_close is not None else '',
        'last_mid_before_close': round(last_mid_before_close, 4) if last_mid_before_close is not None else '',
        'n_trades_total': len(trades),
        'n_ticker_msgs_total': len(ticks),
    })

fieldnames = list(rows[0].keys())
with open('windows.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to windows.csv\n")
covered = [r for r in rows if r['covered_from_open']]
print(f"Fully covered windows (safe for H1 first-30s analysis): {len(covered)} / {len(rows)}\n")
for r in rows:
    print(r)
