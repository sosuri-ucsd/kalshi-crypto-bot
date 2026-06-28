import csv, json, datetime

WINDOWS_CSV = "windows_backfill.csv"
GRID_CSV = "h1_comprehensive_features.csv"
BUCKET_5S_JSON = "kalshi_trades_5s.json"
BASE_K = 120
HALF = 12  # 12 buckets of 5s = 60s

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

with open(WINDOWS_CSV) as f:
    wrows = {r["ticker"]: r for r in csv.DictReader(f) if r["covered_from_open"]=="True" and r["covered_to_close"]=="True"}
with open(GRID_CSV) as f:
    grows = {r["ticker"]: r for r in csv.DictReader(f)}
with open(BUCKET_5S_JSON) as f:
    buckets = json.load(f)

rows = []
for ticker, wr in wrows.items():
    g = grows.get(ticker)
    if not g: continue
    b = buckets.get(ticker)
    if not b or len(b) < BASE_K//5: continue
    base_raw = g.get(f"avg_price_t{BASE_K}s","")
    if base_raw=="": continue
    base_price = float(base_raw)
    early = b[:BASE_K//5]
    fh = [bb["price"] for bb in early[:HALF] if bb.get("price") is not None]
    sh = [bb["price"] for bb in early[HALF:] if bb.get("price") is not None]
    if not fh or not sh: continue
    momentum = (sum(sh)/len(sh)) - (sum(fh)/len(fh))
    rows.append({"base": base_price, "mom": momentum, "y": int(g["y"])})

# Focus on the "near coin-flip" zone where the average price alone is least informative: 0.40-0.60
zone = [r for r in rows if 0.40 <= r["base"] <= 0.60]
up = [r for r in zone if r["mom"] > 0.01]
down = [r for r in zone if r["mom"] < -0.01]
flat = [r for r in zone if -0.01 <= r["mom"] <= 0.01]

def rate(lst):
    return sum(r["y"] for r in lst)/len(lst) if lst else float("nan")

print(f"Windows with base_price (120s avg) between 0.40 and 0.60 -- i.e. the model has barely any idea yet: n={len(zone)}")
print(f"  trending UP   (momentum > +0.01): n={len(up):4d}  YES-rate={rate(up)*100:.1f}%")
print(f"  flat          (momentum ~ 0):     n={len(flat):4d}  YES-rate={rate(flat)*100:.1f}%")
print(f"  trending DOWN (momentum < -0.01): n={len(down):4d}  YES-rate={rate(down)*100:.1f}%")

# tighter zone right at the coin-flip line
zone2 = [r for r in rows if 0.45 <= r["base"] <= 0.55]
up2 = [r for r in zone2 if r["mom"] > 0.01]
down2 = [r for r in zone2 if r["mom"] < -0.01]
print(f"\nEven tighter: base_price between 0.45 and 0.55 (truly looks like a coin flip): n={len(zone2)}")
print(f"  trending UP:   n={len(up2):4d}  YES-rate={rate(up2)*100:.1f}%")
print(f"  trending DOWN: n={len(down2):4d}  YES-rate={rate(down2)*100:.1f}%")
