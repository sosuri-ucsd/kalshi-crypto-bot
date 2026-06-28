import csv, json, math, datetime

WINDOWS_CSV = "windows_backfill.csv"
GRID_CSV = "h1_comprehensive_features.csv"
BUCKET_5S_JSON = "kalshi_trades_5s.json"
BASE_K = 120
HALF = 12

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

n = len(rows)
print(f"Full sample: n={n}\n")

# ---- 1. Two-proportion z-test on the 0.45-0.55 "coin flip" zone ----
zone2 = [r for r in rows if 0.45 <= r["base"] <= 0.55]
up2 = [r for r in zone2 if r["mom"] > 0.01]
down2 = [r for r in zone2 if r["mom"] < -0.01]
p1 = sum(r["y"] for r in up2)/len(up2)
p2 = sum(r["y"] for r in down2)/len(down2)
n1, n2 = len(up2), len(down2)
p_pool = (sum(r["y"] for r in up2) + sum(r["y"] for r in down2)) / (n1+n2)
se = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
z = (p1-p2)/se
p_two_sided = 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
print("=== Test 1: two-proportion z-test, UP vs DOWN momentum in the 0.45-0.55 'coin flip' zone ===")
print(f"UP:   n={n1}, YES-rate={p1*100:.1f}%")
print(f"DOWN: n={n2}, YES-rate={p2*100:.1f}%")
print(f"z={z:.3f}, two-sided p={p_two_sided:.6f}\n")

# ---- 2. Point-biserial correlation between momentum and outcome, FULL sample (not just the tight zone) ----
mom = [r["mom"] for r in rows]
y = [r["y"] for r in rows]
mbar = sum(mom)/n
ybar = sum(y)/n
cov = sum((mom[i]-mbar)*(y[i]-ybar) for i in range(n))/n
sx = math.sqrt(sum((m-mbar)**2 for m in mom)/n)
sy = math.sqrt(sum((yi-ybar)**2 for yi in y)/n)
r_corr = cov/(sx*sy)
t_corr = r_corr*math.sqrt(n-2)/math.sqrt(1-r_corr**2)
p_corr = 2*(1-0.5*(1+math.erf(abs(t_corr)/math.sqrt(2))))
print("=== Test 2: point-biserial correlation, momentum vs outcome, FULL sample (n={}) ===".format(n))
print(f"r = {r_corr:.4f}   (t={t_corr:.2f}, p={p_corr:.6f})\n")

# ---- 3. AUC of momentum ALONE (no base price), full sample, for comparison to "0.7-0.8" expectation ----
def auc_score(p, y):
    n = len(p)
    combined = sorted(zip(p, y))
    ranks = [0.0]*n
    i=0
    while i<n:
        j=i
        while j<n and combined[j][0]==combined[i][0]: j+=1
        avg_rank=(i+1+j)/2
        for k in range(i,j): ranks[k]=avg_rank
        i=j
    npos=sum(y); nneg=n-npos
    rsum=sum(ranks[i] for i in range(n) if combined[i][1]==1)
    return (rsum - npos*(npos+1)/2)/(npos*nneg)

auc_mom_alone = auc_score(mom, y)
print(f"=== AUC of momentum ALONE (no price level at all) predicting outcome: {auc_mom_alone:.4f} ===")
