import csv, json, math, datetime

WINDOWS_CSV = "windows_backfill.csv"
GRID_CSV = "h1_comprehensive_features.csv"
BUCKET_5S_JSON = "kalshi_trades_5s.json"
BUCKET_SECONDS = 5
K = 600          # best backtest candidate from before
THRESH = 0.10

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

with open(WINDOWS_CSV) as f:
    wrows = {r["ticker"]: r for r in csv.DictReader(f) if r["covered_from_open"]=="True" and r["covered_to_close"]=="True"}
with open(GRID_CSV) as f:
    grows = {r["ticker"]: r for r in csv.DictReader(f)}
with open(BUCKET_5S_JSON) as f:
    buckets = json.load(f)

def entry_price_at(b, K):
    idx = K // BUCKET_SECONDS
    if idx >= len(b): idx = len(b)-1
    if idx < 0: return None
    return b[idx].get("price")

by_day = {}
for ticker, wr in wrows.items():
    g = grows.get(ticker)
    if not g: continue
    sig_raw = g.get(f"avg_price_t{K}s", "")
    if sig_raw == "": continue
    sig = float(sig_raw)
    b = buckets.get(ticker)
    if not b: continue
    entry = entry_price_at(b, K)
    if entry is None or entry <= 0.001 or entry >= 0.999: continue
    if abs(sig - 0.5) < THRESH: continue
    bet_yes = sig > 0.5
    cost = entry if bet_yes else (1-entry)
    y = int(g["y"])
    win = (y==1) if bet_yes else (y==0)
    profit = (1-cost) if win else -cost
    profit_costed = profit - 0.01
    day = wr["open_time"][:10]
    by_day.setdefault(day, []).append(profit_costed)

print("=== Per-day average costed profit/bet, K=600s thresh=0.10 (the best backtest candidate) ===")
day_means = []
for day in sorted(by_day):
    vals = by_day[day]
    m = sum(vals)/len(vals)
    day_means.append(m)
    print(f"  {day}: n={len(vals):4d} bets, mean_profit_costed={m:.4f}")

n_days = len(day_means)
grand_mean = sum(day_means)/n_days
sd_days = math.sqrt(sum((m-grand_mean)**2 for m in day_means)/(n_days-1))
print(f"\nAcross {n_days} days: mean={grand_mean:.4f}, between-day SD={sd_days:.4f}")

# sample-size calc: need n_days_required such that mean / (sd_days/sqrt(n)) >= z_crit (one-sided, since we have a directional prior)
for z, label in [(1.645, "90% one-sided"), (1.96, "95% two-sided")]:
    n_req = (z * sd_days / grand_mean) ** 2
    print(f"  Days needed for CI to exclude 0 at {label} (assuming same mean & SD hold): ~{math.ceil(n_req)} days")

# ---- same calc but for the momentum signal's Brier improvement (the other headline finding) ----
print("\n=== Per-day momentum Brier-improvement (base_price+momentum vs base_price alone) ===")
H2_CSV = "h1_h2_features.csv"
with open(H2_CSV) as f:
    h2rows = {r["ticker"]: r for r in csv.DictReader(f)}

def sigmoid(z):
    if z>=0:
        ez=math.exp(-z); return 1/(1+ez)
    ez=math.exp(z); return ez/(1+ez)

# crude per-day Brier diff using a GLOBAL fit (not re-fit per day, just to see day-to-day variability
# of the existing momentum signal's error reduction) -- fit once on all data, look at error reduction by day
BASE_K = 120
HALF = 12
data = []
for ticker, wr in wrows.items():
    g = grows.get(ticker)
    if not g: continue
    base_raw = g.get(f"avg_price_t{BASE_K}s","")
    if base_raw=="": continue
    base_price = float(base_raw)
    b = buckets.get(ticker)
    if not b or len(b) < BASE_K//5: continue
    early = b[:BASE_K//5]
    fh=[bb["price"] for bb in early[:HALF] if bb.get("price") is not None]
    sh=[bb["price"] for bb in early[HALF:] if bb.get("price") is not None]
    if not fh or not sh: continue
    momentum=(sum(sh)/len(sh))-(sum(fh)/len(fh))
    data.append({"day": wr["open_time"][:10], "base": base_price, "mom": momentum, "y": int(g["y"])})

# fit baseline and augmented on ALL data (in-sample, just to look at per-day error patterns of an already-validated effect)
def fit_simple(rows, cols):
    X=[[1.0]+[r[c] for c in cols] for r in rows]
    y=[r["y"] for r in rows]
    n,k=len(X),len(X[0])
    beta=[0.0]*k
    for _ in range(100):
        eta=[sum(X[i][j]*beta[j] for j in range(k)) for i in range(n)]
        mu=[sigmoid(e) for e in eta]
        w=[max(m*(1-m),1e-10) for m in mu]
        grad=[sum(X[i][j]*(y[i]-mu[i]) for i in range(n)) for j in range(k)]
        XtWX=[[0.0]*k for _ in range(k)]
        for i in range(n):
            wi=w[i]; Xi=X[i]
            for a in range(k):
                xa=Xi[a]*wi
                for b_ in range(k): XtWX[a][b_]+=xa*Xi[b_]
        # gauss-jordan inverse
        M=[row[:]+[1.0 if i==j else 0.0 for j in range(k)] for i,row in enumerate(XtWX)]
        for col in range(k):
            pr=max(range(col,k), key=lambda r: abs(M[r][col]))
            M[col],M[pr]=M[pr],M[col]
            piv=M[col][col]
            M[col]=[x/piv for x in M[col]]
            for r_ in range(k):
                if r_!=col:
                    f=M[r_][col]
                    if f!=0: M[r_]=[M[r_][kk]-f*M[col][kk] for kk in range(2*k)]
        inv=[row[k:] for row in M]
        delta=[sum(inv[a][b_]*grad[b_] for b_ in range(k)) for a in range(k)]
        beta=[beta[j]+delta[j] for j in range(k)]
        if max(abs(d) for d in delta)<1e-10: break
    return beta

beta_base = fit_simple(data, ["base"])
beta_aug = fit_simple(data, ["base","mom"])

by_day_brier = {}
for r in data:
    pA = sigmoid(beta_base[0]+beta_base[1]*r["base"])
    pB = sigmoid(beta_aug[0]+beta_aug[1]*r["base"]+beta_aug[2]*r["mom"])
    diff = (r["y"]-pA)**2 - (r["y"]-pB)**2   # positive = augmented better that obs
    by_day_brier.setdefault(r["day"], []).append(diff)

day_means2 = []
for day in sorted(by_day_brier):
    vals = by_day_brier[day]
    m = sum(vals)/len(vals)
    day_means2.append(m)
    print(f"  {day}: n={len(vals):4d}, mean_brier_improvement={m:.4f}")

n_days2 = len(day_means2)
gm2 = sum(day_means2)/n_days2
sd2 = math.sqrt(sum((m-gm2)**2 for m in day_means2)/(n_days2-1))
print(f"\nAcross {n_days2} days: mean={gm2:.5f}, between-day SD={sd2:.5f}")
for z, label in [(1.645, "90% one-sided"), (1.96, "95% two-sided")]:
    n_req = (z*sd2/gm2)**2
    print(f"  Days needed for CI to exclude 0 at {label}: ~{math.ceil(n_req)} days")
