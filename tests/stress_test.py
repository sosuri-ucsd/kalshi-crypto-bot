import csv, json, math, datetime, random

WINDOWS_CSV = "windows_backfill.csv"
GRID_CSV = "h1_comprehensive_features.csv"
BUCKET_5S_JSON = "kalshi_trades_5s.json"
BASE_K = 120

def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))

def sigmoid(z):
    if z>=0:
        ez=math.exp(-z); return 1/(1+ez)
    ez=math.exp(z); return ez/(1+ez)

def mat_inverse(A):
    n=len(A)
    M=[row[:]+[1.0 if i==j else 0.0 for j in range(n)] for i,row in enumerate(A)]
    for col in range(n):
        pr=max(range(col,n), key=lambda r: abs(M[r][col]))
        M[col],M[pr]=M[pr],M[col]
        piv=M[col][col]
        M[col]=[x/piv for x in M[col]]
        for r in range(n):
            if r!=col:
                f=M[r][col]
                if f!=0: M[r]=[M[r][k]-f*M[col][k] for k in range(2*n)]
    return [row[n:] for row in M]

def fit(X,y,max_iter=100,tol=1e-10):
    n,k=len(X),len(X[0])
    beta=[0.0]*k
    for _ in range(max_iter):
        eta=[sum(X[i][j]*beta[j] for j in range(k)) for i in range(n)]
        mu=[sigmoid(e) for e in eta]
        w=[max(m*(1-m),1e-10) for m in mu]
        grad=[sum(X[i][j]*(y[i]-mu[i]) for i in range(n)) for j in range(k)]
        XtWX=[[0.0]*k for _ in range(k)]
        for i in range(n):
            wi=w[i]; Xi=X[i]
            for a in range(k):
                xa=Xi[a]*wi
                for b in range(k): XtWX[a][b]+=xa*Xi[b]
        inv=mat_inverse(XtWX)
        delta=[sum(inv[a][b]*grad[b] for b in range(k)) for a in range(k)]
        beta=[beta[j]+delta[j] for j in range(k)]
        if max(abs(d) for d in delta)<tol: break
    return beta

def auc_score(p,y):
    n=len(p)
    combined=sorted(zip(p,y))
    ranks=[0.0]*n
    i=0
    while i<n:
        j=i
        while j<n and combined[j][0]==combined[i][0]: j+=1
        avg_rank=(i+1+j)/2
        for k in range(i,j): ranks[k]=avg_rank
        i=j
    npos=sum(y); nneg=n-npos
    if npos==0 or nneg==0: return float("nan")
    rsum=sum(ranks[i] for i in range(n) if combined[i][1]==1)
    return (rsum-npos*(npos+1)/2)/(npos*nneg)

def brier(p,y): return sum((pi-yi)**2 for pi,yi in zip(p,y))/len(p)
def mean(xs): return sum(xs)/len(xs)
def sd(xs):
    m=mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/max(len(xs)-1,1))

# ---- load ----
with open(WINDOWS_CSV) as f:
    wrows = {r["ticker"]: r for r in csv.DictReader(f) if r["covered_from_open"]=="True" and r["covered_to_close"]=="True"}
with open(GRID_CSV) as f:
    grows = {r["ticker"]: r for r in csv.DictReader(f)}
with open(BUCKET_5S_JSON) as f:
    buckets = json.load(f)

def momentum_at(b, K):
    n_buckets = K//5
    if len(b) < n_buckets: return None
    early = b[:n_buckets]
    half = n_buckets//2
    fh=[bb["price"] for bb in early[:half] if bb.get("price") is not None]
    sh=[bb["price"] for bb in early[half:] if bb.get("price") is not None]
    if not fh or not sh: return None
    return mean(sh)-mean(fh)

data=[]
for ticker, wr in wrows.items():
    g=grows.get(ticker)
    if not g: continue
    base_raw = g.get(f"avg_price_t{BASE_K}s","")
    if base_raw=="": continue
    b = buckets.get(ticker)
    if not b: continue
    mom = momentum_at(b, BASE_K)
    if mom is None: continue
    data.append({"ticker":ticker, "series":wr["series"], "day":wr["open_time"][:10],
                 "base":float(base_raw), "mom":mom, "y":int(g["y"]), "buckets":b})

print(f"n={len(data)}\n")

# ============ STRESS TEST 1: Leave-one-day-out cross validation ============
print("=== Stress test 1: leave-one-day-out (fit on 5 days, test on the held-out day, true generalization) ===")
days = sorted(set(r["day"] for r in data))
lodo_results=[]
for held_out in days:
    train=[r for r in data if r["day"]!=held_out]
    test=[r for r in data if r["day"]==held_out]
    Xb_tr=[[1.0,r["base"]] for r in train]; y_tr=[r["y"] for r in train]
    Xa_tr=[[1.0,r["base"],r["mom"]] for r in train]
    beta_b=fit(Xb_tr,y_tr); beta_a=fit(Xa_tr,y_tr)
    y_te=[r["y"] for r in test]
    p_b=[sigmoid(beta_b[0]+beta_b[1]*r["base"]) for r in test]
    p_a=[sigmoid(beta_a[0]+beta_a[1]*r["base"]+beta_a[2]*r["mom"]) for r in test]
    auc_b=auc_score(p_b,y_te); auc_a=auc_score(p_a,y_te)
    brier_b=brier(p_b,y_te); brier_a=brier(p_a,y_te)
    lodo_results.append({"day":held_out,"n":len(test),"auc_base":auc_b,"auc_aug":auc_a,
                          "brier_base":brier_b,"brier_aug":brier_a,"d_brier":brier_b-brier_a,"d_auc":auc_a-auc_b})
    print(f"  held-out {held_out}: n={len(test):4d}  AUC base={auc_b:.4f} aug={auc_a:.4f} (d={auc_a-auc_b:+.4f})  "
          f"Brier base={brier_b:.4f} aug={brier_a:.4f} (improvement={brier_b-brier_a:+.4f})")

n_better = sum(1 for r in lodo_results if r["d_brier"]>0)
print(f"\n  Momentum improved held-out Brier on {n_better}/{len(days)} days")
print(f"  Mean d(AUC) across held-out days: {mean([r['d_auc'] for r in lodo_results]):+.4f}")
print(f"  Mean Brier improvement across held-out days: {mean([r['d_brier'] for r in lodo_results]):+.4f} (sd={sd([r['d_brier'] for r in lodo_results]):.4f})")

# ============ STRESS TEST 2: permutation test (could this arise by pure chance?) ============
print("\n=== Stress test 2: permutation test on momentum-outcome correlation ===")
mom_vals=[r["mom"] for r in data]; y_vals=[r["y"] for r in data]
n=len(data)
def pearson(xs,ys):
    mx,my=mean(xs),mean(ys)
    cov=sum((xs[i]-mx)*(ys[i]-my) for i in range(len(xs)))/len(xs)
    sx=math.sqrt(sum((x-mx)**2 for x in xs)/len(xs)); sy=math.sqrt(sum((yv-my)**2 for yv in ys)/len(ys))
    return cov/(sx*sy)
r_obs = pearson(mom_vals, y_vals)
rng=random.Random(7)
n_perm=5000
perm_rs=[]
shuffled=y_vals[:]
for _ in range(n_perm):
    rng.shuffle(shuffled)
    perm_rs.append(pearson(mom_vals, shuffled))
n_exceed = sum(1 for pr in perm_rs if abs(pr)>=abs(r_obs))
print(f"  observed r={r_obs:.4f}; out of {n_perm} random shuffles, {n_exceed} matched or exceeded |r_obs|")
print(f"  empirical p-value = {n_exceed/n_perm:.5f} (this makes NO distributional assumptions -- direct check against pure chance)")

# ============ STRESS TEST 3: per-coin breakdown ============
print("\n=== Stress test 3: does momentum's edge hold for all three coins separately? ===")
for series in ["KXBTC15M","KXETH15M","KXSOL15M"]:
    sub=[r for r in data if r["series"]==series]
    mom_s=[r["mom"] for r in sub]; y_s=[r["y"] for r in sub]
    r_s=pearson(mom_s,y_s)
    auc_mom_only = auc_score(mom_s, y_s)
    print(f"  {series}: n={len(sub):4d}  r(momentum,y)={r_s:.4f}  AUC(momentum alone)={auc_mom_only:.4f}")

# ============ STRESS TEST 4: is the effect specific to K=120s, or does it hold at nearby boundaries? ============
print("\n=== Stress test 4: does the momentum effect hold at nearby boundaries (not just K=120 specifically)? ===")
for K in [60, 90, 120, 180, 300]:
    sub=[]
    for ticker, wr in wrows.items():
        g=grows.get(ticker)
        if not g: continue
        base_raw=g.get(f"avg_price_t{K}s","")
        if base_raw=="": continue
        b=buckets.get(ticker)
        if not b: continue
        mom=momentum_at(b,K)
        if mom is None: continue
        sub.append({"base":float(base_raw),"mom":mom,"y":int(g["y"])})
    if len(sub)<50: continue
    mom_k=[r["mom"] for r in sub]; y_k=[r["y"] for r in sub]
    r_k=pearson(mom_k,y_k)
    Xb=[[1.0,r["base"]] for r in sub]; Xa=[[1.0,r["base"],r["mom"]] for r in sub]
    yb=[r["y"] for r in sub]
    bb=fit(Xb,yb); ba=fit(Xa,yb)
    pb=[sigmoid(bb[0]+bb[1]*r["base"]) for r in sub]
    pa=[sigmoid(ba[0]+ba[1]*r["base"]+ba[2]*r["mom"]) for r in sub]
    auc_b=auc_score(pb,yb); auc_a=auc_score(pa,yb)
    print(f"  K={K:4d}s: n={len(sub):4d}  r(momentum,y)={r_k:+.4f}  in-sample AUC base={auc_b:.4f} -> +momentum={auc_a:.4f}")
