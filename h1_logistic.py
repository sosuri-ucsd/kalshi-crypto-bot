import csv, math

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)

def mat_inverse(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise ValueError("singular matrix")
        M[col], M[pivot_row] = M[pivot_row], M[col]
        pivot = M[col][col]
        M[col] = [x / pivot for x in M[col]]
        for r in range(n):
            if r != col:
                factor = M[r][col]
                if factor != 0.0:
                    M[r] = [M[r][k] - factor * M[col][k] for k in range(2 * n)]
    return [row[n:] for row in M]

def logistic_irls(X, y, max_iter=100, tol=1e-10):
    n, k = len(X), len(X[0])
    beta = [0.0] * k
    for _ in range(max_iter):
        eta = [sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]
        mu = [sigmoid(e) for e in eta]
        w = [max(m * (1 - m), 1e-10) for m in mu]
        grad = [sum(X[i][j] * (y[i] - mu[i]) for i in range(n)) for j in range(k)]
        XtWX = [[0.0] * k for _ in range(k)]
        for i in range(n):
            wi = w[i]
            Xi = X[i]
            for a in range(k):
                xa = Xi[a] * wi
                if xa == 0:
                    continue
                for b in range(k):
                    XtWX[a][b] += xa * Xi[b]
        XtWX_inv = mat_inverse(XtWX)
        delta = [sum(XtWX_inv[a][b] * grad[b] for b in range(k)) for a in range(k)]
        beta = [beta[j] + delta[j] for j in range(k)]
        if max(abs(d) for d in delta) < tol:
            break
    eta = [sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]
    mu = [sigmoid(e) for e in eta]
    w = [max(m * (1 - m), 1e-10) for m in mu]
    XtWX = [[0.0] * k for _ in range(k)]
    for i in range(n):
        wi = w[i]
        Xi = X[i]
        for a in range(k):
            xa = Xi[a] * wi
            for b in range(k):
                XtWX[a][b] += xa * Xi[b]
    cov = mat_inverse(XtWX)
    se = [math.sqrt(max(cov[j][j], 0.0)) for j in range(k)]
    ll = sum(y[i] * math.log(max(mu[i], 1e-15)) + (1 - y[i]) * math.log(max(1 - mu[i], 1e-15)) for i in range(n))
    return beta, se, ll, mu

def report(label, names, beta, se, ll, ll_null, n):
    pseudo_r2 = 1 - ll / ll_null
    print(f"\n=== {label} (n={n}) ===")
    print(f"{'term':18s} {'coef':>10s} {'se':>10s} {'z':>8s} {'p':>8s}")
    for nm, b, s in zip(names, beta, se):
        z = b / s if s > 0 else float("nan")
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        print(f"{nm:18s} {b:10.4f} {s:10.4f} {z:8.2f} {p:8.4f}")
    print(f"log-likelihood: {ll:.2f}  (null: {ll_null:.2f})  McFadden pseudo-R^2: {pseudo_r2:.4f}")

def auc_score(p, y):
    n = len(p)
    combined = sorted(zip(p, y))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for kx in range(i, j):
            ranks[kx] = avg_rank
        i = j
    n_pos = sum(y)
    n_neg = n - n_pos
    rank_sum_pos = sum(ranks[i] for i in range(n) if combined[i][1] == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def main():
    rows = list(csv.DictReader(open("h1_h2_features.csv")))
    n = len(rows)
    y = [int(r["y"]) for r in rows]
    p = [float(r["early_price"]) for r in rows]
    sigma = [float(r["sigma_annual"]) for r in rows]
    is_eth = [1.0 if r["series"] == "KXETH15M" else 0.0 for r in rows]
    is_sol = [1.0 if r["series"] == "KXSOL15M" else 0.0 for r in rows]

    ybar = sum(y) / n
    ll_null = n * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))

    auc = auc_score(p, y)
    print(f"AUC (early_price discriminating y=1 vs y=0): {auc:.4f}  (0.5=coinflip, 1.0=perfect separation)")

    # Model 1: simple logistic, y ~ early_price
    X1 = [[1.0, p[i]] for i in range(n)]
    beta1, se1, ll1, _ = logistic_irls(X1, y)
    report("Model 1: y ~ early_price", ["intercept", "early_price"], beta1, se1, ll1, ll_null, n)

    # Model 2: with series dummies (baseline=BTC) + sigma_annual control
    X2 = [[1.0, p[i], is_eth[i], is_sol[i], sigma[i]] for i in range(n)]
    beta2, se2, ll2, _ = logistic_irls(X2, y)
    report("Model 2: y ~ early_price + series(ETH,SOL) + sigma_annual",
           ["intercept", "early_price", "is_ETH", "is_SOL", "sigma_annual"], beta2, se2, ll2, ll_null, n)

if __name__ == "__main__":
    main()
