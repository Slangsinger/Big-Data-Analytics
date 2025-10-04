# author: aleks — SERIAL 
import argparse, time, random, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from kneed import KneeLocator
plt.switch_backend("Agg")

def load_data():
    df = pd.read_csv("proteins.csv", usecols=["protid","enzyme","hydrofob","sequence"])
    X = df[["enzyme","hydrofob"]].to_numpy(float)
    seq_len = df["sequence"].str.len().to_numpy()
    protid = df["protid"].to_numpy()
    return X, seq_len, protid

def ensure_outdir(p): os.makedirs(p, exist_ok=True)
def save_fig(fig, path): fig.savefig(path, bbox_inches="tight"); plt.close(fig)

def assign_labels_chunked(X, C, batch=200_000):
    n, k = X.shape[0], C.shape[0]
    labels = np.empty(n, np.int32)
    c2 = np.sum(C**2, axis=1)
    for s in range(0, n, batch):
        e = min(s+batch, n)
        Xb = X[s:e]
        x2 = np.sum(Xb**2, axis=1, keepdims=True)
        d2 = x2 + c2[None,:] - 2.0*(Xb @ C.T)
        labels[s:e] = np.argmin(d2, axis=1)
    return labels

def recompute_centroids(X, y, k, rng):
    C = np.zeros((k, X.shape[1]))
    for cid in range(k):
        m = (y == cid)
        C[cid] = X[rng.integers(0, X.shape[0])] if not np.any(m) else X[m].mean(axis=0)
    return C

def inertia_wcss(X, y, C, batch=200_000):
    n = X.shape[0]; c2 = np.sum(C**2, axis=1); tot = 0.0
    for s in range(0, n, batch):
        e = min(s+batch, n); Xb = X[s:e]
        x2 = np.sum(Xb**2, axis=1, keepdims=True)
        d2 = x2 + c2[None,:] - 2.0*(Xb @ C.T)
        tot += np.min(d2, axis=1).sum()
    return float(tot)

def kmeans_serial(X, k, seed=42, max_iter=100, tol=1e-4, batch=200_000):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(X.shape[0], size=k, replace=False)].astype(float, copy=True)
    for _ in range(max_iter):
        y = assign_labels_chunked(X, C, batch)
        C_new = recompute_centroids(X, y, k, rng)
        if np.linalg.norm(C_new - C) < tol:
            C = C_new; break
        C = C_new
    J = inertia_wcss(X, y, C, batch)
    return y, C, J

def elbow_wcss(X, kmax, repeats=1, seed=42, batch=200_000):
    W = []
    for k in range(1, kmax+1):
        best = None
        for r in range(repeats):
            _, _, J = kmeans_serial(X, k, seed=seed+r, batch=batch)
            best = J if best is None else min(best, J)
        W.append(best)
    return W

def pick_k_by_knee(W):
    xs = np.arange(1, len(W)+1)
    knee = KneeLocator(xs, W, curve="convex", direction="decreasing")
    return int(knee.knee) if knee.knee else max(2, len(W)//3)

def plot_elbow(xs, W):
    f,a = plt.subplots(); a.plot(xs,W,marker="o"); a.set(xlabel="k", ylabel="WCSS", title="Elbow"); return f
def plot_clusters(X, y, C, sample=50_000):
    n = X.shape[0]; idx = np.arange(n) if n<=sample else np.random.default_rng(0).choice(n,size=sample,replace=False)
    f,a = plt.subplots(); a.scatter(X[idx,0],X[idx,1],s=5,c=y[idx],alpha=0.6); a.scatter(C[:,0],C[:,1],s=120,marker="X",edgecolor="black")
    a.set(xlabel="enzyme", ylabel="hydrofob", title="Clusters (sampled)"); return f
def plot_centers_heatmap(C):
    f,a = plt.subplots(); sns.heatmap(C, annot=True, fmt=".2f", ax=a); a.set_title("Centroids heatmap"); a.set_xlabel("features"); a.set_ylabel("cluster"); return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--outdir", type=str, default="Serial/results/aleks")
    args = ap.parse_args()

    np.random.seed(args.seed); random.seed(args.seed)
    t0 = time.time()
    X, seq_len, protid = load_data()

    W = elbow_wcss(X, args.kmax, repeats=args.repeats, seed=args.seed, batch=args.batch)
    xs = np.arange(1, args.kmax+1)
    k_opt = pick_k_by_knee(W)

    y, C, J = kmeans_serial(X, k_opt, seed=args.seed, batch=args.batch)

    idx_long = int(np.argmax(seq_len))
    cid = int(y[idx_long]); longest = int(seq_len[idx_long]); avg = float(seq_len[y==cid].mean()); pid = protid[idx_long]
    elapsed = time.time() - t0

    ensure_outdir(args.outdir)
    save_fig(plot_elbow(xs,W),             f"{args.outdir}/elbow.png")
    save_fig(plot_clusters(X,y,C),         f"{args.outdir}/clusters.png")
    save_fig(plot_centers_heatmap(C),      f"{args.outdir}/centers_heatmap.png")

    print(f"[serial|aleks] k_opt={k_opt} inertia={J:.2f} cluster={cid} longest={longest} "
          f"avg={avg:.2f} protid={pid} elapsed={elapsed:.2f}s")

if __name__ == "__main__":
    import numpy as np
    main()
