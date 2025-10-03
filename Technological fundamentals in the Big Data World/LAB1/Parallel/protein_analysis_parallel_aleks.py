# author: aleks — PARALLEL (multiprocessing)


import argparse, os, time, math
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, set_start_method

# ---------- I/O ----------
def load_data():
    df = pd.read_csv("proteins.csv", usecols=["protid","enzyme","hydrofob","sequence"])
    X = df[["enzyme","hydrofob"]].to_numpy(dtype=float)
    seq_len = df["sequence"].str.len().to_numpy()
    protid = df["protid"].to_numpy()
    return X, seq_len, protid

def ensure_outdir(p): os.makedirs(p, exist_ok=True)

# ---------- helpers ----------
def chunk_ranges(n, parts):
    """Split range [0,n) into 'parts' approximately equal chunks."""
    step = math.ceil(n / parts)
    for s in range(0, n, step):
        yield s, min(s + step, n)

def assign_labels_block(args):
    """Worker function: assign labels for a data block given centroids C."""
    Xb, C = args
    c2 = np.sum(C**2, axis=1)                      # (k,)
    x2 = np.sum(Xb**2, axis=1, keepdims=True)      # (B,1)
    d2 = x2 + c2[None, :] - 2.0 * (Xb @ C.T)       # (B,k)
    return np.argmin(d2, axis=1)

def recompute_centroids(X, y, k, rng):
    """Recompute cluster centroids given assignments y."""
    C = np.zeros((k, X.shape[1]), dtype=float)
    for cid in range(k):
        m = (y == cid)
        C[cid] = X[rng.integers(0, X.shape[0])] if not np.any(m) else X[m].mean(axis=0)
    return C

def inertia_wcss(X, y, C, parts):
    """Compute within-cluster sum of squares (WCSS) using chunked computation."""
    total = 0.0
    c2 = np.sum(C**2, axis=1)
    for s, e in chunk_ranges(X.shape[0], parts):
        Xb = X[s:e]
        x2 = np.sum(Xb**2, axis=1, keepdims=True)
        d2 = x2 + c2[None, :] - 2.0 * (Xb @ C.T)
        total += np.min(d2, axis=1).sum()
    return float(total)

# ---------- k-means (multiprocessing) ----------
def kmeans_mp(X, k, seed=42, max_iter=100, tol=1e-4, workers=None):
    """K-means clustering using multiprocessing for label assignment."""
    if workers is None:
        workers = max(1, cpu_count() - 1)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = X[rng.choice(n, size=k, replace=False)].astype(float, copy=True)

    with Pool(processes=workers) as pool:
        for _ in range(max_iter):
            # assign labels in parallel for data chunks
            tasks = [(X[s:e], C) for s, e in chunk_ranges(n, workers)]
            parts = pool.map(assign_labels_block, tasks)

            y = np.empty(n, dtype=np.int32)
            for (s, e), lbl in zip(chunk_ranges(n, workers), parts):
                y[s:e] = lbl

            C_new = recompute_centroids(X, y, k, rng)
            if np.linalg.norm(C_new - C) < tol:
                C = C_new
                break
            C = C_new

        J = inertia_wcss(X, y, C, workers)
    return y, C, J

def elbow_wcss_mp(X, kmax, repeats=1, seed=42, workers=None):
    """Compute elbow curve using multiprocessing-based k-means."""
    W = []
    for k in range(1, kmax + 1):
        best = None
        for r in range(repeats):
            _, _, J = kmeans_mp(X, k, seed=seed + r, workers=workers)
            best = J if best is None else min(best, J)
        W.append(best)
    return W

def pick_k_by_knee(xs, W):
    """Pick optimal k using 'max distance to chord' heuristic (no external libs)."""
    x1, y1 = xs[0], W[0]
    x2, y2 = xs[-1], W[-1]
    line = np.array([x2 - x1, y2 - y1], float)
    L = np.linalg.norm(line) + 1e-12
    best_k, best_d = xs[0], -1.0
    for x, y in zip(xs, W):
        v = np.array([x - x1, y - y1], float)
        d = abs(np.cross(line, v)) / L
        if d > best_d:
            best_d, best_k = d, x
    return int(best_k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="Parallel/results/aleks")
    args = ap.parse_args()

    t0 = time.time()
    X, seq_len, protid = load_data()

    # elbow curve (each k-means run is parallelized)
    xs = np.arange(1, args.kmax + 1)
    W = elbow_wcss_mp(X, args.kmax, repeats=args.repeats, seed=args.seed, workers=args.workers)
    k_opt = pick_k_by_knee(xs, W)

    # final clustering with k_opt
    y, C, J = kmeans_mp(X, k_opt, seed=args.seed, workers=args.workers)

    # find cluster of the protein with the longest sequence
    idx_long = int(np.argmax(seq_len))
    cid = int(y[idx_long])
    longest = int(seq_len[idx_long])
    avg = float(seq_len[y == cid].mean())
    pid = protid[idx_long]

    elapsed = time.time() - t0
    ensure_outdir(args.outdir)

    print(f"[parallel|mp|aleks] k_opt={k_opt} inertia={J:.2f} "
          f"cluster={cid} longest={longest} avg={avg:.2f} protid={pid} "
          f"workers={args.workers or cpu_count()-1} elapsed={elapsed:.2f}s")

if __name__ == "__main__":
    # required for Windows multiprocessing
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    import numpy as np
    main()
