# author: aleks — PARALLEL (threads)


import argparse, os, time, math

# Limit NumPy/MKL to a single internal thread to avoid conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

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
    """Split [0, n) into ~equal contiguous chunks."""
    step = max(1, math.ceil(n / parts))
    for s in range(0, n, step):
        yield s, min(s + step, n)

def assign_labels_block(Xb, C):
    """
    Assign labels for a data block Xb using fixed centroids C.
    Uses squared Euclidean distance via (x - c)^2 = |x|^2 + |c|^2 - 2 x·c
    """
    c2 = np.sum(C**2, axis=1)                      # (k,)
    x2 = np.sum(Xb**2, axis=1, keepdims=True)      # (B,1)
    d2 = x2 + c2[None, :] - 2.0 * (Xb @ C.T)       # (B,k)
    return np.argmin(d2, axis=1)

def recompute_centroids(X, y, k, rng):
    """Recompute centroids; re-seed empty clusters from random points."""
    C = np.zeros((k, X.shape[1]), dtype=float)
    for cid in range(k):
        m = (y == cid)
        C[cid] = X[rng.integers(0, X.shape[0])] if not np.any(m) else X[m].mean(axis=0)
    return C

def inertia_wcss(X, y, C, parts):
    """Compute WCSS in chunks (sequential; fast enough vs assignment)."""
    total = 0.0
    c2 = np.sum(C**2, axis=1)
    for s, e in chunk_ranges(X.shape[0], parts):
        Xb = X[s:e]
        x2 = np.sum(Xb**2, axis=1, keepdims=True)
        d2 = x2 + c2[None, :] - 2.0 * (Xb @ C.T)
        total += np.min(d2, axis=1).sum()
    return float(total)

# ---------- k-means (threads) ----------
def kmeans_threads(X, k, seed=42, max_iter=100, tol=1e-4, workers=None):
    """
    K-means where the expensive label assignment step runs in multiple threads.
    Note: many NumPy BLAS ops release the GIL, so threads can speed up this part.
    """
    if workers is None:
        workers = max(1, mp.cpu_count())
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = X[rng.choice(n, size=k, replace=False)].astype(float, copy=True)

    for _ in range(max_iter):
        # parallel label assignment over contiguous chunks
        y = np.empty(n, dtype=np.int32)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for s, e in chunk_ranges(n, workers):
                futures[ex.submit(assign_labels_block, X[s:e], C)] = (s, e)
            for fut in as_completed(futures):
                s, e = futures[fut]
                y[s:e] = fut.result()

        C_new = recompute_centroids(X, y, k, rng)
        if np.linalg.norm(C_new - C) < tol:
            C = C_new
            break
        C = C_new

    J = inertia_wcss(X, y, C, workers)
    return y, C, J

def elbow_wcss_threads(X, kmax, repeats=1, seed=42, workers=None):
    """Elbow curve using threaded k-means."""
    W = []
    for k in range(1, kmax + 1):
        best = None
        for r in range(repeats):
            _, _, J = kmeans_threads(X, k, seed=seed + r, workers=workers)
            best = J if best is None else min(best, J)
        W.append(best)
    return W

def pick_k_by_knee(xs, W):
    """
    Pick k via 'max distance to chord' (no external deps).
    Avoid np.cross deprecation warnings by using 3D vectors.
    """
    x1, y1 = xs[0], W[0]
    x2, y2 = xs[-1], W[-1]
    line = np.array([x2 - x1, y2 - y1, 0.0], float)            # 3D to satisfy NumPy>=2.0
    L = np.linalg.norm(line) + 1e-12
    best_k, best_d = xs[0], -1.0
    for x, y in zip(xs, W):
        v = np.array([x - x1, y - y1, 0.0], float)
        d = np.linalg.norm(np.cross(line, v)) / L
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

    xs = np.arange(1, args.kmax + 1)
    W = elbow_wcss_threads(X, args.kmax, repeats=args.repeats, seed=args.seed, workers=args.workers)
    k_opt = pick_k_by_knee(xs, W)

    y, C, J = kmeans_threads(X, k_opt, seed=args.seed, workers=args.workers)

    # cluster of the protein with the longest sequence (LAB1 requirement)
    idx_long = int(np.argmax(seq_len))
    cid = int(y[idx_long])
    longest = int(seq_len[idx_long])
    avg = float(seq_len[y == cid].mean())
    pid = protid[idx_long]

    elapsed = time.time() - t0
    ensure_outdir(args.outdir)
    print(f"[parallel|threads|aleks] k_opt={k_opt} inertia={J:.2f} "
          f"cluster={cid} longest={longest} avg={avg:.2f} protid={pid} "
          f"workers={args.workers or mp.cpu_count()} elapsed={elapsed:.2f}s")

if __name__ == "__main__":
    import numpy as np
    main()
