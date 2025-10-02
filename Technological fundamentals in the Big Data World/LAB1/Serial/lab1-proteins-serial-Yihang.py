import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- K-means from scratch ----------
def kmeans(raw_data, k_num, max_iters=300, tol=1e-4, num_gen=None):
    """
    X: (n, d) data
    k: clusters
    returns labels (n,), centroids (k, d), inertia (sum of squared distances)
    """
    n, d = raw_data.shape
    if num_gen is None:
        num_gen = np.random.default_rng()
    # Init: choose k random points as centroids
    idx = num_gen.choice(n, size=k_num, replace=False)
    cluster_centroids = raw_data[idx]

    cluster_labels = np.empty([n, 1])
    for _ in range(max_iters):
        # assign
        dists = np.linalg.norm(raw_data[:, None, :] - cluster_centroids[None, :, :], axis=2)  # (n, k)
        cluster_labels = np.argmin(dists, axis=1)

        # recompute
        new_centroids = np.zeros_like(cluster_centroids)
        for k_iterator in range(k_num):
            cluster_pts = raw_data[cluster_labels == k_iterator]
            if len(cluster_pts) == 0:
                # re-seed an empty cluster to a random point
                new_centroids[k_iterator] = raw_data[num_gen.integers(0, n)]
            else:
                new_centroids[k_iterator] = cluster_pts.mean(axis=0)

        shift = np.linalg.norm(new_centroids - cluster_centroids)
        cluster_centroids = new_centroids
        if shift < tol:
            break

    # inertia (WSS)
    cluster_inertia = 0.0
    for k_iterator in range(k_num):
        cluster_pts = raw_data[cluster_labels == k_iterator]
        if len(cluster_pts):
            cluster_inertia += np.sum((cluster_pts - cluster_centroids[k_iterator]) ** 2)
    return cluster_labels, cluster_centroids, cluster_inertia

# Elbow: choose k by "max distance to the line" heuristic
def choose_k_by_elbow(k_list, inertia_list):
    x1, y1 = k_list[0], inertia_list[0]
    x2, y2 = k_list[-1], inertia_list[-1]

    # line from (x1,y1) to (x2,y2)
    line = np.array([x2 - x1, y2 - y1], dtype=float)
    line_norm = np.linalg.norm(line)
    best_k, best_dist = k_list[0], -1.0
    for x, y in zip(k_list, inertia_list):
        vec = np.array([x - x1, y - y1], dtype=float)

        # distance from point to line
        safe_minimal_dist = 1e-12
        dist = np.abs(np.cross(line, vec)) / (line_norm + safe_minimal_dist)
        if dist > best_dist:
            best_dist = dist
            best_k = x
    return best_k


if __name__ == "__main__":
    global_start = time.time()
    csv_path = Path("proteins.csv")
    df = pd.read_csv(csv_path)

    features = df[["enzyme", "hydrofob"]].astype(float).to_numpy()
    lengths  = df["sequence"].str.len().to_numpy()

    # 1) Elbow
    elbow_start = time.time()
    rng = np.random.default_rng(42)
    ks = list(range(1, 16))
    inertia = []
    for k in ks:
        _, _, wss = kmeans(features, k, num_gen=rng)
        inertia.append(wss)
    k_opt = choose_k_by_elbow(ks, inertia)
    elbow_end = time.time()

    # 2) Cluster with optimal k
    labels, centroids, _ = kmeans(features, k_opt, num_gen=rng)

    # 3) Find the cluster with the highest average sequence length
    avg_len_by_cluster = []
    for i in range(k_opt):
        cl_idx = (labels == i)
        if np.any(cl_idx):
            avg_len = lengths[cl_idx].mean()
            avg_len_by_cluster.append((i, avg_len, cl_idx.sum()))
        else:
            avg_len_by_cluster.append((i, float("-inf"), 0))
    best_cluster, best_avg_len, best_count = max(avg_len_by_cluster, key=lambda t: t[1])
    global_end = time.time()

    # ---------- Plots ----------
    plt.figure(figsize=(5,4))
    plt.plot(ks, inertia, marker='o')
    plt.title("Elbow Plot (WSS vs k)")
    plt.xlabel("k")
    plt.ylabel("Within-Cluster Sum of Squares (WSS)")
    plt.grid(True)

    # Scatter of clusters with centroids
    plt.figure(figsize=(6,5))
    for i in range(k_opt):
        pts = features[labels == i]
        if len(pts):
            plt.scatter(pts[:,0], pts[:,1], s=20, label=f"Cluster {i}")
    plt.scatter(centroids[:,0], centroids[:,1], s=200, marker='X', edgecolor='black', linewidths=1.2, label="Centroids")
    plt.title("K-means Clusters (Enzyme vs Hydrofob)")
    plt.xlabel("enzyme")
    plt.ylabel("hydrofob")
    plt.legend(loc="best")
    plt.grid(True)

    # Heat map of centroid values (features × clusters)
    # Rows: features (Enzyme, Hydrofob); Cols: cluster IDs
    plt.figure(figsize=(1.2*k_opt + 2.5, 3.2))
    mat = centroids.T  # shape (2, k)
    im = plt.imshow(mat, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks([0,1], ["enzyme", "hydrofob"])
    plt.xticks(range(k_opt), [f"C{i}" for i in range(k_opt)])
    plt.title("Centroid Heat Map")

    # annotate
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            plt.text(c, r, f"{mat[r,c]:.2f}", ha="center", va="center")

    # Print the winner cluster info at the end (per instructions)
    print(f"Optimal k (elbow): {k_opt}")
    print(f"Cluster with highest average sequence length: {best_cluster}")
    print(f"  Avg sequence length: {best_avg_len:.4f} (n={best_count})")
    print(f"Elbow time (serial): {elbow_end - elbow_start:.3f} s")
    print(f"Total execution time: {global_end - global_start:.3f} seconds")

    # Show all figures at the very end
    plt.show()
