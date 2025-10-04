import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class kmeans():
    def __init__(self, max_k):
        self.max_k = max_k
        self.k_values = range(1, max_k + 1)
        self.wcss_values = None
        self.optimal_k = None
        self.centroids = None
        self.labels = None

    # ---------- Helper methods ----------
    @staticmethod
    def compute_WCSS(data, labels, centroids):
        sse = 0
        for i in range(len(centroids)):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - centroids[i]) ** 2)
        return sse

    @staticmethod
    def find_optimal_k(k_values, wcss_values):
        x1, y1 = k_values[0], wcss_values[0]
        x2, y2 = k_values[-1], wcss_values[-1]
        distances = []
        for i in range(len(k_values)):
            x0, y0 = k_values[i], wcss_values[i]
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + (x2 * y1 - y2 * x1))
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(numerator / denominator)
        optimal_k_index = np.argmax(distances)
        return k_values[optimal_k_index]

    # ---------- Threaded helper ----------
    @staticmethod
    def _run_kmeans_for_k(args):
        """Run one full k-means instance for a specific k value (vectorized)."""
        k, data_np, max_iter = args

        # Initialize random centroids
        centroids = np.random.uniform(np.min(data_np, axis=0),
                                      np.max(data_np, axis=0),
                                      size=(k, data_np.shape[1]))

        for _ in range(max_iter):
            distances = np.linalg.norm(data_np[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                data_np[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(k)
            ])

            if np.max(np.abs(centroids - new_centroids)) < 1e-4:
                break
            centroids = new_centroids

        wcss = kmeans.compute_WCSS(data_np, labels, centroids)
        return wcss, centroids, labels

    # ---------- Threaded version ----------
    def threaded(self, data, max_iter=200):
        """Threaded + vectorized K-Means for all k."""
        data_np = data[['enzyme', 'hydrofob']].values
        args = [(k, data_np, max_iter) for k in self.k_values]

        t1 = time.time()

        results = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._run_kmeans_for_k, arg) for arg in args]
            for f in as_completed(futures):
                results.append(f.result())

        # Maintain order (as as_completed may change it)
        results = sorted(zip(self.k_values, results), key=lambda x: x[0])
        wcss_list, all_centroids, all_labels = zip(*[r[1] for r in results])

        self.wcss_values = np.array(wcss_list)
        self.optimal_k = self.find_optimal_k(self.k_values, self.wcss_values)
        opt_idx = self.k_values.index(self.optimal_k)
        self.centroids = all_centroids[opt_idx]
        self.labels = all_labels[opt_idx]

        data["cluster"] = self.labels
        data["seq_length"] = data["sequence"].str.len()
        avg_lengths = data.groupby("cluster")["seq_length"].mean()
        best_cluster = avg_lengths.idxmax()
        best_avg = avg_lengths.max()

        t2 = time.time()
        return self.centroids, self.labels, best_avg, best_cluster, t2 - t1


if __name__ == "__main__":
    proteins = pd.read_csv('proteins.csv', sep=',')
    kmeans_try = kmeans(10)
    results = kmeans_try.threaded(proteins)

    print(f"The cluster with the longest average sequence length is cluster {results[3]} with length {results[2]:.2f}")
    print(f"It took {results[4]:.2f} seconds to execute the code using threading")
