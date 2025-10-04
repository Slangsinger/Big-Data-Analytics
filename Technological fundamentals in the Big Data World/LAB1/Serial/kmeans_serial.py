import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


class kmeans():
    def __init__(self, max_k):
        self.max_k = max_k
        self.k_values = range(1, max_k + 1)
        self.wcss_values = None
        self.optimal_k = None
        self.centroids = None
        self.labels = None

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

    def serial(self, data, max_iter=200):
        wcss_list = []
        all_labels = []
        all_centroids = []

        data_np = data[['enzyme', 'hydrofob']].values
        t1 = time.time()

        for i in self.k_values:
            # Randomly initialize centroids
            centroids = np.random.uniform(np.min(data_np, axis=0),
                                          np.max(data_np, axis=0),
                                          size=(i, data_np.shape[1]))

            for _ in range(max_iter):
                #vectorized distance computation
                distances = np.linalg.norm(data_np[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

                # Assign each point to closest centroid
                y = np.argmin(distances, axis=1)

                #update centroids
                new_centroids = np.array([
                    data_np[y == j].mean(axis=0) if np.any(y == j) else centroids[j]
                    for j in range(i)
                ])

                # Check convergence
                if np.max(np.abs(centroids - new_centroids)) < 0.0001:
                    break
                centroids = new_centroids

            # Compute WCSS for this k
            wcss = self.compute_WCSS(data_np, y, centroids)
            wcss_list.append(wcss)
            all_centroids.append(centroids)
            all_labels.append(y)

        # Store results
        self.wcss_values = np.array(wcss_list)
        self.optimal_k = self.find_optimal_k(self.k_values, self.wcss_values)

        # Retrieve best k
        opt_idx = self.k_values.index(self.optimal_k)
        self.centroids = all_centroids[opt_idx]
        self.labels = all_labels[opt_idx]

        # Add cluster info to DataFrame
        data["cluster"] = self.labels
        data["seq_length"] = data["sequence"].str.len()
        avg_lengths = data.groupby("cluster")["seq_length"].mean()

        best_cluster = avg_lengths.idxmax()
        best_avg = avg_lengths.max()

        t2 = time.time()
        return self.centroids, self.labels, best_avg, best_cluster, t2 - t1

proteins = pd.read_csv('proteins.csv', sep=',')
kmeans_try = kmeans(10)
results = kmeans_try.serial(proteins)

print(f'The cluster with the longest average sequence length is cluster {results[3]} with length {results[2]}')
print(f'It took {results[4]:.2f} seconds to execute the code')