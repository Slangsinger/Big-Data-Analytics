import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count


class kmeans():
    def __init__(self, max_k):
        self.max_k = max_k
        self.k_values = range(1, max_k + 1)
        self.wcss_values = None  # for all k
        self.optimal_k = None
        self.centroids = None  # for optimal k
        self.labels = None  # for optimal k

    @staticmethod
    def euclidean_distance(centroids, data_point):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

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

    #Worker function
    @staticmethod
    def _run_single_k(args):
   
        k, data_np, max_iter = args

        # Initialize random centroids
        centroids = np.random.uniform(np.amin(data_np, axis=0), np.amax(data_np, axis=0),
                                      size=(k, data_np.shape[1]))

        for _ in range(max_iter):
            y = []
            for data_point in data_np:
                distances = np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            y = np.array(y)

            # Update centroids
            new_cluster_centers = []
            for a in range(k):
                cluster_points = data_np[y == a]
                if len(cluster_points) == 0:
                    new_cluster_centers.append(centroids[a])
                else:
                    new_cluster_centers.append(np.mean(cluster_points, axis=0))

            new_cluster_centers = np.array(new_cluster_centers)
            if np.max(np.abs(centroids - new_cluster_centers)) < 0.0001:
                break
            centroids = new_cluster_centers

        wcss = kmeans.compute_WCSS(data_np, y, centroids)
        return wcss, centroids, y

    #Parallel version
    def parallel(self, data, max_iter=200):
        wcss_list = []
        all_centroids = []
        all_labels = []

        data_np = data[['enzyme', 'hydrofob']].values

        t1 = time.time()

        # Prepare arguments for each process
        args = [(k, data_np, max_iter) for k in self.k_values]

        # Run in parallel across available CPU cores
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self._run_single_k, args)

        # Unpack results
        wcss_list, all_centroids, all_labels = zip(*results)
        self.wcss_values = np.array(wcss_list)

        # Find optimal k
        self.optimal_k = self.find_optimal_k(self.k_values, self.wcss_values)
        opt_idx = self.k_values.index(self.optimal_k)
        self.centroids = all_centroids[opt_idx]
        self.labels = all_labels[opt_idx]

        # Cluster stats
        data["cluster"] = self.labels
        data["seq_length"] = data["sequence"].str.len()
        avg_lengths = data.groupby("cluster")["seq_length"].mean()

        best_cluster = avg_lengths.idxmax()
        best_avg = avg_lengths.max()

        t2 = time.time()
        return self.centroids, self.labels, best_avg, best_cluster, t2 - t1



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    proteins = pd.read_csv('proteins.csv', sep=',')
    kmeans_try = kmeans(10)
    results = kmeans_try.parallel(proteins)

    print(f"The cluster with the longest average sequence length is cluster {results[3]} with length {results[2]}")
    print(f"It took {results[4]:.2f} seconds to execute the code (parallel, non-vectorized)")

    show_plots = input("Please press y/n to show the plots: ").strip().lower()

    if show_plots == 'y':
        # Elbow plot
        plt.figure()
        plt.plot(kmeans_try.k_values, kmeans_try.wcss_values, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.show()

        # Cluster visualization
        plt.figure()
        plt.scatter(proteins['enzyme'], proteins['hydrofob'],
                    c=kmeans_try.labels, cmap='viridis', alpha=0.7)
        plt.scatter(kmeans_try.centroids[:, 0], kmeans_try.centroids[:, 1],
                    c='red', marker='x', s=200, label='Centroids')
        plt.title(f'K-Means Clustering (k={kmeans_try.optimal_k})')
        plt.xlabel('Enzyme')
        plt.ylabel('Hydrofob')
        plt.legend()
        plt.show()

        # Heatmap of centroids
        plt.figure(figsize=(6, 4))
        plt.imshow(kmeans_try.centroids, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Value')
        plt.title('Centroid Feature Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Clusters')
        plt.xticks(ticks=[0, 1], labels=['enzyme', 'hydrofob'])
        plt.yticks(ticks=range(kmeans_try.optimal_k),
                   labels=[f'Cluster {i}' for i in range(kmeans_try.optimal_k)])
        plt.tight_layout()
        plt.show()
    else:
        print("Plots skipped.")
