import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class kmeans():
    def __init__(self, max_k):
        self.max_k = max_k
        self.k_values = range(1, max_k + 1)
        self.wcss_values = None #for all k
        self.optimal_k = None
        self.centroids = None #for optimal k
        self.labels = None #for optimal k

    @staticmethod
    def euclidean_distance(centroids, data_point):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))
    
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
            numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + (x2*y1 - y2*x1))
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator / denominator)

        optimal_k_index = np.argmax(distances)
        return k_values[optimal_k_index]

    def serial(self, data, max_iter=200):
        wcss_list = []
        all_labels = []
        all_centroids = []
        t1 = time.time()
        for i in self.k_values:
            #create random centroids within the range of the data
            centroids = np.random.uniform(np.amin(data[['enzyme', 'hydrofob']].values, axis=0), np.amax(data[['enzyme', 'hydrofob']].values, axis=0),
                                      size=(i, data[['enzyme', 'hydrofob']].values.shape[1]))
            
            #now that we have the random centroids we calculate the distance from each datapoint to the centroids
            #the distances will be stored in a list. Then we will see which distance is the leasnt and append the id of the cluster to y for that datapoint
            for _ in range(max_iter):
                y = []
                for data_point in data[['enzyme', 'hydrofob']].values:
                    distances = self.euclidean_distance(centroids, data_point)
                    cluster_num = np.argmin(distances)
                    y.append(cluster_num)
                y = np.array(y)

                #now that we have the cluster index for each datapoint we want to update the centroids
                #I do that by computing the mean position of all data points in one cluster
                #this outputs a list of arrays with each new centroid
                new_cluster_centers = []
                for a in range(i):
                    cluster_points = data[['enzyme', 'hydrofob']].values[y == a]
                    if len(cluster_points) == 0:
                        new_cluster_centers.append(centroids[a])
                    else:
                        new_cluster_centers.append(np.mean(cluster_points, axis=0))
                
                #now its time to check if the algorithm has finished. If the centroids moved less than 0.0001 then stop. If not, then repeat process
                new_cluster_centers = np.array(new_cluster_centers)
                if np.max(np.abs(centroids - new_cluster_centers)) < 0.0001:
                    break
                else:
                    centroids = new_cluster_centers

            #now that we have the proper centers and the labels of each point for each cluster we will procede with the elbow plot
            #for each k in k_values we will calculate how good of a fit it is. 
            #for that we will calculate the within-cluster sum of squares (WCSS) and append it to the wcss list
            wcss_list.append(self.compute_WCSS(data[['enzyme', 'hydrofob']].values, y, centroids))
            all_centroids.append(centroids)
            all_labels.append(y)

        #updating the wcss_values and calculate the optimal k with the elbow method
        self.wcss_values = np.array(wcss_list)
        self.optimal_k = self.find_optimal_k(self.k_values, self.wcss_values)

        # Retrieve centroids and labels for the optimal k
        opt_idx = self.k_values.index(self.optimal_k)
        self.centroids = all_centroids[opt_idx]
        self.labels = all_labels[opt_idx]

        # Add cluster labels and compute sequence lengths
        data["cluster"] = self.labels
        data["seq_length"] = data["sequence"].str.len()

        # Compute average length per cluster
        avg_lengths = data.groupby("cluster")["seq_length"].mean()

        # Identify the cluster with the longest average sequence
        best_cluster = avg_lengths.idxmax()
        best_avg = avg_lengths.max()


        t2 = time.time()
        return self.centroids, self.labels, best_avg, best_cluster, t2-t1



proteins = pd.read_csv('proteins.csv', sep = ',')
kmeans_try = kmeans(10)
results = kmeans_try.serial(proteins)

print(f'the cluster with the longest average sequence length is cluster {results[3]} with length {results[2]}')
print(f'it took {results[4]} seconds to excute the code')

# Ask user whether to show the plots
show_plots = input("Please press y/n to show the plots: ").strip().lower()

if show_plots == 'y':
    # elbow plot
    plt.figure()
    plt.plot(kmeans_try.k_values, kmeans_try.wcss_values, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.show()

    # clusters with centroids
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

    # heatmap of centroids
    plt.figure(figsize=(6, 4))
    plt.imshow(kmeans_try.centroids, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Centroid Feature Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Clusters')

    # Add feature and cluster labels
    plt.xticks(ticks=[0, 1], labels=['enzyme', 'hydrofob'])
    plt.yticks(ticks=range(kmeans_try.optimal_k),
               labels=[f'Cluster {i}' for i in range(kmeans_try.optimal_k)])
    plt.tight_layout()
    plt.show()
else:
    print("Plots skipped.")
