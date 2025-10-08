import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator
import seaborn as sns
import time
from multiprocessing import Pool


def find_clusters(data, max_k, plot=True):
    t1 = time.time()
    means = []
    inertias = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[['enzyme','hydrofob']])
        means.append(k)
        inertias.append(kmeans.inertia_)

    # find optimal k with elbow method (using kneed library)
    kl = KneeLocator(means, inertias, curve="convex", direction="decreasing")
    optimal_k = kl.knee

    # clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[['enzyme','hydrofob']])
    centroids = kmeans.cluster_centers_
    centroids_df = pd.DataFrame(centroids, columns=['enzyme', 'hydrofob'])
       
    # compute average seq length and best cluster
    data["seq_length"] = data["sequence"].str.len()
    avg_lengths = data.groupby("cluster")["seq_length"].mean()
    best_cluster = avg_lengths.idxmax()
    best_avg = avg_lengths.max()
    longest_len = data["seq_length"].max()
    t2 = time.time()

    t3 = time.time()
    if plot:
        # elbow plot
        plt.plot(means, inertias, 'bx-')
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.title("Elbow Method")
        plt.savefig("elbow.png", dpi=120, bbox_inches="tight")
        plt.close()

        # cluster plot
        plt.scatter(data["enzyme"], data["hydrofob"], c=data["cluster"], cmap='rainbow')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100, c='black')
        plt.xlabel("enzyme")
        plt.ylabel("hydrofob")
        plt.title("Clusters with Centroids")
        plt.savefig("clusters.png", dpi=120, bbox_inches="tight")
        plt.close()

        # heat map
        plt.figure(figsize=(6,4))
        sns.heatmap(centroids_df, annot=True, cmap="viridis", cbar=True)
        plt.xlabel("Features")
        plt.ylabel("Cluster")
        plt.title("Heatmap of Cluster Centroids")
        plt.savefig("heatmap.png", dpi=120, bbox_inches="tight")
        plt.close()
    t4 = time.time()


    # summary table
    summary = pd.DataFrame({
        "Best Cluster": [best_cluster],
        "Average Sequence Length": [best_avg],
        "Longest Seq Length": [longest_len],
        "Optimal k": [optimal_k]
    })
    #time summary
    time_summary = pd.DataFrame({
        "Kmeans and sequence analysis time (s)": [t2 - t1],
        "Plotting Time (s)": [t4 - t3]
    })
    return summary, time_summary


proteins = pd.read_csv("proteins.csv")
result = find_clusters(proteins, 10, True)
print(result[1])
