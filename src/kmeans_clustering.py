from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_optimal_k(data,max_k=10):
    sse=[]
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(8,4))
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.title('Elbow Method for optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters,kmeans

def add_cluster_lables(data, clusters):
    data_with_labels = data.copy()
    data_with_labels['Cluster'] = clusters
    return data_with_labels