import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import _LINKAGE_METHODS, dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

def plot_dendrogram(data, method='ward'):
    linked = linkage(data, method = method)
    plt.figure(figsize=(10,5))
    dendrogram(linked)
    plt.title('Hierarchical clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

def run_hierarchical_clustering(data,n_clusters=5,linkage_method = 'ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage= linkage_method)
    clusters = model.fit_predict(data)
    return clusters,model

def add_cluster_labels(data, clusters):
    data_with_labels = data.copy()
    data_with_labels['Cluster']=clusters
    return data_with_labels