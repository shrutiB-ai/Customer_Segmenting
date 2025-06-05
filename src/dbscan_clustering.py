from sklearn import neighbors
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_k_distance_graph(data, k=4):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices= neighbors_fit.kneighbors(data)

    # sort distance to find the knee ( optimal epsilon)
    distances = np.sort(distances[:,k-1],axis=0)
    plt.figure(figsize=(8,4))
    plt.plot(distances)
    plt.title('K-distance Graph for DBSCAN')
    plt.xlabel('Data points sorted by distance')
    plt.ylabel(f'{k}th Nearest Neighbour Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_dbscan(data, eps=0.5, min_samples = 5):
    db= DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(data)
    return clusters, db

def add_cluster_labels(data, clusters):
    data_with_labels =  data.copy()
    data_with_labels['Cluster']= clusters
    return data_with_labels