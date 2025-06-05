from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

def find_optimal_gmm_components(data, max_components =10):
    aic=[]
    bic=[]

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full',random_state=42)
        gmm.fit(data)
        aic.append(gmm.aic(data))
        bic.append(gmm.bic(data))
        
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, max_components + 1), aic, marker = 'o')
    plt.title('AIC vs No. of Components')
    plt.xlabel('Components')
    plt.ylabel('AIC')

    plt.subplot(1,2,2)
    plt.plot(range(1, max_components + 1), bic, marker = 'o', color = 'orange')
    plt.title('BIC vs No. of Components')
    plt.xlabel('Components')
    plt.ylabel('BIC')

    plt.tight_layout()
    plt.show()

def run_gmm(data, n_components):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',random_state=42)
    clusters = gmm.fit_predict(data)
    return clusters,gmm

def add_cluster_labels(data, clusters):
    data_with_labels = data.copy()
    data_with_labels['Cluster'] = clusters
    return data_with_labels