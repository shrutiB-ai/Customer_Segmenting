from sys import hash_info
import yaml
from src.preprocessing import preprocess_pipeline
from src.kmeans_clustering import run_kmeans, add_cluster_lables as add_kmeans_labels
from src.gmm_clustering import run_gmm, add_cluster_labels as add_gmm_labels
from src.hierarchical_clustering import run_hierarchical_clustering, add_cluster_labels as add_hr_labels
from src.dbscan_clustering import run_dbscan, add_cluster_labels as add_dbscan_labels
from src.visualization import plot_clusters, compare_cluster_distributions
from src.comparison import evaluate_clustering
import pandas as pd

def load_config(config_path = 'config.yaml'):
    print("Loading Config")
    with open(config_path,'r') as f:
	    return yaml.safe_load(f)

def run_all_models(config):
    raw_df, processed_df= preprocess_pipeline(config)
    scores=[]
    #kmeans
    print("Kmeans")
    k_clusters, _ = run_kmeans(processed_df, config['kmeans']['n_clusters'])
    labeled_df = add_kmeans_labels(raw_df.copy(), k_clusters)
    scores.append(evaluate_clustering(processed_df,k_clusters,"Kmeans"))
    
    #GMM
    print("GMM")
    g_clusters, _ = run_gmm(processed_df, config['GMM']['n_components'])
    labeled_df = add_gmm_labels(raw_df.copy(), g_clusters)
    scores.append(evaluate_clustering(processed_df,g_clusters,"GMM"))

    #Hierarchical
    print("Heir..")
    h_clusters, _ = run_hierarchical_clustering(processed_df, n_clusters=config['hierarchical']['n_clusters'],
                                              linkage_method=config['hierarchical'].get('linkage','ward'))
    labeled_df = add_hr_labels(raw_df.copy(), h_clusters)
    scores.append(evaluate_clustering(processed_df,h_clusters,"hierarchical"))

    #DBScan
    print("DBSCAN")
    d_clusters, _ = run_dbscan(processed_df, config['dbscan']['eps'],
                             min_samples=config['dbscan']['min_samples'])
    labeled_df = add_dbscan_labels(raw_df.copy(), d_clusters)
    scores.append(evaluate_clustering(processed_df,d_clusters,"dbscan"))
    df_scores = pd.DataFrame([s for s in scores if s is not None])
    df_scores.to_csv("output/model_comparison.csv",index=False)
    print(df_scores)

def main():
    config = load_config()
    mode = config.get('mode','single').lower()
    #Load and preprocess data
    print("Starting PreProcessing")
    raw_df, processed_df= preprocess_pipeline(config)
    
    if mode == 'single':
        method = config['model']['type']
        if method == 'kmeans':
            print("Kmeans")
            clusters, _ = run_kmeans(processed_df, config['kmeans']['n_clusters'])
            labeled_df = add_kmeans_labels(raw_df, clusters)
        elif method == 'gmm':
            print("GMM")
            clusters, _ = run_gmm(processed_df, config['GMM']['n_components'])
            labeled_df = add_gmm_labels(raw_df, clusters)

        elif method == 'hierarchical':
            print("Heir..")
            clusters, _ = run_hierarchical_clustering(processed_df, n_clusters=config['hierarchical']['n_clusters'],
                                                      linkage_method=config['hierarchical'].get('linkage','ward'))
            labeled_df = add_hr_labels(raw_df, clusters)
        elif method == 'dbscan':
            print("DBSCAN")
            clusters, _ = run_dbscan(processed_df, config['dbscan']['eps'],
                                     min_samples=config['dbscan']['min_samples'])
            labeled_df = add_dbscan_labels(raw_df, clusters)
        else:
            raise ValueError(f'Unsupported model type : {method}')

        # visualize

        x_col = config['visualization']['x']
        y_col = config['visualization']['y']
        
        plot_clusters(labeled_df,x_col,y_col,title=f'{method.upper()} Clustering', save =True,model_name=method)
        compare_cluster_distributions(labeled_df,[x_col,y_col])
    elif mode == 'all':
        run_all_models(config)
    else:
        raise ValueError(f'Unsupported mode:{mode}')

if __name__ == "__main__":
    main()


    
