import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_clusters(data_with_labels, x_col, y_col, title = 'Cluster Plot', save = False, model_name=''):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data = data_with_labels,x=x_col,y=y_col,hue='Cluster',
                    palette='viridis')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='CLuster')
    plt.grid(True)
    plt.tight_layout()
    if save and model_name:
        os.makedirs("output",exist_ok=True)
        plot_path = f"output/{model_name}_clusters.png"
        plt.savefig(plot_path)
        print("Plot saved")

    plt.show()

def compare_cluster_distributions(data_with_labels, feature_cols):
    for col in feature_cols:
        plt.figure(figsize=(7,4))
        sns.boxplot(x='Cluster', y=col, data= data_with_labels)
        plt.title(f'Distribution of {col} by Cluster')
        plt.tight_layout()
        plt.show()
