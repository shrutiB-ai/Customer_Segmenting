# Customer Segmentation using Clustering Algorithms

This project performs Customer Segmentation using multiple unsupervised machine learning clustering techniques to identify distinct customer groups based on purchasing and behavioral patterns.

The project compares different clustering algorithms to understand how segmentation quality varies across methods and business scenarios.

Implemented clustering approaches include:

* K-Means Clustering
* Gaussian Mixture Models (GMM)
* Hierarchical Clustering
* DBSCAN Clustering

---

# Project Objectives

* Segment customers into meaningful behavioral groups
* Compare clustering algorithms on the same dataset
* Evaluate clustering quality using unsupervised metrics
* Visualize customer clusters and distributions
* Build reusable clustering pipelines for business analytics

---

# Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* YAML configuration management

---

# Clustering Algorithms Implemented

## 1. K-Means Clustering

K-Means partitions customers into predefined clusters based on feature similarity.

### Features

* Fast and scalable clustering
* Suitable for well-separated customer groups
* Centroid-based segmentation

### Use Cases

* Retail customer segmentation
* Marketing campaign targeting
* Customer behavior analysis

---

## 2. Gaussian Mixture Models (GMM)

GMM performs probabilistic clustering and allows soft cluster assignments.

### Features

* Handles overlapping customer segments
* Captures more complex distributions
* Flexible cluster boundaries

### Use Cases

* Customer probability scoring
* Behavioral overlap analysis
* Personalized targeting systems

---

## 3. Hierarchical Clustering

Hierarchical clustering creates nested customer groups using linkage-based clustering.

### Features

* Dendrogram-based hierarchy creation
* No requirement for centroid assumptions
* Useful for exploratory segmentation

### Use Cases

* Business hierarchy analysis
* Exploratory customer grouping
* Relationship-based segmentation

---

## 4. DBSCAN Clustering

DBSCAN identifies dense customer regions while detecting outliers/noise points.

### Features

* Density-based clustering
* Handles irregular cluster shapes
* Automatically identifies anomalies

### Use Cases

* Fraud/outlier detection
* Noise filtering
* Sparse customer analysis

---

# Workflow

1. Load customer dataset
2. Preprocess and normalize features
3. Train clustering models
4. Generate cluster labels
5. Evaluate clustering quality
6. Visualize cluster distributions
7. Compare model performance

---

# Evaluation Metrics

The project compares clustering quality using standard unsupervised learning metrics such as:

* Silhouette Score
* Davies-Bouldin Index
* Cluster Distribution Analysis

Model comparison results are exported to:

```bash id="c4f3rf"
output/model_comparison.csv
```

---

# Visualization Features

The project includes visualization modules for:

* Cluster scatter plots
* Cluster distribution comparison
* Feature-wise segmentation analysis

These visualizations help interpret customer behavior patterns across segments.

---

# Project Structure

```bash id="e8j7xz"
customer-segmentation/
│
├── data/
├── output/
├── src/
│   ├── preprocessing.py
│   ├── kmeans_clustering.py
│   ├── gmm_clustering.py
│   ├── hierarchical_clustering.py
│   ├── dbscan_clustering.py
│   ├── visualization.py
│   ├── comparison.py
│
├── config.yaml
├── main.py
└── README.md
```

---

# Running Modes

## Single Model Mode

Run a specific clustering algorithm configured in `config.yaml`.

Supported models:

* kmeans
* gmm
* hierarchical
* dbscan

---

## All Models Mode

Run all clustering algorithms together and compare results automatically.

---

# Example Business Applications

* Customer segmentation
* Personalized marketing
* Customer lifetime value analysis
* Retail targeting strategies
* Fraud/anomaly detection
* Recommendation system preprocessing

---

# Key Learnings

* Understanding strengths and limitations of clustering algorithms
* Importance of preprocessing and feature scaling in unsupervised learning
* Comparing centroid-based vs density-based clustering approaches
* Interpreting customer behavior using visual analytics
* Evaluating unsupervised models using clustering metrics

---

# Future Improvements

* PCA/t-SNE dimensionality reduction
* Automated hyperparameter tuning
* Interactive dashboard deployment
* Real-time segmentation pipelines
* Hybrid clustering approaches
* Customer churn prediction integration

---

# How to Run

## Install Dependencies

```bash id="n8uvz3"
pip install -r requirements.txt
```

## Run Single Model

```bash id="h2tq5z"
python main.py
```

## Run All Clustering Models

Update `config.yaml`:

```yaml id="6t1s4e"
mode: all
```

Then execute:

```bash id="r9u2wm"
python main.py
```

---

# Author

Shruti Bhosale

Applied Machine Learning | AI Engineering | Customer Analytics | Production ML Systems
