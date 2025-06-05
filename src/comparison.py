from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score

def evaluate_clustering(df, labels, model_name):
    scores={}

    if len(set(labels)) <=1:
        print(f"[WARN] {model_name}: Only 1 cluster found. Skipping evaluation")
        return None
    scores['Model']=model_name
    scores['Silhouette']=silhouette_score(df,labels)
    scores['Davies-Bouldin']=davies_bouldin_score(df,labels)
    scores['Calinski-Harabasz']=calinski_harabasz_score(df,labels)

    return scores