import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage

class GeoClustering:

    def __init__(self):
        pass

    # --------------------
    # KMeans
    # --------------------

    def train_kmeans(self,X,n_clusters=6):
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X)
        return model,labels
    
    # ----------------------
    # DBSCAN
    # ----------------------

    def train_dbscan(self,X,eps=.15,min_samples=27):
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )
        labels = model.fit_predict(X)
        return model, labels

    # --------------------
    # Hierical
    # --------------------

    def hierarchical_linkage(self,X):
        Z = linkage(X,method='ward')
        return Z
    
    # ------------------- 
    # Evaluation
    # -------------------

    def evaluate(self, X, labels):
        sil = silhouette_score(X,labels)
        db = davies_bouldin_score(X, labels)
        return sil, db

    # --------------------
    # Elbow Method
    # -------------------

    def elbow(self,X, k_range=range(2,10)):
        intertias = {}

        for k in k_range:
            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )
            model.fit(X)
            intertias[k] = model.inertia_

        return  intertias
      