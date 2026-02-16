import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DimensionalityReduction:
    def __init__(self):
        self.pca_model = None

    # ----------------
    # PCA
    # ---------------

    def train_pca(self, X, n_components=None):
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X)

        self.pca_model = pca
        return pca, transformed
    
    def explained_variance(self):
        if self.pca_model is None:
            raise ValueError('PCA model not trained')
        
        variance_ratio = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)

        return variance_ratio, cumulative_variance
    
    # ---------------------
    # Feature Importance
    # ---------------------

    def feature_importance(self, feature_names):
        if self.pca_model is None:
            raise ValueError('PCA model not trained')
        
        loadings = pd.DataFrame(
            self.pca_model.components_,
            columns=feature_names
        )

        importance = (
            loadings.abs()
            .mean(axis=0)
            .sort_values(ascending=False)

        )
        return importance
    
    # ------------------------
    # t-SNE (for visualization)
    # -------------------------

    def run_tsne(self, X, n_samples=10000):
        X_sample = X[:n_samples]

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            n_iter=1000
        )

        transformed = tsne.fit_transform(X_sample)
        return transformed

