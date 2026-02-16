import joblib
import pandas as pd


class InferencePipeline:

    def __init__(self):
        self.kmeans_model = joblib.load("artifacts/geo_kmeans_model.pkl")
        self.pca_model = joblib.load("artifacts/pca_model.pkl")
        self.dataset = pd.read_csv("artifacts/deployment_dataset.csv")

    def get_dataset(self):
        return self.dataset

    def get_models(self):
        return self.kmeans_model, self.pca_model
