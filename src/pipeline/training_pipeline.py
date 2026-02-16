import os
import joblib
import mlflow
import mlflow.sklearn

from src.logger import get_logger
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering
from src.components.clustering import GeoClustering
from src.components.dimensionality_reduction import DimensionalityReduction


class TrainingPipeline:

    def __init__(self):
        self.logger = get_logger()

    def run(self):

        self.logger.info("Starting PatrolIQ training pipeline")

        # MLflow Setup
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("PatrolIQ_Clustering")

        # Data Ingestion
        ingestion = DataIngestion(
            raw_path="data/raw/Crimes_-_2001_to_Present_20260115.csv",
            processed_path="data/processed/sample.csv"
        )
        df_sample = ingestion.run()

        # Preprocessing
        preprocessing = DataPreprocessing()
        df_clean = preprocessing.run(df_sample)

        # Feature Engineering
        feature_engineering = FeatureEngineering()
        df_featured = feature_engineering.run(df_clean)

        # Clustering
        geo_X = df_featured[['Lat_scaled', 'Long_scaled']]
        geo_sample = geo_X.sample(n=50000, random_state=42)

        clustering = GeoClustering()
        kmeans_model, _ = clustering.train_kmeans(geo_X, n_clusters=6)

        sample_labels = kmeans_model.predict(geo_sample)
        sil, db = clustering.evaluate(geo_sample, sample_labels)

        # PCA
        pca_features = df_featured[
            [
                'Lat_scaled',
                'Long_scaled',
                'Hour',
                'DayOfWeek_Num',
                'Month',
                'Is_Weekend',
                'Crime_Severity_Score'
            ]
        ].dropna()

        dr = DimensionalityReduction()
        pca_model, _ = dr.train_pca(pca_features, n_components=2)

        variance_ratio, cumulative_variance = dr.explained_variance()

        # MLflow Logging
        with mlflow.start_run():

            # Parameters
            mlflow.log_param("geo_kmeans_clusters", 6)
            mlflow.log_param("pca_components", 2)

            # PCA Metrics
            mlflow.log_metric("pca_component_1_variance", variance_ratio[0])
            mlflow.log_metric("pca_component_2_variance", variance_ratio[1])
            mlflow.log_metric("pca_cumulative_variance", cumulative_variance[-1])

            # Clustering Metrics
            mlflow.log_metric("silhouette_score", sil)
            mlflow.log_metric("davies_bouldin_score", db)

            # Log Models
            mlflow.sklearn.log_model(
                kmeans_model,
                artifact_path="geo_kmeans_model"
            )

            mlflow.sklearn.log_model(
                pca_model,
                artifact_path="pca_model"
            )


        # Save artifacts
        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(kmeans_model, "artifacts/geo_kmeans_model.pkl")
        joblib.dump(pca_model, "artifacts/pca_model.pkl")

        df_featured.to_csv(
            "artifacts/processed_dataset.csv",
            index=False
        )

        self.logger.info("Training pipeline completed successfully")
