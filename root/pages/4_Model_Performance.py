import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import mlflow
import pandas as pd

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("📊 Model Performance Monitoring")

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Get experiment
experiment = mlflow.get_experiment_by_name("PatrolIQ_Clustering")

if experiment is None:
    st.error("Experiment not found.")
else:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    if runs.empty:
        st.warning("No runs found.")
    else:
        latest_run = runs.iloc[0]

        st.subheader("Latest Run Parameters")

        params = {
            "Geo KMeans Clusters": latest_run["params.geo_kmeans_clusters"],
            "PCA Components": latest_run["params.pca_components"]
        }

        st.json(params)

        st.subheader("Evaluation Metrics")

        metrics = {
            "Silhouette Score": latest_run["metrics.silhouette_score"],
            "Davies-Bouldin Score": latest_run["metrics.davies_bouldin_score"],
            "PCA Component 1 Variance": latest_run["metrics.pca_component_1_variance"],
            "PCA Component 2 Variance": latest_run["metrics.pca_component_2_variance"],
            "Total PCA Variance": latest_run["metrics.pca_cumulative_variance"]
        }

        st.json(metrics)
