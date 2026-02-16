import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import plotly.express as px
import pandas as pd

from src.pipeline.inference_pipeline import InferencePipeline

st.set_page_config(page_title="PCA Visualization", layout="wide")

st.title("📉 PCA Crime Pattern Visualization")

pipeline = InferencePipeline()
df = pipeline.get_dataset()
_, pca_model = pipeline.get_models()

# Prepare PCA features
pca_features = df[
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

# Transform using trained PCA
pca_transformed = pca_model.transform(pca_features)

pca_df = pd.DataFrame(
    pca_transformed,
    columns=["PC1", "PC2"]
)

# Add cluster if exists
if "Cluster" in df.columns:
    pca_df["Cluster"] = df.loc[pca_features.index, "Cluster"]

st.subheader("2D PCA Projection")

fig = px.scatter(
    pca_df.sample(20000, random_state=42),
    x="PC1",
    y="PC2",
    color="Cluster" if "Cluster" in pca_df.columns else None,
    opacity=0.6
)

st.plotly_chart(fig, use_container_width=True)

# Show variance information
explained = pca_model.explained_variance_ratio_

st.subheader("Explained Variance")

st.write({
    "PC1 Variance": explained[0],
    "PC2 Variance": explained[1],
    "Total Variance": explained.sum()
})
