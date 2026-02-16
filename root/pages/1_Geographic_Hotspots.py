import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import streamlit as st
import plotly.express as px
import pandas as pd

from src.pipeline.inference_pipeline import InferencePipeline


st.set_page_config(page_title="Geographic Hotspots", layout="wide")

st.title("📍 Geographic Crime Hotspots")

# Load inference pipeline
pipeline = InferencePipeline()
df = pipeline.get_dataset()
kmeans_model, _ = pipeline.get_models()

# Assign clusters using trained model
geo_features = df[['Lat_scaled', 'Long_scaled']]
df['Cluster'] = kmeans_model.predict(geo_features)

st.subheader("Crime Hotspot Clusters")

fig = px.scatter_mapbox(
    df.sample(20000, random_state=42),
    lat="Latitude",
    lon="Longitude",
    color="Cluster",
    zoom=10,
    height=700
)

fig.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig, use_container_width=True)

# Cluster Summary
st.subheader("Cluster Summary")

cluster_summary = (
    df.groupby("Cluster")
    .size()
    .reset_index(name="Total Crimes")
    .sort_values(by="Total Crimes", ascending=False)
)

st.dataframe(cluster_summary)
