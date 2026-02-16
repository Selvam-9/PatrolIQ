import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.pipeline.inference_pipeline import InferencePipeline

st.set_page_config(page_title="Temporal Patterns", layout="wide")

st.title("⏰ Temporal Crime Patterns")

pipeline = InferencePipeline()
df = pipeline.get_dataset()

# -------------------------
# Hourly Distribution
# -------------------------
st.subheader("Hourly Crime Distribution")

hourly = df.groupby("Hour").size().reset_index(name="Total Crimes")

fig_hour = px.bar(
    hourly,
    x="Hour",
    y="Total Crimes",
)

st.plotly_chart(fig_hour, use_container_width=True)

# -------------------------
# Weekday vs Weekend
# -------------------------
st.subheader("Weekday vs Weekend")

weekend_data = (
    df.groupby("Is_Weekend")
    .size()
    .reset_index(name="Total Crimes")
)

fig_weekend = px.bar(
    weekend_data,
    x="Is_Weekend",
    y="Total Crimes",
)

st.plotly_chart(fig_weekend, use_container_width=True)

# -------------------------
# Monthly Trend
# -------------------------
st.subheader("Monthly Crime Trend")

monthly = df.groupby("Month").size().reset_index(name="Total Crimes")

fig_month = px.line(
    monthly,
    x="Month",
    y="Total Crimes",
)

st.plotly_chart(fig_month, use_container_width=True)
