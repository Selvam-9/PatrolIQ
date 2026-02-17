import streamlit as st

st.set_page_config(
    page_title="PatrolIQ",
    layout="wide"
)

st.title("🚔 PatrolIQ – Crime Intelligence Platform")

st.markdown("""
### 🔍 Overview

PatrolIQ is an AI-powered crime hotspot intelligence system built using:

- 📍 Geographic Clustering (KMeans)
- 📉 Dimensionality Reduction (PCA)
- ⏰ Temporal Crime Pattern Analysis
- 📊 MLflow Experiment Tracking
- 🌐 Interactive Streamlit Dashboard

---

### 🎯 Project Objectives

✔ Identify geographic crime hotspots  
✔ Detect high-risk time periods  
✔ Reduce high-dimensional crime features into interpretable components  
✔ Compare clustering performance using evaluation metrics  
✔ Deploy a production-ready safety intelligence platform  

---

### 🧠 Machine Learning Techniques Used

- **K-Means Clustering** for hotspot detection  
- **DBSCAN & Hierarchical Clustering** (offline evaluation)  
- **PCA (Principal Component Analysis)** for feature reduction  

---

### 📊 Dashboard Pages

Use the sidebar to explore:

1. **Geographic Hotspots** – Crime cluster map  
2. **Temporal Patterns** – Hourly & seasonal crime trends  
3. **PCA Visualization** – 2D projection of crime features  

---

### 🏙 Dataset

Chicago Crime Dataset (2001–Present)  
Sample Size: ~500,000 records  
Features Used: 22+ engineered variables  

---

Built for production deployment using modular architecture and MLflow tracking.
""")
