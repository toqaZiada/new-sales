# pages/1_Home.py
import streamlit as st

st.title("🏠 Home")
st.write("""
This project shows a pipeline for predicting sales revenue:
- Data cleaning & preprocessing (IQR outlier removal, encoding).
- Training multiple models (CART, RandomForest, LinearRegression, SVM, KNN, XGB) with GridSearch.
- Saving models, feature names, label encoder, and metrics for a Streamlit app.
""")
st.markdown("**How to run**: 1) `python train_models.py`  2) `streamlit run app.py`")
