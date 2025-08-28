# pages/3_Prediction.py
import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title("🔮 Prediction")

MODELS_DIR = "models"
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_sales_method.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Load metrics & model list
if not os.path.exists(METRICS_PATH):
    st.error("Metrics file not found. Run train_models.py first to create models/ and metrics.json")
    st.stop()

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

model_names = list(metrics.keys())
model_choice = st.selectbox("Choose model", model_names)

st.markdown("**Model performance (on test set)**")
st.json(metrics[model_choice])

# Load feature names
if not os.path.exists(FEATURES_PATH):
    st.error("feature_names.pkl not found. Run train_models.py first.")
    st.stop()

feature_names = joblib.load(FEATURES_PATH)

# Load label encoder if exists
le = None
if os.path.exists(ENCODER_PATH):
    le = joblib.load(ENCODER_PATH)

# Load model
model_path = os.path.join(MODELS_DIR, f"best_{model_choice}_model.pkl")
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Run train_models.py first.")
    st.stop()

model = joblib.load(model_path)
st.success(f"Loaded model: {model_choice}")

# Prediction mode
mode = st.radio("Prediction mode", ["Batch (last 5 rows from cleaned CSV)", "Single Input"])

# Helper to preprocess a dataframe to the model feature order
def preprocess_input(df_in: pd.DataFrame) -> pd.DataFrame:
    X = df_in.copy()
    # If original CSV had sales_method string and we saved encoder, transform it
    if "sales_method" in X.columns and le is not None and X["sales_method"].dtype == object:
        X["sales_method"] = le.transform(X["sales_method"].astype(str))
    # Drop any columns not used by model
    for col in X.columns:
        if col not in feature_names:
            X = X.drop(columns=[col])
    # Reindex columns to match training (fill missing with 0)
    X = X.reindex(columns=feature_names, fill_value=0)
    return X

if mode.startswith("Batch"):
    # Use cleaned CSV if exists
    clean_path = os.path.join("data", "clean_sales_data.csv")
    if os.path.exists(clean_path):
        df_all = pd.read_csv(clean_path)
        st.info("Using cleaned dataset: data/clean_sales_data.csv")
    else:
        df_all = pd.read_csv(os.path.join("data", "sales_data.csv"))
        st.warning("Cleaned dataset not found — using original CSV (may cause mismatch)")

    if len(df_all) < 5:
        st.warning("Dataset has less than 5 rows.")
    else:
        df_tail = df_all.tail(5).copy()
        st.subheader("Input (last 5 rows)")
        st.dataframe(df_tail)

        X_batch = df_tail.drop(columns=["revenue"], errors="ignore")
        X_prep = preprocess_input(X_batch)
        # Predict
        try:
            preds = model.predict(X_prep)
            out = df_tail.copy()
            out["Predicted_Revenue"] = preds
            st.subheader("Predictions")
            st.dataframe(out.reset_index(drop=True))
            # Option to download
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.subheader("Single input prediction")
    # Build UI for required features
    # We'll try to infer common features: nb_sold, week, sales_method — otherwise allow numeric inputs for each feature
    input_dict = {}
    if "nb_sold" in feature_names:
        input_dict["nb_sold"] = st.number_input("nb_sold", min_value=0, value=10)
    if "week" in feature_names:
        if os.path.exists(os.path.join("data", "clean_sales_data.csv")):
            weeks = pd.read_csv(os.path.join("data", "clean_sales_data.csv"))["week"].dropna().unique().tolist()
            input_dict["week"] = st.selectbox("week", sorted(weeks))
        else:
            input_dict["week"] = st.number_input("week", min_value=1, max_value=52, value=1)
    if "sales_method" in feature_names:
        if le is not None:
            # Show original categories if label encoder exists
            classes = le.classes_.tolist()
            sel = st.selectbox("sales_method", classes)
            input_dict["sales_method"] = sel
        else:
            # Let user type or choose common methods
            sel = st.selectbox("sales_method", ["Email", "Call", "Email + Call"])
            input_dict["sales_method"] = sel

    # For any remaining feature_names that are not covered, give numeric inputs default 0
    for feat in feature_names:
        if feat not in input_dict:
            # default numeric input
            input_dict[feat] = st.number_input(f"{feat}", value=0.0)

    if st.button("Predict"):
        X_new = pd.DataFrame([input_dict])
        X_prep = preprocess_input(X_new)
        try:
            pred = model.predict(X_prep)[0]
            st.success(f"Predicted revenue: {pred:.2f}")
            st.write("Model test metrics:")
            st.json(metrics[model_choice])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
