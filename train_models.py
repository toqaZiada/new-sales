# train_models.py
import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

# -----------------------
# Paths & create folders
# -----------------------
DATA_PATH = os.path.join("data", "sales_data.csv")   # place your CSV here
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------
# Load data
# -----------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

# -----------------------
# Basic cleaning & info
# -----------------------
# Drop obvious ID
if "customer_id" in df.columns:
    df = df.drop(columns=["customer_id"])

# Inspect missing values
print("Missing values per column:\n", df.isnull().sum())

# --- Outlier removal using IQR for numeric columns (robust, but optional)
numeric_cols = [c for c in ["nb_sold", "revenue", "years_as_customer", "nb_site_visits"] if c in df.columns]
df_clean = df.copy()
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
print("After IQR outlier removal shape:", df_clean.shape)

# -----------------------
# Categorical handling
# -----------------------
# Label encode sales_method (save encoder)
if "sales_method" in df_clean.columns:
    le = LabelEncoder()
    df_clean["sales_method_encoded"] = le.fit_transform(df_clean["sales_method"].astype(str))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_sales_method.pkl"))
    # If you prefer to replace original column:
    df_clean = df_clean.drop(columns=["sales_method"])
    df_clean = df_clean.rename(columns={"sales_method_encoded": "sales_method"})
    print("Saved label encoder: models/label_encoder_sales_method.pkl")
else:
    print("Warning: 'sales_method' column not found.")

# -----------------------
# Drop high-cardinality or low-importance
# -----------------------
if "state" in df_clean.columns:
    df_clean = df_clean.drop(columns=["state"])

# As you decided earlier: drop years_as_customer and nb_site_visits if present
for col in ["years_as_customer", "nb_site_visits"]:
    if col in df_clean.columns:
        df_clean = df_clean.drop(columns=[col])

print("Columns used for modeling:", df_clean.columns.tolist())

# Save cleaned CSV for app convenience
os.makedirs("data", exist_ok=True)
df_clean.to_csv(os.path.join("data", "clean_sales_data.csv"), index=False)
print("Saved cleaned dataset: data/clean_sales_data.csv")

# -----------------------
# Prepare train/holdout
# -----------------------
# Keep last 5 rows as holdout if dataset is ordered (like in your notebook)
if len(df_clean) < 10:
    raise ValueError("Dataset too small after cleaning.")
df_train = df_clean.iloc[:-5, :].copy()
df_holdout = df_clean.iloc[-5:, :].copy()

X = df_train.drop(columns=["revenue"])
y = df_train["revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train/Test shapes:", X_train.shape, X_test.shape)

# Save feature names (columns order used for training)
feature_names = X.columns.tolist()
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
print("Saved feature names to models/feature_names.pkl")

# -----------------------
# Models & parameter grids
# -----------------------
models = {
    "CART": DecisionTreeRegressor(random_state=42),
    "RF": RandomForestRegressor(random_state=42),
    "LR": LinearRegression(),
    "SVM": SVR(),
    "KNN": KNeighborsRegressor(),
    "XGB": XGBRegressor(random_state=42, verbosity=0)
}

scalers = [StandardScaler(), RobustScaler(), MinMaxScaler()]

param_grids = {
    "CART": {
        "Scaler": scalers,
        "CART__max_depth": [3, 6, 9, None],
        "CART__min_samples_split": [2, 5, 10],
        "CART__min_samples_leaf": [1, 2, 4]
    },
    "RF": {
        "Scaler": scalers,
        "RF__n_estimators": [50, 100],
        "RF__max_depth": [None, 5, 10],
        "RF__min_samples_split": [2, 5],
        "RF__min_samples_leaf": [1, 2]
    },
    "LR": {
        "Scaler": scalers,
        "LR__fit_intercept": [True]
    },
    "SVM": {
        "Scaler": scalers,
        "SVM__C": [0.1, 1, 10],
        "SVM__kernel": ["rbf"],
        "SVM__gamma": ["scale", "auto"]
    },
    "KNN": {
        "Scaler": scalers,
        "KNN__n_neighbors": [3, 5, 7],
        "KNN__weights": ["uniform", "distance"],
        "KNN__p": [1, 2]
    },
    "XGB": {
        "Scaler": scalers,
        "XGB__n_estimators": [50, 100],
        "XGB__max_depth": [3, 6],
        "XGB__learning_rate": [0.01, 0.1],
        "XGB__subsample": [0.8, 1.0],
        "XGB__colsample_bytree": [0.8, 1.0]
    }
}

scoring = {
    "R2": make_scorer(r2_score),
    "MAE": make_scorer(mean_absolute_error)
}
# Train, evaluate, save
# -----------------------
all_metrics = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    steps = [("Scaler", MinMaxScaler()), (name, model)]
    pipeline = Pipeline(steps=steps)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=5,
        scoring=scoring,
        refit="R2",
        return_train_score=True,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Predictions and metrics
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # --- convert best_params to JSON safe dict ---
    safe_params = {}
    for k, v in grid.best_params_.items():
        if hasattr(v, "__class__"):
            safe_params[k] = v.__class__.__name__   # e.g. "StandardScaler"
        else:
            safe_params[k] = v

    all_metrics[name] = {
        "R2_train": float(r2_train),
        "R2_test": float(r2_test),
        "MAE_train": float(mae_train),
        "MAE_test": float(mae_test),
        "best_params": safe_params
    }

    # Save best model
    model_path = os.path.join(MODELS_DIR, f"best_{name}_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Saved model: {model_path}")

    # Optional: cross_val_score summary
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        print(f"{name} CV R2 mean: {cv_scores.mean():.3f}, std: {cv_scores.std():.3f}")
    except Exception as e:
        print("CV failed:", e)

# Save metrics safely
metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=4)
print(f"Saved metrics: {metrics_path}")