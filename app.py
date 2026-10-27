"""
Streamlit app — Breast Cancer Wisconsin classification demo.

Loads five pre-trained models (saved as scikit-learn Pipelines) and evaluates
them on an uploaded TEST csv. The app never retrains: it only loads the saved
pipeline, runs predictions, and shows the metrics, confusion matrix and
classification report.

Required app features implemented:
  (a) Dataset upload option (CSV, test data)
  (b) Model selection dropdown
  (c) Display of evaluation metrics
  (d) Confusion matrix + classification report
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
DEFAULT_TEST = os.path.join(HERE, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression":    "logistic_regression.joblib",
    "Decision Tree":          "decision_tree.joblib",
    "KNN":                    "knn.joblib",
    "Naive Bayes":            "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="ML Classifier Comparison", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load model metadata and all five fitted pipelines once (cached)."""
    meta = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    models = {name: joblib.load(os.path.join(MODEL_DIR, fn))
              for name, fn in MODEL_FILES.items()}
    return meta, models


meta, MODELS = load_artifacts()
FEATURES = meta["feature_names"]
TARGET = meta["target"]

st.title("🩺 Breast Cancer Classification — Model Comparison")
st.markdown(
    "Upload a **test CSV** (feature columns + the `target` column). The app loads a "
    "pre-trained model and reports its evaluation metrics, confusion matrix and "
    "classification report. **No retraining happens** — only the saved model is used.")

# ---- (a) dataset upload ----
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader("Test CSV (features + target)", type=["csv"])

# ---- (b) model selection dropdown ----
st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

# Load the test dataframe (uploaded, else the bundled sample test set)
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded uploaded file with shape {df.shape}.")
else:
    df = pd.read_csv(DEFAULT_TEST)
    st.info("No file uploaded — using the bundled sample `test_data.csv`. "
            "Upload your own test CSV from the sidebar to evaluate it.")

# Validate columns
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"The uploaded CSV is missing {len(missing)} required feature column(s): "
             f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    st.stop()
if TARGET not in df.columns:
    st.error(f"The uploaded CSV must contain the target column '{TARGET}'.")
    st.stop()

X = df[FEATURES].values
y = df[TARGET].values

st.subheader("Preview of test data")
st.dataframe(df.head(), use_container_width=True)

# Predict with the selected (already-trained) model
model = MODELS[model_name]
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# ---- (c) evaluation metrics ----
metrics = {
    "Accuracy":  accuracy_score(y, y_pred),
    "AUC":       roc_auc_score(y, y_proba),
    "Precision": precision_score(y, y_pred),
    "Recall":    recall_score(y, y_pred),
    "F1":        f1_score(y, y_pred),
    "MCC":       matthews_corrcoef(y, y_pred),
}

st.subheader(f"Evaluation metrics — {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

st.table(pd.DataFrame({k: [f"{v:.4f}"] for k, v in metrics.items()}, index=["score"]))

# ---- (d) confusion matrix + classification report ----
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, digits=4)
    st.code(report)

st.caption("Models are loaded from the saved .joblib pipelines in /model. "
           "Metrics are computed on the uploaded test data only.")
