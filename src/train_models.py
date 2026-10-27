"""
train_models.py
---------------
Trains five classification models on the chosen dataset, evaluates them on a
held-out test set using six metrics, and saves:
  - one fitted pipeline per model  -> ../model/<name>.joblib
  - the held-out test split (features + target) -> ../test_data.csv
  - a metrics summary (for the README table)     -> ../model/metrics.json

Each saved model is a scikit-learn Pipeline (StandardScaler + classifier), so the
Streamlit app can call predict / predict_proba directly on raw feature columns
without doing any preprocessing itself and without retraining.

To use a different dataset, change DATA_PATH / TARGET_COLUMN below and re-run.
The dataset must have >= 12 feature columns and >= 500 rows (assignment rule).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "data", "breast_cancer.csv")
TARGET_COLUMN = "target"
MODEL_DIR = os.path.join(ROOT, "model")
TEST_CSV = os.path.join(ROOT, "test_data.csv")
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
assert TARGET_COLUMN in df.columns, f"target column '{TARGET_COLUMN}' not found"
feature_names = [c for c in df.columns if c != TARGET_COLUMN]
assert len(feature_names) >= 12, "dataset must have at least 12 features"
assert len(df) >= 500, "dataset must have at least 500 instances"

X = df[feature_names].values
y = df[TARGET_COLUMN].values

# 80/20 stratified split; the test split is what the Streamlit app will consume
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

# --------------------------------------------------------------------------
# Define the five models (each wrapped with a StandardScaler in a Pipeline)
# --------------------------------------------------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "decision_tree":       DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "knn":                 KNeighborsClassifier(n_neighbors=7),
    "naive_bayes":         GaussianNB(),
    "svm":                 SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

pretty = {
    "logistic_regression": "Logistic Regression",
    "decision_tree":       "Decision Tree",
    "knn":                 "KNN",
    "naive_bayes":         "Naive Bayes",
    "svm":                 "Support Vector Machine",
}


def evaluate(y_true, y_pred, y_proba):
    """Compute the six required metrics for a binary classifier."""
    return {
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC":       round(float(roc_auc_score(y_true, y_proba)), 4),
        "Precision": round(float(precision_score(y_true, y_pred)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred)), 4),
        "F1":        round(float(f1_score(y_true, y_pred)), 4),
        "MCC":       round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


# --------------------------------------------------------------------------
# Train, evaluate and save each model
# --------------------------------------------------------------------------
all_metrics = {}
for key, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]     # probability of the positive class
    all_metrics[pretty[key]] = evaluate(y_test, y_pred, y_proba)

    joblib.dump(pipe, os.path.join(MODEL_DIR, f"{key}.joblib"))
    print(f"saved {key}.joblib  ->  {all_metrics[pretty[key]]}")

# Save the metrics summary and the list of feature names / target name
joblib.dump({"feature_names": feature_names, "target": TARGET_COLUMN,
             "pretty": pretty}, os.path.join(MODEL_DIR, "metadata.joblib"))
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

# --------------------------------------------------------------------------
# Save the held-out test set (features + target) for upload to the app
# --------------------------------------------------------------------------
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df[TARGET_COLUMN] = y_test
test_df.to_csv(TEST_CSV, index=False)
print(f"\nsaved test_data.csv with shape {test_df.shape}")
print("\n==== metrics summary ====")
print(json.dumps(all_metrics, indent=2))
