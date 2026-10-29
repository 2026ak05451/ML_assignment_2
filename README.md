# Breast Cancer Classification — Model Comparison & Streamlit App

An end-to-end machine-learning project that trains **five classification models**
on the Breast Cancer Wisconsin (Diagnostic) dataset, compares them on six
evaluation metrics, and serves an **interactive Streamlit app** that loads the
saved models and evaluates an uploaded test CSV (without retraining).

---

## a. Problem Statement

Given 30 numeric features computed from digitised images of fine-needle-aspiration
breast biopsies (cell radius, texture, perimeter, area, smoothness, concavity,
symmetry, etc.), predict whether a tumour is **benign (1)** or **malignant (0)**.
This is a **binary classification** problem. The goal is to train several standard
classifiers, compare their performance, and deploy an app where a user can upload
a test set and see each model's metrics, confusion matrix and classification report.

## b. Dataset Description

| Property | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository |
| Task | Binary classification |
| Instances | 569 |
| Features | 30 (all numeric) |
| Target | `target` — 1 = benign, 0 = malignant (sklearn encoding) |
| Class balance | 357 benign / 212 malignant |
| Missing values | None |

The 30 features are the **mean**, **standard error**, and **worst** values of ten
cell-nucleus measurements. Features are standardised (`StandardScaler`) inside each
model pipeline. The data is split **80 % train / 20 % test** (stratified,
`random_state=42`); the 20 % test split is saved as `test_data.csv` and is what the
Streamlit app consumes.

> Minimum-size requirements satisfied: **30 features ≥ 12** and **569 instances ≥ 500**.

## c. GitHub Repository Link

https://github.com/2026ak05451/ML_assignment_2

## d. Models Used — Evaluation Metrics

All five models were trained on the **same** dataset and evaluated on the **same**
held-out test set. Each model is a scikit-learn `Pipeline(StandardScaler → classifier)`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9147 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

*(Metrics are computed for the positive class = benign (1), scikit-learn's default.
Reproduce exactly by running `python src/train_models.py`.)*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest AUC (0.9954) and top accuracy/F1/MCC. After scaling, the classes are nearly linearly separable, so a simple linear model generalises extremely well and is cheap to train. |
| Decision Tree | Weakest model here. Lowest AUC (0.9147) because a single tree produces coarse, step-like probability estimates, and its lower recall (0.9028) means it misses more positives. Prone to overfitting without an ensemble. |
| KNN | Very strong after standardisation (AUC 0.9884). Notably **recall = 1.0** — it misses no positive cases — though a slightly lower precision (0.96) shows a few false positives. Distance-based methods benefit a lot from feature scaling. |
| Naive Bayes | High AUC (0.9868, good ranking) but lower accuracy (0.9298). The features are correlated, which violates the conditional-independence assumption, so hard-label accuracy suffers even though probability ranking stays good. |
| Support Vector Machine | Ties Logistic Regression for the best accuracy/F1/MCC with a very high AUC (0.9950). The RBF kernel with scaled inputs captures the decision boundary cleanly. |
| **Overall winner for this dataset** | **Logistic Regression** — marginally the highest AUC and tied for the best Accuracy/F1/MCC, while being the simplest, fastest and most interpretable model. Support Vector Machine is an essentially equal runner-up. |

---

## Repository Structure

```
project-folder/
├── app.py                 # Streamlit app (loads saved models, evaluates uploaded test CSV)
├── requirements.txt
├── README.md
├── test_data.csv          # 20% held-out test split (features + target) for the app
├── data/
│   └── breast_cancer.csv  # full dataset used for training
├── src/
│   └── train_models.py    # trains all 5 models, computes metrics, saves artifacts
└── model/                 # saved trained pipelines + metadata
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metadata.joblib
    └── metrics.json
```

## How to Run Locally

```bash
pip install -r requirements.txt
python src/train_models.py      # (re)generates the model/ files, test_data.csv, metrics
streamlit run app.py
```

Then open the local URL Streamlit prints. Upload `test_data.csv` (or any test CSV with
the 30 feature columns + a `target` column), pick a model from the dropdown, and the
app shows the six metrics, the confusion matrix and the classification report.

## Live Streamlit App

https://2026ak05451-ml-assignment-2-app-295a4h.streamlit.app/

## Deploying on Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. **New app** → select this repo → branch `main` → main file `app.py` → **Deploy**.
4. Once it builds, copy the public URL and paste it above and into your submission PDF.
