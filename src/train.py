from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "creditcard.csv"
MODEL = ROOT / "models" / "fraud_model.joblib"
METRICS = ROOT / "models" / "metrics.json"

def load_data():
    if not DATA.exists():
        raise FileNotFoundError("Run `python src/download_data.py` first.")
    df = pd.read_csv(DATA)
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").astype(int)
    return df

def make_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

def score(y_true, probability, threshold):
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4),
        "average_precision_pr_auc": round(float(average_precision_score(y_true, probability)), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

df = load_data()
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
}

results, fitted = {}, {}

for name, estimator in models.items():
    print(f"Training {name}...")
    pipe = make_pipeline(estimator)
    pipe.fit(X_train, y_train)

    valid_proba = pipe.predict_proba(X_valid)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    validation_scores = [score(y_valid, valid_proba, t) for t in thresholds]
    best_validation = max(validation_scores, key=lambda x: x["f1"])

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_score = score(y_test, test_proba, best_validation["threshold"])

    results[name] = {
        "validation": best_validation,
        "test": test_score,
    }
    fitted[name] = pipe
    print(name, test_score)

best_name = max(results, key=lambda n: results[n]["validation"]["f1"])
best_threshold = results[best_name]["validation"]["threshold"]

MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"pipeline": fitted[best_name], "threshold": best_threshold},
    MODEL
)
METRICS.write_text(json.dumps({
    "best_model": best_name,
    "decision_threshold": best_threshold,
    "models": results
}, indent=2))

print(f"Best model: {best_name}")
print(f"Decision threshold: {best_threshold}")
print(f"Saved model: {MODEL}")
