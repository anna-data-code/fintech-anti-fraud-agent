from pathlib import Path
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA = Path("data/PS_20174392719_1491204439457_log.csv")
OUTPUT = Path("validation/baseline_metrics.json")

FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type_TRANSFER",
]

print("Loading PaySim data...", flush=True)

df = pd.read_csv(
    DATA,
    usecols=[
        "step", "type", "amount",
        "oldbalanceOrg", "oldbalanceDest", "isFraud"
    ],
    dtype={
        "step": "int32",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "oldbalanceDest": "float32",
        "isFraud": "int8",
    },
)

df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype("int8")

# Temporal split: first 80% of time steps for training.
# Remaining time steps are reserved for testing.
steps = np.sort(df["step"].unique())
cutoff = steps[int(len(steps) * 0.8)]

train = df[df["step"] < cutoff]
test = df[df["step"] >= cutoff]

X_train = train[FEATURES]
y_train = train["isFraud"]
X_test = test[FEATURES]
y_test = test["isFraud"]

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError("Both train and test must contain both classes.")

weight = (len(y_train) - int(y_train.sum())) / int(y_train.sum())

print(f"Training rows: {len(train):,}", flush=True)
print(f"Testing rows: {len(test):,}", flush=True)
print(f"Temporal cutoff step: {cutoff}", flush=True)
print(f"Train fraud: {int(y_train.sum()):,}", flush=True)
print(f"Test fraud: {int(y_test.sum()):,}", flush=True)
print("Training baseline XGBoost...", flush=True)

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=weight,
    random_state=42,
    n_jobs=4,
    tree_method="hist",
    eval_metric="logloss",
)

model.fit(X_train, y_train)

scores = model.predict_proba(X_test)[:, 1]
predictions = (scores >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

metrics = {
    "model": "Leakage-aware baseline XGBoost",
    "dataset": "PaySim",
    "features": FEATURES,
    "split": "Temporal, by transaction step",
    "cutoff_step": int(cutoff),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "train_fraud": int(y_train.sum()),
    "test_fraud": int(y_test.sum()),
    "threshold": 0.5,
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "pr_auc": float(average_precision_score(y_test, scores)),
    "roc_auc": float(roc_auc_score(y_test, scores)),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("\n=== BASELINE VALIDATION RESULTS ===")
print(json.dumps(metrics, indent=2))
print(f"\nSaved: {OUTPUT}")
