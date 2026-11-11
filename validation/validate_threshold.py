
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
OUTPUT = Path("validation/threshold_metrics.json")

FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type_TRANSFER",
]

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

steps = np.sort(df["step"].unique())
cutoff_train = steps[int(len(steps) * 0.6)]
cutoff_test = steps[int(len(steps) * 0.8)]

train = df[df["step"] < cutoff_train]
validation = df[
    (df["step"] >= cutoff_train) &
    (df["step"] < cutoff_test)
]
test = df[df["step"] >= cutoff_test]

for name, part in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    print(
        f"{name}: {len(part):,} rows, "
        f"{int(part['isFraud'].sum()):,} fraud, "
        f"fraud rate {part['isFraud'].mean():.2%}",
        flush=True,
    )
    if part["isFraud"].nunique() != 2:
        raise ValueError(f"{name} must contain both classes.")

X_train = train[FEATURES]
y_train = train["isFraud"]

weight = (len(train) - int(y_train.sum())) / int(y_train.sum())

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

print("\nTraining XGBoost...", flush=True)
model.fit(X_train, y_train)

val_scores = model.predict_proba(validation[FEATURES])[:, 1]
val_labels = validation["isFraud"].to_numpy()

# Business rule: maximize precision while keeping
# validation recall at or above 95%.
thresholds = np.arange(0.05, 0.951, 0.05)
candidates = []

for threshold in thresholds:
    pred = (val_scores >= threshold).astype(int)
    candidates.append({
        "threshold": round(float(threshold), 2),
        "precision": float(precision_score(
            val_labels, pred, zero_division=0
        )),
        "recall": float(recall_score(
            val_labels, pred, zero_division=0
        )),
        "f1": float(f1_score(
            val_labels, pred, zero_division=0
        )),
        "false_positives": int(
            ((pred == 1) & (val_labels == 0)).sum()
        ),
        "false_negatives": int(
            ((pred == 0) & (val_labels == 1)).sum()
        ),
    })

eligible = [x for x in candidates if x["recall"] >= 0.95]

if eligible:
    best = max(
        eligible,
        key=lambda x: (x["precision"], x["recall"])
    )
else:
    best = max(candidates, key=lambda x: x["recall"])
    print(
        "WARNING: No tested threshold met 95% validation recall."
    )

print("\n=== VALIDATION THRESHOLDS ===")
for row in candidates:
    print(
        f"Threshold {row['threshold']:.2f} | "
        f"Precision {row['precision']:.3f} | "
        f"Recall {row['recall']:.3f} | "
        f"FP {row['false_positives']:,} | "
        f"FN {row['false_negatives']:,}"
    )

selected_threshold = best["threshold"]

test_scores = model.predict_proba(test[FEATURES])[:, 1]
test_labels = test["isFraud"].to_numpy()
test_pred = (test_scores >= selected_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    test_labels, test_pred, labels=[0, 1]
).ravel()

results = {
    "model": "Leakage-aware XGBoost",
    "features": FEATURES,
    "split": "Temporal train/validation/test",
    "train_cutoff": int(cutoff_train),
    "test_cutoff": int(cutoff_test),
    "selection_rule": "Maximize validation precision with recall >= 0.95",
    "constraint_met": bool(eligible),
    "selected_threshold": selected_threshold,
    "validation_selection": best,
    "test": {
        "rows": int(len(test)),
        "fraud": int(test["isFraud"].sum()),
        "precision": float(precision_score(
            test_labels, test_pred, zero_division=0
        )),
        "recall": float(recall_score(
            test_labels, test_pred, zero_division=0
        )),
        "f1": float(f1_score(
            test_labels, test_pred, zero_division=0
        )),
        "pr_auc": float(average_precision_score(
            test_labels, test_scores
        )),
        "roc_auc": float(roc_auc_score(
            test_labels, test_scores
        )),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },
}


# Compare thresholds on the same trained model and test set.
# This is a diagnostic comparison, not a new threshold selection.

print("\n=== SAME-MODEL THRESHOLD COMPARISON ===")

threshold_comparison = []

for threshold in [0.50, 0.95]:
    predictions = (test_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        test_labels, predictions, labels=[0, 1]
    ).ravel()

    row = {
        "threshold": threshold,
        "precision": float(precision_score(
            test_labels, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            test_labels, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            test_labels, predictions, zero_division=0
        )),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }

    threshold_comparison.append(row)

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision {row['precision']:.4f} | "
        f"Recall {row['recall']:.4f} | "
        f"F1 {row['f1']:.4f} | "
        f"FP {row['false_positives']:,} | "
        f"FN {row['false_negatives']:,}"
    )

results["test_threshold_comparison"] = threshold_comparison

OUTPUT.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("\n=== FINAL TEST RESULTS ===")
print(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT}")
