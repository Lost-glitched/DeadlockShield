"""
Evaluate the trained model STRICTLY on the held-out test rows saved by
train_model.py (split column == "test"). This guarantees no data leakage:
the model never saw these rows during training.

Output: results/ml_evaluation_metrics.csv   (accuracy, precision, recall, F1, FPR, FNR)
        results/ml_confusion_matrix.csv
"""

import csv
import os
import pickle
import time

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLIT_INFO_PATH = os.path.join(BASE, "results", "train_test_split_info.csv")
MODEL_PATH = os.path.join(BASE, "models", "model.pkl")
METRICS_OUT = os.path.join(BASE, "results", "ml_evaluation_metrics.csv")
CONFMAT_OUT = os.path.join(BASE, "results", "ml_confusion_matrix.csv")


def main():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    X_test, y_test = [], []
    with open(SPLIT_INFO_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "test":
                continue
            X_test.append([float(row[c]) for c in feature_names])
            y_test.append(int(row["label"]))

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    total_pred_time = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    with open(METRICS_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["test_set_size", len(y_test)])
        writer.writerow(["accuracy", acc])
        writer.writerow(["precision", prec])
        writer.writerow(["recall", rec])
        writer.writerow(["f1_score", f1])
        writer.writerow(["false_positive_rate", fpr])
        writer.writerow(["false_negative_rate", fnr])
        writer.writerow(["total_prediction_time_sec", total_pred_time])
        writer.writerow(["avg_prediction_time_sec_per_sample", total_pred_time / max(1, len(y_test))])

    with open(CONFMAT_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "predicted_unsafe(0)", "predicted_safe(1)"])
        writer.writerow(["actual_unsafe(0)", tn, fp])
        writer.writerow(["actual_safe(1)", fn, tp])

    print(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
    print(f"FPR={fpr:.3f} FNR={fnr:.3f}")
    print(f"Saved metrics to {METRICS_OUT}")
    print(f"Saved confusion matrix to {CONFMAT_OUT}")


if __name__ == "__main__":
    main()
