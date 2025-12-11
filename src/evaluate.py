# evaluate.py
"""
Reusable evaluation functions for multiclass and binary models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from utils import ensure_dir, save_json


def evaluate_multiclass(model, X_test, y_test, results_dir: str, model_name: str):
    """
    Evaluate a multiclass classifier.

    Saves:
        - metrics_<model_name>.json
        - confusion_<model_name>.png
        - roc_pr_<model_name>.png  (per-class ROC / PR curves)
    """
    ensure_dir(results_dir)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    class_names = np.unique(y_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report,
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_path = os.path.join(results_dir, f"confusion_{model_name}.png")
    from utils import plot_confusion_matrix

    plot_confusion_matrix(cm, class_names, f"Confusion Matrix - {model_name}", cm_path)

    # ROC + PR per class
    y_test_bin = label_binarize(y_test, classes=class_names)
    n_classes = y_test_bin.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}
    precision = {}
    recall = {}
    pr_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        precision[i], recall[i], _ = precision_recall_curve(
            y_test_bin[:, i], y_proba[:, i]
        )
        pr_auc[i] = average_precision_score(y_test_bin[:, i], y_proba[:, i])

    metrics["roc_auc_per_class"] = {
        str(cls): float(roc_auc[i]) for i, cls in enumerate(class_names)
    }
    metrics["pr_auc_per_class"] = {
        str(cls): float(pr_auc[i]) for i, cls in enumerate(class_names)
    }

    # Plot ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], alpha=0.3, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {model_name}")
    plt.legend(fontsize="small")
    plt.tight_layout()
    roc_path = os.path.join(results_dir, f"roc_pr_{model_name}.png")
    plt.savefig(roc_path)
    plt.close()

    # Save metrics
    metrics_path = os.path.join(results_dir, f"metrics_{model_name}.json")
    save_json(metrics, metrics_path)


def evaluate_binary_scores(
    scores,
    y_true_binary,
    results_dir: str,
    model_name: str,
    positive_label_name: str = "malicious",
    negative_label_name: str = "benign",
):
    """
    Evaluate a *binary* detector given anomaly scores.

    scores: array-like, higher means *more malicious*.
    y_true_binary: 0 (benign) / 1 (malicious)
    """
    ensure_dir(results_dir)

    # Chosen threshold = 0 (you can vary this if you want)
    y_pred_binary = (scores > 0).astype(int)

    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary)
    rec = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    labels = [negative_label_name, positive_label_name]
    from utils import plot_confusion_matrix

    cm_path = os.path.join(results_dir, f"confusion_{model_name}.png")
    plot_confusion_matrix(cm, labels, f"Confusion Matrix - {model_name}", cm_path)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true_binary, scores)
    roc_auc = auc(fpr, tpr)
    metrics["roc_auc"] = float(roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(results_dir, f"roc_{model_name}.png")
    plt.savefig(roc_path)
    plt.close()

    metrics_path = os.path.join(results_dir, f"metrics_{model_name}.json")
    save_json(metrics, metrics_path)
