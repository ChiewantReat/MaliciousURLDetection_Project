# train_isolation_forest.py
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

import preprocess

DATA_PATH = "../data/malicious_phish.csv"
RESULTS_DIR = "../results"
MODEL_NAME = "isolation_forest"


def evaluate_and_save(model, X_test, y_test_binary, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)

    # IsolationForest predict: 1 = normal, -1 = anomaly
    y_pred_iso = model.predict(X_test)
    # Map to 0 (benign) and 1 (malicious)
    y_pred_binary = np.where(y_pred_iso == -1, 1, 0)

    acc = accuracy_score(y_test_binary, y_pred_binary)
    prec = precision_score(y_test_binary, y_pred_binary)
    rec = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

    cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malicious"])
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name} (binary)")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, f"confusion_{model_name}.png")
    plt.savefig(cm_path)
    plt.close()

    # ROC curve using decision_function scores
    scores = model.decision_function(X_test)  # higher = more normal
    # Flip sign so higher means more malicious
    fpr, tpr, _ = roc_curve(y_test_binary, -scores)
    roc_auc = auc(fpr, tpr)
    metrics["roc_auc"] = float(roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name} (binary)")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(results_dir, f"roc_{model_name}.png")
    plt.savefig(roc_path)
    plt.close()

    metrics_path = os.path.join(results_dir, f"metrics_{model_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = preprocess.load_and_clean(DATA_PATH)
    df, feature_cols = preprocess.extract_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess.scale_and_split(df, feature_cols)

    # For IsolationForest we ignore labels when fitting, but we still use them for eval
    X_train_full = np.vstack([X_train, X_val])

    # Build binary labels: 0 = benign, 1 = malicious (phishing/malware/defacement)
    def to_binary(y_array):
        return np.array([0 if label == "benign" else 1 for label in y_array])

    y_test_binary = to_binary(y_test)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",  # lets model estimate anomaly proportion
        random_state=42,
        n_jobs=-1
    )

    print("Training Isolation Forest on train+val (unsupervised)...")
    start = time.time()
    model.fit(X_train_full)
    end = time.time()
    print(f"Isolation Forest training done in {end - start:.2f} seconds")

    print("Evaluating on test set (binary benign vs malicious)...")
    evaluate_and_save(model, X_test, y_test_binary, RESULTS_DIR, MODEL_NAME)


if __name__ == "__main__":
    main()
