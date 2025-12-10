# train_logistic_regression.py
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

import preprocess  # assumes this file sits in the same src/ folder

DATA_PATH = "../data/malicious_phish.csv"
RESULTS_DIR = "../results"
MODEL_NAME = "logistic"


def evaluate_and_save(model, X_test, y_test, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)

    # Predict class labels and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    class_names = np.unique(y_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, f"confusion_{model_name}.png")
    plt.savefig(cm_path)
    plt.close()

    # ROC + PR curves (macro-averaged)
    y_test_bin = label_binarize(y_test, classes=class_names)
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        precision[i], recall[i], _ = precision_recall_curve(
            y_test_bin[:, i], y_proba[:, i]
        )
        pr_auc[i] = average_precision_score(y_test_bin[:, i], y_proba[:, i])

    # Macro average AUCs
    metrics["roc_auc_per_class"] = {str(c): float(roc_auc[i]) for i, c in enumerate(class_names)}
    metrics["pr_auc_per_class"] = {str(c): float(pr_auc[i]) for i, c in enumerate(class_names)}

    # Plot macro-averaged ROC + PR
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

    # Save metrics JSON
    metrics_path = os.path.join(results_dir, f"metrics_{model_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = preprocess.load_and_clean(DATA_PATH)
    df, feature_cols = preprocess.extract_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess.scale_and_split(df, feature_cols)

    # Combine train + val for final training after hyperparameter search
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Base model
    base_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)

    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"]
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    print("Starting Logistic Regression grid search...")
    start = time.time()
    grid.fit(X_train, y_train)
    end = time.time()
    print(f"Grid search done in {end - start:.2f} seconds")
    print("Best params:", grid.best_params_)

    # Retrain on train+val
    best_params = grid.best_params_
    best_model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        **best_params
    )
    print("Training final Logistic Regression on train+val...")
    best_model.fit(X_train_full, y_train_full)

    print("Evaluating on test set...")
    evaluate_and_save(best_model, X_test, y_test, RESULTS_DIR, MODEL_NAME)


if __name__ == "__main__":
    main()
