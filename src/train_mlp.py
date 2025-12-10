# train_mlp.py
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import preprocess

DATA_PATH = "../data/malicious_phish.csv"
RESULTS_DIR = "../results"
MODEL_NAME = "mlp"


def plot_learning_curve(loss_curve, results_dir, model_name):
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(f"MLP Training Loss - {model_name}")
    plt.tight_layout()
    path = os.path.join(results_dir, f"loss_{model_name}.png")
    plt.savefig(path)
    plt.close()


def evaluate_and_save(model, X_test, y_test, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    class_names = np.unique(y_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report
    }

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, f"confusion_{model_name}.png")
    plt.savefig(cm_path)
    plt.close()

    # Multi-class ROC (macro-ish visualization)
    y_test_bin = label_binarize(y_test, classes=class_names)
    n_classes = y_test_bin.shape[1]

    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, alpha=0.3, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {model_name}")
    plt.legend(fontsize="small")
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

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    base_model = MLPClassifier(max_iter=100, random_state=42)

    param_grid = {
        "hidden_layer_sizes": [(64,), (128, 64)],
        "activation": ["relu"],
        "learning_rate_init": [0.001, 0.0005],
        "batch_size": [64, 128]
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    print("Starting MLP grid search...")
    start = time.time()
    grid.fit(X_train, y_train)
    end = time.time()
    print(f"Grid search done in {end - start:.2f} seconds")
    print("Best params:", grid.best_params_)

    best_params = grid.best_params_
    best_model = MLPClassifier(
        max_iter=200,
        random_state=42,
        **best_params
    )

    print("Training final MLP on train+val...")
    best_model.fit(X_train_full, y_train_full)

    # Plot training loss curve
    plot_learning_curve(best_model.loss_curve_, RESULTS_DIR, MODEL_NAME)

    print("Evaluating on test set...")
    evaluate_and_save(best_model, X_test, y_test, RESULTS_DIR, MODEL_NAME)


if __name__ == "__main__":
    main()
