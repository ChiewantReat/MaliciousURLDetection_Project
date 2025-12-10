# train_random_forest.py
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import preprocess

DATA_PATH = "../data/malicious_phish.csv"
RESULTS_DIR = "../results"
MODEL_NAME = "random_forest"


def plot_feature_importances(model, feature_names, results_dir, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(sorted_features)), sorted_importances)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(f"Feature Importances - {model_name}")
    plt.tight_layout()
    path = os.path.join(results_dir, f"feature_importance_{model_name}.png")
    plt.savefig(path)
    plt.close()


def evaluate_and_save(model, X_test, y_test, feature_cols, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)

    y_pred = model.predict(X_test)
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

    # Feature importances plot
    plot_feature_importances(model, feature_cols, results_dir, model_name)

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

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20, 40],
        "class_weight": [None, "balanced"]
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    print("Starting Random Forest grid search...")
    start = time.time()
    grid.fit(X_train, y_train)
    end = time.time()
    print(f"Grid search done in {end - start:.2f} seconds")
    print("Best params:", grid.best_params_)

    best_params = grid.best_params_
    best_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **best_params
    )

    print("Training final Random Forest on train+val...")
    best_model.fit(X_train_full, y_train_full)

    print("Evaluating on test set...")
    evaluate_and_save(best_model, X_test, y_test, feature_cols, RESULTS_DIR, MODEL_NAME)


if __name__ == "__main__":
    main()
