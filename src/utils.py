# utils.py
"""
General utility functions shared across training / evaluation scripts.
"""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_config(path: str = "config.json") -> dict:
    """Load JSON configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str):
    """Save dictionary as pretty JSON."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def set_random_seed(seed: int = 42):
    """Set RNG seeds for reproducibility (numpy + random)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch not installed; ignore
        pass


def plot_confusion_matrix(cm, labels, title: str, save_path: str):
    """Simple wrapper to plot and save a confusion matrix."""
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path)
    plt.close()
