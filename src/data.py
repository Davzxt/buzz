"""Data loading and protocol helpers. Uses the bundled scikit-learn digits set (offline)."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digits_split(seed, test_size=0.3):
    """Return (Xtr, ytr, Xte, yte) with pixel intensities scaled to [0, 1]."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return Xtr, ytr, Xte, yte


def split_validation(X, y, seed, test_size=0.2):
    """Inner split of the training set, used only for ablations. The test set is never touched.

    Returns (Xa, ya, Xv, yv).
    """
    Xa, Xv, ya, yv = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return Xa, ya, Xv, yv


def class_incremental_tasks(classes_per_task=2, n_classes=10):
    order = list(range(n_classes))
    return [order[i : i + classes_per_task] for i in range(0, n_classes, classes_per_task)]


def sample_few_shot(y, k, classes, rng):
    indices = [rng.choice(np.flatnonzero(y == c), size=k, replace=False) for c in classes]
    return np.concatenate(indices)
