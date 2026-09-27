"""Evaluation metrics."""
import numpy as np
from sklearn.metrics import average_precision_score


def mean_average_precision(y_true, y_score):
    """Macro mAP (the FSD50K standard) over classes that have at least one positive."""
    y_true = np.asarray(y_true)
    keep = y_true.sum(axis=0) > 0
    if not keep.any():
        raise ValueError("No class has a positive example; mAP is undefined.")
    return float(average_precision_score(y_true[:, keep], np.asarray(y_score)[:, keep], average="macro"))
