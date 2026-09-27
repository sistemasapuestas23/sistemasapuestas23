import numpy as np
from sklearn.metrics import log_loss, accuracy_score


def multiclass_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(log_loss(y_true, y_prob, labels=[0, 1, 2]))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))
