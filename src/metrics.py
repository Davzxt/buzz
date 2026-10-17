"""Metrics for class-incremental evaluation."""
import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == np.asarray(y_pred)))


def class_incremental_metrics(acc_matrix, overall_history):
    """Summarise a run.

    acc_matrix[t][s]: accuracy on task s's test samples after training task t
        (predictions are made over all classes seen so far). NaN where s > t.
    overall_history[t]: accuracy on all test samples of the classes seen up to task t.

    forgetting: mean over earlier tasks s of (best accuracy on s before the last task)
        minus (accuracy on s after the last task). Larger means more forgetting.
    """
    A = np.asarray(acc_matrix, dtype=float)
    T = A.shape[0]
    final_acc = float(overall_history[-1])
    avg_incremental_acc = float(np.mean(overall_history))
    if T < 2:
        forgetting = 0.0
    else:
        drops = [np.max(A[s : T - 1, s]) - A[T - 1, s] for s in range(T - 1)]
        forgetting = float(np.mean(drops))
    return {
        "final_acc": final_acc,
        "avg_incremental_acc": avg_incremental_acc,
        "forgetting": forgetting,
    }
