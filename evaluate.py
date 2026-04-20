"""Evaluation metrics for reconstructed trajectories."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_smoothness_metric(pred: np.ndarray) -> float:
    """Compute mean squared second-order differences."""
    if pred.shape[0] < 3:
        return 0.0
    second_diff = pred[:-2] - 2.0 * pred[1:-1] + pred[2:]
    return float(np.mean(np.sum(second_diff**2, axis=1)))


def compute_metrics(
    pred: np.ndarray,
    clean: np.ndarray,
    final_train_loss: float,
    train_time_sec: float,
    avg_epoch_time_sec: float,
) -> Dict[str, float]:
    """Compute required final run-level metrics."""
    sq_err = (pred - clean) ** 2
    reconstruction_mse = float(np.mean(sq_err))
    pointwise_l2 = np.linalg.norm(pred - clean, axis=1)
    max_pointwise_deviation = float(np.max(pointwise_l2))
    endpoint_error = float(0.5 * (pointwise_l2[0] + pointwise_l2[-1]))
    smoothness = compute_smoothness_metric(pred)
    return {
        "reconstruction_mse": reconstruction_mse,
        "max_pointwise_deviation": max_pointwise_deviation,
        "endpoint_error": endpoint_error,
        "smoothness_metric": smoothness,
        "final_training_loss": float(final_train_loss),
        "training_time_sec": float(train_time_sec),
        "avg_epoch_time_sec": float(avg_epoch_time_sec),
    }
