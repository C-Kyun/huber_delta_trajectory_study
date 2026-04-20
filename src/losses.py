"""Loss functions for robust trajectory reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    """Configuration for main and auxiliary losses."""

    loss_name: str
    huber_delta: float
    endpoint_weight: float
    smoothness_weight: float


def regression_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str, huber_delta: float) -> torch.Tensor:
    """Compute MSE or Huber regression loss."""
    normalized = loss_name.lower()
    if normalized == "mse":
        return F.mse_loss(pred, target)
    if normalized == "huber":
        return F.huber_loss(pred, target, delta=huber_delta, reduction="mean")
    raise ValueError(f"Unsupported loss: {loss_name}")


def endpoint_loss(pred: torch.Tensor, clean_target: torch.Tensor) -> torch.Tensor:
    """Force first and last predicted points to match clean trajectory endpoints."""
    pred_endpoints = torch.stack([pred[0], pred[-1]], dim=0)
    clean_endpoints = torch.stack([clean_target[0], clean_target[-1]], dim=0)
    return F.mse_loss(pred_endpoints, clean_endpoints)


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalize second-order differences to encourage smooth trajectories."""
    if pred.shape[0] < 3:
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    second_diff = pred[:-2] - 2.0 * pred[1:-1] + pred[2:]
    return torch.mean(torch.sum(second_diff**2, dim=1))


def total_loss(
    pred: torch.Tensor,
    noisy_target: torch.Tensor,
    clean_target: torch.Tensor,
    cfg: LossConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine main regression and auxiliary constraints."""
    reg = regression_loss(pred, noisy_target, cfg.loss_name, cfg.huber_delta)
    endp = endpoint_loss(pred, clean_target)
    smooth = smoothness_loss(pred)
    total = reg + cfg.endpoint_weight * endp + cfg.smoothness_weight * smooth
    return total, {
        "regression_loss": float(reg.detach().item()),
        "endpoint_loss": float(endp.detach().item()),
        "smoothness_loss": float(smooth.detach().item()),
        "total_loss": float(total.detach().item()),
    }
