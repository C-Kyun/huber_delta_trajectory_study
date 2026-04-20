"""Training utilities for trajectory reconstruction models."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import Adam

from .losses import LossConfig, total_loss


@dataclass
class TrainOutput:
    """Container for training artifacts."""

    history: List[float]
    prediction: np.ndarray
    final_train_loss: float
    train_time_sec: float
    avg_epoch_time_sec: float
    stopped_early: bool
    best_epoch: Optional[int]


def train_model(
    model: torch.nn.Module,
    t_input: torch.Tensor,
    clean_target: torch.Tensor,
    noisy_target: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    loss_cfg: LossConfig,
    early_stopping_enabled: bool = False,
    early_stopping_patience: int = 500,
    early_stopping_min_delta: float = 1e-8,
    checkpoint_path: Path | None = None,
) -> TrainOutput:
    """Train the model with optional early stopping."""
    model = model.to(device)
    t_input = t_input.to(device)
    clean_target = clean_target.to(device)
    noisy_target = noisy_target.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[float] = []
    best_loss = float("inf")
    best_epoch: Optional[int] = None
    wait = 0
    stopped_early = False
    best_state: Optional[Dict[str, torch.Tensor]] = None

    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(t_input)
        loss, _ = total_loss(pred, noisy_target, clean_target, loss_cfg)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().item())
        history.append(current_loss)

        if early_stopping_enabled:
            if current_loss < best_loss - early_stopping_min_delta:
                best_loss = current_loss
                best_epoch = epoch + 1
                wait = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= early_stopping_patience:
                    stopped_early = True
                    break

    train_time = time.perf_counter() - start_time
    epochs_ran = len(history)
    avg_epoch_time = train_time / max(epochs_ran, 1)

    if early_stopping_enabled and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(t_input).detach().cpu().numpy()

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    final_train_loss = history[-1] if history else float("nan")
    return TrainOutput(
        history=history,
        prediction=pred,
        final_train_loss=final_train_loss,
        train_time_sec=train_time,
        avg_epoch_time_sec=avg_epoch_time,
        stopped_early=stopped_early,
        best_epoch=best_epoch,
    )
