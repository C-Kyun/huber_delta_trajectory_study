"""Synthetic 3D trajectory data generation with configurable outlier corruption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class TrajectoryData:
    """Container for trajectory tensors and metadata."""

    t_input: torch.Tensor
    clean: torch.Tensor
    noisy: torch.Tensor
    outlier_indices: np.ndarray
    t_raw: np.ndarray


def generate_clean_trajectory(num_points: int, t_max: float) -> Dict[str, np.ndarray]:
    """
    Generate clean 3D helix-like trajectory.

    x = cos(t), y = sin(t), z = 0.2 * t
    """
    t = np.linspace(0.0, t_max, num_points, dtype=np.float64)
    x = np.cos(t)
    y = np.sin(t)
    z = 0.2 * t
    clean = np.stack([x, y, z], axis=1)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-12)
    return {"t_raw": t, "t_norm": t_norm[:, None], "clean": clean}


def _sample_outlier_indices(
    num_points: int,
    outlier_indices: List[int],
    outlier_ratio: float,
    random_positions: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose outlier indices either from a fixed list or by random ratio."""
    if random_positions:
        k = max(1, int(round(outlier_ratio * num_points)))
        k = min(k, num_points)
        idx = rng.choice(num_points, size=k, replace=False)
    else:
        idx = np.array([i for i in outlier_indices if 0 <= i < num_points], dtype=np.int64)
        if idx.size == 0:
            idx = np.array([], dtype=np.int64)
    return np.sort(idx)


def inject_outliers(
    clean: np.ndarray,
    noise_std: float,
    outlier_magnitude: float,
    outlier_indices: List[int],
    outlier_ratio: float,
    random_positions: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise and sparse large outlier perturbations."""
    noisy = clean + rng.normal(loc=0.0, scale=noise_std, size=clean.shape)
    idx = _sample_outlier_indices(
        num_points=clean.shape[0],
        outlier_indices=outlier_indices,
        outlier_ratio=outlier_ratio,
        random_positions=random_positions,
        rng=rng,
    )
    if idx.size > 0:
        outlier_noise = rng.normal(loc=0.0, scale=outlier_magnitude, size=(idx.size, clean.shape[1]))
        noisy[idx] += outlier_noise
    return noisy, idx


def build_trajectory_data(
    num_points: int,
    t_max: float,
    noise_std: float,
    outlier_magnitude: float,
    outlier_indices: Optional[List[int]],
    outlier_ratio: float,
    random_positions: bool,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> TrajectoryData:
    """Create full training data (time input, clean target, noisy observation)."""
    outlier_indices = outlier_indices or []
    rng = np.random.default_rng(seed)
    generated = generate_clean_trajectory(num_points=num_points, t_max=t_max)
    noisy, idx = inject_outliers(
        clean=generated["clean"],
        noise_std=noise_std,
        outlier_magnitude=outlier_magnitude,
        outlier_indices=outlier_indices,
        outlier_ratio=outlier_ratio,
        random_positions=random_positions,
        rng=rng,
    )
    return TrajectoryData(
        t_input=torch.tensor(generated["t_norm"], dtype=dtype),
        clean=torch.tensor(generated["clean"], dtype=dtype),
        noisy=torch.tensor(noisy, dtype=dtype),
        outlier_indices=idx,
        t_raw=generated["t_raw"],
    )
