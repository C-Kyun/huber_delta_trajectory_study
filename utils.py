"""Utility helpers for configuration, reproducibility, and file I/O."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


@dataclass
class Paths:
    """Container for canonical output folders."""

    root: Path
    csv: Path
    figures: Path
    configs: Path
    checkpoints: Path


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(root: Path) -> Paths:
    """Create output directories if they do not exist."""
    csv = root / "csv"
    figures = root / "figures"
    configs = root / "configs"
    checkpoints = root / "checkpoints"
    for directory in [root, csv, figures, configs, checkpoints]:
        directory.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, csv=csv, figures=figures, configs=configs, checkpoints=checkpoints)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as JSON with sorted keys for reproducibility."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def to_serializable(obj: Any) -> Any:
    """Convert non-JSON objects to JSON-serializable values."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def apply_cli_overrides(config: Dict[str, Any], epochs: int | None, seeds: List[int] | None) -> Dict[str, Any]:
    """Apply optional command-line overrides to loaded config."""
    cfg = dict(config)
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    if seeds is not None and len(seeds) > 0:
        cfg["seeds"] = [int(s) for s in seeds]
    return cfg
