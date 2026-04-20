"""Matplotlib plotting utilities for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prep_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_trajectory_3d(
    clean: np.ndarray,
    noisy: np.ndarray,
    pred: np.ndarray,
    outlier_indices: Sequence[int],
    save_path: Path,
    title: str,
) -> None:
    """Save 3D trajectory comparison figure."""
    _prep_path(save_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(clean[:, 0], clean[:, 1], clean[:, 2], label="Clean", linewidth=2.0, color="#1f77b4")
    ax.scatter(noisy[:, 0], noisy[:, 1], noisy[:, 2], label="Noisy obs", s=12, alpha=0.6, color="#7f7f7f")
    if len(outlier_indices) > 0:
        idx = np.asarray(outlier_indices, dtype=int)
        ax.scatter(
            noisy[idx, 0],
            noisy[idx, 1],
            noisy[idx, 2],
            label="Outliers",
            s=35,
            color="#d62728",
            marker="x",
        )
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], label="Prediction", linewidth=2.0, color="#2ca02c")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_trajectory_projections(
    clean: np.ndarray,
    noisy: np.ndarray,
    pred: np.ndarray,
    outlier_indices: Sequence[int],
    save_path: Path,
    title: str,
) -> None:
    """Save XY, YZ, XZ projection plots."""
    _prep_path(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    planes = [("XY", 0, 1, "X", "Y"), ("YZ", 1, 2, "Y", "Z"), ("XZ", 0, 2, "X", "Z")]
    for ax, (name, i, j, xlab, ylab) in zip(axes, planes):
        ax.plot(clean[:, i], clean[:, j], label="Clean", linewidth=2.0, color="#1f77b4")
        ax.scatter(noisy[:, i], noisy[:, j], s=10, alpha=0.6, color="#7f7f7f", label="Noisy")
        if len(outlier_indices) > 0:
            idx = np.asarray(outlier_indices, dtype=int)
            ax.scatter(noisy[idx, i], noisy[idx, j], s=28, color="#d62728", marker="x", label="Outliers")
        ax.plot(pred[:, i], pred[:, j], label="Prediction", linewidth=2.0, color="#2ca02c")
        ax.set_title(name)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(loss_history: Sequence[float], save_path: Path, title: str) -> None:
    """Save training loss curve."""
    _prep_path(save_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(loss_history) + 1), loss_history, color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Training Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_delta_sensitivity(summary_df: pd.DataFrame, save_path: Path) -> None:
    """Plot delta vs reconstruction MSE and max deviation for Huber sweeps."""
    _prep_path(save_path)
    sub = summary_df[summary_df["test_family"] == "B_delta"].copy()
    if sub.empty:
        return
    sub["delta"] = pd.to_numeric(sub["huber_delta"], errors="coerce")
    sub = sub.dropna(subset=["delta"]).sort_values("delta")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(
        sub["delta"],
        sub["reconstruction_mse_mean"],
        yerr=sub["reconstruction_mse_std"],
        marker="o",
        linewidth=2.0,
        capsize=4,
        color="#1f77b4",
    )
    axes[0].set_xlabel("Huber delta")
    axes[0].set_ylabel("Reconstruction MSE")
    axes[0].set_title("Delta vs Reconstruction MSE")
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(
        sub["delta"],
        sub["max_pointwise_deviation_mean"],
        yerr=sub["max_pointwise_deviation_std"],
        marker="o",
        linewidth=2.0,
        capsize=4,
        color="#d62728",
    )
    axes[1].set_xlabel("Huber delta")
    axes[1].set_ylabel("Max Pointwise Deviation")
    axes[1].set_title("Delta vs Max Deviation")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_efficiency(summary_df: pd.DataFrame, save_path: Path) -> None:
    """Plot training time by model/loss configuration."""
    _prep_path(save_path)
    sub = summary_df.sort_values("training_time_sec_mean").copy()
    if sub.empty:
        return
    labels = sub["config_id"].astype(str).tolist()
    x = np.arange(len(labels))
    y = sub["training_time_sec_mean"].to_numpy()
    err = sub["training_time_sec_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(labels)), 5))
    ax.bar(x, y, yerr=err, capsize=3, color="#17becf")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Training Time (sec)")
    ax.set_title("Computational Efficiency by Configuration")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(
    summary_df: pd.DataFrame,
    metric: str,
    save_path: Path,
    title: str,
    filter_families: Iterable[str] | None = None,
) -> None:
    """Plot bar chart with mean +/- std across seeds for a chosen metric."""
    _prep_path(save_path)
    sub = summary_df.copy()
    if filter_families is not None:
        sub = sub[sub["test_family"].isin(list(filter_families))]
    if sub.empty:
        return

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    sub = sub.sort_values(mean_col)
    labels = sub["config_id"].astype(str).tolist()
    x = np.arange(len(labels))
    y = sub[mean_col].to_numpy()
    err = sub[std_col].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 5))
    ax.bar(x, y, yerr=err, capsize=3, color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
