"""Generate compact report-ready tables and figure index files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _fmt_mean_std(mean: float, std: float, precision: int = 6) -> str:
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def build_compact_summary(summary_df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Convert mean/std columns into compact report-ready strings."""
    base_cols = [
        "config_id",
        "test_family",
        "method_label",
        "loss_name",
        "huber_delta",
        "depth",
        "outlier_magnitude",
        "outlier_ratio",
        "n_seeds",
    ]
    cols = [c for c in base_cols if c in summary_df.columns]
    out = summary_df[cols].copy()
    for metric in metrics:
        m_col = f"{metric}_mean"
        s_col = f"{metric}_std"
        if m_col in summary_df.columns and s_col in summary_df.columns:
            out[metric] = [
                _fmt_mean_std(float(m), float(s)) for m, s in zip(summary_df[m_col], summary_df[s_col])
            ]
    return out


def build_figure_index(figures_dir: Path) -> pd.DataFrame:
    """Index generated PNG files for easy report assembly."""
    files = sorted(figures_dir.glob("*.png"))
    rows: List[dict] = []
    for i, path in enumerate(files, start=1):
        rows.append(
            {
                "figure_id": f"F{i:03d}",
                "filename": path.name,
                "path": str(path.resolve()),
                "size_kb": round(path.stat().st_size / 1024.0, 2),
            }
        )
    return pd.DataFrame(rows)


def generate_report_assets(
    summary_df: pd.DataFrame,
    figures_dir: Path,
    out_dir: Path,
    metrics: Iterable[str],
) -> tuple[Path, Path]:
    """Save compact summary and figure index CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    compact = build_compact_summary(summary_df, metrics=metrics)
    figure_index = build_figure_index(figures_dir=figures_dir)
    compact_path = out_dir / "report_summary_compact.csv"
    fig_index_path = out_dir / "figure_index.csv"
    compact.to_csv(compact_path, index=False)
    figure_index.to_csv(fig_index_path, index=False)
    return compact_path, fig_index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready summary CSV and figure index.")
    parser.add_argument("--summary_csv", type=Path, required=True, help="Path to summary_metrics.csv")
    parser.add_argument("--figures_dir", type=Path, required=True, help="Path to figures directory")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for report CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    metrics = [
        "reconstruction_mse",
        "max_pointwise_deviation",
        "endpoint_error",
        "smoothness_metric",
        "final_training_loss",
        "training_time_sec",
        "avg_epoch_time_sec",
    ]
    generate_report_assets(summary_df, args.figures_dir, args.out_dir, metrics)


if __name__ == "__main__":
    main()
