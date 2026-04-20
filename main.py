"""Main experiment runner for Huber delta sensitivity study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .data import build_trajectory_data
from .evaluate import compute_metrics
from .losses import LossConfig
from .models import build_mlp
from .plots import (
    plot_delta_sensitivity,
    plot_efficiency,
    plot_summary_bars,
    plot_training_curve,
    plot_trajectory_3d,
    plot_trajectory_projections,
)
from .report import generate_report_assets
from .stats import run_paired_tests, summarize_metrics
from .train import train_model
from .utils import apply_cli_overrides, ensure_output_dirs, get_device, load_yaml, save_json, set_seed, to_serializable

METRIC_COLS = [
    "reconstruction_mse",
    "max_pointwise_deviation",
    "endpoint_error",
    "smoothness_metric",
    "final_training_loss",
    "training_time_sec",
    "avg_epoch_time_sec",
]


@dataclass
class RunSpec:
    """Single training run configuration."""

    run_id: str
    config_id: str
    seed: int
    test_family: str
    pair_key: str
    method_label: str
    loss_name: str
    huber_delta: Optional[float]
    depth: int
    outlier_magnitude: float
    outlier_ratio: float
    random_outlier_positions: bool
    outlier_indices: List[int]
    representative_plot: bool


def _slug(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("=", "-")
        .replace("/", "_")
        .replace(",", "_")
        .replace(".", "p")
        .replace(":", "_")
    )


def _method_label(loss_name: str, huber_delta: Optional[float]) -> str:
    if loss_name == "mse":
        return "mse"
    if huber_delta is None:
        return "huber"
    return f"huber_d{huber_delta:g}"


def _add_specs(
    specs: List[RunSpec],
    seeds: Iterable[int],
    family: str,
    pair_key: str,
    depth: int,
    loss_name: str,
    huber_delta: Optional[float],
    outlier_magnitude: float,
    outlier_ratio: float,
    random_outlier_positions: bool,
    outlier_indices: List[int],
) -> None:
    method = _method_label(loss_name, huber_delta)
    config_id = _slug(
        f"{family}_depth{depth}_{method}_mag{outlier_magnitude:g}_ratio{outlier_ratio:g}_rand{int(random_outlier_positions)}"
    )
    seeds_list = list(seeds)
    representative_seed = seeds_list[0]
    for seed in seeds_list:
        run_id = _slug(f"{config_id}_seed{seed}")
        specs.append(
            RunSpec(
                run_id=run_id,
                config_id=config_id,
                seed=seed,
                test_family=family,
                pair_key=pair_key,
                method_label=method,
                loss_name=loss_name,
                huber_delta=huber_delta,
                depth=depth,
                outlier_magnitude=outlier_magnitude,
                outlier_ratio=outlier_ratio,
                random_outlier_positions=random_outlier_positions,
                outlier_indices=outlier_indices,
                representative_plot=(seed == representative_seed),
            )
        )


def build_run_specs(cfg: Dict[str, Any]) -> List[RunSpec]:
    """Build all experiment configurations for groups A-E."""
    seeds = [int(s) for s in cfg["seeds"]]
    data_cfg = cfg["data"]
    exps = cfg["experiments"]

    specs: List[RunSpec] = []

    if exps["main_comparison"]["enabled"]:
        depth = int(exps["main_comparison"]["depth"])
        huber_delta = float(exps["main_comparison"]["huber_delta"])
        pair_key = "main_depth_baseline"
        for loss_name in ["mse", "huber"]:
            _add_specs(
                specs=specs,
                seeds=seeds,
                family="A_main",
                pair_key=pair_key,
                depth=depth,
                loss_name=loss_name,
                huber_delta=huber_delta if loss_name == "huber" else None,
                outlier_magnitude=float(data_cfg["outlier_magnitude"]),
                outlier_ratio=float(data_cfg["outlier_ratio"]),
                random_outlier_positions=bool(data_cfg["random_outlier_positions"]),
                outlier_indices=[int(x) for x in data_cfg["outlier_indices"]],
            )

    if exps["delta_sensitivity"]["enabled"]:
        depth = int(exps["delta_sensitivity"]["depth"])
        pair_key = "delta_sensitivity"
        for delta in exps["delta_sensitivity"]["deltas"]:
            _add_specs(
                specs=specs,
                seeds=seeds,
                family="B_delta",
                pair_key=pair_key,
                depth=depth,
                loss_name="huber",
                huber_delta=float(delta),
                outlier_magnitude=float(data_cfg["outlier_magnitude"]),
                outlier_ratio=float(data_cfg["outlier_ratio"]),
                random_outlier_positions=bool(data_cfg["random_outlier_positions"]),
                outlier_indices=[int(x) for x in data_cfg["outlier_indices"]],
            )

    if exps["depth_sensitivity"]["enabled"]:
        huber_delta = float(exps["depth_sensitivity"]["huber_delta"])
        losses = [str(v).lower() for v in exps["depth_sensitivity"]["losses"]]
        for depth in exps["depth_sensitivity"]["depths"]:
            pair_key = f"depth={depth}"
            for loss_name in losses:
                _add_specs(
                    specs=specs,
                    seeds=seeds,
                    family="C_depth",
                    pair_key=pair_key,
                    depth=int(depth),
                    loss_name=loss_name,
                    huber_delta=huber_delta if loss_name == "huber" else None,
                    outlier_magnitude=float(data_cfg["outlier_magnitude"]),
                    outlier_ratio=float(data_cfg["outlier_ratio"]),
                    random_outlier_positions=bool(data_cfg["random_outlier_positions"]),
                    outlier_indices=[int(x) for x in data_cfg["outlier_indices"]],
                )

    if exps["outlier_study"]["enabled"]:
        depth = int(exps["outlier_study"]["depth"])
        huber_delta = float(exps["outlier_study"]["huber_delta"])
        losses = [str(v).lower() for v in exps["outlier_study"]["losses"]]
        random_positions = bool(exps["outlier_study"]["random_positions"])

        for mag in exps["outlier_study"]["magnitudes"]:
            pair_key = f"magnitude={mag:g}"
            for loss_name in losses:
                _add_specs(
                    specs=specs,
                    seeds=seeds,
                    family="D_magnitude",
                    pair_key=pair_key,
                    depth=depth,
                    loss_name=loss_name,
                    huber_delta=huber_delta if loss_name == "huber" else None,
                    outlier_magnitude=float(mag),
                    outlier_ratio=float(data_cfg["outlier_ratio"]),
                    random_outlier_positions=random_positions,
                    outlier_indices=[int(x) for x in data_cfg["outlier_indices"]],
                )

        for ratio in exps["outlier_study"]["ratios"]:
            pair_key = f"ratio={ratio:g}"
            for loss_name in losses:
                _add_specs(
                    specs=specs,
                    seeds=seeds,
                    family="D_ratio",
                    pair_key=pair_key,
                    depth=depth,
                    loss_name=loss_name,
                    huber_delta=huber_delta if loss_name == "huber" else None,
                    outlier_magnitude=float(data_cfg["outlier_magnitude"]),
                    outlier_ratio=float(ratio),
                    random_outlier_positions=random_positions,
                    outlier_indices=[int(x) for x in data_cfg["outlier_indices"]],
                )

    return specs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run robust 3D trajectory reconstruction experiments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for training epochs")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed list, e.g. 0,1,2,3,4",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional override for output root directory")
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="If set, save model checkpoints for each run.",
    )
    parser.add_argument("--skip_plots", action="store_true", help="If set, skip figure generation.")
    return parser.parse_args()


def run_experiments(cfg: Dict[str, Any], skip_plots: bool = False) -> Dict[str, Path]:
    """Run all experiments and save outputs."""
    output_root = Path(cfg["output"]["results_dir"])
    paths = ensure_output_dirs(output_root)
    device = get_device()

    save_json(to_serializable(cfg), paths.configs / "resolved_config.json")
    run_specs = build_run_specs(cfg)

    run_records: List[Dict[str, Any]] = []
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    loss_cfg_root = cfg["loss"]
    data_cfg = cfg["data"]
    save_ckpt = bool(cfg["output"]["save_checkpoints"])

    for spec in run_specs:
        set_seed(spec.seed)
        data = build_trajectory_data(
            num_points=int(data_cfg["num_points"]),
            t_max=float(data_cfg["t_max"]),
            noise_std=float(data_cfg["noise_std"]),
            outlier_magnitude=float(spec.outlier_magnitude),
            outlier_indices=spec.outlier_indices,
            outlier_ratio=float(spec.outlier_ratio),
            random_positions=bool(spec.random_outlier_positions),
            seed=spec.seed,
        )

        model = build_mlp(hidden_width=int(model_cfg["hidden_width"]), depth=int(spec.depth))
        losses = LossConfig(
            loss_name=spec.loss_name,
            huber_delta=float(spec.huber_delta) if spec.huber_delta is not None else float(loss_cfg_root["default_huber_delta"]),
            endpoint_weight=float(loss_cfg_root["endpoint_weight"]),
            smoothness_weight=float(loss_cfg_root["smoothness_weight"]),
        )

        ckpt_path = paths.checkpoints / f"{spec.run_id}.pt" if save_ckpt else None
        train_out = train_model(
            model=model,
            t_input=data.t_input,
            clean_target=data.clean,
            noisy_target=data.noisy,
            device=device,
            epochs=int(training_cfg["epochs"]),
            lr=float(training_cfg["lr"]),
            weight_decay=float(training_cfg["weight_decay"]),
            loss_cfg=losses,
            early_stopping_enabled=bool(training_cfg["early_stopping"]["enabled"]),
            early_stopping_patience=int(training_cfg["early_stopping"]["patience"]),
            early_stopping_min_delta=float(training_cfg["early_stopping"]["min_delta"]),
            checkpoint_path=ckpt_path,
        )

        metrics = compute_metrics(
            pred=train_out.prediction,
            clean=data.clean.numpy(),
            final_train_loss=train_out.final_train_loss,
            train_time_sec=train_out.train_time_sec,
            avg_epoch_time_sec=train_out.avg_epoch_time_sec,
        )
        record = {
            "run_id": spec.run_id,
            "config_id": spec.config_id,
            "seed": spec.seed,
            "test_family": spec.test_family,
            "pair_key": spec.pair_key,
            "method_label": spec.method_label,
            "loss_name": spec.loss_name,
            "huber_delta": spec.huber_delta,
            "depth": spec.depth,
            "outlier_magnitude": spec.outlier_magnitude,
            "outlier_ratio": spec.outlier_ratio,
            "random_outlier_positions": spec.random_outlier_positions,
            "n_outliers": int(len(data.outlier_indices)),
            "stopped_early": train_out.stopped_early,
            "best_epoch": train_out.best_epoch,
            "epochs_ran": int(len(train_out.history)),
            "device": str(device),
            **metrics,
        }
        run_records.append(record)

        run_cfg_dict = to_serializable(
            {
                "run": record,
                "outlier_indices_actual": data.outlier_indices.tolist(),
            }
        )
        save_json(run_cfg_dict, paths.configs / f"{spec.run_id}.json")

        if not skip_plots and spec.representative_plot:
            base_name = spec.run_id
            plot_trajectory_3d(
                clean=data.clean.numpy(),
                noisy=data.noisy.numpy(),
                pred=train_out.prediction,
                outlier_indices=data.outlier_indices,
                save_path=paths.figures / f"{base_name}_trajectory3d.png",
                title=f"{spec.test_family} | {spec.method_label} | depth={spec.depth} | seed={spec.seed}",
            )
            plot_trajectory_projections(
                clean=data.clean.numpy(),
                noisy=data.noisy.numpy(),
                pred=train_out.prediction,
                outlier_indices=data.outlier_indices,
                save_path=paths.figures / f"{base_name}_projections.png",
                title=f"{spec.test_family} | {spec.method_label} projections",
            )
            plot_training_curve(
                loss_history=train_out.history,
                save_path=paths.figures / f"{base_name}_loss_curve.png",
                title=f"Training Loss | {spec.test_family} | {spec.method_label}",
            )

    run_df = pd.DataFrame(run_records)
    run_metrics_path = paths.csv / "run_metrics.csv"
    run_df.to_csv(run_metrics_path, index=False)

    group_cols = [
        "config_id",
        "test_family",
        "pair_key",
        "method_label",
        "loss_name",
        "huber_delta",
        "depth",
        "outlier_magnitude",
        "outlier_ratio",
        "random_outlier_positions",
    ]
    summary_df = summarize_metrics(run_df=run_df, group_cols=group_cols, metric_cols=METRIC_COLS)
    summary_path = paths.csv / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    stat_metrics = ["reconstruction_mse", "max_pointwise_deviation", "endpoint_error", "training_time_sec"]
    stat_df = run_paired_tests(run_df=run_df, metric_cols=stat_metrics)
    stat_path = paths.csv / "statistical_tests.csv"
    stat_df.to_csv(stat_path, index=False)

    if not skip_plots:
        plot_delta_sensitivity(summary_df, paths.figures / "delta_sensitivity.png")
        plot_efficiency(summary_df, paths.figures / "efficiency_by_configuration.png")
        plot_summary_bars(
            summary_df,
            metric="reconstruction_mse",
            save_path=paths.figures / "summary_reconstruction_mse.png",
            title="Reconstruction MSE Across Configurations (mean ± std)",
        )
        plot_summary_bars(
            summary_df,
            metric="max_pointwise_deviation",
            save_path=paths.figures / "summary_max_deviation.png",
            title="Max Pointwise Deviation Across Configurations (mean ± std)",
        )

    compact_path, fig_index_path = generate_report_assets(
        summary_df=summary_df,
        figures_dir=paths.figures,
        out_dir=paths.csv,
        metrics=METRIC_COLS,
    )

    return {
        "run_metrics_csv": run_metrics_path,
        "summary_csv": summary_path,
        "stat_tests_csv": stat_path,
        "report_compact_csv": compact_path,
        "figure_index_csv": fig_index_path,
        "figures_dir": paths.figures,
    }


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_yaml(args.config)
    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else None
    cfg = apply_cli_overrides(cfg, epochs=args.epochs, seeds=seeds)
    if args.output_dir is not None:
        cfg["output"]["results_dir"] = str(args.output_dir)
    if args.save_checkpoints:
        cfg["output"]["save_checkpoints"] = True

    outputs = run_experiments(cfg, skip_plots=args.skip_plots)
    print("Experiment run complete.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
