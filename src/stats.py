"""Statistical testing for seed-level final metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def summarize_metrics(
    run_df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """Create mean/std summary tables across random seeds."""
    grouped = run_df.groupby(group_cols, dropna=False)
    rows: List[Dict[str, object]] = []
    for key, group in grouped:
        row: Dict[str, object] = {}
        if len(group_cols) == 1:
            row[group_cols[0]] = key
        else:
            for col, val in zip(group_cols, key):
                row[col] = val
        row["n_seeds"] = int(group.shape[0])
        for metric in metric_cols:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _normality_acceptable(differences: np.ndarray) -> bool:
    """Check if paired differences are close enough to normal for t-test usage."""
    if differences.size < 3:
        return False
    if np.allclose(differences, 0.0):
        return False
    try:
        _, p_value = stats.shapiro(differences)
        return bool(p_value >= 0.05)
    except Exception:
        return False


def run_paired_tests(
    run_df: pd.DataFrame,
    metric_cols: List[str],
    family_col: str = "test_family",
    pair_key_col: str = "pair_key",
    method_col: str = "method_label",
    seed_col: str = "seed",
) -> pd.DataFrame:
    """
    Run paired tests using final seed-level metrics.

    For each family/pair_key subset, compare all method pairs:
    - Paired t-test if normality of paired differences is acceptable.
    - Otherwise Wilcoxon signed-rank test.
    """
    results: List[Dict[str, object]] = []
    grouped = run_df.groupby([family_col, pair_key_col], dropna=False)
    for (family, pair_key), subdf in grouped:
        methods = sorted(subdf[method_col].dropna().unique().tolist())
        if len(methods) < 2:
            continue

        for m1, m2 in combinations(methods, 2):
            pair_df = subdf[subdf[method_col].isin([m1, m2])]
            for metric in metric_cols:
                pivot = pair_df.pivot_table(index=seed_col, columns=method_col, values=metric, aggfunc="mean")
                if m1 not in pivot.columns or m2 not in pivot.columns:
                    continue
                pivot = pivot[[m1, m2]].dropna()
                n = int(pivot.shape[0])
                if n < 3:
                    continue

                x = pivot[m1].to_numpy(dtype=float)
                y = pivot[m2].to_numpy(dtype=float)
                diff = x - y

                method = "paired_ttest" if _normality_acceptable(diff) else "wilcoxon_signed_rank"
                statistic = np.nan
                p_value = np.nan

                try:
                    if method == "paired_ttest":
                        statistic, p_value = stats.ttest_rel(x, y, nan_policy="omit")
                    else:
                        if np.allclose(diff, 0.0):
                            statistic, p_value = 0.0, 1.0
                        else:
                            statistic, p_value = stats.wilcoxon(x, y, zero_method="wilcox")
                except Exception:
                    method = "failed"

                results.append(
                    {
                        family_col: family,
                        pair_key_col: pair_key,
                        "metric": metric,
                        "method_a": m1,
                        "method_b": m2,
                        "n_pairs": n,
                        "test_type": method,
                        "statistic": float(statistic) if not np.isnan(statistic) else np.nan,
                        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                        "mean_diff_a_minus_b": float(np.mean(diff)),
                        "std_diff_a_minus_b": float(np.std(diff, ddof=1)) if n > 1 else 0.0,
                    }
                )

    return pd.DataFrame(results)
