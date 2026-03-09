"""Shared test helpers — synthetic data generators and config builders.

All test modules should import from here instead of defining local copies.
Functions (not fixtures) so they can be called at module level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_regression_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Create a regression DataFrame with 2 features and a linear target."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"] + rng.normal(0, 0.1, n)
    return df


def make_binary_df(
    n: int = 200,
    seed: int = 1,
    *,
    group_col: str | None = None,
    n_groups: int = 10,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Create a binary classification DataFrame with threshold at feat_a > 5.

    Args:
        group_col: If set, adds a group column with *n_groups* distinct values.
        time_col: If set, adds a monotonically increasing time column.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    if group_col is not None:
        df[group_col] = rng.integers(0, n_groups, n)
    if time_col is not None:
        df[time_col] = np.arange(n)
    return df


def make_multiclass_df(n: int = 300, seed: int = 2) -> pd.DataFrame:
    """Create a 3-class classification DataFrame via binning feat_a."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = pd.cut(df["feat_a"], bins=3, labels=[0, 1, 2]).astype(int)
    return df


def make_config(
    task: str,
    *,
    n_estimators: int = 10,
    n_splits: int = 3,
    split_method: str = "kfold",
    group_col: str | None = None,
    time_col: str | None = None,
    split_overrides: dict[str, Any] | None = None,
    calibration: str | None = None,
    calibration_n_splits: int = 5,
    calibration_params: dict[str, Any] | None = None,
    tuning_n_trials: int | None = None,
    seed: int = 0,
    **model_overrides: Any,
) -> dict[str, Any]:
    """Build a minimal LizyML config dict.

    Args:
        task: ``"regression"``, ``"binary"``, or ``"multiclass"``.
        n_estimators: Number of boosting rounds.
        n_splits: Number of CV folds.
        split_method: Split method name (e.g. ``"kfold"``, ``"group_kfold"``).
        group_col: Group column name for group-based splits.
        time_col: Time column name for time-based splits.
        split_overrides: Extra keys merged into the split config.
        calibration: Calibration method (e.g. ``"platt"``). ``None`` disables.
        calibration_n_splits: Number of folds for calibration cross-fit.
        calibration_params: Extra params for calibration (e.g. ``{"degree": 3}``).
        tuning_n_trials: When set, adds an ``optuna`` tuning section.
        seed: Training random seed.
        **model_overrides: Extra keys merged into ``model.params``.
    """
    params: dict[str, Any] = {"n_estimators": n_estimators, **model_overrides}
    split_cfg: dict[str, Any] = {
        "method": split_method,
        "n_splits": n_splits,
    }
    # Add random_state for methods that support it
    if split_method in ("kfold", "stratified_kfold"):
        split_cfg["random_state"] = 42
    if split_overrides:
        split_cfg.update(split_overrides)

    data_cfg: dict[str, Any] = {"target": "target"}
    if group_col is not None:
        data_cfg["group_col"] = group_col
    if time_col is not None:
        data_cfg["time_col"] = time_col

    cfg: dict[str, Any] = {
        "config_version": 1,
        "task": task,
        "data": data_cfg,
        "split": split_cfg,
        "model": {"name": "lgbm", "params": params},
        "training": {"seed": seed},
    }
    if calibration is not None:
        cal: dict[str, Any] = {
            "method": calibration,
            "n_splits": calibration_n_splits,
        }
        if calibration_params is not None:
            cal["params"] = calibration_params
        cfg["calibration"] = cal
    if tuning_n_trials is not None:
        cfg["tuning"] = {
            "optuna": {
                "params": {"n_trials": tuning_n_trials, "direction": "minimize"},
            }
        }
    return cfg
