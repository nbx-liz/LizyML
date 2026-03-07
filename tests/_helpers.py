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


def make_binary_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    """Create a binary classification DataFrame with threshold at feat_a > 5."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
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
    calibration: str | None = None,
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
        calibration: Calibration method (e.g. ``"platt"``). ``None`` disables.
        calibration_params: Extra params for calibration (e.g. ``{"degree": 3}``).
        tuning_n_trials: When set, adds an ``optuna`` tuning section.
        seed: Training random seed.
        **model_overrides: Extra keys merged into ``model.params``.
    """
    params: dict[str, Any] = {"n_estimators": n_estimators, **model_overrides}
    cfg: dict[str, Any] = {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": n_splits, "random_state": 42},
        "model": {"name": "lgbm", "params": params},
        "training": {"seed": seed},
    }
    if calibration is not None:
        cal: dict[str, Any] = {"method": calibration}
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
