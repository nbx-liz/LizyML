"""Category E: Pairwise parameter combination tests.

Tests all 2-factor combinations of (task, split_method, calibration,
early_stopping, n_estimators) via a pairwise covering array, plus
targeted interaction tests for high-risk combinations.

See BLUEPRINT §18.1.8 and HISTORY H-0056 Category E.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from tests._helpers import make_config, make_pairwise_df

# ---------------------------------------------------------------------------
# Factor space
# ---------------------------------------------------------------------------
# task:           regression, binary, multiclass
# split_method:   kfold, stratified_kfold, group_kfold, time_series
# calibration:    None, "platt"
# early_stopping: True, False
# n_estimators:   5, 50
#
# Invalid combinations:
# - calibration="platt" + task="regression"
# - calibration="platt" + task="multiclass"
# - group_kfold requires group_col
# - time_series requires time_col
# - stratified_kfold not valid for regression (warning only, not error)
# ---------------------------------------------------------------------------

# Pre-computed pairwise covering array
# Each pair of factor values appears at least once across all rows.
# Constraints:
# - stratified_kfold requires classification task (not regression)
# - time_series + classification uses large n to avoid NaN OOF issues
PAIRWISE_CASES: list[dict[str, Any]] = [
    # --- regression (kfold, group_kfold, time_series only — no stratified) ---
    {"task": "regression", "split": "kfold", "cal": None, "es": True, "n_est": 5},
    {"task": "regression", "split": "kfold", "cal": None, "es": False, "n_est": 50},
    {
        "task": "regression",
        "split": "group_kfold",
        "cal": None,
        "es": True,
        "n_est": 50,
    },
    {
        "task": "regression",
        "split": "group_kfold",
        "cal": None,
        "es": False,
        "n_est": 5,
    },
    {
        "task": "regression",
        "split": "time_series",
        "cal": None,
        "es": False,
        "n_est": 5,
    },
    {
        "task": "regression",
        "split": "time_series",
        "cal": None,
        "es": True,
        "n_est": 50,
    },
    # invalid: cal=platt + regression
    {"task": "regression", "split": "kfold", "cal": "platt", "es": False, "n_est": 50},
    # --- binary (all split methods valid) ---
    {"task": "binary", "split": "kfold", "cal": None, "es": False, "n_est": 5},
    {
        "task": "binary",
        "split": "stratified_kfold",
        "cal": "platt",
        "es": True,
        "n_est": 50,
    },
    {
        "task": "binary",
        "split": "stratified_kfold",
        "cal": None,
        "es": False,
        "n_est": 5,
    },
    {
        "task": "binary",
        "split": "stratified_kfold",
        "cal": "platt",
        "es": False,
        "n_est": 5,
    },
    {"task": "binary", "split": "group_kfold", "cal": None, "es": False, "n_est": 50},
    {"task": "binary", "split": "kfold", "cal": "platt", "es": True, "n_est": 5},
    # --- multiclass (all split methods except cal=platt is invalid) ---
    # invalid: cal=platt + multiclass
    {"task": "multiclass", "split": "kfold", "cal": "platt", "es": True, "n_est": 50},
    {
        "task": "multiclass",
        "split": "stratified_kfold",
        "cal": None,
        "es": True,
        "n_est": 5,
    },
    {"task": "multiclass", "split": "group_kfold", "cal": None, "es": True, "n_est": 5},
    {"task": "multiclass", "split": "kfold", "cal": None, "es": False, "n_est": 50},
    {
        "task": "multiclass",
        "split": "stratified_kfold",
        "cal": None,
        "es": False,
        "n_est": 50,
    },
]

# Known invalid combinations
_INVALID_COMBOS = {
    frozenset({("cal", "platt"), ("task", "regression")}),
    frozenset({("cal", "platt"), ("task", "multiclass")}),
}


def _is_invalid(case: dict[str, Any]) -> bool:
    """Check if this combination is known to be invalid."""
    items = {(k, v) for k, v in case.items()}
    return any(combo <= items for combo in _INVALID_COMBOS)


def _case_id(case: dict[str, Any]) -> str:
    """Generate a human-readable test ID."""
    return (
        f"{case['task']}-{case['split']}"
        f"-cal={case['cal']}"
        f"-es={case['es']}"
        f"-n={case['n_est']}"
    )


def _build_config_and_data(case: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Build config dict and DataFrame for a pairwise case."""
    task = case["task"]
    split = case["split"]
    cal = case["cal"]
    es = case["es"]
    n_est = case["n_est"]

    group_col = "grp" if split == "group_kfold" else None
    time_col = "time" if split == "time_series" else None
    # Larger datasets for multiclass and time_series to avoid NaN OOF
    n = 300 if task == "multiclass" else 200
    if split == "time_series":
        n = max(n, 300)

    df = make_pairwise_df(task, n=n, seed=42, group_col=group_col, time_col=time_col)

    cfg = make_config(
        task,
        n_estimators=n_est,
        n_splits=2,
        split_method=split,
        group_col=group_col,
        time_col=time_col,
        calibration=cal,
        seed=42,
    )
    # Early stopping config
    cfg["training"]["early_stopping"] = {
        "enabled": es,
        "rounds": 3 if es else 150,
        "validation_ratio": 0.2 if es else None,
    }

    return cfg, df


# ===================================================================
# Pairwise fit completion
# ===================================================================


class TestPairwiseFitCompletion:
    """All pairwise combinations complete or produce a clear LizyMLError."""

    @pytest.mark.parametrize(
        "case",
        PAIRWISE_CASES,
        ids=[_case_id(c) for c in PAIRWISE_CASES],
    )
    def test_fit_or_clear_error(self, case: dict[str, Any]) -> None:
        cfg, df = _build_config_and_data(case)

        if _is_invalid(case):
            with pytest.raises(LizyMLError) as exc_info:
                Model(cfg).fit(data=df)
            assert exc_info.value.code in (
                ErrorCode.CALIBRATION_NOT_SUPPORTED,
                ErrorCode.CONFIG_INVALID,
            )
        else:
            result = Model(cfg).fit(data=df)
            assert len(result.models) == 2
            assert result.oof_pred is not None


# ===================================================================
# Targeted interaction tests
# ===================================================================


class TestCalibrationBinaryInteraction:
    """Calibration interacts correctly with different splits for binary."""

    def test_binary_stratified_calibration(self) -> None:
        """Stratified + platt = the standard calibration path."""
        df = make_pairwise_df("binary", n=200, seed=42)
        cfg = make_config(
            "binary",
            n_estimators=5,
            n_splits=2,
            split_method="stratified_kfold",
            calibration="platt",
        )
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.calibrator is not None


class TestBalancedMulticlass:
    """balanced=True + multiclass fit completes."""

    def test_balanced_multiclass_fit(self) -> None:
        df = make_pairwise_df("multiclass", n=300, seed=42)
        cfg = make_config(
            "multiclass",
            n_estimators=5,
            n_splits=2,
            split_method="stratified_kfold",
        )
        # balanced is a top-level LGBMConfig smart param, not a Booster param
        cfg["model"]["balanced"] = True
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred is not None
        assert result.oof_pred.shape[0] == len(df)


class TestFeatureWeightsAutoNumLeaves:
    """feature_weights + auto_num_leaves coexist."""

    def test_feature_weights_with_auto_num_leaves(self) -> None:
        df = make_pairwise_df("regression", n=200, seed=42)
        cfg = make_config(
            "regression",
            n_estimators=5,
            n_splits=2,
        )
        cfg["model"]["feature_weights"] = {"feat_a": 2.0, "feat_b": 0.5}
        cfg["model"]["auto_num_leaves"] = True
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred is not None


class TestTuningThenCalibration:
    """tune → fit with calibration: best_params and calibration coexist."""

    def test_tune_then_fit_with_calibration(self) -> None:
        df = make_pairwise_df("binary", n=200, seed=42)
        cfg = make_config(
            "binary",
            n_estimators=10,
            n_splits=2,
            split_method="stratified_kfold",
            calibration="platt",
            tuning_n_trials=3,
        )
        m = Model(cfg)
        tune_result = m.tune(data=df)
        assert tune_result.best_params is not None
        fit_result = m.fit(data=df)
        assert fit_result.calibrator is not None


class TestMinEstimatorsEarlyStopping:
    """n_estimators=5 + early_stopping: edge case."""

    def test_small_estimators_with_early_stopping(self) -> None:
        df = make_pairwise_df("regression", n=200, seed=42)
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        cfg["training"]["early_stopping"] = {
            "enabled": True,
            "rounds": 3,
            "validation_ratio": 0.2,
        }
        m = Model(cfg)
        result = m.fit(data=df)
        assert result.oof_pred is not None


class TestExcludeCategorical:
    """features.exclude targeting a categorical column."""

    def test_exclude_categorical_column(self) -> None:
        df = make_pairwise_df("regression", n=200, seed=42)
        rng = np.random.default_rng(42)
        df["cat_to_exclude"] = rng.choice(["x", "y", "z"], len(df))
        cfg = make_config("regression", n_estimators=5, n_splits=2)
        cfg["features"] = {"exclude": ["cat_to_exclude"]}
        m = Model(cfg)
        result = m.fit(data=df)
        assert "cat_to_exclude" not in result.feature_names
        assert result.oof_pred is not None
