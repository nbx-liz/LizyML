"""Regression tests for issue #95 — Model.save() / Model.load()
round-trip with non-holdout ``inner_valid``.

Pre-H-0069 the round-trip allowance in ``EarlyStoppingConfig`` only
accepted ``HoldoutInnerValidConfig``, so saving a model fit with
``group_holdout`` or ``time_holdout`` produced an artifact that could
not be loaded back (``CONFIG_INVALID``).  H-0069 makes
``validation_ratio`` a computed field derived from
``inner_valid.ratio``, eliminating the dual-write inconsistency at
its root.

These tests exercise the full save → load cycle for every
``InnerValidConfig`` discriminant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from tests._helpers import make_binary_df


def _build_config(
    *,
    split_method: str,
    inner_valid: dict[str, Any] | None,
    extra_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_cfg: dict[str, Any] = {"target": "target"}
    if extra_data:
        data_cfg.update(extra_data)
    split_cfg: dict[str, Any] = {"method": split_method, "n_splits": 3}
    if split_method in ("kfold", "stratified_kfold"):
        split_cfg["random_state"] = 42
    if split_method == "group_kfold":
        # GroupKFold has no random_state.
        pass

    early_stopping: dict[str, Any] = {"enabled": True, "rounds": 30}
    if inner_valid is not None:
        early_stopping["inner_valid"] = inner_valid

    return {
        "config_version": 1,
        "task": "binary",
        "data": data_cfg,
        "split": split_cfg,
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0, "early_stopping": early_stopping},
        "evaluation": {"metrics": ["auc"]},
    }


def _make_grouped_binary_df(n: int = 90, n_groups: int = 6) -> pd.DataFrame:
    df = make_binary_df(n=n)
    rng = np.random.default_rng(42)
    df["g"] = rng.integers(0, n_groups, size=len(df))
    return df


def _make_time_binary_df(n: int = 90) -> pd.DataFrame:
    df = make_binary_df(n=n)
    df["t"] = np.arange(len(df), dtype=int)
    return df


class TestSaveLoadRoundtrip:
    """Save → load preserves every ``InnerValidConfig`` discriminant."""

    def test_holdout_roundtrip(self, tmp_path: Path) -> None:
        cfg = _build_config(
            split_method="stratified_kfold",
            inner_valid={"method": "holdout", "ratio": 0.2, "random_state": 42},
        )
        df = make_binary_df(n=80)
        m = Model(cfg)
        m.fit(data=df)

        out = tmp_path / "model"
        m.export(out)

        loaded = Model.load(out)
        assert loaded._cfg.training.early_stopping.inner_valid is not None
        assert loaded._cfg.training.early_stopping.inner_valid.method == "holdout"
        assert loaded._cfg.training.early_stopping.validation_ratio == 0.2

    def test_group_holdout_roundtrip(self, tmp_path: Path) -> None:
        cfg = _build_config(
            split_method="group_kfold",
            inner_valid={"method": "group_holdout", "ratio": 0.2, "random_state": 42},
            extra_data={"group_col": "g"},
        )
        df = _make_grouped_binary_df()
        m = Model(cfg)
        m.fit(data=df)

        out = tmp_path / "model"
        m.export(out)

        loaded = Model.load(out)
        assert loaded._cfg.training.early_stopping.inner_valid is not None
        assert loaded._cfg.training.early_stopping.inner_valid.method == "group_holdout"
        assert loaded._cfg.training.early_stopping.validation_ratio == 0.2

    def test_time_holdout_roundtrip(self, tmp_path: Path) -> None:
        cfg = _build_config(
            split_method="time_series",
            inner_valid={"method": "time_holdout", "ratio": 0.2},
            extra_data={"time_col": "t"},
        )
        df = _make_time_binary_df()
        m = Model(cfg)
        m.fit(data=df)

        out = tmp_path / "model"
        m.export(out)

        loaded = Model.load(out)
        assert loaded._cfg.training.early_stopping.inner_valid is not None
        assert loaded._cfg.training.early_stopping.inner_valid.method == "time_holdout"
        assert loaded._cfg.training.early_stopping.validation_ratio == 0.2


class TestNonDefaultRatioRoundtrip:
    """Non-default ratios round-trip correctly (was silently broken
    pre-H-0069 even for ``holdout`` because ``validation_ratio`` was
    not synced from ``inner_valid.ratio``)."""

    @pytest.mark.parametrize("ratio", [0.15, 0.3])
    def test_holdout_with_non_default_ratio(self, tmp_path: Path, ratio: float) -> None:
        cfg = _build_config(
            split_method="stratified_kfold",
            inner_valid={"method": "holdout", "ratio": ratio, "random_state": 42},
        )
        df = make_binary_df(n=80)
        m = Model(cfg)
        m.fit(data=df)

        out = tmp_path / "model"
        m.export(out)

        loaded = Model.load(out)
        assert loaded._cfg.training.early_stopping.inner_valid is not None
        assert loaded._cfg.training.early_stopping.inner_valid.ratio == ratio
        assert loaded._cfg.training.early_stopping.validation_ratio == ratio
