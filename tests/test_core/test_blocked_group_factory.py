"""Tests for factory integration of blocked_group_kfold (H-0060).

TDD RED phase: factory dispatch and inner valid auto-resolution.
"""

from __future__ import annotations

import numpy as np

from lizyml.config.schema import (
    LizyMLConfig,
)
from lizyml.core._model_factories import (
    _resolve_auto_inner_valid,
    build_inner_valid,
    build_splitter,
)
from lizyml.splitters.blocked_group_kfold import BlockedGroupKFoldSplitter
from lizyml.training.inner_valid import (
    BlockedGroupInnerValid,
)


def _make_cfg(**split_overrides: object) -> LizyMLConfig:
    split = {
        "method": "blocked_group_kfold",
        "blocks": {"col": "date", "cutoffs": ["2025-03"]},
        "groups": {"col": "user_id", "n_splits": 3},
    }
    split.update(split_overrides)
    return LizyMLConfig(
        config_version=1,
        task="binary",
        data={"target": "y"},
        split=split,
        model={"name": "lgbm"},
    )


class TestBuildSplitter:
    """build_splitter dispatches BlockedGroupKFoldConfig correctly."""

    def test_returns_blocked_group_kfold_splitter(self) -> None:
        cfg = _make_cfg()
        block_values = np.array([0, 0, 1, 1, 2, 2])
        splitter = build_splitter(
            cfg, block_values=block_values, task="binary", seed=42
        )
        assert isinstance(splitter, BlockedGroupKFoldSplitter)

    def test_existing_splitters_unaffected(self) -> None:
        """build_splitter with no block_values still works for existing methods."""
        cfg = LizyMLConfig(
            config_version=1,
            task="binary",
            data={"target": "y"},
            split={"method": "stratified_kfold", "n_splits": 5},
            model={"name": "lgbm"},
        )
        splitter = build_splitter(cfg)
        assert not isinstance(splitter, BlockedGroupKFoldSplitter)


class TestAutoResolveInnerValid:
    """_resolve_auto_inner_valid for blocked_group_kfold."""

    def test_binary_returns_blocked_group_inner_valid(self) -> None:
        iv = _resolve_auto_inner_valid("blocked_group_kfold", 0.1, 42, task="binary")
        assert isinstance(iv, BlockedGroupInnerValid)
        assert iv.task == "binary"

    def test_regression_returns_blocked_group_inner_valid(self) -> None:
        iv = _resolve_auto_inner_valid(
            "blocked_group_kfold", 0.2, 42, task="regression"
        )
        assert isinstance(iv, BlockedGroupInnerValid)
        assert iv.task == "regression"

    def test_existing_methods_unaffected(self) -> None:
        """Existing auto-resolve rules still work."""
        iv = _resolve_auto_inner_valid("stratified_kfold", 0.1, 42)
        assert not isinstance(iv, BlockedGroupInnerValid)


class TestBuildInnerValid:
    """build_inner_valid integrates with blocked_group_kfold."""

    def test_auto_resolve_binary(self) -> None:
        cfg = _make_cfg()
        iv = build_inner_valid(cfg)
        assert isinstance(iv, BlockedGroupInnerValid)
        assert iv.task == "binary"

    def test_auto_resolve_regression(self) -> None:
        cfg = LizyMLConfig(
            config_version=1,
            task="regression",
            data={"target": "y"},
            split={
                "method": "blocked_group_kfold",
                "blocks": {"col": "date", "cutoffs": ["2025-03"]},
                "groups": {"col": "user_id", "n_splits": 3},
            },
            model={"name": "lgbm"},
        )
        iv = build_inner_valid(cfg)
        assert isinstance(iv, BlockedGroupInnerValid)
        assert iv.task == "regression"
