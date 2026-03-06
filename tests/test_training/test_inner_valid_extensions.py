"""Tests for InnerValid extensions: stratified/group/time holdout (H-0020)."""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.config.loader import load_config
from lizyml.core.exceptions import LizyMLError
from lizyml.core.model import Model
from lizyml.training.inner_valid import (
    GroupHoldoutInnerValid,
    HoldoutInnerValid,
    TimeHoldoutInnerValid,
)


class TestStratifiedHoldout:
    def test_class_balance_preserved(self) -> None:
        """Stratified holdout preserves class proportions."""
        y = np.array([0] * 80 + [1] * 20)
        iv = HoldoutInnerValid(ratio=0.2, random_state=42, stratify=True)
        train_idx, valid_idx = iv.split(len(y), y=y)

        # Check proportions in valid set
        valid_ratio = np.mean(y[valid_idx])
        # Original ratio is 0.2, valid should be close
        assert 0.1 <= valid_ratio <= 0.3

    def test_no_overlap(self) -> None:
        y = np.array([0] * 50 + [1] * 50)
        iv = HoldoutInnerValid(ratio=0.2, random_state=42, stratify=True)
        train_idx, valid_idx = iv.split(len(y), y=y)
        assert len(set(train_idx) & set(valid_idx)) == 0

    def test_covers_all_samples(self) -> None:
        y = np.array([0] * 50 + [1] * 50)
        iv = HoldoutInnerValid(ratio=0.2, random_state=42, stratify=True)
        train_idx, valid_idx = iv.split(len(y), y=y)
        assert len(train_idx) + len(valid_idx) == len(y)

    def test_non_stratified_still_works(self) -> None:
        """Non-stratified holdout (default) still functions normally."""
        iv = HoldoutInnerValid(ratio=0.2, random_state=42, stratify=False)
        train_idx, valid_idx = iv.split(100)
        assert len(train_idx) + len(valid_idx) == 100


class TestGroupHoldout:
    def test_no_group_overlap(self) -> None:
        """Groups in valid should not appear in train."""
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        iv = GroupHoldoutInnerValid(ratio=0.25, random_state=42)
        train_idx, valid_idx = iv.split(len(groups), groups=groups)

        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])
        assert len(train_groups & valid_groups) == 0

    def test_covers_all_samples(self) -> None:
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        iv = GroupHoldoutInnerValid(ratio=0.25, random_state=42)
        train_idx, valid_idx = iv.split(len(groups), groups=groups)
        assert len(train_idx) + len(valid_idx) == len(groups)

    def test_requires_groups(self) -> None:
        iv = GroupHoldoutInnerValid(ratio=0.25, random_state=42)
        with pytest.raises(LizyMLError) as exc_info:
            iv.split(10, groups=None)
        assert exc_info.value.code.value == "CONFIG_INVALID"

    def test_reproducible(self) -> None:
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        iv = GroupHoldoutInnerValid(ratio=0.2, random_state=42)
        t1, v1 = iv.split(len(groups), groups=groups)
        t2, v2 = iv.split(len(groups), groups=groups)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)

    def test_tail_groups_selected_by_input_order(self) -> None:
        """Validation should contain the LAST groups in input order."""
        # Groups appear in order: A, B, C, D, E
        groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"])
        iv = GroupHoldoutInnerValid(ratio=0.4, random_state=0)
        _, valid_idx = iv.split(len(groups), groups=groups)
        valid_groups = set(groups[valid_idx])
        # Last 40% of 5 groups = 2 groups → should be D, E
        assert valid_groups == {"D", "E"}

    def test_tail_groups_non_sequential_input(self) -> None:
        """Groups not in sorted order: tail by first-appearance order."""
        # First appearance order: C, A, B (not alphabetical)
        groups = np.array(["C", "C", "A", "A", "B", "B"])
        iv = GroupHoldoutInnerValid(ratio=0.34, random_state=0)
        _, valid_idx = iv.split(len(groups), groups=groups)
        valid_groups = set(groups[valid_idx])
        # 1 tail group by first-appearance → B (last seen group)
        assert valid_groups == {"B"}


class TestTimeHoldout:
    def test_last_rows_are_valid(self) -> None:
        """Validation rows should be the last rows."""
        iv = TimeHoldoutInnerValid(ratio=0.2)
        train_idx, valid_idx = iv.split(100)
        assert valid_idx[-1] == 99
        assert valid_idx[0] == 80
        assert train_idx[-1] == 79
        assert train_idx[0] == 0

    def test_no_shuffle(self) -> None:
        """Results should be identical across repeated calls."""
        iv = TimeHoldoutInnerValid(ratio=0.2)
        t1, v1 = iv.split(100)
        t2, v2 = iv.split(100)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)

    def test_covers_all_samples(self) -> None:
        iv = TimeHoldoutInnerValid(ratio=0.3)
        train_idx, valid_idx = iv.split(50)
        assert len(train_idx) + len(valid_idx) == 50

    def test_monotonic_order(self) -> None:
        """Train and valid indices should be in sorted order."""
        iv = TimeHoldoutInnerValid(ratio=0.2)
        train_idx, valid_idx = iv.split(100)
        np.testing.assert_array_equal(train_idx, np.sort(train_idx))
        np.testing.assert_array_equal(valid_idx, np.sort(valid_idx))


class TestAutoResolve:
    """Test auto-resolve of inner_valid from outer split method."""

    @staticmethod
    def _make_model(task: str, split_method: str, es_enabled: bool = True) -> Model:
        raw = {
            "config_version": 1,
            "task": task,
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": split_method},
            "training": {"early_stopping": {"enabled": es_enabled, "rounds": 10}},
        }
        cfg = load_config(raw)
        return Model(cfg)

    def test_stratified_kfold_auto_resolves(self) -> None:
        model = self._make_model("binary", "stratified_kfold")
        iv = model._build_inner_valid()
        assert isinstance(iv, HoldoutInnerValid)
        assert iv.stratify is True

    def test_group_kfold_auto_resolves(self) -> None:
        model = self._make_model("binary", "group_kfold")
        iv = model._build_inner_valid()
        assert isinstance(iv, GroupHoldoutInnerValid)

    def test_time_series_auto_resolves(self) -> None:
        model = self._make_model("regression", "time_series")
        iv = model._build_inner_valid()
        assert isinstance(iv, TimeHoldoutInnerValid)

    def test_kfold_auto_resolves_plain(self) -> None:
        model = self._make_model("regression", "kfold")
        iv = model._build_inner_valid()
        assert isinstance(iv, HoldoutInnerValid)
        assert iv.stratify is False


class TestExplicitOverride:
    """Explicit inner_valid config overrides auto-resolve."""

    def test_explicit_holdout_overrides_auto(self) -> None:
        raw = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "stratified_kfold"},
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "rounds": 10,
                    "inner_valid": {
                        "method": "holdout",
                        "ratio": 0.15,
                        "stratify": False,
                    },
                }
            },
        }
        cfg = load_config(raw)
        model = Model(cfg)
        iv = model._build_inner_valid()
        assert isinstance(iv, HoldoutInnerValid)
        assert iv.stratify is False
        assert iv.ratio == 0.15

    def test_validation_ratio_still_works(self) -> None:
        raw = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "stratified_kfold"},
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "rounds": 10,
                    "validation_ratio": 0.2,
                }
            },
        }
        cfg = load_config(raw)
        model = Model(cfg)
        iv = model._build_inner_valid()
        assert isinstance(iv, HoldoutInnerValid)
        assert iv.ratio == 0.2

    def test_explicit_group_holdout(self) -> None:
        raw = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "y", "group_col": "g"},
            "model": {"name": "lgbm"},
            "split": {"method": "stratified_kfold"},
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "rounds": 10,
                    "inner_valid": {"method": "group_holdout", "ratio": 0.2},
                }
            },
        }
        cfg = load_config(raw)
        model = Model(cfg)
        iv = model._build_inner_valid()
        assert isinstance(iv, GroupHoldoutInnerValid)
        assert iv.ratio == 0.2

    def test_explicit_time_holdout(self) -> None:
        raw = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": {"method": "kfold"},
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "rounds": 10,
                    "inner_valid": {"method": "time_holdout", "ratio": 0.15},
                }
            },
        }
        cfg = load_config(raw)
        model = Model(cfg)
        iv = model._build_inner_valid()
        assert isinstance(iv, TimeHoldoutInnerValid)
        assert iv.ratio == 0.15
