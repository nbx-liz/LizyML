"""Tests for BlockedGroupInnerValid and StratifiedTimeHoldoutInnerValid (H-0060).

TDD RED→GREEN: inner valid strategies for blocked_group_kfold.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from lizyml.training.inner_valid import (
    BlockedGroupInnerValid,
    StratifiedTimeHoldoutInnerValid,
    TimeHoldoutInnerValid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inner_data(
    n_groups: int = 8,
    rows_per_group: int = 3,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (y, groups, block_order) for inner valid testing.

    Groups are ordered by "time" (group 0 is earliest, group N is latest).
    """
    rng = np.random.RandomState(seed)
    groups_arr: list[Any] = []
    y_arr: list[int] = []
    for g in range(n_groups):
        for _ in range(rows_per_group):
            groups_arr.append(f"g{g}")
            y_arr.append(rng.randint(0, n_classes))
    return np.array(y_arr), np.array(groups_arr), np.arange(len(y_arr))


# ---------------------------------------------------------------------------
# BlockedGroupInnerValid
# ---------------------------------------------------------------------------


class TestBlockedGroupInnerValid:
    """Group-isolated, time-ordered, stratified inner valid."""

    def test_group_isolation(self) -> None:
        """No group appears in both inner train and inner valid."""
        y, groups, _ = _make_inner_data(n_groups=10)
        strategy = BlockedGroupInnerValid(ratio=0.3, task="binary")
        result = strategy.split(len(y), y=y, groups=groups)
        assert result is not None
        train_idx, valid_idx = result
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])
        assert train_groups.isdisjoint(valid_groups)

    def test_time_ordering(self) -> None:
        """Inner valid groups are from later time (higher group indices)."""
        y, groups, _ = _make_inner_data(n_groups=10)
        strategy = BlockedGroupInnerValid(ratio=0.3, task="regression")
        result = strategy.split(len(y), y=y, groups=groups)
        assert result is not None
        _, valid_idx = result
        valid_groups = set(groups[valid_idx])
        # Valid groups should be the "later" ones (higher numbered)
        # With 10 groups and ratio=0.3, expect ~3 groups in valid
        # These should be from the tail (g7, g8, g9)
        for vg in valid_groups:
            group_num = int(vg[1:])
            assert group_num >= 7, f"Expected tail groups, got {vg}"

    def test_classification_all_classes_covered(self) -> None:
        """Each class has at least 1 group in inner valid (binary)."""
        # Construct data: g0-g4 are class 0, g5-g9 are class 1
        n_per = 3
        groups = np.repeat([f"g{i}" for i in range(10)], n_per)
        y = np.array([0] * (5 * n_per) + [1] * (5 * n_per))
        strategy = BlockedGroupInnerValid(ratio=0.2, task="binary")
        result = strategy.split(len(y), y=y, groups=groups)
        assert result is not None
        _, valid_idx = result
        valid_classes = set(y[valid_idx])
        assert 0 in valid_classes, "Class 0 missing from inner valid"
        assert 1 in valid_classes, "Class 1 missing from inner valid"

    def test_regression_tail_groups(self) -> None:
        """Regression: tail ratio groups go to inner valid."""
        y, groups, _ = _make_inner_data(n_groups=10)
        y_reg = y.astype(float)
        strategy = BlockedGroupInnerValid(ratio=0.2, task="regression")
        result = strategy.split(len(y_reg), y=y_reg, groups=groups)
        assert result is not None
        _, valid_idx = result
        valid_groups = set(groups[valid_idx])
        # 10 groups * 0.2 = 2 groups in valid (tail)
        assert len(valid_groups) >= 1
        assert len(valid_groups) <= 3

    def test_groups_none_raises(self) -> None:
        y, _, _ = _make_inner_data()
        strategy = BlockedGroupInnerValid(ratio=0.2, task="binary")
        with pytest.raises(Exception, match="groups"):
            strategy.split(len(y), y=y, groups=None)

    def test_reproducibility(self) -> None:
        y, groups, _ = _make_inner_data(n_groups=10)
        s1 = BlockedGroupInnerValid(ratio=0.3, task="binary")
        s2 = BlockedGroupInnerValid(ratio=0.3, task="binary")
        r1 = s1.split(len(y), y=y, groups=groups)
        r2 = s2.split(len(y), y=y, groups=groups)
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])

    def test_complete_coverage(self) -> None:
        """All rows are in either inner train or inner valid."""
        y, groups, _ = _make_inner_data(n_groups=8)
        strategy = BlockedGroupInnerValid(ratio=0.25, task="binary")
        result = strategy.split(len(y), y=y, groups=groups)
        assert result is not None
        train_idx, valid_idx = result
        all_idx = np.sort(np.concatenate([train_idx, valid_idx]))
        np.testing.assert_array_equal(all_idx, np.arange(len(y)))


# ---------------------------------------------------------------------------
# Fallback to StratifiedTimeHoldout
# ---------------------------------------------------------------------------


class TestBlockedGroupInnerValidFallback:
    """Fallback when n_unique_groups < 4."""

    def test_fallback_with_3_groups(self) -> None:
        groups = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1])
        strategy = BlockedGroupInnerValid(ratio=0.3, task="binary")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = strategy.split(len(y), y=y, groups=groups)
            assert result is not None
            assert any(
                "fallback" in str(warning.message).lower()
                or "few" in str(warning.message).lower()
                for warning in w
            )

    def test_fallback_with_2_groups(self) -> None:
        groups = np.array(["a", "a", "a", "b", "b", "b"])
        y = np.array([0, 0, 1, 1, 0, 1])
        strategy = BlockedGroupInnerValid(ratio=0.3, task="binary")
        result = strategy.split(len(y), y=y, groups=groups)
        assert result is not None
        train_idx, valid_idx = result
        assert len(train_idx) > 0
        assert len(valid_idx) > 0


# ---------------------------------------------------------------------------
# StratifiedTimeHoldoutInnerValid
# ---------------------------------------------------------------------------


class TestStratifiedTimeHoldout:
    """Per-class tail selection for inner valid."""

    def test_all_classes_in_valid(self) -> None:
        """Each class has at least 1 row in inner valid."""
        y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        strategy = StratifiedTimeHoldoutInnerValid(ratio=0.3)
        result = strategy.split(len(y), y=y)
        assert result is not None
        _, valid_idx = result
        valid_classes = set(y[valid_idx])
        assert 0 in valid_classes
        assert 1 in valid_classes

    def test_tail_rows_selected(self) -> None:
        """Valid rows come from the tail of each class."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        strategy = StratifiedTimeHoldoutInnerValid(ratio=0.4)
        result = strategy.split(len(y), y=y)
        assert result is not None
        _, valid_idx = result
        # Class 0 indices: [0, 2, 4, 6, 8], tail 40% = [6, 8]
        # Class 1 indices: [1, 3, 5, 7, 9], tail 40% = [7, 9]
        assert 8 in valid_idx  # tail of class 0
        assert 9 in valid_idx  # tail of class 1

    def test_multiclass(self) -> None:
        """Works with 3+ classes."""
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        strategy = StratifiedTimeHoldoutInnerValid(ratio=0.4)
        result = strategy.split(len(y), y=y)
        assert result is not None
        _, valid_idx = result
        assert set(y[valid_idx]) == {0, 1, 2}

    def test_regression_fallback(self) -> None:
        """Regression (no y): falls back to simple tail holdout."""
        strategy = StratifiedTimeHoldoutInnerValid(ratio=0.2)
        result = strategy.split(10, y=None)
        assert result is not None
        train_idx, valid_idx = result
        # Last 20% = 2 rows
        np.testing.assert_array_equal(valid_idx, np.array([8, 9]))

    def test_regression_fallback_raises_when_ratio_consumes_all(self) -> None:
        """Inline fallback (#124) must reject configurations that would
        consume the entire training set, matching the pre-refactor contract
        of ``TimeHoldoutInnerValid``. The trip wire fires for n_samples=1
        because ``max(1, int(1*r))`` is always 1, leaving zero training rows.
        """
        strategy = StratifiedTimeHoldoutInnerValid(ratio=0.5)
        with pytest.raises(ValueError, match="would consume all"):
            strategy.split(1, y=None)

    def test_complete_coverage(self) -> None:
        y = np.array([0, 1, 0, 1, 0, 1])
        strategy = StratifiedTimeHoldoutInnerValid(ratio=0.3)
        result = strategy.split(len(y), y=y)
        assert result is not None
        train_idx, valid_idx = result
        all_idx = np.sort(np.concatenate([train_idx, valid_idx]))
        np.testing.assert_array_equal(all_idx, np.arange(len(y)))

    def test_min_one_per_class(self) -> None:
        """Even with very small ratio, at least 1 per class."""
        y = np.array([0, 0, 0, 0, 0, 1])  # Only 1 instance of class 1
        strategy = StratifiedTimeHoldoutInnerValid(ratio=0.01)
        result = strategy.split(len(y), y=y)
        assert result is not None
        _, valid_idx = result
        assert 1 in y[valid_idx], "Class 1 must be in valid"


class TestRegressionFallbackIsTimeOrdered:
    """The <4-group fallback must not stratify a continuous target (H-0092).

    BLUEPRINT 10.3.3 fixes the regression fallback as equivalent to
    ``TimeHoldoutInnerValid``. It was not: ``np.unique`` on a continuous target
    yields one "class" per row, each contributing at least one validation row,
    so every row landed in validation and inner-train came back empty. Nothing
    raised -- the empty split was handed to the estimator.
    """

    @staticmethod
    def _continuous(n: int) -> np.ndarray:
        return np.arange(n, dtype=float) * 1.5 + 0.25  # every value distinct

    def test_regression_fallback_matches_time_holdout(self) -> None:
        n = 6
        groups = np.array([0, 0, 1, 1, 2, 2])  # 3 groups: below the threshold
        with pytest.warns(UserWarning, match="Falling back to TimeHoldout"):
            train_idx, valid_idx = BlockedGroupInnerValid(
                ratio=0.3, task="regression"
            ).split(n, y=self._continuous(n), groups=groups)

        expected_train, expected_valid = TimeHoldoutInnerValid(ratio=0.3).split(n)
        assert train_idx.tolist() == expected_train.tolist()
        assert valid_idx.tolist() == expected_valid.tolist()
        assert train_idx.size > 0

    def test_classification_fallback_still_stratifies(self) -> None:
        n = 6
        groups = np.array([0, 0, 1, 1, 2, 2])
        y = np.array([0, 1, 0, 1, 0, 1])
        with pytest.warns(UserWarning, match="Falling back to StratifiedTimeHoldout"):
            train_idx, valid_idx = BlockedGroupInnerValid(
                ratio=0.3, task="binary"
            ).split(n, y=y, groups=groups)
        # One tail row per class, so both classes are represented in validation.
        assert sorted(y[valid_idx].tolist()) == [0, 1]
        assert train_idx.size > 0

    def test_stratifying_a_continuous_target_directly_is_refused(self) -> None:
        n = 6
        with pytest.raises(ValueError, match="cannot be stratified"):
            StratifiedTimeHoldoutInnerValid(ratio=0.3).split(n, y=self._continuous(n))
