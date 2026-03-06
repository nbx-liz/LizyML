"""Tests for Phase 5: splitters and SplitPlan."""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.core.specs.calibration_spec import CalibrationSpec
from lizyml.core.specs.split_plan import SplitPlan
from lizyml.core.specs.split_spec import SplitSpec
from lizyml.core.specs.training_spec import (
    EarlyStoppingSpec,
    InnerValidSpec,
    TrainingSpec,
)
from lizyml.splitters import (
    GroupKFoldSplitter,
    GroupTimeSeriesSplitter,
    KFoldSplitter,
    PurgedTimeSeriesSplitter,
    StratifiedGroupKFoldSplitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 50  # default sample count


def _collect(splitter, n=N, y=None, groups=None):
    return list(splitter.split(n, y=y, groups=groups))


def _no_leakage(folds):
    """Assert train and valid indices are disjoint in every fold."""
    for train, valid in folds:
        assert len(set(train) & set(valid)) == 0, "Train/valid index overlap detected."


def _covers_all(folds, n=N):
    """Assert every sample appears in at least one validation fold."""
    seen = set()
    for _, valid in folds:
        seen.update(valid.tolist())
    assert len(seen) == n


# ---------------------------------------------------------------------------
# KFoldSplitter
# ---------------------------------------------------------------------------


class TestKFoldSplitter:
    def test_correct_fold_count(self) -> None:
        folds = _collect(KFoldSplitter(n_splits=5))
        assert len(folds) == 5

    def test_no_leakage(self) -> None:
        _no_leakage(_collect(KFoldSplitter(n_splits=5)))

    def test_covers_all_samples(self) -> None:
        _covers_all(_collect(KFoldSplitter(n_splits=5)))

    def test_seed_reproducibility(self) -> None:
        s1 = KFoldSplitter(n_splits=5, shuffle=True, random_state=42)
        s2 = KFoldSplitter(n_splits=5, shuffle=True, random_state=42)
        for (t1, v1), (t2, v2) in zip(_collect(s1), _collect(s2), strict=True):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_differ(self) -> None:
        s1 = KFoldSplitter(n_splits=5, shuffle=True, random_state=1)
        s2 = KFoldSplitter(n_splits=5, shuffle=True, random_state=2)
        folds1 = _collect(s1)
        folds2 = _collect(s2)
        # At least one fold should differ
        pairs = zip(folds1, folds2, strict=True)
        any_diff = any(not np.array_equal(v1, v2) for (_, v1), (_, v2) in pairs)
        assert any_diff


# ---------------------------------------------------------------------------
# StratifiedKFoldSplitter
# ---------------------------------------------------------------------------


class TestStratifiedKFoldSplitter:
    def test_correct_fold_count(self) -> None:
        y = np.array([0, 1] * (N // 2))
        folds = _collect(StratifiedKFoldSplitter(n_splits=5), y=y)
        assert len(folds) == 5

    def test_no_leakage(self) -> None:
        y = np.array([0, 1] * (N // 2))
        _no_leakage(_collect(StratifiedKFoldSplitter(n_splits=5), y=y))

    def test_seed_reproducibility(self) -> None:
        y = np.array([0, 1] * (N // 2))
        s1 = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        s2 = StratifiedKFoldSplitter(n_splits=5, random_state=42)
        for (t1, _), (t2, _) in zip(_collect(s1, y=y), _collect(s2, y=y), strict=True):
            np.testing.assert_array_equal(t1, t2)

    def test_requires_y(self) -> None:
        with pytest.raises(ValueError, match="requires y"):
            _collect(StratifiedKFoldSplitter(n_splits=5))


# ---------------------------------------------------------------------------
# GroupKFoldSplitter
# ---------------------------------------------------------------------------


class TestGroupKFoldSplitter:
    def test_correct_fold_count(self) -> None:
        groups = np.repeat(np.arange(10), N // 10)
        folds = _collect(GroupKFoldSplitter(n_splits=5), groups=groups)
        assert len(folds) == 5

    def test_no_group_overlap(self) -> None:
        groups = np.repeat(np.arange(10), N // 10)
        for train, valid in _collect(GroupKFoldSplitter(n_splits=5), groups=groups):
            train_groups = set(groups[train])
            valid_groups = set(groups[valid])
            assert train_groups & valid_groups == set()

    def test_requires_groups(self) -> None:
        with pytest.raises(ValueError, match="requires groups"):
            _collect(GroupKFoldSplitter(n_splits=5))


# ---------------------------------------------------------------------------
# StratifiedGroupKFoldSplitter
# ---------------------------------------------------------------------------


class TestStratifiedGroupKFoldSplitter:
    def test_correct_fold_count(self) -> None:
        groups = np.repeat(np.arange(10), N // 10)
        y = np.array([0, 1] * (N // 2))
        folds = _collect(StratifiedGroupKFoldSplitter(n_splits=5), y=y, groups=groups)
        assert len(folds) == 5

    def test_no_group_overlap(self) -> None:
        groups = np.repeat(np.arange(10), N // 10)
        y = np.array([0, 1] * (N // 2))
        for train, valid in _collect(
            StratifiedGroupKFoldSplitter(n_splits=5), y=y, groups=groups
        ):
            assert set(groups[train]) & set(groups[valid]) == set()

    def test_requires_y(self) -> None:
        groups = np.repeat(np.arange(10), N // 10)
        with pytest.raises(ValueError, match="requires y"):
            _collect(StratifiedGroupKFoldSplitter(), groups=groups)

    def test_requires_groups(self) -> None:
        y = np.array([0, 1] * (N // 2))
        with pytest.raises(ValueError, match="requires groups"):
            _collect(StratifiedGroupKFoldSplitter(), y=y)


# ---------------------------------------------------------------------------
# TimeSeriesSplitter
# ---------------------------------------------------------------------------


class TestTimeSeriesSplitter:
    def test_correct_fold_count(self) -> None:
        folds = _collect(TimeSeriesSplitter(n_splits=5))
        assert len(folds) == 5

    def test_no_leakage(self) -> None:
        _no_leakage(_collect(TimeSeriesSplitter(n_splits=5)))

    def test_train_precedes_valid(self) -> None:
        for train, valid in _collect(TimeSeriesSplitter(n_splits=5)):
            assert train.max() < valid.min()

    def test_seed_reproducibility(self) -> None:
        """TimeSeriesSplitter is deterministic (no shuffle)."""
        s1 = TimeSeriesSplitter(n_splits=5)
        s2 = TimeSeriesSplitter(n_splits=5)
        for (t1, v1), (t2, v2) in zip(_collect(s1), _collect(s2), strict=True):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_gap_removes_samples(self) -> None:
        folds_no_gap = _collect(TimeSeriesSplitter(n_splits=5, gap=0))
        folds_gap = _collect(TimeSeriesSplitter(n_splits=5, gap=3))
        for (t_no, _), (t_gap, _) in zip(folds_no_gap, folds_gap, strict=True):
            # With gap, valid set starts later so train and valid have space
            assert len(t_gap) <= len(t_no)


# ---------------------------------------------------------------------------
# PurgedTimeSeriesSplitter
# ---------------------------------------------------------------------------


class TestPurgedTimeSeriesSplitter:
    def test_correct_fold_count(self) -> None:
        folds = _collect(PurgedTimeSeriesSplitter(n_splits=5))
        assert len(folds) == 5

    def test_no_leakage(self) -> None:
        _no_leakage(_collect(PurgedTimeSeriesSplitter(n_splits=5)))

    def test_train_precedes_valid(self) -> None:
        for train, valid in _collect(PurgedTimeSeriesSplitter(n_splits=5)):
            assert train.max() < valid.min()

    def test_purge_shrinks_train(self) -> None:
        folds_no_purge = _collect(PurgedTimeSeriesSplitter(n_splits=5, purge_gap=0))
        folds_purged = _collect(PurgedTimeSeriesSplitter(n_splits=5, purge_gap=3))
        for (t_no, _), (t_p, _) in zip(folds_no_purge, folds_purged, strict=True):
            assert len(t_p) <= len(t_no)

    def test_seed_reproducibility(self) -> None:
        s1 = PurgedTimeSeriesSplitter(n_splits=5, purge_gap=2)
        s2 = PurgedTimeSeriesSplitter(n_splits=5, purge_gap=2)
        for (t1, v1), (t2, v2) in zip(_collect(s1), _collect(s2), strict=True):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_invalid_purge_gap(self) -> None:
        with pytest.raises(ValueError, match="purge_gap must be >= 0"):
            PurgedTimeSeriesSplitter(purge_gap=-1)


# ---------------------------------------------------------------------------
# GroupTimeSeriesSplitter
# ---------------------------------------------------------------------------


class TestGroupTimeSeriesSplitter:
    def _groups(self, n_groups: int = 12) -> np.ndarray:
        """Create groups array with n_groups distinct groups."""
        per = N // n_groups
        groups = np.repeat(np.arange(n_groups), per)
        # Pad to N if needed
        if len(groups) < N:
            groups = np.concatenate([groups, np.full(N - len(groups), n_groups - 1)])
        return groups[:N]

    def test_correct_fold_count(self) -> None:
        groups = self._groups()
        folds = _collect(GroupTimeSeriesSplitter(n_splits=5), groups=groups)
        assert len(folds) == 5

    def test_no_group_overlap(self) -> None:
        groups = self._groups()
        for train, valid in _collect(
            GroupTimeSeriesSplitter(n_splits=5), groups=groups
        ):
            assert set(groups[train]) & set(groups[valid]) == set()

    def test_train_groups_precede_valid_groups(self) -> None:
        groups = self._groups()
        unique = np.unique(groups)
        group_order = {g: i for i, g in enumerate(unique)}
        for train, valid in _collect(
            GroupTimeSeriesSplitter(n_splits=5), groups=groups
        ):
            max_train_order = max(group_order[g] for g in groups[train])
            min_valid_order = min(group_order[g] for g in groups[valid])
            assert max_train_order < min_valid_order

    def test_requires_groups(self) -> None:
        with pytest.raises(ValueError, match="requires groups"):
            _collect(GroupTimeSeriesSplitter(n_splits=5))

    def test_too_few_groups_raises(self) -> None:
        # Only 2 groups but need at least 6 for n_splits=5
        groups = np.array([0, 0, 1, 1] * 10)
        with pytest.raises(ValueError, match="Not enough groups"):
            _collect(GroupTimeSeriesSplitter(n_splits=5), groups=groups)


# ---------------------------------------------------------------------------
# SplitPlan
# ---------------------------------------------------------------------------


class TestSplitPlan:
    def test_create_kfold_no_early_stopping(self) -> None:
        split_spec = SplitSpec(
            method="kfold",
            n_splits=5,
            random_state=42,
            shuffle=True,
            gap=0,
        )
        training_spec = TrainingSpec(
            seed=42,
            early_stopping=EarlyStoppingSpec(
                enabled=False, rounds=50, inner_valid=None
            ),
        )
        plan = SplitPlan.create(split_spec, training_spec)
        assert isinstance(plan.outer, KFoldSplitter)
        assert plan.inner is None
        assert plan.calibration is None

    def test_create_with_inner_valid(self) -> None:
        split_spec = SplitSpec(
            method="kfold",
            n_splits=5,
            random_state=42,
            shuffle=True,
            gap=0,
        )
        training_spec = TrainingSpec(
            seed=42,
            early_stopping=EarlyStoppingSpec(
                enabled=True,
                rounds=50,
                inner_valid=InnerValidSpec(
                    method="holdout", ratio=0.1, random_state=42
                ),
            ),
        )
        plan = SplitPlan.create(split_spec, training_spec)
        assert plan.inner is not None

    def test_create_with_calibration(self) -> None:
        split_spec = SplitSpec(
            method="kfold",
            n_splits=5,
            random_state=42,
            shuffle=True,
            gap=0,
        )
        training_spec = TrainingSpec(
            seed=42,
            early_stopping=EarlyStoppingSpec(
                enabled=False, rounds=50, inner_valid=None
            ),
        )
        cal_spec = CalibrationSpec(method="platt", n_splits=3)
        plan = SplitPlan.create(split_spec, training_spec, cal_spec)
        assert plan.calibration is not None

    def test_outer_split_reproducibility(self) -> None:
        split_spec = SplitSpec(
            method="kfold",
            n_splits=5,
            random_state=42,
            shuffle=True,
            gap=0,
        )
        training_spec = TrainingSpec(
            seed=42,
            early_stopping=EarlyStoppingSpec(
                enabled=False, rounds=50, inner_valid=None
            ),
        )
        plan1 = SplitPlan.create(split_spec, training_spec)
        plan2 = SplitPlan.create(split_spec, training_spec)
        folds1 = list(plan1.outer.split(N))
        folds2 = list(plan2.outer.split(N))
        for (t1, v1), (t2, v2) in zip(folds1, folds2, strict=True):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)
