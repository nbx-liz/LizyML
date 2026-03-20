"""Tests for BlockedGroupKFoldSplitter (H-0060).

TDD RED phase: these tests must fail until the splitter is implemented.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lizyml.splitters.blocked_group_kfold import BlockedGroupKFoldSplitter

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_data(
    n_users: int = 7,
    n_periods: int = 5,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (block_values, groups, y) for testing.

    Each user appears in a random subset of periods.
    Returns arrays sorted by block_values.
    """
    rng = np.random.RandomState(seed)
    rows_block = []
    rows_group = []
    rows_y = []
    for period in range(n_periods):
        # Each user appears in this period with 70% probability
        for user in range(n_users):
            if rng.random() < 0.7:
                rows_block.append(period)
                rows_group.append(f"user_{user}")
                rows_y.append(rng.randint(0, n_classes))
    block_values = np.array(rows_block)
    groups = np.array(rows_group)
    y = np.array(rows_y)
    # Already sorted by period
    return block_values, groups, y


def _make_deterministic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed small dataset matching the spec discussion example."""
    # 7 users, 5 months (0=Jan, 1=Feb, 2=Mar, 3=Apr, 4=May)
    block_values = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    groups = np.array(
        ["A", "B", "C", "A", "B", "D", "C", "D", "E", "A", "E", "F", "B", "F", "G"]
    )
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return block_values, groups, y


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestFoldCount:
    """Fold count = len(cutoffs) * n_splits - skipped."""

    def test_expanding_fold_count(self) -> None:
        block_values, groups, y = _make_deterministic_data()
        cutoffs = [1, 2, 3]  # 3 time folds
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        folds = list(splitter.split(len(block_values), y=y, groups=groups))
        # 3 time folds * 2 group splits = 6, minus any skipped
        assert len(folds) <= 6
        assert len(folds) >= 1  # At least some folds should work

    def test_sliding_fold_count(self) -> None:
        block_values, groups, y = _make_deterministic_data()
        cutoffs = [1, 2, 3]
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="sliding",
            train_window=2,
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        folds = list(splitter.split(len(block_values), y=y, groups=groups))
        assert len(folds) <= 6
        assert len(folds) >= 1

    def test_single_cutoff(self) -> None:
        block_values, groups, y = _make_deterministic_data()
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=[2],  # 1 time fold
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        folds = list(splitter.split(len(block_values), y=y, groups=groups))
        assert len(folds) <= 2  # 1 time fold * 2 group splits


# ---------------------------------------------------------------------------
# Leak prevention (CRITICAL)
# ---------------------------------------------------------------------------


class TestLeakPrevention:
    """No user appears in both train and valid in any fold."""

    def test_no_user_leak(self) -> None:
        block_values, groups, y = _make_data(n_users=10, n_periods=5, seed=42)
        cutoffs = [1, 2, 3]
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="expanding",
            n_splits=3,
            stratify=False,
            shuffle=True,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        for train_idx, valid_idx in splitter.split(
            len(block_values), y=y, groups=groups
        ):
            train_users = set(groups[train_idx])
            valid_users = set(groups[valid_idx])
            assert train_users.isdisjoint(valid_users), (
                f"User leak: {train_users & valid_users}"
            )

    def test_no_temporal_leak(self) -> None:
        block_values, groups, y = _make_deterministic_data()
        cutoffs = [2]  # train: <2 (Jan,Feb), valid: >=2 (Mar+)
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        for train_idx, valid_idx in splitter.split(
            len(block_values), y=y, groups=groups
        ):
            # All train rows must be from periods < cutoff
            assert np.all(block_values[train_idx] < 2)
            # All valid rows must be from periods >= cutoff
            assert np.all(block_values[valid_idx] >= 2)

    def test_no_leak_with_stratify(self) -> None:
        block_values, groups, y = _make_data(n_users=15, n_periods=4, seed=99)
        cutoffs = [1, 2]
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="expanding",
            n_splits=3,
            stratify=True,
            shuffle=True,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        for train_idx, valid_idx in splitter.split(
            len(block_values), y=y, groups=groups
        ):
            train_users = set(groups[train_idx])
            valid_users = set(groups[valid_idx])
            assert train_users.isdisjoint(valid_users)

    def test_no_leak_sliding_mode(self) -> None:
        block_values, groups, y = _make_data(n_users=10, n_periods=6, seed=123)
        cutoffs = [2, 3, 4]
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="sliding",
            train_window=2,
            n_splits=2,
            stratify=False,
            shuffle=True,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        for fold_i, (train_idx, valid_idx) in enumerate(
            splitter.split(len(block_values), y=y, groups=groups)
        ):
            train_users = set(groups[train_idx])
            valid_users = set(groups[valid_idx])
            assert train_users.isdisjoint(valid_users), f"Leak in fold {fold_i}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same seed produces identical fold indices."""

    def test_same_seed_same_folds(self) -> None:
        block_values, groups, y = _make_data(seed=42)
        kwargs = dict(
            block_values=block_values,
            cutoffs=[1, 2],
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=True,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        folds1 = list(
            BlockedGroupKFoldSplitter(**kwargs).split(
                len(block_values), y=y, groups=groups
            )
        )
        folds2 = list(
            BlockedGroupKFoldSplitter(**kwargs).split(
                len(block_values), y=y, groups=groups
            )
        )
        assert len(folds1) == len(folds2)
        for (t1, v1), (t2, v2) in zip(folds1, folds2, strict=True):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_different_seed_different_folds(self) -> None:
        block_values, groups, y = _make_data(n_users=20, n_periods=4, seed=42)
        common = dict(
            block_values=block_values,
            cutoffs=[1, 2],
            mode="expanding",
            n_splits=3,
            stratify=False,
            shuffle=True,
            min_train_rows=1,
            min_valid_rows=1,
        )
        folds1 = list(
            BlockedGroupKFoldSplitter(**common, random_state=1).split(
                len(block_values), y=y, groups=groups
            )
        )
        folds2 = list(
            BlockedGroupKFoldSplitter(**common, random_state=99).split(
                len(block_values), y=y, groups=groups
            )
        )
        # At least one fold should differ
        any_diff = any(
            not (np.array_equal(t1, t2) and np.array_equal(v1, v2))
            for (t1, v1), (t2, v2) in zip(folds1, folds2, strict=False)
        )
        assert any_diff, "Different seeds should produce different folds"


# ---------------------------------------------------------------------------
# Mode tests
# ---------------------------------------------------------------------------


class TestModes:
    """Expanding vs sliding train period assignment."""

    def test_expanding_cumulative_train(self) -> None:
        """Expanding: fold k includes all periods P0..Pk in train."""
        block_values, groups, y = _make_deterministic_data()
        cutoffs = [1, 2, 3]
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        folds = list(splitter.split(len(block_values), y=y, groups=groups))
        # 3 time folds * 2 group splits = up to 6 folds
        assert len(folds) >= 3  # At least some per time fold

        # Group all folds by time fold (check train period boundaries)
        for train_idx, valid_idx in folds:
            train_periods = set(block_values[train_idx])
            valid_periods = set(block_values[valid_idx])
            # Train periods must be strictly before valid periods
            if train_periods and valid_periods:
                assert max(train_periods) < min(valid_periods)

    def test_sliding_windowed_train(self) -> None:
        """Sliding: fold k uses only last train_window periods."""
        block_values, groups, y = _make_deterministic_data()
        cutoffs = [1, 2, 3]
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="sliding",
            train_window=2,
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        folds = list(splitter.split(len(block_values), y=y, groups=groups))
        assert len(folds) >= 1

        # For sliding with window=2: train should span at most 2 periods
        for train_idx, _ in folds:
            train_periods = set(block_values[train_idx])
            assert len(train_periods) <= 2, (
                f"Sliding window=2 but train has {len(train_periods)} periods"
            )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


class TestStratification:
    """Binary/multiclass: user folds have balanced class distribution."""

    def test_stratified_class_balance(self) -> None:
        # Large enough dataset for meaningful stratification
        block_values, groups, y = _make_data(
            n_users=30, n_periods=4, n_classes=2, seed=42
        )
        cutoffs = [2]
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=cutoffs,
            mode="expanding",
            n_splits=3,
            stratify=True,
            shuffle=True,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        valid_ratios = []
        for _, valid_idx in splitter.split(len(block_values), y=y, groups=groups):
            if len(valid_idx) > 0:
                valid_ratios.append(y[valid_idx].mean())

        if len(valid_ratios) >= 2:
            # Class ratios across folds should be roughly similar
            max_diff = max(valid_ratios) - min(valid_ratios)
            assert max_diff < 0.5, f"Class balance too uneven: {valid_ratios}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_groups_none_raises(self) -> None:
        block_values, _, y = _make_deterministic_data()
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=[2],
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
        )
        with pytest.raises(ValueError, match="groups"):
            list(splitter.split(len(block_values), y=y, groups=None))

    def test_min_rows_skip_warns(self) -> None:
        """Folds below min_rows thresholds are skipped with warning."""
        block_values, groups, y = _make_deterministic_data()
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=[1],
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=100,  # Very high threshold
            min_valid_rows=100,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            folds = list(splitter.split(len(block_values), y=y, groups=groups))
            assert len(folds) == 0  # All folds skipped
            assert any("skip" in str(warning.message).lower() for warning in w)

    def test_all_users_all_periods(self) -> None:
        """Every user in every period — high exclusion but should work."""
        n_users, n_periods = 6, 3
        block_values = np.repeat(np.arange(n_periods), n_users)
        groups = np.tile([f"u{i}" for i in range(n_users)], n_periods)
        y = np.random.RandomState(42).randint(0, 2, len(block_values))
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=[1],
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        folds = list(splitter.split(len(block_values), y=y, groups=groups))
        assert len(folds) == 2
        for train_idx, valid_idx in folds:
            assert set(groups[train_idx]).isdisjoint(set(groups[valid_idx]))

    def test_index_dtype_is_intp(self) -> None:
        block_values, groups, y = _make_deterministic_data()
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=[2],
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=False,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        for train_idx, valid_idx in splitter.split(
            len(block_values), y=y, groups=groups
        ):
            assert train_idx.dtype == np.intp
            assert valid_idx.dtype == np.intp

    def test_no_overlap_between_train_and_valid_indices(self) -> None:
        """Train and valid index arrays never share row indices."""
        block_values, groups, y = _make_data(n_users=10, n_periods=5, seed=42)
        splitter = BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=[1, 2, 3],
            mode="expanding",
            n_splits=2,
            stratify=False,
            shuffle=True,
            random_state=42,
            min_train_rows=1,
            min_valid_rows=1,
        )
        for train_idx, valid_idx in splitter.split(
            len(block_values), y=y, groups=groups
        ):
            assert set(train_idx).isdisjoint(set(valid_idx))
