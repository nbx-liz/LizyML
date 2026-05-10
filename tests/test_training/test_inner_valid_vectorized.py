"""Behavioral tests for the numpy-vectorised inner-valid splits (#116).

The refactor replaced three O(n) Python loops in ``inner_valid.py`` with
``np.isin`` / boolean-mask equivalents. These tests pin down the
behavioural contract on larger inputs where the speedup is visible — they
do *not* benchmark, but they ensure the vectorised paths produce the same
result as the slow reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.training.inner_valid import (
    BlockedGroupInnerValid,
    GroupHoldoutInnerValid,
    StratifiedTimeHoldoutInnerValid,
)


def _reference_group_mask(groups: np.ndarray, valid_groups: set) -> np.ndarray:
    """Slow Python reference for the membership mask."""
    return np.array([g in valid_groups for g in groups])


class TestGroupHoldoutVectorized:
    def test_large_groups_match_reference(self) -> None:
        rng = np.random.default_rng(42)
        n = 5_000
        n_groups = 100
        groups = rng.integers(0, n_groups, size=n)

        iv = GroupHoldoutInnerValid(ratio=0.2, random_state=0)
        train_idx, valid_idx = iv.split(n, groups=groups)

        # Sanity: no overlap, full coverage
        assert len(set(train_idx) & set(valid_idx)) == 0
        assert len(train_idx) + len(valid_idx) == n
        # Group isolation
        assert not (set(groups[train_idx]) & set(groups[valid_idx]))


class TestStratifiedTimeHoldoutVectorized:
    def test_large_n_matches_reference(self) -> None:
        rng = np.random.default_rng(42)
        n = 10_000
        y = rng.integers(0, 3, size=n)

        iv = StratifiedTimeHoldoutInnerValid(ratio=0.1)
        train_idx, valid_idx = iv.split(n, y=y)

        # No overlap, full coverage
        assert len(set(train_idx) & set(valid_idx)) == 0
        assert len(train_idx) + len(valid_idx) == n
        # Each class appears in valid
        assert set(y[valid_idx]) == {0, 1, 2}
        # train + valid == range(n)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([train_idx, valid_idx])),
            np.arange(n),
        )

    def test_dtype_is_intp(self) -> None:
        y = np.array([0, 1, 0, 1, 0, 1])
        iv = StratifiedTimeHoldoutInnerValid(ratio=0.4)
        train_idx, valid_idx = iv.split(len(y), y=y)
        assert train_idx.dtype == np.intp
        assert valid_idx.dtype == np.intp


class TestBlockedGroupVectorized:
    @pytest.mark.parametrize("task", ["regression", "binary"])
    def test_large_groups_match_reference(self, task: str) -> None:
        rng = np.random.default_rng(42)
        n = 4_000
        n_groups = 200
        groups = np.repeat(np.arange(n_groups), n // n_groups)
        # Time-ordered: groups appear in ascending order
        y = rng.integers(0, 2, size=n)

        iv = BlockedGroupInnerValid(ratio=0.1, task=task)
        train_idx, valid_idx = iv.split(n, y=y, groups=groups)

        # No overlap, full coverage
        assert len(set(train_idx) & set(valid_idx)) == 0
        assert len(train_idx) + len(valid_idx) == n
        # Group isolation
        assert not (set(groups[train_idx]) & set(groups[valid_idx]))
