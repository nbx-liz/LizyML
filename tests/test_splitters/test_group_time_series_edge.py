"""GroupTimeSeriesSplitter regressions + edge cases.

- Trailing groups must not be silently dropped (non-round n_groups / n_splits).
- gap must be respected; train/valid groups disjoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.splitters import GroupTimeSeriesSplitter


class TestGroupTimeSeriesTrailingGroups:
    def test_all_groups_covered_non_round(self) -> None:
        groups = np.repeat(np.arange(11), 5)
        folds = list(
            GroupTimeSeriesSplitter(n_splits=3).split(len(groups), groups=groups)
        )
        covered: set[int] = set()
        for train_idx, valid_idx in folds:
            covered.update(groups[valid_idx].tolist())
            covered.update(groups[train_idx].tolist())
        assert covered == set(range(11))

    def test_last_fold_extends_to_end(self) -> None:
        groups = np.repeat(np.arange(7), 3)
        folds = list(
            GroupTimeSeriesSplitter(n_splits=3).split(len(groups), groups=groups)
        )
        _, last_valid_idx = folds[-1]
        assert 6 in set(groups[last_valid_idx].tolist())

    def test_no_leakage_between_folds(self) -> None:
        groups = np.repeat(np.arange(10), 4)
        for train_idx, valid_idx in GroupTimeSeriesSplitter(n_splits=3).split(
            len(groups), groups=groups
        ):
            assert len(set(groups[train_idx]) & set(groups[valid_idx])) == 0

    def test_gap_respected_with_trailing(self) -> None:
        groups = np.repeat(np.arange(10), 3)
        for train_idx, valid_idx in GroupTimeSeriesSplitter(n_splits=2, gap=1).split(
            len(groups), groups=groups
        ):
            train_groups = set(groups[train_idx])
            valid_groups = set(groups[valid_idx])
            if train_groups and valid_groups:
                assert max(train_groups) + 1 < min(valid_groups)


class TestGroupTimeSeriesEdge:
    def test_negative_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="gap must be >= 0"):
            GroupTimeSeriesSplitter(gap=-1)

    def test_large_gap_skips_folds(self) -> None:
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        folds = list(
            GroupTimeSeriesSplitter(n_splits=3, gap=3).split(len(groups), groups=groups)
        )
        assert len(folds) < 3

    def test_valid_end_clamp(self) -> None:
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        for train, valid in GroupTimeSeriesSplitter(n_splits=3, gap=0).split(
            len(groups), groups=groups
        ):
            assert len(train) > 0
            assert len(valid) > 0
