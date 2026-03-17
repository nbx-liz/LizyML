"""Unit tests for compute_oof_valid_mask (H-0057).

Tests the pure function that derives OOF coverage from split indices,
NOT from NaN detection. This ensures:
- KFold-style splits produce full coverage (all True).
- TimeSeriesCV-style splits produce partial coverage.
- Edge cases (empty, overlapping, zero samples) are handled.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from lizyml.training.oof_assembly import compute_oof_valid_mask


def _make_outer(
    pairs: list[tuple[list[int], list[int]]],
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    """Helper to build SplitIndices.outer-compatible list."""
    return [(np.array(t, dtype=np.intp), np.array(v, dtype=np.intp)) for t, v in pairs]


class TestComputeOofValidMask:
    """Tests for compute_oof_valid_mask."""

    def test_full_coverage_kfold(self) -> None:
        """KFold: every sample appears in exactly one valid fold -> all True."""
        # 3-fold KFold on 9 samples
        outer = _make_outer(
            [
                ([3, 4, 5, 6, 7, 8], [0, 1, 2]),
                ([0, 1, 2, 6, 7, 8], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
            ]
        )
        mask = compute_oof_valid_mask(outer, n_samples=9)
        assert mask.dtype == np.bool_
        assert mask.shape == (9,)
        assert mask.all()

    def test_partial_coverage_time_series(self) -> None:
        """TimeSeriesCV: first rows never in valid -> False for those rows."""
        # Simulates TimeSeriesSplit(n_splits=3) on 12 samples
        # fold 0: train=[0,1,2], valid=[3,4,5]
        # fold 1: train=[0..5],  valid=[6,7,8]
        # fold 2: train=[0..8],  valid=[9,10,11]
        # -> rows 0,1,2 never in valid
        outer = _make_outer(
            [
                ([0, 1, 2], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
                ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]),
            ]
        )
        mask = compute_oof_valid_mask(outer, n_samples=12)
        assert not mask[0] and not mask[1] and not mask[2]
        assert mask[3:].all()
        assert mask.sum() == 9

    def test_coverage_ratio(self) -> None:
        """Coverage ratio matches expected value."""
        outer = _make_outer(
            [
                ([0, 1, 2], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
                ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]),
            ]
        )
        mask = compute_oof_valid_mask(outer, n_samples=12)
        coverage = float(mask.sum()) / len(mask)
        assert coverage == pytest.approx(9 / 12)

    def test_empty_splits(self) -> None:
        """Empty outer list -> mask is all False."""
        mask = compute_oof_valid_mask([], n_samples=5)
        assert mask.shape == (5,)
        assert not mask.any()

    def test_single_fold(self) -> None:
        """Single fold covering a subset -> correct mask."""
        outer = _make_outer([([0, 1, 2], [3, 4])])
        mask = compute_oof_valid_mask(outer, n_samples=5)
        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_overlapping_valid_idx(self) -> None:
        """Overlapping valid indices -> union (no double-counting)."""
        outer = _make_outer(
            [
                ([2, 3, 4], [0, 1]),
                ([0, 3, 4], [1, 2]),
            ]
        )
        mask = compute_oof_valid_mask(outer, n_samples=5)
        # Union of {0,1} and {1,2} = {0,1,2}
        expected = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_n_samples_zero(self) -> None:
        """n_samples=0 -> returns empty boolean array."""
        mask = compute_oof_valid_mask([], n_samples=0)
        assert mask.shape == (0,)
        assert mask.dtype == np.bool_

    def test_return_type_is_bool_ndarray(self) -> None:
        """Verify dtype is np.bool_."""
        outer = _make_outer([([0], [1])])
        mask = compute_oof_valid_mask(outer, n_samples=2)
        assert mask.dtype == np.bool_
        assert isinstance(mask, np.ndarray)
