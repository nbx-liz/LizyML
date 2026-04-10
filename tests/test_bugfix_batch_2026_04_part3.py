"""Regression tests for issues #8, #9 discovered 2026-04-10.

#8: HoldoutInnerValid can produce empty train set.
#9: TimeHoldoutInnerValid can produce empty train set.
"""

from __future__ import annotations

import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.training.inner_valid import HoldoutInnerValid, TimeHoldoutInnerValid


class TestHoldoutInnerValidEmptyTrain:
    """HoldoutInnerValid must raise when n_valid >= n_samples."""

    def test_n_samples_1_raises(self) -> None:
        """n_samples=1 with any ratio produces n_valid=1, train=0."""
        iv = HoldoutInnerValid(ratio=0.1)
        with pytest.raises((ValueError, LizyMLError)):
            iv.split(1)

    def test_high_ratio_small_n_raises(self) -> None:
        """n=5, ratio=0.9 -> ceil(4.5)=5 -> train=0."""
        iv = HoldoutInnerValid(ratio=0.9)
        with pytest.raises((ValueError, LizyMLError)):
            iv.split(5)

    def test_normal_case_works(self) -> None:
        """n=100, ratio=0.2 -> n_valid=20, train=80."""
        iv = HoldoutInnerValid(ratio=0.2)
        train, valid = iv.split(100)
        assert len(train) > 0
        assert len(valid) > 0
        assert len(train) + len(valid) == 100


class TestTimeHoldoutInnerValidEmptyTrain:
    """TimeHoldoutInnerValid must raise when n_valid >= n_samples."""

    def test_n_samples_1_raises(self) -> None:
        """n_samples=1 -> n_valid=max(1,0)=1 -> train=0."""
        iv = TimeHoldoutInnerValid(ratio=0.1)
        with pytest.raises((ValueError, LizyMLError)):
            iv.split(1)

    def test_normal_case_works(self) -> None:
        """n=100, ratio=0.1 -> n_valid=10, train=90."""
        iv = TimeHoldoutInnerValid(ratio=0.1)
        train, valid = iv.split(100)
        assert len(train) > 0
        assert len(valid) > 0
        assert len(train) + len(valid) == 100
