"""Regression tests: inner-valid splitters must reject empty train sets (#8, #9)."""

from __future__ import annotations

import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.training.inner_valid import HoldoutInnerValid, TimeHoldoutInnerValid


class TestHoldoutInnerValidEmptyTrain:
    def test_n_samples_1_raises(self) -> None:
        with pytest.raises((ValueError, LizyMLError)):
            HoldoutInnerValid(ratio=0.1).split(1)

    def test_high_ratio_small_n_raises(self) -> None:
        with pytest.raises((ValueError, LizyMLError)):
            HoldoutInnerValid(ratio=0.9).split(5)

    def test_normal_case_works(self) -> None:
        train, valid = HoldoutInnerValid(ratio=0.2).split(100)
        assert len(train) > 0
        assert len(valid) > 0
        assert len(train) + len(valid) == 100


class TestTimeHoldoutInnerValidEmptyTrain:
    def test_n_samples_1_raises(self) -> None:
        with pytest.raises((ValueError, LizyMLError)):
            TimeHoldoutInnerValid(ratio=0.1).split(1)

    def test_normal_case_works(self) -> None:
        train, valid = TimeHoldoutInnerValid(ratio=0.1).split(100)
        assert len(train) > 0
        assert len(valid) > 0
        assert len(train) + len(valid) == 100
