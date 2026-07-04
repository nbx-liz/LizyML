"""Edge-case coverage for splitters/purged_time_series.py."""

from __future__ import annotations

import pytest

from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter


class TestPurgedTimeSeries:
    def test_negative_purge_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="purge_gap must be >= 0"):
            PurgedTimeSeriesSplitter(purge_gap=-1)

    def test_negative_embargo_raises(self) -> None:
        with pytest.raises(ValueError, match="embargo must be >= 0"):
            PurgedTimeSeriesSplitter(embargo=-1)

    def test_too_few_samples_raises(self) -> None:
        sp = PurgedTimeSeriesSplitter(n_splits=5)
        with pytest.raises(ValueError, match="too small"):
            list(sp.split(3))

    def test_large_purge_gap_skips_folds(self) -> None:
        sp = PurgedTimeSeriesSplitter(n_splits=3, purge_gap=100)
        assert list(sp.split(40)) == []
