"""purge_gap propagation into the auto-resolved inner valid (#212).

H-0085 decision (b): ``purge_gap`` (and ``gap`` for ``time_series``)
propagates into the auto-resolved inner-valid split so the
early-stopping boundary gets the same look-ahead guard as the outer split.
Previously ``TimeHoldoutInnerValid`` placed inner-valid directly adjacent to
inner-train (zero gap), leaking look-ahead-constructed targets at the boundary
and biasing ``best_iteration`` for every fold.

These tests fail closed on a regression to the zero-gap inner boundary.
H-0115 merged the deprecated ``embargo`` into ``purge_gap``, so the
propagated gap is ``purge_gap`` alone and still includes any ``embargo``.
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.config.loader import load_config
from lizyml.core._model_factories import build_inner_valid
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.training.inner_valid import TimeHoldoutInnerValid


def _gap_between(train_idx: np.ndarray, valid_idx: np.ndarray) -> int:
    """Number of purged rows between inner-train end and inner-valid start."""
    return int(valid_idx.min() - train_idx.max() - 1)


class TestTimeHoldoutGap:
    def test_gap_purges_boundary_rows(self) -> None:
        iv = TimeHoldoutInnerValid(ratio=0.1, gap=5)
        train_idx, valid_idx = iv.split(100)
        # n_valid = 10 → valid = [90..99]; gap=5 → train = [0..84]; [85..89] purged
        assert valid_idx.tolist() == list(range(90, 100))
        assert train_idx.max() == 84
        assert _gap_between(train_idx, valid_idx) == 5
        # purged rows belong to neither set
        assert set(range(85, 90)).isdisjoint(train_idx.tolist())
        assert set(range(85, 90)).isdisjoint(valid_idx.tolist())

    def test_zero_gap_is_contiguous(self) -> None:
        iv = TimeHoldoutInnerValid(ratio=0.1, gap=0)
        train_idx, valid_idx = iv.split(100)
        assert _gap_between(train_idx, valid_idx) == 0

    def test_gap_consuming_all_rows_raises(self) -> None:
        iv = TimeHoldoutInnerValid(ratio=0.5, gap=60)
        with pytest.raises(ValueError):
            iv.split(100)


class TestAutoResolvePropagatesGap:
    @staticmethod
    def _cfg(split: dict) -> object:
        raw = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": split,
            "training": {"early_stopping": {"enabled": True, "rounds": 10}},
        }
        return load_config(raw)

    def test_purged_time_series_propagates_purge_gap(self) -> None:
        cfg = self._cfg({"method": "purged_time_series", "purge_gap": 5})
        iv = build_inner_valid(cfg)
        assert isinstance(iv, TimeHoldoutInnerValid)
        train_idx, valid_idx = iv.split(100)
        assert _gap_between(train_idx, valid_idx) == 5

    def test_deprecated_embargo_is_propagated_inside_purge_gap(self) -> None:
        with pytest.warns(DeprecationWarning, match="purge_gap"):
            cfg = self._cfg(
                {"method": "purged_time_series", "purge_gap": 3, "embargo": 2}
            )
        iv = build_inner_valid(cfg)
        assert isinstance(iv, TimeHoldoutInnerValid)
        train_idx, valid_idx = iv.split(100)
        assert _gap_between(train_idx, valid_idx) == 5  # 3 + 2, merged (H-0115)

    def test_time_series_propagates_gap(self) -> None:
        cfg = self._cfg({"method": "time_series", "gap": 4})
        iv = build_inner_valid(cfg)
        assert isinstance(iv, TimeHoldoutInnerValid)
        train_idx, valid_idx = iv.split(100)
        assert _gap_between(train_idx, valid_idx) == 4

    def test_plain_time_series_no_gap(self) -> None:
        cfg = self._cfg({"method": "time_series"})
        iv = build_inner_valid(cfg)
        assert isinstance(iv, TimeHoldoutInnerValid)
        train_idx, valid_idx = iv.split(100)
        assert _gap_between(train_idx, valid_idx) == 0


class TestExplicitInnerValidDoesNotInheritGap:
    """An explicitly configured inner valid keeps its own settings (BLUEPRINT 10.3.3).

    Auto-resolution inherits the outer split's boundary gap; an explicit
    ``training.early_stopping.inner_valid`` does not, per 10.3.1's rule that an
    explicit spec does not consult the outer ``split.method``. The two paths
    therefore disagree on purpose for the same outer configuration, and that
    difference is what these tests pin -- it is the sentence 10.3.3 states, and
    nothing asserted it before.
    """

    @staticmethod
    def _cfg(split: dict, inner_valid: dict | None) -> object:
        early_stopping: dict = {"enabled": True, "rounds": 10}
        if inner_valid is not None:
            early_stopping["inner_valid"] = inner_valid
        raw = {
            "config_version": 1,
            "task": "regression",
            "data": {"target": "y"},
            "model": {"name": "lgbm"},
            "split": split,
            "training": {"early_stopping": early_stopping},
        }
        return load_config(raw)

    #: One outer configuration whose auto-resolved inner gap is non-zero.
    SPLIT = {"method": "purged_time_series", "purge_gap": 5}

    def test_explicit_time_holdout_gets_no_gap(self) -> None:
        cfg = self._cfg(self.SPLIT, {"method": "time_holdout", "ratio": 0.1})
        iv = build_inner_valid(cfg)
        assert isinstance(iv, TimeHoldoutInnerValid)
        assert iv.gap == 0
        train_idx, valid_idx = iv.split(100)
        assert _gap_between(train_idx, valid_idx) == 0

    def test_auto_and_explicit_differ_for_the_same_outer_split(self) -> None:
        auto = build_inner_valid(self._cfg(self.SPLIT, None))
        explicit = build_inner_valid(
            self._cfg(self.SPLIT, {"method": "time_holdout", "ratio": 0.1})
        )
        assert isinstance(auto, TimeHoldoutInnerValid)
        assert isinstance(explicit, TimeHoldoutInnerValid)
        assert auto.gap == 5
        assert explicit.gap == 0

    def test_gap_is_not_a_config_field(self) -> None:
        """``gap`` is resolver-only (H-0101): an explicit spec cannot carry one.

        BLUEPRINT 10.3.3 once said an explicit inner valid uses "the given gap,
        default 0" -- a value no Config can give. The disposition is that the
        explicit path always gets 0, and a ``gap`` key is refused rather than
        silently dropped.
        """
        with pytest.raises(LizyMLError) as exc:
            self._cfg(self.SPLIT, {"method": "time_holdout", "ratio": 0.1, "gap": 4})
        assert exc.value.code is ErrorCode.CONFIG_INVALID
        # The refusal is pydantic's extra-forbid on the gap key itself, not some
        # other invalid value in the spec.
        detail = repr(exc.value)
        assert "gap" in detail and "extra_forbidden" in detail, detail
