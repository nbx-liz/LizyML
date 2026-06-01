"""Seed propagation invariants (H-0080, issue #169).

`training.seed` must reach the outer splitter and the isotonic calibrator.
`split.random_state` is a sentinel: ``None`` (the default) inherits
``training.seed``; an explicit value overrides it. Because ``training.seed``
also defaults to 42, fully-default configs reproduce the historical seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lizyml import Model
from lizyml.config.loader import load_config
from lizyml.core._model_factories import build_splitter
from tests._helpers import make_binary_df, make_config


def _outer_valid_indices(
    cfg_dict: dict[str, Any], n: int = 60
) -> list[tuple[int, ...]]:
    splitter = build_splitter(load_config(cfg_dict))
    return [tuple(int(i) for i in valid) for _, valid in splitter.split(n)]


class TestBuildSplitterSeedResolution:
    def test_default_inherits_training_seed_backward_compat(self) -> None:
        # split.random_state=None + training.seed=42 must reproduce the folds
        # of an explicit random_state=42 (historical behavior).
        inherited = _outer_valid_indices(
            make_config("regression", seed=42, split_overrides={"random_state": None})
        )
        explicit_42 = _outer_valid_indices(
            make_config("regression", seed=999, split_overrides={"random_state": 42})
        )
        assert inherited == explicit_42

    def test_training_seed_propagates_to_splitter(self) -> None:
        a = _outer_valid_indices(
            make_config("regression", seed=42, split_overrides={"random_state": None})
        )
        b = _outer_valid_indices(
            make_config("regression", seed=123, split_overrides={"random_state": None})
        )
        # Pre-fix these were identical (both used random_state=42 regardless).
        assert a != b

    def test_explicit_random_state_overrides_training_seed(self) -> None:
        a = _outer_valid_indices(
            make_config("regression", seed=123, split_overrides={"random_state": 7})
        )
        b = _outer_valid_indices(
            make_config("regression", seed=999, split_overrides={"random_state": 7})
        )
        assert a == b


class TestSeedSensitivityE2E:
    def test_oof_splits_differ_by_training_seed(self) -> None:
        df = make_binary_df()
        r1 = Model(
            make_config(
                "binary",
                seed=42,
                split_overrides={"random_state": None},
                n_estimators=10,
            )
        ).fit(data=df)
        r2 = Model(
            make_config(
                "binary",
                seed=123,
                split_overrides={"random_state": None},
                n_estimators=10,
            )
        ).fit(data=df)
        v1 = [tuple(int(i) for i in v) for _, v in r1.splits.outer]
        v2 = [tuple(int(i) for i in v) for _, v in r2.splits.outer]
        assert v1 != v2

    def test_same_training_seed_identical_oof(self) -> None:
        df = make_binary_df()
        r1 = Model(
            make_config(
                "binary",
                seed=42,
                split_overrides={"random_state": None},
                n_estimators=10,
            )
        ).fit(data=df)
        r2 = Model(
            make_config(
                "binary",
                seed=42,
                split_overrides={"random_state": None},
                n_estimators=10,
            )
        ).fit(data=df)
        np.testing.assert_array_equal(r1.oof_pred, r2.oof_pred)


class TestIsotonicCalibratorInheritsSeed:
    """With folds fixed (explicit split seed) and the base-model seed held
    constant, the isotonic calibrator's internal seed is the only variable."""

    def _calibrated_oof(self, cal_params: dict[str, Any] | None) -> np.ndarray:
        df = make_binary_df()
        cfg = make_config(
            "binary",
            seed=7,  # base model + (explicit) folds identical across runs
            split_overrides={"random_state": 42},
            calibration="isotonic",
            calibration_params=cal_params,
            n_estimators=30,
        )
        result = Model(cfg).fit(data=df)
        return np.asarray(result.calibrator.calibrated_oof)

    def test_inherited_seed_equals_explicit_match(self) -> None:
        # No calibration seed -> inherits training.seed (7); explicit 7 must match.
        inherited = self._calibrated_oof(None)
        explicit_7 = self._calibrated_oof({"seed": 7})
        np.testing.assert_array_equal(inherited, explicit_7)

    def test_calibrator_seed_actually_matters(self) -> None:
        # Negative control: a different calibrator seed changes the output,
        # so the equivalence above is not vacuous.
        inherited = self._calibrated_oof(None)
        explicit_999 = self._calibrated_oof({"seed": 999})
        assert not np.array_equal(inherited, explicit_999)
