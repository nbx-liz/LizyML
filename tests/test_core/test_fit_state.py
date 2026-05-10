"""Tests for ``FitState`` and ``Model._get_fit_state()`` (#112, H-0074).

Phase 1 of the FitState rollout. Verifies:

- ``FitState`` is a frozen dataclass.
- ``Model._get_fit_state()`` returns a populated ``FitState`` after fit.
- The error path raises ``LizyMLError(MODEL_NOT_FIT)`` before fit.
- Mixin-style helpers can be unit-tested by constructing a ``FitState``
  with mock components — no full Model fit required.
"""

from __future__ import annotations

from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.fit_state import FitState
from tests._helpers import make_config, make_regression_df


class TestFitStateContract:
    def test_is_frozen(self) -> None:
        """Frozen dataclass — runtime mutation must fail."""
        df = make_regression_df(n=80)
        cfg = make_config("regression")
        model = Model(cfg, data=df)
        model.fit()
        state = model._get_fit_state()
        with pytest.raises((AttributeError, TypeError)):
            state.fit_result = None  # type: ignore[misc]

    def test_required_fields_populated_after_fit(self) -> None:
        df = make_regression_df(n=80)
        cfg = make_config("regression")
        model = Model(cfg, data=df)
        model.fit()
        state = model._get_fit_state()
        # Required fields
        assert state.cfg is model._cfg
        assert state.fit_result is not None
        assert state.provider is not None
        # Refit produced by default fit()
        assert state.refit_result is not None
        # Tuning was not called
        assert state.tuning_result is None

    def test_y_and_x_present_after_fit(self) -> None:
        df = make_regression_df(n=80)
        cfg = make_config("regression")
        model = Model(cfg, data=df)
        model.fit()
        state = model._get_fit_state()
        assert state.y is not None
        assert state.X is not None

    def test_metrics_populated_after_fit(self) -> None:
        df = make_regression_df(n=80)
        cfg = make_config("regression")
        model = Model(cfg, data=df)
        model.fit()
        state = model._get_fit_state()
        assert state.metrics is not None
        assert "raw" in state.metrics


class TestGetFitStatePreFit:
    def test_raises_before_fit(self) -> None:
        cfg = make_config("regression")
        model = Model(cfg)
        with pytest.raises(LizyMLError) as exc_info:
            model._get_fit_state()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT


class TestFitStateForMixinUnitTest:
    """Demonstrate that a downstream consumer can build a ``FitState`` with
    mocks and exercise Mixin-style logic without running a full Model fit.

    Phase 2 of H-0074 will migrate Mixin signatures to actually accept
    ``state: FitState``. This test pins down the data-only nature of the
    snapshot so the future migration is mechanical.
    """

    def test_mocked_state_is_constructible(self) -> None:
        from dataclasses import FrozenInstanceError

        # Mock minimal pieces for the dataclass — actual values are
        # consumer-specific; we only assert constructibility + immutability.
        class _Sentinel:
            pass

        state = FitState(
            cfg=_Sentinel(),  # type: ignore[arg-type]
            fit_result=_Sentinel(),  # type: ignore[arg-type]
            refit_result=None,
            tuning_result=None,
            provider=_Sentinel(),  # type: ignore[arg-type]
            metrics={"raw": {}},
            y=None,
            X=None,
            run_dir=None,
            output_dir=None,
        )
        assert state.metrics == {"raw": {}}
        with pytest.raises((AttributeError, FrozenInstanceError)):
            state.metrics = {"raw": {"new": 1}}  # type: ignore[misc, assignment]

    @pytest.mark.parametrize(
        "missing_field",
        ["cfg", "fit_result", "provider"],
    )
    def test_missing_required_field_raises_typerror(self, missing_field: str) -> None:
        """Required positional/keyword fields cannot default — guarantees
        Phase 2 mock builders catch missing pieces at construction time."""
        kwargs: dict[str, Any] = {
            "cfg": object(),
            "fit_result": object(),
            "refit_result": None,
            "tuning_result": None,
            "provider": object(),
            "metrics": None,
            "y": None,
            "X": None,
            "run_dir": None,
            "output_dir": None,
        }
        kwargs.pop(missing_field)
        with pytest.raises(TypeError):
            FitState(**kwargs)  # type: ignore[arg-type]
