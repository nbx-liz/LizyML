"""``calibration.params`` reaches the calibrator that consumes it, or is refused first.

H-0100 (#277). Through the facade: the parameters change the fitted calibrator,
reach every cross-fit calibrator and ``C_final``, are refused before any Booster
or study trains -- in ``fit()`` and in ``tune()`` -- and are prepared per method:
values normalised for every calibrator, LightGBM alias canonicalisation only for
the LightGBM-backed one.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lizyml import Model
from lizyml.calibration import registry
from lizyml.core import _model_factories
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import CalibratorRegistry
from tests._helpers import make_binary_df, make_config
from tests._train_spy import record_lightgbm_calls

REPO = Path(__file__).resolve().parents[2]


def _fit(method: str, params: dict[str, Any] | None) -> Model:
    cfg = make_config(
        "binary",
        n_estimators=5,
        n_splits=3,
        calibration=method,
        calibration_params=params,
    )
    model = Model(cfg, data=make_binary_df(n=240, seed=4))
    model.fit()
    return model


def _c_final_export(model: Model) -> dict[str, Any]:
    return model.fit_result.calibrator.c_final.export_params()


# ---------------------------------------------------------------------------
# Effect
# ---------------------------------------------------------------------------


def test_platt_params_change_the_fitted_calibrator() -> None:
    default = _c_final_export(_fit("platt", None))
    unsmoothed = _c_final_export(_fit("platt", {"target_smoothing": False}))
    assert (default["a"], default["b"]) != pytest.approx(
        (unsmoothed["a"], unsmoothed["b"]), abs=1e-6
    )


def test_beta_params_change_the_fitted_calibrator() -> None:
    default = _c_final_export(_fit("beta", None))
    bounded = _c_final_export(
        _fit("beta", {"bounds": [[0.0, 0.05], [None, None], [None, None]]})
    )
    assert bounded["a"] <= 0.05 + 1e-9
    assert default["a"] != pytest.approx(bounded["a"], abs=1e-6)


# ---------------------------------------------------------------------------
# Reach: every cross-fit calibrator and C_final
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,params",
    [("platt", {"target_smoothing": False}), ("beta", {"options": {"maxiter": 200}})],
)
def test_every_calibrator_built_is_given_the_params(
    method: str, params: dict[str, Any]
) -> None:
    calls: list[tuple[str, Any]] = []
    real = registry.get_calibrator

    def spy(name: str, params: dict[str, Any] | None = None) -> Any:
        calls.append((name, params))
        return real(name, params=params)

    with mock.patch.object(registry, "get_calibrator", spy):
        _fit(method, params)

    # three outer folds, then C_final
    assert len(calls) == 4, calls
    assert all(call == (method, params) for call in calls), calls


# ---------------------------------------------------------------------------
# Preparation is per method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,params",
    [("platt", {"x0": [0.0, 0.0], "tol": 1e-7}), ("beta", {"tol": 1e-7})],
)
def test_platt_and_beta_are_not_given_lightgbm_canonicalisation(
    method: str, params: dict[str, Any]
) -> None:
    """Canonicalising by LightGBM's alias table must not run for these two.

    None of their accepted names happens to be a LightGBM alias today, so
    comparing names would pass whether or not canonicalisation ran. The claim is
    about what runs, so it is asserted on the call.
    """
    real = _model_factories.canonicalise_calibration_params
    with mock.patch.object(
        _model_factories, "canonicalise_calibration_params", side_effect=real
    ) as spy:
        _fit(method, params)
    spy.assert_not_called()


def test_isotonic_is_given_lightgbm_canonicalisation() -> None:
    real = _model_factories.canonicalise_calibration_params
    with mock.patch.object(
        _model_factories, "canonicalise_calibration_params", side_effect=real
    ) as spy:
        _fit("isotonic", {"eta": 0.05})
    spy.assert_called()


def test_isotonic_aliases_are_still_canonicalised() -> None:
    """H-0094 decision 8 must not regress while the other two stop being renamed."""
    calls: list[Any] = []
    real = registry.get_calibrator

    def spy(name: str, params: dict[str, Any] | None = None) -> Any:
        calls.append(params)
        return real(name, params=params)

    with mock.patch.object(registry, "get_calibrator", spy):
        _fit("isotonic", {"eta": 0.05})

    assert calls and all("learning_rate" in p and "eta" not in p for p in calls), calls


# ---------------------------------------------------------------------------
# Refusal before any training, at both entrances
# ---------------------------------------------------------------------------


REFUSED = [
    ("platt", {"C": 0.001}),
    ("platt", {"method": "BFGS", "bounds": [[0, 1], [0, 1]]}),
    ("beta", {"x0": [1.0, 1.0]}),
    ("beta", {"options": {"not_an_option": 1}}),
    # An unhashable method once escaped as TypeError instead of the refusal.
    ("platt", {"method": []}),
]


@pytest.mark.parametrize("method,params", REFUSED)
def test_fit_refuses_before_training(method: str, params: dict[str, Any]) -> None:
    cfg = make_config(
        "binary",
        n_estimators=5,
        n_splits=3,
        calibration=method,
        calibration_params=params,
    )
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        Model(cfg, data=make_binary_df(n=240, seed=4)).fit()

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert "calibration.params" in caught.value.user_message
    assert not seen["train_params"], "a Booster trained before the refusal"


@pytest.mark.parametrize("method,params", REFUSED)
def test_tune_refuses_before_any_study(method: str, params: dict[str, Any]) -> None:
    cfg = make_config(
        "binary",
        n_estimators=5,
        n_splits=3,
        calibration=method,
        calibration_params=params,
        tuning_n_trials=1,
    )
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        Model(cfg, data=make_binary_df(n=240, seed=4)).tune()

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert "calibration.params" in caught.value.user_message
    assert not seen["train_params"], "a trial trained before the refusal"


# ---------------------------------------------------------------------------
# Positions: every registered calibrator declares its contract and has a
# generated fitter
# ---------------------------------------------------------------------------


def _generated_fitters() -> set[str]:
    source = (REPO / "lizyml" / "codegen" / "templates.py").read_text(encoding="utf-8")
    marker = "_CAL_FITTERS = "
    line = next(ln for ln in source.splitlines() if ln.startswith(marker))
    table = ast.parse(line[len(marker) :], mode="eval").body
    assert isinstance(table, ast.Dict)
    return {k.value for k in table.keys if isinstance(k, ast.Constant)}


def test_every_registered_calibrator_declares_its_params_contract() -> None:
    # `.keys()` is this registry's own accessor returning a list, not a dict view.
    names = CalibratorRegistry.keys()
    for name in names:
        cls = CalibratorRegistry.get(name)
        assert "validate_params" in vars(cls), (
            f"calibrator {name!r} inherits validate_params instead of declaring "
            "what it accepts, so calibration.params for it is not checked"
        )


def test_every_registered_calibrator_has_a_generated_fitter() -> None:
    assert _generated_fitters() == set(CalibratorRegistry.keys())
