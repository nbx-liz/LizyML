"""What each calibrator accepts in ``calibration.params`` (H-0100, #277).

``platt`` and ``beta`` took ``params`` and threw it away. They now declare the
names they accept, the shapes those values must have, and which optimisation
methods can honour them, and refuse the rest with ``CONFIG_INVALID`` naming
``calibration.params``. The facade calls these declarations before any training
(see ``tests/test_core/test_calibration_params_reach.py``); this file tests the
declarations themselves and the optimiser settings they resolve to.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from lizyml.calibration._optimizer import resolve_minimize_kwargs
from lizyml.calibration.beta import BetaCalibrator
from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError

LBFGSB_DEFAULTS = {"L-BFGS-B": {"gtol": 1e-6, "ftol": 64 * np.finfo(float).eps}}


def _refused(cls: type, params: dict[str, Any]) -> LizyMLError:
    with pytest.raises(LizyMLError) as caught:
        cls.validate_params(params)
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert "calibration.params" in caught.value.user_message
    return caught.value


# ---------------------------------------------------------------------------
# Accepted surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"target_smoothing": False},
        {"x0": [0.0, 0.0]},
        {"method": "TNC", "bounds": [[None, None], [-3.0, 3.0]]},
        {"method": "BFGS", "tol": 1e-8},
        {"options": {"maxiter": 50}},
    ],
)
def test_platt_accepts_its_surface(params: dict[str, Any]) -> None:
    PlattCalibrator.validate_params(params)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"x0": [1.0, 1.0, 0.0]},
        {"bounds": [[0.0, None], [0.0, None], [None, None]]},
        {"method": "Powell", "tol": 1e-6},
        {"options": {"maxiter": 100}},
    ],
)
def test_beta_accepts_its_surface(params: dict[str, Any]) -> None:
    BetaCalibrator.validate_params(params)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params,reason",
    [
        ({"C": 0.001}, "LogisticRegression names are not Platt's surface"),
        ({"not_a_real_option": 123}, "unknown name"),
        ({"x0": [0.0]}, "x0 must have one value per coefficient"),
        ({"bounds": [[None, None]]}, "bounds must have one pair per coefficient"),
        (
            {"bounds": [(None, None), (None, None)]},
            "a bound pair is a list, not a tuple",
        ),
        ({"method": "Newton-CG"}, "method outside the table"),
        ({"method": "BFGS", "bounds": [[0, 1], [0, 1]]}, "BFGS cannot honour bounds"),
        ({"options": {"not_an_option": 1}}, "scipy does not know this option"),
        ({"target_smoothing": "yes"}, "target_smoothing is a bool"),
    ],
)
def test_platt_refuses(params: dict[str, Any], reason: str) -> None:
    _refused(PlattCalibrator, params)


@pytest.mark.parametrize(
    "params,reason",
    [
        ({"target_smoothing": False}, "not part of beta's approved surface"),
        ({"x0": [1.0, 1.0]}, "beta has three coefficients"),
        ({"bounds": [[0, None], [0, None]]}, "three pairs"),
        (
            {"method": "CG", "bounds": [[0, 1], [0, 1], [0, 1]]},
            "CG cannot honour bounds",
        ),
        ({"fun": "x"}, "callables are outside the approved surface"),
    ],
)
def test_beta_refuses(params: dict[str, Any], reason: str) -> None:
    _refused(BetaCalibrator, params)


# ---------------------------------------------------------------------------
# tol, options and method-specific defaults
# ---------------------------------------------------------------------------


def test_defaults_apply_when_nothing_is_written() -> None:
    kw = resolve_minimize_kwargs(
        {}, n_coef=2, default_x0=[0.0, 0.0], default_options=LBFGSB_DEFAULTS
    )
    assert kw["method"] == "L-BFGS-B"
    assert kw["options"] == LBFGSB_DEFAULTS["L-BFGS-B"]
    assert "tol" not in kw


def test_a_written_tol_is_not_defeated_by_the_default_options() -> None:
    """scipy passes ``tol`` through ``options.setdefault``; a default key would win."""
    kw = resolve_minimize_kwargs(
        {"tol": 1e-3}, n_coef=2, default_x0=[0.0, 0.0], default_options=LBFGSB_DEFAULTS
    )
    assert kw["tol"] == 1e-3
    assert "gtol" not in kw["options"] and "ftol" not in kw["options"]


def test_written_options_beat_written_tol_key_by_key() -> None:
    kw = resolve_minimize_kwargs(
        {"tol": 1e-3, "options": {"gtol": 1e-9}},
        n_coef=2,
        default_x0=[0.0, 0.0],
        default_options=LBFGSB_DEFAULTS,
    )
    assert kw["options"] == {"gtol": 1e-9}
    assert kw["tol"] == 1e-3


def test_written_options_merge_over_the_defaults() -> None:
    kw = resolve_minimize_kwargs(
        {"options": {"maxiter": 5}},
        n_coef=2,
        default_x0=[0.0, 0.0],
        default_options=LBFGSB_DEFAULTS,
    )
    assert kw["options"] == {**LBFGSB_DEFAULTS["L-BFGS-B"], "maxiter": 5}


def test_defaults_belong_to_their_method() -> None:
    """L-BFGS-B's ``ftol`` is an unknown option to BFGS and would warn."""
    kw = resolve_minimize_kwargs(
        {"method": "BFGS"},
        n_coef=2,
        default_x0=[0.0, 0.0],
        default_options=LBFGSB_DEFAULTS,
    )
    assert kw["options"] == {}


# ---------------------------------------------------------------------------
# Every method in the table fits, and the parameters take effect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["L-BFGS-B", "TNC", "SLSQP", "trust-constr", "Powell", "Nelder-Mead", "BFGS", "CG"],
)
def test_every_method_in_the_table_fits_platt(method: str) -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(0, 2, 600)
    y = (rng.random(600) < expit(0.5 * s + 1.0)).astype(float)
    reference = PlattCalibrator().fit(s, y).export_params()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = PlattCalibrator({"method": method}).fit(s, y).export_params()
    assert got["a"] == pytest.approx(reference["a"], abs=5e-3)
    assert got["b"] == pytest.approx(reference["b"], abs=5e-3)


def test_beta_bounds_take_effect() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400).astype(float)
    logits = y * 2.0 - 1.0 + rng.normal(0, 0.8, 400)

    free = BetaCalibrator().fit(logits, y).export_params()
    bounded = BetaCalibrator({"bounds": [[0.0, 0.1], [None, None], [None, None]]}).fit(
        logits, y
    )
    a = bounded.export_params()["a"]

    assert free["a"] > 0.1
    assert 0.0 <= a <= 0.1 + 1e-9


def test_beta_default_fit_emits_no_warning() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300).astype(float)
    logits = y * 2.0 - 1.0 + rng.normal(0, 0.5, 300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BetaCalibrator().fit(logits, y)
