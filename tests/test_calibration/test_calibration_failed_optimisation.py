"""A calibration whose optimiser did not converge is refused (H-0113, #297).

``PlattCalibrator.fit`` / ``BetaCalibrator.fit`` used ``minimize``'s ``x``
without looking at ``success``, so a failed optimisation shipped its
intermediate (or initial) coefficients as a fitted calibrator. The failures
here are real: ``{"options": {"maxiter": 1}}`` makes L-BFGS-B stop with
``success=False`` for both calibrators on this data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from lizyml import Model
from lizyml.calibration.beta import BetaCalibrator
from lizyml.calibration.cross_fit import cross_fit_calibrate
from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_binary_df, make_config

_STOP_EARLY = {"options": {"maxiter": 1}}
_CALIBRATORS = {"platt": PlattCalibrator, "beta": BetaCalibrator}


def _scores(seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 2.0, n)
    y = (rng.random(n) < expit(0.6 * s - 0.5)).astype(float)
    return s, y


@pytest.mark.parametrize("name", sorted(_CALIBRATORS))
def test_failed_minimize_raises_and_leaves_no_coefficients(name: str) -> None:
    s, y = _scores()
    calibrator = _CALIBRATORS[name](_STOP_EARLY)
    with pytest.raises(LizyMLError) as exc:
        calibrator.fit(s, y)
    assert exc.value.code == ErrorCode.CALIBRATION_FAILED
    context = exc.value.context
    assert context["calibrator"] == name
    assert context["method"] == "L-BFGS-B"
    assert "ITERATIONS" in context["message"]
    assert isinstance(context["status"], int)
    assert context["nit"] == 1
    with pytest.raises(LizyMLError) as not_fitted:
        calibrator.predict(s)
    assert not_fitted.value.code == ErrorCode.CALIBRATION_NOT_FITTED


@pytest.mark.parametrize("name", sorted(_CALIBRATORS))
def test_failed_refit_does_not_keep_the_earlier_fit(name: str) -> None:
    """A success followed by a failed refit leaves the calibrator unfitted (review r1).

    The first fit's data make the start point the optimum, so ``maxiter=1``
    converges; the refit on informative scores needs more iterations and fails.
    """
    calibrator = _CALIBRATORS[name](_STOP_EARLY)
    s, y = _scores()
    if name == "platt":
        # x0 = [0, prior log-odds]: constant scores, balanced labels.
        calibrator.fit(np.zeros(400), (np.arange(400) % 2).astype(float))
    else:
        # x0 = [1, 1, 0]: soft labels equal to the start point's prediction.
        p = np.clip(expit(s), 1e-10, 1 - 1e-10)
        calibrator.fit(s, expit(np.log(p) + np.log(1 - p)))
    calibrator.export_params()  # the first fit succeeded
    with pytest.raises(LizyMLError) as exc:
        calibrator.fit(s, y)
    assert exc.value.code == ErrorCode.CALIBRATION_FAILED
    for call in (lambda: calibrator.predict(s), calibrator.export_params):
        with pytest.raises(LizyMLError) as not_fitted:
            call()
        assert not_fitted.value.code == ErrorCode.CALIBRATION_NOT_FITTED


@pytest.mark.parametrize("name", sorted(_CALIBRATORS))
def test_refit_failing_before_minimize_does_not_keep_the_earlier_fit(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refit that fails before ``minimize`` (scipy missing) also clears the fit.

    Review round 2: the reset ran after the scipy import / check, so a refit
    that failed there left the earlier coefficients usable.
    """
    import sys

    from lizyml.calibration import beta as beta_module

    s, y = _scores()
    calibrator = _CALIBRATORS[name]().fit(s, y)
    calibrator.export_params()  # the first fit succeeded
    if name == "platt":
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)
        expected: type[BaseException] = ImportError
    else:
        monkeypatch.setattr(beta_module, "_scipy", None)
        expected = LizyMLError
    with pytest.raises(expected):
        calibrator.fit(s, y)
    monkeypatch.undo()
    for call in (lambda: calibrator.predict(s), calibrator.export_params):
        with pytest.raises(LizyMLError) as not_fitted:
            call()
        assert not_fitted.value.code == ErrorCode.CALIBRATION_NOT_FITTED


@pytest.mark.parametrize("name", sorted(_CALIBRATORS))
def test_default_settings_still_fit(name: str) -> None:
    s, y = _scores()
    probs = _CALIBRATORS[name]().fit(s, y).predict(s)
    assert np.all((probs > 0) & (probs < 1))


def _splits(n: int, k: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    folds = np.array_split(idx, k)
    return [(np.setdiff1d(idx, f), f) for f in folds]


def _factory_failing_on(call: int) -> Any:
    calls = {"n": 0}

    def factory() -> PlattCalibrator:
        calls["n"] += 1
        return PlattCalibrator(_STOP_EARLY if calls["n"] == call else None)

    return factory


def test_cross_fit_names_the_failing_fold() -> None:
    s, y = _scores()
    with pytest.raises(LizyMLError) as exc:
        cross_fit_calibrate(s, y, _factory_failing_on(2), split_indices=_splits(len(s)))
    assert exc.value.code == ErrorCode.CALIBRATION_FAILED
    assert exc.value.context["stage"] == "cross_fit"
    assert exc.value.context["fold"] == 1
    assert exc.value.context["calibrator"] == "platt"
    assert isinstance(exc.value.cause, LizyMLError)


def test_cross_fit_names_c_final() -> None:
    s, y = _scores()
    splits = _splits(len(s))
    with pytest.raises(LizyMLError) as exc:
        cross_fit_calibrate(
            s, y, _factory_failing_on(len(splits) + 1), split_indices=splits
        )
    assert exc.value.code == ErrorCode.CALIBRATION_FAILED
    assert exc.value.context["stage"] == "c_final"
    assert "fold" not in exc.value.context


@pytest.mark.parametrize("method", ["platt", "beta"])
def test_model_fit_refuses_a_non_converged_calibration(method: str) -> None:
    cfg = make_config(
        "binary",
        n_estimators=5,
        n_splits=3,
        calibration=method,
        calibration_params=_STOP_EARLY,
    )
    with pytest.raises(LizyMLError) as exc:
        Model(cfg, data=make_binary_df(n=240, seed=4)).fit()
    assert exc.value.code == ErrorCode.CALIBRATION_FAILED
    assert exc.value.context["stage"] == "cross_fit"
    assert exc.value.context["fold"] == 0
