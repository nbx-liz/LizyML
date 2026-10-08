"""Platt scaling fitted as Platt defined it (H-0100, #277).

``PlattCalibrator`` used ``LogisticRegression(C=1.0)``: an L2 penalty and 0/1
targets, neither of which is Platt's method. Platt (1999) fits the slope and the
intercept of ``1 / (1 + exp(A f + B))`` jointly by maximum likelihood, with
smoothed targets and no penalty. scikit-learn implements exactly that in
``sklearn.calibration._sigmoid_calibration``, which these tests use as the
reference -- in tests only; production code does not import a private function.

The exported form stays ``sigmoid(a * s + b)``, so ``a = -A`` and ``b = -B``.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from scipy.special import expit
from sklearn.calibration import _sigmoid_calibration
from sklearn.linear_model import LogisticRegression

from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError


def _scores(
    n: int, slope: float, offset: float, scale: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores whose true log-odds are ``slope * s + offset``."""
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, scale, size=n)
    y = (rng.random(n) < expit(slope * s + offset)).astype(float)
    return s, y


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# The default is Platt's method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slope,offset,n,scale",
    [
        (0.5, 1.0, 2000, 2.0),
        (0.5, 1.0, 100, 2.0),
        (2.0, -1.5, 2000, 2.0),
        (0.3, 0.5, 2000, 40.0),
    ],
)
def test_default_matches_platt_as_sklearn_implements_it(
    slope: float, offset: float, n: int, scale: float
) -> None:
    s, y = _scores(n, slope, offset, scale)
    cal = PlattCalibrator().fit(s, y)
    exported = cal.export_params()

    big_a, big_b = _sigmoid_calibration(s, y)
    assert exported["a"] == pytest.approx(-big_a, abs=1e-3)
    assert exported["b"] == pytest.approx(-big_b, abs=1e-3)


def test_the_intercept_is_estimated_and_it_matters() -> None:
    """A score whose zero is not probability one half needs the intercept.

    Fixing ``b`` at zero through ``bounds`` is the only way to remove it now, and
    doing so on offset data must fit worse -- which is what the intercept is for.
    """
    s, y = _scores(2000, slope=0.5, offset=1.0)
    free = PlattCalibrator().fit(s, y)
    pinned = PlattCalibrator({"bounds": [[None, None], [0.0, 0.0]]}).fit(s, y)

    assert abs(free.export_params()["b"]) > 0.5
    assert pinned.export_params()["b"] == pytest.approx(0.0, abs=1e-12)
    assert _log_loss(free.predict(s), y) < _log_loss(pinned.predict(s), y) - 0.05


def test_target_smoothing_can_be_turned_off_and_changes_the_fit() -> None:
    s, y = _scores(40, slope=1.0, offset=0.3, seed=3)
    smoothed = PlattCalibrator().fit(s, y).export_params()
    plain = PlattCalibrator({"target_smoothing": False}).fit(s, y).export_params()

    assert (smoothed["a"], smoothed["b"]) != pytest.approx(
        (plain["a"], plain["b"]), abs=1e-4
    )


def test_default_fit_emits_no_warning() -> None:
    s, y = _scores(500, slope=0.7, offset=-0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PlattCalibrator().fit(s, y)


# ---------------------------------------------------------------------------
# Output contract (unchanged)
# ---------------------------------------------------------------------------


def test_export_form_and_predict_agree() -> None:
    s, y = _scores(500, slope=0.7, offset=-0.4)
    cal = PlattCalibrator().fit(s, y)
    exported = cal.export_params()

    assert set(exported) == {"method", "a", "b"}
    assert exported["method"] == "platt"
    np.testing.assert_allclose(cal.predict(s), expit(exported["a"] * s + exported["b"]))


def test_predict_before_fit_raises() -> None:
    with pytest.raises(LizyMLError) as caught:
        PlattCalibrator().predict(np.array([0.0]))
    assert caught.value.code is ErrorCode.CALIBRATION_NOT_FITTED


# ---------------------------------------------------------------------------
# Large scores are rescaled without changing the problem
# ---------------------------------------------------------------------------


def test_bounds_on_the_slope_hold_in_the_written_coordinates_for_large_scores() -> None:
    """Rescaling divides the scores by ``k``; the slope bound must follow.

    A bound applied to the rescaled slope without multiplying it by ``k`` would
    let the returned slope land ``k`` times outside what the user wrote.
    """
    s, y = _scores(2000, slope=0.05, offset=0.2, scale=60.0)
    assert np.max(np.abs(s)) >= 30

    cal = PlattCalibrator({"x0": [0.015, 0.0], "bounds": [[0.01, 0.02], [None, None]]})
    a = cal.fit(s, y).export_params()["a"]

    assert 0.01 - 1e-9 <= a <= 0.02 + 1e-9
    assert a == pytest.approx(0.02, abs=1e-6)  # the unconstrained slope is 0.05


def test_rescaling_is_the_same_problem() -> None:
    """Halving every score and doubling the slope's start and bounds is the same fit."""
    s, y = _scores(2000, slope=0.03, offset=-0.3, scale=80.0, seed=5)
    one = PlattCalibrator({"x0": [0.01, 0.1], "bounds": [[0.0, 0.025], [-2, 2]]})
    two = PlattCalibrator({"x0": [0.02, 0.1], "bounds": [[0.0, 0.05], [-2, 2]]})

    a1, b1 = (one.fit(s, y).export_params()[k] for k in ("a", "b"))
    a2, b2 = (two.fit(s / 2.0, y).export_params()[k] for k in ("a", "b"))

    assert a1 == pytest.approx(a2 / 2.0, abs=1e-5)
    assert b1 == pytest.approx(b2, abs=1e-4)


# ---------------------------------------------------------------------------
# Artifacts saved before H-0100 still predict
# ---------------------------------------------------------------------------


def _legacy_calibrator(model: LogisticRegression | None) -> PlattCalibrator:
    """An instance carrying the pre-H-0100 state, as an old pickle would restore it."""
    legacy = object.__new__(PlattCalibrator)
    legacy.__dict__.update({"_model": model})
    return legacy


def test_a_legacy_calibrator_predicts_as_it_did() -> None:
    s, y = _scores(500, slope=0.7, offset=-0.4)
    old = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200).fit(
        s.reshape(-1, 1), y
    )
    restored = pickle.loads(pickle.dumps(_legacy_calibrator(old)))

    probes = np.concatenate([s, [-1e3, -50.0, -30.0, 0.0, 30.0, 50.0, 1e3]])
    expected = old.predict_proba(probes.reshape(-1, 1))[:, 1]
    np.testing.assert_allclose(restored.predict(probes), expected, rtol=0, atol=1e-12)
    assert restored.export_params() == {
        "method": "platt",
        "a": float(old.coef_[0, 0]),
        "b": float(old.intercept_[0]),
    }


def test_an_unfitted_legacy_calibrator_stays_unfitted() -> None:
    restored = pickle.loads(pickle.dumps(_legacy_calibrator(None)))
    with pytest.raises(LizyMLError) as caught:
        restored.predict(np.array([0.0]))
    assert caught.value.code is ErrorCode.CALIBRATION_NOT_FITTED


def test_a_current_calibrator_survives_a_pickle_round_trip() -> None:
    s, y = _scores(300, slope=1.2, offset=0.1)
    cal = PlattCalibrator({"target_smoothing": False}).fit(s, y)
    restored = pickle.loads(pickle.dumps(cal))

    np.testing.assert_array_equal(restored.predict(s), cal.predict(s))
    assert restored.export_params() == cal.export_params()
