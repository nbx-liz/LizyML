"""#298: the PR 3c (H-0100) acceptance rows, each with a test that fails with it.

The criteria table is ``docs/audits/2026-09-defect-discovery/results/
pr3c_acceptance_criteria.md`` §2. Each section names its row. Where the
earlier test inspected a helper, the tests here observe the call the claim is
about: the keyword arguments ``scipy.optimize.minimize`` and ``lightgbm.train``
actually receive, the ``config.json`` ``export_code`` writes, and a real
artifact written by LizyML 0.17.1, the last release before H-0100.

The ``minimize`` spy keeps only the fit's own solve. That solve passes ``x0``
by keyword. The options probe that validation runs passes ``x0``
positionally (``_optimizer._check_options``), so it is left out.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import lightgbm
import numpy as np
import pytest
import scipy.optimize

import lizyml.calibration.platt as platt_module
from lizyml import Model
from lizyml.calibration.beta import BetaCalibrator
from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_binary_df, make_config

pytest.importorskip("scipy")

FIXTURE = (
    Path(__file__).resolve().parents[1] / "test_persistence" / "data" / "v0_17_1_platt"
)


class _Spy:
    """Record the keyword arguments of every call, then run the real function."""

    def __init__(self, real: Any) -> None:
        self.real = real
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, dict(kwargs)))
        return self.real(*args, **kwargs)


@pytest.fixture
def minimize_spy() -> Iterator[_Spy]:
    spy = _Spy(scipy.optimize.minimize)
    with mock.patch("scipy.optimize.minimize", spy):
        yield spy


def _solves(spy: _Spy) -> list[dict[str, Any]]:
    """The fits' own solves (``x0`` by keyword), not the options probes."""
    return [kwargs for _, kwargs in spy.calls if "x0" in kwargs]


def _scores(
    n: int = 400, scale: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, scale, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.6 * s / scale * 2.0 - 0.3)))).astype(
        float
    )
    return s, y


def _fit_model(method: str, params: dict[str, Any] | None, seed: int = 0) -> Model:
    cfg = make_config(
        "binary",
        n_estimators=5,
        n_splits=3,
        calibration=method,
        calibration_params=params,
        seed=seed,
    )
    model = Model(cfg, data=make_binary_df(n=240, seed=4))
    model.fit()
    return model


# ---------------------------------------------------------------------------
# Row 3c: the params reach every cross-fit calibrator and C_final, in the fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["platt", "beta"])
def test_3c_every_fold_and_c_final_solve_with_the_written_options(
    method: str, minimize_spy: _Spy
) -> None:
    _fit_model(method, {"options": {"maxiter": 321}})
    solves = _solves(minimize_spy)
    # three outer folds, then C_final
    assert len(solves) == 4, solves
    assert all(kw["options"].get("maxiter") == 321 for kw in solves), solves


def test_3c_every_fold_and_c_final_train_isotonic_with_the_prepared_params() -> None:
    spy = _Spy(lightgbm.train)
    with mock.patch.object(lightgbm, "train", spy):
        _fit_model("isotonic", {"eta": 0.2, "num_boost_round": 5}, seed=13)
    # The model's own boosters go through lightgbm.train too; the calibrator's
    # are the ones with the monotone constraint it forces.
    calibrators = [(a, kw) for a, kw in spy.calls if "monotone_constraints" in a[0]]
    assert len(calibrators) == 4, len(calibrators)
    for args, kwargs in calibrators:
        params = args[0]
        assert params["learning_rate"] == 0.2 and "eta" not in params, params
        assert params["seed"] == 13, params
        assert kwargs["num_boost_round"] == 5, kwargs


# ---------------------------------------------------------------------------
# Rows 4a-4c: tol / options precedence and per-method defaults, as minimize
# receives them from PlattCalibrator.fit and BetaCalibrator.fit
# ---------------------------------------------------------------------------


#: Every method calibration.params accepts, and the options keys scipy fills
#: from ``tol`` for it. Written out here rather than read from the code under
#: test, so a change to either side shows up as a failure.
TOL_KEYS: dict[str, tuple[str, ...]] = {
    "L-BFGS-B": ("ftol", "gtol"),
    "TNC": ("ftol", "gtol", "xtol"),
    "SLSQP": ("ftol",),
    "trust-constr": ("barrier_tol", "gtol", "xtol"),
    "Powell": ("ftol", "xtol"),
    "Nelder-Mead": ("fatol", "xatol"),
    "BFGS": ("gtol",),
    "CG": ("gtol",),
}
#: The only per-method default options: Platt's, for L-BFGS-B (scikit-learn's).
PLATT_L_BFGS_B_DEFAULTS = {"gtol": 1e-6, "ftol": 64 * float(np.finfo(float).eps)}
CALIBRATORS = {"platt": PlattCalibrator, "beta": BetaCalibrator}


def _defaults(calibrator: str, method: str) -> dict[str, float]:
    if calibrator == "platt" and method == "L-BFGS-B":
        return dict(PLATT_L_BFGS_B_DEFAULTS)
    return {}


def _one_solve(calibrator: str, params: dict[str, Any], spy: _Spy) -> dict[str, Any]:
    """The kwargs of the one solve ``fit`` makes. What the fit concludes does
    not matter here; a method that does not converge on these scores still
    made the call, and the refusal it raises is the documented one."""
    s, y = _scores()
    try:
        CALIBRATORS[calibrator](params).fit(s, y)
    except LizyMLError as exc:
        assert exc.code == ErrorCode.CALIBRATION_FAILED, exc
    solves = _solves(spy)
    assert len(solves) == 1, solves
    return solves[0]


def test_4_the_tol_key_table_matches_the_accepted_methods() -> None:
    from lizyml.calibration._optimizer import _TOL_KEYS, METHODS

    assert set(TOL_KEYS) == set(METHODS)
    assert {m: tuple(sorted(k)) for m, k in _TOL_KEYS.items()} == {
        m: tuple(sorted(k)) for m, k in TOL_KEYS.items()
    }


@pytest.mark.parametrize("calibrator", sorted(CALIBRATORS))
@pytest.mark.parametrize("method", sorted(TOL_KEYS))
def test_4a_written_tol_is_not_defeated_by_the_default_options(
    calibrator: str, method: str, minimize_spy: _Spy
) -> None:
    kwargs = _one_solve(calibrator, {"method": method, "tol": 1e-7}, minimize_spy)
    assert kwargs["method"] == method and kwargs["tol"] == 1e-7
    # Every default is a key tol sets, so none of them may stay.
    expected = {
        k: v
        for k, v in _defaults(calibrator, method).items()
        if k not in TOL_KEYS[method]
    }
    assert kwargs["options"] == expected


@pytest.mark.parametrize("calibrator", sorted(CALIBRATORS))
@pytest.mark.parametrize("method", sorted(TOL_KEYS))
def test_4b_written_options_beat_written_tol_key_by_key(
    calibrator: str, method: str, minimize_spy: _Spy
) -> None:
    key = TOL_KEYS[method][0]
    kwargs = _one_solve(
        calibrator,
        {"method": method, "tol": 1e-7, "options": {key: 1e-9}},
        minimize_spy,
    )
    assert kwargs["tol"] == 1e-7
    assert kwargs["options"] == {key: 1e-9}


@pytest.mark.parametrize("calibrator", sorted(CALIBRATORS))
@pytest.mark.parametrize("method", ["L-BFGS-B", "BFGS"])
def test_4b_written_options_merge_over_the_defaults(
    calibrator: str, method: str, minimize_spy: _Spy
) -> None:
    kwargs = _one_solve(
        calibrator, {"method": method, "options": {"maxiter": 77}}, minimize_spy
    )
    assert "tol" not in kwargs
    assert kwargs["options"] == {**_defaults(calibrator, method), "maxiter": 77}


@pytest.mark.parametrize("calibrator", sorted(CALIBRATORS))
@pytest.mark.parametrize("method", sorted(TOL_KEYS))
def test_4c_defaults_belong_to_their_method(
    calibrator: str, method: str, minimize_spy: _Spy
) -> None:
    kwargs = _one_solve(calibrator, {"method": method}, minimize_spy)
    assert kwargs["method"] == method
    # L-BFGS-B's ftol is not handed to BFGS, or to any other method.
    assert kwargs["options"] == _defaults(calibrator, method)


def test_4c_the_unwritten_method_is_l_bfgs_b(minimize_spy: _Spy) -> None:
    for calibrator in sorted(CALIBRATORS):
        minimize_spy.calls.clear()
        kwargs = _one_solve(calibrator, {}, minimize_spy)
        assert kwargs["method"] == "L-BFGS-B"
        assert kwargs["options"] == _defaults(calibrator, "L-BFGS-B")


# ---------------------------------------------------------------------------
# Row 5b: rescaling is the same problem, against an unscaled reference solve
# ---------------------------------------------------------------------------


def test_5b_rescaled_solve_equals_the_unscaled_one_with_x0_and_slope_bounds(
    minimize_spy: _Spy, monkeypatch: pytest.MonkeyPatch
) -> None:
    s, y = _scores(scale=12.0, seed=5)
    largest = float(np.max(np.abs(s)))
    assert largest > platt_module._RESCALE_THRESHOLD
    # The slope bound is active at the solution (the fitted slope is about
    # 0.1 unbounded), so a bound left untransformed changes the answer.
    params = {"x0": [0.02, -0.2], "bounds": [[0.0, 0.03], [None, None]]}

    rescaled = PlattCalibrator(params).fit(s, y)
    call = _solves(minimize_spy)[-1]
    # The written start and slope bounds, moved into the rescaled coordinates.
    assert call["x0"] == pytest.approx([0.02 * largest, -0.2], rel=1e-12)
    (a_low, a_high), b_bounds = call["bounds"]
    assert (a_low, a_high) == pytest.approx((0.0, 0.03 * largest), rel=1e-12)
    assert tuple(b_bounds) == (None, None)

    monkeypatch.setattr(platt_module, "_RESCALE_THRESHOLD", math.inf)
    reference = PlattCalibrator(params).fit(s, y)
    a_r, b_r = rescaled.export_params()["a"], rescaled.export_params()["b"]
    a_u, b_u = reference.export_params()["a"], reference.export_params()["b"]
    assert a_r == pytest.approx(0.03, abs=1e-9)  # at the written bound
    assert (a_r, b_r) == pytest.approx((a_u, b_u), rel=1e-6, abs=1e-8)


# ---------------------------------------------------------------------------
# Row 8a: config.json carries the effective, prepared params
# ---------------------------------------------------------------------------


def test_8a_config_json_carries_the_prepared_isotonic_params(tmp_path: Path) -> None:
    # "eta" is a LightGBM alias; no seed is written, so the training seed (13,
    # not any default) is inherited.
    model = _fit_model("isotonic", {"eta": 0.2, "num_boost_round": 5}, seed=13)
    out = tmp_path / "code"
    model.export_code(out)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["calibration_params"] == {
        "learning_rate": 0.2,
        "num_boost_round": 5,
        "seed": 13,
    }


# ---------------------------------------------------------------------------
# Rows 9a-9b: an artifact written by LizyML 0.17.1 predicts as it did
# ---------------------------------------------------------------------------


def _expected() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURE / "expected.json").read_text())
    return data


#: SHA-256 of every fixture file, frozen when the fixture was generated by
#: ``docs/audits/2026-10-i298/probes/make_v0171_platt_artifact.py`` under
#: LizyML 0.17.1. They live in this file, not beside the fixture, so replacing
#: the fixture (and its own checksums and expectations) also needs a change
#: here that a reviewer sees.
FIXTURE_SHA256 = {
    "artifact/analysis_context.pkl": (
        "232b85526afc2499b3fc6b78c60bb61733abd898e176ada1204c2e5d209dc25d"
    ),
    "artifact/fit_result.pkl": (
        "757bfcea25887ea36011c24514cf57d7c27866b550499f6f7c0568e02553a363"
    ),
    "artifact/metadata.json": (
        "3065a29435cb19093c77ac0bf42995db9c6113994c3c0b9d2b7e3d38527cd969"
    ),
    "artifact/refit_model.pkl": (
        "dd0076046c2c1c0b183d5459035b8dffa83b6f49ad0f8a35d9512f1b5f79d58a"
    ),
    "expected.json": (
        "33847c9f89026339d1f666764b911db6c8ac97c5d156921eea27b173cf30f5bf"
    ),
}


def test_9a_the_fixture_is_the_frozen_v0_17_1_artifact() -> None:
    import hashlib

    found = {
        str(path.relative_to(FIXTURE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(FIXTURE.rglob("*"))
        if path.is_file()
    }
    assert found == FIXTURE_SHA256
    expected = _expected()
    assert expected["lizyml"] == "0.17.1"
    assert expected["numpy"] == "1.24.0"  # readable by every CI lane
    meta = json.loads((FIXTURE / "artifact" / "metadata.json").read_text())
    assert meta["format_version"] == 2 and meta["lizyml_version"] == "0.17.1"
    assert meta["config"]["calibration"]["method"] == "platt"


@pytest.mark.filterwarnings("ignore")
def test_9a_loading_runs_the_legacy_setstate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pickled calibrators carry the pre-H-0100 ``{"_model":
    LogisticRegression}`` state, so loading goes through the migration in
    ``PlattCalibrator.__setstate__``, not through the current state."""
    states: list[dict[str, Any]] = []
    real = PlattCalibrator.__setstate__

    def record(self: PlattCalibrator, state: dict[str, Any]) -> None:
        states.append(dict(state))
        real(self, state)

    monkeypatch.setattr(PlattCalibrator, "__setstate__", record)
    Model.load(FIXTURE / "artifact")
    expected = _expected()
    assert states, "no PlattCalibrator was unpickled"
    for state in states:
        assert set(state) == {"_model"}, sorted(state)
        assert type(state["_model"]).__name__ == "LogisticRegression"
    coefs = {
        (float(s["_model"].coef_[0, 0]), float(s["_model"].intercept_[0]))
        for s in states
    }
    assert (expected["platt_coef"], expected["platt_intercept"]) in coefs


# Unpickling the scikit-learn 1.3.0 estimator inside the legacy state may warn.
@pytest.mark.filterwarnings("ignore")
def test_9a_a_v0_17_1_platt_model_predicts_as_it_did() -> None:
    import pandas as pd

    expected = _expected()
    model = Model.load(FIXTURE / "artifact")
    probe = pd.DataFrame(expected["probe"])
    pred = model.predict(probe)
    np.testing.assert_array_equal(np.asarray(pred.pred), np.asarray(expected["pred"]))
    np.testing.assert_allclose(
        np.asarray(pred.proba), np.asarray(expected["proba"]), rtol=1e-9, atol=1e-12
    )


@pytest.mark.filterwarnings("ignore")
def test_9b_the_legacy_platt_calibrator_maps_scores_as_it_did() -> None:
    expected = _expected()
    model = Model.load(FIXTURE / "artifact")
    c_final = model.fit_result.calibrator.c_final
    assert isinstance(c_final, PlattCalibrator)
    exported = c_final.export_params()
    assert exported["a"] == expected["platt_coef"]
    assert exported["b"] == expected["platt_intercept"]
    np.testing.assert_allclose(
        c_final.predict(np.asarray(expected["calibrator_scores"])),
        np.asarray(expected["calibrator_proba"]),
        rtol=1e-9,
        atol=1e-12,
    )
