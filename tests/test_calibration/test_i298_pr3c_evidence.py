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
from lizyml.calibration.platt import _DEFAULT_OPTIONS, PlattCalibrator
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


def _one_solve(calibrator: Any, spy: _Spy) -> dict[str, Any]:
    s, y = _scores()
    calibrator.fit(s, y)
    solves = _solves(spy)
    assert len(solves) == 1, solves
    return solves[0]


def test_4a_written_tol_replaces_the_tolerance_options_it_sets(
    minimize_spy: _Spy,
) -> None:
    kwargs = _one_solve(PlattCalibrator({"tol": 1e-8}), minimize_spy)
    assert kwargs["tol"] == 1e-8
    # L-BFGS-B's tolerance keys come from tol, not from the platt defaults.
    assert "ftol" not in kwargs["options"] and "gtol" not in kwargs["options"]


def test_4b_written_options_win_over_tol_and_over_the_defaults(
    minimize_spy: _Spy,
) -> None:
    kwargs = _one_solve(
        PlattCalibrator({"tol": 1e-8, "options": {"gtol": 1e-9, "maxiter": 77}}),
        minimize_spy,
    )
    assert kwargs["tol"] == 1e-8
    assert kwargs["options"] == {"gtol": 1e-9, "maxiter": 77}


def test_4b_written_options_merge_with_the_defaults(minimize_spy: _Spy) -> None:
    kwargs = _one_solve(PlattCalibrator({"options": {"maxiter": 77}}), minimize_spy)
    assert "tol" not in kwargs
    assert kwargs["options"] == {**_DEFAULT_OPTIONS["L-BFGS-B"], "maxiter": 77}


def test_4c_platt_default_options_apply_to_l_bfgs_b_only(minimize_spy: _Spy) -> None:
    default = _one_solve(PlattCalibrator(), minimize_spy)
    assert default["method"] == "L-BFGS-B"
    assert default["options"] == _DEFAULT_OPTIONS["L-BFGS-B"]
    minimize_spy.calls.clear()
    tnc = _one_solve(PlattCalibrator({"method": "TNC"}), minimize_spy)
    assert tnc["method"] == "TNC" and tnc["options"] == {}


def test_4c_beta_has_no_default_options(minimize_spy: _Spy) -> None:
    kwargs = _one_solve(BetaCalibrator(), minimize_spy)
    assert kwargs["method"] == "L-BFGS-B" and kwargs["options"] == {}


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


def test_9a_the_v0_17_1_artifact_is_what_the_fixture_says() -> None:
    expected = _expected()
    assert expected["lizyml"] == "0.17.1"
    assert expected["numpy"] == "1.24.0"  # readable by every CI lane
    meta = json.loads((FIXTURE / "artifact" / "metadata.json").read_text())
    assert meta["format_version"] == 2 and meta["lizyml_version"] == "0.17.1"
    assert meta["config"]["calibration"]["method"] == "platt"


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
