"""The generated project rebuilds the calibrator with the same settings (H-0100, #277).

H-0059 promises that the exported ``train.py`` retrains "with the same settings,
including the calibrator". ``config.json`` carried the calibration method but not
``calibration.params``, and every generated fitter hard-coded its defaults -- so a
retrain ignored the parameters even for ``isotonic``, whose runtime path already
honoured them.

Agreement between runtime and generated code is not enough on its own: both could
ignore the parameters and still agree. So the tests also require the parameters
to change the generated fit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from lizyml import Model
from lizyml.calibration.registry import get_calibrator
from tests._helpers import make_binary_df, make_config

REPO = Path(__file__).resolve().parents[2]


def _export(
    tmp_path: Path, method: str, params: dict[str, Any] | None, name: str
) -> Path:
    cfg = make_config(
        "binary",
        n_estimators=5,
        n_splits=3,
        calibration=method,
        calibration_params=params,
    )
    model = Model(cfg, data=make_binary_df(n=240, seed=6))
    model.fit()
    out = tmp_path / name
    model.export_code(out)
    return out


def _load_train(export_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"generated_train_{export_dir.name}", export_dir / "train.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scores(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 2.0, 400)
    y = (rng.random(400) < expit(0.6 * s - 0.5)).astype(float)
    return s, y


def test_config_json_carries_the_effective_calibration_params(tmp_path: Path) -> None:
    out = _export(tmp_path, "platt", {"target_smoothing": False}, "cfg")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["calibration_method"] == "platt"
    assert config["calibration_params"] == {"target_smoothing": False}


@pytest.mark.parametrize(
    "method,params",
    [
        ("platt", {}),
        ("platt", {"target_smoothing": False, "method": "TNC"}),
        ("beta", {}),
        ("beta", {"bounds": [[0.0, 0.2], [None, None], [None, None]]}),
    ],
)
def test_generated_fitter_matches_the_runtime_calibrator(
    tmp_path: Path, method: str, params: dict[str, Any]
) -> None:
    out = _export(tmp_path, method, params, f"eq_{method}_{len(params)}")
    train = _load_train(out)
    s, y = _scores()

    generated = train._CAL_FITTERS[method](s, y, params)
    runtime = get_calibrator(method, params=params or None).fit(s, y).export_params()

    for key in runtime:
        if key == "method":
            assert generated[key] == runtime[key]
        else:
            assert generated[key] == pytest.approx(runtime[key], abs=1e-6), key


def test_generated_isotonic_fitter_honours_its_params(tmp_path: Path) -> None:
    out = _export(
        tmp_path, "isotonic", {"learning_rate": 0.2, "num_boost_round": 5}, "iso"
    )
    train = _load_train(out)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    s, y = _scores(1)

    calibrator = get_calibrator("isotonic", params=config["calibration_params"]).fit(
        s, y
    )
    train._CAL_FITTERS["isotonic"](s, y, config["calibration_params"])

    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(out / "artifacts" / "calibrator_model.txt"))
    np.testing.assert_allclose(
        booster.predict(s.reshape(-1, 1)), calibrator.predict(s), rtol=0, atol=1e-9
    )
    assert "[learning_rate: 0.2]" in booster.model_to_string()


def test_generated_platt_fitter_is_changed_by_its_params(tmp_path: Path) -> None:
    out = _export(tmp_path, "platt", None, "effect")
    train = _load_train(out)
    s, y = _scores(2)
    default = train._CAL_FITTERS["platt"](s[:30], y[:30], {})
    plain = train._CAL_FITTERS["platt"](s[:30], y[:30], {"target_smoothing": False})
    assert (default["a"], default["b"]) != pytest.approx(
        (plain["a"], plain["b"]), abs=1e-6
    )


def test_generated_retrain_uses_the_params(tmp_path: Path) -> None:
    df = make_binary_df(n=240, seed=6)
    results = {}
    for label, params in (("smooth", None), ("plain", {"target_smoothing": False})):
        out = _export(tmp_path, "platt", params, f"retrain_{label}")
        train = _load_train(out)
        train.train(df, calibrate=True)
        results[label] = json.loads(
            (out / "artifacts" / "calibrator.json").read_text("utf-8")
        )

    assert results["smooth"]["b"] != pytest.approx(results["plain"]["b"], abs=1e-6)


@pytest.mark.parametrize("method", ["platt", "beta"])
def test_requirements_list_scipy_when_the_generated_code_imports_it(
    tmp_path: Path, method: str
) -> None:
    out = _export(tmp_path, method, None, f"req_{method}")
    lines = (out / "requirements.txt").read_text(encoding="utf-8").split()
    assert "scipy" in lines


def test_readme_names_both_calibrators_that_need_scipy() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    line = next(ln for ln in readme.splitlines() if ln.startswith("Dependencies:"))
    assert "scipy" in line and "platt" in line and "beta" in line, line
