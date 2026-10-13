"""The generated train.py and predict.py encode float16 / longdouble arrivals as
LizyML does (H-0127, #309).

Each generated file carries its own copy of ``encode_column``; both are checked
against the runtime encoder, separately, so leaving one copy behind fails here.
"""

from __future__ import annotations

import importlib.util
import json
import typing
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.core.model import Model
from lizyml.features.encoders.categorical_encoder import (
    CategoricalEncoder,
    UnseenPolicy,
)
from tests._helpers import make_config

POLICIES: tuple[str, ...] = typing.get_args(UnseenPolicy)
WIDE_LONGDOUBLE = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
NEAR_THREE = np.longdouble(3) + np.finfo(np.longdouble).eps * 3


def _df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ints = rng.integers(0, 5, size=n)
    df = pd.DataFrame(
        {"num": rng.normal(size=n), "cat": pd.Series(list(ints), dtype=object)}
    )
    df["target"] = df["num"] + (ints == 3).astype(float)
    return df


@pytest.fixture(scope="module", params=POLICIES)
def exported(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Model, Path]:
    policy = request.param
    raw = make_config("regression", n_estimators=10)
    raw["features"] = {"unseen_policy": policy}
    model = Model(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data=_df())
    out = tmp_path_factory.mktemp(f"export_{policy}")
    model.export_code(out)
    return policy, model, out


def _load(export_dir: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"generated_{name}_{export_dir.name}", export_dir / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arrivals() -> dict[str, pd.Series]:
    arrivals = {
        "float16": pd.Series([0, 1, np.nan, 2.5, 4], dtype=np.float16),
        "longdouble": pd.Series([0, 1, np.nan, 2.5, 4], dtype=np.longdouble),
        "object of np.float16": pd.Series(
            [np.float16(v) for v in (0, 1, np.nan, 2.5, 4)], dtype=object
        ),
    }
    if WIDE_LONGDOUBLE:
        arrivals["longdouble, a value float64 changes"] = pd.Series(
            [0, 1, np.nan, NEAR_THREE, 4], dtype=np.longdouble
        )
        arrivals["object of np.longdouble"] = pd.Series(
            [np.longdouble(v) for v in (0, 1, np.nan)] + [NEAR_THREE, np.longdouble(4)],
            dtype=object,
        )
    return arrivals


def _runtime_codes(
    entry: dict[str, Any], policy: str, series: pd.Series
) -> list[float]:
    encoder = CategoricalEncoder(unseen_policy=policy)  # type: ignore[arg-type]
    encoder.load_state(
        {
            "unseen_policy": policy,
            "categories": {"cat": entry["categories"]},
            "modes": {"cat": entry["mode"]},
        }
    )
    out = encoder.transform(pd.DataFrame({"cat": series}))
    codes = out["cat"].cat.codes.to_numpy().astype(np.float64)
    codes[codes < 0] = np.nan
    return codes.tolist()


@pytest.mark.parametrize("module_name", ["train", "predict"])
@pytest.mark.parametrize("arrival", sorted(_arrivals()))
def test_each_generated_copy_encodes_as_the_runtime(
    exported: tuple[str, Model, Path], module_name: str, arrival: str
) -> None:
    """Criterion 3: train.py's and predict.py's encode_column, one at a time."""
    policy, _, root = exported
    module = _load(root, module_name)
    state = json.loads(
        (root / "artifacts" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    entry = state["categories"]["cat"]
    series = _arrivals()[arrival]
    if policy == "error":
        with pytest.raises(LizyMLError):
            _runtime_codes(entry, policy, series)
        with pytest.raises(ValueError, match="unseen categories"):
            module.encode_column(series, entry, policy, "cat")
        return
    expected = _runtime_codes(entry, policy, series)
    got = module.encode_column(series, entry, policy, "cat").tolist()
    np.testing.assert_array_equal(np.array(got), np.array(expected))
    assert np.isnan(got[2])  # the missing value stays missing


@pytest.mark.parametrize("arrival", ["float16", "longdouble"])
def test_generated_predict_matches_the_model(
    exported: tuple[str, Model, Path], arrival: str
) -> None:
    """Criterion 3: whole predictions, generated predict.py against Model.predict."""
    _, model, root = exported
    # Seen values and a missing one, so 'error' predicts too.
    cat = pd.Series([0, 1, np.nan, 3, 4], dtype=np.dtype(arrival))
    frame = pd.DataFrame({"num": np.linspace(-1, 1, 5), "cat": cat})
    predict = _load(root, "predict")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = model.predict(frame).pred
    got = predict.predict(frame)["pred"]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
