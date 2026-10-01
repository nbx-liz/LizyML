"""The generated project keeps the unseen-category policy and reports substitutions
(H-0104).

Three positions of the same rules: ``export_code`` carries the policy the fit
applied; the generated ``train.py`` rebuilds ``pipeline_state.json`` -- before
H-0104 without the policy, so ``predict.py`` fell back to ``"nan"`` after a
retrain; and the generated ``predict.py`` logs every substitution, as the
runtime reports it in ``PredictionResult.warnings``. A missing value is not an
unseen category: it stays missing under every policy, as in the runtime.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import typing
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.model import Model
from lizyml.features.encoders.categorical_encoder import UnseenPolicy
from tests._helpers import make_config

POLICIES: tuple[str, ...] = typing.get_args(UnseenPolicy)


def _df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.25, 0.15])
    df = pd.DataFrame({"num": rng.normal(size=n), "cat": cat})
    df["target"] = df["num"] + (df["cat"] == "b").astype(float)
    return df


def _export(policy: str, out: Path) -> Path:
    raw = make_config("regression", n_estimators=10)
    raw["features"] = {"unseen_policy": policy}
    model = Model(raw)
    model.fit(data=_df())
    model.export_code(out)
    return out


def _load(export_dir: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"generated_{name}_{export_dir.name}", export_dir / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(values: list[Any]) -> pd.DataFrame:
    return pd.DataFrame({"num": [0.0] * len(values), "cat": values})


@pytest.mark.parametrize("policy", POLICIES)
def test_export_carries_the_policy_the_fit_applied(policy: str, tmp_path: Path) -> None:
    root = _export(policy, tmp_path / policy)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    state = json.loads(
        (root / "artifacts" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    assert config["unseen_policy"] == policy
    assert state["unseen_policy"] == policy


@pytest.mark.parametrize("policy", POLICIES)
def test_generated_predict_logs_substitutions(
    policy: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = _export(policy, tmp_path / policy)
    predict = _load(root, "predict")
    frame = _frame(["TYPO", "a", None])

    if policy == "error":
        with pytest.raises(ValueError, match="TYPO"):
            predict.transform(frame)
        return

    with caplog.at_level(logging.WARNING):
        X = predict.transform(frame)
    messages = [r.getMessage() for r in caplog.records if "unseen" in r.getMessage()]
    assert len(messages) == 1, messages
    assert "TYPO" in messages[0] and f"unseen_policy='{policy}'" in messages[0]

    state = json.loads(
        (root / "artifacts" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    mapping = state["category_mappings"]["cat"]
    if policy == "mode":
        assert X["cat"].iloc[0] == state["unseen_codes"]["cat"]
    else:
        assert np.isnan(X["cat"].iloc[0])
    assert X["cat"].iloc[1] == mapping["a"]
    # A missing value is not unseen: it stays missing and is not reported.
    assert np.isnan(X["cat"].iloc[2])


def test_missing_values_are_not_refused_under_error(tmp_path: Path) -> None:
    root = _export("error", tmp_path / "error")
    predict = _load(root, "predict")
    X = predict.transform(_frame(["a", None]))
    assert np.isnan(X["cat"].iloc[1])


@pytest.mark.parametrize("policy", ["mode", "error"])
def test_retrain_keeps_the_policy(policy: str, tmp_path: Path) -> None:
    """The generated ``train.py`` rewrites ``pipeline_state.json``; the policy
    and the mode codes must survive it (design review round 1, blocking 2)."""
    root = _export(policy, tmp_path / policy)
    train = _load(root, "train")
    exported = json.loads(
        (root / "artifacts" / "pipeline_state.json").read_text(encoding="utf-8")
    )

    rebuilt = train.fit_pipeline(_df().drop(columns=["target"]))
    assert rebuilt["unseen_policy"] == policy
    assert rebuilt["unseen_codes"] == exported["unseen_codes"]

    predict = _load(root, "predict")
    if policy == "error":
        with pytest.raises(ValueError, match="TYPO"):
            predict.transform(_frame(["TYPO"]))
    else:
        X = predict.transform(_frame(["TYPO"]))
        assert X["cat"].iloc[0] == rebuilt["unseen_codes"]["cat"]


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(pd.Series([2, 10, 2, 10, 2, 10]), id="numeric-tie"),
        pytest.param(
            pd.Series(pd.Categorical(["b", "a", "b", "a"], categories=["b", "a"])),
            id="category-order-tie",
        ),
        # Review round 2: the mode's str() ("0.1") differed from the mapping
        # key built from the category values ("0.10000000149011612").
        pytest.param(
            pd.Series(pd.Categorical(np.array([0.1, 0.1, 0.2], dtype="float32"))),
            id="categorical-float32",
        ),
    ],
)
def test_retrain_picks_the_same_mode_as_the_runtime(
    values: pd.Series, tmp_path: Path
) -> None:
    """On a tie the runtime encoder takes the first mode in the column's own
    order (numeric, or category order). The generated ``fit_pipeline`` must pick
    the same value, not the first after converting to strings (code review
    round 1, blocking 1)."""
    from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

    root = _export("mode", tmp_path / "mode")
    train = _load(root, "train")
    frame = pd.DataFrame({"num": [0.0] * len(values), "cat": values})

    # The runtime path: the data builder casts to category, the encoder learns.
    as_category = pd.DataFrame({"cat": frame["cat"].astype("category")})
    runtime_mode = (
        CategoricalEncoder().fit(as_category, ["cat"]).get_state()["modes"]["cat"]
    )

    rebuilt = train.fit_pipeline(frame)
    mapping = rebuilt["category_mappings"]["cat"]
    # Decode the chosen code back to the original value through the mapping's
    # own keys; a str() of the runtime mode need not equal those keys.
    key = {code: k for k, code in mapping.items()}[rebuilt["unseen_codes"]["cat"]]
    chosen = next(v for v in frame["cat"].dropna().unique() if str(v) == key)
    assert chosen == runtime_mode, (chosen, runtime_mode, mapping)
