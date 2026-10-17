"""MASE in the generated train.py (H-0132, criterion 5)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml.codegen.templates import render_train_py
from lizyml.core.model import Model
from lizyml.estimators.lgbm.metric_bridge import _FEVAL_METRICS, resolve_metrics
from tests._helpers import make_config


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"gen_{name}_{path.name}", path / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_feval_metric_reaches_the_generated_registry() -> None:
    """Derived from ``_FEVAL_METRICS``, so a new feval metric cannot take the
    generated code's warning-and-skip path for unknown names."""
    src = render_train_py()
    names = sorted({name for names in _FEVAL_METRICS.values() for name in names})
    assert "mase" in names
    for name in names:
        assert f'"{name}": _feval_{name},' in src, name


def test_the_generated_feval_receives_the_training_rows_of_the_inner_split() -> None:
    src = render_train_py()
    assert "build_feval_from_config(y.iloc[tr] if iv is not None else y)" in src


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["target"] = df["a"] * 3.0 + rng.normal(scale=0.5, size=n)
    cfg = make_config("regression", n_estimators=10, n_splits=2)
    cfg["model"]["params"]["metric"] = ["mase"]
    model = Model(cfg)
    model.fit(data=df)
    out = tmp_path_factory.mktemp("gen")
    model.export_code(out / "code")
    return _load(out / "code", "train")


def test_generated_and_library_fevals_agree(generated: ModuleType) -> None:
    rng = np.random.default_rng(4)
    y_train = rng.normal(scale=1.0, size=80)
    y_valid = rng.normal(scale=10.0, size=40)  # labels differ from y_train
    pred = y_valid + rng.normal(size=40)
    ds = lgb.Dataset(np.zeros((40, 1)), y_valid, free_raw_data=False)
    ds.construct()

    gen = generated.build_feval_from_config(y_train)
    _, lib, _ = resolve_metrics(["mase"], "regression", y_train=y_train)
    (_, gen_value, gen_higher) = gen[0](pred, ds)
    (_, lib_value, lib_higher) = lib[0](pred, ds)
    assert gen_value == pytest.approx(lib_value, rel=1e-12)
    assert gen_higher is lib_higher is False
    # LightGBM keeps labels as float32: the scored labels are what it returns.
    labels = np.asarray(ds.get_label())
    expected = np.mean(np.abs(labels - pred)) / np.mean(
        np.abs(y_train - y_train.mean())
    )
    assert gen_value == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "y_train",
    [np.full(5, 2.0), np.array([1.0, np.inf, 2.0])],
    ids=["constant", "infinite"],
)
def test_the_generated_scale_refuses_zero_and_non_finite(
    generated: ModuleType, y_train: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="MASE"):
        generated._train_scale_mase(y_train)


def test_the_generated_feval_needs_the_training_targets(generated: ModuleType) -> None:
    with pytest.raises(ValueError, match="training"):
        generated.build_feval_from_config()
