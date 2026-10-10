"""Harness for the H-0120 reproduction tests: fit, export, retrain, compare.

The promise under test (H-0120): after ``Model.fit(df)`` and
``export_code(path)``, running the generated ``train.py`` on the same data
writes ``artifacts/`` whose uncalibrated predictions match the LizyML refit
model's at ``rtol=1e-7``. The configs built here carry no calibration block, so
``Model.predict`` returns the refit model's uncalibrated output.

``train.py`` runs in-process through :func:`runpy.run_path` (the same code path
as ``python train.py``: ``__name__ == "__main__"`` and ``sys.argv``); a few
tests run it as a real subprocess to anchor the command-line claim.
"""

from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from lizyml import Model

#: LightGBM's own determinism settings (H-0120 premises). Without them the
#: library picks col-wise or row-wise by timing, and the promise holds only as
#: far as LightGBM itself is deterministic.
DETERMINISTIC: dict[str, Any] = {
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
    "verbose": -1,
}

TASKS = ("regression", "binary", "multiclass")
SPLIT_METHODS = (
    "kfold",
    "stratified_kfold",
    "group_kfold",
    "stratified_group_kfold",
    "time_series",
    "purged_time_series",
    "group_time_series",
    "blocked_group_kfold",
)


def make_frame(
    task: str, n: int = 240, seed: int = 0, *, tied_time: bool = False
) -> pd.DataFrame:
    """Rows deliberately out of time order, with group and period columns."""
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    df = pd.DataFrame({"f0": f0, "f1": f1})
    df["t"] = rng.integers(0, n // 6, n) if tied_time else rng.permutation(n)
    df["g"] = rng.integers(0, 12, n)
    df["p"] = rng.integers(0, 6, n)
    if task == "regression":
        df["target"] = 2.0 * f0 + f1 + rng.normal(0, 0.3, n)
    elif task == "binary":
        df["target"] = (f0 + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    else:
        # Unequal class sizes, so `balanced` weights are not all 1.
        score = f0 + 0.3 * rng.normal(size=n)
        df["target"] = np.digitize(score, [-0.2, 0.9]).astype(np.int64)
    return df


def split_block(method: str) -> dict[str, Any]:
    """A valid outer split config for *method* over :func:`make_frame` columns."""
    if method == "blocked_group_kfold":
        return {
            "method": method,
            "blocks": {"col": "p", "cutoffs": [2, 4], "mode": "expanding"},
            "groups": {"col": "g", "n_splits": 2, "shuffle": True},
            "min_train_rows": 1,
            "min_valid_rows": 1,
        }
    block: dict[str, Any] = {"method": method, "n_splits": 3}
    if method in ("kfold", "stratified_kfold"):
        block["random_state"] = 42
    if method == "stratified_group_kfold":
        block["shuffle"] = True
        block["random_state"] = 11
    return block


def data_block(method: str) -> dict[str, Any]:
    data: dict[str, Any] = {"target": "target"}
    if method in ("group_kfold", "stratified_group_kfold", "group_time_series"):
        data["group_col"] = "g"
    if method in ("time_series", "purged_time_series", "group_time_series"):
        data["time_col"] = "t"
    return data


def make_config(
    task: str,
    method: str = "kfold",
    *,
    early_stopping: dict[str, Any] | None = None,
    model_extra: dict[str, Any] | None = None,
    split_extra: dict[str, Any] | None = None,
    n_estimators: int = 60,
    seed: int = 3,
) -> dict[str, Any]:
    """A deterministic config with early stopping on (patience 5) by default."""
    es: dict[str, Any] = {"enabled": True, "rounds": 5}
    if early_stopping is not None:
        es = {**es, **early_stopping}
    split = {**split_block(method), **(split_extra or {})}
    model: dict[str, Any] = {
        "name": "lgbm",
        "params": {"n_estimators": n_estimators, **DETERMINISTIC},
        **(model_extra or {}),
    }
    return {
        "config_version": 1,
        "task": task,
        "data": data_block(method),
        "split": split,
        "model": model,
        "training": {"seed": seed, "early_stopping": es},
    }


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_train(project: Path, data_path: Path, *, subprocess_run: bool = False) -> None:
    """Run the generated ``train.py`` on *data_path* without calibration."""
    argv = [str(project / "train.py"), str(data_path), "--no-calibration"]
    if subprocess_run:
        subprocess.run(
            [sys.executable, *argv], check=True, capture_output=True, text=True
        )
        return
    saved = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(argv[0], run_name="__main__")
    finally:
        sys.argv = saved


def write_data(df: pd.DataFrame, project: Path, fmt: str) -> Path:
    path = project / f"train_data.{fmt}"
    if fmt == "parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)
    return path


def generated_prediction(project: Path, X: pd.DataFrame, task: str) -> np.ndarray:
    """Uncalibrated prediction of the artifacts ``train.py`` wrote."""
    module = load_module(project / "predict.py", f"gen_predict_{id(project)}")
    out = module.predict(X)
    return np.asarray(out["pred"] if task == "regression" else out["proba"])


def lizyml_prediction(model: Model, X: pd.DataFrame, task: str) -> np.ndarray:
    result = model.predict(X)
    return np.asarray(result.pred if task == "regression" else result.proba)


def assert_retrain_reproduces(
    model: Model,
    df: pd.DataFrame,
    project: Path,
    *,
    fmt: str = "parquet",
    subprocess_run: bool = False,
    export: bool = True,
) -> None:
    """Export *model*, retrain on *df* with the generated code, compare."""
    task = model._cfg.task
    if export:
        model.export_code(project)
    data_path = write_data(df, project, fmt)
    run_train(project, data_path, subprocess_run=subprocess_run)
    X = df.drop(columns=["target"])
    np.testing.assert_allclose(
        generated_prediction(project, X, task),
        lizyml_prediction(model, X, task),
        rtol=1e-7,
        atol=0.0,
    )
