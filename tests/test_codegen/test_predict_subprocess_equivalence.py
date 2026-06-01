"""Real-subprocess equivalence for the generated predict.py (H-0178 item 1).

The existing equivalence suite re-implements the predict logic in-test, so a
regression in ``lizyml/codegen/templates.py`` would not be caught. These tests
execute the *generated* ``predict.py`` as a user would (subprocess + file I/O)
and assert numeric equality against ``Model.predict()``. Inference data is
written as Parquet so the input reaches the script without CSV precision loss.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)


def _predict_via_subprocess(
    codegen_dir: Path, X: pd.DataFrame, tmp_path: Path
) -> pd.DataFrame:
    infer = tmp_path / "infer.parquet"
    X.to_parquet(infer)
    out_csv = tmp_path / "preds.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(codegen_dir / "predict.py"),
            str(infer),
            "-o",
            str(out_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out_csv.exists(), result.stderr
    return pd.read_csv(out_csv)


def test_regression_predict_script_matches_model(tmp_path: Path) -> None:
    df = make_regression_df(n=200)
    m = Model(make_config("regression", n_estimators=40))
    m.fit(data=df)
    X = df.drop(columns=["target"]).iloc[:50].reset_index(drop=True)

    codegen_dir = tmp_path / "codegen"
    m.export_code(codegen_dir)
    out = _predict_via_subprocess(codegen_dir, X, tmp_path)

    np.testing.assert_allclose(out["pred"].to_numpy(), m.predict(X).pred, rtol=1e-7)


def test_binary_calibrated_predict_script_matches_model(tmp_path: Path) -> None:
    df = make_binary_df(n=300)
    m = Model(
        make_config(
            "binary",
            n_estimators=40,
            split_method="stratified_kfold",
            calibration="platt",
        )
    )
    m.fit(data=df)
    X = df.drop(columns=["target"]).iloc[:60].reset_index(drop=True)

    codegen_dir = tmp_path / "codegen"
    m.export_code(codegen_dir)
    out = _predict_via_subprocess(codegen_dir, X, tmp_path)

    ref = m.predict(X)
    np.testing.assert_array_equal(out["pred"].to_numpy(), ref.pred)
    assert ref.proba is not None
    np.testing.assert_allclose(out["proba"].to_numpy(), ref.proba, rtol=1e-6)


def test_multiclass_predict_script_matches_model(tmp_path: Path) -> None:
    df = make_multiclass_df(n=300)
    m = Model(make_config("multiclass", n_estimators=40))
    m.fit(data=df)
    X = df.drop(columns=["target"]).iloc[:60].reset_index(drop=True)

    codegen_dir = tmp_path / "codegen"
    m.export_code(codegen_dir)
    out = _predict_via_subprocess(codegen_dir, X, tmp_path)

    ref = m.predict(X)
    np.testing.assert_array_equal(out["pred"].to_numpy(), ref.pred)
    assert ref.proba is not None
    proba_cols = [c for c in out.columns if c.startswith("proba_")]
    assert len(proba_cols) == ref.proba.shape[1]
    out_proba = out[sorted(proba_cols, key=lambda c: int(c.split("_")[1]))].to_numpy()
    np.testing.assert_allclose(out_proba, ref.proba, rtol=1e-6)
