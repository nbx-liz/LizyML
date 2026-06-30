"""Non-ASCII portability regression for the generated scripts (#192).

The generated ``train.py`` / ``predict.py`` / ``test_equivalence.py`` open the
JSON artifacts (``config.json``, ``pipeline_state.json``, ``calibrator.json``)
with ``open()``. If the encoding is not pinned, the default text encoding comes
from the process locale -- ``cp1252`` on Windows, ``ascii`` under a ``C`` locale.
A category value that contains non-ASCII characters is then written/read as
raw UTF-8 by ``json`` (``ensure_ascii=False``), which raises
``UnicodeEncodeError`` / ``UnicodeDecodeError`` under such a locale.

These tests run the *generated* scripts as a subprocess with the default text
encoding forced to ASCII (``PYTHONUTF8=0`` + ``LC_ALL=C``) over data carrying a
non-ASCII categorical value, and assert they round-trip. They fail against the
pre-fix templates (default ``open()``) and pass once every ``open()`` pins
``encoding="utf-8"``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model
from tests._helpers import make_config

# Default text encoding forced to ASCII to emulate a non-UTF-8 locale
# (Windows cp1252 / POSIX C). PYTHONUTF8=0 disables the 3.15+ UTF-8 mode so the
# locale actually governs open()'s default encoding; PYTHONIOENCODING keeps
# stdio on utf-8 so only the file-open paths under test can fail.
_ASCII_LOCALE_ENV = {
    **os.environ,
    "PYTHONUTF8": "0",
    "PYTHONIOENCODING": "utf-8",
    "LC_ALL": "C",
    "LANG": "C",
}

# Non-ASCII categorical values spanning Latin-1 and CJK code points.
_NON_ASCII_CATS = ["café", "naïve", "日本語", "Über"]


def _make_non_ascii_binary_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    """Binary DataFrame with a categorical column of non-ASCII values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"feat_a": rng.uniform(0, 10, n)})
    df["feat_cat"] = rng.choice(_NON_ASCII_CATS, n)
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _export_codegen(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Fit a calibrated binary model on non-ASCII data and export the scripts."""
    df = _make_non_ascii_binary_df()
    m = Model(
        make_config(
            "binary",
            n_estimators=20,
            split_method="stratified_kfold",
            calibration="platt",
        )
    )
    m.fit(data=df)
    codegen_dir = tmp_path / "codegen"
    m.export_code(codegen_dir)
    return codegen_dir, df


def test_train_script_round_trips_non_ascii_under_ascii_locale(
    tmp_path: Path,
) -> None:
    """train.py reads config.json and writes the JSON artifacts as UTF-8.

    Exercises the write paths (pipeline_state.json with ensure_ascii=False,
    calibrator.json) and the config.json read path. Pre-fix this raised
    UnicodeEncodeError when dumping the non-ASCII category mappings.
    """
    codegen_dir, df = _export_codegen(tmp_path)
    train_data = codegen_dir / "train.parquet"
    df.to_parquet(train_data)

    # Remove the LizyML-written artifacts so train.py must regenerate them
    # itself under the ASCII locale.
    for art in (codegen_dir / "artifacts").glob("*"):
        art.unlink()

    result = subprocess.run(
        [sys.executable, "train.py", "train.parquet"],
        cwd=codegen_dir,
        env=_ASCII_LOCALE_ENV,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    state_path = codegen_dir / "artifacts" / "pipeline_state.json"
    raw = state_path.read_bytes()
    assert any(b > 127 for b in raw), "expected non-ASCII bytes in pipeline_state"
    mappings = json.loads(raw.decode("utf-8"))["category_mappings"]["feat_cat"]
    assert set(mappings) == set(_NON_ASCII_CATS)
    assert (codegen_dir / "artifacts" / "calibrator.json").exists()


def test_predict_script_reads_non_ascii_artifacts_under_ascii_locale(
    tmp_path: Path,
) -> None:
    """predict.py reads config.json / pipeline_state.json / calibrator.json.

    Runs the full train.py -> predict.py flow a user would (so the model and
    artifacts are consistent), both under the ASCII locale. The artifacts carry
    non-ASCII category keys; pre-fix the default open() raised
    UnicodeDecodeError when predict.py loaded pipeline_state.json.
    """
    codegen_dir, df = _export_codegen(tmp_path)
    train_data = codegen_dir / "train.parquet"
    df.to_parquet(train_data)
    infer_data = codegen_dir / "infer.parquet"
    df.drop(columns=["target"]).to_parquet(infer_data)

    # Regenerate artifacts via the generated train.py so the categorical
    # signature matches what predict.py reconstructs (decoupled from the
    # LizyML-trained model.txt's categorical-feature predict semantics).
    for art in (codegen_dir / "artifacts").glob("*"):
        art.unlink()
    train = subprocess.run(
        [sys.executable, "train.py", "train.parquet"],
        cwd=codegen_dir,
        env=_ASCII_LOCALE_ENV,
        capture_output=True,
        text=True,
    )
    assert train.returncode == 0, train.stderr

    result = subprocess.run(
        [sys.executable, "predict.py", "infer.parquet", "-o", "preds.csv"],
        cwd=codegen_dir,
        env=_ASCII_LOCALE_ENV,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    preds = pd.read_csv(codegen_dir / "preds.csv")
    assert len(preds) == len(df)
    assert "pred" in preds.columns
