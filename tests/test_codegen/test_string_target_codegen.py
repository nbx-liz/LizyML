"""Codegen E2E tests for non-numeric classification targets (H-0070).

Verifies that ``Model.export_code()`` produces train.py / predict.py that:
- bake the original class labels into config.json
- encode str y to int before training (train.py)
- decode int predictions back to original labels (predict.py)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml.core.model import Model


def _binary_string_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = np.where(df["feat_a"] > 5, "yes", "no")
    return df


def _multiclass_string_df(n: int = 240, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    bins = pd.cut(df["feat_a"], bins=3, labels=["Adelie", "Chinstrap", "Gentoo"])
    df["target"] = bins.astype(str)
    return df


def _config(task: str, n_estimators: int = 20) -> dict:
    return {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {
            "method": "stratified_kfold",
            "n_splits": 3,
            "random_state": 42,
        },
        "model": {"name": "lgbm", "params": {"n_estimators": n_estimators}},
        "training": {"seed": 42},
    }


class TestConfigBakesClasses:
    def test_binary_string_classes_in_config(self, tmp_path: Path) -> None:
        df = _binary_string_df()
        m = Model(_config("binary"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        with open(codegen_dir / "config.json") as f:
            cfg = json.load(f)
        assert cfg["target_encoder"]["needs_encoding"] is True
        assert cfg["target_encoder"]["classes"] == ["no", "yes"]

    def test_multiclass_string_classes_in_config(self, tmp_path: Path) -> None:
        df = _multiclass_string_df()
        m = Model(_config("multiclass"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        with open(codegen_dir / "config.json") as f:
            cfg = json.load(f)
        assert cfg["target_encoder"]["classes"] == ["Adelie", "Chinstrap", "Gentoo"]

    def test_numeric_target_emits_no_op_block(self, tmp_path: Path) -> None:
        from tests._helpers import make_binary_df

        df = make_binary_df(n=120)
        m = Model(_config("binary"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        with open(codegen_dir / "config.json") as f:
            cfg = json.load(f)
        assert cfg["target_encoder"]["needs_encoding"] is False
        assert cfg["target_encoder"]["classes"] == []


class TestPredictScriptRoundTrip:
    def test_binary_string_predict_script_decodes_labels(self, tmp_path: Path) -> None:
        df = _binary_string_df()
        m = Model(_config("binary"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        # Write inference data without the target column
        infer_csv = tmp_path / "infer.csv"
        df.drop(columns=["target"]).iloc[:30].to_csv(infer_csv, index=False)
        out_csv = tmp_path / "preds.csv"

        # Run the generated predict.py as the user would
        result = subprocess.run(
            [
                sys.executable,
                str(codegen_dir / "predict.py"),
                str(infer_csv),
                "-o",
                str(out_csv),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert out_csv.exists(), result.stderr

        out_df = pd.read_csv(out_csv)
        assert set(out_df["pred"].unique()).issubset({"yes", "no"})

    def test_multiclass_string_predict_script_decodes_labels(
        self, tmp_path: Path
    ) -> None:
        df = _multiclass_string_df()
        m = Model(_config("multiclass"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        infer_csv = tmp_path / "infer.csv"
        df.drop(columns=["target"]).iloc[:40].to_csv(infer_csv, index=False)
        out_csv = tmp_path / "preds.csv"

        result = subprocess.run(
            [
                sys.executable,
                str(codegen_dir / "predict.py"),
                str(infer_csv),
                "-o",
                str(out_csv),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert out_csv.exists(), result.stderr

        out_df = pd.read_csv(out_csv)
        assert set(out_df["pred"].unique()).issubset({"Adelie", "Chinstrap", "Gentoo"})


class TestTrainScriptRoundTrip:
    def test_binary_string_train_script_completes(self, tmp_path: Path) -> None:
        df = _binary_string_df()
        m = Model(_config("binary"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        train_csv = tmp_path / "train.csv"
        df.to_csv(train_csv, index=False)

        # Re-train via the generated script — must accept str target via
        # the baked-in encoder.
        result = subprocess.run(
            [
                sys.executable,
                str(codegen_dir / "train.py"),
                str(train_csv),
                "--no-calibration",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(codegen_dir),
        )
        assert (codegen_dir / "artifacts" / "model.txt").exists(), result.stderr


class TestUnseenLabelRejection:
    """Generated train.py raises a clear error on unseen labels (H-0070)."""

    def test_train_script_rejects_unseen_label(self, tmp_path: Path) -> None:
        df = _binary_string_df()
        m = Model(_config("binary"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        # Inject a third label the encoder has never seen.
        bad_df = df.copy()
        bad_df.loc[bad_df.index[:5], "target"] = "maybe"
        bad_csv = tmp_path / "bad_train.csv"
        bad_df.to_csv(bad_csv, index=False)

        result = subprocess.run(
            [
                sys.executable,
                str(codegen_dir / "train.py"),
                str(bad_csv),
                "--no-calibration",
            ],
            capture_output=True,
            text=True,
            cwd=str(codegen_dir),
        )
        assert result.returncode != 0
        # The encoder raises ValueError with the unseen label name.
        assert "maybe" in (result.stderr + result.stdout)
