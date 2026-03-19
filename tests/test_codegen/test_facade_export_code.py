"""Tests for Model.export_code() facade integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lizyml.core.model import Model

# ── Fixtures ─────────────────────────────────────────────────


def _make_binary_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(["a", "b", "c"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


def _make_regression_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )


def _binary_config(*, calibration: str | None = "platt") -> dict:
    cfg: dict = {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "split": {
            "method": "stratified_kfold",
            "n_splits": 3,
            "random_state": 42,
        },
        "model": {"name": "lgbm", "params": {"n_estimators": 50}},
        "training": {"seed": 42},
    }
    if calibration:
        cfg["calibration"] = {"method": calibration}
    return cfg


def _regression_config() -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {
            "method": "kfold",
            "n_splits": 3,
            "random_state": 42,
        },
        "model": {"name": "lgbm", "params": {"n_estimators": 50}},
        "training": {"seed": 42},
    }


# ── Tests ────────────────────────────────────────────────────


class TestExportCodeBinary:
    def test_creates_all_expected_files(self, tmp_path: Path) -> None:
        df = _make_binary_data()
        m = Model(_binary_config())
        m.fit(df)

        out = tmp_path / "codegen_out"
        result = m.export_code(out)

        assert result == out
        assert (out / "config.json").exists()
        assert (out / "train.py").exists()
        assert (out / "predict.py").exists()
        assert (out / "test_equivalence.py").exists()
        assert (out / "requirements.txt").exists()
        assert (out / "artifacts" / "model.txt").exists()
        assert (out / "artifacts" / "pipeline_state.json").exists()

    def test_config_json_has_correct_task(self, tmp_path: Path) -> None:
        df = _make_binary_data()
        m = Model(_binary_config())
        m.fit(df)

        out = tmp_path / "codegen_out"
        m.export_code(out)

        with open(out / "config.json") as f:
            cfg = json.load(f)
        assert cfg["_task"] == "binary"
        assert cfg["_target_col"] == "target"

    def test_platt_calibrator_exported(self, tmp_path: Path) -> None:
        df = _make_binary_data()
        m = Model(_binary_config(calibration="platt"))
        m.fit(df)

        out = tmp_path / "codegen_out"
        m.export_code(out)

        cal_path = out / "artifacts" / "calibrator.json"
        assert cal_path.exists()
        with open(cal_path) as f:
            params = json.load(f)
        assert params["method"] == "platt"

    def test_no_calibration_no_calibrator_files(self, tmp_path: Path) -> None:
        df = _make_binary_data()
        m = Model(_binary_config(calibration=None))
        m.fit(df)

        out = tmp_path / "codegen_out"
        m.export_code(out)

        assert not (out / "artifacts" / "calibrator.json").exists()

    def test_unfitted_raises(self, tmp_path: Path) -> None:
        from lizyml.core.exceptions import LizyMLError

        m = Model(_binary_config())
        with pytest.raises(LizyMLError):
            m.export_code(tmp_path / "codegen_out")


class TestExportCodeRegression:
    def test_creates_files(self, tmp_path: Path) -> None:
        df = _make_regression_data()
        m = Model(_regression_config())
        m.fit(df)

        out = tmp_path / "codegen_out"
        m.export_code(out)

        assert (out / "config.json").exists()
        assert (out / "train.py").exists()
        assert (out / "predict.py").exists()

    def test_regression_no_calibrator(self, tmp_path: Path) -> None:
        df = _make_regression_data()
        m = Model(_regression_config())
        m.fit(df)

        out = tmp_path / "codegen_out"
        m.export_code(out)

        assert not (out / "artifacts" / "calibrator.json").exists()

        with open(out / "config.json") as f:
            cfg = json.load(f)
        assert cfg["_task"] == "regression"
        assert cfg["calibration_method"] is None
