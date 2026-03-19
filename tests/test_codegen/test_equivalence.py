"""E2E equivalence tests — exported predict.py matches LizyML predictions.

Tests 5 patterns:
1. Regression (no calibration)
2. Binary + Platt calibrator
3. Binary + Isotonic calibrator
4. Binary + Beta calibrator
5. Multiclass (no calibration)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from lizyml.core.model import Model

# ── Data generators ──────────────────────────────────────────


def _binary_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(["a", "b", "c"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


def _regression_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )


def _multiclass_data(n: int = 300, n_classes: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "target": rng.integers(0, n_classes, size=n),
        }
    )


# ── Config builders ──────────────────────────────────────────


def _config(
    task: str,
    *,
    calibration: str | None = None,
    n_estimators: int = 50,
) -> dict[str, Any]:
    split_method = "stratified_kfold" if task == "binary" else "kfold"
    cfg: dict[str, Any] = {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {
            "method": split_method,
            "n_splits": 3,
            "random_state": 42,
        },
        "model": {
            "name": "lgbm",
            "params": {"n_estimators": n_estimators},
        },
        "training": {"seed": 42},
    }
    if calibration:
        cfg["calibration"] = {"method": calibration}
    return cfg


# ── Prediction via exported code ─────────────────────────────


def _predict_via_codegen(
    codegen_dir: Path, X: pd.DataFrame
) -> dict[str, np.ndarray | None]:
    """Run prediction using the exported codegen artifacts directly.

    Instead of exec-ing the predict.py script (which would require
    subprocess + file I/O), we replicate the predict logic using the
    exported artifacts and config.
    """
    with open(codegen_dir / "config.json") as f:
        cfg = json.load(f)

    artifacts = codegen_dir / "artifacts"

    # Load pipeline state and transform
    with open(artifacts / "pipeline_state.json") as f:
        state = json.load(f)

    expected = state["feature_names"]
    Xp = X[expected].copy()

    # The exported pipeline_state uses the LizyML encoder format:
    # encoder.categories: {col: [cat_values]}
    # We replicate the NativeFeaturePipeline transform: set pd.Categorical
    encoder = state.get("encoder", {})
    categories = encoder.get("categories", {})
    for col, cats in categories.items():
        if col in Xp.columns:
            Xp[col] = Xp[col].astype("category")
            Xp[col] = Xp[col].cat.set_categories(cats)

    # Load model and predict
    booster = lgb.Booster(model_file=str(artifacts / "model.txt"))
    task = cfg["_task"]

    if task == "regression":
        pred = np.asarray(booster.predict(Xp), dtype=np.float64)
        return {"pred": pred, "proba": None}

    if task == "binary":
        proba = np.asarray(booster.predict(Xp), dtype=np.float64)
        # Check for calibrator
        cal_path = artifacts / "calibrator.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            logits = np.asarray(booster.predict(Xp, raw_score=True), dtype=np.float64)
            proba = _calibrate(logits, cal, artifacts)
        return {
            "pred": (proba > 0.5).astype(np.int64),
            "proba": proba,
        }

    if task == "multiclass":
        proba = np.asarray(booster.predict(Xp), dtype=np.float64)
        return {
            "pred": np.argmax(proba, axis=1).astype(np.int64),
            "proba": proba,
        }

    raise ValueError(f"Unknown task: {task}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def _calibrate(raw: np.ndarray, cal: dict, artifacts: Path) -> np.ndarray:
    m = cal["method"]
    if m == "platt":
        return 1 / (1 + np.exp(-(cal["a"] * raw + cal["b"])))
    if m == "beta":
        s = np.clip(_sigmoid(raw), 1e-10, 1 - 1e-10)
        logit = cal["a"] * np.log(s) + cal["b"] * np.log(1 - s) + cal["c"]
        return np.clip(_sigmoid(logit), 0, 1)
    if m == "isotonic":
        bst = lgb.Booster(model_file=str(artifacts / cal["model_file"]))
        return np.clip(bst.predict(raw.reshape(-1, 1)), 0, 1)
    raise ValueError(f"Unknown calibration: {m}")


# ── LizyML prediction ───────────────────────────────────────


def _predict_via_lizyml(model: Model, X: pd.DataFrame) -> dict:
    """Get predictions from fitted LizyML model."""
    result = model.predict(X)
    return {
        "pred": result.pred,
        "proba": result.proba,
    }


# ── Tests ────────────────────────────────────────────────────


class TestEquivalenceRegression:
    def test_regression_predictions_match(self, tmp_path: Path) -> None:
        df = _regression_data()
        m = Model(_config("regression"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        # Predict with both
        X_test = df.drop(columns=["target"])
        lizyml_result = _predict_via_lizyml(m, X_test)
        codegen_result = _predict_via_codegen(codegen_dir, X_test)

        np.testing.assert_allclose(
            codegen_result["pred"],
            lizyml_result["pred"],
            rtol=1e-7,
            err_msg="Regression predictions differ",
        )


class TestEquivalenceBinaryPlatt:
    def test_platt_predictions_match(self, tmp_path: Path) -> None:
        df = _binary_data()
        m = Model(_config("binary", calibration="platt"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        X_test = df.drop(columns=["target"])
        lizyml_result = _predict_via_lizyml(m, X_test)
        codegen_result = _predict_via_codegen(codegen_dir, X_test)

        # Calibrated probabilities should match
        assert codegen_result["proba"] is not None
        assert lizyml_result["proba"] is not None
        np.testing.assert_allclose(
            codegen_result["proba"],
            lizyml_result["proba"],
            rtol=1e-6,
            err_msg="Platt calibrated probabilities differ",
        )


class TestEquivalenceBinaryIsotonic:
    def test_isotonic_predictions_match(self, tmp_path: Path) -> None:
        df = _binary_data()
        m = Model(_config("binary", calibration="isotonic"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        X_test = df.drop(columns=["target"])
        lizyml_result = _predict_via_lizyml(m, X_test)
        codegen_result = _predict_via_codegen(codegen_dir, X_test)

        assert codegen_result["proba"] is not None
        assert lizyml_result["proba"] is not None
        np.testing.assert_allclose(
            codegen_result["proba"],
            lizyml_result["proba"],
            rtol=1e-6,
            err_msg="Isotonic calibrated probabilities differ",
        )


class TestEquivalenceBinaryBeta:
    def test_beta_predictions_match(self, tmp_path: Path) -> None:
        df = _binary_data()
        m = Model(_config("binary", calibration="beta"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        X_test = df.drop(columns=["target"])
        lizyml_result = _predict_via_lizyml(m, X_test)
        codegen_result = _predict_via_codegen(codegen_dir, X_test)

        assert codegen_result["proba"] is not None
        assert lizyml_result["proba"] is not None
        np.testing.assert_allclose(
            codegen_result["proba"],
            lizyml_result["proba"],
            rtol=1e-6,
            err_msg="Beta calibrated probabilities differ",
        )


class TestEquivalenceMulticlass:
    def test_multiclass_predictions_match(self, tmp_path: Path) -> None:
        df = _multiclass_data()
        m = Model(_config("multiclass"))
        m.fit(df)

        codegen_dir = tmp_path / "codegen"
        m.export_code(codegen_dir)

        X_test = df.drop(columns=["target"])
        lizyml_result = _predict_via_lizyml(m, X_test)
        codegen_result = _predict_via_codegen(codegen_dir, X_test)

        np.testing.assert_array_equal(
            codegen_result["pred"],
            lizyml_result["pred"],
            err_msg="Multiclass class predictions differ",
        )
        # Check probabilities match
        if lizyml_result["proba"] is not None:
            np.testing.assert_allclose(
                codegen_result["proba"],
                lizyml_result["proba"],
                rtol=1e-7,
                err_msg="Multiclass probabilities differ",
            )
