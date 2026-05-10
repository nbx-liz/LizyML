"""Tests for codegen feval metric support (H-0066).

Covers:
- config.json feval_metrics field
- Backward compatibility (empty feval_metrics)
- Numerical equivalence between codegen feval and LizyML metric implementations
- feval metadata extraction from adapter params
- Template contains feval infrastructure
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from lizyml.codegen.config_writer import build_config

# ── Helpers ──────────────────────────────────────────────────


def _make_run_meta(
    *,
    task: str = "binary",
    target_col: str = "y",
) -> dict[str, Any]:
    return {
        "lizyml_version": "0.7.3",
        "run_id": "test-run-id",
        "timestamp": "2026-04-02T00:00:00",
        "config_normalized": {
            "task": task,
            "data": {"target_col": target_col},
        },
    }


def _make_lgbm_params() -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "None",
        "num_leaves": 31,
        "verbosity": -1,
    }


def _sample_feval_metrics() -> list[dict[str, Any]]:
    return [
        {
            "name": "f1",
            "params": {},
            "greater_is_better": True,
            "needs_proba": False,
        },
        {
            "name": "precision_at_k",
            "params": {"k": 20},
            "greater_is_better": True,
            "needs_proba": True,
        },
    ]


# ── config.json feval_metrics field ──────────────────────────


class TestConfigFeval:
    """Tests for feval_metrics in config.json."""

    def test_feval_metrics_present_in_config(self) -> None:
        fevals = _sample_feval_metrics()
        config = build_config(
            run_meta=_make_run_meta(),
            feature_names=["a", "b"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=100,
            early_stopping_rounds=10,
            validation_ratio=0.2,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
            feval_metrics=fevals,
        )
        assert "feval_metrics" in config
        assert len(config["feval_metrics"]) == 2
        assert config["feval_metrics"][0]["name"] == "f1"
        assert config["feval_metrics"][1]["name"] == "precision_at_k"
        assert config["feval_metrics"][1]["params"] == {"k": 20}

    def test_feval_metrics_empty_when_none(self) -> None:
        config = build_config(
            run_meta=_make_run_meta(),
            feature_names=["a"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=100,
            early_stopping_rounds=None,
            validation_ratio=0.0,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
        )
        assert config["feval_metrics"] == []

    def test_feval_metrics_empty_list_explicit(self) -> None:
        config = build_config(
            run_meta=_make_run_meta(),
            feature_names=["a"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=100,
            early_stopping_rounds=None,
            validation_ratio=0.0,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
            feval_metrics=[],
        )
        assert config["feval_metrics"] == []

    def test_feval_metrics_json_serializable(self) -> None:
        fevals = _sample_feval_metrics()
        config = build_config(
            run_meta=_make_run_meta(),
            feature_names=["a"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=100,
            early_stopping_rounds=None,
            validation_ratio=0.0,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
            feval_metrics=fevals,
        )
        # Must not raise
        serialized = json.dumps(config)
        roundtrip = json.loads(serialized)
        assert roundtrip["feval_metrics"] == fevals

    def test_feval_metrics_key_ordering(self) -> None:
        """feval_metrics comes after seed and before calibration_method."""
        fevals = _sample_feval_metrics()
        config = build_config(
            run_meta=_make_run_meta(),
            feature_names=["a"],
            categorical_features=[],
            lgbm_params=_make_lgbm_params(),
            num_boost_round=100,
            early_stopping_rounds=None,
            validation_ratio=0.0,
            seed=42,
            calibration_method=None,
            calibration_n_splits=5,
            feval_metrics=fevals,
        )
        keys = list(config.keys())
        assert keys.index("feval_metrics") < keys.index("calibration_method")
        assert keys.index("seed") < keys.index("feval_metrics")


# ── Numerical equivalence: codegen feval vs LizyML metrics ──


class TestFevalNumericalEquivalence:
    """Each codegen feval must match the LizyML metric within rtol=1e-10."""

    @pytest.fixture()
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    @pytest.fixture()
    def binary_data(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        y_true = rng.integers(0, 2, size=200).astype(np.float64)
        y_pred = rng.uniform(0, 1, size=200).astype(np.float64)
        return y_true, y_pred

    @pytest.fixture()
    def regression_data(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        y_true = rng.uniform(0, 10, size=200).astype(np.float64)
        y_pred = rng.uniform(0, 10, size=200).astype(np.float64)
        return y_true, y_pred

    @pytest.fixture()
    def multiclass_proba(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n, k = 200, 4
        y_true = rng.integers(0, k, size=n).astype(np.float64)
        logits = rng.standard_normal((n, k))
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y_pred = e_x / e_x.sum(axis=1, keepdims=True)
        return y_true, y_pred

    @pytest.fixture(autouse=True, scope="class")
    def _feval_ns(self) -> None:
        """Extract only the feval functions from the train.py template."""
        import re

        from lizyml.codegen.templates import render_train_py

        source = render_train_py()
        # Extract feval function block: from _sigmoid through _FEVAL_REGISTRY
        pattern = (
            r"(def _sigmoid\(x.*?\n)"  # _sigmoid (already exists for calibration)
            r"(.*?)"  # everything between
            r"(_FEVAL_REGISTRY.*?\})"  # up to end of registry
        )
        match = re.search(pattern, source, re.DOTALL)
        assert match is not None, "Could not find feval block in template"
        feval_block = match.group(0)

        # Build a minimal namespace with required imports
        ns: dict[str, Any] = {"__builtins__": __builtins__}
        exec("import numpy as np", ns)  # noqa: S102
        exec(compile(feval_block, "<feval_block>", "exec"), ns)  # noqa: S102
        type(self)._feval_ns_dict = ns

    def _get_codegen_fn(self, name: str):  # noqa: ANN202
        return self._feval_ns_dict[f"_feval_{name}"]

    def _get_lizyml_metric(self, name: str, **kwargs: Any):  # noqa: ANN202
        from lizyml.metrics.registry import get_metric

        return get_metric(name, **kwargs)

    def test_rmsle(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = regression_data
        codegen_fn = self._get_codegen_fn("rmsle")
        lizyml_metric = self._get_lizyml_metric("rmsle")
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_r2(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = regression_data
        codegen_fn = self._get_codegen_fn("r2")
        lizyml_metric = self._get_lizyml_metric("r2")
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_smape(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """H-0071: codegen sMAPE numerically matches lizyml.metrics.SMAPE."""
        y_true, y_pred = regression_data
        codegen_fn = self._get_codegen_fn("smape")
        lizyml_metric = self._get_lizyml_metric("smape")
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_smape_with_zero_zero_row(self) -> None:
        """H-0071: codegen sMAPE handles |y_true|+|y_pred|==0 row."""
        codegen_fn = self._get_codegen_fn("smape")
        lizyml_metric = self._get_lizyml_metric("smape")
        y_true = np.array([0.0, 5.0, 4.0])
        y_pred = np.array([0.0, 6.0, 6.0])
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_wape(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """H-0071: codegen WAPE numerically matches lizyml.metrics.WAPE."""
        y_true, y_pred = regression_data
        codegen_fn = self._get_codegen_fn("wape")
        lizyml_metric = self._get_lizyml_metric("wape")
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_wape_all_zero_y_true_raises(self) -> None:
        """H-0071: codegen WAPE raises on sum(|y_true|)==0."""
        codegen_fn = self._get_codegen_fn("wape")
        with pytest.raises(ValueError, match="WAPE is undefined"):
            codegen_fn(np.array([0.0, 0.0]), np.array([1.0, 2.0]))

    def test_f1_binary(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = binary_data
        codegen_fn = self._get_codegen_fn("f1")
        lizyml_metric = self._get_lizyml_metric("f1")
        # Both use threshold at 0.5
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_f1_multiclass(
        self, multiclass_proba: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_proba
        codegen_fn = self._get_codegen_fn("f1")
        lizyml_metric = self._get_lizyml_metric("f1")
        # Convert proba to hard labels (same as what feval wrapper does)
        pred = y_pred.argmax(axis=1).astype(np.int64)
        np.testing.assert_allclose(
            codegen_fn(y_true, pred),
            lizyml_metric(y_true, pred),
            rtol=1e-10,
        )

    def test_brier_binary(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = binary_data
        codegen_fn = self._get_codegen_fn("brier")
        lizyml_metric = self._get_lizyml_metric("brier")
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_brier_multiclass(
        self, multiclass_proba: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_proba
        codegen_fn = self._get_codegen_fn("brier")
        lizyml_metric = self._get_lizyml_metric("brier")
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_ece(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = binary_data
        codegen_fn = self._get_codegen_fn("ece")
        lizyml_metric = self._get_lizyml_metric("ece")
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_precision_at_k(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = binary_data
        codegen_fn = self._get_codegen_fn("precision_at_k")
        lizyml_metric = self._get_lizyml_metric("precision_at_k", k=20)
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred, k=20),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_accuracy_binary(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = binary_data
        codegen_fn = self._get_codegen_fn("accuracy")
        lizyml_metric = self._get_lizyml_metric("accuracy")
        np.testing.assert_allclose(
            codegen_fn(y_true, y_pred),
            lizyml_metric(y_true, y_pred),
            rtol=1e-10,
        )

    def test_accuracy_multiclass(
        self, multiclass_proba: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_proba
        codegen_fn = self._get_codegen_fn("accuracy")
        lizyml_metric = self._get_lizyml_metric("accuracy")
        pred = y_pred.argmax(axis=1).astype(np.int64)
        np.testing.assert_allclose(
            codegen_fn(y_true, pred),
            lizyml_metric(y_true, pred),
            rtol=1e-10,
        )


# ── Template content checks ─────────────────────────────────


class TestTemplateFevalContent:
    """Verify train.py template contains feval infrastructure."""

    def test_contains_feval_registry(self) -> None:
        from lizyml.codegen.templates import render_train_py

        src = render_train_py()
        assert "_FEVAL_REGISTRY" in src

    def test_contains_build_feval_from_config(self) -> None:
        from lizyml.codegen.templates import render_train_py

        src = render_train_py()
        assert "build_feval_from_config" in src

    def test_contains_softmax(self) -> None:
        from lizyml.codegen.templates import render_train_py

        src = render_train_py()
        assert "_softmax" in src

    def test_train_lgbm_uses_feval(self) -> None:
        from lizyml.codegen.templates import render_train_py

        src = render_train_py()
        assert "feval=fevals" in src

    def test_contains_all_feval_functions(self) -> None:
        from lizyml.codegen.templates import render_train_py

        src = render_train_py()
        expected = [
            "_feval_rmsle",
            "_feval_r2",
            "_feval_f1",
            "_feval_brier",
            "_feval_ece",
            "_feval_precision_at_k",
            "_feval_accuracy",
            # H-0071: zero-tolerant percentage-style regression metrics
            "_feval_smape",
            "_feval_wape",
        ]
        for name in expected:
            assert name in src, f"{name} not found in train.py template"


# ── Feval metadata extraction ────────────────────────────────


class TestExtractFevalMetadata:
    """Tests for _extract_feval_metadata in lgbm.provider (moved in H-0073)."""

    def _make_adapter(
        self,
        task: str = "binary",
        params: dict[str, Any] | None = None,
    ):  # noqa: ANN202
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        return LGBMAdapter(
            task=task,
            params=params or {},
            random_state=42,
        )

    def test_no_metric_returns_empty(self) -> None:
        from lizyml.estimators.lgbm.provider import _extract_feval_metadata

        adapter = self._make_adapter(params={})
        assert _extract_feval_metadata(adapter) == []

    def test_native_only_returns_empty(self) -> None:
        from lizyml.estimators.lgbm.provider import _extract_feval_metadata

        adapter = self._make_adapter(params={"metric": ["auc", "binary_logloss"]})
        assert _extract_feval_metadata(adapter) == []

    def test_feval_metric_extracted(self) -> None:
        from lizyml.estimators.lgbm.provider import _extract_feval_metadata

        adapter = self._make_adapter(params={"metric": ["auc", "f1"]})
        result = _extract_feval_metadata(adapter)
        assert len(result) == 1
        assert result[0]["name"] == "f1"
        assert result[0]["greater_is_better"] is True
        assert result[0]["needs_proba"] is False
        assert result[0]["params"] == {}

    def test_feval_with_params(self) -> None:
        from lizyml.estimators.lgbm.provider import _extract_feval_metadata

        adapter = self._make_adapter(params={"metric": [{"precision_at_k": {"k": 20}}]})
        result = _extract_feval_metadata(adapter)
        assert len(result) == 1
        assert result[0]["name"] == "precision_at_k"
        assert result[0]["params"] == {"k": 20}
        assert result[0]["greater_is_better"] is True
        assert result[0]["needs_proba"] is True

    def test_mixed_native_and_feval(self) -> None:
        from lizyml.estimators.lgbm.provider import _extract_feval_metadata

        adapter = self._make_adapter(params={"metric": ["auc", "f1", "brier"]})
        result = _extract_feval_metadata(adapter)
        names = [m["name"] for m in result]
        assert "f1" in names
        assert "brier" in names
        assert "auc" not in names

    def test_regression_feval(self) -> None:
        from lizyml.estimators.lgbm.provider import _extract_feval_metadata

        adapter = self._make_adapter(
            task="regression", params={"metric": ["rmse", "rmsle"]}
        )
        result = _extract_feval_metadata(adapter)
        assert len(result) == 1
        assert result[0]["name"] == "rmsle"
        assert result[0]["greater_is_better"] is False

    def test_string_metric(self) -> None:
        from lizyml.estimators.lgbm.provider import _extract_feval_metadata

        adapter = self._make_adapter(params={"metric": "f1"})
        result = _extract_feval_metadata(adapter)
        assert len(result) == 1
        assert result[0]["name"] == "f1"

    def test_r2_regression_feval(self) -> None:
        from lizyml.estimators.lgbm.provider import _extract_feval_metadata

        adapter = self._make_adapter(
            task="regression", params={"metric": ["rmse", "r2"]}
        )
        result = _extract_feval_metadata(adapter)
        assert len(result) == 1
        assert result[0]["name"] == "r2"
        assert result[0]["greater_is_better"] is True
        assert result[0]["needs_proba"] is False


# ── build_feval_from_config() unit tests ─────────────────────


class TestBuildFevalFromConfig:
    """Tests for build_feval_from_config in the generated train.py template.

    Extracts the function via exec and invokes it with mocked CFG globals.
    """

    @staticmethod
    def _extract_feval_block() -> str:
        """Extract the feval block + build_feval_from_config from template."""
        import re

        from lizyml.codegen.templates import render_train_py

        source = render_train_py()
        # Match from _sigmoid to the end of build_feval_from_config
        pattern = (
            r"(def _sigmoid\(x.*?\n)"
            r"(.*?)"
            r"(def build_feval_from_config.*?return fevals\n)"
        )
        match = re.search(pattern, source, re.DOTALL)
        assert match is not None
        return match.group(0)

    def _build(
        self,
        feval_metrics: list[dict[str, Any]],
        task: str = "binary",
        num_class: int = 2,
    ) -> list[Any]:
        """Build feval list from mocked CFG."""
        block = self._extract_feval_block()
        ns: dict[str, Any] = {}
        exec("import numpy as np", ns)  # noqa: S102
        exec("import logging; log = logging.getLogger('test')", ns)  # noqa: S102
        ns["CFG"] = {
            "_task": task,
            "feval_metrics": feval_metrics,
            "lgbm_params": {"num_class": num_class},
        }
        exec(compile(block, "<feval>", "exec"), ns)  # noqa: S102
        return ns["build_feval_from_config"]()

    def test_empty_feval_metrics_returns_empty(self) -> None:
        result = self._build(feval_metrics=[])
        assert result == []

    def test_missing_feval_metrics_key_returns_empty(self) -> None:
        block = self._extract_feval_block()
        ns: dict[str, Any] = {}
        exec("import numpy as np", ns)  # noqa: S102
        exec("import logging; log = logging.getLogger('test')", ns)  # noqa: S102
        ns["CFG"] = {"_task": "binary", "lgbm_params": {}}
        exec(compile(block, "<feval>", "exec"), ns)  # noqa: S102
        result = ns["build_feval_from_config"]()
        assert result == []

    def test_single_feval_returns_callable(self) -> None:
        fevals = self._build(
            feval_metrics=[
                {
                    "name": "f1",
                    "params": {},
                    "greater_is_better": True,
                    "needs_proba": False,
                }
            ]
        )
        assert len(fevals) == 1
        assert callable(fevals[0])

    def test_unknown_metric_skipped(self) -> None:
        fevals = self._build(
            feval_metrics=[
                {
                    "name": "nonexistent_metric",
                    "params": {},
                    "greater_is_better": True,
                    "needs_proba": False,
                }
            ]
        )
        assert fevals == []

    def test_multiple_fevals(self) -> None:
        fevals = self._build(
            feval_metrics=[
                {
                    "name": "f1",
                    "params": {},
                    "greater_is_better": True,
                    "needs_proba": False,
                },
                {
                    "name": "brier",
                    "params": {},
                    "greater_is_better": False,
                    "needs_proba": True,
                },
            ]
        )
        assert len(fevals) == 2
