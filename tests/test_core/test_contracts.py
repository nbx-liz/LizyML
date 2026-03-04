"""Golden tests for Phase 4 type contracts.

Covers FitResult, PredictionResult, SplitIndices, RunMeta.
Fixes field names, types, and hierarchy of the public result contracts.
Any unintentional schema change will cause these tests to fail.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lizyml.core.types import FitResult, PredictionResult, RunMeta, SplitIndices
from lizyml.data.fingerprint import DataFingerprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _field_names(cls: type) -> list[str]:
    return [f.name for f in dataclasses.fields(cls)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def split_indices() -> SplitIndices:
    outer = [(np.array([0, 1]), np.array([2, 3]))]
    return SplitIndices(outer=outer, inner=None, calibration=None)


@pytest.fixture()
def run_meta() -> RunMeta:
    return RunMeta(
        lizyml_version="0.1.0",
        python_version="3.11.0",
        deps_versions={"lightgbm": "4.0.0"},
        config_normalized={"task": "binary"},
        config_version=1,
        run_id="test-uuid",
        timestamp="2026-03-04T00:00:00",
    )


@pytest.fixture()
def fingerprint() -> DataFingerprint:
    return DataFingerprint(row_count=10, column_hash="abc", file_hash=None)


@pytest.fixture()
def fit_result(
    split_indices: SplitIndices,
    run_meta: RunMeta,
    fingerprint: DataFingerprint,
) -> FitResult:
    oof = np.array([0.1, 0.9, 0.2])
    return FitResult(
        oof_pred=oof,
        if_pred_per_fold=[oof],
        metrics={
            "raw": {
                "oof": {"rmse": 0.5},
                "if_mean": {"rmse": 0.4},
                "if_per_fold": [{"rmse": 0.4}],
            }
        },
        models=[object()],
        history=[{"eval_history": {}, "best_iteration": 10}],
        feature_names=["a", "b"],
        dtypes={"a": "float64", "b": "category"},
        categorical_features=["b"],
        splits=split_indices,
        data_fingerprint=fingerprint,
        pipeline_state=None,
        calibrator=None,
        run_meta=run_meta,
    )


@pytest.fixture()
def prediction_result() -> PredictionResult:
    return PredictionResult(
        pred=np.array([0, 1, 0]),
        proba=np.array([0.1, 0.9, 0.2]),
        shap_values=None,
        used_features=["a", "b"],
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Golden: SplitIndices schema
# ---------------------------------------------------------------------------


class TestSplitIndicesSchema:
    def test_field_names(self) -> None:
        assert _field_names(SplitIndices) == ["outer", "inner", "calibration"]

    def test_instantiation_no_inner_no_calibration(self) -> None:
        si = SplitIndices(
            outer=[(np.array([0]), np.array([1]))],
            inner=None,
            calibration=None,
        )
        assert si.inner is None
        assert si.calibration is None
        assert len(si.outer) == 1

    def test_instantiation_with_inner_and_calibration(self) -> None:
        idx = [(np.array([0]), np.array([1]))]
        si = SplitIndices(outer=idx, inner=idx, calibration=idx)
        assert si.inner is not None
        assert si.calibration is not None


# ---------------------------------------------------------------------------
# Golden: RunMeta schema
# ---------------------------------------------------------------------------


class TestRunMetaSchema:
    def test_field_names(self) -> None:
        expected = [
            "lizyml_version",
            "python_version",
            "deps_versions",
            "config_normalized",
            "config_version",
            "run_id",
            "timestamp",
        ]
        assert _field_names(RunMeta) == expected

    def test_instantiation(self, run_meta: RunMeta) -> None:
        assert run_meta.lizyml_version == "0.1.0"
        assert run_meta.config_version == 1
        assert isinstance(run_meta.deps_versions, dict)
        assert isinstance(run_meta.config_normalized, dict)


# ---------------------------------------------------------------------------
# Golden: FitResult schema
# ---------------------------------------------------------------------------


class TestFitResultSchema:
    def test_field_names(self) -> None:
        expected = [
            "oof_pred",
            "if_pred_per_fold",
            "metrics",
            "models",
            "history",
            "feature_names",
            "dtypes",
            "categorical_features",
            "splits",
            "data_fingerprint",
            "pipeline_state",
            "calibrator",
            "run_meta",
        ]
        assert _field_names(FitResult) == expected

    def test_metrics_raw_structure(self, fit_result: FitResult) -> None:
        raw = fit_result.metrics["raw"]
        assert "oof" in raw
        assert "if_mean" in raw
        assert "if_per_fold" in raw
        assert isinstance(raw["if_per_fold"], list)

    def test_calibrated_key_absent_when_no_calibrator(
        self, fit_result: FitResult
    ) -> None:
        assert fit_result.calibrator is None
        assert "calibrated" not in fit_result.metrics

    def test_oof_pred_is_ndarray(self, fit_result: FitResult) -> None:
        assert isinstance(fit_result.oof_pred, np.ndarray)

    def test_if_pred_per_fold_len_equals_n_splits(self, fit_result: FitResult) -> None:
        assert len(fit_result.if_pred_per_fold) == len(fit_result.splits.outer)

    def test_splits_type(self, fit_result: FitResult) -> None:
        assert isinstance(fit_result.splits, SplitIndices)

    def test_data_fingerprint_type(self, fit_result: FitResult) -> None:
        assert isinstance(fit_result.data_fingerprint, DataFingerprint)

    def test_run_meta_type(self, fit_result: FitResult) -> None:
        assert isinstance(fit_result.run_meta, RunMeta)

    def test_history_has_required_keys(self, fit_result: FitResult) -> None:
        for fold_hist in fit_result.history:
            assert "eval_history" in fold_hist
            assert "best_iteration" in fold_hist


# ---------------------------------------------------------------------------
# Golden: PredictionResult schema
# ---------------------------------------------------------------------------


class TestPredictionResultSchema:
    def test_field_names(self) -> None:
        expected = [
            "pred",
            "proba",
            "shap_values",
            "used_features",
            "warnings",
        ]
        assert _field_names(PredictionResult) == expected

    def test_pred_is_ndarray(self, prediction_result: PredictionResult) -> None:
        assert isinstance(prediction_result.pred, np.ndarray)

    def test_proba_optional(self) -> None:
        r = PredictionResult(
            pred=np.array([1, 2]),
            proba=None,
            shap_values=None,
            used_features=["x"],
            warnings=[],
        )
        assert r.proba is None

    def test_shap_values_optional(self, prediction_result: PredictionResult) -> None:
        assert prediction_result.shap_values is None

    def test_warnings_is_list(self, prediction_result: PredictionResult) -> None:
        assert isinstance(prediction_result.warnings, list)


# ---------------------------------------------------------------------------
# Re-export surface
# ---------------------------------------------------------------------------


class TestReExports:
    def test_imports_from_package(self) -> None:
        from lizyml.core.types import (  # noqa: F401
            FitResult,
            PredictionResult,
            RunMeta,
            SplitIndices,
        )
