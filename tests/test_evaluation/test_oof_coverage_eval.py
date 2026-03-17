"""Integration tests for Evaluator OOF coverage mask (H-0057).

Uses synthetic FitResult objects with hand-crafted SplitIndices to test:
- Full coverage (KFold): backward-compatible, oof_coverage == 1.0
- Partial coverage (TimeSeriesCV): oof metrics finite, oof_coverage < 1.0
- NaN in covered rows: ValueError (bug detection)
- NaN in uncovered rows: silently excluded (expected)
- oof_per_fold / IF metrics unchanged
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from lizyml.core.types.artifacts import DataFingerprint, RunMeta, SplitIndices
from lizyml.core.types.fit_result import FitResult
from lizyml.evaluation.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_META = RunMeta(
    lizyml_version="0.1.0",
    python_version="3.11",
    deps_versions={},
    config_normalized={},
    config_version=1,
    run_id="test-coverage",
    timestamp="2026-01-01T00:00:00",
)

_FP = DataFingerprint(row_count=0, column_hash="test")


def _make_outer(
    pairs: list[tuple[list[int], list[int]]],
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    return [(np.array(t, dtype=np.intp), np.array(v, dtype=np.intp)) for t, v in pairs]


def _make_fit_result(
    oof_pred: npt.NDArray[np.float64],
    outer: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]],
    if_pred_per_fold: list[npt.NDArray[np.float64]] | None = None,
) -> FitResult:
    """Build a minimal synthetic FitResult for Evaluator testing."""
    if if_pred_per_fold is None:
        if_pred_per_fold = [oof_pred[train_idx] for train_idx, _ in outer]
    return FitResult(
        oof_pred=oof_pred,
        if_pred_per_fold=if_pred_per_fold,
        metrics={},
        models=[],
        history=[],
        feature_names=["a"],
        dtypes={"a": "float64"},
        categorical_features=[],
        splits=SplitIndices(outer=outer, inner=None, calibration=None),
        data_fingerprint=_FP,
        pipeline_state=None,
        calibrator=None,
        run_meta=_RUN_META,
    )


# ---------------------------------------------------------------------------
# KFold (full coverage) — backward compatibility
# ---------------------------------------------------------------------------


class TestEvaluatorFullCoverageKFold:
    """KFold splits where all rows are covered."""

    @pytest.fixture()
    def kfold_setup(self):
        rng = np.random.default_rng(42)
        n = 12
        y = rng.uniform(0, 10, n)
        oof_pred = y + rng.normal(0, 0.5, n)  # noisy predictions
        outer = _make_outer(
            [
                ([4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3]),
                ([0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7]),
                ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]),
            ]
        )
        fr = _make_fit_result(oof_pred, outer)
        return fr, y

    def test_oof_metrics_finite(self, kfold_setup) -> None:
        fr, y = kfold_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["rmse", "mae"])
        for v in out["raw"]["oof"].values():
            assert np.isfinite(v)

    def test_oof_coverage_is_one(self, kfold_setup) -> None:
        fr, y = kfold_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["rmse"])
        assert out["raw"]["oof_coverage"] == 1.0

    def test_raw_keys_include_oof_coverage(self, kfold_setup) -> None:
        fr, y = kfold_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["rmse"])
        assert "oof_coverage" in out["raw"]

    def test_oof_per_fold_unchanged(self, kfold_setup) -> None:
        """oof_per_fold still uses valid_idx, not mask."""
        fr, y = kfold_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["rmse"])
        assert len(out["raw"]["oof_per_fold"]) == 3
        for fold_dict in out["raw"]["oof_per_fold"]:
            assert "rmse" in fold_dict
            assert np.isfinite(fold_dict["rmse"])

    def test_if_metrics_unchanged(self, kfold_setup) -> None:
        """IF metrics are completely independent of OOF coverage."""
        fr, y = kfold_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["rmse"])
        assert "if_mean" in out["raw"]
        assert np.isfinite(out["raw"]["if_mean"]["rmse"])
        assert len(out["raw"]["if_per_fold"]) == 3


# ---------------------------------------------------------------------------
# TimeSeriesCV (partial coverage)
# ---------------------------------------------------------------------------


class TestEvaluatorPartialCoverageTimeSeries:
    """TimeSeriesCV-like splits where first rows are uncovered."""

    @pytest.fixture()
    def ts_setup(self):
        rng = np.random.default_rng(42)
        n = 12
        y = rng.uniform(0, 10, n)
        # Rows 0-2 never in valid -> their OOF stays NaN
        oof_pred = np.full(n, np.nan)
        oof_pred[3:] = y[3:] + rng.normal(0, 0.5, 9)
        outer = _make_outer(
            [
                ([0, 1, 2], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
                ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]),
            ]
        )
        fr = _make_fit_result(oof_pred, outer)
        return fr, y

    def test_oof_metrics_finite(self, ts_setup) -> None:
        fr, y = ts_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["rmse", "mae"])
        for v in out["raw"]["oof"].values():
            assert np.isfinite(v), f"OOF metric should be finite, got {v}"

    def test_oof_coverage_less_than_one(self, ts_setup) -> None:
        fr, y = ts_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["rmse"])
        assert out["raw"]["oof_coverage"] == pytest.approx(9 / 12)

    def test_oof_coverage_type(self, ts_setup) -> None:
        fr, y = ts_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["rmse"])
        assert isinstance(out["raw"]["oof_coverage"], float)

    def test_oof_computed_on_covered_rows_only(self, ts_setup) -> None:
        """OOF metrics should be computed on rows 3-11 only, not 0-11."""
        fr, y = ts_setup
        ev = Evaluator(task="regression")
        out = ev.evaluate(fr, y, ["mae"])
        # Manually compute expected MAE on covered rows
        covered_y = y[3:]
        covered_pred = fr.oof_pred[3:]
        expected_mae = float(np.mean(np.abs(covered_y - covered_pred)))
        assert out["raw"]["oof"]["mae"] == pytest.approx(expected_mae)


# ---------------------------------------------------------------------------
# NaN assertion in covered rows (bug detection)
# ---------------------------------------------------------------------------


class TestEvaluatorNanBugDetection:
    """NaN in covered rows must raise ValueError."""

    def test_nan_in_covered_rows_raises(self) -> None:
        n = 9
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 10, n)
        oof_pred = y + rng.normal(0, 0.5, n)
        # Inject NaN in a covered row (row 3 is in valid of fold 0)
        oof_pred[3] = np.nan
        outer = _make_outer(
            [
                ([0, 1, 2], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
            ]
        )
        fr = _make_fit_result(oof_pred, outer)
        ev = Evaluator(task="regression")
        with pytest.raises(ValueError, match="covered by validation folds"):
            ev.evaluate(fr, y, ["rmse"])

    def test_nan_in_uncovered_rows_ok(self) -> None:
        """NaN in structurally uncovered rows is expected and must not raise."""
        n = 9
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 10, n)
        oof_pred = np.full(n, np.nan)
        # Only fill covered rows (3-8)
        oof_pred[3:] = y[3:] + rng.normal(0, 0.5, 6)
        outer = _make_outer(
            [
                ([0, 1, 2], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
            ]
        )
        fr = _make_fit_result(oof_pred, outer)
        ev = Evaluator(task="regression")
        # Should not raise
        out = ev.evaluate(fr, y, ["rmse"])
        assert np.isfinite(out["raw"]["oof"]["rmse"])
