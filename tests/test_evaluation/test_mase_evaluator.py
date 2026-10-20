"""MASE in the Evaluator (H-0132, criterion 2).

Each fold's scale comes from that fold's outer-train targets; the pooled OOF
value scales each covered row by the scale of the fold that validated it, which
equals the fold-size-weighted mean of the per-fold values. The fixtures give
the folds different validation sizes and different per-fold values, so an
unweighted mean, or a scale taken from the scored rows, gives another number.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.artifacts import DataFingerprint, RunMeta, SplitIndices
from lizyml.core.types.fit_result import FitResult
from lizyml.evaluation.evaluator import Evaluator

_RUN_META = RunMeta(
    lizyml_version="0.1.0",
    python_version="3.11",
    deps_versions={},
    config_normalized={},
    config_version=1,
    run_id="test-mase",
    timestamp="2026-01-01T00:00:00",
)
_FP = DataFingerprint(row_count=0, column_hash="test")

Outer = list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]


def _outer(pairs: list[tuple[list[int], list[int]]]) -> Outer:
    return [(np.array(t, dtype=np.intp), np.array(v, dtype=np.intp)) for t, v in pairs]


def _fit_result(
    oof_pred: npt.NDArray[np.float64],
    outer: Outer,
    if_pred_per_fold: list[npt.NDArray[np.float64]],
) -> FitResult:
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


def _scale(y: npt.NDArray[np.float64]) -> float:
    return float(np.mean(np.abs(y - y.mean())))


def _mase(y, pred, y_train) -> float:  # type: ignore[no-untyped-def]
    return float(np.mean(np.abs(y - pred)) / _scale(y_train))


# Ten rows; fold 0 validates 4 rows, fold 1 validates 6 rows.
Y = np.array([0.0, 4.0, 1.0, 9.0, 2.0, 30.0, 0.0, 5.0, 8.0, 1.0])
PRED = np.array([1.0, 4.0, 3.0, 8.0, 2.0, 20.0, 1.0, 5.0, 2.0, 1.0])
OUTER = _outer([([4, 5, 6, 7, 8, 9], [0, 1, 2, 3]), ([0, 1, 2, 3], [4, 5, 6, 7, 8, 9])])
IF_PRED = [Y[t] + 0.5 for t, _ in OUTER]


def _evaluate(outer: Outer = OUTER, if_pred=None, y=Y, pred=PRED):  # type: ignore[no-untyped-def]
    if if_pred is None:
        if_pred = [y[t] + 0.5 for t, _ in outer]
    fr = _fit_result(pred, outer, if_pred)
    return Evaluator(task="regression").evaluate(fr, y, ["mase", "rmse"])["raw"]


def test_each_fold_uses_its_own_outer_train_scale() -> None:
    raw = _evaluate()
    for k, (train, valid) in enumerate(OUTER):
        expected = _mase(Y[valid], PRED[valid], Y[train])
        assert raw["oof_per_fold"][k]["mase"] == pytest.approx(expected)
        # The scored rows' own deviation would give another value.
        assert expected != pytest.approx(
            float(np.mean(np.abs(Y[valid] - PRED[valid])) / _scale(Y[valid]))
        )


def test_the_pooled_value_scales_each_row_by_its_folds_scale() -> None:
    raw = _evaluate()
    per_row = np.empty(len(Y))
    for train, valid in OUTER:
        per_row[valid] = np.abs(Y[valid] - PRED[valid]) / _scale(Y[train])
    folds = [raw["oof_per_fold"][k]["mase"] for k in range(len(OUTER))]
    sizes = [len(v) for _, v in OUTER]
    weighted = float(np.average(folds, weights=sizes))
    assert raw["oof"]["mase"] == pytest.approx(float(per_row.mean()))
    assert raw["oof"]["mase"] == pytest.approx(weighted)
    # The fixture separates the weighted from the unweighted mean.
    assert sizes[0] != sizes[1] and folds[0] != pytest.approx(folds[1])
    assert raw["oof"]["mase"] != pytest.approx(float(np.mean(folds)))


def test_in_fold_values_use_the_outer_train_scale() -> None:
    raw = _evaluate()
    for k, (train, _) in enumerate(OUTER):
        expected = _mase(Y[train], IF_PRED[k], Y[train])
        assert raw["if_per_fold"][k]["mase"] == pytest.approx(expected)
    assert raw["if_mean"]["mase"] == pytest.approx(
        float(np.mean([f["mase"] for f in raw["if_per_fold"]]))
    )


def test_other_metrics_are_computed_as_before() -> None:
    raw = _evaluate()
    assert raw["oof"]["rmse"] == pytest.approx(float(np.sqrt(np.mean((Y - PRED) ** 2))))


def test_partial_coverage_divides_by_the_covered_rows() -> None:
    """Time-ordered folds leave rows 0-2 uncovered."""
    outer = _outer([([0, 1, 2], [3, 4, 5]), ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9])])
    pred = PRED.copy()
    pred[:3] = np.nan
    raw = _evaluate(outer=outer, pred=pred)
    covered = np.concatenate([v for _, v in outer])
    per_row = np.concatenate([np.abs(Y[v] - pred[v]) / _scale(Y[t]) for t, v in outer])
    assert len(covered) == 7
    assert raw["oof"]["mase"] == pytest.approx(float(per_row.sum() / 7))
    assert raw["oof_coverage"] == pytest.approx(0.7)


def test_overlapping_validation_folds_are_refused() -> None:
    outer = _outer([([5, 6, 7, 8, 9], [0, 1, 2]), ([0, 6, 7, 8, 9], [2, 3, 4, 5])])
    with pytest.raises(LizyMLError) as caught:
        _evaluate(outer=outer)
    assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC
    assert caught.value.context["metric"] == "mase"
    assert caught.value.context["overlapping_rows"] == 1
    assert caught.value.context["first_row"] == 2


def test_overlapping_folds_still_evaluate_metrics_that_need_no_training_targets() -> (
    None
):
    outer = _outer([([5, 6, 7, 8, 9], [0, 1, 2]), ([0, 6, 7, 8, 9], [2, 3, 4, 5])])
    fr = _fit_result(PRED, outer, [Y[t] for t, _ in outer])
    raw = Evaluator(task="regression").evaluate(fr, Y, ["rmse"])["raw"]
    assert np.isfinite(raw["oof"]["rmse"])


def test_a_constant_outer_train_target_is_refused() -> None:
    y = Y.copy()
    y[[4, 5, 6, 7, 8, 9]] = 3.0  # fold 0's outer-train rows
    with pytest.raises(LizyMLError) as caught:
        _evaluate(y=y)
    assert caught.value.code == ErrorCode.UNSUPPORTED_METRIC
    assert caught.value.context["metric"] == "mase"
