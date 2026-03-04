"""Tests for Phase 9 — Training Core (CVTrainer, RefitTrainer, InnerValid, OOF).

Covers:
- InnerValid: split output shapes and correctness
- OOF helpers: init, fill, leakage guard
- CVTrainer: OOF shape, fold count, history, reproducibility
- OOF leakage test: train-fold rows must not produce their own OOF predictions
- RefitTrainer: produces a fitted model on full data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.types.artifacts import RunMeta
from lizyml.core.types.fit_result import FitResult
from lizyml.data.fingerprint import DataFingerprint
from lizyml.estimators.lgbm import LGBMAdapter
from lizyml.evaluation.oof import fill_oof, init_oof
from lizyml.features.pipelines_native import NativeFeaturePipeline
from lizyml.splitters.kfold import KFoldSplitter
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.inner_valid import HoldoutInnerValid, NoInnerValid
from lizyml.training.refit_trainer import RefitResult, RefitTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_meta() -> RunMeta:
    return RunMeta(
        lizyml_version="0.1.0",
        python_version="3.11",
        deps_versions={},
        config_normalized={},
        config_version=1,
        run_id="test-run",
        timestamp="2026-01-01T00:00:00",
    )


def _make_fingerprint(X: pd.DataFrame) -> DataFingerprint:
    from lizyml.data.fingerprint import compute

    return compute(X, file_path=None)


def _reg_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-1, 1, n)})
    y = pd.Series(X["a"] * 2.0 + X["b"] + rng.normal(0, 0.1, n), name="target")
    return X, y


def _bin_data(n: int = 200, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-1, 1, n)})
    y = pd.Series((X["a"] > 5).astype(int), name="target")
    return X, y


def _multi_data(
    n: int = 300, n_classes: int = 3, seed: int = 2
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-1, 1, n)})
    y = pd.Series(
        pd.cut(X["a"], bins=n_classes, labels=list(range(n_classes))).astype(int),
        name="target",
    )
    return X, y


def _make_cv_trainer(
    n_splits: int = 3, task: str = "regression", n_classes: int | None = None
) -> CVTrainer:
    return CVTrainer(
        outer_splitter=KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=42),
        inner_valid=NoInnerValid(),
        pipeline_factory=NativeFeaturePipeline,
        estimator_factory=lambda: LGBMAdapter(
            task=task,  # type: ignore[arg-type]
            num_class=n_classes if task == "multiclass" else None,
            params={"n_estimators": 20},
            random_state=0,
        ),
        task=task,  # type: ignore[arg-type]
        n_classes=n_classes,
    )


# ---------------------------------------------------------------------------
# InnerValid
# ---------------------------------------------------------------------------


class TestInnerValid:
    def test_no_inner_valid_returns_none(self) -> None:
        iv = NoInnerValid()
        assert iv.split(100) is None

    def test_holdout_split_sizes(self) -> None:
        iv = HoldoutInnerValid(ratio=0.2, random_state=0)
        result = iv.split(100)
        assert result is not None
        train_idx, valid_idx = result
        assert len(train_idx) + len(valid_idx) == 100
        assert len(valid_idx) == 20  # 20% of 100

    def test_holdout_no_overlap(self) -> None:
        iv = HoldoutInnerValid(ratio=0.1, random_state=42)
        result = iv.split(50)
        assert result is not None
        train_idx, valid_idx = result
        assert len(set(train_idx) & set(valid_idx)) == 0

    def test_holdout_reproducible(self) -> None:
        iv1 = HoldoutInnerValid(ratio=0.2, random_state=7)
        iv2 = HoldoutInnerValid(ratio=0.2, random_state=7)
        r1 = iv1.split(100)
        r2 = iv2.split(100)
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])

    def test_holdout_invalid_ratio(self) -> None:
        with pytest.raises(ValueError):
            HoldoutInnerValid(ratio=1.5)


# ---------------------------------------------------------------------------
# OOF helpers
# ---------------------------------------------------------------------------


class TestOOFHelpers:
    def test_init_oof_regression(self) -> None:
        oof = init_oof(100, "regression")
        assert oof.shape == (100,)
        assert np.all(np.isnan(oof))

    def test_init_oof_binary(self) -> None:
        oof = init_oof(50, "binary")
        assert oof.shape == (50,)

    def test_init_oof_multiclass(self) -> None:
        oof = init_oof(60, "multiclass", n_classes=3)
        assert oof.shape == (60, 3)

    def test_init_oof_multiclass_requires_n_classes(self) -> None:
        with pytest.raises(ValueError):
            init_oof(10, "multiclass")

    def test_fill_oof_correct_indices(self) -> None:
        oof = init_oof(10, "regression")
        valid_idx = np.array([2, 5, 7])
        fold_pred = np.array([0.1, 0.2, 0.3])
        fill_oof(oof, valid_idx, fold_pred)
        assert oof[2] == pytest.approx(0.1)
        assert oof[5] == pytest.approx(0.2)
        assert oof[7] == pytest.approx(0.3)
        # Other positions remain NaN
        assert np.isnan(oof[0])

    def test_fill_oof_shape_mismatch(self) -> None:
        oof = init_oof(10, "regression")
        with pytest.raises(ValueError):
            fill_oof(oof, np.array([0, 1, 2]), np.array([0.1, 0.2]))


# ---------------------------------------------------------------------------
# CVTrainer — regression
# ---------------------------------------------------------------------------


class TestCVTrainerRegression:
    def test_oof_shape(self) -> None:
        X, y = _reg_data()
        trainer = _make_cv_trainer(n_splits=3, task="regression")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        assert isinstance(result, FitResult)
        assert result.oof_pred.shape == (len(X),)

    def test_oof_no_nan(self) -> None:
        X, y = _reg_data()
        trainer = _make_cv_trainer(n_splits=3, task="regression")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        assert not np.any(np.isnan(result.oof_pred))

    def test_fold_count(self) -> None:
        X, y = _reg_data()
        trainer = _make_cv_trainer(n_splits=4, task="regression")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        assert len(result.models) == 4
        assert len(result.history) == 4
        assert len(result.if_pred_per_fold) == 4

    def test_if_pred_shapes(self) -> None:
        X, y = _reg_data(n=120)
        n_splits = 3
        trainer = _make_cv_trainer(n_splits=n_splits, task="regression")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        # IF predictions cover the train fold (n_samples - n_valid per fold)
        for i, (train_idx, _) in enumerate(result.splits.outer):
            assert result.if_pred_per_fold[i].shape == (len(train_idx),)

    def test_feature_names_stored(self) -> None:
        X, y = _reg_data()
        trainer = _make_cv_trainer(task="regression")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        assert result.feature_names == ["a", "b"]

    def test_metrics_empty(self) -> None:
        X, y = _reg_data()
        trainer = _make_cv_trainer(task="regression")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        # Metrics are populated by Evaluator (Phase 10), not CVTrainer
        assert result.metrics == {}

    def test_reproducibility(self) -> None:
        X, y = _reg_data()
        trainer_a = _make_cv_trainer(task="regression")
        trainer_b = _make_cv_trainer(task="regression")
        fp = _make_fingerprint(X)
        rm = _make_run_meta()
        result_a = trainer_a.fit(X, y, data_fingerprint=fp, run_meta=rm)
        result_b = trainer_b.fit(X, y, data_fingerprint=fp, run_meta=rm)
        np.testing.assert_array_almost_equal(result_a.oof_pred, result_b.oof_pred)


# ---------------------------------------------------------------------------
# CVTrainer — OOF leakage test (MUST: same-row leakage must not occur)
# ---------------------------------------------------------------------------


class TestOOFLeakage:
    def test_oof_only_written_by_valid_fold(self) -> None:
        """For each fold, the OOF at train_idx must NOT be written by that fold's model.

        We verify this by checking that oof[train_idx] was written by OTHER folds'
        models (i.e. every position in oof_pred is non-NaN after all folds complete,
        and oof_pred[valid_idx] matches fold predictions exactly).
        """
        X, y = _reg_data(n=90)
        n_splits = 3
        trainer = _make_cv_trainer(n_splits=n_splits, task="regression")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        # Every sample covered exactly once
        all_valid = np.concatenate([v for _, v in result.splits.outer])
        assert len(np.unique(all_valid)) == len(X)
        assert not np.any(np.isnan(result.oof_pred))

    def test_oof_pred_matches_fold_valid_predictions(self) -> None:
        """Directly verify OOF == fold model's predictions on valid set."""
        X, y = _reg_data(n=60)
        n_splits = 3
        trainer = _make_cv_trainer(n_splits=n_splits, task="regression")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        # Re-predict on valid set from each saved model to verify OOF correctness
        for i, (train_idx, valid_idx) in enumerate(result.splits.outer):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            X_valid = X.iloc[valid_idx].reset_index(drop=True)

            pipeline = NativeFeaturePipeline()
            pipeline.fit(X_train, y_train)
            X_valid_t = pipeline.transform(X_valid)

            expected = result.models[i].predict(X_valid_t)
            np.testing.assert_array_almost_equal(
                result.oof_pred[valid_idx],
                expected,
                err_msg=f"OOF mismatch on fold {i}",
            )


# ---------------------------------------------------------------------------
# CVTrainer — binary classification
# ---------------------------------------------------------------------------


class TestCVTrainerBinary:
    def test_oof_shape_binary(self) -> None:
        X, y = _bin_data()
        trainer = _make_cv_trainer(n_splits=3, task="binary")
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        assert result.oof_pred.shape == (len(X),)
        # Binary OOF are probabilities in [0, 1]
        assert np.all(result.oof_pred >= 0) and np.all(result.oof_pred <= 1)


# ---------------------------------------------------------------------------
# CVTrainer — multiclass
# ---------------------------------------------------------------------------


class TestCVTrainerMulticlass:
    def test_oof_shape_multiclass(self) -> None:
        X, y = _multi_data(n_classes=3)
        trainer = _make_cv_trainer(n_splits=3, task="multiclass", n_classes=3)
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        assert result.oof_pred.shape == (len(X), 3)
        # Each row sums to ~1
        assert np.allclose(result.oof_pred.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# CVTrainer with HoldoutInnerValid (early stopping)
# ---------------------------------------------------------------------------


class TestCVTrainerWithInnerValid:
    def test_early_stopping_runs(self) -> None:
        X, y = _reg_data(n=200)
        trainer = CVTrainer(
            outer_splitter=KFoldSplitter(n_splits=3, shuffle=True, random_state=0),
            inner_valid=HoldoutInnerValid(ratio=0.15, random_state=99),
            pipeline_factory=NativeFeaturePipeline,
            estimator_factory=lambda: LGBMAdapter(
                task="regression",
                params={"n_estimators": 200, "learning_rate": 0.1},
                early_stopping_rounds=10,
                random_state=0,
            ),
            task="regression",
        )
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        assert result.oof_pred.shape == (len(X),)
        # At least one fold should have early stopped (best_iteration < 200)
        best_iters = [h["best_iteration"] for h in result.history]
        assert any(bi is not None and bi < 200 for bi in best_iters)

    def test_inner_splits_recorded_when_used(self) -> None:
        X, y = _reg_data(n=150)
        trainer = CVTrainer(
            outer_splitter=KFoldSplitter(n_splits=3, shuffle=True, random_state=0),
            inner_valid=HoldoutInnerValid(ratio=0.2, random_state=0),
            pipeline_factory=NativeFeaturePipeline,
            estimator_factory=lambda: LGBMAdapter(
                task="regression", params={"n_estimators": 10}, random_state=0
            ),
            task="regression",
        )
        result = trainer.fit(
            X,
            y,
            data_fingerprint=_make_fingerprint(X),
            run_meta=_make_run_meta(),
        )
        assert result.splits.inner is not None
        assert len(result.splits.inner) == 3


# ---------------------------------------------------------------------------
# RefitTrainer
# ---------------------------------------------------------------------------


class TestRefitTrainer:
    def test_refit_produces_result(self) -> None:
        X, y = _reg_data(n=100)
        trainer = RefitTrainer(
            inner_valid=NoInnerValid(),
            pipeline_factory=NativeFeaturePipeline,
            estimator_factory=lambda: LGBMAdapter(
                task="regression", params={"n_estimators": 20}, random_state=0
            ),
            task="regression",
        )
        result = trainer.fit(X, y)
        assert isinstance(result, RefitResult)
        assert result.train_pred.shape == (len(X),)
        assert result.feature_names == ["a", "b"]

    def test_refit_model_can_predict(self) -> None:
        X, y = _reg_data(n=100)
        trainer = RefitTrainer(
            inner_valid=NoInnerValid(),
            pipeline_factory=NativeFeaturePipeline,
            estimator_factory=lambda: LGBMAdapter(
                task="regression", params={"n_estimators": 20}, random_state=0
            ),
            task="regression",
        )
        result = trainer.fit(X, y)
        X_new = X.iloc[:5].reset_index(drop=True)
        pipeline = NativeFeaturePipeline()
        pipeline.load_state(result.pipeline_state)
        X_new_t = pipeline.transform(X_new)
        preds = result.model.predict(X_new_t)
        assert preds.shape == (5,)

    def test_refit_with_early_stopping(self) -> None:
        X, y = _reg_data(n=200)
        trainer = RefitTrainer(
            inner_valid=HoldoutInnerValid(ratio=0.15, random_state=0),
            pipeline_factory=NativeFeaturePipeline,
            estimator_factory=lambda: LGBMAdapter(
                task="regression",
                params={"n_estimators": 200, "learning_rate": 0.1},
                early_stopping_rounds=10,
                random_state=0,
            ),
            task="regression",
        )
        result = trainer.fit(X, y)
        assert result.best_iteration is not None
        assert result.best_iteration < 200
