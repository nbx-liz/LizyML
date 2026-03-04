"""CVTrainer — outer cross-validation training loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from lizyml.core.types.artifacts import RunMeta, SplitIndices
from lizyml.core.types.fit_result import FitResult
from lizyml.data.fingerprint import DataFingerprint
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.evaluation.oof import fill_oof, get_fold_pred, init_oof
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.splitters.base import BaseSplitter
from lizyml.training.inner_valid import BaseInnerValidStrategy

TaskType = Literal["regression", "binary", "multiclass"]


class CVTrainer:
    """Executes an outer cross-validation training loop.

    Responsibilities (per outer fold):
    1. Split data into train / valid via ``outer_splitter``.
    2. Apply ``inner_valid`` to train data for early stopping.
    3. Fit ``pipeline_factory()`` on train data only (leakage prevention).
    4. Transform train, inner-valid, and valid data through the fitted pipeline.
    5. Fit ``estimator_factory()`` with early stopping on inner-valid set.
    6. Assemble OOF predictions (valid set only) and IF predictions (train set).
    7. Collect per-fold history and best_iteration.
    8. Return a :class:`~lizyml.core.types.fit_result.FitResult`.

    Args:
        outer_splitter: Splitter for outer CV folds.
        inner_valid: Strategy for inner holdout (early stopping).
        pipeline_factory: Callable that returns a fresh
            :class:`~lizyml.features.pipeline_base.BaseFeaturePipeline`.
        estimator_factory: Callable that returns a fresh
            :class:`~lizyml.estimators.base.BaseEstimatorAdapter`.
        task: ML task type.
        n_classes: Number of classes (required for multiclass).
    """

    def __init__(
        self,
        outer_splitter: BaseSplitter,
        inner_valid: BaseInnerValidStrategy,
        pipeline_factory: Callable[[], BaseFeaturePipeline],
        estimator_factory: Callable[[], BaseEstimatorAdapter],
        task: TaskType = "regression",
        n_classes: int | None = None,
    ) -> None:
        self.outer_splitter = outer_splitter
        self.inner_valid = inner_valid
        self.pipeline_factory = pipeline_factory
        self.estimator_factory = estimator_factory
        self.task = task
        self.n_classes = n_classes

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray | None = None,
        *,
        data_fingerprint: DataFingerprint,
        run_meta: RunMeta,
    ) -> FitResult:
        """Run the CV training loop and return a :class:`FitResult`.

        Args:
            X: Full feature DataFrame.
            y: Full target Series.
            groups: Optional group labels (same length as X).
            data_fingerprint: Pre-computed fingerprint of the training data.
            run_meta: Runtime/version metadata.

        Returns:
            Populated :class:`~lizyml.core.types.fit_result.FitResult`
            with ``metrics={}`` (metrics populated by Evaluator in Phase 10).
        """
        n_samples = len(X)
        y_arr = y.to_numpy()
        oof = init_oof(n_samples, self.task, self.n_classes)

        outer_splits: list[tuple[np.ndarray, np.ndarray]] = []
        inner_splits: list[tuple[np.ndarray, np.ndarray]] = []
        models: list[BaseEstimatorAdapter] = []
        if_pred_per_fold: list[np.ndarray] = []
        history: list[dict[str, Any]] = []
        last_pipeline: BaseFeaturePipeline | None = None

        fold_iter = list(self.outer_splitter.split(n_samples, y=y_arr, groups=groups))

        for train_idx, valid_idx in fold_iter:
            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            X_valid = X.iloc[valid_idx].reset_index(drop=True)

            # --- Inner validation split (for early stopping) -----------------
            iv_result = self.inner_valid.split(
                len(X_train),
                y=y_train.to_numpy(),
            )
            if iv_result is not None:
                inner_train_rel, inner_valid_rel = iv_result
                inner_splits.append((inner_train_rel, inner_valid_rel))
            else:
                inner_splits.append((np.arange(len(X_train)), np.array([], dtype=int)))

            # --- Feature pipeline fit (train only, leakage prevention) -------
            pipeline = self.pipeline_factory()
            pipeline.fit(X_train, y_train)
            last_pipeline = pipeline

            X_train_t = pipeline.transform(X_train)

            # Prepare inner-valid subsets through the ALREADY-fitted pipeline
            X_iv_train: pd.DataFrame | None = None
            y_iv_train: pd.Series | None = None
            X_iv_valid: pd.DataFrame | None = None
            y_iv_valid: pd.Series | None = None
            if iv_result is not None:
                X_iv_train = X_train_t.iloc[inner_train_rel].reset_index(drop=True)
                y_iv_train = y_train.iloc[inner_train_rel].reset_index(drop=True)
                X_iv_valid = X_train_t.iloc[inner_valid_rel].reset_index(drop=True)
                y_iv_valid = y_train.iloc[inner_valid_rel].reset_index(drop=True)

            X_valid_t = pipeline.transform(X_valid)

            # --- Estimator fit -----------------------------------------------
            estimator = self.estimator_factory()
            cat_cols: list[str] = (
                pipeline.get_state().get("categorical_cols", [])
                if hasattr(pipeline, "get_state")
                else []
            )

            if X_iv_train is not None:
                # y_iv_train and y_iv_valid are set in the same branch as X_iv_train
                assert y_iv_train is not None and y_iv_valid is not None  # noqa: S101
                estimator.fit(
                    X_iv_train,
                    y_iv_train,
                    X_iv_valid,
                    y_iv_valid,
                    categorical_feature=cat_cols or "auto",
                )
            else:
                estimator.fit(
                    X_train_t,
                    y_train,
                    categorical_feature=cat_cols or "auto",
                )
            models.append(estimator)

            # --- OOF predictions ---------------------------------------------
            fold_valid_pred = get_fold_pred(estimator, X_valid_t, self.task)
            fill_oof(oof, valid_idx, fold_valid_pred)

            # --- IF predictions (train fold) ----------------------------------
            fold_train_pred = get_fold_pred(estimator, X_train_t, self.task)
            if_pred_per_fold.append(fold_train_pred)

            # --- Collect history ----------------------------------------------
            native = estimator.get_native_model()
            eval_hist: dict[str, Any] = {}
            if hasattr(native, "evals_result_"):
                eval_hist = dict(native.evals_result_)
            history.append(
                {
                    "best_iteration": estimator.best_iteration,
                    "eval_history": eval_hist,
                }
            )

            # --- Record outer split ------------------------------------------
            outer_splits.append((train_idx, valid_idx))

        assert last_pipeline is not None, "No folds were executed."  # noqa: S101

        feature_names = list(X.columns)
        dtypes = {col: str(X[col].dtype) for col in X.columns}
        categorical_features: list[str] = last_pipeline.get_state().get(
            "categorical_cols", []
        )

        splits = SplitIndices(
            outer=outer_splits,
            inner=inner_splits if any(len(iv[1]) > 0 for iv in inner_splits) else None,
            calibration=None,
        )

        return FitResult(
            oof_pred=oof,
            if_pred_per_fold=if_pred_per_fold,
            metrics={},
            models=models,
            history=history,
            feature_names=feature_names,
            dtypes=dtypes,
            categorical_features=categorical_features,
            splits=splits,
            data_fingerprint=data_fingerprint,
            pipeline_state=last_pipeline.get_state(),
            calibrator=None,
            run_meta=run_meta,
        )
