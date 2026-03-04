"""RefitTrainer — train a single model on the full dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.evaluation.oof import get_fold_pred
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.training.inner_valid import BaseInnerValidStrategy

TaskType = Literal["regression", "binary", "multiclass"]


@dataclass
class RefitResult:
    """Output of a full-data refit.

    Attributes:
        model: Estimator fitted on the complete training set.
        pipeline_state: Serializable state of the final pipeline.
        feature_names: Ordered feature column names.
        categorical_features: Names of categorical features.
        best_iteration: Best iteration from early stopping (None if not used).
        train_pred: Predictions on the training set (for sanity checks).
        history: Training history dict from the native model.
    """

    model: BaseEstimatorAdapter
    pipeline_state: Any
    feature_names: list[str]
    categorical_features: list[str]
    best_iteration: int | None
    train_pred: np.ndarray
    history: dict[str, Any]


class RefitTrainer:
    """Trains a single model on the full dataset.

    Intended for use after CV to produce the final model for inference.

    Args:
        inner_valid: Inner validation strategy for early stopping
            (``NoInnerValid()`` to skip).
        pipeline_factory: Callable returning a fresh
            :class:`~lizyml.features.pipeline_base.BaseFeaturePipeline`.
        estimator_factory: Callable returning a fresh
            :class:`~lizyml.estimators.base.BaseEstimatorAdapter`.
        task: ML task type.
    """

    def __init__(
        self,
        inner_valid: BaseInnerValidStrategy,
        pipeline_factory: Callable[[], BaseFeaturePipeline],
        estimator_factory: Callable[[], BaseEstimatorAdapter],
        task: TaskType = "regression",
    ) -> None:
        self.inner_valid = inner_valid
        self.pipeline_factory = pipeline_factory
        self.estimator_factory = estimator_factory
        self.task = task

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray | None = None,
    ) -> RefitResult:
        """Fit pipeline and estimator on the full dataset.

        Args:
            X: Full feature DataFrame.
            y: Full target Series.
            groups: Optional group labels.

        Returns:
            :class:`RefitResult` with the fitted model and pipeline state.
        """
        n_samples = len(X)

        # Fit pipeline on all data
        pipeline = self.pipeline_factory()
        pipeline.fit(X, y)
        X_t = pipeline.transform(X)

        cat_cols: list[str] = (
            pipeline.get_state().get("categorical_cols", [])
            if hasattr(pipeline, "get_state")
            else []
        )

        # Inner validation split for early stopping
        iv_result = self.inner_valid.split(n_samples, y=y.to_numpy(), groups=groups)

        estimator = self.estimator_factory()

        if iv_result is not None:
            inner_train_rel, inner_valid_rel = iv_result
            X_iv_train = X_t.iloc[inner_train_rel].reset_index(drop=True)
            y_iv_train = y.iloc[inner_train_rel].reset_index(drop=True)
            X_iv_valid = X_t.iloc[inner_valid_rel].reset_index(drop=True)
            y_iv_valid = y.iloc[inner_valid_rel].reset_index(drop=True)
            estimator.fit(
                X_iv_train,
                y_iv_train,
                X_iv_valid,
                y_iv_valid,
                categorical_feature=cat_cols or "auto",
            )
        else:
            estimator.fit(
                X_t,
                y,
                categorical_feature=cat_cols or "auto",
            )

        train_pred = get_fold_pred(estimator, X_t, self.task)

        native = estimator.get_native_model()
        eval_hist: dict[str, Any] = {}
        if hasattr(native, "evals_result_"):
            eval_hist = dict(native.evals_result_)

        return RefitResult(
            model=estimator,
            pipeline_state=pipeline.get_state(),
            feature_names=list(X.columns),
            categorical_features=cat_cols,
            best_iteration=estimator.best_iteration,
            train_pred=train_pred,
            history=eval_hist,
        )
