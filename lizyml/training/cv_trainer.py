"""CVTrainer — outer cross-validation training loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.types.artifacts import RunMeta, SplitIndices
from lizyml.core.types.fit_result import FitResult
from lizyml.data.fingerprint import DataFingerprint
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.evaluation.oof import fill_oof, get_fold_pred, get_fold_raw, init_oof
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.splitters.base import BaseSplitter
from lizyml.training.inner_valid import BaseInnerValidStrategy

TaskType = Literal["regression", "binary", "multiclass"]

# Type alias for inner-valid subsets (X_train, y_train, X_valid, y_valid)
_IVSubsets = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


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
        *,
        ratio_param_resolver: Callable[[int], dict[str, Any]] | None = None,
        collect_raw_scores: bool = False,
    ) -> None:
        self.outer_splitter = outer_splitter
        self.inner_valid = inner_valid
        self.pipeline_factory = pipeline_factory
        self.estimator_factory = estimator_factory
        self.task = task
        self.n_classes = n_classes
        self.ratio_param_resolver = ratio_param_resolver
        self.collect_raw_scores = collect_raw_scores

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: npt.NDArray[Any] | None = None,
        *,
        data_fingerprint: DataFingerprint,
        run_meta: RunMeta,
        sample_weight: npt.NDArray[Any] | None = None,
        time_values: pd.Series | None = None,
    ) -> FitResult:
        """Run the CV training loop and return a :class:`FitResult`."""
        n_samples = len(X)
        y_arr = y.to_numpy()
        oof = init_oof(n_samples, self.task, self.n_classes)
        oof_raw: npt.NDArray[np.float64] | None = None
        if self.collect_raw_scores:
            oof_raw = init_oof(n_samples, self.task, self.n_classes)

        outer_splits: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] = []
        inner_splits: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] = []
        models: list[BaseEstimatorAdapter] = []
        if_pred_per_fold: list[npt.NDArray[np.float64]] = []
        history: list[dict[str, Any]] = []
        time_ranges: list[dict[str, Any]] = []
        last_pipeline: BaseFeaturePipeline | None = None

        fold_iter = list(self.outer_splitter.split(n_samples, y=y_arr, groups=groups))

        for train_idx, valid_idx in fold_iter:
            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            X_valid = X.iloc[valid_idx].reset_index(drop=True)

            # --- Inner validation split --------------------------------------
            iv_result, iv_entry = self._split_inner(X_train, y_train, groups, train_idx)
            inner_splits.append(iv_entry)

            # --- Feature pipeline fit (train only, leakage prevention) -------
            pipeline = self.pipeline_factory()
            pipeline.fit(X_train, y_train)
            last_pipeline = pipeline

            X_train_t = pipeline.transform(X_train)
            iv_subsets = self._build_iv_subsets(X_train_t, y_train, iv_result)
            X_valid_t = pipeline.transform(X_valid)

            # --- Estimator fit -----------------------------------------------
            estimator = self._fit_estimator(
                X_train_t,
                y_train,
                iv_subsets,
                iv_result,
                pipeline,
                sample_weight,
                train_idx,
            )
            models.append(estimator)

            # --- Predictions -------------------------------------------------
            fill_oof(oof, valid_idx, get_fold_pred(estimator, X_valid_t, self.task))
            if oof_raw is not None:
                raw = get_fold_raw(estimator, X_valid_t, self.task)
                fill_oof(oof_raw, valid_idx, raw)
            if_pred_per_fold.append(get_fold_pred(estimator, X_train_t, self.task))

            # --- Collect history & splits ------------------------------------
            history.append(
                {
                    "best_iteration": estimator.best_iteration,
                    "eval_history": dict(estimator.eval_results),
                }
            )
            outer_splits.append((train_idx, valid_idx))

            if time_values is not None:
                time_ranges.append(
                    {
                        "fold": len(outer_splits) - 1,
                        "train_start": time_values.iloc[train_idx].min(),
                        "train_end": time_values.iloc[train_idx].max(),
                        "valid_start": time_values.iloc[valid_idx].min(),
                        "valid_end": time_values.iloc[valid_idx].max(),
                    }
                )

        assert last_pipeline is not None, "No folds were executed."  # noqa: S101

        return self._build_result(
            X,
            oof,
            oof_raw,
            outer_splits,
            inner_splits,
            models,
            if_pred_per_fold,
            history,
            time_ranges,
            last_pipeline,
            data_fingerprint,
            run_meta,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_inner(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups: npt.NDArray[Any] | None,
        train_idx: npt.NDArray[np.intp],
    ) -> tuple[
        tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]] | None,
        tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    ]:
        """Split train data for inner validation (early stopping)."""
        groups_train = groups[train_idx] if groups is not None else None
        iv_result = self.inner_valid.split(
            len(X_train),
            y=y_train.to_numpy(),
            groups=groups_train,
        )
        if iv_result is not None:
            return iv_result, iv_result
        return None, (np.arange(len(X_train)), np.array([], dtype=int))

    @staticmethod
    def _build_iv_subsets(
        X_train_t: pd.DataFrame,
        y_train: pd.Series,
        iv_result: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]] | None,
    ) -> _IVSubsets | None:
        """Slice transformed train data into inner-valid subsets."""
        if iv_result is None:
            return None
        inner_train_rel, inner_valid_rel = iv_result
        return (
            X_train_t.iloc[inner_train_rel].reset_index(drop=True),
            y_train.iloc[inner_train_rel].reset_index(drop=True),
            X_train_t.iloc[inner_valid_rel].reset_index(drop=True),
            y_train.iloc[inner_valid_rel].reset_index(drop=True),
        )

    def _fit_estimator(
        self,
        X_train_t: pd.DataFrame,
        y_train: pd.Series,
        iv_subsets: _IVSubsets | None,
        iv_result: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]] | None,
        pipeline: BaseFeaturePipeline,
        sample_weight: npt.NDArray[Any] | None,
        train_idx: npt.NDArray[np.intp],
    ) -> BaseEstimatorAdapter:
        """Create, configure, and fit a single fold estimator."""
        estimator = self.estimator_factory()

        # Resolve n_rows-dependent ratio params (H-0036)
        if self.ratio_param_resolver is not None:
            n_train = len(iv_subsets[0]) if iv_subsets is not None else len(X_train_t)
            estimator.update_params(self.ratio_param_resolver(n_train))

        cat_cols: list[str] = (
            pipeline.get_state().get("categorical_cols", [])
            if hasattr(pipeline, "get_state")
            else []
        )

        # Prepare fit kwargs (sample_weight slicing)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fold_sw = sample_weight[train_idx]
            if iv_result is not None:
                fit_kwargs["sample_weight"] = fold_sw[iv_result[0]]
            else:
                fit_kwargs["sample_weight"] = fold_sw

        # Single estimator.fit() call with conditional eval set
        if iv_subsets is not None:
            X_iv_train, y_iv_train, X_iv_valid, y_iv_valid = iv_subsets
            estimator.fit(
                X_iv_train,
                y_iv_train,
                X_iv_valid,
                y_iv_valid,
                categorical_feature=cat_cols or "auto",
                **fit_kwargs,
            )
        else:
            estimator.fit(
                X_train_t,
                y_train,
                categorical_feature=cat_cols or "auto",
                **fit_kwargs,
            )
        return estimator

    @staticmethod
    def _build_result(
        X: pd.DataFrame,
        oof: npt.NDArray[np.float64],
        oof_raw: npt.NDArray[np.float64] | None,
        outer_splits: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]],
        inner_splits: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]],
        models: list[BaseEstimatorAdapter],
        if_pred_per_fold: list[npt.NDArray[np.float64]],
        history: list[dict[str, Any]],
        time_ranges: list[dict[str, Any]],
        last_pipeline: BaseFeaturePipeline,
        data_fingerprint: DataFingerprint,
        run_meta: RunMeta,
    ) -> FitResult:
        """Assemble the final FitResult from per-fold collections."""
        feature_names = list(X.columns)
        dtypes = {col: str(X[col].dtype) for col in X.columns}
        categorical_features: list[str] = last_pipeline.get_state().get(
            "categorical_cols", []
        )
        splits = SplitIndices(
            outer=outer_splits,
            inner=inner_splits if any(len(iv[1]) > 0 for iv in inner_splits) else None,
            calibration=None,
            time_range=time_ranges if time_ranges else None,
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
            oof_raw_scores=oof_raw,
        )
