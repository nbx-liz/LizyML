"""ModelTablesMixin — table/accessor methods extracted from Model facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.config.schema import LGBMConfig
from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.fit_result import FitResult
    from lizyml.core.types.tuning_result import TuningResult


class ModelTablesMixin:
    """Mixin providing table/accessor methods for :class:`Model`."""

    # Attributes provided by Model — declared for type checking only.
    if TYPE_CHECKING:
        _cfg: LizyMLConfig
        _fit_result: FitResult | None
        _y: pd.Series | None
        _X: pd.DataFrame | None
        _metrics: dict[str, Any] | None
        _tuning_result: TuningResult | None

        def _require_fit(self) -> FitResult: ...

    def evaluate_table(self) -> pd.DataFrame:
        """Return evaluation metrics as a formatted DataFrame.

        Rows are metric names, columns are ``if_mean``, ``oof``,
        ``fold_0`` … ``fold_N-1`` (OOF per-fold on valid_idx),
        and ``cal_oof`` when calibrated.

        Returns:
            :class:`pd.DataFrame` with metric values.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        self._require_fit()
        from lizyml.evaluation.table_formatter import format_metrics_table

        assert self._metrics is not None  # noqa: S101 — set by fit()
        return format_metrics_table(self._metrics)

    def residuals(self) -> npt.NDArray[np.float64]:
        """Return OOF residuals ``(y_true - oof_pred)``.  Regression only.

        Returns:
            1-D array of shape ``(n_samples,)``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-regression tasks.
        """
        fit_result = self._require_fit()
        if self._cfg.task != "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message=(
                    "residuals() is only supported for regression tasks. "
                    f"Got task='{self._cfg.task}'."
                ),
                context={"task": self._cfg.task},
            )
        if self._y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version to enable "
                    "diagnostic APIs after Model.load()."
                ),
                context={},
            )
        result: npt.NDArray[np.float64] = np.asarray(self._y) - fit_result.oof_pred
        return result

    def confusion_matrix(self, threshold: float = 0.5) -> dict[str, pd.DataFrame]:
        """Return IS/OOS confusion matrices.

        Args:
            threshold: Binary decision boundary (binary only).

        Returns:
            ``{"is": DataFrame, "oos": DataFrame}``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for regression.
        """
        fit_result = self._require_fit()
        if self._y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version "
                    "to enable diagnostic APIs after Model.load()."
                ),
                context={},
            )
        if self._cfg.task == "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="confusion_matrix() requires a binary or multiclass task.",
                context={"task": self._cfg.task},
            )
        from lizyml.evaluation.confusion import confusion_matrix_table

        return confusion_matrix_table(
            fit_result,
            np.asarray(self._y),
            threshold=threshold,
            task=self._cfg.task,
        )

    def importance(self, kind: str = "split") -> dict[str, float]:
        """Return averaged feature importance across CV fold models.

        Args:
            kind: ``"split"``, ``"gain"``, or ``"shap"``.
                ``"shap"`` computes mean(|SHAP|) per feature across folds.
                Requires ``shap`` to be installed and training data to be
                available (or ``analysis_context`` to be restored after load).

        Returns:
            Dict mapping feature name → importance score.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit`` or (for ``"shap"``)
            when loaded artifacts lack ``analysis_context``.
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``OPTIONAL_DEP_MISSING`` when ``kind="shap"`` and shap is
            not installed.
        """
        fit_result = self._require_fit()

        if kind == "shap":
            if self._X is None:
                raise LizyMLError(
                    code=ErrorCode.MODEL_NOT_FIT,
                    user_message=(
                        "Training data not available. "
                        "Re-export the model with the latest version to enable "
                        "diagnostic APIs after Model.load()."
                    ),
                    context={},
                )
            from lizyml.explain.shap_explainer import compute_shap_importance

            return compute_shap_importance(
                models=fit_result.models,
                X=self._X,
                splits_outer=fit_result.splits.outer,
                task=self._cfg.task,
                feature_names=fit_result.feature_names,
                pipeline_state=fit_result.pipeline_state,
            )

        models = fit_result.models
        if not models:
            return {}

        agg: dict[str, float] = {}
        for m in models:
            for feat, val in m.importance(kind=kind).items():
                agg[feat] = agg.get(feat, 0.0) + val / len(models)
        return agg

    def tuning_table(self) -> pd.DataFrame:
        """Return a DataFrame of all tuning trial results.

        Columns: ``trial``, metric name, and each searched parameter name.

        Returns:
            DataFrame with one row per trial.

        Raises:
            LizyMLError with MODEL_NOT_FIT when ``tune()`` has not been called.
        """
        if self._tuning_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="tune() has not been called yet.",
                context={},
            )
        tr = self._tuning_result
        rows = []
        for t in tr.trials:
            row: dict[str, Any] = {
                "trial": t.number,
                tr.metric_name: t.score,
                **t.params,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def params_table(self) -> pd.DataFrame:
        """Return resolved parameters as a single-column DataFrame.

        Merges Config smart params, training settings, resolved booster
        params (fold 0), and per-fold ``best_iteration`` into one table.

        Returns:
            :class:`pd.DataFrame` with index ``parameter`` and column ``value``.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        fr = self._require_fit()
        if not fr.models:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="No trained models available.",
                context={},
            )
        model_cfg = self._cfg.model

        rows: list[dict[str, Any]] = []

        # --- Config smart params ---
        if isinstance(model_cfg, LGBMConfig):
            smart = {
                "auto_num_leaves": model_cfg.auto_num_leaves,
                "num_leaves_ratio": model_cfg.num_leaves_ratio,
                "min_data_in_leaf_ratio": model_cfg.min_data_in_leaf_ratio,
                "min_data_in_bin_ratio": model_cfg.min_data_in_bin_ratio,
                "balanced": model_cfg.balanced,
                "feature_weights": model_cfg.feature_weights,
            }
            for k, v in smart.items():
                rows.append({"parameter": k, "value": v})

        # --- Config training params ---
        es = self._cfg.training.early_stopping
        if es is not None:
            rows.append({"parameter": "early_stopping_rounds", "value": es.rounds})
            rows.append({"parameter": "validation_ratio", "value": es.validation_ratio})

        # --- Resolved booster params (fold 0) ---
        booster = fr.models[0].get_native_model()
        for k in [
            "objective",
            "learning_rate",
            "max_depth",
            "num_leaves",
            "min_data_in_leaf",
            "min_data_in_bin",
            "max_bin",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "num_iterations",
        ]:
            v = booster.params.get(k)
            if v is not None:
                rows.append({"parameter": k, "value": v})

        # task-specific params
        for k in ["scale_pos_weight", "num_class"]:
            v = booster.params.get(k)
            if v is not None:
                rows.append({"parameter": k, "value": v})

        # --- Best iteration per fold ---
        for i, m in enumerate(fr.models):
            rows.append({"parameter": f"best_iteration_{i}", "value": m.best_iteration})

        df = pd.DataFrame(rows)
        return df.set_index("parameter")

    def split_summary(self) -> pd.DataFrame:
        """Return per-fold split summary as a DataFrame.

        Columns always include ``fold``, ``train_size``, ``valid_size``.
        For time-series splits with ``time_col``, also includes
        ``train_start``, ``train_end``, ``valid_start``, ``valid_end``.

        Returns:
            :class:`pd.DataFrame` with one row per fold.

        Raises:
            :class:`~lizyml.core.exceptions.LizyMLError` with
            ``MODEL_NOT_FIT`` when called before ``fit``.
        """
        fr = self._require_fit()
        rows: list[dict[str, Any]] = []
        for i, (train_idx, valid_idx) in enumerate(fr.splits.outer):
            row: dict[str, Any] = {
                "fold": i,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
            }
            if fr.splits.time_range is not None and i < len(fr.splits.time_range):
                tr = fr.splits.time_range[i]
                row["train_start"] = tr["train_start"]
                row["train_end"] = tr["train_end"]
                row["valid_start"] = tr["valid_start"]
                row["valid_end"] = tr["valid_end"]
            rows.append(row)
        return pd.DataFrame(rows)
