"""ModelTablesMixin — table/accessor methods extracted from Model facade.

After H-0077 (Phase 2) every method reads state exclusively through
``self._get_fit_state()`` / ``self._get_tuning_state()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_state import FitState, TuningState


class ModelTablesMixin:
    """Mixin providing table/accessor methods for :class:`Model`."""

    # Facade entry points provided by Model — declared for type checking only.
    if TYPE_CHECKING:

        def _get_fit_state(self) -> FitState: ...

        def _get_tuning_state(self) -> TuningState: ...

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
        state = self._get_fit_state()
        from lizyml.evaluation.table_formatter import format_metrics_table

        assert state.metrics is not None  # noqa: S101 — populated after fit()
        return format_metrics_table(state.metrics)

    def residuals(self) -> npt.NDArray[np.float64]:
        """Return OOF residuals ``(y_true - oof_pred)``.  Regression only.

        Returns:
            1-D array of shape ``(n_samples,)``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-regression tasks.
        """
        state = self._get_fit_state()
        if state.cfg.task != "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message=(
                    "residuals() is only supported for regression tasks. "
                    f"Got task='{state.cfg.task}'."
                ),
                context={"task": state.cfg.task},
            )
        if state.y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version to enable "
                    "diagnostic APIs after Model.load()."
                ),
                context={
                    "task": state.cfg.task,
                    "loaded_from_artifact": True,
                    "method": "residuals",
                },
            )
        result: npt.NDArray[np.float64] = (
            np.asarray(state.y) - state.fit_result.oof_pred
        )
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
        state = self._get_fit_state()
        if state.y is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "Target values not available. "
                    "Re-export the model with the latest version "
                    "to enable diagnostic APIs after Model.load()."
                ),
                context={"task": state.cfg.task, "loaded_from_artifact": True},
            )
        if state.cfg.task == "regression":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="confusion_matrix() requires a binary or multiclass task.",
                context={"task": state.cfg.task},
            )
        from lizyml.evaluation.confusion import confusion_matrix_table

        return confusion_matrix_table(
            state.fit_result,
            np.asarray(state.y),
            threshold=threshold,
            task=state.cfg.task,
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
        state = self._get_fit_state()

        if kind == "shap":
            if state.X is None:
                raise LizyMLError(
                    code=ErrorCode.MODEL_NOT_FIT,
                    user_message=(
                        "Training data not available. "
                        "Re-export the model with the latest version to enable "
                        "diagnostic APIs after Model.load()."
                    ),
                    context={
                        "task": state.cfg.task,
                        "kind": kind,
                        "method": "importance",
                    },
                )
            from lizyml.explain.shap_explainer import compute_shap_importance

            return compute_shap_importance(
                models=state.fit_result.models,
                X=state.X,
                splits_outer=state.fit_result.splits.outer,
                task=state.cfg.task,
                feature_names=state.fit_result.feature_names,
                pipeline_state=state.fit_result.pipeline_state,
                pipeline_factory=state.provider.build_pipeline_factory(),
            )

        models = state.fit_result.models
        if not models:
            return {}

        agg: dict[str, float] = {}
        for m in models:
            for feat, val in m.importance(kind=kind).items():
                agg[feat] = agg.get(feat, 0.0) + val / len(models)
        return agg

    def tuning_table(self) -> pd.DataFrame:
        """Return a DataFrame of all tuning trial results.

        Columns: ``trial``, ``round``, metric name, each searched parameter
        name, and ``state``.

        Returns:
            DataFrame with one row per trial.

        Raises:
            LizyMLError with MODEL_NOT_FIT when ``tune()`` has not been called.
        """
        state = self._get_tuning_state()
        tr = state.tuning_result
        rows = []
        for t in tr.trials:
            row: dict[str, Any] = {
                "trial": t.number,
                "round": t.round,
                tr.metric_name: t.score,
                **t.params,
                "state": t.state,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def boundary_table(self) -> pd.DataFrame:
        """Return a DataFrame of boundary detection results (H-0068).

        Columns: ``dim``, ``best``, ``low``, ``high``, ``position``,
        ``edge``, ``expanded``, ``new_low``, ``new_high``.

        Returns:
            DataFrame with one row per search dimension.

        Raises:
            LizyMLError with MODEL_NOT_FIT when ``tune(resume=True)`` has
            not been called or no boundary report exists.
        """
        state = self._get_tuning_state()
        if state.tuning_result.boundary_report is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message=(
                    "No boundary report available. "
                    "Run tune(resume=True) to generate a boundary report."
                ),
                context={
                    "method": "boundary_table",
                    "tune_called": True,
                },
            )
        report = state.tuning_result.boundary_report
        rows = []
        for s in report.dims:
            rows.append(
                {
                    "dim": s.name,
                    "best": s.best_value,
                    "low": s.low,
                    "high": s.high,
                    "position": s.position_pct,
                    "edge": s.edge,
                    "expanded": s.expanded,
                    "new_low": s.new_low,
                    "new_high": s.new_high,
                }
            )
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
        state = self._get_fit_state()
        fr = state.fit_result
        if not fr.models:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="No trained models available.",
                context={"method": "params_table", "task": state.cfg.task},
            )

        rows: list[dict[str, Any]] = []

        # --- Estimator-specific params via provider (H-0054) ---
        rows.extend(state.provider.params_summary(fr.models[0], state.cfg.model))

        # --- Config training params ---
        es = state.cfg.training.early_stopping
        if es is not None:
            rows.append({"parameter": "early_stopping_rounds", "value": es.rounds})
            rows.append({"parameter": "validation_ratio", "value": es.validation_ratio})

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
        state = self._get_fit_state()
        fr = state.fit_result
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
