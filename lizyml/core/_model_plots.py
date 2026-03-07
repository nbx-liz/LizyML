"""ModelPlotsMixin — plot methods extracted from Model facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.fit_result import FitResult
    from lizyml.core.types.tuning_result import TuningResult


class ModelPlotsMixin:
    """Mixin providing plot methods for :class:`Model`."""

    # Attributes provided by Model — declared for type checking only.
    if TYPE_CHECKING:
        _cfg: LizyMLConfig
        _fit_result: FitResult | None
        _y: pd.Series | None
        _X: pd.DataFrame | None
        _tuning_result: TuningResult | None

        def _require_fit(self) -> FitResult: ...

        def residuals(self) -> npt.NDArray[np.float64]: ...

        def importance(self, kind: str = "split") -> dict[str, float]: ...

    def residuals_plot(self, *, kind: str = "all") -> Any:
        """Plot residual analysis.  Regression only.

        Args:
            kind: Which plot to render.
                ``"scatter"`` — residuals vs predicted (IS + OOS overlay).
                ``"histogram"`` — residual distribution (IS + OOS overlay).
                ``"qq"`` — QQ plot of OOS residuals.
                ``"all"`` — all three panels in one figure (default).

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-regression tasks.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
            LizyMLError with ``CONFIG_INVALID`` for an unknown ``kind`` value.
        """
        # Validate state (raises MODEL_NOT_FIT / UNSUPPORTED_TASK as needed)
        self.residuals()
        fit_result = self._require_fit()
        from lizyml.plots.residuals import plot_residuals

        return plot_residuals(fit_result, np.asarray(self._y), kind=kind)

    def roc_curve_plot(self) -> Any:
        """Plot ROC Curve. Binary: IS/OOS overlay. Multiclass: OvR subplots.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for regression.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
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
                user_message="roc_curve_plot() requires a binary or multiclass task.",
                context={"task": self._cfg.task},
            )
        from lizyml.plots.classification import plot_roc_curve

        return plot_roc_curve(fit_result, np.asarray(self._y), task=self._cfg.task)

    def calibration_plot(self) -> Any:
        """Plot calibration reliability diagram. Binary + calibration only.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-binary tasks.
            LizyMLError with ``CALIBRATION_NOT_SUPPORTED`` if calibration
                is not enabled.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
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
        if self._cfg.task != "binary":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="calibration_plot() requires a binary task.",
                context={"task": self._cfg.task},
            )
        from lizyml.plots.calibration import plot_calibration_curve

        return plot_calibration_curve(fit_result, np.asarray(self._y))

    def probability_histogram_plot(self) -> Any:
        """Plot raw vs calibrated probability histogram. Binary + calibration only.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when called before ``fit``
                or when loaded artifacts lack ``analysis_context``.
            LizyMLError with ``UNSUPPORTED_TASK`` for non-binary tasks.
            LizyMLError with ``CALIBRATION_NOT_SUPPORTED`` if calibration
                is not enabled.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        """
        self._require_fit()
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
        if self._cfg.task != "binary":
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_TASK,
                user_message="probability_histogram_plot() requires a binary task.",
                context={"task": self._cfg.task},
            )
        fit_result = self._require_fit()
        from lizyml.plots.calibration import plot_probability_histogram

        return plot_probability_histogram(fit_result)

    def importance_plot(self, kind: str = "split", top_n: int | None = 20) -> Any:
        """Plot fold-averaged feature importances as a horizontal bar chart.

        Args:
            kind: ``"split"``, ``"gain"``, or ``"shap"``.
            top_n: Maximum number of features to show.

        Returns:
            A ``plotly.graph_objects.Figure`` object.

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before fit.
            LizyMLError with OPTIONAL_DEP_MISSING when plotly (or shap for
                ``kind="shap"``) is not installed.
        """
        if kind == "shap":
            imp = self.importance(kind="shap")
            from lizyml.plots.importance import plot_importance_from_dict

            return plot_importance_from_dict(imp, top_n=top_n)

        fit_result = self._require_fit()
        from lizyml.plots.importance import plot_importance

        return plot_importance(fit_result, kind=kind, top_n=top_n)

    def plot_learning_curve(self) -> Any:
        """Plot per-fold training/validation loss vs iteration.

        Returns:
            A ``plotly.graph_objects.Figure`` object.

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before fit or when
                no evaluation history is available.
            LizyMLError with OPTIONAL_DEP_MISSING when plotly is not installed.
        """
        fit_result = self._require_fit()
        from lizyml.plots.learning_curve import plot_learning_curve

        return plot_learning_curve(fit_result)

    def plot_oof_distribution(self) -> Any:
        """Plot the distribution of out-of-fold predictions.

        Returns:
            A ``plotly.graph_objects.Figure`` object.

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before fit.
            LizyMLError with OPTIONAL_DEP_MISSING when plotly is not installed.
        """
        fit_result = self._require_fit()
        from lizyml.plots.oof_distribution import plot_oof_distribution

        return plot_oof_distribution(fit_result)

    def tuning_plot(self) -> Any:
        """Plot tuning history. Requires ``tune()`` to have been called.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            LizyMLError with ``MODEL_NOT_FIT`` when ``tune()`` has not been called.
            LizyMLError with ``OPTIONAL_DEP_MISSING`` when plotly is not installed.
        """
        if self._tuning_result is None:
            raise LizyMLError(
                code=ErrorCode.MODEL_NOT_FIT,
                user_message="tune() has not been called yet.",
                context={},
            )
        from lizyml.plots.tuning import plot_tuning_history

        return plot_tuning_history(self._tuning_result)
