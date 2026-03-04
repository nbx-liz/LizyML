"""Public API surface tests.

Verifies that all documented public names are importable from the expected locations.
"""

from __future__ import annotations


class TestPublicImports:
    def test_model_importable_from_lizyml(self) -> None:
        from lizyml import Model

        assert callable(Model)

    def test_version_importable_from_lizyml(self) -> None:
        from lizyml import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_fit_result_importable(self) -> None:
        from lizyml.core.types.fit_result import FitResult

        assert FitResult is not None

    def test_prediction_result_importable(self) -> None:
        from lizyml.core.types.predict_result import PredictionResult

        assert PredictionResult is not None

    def test_error_code_importable(self) -> None:
        from lizyml.core.exceptions import ErrorCode, LizyMLError

        assert ErrorCode is not None
        assert LizyMLError is not None

    def test_plots_importable(self) -> None:
        from lizyml.plots import (
            plot_importance,
            plot_learning_curve,
            plot_oof_distribution,
        )

        assert callable(plot_importance)
        assert callable(plot_learning_curve)
        assert callable(plot_oof_distribution)

    def test_explain_importable(self) -> None:
        from lizyml.explain import compute_shap_values

        assert callable(compute_shap_values)

    def test_model_methods_present(self) -> None:
        from lizyml import Model

        assert hasattr(Model, "fit")
        assert hasattr(Model, "predict")
        assert hasattr(Model, "evaluate")
        assert hasattr(Model, "importance")
        assert hasattr(Model, "tune")
        assert hasattr(Model, "export")
        assert hasattr(Model, "load")
        assert hasattr(Model, "importance_plot")
        assert hasattr(Model, "plot_learning_curve")
        assert hasattr(Model, "plot_oof_distribution")
