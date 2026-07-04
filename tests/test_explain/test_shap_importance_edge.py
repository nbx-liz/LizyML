"""Edge-case: compute_shap_importance must handle an empty models list."""

from __future__ import annotations

import pytest


class TestShapEmptyModels:
    def test_empty_models_no_crash(self) -> None:
        pytest.importorskip("shap")
        from lizyml.explain.shap_explainer import compute_shap_importance

        result = compute_shap_importance(
            models=[],
            X=None,  # type: ignore[arg-type]
            splits_outer=[],
            task="regression",
            feature_names=["a", "b"],
            pipeline_state=None,
        )
        assert result == {"a": 0.0, "b": 0.0}
