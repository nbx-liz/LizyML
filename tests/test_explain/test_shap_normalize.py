"""Unit tests for SHAP output normalization (H-0002 shape contract).

Exercises ``_normalize_shap_output`` directly with plain numpy inputs — no
SHAP install and no MagicMock explainer needed, since the normalization is a
pure function over the raw ``shap_values`` return value.
"""

from __future__ import annotations

import numpy as np

from lizyml.explain.shap_explainer import _normalize_shap_output


class TestNormalizeShapOutput:
    def test_legacy_list_binary_keeps_positive_class(self) -> None:
        arr0 = np.zeros((10, 5))
        arr1 = np.ones((10, 5))
        result = _normalize_shap_output([arr0, arr1], "binary")
        np.testing.assert_array_equal(result, arr1)

    def test_legacy_list_multiclass_mean_abs(self) -> None:
        arrs = [np.random.default_rng(0).standard_normal((10, 5)) for _ in range(3)]
        result = _normalize_shap_output(arrs, "multiclass")
        assert result.shape == (10, 5)
        expected = np.mean(np.abs(np.stack(arrs, axis=0)), axis=0)
        np.testing.assert_allclose(result, expected)

    def test_ndarray_2d_returned_as_is(self) -> None:
        raw = np.arange(20, dtype=np.float64).reshape(4, 5)
        result = _normalize_shap_output(raw, "regression")
        np.testing.assert_array_equal(result, raw)

    def test_ndarray_3d_multiclass_reduced(self) -> None:
        raw = np.random.default_rng(1).standard_normal((6, 4, 3))
        result = _normalize_shap_output(raw, "multiclass")
        assert result.shape == (6, 4)
        np.testing.assert_allclose(result, np.mean(np.abs(raw), axis=2))

    def test_fallback_to_asarray(self) -> None:
        result = _normalize_shap_output(((1.0, 2.0), (3.0, 4.0)), "regression")
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
