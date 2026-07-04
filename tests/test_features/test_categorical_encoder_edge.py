"""Edge-case coverage for features/encoders/categorical_encoder.py."""

from __future__ import annotations

import pandas as pd
import pytest

from lizyml.features.encoders.categorical_encoder import CategoricalEncoder


class TestCategoricalEncoder:
    def test_fit_missing_column(self) -> None:
        enc = CategoricalEncoder()
        enc.fit(pd.DataFrame({"a": [1, 2, 3]}), ["nonexistent"])
        assert "nonexistent" not in enc._categories

    def test_fit_object_dtype(self) -> None:
        enc = CategoricalEncoder()
        enc.fit(pd.DataFrame({"col": ["b", "a", "c", "a"]}), ["col"])
        assert enc._categories["col"] == ["a", "b", "c"]

    def test_fit_all_na_column(self) -> None:
        enc = CategoricalEncoder()
        enc.fit(pd.DataFrame({"col": pd.Categorical([None, None, None])}), ["col"])
        assert enc._modes["col"] is None

    def test_transform_before_fit(self) -> None:
        enc = CategoricalEncoder()
        with pytest.raises(RuntimeError, match="must be fitted"):
            enc.transform(pd.DataFrame({"col": ["a"]}))

    def test_transform_missing_column(self) -> None:
        enc = CategoricalEncoder()
        enc.fit(pd.DataFrame({"col": pd.Categorical(["a", "b"])}), ["col"])
        result = enc.transform(pd.DataFrame({"other": [1, 2]}))
        assert "other" in result.columns
