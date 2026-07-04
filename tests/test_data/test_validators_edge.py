"""Edge-case coverage for data/validators.py (missing-column / non-comparable)."""

from __future__ import annotations

import pandas as pd

from lizyml.data.validators import (
    validate_no_target_leakage,
    validate_time_series_order,
)


class TestValidators:
    def test_time_series_missing_col(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert validate_time_series_order(df, "nonexistent") == []

    def test_leakage_missing_target(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert validate_no_target_leakage(df, "nonexistent") == []

    def test_leakage_type_error(self) -> None:
        df = pd.DataFrame(
            {
                "target": [1, 2, 3],
                "mixed": [object(), object(), object()],
            }
        )
        result = validate_no_target_leakage(df, "target", raise_on_violation=False)
        assert isinstance(result, list)
