"""Edge-case coverage for data/validators.py (missing-column / non-comparable)."""

from __future__ import annotations

import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.data.validators import (
    validate_no_target_leakage,
    validate_time_series_order,
)


class TestValidators:
    def test_time_series_missing_col(self) -> None:
        """A missing time column is refused, not reported clean (H-0112)."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(LizyMLError) as exc:
            validate_time_series_order(df, "nonexistent")
        assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_leakage_missing_target(self) -> None:
        """A missing target column is refused, not reported clean (H-0112)."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(LizyMLError) as exc:
            validate_no_target_leakage(df, "nonexistent")
        assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID

    def test_leakage_type_error(self) -> None:
        df = pd.DataFrame(
            {
                "target": [1, 2, 3],
                "mixed": [object(), object(), object()],
            }
        )
        result = validate_no_target_leakage(df, "target", raise_on_violation=False)
        assert isinstance(result, list)
