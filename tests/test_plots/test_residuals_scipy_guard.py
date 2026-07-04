"""QQ plots must raise LizyMLError(OPTIONAL_DEP_MISSING) when scipy is absent."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError


class TestScipyOptionalDepGuard:
    def test_scipy_missing_raises_lizyml_error(self) -> None:
        pytest.importorskip("plotly")
        with patch("lizyml.plots.residuals._scipy_stats", None):
            from lizyml.plots.residuals import _add_qq_traces

            with pytest.raises(LizyMLError) as exc_info:
                _add_qq_traces(None, np.array([1.0, 2.0, 3.0]))
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING
