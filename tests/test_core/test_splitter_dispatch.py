"""Regression tests for splitter dispatch fallback (#119).

The dispatch in ``_build_splitter_for_method`` previously ended with a silent
``KFoldSplitter`` fallback for unmatched ``SplitConfig`` variants. Adding a
new variant without updating the dispatch would have silently produced KFold
splits instead of failing loudly. The fallback now raises ``LizyMLError``.
"""

from __future__ import annotations

import pytest

from lizyml.core._model_factories import _build_splitter_for_method
from lizyml.core.exceptions import ErrorCode, LizyMLError


class _UnknownSplitConfig:
    """Sentinel object that does not match any recognised ``SplitConfig``
    isinstance check, simulating a future variant added without updating
    the dispatch."""


class TestUnhandledSplitConfigRaises:
    def test_unknown_split_config_raises_lizyml_error(self) -> None:
        unknown = _UnknownSplitConfig()
        with pytest.raises(LizyMLError) as exc_info:
            _build_splitter_for_method(unknown, n_splits=5)  # type: ignore[arg-type]
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

    def test_error_message_names_the_unhandled_type(self) -> None:
        unknown = _UnknownSplitConfig()
        with pytest.raises(LizyMLError) as exc_info:
            _build_splitter_for_method(unknown, n_splits=5)  # type: ignore[arg-type]
        # The message and context must surface the unhandled config type
        # so contributors get a clear signal.
        assert "_UnknownSplitConfig" in exc_info.value.user_message
        assert exc_info.value.context.get("split_config_type") == "_UnknownSplitConfig"
