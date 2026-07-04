"""Coverage for core/logging.py log_event formatting."""

from __future__ import annotations

import logging
from unittest.mock import patch

from lizyml.core.logging import log_event


class TestLogEvent:
    def test_log_event_basic(self) -> None:
        logger = logging.getLogger("test.log_event")
        with patch.object(logger, "log") as mock_log:
            log_event(logger, "fit.start", run_id="abc", fold=0)
            mock_log.assert_called_once()
            msg = mock_log.call_args[0][1]
            assert "event='fit.start'" in msg
            assert "run_id='abc'" in msg
            assert "fold=0" in msg
