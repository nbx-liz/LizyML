"""H-0144: say which H-0138 oracle the lane compared, even when all pass."""

from __future__ import annotations

import sys
from typing import Any


def pytest_terminal_summary(terminalreporter: Any) -> None:
    # Only when the oracle module was collected; importing it here would load
    # LightGBM and the probe samples for every run of this directory.
    helper = sys.modules.get("tests.test_features._h0138_oracle")
    summary = getattr(helper, "SUMMARY", None)
    if summary:
        line = ", ".join(f"{k}={v}" for k, v in summary.items())
        terminalreporter.write_line(f"H-0138 oracle: {line}")
