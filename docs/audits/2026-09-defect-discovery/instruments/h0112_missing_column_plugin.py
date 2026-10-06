"""pytest plugin for #311 (H-0112): count validator calls whose named column is missing.

Wraps ``validate_no_target_leakage`` and ``validate_time_series_order`` in both
``lizyml.data.validators`` and the ``lizyml.data`` re-export, before any test
module imports them, and records each call and whether its named column
(``target`` / ``time_col``) was absent from the frame. A call with an absent
column is one whose behaviour H-0112 changes (``[]`` -> ``DATA_SCHEMA_INVALID``).

Run from the repo root:

    PYTHONPATH=docs/audits/2026-09-defect-discovery/instruments \
        uv run pytest -p h0112_missing_column_plugin -q

Totals and the tests that made an absent-column call are printed at the end.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

import lizyml.data as _pkg
import lizyml.data.validators as _v

_calls: Counter[str] = Counter()
_absent_tests: set[str] = set()


def _wrap(name: str, column_arg: str, position: int) -> None:
    real = getattr(_v, name)

    def counting(*args: Any, **kwargs: Any) -> Any:
        df = args[0] if args else kwargs["df"]
        column = args[position] if len(args) > position else kwargs[column_arg]
        _calls[f"{name}:all"] += 1
        if column not in df.columns:
            _calls[f"{name}:absent"] += 1
            _absent_tests.add(os.environ.get("PYTEST_CURRENT_TEST", "?"))
        return real(*args, **kwargs)

    setattr(_v, name, counting)
    setattr(_pkg, name, counting)


_wrap("validate_no_target_leakage", "target", 1)
_wrap("validate_time_series_order", "time_col", 1)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:  # noqa: ARG001
    terminalreporter.write_line(f"h0112 validator calls: {dict(sorted(_calls.items()))}")
    for name in sorted(_absent_tests):
        terminalreporter.write_line(f"  absent column in: {name}")
