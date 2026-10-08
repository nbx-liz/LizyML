"""pytest plugin for #267: count guarded calls in the suite that the handler swallows.

Wraps ``lizyml.data.validators._series_perfectly_correlated`` and records each
call and each ``TypeError`` / ``ValueError`` it raises (the exceptions the
handler in ``validate_no_target_leakage`` catches today). A raise counted here
is a call whose behaviour changes when the handler is removed.

Run: ``PYTHONPATH=<this dir> pytest -p pr7_swallow_plugin``; totals are printed
at the end of the session.
"""

from __future__ import annotations

import os

import lizyml.data.validators as _v

_calls = {"guarded": 0, "raised": 0}
_raised_tests: list[str] = []
_real = _v._series_perfectly_correlated


def _counting(col, y):  # noqa: ANN001, ANN202
    _calls["guarded"] += 1
    try:
        return _real(col, y)
    except (TypeError, ValueError):
        _calls["raised"] += 1
        _raised_tests.append(os.environ.get("PYTEST_CURRENT_TEST", ""))
        raise


_v._series_perfectly_correlated = _counting


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ANN001, ARG001
    terminalreporter.write_line(f"pr7 guarded calls: {_calls}")
    for name in sorted(set(_raised_tests)):
        terminalreporter.write_line(f"  raised in: {name}")
