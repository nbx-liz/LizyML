"""#391 census plugin: how often the frozenset-collapse refusal is asked and fires.

    pytest -p docs.audits... is awkward; run with
    PYTHONPATH=.:docs/audits/2026-10-i391/probes python -m pytest -p collapse_census ...

Wraps ``lizyml.core.category_values._collapse_refusal`` (the prototype rule) in
the pytest process, counts calls and refusals with the test that made them,
and writes ``$COLLAPSE_CENSUS_OUT`` (JSON) at the end of the session. Calls in
subprocesses (the H-0138 answers, notebook kernels) are not counted.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import pytest

from lizyml.core import category_values as cv

_CALLS: Counter[str] = Counter()
_FIRES: Counter[str] = Counter()
_EXAMPLES: dict[str, str] = {}
_CURRENT = {"test": "<collection>"}
_ORIGINAL = cv._collapse_refusal


def _counted(value: frozenset[Any], depth: int) -> str | None:
    reason = _ORIGINAL(value, depth)
    test = _CURRENT["test"]
    _CALLS[test] += 1
    if reason is not None:
        _FIRES[test] += 1
        _EXAMPLES.setdefault(test, f"{cv.shown(value)[:200]} -> {reason}")
    return reason


cv._collapse_refusal = _counted


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: Any) -> Any:
    _CURRENT["test"] = item.nodeid
    yield
    _CURRENT["test"] = "<between tests>"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("COLLAPSE_CENSUS_OUT", "collapse_census.json")
    payload = {
        "calls": sum(_CALLS.values()),
        "fires": sum(_FIRES.values()),
        "tests_calling": len(_CALLS),
        "tests_firing": dict(_FIRES),
        "examples": _EXAMPLES,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
