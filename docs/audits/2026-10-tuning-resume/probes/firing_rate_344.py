"""Firing rate of the #344 search-space shape checks, measured on the suite.

Loaded as a pytest plugin (``-p``) with ``docs/audits/2026-10-tuning-resume/probes``
on ``PYTHONPATH``. It wraps every module-level reference to ``parse_space`` and,
without changing the outcome, records for each dimension spec whether the
proposed strict shape rule would refuse it, and why:

- a key other than ``type``, ``low``, ``high``, ``log``, ``category`` (float/int)
  or ``type``, ``choices``, ``category`` (categorical);
- ``log`` present and not a ``bool``;
- ``low`` / ``high`` not a real number, or a ``bool``; for ``int``, not integral;
- ``category`` present and not one of ``model``, ``smart``, ``training``;
- ``categorical`` whose ``choices`` is not a list or tuple.

Population: every dimension spec passed to ``parse_space`` in the suite. Writes
``<nodeid>\t<dim>\t<reasons or "">`` per spec to ``$FIRING_RATE_344_OUT``.
"""

from __future__ import annotations

import numbers
import os
from collections.abc import Mapping
from typing import Any

import pytest

from lizyml.core import _tuning_admission, _tuning_plan, _tuning_resume
from lizyml.tuning import search_space

_current = {"nodeid": "<none>"}
_rows: list[str] = []
_original = search_space.parse_space

_NUMERIC_KEYS = {"type", "low", "high", "log", "category"}
_CATEGORICAL_KEYS = {"type", "choices", "category"}
_CATEGORIES = {"model", "smart", "training"}


def _reasons(spec: Any) -> list[str]:
    if not isinstance(spec, Mapping):
        return ["not_a_mapping"]
    out: list[str] = []
    kind = spec.get("type")
    allowed = _CATEGORICAL_KEYS if kind == "categorical" else _NUMERIC_KEYS
    extra = sorted(str(k) for k in spec if k not in allowed)
    if extra:
        out.append("extra_keys:" + ",".join(extra))
    if "category" in spec and spec["category"] not in _CATEGORIES:
        out.append("category")
    if kind == "categorical" and not isinstance(spec.get("choices"), (list, tuple)):
        out.append("choices_not_sequence")
    if kind in ("float", "int"):
        if "log" in spec and type(spec["log"]) is not bool:
            out.append("log_not_bool")
        for edge in ("low", "high"):
            if edge not in spec:
                continue
            value = spec[edge]
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                out.append(f"{edge}_not_number")
            elif kind == "int" and float(value) != int(value):
                out.append(f"{edge}_not_integral")
    return out


def _wrapped(space: Any) -> Any:
    if isinstance(space, Mapping):
        for name, spec in space.items():
            reasons = _reasons(spec)
            _rows.append(f"{_current['nodeid']}\t{name}\t{';'.join(reasons)}")
    return _original(space)


for module in (search_space, _tuning_admission, _tuning_plan, _tuning_resume):
    if getattr(module, "parse_space", None) is _original:
        module.parse_space = _wrapped  # type: ignore[attr-defined]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("FIRING_RATE_344_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_rows) + "\n")
