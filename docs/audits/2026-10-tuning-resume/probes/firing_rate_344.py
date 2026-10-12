"""Firing rate of the H-0123 search-space format rule (#344, v4), measured on the suite.

Loaded as a pytest plugin (``-p``) with ``docs/audits/2026-10-tuning-resume/probes``
on ``PYTHONPATH``. It wraps every module-level reference to ``parse_space`` and,
without changing the outcome, records for each dimension spec whether H-0123's
policy 3 (the documented format) would newly refuse it, and why. Refusals
``parse_space`` already makes (unknown ``type``, missing bound, empty or
non-iterable ``choices``, ``low >= high``, a log range with ``low <= 0``) are
not counted:

- a dimension name that is not a ``str``;
- a spec that is not a mapping;
- a key other than ``type``, ``low``, ``high``, ``log``, ``category`` (float/int)
  or ``type``, ``choices``, ``category`` (categorical) -- ``step`` included;
- ``category`` present and not one of ``model``, ``smart``, ``training``;
- ``log`` present and not a ``bool``;
- a bound that is not an ``int`` / ``float`` (``bool`` excluded), a numpy
  integer / floating scalar, or a ``str`` that fully matches a number literal;
  or one that is not finite; or, for ``int``, one that is not integral;
- ``choices`` that is not a ``list`` or ``tuple``.

Population: every dimension spec passed to ``parse_space`` in the suite. Writes
``<nodeid>\t<dim>\t<reasons or "">`` per spec to ``$FIRING_RATE_344_OUT``.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from lizyml.core import _tuning_admission, _tuning_plan, _tuning_resume
from lizyml.tuning import search_space

_current = {"nodeid": "<none>"}
_rows: list[str] = []
_original = search_space.parse_space

_NUMERIC_KEYS = {"type", "low", "high", "log", "category"}
_CATEGORICAL_KEYS = {"type", "choices", "category"}
_CATEGORIES = {"model", "smart", "training"}
_NUMBER_LITERAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")


def _bound_reason(value: Any, kind: str) -> str | None:
    if isinstance(value, str):
        if not _NUMBER_LITERAL.fullmatch(value):
            return "string"
        number: Any = float(value)
    elif type(value) in (int, float) or isinstance(value, (np.integer, np.floating)):
        number = value
    else:
        return "type"
    if isinstance(number, (int, np.integer)):
        return None
    if not math.isfinite(float(number)):
        return "not_finite"
    if kind == "int" and not float(number).is_integer():
        return "not_integral"
    return None


def _choices_reason(choices: Any) -> str | None:
    if type(choices) in (list, tuple):
        return None
    try:
        iter(choices)
    except TypeError:
        return None  # not iterable: parse_space already refuses it (raw today)
    return type(choices).__name__


def _reasons(spec: Any) -> list[str]:
    if not isinstance(spec, Mapping):
        return ["not_a_mapping"]
    out: list[str] = []
    kind = spec.get("type")
    allowed = _CATEGORICAL_KEYS if kind == "categorical" else _NUMERIC_KEYS
    extra = sorted(str(k) for k in spec if k not in allowed)
    if extra:
        out.append("extra_keys:" + ",".join(extra))
    category = spec.get("category", "model")
    if not (isinstance(category, str) and category in _CATEGORIES):
        out.append("category")
    if kind == "categorical" and "choices" in spec:
        reason = _choices_reason(spec["choices"])
        if reason:
            out.append(f"choices_{reason}")
    if kind in ("float", "int"):
        if "log" in spec and type(spec["log"]) is not bool:
            out.append("log_not_bool")
        for edge in ("low", "high"):
            if edge in spec:
                reason = _bound_reason(spec[edge], kind)
                if reason:
                    out.append(f"{edge}_{reason}")
    return out


def _wrapped(space: Any) -> Any:
    if isinstance(space, Mapping):
        for name, spec in space.items():
            reasons = _reasons(spec)
            if not isinstance(name, str):
                reasons = ["name", *reasons]
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
