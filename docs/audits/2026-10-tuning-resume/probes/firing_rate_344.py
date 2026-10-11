"""Firing rate of the H-0123 search-space shape rule (#344), measured on the suite.

Loaded as a pytest plugin (``-p``) with ``docs/audits/2026-10-tuning-resume/probes``
on ``PYTHONPATH``. It wraps every module-level reference to ``parse_space`` and,
without changing the outcome, records for each dimension spec whether H-0123's
policy 3 would newly refuse it, and why. Refusals ``parse_space`` already makes
(unknown ``type``, missing bound, empty or non-iterable ``choices``,
``low >= high``, a log range with ``low <= 0``) are not counted:

- a key other than ``type``, ``low``, ``high``, ``log``, ``category`` (float/int)
  or ``type``, ``choices``, ``category`` (categorical);
- ``category`` present and not one of ``model``, ``smart``, ``training``;
- ``log`` present and not a ``bool``;
- a bound that is a ``bool``, ``str`` or ``bytes``; for ``float``, one whose
  ``float()`` fails or is not finite; for ``int``, one whose ``int()`` fails or
  is not equal to the value (compared without passing through ``float``);
- ``choices`` that is a ``str`` / ``bytes``, a mapping, a set, or an iterator.

Population: every dimension spec passed to ``parse_space`` in the suite. Writes
``<nodeid>\t<dim>\t<reasons or "">`` per spec to ``$FIRING_RATE_344_OUT``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
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


def _bound_reason(value: Any, kind: str) -> str | None:
    if isinstance(value, (bool, str, bytes)):
        return "type"
    if kind == "float":
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return "not_number"
        return None if math.isfinite(converted) else "not_finite"
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return "not_integer"
    try:
        same = bool(integer == value)
    except Exception:  # noqa: BLE001 -- an incomparable object is "not the same value"
        same = False
    return None if same else "not_integral"


def _choices_reason(choices: Any) -> str | None:
    if isinstance(choices, (str, bytes)):
        return "string"
    if isinstance(choices, Mapping):
        return "mapping"
    if isinstance(choices, AbstractSet):
        return "set"
    try:
        iterator = iter(choices)
    except TypeError:
        return None  # not iterable: parse_space already refuses it (raw today)
    return "iterator" if iterator is choices else None


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
