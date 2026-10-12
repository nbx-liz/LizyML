"""Which search-space specs in the suite fall outside the documented format.

Pytest plugin (``-p space_format_strict``, with this directory on ``PYTHONPATH``).
It wraps every module-level reference to ``parse_space`` and records, without
changing the outcome, each dimension spec that is outside the format
``docs/config-reference.md`` documents (``type``, ``low``, ``high``, ``log``,
``choices``, ``category``) read with the types YAML and JSON produce:

- a key outside the documented set for its ``type``;
- ``low`` / ``high`` that is not a finite ``int`` or ``float`` (``bool``
  excluded; numpy scalars reported separately as ``numpy``);
- an ``int`` dimension with a non-integral bound;
- ``log`` that is not a ``bool``;
- ``choices`` that is not a ``list`` (a tuple reported as ``tuple``);
- ``category`` that is not one of ``model`` / ``smart`` / ``training``.

Writes ``<nodeid>\t<dim>\t<reasons>`` for every spec to ``$SPACE_FORMAT_OUT``.
"""

from __future__ import annotations

import math
import os
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


def _bound(value: Any, kind: str) -> str | None:
    if isinstance(value, np.generic):
        return "numpy"
    if type(value) not in (int, float):
        return f"type:{type(value).__name__}"
    if not math.isfinite(value):
        return "not_finite"
    if kind == "int" and float(value) != int(value):
        return "not_integral"
    return None


def _reasons(spec: Any) -> list[str]:
    if not isinstance(spec, Mapping):
        return [f"spec:{type(spec).__name__}"]
    if type(spec) is not dict:
        out = [f"spec:{type(spec).__name__}"]
    else:
        out = []
    kind = spec.get("type")
    allowed = _CATEGORICAL_KEYS if kind == "categorical" else _NUMERIC_KEYS
    extra = sorted(str(k) for k in spec if k not in allowed)
    if extra:
        out.append("extra:" + ",".join(extra))
    cat = spec.get("category", "model")
    if not (isinstance(cat, str) and cat in {"model", "smart", "training"}):
        out.append("category")
    if kind == "categorical":
        ch = spec.get("choices")
        if type(ch) is not list:
            out.append(f"choices:{type(ch).__name__}")
    elif kind in ("float", "int"):
        if "log" in spec and type(spec["log"]) is not bool:
            out.append("log")
        for edge in ("low", "high"):
            if edge in spec:
                reason = _bound(spec[edge], kind)
                if reason:
                    out.append(f"{edge}:{reason}")
    return out


def _wrapped(space: Any) -> Any:
    if isinstance(space, Mapping):
        for name, spec in space.items():
            label = name if isinstance(name, str) else f"<{type(name).__name__}>{name!r}"
            _rows.append(f"{_current['nodeid']}\t{label}\t{';'.join(_reasons(spec))}")
    return _original(space)


for module in (search_space, _tuning_admission, _tuning_plan, _tuning_resume):
    if getattr(module, "parse_space", None) is _original:
        module.parse_space = _wrapped  # type: ignore[attr-defined]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("SPACE_FORMAT_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_rows) + "\n")
