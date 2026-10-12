"""Firing rate of the H-0124 amendment's magnitude rule (policy 9), measured on the suite.

Loaded as a pytest plugin (``-p``) with ``docs/audits/2026-10-tuning-resume/probes``
on ``PYTHONPATH``. Without changing any outcome it records:

- for every dimension spec ``parse_space`` accepts today, whether policy 9
  would refuse it: a ``float`` dimension whose width ``high - low`` is not a
  finite float (``range_not_finite``), or an ``int`` dimension whose ``low``,
  ``high`` or ``step`` exceeds ``2**52`` in absolute value (``bound_too_large``);
- for every dimension ``expand_dims`` returns, whether an expanded edge lies
  outside those limits (``expanded_beyond``) -- the expansions policy 9 caps.

Specs ``parse_space`` already refuses are counted in the denominator with an
empty reason (the rule does not change their outcome).

Writes ``<kind>\t<nodeid>\t<dim>\t<reasons or "">`` rows to
``$FIRING_RATE_H0124_OUT``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

import pytest

from lizyml.core import _tuning_admission, _tuning_plan, _tuning_resume
from lizyml.core.types.search_dim import FloatDim, IntDim
from lizyml.tuning import search_space

INT_LIMIT = 2**52
_current = {"nodeid": "<none>"}
_rows: list[str] = []
_parse = search_space.parse_space
_expand = search_space.expand_dims


def _magnitude(dim: Any) -> list[str]:
    if isinstance(dim, FloatDim):
        try:
            width = float(dim.high) - float(dim.low)
        except OverflowError:
            return ["range_not_finite"]
        return [] if math.isfinite(width) else ["range_not_finite"]
    if isinstance(dim, IntDim):
        values = [dim.low, dim.high] + ([dim.step] if dim.step is not None else [])
        return ["bound_too_large"] if any(abs(v) > INT_LIMIT for v in values) else []
    return []


def _wrapped_parse(space: Any, *args: Any, **kwargs: Any) -> Any:
    result = _parse(space, *args, **kwargs)  # raises exactly as today
    for dim in result:
        _rows.append(f"spec\t{_current['nodeid']}\t{dim.name}\t{';'.join(_magnitude(dim))}")
    return result


def _wrapped_parse_counting_refusals(space: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return _wrapped_parse(space, *args, **kwargs)
    except Exception:
        if isinstance(space, Mapping):
            for name in space:
                _rows.append(f"spec\t{_current['nodeid']}\t{name}\t")
        raise


def _wrapped_expand(dims: Any, *args: Any, **kwargs: Any) -> Any:
    result = _expand(dims, *args, **kwargs)
    for dim in result:
        reasons = ["expanded_beyond"] if _magnitude(dim) else []
        if isinstance(dim, FloatDim) and not (math.isfinite(dim.low) and math.isfinite(dim.high)):
            reasons = ["expanded_beyond"]
        _rows.append(f"expand\t{_current['nodeid']}\t{dim.name}\t{';'.join(reasons)}")
    return result


for module in (search_space, _tuning_admission, _tuning_plan, _tuning_resume):
    if getattr(module, "parse_space", None) is _parse:
        module.parse_space = _wrapped_parse_counting_refusals  # type: ignore[attr-defined]
    if getattr(module, "expand_dims", None) is _expand:
        module.expand_dims = _wrapped_expand  # type: ignore[attr-defined]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("FIRING_RATE_H0124_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_rows) + "\n")
