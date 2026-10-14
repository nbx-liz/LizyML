"""Firing rate of the H-0124 amendment 1 (policies 9 and 10), measured on the suite.

Loaded as a pytest plugin (``-p``) with ``docs/audits/2026-10-tuning-resume/probes``
on ``PYTHONPATH``. Without changing any outcome it records:

- ``spec`` rows: for every dimension ``parse_space`` accepts today, whether
  policy 9 would refuse it -- a ``float`` dimension whose exact width
  ``high - low`` exceeds ``sys.float_info.max`` (``range_not_finite``), or an
  ``int`` dimension whose ``low``, ``high`` or ``step`` exceeds ``2**52`` in
  absolute value (``bound_too_large``). Specs ``parse_space`` refuses today are
  in the denominator with an empty reason.
- ``expand`` rows: for every edge ``expand_dims`` expands today, whether
  policy 10 would not expand it. The candidate is the edge before today's
  float-range cap (H-0124 ``_expand_on_grid`` stops a stepped float edge at
  the last grid point inside the float range; policy 10 drops that cap), after
  today's provider clamp. Policy 10 vetoes the edge when, as a value of the
  dimension type, it is not finite or its width with the fixed edge exceeds
  ``sys.float_info.max`` (``float``), exceeds ``2**52`` in absolute value
  (``int``), or -- for a stepped dimension -- is off the grid as read back
  (``Fraction(Decimal(str(x)))``).

Writes ``<kind>\t<nodeid>\t<dim>\t<reasons or "">`` rows to
``$FIRING_RATE_H0124_OUT``.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest

from lizyml.core import _tuning_admission, _tuning_plan, _tuning_resume
from lizyml.core.types.search_dim import FloatDim, IntDim
from lizyml.tuning import search_space

INT_LIMIT = 2**52
F = Fraction(sys.float_info.max)
_current = {"nodeid": "<none>"}
_rows: list[str] = []
_parse = search_space.parse_space
_expand = search_space.expand_dims


def _exact(x: Any) -> Fraction:
    return Fraction(x) if isinstance(x, int) else Fraction(Decimal(str(float(x))))


def _spec_reasons(dim: Any) -> list[str]:
    if isinstance(dim, FloatDim):
        return ["range_not_finite"] if _exact(dim.high) - _exact(dim.low) > F else []
    if isinstance(dim, IntDim):
        values = [dim.low, dim.high] + ([dim.step] if dim.step is not None else [])
        return ["bound_too_large"] if any(abs(v) > INT_LIMIT for v in values) else []
    return []


def _natural_float_upper(dim: FloatDim) -> Fraction:
    """Today's stepped upper candidate before the float-range cap, after the clamp."""
    low, high = _exact(dim.low), _exact(dim.high)
    cand = high * 3 if dim.log and high > 0 else high + (high - low)
    if dim.step is not None:
        s = _exact(dim.step)
        cand = -((-cand) // s) * s
    if dim.max_allowed is not None and cand > max(high, _exact(dim.max_allowed)):
        cand = max(high, _exact(dim.max_allowed))
        if dim.step is not None:
            cand = max(high, (cand // _exact(dim.step)) * _exact(dim.step))
    return cand


def _veto(dim: Any, edge: str, new_value: Any) -> list[str]:
    """Policy 10's reasons not to expand *edge* of *dim* to today's result."""
    if isinstance(dim, IntDim):
        return ["int_beyond_2**52"] if abs(new_value) > INT_LIMIT else []
    reasons: list[str] = []
    if dim.step is not None and edge == "upper":
        candidate = _natural_float_upper(dim)  # the edge before today's cap
    else:
        candidate = None
    try:
        x = new_value if candidate is None else float(candidate)
    except OverflowError:
        return ["float_not_finite"]
    if not math.isfinite(x):
        return ["float_not_finite"]
    fixed = _exact(dim.low if edge == "upper" else dim.high)
    if abs(_exact(x) - fixed) > F:
        reasons.append("width_beyond_float")
    if dim.step is not None and (_exact(x) / _exact(dim.step)).denominator != 1:
        reasons.append("off_grid_as_float")
    return reasons


def _wrapped_parse(space: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        result = _parse(space, *args, **kwargs)  # raises exactly as today
    except Exception:
        if isinstance(space, Mapping):
            for name in space:
                _rows.append(f"spec\t{_current['nodeid']}\t{name}\t")
        raise
    for dim in result:
        _rows.append(f"spec\t{_current['nodeid']}\t{dim.name}\t{';'.join(_spec_reasons(dim))}")
    return result


def _wrapped_expand(dims: Any, report: Any, *args: Any, **kwargs: Any) -> Any:
    result = _expand(dims, report, *args, **kwargs)
    by_name = {d.name: d for d in dims}
    for status in report.dims:
        if not status.expanded or status.name not in by_name:
            continue
        dim = by_name[status.name]
        if not isinstance(dim, (FloatDim, IntDim)):
            continue  # a hand-made report may mark a categorical dim expanded
        new_value = status.new_high if status.edge == "upper" else status.new_low
        reasons = _veto(dim, status.edge, new_value)
        _rows.append(f"expand\t{_current['nodeid']}\t{dim.name}\t{';'.join(reasons)}")
    return result


for module in (search_space, _tuning_admission, _tuning_plan, _tuning_resume):
    if getattr(module, "parse_space", None) is _parse:
        module.parse_space = _wrapped_parse  # type: ignore[attr-defined]
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
