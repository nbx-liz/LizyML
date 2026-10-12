"""Firing rate of the H-0121 amendment 2 smart-dimension value rule (#347), on the suite.

Loaded as a pytest plugin (``-p``) with ``docs/audits/2026-10-tuning-resume/probes``
on ``PYTHONPATH``. Without changing any outcome it records:

- ``dim`` rows: every ``category: smart`` dimension of a round's effective space
  that ``validate_tuning_dimensions`` sees (config space, ``tune(space=...)``,
  provider default dims, expanded dims), and whether the proposed rule would
  refuse it:
  - ``auto_num_leaves``: a categorical whose choices are all ``bool``;
  - ``balanced``: a categorical whose choices are ``bool`` or ``None``, and no
    ``True`` for a regression task;
  - ``num_leaves_ratio``: every value a number (``int`` / ``float``, not
    ``bool``) in ``(0, 1]`` -- a numeric dimension's ``low > 0`` and
    ``high <= 1``, a categorical's every choice;
  - ``min_data_in_leaf_ratio`` / ``min_data_in_bin_ratio``: every value in
    ``(0, 1)``, or ``None`` as a categorical choice;
  - ``feature_weights``: refused as a dimension.
- ``expand`` rows: every ``num_leaves_ratio`` edge ``expand_dims`` expands
  today, and whether the new edge passes ``1.0`` (the provider bound the
  amendment lowers from ``2.0``).

Writes ``<kind>\t<nodeid>\t<dim>\t<reasons or "">`` rows to
``$FIRING_RATE_347_OUT``.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest

from lizyml.core import _tuning_admission, _tuning_plan
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.tuning import search_space

_current = {"nodeid": "<none>"}
_rows: list[str] = []
_validate = _tuning_admission.validate_tuning_dimensions
_expand = search_space.expand_dims


def _number(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _in(v: Any, upper_inclusive: bool) -> bool:
    return _number(v) and v > 0 and (v <= 1 if upper_inclusive else v < 1)


def _reasons(dim: Any, task: str) -> list[str]:
    name = dim.name
    if name == "feature_weights":
        return ["feature_weights_dim"]
    if name in ("auto_num_leaves", "balanced"):
        if not isinstance(dim, CategoricalDim):
            return ["bool_param_numeric_dim"]
        allowed = (bool,) if name == "auto_num_leaves" else (bool, type(None))
        bad = [c for c in dim.choices if type(c) not in allowed]
        out = [f"choice_not_bool:{bad!r}"] if bad else []
        if name == "balanced" and task == "regression" and True in [
            c for c in dim.choices if type(c) is bool
        ]:
            out.append("balanced_true_regression")
        return out
    if name in ("num_leaves_ratio", "min_data_in_leaf_ratio", "min_data_in_bin_ratio"):
        inclusive = name == "num_leaves_ratio"
        if isinstance(dim, (FloatDim, IntDim)):
            ok = _in(dim.low, inclusive) and _in(dim.high, inclusive)
            return [] if ok else [f"range_outside:{dim.low!r}..{dim.high!r}"]
        allow_none = name != "num_leaves_ratio"
        bad = [
            c
            for c in dim.choices
            if not (_in(c, inclusive) or (allow_none and c is None))
        ]
        return [f"choice_outside:{bad!r}"] if bad else []
    return []


def _wrapped_validate(provider: Any, space: Any, smart: Any, task: Any, *args: Any, **kwargs: Any) -> Any:
    for dim in space:
        if getattr(dim, "category", None) == "smart":
            reasons = _reasons(dim, task)
            _rows.append(f"dim\t{_current['nodeid']}\t{dim.name}\t{';'.join(reasons)}")
    return _validate(provider, space, smart, task, *args, **kwargs)


def _wrapped_expand(dims: Any, report: Any, *args: Any, **kwargs: Any) -> Any:
    result = _expand(dims, report, *args, **kwargs)
    for status in report.dims:
        if status.name == "num_leaves_ratio" and status.expanded:
            past = status.new_high is not None and status.new_high > 1.0
            _rows.append(
                f"expand\t{_current['nodeid']}\t{status.name}\t{'past_1.0' if past else ''}"
            )
    return result


_tuning_admission.validate_tuning_dimensions = _wrapped_validate  # type: ignore[assignment]
for module in (search_space, _tuning_plan):
    if getattr(module, "expand_dims", None) is _expand:
        module.expand_dims = _wrapped_expand  # type: ignore[attr-defined]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("FIRING_RATE_347_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_rows) + "\n")
