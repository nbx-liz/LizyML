"""The grid of a stepped search dimension (H-0124).

A ``FloatDim`` / ``IntDim`` with a ``step`` searches the multiples of ``step``
from 0 between ``low`` and ``high``; both bounds must be multiples. The value a
trial trains on -- and the value LizyML records for it -- is the grid point its
raw sample rounds to (:func:`grid_value`), decided from the raw value and the
dimension of the trial's round, never stored (:func:`recorded_params`).

All grid arithmetic is exact: a ``float`` ``x`` is read as
``Fraction(Decimal(str(x)))`` -- ``str`` is the shortest form, so ``0.3`` and
``0.1`` are 3/10 and 1/10, and ``0.3`` is a multiple of ``0.1`` even though
``0.3 % 0.1`` is not 0 in binary. ``Fraction`` does not depend on the
``decimal`` context, so no precision setting changes a result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np

from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim, SearchDim
from lizyml.tuning._space_format import (
    format_error,
    read_float_bound,
    read_float_step,
    read_int_bound,
    read_int_step,
)

__all__ = [
    "check_dim_step",
    "check_grid",
    "exact",
    "grid_value",
    "recorded_params",
    "snap_down",
    "snap_up",
    "suggest_on_grid",
]


def exact(value: Any) -> Fraction:
    """*value* as an exact rational: an integer as itself, else its float's text.

    A ``float`` (or a numpy floating scalar) is read through the shortest
    ``str`` of its ``float`` value, the reading Optuna's own step arithmetic
    uses; an ``int`` or numpy integer is exact already.
    """
    if type(value) is int or isinstance(value, np.integer):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    return Fraction(Decimal(str(float(value))))


def snap_down(value: Fraction, step: Fraction) -> Fraction:
    """The largest grid point ``k * step`` not above *value*."""
    return math.floor(value / step) * step


def snap_up(value: Fraction, step: Fraction) -> Fraction:
    """The smallest grid point ``k * step`` not below *value*."""
    return math.ceil(value / step) * step


def _plain(value: Fraction, kind: str) -> float | int:
    """A grid point as the dimension's plain type."""
    if kind == "int":
        return int(value)
    return float(value)


def _range_on_grid(
    low: Fraction, high: Fraction, step: Fraction, *, log: bool
) -> tuple[Fraction, Fraction, bool]:
    """The range the ``bound_off_grid`` message shows, and whether it is inside.

    The inner range ``[ceil(low/step), floor(high/step)] * step`` when it holds
    at least two grid points, else the enclosing one
    ``[floor(low/step), ceil(high/step)] * step``. On a log scale the
    enclosing lower edge is at least ``step`` (0 is not on a log scale), and a
    range that collapses to one point is widened by one step upward.
    """
    inner_low, inner_high = snap_up(low, step), snap_down(high, step)
    if inner_low < inner_high:
        return inner_low, inner_high, True
    outer_low, outer_high = snap_down(low, step), snap_up(high, step)
    if log and outer_low <= 0:
        outer_low = step
        if outer_high <= outer_low:
            outer_high = outer_low + step
    return outer_low, outer_high, False


def check_grid(
    name: str,
    low: float | int,
    high: float | int,
    step: float | int,
    *,
    log: bool,
    kind: str,
    surface: str,
) -> None:
    """Refuse a ``low`` or ``high`` that is not a multiple of ``step``.

    The values are the plain ones the format readers return. ``low`` is
    reported before ``high``. The message shows a range on the grid
    (:func:`_range_on_grid`) instead of silently truncating it the way Optuna
    does.

    Raises:
        LizyMLError: ``CONFIG_INVALID`` with ``reason="bound_off_grid"`` and
            ``key`` ``"low"`` or ``"high"``.
    """
    s, lo, hi = exact(step), exact(low), exact(high)
    off_grid = [
        (key, value)
        for key, value, bound in (("low", low, lo), ("high", high, hi))
        if (bound / s).denominator != 1
    ]
    if not off_grid:
        return
    key, value = off_grid[0]
    grid_low, grid_high, inside = _range_on_grid(lo, hi, s, log=log)
    shown_low, shown_high = _plain(grid_low, kind), _plain(grid_high, kind)
    where = (
        "inside the declared range"
        if inside
        else "encloses the declared range, which holds fewer than two grid points"
    )
    raise format_error(
        surface,
        f"dimension '{name}' has {key}={value!r}, which is not a multiple of "
        f"step={step!r}: a stepped dimension searches the multiples of its step "
        f"from 0, and both bounds must be on that grid. "
        f"[{shown_low}, {shown_high}] is on it ({where}).",
        reason="bound_off_grid",
        dim=name,
        key=key,
        grid_low=shown_low,
        grid_high=shown_high,
    )


def check_dim_step(dim: SearchDim, *, surface: str) -> None:
    """The ``step`` checks of ``parse_space`` for a dimension built directly.

    A ``FloatDim`` / ``IntDim`` made in Python skips the format readers, so its
    ``step`` is read by them here -- type, finite, positive, whole for an
    ``int`` dimension -- and its bounds with it, then the grid is checked.
    A dimension without ``step``, and a ``CategoricalDim``, pass unchecked.

    Raises:
        LizyMLError: ``CONFIG_INVALID``, the same reasons as ``parse_space``.
    """
    if isinstance(dim, CategoricalDim) or dim.step is None:
        return
    if isinstance(dim, IntDim):
        step_i = read_int_step(dim.name, dim.step, surface=surface)
        low_i = read_int_bound(dim.name, "low", dim.low, surface=surface)
        high_i = read_int_bound(dim.name, "high", dim.high, surface=surface)
        check_grid(
            dim.name, low_i, high_i, step_i, log=dim.log, kind="int", surface=surface
        )
        return
    step_f = read_float_step(dim.name, dim.step, surface=surface)
    low_f = read_float_bound(dim.name, "low", dim.low, surface=surface)
    high_f = read_float_bound(dim.name, "high", dim.high, surface=surface)
    check_grid(
        dim.name, low_f, high_f, step_f, log=dim.log, kind="float", surface=surface
    )


def grid_value(dim: FloatDim | IntDim, raw: Any) -> float | int:
    """The grid point a stepped dimension's raw sample *raw* trains on.

    ``k = round(raw / step)`` -- Python rounds a ``Fraction`` half to even,
    exactly -- clamped to the grid points in ``[low, high]``; the value is
    ``k * step`` as a ``float`` (the nearest one) or, for an ``IntDim``, an
    ``int``. Rounded once: an ``int`` dimension sampled as a float is not
    rounded to an integer first.

    Raises:
        ValueError: *dim* has no ``step``.
    """
    if dim.step is None:
        raise ValueError(f"dimension '{dim.name}' has no step")
    s = exact(dim.step)
    k = round(exact(raw) / s)
    k = max(math.ceil(exact(dim.low) / s), min(k, math.floor(exact(dim.high) / s)))
    return _plain(k * s, "int" if isinstance(dim, IntDim) else "float")


def recorded_params(
    params: Mapping[str, Any], dims: Sequence[SearchDim]
) -> dict[str, Any]:
    """The values a trial trained on, from its raw *params* and its round's *dims*.

    A stepped dimension's value is its :func:`grid_value`; any other value is
    the raw one. *dims* must be the space of the round the trial belongs to.
    *params* may hold only some of the dims (a failed trial), but every name
    in it must have a dimension.

    Raises:
        ValueError: A parameter has no dimension in *dims*.
    """
    by_name = {dim.name: dim for dim in dims}
    out: dict[str, Any] = {}
    for name, value in params.items():
        dim = by_name.get(name)
        if dim is None:
            raise ValueError(
                f"trial parameter {name!r} has no dimension in its round's space "
                f"{sorted(by_name)}"
            )
        if isinstance(dim, CategoricalDim) or dim.step is None:
            out[name] = value
        else:
            out[name] = grid_value(dim, value)
    return out


def suggest_on_grid(trial: Any, dim: FloatDim | IntDim) -> float | int:
    """Sample stepped *dim* from an Optuna *trial* and return its grid value.

    Off a log scale Optuna samples the grid itself (``step=``); the grid value
    then only removes binary noise (``0.30000000000000004`` is ``0.3``). Optuna
    cannot combine ``log`` and ``step``, so a log dim is sampled without a step
    and rounded once by :func:`grid_value`: a ``float`` one through
    ``suggest_float(log=True)``, an ``int`` one with ``step > 1`` through
    ``suggest_float(log=True)`` too (``suggest_int`` would round to an integer
    first), and an ``int`` one with ``step == 1`` through
    ``suggest_int(log=True)``, as without a step.
    """
    if isinstance(dim, FloatDim):
        if dim.log:
            raw = trial.suggest_float(dim.name, dim.low, dim.high, log=True)
        else:
            raw = trial.suggest_float(
                dim.name, dim.low, dim.high, log=False, step=dim.step
            )
    elif not dim.log:
        raw = trial.suggest_int(dim.name, dim.low, dim.high, log=False, step=dim.step)
    elif dim.step == 1:
        raw = trial.suggest_int(dim.name, dim.low, dim.high, log=True)
    else:
        raw = trial.suggest_float(dim.name, float(dim.low), float(dim.high), log=True)
    return grid_value(dim, raw)
