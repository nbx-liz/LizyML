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
from fractions import Fraction
from typing import Any

import numpy as np

from lizyml.core.magnitude import (
    exact,
    finite_float,
    int_too_large,
    width_too_large,
)
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim, SearchDim
from lizyml.tuning._space_format import (
    check_float_width,
    format_error,
    int_too_large_error,
    read_float_bound,
    read_float_step,
    read_int_bound,
    read_int_step,
)

__all__ = [
    "check_dim_magnitude",
    "check_dim_step",
    "check_grid",
    "exact",
    "expansion_edge",
    "grid_value",
    "recorded_params",
    "snap_down",
    "snap_up",
    "suggest_on_grid",
]


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


def _shown(value: Fraction, kind: str) -> float | int | str:
    """A grid edge for the refusal message.

    The enclosing range of finite bounds near the float limit can reach past
    it (``[-1.7e308, -1e308]`` step ``1e308`` encloses ``-2e308``), so such an
    edge is shown as its exact decimal text instead of raising
    ``OverflowError``.
    """
    try:
        return _plain(value, kind)
    except OverflowError:
        return str(value.numerator // value.denominator)


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
    shown_low, shown_high = _shown(grid_low, kind), _shown(grid_high, kind)
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


def _refuse_text(dim: FloatDim | IntDim, keys: Sequence[str], *, surface: str) -> None:
    """Refuse a number written as text on a dimension built in Python.

    Text is read as a number only from a config or ``space=`` mapping; a
    dimension built in Python hands its values to Optuna as they are, where a
    string fails every trial after the study exists, or (a big ``int`` as text)
    is sampled coarsely without a word.
    """
    for key in keys:
        value = getattr(dim, key)
        if isinstance(value, str):
            raise format_error(
                surface,
                f"dimension '{dim.name}' has {key}={value!r} (str): a dimension "
                "built in Python takes numbers; text is read as a number only "
                "in tuning.optuna.space or tune(space=...).",
                reason="bound_not_a_number",
                dim=dim.name,
                key=key,
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
    _refuse_text(dim, ("step", "low", "high"), surface=surface)
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

    The result is in the order of *dims*, not of *params*: Optuna 3.1 returns a
    stored trial's parameters in a different order from the live trial's, and
    the live and restored ``tuning_table`` columns must agree (H-0126).

    Raises:
        ValueError: A parameter has no dimension in *dims*.
    """
    by_name = {dim.name: dim for dim in dims}
    for name in params:
        if name not in by_name:
            raise ValueError(
                f"trial parameter {name!r} has no dimension in its round's space "
                f"{sorted(by_name)}"
            )
    out: dict[str, Any] = {}
    for dim in dims:
        if dim.name not in params:
            continue
        value = params[dim.name]
        if isinstance(dim, CategoricalDim) or dim.step is None:
            out[dim.name] = value
        else:
            out[dim.name] = grid_value(dim, value)
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


def _is_number(value: Any) -> bool:
    """A real number by type: ``int`` / ``float`` or numpy's, not ``bool``."""
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def check_dim_magnitude(dim: SearchDim, *, surface: str) -> None:
    """Policy 9 of H-0124 amendment 1 for a dimension built directly.

    ``Tuner.tune`` and the round admission check run it on every numeric
    dimension, stepped or not: an ``int`` dimension's ``low``, ``high`` and
    ``step`` (checked in that order) are at most ``2**52`` in absolute value,
    and a ``float`` dimension's exact width ``high - low`` is at most the
    largest float (a bound that is not finite has no finite width).

    A number written as text is refused (``bound_not_a_number``): it is read
    as a number only from a mapping, and Optuna would get the string. Other
    values that are not numbers are left to Optuna as before; a NaN is
    unordered and is left the same way. A ``float`` bound that is not finite
    is ``bound_not_finite`` with its own key.

    Raises:
        LizyMLError: ``CONFIG_INVALID`` with ``reason`` ``bound_not_a_number``
            or ``bound_too_large`` (``key`` ``low`` / ``high`` / ``step``),
            ``bound_not_finite`` (``key`` ``low`` / ``high``) or
            ``range_not_finite`` (``key`` ``high``).
    """
    if isinstance(dim, CategoricalDim):
        return
    _refuse_text(dim, ("low", "high", "step"), surface=surface)
    if isinstance(dim, IntDim):
        for key in ("low", "high", "step"):
            value = getattr(dim, key)
            if _is_number(value) and int_too_large(value):
                raise int_too_large_error(dim.name, key, value, surface=surface)
        return
    low, high = dim.low, dim.high
    if not (_is_number(low) and _is_number(high)):
        return
    if any(type(v) is not int and math.isnan(v) for v in (low, high)):
        return
    for key, value in (("low", low), ("high", high)):
        if not finite_float(value):
            raise format_error(
                surface,
                f"dimension '{dim.name}' has {key}={value!r}, which is not a "
                "finite float.",
                reason="bound_not_finite",
                dim=dim.name,
                key=key,
            )
    check_float_width(dim.name, low, high, surface=surface)


def expansion_edge(
    dim: FloatDim | IntDim, edge: str, candidate: float | int | Fraction
) -> float | int | None:
    """The expanded *edge* as *dim*'s type, or ``None`` when it is not taken.

    H-0124 amendment 1, policy 10: *candidate* is the moved edge after every
    expansion rule (the provider bounds included). It becomes ``x`` -- the
    nearest ``float`` for a ``float`` dimension, the integer for an ``int``
    one -- and is taken only when ``x`` is finite and the exact width to the
    edge that does not move is at most the largest float (``float``), or
    ``|x| <= 2**52`` (``int``); on the grid as read back,
    ``Fraction(Decimal(str(x)))`` (a stepped dimension: a grid point near the
    float limit can leave the grid once it is a ``float``); and above 0 on a
    log scale (a subnormal lower edge divided by 3 rounds to 0.0). Whether
    ``x`` is outside the old edge is the caller's no-op test (H-0111).
    """
    x: float | int
    if isinstance(dim, IntDim):
        x = int(candidate)
        if int_too_large(x):
            return None
    else:
        try:
            x = float(candidate)
        except OverflowError:
            return None
        if not math.isfinite(x):
            return None
        low, high = (x, dim.high) if edge == "lower" else (dim.low, x)
        if width_too_large(low, high):
            return None
    if dim.step is not None and (exact(x) / exact(dim.step)).denominator != 1:
        return None
    if dim.log and x <= 0:
        return None
    return x
