"""SearchSpace — optuna-independent representation of hyperparameter search spaces.

Type definitions (``SearchDim``, ``FloatDim``, ``IntDim``, ``CategoricalDim``,
``DimCategory``) live in ``core/types/search_dim.py`` (Foundation layer) and are
re-exported here for backward compatibility.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, TypeVar

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import (
    CategoricalDim,
    DimCategory,
    FloatDim,
    IntDim,
    SearchDim,
)
from lizyml.core.types.tuning_result import BoundaryDimStatus, BoundaryReport
from lizyml.tuning._space_format import (
    DIM_KEYS,
    check_dim_keys,
    check_float_width,
    check_space_shape,
    format_error,
    read_choices,
    read_float_bound,
    read_float_step,
    read_int_bound,
    read_int_step,
    read_log,
)
from lizyml.tuning._step_grid import (
    check_grid,
    exact,
    expansion_edge,
    snap_down,
    snap_up,
    suggest_on_grid,
)

__all__ = [
    "CategoricalDim",
    "FloatDim",
    "IntDim",
    "SearchDim",
    "attach_bounds",
    "detect_boundary",
    "expand_dims",
    "parse_space",
    "split_by_category",
    "suggest_params",
]


_ALLOWED_CHOICE_TYPES = (type(None), bool, int, float, str)
"""The exact types a categorical choice may have.

Two boundaries meet here, and neither is ``param_domain``'s accepted set:
Optuna's ``CategoricalDistribution`` takes a plain scalar, and a sampled value
reaches ``lgb.train`` **without passing an entrance normaliser**, so it also has
to be a value the exit assertion in the adapter accepts as it stands. The
accepted set is the wider one -- it admits a numpy scalar and converts it -- and
restating it here would be two declarations of one boundary (BLUEPRINT 14.4).
"""


def _validate_numeric_range(
    name: str,
    low: float,
    high: float,
    *,
    log: bool,
    surface: str,
    float_width: bool = False,
) -> None:
    """Validate ``low < high``, (when ``log`` is True) ``low > 0`` and the width.

    Optuna only raises these errors at trial-time with generic messages.
    Catching them at parse time gives users a clear, configuration-level
    diagnostic with the offending bounds in ``context``. With *float_width*
    (a ``float`` dimension), the exact width ``high - low`` must be at most the
    largest float (H-0124 amendment 1, policy 9; ``range_not_finite``).
    """
    if low >= high:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"{surface}: Search dim '{name}' has low >= high "
                f"(low={low}, high={high}). "
                f"Min must be strictly less than Max."
            ),
            context={
                "param": name,
                "low": low,
                "high": high,
                "log": log,
                "surface": surface,
                "reason": "empty_range",
                "dim": name,
                "key": "low",
            },
        )
    if log and low <= 0:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"{surface}: Search dim '{name}' uses log distribution but "
                f"low={low} <= 0. "
                f"Log requires a strictly positive lower bound."
            ),
            context={
                "param": name,
                "low": low,
                "high": high,
                "log": log,
                "surface": surface,
                "reason": "log_low_not_positive",
                "dim": name,
                "key": "low",
            },
        )
    if float_width:
        check_float_width(name, low, high, surface=surface)


def _validate_categorical_choices(
    name: str, choices: list[Any] | tuple[Any, ...], *, surface: str
) -> None:
    """Validate that every element in *choices* is a plain scalar.

    Non-scalar values (e.g. a nested list produced by YAML ``- [a, b]``) are
    rejected early with a clear error message. So is a numpy scalar, and the
    reason is not cosmetic: a sampled value is overlaid onto the trial params
    without passing an entrance normaliser, so a numpy one reached the exit
    assertion in the adapter and failed **every** trial. The user was then told
    ``TUNING_FAILED: All tuning trials failed. Check parameter ranges.`` with
    ranges that were fine (#287).

    The type is judged by **identity**, for two separate reasons:

    * ``isinstance`` admits a subclass, and ``np.float64`` subclasses ``float``
      while ``np.str_`` subclasses ``str``. Those two were the whole hole: every
      other numpy scalar type was already refused here.
    * ``type(val) in _ALLOWED_CHOICE_TYPES`` asks the tuple whether any member
      *equals* the type, and a class's ``__eq__`` comes from its metaclass, which
      the caller writes. Measured: a class whose metaclass returns ``True`` for
      ``float`` passes that membership test, and ``__hash__`` is never consulted
      because this container is a tuple -- a ``set`` would hash first (BLUEPRINT
      14.4 states the ``set`` case, from review round 23).
    """
    for i, val in enumerate(choices):
        if not any(type(val) is allowed for allowed in _ALLOWED_CHOICE_TYPES):
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    f"{surface}: Categorical dim '{name}' has invalid choice at "
                    f"index {i}: "
                    f"got {val!r} (type={type(val).__name__}). "
                    f"Each choice must be a plain Python scalar (str, int, float, "
                    f"bool, or None) -- a numpy scalar is refused because a "
                    f"sampled choice is not normalised before training. "
                    f"Hint: write float(x) or int(x), and flatten nested lists in "
                    f'your YAML config — use "- value" instead of "- [v1, v2]".'
                ),
                context={
                    "param": name,
                    "index": i,
                    "bad_value": str(val),
                    "surface": surface,
                    "reason": "invalid_choice",
                    "dim": name,
                    "key": "choices",
                },
            )


def parse_space(
    space: Mapping[str, Any], *, surface: str = "tuning.optuna.space"
) -> list[SearchDim]:
    """Parse a config-style space dict into typed SearchDim instances.

    Space dict format (docs/config-reference.md, BLUEPRINT §11.1)::

        {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 16, "high": 256},
            "subsample": {"type": "categorical", "choices": [0.6, 0.8, 1.0]},
        }

    Only that format is accepted (H-0123): a spec outside it is refused, not
    converted into another search. ``tuning.optuna.space`` and
    ``tune(space=...)`` both pass here, so they share one rule; the checks are
    in :mod:`lizyml.tuning._space_format`.

    Args:
        space: Raw search space mapping, from config or ``tune(space=...)``.
        surface: Where *space* was declared, quoted in every refusal's
            message and ``context["surface"]``.

    A ``step`` (H-0124) is read like a bound and must be positive (at least 1
    on an ``int`` dimension); ``low`` and ``high`` must then be multiples of it
    (:func:`lizyml.tuning._step_grid.check_grid`, after the range checks).

    Returns:
        List of typed SearchDim instances, with plain ``int`` / ``float``
        bounds and step, and a ``tuple`` of choices.

    Raises:
        LizyMLError with CONFIG_INVALID for a space outside the format, with
            ``context`` naming the dimension (``dim``), the ``key``, a
            ``reason`` and the ``surface``.
    """
    checked = check_space_shape(space, surface=surface)
    dims: list[SearchDim] = []
    for name, spec in checked.items():
        dim_type = spec.get("type", "")
        if type(dim_type) is not str or dim_type not in DIM_KEYS:
            raise format_error(
                surface,
                f"Unknown search space type '{dim_type}' for param '{name}'. "
                f"Expected 'float', 'int', or 'categorical'.",
                reason="unknown_type",
                dim=name,
                key="type",
                param=name,
                type=dim_type,
            )
        check_dim_keys(name, spec, dim_type, surface=surface)
        category: DimCategory = spec.get("category", "model")
        if dim_type == "float":
            low_f = read_float_bound(name, "low", spec["low"], surface=surface)
            high_f = read_float_bound(name, "high", spec["high"], surface=surface)
            log_f = read_log(name, spec, surface=surface)
            _validate_numeric_range(
                name, low_f, high_f, log=log_f, surface=surface, float_width=True
            )
            step_f = None
            if "step" in spec:
                step_f = read_float_step(name, spec["step"], surface=surface)
                check_grid(
                    name,
                    low_f,
                    high_f,
                    step_f,
                    log=log_f,
                    kind="float",
                    surface=surface,
                )
            dims.append(
                FloatDim(
                    name=name,
                    low=low_f,
                    high=high_f,
                    log=log_f,
                    category=category,
                    step=step_f,
                )
            )
        elif dim_type == "int":
            low_i = read_int_bound(name, "low", spec["low"], surface=surface)
            high_i = read_int_bound(name, "high", spec["high"], surface=surface)
            log_i = read_log(name, spec, surface=surface)
            _validate_numeric_range(name, low_i, high_i, log=log_i, surface=surface)
            step_i = None
            if "step" in spec:
                step_i = read_int_step(name, spec["step"], surface=surface)
                check_grid(
                    name,
                    low_i,
                    high_i,
                    step_i,
                    log=log_i,
                    kind="int",
                    surface=surface,
                )
            dims.append(
                IntDim(
                    name=name,
                    low=low_i,
                    high=high_i,
                    log=log_i,
                    category=category,
                    step=step_i,
                )
            )
        else:
            choices = read_choices(name, spec, surface=surface)
            _validate_categorical_choices(name, choices, surface=surface)
            dims.append(
                CategoricalDim(name=name, choices=tuple(choices), category=category)
            )
    return dims


def split_by_category(
    trial_params: dict[str, Any],
    dims: list[SearchDim],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split trial params into (model, smart, training) dicts by dim category.

    Args:
        trial_params: Sampled parameters from a trial.
        dims: Search dimensions with category attributes.

    Returns:
        Tuple of (model_params, smart_params, training_params).
    """
    cat_map = {d.name: d.category for d in dims}
    model_p: dict[str, Any] = {}
    smart_p: dict[str, Any] = {}
    training_p: dict[str, Any] = {}
    for name, val in trial_params.items():
        cat = cat_map.get(name, "model")
        if cat == "smart":
            smart_p[name] = val
        elif cat == "training":
            training_p[name] = val
        else:
            model_p[name] = val
    return model_p, smart_p, training_p


def suggest_params(trial: Any, dims: list[SearchDim]) -> dict[str, Any]:
    """Sample a parameter dict from an optuna trial.

    A dimension without ``step`` is sampled as before, without ``step=``. A
    stepped one (H-0124) is sampled by
    :func:`lizyml.tuning._step_grid.suggest_on_grid` and returns its grid
    value, the value the trial trains on.

    Args:
        trial: An ``optuna.Trial`` instance.
        dims: List of search dimensions.

    Returns:
        Dict of the values to train on, one per dimension.
    """
    params: dict[str, Any] = {}
    for dim in dims:
        if isinstance(dim, CategoricalDim):
            params[dim.name] = trial.suggest_categorical(dim.name, dim.choices)
        elif dim.step is not None:
            params[dim.name] = suggest_on_grid(trial, dim)
        elif isinstance(dim, FloatDim):
            params[dim.name] = trial.suggest_float(
                dim.name, dim.low, dim.high, log=dim.log
            )
        else:  # IntDim
            params[dim.name] = trial.suggest_int(
                dim.name, dim.low, dim.high, log=dim.log
            )
    return params


# ---------------------------------------------------------------------------
# Boundary detection and expansion (H-0068)
# ---------------------------------------------------------------------------


_LOG_EXPANSION_FACTOR = 3.0
"""Expansion factor for log-scale dimensions (applied in log space)."""

_Num = TypeVar("_Num", int, float, Fraction)


def _position_pct(best: float, low: float, high: float, *, log: bool) -> float:
    """Compute relative position of *best* within [low, high] as 0.0–1.0."""
    if log and low > 0 and high > 0 and best > 0:
        log_low, log_high, log_best = math.log(low), math.log(high), math.log(best)
        span = log_high - log_low
        if span <= 0:
            return 0.5
        return (log_best - log_low) / span
    span = high - low
    if span <= 0:
        return 0.5
    return (best - low) / span


def _detect_edge(position: float, threshold: float) -> str:
    """Return 'lower', 'upper', or 'none' based on position and threshold."""
    if position < threshold:
        return "lower"
    if position > (1.0 - threshold):
        return "upper"
    return "none"


def _expand_range(
    low: float,
    high: float,
    edge: str,
    *,
    log: bool,
    min_allowed: float | None = None,
    max_allowed: float | None = None,
) -> tuple[float, float, bool]:
    """Expand *low*/*high* asymmetrically toward *edge*.

    Linear lower-edge expansion is clamped at 0.0 (#110): negative bounds are
    physically meaningless for hyperparameters such as ``learning_rate`` and
    crash downstream samplers. Log-scale expansion is already safe because
    division by ``_LOG_EXPANSION_FACTOR`` cannot make a positive value cross
    zero. ``IntDim`` retains its own ``max(1, ...)`` clamp at the call site.

    When ``min_allowed`` / ``max_allowed`` are provided (H-0078), the
    expansion is additionally clamped to those parameter-meaningful
    bounds. The third return value is ``True`` iff such a clamp actually
    bit (i.e. the natural expansion would have crossed the bound).

    Expansion never narrows (#342): a user range may lie outside the bounds
    (H-0111), and neither a bound nor the ``0.0`` floor moves an edge inward
    past the declared one. Such an edge stays where it is, and the caller
    reports the unchanged range as the H-0111 no-op.
    """
    new_low = low
    new_high = high
    if edge == "lower":
        if log and low > 0:
            new_low = low / _LOG_EXPANSION_FACTOR
        else:
            new_low = min(low, max(0.0, low - (high - low)))
    elif edge == "upper":
        new_high = (
            high * _LOG_EXPANSION_FACTOR if log and high > 0 else high + (high - low)
        )
    return _clamp_outward(low, high, new_low, new_high, min_allowed, max_allowed)


def _clamp_outward(
    low: _Num,
    high: _Num,
    new_low: _Num,
    new_high: _Num,
    min_allowed: _Num | None,
    max_allowed: _Num | None,
) -> tuple[_Num, _Num, bool]:
    """Clamp an expanded edge to its bound, never inward past *low* / *high*.

    Only an edge the expansion moved outward is clamped, and only as far as
    the declared edge: a user ``low`` already below ``min_allowed`` stays
    (#342). ``clamped`` is ``True`` iff a bound stopped an outward move.
    Exact for ints and fractions: ``min`` / ``max`` return one of their
    arguments.
    """
    clamped = False
    if min_allowed is not None and new_low < low and new_low < min_allowed:
        new_low = min(low, min_allowed)
        clamped = True
    if max_allowed is not None and new_high > high and new_high > max_allowed:
        new_high = max(high, max_allowed)
        clamped = True
    return new_low, new_high, clamped


def _expand_int_range(
    low: int,
    high: int,
    edge: str,
    *,
    log: bool,
    min_allowed: int | None = None,
    max_allowed: int | None = None,
) -> tuple[int, int, bool]:
    """Integer counterpart of :func:`_expand_range` for ``IntDim`` (H-0111).

    Same rules, in integer arithmetic so that the result -- and the H-0111
    "range unchanged" comparison -- stays exact beyond ``2**53``, where a
    float round trip merges neighbouring integers. The low edge is floored
    and guarded by ``max(1, ...)``, the high edge ceiled, as before. Neither
    the guard nor a bound moves an edge inward past the declared one (#342):
    ``max_depth`` keeps ``-1`` and ``bagging_freq`` keeps ``0``.
    """
    factor = int(_LOG_EXPANSION_FACTOR)
    new_low: int = low
    new_high: int = high
    if edge == "lower":
        new_low = low // factor if log and low > 0 else max(0, low - (high - low))
        new_low = min(low, max(1, new_low))
    elif edge == "upper":
        new_high = high * factor if log and high > 0 else high + (high - low)
    return _clamp_outward(low, high, new_low, new_high, min_allowed, max_allowed)


def _expand_on_grid(
    dim: FloatDim | IntDim, edge: str
) -> tuple[Fraction, Fraction, bool]:
    """:func:`_expand_range` / :func:`_expand_int_range` for a stepped dim (H-0124).

    The same rule in exact arithmetic (``0.2 + (0.2 - 0.1)`` is ``0.3``, not
    ``0.30000000000000004``); the moved edge is then snapped outward to the
    grid, kept above its floor (a log lower edge at ``step``; an ``int`` one at
    the grid point the ``max(1, ...)`` guard allows), clamped to a provider
    bound (:func:`_clamp_outward`) and, if clamped, snapped inward -- never
    inside the declared edge (#342). The edges are returned exact: the moved
    one can be past the float range, and :func:`expansion_edge` decides
    whether it is taken (amendment 1, policy 10).
    """
    is_int = isinstance(dim, IntDim)
    assert dim.step is not None
    step = exact(dim.step)
    low, high = exact(dim.low), exact(dim.high)
    new_low, new_high = low, high
    factor = int(_LOG_EXPANSION_FACTOR)
    if edge == "lower":
        if dim.log and low > 0:
            new_low = Fraction(dim.low // factor) if is_int else low / factor
        else:
            new_low = min(low, max(Fraction(0), low - (high - low)))
        if is_int:
            new_low = min(low, max(Fraction(1), new_low))
        new_low = snap_down(new_low, step)
        if dim.log and new_low < step:
            new_low = step
        if is_int and new_low < 1:
            new_low = snap_up(Fraction(1), step)
        new_low = min(low, new_low)
    elif edge == "upper":
        new_high = high * factor if dim.log and high > 0 else high + (high - low)
        new_high = snap_up(new_high, step)
    min_allowed = None if dim.min_allowed is None else exact(dim.min_allowed)
    max_allowed = None if dim.max_allowed is None else exact(dim.max_allowed)
    new_low, new_high, clamped = _clamp_outward(
        low, high, new_low, new_high, min_allowed, max_allowed
    )
    if clamped and edge == "lower":
        new_low = min(low, snap_up(new_low, step))
    elif clamped and edge == "upper":
        new_high = max(high, snap_down(new_high, step))
    return new_low, new_high, clamped


def detect_boundary(
    dims: list[SearchDim],
    best_params: dict[str, Any],
    threshold: float = 0.05,
) -> BoundaryReport:
    """Detect which dimensions have best params near their boundary (H-0068).

    Args:
        dims: Current search space dimensions.
        best_params: Best parameter values from the previous tuning round.
        threshold: Edge detection threshold (0.0–1.0). A best value within
            this fraction of the range from either edge is considered near
            the boundary.

    Returns:
        BoundaryReport with per-dimension analysis.
    """
    statuses: list[BoundaryDimStatus] = []
    expanded_names: list[str] = []

    for dim in dims:
        best_val = best_params.get(dim.name)

        if isinstance(dim, CategoricalDim):
            statuses.append(
                BoundaryDimStatus(
                    name=dim.name,
                    best_value=best_val,
                    low=None,
                    high=None,
                    position_pct=None,
                    edge="none",
                    expanded=False,
                    new_low=None,
                    new_high=None,
                )
            )
            continue

        # FloatDim or IntDim
        low: float = float(dim.low)
        high: float = float(dim.high)
        is_log = dim.log
        best_num = float(best_val) if best_val is not None else (low + high) / 2

        position = _position_pct(best_num, low, high, log=is_log)
        edge = _detect_edge(position, threshold)
        should_expand = edge != "none"

        new_low: float | int | None = None
        new_high: float | int | None = None
        clamped = False
        if should_expand:
            nl: float | int | Fraction
            nh: float | int | Fraction
            if dim.step is not None:
                nl, nh, clamped = _expand_on_grid(dim, edge)
            elif isinstance(dim, IntDim):
                nl, nh, clamped = _expand_int_range(
                    dim.low,
                    dim.high,
                    edge,
                    log=is_log,
                    min_allowed=dim.min_allowed,
                    max_allowed=dim.max_allowed,
                )
            else:
                nl, nh, clamped = _expand_range(
                    low,
                    high,
                    edge,
                    log=is_log,
                    min_allowed=(
                        float(dim.min_allowed) if dim.min_allowed is not None else None
                    ),
                    max_allowed=(
                        float(dim.max_allowed) if dim.max_allowed is not None else None
                    ),
                )
            # Amendment 1, policy 10: the moved edge, as the dim's type, is
            # taken only inside the limits Optuna samples and on the grid as
            # read back; otherwise the edge stays and no bound was reached.
            moved = expansion_edge(dim, edge, nl if edge == "lower" else nh)
            old = dim.low if edge == "lower" else dim.high
            if moved is None:
                should_expand = False
                clamped = False
            elif moved == old:
                # H-0078 item 4 (H-0111): an expansion that leaves the range
                # unchanged -- the edge already sits on a clamp (min/max_allowed,
                # the linear 0.0 floor, the IntDim max(1, ...) guard) -- is not
                # an expansion; ``clamped_to_bound`` keeps what the clamp said.
                should_expand = False
            else:
                plain: type[int] | type[float] = (
                    int if isinstance(dim, IntDim) else float
                )
                new_low = moved if edge == "lower" else plain(dim.low)
                new_high = moved if edge == "upper" else plain(dim.high)
                expanded_names.append(dim.name)

        statuses.append(
            BoundaryDimStatus(
                name=dim.name,
                best_value=best_val,
                low=dim.low,
                high=dim.high,
                position_pct=round(position, 4),
                edge=edge,
                expanded=should_expand,
                new_low=new_low,
                new_high=new_high,
                clamped_to_bound=clamped,
            )
        )

    return BoundaryReport(
        dims=tuple(statuses),
        expanded_names=tuple(expanded_names),
    )


def attach_bounds(
    dims: list[SearchDim],
    bounds: dict[str, dict[str, float | int]],
) -> list[SearchDim]:
    """Return a new list of dims with ``min_allowed`` / ``max_allowed`` injected.

    For each ``FloatDim`` / ``IntDim`` whose ``name`` is a key in *bounds*,
    a copy is returned with ``min_allowed`` / ``max_allowed`` populated
    from ``{"min": ..., "max": ...}``, keeping ``step``. Either key may be
    absent, and that side stays unbounded (H-0140). Dims that already
    carry user-set bounds are left untouched (user override wins). ``CategoricalDim`` is
    always returned unchanged. Dims with no entry in *bounds* are
    unchanged.

    Used by ``Model.tune`` to wire ``EstimatorProvider.parameter_bounds(task)``
    onto the search space (H-0078).
    """
    if not bounds:
        return list(dims)

    new_dims: list[SearchDim] = []
    for dim in dims:
        if isinstance(dim, CategoricalDim):
            new_dims.append(dim)
            continue
        b = bounds.get(dim.name)
        if b is None:
            new_dims.append(dim)
            continue
        # Respect user-set bounds — do not overwrite.
        if dim.min_allowed is not None or dim.max_allowed is not None:
            new_dims.append(dim)
            continue
        # A one-sided bound (H-0140) leaves the other side unbounded.
        low_b, high_b = b.get("min"), b.get("max")
        if isinstance(dim, FloatDim):
            new_dims.append(
                FloatDim(
                    name=dim.name,
                    low=dim.low,
                    high=dim.high,
                    log=dim.log,
                    category=dim.category,
                    min_allowed=None if low_b is None else float(low_b),
                    max_allowed=None if high_b is None else float(high_b),
                    step=dim.step,
                )
            )
        elif isinstance(dim, IntDim):
            new_dims.append(
                IntDim(
                    name=dim.name,
                    low=dim.low,
                    high=dim.high,
                    log=dim.log,
                    category=dim.category,
                    min_allowed=None if low_b is None else int(low_b),
                    max_allowed=None if high_b is None else int(high_b),
                    step=dim.step,
                )
            )
        else:  # pragma: no cover — exhaustive over SearchDim
            new_dims.append(dim)
    return new_dims


def expand_dims(
    dims: list[SearchDim],
    report: BoundaryReport,
) -> list[SearchDim]:
    """Return a new list of SearchDim with boundary-expanded ranges (H-0068).

    Only dimensions flagged ``expanded=True`` in *report* are modified, and
    keep their ``step``: :func:`detect_boundary` puts a stepped dimension's new
    edges on its grid (H-0124).
    Non-expanded dimensions are returned unchanged. An expanded range only
    ever widens (#342): each edge is the outer of the old and the reported
    one, so even a hand-made report cannot narrow or invert a range.

    Args:
        dims: Original search space dimensions.
        report: Boundary report from :func:`detect_boundary`.

    Returns:
        New list of SearchDim with expanded ranges where applicable.
    """
    expansion_map: dict[str, BoundaryDimStatus] = {
        s.name: s for s in report.dims if s.expanded
    }
    new_dims: list[SearchDim] = []
    for dim in dims:
        status = expansion_map.get(dim.name)
        if status is None:
            new_dims.append(dim)
            continue

        if (
            isinstance(dim, FloatDim)
            and status.new_low is not None
            and status.new_high is not None
        ):
            new_dims.append(
                FloatDim(
                    name=dim.name,
                    low=min(dim.low, float(status.new_low)),
                    high=max(dim.high, float(status.new_high)),
                    log=dim.log,
                    category=dim.category,
                    min_allowed=dim.min_allowed,
                    max_allowed=dim.max_allowed,
                    step=dim.step,
                )
            )
        elif (
            isinstance(dim, IntDim)
            and status.new_low is not None
            and status.new_high is not None
        ):
            new_dims.append(
                IntDim(
                    name=dim.name,
                    low=min(dim.low, int(status.new_low)),
                    high=max(dim.high, int(status.new_high)),
                    log=dim.log,
                    category=dim.category,
                    min_allowed=dim.min_allowed,
                    max_allowed=dim.max_allowed,
                    step=dim.step,
                )
            )
        else:
            new_dims.append(dim)

    return new_dims
