"""Shared optimiser contract for calibrators fitted with ``scipy.optimize.minimize``.

``platt`` and ``beta`` both fit a few coefficients by maximum likelihood, and both
accept the same optimiser settings in ``calibration.params`` (H-0100, #277):
``x0``, ``method``, ``bounds``, ``tol`` and ``options``. This module is where those
settings are validated and resolved, so the two calibrators cannot drift apart.

Three things are decided here rather than left to scipy:

* **Which methods are accepted, and which of them can honour ``bounds``.** scipy
  ignores ``bounds`` for ``BFGS`` and ``CG`` with only a ``RuntimeWarning``, so an
  approved setting would silently do nothing. The table is closed and limited to
  methods scipy 1.10 provides, so the accepted set does not move with the
  installed scipy.
* **How ``tol`` and ``options`` combine.** scipy applies ``tol`` through
  ``options.setdefault``, so a default key the calibrator put in ``options`` would
  defeat a ``tol`` the user wrote. The order is: the user's ``options`` keys, then
  the user's ``tol``, then the calibrator's per-method defaults.
* **Whether an ``options`` key exists.** That is asked of the real scipy, by
  running ``minimize`` once on a tiny problem and turning its
  ``Unknown solver options`` warning into a refusal. Copying the option names
  into a table would be a declaration nothing executes.

``lizyml/calibration/`` may not import ``lizyml/estimators/``; nothing here does.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lizyml.core.exceptions import ErrorCode, LizyMLError

SURFACE = "calibration.params"

#: The optimiser settings every scipy-fitted calibrator accepts.
OPTIMIZER_NAMES: frozenset[str] = frozenset(
    {"x0", "method", "bounds", "tol", "options"}
)

DEFAULT_METHOD = "L-BFGS-B"


@dataclass(frozen=True)
class MethodSpec:
    """What a ``minimize`` method can do with the settings a user may write."""

    bounds: bool
    gradient: bool


#: The accepted methods. Methods that require a Hessian are deliberately absent.
METHODS: Mapping[str, MethodSpec] = {
    "L-BFGS-B": MethodSpec(bounds=True, gradient=True),
    "TNC": MethodSpec(bounds=True, gradient=True),
    "SLSQP": MethodSpec(bounds=True, gradient=True),
    "trust-constr": MethodSpec(bounds=True, gradient=True),
    "Powell": MethodSpec(bounds=True, gradient=False),
    "Nelder-Mead": MethodSpec(bounds=True, gradient=False),
    "BFGS": MethodSpec(bounds=False, gradient=True),
    "CG": MethodSpec(bounds=False, gradient=True),
}

#: The ``options`` keys scipy sets from ``tol`` for each method
#: (``scipy/optimize/_minimize.py``, "set default tolerances").
_TOL_KEYS: Mapping[str, frozenset[str]] = {
    "L-BFGS-B": frozenset({"ftol", "gtol"}),
    "TNC": frozenset({"xtol", "ftol", "gtol"}),
    "SLSQP": frozenset({"ftol"}),
    "trust-constr": frozenset({"xtol", "gtol", "barrier_tol"}),
    "Powell": frozenset({"xtol", "ftol"}),
    "Nelder-Mead": frozenset({"xatol", "fatol"}),
    "BFGS": frozenset({"gtol"}),
    "CG": frozenset({"gtol"}),
}


def refuse(calibrator: str, parameter: str, detail: str) -> LizyMLError:
    """Build the refusal for one setting, naming the surface and the calibrator."""
    return LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=f"{SURFACE} for calibrator '{calibrator}': '{parameter}' {detail}",
        context={"surface": SURFACE, "calibrator": calibrator, "parameter": parameter},
    )


def _is_number(value: Any) -> bool:
    """A real number that scipy can hold as a float.

    An integer too large for a float is refused here: ``math.isfinite`` would
    raise ``OverflowError`` on it instead of letting the setting be refused.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        float(value)
    except OverflowError:
        return False
    return True


def validate_optimizer_params(
    params: Mapping[str, Any],
    *,
    calibrator: str,
    n_coef: int,
    extra: Mapping[str, Callable[[Any], str | None]] | None = None,
) -> None:
    """Refuse any setting the calibrator cannot honour, before anything is fitted.

    Args:
        params: The calibrator's ``calibration.params``.
        calibrator: The calibrator's registered name, for the message.
        n_coef: How many coefficients ``x0`` and ``bounds`` describe.
        extra: Calibrator-specific names mapped to a check returning an error
            detail, or ``None`` when the value is acceptable.

    Raises:
        LizyMLError: ``CONFIG_INVALID`` naming ``calibration.params``, the
            calibrator and the offending setting.
    """
    extra = extra or {}
    accepted = OPTIMIZER_NAMES | set(extra)
    for name in params:
        if name not in accepted:
            raise refuse(
                calibrator,
                name,
                f"is not an accepted setting. Accepted: {sorted(accepted)}.",
            )

    method = params.get("method", DEFAULT_METHOD)
    # The type is checked first: a list or dict survives value normalisation and
    # is unhashable, so the membership test alone would escape as TypeError.
    if not isinstance(method, str) or method not in METHODS:
        raise refuse(
            calibrator, "method", f"must be one of {sorted(METHODS)}; got {method!r}."
        )
    spec = METHODS[method]

    if "x0" in params:
        _check_x0(params["x0"], calibrator=calibrator, n_coef=n_coef)
    if "bounds" in params:
        if not spec.bounds:
            raise refuse(
                calibrator,
                "bounds",
                f"cannot be honoured by method {method!r}; scipy ignores bounds "
                f"for it. Methods that honour bounds: "
                f"{sorted(m for m, s in METHODS.items() if s.bounds)}.",
            )
        _check_bounds(params["bounds"], calibrator=calibrator, n_coef=n_coef)
    if "tol" in params:
        tol = params["tol"]
        if not _is_number(tol) or not math.isfinite(tol) or tol <= 0:
            raise refuse(
                calibrator, "tol", f"must be a positive finite number; got {tol!r}."
            )
    if "options" in params:
        _check_options(
            params["options"], calibrator=calibrator, method=method, n_coef=n_coef
        )

    for name, check in extra.items():
        if name in params:
            detail = check(params[name])
            if detail is not None:
                raise refuse(calibrator, name, detail)


def _check_x0(value: Any, *, calibrator: str, n_coef: int) -> None:
    if not isinstance(value, list | tuple) or len(value) != n_coef:
        raise refuse(
            calibrator, "x0", f"must be a list of {n_coef} numbers; got {value!r}."
        )
    if not all(_is_number(v) and math.isfinite(v) for v in value):
        raise refuse(
            calibrator, "x0", f"must contain finite numbers only; got {value!r}."
        )


def _check_bounds(value: Any, *, calibrator: str, n_coef: int) -> None:
    shape = (
        f"must be a list of {n_coef} [lower, upper] lists, each bound a number "
        f"or null; got {value!r}."
    )
    if not isinstance(value, list | tuple) or len(value) != n_coef:
        raise refuse(calibrator, "bounds", shape)
    for pair in value:
        # A pair is a list: the accepted value set (H-0095) admits a list as a
        # member of a sequence, and a tuple there is refused at the entrance.
        if not isinstance(pair, list) or len(pair) != 2:
            raise refuse(calibrator, "bounds", shape)
        low, high = pair
        for bound in (low, high):
            if bound is not None and not (_is_number(bound) and not math.isnan(bound)):
                raise refuse(calibrator, "bounds", shape)
        if low is not None and high is not None and low > high:
            raise refuse(
                calibrator,
                "bounds",
                f"has a lower bound above its upper bound: {pair!r}.",
            )


def _check_options(value: Any, *, calibrator: str, method: str, n_coef: int) -> None:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise refuse(
            calibrator, "options", f"must be a mapping with string keys; got {value!r}."
        )
    if not value:
        return

    from scipy.optimize import OptimizeWarning, minimize

    spec = METHODS[method]

    def fun(x: Any) -> Any:
        loss = float(sum((xi - 1.0) ** 2 for xi in x))
        if spec.gradient:
            return loss, [2.0 * (xi - 1.0) for xi in x]
        return loss

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            minimize(
                fun,
                [0.0] * n_coef,
                jac=True if spec.gradient else None,
                method=method,
                options=dict(value),
            )
        except (TypeError, ValueError) as err:
            raise refuse(
                calibrator,
                "options",
                f"is not accepted by scipy method {method!r}: {err}",
            ) from err
    unknown = [w for w in caught if issubclass(w.category, OptimizeWarning)]
    if unknown:
        raise refuse(
            calibrator,
            "options",
            f"has keys scipy method {method!r} does not know: {unknown[0].message}",
        )


def resolve_minimize_kwargs(
    params: Mapping[str, Any],
    *,
    n_coef: int,
    default_x0: Sequence[float],
    default_options: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve validated settings into keyword arguments for ``minimize``.

    Args:
        params: Validated ``calibration.params``.
        n_coef: Number of coefficients.
        default_x0: The calibrator's starting point, used when ``x0`` is absent.
        default_options: The calibrator's default ``options`` per method.

    Returns:
        ``method``, ``x0`` and ``options``, plus ``bounds`` and ``tol`` when written.
    """
    method = params.get("method", DEFAULT_METHOD)
    options = dict(default_options.get(method, {}))
    kwargs: dict[str, Any] = {"method": method}
    if "tol" in params:
        # scipy fills these from tol with setdefault; a default left here would win.
        for key in _TOL_KEYS[method]:
            options.pop(key, None)
        kwargs["tol"] = params["tol"]
    options.update(params.get("options", {}))
    kwargs["options"] = options

    x0 = params.get("x0", default_x0)
    kwargs["x0"] = [float(v) for v in x0]
    if len(kwargs["x0"]) != n_coef:  # validated; kept as a guard for internal callers
        raise ValueError(f"x0 must have {n_coef} values")
    if "bounds" in params:
        kwargs["bounds"] = [tuple(pair) for pair in params["bounds"]]
    return kwargs
