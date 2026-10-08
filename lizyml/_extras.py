"""Which optional extra each public ``Model`` method needs (H-0119, private).

The table states package behaviour, so it lives next to the code it describes.
``scripts/examples_index.py`` derives each notebook's extras from it, and CI
checks every row by calling it in an environment without that extra (H-0119
6.(a)).

scipy is not listed: scikit-learn and lightgbm require it unconditionally, so
the base install always has it (Beta calibration and the qq / all residual
plots therefore need no extra).

This module imports nothing outside the standard library, so the CI scripts can
load it by path before the package's dependencies are installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: Extra name -> import name of the package it installs.
EXTRA_PACKAGES: Mapping[str, str] = MappingProxyType(
    {"explain": "shap", "plots": "plotly", "tuning": "optuna"}
)


@dataclass(frozen=True)
class ExtraRule:
    """``method`` needs ``extra`` always, or when ``condition`` holds.

    ``condition`` is ``(argument, value)``: the rule applies when the call
    passes ``value`` for ``argument`` (or omits it and the default is
    ``value``).
    """

    method: str
    extra: str
    condition: tuple[str, object] | None = None


@dataclass(frozen=True)
class ConditionArgument:
    """An argument a rule is conditioned on.

    ``domain`` lists every value the method accepts, so that any other value
    fails instead of being read as "condition not met". ``position`` is the
    positional index (after ``self``) at which the value may be passed, or
    ``None`` when it must be passed by keyword.
    """

    default: object
    domain: tuple[object, ...]
    position: int | None


_PLOTS_ALWAYS = (
    "residuals_plot",
    "roc_curve_plot",
    "calibration_plot",
    "probability_histogram_plot",
    "plot_learning_curve",
    "plot_oof_distribution",
    "tuning_plot",
)

RULES: tuple[ExtraRule, ...] = (
    ExtraRule("tune", "tuning"),
    ExtraRule("predict", "explain", ("return_shap", True)),
    ExtraRule("importance", "explain", ("kind", "shap")),
    ExtraRule("importance_plot", "plots"),
    ExtraRule("importance_plot", "explain", ("kind", "shap")),
    *(ExtraRule(method, "plots") for method in _PLOTS_ALWAYS),
)

_KINDS = ("split", "gain", "shap")

CONDITION_ARGUMENTS: Mapping[tuple[str, str], ConditionArgument] = MappingProxyType(
    {
        ("predict", "return_shap"): ConditionArgument(False, (False, True), None),
        ("importance", "kind"): ConditionArgument("split", _KINDS, 0),
        ("importance_plot", "kind"): ConditionArgument("split", _KINDS, 0),
    }
)

#: Defaults H-0119 records for the arguments the table is read against. The
#: ``residuals_plot`` entry is recorded because its default (``"all"``) draws
#: the qq plot, which uses scipy from the base install.
ARGUMENT_DEFAULTS: Mapping[tuple[str, str], object] = MappingProxyType(
    {
        **{key: spec.default for key, spec in CONDITION_ARGUMENTS.items()},
        ("residuals_plot", "kind"): "all",
    }
)


def _same(a: object, b: object) -> bool:
    """Equal and of the same type, so ``1`` is not ``True``."""
    return type(a) is type(b) and a == b


def condition_value(
    method: str, argument: str, arguments: Mapping[str, object]
) -> object:
    """The value ``argument`` takes in a call passing ``arguments``.

    Raises:
        ValueError: The value is not one the method accepts, so whether the
            rule applies cannot be decided.
    """
    spec = CONDITION_ARGUMENTS[(method, argument)]
    value = arguments.get(argument, spec.default)
    if not any(_same(value, allowed) for allowed in spec.domain):
        raise ValueError(
            f"{method}({argument}={value!r}): cannot decide the extras it needs; "
            f"pass one of {list(spec.domain)!r}"
        )
    return value


def extras_for(method: str, arguments: Mapping[str, object]) -> frozenset[str]:
    """Extras a call of ``method`` with ``arguments`` needs.

    Args:
        method: Public ``Model`` method name.
        arguments: The condition arguments the call passes, by name. An
            omitted one takes its default.

    Raises:
        ValueError: A condition argument has a value outside its domain.
    """
    needed: set[str] = set()
    for rule in RULES:
        if rule.method != method:
            continue
        if rule.condition is None:
            needed.add(rule.extra)
            continue
        argument, value = rule.condition
        if _same(condition_value(method, argument, arguments), value):
            needed.add(rule.extra)
    return frozenset(needed)
