"""Which optional extra each public ``Model`` method needs (H-0119, private).

The table states package behaviour, so it lives next to the code it describes.
``scripts/examples_index.py`` derives each notebook's extras from it, and CI
checks every row by calling it in an environment without that extra (H-0119
6.(a)).

A conditional row carries the predicate the method itself evaluates on the
argument, with a pointer to that check; it is not a list of accepted values.
An omitted argument takes the signature default.

scipy is not listed: scikit-learn and lightgbm require it unconditionally, so
the base install always has it (Beta calibration and the qq / all residual
plots therefore need no extra).

This module imports nothing outside the standard library, so the CI scripts can
load it by path before the package's dependencies are installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: Extra name -> import name of the package it installs.
EXTRA_PACKAGES: Mapping[str, str] = MappingProxyType(
    {"explain": "shap", "plots": "plotly", "tuning": "optuna"}
)


@dataclass(frozen=True)
class ExtraRule:
    """``method`` needs ``extra`` always, or when ``condition`` says so.

    ``condition`` names an argument of ``method``; the row applies when that
    argument's predicate in :data:`CONDITION_ARGUMENTS` holds for the value the
    call passes (or for the default when the call omits it).
    """

    method: str
    extra: str
    condition: str | None = None


@dataclass(frozen=True)
class ConditionArgument:
    """An argument a rule is conditioned on.

    ``needs_extra`` mirrors the runtime check at ``mirrors`` (``path:line``)
    exactly. ``position`` is the positional index (after ``self``) at which the
    value may be passed, or ``None`` when it must be passed by keyword.
    """

    default: object
    position: int | None
    needs_extra: Callable[[object], bool]
    mirrors: str


def _truthy(value: object) -> bool:
    return bool(value)


def _is_shap(value: object) -> bool:
    return value == "shap"


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
    ExtraRule("predict", "explain", "return_shap"),
    ExtraRule("importance", "explain", "kind"),
    ExtraRule("importance_plot", "plots"),
    ExtraRule("importance_plot", "explain", "kind"),
    *(ExtraRule(method, "plots") for method in _PLOTS_ALWAYS),
)

CONDITION_ARGUMENTS: Mapping[tuple[str, str], ConditionArgument] = MappingProxyType(
    {
        # `if return_shap:`
        ("predict", "return_shap"): ConditionArgument(
            False, None, _truthy, "lizyml/core/_model_predict.py:93"
        ),
        # `if kind == "shap":`
        ("importance", "kind"): ConditionArgument(
            "split", 0, _is_shap, "lizyml/core/_model_tables.py:153"
        ),
        # `if kind == "shap":` (then `self.importance(kind="shap")`)
        ("importance_plot", "kind"): ConditionArgument(
            "split", 0, _is_shap, "lizyml/core/_model_plots.py:184"
        ),
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


def rule_applies(rule: ExtraRule, arguments: Mapping[str, object]) -> bool:
    """Whether a call passing ``arguments`` (by name) needs ``rule.extra``."""
    if rule.condition is None:
        return True
    spec = CONDITION_ARGUMENTS[(rule.method, rule.condition)]
    return spec.needs_extra(arguments.get(rule.condition, spec.default))


def extras_for(method: str, arguments: Mapping[str, object]) -> frozenset[str]:
    """Extras a call of ``method`` with ``arguments`` needs.

    Args:
        method: Public ``Model`` method name.
        arguments: The condition arguments the call passes, by name. An
            omitted one takes its default.
    """
    return frozenset(
        rule.extra
        for rule in RULES
        if rule.method == method and rule_applies(rule, arguments)
    )
