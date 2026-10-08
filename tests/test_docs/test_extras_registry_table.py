"""The method-to-extra registry in ``lizyml/_extras.py`` (H-0119 section 4).

These tests pin the table as H-0119 states it and check it against the code it
describes: every method it names is a public ``Model`` method, and every
default it records is the default of that method's signature. Whether each row
is really needed is checked by running it without its extra (H-0119 6.(a),
``tests/test_notebooks/probe_extras_registry.py``).
"""

from __future__ import annotations

import inspect

import pytest

from lizyml import Model
from lizyml._extras import (
    ARGUMENT_DEFAULTS,
    CONDITION_ARGUMENTS,
    EXTRA_PACKAGES,
    RULES,
    extras_for,
)

PLOT_METHODS = (
    "residuals_plot",
    "roc_curve_plot",
    "calibration_plot",
    "probability_histogram_plot",
    "plot_learning_curve",
    "plot_oof_distribution",
    "tuning_plot",
)


def test_extras_and_their_packages() -> None:
    assert dict(EXTRA_PACKAGES) == {
        "explain": "shap",
        "plots": "plotly",
        "tuning": "optuna",
    }
    assert {rule.extra for rule in RULES} == set(EXTRA_PACKAGES)


def test_the_table_is_the_one_h0119_states() -> None:
    rows = {(r.method, r.condition, r.extra) for r in RULES}
    expected = {
        ("tune", None, "tuning"),
        ("predict", ("return_shap", True), "explain"),
        ("importance", ("kind", "shap"), "explain"),
        ("importance_plot", None, "plots"),
        ("importance_plot", ("kind", "shap"), "explain"),
    } | {(m, None, "plots") for m in PLOT_METHODS}
    assert rows == expected
    assert len(RULES) == len(rows), "a row is listed twice"


def test_every_method_in_the_table_is_a_public_model_method() -> None:
    public = {
        name
        for name, value in inspect.getmembers(Model)
        if not name.startswith("_") and callable(value)
    }
    assert {r.method for r in RULES} <= public
    assert {m for m, _ in ARGUMENT_DEFAULTS} <= public


@pytest.mark.parametrize("key", sorted(ARGUMENT_DEFAULTS))
def test_recorded_defaults_match_the_signatures(key: tuple[str, str]) -> None:
    method, argument = key
    parameter = inspect.signature(getattr(Model, method)).parameters[argument]
    assert parameter.default == ARGUMENT_DEFAULTS[key]
    assert type(parameter.default) is type(ARGUMENT_DEFAULTS[key])


def test_defaults_h0119_lists() -> None:
    assert dict(ARGUMENT_DEFAULTS) == {
        ("predict", "return_shap"): False,
        ("importance", "kind"): "split",
        ("importance_plot", "kind"): "split",
        ("residuals_plot", "kind"): "all",
    }


def test_every_condition_argument_is_described() -> None:
    conditioned = {(r.method, r.condition[0]) for r in RULES if r.condition}
    assert conditioned == set(CONDITION_ARGUMENTS)
    for key, spec in CONDITION_ARGUMENTS.items():
        assert spec.default == ARGUMENT_DEFAULTS[key]
        assert any(
            type(v) is type(spec.default) and v == spec.default for v in spec.domain
        )


@pytest.mark.parametrize(
    ("key", "position"),
    [
        (("predict", "return_shap"), None),
        (("importance", "kind"), 0),
        (("importance_plot", "kind"), 0),
    ],
)
def test_positional_kind_only_where_h0119_allows_it(
    key: tuple[str, str], position: int | None
) -> None:
    spec = CONDITION_ARGUMENTS[key]
    assert spec.position == position
    if position is not None:
        method, argument = key
        names = list(inspect.signature(getattr(Model, method)).parameters)
        # ``self`` is parameter 0 of the unbound function.
        assert names[position + 1] == argument


@pytest.mark.parametrize(
    ("method", "arguments", "expected"),
    [
        ("fit", {}, set()),
        ("tune", {}, {"tuning"}),
        ("predict", {}, set()),
        ("predict", {"return_shap": False}, set()),
        ("predict", {"return_shap": True}, {"explain"}),
        ("importance", {}, set()),
        ("importance", {"kind": "gain"}, set()),
        ("importance", {"kind": "shap"}, {"explain"}),
        ("importance_plot", {}, {"plots"}),
        ("importance_plot", {"kind": "split"}, {"plots"}),
        ("importance_plot", {"kind": "shap"}, {"plots", "explain"}),
        ("residuals_plot", {}, {"plots"}),
        ("residuals_plot", {"kind": "qq"}, {"plots"}),
        ("tuning_plot", {}, {"plots"}),
    ],
)
def test_extras_for(
    method: str, arguments: dict[str, object], expected: set[str]
) -> None:
    assert extras_for(method, arguments) == frozenset(expected)


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("predict", {"return_shap": 1}),
        ("predict", {"return_shap": "yes"}),
        ("importance", {"kind": "SHAP"}),
        ("importance_plot", {"kind": None}),
    ],
)
def test_a_condition_value_outside_its_domain_fails(
    method: str, arguments: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="cannot decide"):
        extras_for(method, arguments)


def test_every_row_has_a_probe_for_ci() -> None:
    """The 6.(a) probe exercises every row (it runs only in the CI matrix)."""
    from tests.test_notebooks import probe_extras_registry as probe

    assert probe.uncovered_rules() == []
    assert {f.method for f in probe.FORMS} == {r.method for r in RULES}
    for extra in EXTRA_PACKAGES:
        assert any(extra in extras_for(f.method, f.conditions) for f in probe.FORMS)
