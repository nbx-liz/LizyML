"""The method-to-extra registry in ``lizyml/_extras.py`` (H-0119 section 4).

These tests pin the table as H-0119 states it and check it against the code it
describes: every method it names is a public instance method of ``Model``,
every default it records is the default of that method's signature, and each
condition is the predicate the method itself evaluates. That last point is
checked by behaviour: with shap made unavailable, a call raises
``OPTIONAL_DEP_MISSING`` exactly when the registry's predicate says the call
needs ``explain``. Whether each row is needed at all is checked by running it
without its extra (H-0119 6.(a), ``tests/test_notebooks/probe_extras_registry.py``).
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

import lizyml.explain.shap_explainer as shap_explainer
from lizyml import Model
from lizyml._extras import (
    ARGUMENT_DEFAULTS,
    CONDITION_ARGUMENTS,
    EXTRA_PACKAGES,
    RULES,
    extras_for,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config, make_regression_df

ROOT = pathlib.Path(__file__).resolve().parents[2]

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
        ("predict", "return_shap", "explain"),
        ("importance", "kind", "explain"),
        ("importance_plot", None, "plots"),
        ("importance_plot", "kind", "explain"),
    } | {(m, None, "plots") for m in PLOT_METHODS}
    assert rows == expected
    assert len(RULES) == len(rows), "a row is listed twice"


def test_every_method_in_the_table_is_a_public_instance_method() -> None:
    for method in {r.method for r in RULES} | {m for m, _ in ARGUMENT_DEFAULTS}:
        assert not method.startswith("_")
        assert inspect.isfunction(inspect.getattr_static(Model, method)), method


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
    conditioned = {(r.method, r.condition) for r in RULES if r.condition}
    assert conditioned == set(CONDITION_ARGUMENTS)
    for key, spec in CONDITION_ARGUMENTS.items():
        assert spec.default == ARGUMENT_DEFAULTS[key]
        assert not spec.needs_extra(spec.default), "the default needs no extra"


@pytest.mark.parametrize(
    ("key", "pointer", "line"),
    [
        (
            ("predict", "return_shap"),
            "lizyml/core/_model_predict.py:93",
            "if return_shap:",
        ),
        (
            ("importance", "kind"),
            "lizyml/core/_model_tables.py:153",
            'if kind == "shap":',
        ),
        (
            ("importance_plot", "kind"),
            "lizyml/core/_model_plots.py:184",
            'if kind == "shap":',
        ),
    ],
)
def test_each_predicate_points_at_the_runtime_check(
    key: tuple[str, str], pointer: str, line: str
) -> None:
    assert CONDITION_ARGUMENTS[key].mirrors == pointer
    path, number = pointer.split(":")
    source = (ROOT / path).read_text(encoding="utf-8").splitlines()
    assert source[int(number) - 1].strip() == line


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
        # The runtime check is `if return_shap:`, so truthiness decides.
        ("predict", {"return_shap": 1}, {"explain"}),
        ("predict", {"return_shap": "yes"}, {"explain"}),
        ("predict", {"return_shap": 0}, set()),
        ("predict", {"return_shap": None}, set()),
        ("importance", {}, set()),
        ("importance", {"kind": "gain"}, set()),
        ("importance", {"kind": "shap"}, {"explain"}),
        # The runtime check is `kind == "shap"`, so any other value needs none.
        ("importance", {"kind": "SHAP"}, set()),
        ("importance", {"kind": None}, set()),
        ("importance_plot", {}, {"plots"}),
        ("importance_plot", {"kind": "split"}, {"plots"}),
        ("importance_plot", {"kind": "shap"}, {"plots", "explain"}),
        ("importance_plot", {"kind": "SHAP"}, {"plots"}),
        ("importance_plot", {"kind": None}, {"plots"}),
        ("residuals_plot", {}, {"plots"}),
        ("residuals_plot", {"kind": "qq"}, {"plots"}),
        ("tuning_plot", {}, {"plots"}),
    ],
)
def test_extras_for(
    method: str, arguments: dict[str, object], expected: set[str]
) -> None:
    assert extras_for(method, arguments) == frozenset(expected)


# --- The predicates against the running code ------------------------------------

SAMPLES: dict[str, tuple[object, ...]] = {
    "return_shap": (True, False, 1, 0, "yes", "", None),
    "kind": ("shap", "SHAP", "Shap", "split", "gain", None, 1),
}


@pytest.fixture(scope="module")
def fitted() -> Model:
    model = Model(make_config("regression", n_estimators=5))
    model.fit(data=make_regression_df(n=120))
    return model


def _shap_guard_raised(model: Model, method: str, argument: str, value: object) -> bool:
    call = getattr(model, method)
    try:
        if method == "predict":
            X = make_regression_df(n=10).drop(columns=["target"])
            call(X, **{argument: value})
        else:
            call(**{argument: value})
    except LizyMLError as exc:
        return (
            exc.code is ErrorCode.OPTIONAL_DEP_MISSING
            and exc.context.get("package") == "shap"
        )
    except Exception:  # noqa: BLE001 - another error is "no shap guard"
        return False
    return False


@pytest.mark.parametrize(
    ("key", "value"),
    [(key, value) for key in sorted(CONDITION_ARGUMENTS) for value in SAMPLES[key[1]]],
)
def test_each_predicate_matches_the_runtime_check(
    key: tuple[str, str],
    value: object,
    fitted: Model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shap_explainer, "_shap", None)  # shap "not installed"
    method, argument = key
    expected = CONDITION_ARGUMENTS[key].needs_extra(value)
    assert _shap_guard_raised(fitted, method, argument, value) is expected


def test_every_row_has_a_probe_for_ci() -> None:
    """The 6.(a) probe exercises every row (it runs only in the CI matrix)."""
    from tests.test_notebooks import probe_extras_registry as probe

    assert probe.uncovered_rules() == []
    assert {f.method for f in probe.FORMS} == {r.method for r in RULES}
    for extra in EXTRA_PACKAGES:
        assert any(extra in extras_for(f.method, f.conditions) for f in probe.FORMS)
    for form in probe.NEGATIVE_FORMS:
        assert "explain" not in extras_for(form.method, form.conditions)
