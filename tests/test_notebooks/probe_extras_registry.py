"""Call every registry row without its extra (H-0119 6.(a)).

Run by CI in an environment built without one extra::

    uv sync --frozen --no-dev --group notebooks <--extra for the other two>
    uv run --no-sync --no-dev python -m \
        tests.test_notebooks.probe_extras_registry --without <extra>

First the environment is checked: the removed extra's package is absent and the
others import. Then every call form below whose registry extras include the
removed extra is run on a model fitted for it, and must raise
``LizyMLError(OPTIONAL_DEP_MISSING)`` with ``context["package"]`` naming that
extra's package. Any other exception, in the prerequisite or in the call, and a
call that succeeds, fail the probe: a row that never reaches the dependency
guard is not evidence for the row.

Every registry row must be exercised by at least one call form, so a row added
to ``lizyml/_extras.py`` without a probe fails here as well.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from lizyml import Model
from lizyml._extras import EXTRA_PACKAGES, RULES, condition_value, extras_for
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_binary_df, make_config, make_regression_df
from tests.test_notebooks.check_installed_extras import assert_installed


def _regression() -> Model:
    model = Model(make_config("regression", n_estimators=10))
    model.fit(data=make_regression_df(n=150))
    return model


def _regression_early_stopping() -> Model:
    config = make_config("regression", n_estimators=50)
    config["training"]["early_stopping"] = {
        "enabled": True,
        "rounds": 5,
        "validation_ratio": 0.2,
    }
    model = Model(config)
    model.fit(data=make_regression_df(n=150))
    return model


def _binary() -> Model:
    model = Model(make_config("binary", n_estimators=10))
    model.fit(data=make_binary_df(n=200))
    return model


def _binary_isotonic() -> Model:
    model = Model(make_config("binary", n_estimators=10, calibration="isotonic"))
    model.fit(data=make_binary_df(n=200))
    return model


def _tuning() -> Model:
    return Model(make_config("regression", n_estimators=10, tuning_n_trials=2))


def _tuned() -> Model:
    model = _tuning()
    model.tune(data=make_regression_df(n=150))
    return model


@dataclass(frozen=True)
class CallForm:
    """``method(**kwargs)`` on the model ``prerequisite`` builds."""

    method: str
    prerequisite: Callable[[], Model]
    kwargs: dict[str, Any] = field(default_factory=dict)
    args: tuple[Any, ...] = ()

    @property
    def conditions(self) -> dict[str, object]:
        return {k: v for k, v in self.kwargs.items() if k in ("kind", "return_shap")}

    def label(self) -> str:
        shown = ", ".join(f"{k}={v!r}" for k, v in self.conditions.items())
        return f"{self.method}({shown})"


X_NEW = make_regression_df(n=20).drop(columns=["target"])

FORMS: tuple[CallForm, ...] = (
    CallForm("tune", _tuning, {"data": make_regression_df(n=150)}),
    CallForm("predict", _regression, {"return_shap": True}, (X_NEW,)),
    CallForm("importance", _regression, {"kind": "shap"}),
    CallForm("importance_plot", _regression),
    CallForm("importance_plot", _regression, {"kind": "shap"}),
    CallForm("residuals_plot", _regression, {"kind": "scatter"}),
    CallForm("plot_oof_distribution", _regression),
    CallForm("plot_learning_curve", _regression_early_stopping),
    CallForm("roc_curve_plot", _binary),
    CallForm("calibration_plot", _binary_isotonic),
    CallForm("probability_histogram_plot", _binary_isotonic),
    CallForm("tuning_plot", _tuned),
)


def uncovered_rules() -> list[str]:
    """Registry rows no call form triggers."""
    missing = []
    for rule in RULES:
        covered = False
        for form in FORMS:
            if form.method != rule.method:
                continue
            if rule.condition is None:
                covered = True
            else:
                argument, value = rule.condition
                actual = condition_value(form.method, argument, form.conditions)
                covered = covered or (type(actual) is type(value) and actual == value)
        if not covered:
            missing.append(f"{rule.method} -> {rule.extra} when {rule.condition}")
    return missing


def probe(form: CallForm, removed: str) -> str | None:
    """Run ``form``; return a failure message or ``None`` when it behaved."""
    try:
        model = form.prerequisite()
    except Exception as exc:  # noqa: BLE001 - any prerequisite failure is reported
        return f"{form.label()}: the prerequisite failed with {exc!r}"
    try:
        getattr(model, form.method)(*form.args, **form.kwargs)
    except LizyMLError as exc:
        package = exc.context.get("package")
        if (
            exc.code is ErrorCode.OPTIONAL_DEP_MISSING
            and package == EXTRA_PACKAGES[removed]
        ):
            return None
        return f"{form.label()}: raised {exc.code} with package {package!r}"
    except Exception as exc:  # noqa: BLE001 - not the dependency guard
        return f"{form.label()}: raised {exc!r}, not OPTIONAL_DEP_MISSING"
    return f"{form.label()}: succeeded without {removed!r}; the row is not needed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--without", required=True, choices=sorted(EXTRA_PACKAGES))
    removed = parser.parse_args(argv).without
    for line in assert_installed(set(EXTRA_PACKAGES) - {removed}, {removed}):
        print(line)
    missing = uncovered_rules()
    if missing:
        print("registry rows without a probe:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    selected = [f for f in FORMS if removed in extras_for(f.method, f.conditions)]
    if not selected:
        print(f"no call form needs {removed!r}", file=sys.stderr)
        return 1
    failures = []
    for form in selected:
        failure = probe(form, removed)
        print(
            f"{'FAIL' if failure else 'ok  '} {form.label()}"
            + (f": {failure}" if failure else "")
        )
        if failure:
            failures.append(failure)
    print(f"{len(selected) - len(failures)}/{len(selected)} rows raised as declared")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
