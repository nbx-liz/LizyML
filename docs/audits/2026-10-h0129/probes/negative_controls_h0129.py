"""H-0129 negative controls: each wrong implementation the acceptance criteria
name is put in place of ``exporter._saved_pipeline_states`` /
``check_category_values``, and the criterion's input must then pass silently
(so the test asserting a refusal would fail).

Run: uv run python docs/audits/2026-10-h0129/probes/negative_controls_h0129.py
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from lizyml.core.exceptions import LizyMLError
from lizyml.estimators.base import BaseEstimatorAdapter, UnsaveableCategory
from lizyml.estimators.lgbm.category_json import encoder_categories, unwritable_categories
from lizyml.persistence import exporter

GOOD = {"encoder": {"categories": {"c": ["a", "b"]}}}
BAD = {"encoder": {"categories": {"c": ["a", b"x"]}}}


class Silent(BaseEstimatorAdapter):
    def fit(self, *a: Any, **k: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...
    def predict_raw(self, X: Any) -> Any: ...
    def importance(self, kind: Any = "split") -> Any: ...
    def get_native_model(self) -> Any: ...
    def save_model_text(self, path: Any) -> Any: ...


class Refusing(Silent):
    def unsaveable_categories(self, pipeline_state: Any) -> list[UnsaveableCategory]:
        return [UnsaveableCategory(column="c", shown="'z'", type_name="str")]


def results(folds: Any, models: list[Any], refit_model: Any, last: Any = GOOD) -> Any:
    fit = SimpleNamespace(pipeline_state_per_fold=folds, pipeline_state=last, models=models)
    return fit, SimpleNamespace(pipeline_state=GOOD, model=refit_model)


def refuses(fit: Any, refit: Any) -> bool:
    try:
        exporter.check_category_values(fit, refit)
    except LizyMLError:
        return True
    return False


real_states = exporter._saved_pipeline_states
real_check = exporter.check_category_values
rows = []

# Criterion 5b: the fold-2 adapter alone refuses.
case_5b = results([GOOD, GOOD, GOOD], [Silent(), Refusing(), Silent()], Silent())
rows.append(("5b real implementation refuses", refuses(*case_5b), True))


def refit_only(fit: Any, refit: Any) -> list[Any]:
    return [(name, state, refit.model) for name, state, _ in real_states(fit, refit)]


exporter._saved_pipeline_states = refit_only
rows.append(("5b control: ask the refit adapter only", refuses(*case_5b), False))
exporter._saved_pipeline_states = real_states

# Criterion 5b: unpaired states.
case_unpaired = results([GOOD, BAD, GOOD], [Refusing(), Refusing()], Silent())
rows.append(("5b real implementation refuses unpaired", refuses(*case_unpaired), True))


def skip_unpaired(fit: Any, refit: Any) -> list[Any]:
    named = [("refit", refit.pipeline_state, refit.model)]
    per_fold = fit.pipeline_state_per_fold or [fit.pipeline_state]
    named += [(f"fold {k}", s, m) for k, (s, m) in enumerate(zip(per_fold, fit.models), 1)]
    return named


exporter._saved_pipeline_states = skip_unpaired
case_unpaired_quiet = results([GOOD, GOOD, BAD], [Silent(), Silent()], Silent())
rows.append(("5b control: zip drops the unpaired state", refuses(*case_unpaired_quiet), False))
exporter._saved_pipeline_states = real_states

# Criterion 6: an adapter without the method refuses nothing.
case_6 = results([BAD, BAD], [Silent(), Silent()], Silent(), last=BAD)
rows.append(("6 real implementation passes", refuses(*case_6), False))


def lightgbm_rule_in_persistence(fit: Any, refit: Any) -> None:
    for _, state, _ in real_states(fit, refit):
        if unwritable_categories(encoder_categories(state)):
            raise LizyMLError(code=exporter.ErrorCode.SERIALIZATION_FAILED, user_message="x")


exporter.check_category_values = lightgbm_rule_in_persistence
rows.append(("6 control: LightGBM rule kept in persistence", refuses(*case_6), True))
exporter.check_category_values = real_check

failures = 0
for label, got, want in rows:
    ok = got == want
    failures += not ok
    print(f"{'ok ' if ok else 'BAD'} {label}: refuses={got} (expected {want})")
print(f"failures: {failures}")
raise SystemExit(1 if failures else 0)
