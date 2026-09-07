"""``Model.fit(params=...)`` must reach the trained model (H-0094, #264).

``fit()`` accepted and documented a ``params`` argument and never forwarded it.
The overlay it was meant to reach (``_merge_params``'s ``override``) was correct
and had no caller, so every override was discarded in silence: no error, no
effect, and a booster trained on the config value.

The tests here assert at the level the defect lived at. Asserting on the merged
params dict alone would have passed against the shipped code, because that dict
was already right -- nothing handed it the override. So the claims are made
against the **trained Booster** and against what ``lgb.train`` received.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
from typing import Any
from unittest import mock

import numpy as np
import pytest

from lizyml import Model
from lizyml.core._model_factories import (
    check_duplicate_identities,
    overlay_params,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuningResult
from lizyml.core.value_equality import values_differ
from lizyml.estimators.lgbm.adapter import _pop_by_identity
from lizyml.estimators.lgbm.param_names import accepted_spellings
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.estimators.lgbm.smart_params import SMART_PARAM_TARGETS
from tests._helpers import make_binary_df, make_config, make_multiclass_df
from tests._train_spy import record_lightgbm_calls

REPO = pathlib.Path(__file__).resolve().parents[2]

#: A native LightGBM name whose value is visible in the booster text, so a fit
#: that honoured the override can be told from one that did not by reading the
#: model rather than by trusting a dict LizyML built.
OVERRIDDEN = "learning_rate"
CONFIG_VALUE = 0.001
OVERRIDE_VALUE = 0.5


def _fit(params: dict[str, Any] | None, **cfg_overrides: Any) -> Any:
    cfg = make_config(
        "binary",
        n_estimators=5,
        n_splits=2,
        learning_rate=CONFIG_VALUE,
        **cfg_overrides,
    )
    model = Model(cfg, data=make_binary_df(n=120))
    model.fit(params=params)
    return model


def _booster_text(model: Model) -> str:
    return str(model.fit_result.models[0].get_native_model().model_to_string())


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_fit_params_changes_the_trained_booster() -> None:
    """The DoD claim: two fits differing only in ``params`` must differ.

    Red before the fix -- the two booster texts were byte-identical and both
    carried the config's ``learning_rate``, which is the whole defect.
    """
    without = _booster_text(_fit(None))
    with_override = _booster_text(_fit({OVERRIDDEN: OVERRIDE_VALUE}))

    assert without != with_override, (
        "two fits differing only in fit(params=...) produced identical "
        "boosters, so the override reached nothing"
    )
    assert f"[{OVERRIDDEN}: {OVERRIDE_VALUE}]" in with_override, (
        f"the booster does not carry the overridden {OVERRIDDEN}; "
        "it was trained on some other value"
    )
    assert f"[{OVERRIDDEN}: {CONFIG_VALUE}]" in without, (
        "the control fit did not carry the config value either, so the "
        "comparison above proves nothing"
    )


def test_the_override_reaches_lgb_train_itself() -> None:
    """Read it where it lands, not where LizyML assembled it."""
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=120))

    with record_lightgbm_calls() as seen:
        model.fit(params={OVERRIDDEN: OVERRIDE_VALUE})

    assert seen["train_params"], "no lgb.train call was recorded"
    values = {call.get(OVERRIDDEN) for call in seen["train_params"]}
    assert values == {OVERRIDE_VALUE}, (
        f"lgb.train received {OVERRIDDEN}={values}, expected only {OVERRIDE_VALUE}"
    )


# ---------------------------------------------------------------------------
# The priority the docstring declares: config < tune best < fit() args
# ---------------------------------------------------------------------------


def test_fit_params_outrank_the_tuning_result() -> None:
    """``fit(params=)`` is documented as the highest priority. Execute that.

    A tuned model whose user re-fits with an explicit override must get the
    override, not the tuned value. This is the one ordering claim that cannot
    be read off the config.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=120))
    model._tuning_result = TuningResult(
        best_model_params={OVERRIDDEN: 0.25},
        best_smart_params={},
        best_training_params={},
        best_score=0.0,
        metric_name="auc",
        direction="maximize",
        trials=(),
        rounds=(),
    )

    with record_lightgbm_calls() as seen:
        model.fit(params={OVERRIDDEN: OVERRIDE_VALUE})

    values = {call.get(OVERRIDDEN) for call in seen["train_params"]}
    assert values == {OVERRIDE_VALUE}, (
        f"the tuned value won over the fit() override: lgb.train saw {values}"
    )


def test_the_tuning_result_still_outranks_the_config() -> None:
    """The rung below, so the ordering is pinned and not just its top."""
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=120))
    model._tuning_result = TuningResult(
        best_model_params={OVERRIDDEN: 0.25},
        best_smart_params={},
        best_training_params={},
        best_score=0.0,
        metric_name="auc",
        direction="maximize",
        trials=(),
        rounds=(),
    )

    with record_lightgbm_calls() as seen:
        model.fit()

    values = {call.get(OVERRIDDEN) for call in seen["train_params"]}
    assert values == {0.25}, f"expected the tuned value, lgb.train saw {values}"


# ---------------------------------------------------------------------------
# The boundary PR 1 closed must stay closed through the new route
# ---------------------------------------------------------------------------


def test_an_unknown_name_in_fit_params_is_refused_before_training() -> None:
    """Forwarding without checking would reopen H-0093 through a new door.

    Red at `5712f41` *and after PR 1*: the override was inert, so nothing
    reached any check and nothing was refused. The name check sits on the dict
    ``_merge_params`` returns, which is where the override lands.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit(params={"not_a_lightgbm_parameter": 1})

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert not seen["train_params"], (
        f"{len(seen['train_params'])} Booster(s) were trained before the "
        "refusal; the check must fire before any training"
    )


def test_the_refusal_names_the_fit_params_surface() -> None:
    """A message pointing at ``model.params`` sends the user to the wrong file.

    The offending name is not in the config at all here -- it came from the
    call. Naming the surface is what makes the error actionable.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))

    with pytest.raises(LizyMLError) as exc:
        model.fit(params={"not_a_lightgbm_parameter": 1})

    message = str(exc.value)
    assert "fit(params=)" in message, (
        f"the rejection must name the surface the name came from; got: {message}"
    )
    surfaces = [entry["surface"] for entry in exc.value.context.get("unknown", [])]
    assert surfaces == ["fit(params=)"], (
        f"the machine-readable context names {surfaces}"
    )


def test_a_smart_name_in_fit_params_is_reported_as_smart() -> None:
    """LizyML's own namespace has its own message; the new surface keeps it."""
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))

    with pytest.raises(LizyMLError) as exc:
        model.fit(params={"num_leaves_ratio": 0.5})

    message = str(exc.value)
    assert "smart parameter" in message, message
    assert "fit(params=)" in message, message


def test_each_input_is_named_by_its_own_surface() -> None:
    """Three inputs merge into one dict; the error must not blame one of them.

    ``model.params``, a restored ``best_model_params`` and the ``fit()``
    override all land in the same dict, and before this the whole dict was
    reported as ``model.params``. Two of the three would have sent the user to
    a file that does not contain the name.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    cfg["model"]["params"]["not_a_lightgbm_parameter"] = 1
    model = Model(cfg, data=make_binary_df(n=120))
    model._tuning_result = TuningResult(
        best_model_params={"also_not_a_parameter": 2},
        best_smart_params={},
        best_training_params={},
        best_score=0.0,
        metric_name="auc",
        direction="maximize",
        trials=(),
        rounds=(),
    )

    with pytest.raises(LizyMLError) as exc:
        model.fit(params={"third_invented_name": 3})

    by_name = {
        entry["name"]: entry["surface"] for entry in exc.value.context["unknown"]
    }
    assert by_name == {
        "not_a_lightgbm_parameter": "model.params",
        "also_not_a_parameter": "tuning best_model_params",
        "third_invented_name": "fit(params=)",
    }, by_name


# ---------------------------------------------------------------------------
# The other side of the gate: it must change nothing it was not asked to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [None, {}])
def test_no_override_leaves_the_config_value_in_place(
    params: dict[str, Any] | None,
) -> None:
    """``None`` and ``{}`` are both "no override", and must stay that way."""
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=120))

    with record_lightgbm_calls() as seen:
        model.fit(params=params)

    values = {call.get(OVERRIDDEN) for call in seen["train_params"]}
    assert values == {CONFIG_VALUE}, (
        f"passing params={params!r} changed the training values: {values}"
    )


def test_the_override_does_not_outlive_the_call() -> None:
    """It overrides *this* fit, not the model's configuration.

    ``_merge_params`` builds a new dict rather than mutating ``cfg.model``, and
    a caller who overrides once and re-fits must get the config value back. The
    config object is the caller's own -- silently rewriting it would be a
    second, invisible effect of a documented argument.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=120))
    model.fit(params={OVERRIDDEN: OVERRIDE_VALUE})

    assert model._cfg.model.params[OVERRIDDEN] == CONFIG_VALUE, (
        "fit(params=) mutated the caller's config"
    )

    with record_lightgbm_calls() as seen:
        model.fit()
    values = {call.get(OVERRIDDEN) for call in seen["train_params"]}
    assert values == {CONFIG_VALUE}, (
        f"the previous call's override persisted into the next fit: {values}"
    )


# ---------------------------------------------------------------------------
# Names a smart parameter is going to overwrite (review round 1)
# ---------------------------------------------------------------------------

#: Every native name an active smart parameter writes, with the config that
#: switches that smart parameter on and off and a value to try.
#:
#: Review round 1 measured the defect this closes: forwarding alone made
#: ``fit(params={"num_leaves": 12})`` train at 32, ``min_data_in_leaf=3`` at 1
#: and ``scale_pos_weight=10`` at 0.935. Smart resolution runs downstream of the
#: merge and wins, so those overrides were accepted and replaced -- the same
#: silence this whole PR is about, one level down.
#:
#: The population is not this dict: ``test_the_managed_table_matches_the_code``
#: derives the native names from the assignments in ``smart_params.py``.
MANAGED_CASES: dict[str, dict[str, Any]] = {
    "num_leaves": {
        "smart": "auto_num_leaves",
        "enable": {},
        "disable": {"auto_num_leaves": False},
        "value": 12,
    },
    "min_data_in_leaf": {
        "smart": "min_data_in_leaf_ratio",
        "enable": {},
        "disable": {"min_data_in_leaf_ratio": None},
        "value": 3,
    },
    "min_data_in_bin": {
        "smart": "min_data_in_bin_ratio",
        "enable": {},
        "disable": {"min_data_in_bin_ratio": None},
        "value": 3,
    },
    "scale_pos_weight": {
        "smart": "balanced",
        "enable": {},
        "disable": {"balanced": False},
        "value": 10,
    },
    "feature_contri": {
        "smart": "feature_weights",
        "enable": {"feature_weights": {"feat_a": 2.0}},
        "disable": {},
        "value": [2.0, 1.0],
    },
    "feature_pre_filter": {
        "smart": "feature_weights",
        "enable": {"feature_weights": {"feat_a": 2.0}},
        "disable": {},
        "value": False,
    },
}

#: Smart parameters that write no native name of their own. ``num_leaves_ratio``
#: is an input to ``auto_num_leaves``'s computation, not a writer, so it cannot
#: collide with anything. Declared so the two sets can be asserted to cover the
#: provider's whole smart surface.
SMART_PARAMS_THAT_WRITE_NOTHING: frozenset[str] = frozenset({"num_leaves_ratio"})


#: Every spelling LightGBM accepts for a managed name, paired with the canonical
#: one, read from LightGBM's own alias registry rather than listed here.
#:
#: Review round 2 measured why: the refusal compared literal names, so
#: ``max_leaves`` passed it, and LightGBM -- which treats the alias as
#: ``num_leaves`` -- then preferred the value smart resolution had supplied.
#: ``fit(params={"max_leaves": 12})`` trained at 32 and said nothing. Refusing
#: one spelling of a parameter and admitting another refuses nothing.
MANAGED_SPELLINGS: list[tuple[str, str]] = sorted(
    (spelling, canonical)
    for canonical in MANAGED_CASES
    for spelling in accepted_spellings(canonical)
)


def _fit_with(smart: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    cfg["model"].update(smart)
    model = Model(cfg, data=make_binary_df(n=160))
    with record_lightgbm_calls() as seen:
        model.fit(params=params)
    return list(seen["train_params"])


@pytest.mark.parametrize(("spelling", "native"), MANAGED_SPELLINGS)
def test_a_managed_name_is_refused_rather_than_replaced(
    spelling: str, native: str
) -> None:
    """Accepting a value that is then discarded is the defect, not the fix.

    Parametrized over every spelling the library accepts, because the estimator
    resolves aliases and a check that does not would leave the same parameter
    reachable under another name.
    """
    case = MANAGED_CASES[native]
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    cfg["model"].update(case["enable"])
    model = Model(cfg, data=make_binary_df(n=160))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit(params={spelling: case["value"]})

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    message = str(exc.value)
    assert spelling in message and case["smart"] in message, (
        "the refusal must name both the parameter as written and the smart "
        f"parameter that manages it; got: {message}"
    )
    if spelling != native:
        assert native in message, (
            f"an alias must be told what it names; {spelling!r} was refused "
            f"without mentioning {native!r}: {message}"
        )
    assert not seen["train_params"], (
        f"{len(seen['train_params'])} Booster(s) were trained before the refusal"
    )


@pytest.mark.parametrize(("spelling", "native"), MANAGED_SPELLINGS)
def test_the_same_name_applies_once_its_smart_parameter_is_off(
    spelling: str, native: str
) -> None:
    """The other direction, and the reason the table is not merely a list.

    Each entry claims "an active smart parameter overwrites this name". Switch
    that smart parameter off and the very same override must reach ``lgb.train``
    untouched. Without this, the table could name anything at all and every
    refusal above would still pass.
    """
    case = MANAGED_CASES[native]
    calls = _fit_with(case["disable"], {spelling: case["value"]})

    assert calls, "no lgb.train call was recorded"
    got = [call.get(spelling) for call in calls]
    assert all(value == case["value"] for value in got), (
        f"with {case['smart']} disabled, {spelling} should reach lgb.train as "
        f"{case['value']!r}; it arrived as {got}"
    )


def test_the_alias_review_round_2_measured_is_refused() -> None:
    """The exact case, kept as itself so the regression has a name.

    ``max_leaves`` is LightGBM's alias for ``num_leaves``. Before this, it
    passed the refusal, reached ``lgb.train`` alongside the smart-resolved
    ``num_leaves``, and LightGBM used the canonical one: measured as
    ``[(12, 32), (12, 32), (12, 32)]`` with ``[num_leaves: 32]`` in the booster.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=160))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit(params={"max_leaves": 12})

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert not seen["train_params"]
    names = {entry["name"] for entry in exc.value.context["managed"]}
    canonicals = {entry["canonical"] for entry in exc.value.context["managed"]}
    assert names == {"max_leaves"} and canonicals == {"num_leaves"}, (
        f"context reported names={names} canonicals={canonicals}"
    )


def test_every_alias_of_a_managed_name_is_covered() -> None:
    """The spelling population must come from the library, not from a list.

    If ``accepted_spellings`` ever returned only the canonical name, every
    parametrized cell above would still pass and the alias hole would be back.
    """
    for canonical in MANAGED_CASES:
        spellings = {s for s, c in MANAGED_SPELLINGS if c == canonical}
        assert canonical in spellings
        assert spellings == accepted_spellings(canonical)
    assert any(s != c for s, c in MANAGED_SPELLINGS), (
        "no alias appears in the population at all, so the alias direction is "
        "untested; LightGBM defines aliases for num_leaves, min_data_in_leaf "
        "and feature_contri"
    )


def test_an_unmanaged_name_is_never_refused() -> None:
    """The control: the gate must not swallow ordinary overrides."""
    calls = _fit_with({}, {OVERRIDDEN: OVERRIDE_VALUE})
    assert {call.get(OVERRIDDEN) for call in calls} == {OVERRIDE_VALUE}


def test_scale_pos_weight_is_not_managed_for_multiclass() -> None:
    """``balanced`` writes a sample weight there, not a parameter name.

    A table keyed only by smart parameter would refuse this, which would be a
    refusal with no defect behind it.
    """
    cfg = make_config("multiclass", n_estimators=5, n_splits=2)
    model = Model(cfg, data=make_multiclass_df(n=180))
    with record_lightgbm_calls() as seen:
        model.fit(params={"scale_pos_weight": 10})
    assert {call.get("scale_pos_weight") for call in seen["train_params"]} == {10}


def test_the_managed_table_matches_the_code_that_writes_the_names() -> None:
    """Close the table against the assignments, not against a reading of them.

    ``SMART_PARAM_TARGETS`` claims to name every native parameter smart
    resolution writes. That claim would go stale the day a smart parameter
    learns to write a fourth one, and nothing else would notice: the refusal
    would simply not fire and the override would be silently replaced again.
    """
    source = (REPO / "lizyml/estimators/lgbm/smart_params.py").read_text()
    tree = ast.parse(source)

    written: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in {"resolve_smart_params", "resolve_ratio_params"}:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Assign):
                continue
            for target in inner.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "resolved"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    written.add(target.slice.value)

    declared = {name for names in SMART_PARAM_TARGETS.values() for name in names}
    assert written, "the scan found no `resolved[...] = ...` assignment at all"
    assert declared == written, (
        f"SMART_PARAM_TARGETS declares {sorted(declared)}; the resolvers write "
        f"{sorted(written)}. A name written but not declared is silently "
        "overwritten again; a name declared but not written is a refusal with "
        "no defect behind it."
    )


def test_every_smart_parameter_is_classified() -> None:
    """The provider's smart surface must be covered by the two sets."""
    smart_names = LGBMProvider().smart_param_names()
    classified = set(SMART_PARAM_TARGETS) | SMART_PARAMS_THAT_WRITE_NOTHING

    unclassified = sorted(smart_names - classified)
    assert not unclassified, (
        f"smart parameter(s) classified by neither set: {unclassified}. If it "
        "writes a native parameter, add it to SMART_PARAM_TARGETS; if it does "
        "not, declare it in SMART_PARAMS_THAT_WRITE_NOTHING."
    )
    gone = sorted(classified - smart_names)
    assert not gone, f"classified name(s) the provider does not report as smart: {gone}"


# ---------------------------------------------------------------------------
# One parameter, two spellings (review round 3)
# ---------------------------------------------------------------------------

#: LightGBM's alias for ``learning_rate``. Not managed by any smart parameter,
#: so these tests are about the merge itself and nothing else.
ALIAS = "eta"

#: ``_COMMON_DEFAULTS`` carries ``learning_rate``, so the canonical spelling is
#: in the parameter dict of every fit whether the user wrote it or not. That is
#: what made this defect reachable without any config entry at all.
DEFAULTED = 0.001


def _learning_rate_in_booster(model: Model) -> str:
    line = [
        s for s in _booster_text(model).splitlines() if s.startswith(f"[{OVERRIDDEN}:")
    ]
    assert line, "the booster does not report learning_rate at all"
    return line[0]


def test_an_alias_override_wins_over_a_canonical_config_value() -> None:
    """The round-3 finding: one parameter, two spellings, override lost.

    ``{**config, **override}`` keeps both keys, and LightGBM prefers the
    canonical one. Measured before the fix: ``fit(params={"eta": 0.5})`` on a
    config carrying ``learning_rate: 0.001`` trained at 0.001.
    """
    model = _fit({ALIAS: OVERRIDE_VALUE})
    assert _learning_rate_in_booster(model) == f"[{OVERRIDDEN}: {OVERRIDE_VALUE}]"


def test_an_alias_override_wins_over_the_estimator_defaults() -> None:
    """No config entry at all, and the override still has to win.

    The defaults are canonical and are merged in below the user's parameters,
    so an alias override was beaten by a value the user never wrote.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=160))
    model.fit(params={ALIAS: OVERRIDE_VALUE})
    assert _learning_rate_in_booster(model) == f"[{OVERRIDDEN}: {OVERRIDE_VALUE}]"


def test_the_estimator_never_sees_two_spellings_of_one_parameter() -> None:
    """Do not rely on which spelling the estimator prefers -- send one.

    Asserting only on the outcome would pass while the merged dict still
    carried both keys, leaving the result at the mercy of a library rule this
    code does not own.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=160))

    with record_lightgbm_calls() as seen:
        model.fit(params={ALIAS: OVERRIDE_VALUE})

    assert seen["train_params"], "no lgb.train call was recorded"
    for call in seen["train_params"]:
        spellings = sorted({ALIAS, OVERRIDDEN} & set(call))
        assert spellings == [ALIAS], (
            f"lgb.train received {spellings}; it must receive exactly the "
            "spelling the caller used, with the other one removed"
        )


def test_an_alias_in_the_tuning_result_wins_over_the_config() -> None:
    """The same seam one rung down, and the rung a real workflow uses.

    ``fit(params=tuning_result.best_model_params)`` is why the argument was
    kept rather than removed, so a tuned value spelled differently from the
    config must not lose to it either.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=160))
    model._tuning_result = TuningResult(
        best_model_params={ALIAS: 0.25},
        best_smart_params={},
        best_training_params={},
        best_score=0.0,
        metric_name="auc",
        direction="maximize",
        trials=(),
        rounds=(),
    )

    model.fit()
    assert _learning_rate_in_booster(model) == f"[{OVERRIDDEN}: 0.25]"


def test_an_alias_in_the_config_now_applies_too() -> None:
    """A consequence of the fix, asserted rather than left to be discovered.

    Before it, ``model.params: {"eta": 0.07}`` trained at 0.001 -- the
    canonical default shadowed the alias the user wrote, silently. Fixing the
    merge for the override fixes this seam as well, which changes what an
    existing config does and is recorded in the CHANGELOG for that reason.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    cfg["model"]["params"][ALIAS] = 0.07
    model = Model(cfg, data=make_binary_df(n=160))
    model.fit()
    assert _learning_rate_in_booster(model) == f"[{OVERRIDDEN}: 0.07]"


def test_overlay_params_keeps_a_name_the_estimator_does_not_know() -> None:
    """Canonicalisation must not double as the accepted-name gate.

    An unknown name has no canonical form; if ``overlay_params`` dropped it,
    the refusal that H-0093 owns would never see it and an invented name would
    become a silent no-op again -- the exact defect this PR sits on top of.
    """
    provider = LGBMProvider()
    merged = overlay_params(
        provider,
        {"learning_rate": 0.1, "invented_name": 1},
        {"eta": 0.5},
    )
    assert merged == {"invented_name": 1, "eta": 0.5}, merged


# ---------------------------------------------------------------------------
# The adapter's own special handling, by identity (review round 4)
# ---------------------------------------------------------------------------

#: The parameters the adapter pulls out of the merged dict and treats specially
#: -- validating, renaming, or passing them as a call argument -- with a value
#: to try and how to read the outcome back.
#:
#: Each was matched by one literal name. That was survivable while a canonical
#: default sat beside the alias and won; once the merge became identity-aware
#: and dropped the shadowing default, the alias became the value that trained
#: **and skipped the handling**. Measured on a binary task:
#: ``fit(params={"application": "regression"})`` trained a regression objective.
SPECIAL_HANDLING_ALIASES: dict[str, str] = {
    "objective": "application",
    "metric": "metrics",
    "num_iterations": "num_round",
}


def _rounds_handed_to_lgb_train(params: dict[str, Any] | None) -> list[int]:
    """``num_boost_round`` is a call argument, not a key in the params dict."""
    import lightgbm as lgb

    seen: list[int] = []
    real = lgb.train

    def spy(
        params_: Any, train_set: Any, num_boost_round: int = 100, *a: Any, **kw: Any
    ) -> Any:
        seen.append(num_boost_round)
        return real(params_, train_set, num_boost_round, *a, **kw)

    cfg = make_config("binary", n_estimators=3, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))
    lgb.train = spy  # type: ignore[assignment]
    try:
        model.fit(params=params)
    finally:
        lgb.train = real  # type: ignore[assignment]
    return seen


def test_an_objective_alias_gets_the_same_task_check() -> None:
    """The check must follow the parameter, not the spelling.

    ``application`` is LightGBM's alias for ``objective``. Before this, a
    cross-task objective written that way skipped ``_check_objective_compatible``
    entirely and trained: measured as ``[objective: regression]`` on a binary
    task.
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit(params={"application": "regression"})

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "regression" in str(exc.value)
    assert not seen["train_params"], "it trained before refusing"


def test_a_compatible_objective_alias_still_trains() -> None:
    """The other side: the check must not refuse what it should accept."""
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))
    model.fit(params={"application": "binary"})
    assert "[objective: binary]" in _booster_text(model)


def test_two_spellings_of_one_parameter_with_the_same_value_are_fine() -> None:
    """Redundant is not ambiguous, and must not be an internal error.

    This raised ``KeyError: 'objective'``: the adapter validated the popped
    ``objective`` into its params, and the shadow-drop then deleted it as if it
    were a default, because the alias was still in the user dict.
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))
    model.fit(params={"objective": "binary", "application": "binary"})
    assert "[objective: binary]" in _booster_text(model)


def test_two_spellings_with_different_values_are_refused() -> None:
    """Ambiguous is refused, and the message names both spellings.

    Resolving it by dictionary order would decide the training run on something
    the caller cannot see -- the class of defect this change exists to remove.
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit(params={"objective": "binary", "application": "cross_entropy"})

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    message = str(exc.value)
    assert "objective" in message and "application" in message, message
    assert not seen["train_params"], "it trained before refusing"


def test_two_spellings_of_an_ordinary_parameter_are_refused_too() -> None:
    """The refusal cannot live only in the adapter special handling.

    ``objective`` is caught there because it is popped and validated; an
    ordinary parameter is not popped by anything, so both spellings would
    survive into the dict and the estimator would pick one. Removing the facade
    check leaves the objective case still passing and this one failing, which is
    what makes this test the one that holds it.
    """
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=160))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit(params={OVERRIDDEN: 0.1, ALIAS: 0.2})

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    message = str(exc.value)
    assert OVERRIDDEN in message and ALIAS in message, message
    assert not seen["train_params"], "it trained before refusing"

    written = exc.value.context["conflicts"][0]["written"]
    assert written == {OVERRIDDEN: 0.1, ALIAS: 0.2}, written


def test_the_same_ordinary_value_under_two_spellings_is_accepted() -> None:
    """Redundant is not ambiguous here either."""
    calls = _fit_with({}, {OVERRIDDEN: 0.2, ALIAS: 0.2})
    assert calls, "no lgb.train call was recorded"
    assert all(call.get(OVERRIDDEN, call.get(ALIAS)) == 0.2 for call in calls)


@pytest.mark.parametrize("spelling", ["n_estimators", "num_iterations", "num_round"])
def test_a_boosting_round_alias_sets_the_rounds(spelling: str) -> None:
    """``num_boost_round`` is extracted from the params, so identity matters.

    Only the literal ``n_estimators`` was extracted; another spelling stayed in
    the dict and reached ``lgb.train`` as a parameter beside a different
    ``num_boost_round`` argument.
    """
    assert _rounds_handed_to_lgb_train({spelling: 7}) == [7, 7, 7]
    assert _rounds_handed_to_lgb_train(None) == [3, 3, 3]


def test_a_metric_alias_is_taken_as_the_metric() -> None:
    """The third specially handled name, for the same reason."""
    calls = _fit_with({}, {"metrics": "auc"})
    assert calls, "no lgb.train call was recorded"
    assert all(call.get("metric") == ["auc"] for call in calls), (
        f"metric reached lgb.train as {[c.get('metric') for c in calls]}"
    )


def test_every_specially_handled_name_has_an_alias_under_test() -> None:
    """The population is the adapter's special handling, checked against it.

    A parameter the adapter starts treating specially, matched by one literal
    name, is this defect again. This fails when one is added without a case
    here.
    """
    source = (REPO / "lizyml/estimators/lgbm/adapter.py").read_text()
    popped = set(re.findall(r'_pop_by_identity\(user_params, "([^"]+)"\)', source))
    assert popped == set(SPECIAL_HANDLING_ALIASES), (
        f"the adapter pops {sorted(popped)} by identity; the cases here cover "
        f"{sorted(SPECIAL_HANDLING_ALIASES)}"
    )
    for canonical, alias in SPECIAL_HANDLING_ALIASES.items():
        assert alias in accepted_spellings(canonical), (
            f"{alias!r} is not a spelling of {canonical!r}, so the case named "
            "for it tests nothing"
        )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (1, 1.0),
        (0.5, 0.5),
    ],
)
def test_equal_values_under_two_spellings_are_accepted_whatever_the_type(
    first: Any, second: Any
) -> None:
    """Equal is equal, even when the two are written differently.

    The refusal compared ``repr`` at first, so ``1`` and ``1.0`` read as two
    values and a call that meant one thing twice was refused (review round 5).
    A gate that refuses valid input is worse here than the ambiguity it exists
    to catch, and it disagreed with ``_pop_by_identity``, which compares by
    equality.
    """
    calls = _fit_with({}, {OVERRIDDEN: first, ALIAS: second})
    assert calls, "no lgb.train call was recorded"
    assert all(call.get(OVERRIDDEN, call.get(ALIAS)) == first for call in calls), [
        c.get(OVERRIDDEN, c.get(ALIAS)) for c in calls
    ]


def test_the_two_refusals_agree_about_what_equal_means() -> None:
    """The adapter and the facade must not disagree on one input.

    ``_pop_by_identity`` refuses a conflicting objective; the facade refuses a
    conflicting anything. If they used different notions of equality, the same
    call would be accepted or refused depending on which parameter it named.
    """
    provider = LGBMProvider()
    # ``True == 1`` in Python, and this is where that belongs: LightGBM cannot
    # parse a bool as a learning rate, so the pair cannot be checked through a
    # real fit without testing the library's parser instead of the refusal.
    for first, second in ((1, 1.0), (0.5, 0.5), (True, 1)):
        check_duplicate_identities(
            provider, {OVERRIDDEN: first, ALIAS: second}, surface="probe"
        )
        assert _pop_by_identity(
            {"objective": first, "application": second}, "objective"
        ) == (first, "objective")

    with pytest.raises(LizyMLError):
        check_duplicate_identities(provider, {OVERRIDDEN: 1, ALIAS: 2}, surface="probe")
    with pytest.raises(LizyMLError):
        _pop_by_identity(
            {"objective": "binary", "application": "xentropy"}, "objective"
        )


def test_an_unhashable_value_does_not_break_the_refusal() -> None:
    """``feature_contri`` is a list, and a set of values would raise on it."""
    provider = LGBMProvider()
    check_duplicate_identities(
        provider,
        {"feature_contri": [1.0, 2.0], "feature_contrib": [1.0, 2.0]},
        surface="probe",
    )
    with pytest.raises(LizyMLError):
        check_duplicate_identities(
            provider,
            {"feature_contri": [1.0, 2.0], "feature_contrib": [2.0, 1.0]},
            surface="probe",
        )


# ---------------------------------------------------------------------------
# What a parameter's *value* can be (self-review + rounds 4-5 monitor)
# ---------------------------------------------------------------------------

#: Values a LightGBM parameter plausibly arrives as, beyond a number or string.
#:
#: Five rounds went over *which name* a parameter is written under. Nothing had
#: looked at what its value can be, and both refusals compared values with a
#: bare ``!=``: for a numpy array that yields an array, and ``bool()`` of it
#: raises. It raised even for a value written **once**, because the comparison
#: was made against itself. ``feature_contri`` and ``monotone_constraints`` both
#: plausibly arrive as arrays.
AWKWARD_VALUES: dict[str, Any] = {
    "numpy array": np.array([1.0, 2.0]),
    "list": [1.0, 2.0],
    "tuple": (1.0, 2.0),
    "none": None,
    "bool": True,
    "empty list": [],
}


@pytest.mark.parametrize("label", sorted(AWKWARD_VALUES))
def test_a_single_value_of_any_shape_passes_the_duplicate_refusal(label: str) -> None:
    """One spelling cannot be a duplicate, whatever the value is.

    The refusal compared each value against the group's first -- itself, when
    the group has one member -- so an array value raised ``ValueError`` on a
    call that named nothing twice.
    """
    check_duplicate_identities(
        LGBMProvider(), {"feature_contri": AWKWARD_VALUES[label]}, surface="probe"
    )


@pytest.mark.parametrize("label", sorted(AWKWARD_VALUES))
def test_equal_values_of_any_shape_are_accepted_under_two_spellings(
    label: str,
) -> None:
    """And two spellings of the same value are the same value."""
    value = AWKWARD_VALUES[label]
    provider = LGBMProvider()
    check_duplicate_identities(
        provider,
        {"feature_contri": value, "feature_contrib": value},
        surface="probe",
    )
    assert _pop_by_identity(
        {"feature_contri": value, "feature_contrib": value}, "feature_contri"
    ) == (value, "feature_contri")


def test_arrays_that_differ_are_still_refused() -> None:
    """The refusal must not be bought by making everything compare equal."""
    provider = LGBMProvider()
    with pytest.raises(LizyMLError):
        check_duplicate_identities(
            provider,
            {
                "feature_contri": np.array([1.0, 2.0]),
                "feature_contrib": np.array([2.0, 1.0]),
            },
            surface="probe",
        )
    with pytest.raises(LizyMLError):
        _pop_by_identity(
            {
                "feature_contri": np.array([1.0, 2.0]),
                "feature_contrib": np.array([2.0, 1.0]),
            },
            "feature_contri",
        )


def test_an_array_valued_parameter_survives_a_real_fit() -> None:
    """The path the defect was reachable on: an override, before any training.

    ``check_duplicate_identities`` runs unconditionally on the override, so the
    crash needed no duplicate and no unusual config -- just an array value.
    """
    df = make_binary_df(n=160)
    n_features = len([c for c in df.columns if c != "target"])
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    model = Model(cfg, data=df)

    with record_lightgbm_calls() as seen:
        model.fit(params={"feature_contri": np.ones(n_features)})

    assert seen["train_params"], "no lgb.train call was recorded"


def test_the_two_refusals_agree_on_the_awkward_values_too() -> None:
    """One notion of equality, shared, so neither can drift from the other."""
    for value in AWKWARD_VALUES.values():
        assert not values_differ(value, value)
    assert values_differ(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
    assert values_differ([1.0], [1.0, 2.0])
    assert not values_differ(1, 1.0)


# ---------------------------------------------------------------------------
# What the override does to what is saved (rounds 5-6 monitor)
# ---------------------------------------------------------------------------
# Six review rounds went over the path from `fit(params=...)` to `lgb.train`.
# None of them, and none of the 41 tests above, paired an override with
# `export`, `load` or `export_code` -- the monitor found no occurrence of those
# words in this file. The behaviour turns out to be right; it was simply never
# pinned, so these say what it is.


def test_the_override_reaches_the_exported_booster(tmp_path: pathlib.Path) -> None:
    """The artifact must carry the model that was trained, not the config's.

    Read back from the artifact, not from the model in memory. The first version
    of this test asserted on ``_booster_text(model)`` and passed with ``export``
    replaced by a no-op -- green because it never looked at what was written
    (review round 7). Both persisted surfaces are checked: the CV boosters and
    the refit model, which is the one ``predict`` uses.
    """
    out = tmp_path / "artifact"
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=160))
    model.fit(params={OVERRIDDEN: OVERRIDE_VALUE})
    model.export(out)

    restored = Model.load(out)
    # Pinned rather than assumed: the assertions below must be about what was
    # written, not about the model still in memory. The first version of this
    # test read the in-memory model and stayed green with `export` replaced by
    # a no-op, and only this line makes that substitution fail.
    assert restored is not model and restored.fit_result is not model.fit_result

    expected = f"[{OVERRIDDEN}: {OVERRIDE_VALUE}]"

    persisted = [
        fold.get_native_model().model_to_string() for fold in restored.fit_result.models
    ]
    assert persisted, "the artifact carries no CV boosters"
    for index, text in enumerate(persisted):
        assert expected in text, f"CV booster {index} was saved with other params"

    refit = restored._refit_result.model.get_native_model().model_to_string()
    assert expected in refit, (
        "the refit model predict() uses was saved with other params"
    )


def test_the_override_does_not_survive_a_load(tmp_path: pathlib.Path) -> None:
    """`fit(params=)` is documented as applying to that call only.

    So a model restored from the artifact must re-fit on the config's values.
    That is the same contract ``test_the_override_does_not_outlive_the_call``
    pins in process, asserted across the artifact boundary, where it could
    plausibly have been persisted instead.
    """
    out = tmp_path / "artifact"
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    df = make_binary_df(n=160)
    model = Model(cfg, data=df)
    model.fit(params={OVERRIDDEN: OVERRIDE_VALUE})
    model.export(out)

    restored = Model.load(out)
    assert restored._cfg.model.params[OVERRIDDEN] == CONFIG_VALUE

    with record_lightgbm_calls() as seen:
        restored.fit(data=df)
    values = {call.get(OVERRIDDEN) for call in seen["train_params"]}
    assert values == {CONFIG_VALUE}, (
        f"a re-fit after load trained at {values}; the override was persisted "
        "when it is documented as applying to one call"
    )


def test_export_code_generates_the_overridden_value(tmp_path: pathlib.Path) -> None:
    """The generated project must reproduce the model that was fitted.

    Asserted on the generated ``config.json`` rather than on ``train.py``: the
    template carries a fixed example line mentioning ``learning_rate`` that is
    identical whether or not an override was passed, so matching the source text
    would pass for the wrong reason.
    """
    out = tmp_path / "generated"
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=160))
    model.fit(params={OVERRIDDEN: OVERRIDE_VALUE})
    model.export_code(out)

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    # The generated project keeps the resolved estimator parameters under
    # `lgbm_params`; `model.params` is the input shape, not the emitted one.
    assert config["lgbm_params"][OVERRIDDEN] == OVERRIDE_VALUE, config["lgbm_params"]

    boosters = list(out.rglob("model.txt"))
    assert len(boosters) == 1, f"expected one exported booster, found {boosters}"
    assert f"[{OVERRIDDEN}: {OVERRIDE_VALUE}]" in boosters[0].read_text(
        encoding="utf-8"
    )


def test_export_code_without_an_override_carries_the_config_value(
    tmp_path: pathlib.Path,
) -> None:
    """The control, without which the test above passes on a constant."""
    out = tmp_path / "generated"
    cfg = make_config("binary", n_estimators=5, n_splits=2, learning_rate=CONFIG_VALUE)
    model = Model(cfg, data=make_binary_df(n=160))
    model.fit()
    model.export_code(out)

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    # The generated project keeps the resolved estimator parameters under
    # `lgbm_params`; `model.params` is the input shape, not the emitted one.
    assert config["lgbm_params"][OVERRIDDEN] == CONFIG_VALUE, config["lgbm_params"]


# ---------------------------------------------------------------------------
# Every artifact-reading test must fail when nothing is written
# ---------------------------------------------------------------------------
# Round 7 found one test that passed with `Model.export` replaced by a no-op:
# it asserted on the model in memory and never read what was written. The fix
# was made for that one test. The rounds 6-7 monitor named the repair -- assert
# the property over **every** test that claims to read the artifact, not over
# the one that was caught -- which is what stops the next round finding the
# second instance of a class already found.


def _artifact_reading_tests() -> list[str]:
    """Tests whose body reads an exported artifact, found by reading them.

    Derived rather than listed: a test added later that loads an artifact joins
    this population without anyone remembering to add it, which is the whole
    reason the population is not a constant here.
    """
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        body = ast.get_source_segment(source, node) or ""
        if any(
            marker in body
            for marker in ("Model.load(", "rglob(", 'read_text(encoding="utf-8")')
        ):
            found.append(node.name)
    return sorted(found)


def test_artifact_reading_tests_fail_when_nothing_is_exported(
    tmp_path: pathlib.Path,
) -> None:
    """Substitute the export and every one of them must notice.

    A test that still passes is not reading the artifact, whatever its name
    says. This is the property round 7 used to expose one such test, applied to
    the population instead of to the instance.
    """
    names = _artifact_reading_tests()
    assert names, "no artifact-reading test was found; the scan has gone blind"

    still_green: list[str] = []
    for name in names:
        target = globals()[name]
        # Both writers, because a test reads whichever one it called: the first
        # run of this instrument patched only `export` and reported the
        # `export_code` test as not reading its artifact, when in fact the
        # substitution had missed it. An instrument that names the wrong test is
        # the same defect as a test that checks nothing.
        with (
            mock.patch.object(
                Model, "export", autospec=True, return_value=tmp_path / "never-written"
            ),
            mock.patch.object(
                Model,
                "export_code",
                autospec=True,
                return_value=tmp_path / "never-written",
            ),
        ):
            try:
                target(tmp_path / f"probe-{name}")
            except Exception:  # noqa: BLE001 - failing is the expected outcome
                continue
            still_green.append(name)

    assert not still_green, (
        f"these tests passed with export replaced by a no-op, so they are not "
        f"reading the artifact: {still_green}"
    )


def test_the_artifact_test_population_is_not_empty_by_accident() -> None:
    """The scan must find the tests it is named for.

    A scan that silently matches nothing would make the assertion above pass
    vacuously -- the failure this PR has been fixing, in the instrument built to
    prevent it.
    """
    names = _artifact_reading_tests()
    assert "test_the_override_reaches_the_exported_booster" in names, names
    assert "test_the_override_does_not_survive_a_load" in names, names
    assert "test_export_code_generates_the_overridden_value" in names, names
    assert len(names) >= 3, names
