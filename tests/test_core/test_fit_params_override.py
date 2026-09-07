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
import pathlib
from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuningResult
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


def _fit_with(smart: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    cfg["model"].update(smart)
    model = Model(cfg, data=make_binary_df(n=160))
    with record_lightgbm_calls() as seen:
        model.fit(params=params)
    return list(seen["train_params"])


@pytest.mark.parametrize("native", sorted(MANAGED_CASES))
def test_a_managed_name_is_refused_rather_than_replaced(native: str) -> None:
    """Accepting a value that is then discarded is the defect, not the fix."""
    case = MANAGED_CASES[native]
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    cfg["model"].update(case["enable"])
    model = Model(cfg, data=make_binary_df(n=160))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit(params={native: case["value"]})

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    message = str(exc.value)
    assert native in message and case["smart"] in message, (
        "the refusal must name both the parameter and the smart parameter that "
        f"manages it; got: {message}"
    )
    assert not seen["train_params"], (
        f"{len(seen['train_params'])} Booster(s) were trained before the refusal"
    )


@pytest.mark.parametrize("native", sorted(MANAGED_CASES))
def test_the_same_name_applies_once_its_smart_parameter_is_off(native: str) -> None:
    """The other direction, and the reason the table is not merely a list.

    Each entry claims "an active smart parameter overwrites this name". Switch
    that smart parameter off and the very same override must reach ``lgb.train``
    untouched. Without this, the table could name anything at all and every
    refusal above would still pass.
    """
    case = MANAGED_CASES[native]
    calls = _fit_with(case["disable"], {native: case["value"]})

    assert calls, "no lgb.train call was recorded"
    got = [call.get(native) for call in calls]
    assert all(value == case["value"] for value in got), (
        f"with {case['smart']} disabled, {native} should reach lgb.train as "
        f"{case['value']!r}; it arrived as {got}"
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
