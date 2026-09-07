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

import contextlib
import inspect
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
from lizyml.estimators.lgbm.smart_params import (
    SMART_PARAM_TARGETS,
    resolve_ratio_params,
    resolve_smart_params,
)
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


def test_tuning_evaluates_the_parameters_it_then_selects() -> None:
    """A trial must train on the value the study records for it.

    The trial merge was the fourth seam and the only one still merging by
    spelling. A config `learning_rate` and a search dimension named `eta` are
    one parameter to LightGBM, so a plain dict merge kept both and the library
    preferred the canonical one: **the trials trained at the config's value
    while the study recorded the trial's**, and the fit afterwards used the
    recorded one. Tuning selected a model it had never evaluated (review round
    11).
    """
    cfg = make_config(
        "binary",
        n_estimators=3,
        n_splits=2,
        learning_rate=CONFIG_VALUE,
        tuning_n_trials=1,
        num_threads=1,
    )
    cfg["tuning"]["optuna"]["space"] = {
        "eta": {
            "type": "categorical",
            "choices": [OVERRIDE_VALUE],
            "category": "model",
        }
    }
    model = Model(cfg, data=make_binary_df(n=120))

    def rates(calls: list[dict[str, Any]]) -> set[Any]:
        # Under whichever spelling reached the library: the overlay keeps the
        # spelling that was written last, and LightGBM resolves the alias.
        spellings = accepted_spellings(OVERRIDDEN)
        return {
            value for call in calls for name, value in call.items() if name in spellings
        }

    with record_lightgbm_calls() as seen:
        result = model.tune()
    tuned = rates(seen["train_params"])

    assert result.best_model_params == {"eta": OVERRIDE_VALUE}
    assert tuned == {OVERRIDE_VALUE}, (
        f"the trials trained at {tuned} while the study recorded "
        f"{result.best_model_params}; tuning selected a model it never "
        "evaluated"
    )

    with record_lightgbm_calls() as seen:
        model.fit()
    fitted = rates(seen["train_params"])
    assert fitted == tuned, (
        f"the fit trained at {fitted} and the trials at {tuned}; the selected "
        "model is not the one the fit reproduces"
    )


def test_two_spellings_in_the_config_are_refused_before_training() -> None:
    """The same-layer rule applies to the layer it was declared for.

    H-0094 decision 6 states that one parameter written twice under two
    spellings with different values is refused. Until review round 11 only
    `fit(params=)` was checked, so a config carrying both `learning_rate` and
    `eta` sent both to `lgb.train`, which silently kept the canonical one -- the
    exact shape the rule exists to prevent, on the layer most callers use.
    """
    cfg = make_config(
        "binary",
        n_estimators=3,
        n_splits=2,
        learning_rate=CONFIG_VALUE,
        eta=OVERRIDE_VALUE,
    )
    model = Model(cfg, data=make_binary_df(n=120))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit()

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "model.params" in exc.value.user_message, exc.value.user_message
    assert not seen["train_params"], (
        f"{len(seen['train_params'])} Booster(s) were trained before the "
        "refusal; the check must fire before any training"
    )


def test_two_spellings_in_calibration_params_are_refused_before_training() -> None:
    """The fourth layer, which had a name check and no identity check.

    `calibration.params` reaches the calibrator's `lgbm.train` and is a layer
    like any other, so the same-layer rule applies to it. It did not: measured
    before this, `{"learning_rate": 0.001, "eta": 0.5}` sent **both** spellings
    to the calibrator and LightGBM kept the canonical one in silence. Found by
    the rounds 10-11 monitor asking which layers the rule was wired to.
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    cfg["calibration"] = {
        "method": "isotonic",
        "params": {OVERRIDDEN: CONFIG_VALUE, "eta": OVERRIDE_VALUE},
    }

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        Model(cfg, data=make_binary_df(n=160)).fit()

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "calibration.params" in exc.value.user_message, exc.value.user_message
    assert not seen["train_params"], (
        f"{len(seen['train_params'])} Booster(s) were trained before the "
        "refusal; the check must fire before any training"
    )


def test_two_spellings_of_one_value_in_calibration_params_are_accepted() -> None:
    """The other direction, so the fourth layer is not refusing everything."""
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    cfg["calibration"] = {
        "method": "isotonic",
        "params": {OVERRIDDEN: OVERRIDE_VALUE, "eta": OVERRIDE_VALUE},
    }

    with record_lightgbm_calls() as seen:
        Model(cfg, data=make_binary_df(n=160)).fit()

    assert seen["train_params"], "the call was refused, or nothing trained"


def test_two_search_dimensions_naming_one_parameter_are_refused() -> None:
    """The same-layer rule on the layer decision 6 had not reached.

    `sample_params` writes one key per dimension, so `learning_rate` and `eta`
    as two dimensions both land in every trial dict. LightGBM resolves them to
    one parameter and keeps the canonical spelling, so measured before this
    check: every trial trained at the `learning_rate` value, `eta` was sampled
    and optimised over without affecting anything, and `best_model_params`
    recorded both -- so the `fit` afterwards carried the dead spelling too.

    There is no equal-values escape here, unlike the dict surfaces: two
    dimensions sample independently.
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2, tuning_n_trials=3)
    cfg["tuning"]["optuna"]["space"] = {
        OVERRIDDEN: {
            "type": "float",
            "low": 0.001,
            "high": 0.01,
            "category": "model",
        },
        "eta": {"type": "float", "low": 0.4, "high": 0.5, "category": "model"},
    }

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        Model(cfg, data=make_binary_df(n=160)).tune()

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "tuning.optuna.space" in exc.value.user_message, exc.value.user_message
    assert not seen["train_params"], (
        f"{len(seen['train_params'])} Booster(s) were trained before the "
        "refusal; the check must fire before the study starts"
    )


def test_two_search_dimensions_of_different_parameters_are_accepted() -> None:
    """The other direction, so the space check is not refusing every study."""
    cfg = make_config("binary", n_estimators=3, n_splits=2, tuning_n_trials=2)
    cfg["tuning"]["optuna"]["space"] = {
        OVERRIDDEN: {
            "type": "float",
            "low": 0.001,
            "high": 0.01,
            "category": "model",
        },
        "lambda_l2": {
            "type": "float",
            "low": 0.1,
            "high": 1.0,
            "category": "model",
        },
    }

    with record_lightgbm_calls() as seen:
        Model(cfg, data=make_binary_df(n=160)).tune()

    assert seen["train_params"], "the study was refused, or nothing trained"


@pytest.mark.parametrize("surface", ["model.params", "fit(params=)"])
def test_one_sequence_written_in_two_containers_is_not_refused(surface: str) -> None:
    """A false refusal on ordinary input, one container past round 11's.

    ``feature_contri`` and ``feature_penalty`` are one LightGBM parameter, so
    writing both is the same-layer case the identity refusal exists for --
    correctly, when the values differ. Here they do not: ``np.array([1., 2.])``
    and ``(1., 2.)`` produce the byte-identical booster. The refusal fired
    anyway, on both surfaces, with nothing trained (H-0094 decision 8, review
    round 12).

    Each spelling alone is asserted to train first, so a failure distinguishes
    "the pair is refused" from "the parameter is unusable here".
    """
    combined = {"feature_contri": np.array([1.0, 2.0]), "feature_penalty": (1.0, 2.0)}
    for written in ({k: v} for k, v in combined.items()), (combined,):
        for params in written:
            cfg = make_config("binary", n_estimators=3, n_splits=2)
            if surface == "model.params":
                cfg["model"]["params"].update(params)
            with record_lightgbm_calls() as seen:
                Model(cfg, data=make_binary_df(n=160)).fit(
                    params=dict(params) if surface == "fit(params=)" else None
                )
            assert seen["train_params"], (
                f"{sorted(params)} on {surface} trained nothing"
            )
            reached = seen["train_params"][0]
            assert any(name in reached for name in params), (
                f"{sorted(params)} did not reach lgb.train: {sorted(reached)}"
            )


@pytest.mark.parametrize("alias", ["eta", "shrinkage_rate", "learning_rate"])
def test_a_calibration_alias_reaches_the_calibrator(alias: str) -> None:
    """The override must not be defeated by the default it was written over.

    The calibrator merges ``calibration.params`` over its own defaults **by
    spelling**, and those defaults are canonical. So ``{"eta": 0.5}`` passed the
    name check and the identity check -- the caller wrote the parameter once --
    and then arrived at ``lgbm.train`` beside the default ``learning_rate:
    0.03``, which LightGBM preferred. Measured before the fix: the
    ``learning_rate`` spelling trained at 0.5 and every alias trained at 0.03
    (H-0094 decision 8, review round 12).

    ``learning_rate`` itself is in the parametrisation because a test that only
    exercises the aliases cannot show the two spellings now agree.
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    cfg["calibration"] = {"method": "isotonic", "params": {alias: OVERRIDE_VALUE}}

    with record_lightgbm_calls() as seen:
        Model(cfg, data=make_binary_df(n=160)).fit()

    calibrator_calls = [
        call for call in seen["train_params"] if call.get("monotone_constraints") == [1]
    ]
    assert calibrator_calls, "no calibrator Booster was trained"
    for call in calibrator_calls:
        assert call.get("learning_rate") == OVERRIDE_VALUE, (
            f"the calibrator trained at {call.get('learning_rate')!r} while "
            f"'{alias}' asked for {OVERRIDE_VALUE!r}"
        )
        assert not any(
            spelling in call
            for spelling in accepted_spellings(OVERRIDDEN)
            if spelling != OVERRIDDEN
        ), f"two spellings of one parameter reached lgb.train: {call!r}"


@pytest.mark.parametrize(
    "forced,spelling",
    [
        ("monotone_constraints", "monotone_constraints"),
        ("monotone_constraints", "monotone_constraint"),
        ("verbosity", "verbosity"),
        ("verbosity", "verbose"),
    ],
)
def test_a_calibration_parameter_the_calibrator_forces_reaches_it_once(
    forced: str, spelling: str
) -> None:
    """One spelling of a forced parameter reaches lgb.train, whatever was written.

    The calibrator forces `monotone_constraints` and `verbosity` after merging
    the caller's dict, and it cannot see aliases -- `lizyml/calibration/` may
    not import `lizyml/estimators/`. The facade canonicalises first, so the
    force lands on the same key the caller's value did and overwrites it,
    instead of both reaching LightGBM and the estimator's preference deciding
    (H-0094 decision 8, review round 12).
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    cfg["calibration"] = {"method": "isotonic", "params": {spelling: 99}}

    with record_lightgbm_calls() as seen:
        Model(cfg, data=make_binary_df(n=160)).fit()

    calibrator_calls = [
        call for call in seen["train_params"] if call.get("monotone_constraints")
    ]
    assert calibrator_calls, "no calibrator Booster was trained"
    for call in calibrator_calls:
        reached = {s: call[s] for s in accepted_spellings(forced) if s in call}
        assert list(reached) == [forced], (
            f"'{spelling}' left {sorted(reached)} for lgb.train; which one "
            "applies would then be LightGBM's choice rather than the code's"
        )
        assert reached[forced] != 99, (
            f"'{spelling}' overrode a value the calibrator forces"
        )


def test_every_calibration_alias_is_canonical_before_the_defaults_merge() -> None:
    """The property, quantified over the calibrator's own defaults.

    The defect above is not about ``learning_rate``: it is available for
    **every** default the calibrator carries, because the merge is by spelling
    and the caller may write any of them under an alias. Checking one name
    would leave the rest to the next round, so this asks it of all of them.

    ``CALIBRATOR_OWN_PARAM_NAMES`` are excluded deliberately and asserted to be
    excluded: ``num_boost_round`` is a LightGBM alias of ``num_iterations``, and
    renaming it would take the key the calibrator pops for its boosting rounds.
    """
    from lizyml.calibration.isotonic import (
        _ISOTONIC_DEFAULTS,
        CALIBRATOR_OWN_PARAM_NAMES,
    )
    from lizyml.core._model_factories import canonicalise_calibration_params

    canonical = LGBMProvider().canonical_param_names(_ISOTONIC_DEFAULTS)
    checked = 0
    for default_name in _ISOTONIC_DEFAULTS:
        if default_name in CALIBRATOR_OWN_PARAM_NAMES:
            continue
        for spelling in accepted_spellings(canonical[default_name]):
            written = canonicalise_calibration_params({spelling: "sentinel"})
            assert written == {canonical[default_name]: "sentinel"}, (
                f"'{spelling}' reaches the calibrator as {list(written)}, so it "
                f"would sit beside the default '{default_name}' instead of "
                "replacing it"
            )
            checked += 1
    assert checked > len(_ISOTONIC_DEFAULTS), (
        "the population collapsed to one spelling per default; the aliases are "
        "the whole point of the check"
    )

    for own in CALIBRATOR_OWN_PARAM_NAMES:
        assert canonicalise_calibration_params({own: 7}) == {own: 7}, (
            f"'{own}' is the calibrator's own key and was renamed, which takes "
            "it away from the code that pops it"
        )


def test_no_smart_parameter_name_has_an_estimator_alias() -> None:
    """Why the smart layer is merged by spelling and needs no identity overlay.

    Every other layer merges by parameter identity because LightGBM resolves
    aliases. The smart layer does not, and this is the reason rather than an
    oversight: smart parameter names are LizyML's own and the library has never
    heard of them, so there is no second spelling for one of them to arrive
    under. Asserted rather than assumed, because "no aliases" is exactly the
    kind of claim that goes stale when a name is added.
    """
    aliased = {}
    for name in LGBMProvider().smart_param_names():
        try:
            spellings = accepted_spellings(name)
        except Exception:  # noqa: BLE001 - not a canonical LightGBM name at all
            continue
        if spellings - {name}:
            aliased[name] = sorted(spellings - {name})

    assert not aliased, (
        f"these smart parameters now have estimator aliases: {aliased}. The "
        "smart layer is merged by spelling, so it would keep both and the "
        "resolver would read whichever it names."
    )


def test_two_spellings_of_one_value_in_the_config_are_accepted() -> None:
    """The other direction, so the refusal is not bought by refusing everything.

    Writing one parameter twice with the *same* value names one thing twice;
    there is no ambiguity for LightGBM to resolve, so nothing is refused.
    """
    cfg = make_config(
        "binary",
        n_estimators=3,
        n_splits=2,
        learning_rate=OVERRIDE_VALUE,
        eta=OVERRIDE_VALUE,
    )
    model = Model(cfg, data=make_binary_df(n=120))

    with record_lightgbm_calls() as seen:
        model.fit()

    values = {call.get(OVERRIDDEN) for call in seen["train_params"]}
    assert values == {OVERRIDE_VALUE}, values


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


#: The tasks smart resolution runs under. Not "every task LizyML has" -- these
#: are the three the resolvers branch on.
SMART_TASKS: tuple[str, ...] = ("binary", "multiclass", "regression")

#: The feature the ``feature_weights`` activation names. It has to exist in the
#: frame, or the resolver refuses; naming a fictional feature and rewriting it
#: inside the observer meant the declared value was never the one used.
WEIGHTED_FEATURE = "feat_a"

#: A value that switches each smart parameter on. The keys are asserted below
#: against the provider's own list, so a new smart parameter fails here for
#: having no activation rather than passing unobserved, and every value is
#: asserted to make a difference, so an activation cannot go inert.
SMART_ACTIVATIONS: dict[str, Any] = {
    "auto_num_leaves": True,
    "num_leaves_ratio": 0.25,
    "min_data_in_leaf_ratio": 0.01,
    "min_data_in_bin_ratio": 0.01,
    "feature_weights": {WEIGHTED_FEATURE: 2.0},
    "balanced": True,
}

#: What a smart parameter needs switched on before it does anything.
#: ``num_leaves_ratio`` is read only inside the ``auto_num_leaves`` branch, so
#: activating it alone exercised a combination the resolver never reaches --
#: dead in practice, and the observation said nothing about it (review round 10).
SMART_PREREQUISITES: dict[str, dict[str, Any]] = {
    "num_leaves_ratio": {"auto_num_leaves": True},
}


def _resolve_smart(smart: dict[str, Any], task: str) -> dict[str, Any]:
    """Both resolvers, the way the training path calls them.

    ``resolve_ratio_params`` is a separate entry point called per fold, so a
    parameter that goes through it is invisible to the other one. Running both
    puts every smart parameter through the same door here.
    """
    frame = make_binary_df(n=40)
    feature_names = [column for column in frame.columns if column != "target"]
    target = frame["target"] if task != "regression" else frame["target"] * 1.0

    resolved, _ = resolve_smart_params(
        smart=smart,
        effective_params={"max_depth": 5},
        n_rows=len(frame),
        feature_names=feature_names,
        y=target,
        task=task,
    )
    resolved.update(
        resolve_ratio_params(
            min_data_in_leaf_ratio=smart.get("min_data_in_leaf_ratio"),
            min_data_in_bin_ratio=smart.get("min_data_in_bin_ratio"),
            n_rows=1000,
        )
    )
    return resolved


def _smart_input(name: str | None) -> dict[str, Any]:
    """One smart parameter switched on, with whatever it needs beneath it.

    ``balanced`` is switched off unless it is the subject: it defaults to on for
    the classification tasks, so otherwise every case would write
    ``scale_pos_weight`` and the individual activations would say nothing.
    """
    smart: dict[str, Any] = {"balanced": False}
    if name is None:
        return smart
    smart.update(SMART_PREREQUISITES.get(name, {}))
    smart[name] = SMART_ACTIVATIONS[name]
    return smart


def _names_the_resolvers_write() -> set[str]:
    """Run the resolvers and record what they actually put in the dict.

    Executed, not parsed. The previous form walked the source for
    ``resolved["<literal>"] = ...`` inside two named functions, which is a
    hypothesis about how an assignment is spelled: a fourth native name written
    through ``resolved.update({...})`` was invisible to it and the test still
    passed (rounds 8-9 monitor). Running the code has no spelling to guess.

    **This is a bounded set of executions, not a closed input domain.** Every
    declared smart parameter is run on its own with its prerequisites, and all
    of them are run together, across the three tasks. Enumerating the parameter
    *names* does not enumerate the combinations the resolvers accept, and saying
    it did was the overclaim round 10 found. What it does establish is that a
    native name written under any of these executions is declared.
    """
    written: set[str] = set()
    inputs = [_smart_input(name) for name in SMART_ACTIVATIONS]
    everything = {"balanced": SMART_ACTIVATIONS["balanced"]}
    for name in SMART_ACTIVATIONS:
        everything.update(SMART_PREREQUISITES.get(name, {}))
        everything[name] = SMART_ACTIVATIONS[name]
    inputs.append(everything)

    for task in SMART_TASKS:
        for smart in inputs:
            if smart.get("balanced") and task == "regression":
                # The one refusal this observation expects. Every other
                # `LizyMLError` is a failure, not a case to skip.
                with pytest.raises(LizyMLError):
                    _resolve_smart(smart, task)
                continue
            written |= set(_resolve_smart(smart, task))
    return written


def test_every_smart_parameter_has_an_activation() -> None:
    """Every smart parameter the provider declares is one this observation runs.

    A new smart parameter with no activation would be observed writing nothing,
    and the table would agree with a measurement that never took place.
    """
    assert set(SMART_ACTIVATIONS) == LGBMProvider().smart_param_names(), (
        f"declared activations {sorted(SMART_ACTIVATIONS)} do not match the "
        f"provider's smart parameters {sorted(LGBMProvider().smart_param_names())}"
    )
    assert set(SMART_PREREQUISITES) <= set(SMART_ACTIVATIONS), SMART_PREREQUISITES


@pytest.mark.parametrize("name", sorted(SMART_ACTIVATIONS))
def test_every_activation_changes_what_the_resolvers_produce(name: str) -> None:
    """An activation that does nothing observes nothing, and says so quietly.

    Two ways that happened at once (review round 10): ``feature_weights`` named
    a feature the frame does not have, so the observer silently substituted its
    own value and the declared one was never used; and ``num_leaves_ratio`` was
    supplied without ``auto_num_leaves``, so the resolver never read it. Setting
    either to ``None`` left both closure tests green.

    Compared against the same input **with its prerequisites but without the
    parameter itself**, so the difference is the parameter and nothing else.
    """
    baseline_input = _smart_input(None)
    baseline_input.update(SMART_PREREQUISITES.get(name, {}))
    activated = _smart_input(name)

    differences = []
    for task in SMART_TASKS:
        if activated.get("balanced") and task == "regression":
            continue
        baseline = _resolve_smart(baseline_input, task)
        with_it = _resolve_smart(activated, task)
        if baseline != with_it:
            differences.append(task)

    assert differences, (
        f"activating {name!r} with {SMART_ACTIVATIONS[name]!r} produced exactly "
        "what leaving it out produced, on every task. The observation that "
        "backs SMART_PARAM_TARGETS therefore never exercised it."
    )


def test_the_managed_table_matches_the_names_the_resolvers_write() -> None:
    """Close the table against what running the resolvers produces.

    ``SMART_PARAM_TARGETS`` claims to name every native parameter smart
    resolution writes. That claim would go stale the day a smart parameter
    learns to write a fourth one, and nothing else would notice: the refusal
    would simply not fire and the override would be silently replaced again.
    """
    written = _names_the_resolvers_write()
    declared = {name for names in SMART_PARAM_TARGETS.values() for name in names}

    assert written, "running the resolvers produced no native parameter at all"
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
#: Factories, not values. Binding one object under both spellings made every
#: case resolve at the identity step and reach nothing else, so the claim "equal
#: values under two spellings are accepted whatever the type" was carried by six
#: cases that never compared anything (found by the rounds 7-8 monitor).
AWKWARD_VALUES: dict[str, Any] = {
    "numpy array": lambda: np.array([1.0, 2.0]),
    "list": lambda: [1.0, 2.0],
    # Built rather than written as a literal: CPython folds a constant tuple
    # into the code object, so two calls returned the same object and the case
    # went no further than the identity step.
    "tuple": lambda: tuple([1.0, 2.0]),
    "none": lambda: None,
    "bool": lambda: True,
    "empty list": lambda: [],
}

#: The labels whose value is a singleton, where two calls cannot produce two
#: objects. Named rather than skipped, so the exception is visible.
SINGLETON_VALUES = frozenset({"none", "bool"})


@pytest.mark.parametrize("label", sorted(AWKWARD_VALUES))
def test_a_single_value_of_any_shape_passes_the_duplicate_refusal(label: str) -> None:
    """One spelling cannot be a duplicate, whatever the value is.

    The refusal compared each value against the group's first -- itself, when
    the group has one member -- so an array value raised ``ValueError`` on a
    call that named nothing twice.
    """
    check_duplicate_identities(
        LGBMProvider(), {"feature_contri": AWKWARD_VALUES[label]()}, surface="probe"
    )


@pytest.mark.parametrize("label", sorted(AWKWARD_VALUES))
def test_equal_values_of_any_shape_are_accepted_under_two_spellings(
    label: str,
) -> None:
    """And two spellings of an equal value are the same value.

    Two *objects*, built separately. One object bound under both keys is
    answered by the identity step and reaches nothing else, so the version of
    this test that did that asserted only that ``x is x`` (rounds 7-8 monitor).
    """
    first, second = AWKWARD_VALUES[label](), AWKWARD_VALUES[label]()
    if label not in SINGLETON_VALUES:
        assert first is not second, (
            f"{label} produced one object twice, so this case cannot reach "
            "past the identity step"
        )

    provider = LGBMProvider()
    check_duplicate_identities(
        provider,
        {"feature_contri": first, "feature_contrib": second},
        surface="probe",
    )
    kept, spelling = _pop_by_identity(
        {"feature_contri": first, "feature_contrib": second}, "feature_contri"
    )
    assert kept is first and spelling == "feature_contri"


def test_equal_arrays_of_different_dtypes_are_one_value() -> None:
    """Through the production entrypoint, because that is where it was refused.

    `np.array([1, 2])` and `np.array([1.0, 2.0])` are the same value and each
    trains on its own. Deciding array-likes by their printed forms made them
    two values, so naming one parameter twice with them was refused with
    `CONFIG_INVALID` -- a gate refusing legitimate input, on ordinary arrays
    rather than adversarial objects (review round 11).
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=120))

    with record_lightgbm_calls() as seen:
        model.fit(
            params={
                "feature_contri": np.array([1, 2]),
                "feature_contrib": np.array([1.0, 2.0]),
            }
        )

    assert seen["train_params"], "the call was refused, or nothing trained"


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
# Every test that exports must fail when nothing is written
# ---------------------------------------------------------------------------
# Round 7 found one test that passed with `Model.export` replaced by a no-op:
# it asserted on the model in memory and never read what was written. The fix
# was made for that one test. The rounds 6-7 monitor named the repair -- assert
# the property over **every** test that claims something about the artifact,
# not over the one that was caught.
#
# Rounds 8 and 9 then found that "every" was a claim no scanner can keep. A
# source scan looking for `model.export(...)` was defeated by binding the
# attribute, then by `getattr(model, "export")`, then by `getattr(model, name)`
# -- and `operator.methodcaller`, `functools.partial` and `Model.__dict__` are
# next. Python's dispatch is an open grammar, so the completeness claim was
# always a hypothesis about spelling and each round refuted one more spelling.
#
# So the scanner is deleted rather than taught a fourth form. The population is
# named here, by hand, and says so. A bounded claim that is honestly stated
# cannot be reproduced against; an unbounded one was reproduced against three
# times. What it costs is real and is not hidden: a new exporting test has to be
# added to this tuple by hand, and nothing detects a failure to do that.

#: The tests whose subject is the exported artifact. Hand-maintained.
ARTIFACT_TESTS: tuple[str, ...] = (
    "test_the_override_reaches_the_exported_booster",
    "test_the_override_does_not_survive_a_load",
    "test_export_code_generates_the_overridden_value",
    "test_export_code_without_an_override_carries_the_config_value",
)

#: Writer method -> the function beneath it that actually touches the disk.
#: The substitution replaces the *writing*, not the method, so the method still
#: runs and still returns the path it resolved. Substituting the method changed
#: its return value too, and a target asserting only on that return failed for a
#: reason that had nothing to do with the artifact (review round 9).
ARTIFACT_WRITERS: dict[str, str] = {
    "export": "lizyml.persistence.exporter.export",
    "export_code": "lizyml.codegen.generator.generate_code",
}


def _declared_writers() -> frozenset[str]:
    """``Model``'s callable attributes whose name begins with ``export``.

    Asked of the **class**, not of its source. The source form matched
    ``ast.FunctionDef`` in one class body, so an ``async def``, an assignment,
    or a method inherited from elsewhere was absent from this side as well as
    from the map, the equality below held, and that writer was never
    substituted (rounds 8-9 monitor).

    The one assumption left is the name: a writer called something that does not
    begin with ``export`` is not covered here, and no check can find it, because
    "writes to disk" is not a property of a name. That is a stated limit, not a
    silence.
    """
    return frozenset(
        name
        for name in dir(Model)
        if name.startswith("export") and callable(getattr(Model, name, None))
    )


def _keep_the_output_path(*args: Any, **kwargs: Any) -> Any:
    """Return the directory the caller asked for, writing nothing."""
    return pathlib.Path(kwargs["output_dir"])


def _probe(target: Any, tmp_path: pathlib.Path, name: str) -> str:
    """Run one target twice and say what its behaviour under substitution means.

    A separate function because the three outcomes have to be testable on
    synthetic targets. An instrument whose own decision rule is only exercised
    by the four tests it happens to select is verified by a table again.

    ``failed-after-writing`` -- it failed, and the writing it depends on had
    been reached first.
    ``never-reached`` -- it failed before any writing, so its failure is evidence
    of nothing. Counting that as noticing would be a silent pass in the
    instrument written to catch silent passes (review round 8).
    ``green`` -- it passed, so it asserts nothing about what was written.

    **``failed-after-writing`` is not "it inspected the artifact", and the
    verdict is named for what was observed rather than for what it would be
    convenient to conclude.** Two invocations of the same target can differ in
    more than the writing -- in the path each is given, and in whatever state
    the first left behind -- so ordering alone establishes nothing; a target that
    reaches the exporter and then fails for its own reasons lands here (review
    round 10). What the verdict rules out is the shape this instrument exists
    for: a test that stays green when nothing is written. That the remaining
    difference is *only* the writing is what the preserved return contract buys,
    and it is a limit on this instrument, not a claim of it. The artifact
    assertions in the named tests are reviewed directly; this does not replace
    reading them.
    """
    # It must pass unpatched first. Without this control, a target that fails
    # for a reason of its own -- a broken fixture, an import error, a bug in
    # this loop -- would be read as having noticed the missing artifact.
    target(tmp_path / f"control-{name}")

    with contextlib.ExitStack() as stack:
        substituted = [
            stack.enter_context(
                mock.patch(where, side_effect=_keep_the_output_path)
                if where.endswith("generate_code")
                else mock.patch(where)
            )
            for where in sorted(ARTIFACT_WRITERS.values())
        ]
        try:
            target(tmp_path / f"probe-{name}")
        except Exception:  # noqa: BLE001 - failing is the expected outcome
            reached = any(writer.called for writer in substituted)
            return "failed-after-writing" if reached else "never-reached"
    return "green"


def test_every_exporting_test_fails_when_nothing_is_written(
    tmp_path: pathlib.Path,
) -> None:
    """Suppress the writing and every test in ``ARTIFACT_TESTS`` must notice.

    A test that still passes made no claim about what was written, whatever its
    name says. This is the property round 7 used to expose one such test,
    applied to the named population instead of to the instance.
    """
    # Refuse a shape this instrument cannot supply, instead of calling it and
    # reading the resulting `TypeError` as "the test noticed". That would be
    # DC1 -- couldn't run, counted as clean -- inside the instrument written to
    # catch DC1. A test that needs another fixture is a real gap, so it has to
    # be visible rather than absorbed.
    unsupported = {
        name: list(inspect.signature(globals()[name]).parameters)
        for name in ARTIFACT_TESTS
        if list(inspect.signature(globals()[name]).parameters) != ["tmp_path"]
    }
    assert not unsupported, (
        f"these exporting tests take arguments this instrument cannot provide, "
        f"so it cannot check them: {unsupported}"
    )

    verdicts = {
        name: _probe(globals()[name], tmp_path, name) for name in ARTIFACT_TESTS
    }
    still_green = [name for name, verdict in verdicts.items() if verdict == "green"]
    never_reached = [
        name for name, verdict in verdicts.items() if verdict == "never-reached"
    ]

    assert not never_reached, (
        f"these tests failed before any writing was reached, so their failure "
        f"says nothing about the artifact: {never_reached}"
    )
    assert not still_green, (
        f"these tests passed with the writing suppressed, so they assert "
        f"nothing about what was written: {still_green}"
    )


def test_the_named_population_names_tests_that_exist() -> None:
    """A stale name would drop a test from the check without a word.

    This is the whole guarantee the deleted scanner used to claim: it is smaller
    than that claim, and unlike it, it is true. Nothing here detects a *new*
    exporting test that was not added to the tuple -- that limit is stated where
    the tuple is defined.
    """
    assert ARTIFACT_TESTS, "the population is empty"
    assert len(set(ARTIFACT_TESTS)) == len(ARTIFACT_TESTS), ARTIFACT_TESTS
    for name in ARTIFACT_TESTS:
        assert name in globals(), f"{name} is named here but does not exist"


def test_every_declared_writer_has_a_substitution() -> None:
    """A writer added to the mixin must not be left running for real.

    The map is written by hand, but the *keys* are checked against the class
    that defines the methods, so a third writer fails here instead of quietly
    going unsuppressed while the instrument still reports clean.
    """
    assert set(ARTIFACT_WRITERS) == _declared_writers(), ARTIFACT_WRITERS
    assert all(hasattr(Model, writer) for writer in ARTIFACT_WRITERS)


# ---------------------------------------------------------------------------
# The instrument's own decision rule and grammar, on synthetic inputs
# ---------------------------------------------------------------------------
# Round 8 found two defects here: a target that failed before reaching either
# writer was counted as having noticed, and a writer named under an alias left
# the population without a word. Both are the classes this PR keeps fixing, in
# the instrument built to catch them, so both are pinned against inputs made
# for the purpose rather than against the four tests that happen to be selected.


def _synthetic_reader(path: pathlib.Path) -> None:
    """A target that reads what it wrote. Not a ``test_``, so not a population
    member -- the scan is about the tests under review, and this one exists to
    exercise the probe."""
    model = Model(
        make_config("binary", n_estimators=3, n_splits=2),
        data=make_binary_df(n=80),
    )
    model.fit()
    out = model.export(path)
    restored = Model.load(out)
    assert restored is not model and restored.fit_result is not model.fit_result


def _synthetic_in_memory_asserter(path: pathlib.Path) -> None:
    """A target that exports and then asserts on the model still in memory.

    Round 7's shape exactly, reproduced so the probe's ``green`` verdict is
    reached by something and not only declared.
    """
    model = Model(
        make_config("binary", n_estimators=3, n_splits=2),
        data=make_binary_df(n=80),
    )
    model.fit()
    model.export(path)
    assert model.fit_result is not None


def _synthetic_return_path_only(path: pathlib.Path) -> None:
    """A target that asserts only on the path the writer returned.

    It inspects no artifact, so it must be reported ``green``. Substituting the
    *method* rather than the writing changed this return value too, and the
    target failed for a reason that had nothing to do with the artifact and was
    reported as having noticed one (review round 9).
    """
    model = Model(
        make_config("binary", n_estimators=3, n_splits=2),
        data=make_binary_df(n=80),
    )
    model.fit()
    assert model.export(path) == path


def _synthetic_broken(path: pathlib.Path) -> None:
    """A target that fails before it could reach any writer."""
    raise RuntimeError("failed before any export")


def test_the_probe_tells_the_three_outcomes_apart(tmp_path: pathlib.Path) -> None:
    """Each verdict is reached by a target constructed to produce it."""
    assert _probe(_synthetic_reader, tmp_path, "reader") == "failed-after-writing"
    assert _probe(_synthetic_in_memory_asserter, tmp_path, "in-memory") == "green"
    assert _probe(_synthetic_return_path_only, tmp_path, "return-path") == "green"

    # This one fails unpatched too, so the control run refuses it before any
    # verdict is reached, which is the stronger of the two guards.
    with pytest.raises(RuntimeError, match="failed before any export"):
        _probe(_synthetic_broken, tmp_path, "broken")


def test_the_probe_will_not_read_a_failure_before_the_writer_as_noticing(
    tmp_path: pathlib.Path,
) -> None:
    """The other guard, on a target that passes unpatched and fails patched.

    The control run cannot catch this one: the target is healthy, and it fails
    under substitution for a reason that has nothing to do with the artifact.
    Only the record of whether a writer was reached separates the two.
    """
    calls = {"n": 0}

    def target(path: pathlib.Path) -> None:
        # The first call is the control and passes; the second runs under the
        # substitution and fails for a reason that never reaches a writer.
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("unrelated failure, no writer involved")

    assert _probe(target, tmp_path, "unrelated") == "never-reached"
    assert calls["n"] == 2
