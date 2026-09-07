"""``calibration.params`` names must be checked, not silently discarded (H-0093).

``IsotonicCalibrator`` merges ``cfg.calibration.params`` over its defaults and
hands the result to ``lgbm.train``. LightGBM drops a name it does not know
without raising, and the calibrator ships ``verbose=-1``, so ``{"num_leave":
7}`` trained with the default ``num_leaves`` and reported success. This is the
fourth route H-0093 covers; it was found by review after the first three were
gated, and ``test_the_calibrator_itself_forwards_an_unknown_name`` is the
negative control showing the gate is the only thing standing in front of it.

The other side of the gate matters as much: it must refuse nothing legitimate.
Every shipped default, every name the calibrator consumes itself, and every
calibrator that does not use LightGBM at all are checked here by execution.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

from lizyml import Model
from lizyml.calibration.isotonic import (
    _ISOTONIC_DEFAULTS,
    CALIBRATOR_OWN_PARAM_NAMES,
    IsotonicCalibrator,
)
from lizyml.core._model_factories import (
    LGBM_BACKED_CALIBRATORS,
    check_calibration_param_names,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.registries import CalibratorRegistry
from tests._ast_scan import lightgbm_bindings
from tests._helpers import make_binary_df, make_config

REPO = pathlib.Path(__file__).resolve().parents[2]

#: A name LightGBM has never accepted, one character from a real one. Anchored
#: so a substring match on `num_leaves` would not make the test pass anyway.
UNKNOWN_NAME = "num_leave"

#: A value each self-consumed name can plausibly take, so the calibrator can be
#: constructed with it and asked what it forwarded.
OWN_PARAM_VALUES: dict[str, Any] = {
    "num_boost_round": 7,
    "validation_ratio": 0.2,
    "min_data_in_leaf_ratio": 0.05,
}


class _TrainSpy:
    """Record the params dict every ``lgbm.train`` call receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import lizyml.calibration.isotonic as iso

        real = iso.lgbm.train

        def spy(params: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            self.calls.append(dict(params))
            return real(params, *args, **kwargs)

        monkeypatch.setattr(iso.lgbm, "train", spy)

    @property
    def names(self) -> set[str]:
        return {name for call in self.calls for name in call}


# ---------------------------------------------------------------------------
# The defect, and the gate that closes it
# ---------------------------------------------------------------------------


def test_the_calibrator_itself_forwards_an_unknown_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: without the gate, the name reaches LightGBM.

    If this ever stops holding -- because the calibrator grew its own check, or
    LightGBM started refusing unknown names -- the gate below is no longer the
    thing that protects this surface and the reason for it must be re-read.
    """
    spy = _TrainSpy()
    spy.install(monkeypatch)
    scores, y = _scores_and_labels()

    IsotonicCalibrator({UNKNOWN_NAME: 7}).fit(scores, y)

    assert spy.calls, "the calibrator did not reach lgbm.train at all"
    assert UNKNOWN_NAME in spy.names, (
        "the unknown name did not reach lgbm.train; the gate is no longer the "
        "only thing standing between calibration.params and the Booster"
    )


def test_unknown_calibration_param_is_refused_before_any_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped path refuses it, and **nothing is trained at all**.

    The second assertion is the load-bearing one, and it is not incidental
    strictness. H-0093 and ``BLUEPRINT.md`` §12.2 both state the check fires
    before training starts; while the call sat in ``_run_calibration`` that was
    false -- the whole outer CV ran first, measured at two ``lgb.train`` calls
    before the raise, and the config could never have produced a usable model.
    Asserting only that the bad name never reached LightGBM passed throughout,
    which is why the ordering is asserted separately.

    The spy patches the ``lightgbm`` module object, which the adapter and the
    calibrator both resolve at call time, so it sees every Booster either would
    train.
    """
    spy = _TrainSpy()
    spy.install(monkeypatch)
    df = make_binary_df(n=120)
    cfg = make_config(
        "binary",
        calibration="isotonic",
        calibration_n_splits=2,
        calibration_params={UNKNOWN_NAME: 7},
    )

    with pytest.raises(LizyMLError) as err:
        Model(cfg).fit(df)

    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert UNKNOWN_NAME in str(err.value)
    assert "calibration.params" in str(err.value)
    assert UNKNOWN_NAME not in spy.names
    assert spy.calls == [], (
        f"{len(spy.calls)} Booster(s) were trained before the config was "
        "refused; the check is downstream of training, and the specification "
        "says it fires before any training starts"
    )


#: The public methods that train a Booster, and how to invoke each.
#:
#: The check has now been missed on an entry point twice -- once by living at
#: construction time, once by living on the fit path only while ``tune()`` has
#: its own -- so the entry points are enumerated and each is exercised rather
#: than one standing in for the rest. ``predict`` and ``export_code`` are absent
#: deliberately: neither trains, so neither can train before a refusal.
TRAINING_ENTRY_POINTS: dict[str, Any] = {
    "fit": lambda model, df: model.fit(df),
    "tune": lambda model, df: model.tune(df),
}


@pytest.mark.parametrize("entry_point", sorted(TRAINING_ENTRY_POINTS))
def test_no_entry_point_trains_before_refusing(
    entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every way in must refuse first and train second.

    ``tune()`` was the second miss: it validates ``model.params`` and the search
    space itself and did not ask about ``calibration.params``, so a config that
    ``fit()`` would refuse completed an entire study first. Measured before the
    fix: ``calls_after_tune=2``.
    """
    spy = _TrainSpy()
    spy.install(monkeypatch)
    df = make_binary_df(n=120)
    cfg = make_config(
        "binary",
        calibration="isotonic",
        calibration_params={UNKNOWN_NAME: 7},
        tuning_n_trials=1,
    )

    with pytest.raises(LizyMLError) as err:
        TRAINING_ENTRY_POINTS[entry_point](Model(cfg), df)

    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert spy.calls == [], (
        f"Model.{entry_point}() trained {len(spy.calls)} Booster(s) before "
        "refusing a config it can never honour"
    )


def test_the_gate_names_the_surface_and_the_offender() -> None:
    """The message must be actionable without the user reading our source."""
    with pytest.raises(LizyMLError) as err:
        check_calibration_param_names(
            _CalibrationCfg("isotonic", {UNKNOWN_NAME: 7, "num_leaves": 7})
        )

    message = str(err.value)
    assert UNKNOWN_NAME in message
    assert "num_leaves" not in message, "a legitimate name was reported as unknown"
    context = err.value.context
    assert context["unknown"] == [
        {"surface": "calibration.params", "name": UNKNOWN_NAME}
    ]


# ---------------------------------------------------------------------------
# The gate must refuse nothing legitimate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_ISOTONIC_DEFAULTS))
def test_every_shipped_default_is_accepted(name: str) -> None:
    """A default the gate would refuse is a config nobody could write.

    The defaults are the names the calibrator itself considers legitimate, so
    they are the tightest available check that the accepted set is not too
    small -- and they drift independently of it.
    """
    check_calibration_param_names(
        _CalibrationCfg("isotonic", {name: _ISOTONIC_DEFAULTS[name]})
    )


@pytest.mark.parametrize("name", sorted(CALIBRATOR_OWN_PARAM_NAMES))
def test_self_consumed_names_are_accepted_and_never_forwarded(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each declared exception must really be consumed by the calibrator.

    ``CALIBRATOR_OWN_PARAM_NAMES`` widens what the gate accepts, so a name
    listed there but *not* popped would be admitted and then silently dropped
    by LightGBM -- reintroducing the defect through the exception list. This
    asserts on what ``lgbm.train`` actually received.
    """
    check_calibration_param_names(
        _CalibrationCfg("isotonic", {name: OWN_PARAM_VALUES[name]})
    )

    spy = _TrainSpy()
    spy.install(monkeypatch)
    scores, y = _scores_and_labels()
    IsotonicCalibrator({name: OWN_PARAM_VALUES[name]}).fit(scores, y)

    assert spy.calls
    assert name not in spy.names, (
        f"'{name}' is declared as consumed by the calibrator but reached "
        "lgbm.train, where LightGBM would discard it in silence"
    )


def test_seed_is_not_an_exception_because_it_reaches_the_booster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``seed`` is popped in ``__init__`` and then put straight back.

    It looks like a self-consumed name and is not one: ``merged["seed"]``
    reinstates it, and LightGBM knows the name, so the base registry accepts it
    already. This pins the reason it is absent from the exception list, since
    the ``pop`` alone reads as sufficient grounds to add it.
    """
    assert "seed" not in CALIBRATOR_OWN_PARAM_NAMES
    check_calibration_param_names(_CalibrationCfg("isotonic", {"seed": 5}))

    spy = _TrainSpy()
    spy.install(monkeypatch)
    scores, y = _scores_and_labels()
    IsotonicCalibrator({"seed": 5}).fit(scores, y)

    assert spy.calls[0]["seed"] == 5


def test_every_own_param_name_has_a_test_value() -> None:
    """The parametrization above must cover the declaration, not a subset."""
    assert set(OWN_PARAM_VALUES) == set(CALIBRATOR_OWN_PARAM_NAMES)


def test_a_calibrated_run_still_succeeds() -> None:
    """The gate is on the path of every calibrated fit: it must not break one."""
    df = make_binary_df(n=120)
    cfg = make_config(
        "binary",
        calibration="isotonic",
        calibration_n_splits=2,
        calibration_params={"num_leaves": 5, "num_boost_round": 20},
    )
    result = Model(cfg).fit(df)
    assert result.calibrator is not None


@pytest.mark.parametrize("method", ["platt", "beta"])
def test_calibrators_that_do_not_use_lightgbm_are_not_checked(method: str) -> None:
    """Their params are not LightGBM's, so LightGBM's registry cannot judge them.

    Note this says nothing good about those surfaces: ``PlattCalibrator``
    ignores ``params`` entirely. That is a separate defect on a separate route
    and is recorded as such; what is asserted here is only that this gate does
    not refuse them.
    """
    check_calibration_param_names(_CalibrationCfg(method, {UNKNOWN_NAME: 7}))


def test_no_calibration_and_no_params_are_no_ops() -> None:
    check_calibration_param_names(None)
    check_calibration_param_names(_CalibrationCfg("isotonic", None))
    check_calibration_param_names(_CalibrationCfg("isotonic", {}))


# ---------------------------------------------------------------------------
# The declaration of which calibrators use LightGBM is scanned, not asserted
# ---------------------------------------------------------------------------


def _calibrator_modules_binding_lightgbm() -> set[str]:
    """Registered calibrator names whose module imports LightGBM."""
    found: set[str] = set()
    # `.keys()` is this registry's own accessor returning a list, not a dict view.
    methods = CalibratorRegistry.keys()
    for method in methods:
        module = CalibratorRegistry.get(method).__module__
        path = REPO / (module.replace(".", "/") + ".py")
        bindings = lightgbm_bindings(ast.parse(path.read_text(encoding="utf-8")))
        if bindings.modules or bindings.attrs:
            found.add(method)
    return found


def test_lgbm_backed_calibrators_matches_the_scan() -> None:
    """A new LightGBM-backed calibrator must not arrive ungated.

    ``LGBM_BACKED_CALIBRATORS`` decides which methods the gate looks at. Left
    as prose it would be true only until someone adds a calibrator -- exactly
    how ``isotonic`` itself went ungated. The scan reads each registered
    calibrator's own imports, so both directions fail here: a module that binds
    LightGBM and is not declared, and a declared method whose module does not.
    """
    scanned = _calibrator_modules_binding_lightgbm()
    assert scanned == set(LGBM_BACKED_CALIBRATORS), (
        f"scanned {sorted(scanned)} but LGBM_BACKED_CALIBRATORS declares "
        f"{sorted(LGBM_BACKED_CALIBRATORS)}. A calibrator that reaches "
        "LightGBM with user parameters and is not declared here is ungated."
    )
    assert scanned, "the scan found no LightGBM-backed calibrator; it is not looking"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CalibrationCfg:
    """The two attributes the gate reads, without building a whole config."""

    def __init__(self, method: str, params: dict[str, Any] | None) -> None:
        self.method = method
        self.params = params


def _scores_and_labels() -> tuple[Any, Any]:
    import numpy as np

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(float)
    scores = y * 2.0 - 1.0 + rng.normal(0, 0.5, 200)
    return scores, y
