"""The parameter layer x refusal grid, as an executable declaration.

Rounds 11, 12 and 14 each found one empty cell of this grid, one round apart.
The rounds 13-14 monitor observed that a loop finding one empty cell per round
is either converging or mining a blind spot, and that enumerating the grid once
settles it either way. This file is that enumeration.

It is deliberately **not** a copy of what the code does: each `"wired"` cell is
reached by an executed case in `test_fit_params_override.py` or in
`test_calibration_param_names.py`, and the tests here assert that the table is
rectangular, that every open cell names an issue, and that no refusal exists in
`_model_factories` without a column here. A table nobody can lose a row from is
the difference between a declaration and a note.
"""

from __future__ import annotations

from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuningResult
from tests._helpers import make_binary_df, make_config
from tests._train_spy import record_lightgbm_calls

#: The grid this PR's defects came out of: **parameter layer x refusal**.
#:
#: Rounds 11, 12 and 14 each found one empty cell of this grid, one round apart,
#: and the rounds 13-14 monitor pointed out that a loop finding one empty cell
#: per round is either converging or mining a blind spot -- and that enumerating
#: the grid once ends the question either way. This is that enumeration, as an
#: executable declaration rather than the prose the last three rounds relied on.
#:
#: Each value is one of:
#:   ``"wired"``   -- the refusal is called for this layer, asserted by execution
#:                    below;
#:   ``"n/a: ..."``-- the refusal does not apply here, with the reason;
#:   ``"open: ..."``-- a known gap with an issue and a measured rate.
#:
#: A cell may not say ``"wired"`` unless a case in this file reaches it. The
#: point of the table is that a future reader can tell an empty cell from a
#: deliberate one without re-deriving either.
REFUSAL_MATRIX: dict[str, dict[str, str]] = {
    "model.params": {
        "check_param_values": "wired",
        "normalise_params": "wired",
        "check_param_names": "wired",
        "check_duplicate_identities": "wired",
        "check_training_managed_overrides": "wired",
        "check_smart_managed_overrides": "open: #280, 3/18 of the surface refused",
        "canonicalisation": "n/a: merged by identity, not by spelling",
    },
    "fit(params=)": {
        "check_param_values": "wired",
        "normalise_params": "wired",
        "check_param_names": "wired",
        "check_duplicate_identities": "wired",
        "check_training_managed_overrides": "wired",
        "check_smart_managed_overrides": "wired",
        "canonicalisation": "n/a: merged by identity, not by spelling",
    },
    "tuning best_model_params": {
        "check_param_values": "wired",
        "normalise_params": "wired",
        "check_param_names": "wired",
        "check_duplicate_identities": "wired",
        "check_training_managed_overrides": "wired",
        "check_smart_managed_overrides": "open: #279, 54/67 of the population",
        "canonicalisation": "n/a: merged by identity, not by spelling",
    },
    "tuning.optuna.space": {
        "check_param_values": (
            "n/a: merged-value gate; trial overlays use adapter validation"
        ),
        "normalise_params": (
            "n/a: dimensions carry bounds, and the values sampled from them "
            "arrive at `tuning best_model_params`"
        ),
        "check_param_names": "wired",
        "check_duplicate_identities": "wired",
        "check_training_managed_overrides": "wired",
        "check_smart_managed_overrides": "open: #279, 54/67 of the population",
        "canonicalisation": "n/a: dimensions carry names, not values",
    },
    "calibration.params": {
        "check_param_values": (
            "n/a: separate calibrator boundary, not merged model inputs"
        ),
        "normalise_params": "wired",
        "check_param_names": "wired",
        "check_duplicate_identities": "wired",
        "check_training_managed_overrides": (
            "n/a: the calibrator takes its own seed by documented precedence "
            "(H-0080) and has no early stopping of its own"
        ),
        "check_smart_managed_overrides": (
            "n/a: smart resolution does not reach the calibrator"
        ),
        "canonicalisation": "wired",
    },
}

#: Cells whose value must be reached by an executed case in this file.
_WIRED = {
    (layer, check)
    for layer, checks in REFUSAL_MATRIX.items()
    for check, state in checks.items()
    if state == "wired"
}


def test_the_refusal_matrix_covers_every_layer_and_every_check() -> None:
    """The grid is rectangular, and nothing has been quietly dropped from it.

    A table that loses a row or a column stops reporting the gap it was written
    to report -- which is how three of this PR's own findings survived to be
    found by review (H-0094 decision 10).
    """
    checks = {check for row in REFUSAL_MATRIX.values() for check in row}
    assert len(checks) == 7, checks
    for layer, row in REFUSAL_MATRIX.items():
        assert set(row) == checks, f"{layer} is missing {checks - set(row)}"
        for check, state in row.items():
            assert state == "wired" or state.startswith(("n/a: ", "open: ")), (
                f"{layer} x {check} says {state!r}; a cell is wired, n/a with a "
                "reason, or open with an issue and a measured rate"
            )


def test_every_open_cell_names_an_issue() -> None:
    """An open cell is a deferral the maintainer holds, not a note to ourselves."""
    for layer, row in REFUSAL_MATRIX.items():
        for check, state in row.items():
            if state.startswith("open: "):
                assert "#" in state, f"{layer} x {check} is open with no issue"


def test_no_refusal_exists_that_the_matrix_does_not_name() -> None:
    """The columns are derived from the code, not from what we remembered.

    Reading the module for `check_*` functions is what makes a sixth refusal
    arriving without a column a failure rather than a silent hole -- the shape
    the seam scan was twice caught in (decision 8, decision 9).
    """
    import lizyml.core._model_factories as factories

    named = {check for row in REFUSAL_MATRIX.values() for check in row}
    refusals = {
        name
        for name in dir(factories)
        if name.startswith(("check_", "normalise_"))
        and name not in {"check_param_names"}
    }
    # The surfaces call one entry point that normalises and then checks; the
    # column is named for the half that is new.
    refusals -= {"normalise_and_check"}
    refusals |= {"normalise_params"}
    # The space-level wrappers delegate to the row they belong to.
    refusals -= {"check_duplicate_space_dimensions", "check_training_managed_space"}
    refusals -= {"check_calibration_param_names"}
    refusals |= {"check_param_names", "canonicalisation"}
    assert refusals <= named, (
        f"refusals with no column in the matrix: {refusals - named}"
    )


# --------------------------------------------------------------------------
# Executing the grid
# --------------------------------------------------------------------------
#
# The table above would be prose if nothing reached its cells. Each entry here
# builds a config that violates one (layer, refusal) pair and asserts that the
# refusal fires, names the layer, and fires **before training** -- the property
# round 14's finding was about, where the refusal existed but arrived after a
# study had trained two boosters.
#
# `_CELL_INPUTS` is keyed by the same (layer, check) pairs as `REFUSAL_MATRIX`,
# and a test below asserts the two agree, so a cell cannot be marked wired
# without an input that reaches it.

#: A native name whose value is visible in the booster text.
_OVERRIDDEN = "learning_rate"


class _Unaccepted:
    """A value LightGBM would serialise and H-0095 refuses at the surface.

    ``_is_numeric`` accepts anything ``float()`` survives, so the serialiser
    would have written ``learning_rate=0.5`` for this. The accepted set is
    narrower on purpose: what a value *is* has to be knowable before the
    comparison, not discovered by asking the value.
    """

    def __float__(self) -> float:
        return 0.5


#: One instance, shared by the cells below.
_UNACCEPTED = _Unaccepted()


def _base(**kwargs: Any) -> dict[str, Any]:
    return make_config("binary", n_estimators=3, n_splits=2, **kwargs)


def _model_params(**params: Any) -> dict[str, Any]:
    cfg = _base()
    cfg["model"]["params"].update(params)
    return cfg


def _space(**dims: Any) -> dict[str, Any]:
    cfg = _base(tuning_n_trials=1)
    cfg["tuning"]["optuna"]["space"] = dims
    return cfg


_FLOAT_DIM = {"type": "float", "low": 0.001, "high": 0.01, "category": "model"}
_INT_DIM = {"type": "int", "low": 11, "high": 12, "category": "model"}


def _calibration(**params: Any) -> dict[str, Any]:
    cfg = _base()
    cfg["calibration"] = {"method": "isotonic", "params": params}
    return cfg


#: ``(layer, check) -> (config, fit kwargs, the entry point to call)``.
_CELL_INPUTS: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], str]] = {
    ("model.params", "check_param_values"): (
        _model_params(application="regression"),
        {},
        "fit",
    ),
    ("fit(params=)", "check_param_values"): (
        _base(),
        {"params": {"metrics": "rmse"}},
        "fit",
    ),
    ("model.params", "check_param_names"): (
        _model_params(not_a_lightgbm_parameter=1),
        {},
        "fit",
    ),
    ("model.params", "check_duplicate_identities"): (
        _model_params(learning_rate=0.001, eta=0.5),
        {},
        "fit",
    ),
    ("model.params", "normalise_params"): (
        _model_params(learning_rate=_UNACCEPTED),
        {},
        "fit",
    ),
    ("model.params", "check_training_managed_overrides"): (
        _model_params(seed=7),
        {},
        "fit",
    ),
    ("fit(params=)", "check_param_names"): (
        _base(),
        {"params": {"not_a_lightgbm_parameter": 1}},
        "fit",
    ),
    ("fit(params=)", "check_duplicate_identities"): (
        _base(),
        {"params": {"learning_rate": 0.001, "eta": 0.5}},
        "fit",
    ),
    ("fit(params=)", "normalise_params"): (
        _base(),
        {"params": {"learning_rate": _UNACCEPTED}},
        "fit",
    ),
    ("fit(params=)", "check_training_managed_overrides"): (
        _base(),
        {"params": {"seed": 7}},
        "fit",
    ),
    ("fit(params=)", "check_smart_managed_overrides"): (
        _base(),
        {"params": {"num_leaves": 12}},
        "fit",
    ),
    ("tuning.optuna.space", "check_param_names"): (
        _space(not_a_lightgbm_parameter=_FLOAT_DIM),
        {},
        "tune",
    ),
    ("tuning.optuna.space", "check_duplicate_identities"): (
        _space(learning_rate=_FLOAT_DIM, eta=_FLOAT_DIM),
        {},
        "tune",
    ),
    ("tuning.optuna.space", "check_training_managed_overrides"): (
        _space(seed=_INT_DIM),
        {},
        "tune",
    ),
    ("calibration.params", "check_param_names"): (
        _calibration(num_leave=7),
        {},
        "fit",
    ),
    ("calibration.params", "check_duplicate_identities"): (
        _calibration(learning_rate=0.001, eta=0.5),
        {},
        "fit",
    ),
    ("calibration.params", "normalise_params"): (
        _calibration(learning_rate=_UNACCEPTED),
        {},
        "fit",
    ),
}


@pytest.mark.parametrize("cell", sorted(_CELL_INPUTS))
def test_each_wired_cell_refuses_before_training(cell: tuple[str, str]) -> None:
    """Every cell the table calls wired, executed.

    Asserted on three things, because this PR has produced a defect against each
    of them: that the refusal fires, that it names the layer the caller has to
    change (decision 3), and that **nothing trained first** (decision 10).
    """
    config, fit_kwargs, entry = _CELL_INPUTS[cell]
    layer, _check = cell
    model = Model(config, data=make_binary_df(n=160))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        if entry == "tune":
            model.tune()
        else:
            model.fit(**fit_kwargs)

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert layer in exc.value.user_message, (
        f"{cell} refused without naming {layer}: {exc.value.user_message}"
    )
    assert not seen["train_params"], (
        f"{cell} trained {len(seen['train_params'])} Booster(s) before refusing"
    )


def _with_tuning_result(config: dict[str, Any], best: dict[str, Any]) -> Model:
    """A model carrying a restored tuning result, as `load()` produces one.

    The layer matters because `best_model_params` comes back from an artifact
    written before these refusals existed, so it is the one input the checks
    cannot assume was ever gated at the time it was created.
    """
    model = Model(config, data=make_binary_df(n=160))
    model._tuning_result = TuningResult(
        best_model_params=best,
        best_smart_params={},
        best_training_params={},
        best_score=0.0,
        metric_name="auc",
        direction="maximize",
        trials=[],
        rounds=(),
    )
    return model


@pytest.mark.parametrize(
    "check,best",
    [
        ("check_param_names", {"not_a_lightgbm_parameter": 1}),
        ("check_param_values", {"application": "regression"}),
        ("check_training_managed_overrides", {"seed": 7}),
        # Added in round 15, which falsified this cell's previous `n/a`.
        # `overlay_params` drops competing spellings from the layer it overlays
        # and keeps whatever the **overlay itself** carries, so a restored
        # `best_model_params` naming one parameter twice sent both spellings to
        # `lgb.train`. Measured: `{"learning_rate": 0.1, "eta": 0.8}` trained at
        # 0.1 with both present.
        ("check_duplicate_identities", {"learning_rate": 0.1, "eta": 0.8}),
        # H-0095: an artifact is the one layer written before these refusals
        # existed, so it is the one that can carry a value the accepted set
        # does not contain.
        ("normalise_params", {"learning_rate": _UNACCEPTED}),
    ],
)
def test_a_restored_tuning_result_is_refused_too(
    check: str, best: dict[str, Any]
) -> None:
    """The layer that arrives from disk rather than from the caller.

    `Model.load()` itself does not check -- an artifact is the record of a fit
    that happened, and refusing to read it would help nobody. The refusal
    belongs on the **re-fit**, which is where these run.
    """
    assert REFUSAL_MATRIX["tuning best_model_params"][check] == "wired"
    model = _with_tuning_result(_base(), best)

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit()

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert not seen["train_params"], (
        f"{check} trained {len(seen['train_params'])} Booster(s) before refusing"
    )


def test_calibration_params_are_canonicalised_before_the_calibrator_sees_them() -> None:
    """The one cell that is a rewrite rather than a refusal.

    It cannot use the harness above because nothing raises: the point is that
    the alias *reaches* the calibrator, having been rewritten to the spelling
    the calibrator merges by.
    """
    assert REFUSAL_MATRIX["calibration.params"]["canonicalisation"] == "wired"
    config = _calibration(eta=0.5, num_boost_round=3)

    with record_lightgbm_calls() as seen:
        Model(config, data=make_binary_df(n=160)).fit()

    calibrator_calls = [
        call for call in seen["train_params"] if call.get("monotone_constraints")
    ]
    assert calibrator_calls, "no calibrator Booster was trained"
    for call in calibrator_calls:
        assert call.get(_OVERRIDDEN) == 0.5, call
        assert "eta" not in call, f"two spellings reached lgb.train: {call!r}"


def test_the_executed_cells_are_exactly_the_wired_ones() -> None:
    """A cell cannot be called wired without an input that reaches it.

    This is the assertion that turns the table from a description into a
    declaration. Without it, marking a cell wired would be the same kind of
    unexecuted claim that three consecutive rounds found and closed.

    ``model.params x check_smart_managed_overrides`` is the one wired-looking
    pair deliberately absent: it is **open** (#280), and the table says so.
    """
    executed = set(_CELL_INPUTS) | {
        ("tuning best_model_params", "check_param_values"),
        ("tuning best_model_params", "check_param_names"),
        ("tuning best_model_params", "check_training_managed_overrides"),
        ("tuning best_model_params", "check_duplicate_identities"),
        ("tuning best_model_params", "normalise_params"),
        ("calibration.params", "canonicalisation"),
    }
    missing = _WIRED - executed
    assert not missing, f"cells marked wired with no executed input: {sorted(missing)}"
    extra = executed - _WIRED
    assert not extra, f"executed cells the table does not call wired: {sorted(extra)}"
