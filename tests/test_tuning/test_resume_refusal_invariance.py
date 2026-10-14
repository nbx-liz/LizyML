"""H-0122 criterion 22: a refusal changes nothing (INV-R4).

One table over every refusal row (PR 1 design §6.4, PR 2 design §8.2, spec
§8): each row builds its state, snapshots the study (trial count, states,
values, params and every ``lizyml:`` attribute; a stored study is re-read from
the storage) and the Model's tune-owned state, calls ``tune`` (or
``restore_tuning``), checks the error code and the reason (so a row cannot
pass on an unrelated refusal), and compares both snapshots.

Rows that refuse a Model holding a study use a Model that has tuned, so the
snapshot covers real state, not only ``None`` fields.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from optuna.storages import RDBStorage

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning.lineage import STUDY_KEY
from tests.test_tuning._invariance import (
    CI,
    Case,
    Env,
    _changed_frame,
    _empty_study,
    _holding_c,
    _memory_tuned,
    _message,
    _mismatched,
    _state_is,
    _surface,
    env,
    run_case,
    templates,
)
from tests.test_tuning._resume_helpers import (
    ADMISSION_SPACES,
    CAT_LR,
    INCONSISTENT,
    OPEN_SPACE,
    RATIO,
    admission_refused_config,
    cat_config,
    delete_attrs,
    edge_config,
    fast_config,
    frame,
    load,
    set_attr,
    space_of,
    study_record_for,
)

#: The fixtures this table runs on (imported so pytest finds them here).
__all__ = ["env", "templates"]


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


def _storage_no_name(env: Env, resume: bool) -> Case:
    model = _memory_tuned()
    return Case(
        model,
        model._study,
        {"resume": resume, "storage": env.spare},
        CI,
        _surface("tune(storage=)"),
        {"no_file": env.spare},
    )


def _name_no_storage(env: Env, resume: bool) -> Case:
    model = _memory_tuned()
    return Case(
        model,
        model._study,
        {"resume": resume, "study_name": "s"},
        CI,
        _surface("tune(study_name=)"),
    )


def _resume_false_existing_c(env: Env) -> Case:
    model = _holding_c(env)
    return Case(
        model,
        (env.c, "s"),
        {"storage": env.c, "study_name": "s"},
        CI,
        _message("already exists"),
    )


def _resume_false_existing_o(env: Env) -> Case:
    assert env.state(env.o) == "open"
    return Case(
        _memory_tuned(),
        (env.o, "s"),
        {"storage": env.o, "study_name": "s"},
        CI,
        _message("already exists"),
    )


def _resume_false_existing_e(env: Env) -> Case:
    _empty_study(env)
    return Case(
        _memory_tuned(),
        (env.spare, "s"),
        {"storage": env.spare, "study_name": "s"},
        CI,
        _message("already exists"),
    )


def _other_name(env: Env) -> Case:
    return Case(
        _holding_c(env),
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "other"},
        CI,
        _message("cannot be shown to name it"),
    )


def _other_url_string(env: Env) -> Case:
    path = Path(env.c[len("sqlite:///") :])
    spelled = "sqlite:///./" + os.path.relpath(path)
    assert spelled != env.c
    return Case(
        _holding_c(env),
        (env.c, "s"),
        {"resume": True, "storage": spelled, "study_name": "s"},
        CI,
        _message("cannot be shown to name it"),
    )


def _other_storage_object(env: Env) -> Case:
    model = Model(fast_config(), data=frame())
    model.tune(storage=RDBStorage(env.spare), study_name="x")
    return Case(
        model,
        (env.spare, "x"),
        {"resume": True, "storage": RDBStorage(env.spare), "study_name": "x"},
        CI,
        _message("cannot be shown to name it"),
    )


def _memory_plus_storage(env: Env) -> Case:
    model = _memory_tuned()
    return Case(
        model,
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "s"},
        CI,
        _message("cannot be shown to name it"),
        {"also": model._study},
    )


def _identity_data(env: Env) -> Case:
    return Case(
        _holding_c(env),
        (env.c, "s"),
        {"resume": True, "data": _changed_frame()},
        CI,
        _mismatched("data.values"),
    )


def _identity_config(env: Env) -> Case:
    model = _holding_c(env)
    model._cfg.training.seed = model._cfg.training.seed + 1
    return Case(
        model, (env.c, "s"), {"resume": True}, CI, _mismatched("config.training")
    )


def _identity_metric(env: Env) -> Case:
    model = _holding_c(env)
    model._cfg.evaluation.metrics = ["mae"]
    return Case(model, (env.c, "s"), {"resume": True}, CI, _mismatched("metric"))


def _legacy(env: Env) -> Case:
    assert delete_attrs(env.c, "lizyml:%") > 0
    return Case(
        Model(fast_config(), data=frame()),
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "s"},
        CI,
        _state_is("legacy"),
    )


def _unreadable(env: Env) -> Case:
    set_attr(env.c, "s", STUDY_KEY, {"schema": 999})
    return Case(
        Model(fast_config(), data=frame()),
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "s"},
        CI,
        _state_is("unreadable"),
    )


def _unreadable_fixed(env: Env) -> Case:
    """The stored ``fixed`` no longer digests to the record's ``config.fixed``."""
    record = copy.deepcopy(load(env.c, "s").user_attrs[STUDY_KEY])
    record["fixed"] = {"first_metric_only": False}
    set_attr(env.c, "s", STUDY_KEY, record)
    return Case(
        Model(fast_config(), data=frame()),
        (env.c, "s"),
        {"resume": True, "storage": env.c, "study_name": "s"},
        CI,
        lambda e: _state_is("unreadable")(e) and _message("config.fixed")(e),
    )


def _unreadable_empty(env: Env) -> Case:
    _empty_study(env)
    set_attr(env.spare, "s", STUDY_KEY, {"schema": 999})
    return Case(
        Model(fast_config(), data=frame()),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _state_is("unreadable"),
    )


def _inconsistent(corruption: str) -> Callable[[Env], Case]:
    def build(env: Env) -> Case:
        INCONSISTENT[corruption](env.c)
        assert env.state(env.c) == "inconsistent"
        return Case(
            Model(fast_config(), data=frame()),
            (env.c, "s"),
            {"resume": True, "storage": env.c, "study_name": "s"},
            CI,
            _state_is("inconsistent"),
        )

    return build


def _config_space(env: Env) -> Case:
    model = _holding_c(env)
    assert model._cfg.tuning is not None
    space = copy.deepcopy(model._cfg.tuning.optuna.space)
    space["num_leaves"]["high"] = 9
    model._cfg.tuning.optuna.space = space
    return Case(
        model, (env.c, "s"), {"resume": True}, CI, _surface("tuning.optuna.space")
    )


def _config_space_mode(env: Env) -> Case:
    model = _holding_c(env)
    assert model._cfg.tuning is not None
    model._cfg.tuning.optuna.space_mode = "merge"
    return Case(
        model, (env.c, "s"), {"resume": True}, CI, _surface("tuning.optuna.space_mode")
    )


def _open(kwargs: dict[str, Any], reason: Callable[[LizyMLError], bool]) -> Any:
    def build(env: Env) -> Case:
        assert load(env.o, "s").trials[3].state.name == "RUNNING"
        return Case(
            Model(edge_config(), data=frame()),
            (env.o, "s"),
            {"resume": True, "storage": env.o, "study_name": "s", **kwargs},
            CI,
            reason,
        )

    return build


def _no_round(env: Env, cfg: dict[str, Any], data: pd.DataFrame) -> None:
    _empty_study(env)
    set_attr(env.spare, "s", STUDY_KEY, study_record_for(cfg, data))
    assert env.state(env.spare) == "no_round"


def _no_round_identity(env: Env) -> Case:
    _no_round(env, fast_config(), frame())
    return Case(
        Model(fast_config(), data=_changed_frame()),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _mismatched("data.values"),
    )


def _uninitialised_admission(env: Env) -> Case:
    _empty_study(env)
    return Case(
        Model(admission_refused_config(), data=frame("binary")),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _message(RATIO),
    )


def _no_round_admission(env: Env) -> Case:
    _no_round(env, admission_refused_config(), frame("binary"))
    return Case(
        Model(admission_refused_config(), data=frame("binary")),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _message(RATIO),
    )


def _direction_mismatch(env: Env) -> Case:
    _empty_study(env, direction="maximize")
    return Case(
        Model(fast_config(), data=frame()),
        (env.spare, "s"),
        {"resume": True, "storage": env.spare, "study_name": "s"},
        CI,
        _message("direction disagrees"),
    )


def _no_study_no_storage(env: Env) -> Case:
    return Case(
        Model(fast_config(), data=frame()),
        None,
        {"resume": True},
        ErrorCode.TUNING_FAILED,
        lambda e: e.context.get("storage_given") is False,
    )


# ---------------------------------------------------------------------------
# PR 2: `space=` rows (criteria 7, 8b, 12, 14, 19, 29)
# ---------------------------------------------------------------------------

#: A well-formed `space=`: the fast config's own space.
SPACE = space_of(fast_config())


def _space_reason(e: LizyMLError) -> bool:
    return e.context.get("surface") == "tune(space=)"


def _creates_study(e: LizyMLError) -> bool:
    return _space_reason(e) and e.context.get("creates_study") is True


def _space_on_create_memory(env: Env) -> Case:
    model = _memory_tuned()
    return Case(model, model._study, {"space": SPACE}, CI, _creates_study)


def _space_on_create_storage_absent(env: Env) -> Case:
    model = _memory_tuned()
    return Case(
        model,
        model._study,
        {"storage": env.spare, "study_name": "s", "space": SPACE},
        CI,
        _creates_study,
        {"no_file": env.spare},
    )


def _space_on_u3(env: Env) -> Case:
    """U3 lists the storage's studies (which may create the SQLite file, design
    D-c), so the check is that no study was created, not that no file is."""
    return Case(
        Model(fast_config(), data=frame()),
        None,
        {"resume": True, "storage": env.spare, "study_name": "s", "space": SPACE},
        CI,
        _creates_study,
        {"absent_study": (env.spare, "s")},
    )


def _space_on_stored(prepare: Callable[[Env], None]) -> Callable[[Env], Case]:
    def build(env: Env) -> Case:
        prepare(env)
        return Case(
            Model(fast_config(), data=frame()),
            (env.spare, "s"),
            {
                "resume": True,
                "storage": env.spare,
                "study_name": "s",
                "space": SPACE,
            },
            CI,
            _creates_study,
        )

    return build


def _cat_tuned() -> Model:
    model = Model(cat_config(), data=frame())
    model.tune()
    return model


#: name -> (dim, the changed declaration, ``context["changed"]``).
UNCHANGEABLE: dict[str, tuple[str, dict[str, Any], str]] = {
    "space_kind": ("learning_rate", {"type": "int", "low": 1, "high": 3}, "kind"),
    "space_log": ("learning_rate", {**CAT_LR, "log": False}, "log"),
    "space_choice_type": (
        "lambda_l2",
        {"type": "categorical", "choices": [0.0, 1.0]},
        "choices",
    ),
    "space_choice_order": (
        "lambda_l2",
        {"type": "categorical", "choices": [1, 0]},
        "choices",
    ),
}


def _unchangeable(row: str) -> Callable[[Env], Case]:
    def build(env: Env) -> Case:
        name, spec, changed = UNCHANGEABLE[row]
        model = _cat_tuned()
        space = space_of(cat_config())
        space[name] = spec
        return Case(
            model,
            model._study,
            {"resume": True, "space": space},
            CI,
            lambda e: (
                _space_reason(e)
                and e.context.get("dim") == name
                and e.context.get("changed") == changed
            ),
        )

    return build


def _space_category(env: Env) -> Case:
    model = _cat_tuned()
    space = space_of(cat_config())
    space[RATIO] = {**space[RATIO], "category": "model"}
    return Case(
        model,
        model._study,
        {"resume": True, "space": space},
        CI,
        lambda e: _space_reason(e) and "category" in e.user_message,
    )


#: The new-tune admission checks read `space=` (criterion 29 bullet 2, and
#: H-0121 on a user-written dim): the shared `ADMISSION_SPACES`, plus a
#: malformed shape.
ADMISSION: dict[str, dict[str, Any]] = {
    **{f"space_{name}": space for name, space in ADMISSION_SPACES.items()},
    "space_malformed_shape": {"learning_rate": [0.1, 0.2]},
}


def _admission(row: str) -> Callable[[Env], Case]:
    def build(env: Env) -> Case:
        model = _memory_tuned()
        return Case(
            model,
            model._study,
            {"resume": True, "space": ADMISSION[row]},
            CI,
            _message("tune(space=)"),
        )

    return build


def _open_space(variant: str) -> Callable[[Env], Case]:
    """U2: an open round started with `space=` takes only its own space."""
    given = {
        "range": {
            **OPEN_SPACE,
            "learning_rate": {**OPEN_SPACE["learning_rate"], "high": 0.3},
        },
        "choice_type": {
            **OPEN_SPACE,
            "lambda_l1": {"type": "categorical", "choices": [0, 1.0]},
        },
    }[variant]

    def build(env: Env) -> Case:
        assert env.state(env.os) == "open"
        return Case(
            Model(edge_config(), data=frame()),
            (env.os, "s"),
            {"resume": True, "storage": env.os, "study_name": "s", "space": given},
            CI,
            lambda e: _space_reason(e) and e.context.get("round") == 2,
        )

    return build


# ---------------------------------------------------------------------------
# PR 2: `restore_tuning` rows (criterion 36)
# ---------------------------------------------------------------------------


def _restore(
    storage_of: Callable[[Env], str],
    code: ErrorCode,
    state: str,
    prepare: Callable[[Env], None] | None = None,
) -> Callable[[Env], Case]:
    """``restore_tuning(storage, "s")`` on a Model that has tuned in memory."""

    def build(env: Env) -> Case:
        if prepare is not None:
            prepare(env)
        storage = storage_of(env)
        assert env.state(storage) == state
        return Case(
            _memory_tuned(),
            (storage, "s"),
            {},
            code,
            _state_is(state),
            method="restore_tuning",
            args=(storage, "s"),
        )

    return build


def _restore_absent(env: Env) -> Case:
    return Case(
        _memory_tuned(),
        None,
        {},
        ErrorCode.TUNING_FAILED,
        _state_is("absent"),
        {"absent_study": (env.spare, "s")},
        method="restore_tuning",
        args=(env.spare, "s"),
    )


def _restore_identity_data(env: Env) -> Case:
    return Case(
        _memory_tuned(),
        (env.c, "s"),
        {"data": _changed_frame()},
        CI,
        _mismatched("data.values"),
        method="restore_tuning",
        args=(env.c, "s"),
    )


def _legacy_c(env: Env) -> None:
    assert delete_attrs(env.c, "lizyml:%") > 0


def _unreadable_c(env: Env) -> None:
    set_attr(env.c, "s", STUDY_KEY, {"schema": 999})


ROWS: dict[str, Callable[[Env], Case]] = {
    "storage_no_name[resume=False]": lambda env: _storage_no_name(env, False),
    "storage_no_name[resume=True]": lambda env: _storage_no_name(env, True),
    "name_no_storage[resume=False]": lambda env: _name_no_storage(env, False),
    "name_no_storage[resume=True]": lambda env: _name_no_storage(env, True),
    "resume_false_existing[C]": _resume_false_existing_c,
    "resume_false_existing[O]": _resume_false_existing_o,
    "resume_false_existing[E]": _resume_false_existing_e,
    "other_target[name]": _other_name,
    "other_target[url_string]": _other_url_string,
    "other_target[storage_object]": _other_storage_object,
    "other_target[memory_plus_storage]": _memory_plus_storage,
    "identity[data]": _identity_data,
    "identity[config]": _identity_config,
    "identity[metric]": _identity_metric,
    "legacy": _legacy,
    "unreadable": _unreadable,
    "unreadable[fixed_edited]": _unreadable_fixed,
    "unreadable_empty": _unreadable_empty,
    **{f"inconsistent[{name}]": _inconsistent(name) for name in sorted(INCONSISTENT)},
    "config_space": _config_space,
    "config_space_mode": _config_space_mode,
    "open_n_trials": _open({"n_trials": 4}, _surface("tune(n_trials=)")),
    "open_expand": _open(
        {"expand_boundary": False}, _surface("tune(expand_boundary=)")
    ),
    "open_identity": _open({"data": _changed_frame()}, _mismatched("data.values")),
    "no_round_identity": _no_round_identity,
    "uninitialised_admission": _uninitialised_admission,
    "no_round_admission": _no_round_admission,
    "direction_mismatch": _direction_mismatch,
    "no_study_no_storage": _no_study_no_storage,
    # --- PR 2: `space=` ---
    "space_on_create[memory]": _space_on_create_memory,
    "space_on_create[storage_absent]": _space_on_create_storage_absent,
    "space_on_u3": _space_on_u3,
    "space_on_no_round": _space_on_stored(
        lambda env: _no_round(env, fast_config(), frame())
    ),
    "space_on_uninitialised": _space_on_stored(_empty_study),
    **{row: _unchangeable(row) for row in UNCHANGEABLE},
    "space_category[smart_to_model]": _space_category,
    **{row: _admission(row) for row in ADMISSION},
    "open_space[range]": _open_space("range"),
    "open_space[choice_type]": _open_space("choice_type"),
    # --- PR 2: `restore_tuning` ---
    "restore[open]": _restore(lambda env: env.o, CI, "open"),
    "restore[failed]": _restore(lambda env: env.f, ErrorCode.TUNING_FAILED, "failed"),
    "restore[uninitialised]": _restore(
        lambda env: env.spare,
        ErrorCode.TUNING_FAILED,
        "uninitialized",
        _empty_study,
    ),
    "restore[no_round]": _restore(
        lambda env: env.spare,
        ErrorCode.TUNING_FAILED,
        "no_round",
        lambda env: _no_round(env, fast_config(), frame()),
    ),
    "restore[absent]": _restore_absent,
    "restore[legacy]": _restore(lambda env: env.c, CI, "legacy", _legacy_c),
    "restore[unreadable]": _restore(lambda env: env.c, CI, "unreadable", _unreadable_c),
    "restore[inconsistent:gap]": _restore(
        lambda env: env.c,
        CI,
        "inconsistent",
        lambda env: INCONSISTENT["gap"](env.c),
    ),
    "restore[identity_data]": _restore_identity_data,
}


@pytest.mark.parametrize("row", list(ROWS))
def test_refusal_changes_nothing(env: Env, row: str) -> None:
    run_case(ROWS[row](env))
