"""Shared helpers for the H-0122 resume / reattach integration tests.

* :func:`fast_config` -- a small, fast tuning config (2 folds, 3 trees).
* :func:`url` -- a SQLite storage URL inside ``tmp_path``.
* :func:`snapshot_study` / :func:`snapshot_model` -- what INV-R4 promises not
  to change on a refusal (criterion 22).
* :func:`sqlite_exec` -- edit ``lizyml:`` attributes of a stored study through
  stdlib ``sqlite3``; Optuna has no attribute-delete API.
* :class:`ProcessDeath`, :func:`die_at_attr`, :func:`die_in_objective` --
  simulate a process dying at a given point of a tune (crash tests).
* :func:`crash_round_2`, :func:`run_child_and_kill` -- the criterion-8 crash
  scenario in-process, and in a real child process (this module's
  ``__main__``).
* :func:`spy_tuner`, :func:`study_record_for` -- what the round runs with, and
  a ``lizyml:study`` record built for a given config (states E/N by hand).
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import pytest

from lizyml import Model
from lizyml.tuning import tuner as tuner_module
from lizyml.tuning.lineage import end_key, round_key
from lizyml.tuning.tuner import Tuner
from tests._helpers import make_binary_df, make_config, make_regression_df

REPO_ROOT = Path(__file__).resolve().parents[2]

SPACE: dict[str, Any] = {
    "num_leaves": {"type": "int", "low": 4, "high": 8},
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
}
#: Keeps merge-mode tunes cheap: the default ``n_estimators`` range is 600-2500.
SMALL_TREES: dict[str, Any] = {"type": "int", "low": 3, "high": 5}


def fast_config(
    task: str = "regression", n_trials: int = 2, mode: str = "replace"
) -> dict[str, Any]:
    """A tuning config that trains 3-tree models on 2 folds."""
    cfg = make_config(task, n_estimators=3, n_splits=2, tuning_n_trials=n_trials)
    cfg["model"]["auto_num_leaves"] = False
    space = copy.deepcopy(SPACE)
    if mode == "merge":
        space["n_estimators"] = dict(SMALL_TREES)
    cfg["tuning"]["optuna"]["space"] = space
    cfg["tuning"]["optuna"]["space_mode"] = mode
    return cfg


def frame(task: str = "regression", n: int = 80) -> pd.DataFrame:
    """Training data matching :func:`fast_config`."""
    if task == "binary":
        return make_binary_df(n=n)
    return make_regression_df(n=n)


def url(tmp_path: Path, name: str = "s.db") -> str:
    """A SQLite storage URL inside *tmp_path*."""
    return f"sqlite:///{tmp_path / name}"


def load(storage: Any, name: str) -> Any:
    """Reload a stored study, so what is compared comes from the storage."""
    return optuna.load_study(study_name=name, storage=storage)


# ---------------------------------------------------------------------------
# Snapshots (criterion 22)
# ---------------------------------------------------------------------------


def lizyml_attrs(study: Any) -> dict[str, Any]:
    """The study's ``lizyml:`` attributes."""
    return {k: v for k, v in study.user_attrs.items() if k.startswith("lizyml:")}


def snapshot_study(study_or_url: Any, name: str | None = None) -> Any:
    """Trials (number, state, values, params) and ``lizyml:`` attributes.

    A storage URL (or ``BaseStorage``) is reloaded from the storage; a study
    object is read as it is.
    """
    study = study_or_url if name is None else load(study_or_url, name)
    trials = [(t.number, t.state.name, t.values, t.params) for t in study.trials]
    return trials, json.dumps(lizyml_attrs(study), sort_keys=True)


#: Tune-owned Model state (H-0094 decision 14). Compared by identity; a
#: ``BaseStorage`` names "the same study" only as the same object (§3).
_IDENTITY_FIELDS = (
    "_study",
    "_tuning_result",
    "_X",
    "_y",
    "_fit_result",
    "_tuning_storage",
)
#: Compared by value.
_VALUE_FIELDS = (
    "_space",
    "_round_number",
    "_rounds",
    "_used_default_space",
    "_tuning_fixed_params",
    "_tuning_study_name",
    "_run_dir",
)
# `_block_values`, `_input_categories` and `_provider` are left out on purpose:
# they are per-call transients `_prepare_training_data` / `fit` recompute before
# reading (H-0122 design R8), not state a tune commits.


def snapshot_model(model: Model) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    """Identity of the tune-owned objects and copies of the tune-owned values."""
    ids = tuple(id(getattr(model, f, None)) for f in _IDENTITY_FIELDS)
    values = tuple(copy.deepcopy(getattr(model, f, None)) for f in _VALUE_FIELDS)
    return ids, values


# ---------------------------------------------------------------------------
# Storage surgery
# ---------------------------------------------------------------------------


def sqlite_path(storage_url: str) -> str:
    assert storage_url.startswith("sqlite:///"), storage_url
    return storage_url[len("sqlite:///") :]


def sqlite_exec(storage_url: str, sql: str, params: tuple[Any, ...] = ()) -> int:
    """Run one statement on the storage's SQLite file; return the row count.

    Attributes live in table ``study_user_attributes`` (columns ``key`` and
    ``value_json``).
    """
    with sqlite3.connect(sqlite_path(storage_url)) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.rowcount


def delete_attrs(storage_url: str, pattern: str) -> int:
    """Delete the study attributes whose key matches the SQL LIKE *pattern*."""
    return sqlite_exec(
        storage_url, "DELETE FROM study_user_attributes WHERE key LIKE ?", (pattern,)
    )


def set_attr(storage_url: str, name: str, key: str, value: Any) -> None:
    """Overwrite one attribute through Optuna (what a writer would store)."""
    load(storage_url, name).set_user_attr(key, value)


# ---------------------------------------------------------------------------
# Crash simulation
# ---------------------------------------------------------------------------


class ProcessDeath(BaseException):
    """Stands in for the process dying.

    Neither an ``Exception`` nor a ``KeyboardInterrupt``, so Optuna's
    ``_run_trial`` does not catch it and a trial it escapes stays RUNNING.
    """


def die_at_attr(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    """Raise :class:`ProcessDeath` just before the study attribute *key* is written.

    Every ``lizyml:`` write goes through ``lizyml.tuning.tuner.write_study_attr``
    (looked up on the module at call time), so this stops a tune between any
    two of the §7.1 write positions.
    """
    real = tuner_module.write_study_attr

    def write(study: Any, attr: str, value: Any) -> None:
        if attr == key:
            raise ProcessDeath(f"died before writing {attr}")
        real(study, attr, value)

    monkeypatch.setattr(tuner_module, "write_study_attr", write)


def die_in_objective(
    monkeypatch: pytest.MonkeyPatch, model: Model, at_call: int
) -> None:
    """Raise :class:`ProcessDeath` inside the *at_call*-th trial (1-based).

    ``Model._build_train_components`` runs once per trial, inside the
    objective, so the trial Optuna has already created stays RUNNING.
    """
    real = model._build_train_components
    calls = {"n": 0}

    def build(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == at_call:
            raise ProcessDeath(f"died in trial call {at_call}")
        return real(*args, **kwargs)

    monkeypatch.setattr(model, "_build_train_components", build)


# ---------------------------------------------------------------------------
# The criterion-8 crash scenario
# ---------------------------------------------------------------------------

#: Round 1 runs `ROUND_1_TRIALS` (an argument); round 2 asks for the config's
#: `ROUND_2_TRIALS` with `expand_boundary=True`, and the process dies in round
#: 2's `DIE_AT_CALL`-th trial -- trial #3, left RUNNING.
ROUND_1_TRIALS = 2
ROUND_2_TRIALS = 3
DIE_AT_CALL = 2
CRASHED_TRIAL = ROUND_1_TRIALS + DIE_AT_CALL - 1


def edge_config(n_trials: int = ROUND_2_TRIALS) -> dict[str, Any]:
    """``num_leaves`` in [4, 5]: every best sits on an edge, so round 2 expands
    and its record carries a non-empty boundary report."""
    cfg = fast_config(n_trials=n_trials)
    cfg["tuning"]["optuna"]["space"]["num_leaves"] = {
        "type": "int",
        "low": 4,
        "high": 5,
    }
    return cfg


def crash_round_2(
    monkeypatch: pytest.MonkeyPatch,
    model: Model,
    storage: Any = None,
    name: str | None = None,
) -> None:
    """Run round 1 on *model*, then die in round 2's 2nd trial.

    *monkeypatch* is undone before returning, so the same Model can resume.
    """
    model.tune(n_trials=ROUND_1_TRIALS, storage=storage, study_name=name)
    die_in_objective(monkeypatch, model, at_call=DIE_AT_CALL)
    with pytest.raises(ProcessDeath):
        model.tune(resume=True, expand_boundary=True)
    monkeypatch.undo()


def run_child_and_kill(storage_url: str, name: str) -> subprocess.CompletedProcess[str]:
    """Run the crash scenario in a real child process that ``os._exit(1)``s.

    The child (this module's ``__main__``) runs round 1 and round 2 against
    *storage_url* and ends the process from inside round 2's 2nd trial, so
    nothing after the objective runs: no ``tell``, no ``:end``.
    """
    return subprocess.run(  # noqa: S603 -- fixed argv, this interpreter
        [sys.executable, "-m", "tests.test_tuning._resume_helpers", storage_url, name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _child_main(storage_url: str, name: str) -> None:
    """The child of :func:`run_child_and_kill`."""
    model = Model(edge_config(), data=frame())
    model.tune(n_trials=ROUND_1_TRIALS, storage=storage_url, study_name=name)
    real = model._build_train_components
    calls = {"n": 0}

    def build(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == DIE_AT_CALL:
            os._exit(1)
        return real(*args, **kwargs)

    model._build_train_components = build  # type: ignore[method-assign]
    model.tune(resume=True, expand_boundary=True)
    sys.exit(2)  # reached only when the crash did not fire


# ---------------------------------------------------------------------------
# Spies and hand-built records
# ---------------------------------------------------------------------------


def spy_tuner(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Record every ``Tuner`` construction and ``Tuner.run`` call."""
    seen: dict[str, list[Any]] = {"init": [], "run": []}
    real_init = Tuner.__init__
    real_run = Tuner.run

    def init(self: Tuner, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        seen["init"].append({"n_trials": self.n_trials, "timeout": self.timeout})

    def run(self: Tuner, *args: Any, **kwargs: Any) -> None:
        seen["run"].append(kwargs)
        real_run(self, *args, **kwargs)

    monkeypatch.setattr(Tuner, "__init__", init)
    monkeypatch.setattr(Tuner, "run", run)
    return seen


def study_record_for(cfg_dict: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """The ``lizyml:study`` record a tune of *cfg_dict* on *df* would write.

    Built with the functions ``Model.tune`` uses, so the record's identity is
    that call's and only the checks after identity can refuse it.
    """
    from lizyml.core._model_factories import get_provider
    from lizyml.core._tuning_identity import (
        build_identity,
        config_parts,
        library_versions,
        provider_default_params,
        summarise_frame,
    )
    from lizyml.core._tuning_resume import study_record_payload
    from lizyml.core._tuning_validation import resolve_tuning_direction

    cfg = Model(cfg_dict)._cfg
    assert cfg.tuning is not None
    provider = get_provider(cfg.model)
    fixed = (
        provider.default_fixed_params(cfg.task)
        if cfg.tuning.optuna.space_mode == "merge"
        else {}
    )
    direction = resolve_tuning_direction(cfg)
    parts = config_parts(
        cfg,
        fixed,
        direction,
        provider_defaults=provider_default_params(provider, cfg.task),
    )
    identity = build_identity(summarise_frame(df), parts)
    return study_record_payload(
        cfg,
        identity=identity,
        direction=direction,
        versions=library_versions(provider),
        fixed=fixed,
    )


# ---------------------------------------------------------------------------
# Corruptions (state X) and an admission-refused config (criteria 19, 22)
# ---------------------------------------------------------------------------


#: Each turns a 2-round stored study named ``s`` into state X (§7.4).
def _no_rounds_with_trials(storage: str) -> None:
    assert delete_attrs(storage, "lizyml:round:%") == 4


def _gap(storage: str) -> None:
    for suffix in ("", ":end"):
        renamed = sqlite_exec(
            storage,
            "UPDATE study_user_attributes SET key = ? WHERE key = ?",
            (round_key(3) + suffix, round_key(2) + suffix),
        )
        assert renamed == 1


def _missing_mid_end(storage: str) -> None:
    assert delete_attrs(storage, end_key(1)) == 1


def _start_gt_0(storage: str) -> None:
    study = load(storage, "s")
    record = copy.deepcopy(study.user_attrs[round_key(1)])
    record["start_trial"] = 1
    study.set_user_attr(round_key(1), record)


INCONSISTENT: dict[str, Any] = {
    "no_rounds_with_trials": _no_rounds_with_trials,
    "gap": _gap,
    "missing_mid_end": _missing_mid_end,
    "start_gt_0": _start_gt_0,
}


#: A config whose own dimension H-0121 refuses: `num_leaves_ratio` is read only
#: while `auto_num_leaves` is on, and it is off.
RATIO = "num_leaves_ratio"


def admission_refused_config() -> dict[str, Any]:
    cfg = make_config(
        "binary",
        n_estimators=3,
        n_splits=2,
        split_method="stratified_kfold",
        tuning_n_trials=2,
    )
    cfg["model"]["auto_num_leaves"] = False
    cfg["tuning"]["optuna"]["space_mode"] = "replace"
    cfg["tuning"]["optuna"]["space"] = {
        RATIO: {"type": "float", "low": 0.1, "high": 0.9, "category": "smart"}
    }
    return cfg


if __name__ == "__main__":
    _child_main(sys.argv[1], sys.argv[2])
