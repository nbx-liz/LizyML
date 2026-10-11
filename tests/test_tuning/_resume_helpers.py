"""Shared helpers for the H-0122 resume / reattach integration tests.

* :func:`fast_config` -- a small, fast tuning config (2 folds, 3 trees).
* :func:`url` -- a SQLite storage URL inside ``tmp_path``.
* :func:`snapshot_study` / :func:`snapshot_model` -- what INV-R4 promises not
  to change on a refusal (criterion 22).
* :func:`sqlite_exec` -- edit ``lizyml:`` attributes of a stored study through
  stdlib ``sqlite3``; Optuna has no attribute-delete API.
* :class:`ProcessDeath`, :func:`die_at_attr`, :func:`die_in_objective` --
  simulate a process dying at a given point of a tune (crash tests).
"""

from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import pytest

from lizyml import Model
from lizyml.tuning import tuner as tuner_module
from tests._helpers import make_binary_df, make_config, make_regression_df

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
