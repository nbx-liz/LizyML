"""H-0122 §7.1 / H-0094 decision 14: a tune that fails leaves the Model as it was.

Every case runs with ``output_dir`` set, because the run directory is
allocated at the start of the writes (phase E): without ``output_dir`` the
Model's ``_run_dir`` stays ``None`` either way and a snapshot cannot see it
move (DC5). ``_provider`` is outside :func:`snapshot_model` on purpose (it is
a per-call transient for ``fit``), so it is compared here by identity.

The run directory a failed tune created on disk may remain; only the Model's
fields are promised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests.test_tuning._resume_helpers import (
    ProcessDeath,
    die_in_objective,
    fast_config,
    frame,
    snapshot_model,
)


def _every_trial_fails(monkeypatch: pytest.MonkeyPatch, model: Model) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("every trial fails")

    monkeypatch.setattr(model, "_build_train_components", broken)


def test_failed_close_leaves_run_dir_and_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every trial FAILs: the round closes ``failed`` and ``TUNING_FAILED``."""
    model = Model(fast_config(n_trials=1), data=frame(), output_dir=tmp_path)
    _every_trial_fails(monkeypatch, model)
    before = snapshot_model(model)
    provider_before = model._provider
    with pytest.raises(LizyMLError) as caught:
        model.tune()
    assert caught.value.code is ErrorCode.TUNING_FAILED
    assert snapshot_model(model) == before
    assert model._run_dir is None
    assert model._provider is provider_before


def test_exception_during_optimize_leaves_run_dir_and_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``BaseException`` escaping the trials propagates; the Model is as before."""
    model = Model(fast_config(n_trials=2), data=frame(), output_dir=tmp_path)
    die_in_objective(monkeypatch, model, at_call=1)
    before = snapshot_model(model)
    provider_before = model._provider
    with pytest.raises(ProcessDeath):
        model.tune()
    assert snapshot_model(model) == before
    assert model._run_dir is None
    assert model._provider is provider_before


def test_failed_resume_restores_the_previous_run_dir_and_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuing a held study: round 2 dies, round 1's fields come back."""
    model = Model(fast_config(n_trials=2), data=frame(), output_dir=tmp_path)
    model.tune()
    run_dir_1 = model._run_dir
    provider_1 = model._provider
    assert run_dir_1 is not None and provider_1 is not None
    before = snapshot_model(model)
    die_in_objective(monkeypatch, model, at_call=1)
    with pytest.raises(ProcessDeath):
        model.tune(resume=True)
    assert snapshot_model(model) == before
    assert model._run_dir == run_dir_1
    assert model._provider is provider_1
