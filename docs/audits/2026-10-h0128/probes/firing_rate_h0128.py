"""Firing rate of H-0128's three conditions, measured on the suite before implementation.

Loaded as a pytest plugin (``-p firing_rate_h0128``) with this directory on
``PYTHONPATH``. Each wrapper records, then calls the original; nothing changes.

1. #348 -- a round closes ``failed`` when **that round** has no COMPLETE trial
   (today: when the study has none). For every ``Model._close_round`` call and
   every ``Tuner.run`` call (the public ``Tuner.tune`` and ``Model.tune`` both run
   through it): whether the round has a COMPLETE trial and
   whether the study has one. The new condition fires when the round has none
   and the study has one.
2. #349 (1) -- a ``lightgbm.basic.LightGBMError`` raised inside a trial is
   recorded as a FAIL instead of aborting the round. For every ``Tuner.run``
   call: whether it raised, and whether a LightGBMError is in the cause chain.
3. #349 (2) -- an open round with no COMPLETE trial, given a ``space=`` that
   differs from its record, is closed ``failed`` and a new round starts. For
   every ``plan_open_round`` call: whether ``space=`` was given, whether it
   differs from the record, and whether the round has a COMPLETE trial.

Writes one tab-separated row per event to ``$FIRING_RATE_H0128_OUT``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from lightgbm.basic import LightGBMError

from lizyml.core import _model_tuning, _tuning_plan
from lizyml.core._tuning_space import same_declared_space
from lizyml.tuning import tuner as tuner_module

_current = {"nodeid": "<none>"}
_rows: list[str] = []


def _complete(trials: Any, start: int = 0) -> bool:
    return any(t.state.name == "COMPLETE" and t.number >= start for t in trials)


_orig_close = _model_tuning.ModelTuningMixin._close_round


def _close(self: Any, study: Any, plan: Any) -> Any:
    round_has = _complete(study.trials, plan.start_trial)
    study_has = _complete(study.trials)
    _rows.append(
        f"close\t{_current['nodeid']}\t{plan.round_number}\t{int(round_has)}\t{int(study_has)}"
    )
    return _orig_close(self, study, plan)


_orig_run = tuner_module.Tuner.run


def _run(self: Any, study: Any, *args: Any, **kwargs: Any) -> Any:
    start = len(study.trials)
    try:
        return _orig_run(self, study, *args, **kwargs)
    except BaseException as exc:
        chain, cur = [], exc
        while cur is not None:
            chain.append(cur)
            cur = cur.__cause__ or cur.__context__
        lgb = any(isinstance(e, LightGBMError) for e in chain)
        _rows.append(f"run_raised\t{_current['nodeid']}\t{int(lgb)}\t{type(exc).__name__}")
        raise
    finally:
        round_has = _complete(study.trials, start)
        study_has = _complete(study.trials)
        _rows.append(
            f"run\t{_current['nodeid']}\t{start}\t{int(round_has)}\t{int(study_has)}"
        )


_orig_plan_open = _tuning_plan.plan_open_round


def _plan_open(cfg: Any, **kwargs: Any) -> Any:
    lineage = kwargs["lineage"]
    record = lineage.last_round
    declaration = kwargs.get("space_declaration")
    given = declaration is not None
    differs = given and not same_declared_space(
        declaration.declared, record.declared, lineage.routes
    )
    round_has = _complete(kwargs["trials"], record.start_trial)
    _rows.append(
        f"open\t{_current['nodeid']}\t{int(given)}\t{int(differs)}\t{int(round_has)}"
    )
    return _orig_plan_open(cfg, **kwargs)


_model_tuning.ModelTuningMixin._close_round = _close  # type: ignore[method-assign]
tuner_module.Tuner.run = _run  # type: ignore[method-assign]
_tuning_plan.plan_open_round = _plan_open


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session) -> None:  # noqa: ARG001
    out = os.environ.get("FIRING_RATE_H0128_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(_rows) + "\n")
