"""_tune_round_close -- the round's end record of ``Model.tune`` (H-0128, H-0150).

Moved out of ``_model_tuning`` (file cap); ``ModelTuningMixin._close_round``
delegates here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from lizyml.core._tune_budget import TuneBudget
from lizyml.core._tuning_plan import RoundPlan
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning import tuner as _tuner
from lizyml.tuning.lineage import StoredRoundEnd, encode_round_end, end_key
from lizyml.tuning.trial_failure import round_trial_states, states_text


def close_round(
    study: Any,
    plan: RoundPlan,
    *,
    first_error: str | None = None,
    budget: TuneBudget | None = None,
    interrupted: int = 0,
) -> None:
    """Write ``lizyml:round:<k>:end`` (write position 4).

    ``completed`` when **the round** holds a COMPLETE trial (its trials
    are numbered from ``plan.start_trial``), else ``failed`` followed by
    ``TUNING_FAILED`` (H-0128 policy 1; H-0122 §7.1 looked at the whole
    study). The Model is left unchanged, and the next ``resume=True``
    starts a new round. The error reports the round's trial states and,
    when this call saw one, the first objective error (policy 2). Its
    context also carries ``memory_refused_this_call`` and
    ``memory_budget_bytes`` (H-0150): this call only, ``0`` and ``None``
    without a budget.
    """
    trials = study.trials
    complete = any(
        t.state.name == "COMPLETE" and t.number >= plan.start_trial for t in trials
    )
    ended_at = datetime.now(tz=timezone.utc).isoformat()
    end = (
        StoredRoundEnd("completed", float(study.best_value), ended_at)
        if complete
        else StoredRoundEnd("failed", None, ended_at)
    )
    _tuner.write_study_attr(study, end_key(plan.round_number), encode_round_end(end))
    if complete:
        return
    k = plan.round_number
    states = round_trial_states(trials, plan.start_trial, interrupted=interrupted)
    first = f" First error: {first_error}." if first_error else ""
    earlier = (
        " The study keeps the results of earlier rounds."
        if any(t.state.name == "COMPLETE" for t in trials)
        else ""
    )
    context: dict[str, Any] = {
        "round": k,
        "n_trials": plan.budget_n_trials,
        "status": "failed",
        "trial_states": states,
        **(budget or TuneBudget()).context(),
    }
    if first_error:
        context["first_error"] = first_error
    raise LizyMLError(
        code=ErrorCode.TUNING_FAILED,
        user_message=(
            f"Round {k} ended with no completed trial ("
            + states_text(
                states, n_trials=plan.budget_n_trials, timeout=plan.budget_timeout
            )
            + f").{first}{earlier} Check the parameter ranges, then start "
            f"round {k + 1} with tune(resume=True, space=...)."
        ),
        context=context,
    )
