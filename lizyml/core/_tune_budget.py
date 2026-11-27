"""_tune_budget -- the opt-in memory budget of ``Model.tune`` (H-0150 part B).

``tune(memory_budget_bytes=N)`` refuses, before it trains, every trial whose
provider estimate is over *N*, whose assessment raises, or whose device is not
the CPU. A refused trial raises ``TUNING_FAILED`` so that the objective's
``catch=`` path records it as an ordinary FAIL; nothing is added to the study,
the trial's attributes or ``TrialResult``. The budget is a per-call argument:
it is never stored and may change on resume.

Three seams are reused so a refusal sees what a trial would train on (DC3):
``trial_inputs`` for the parameters, ``estimate_trial`` for the estimate and
the provider's ``device_is_cpu`` for the device.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

from lizyml.core._model_memory import require_memory_provider
from lizyml.core._trial_inputs import trial_inputs
from lizyml.core._tune_memory_envelope import device_may_leave_cpu, estimate_trial
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import get_logger
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.search_dim import SearchDim

_log = get_logger("core.tune_budget")


@dataclass
class TuneBudget:
    """One ``tune()`` call's budget and its count of refused trials.

    ``limit`` is ``None`` when the call has no budget; ``refused`` counts this
    call only, never a previous process's trials.
    """

    limit: int | None = None
    refused: int = 0

    def context(self) -> dict[str, Any]:
        """The additive ``TUNING_FAILED`` context keys (0 and None unbudgeted)."""
        return {
            "memory_refused_this_call": self.refused,
            "memory_budget_bytes": self.limit,
        }


def validate_budget_value(value: Any) -> int | None:
    """*value* when it is ``None`` or a positive ``int``; else ``CONFIG_INVALID``.

    A ``bool`` is not a budget. Runs on every call, before the round starts.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"memory_budget_bytes must be None or a positive int, got {value!r}."
            ),
            context={"reason": "invalid_memory_budget", "value": repr(value)},
        )
    return value


def admit_budget(
    provider: Any,
    *,
    runs_optimize: bool,
    dims: tuple[SearchDim, ...],
    fixed: dict[str, Any],
    base_model_params: dict[str, Any],
) -> None:
    """Refuse a budgeted round that cannot be checked, before any write.

    Only a round that runs a trial is judged (a no-trial round closes as it
    does without a budget): the provider must have the memory capability
    (``memory_estimate_unsupported``), and no value the round can reach may
    leave the CPU (policy 6 (a)), decided from the declared layers.
    """
    if not runs_optimize:
        return
    capable = require_memory_provider(provider)
    if device_may_leave_cpu(capable, dims, fixed, base_model_params):
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                "memory_budget_bytes is for CPU training, but this round can "
                "reach a non-CPU device (a device_type in the search space, "
                "the round's fixed values or model.params)."
            ),
            context={"reason": "memory_budget_device_not_cpu"},
        )


def _device_text(provider: Any, params: dict[str, Any]) -> str:
    return ",".join(str(v) for k, v in params.items() if provider.is_device_name(k))


def check_trial(
    budget: TuneBudget,
    provider: Any,
    dims: list[SearchDim],
    trial_params: dict[str, Any],
    *,
    base_model_params: dict[str, Any],
    base_smart_params: dict[str, Any],
    fixed: dict[str, Any],
    n_boosters: int,
    build_components: Callable[
        [dict[str, Any], dict[str, Any], dict[str, Any]], TrainComponents
    ],
    X: Any,
) -> None:
    """Raise ``TUNING_FAILED`` (and count it) when the trial must not train.

    No-op without a budget. Any exception of the assessment refuses the trial,
    whatever its type, so one that ``catch=`` would not record cannot stop the
    run.
    """
    limit = budget.limit
    if limit is None:
        return
    try:
        merged_model, _, _ = trial_inputs(
            provider,
            dims,
            trial_params,
            base_model_params=base_model_params,
            base_smart_params=base_smart_params,
            fixed=fixed,
        )
        on_cpu = provider.device_is_cpu(merged_model)
        est = (
            estimate_trial(
                provider,
                dims,
                trial_params,
                base_model_params=base_model_params,
                base_smart_params=base_smart_params,
                fixed=fixed,
                n_boosters=n_boosters,
                build_components=build_components,
                X=X,
            )
            if on_cpu
            else None
        )
    except Exception as exc:  # noqa: BLE001 -- refused and logged, never swallowed
        _refuse(
            budget,
            "assessment_failed",
            f"error={type(exc).__name__}: {exc} budget={limit}",
            f"its memory assessment failed ({type(exc).__name__}: {exc})",
        )
    if est is None:
        device = _device_text(provider, merged_model)
        _refuse(
            budget,
            "device_not_cpu",
            f"device={device} budget={limit}",
            f"its device ({device}) is not the CPU",
        )
    elif est.estimate_bytes > limit:
        _refuse(
            budget,
            "over_budget",
            f"estimate_bytes={est.estimate_bytes} budget={limit} "
            f"num_leaves={est.num_leaves}",
            f"its estimated {est.estimate_bytes} bytes exceed the budget "
            f"{limit} (num_leaves={est.num_leaves})",
        )


def _refuse(budget: TuneBudget, reason: str, log_fields: str, why: str) -> NoReturn:
    budget.refused += 1
    _log.info("event='tune.trial.memory_refused' reason=%s %s", reason, log_fields)
    raise LizyMLError(
        code=ErrorCode.TUNING_FAILED,
        user_message=f"Trial refused by memory_budget_bytes: {why}.",
        context={"reason": reason, "memory_budget_bytes": budget.limit},
    )
