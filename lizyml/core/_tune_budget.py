"""_tune_budget -- the opt-in memory budget of ``Model.tune`` (H-0150 part B).

``tune(memory_budget_bytes=N)`` refuses, before it trains, every trial whose
provider estimate is over *N*, whose assessment raises, or whose device is not
the CPU. A refused trial raises ``TUNING_FAILED`` so that the objective's
``catch=`` path records it as an ordinary FAIL; nothing is added to the study,
the trial's attributes or ``TrialResult``. The budget is a per-call argument:
it is never stored and may change on resume.

A trial's parameters and train components are built once (``build_trial``),
and the device check and the estimate read the components that trial trains
with (DC3): the parameters come from ``trial_inputs``, the device from the
provider's ``finalized_device`` (what the adapter hands the library), judged
by its ``device_is_cpu``, and the estimate from its ``estimate_memory``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

from lizyml.core._model_memory import require_memory_provider
from lizyml.core._trial_inputs import trial_inputs
from lizyml.core._tune_memory_envelope import (
    UNSUPPORTED,
    device_may_leave_cpu,
    estimate_trial,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import get_logger
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.memory_estimate import MemoryEstimate
from lizyml.core.types.search_dim import SearchDim
from lizyml.estimators.provider import MemoryEstimateProvider
from lizyml.tuning.trial_failure import describe_exception

_log = get_logger("core.tune_budget")

#: ``(model_params, smart_params, training_overrides) -> TrainComponents``.
BuildComponents = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]], TrainComponents
]


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


@dataclass(frozen=True)
class TrialBuild:
    """The parameters and train components one trial trains with."""

    model: dict[str, Any]
    smart: dict[str, Any]
    training: dict[str, Any]
    components: TrainComponents


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
    base_smart_params: dict[str, Any],
    n_boosters: int,
    build_components: BuildComponents,
    X: Any,
) -> None:
    """Refuse a budgeted round that cannot be checked, before any write.

    Only a round that runs a trial is judged (a no-trial round closes as it
    does without a budget). The provider must have the memory capability and
    estimate the adapter it builds (``memory_estimate_unsupported``, policy
    5), and no value the round can reach may leave the CPU (policy 6 (a)),
    decided from the declared layers.
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
    probe_adapter(
        capable,
        dims,
        base_model_params=base_model_params,
        base_smart_params=base_smart_params,
        fixed=fixed,
        n_boosters=n_boosters,
        build_components=build_components,
        X=X,
    )


def probe_adapter(
    provider: MemoryEstimateProvider,
    dims: tuple[SearchDim, ...],
    *,
    base_model_params: dict[str, Any],
    base_smart_params: dict[str, Any],
    fixed: dict[str, Any],
    n_boosters: int,
    build_components: BuildComponents,
    X: Any,
) -> None:
    """Estimate one representative point -- the base parameters with no
    sampled value -- and re-raise only ``memory_estimate_unsupported``.

    The factory builds the same kind of adapter for every trial, so an
    adapter the provider cannot estimate shows here. Any other failure of
    this point is logged and left to the per-trial guard, which refuses each
    trial it hits.
    """
    try:
        estimate_trial(
            provider,
            list(dims),
            {},
            base_model_params=base_model_params,
            base_smart_params=base_smart_params,
            fixed=fixed,
            n_boosters=n_boosters,
            build_components=build_components,
            X=X,
        )
    except LizyMLError as exc:
        if exc.context.get("reason") == UNSUPPORTED:
            raise
        _probe_failed(exc)
    except Exception as exc:  # noqa: BLE001 -- logged; each trial is judged
        _probe_failed(exc)


def _probe_failed(exc: BaseException) -> None:
    _log.debug(
        "event='tune.budget.probe_failed' error=%s (left to the trial guard)",
        describe_exception(exc),
    )


def build_trial(
    budget: TuneBudget,
    provider: Any,
    dims: list[SearchDim],
    trial_params: dict[str, Any],
    *,
    base_model_params: dict[str, Any],
    base_smart_params: dict[str, Any],
    fixed: dict[str, Any],
    n_boosters: int,
    build_components: BuildComponents,
    X: Any,
) -> TrialBuild:
    """Build the trial once; with a budget, refuse it before it trains.

    Without a budget this is the build alone, and its exceptions propagate as
    before. With one, any exception of the build or of the assessment, of any
    type, refuses the trial (``assessment_failed``), so one that ``catch=``
    would not record cannot stop the run. Then the finalized device
    (``device_not_cpu``) and the estimate (``over_budget``) are judged.

    Raises:
        LizyMLError: ``TUNING_FAILED`` for a refused trial, counted in
            *budget*.
    """

    def build() -> TrialBuild:
        model_p, smart_p, training_p = trial_inputs(
            provider,
            dims,
            trial_params,
            base_model_params=base_model_params,
            base_smart_params=base_smart_params,
            fixed=fixed,
        )
        tc = build_components(model_p, smart_p, training_p)
        return TrialBuild(model_p, smart_p, training_p, tc)

    limit = budget.limit
    if limit is None:
        return build()
    est: MemoryEstimate | None = None
    device_text = ""
    try:
        built = build()
        factory = built.components.estimator_factory
        device = provider.finalized_device(estimator_factory=factory)
        device_text = ",".join(str(v) for v in device.values())
        if provider.device_is_cpu(device):
            est = provider.estimate_memory(
                estimator_factory=factory,
                X=X,
                smart_params=built.smart,
                n_boosters=n_boosters,
            )
    except Exception as exc:  # noqa: BLE001 -- refused and logged, never swallowed
        error = describe_exception(exc)
        _refuse(
            budget,
            "assessment_failed",
            f"error={error}",
            "the memory assessment failed",
        )
    if est is None:
        _refuse(
            budget,
            "device_not_cpu",
            f"device={device_text}",
            "the finalized device is not the CPU",
        )
    if est.estimate_bytes > limit:
        _refuse(
            budget,
            "over_budget",
            f"estimate_bytes={est.estimate_bytes} num_leaves={est.num_leaves}",
            "the estimate is over the budget",
        )
    return built


def _refuse(budget: TuneBudget, reason: str, fields: str, why: str) -> NoReturn:
    """Count, log and raise one refusal; the reason and the budget lead both
    the log line and the message, so a cut ``first_error`` keeps them."""
    budget.refused += 1
    _log.info(
        "event='tune.trial.memory_refused' reason=%s budget=%s %s",
        reason,
        budget.limit,
        fields,
    )
    raise LizyMLError(
        code=ErrorCode.TUNING_FAILED,
        user_message=(
            f"Trial refused by memory_budget_bytes: reason={reason} "
            f"budget={budget.limit} {fields} ({why})."
        ),
        context={"reason": reason, "memory_budget_bytes": budget.limit},
    )
