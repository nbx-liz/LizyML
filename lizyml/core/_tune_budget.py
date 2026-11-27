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

``finalized_device`` is the separate :class:`TrialDeviceProvider` capability:
only a budget requires it, so the two estimates need only
:class:`MemoryEstimateProvider`.

The budget's domain is bounded: ``1 <= memory_budget_bytes <= 2**63 - 1``
(:data:`MAX_MEMORY_BUDGET_BYTES`), so a valid budget always prints. An invalid
value is rendered by :func:`value_text`, and a provider's estimate (which the
bound does not cover) by :func:`int_text`; neither can raise.
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
    evaluate_envelope,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import get_logger
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.memory_estimate import MemoryEstimate
from lizyml.core.types.search_dim import SearchDim
from lizyml.estimators.provider import MemoryEstimateProvider, TrialDeviceProvider
from lizyml.tuning.trial_failure import MAX_TEXT, describe_exception

_log = get_logger("core.tune_budget")

#: ``(model_params, smart_params, training_overrides) -> TrainComponents``.
BuildComponents = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]], TrainComponents
]


#: The largest ``memory_budget_bytes``: a signed 64-bit integer.
MAX_MEMORY_BUDGET_BYTES = 2**63 - 1

#: An ``int`` wider than this many bits is written by its bit length.
_INT_BITS = 64


def int_text(value: int) -> str:
    """*value* in decimal when it fits in 64 bits, else its sign and bit
    length. Never raises: ``str()`` of a wide ``int`` can (Python's limit on
    integer-string conversion), so it is never called for one."""
    bits = value.bit_length()
    if bits <= _INT_BITS:
        return str(value)
    sign = "-" if value < 0 else ""
    return f"<{sign}int of {bits} bits>"


def value_text(value: Any) -> str:
    """A rendering of any *value* that never raises.

    An ``int`` (not a ``bool``) goes through :func:`int_text`; anything else
    is its ``repr`` cut to :data:`MAX_TEXT`. Both carry the type name, and a
    ``repr`` that raises or returns a non-str gives ``<unprintable TYPE>``.
    """
    name = _type_name(value)
    try:
        if isinstance(value, int) and not isinstance(value, bool):
            text = int_text(value)
            return text if text.startswith("<") else f"{text} ({name})"
        text = repr(value)
    except Exception:  # noqa: BLE001 - the rendering must not raise
        return f"<unprintable {name}>"
    if not isinstance(text, str):
        return f"<unprintable {name}>"
    return f"{text[:MAX_TEXT]} ({name})"


def _type_name(value: Any) -> str:
    """The type name of *value*, cut to :data:`MAX_TEXT`; never raises."""
    try:
        name = type.__getattribute__(type(value), "__name__")
    except Exception:  # noqa: BLE001 - the rendering must not raise
        return "<unknown type>"
    return name[:MAX_TEXT] if isinstance(name, str) else "<unknown type>"


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
    """*value* when it is ``None`` or an ``int`` from 1 to ``2**63 - 1``
    (:data:`MAX_MEMORY_BUDGET_BYTES`); else ``CONFIG_INVALID`` (``reason:
    "invalid_memory_budget"``).

    A ``bool`` is not a budget. An ``int`` subclass is taken as the exact
    ``int`` it holds (``int.__int__``, which no subclass can override), so
    the returned budget always prints. Runs on every call, before the round
    is planned. The message and the context's ``value`` render *value*
    through :func:`value_text`, so the refusal itself cannot raise.
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        plain = int.__int__(value)
        if 1 <= plain <= MAX_MEMORY_BUDGET_BYTES:
            return plain
    text = value_text(value)
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            "memory_budget_bytes must be None or an int from 1 to "
            f"2**63 - 1, got {text}."
        ),
        context={"reason": "invalid_memory_budget", "value": text},
    )


def require_trial_device_provider(provider: Any) -> TrialDeviceProvider:
    """*provider* when it has the :class:`TrialDeviceProvider` capability, else
    ``CONFIG_INVALID`` (``reason: "memory_estimate_unsupported"``): without the
    finalized device a budgeted trial's device cannot be checked (DC1).

    ``finalized_device`` must also be callable: the runtime protocol accepts
    any attribute of that name (``False`` included), and a non-callable one
    would fail every trial after the round was written.
    """
    if not isinstance(provider, TrialDeviceProvider) or not callable(
        getattr(provider, "finalized_device", None)
    ):
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                "This model's estimator provider cannot report the device a "
                "trial trains on; memory_budget_bytes needs the "
                "TrialDeviceProvider capability."
            ),
            context={
                "reason": "memory_estimate_unsupported",
                "provider": type(provider).__name__,
            },
        )
    return provider


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
    does without a budget). The provider must have both the memory and the
    device capability, and estimate the adapter every candidate point builds
    (``memory_estimate_unsupported``, policy 5), and no value the round can
    reach may leave the CPU (policy 6 (a)), decided from the declared layers.
    """
    if not runs_optimize:
        return
    capable = require_memory_provider(provider)
    require_trial_device_provider(provider)
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
    """Estimate every candidate point of the next round's envelope -- the
    points ``estimate_tune_memory`` evaluates (``evaluate_envelope``: the
    corners, every categorical choice, the provider's extra candidates) --
    and let only ``memory_estimate_unsupported`` propagate.

    The bound: only these points are probed. A factory may build a different
    adapter for other parameter values, so a foreign adapter that appears
    only at another sampled point is not seen here; the per-trial guard
    refuses that trial as ``assessment_failed``. Any other failure of a
    candidate point is logged and left to the per-trial guard as well.

    Raises:
        LizyMLError: ``CONFIG_INVALID`` (``reason:
            "memory_estimate_unsupported"``) when a candidate point builds an
            adapter the provider cannot estimate.
    """
    try:
        found = evaluate_envelope(
            provider,
            dims,
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
        _probe_failed(describe_exception(exc))
        return
    except Exception as exc:  # noqa: BLE001 -- logged; each trial is judged
        # Outside a point (the candidate values themselves), left to the
        # per-trial guard like a point's failure.
        _probe_failed(describe_exception(exc))
        return
    for failure in found.failures:
        _probe_failed(failure)


def _probe_failed(error: str) -> None:
    _log.debug(
        "event='tune.budget.probe_failed' error=%s (left to the trial guard)",
        error,
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
            f"estimate_bytes={int_text(est.estimate_bytes)} "
            f"num_leaves={int_text(est.num_leaves)}",
            "the estimate is over the budget",
        )
    return built


def _refuse(budget: TuneBudget, reason: str, fields: str, why: str) -> NoReturn:
    """Count, log and raise one refusal; the reason and the budget lead both
    the log line and the message, so a cut ``first_error`` keeps them.

    The budget is within :data:`MAX_MEMORY_BUDGET_BYTES`, so it prints, and
    the trial error's context carries it as an ``int``: Optuna logs ``repr()``
    of every FAIL trial's error.
    """
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
