"""_tune_memory_envelope -- the memory envelope over a tune round (H-0150).

Given the round a tune would run next, this builds the candidate points whose
memory estimate bounds every ordinary CPU trial of that round from above, and
evaluates each one through the path a trial takes (``trial_inputs`` ->
``_build_train_components`` -> the provider's ``estimate_memory``). LightGBM
knowledge stays in the provider: which dimensions matter (``memory_role``),
which special values a range holds beyond its ends (``memory_candidates``),
what the device is (``device_is_cpu`` / ``is_device_name``) and how a point is
estimated (``estimate_memory``).

Why the corners suffice: the estimate is non-decreasing in each
``"increasing"`` dimension and does not read a ``"none"`` one, as long as the
finalized round count and leaf count are at least 1 -- a point where either is
lower cannot be evaluated and makes the result incomplete. A categorical
dimension enumerates every choice, so its role only decides whether the
provider knows the name.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from lizyml.core._trial_inputs import trial_inputs
from lizyml.core.exceptions import LizyMLError
from lizyml.core.train_components import TrainComponents
from lizyml.core.types.memory_estimate import MemoryEstimate
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim, SearchDim
from lizyml.estimators.provider import MemoryEstimateProvider
from lizyml.tuning._step_grid import grid_value

#: The refusal a provider raises for an adapter it cannot estimate (policy 3).
UNSUPPORTED = "memory_estimate_unsupported"


@dataclass(frozen=True)
class Envelope:
    """The result of evaluating every candidate point."""

    envelope: MemoryEstimate | None
    unresolved: tuple[str, ...]
    n_candidates: int
    failures: tuple[str, ...]


@dataclass(frozen=True)
class Candidates:
    """The values one dimension contributes to the candidate points.

    *named*: the values are end values or choices, so a point that fails to
    evaluate names the dimension. *undecided*: the provider cannot say how the
    estimate depends on the name, so the round is incomplete.
    """

    values: tuple[Any, ...]
    named: bool
    undecided: bool


def device_may_leave_cpu(
    provider: MemoryEstimateProvider,
    dims: tuple[SearchDim, ...],
    fixed: Mapping[str, Any],
    base_model_params: Mapping[str, Any],
) -> bool:
    """Whether a trial of the round may train on a non-CPU device (policy 6a).

    Decided from the declared layers alone, before any candidate is assembled,
    so an unrelated assembly failure cannot hide it. The top layer that names
    the device wins: the sampled dimensions (every choice of a categorical one;
    a numeric one cannot be decided, so it counts as non-CPU), then the round's
    fixed values, then ``model.params``; a device named nowhere is the CPU.
    """
    sampled = [d for d in dims if provider.is_device_name(d.name)]
    if sampled:
        for dim in sampled:
            if not isinstance(dim, CategoricalDim):
                return True
            if any(not provider.device_is_cpu({dim.name: c}) for c in dim.choices):
                return True
        return False
    for layer in (fixed, base_model_params):
        named = {k: v for k, v in layer.items() if provider.is_device_name(k)}
        if named:
            return not provider.device_is_cpu(named)
    return False


def _numeric_ends(dim: FloatDim | IntDim) -> list[Any]:
    """Both ends of a numeric dimension, on the grid when it has a step."""
    if dim.step is not None:
        return [grid_value(dim, dim.low), grid_value(dim, dim.high)]
    return [dim.low, dim.high]


def candidate_values(provider: MemoryEstimateProvider, dim: SearchDim) -> Candidates:
    """The candidate values of one dimension (policy 4).

    A categorical dimension: every choice, whatever its role, and undecided
    when the provider does not know the name. A numeric ``"increasing"`` one:
    both ends (stepped ones through the objective's grid) plus the provider's
    extra values for the range (``memory_candidates``). Any other numeric one:
    its lower end, undecided unless the role is ``"none"``.
    """
    role = provider.memory_role(dim.name)
    if isinstance(dim, CategoricalDim):
        return Candidates(tuple(dim.choices), named=True, undecided=role is None)
    if role != "increasing":
        lower = _numeric_ends(dim)[0]
        return Candidates((lower,), named=False, undecided=role != "none")
    extra = provider.memory_candidates(dim.name, dim.low, dim.high)
    values = tuple(dict.fromkeys([*_numeric_ends(dim), *extra]))
    return Candidates(values, named=True, undecided=False)


def evaluate_envelope(
    provider: MemoryEstimateProvider,
    dims: tuple[SearchDim, ...],
    *,
    base_model_params: dict[str, Any],
    base_smart_params: dict[str, Any],
    fixed: dict[str, Any],
    n_boosters: int,
    build_components: Callable[
        [dict[str, Any], dict[str, Any], dict[str, Any]], TrainComponents
    ],
    X: Any,
) -> Envelope:
    """Evaluate every candidate point of *dims* and return the largest estimate.

    Each point whose assembly or estimate raises is one ``failures`` entry,
    names the dimensions that supplied end values or choices at it, and
    leaves the result incomplete; so does a dimension outside the provider's
    closed set.

    Raises:
        LizyMLError: ``CONFIG_INVALID`` (``reason:
            "memory_estimate_unsupported"``) when the provider cannot estimate
            the adapter a point builds: a refusal of the estimate, not a point
            that failed (policy 3).
    """
    per_dim = [candidate_values(provider, d) for d in dims]
    unresolved = [d.name for d, c in zip(dims, per_dim, strict=True) if c.undecided]
    named = [d.name for d, c in zip(dims, per_dim, strict=True) if c.named]
    best: MemoryEstimate | None = None
    failures: list[str] = []
    evaluated = 0
    for point in itertools.product(*(c.values for c in per_dim)):
        trial_params = {d.name: v for d, v in zip(dims, point, strict=True)}
        try:
            est = estimate_trial(
                provider,
                list(dims),
                trial_params,
                base_model_params=base_model_params,
                base_smart_params=base_smart_params,
                fixed=fixed,
                n_boosters=n_boosters,
                build_components=build_components,
                X=X,
            )
        except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
            if (
                isinstance(exc, LizyMLError)
                and exc.context.get("reason") == UNSUPPORTED
            ):
                raise
            failures.append(f"{type(exc).__name__}: {exc}")
            unresolved.extend(named)
            continue
        evaluated += 1
        if best is None or est.estimate_bytes > best.estimate_bytes:
            best = est
    return Envelope(
        envelope=best,
        unresolved=tuple(dict.fromkeys(unresolved)),
        n_candidates=evaluated,
        failures=tuple(failures),
    )


def estimate_trial(
    provider: MemoryEstimateProvider,
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
) -> MemoryEstimate:
    """The estimate of the one trial that sampled *trial_params*.

    The path a tune trial takes: its parameters (``trial_inputs``), its train
    components, then the provider's estimate. Callable per trial.
    """
    merged_model, merged_smart, training_p = trial_inputs(
        provider,
        dims,
        trial_params,
        base_model_params=base_model_params,
        base_smart_params=base_smart_params,
        fixed=fixed,
    )
    tc = build_components(merged_model, merged_smart, training_p)
    return provider.estimate_memory(
        estimator_factory=tc.estimator_factory,
        X=X,
        smart_params=merged_smart,
        n_boosters=n_boosters,
    )
