"""Round planning for ``Model.tune`` (H-0122 §5, §6, §7.2).

A tune runs one round of a study. Before anything is written, this module
decides that round completely -- its declared and effective space, boundary
expansion, budget, first trial number and dimension routes -- as a frozen
:class:`RoundPlan`. Three planners cover the three ways a round starts:

* :func:`plan_first_round` -- round 1 of a new study, or of a stored study in
  state E (uninitialized) or N (no round). The space comes from the config.
* :func:`plan_next_round` -- after a closed round (state C or F). The space is
  the previous round's, expanded from the study's best trial.
* :func:`plan_open_round` -- an open round (state O) finished on reattach. The
  round's record decides everything; arguments are only compared (U2).

Pure: reads the config, the provider and the lineage, and writes nothing --
neither the study nor the Model. Every refusal here is ``CONFIG_INVALID`` and
happens before the first write (INV-R4).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from lizyml.core._tuning_validation import admit_smart_dimensions
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import get_logger
from lizyml.core.types.tuning_result import BoundaryReport
from lizyml.tuning.lineage import (
    Lineage,
    Provenance,
    Route,
    RouteConflict,
    StoredRound,
    TrialLike,
    merge_routes,
    remaining_timeout,
    remaining_trials,
    trials_of_round,
)
from lizyml.tuning.search_space import (
    attach_bounds,
    detect_boundary,
    expand_dims,
    parse_space,
)

if TYPE_CHECKING:
    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.search_dim import SearchDim
    from lizyml.estimators.provider import EstimatorProvider

_log = get_logger("model")

PlanKind = Literal["first", "next", "open"]


# ---------------------------------------------------------------------------
# The config space (round 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedSpace:
    """The config's search space resolved by ``space_mode``, without bounds.

    ``provenance`` says, per declared dimension, whether the user wrote it or
    it came from the provider's default space; ``default_dims`` is the same
    information as the set H-0121 decision 4 reads. ``used_default`` is merge
    mode with no user dimension (the H-0102 ``expand_boundary=None`` rule).
    """

    declared: tuple[SearchDim, ...]
    provenance: tuple[Provenance, ...]
    used_default: bool
    fixed: dict[str, Any]
    default_dims: frozenset[tuple[str, str]]


def resolve_config_space(
    cfg: LizyMLConfig, provider: EstimatorProvider
) -> ResolvedSpace:
    """Resolve ``tuning.optuna.space`` by ``space_mode`` (H-0024, H-0121).

    ``replace`` declares the user's dimensions only, with no fixed parameters.
    ``merge`` lays the user's dimensions over the provider's default space by
    parameter identity (a model dimension's alias meets its canonical name)
    and fixes the provider's ``default_fixed_params``.
    """
    assert cfg.tuning is not None  # noqa: S101 -- validated upstream
    user_space = parse_space(cfg.tuning.optuna.space)
    if cfg.tuning.optuna.space_mode == "replace":
        return ResolvedSpace(
            declared=tuple(user_space),
            provenance=tuple(
                Provenance(d.category, d.name, "user") for d in user_space
            ),
            used_default=False,
            fixed={},
            default_dims=frozenset(),
        )
    defaults = provider.default_space(cfg.task)
    canonical = provider.canonical_param_names(
        [dim.name for dim in [*defaults, *user_space] if dim.category == "model"]
    )

    def identity(dim: SearchDim) -> tuple[str, str]:
        return (
            dim.category,
            canonical[dim.name] if dim.category == "model" else dim.name,
        )

    overrides = {identity(dim): dim for dim in user_space}
    default_dims = frozenset(
        (dim.category, dim.name) for dim in defaults if identity(dim) not in overrides
    )
    declared = [overrides.pop(identity(dim), dim) for dim in defaults]
    declared.extend(overrides.values())
    return ResolvedSpace(
        declared=tuple(declared),
        provenance=tuple(
            Provenance(
                d.category,
                d.name,
                "default" if (d.category, d.name) in default_dims else "user",
            )
            for d in declared
        ),
        used_default=not user_space,
        fixed=provider.default_fixed_params(cfg.task),
        default_dims=default_dims,
    )


def route_entries(
    dims: Iterable[SearchDim], provider: EstimatorProvider
) -> tuple[Route, ...]:
    """The route of each dimension (§4.1b).

    ``name`` is the raw dimension name, which is also the Optuna param key.
    Only ``model`` names are canonicalised: canonicalising the training
    dimension ``early_stopping_rounds`` would give LightGBM's
    ``early_stopping_round``.
    """
    dims = list(dims)
    canonical = provider.canonical_param_names(
        [d.name for d in dims if d.category == "model"]
    )
    return tuple(
        Route(
            name=d.name,
            canonical=canonical[d.name] if d.category == "model" else d.name,
            category=d.category,
        )
        for d in dims
    )


def _admit_routes(existing: Mapping[str, Route], added: Iterable[Route]) -> None:
    """Refuse a route that would re-route a name (§4.1b; append-only)."""
    try:
        merge_routes(existing, added)
    except RouteConflict as exc:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"tuning.optuna.space: {exc}. A dimension's category cannot change "
                "within a study; start a new study (another study_name)."
            ),
            context={"surface": "tuning.optuna.space", "reason": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundPlan:
    """Everything round ``round_number`` does, decided before any write.

    ``budget_*`` is what the round record says the round asked for;
    ``run_*`` is what this call runs (the remainder, for an open round).
    """

    kind: PlanKind
    round_number: int
    declared: tuple[SearchDim, ...]
    provenance: tuple[Provenance, ...]
    effective: tuple[SearchDim, ...]
    used_default: bool
    fixed: Mapping[str, Any]
    expand_boundary: bool
    boundary_threshold: float
    boundary_report: BoundaryReport | None
    budget_n_trials: int
    budget_timeout: float | None
    start_trial: int
    best_before: float | None
    routes_added: tuple[Route, ...]
    run_n_trials: int
    run_timeout: float | None
    running_to_fail: tuple[int, ...] = ()

    @property
    def expanded_names(self) -> tuple[str, ...]:
        """The dimensions this round's report expanded."""
        report = self.boundary_report
        return report.expanded_names if report is not None else ()

    @property
    def runs_optimize(self) -> bool:
        """Whether any trial budget and time remain (§7.2: else close at once)."""
        return self.run_n_trials > 0 and (
            self.run_timeout is None or self.run_timeout > 0
        )

    def round_record(self) -> StoredRound:
        """The ``lizyml:round:<k>`` record this plan writes (not for ``open``)."""
        return StoredRound(
            round=self.round_number,
            source="config" if self.round_number == 1 else "previous",
            space_arg=None,
            provenance=self.provenance,
            declared=self.declared,
            effective=self.effective,
            expand_boundary=self.expand_boundary,
            boundary_threshold=self.boundary_threshold,
            boundary_report=self.boundary_report,
            n_trials=self.budget_n_trials,
            timeout=self.budget_timeout,
            start_trial=self.start_trial,
            best_before=self.best_before,
            routes_added=self.routes_added,
        )


def _config_budget(cfg: LizyMLConfig) -> tuple[int, float | None]:
    assert cfg.tuning is not None  # noqa: S101 -- validated upstream
    params = cfg.tuning.optuna.params
    return params.n_trials, params.timeout


def _uses_default(space_mode: str, provenance: Sequence[Provenance]) -> bool:
    """H-0102: merge mode with no user dimension, from the recorded provenance."""
    return space_mode == "merge" and not any(p.origin == "user" for p in provenance)


def plan_first_round(
    cfg: LizyMLConfig,
    provider: EstimatorProvider,
    *,
    resolved: ResolvedSpace,
    fixed: Mapping[str, Any],
    base_smart: dict[str, Any],
    n_trials: int | None,
    expand_boundary: bool | None,
    boundary_threshold: float,
) -> RoundPlan:
    """Round 1 from the config space (a new study, or state E / N).

    H-0121 runs here and only here: a smart dimension nothing would read is
    refused when the user wrote it and dropped when it is a default, before
    the round records its declared space. Later rounds inherit the drop.
    """
    declared, dropped = admit_smart_dimensions(
        provider, list(resolved.declared), base_smart, resolved.default_dims
    )
    if dropped:
        _log.info(
            "event='tune.drop_dims' reason=inactive_smart_owner dims=%s",
            ",".join(dim.name for dim in dropped),
        )
    kept = {(d.category, d.name) for d in declared}
    routes = route_entries(declared, provider)
    _admit_routes({}, routes)
    config_n_trials, timeout = _config_budget(cfg)
    budget = n_trials if n_trials is not None else config_n_trials
    return RoundPlan(
        kind="first",
        round_number=1,
        declared=tuple(declared),
        provenance=tuple(
            p for p in resolved.provenance if (p.category, p.name) in kept
        ),
        effective=tuple(attach_bounds(declared, provider.parameter_bounds(cfg.task))),
        used_default=resolved.used_default,
        fixed=dict(fixed),
        expand_boundary=(
            expand_boundary if expand_boundary is not None else resolved.used_default
        ),
        boundary_threshold=boundary_threshold,
        boundary_report=None,
        budget_n_trials=budget,
        budget_timeout=timeout,
        start_trial=0,
        best_before=None,
        routes_added=routes,
        run_n_trials=budget,
        run_timeout=timeout,
    )


def expand_for_next_round(
    previous: Sequence[SearchDim],
    best_params: Mapping[str, Any],
    threshold: float,
    *,
    frozen_names: frozenset[str] = frozenset(),
) -> tuple[tuple[SearchDim, ...], BoundaryReport]:
    """Detect near-edge dimensions and expand them (H-0068, H-0078, H-0111).

    The report is returned even when nothing expanded. ``frozen_names`` are
    reported with ``expanded=False`` and left as they are: the hook §5.3
    bullets 1-2 need (a changed declaration, a best outside the range), which
    only ``space=`` can produce (PR 2).
    """
    report = detect_boundary(list(previous), dict(best_params), threshold)
    if frozen_names:
        report = BoundaryReport(
            dims=tuple(
                dataclasses.replace(
                    s,
                    expanded=False,
                    new_low=None,
                    new_high=None,
                    clamped_to_bound=False,
                )
                if s.name in frozen_names
                else s
                for s in report.dims
            ),
            expanded_names=tuple(
                n for n in report.expanded_names if n not in frozen_names
            ),
        )
    if not report.expanded_names:
        # Either no dim is near an edge, or every near-edge dim already sits on
        # a clamp and cannot move (H-0111).
        _log.info("event='tune.resume' no dims expanded")
        return tuple(previous), report
    expanded = expand_dims(list(previous), report)
    for status in report.dims:
        if status.name in report.expanded_names:
            _log.info(
                "event='tune.expand' dim=%s edge=%s old=[%s, %s] new=[%s, %s] best=%s",
                status.name,
                status.edge,
                status.low,
                status.high,
                status.new_low,
                status.new_high,
                status.best_value,
            )
    return tuple(expanded), report


def plan_next_round(
    cfg: LizyMLConfig,
    *,
    lineage: Lineage,
    n_study_trials: int,
    best_params: Mapping[str, Any] | None,
    best_value: float | None,
    n_trials: int | None,
    expand_boundary: bool | None,
    boundary_threshold: float,
) -> RoundPlan:
    """A new round after a closed one (state C or F).

    The declaration is the previous round's; the previous best is the study's
    best COMPLETE trial, never the Model's tuning result (§5.3). A study in
    state F has no COMPLETE trial, so nothing is expanded.
    """
    study, last = lineage.study, lineage.last_round
    assert study is not None and last is not None  # noqa: S101 -- C/F only
    used_default = _uses_default(study.space_mode, last.provenance)
    should = expand_boundary if expand_boundary is not None else used_default
    effective: tuple[SearchDim, ...] = last.effective
    report: BoundaryReport | None = None
    if not should:
        _log.info("event='tune.resume' expand_boundary=False")
    elif best_params is not None:
        effective, report = expand_for_next_round(
            last.effective, best_params, boundary_threshold
        )
    config_n_trials, timeout = _config_budget(cfg)
    budget = n_trials if n_trials is not None else config_n_trials
    return RoundPlan(
        kind="next",
        round_number=last.round + 1,
        declared=last.declared,
        provenance=last.provenance,
        effective=effective,
        used_default=used_default,
        fixed=dict(study.fixed),
        expand_boundary=should,
        boundary_threshold=boundary_threshold,
        boundary_report=report,
        budget_n_trials=budget,
        budget_timeout=timeout,
        start_trial=n_study_trials,
        best_before=best_value if best_params is not None else None,
        routes_added=(),
        run_n_trials=budget,
        run_timeout=timeout,
    )


def _refuse_changed_argument(k: int, name: str, recorded: Any, given: Any) -> None:
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"round {k} was started with {name}={recorded!r} and is not finished; "
            f"finish it without {name} or with the same value (got {given!r})."
        ),
        context={
            "surface": f"tune({name}=)",
            "round": k,
            "recorded": recorded,
            "given": given,
        },
    )


def plan_open_round(
    cfg: LizyMLConfig,
    *,
    lineage: Lineage,
    trials: Sequence[TrialLike],
    n_trials: int | None,
    expand_boundary: bool | None,
    boundary_threshold: float,
) -> RoundPlan:
    """Finish an open round (state O) with its remaining budget (§7.2).

    The round's record decides its space, report and budget. U2: an
    ``n_trials`` or ``expand_boundary`` that is not ``None`` must equal the
    recorded value. Trials left RUNNING are FAILed before the round runs, so
    they count as used.
    """
    study, record = lineage.study, lineage.last_round
    assert study is not None and record is not None  # noqa: S101 -- O only
    k = record.round
    if n_trials is not None and n_trials != record.n_trials:
        _refuse_changed_argument(k, "n_trials", record.n_trials, n_trials)
    if expand_boundary is not None and expand_boundary != record.expand_boundary:
        _refuse_changed_argument(
            k, "expand_boundary", record.expand_boundary, expand_boundary
        )
    config_n_trials, config_timeout = _config_budget(cfg)
    if config_timeout != record.timeout:
        _log.info(
            "event='tune.resume_open' round=%d ignored=timeout given=%s recorded=%s",
            k,
            config_timeout,
            record.timeout,
        )
    if n_trials is None and config_n_trials != record.n_trials:
        warnings.warn(
            f"tuning.optuna.params.n_trials={config_n_trials} is ignored: round {k} "
            f"is open and finishes with its recorded n_trials={record.n_trials}.",
            UserWarning,
            stacklevel=3,
        )
    if boundary_threshold != record.boundary_threshold:
        _log.info(
            "event='tune.resume_open' round=%d ignored=boundary_threshold "
            "given=%s recorded=%s",
            k,
            boundary_threshold,
            record.boundary_threshold,
        )
    round_trials = trials_of_round(lineage, k, trials)
    return RoundPlan(
        kind="open",
        round_number=k,
        declared=record.declared,
        provenance=record.provenance,
        effective=record.effective,
        used_default=_uses_default(study.space_mode, record.provenance),
        fixed=dict(study.fixed),
        expand_boundary=record.expand_boundary,
        boundary_threshold=record.boundary_threshold,
        boundary_report=record.boundary_report,
        budget_n_trials=record.n_trials,
        budget_timeout=record.timeout,
        start_trial=record.start_trial,
        best_before=record.best_before,
        routes_added=record.routes_added,
        run_n_trials=remaining_trials(record, round_trials),
        run_timeout=remaining_timeout(record, round_trials),
        running_to_fail=tuple(
            t.number for t in round_trials if t.state.name == "RUNNING"
        ),
    )
