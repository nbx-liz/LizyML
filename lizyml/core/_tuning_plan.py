"""Round planning for ``Model.tune`` (H-0122 §5, §6, §7.2).

A tune runs one round of a study. Before anything is written, this module
decides that round completely -- its declared and effective space, boundary
expansion, budget, first trial number and dimension routes -- as a frozen
:class:`RoundPlan`. :func:`plan_round` dispatches on the study's state to the
three planners:

* :func:`plan_first_round` -- round 1 of a new study, or of a stored study in
  state E (uninitialized) or N (no round). The space comes from the config.
* :func:`plan_next_round` -- after a closed round (state C or F). The space is
  the previous round's, or a ``tune(space=...)`` declaration, expanded from
  the study's best trial.
* :func:`plan_open_round` -- an open round (state O) finished on reattach. The
  round's record decides everything; arguments are only compared (U2).

Pure: reads the config, the provider, the lineage and the study's best trial,
and writes nothing -- neither the study nor the Model. Every refusal here is
``CONFIG_INVALID`` and happens before the first write (INV-R4).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from lizyml.core._tuning_space import (
    Declaration,
    ResolvedSpace,
    admit_routes,
    admit_smart,
    carry_forward,
    outside_range,
    refuse_unchangeable,
    resolve_declaration,
    resolve_space,
    route_entries,
    same_declared_space,
    space_summary,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import get_logger
from lizyml.core.types.tuning_result import BoundaryReport
from lizyml.tuning.lineage import (
    Lineage,
    Provenance,
    RoundSource,
    Route,
    StoredRound,
    TrialLike,
    remaining_timeout,
    remaining_trials,
    trials_of_round,
)
from lizyml.tuning.rounds import recorded_trial_params
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

__all__ = [
    "ResolvedSpace",
    "RoundPlan",
    "expand_for_next_round",
    "plan_first_round",
    "plan_next_round",
    "plan_open_round",
    "plan_round",
    "resolve_config_space",
    "route_entries",
]

_log = get_logger("model")

PlanKind = Literal["first", "next", "open"]


# ---------------------------------------------------------------------------
# The config space (round 1)
# ---------------------------------------------------------------------------


def resolve_config_space(
    cfg: LizyMLConfig, provider: EstimatorProvider
) -> ResolvedSpace:
    """Resolve ``tuning.optuna.space`` by ``space_mode`` (H-0024, H-0121).

    The config's declaration through :func:`resolve_space`, the same
    resolution a ``tune(space=...)`` round gets.
    """
    assert cfg.tuning is not None  # noqa: S101 -- validated upstream
    return resolve_space(
        cfg.tuning.optuna.space_mode,
        parse_space(cfg.tuning.optuna.space),
        provider,
        cfg.task,
    )


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundPlan:
    """Everything round ``round_number`` does, decided before any write.

    ``budget_*`` is what the round record says the round asked for;
    ``run_*`` is what this call runs (the remainder, for an open round).
    ``source`` / ``space_arg`` say where the declaration came from: the
    config (round 1), the previous round, or ``tune(space=...)``.
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
    source: RoundSource
    space_arg: tuple[SearchDim, ...] | None = None
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
            source=self.source,
            space_arg=self.space_arg,
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

    H-0121 runs before the round records its declared space: a smart
    dimension nothing would read is refused when the user wrote it and
    dropped when it is a default. Later rounds inherit the drop, or apply the
    rule again to a ``space=`` declaration.
    """
    declared = admit_smart(
        provider, resolved.declared, base_smart, resolved.default_dims, task=cfg.task
    )
    kept = {(d.category, d.name) for d in declared}
    routes = route_entries(declared, provider)
    admit_routes({}, routes)
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
        source="config",
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
    reported with ``expanded=False`` and left as they are (§5.3 bullets 1-2:
    a changed declaration, a best outside the range).
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
    provider: EstimatorProvider,
    *,
    lineage: Lineage,
    n_study_trials: int,
    best_params: Mapping[str, Any] | None,
    best_value: float | None,
    n_trials: int | None,
    expand_boundary: bool | None,
    boundary_threshold: float,
    declaration: Declaration | None,
) -> RoundPlan:
    """A new round after a closed one (state C or F).

    Without *declaration* the round declares the previous round's space and
    starts from its effective space. With one (``tune(space=...)``), a dim
    keeps the previous effective range only if its declaration is unchanged
    (5.2-2); the distribution kind, ``log`` and choices may not change
    (5.2-3). A changed or new dim, and a dim whose best lies outside its
    range, is not expanded (5.3). The previous best is the study's best
    COMPLETE trial, never the Model's tuning result (§5.3); a study in state
    F has none, so nothing is expanded.
    """
    study, last = lineage.study, lineage.last_round
    assert study is not None and last is not None  # noqa: S101 -- C/F only
    source: RoundSource = "previous"
    space_arg: tuple[SearchDim, ...] | None = None
    routes_added: tuple[Route, ...] = ()
    declared, provenance = last.declared, last.provenance
    base: tuple[SearchDim, ...] = last.effective
    changed: frozenset[str] = frozenset()
    if declaration is not None:
        refuse_unchangeable(declaration.declared, lineage.rounds, lineage.routes)
        base, changed = carry_forward(
            declaration.declared,
            last,
            provider.parameter_bounds(cfg.task),
            lineage.routes,
        )
        declared, provenance = declaration.declared, declaration.provenance
        source, space_arg = "space_arg", declaration.space_arg
        routes_added = declaration.routes_added
    used_default = _uses_default(study.space_mode, provenance)
    should = expand_boundary if expand_boundary is not None else used_default
    effective, report = base, None
    if not should:
        _log.info("event='tune.resume' expand_boundary=False")
    elif best_params is not None:
        effective, report = expand_for_next_round(
            base,
            best_params,
            boundary_threshold,
            frozen_names=changed | outside_range(base, best_params),
        )
    config_n_trials, timeout = _config_budget(cfg)
    budget = n_trials if n_trials is not None else config_n_trials
    return RoundPlan(
        kind="next",
        round_number=last.round + 1,
        declared=declared,
        provenance=provenance,
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
        routes_added=routes_added,
        run_n_trials=budget,
        run_timeout=timeout,
        source=source,
        space_arg=space_arg,
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
    space_declaration: Declaration | None = None,
) -> RoundPlan:
    """Finish an open round (state O) with its remaining budget (§7.2).

    The round's record decides its space, report and budget. U2: an
    ``n_trials`` or ``expand_boundary`` that is not ``None`` must equal the
    recorded value, and a ``space=`` must resolve to the recorded declared
    space. Trials left RUNNING are FAILed before the round runs, so they
    count as used.
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
    if space_declaration is not None and not same_declared_space(
        space_declaration.declared, record.declared, lineage.routes
    ):
        _refuse_changed_argument(
            k,
            "space",
            space_summary(record.declared, lineage.routes),
            space_summary(space_declaration.declared, lineage.routes),
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
            stacklevel=4,
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
        source=record.source,
        space_arg=record.space_arg,
        running_to_fail=tuple(
            t.number for t in round_trials if t.state.name == "RUNNING"
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch (§7.4)
# ---------------------------------------------------------------------------


def plan_round(
    cfg: LizyMLConfig,
    provider: EstimatorProvider,
    *,
    lineage: Lineage | None,
    trials: Sequence[TrialLike],
    study: Any | None,
    fixed: Mapping[str, Any],
    base_smart: dict[str, Any],
    n_trials: int | None,
    expand_boundary: bool | None,
    boundary_threshold: float,
    space_dims: Sequence[SearchDim] | None,
) -> RoundPlan:
    """Dispatch to the planner for the study's state (§7.4).

    *space_dims* is the parsed ``tune(space=...)``. It reaches only states O,
    C and F: a call that creates a study with ``space=`` is refused before
    planning (§3, §7.4 rows E / N).
    """
    state = lineage.state if lineage is not None else "absent"
    declaration: Declaration | None = None
    if space_dims is not None and lineage is not None:
        assert lineage.study is not None  # noqa: S101 -- states O / C / F
        declaration = resolve_declaration(
            space_dims,
            space_mode=lineage.study.space_mode,
            provider=provider,
            task=cfg.task,
            routes=lineage.routes,
            base_smart=base_smart,
        )
    if state == "open":
        assert lineage is not None  # noqa: S101
        return plan_open_round(
            cfg,
            lineage=lineage,
            trials=trials,
            n_trials=n_trials,
            expand_boundary=expand_boundary,
            boundary_threshold=boundary_threshold,
            space_declaration=declaration,
        )
    if state in ("completed", "failed"):
        assert lineage is not None and study is not None  # noqa: S101
        # The previous best comes from the study, never from the Model's
        # `_tuning_result` (§5.3): a loaded Model's may be another study's.
        # Its values are read with its own round's dims (H-0124 policy 4): the
        # study's best may come from an earlier round with another step.
        complete = any(t.state.name == "COMPLETE" for t in trials)
        best = study.best_trial if complete else None
        return plan_next_round(
            cfg,
            provider,
            lineage=lineage,
            n_study_trials=len(trials),
            best_params=(
                recorded_trial_params(lineage, best) if best is not None else None
            ),
            best_value=best.value if best is not None else None,
            n_trials=n_trials,
            expand_boundary=expand_boundary,
            boundary_threshold=boundary_threshold,
            declaration=declaration,
        )
    # absent (a new study), uninitialized (E) or no_round (N).
    assert space_dims is None  # noqa: S101 -- refused as creating a study
    return plan_first_round(
        cfg,
        provider,
        resolved=resolve_config_space(cfg, provider),
        fixed=fixed,
        base_smart=base_smart,
        n_trials=n_trials,
        expand_boundary=expand_boundary,
        boundary_threshold=boundary_threshold,
    )
