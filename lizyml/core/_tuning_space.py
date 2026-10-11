"""The search space of a round: resolution and the per-dim rules (H-0122 §5.2).

* :func:`resolve_space` -- a declaration resolved by ``space_mode`` (5.2-1),
  the config's (round 1) or a ``tune(space=...)``'s.
* :func:`resolve_declaration` -- a ``space=`` round's declaration: the
  category rule (4.1b), alias names normalised to the routed name (design
  D-b), and H-0121 on the result (5.2-4).
* :func:`refuse_unchangeable` (5.2-3), :func:`carry_forward` (5.2-2) and
  :func:`outside_range` (5.3) -- what a new round may change and inherit.
* :func:`same_declared_space` -- the U2 comparison for an open round (7.2).

Dims are matched by *identity*: ``(category, canonical name)``, where a
routed name's canonical name comes from the lineage's routes and an unrouted
one is its own. Dims are compared through :func:`dim_signature`, which keeps
``True`` / ``1`` / ``1.0`` apart and makes NaN equal to NaN.

Pure: reads the provider and the lineage and writes nothing. Every refusal is
``CONFIG_INVALID`` and comes before any study write (INV-R4).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from lizyml.core._tuning_validation import admit_smart_dimensions
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import get_logger
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.tuning.lineage import (
    Provenance,
    Route,
    RouteConflict,
    StoredRound,
    merge_routes,
)
from lizyml.tuning.lineage_codec import dim_signature, encode_scalar
from lizyml.tuning.search_space import attach_bounds

if TYPE_CHECKING:
    from lizyml.core.types.search_dim import SearchDim
    from lizyml.core.types.task import TaskType
    from lizyml.estimators.provider import EstimatorProvider

_log = get_logger("model")

SPACE_ARG = "tune(space=)"
"""The surface a ``space=`` refusal names."""

Identity = tuple[str, str]


# ---------------------------------------------------------------------------
# Resolution by space_mode (5.2-1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedSpace:
    """A declaration resolved by ``space_mode``, without bounds.

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


def resolve_space(
    space_mode: Literal["merge", "replace"],
    user: Sequence[SearchDim],
    provider: EstimatorProvider,
    task: TaskType,
) -> ResolvedSpace:
    """Resolve the user's dims by *space_mode* (H-0024, H-0121, H-0122 5.2-1).

    ``replace`` declares the user's dimensions only, with no fixed parameters.
    ``merge`` lays the user's dimensions over the provider's default space by
    parameter identity (a model dimension's alias meets its canonical name)
    and fixes the provider's ``default_fixed_params``.
    """
    user_space = list(user)
    if space_mode == "replace":
        return ResolvedSpace(
            declared=tuple(user_space),
            provenance=tuple(
                Provenance(d.category, d.name, "user") for d in user_space
            ),
            used_default=False,
            fixed={},
            default_dims=frozenset(),
        )
    defaults = provider.default_space(task)
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
        fixed=provider.default_fixed_params(task),
        default_dims=default_dims,
    )


def admit_smart(
    provider: EstimatorProvider,
    declared: Sequence[SearchDim],
    base_smart: dict[str, Any],
    default_dims: frozenset[tuple[str, str]],
    *,
    surface: str = "tuning.optuna.space",
) -> list[SearchDim]:
    """H-0121 on a resolved declaration: refuse a user dim, drop a default."""
    kept, dropped = admit_smart_dimensions(
        provider, list(declared), base_smart, default_dims, surface=surface
    )
    if dropped:
        _log.info(
            "event='tune.drop_dims' reason=inactive_smart_owner dims=%s",
            ",".join(dim.name for dim in dropped),
        )
    return kept


# ---------------------------------------------------------------------------
# Routes (4.1b)
# ---------------------------------------------------------------------------


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


def admit_routes(
    existing: Mapping[str, Route],
    added: Iterable[Route],
    *,
    surface: str = "tuning.optuna.space",
) -> None:
    """Refuse a route that would re-route a name (§4.1b; append-only)."""
    try:
        merge_routes(existing, added)
    except RouteConflict as exc:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"{surface}: {exc}. A dimension's category cannot change "
                "within a study; start a new study (another study_name)."
            ),
            context={"surface": surface, "reason": str(exc)},
        ) from exc


def admit_space_routes(
    space_dims: Sequence[SearchDim],
    routes: Mapping[str, Route],
    provider: EstimatorProvider,
) -> None:
    """Refuse a ``space=`` dim that changes a routed name's category (4.1b).

    Run before the name checks, so a smart or training dim re-declared as
    ``model`` gets this message rather than "not a parameter".
    """
    admit_routes(routes, route_entries(space_dims, provider), surface=SPACE_ARG)


def _identity(dim: SearchDim, routes: Mapping[str, Route]) -> Identity:
    route = routes.get(dim.name)
    return (dim.category, route.canonical if route is not None else dim.name)


def normalise_names(
    dims: Sequence[SearchDim],
    routes: Mapping[str, Route],
    provider: EstimatorProvider,
) -> list[SearchDim]:
    """Rename a dim whose identity is routed under another raw name (D-b).

    Optuna keys a trial's params by raw name, and expansion looks the best
    value up by it: a second spelling of a routed parameter would be a new
    parameter to the sampler and would never inherit its expansion. A name
    with no route keeps its spelling.
    """
    canonical = provider.canonical_param_names(
        [d.name for d in dims if d.category == "model"]
    )
    routed: dict[Identity, str] = {}
    for route in routes.values():
        routed.setdefault((route.category, route.canonical), route.name)
    out: list[SearchDim] = []
    for dim in dims:
        key = (
            dim.category,
            canonical[dim.name] if dim.category == "model" else dim.name,
        )
        target = routed.get(key, dim.name)
        out.append(dim if target == dim.name else dataclasses.replace(dim, name=target))
    return out


# ---------------------------------------------------------------------------
# A space= round's declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Declaration:
    """A ``tune(space=...)`` round's declaration, resolved.

    ``space_arg`` is the argument as parsed (raw names and order as written);
    ``declared`` is resolved by the study's ``space_mode``, with names
    normalised to the routes and H-0121 applied; ``provenance`` is aligned
    with ``declared``; ``routes_added`` holds the identities not routed
    before.
    """

    space_arg: tuple[SearchDim, ...]
    declared: tuple[SearchDim, ...]
    provenance: tuple[Provenance, ...]
    routes_added: tuple[Route, ...]


def resolve_declaration(
    space_dims: Sequence[SearchDim],
    *,
    space_mode: Literal["merge", "replace"],
    provider: EstimatorProvider,
    task: TaskType,
    routes: Mapping[str, Route],
    base_smart: dict[str, Any],
) -> Declaration:
    """Resolve a ``space=`` argument into the round's declaration (5.2-1, 5.2-4).

    The category rule runs first (4.1b). The resolved dims -- the user's and,
    in merge mode, the provider's defaults -- are renamed to their routed
    raw names, then H-0121 refuses a user dim nothing would read and drops
    such a default. Provenance depends only on ``space=`` and the provider,
    so a reattach resolves the same declaration (INV-R5).
    """
    admit_space_routes(space_dims, routes, provider)
    resolved = resolve_space(space_mode, space_dims, provider, task)
    renamed = normalise_names(resolved.declared, routes, provider)
    names: dict[tuple[str, str], str] = {
        (d.category, d.name): r.name
        for d, r in zip(resolved.declared, renamed, strict=True)
    }
    default_dims = frozenset(
        (c, names.get((c, n), n)) for c, n in resolved.default_dims
    )
    declared = admit_smart(
        provider, renamed, base_smart, default_dims, surface=SPACE_ARG
    )
    kept = {(d.category, d.name) for d in declared}
    provenance = tuple(
        Provenance(p.category, names[(p.category, p.name)], p.origin)
        for p in resolved.provenance
        if (p.category, names[(p.category, p.name)]) in kept
    )
    entries = route_entries(declared, provider)
    admit_routes(routes, entries, surface=SPACE_ARG)
    return Declaration(
        space_arg=tuple(space_dims),
        declared=tuple(declared),
        provenance=provenance,
        routes_added=tuple(r for r in entries if r.name not in routes),
    )


# ---------------------------------------------------------------------------
# 5.2-3: what one study cannot change
# ---------------------------------------------------------------------------


def _kind(dim: SearchDim) -> str:
    if isinstance(dim, CategoricalDim):
        return "categorical"
    return "int" if isinstance(dim, IntDim) else "float"


def _unchangeable_difference(recorded: SearchDim, given: SearchDim) -> str | None:
    if _kind(recorded) != _kind(given):
        return "kind"
    if isinstance(recorded, CategoricalDim) and isinstance(given, CategoricalDim):
        same = [encode_scalar(c) for c in recorded.choices] == [
            encode_scalar(c) for c in given.choices
        ]
        return None if same else "choices"
    assert isinstance(recorded, FloatDim | IntDim)  # noqa: S101 -- by _kind
    assert isinstance(given, FloatDim | IntDim)  # noqa: S101
    return None if recorded.log == given.log else "log"


def _describe(dim: SearchDim) -> str:
    if isinstance(dim, CategoricalDim):
        return f"categorical {list(dim.choices)!r}"
    return f"{_kind(dim)} log={dim.log}"


def refuse_unchangeable(
    declared: Sequence[SearchDim],
    rounds: Sequence[StoredRound],
    routes: Mapping[str, Route],
) -> None:
    """Refuse a distribution kind, ``log`` or choice change (5.2-3).

    Each dim is compared with the latest declaration of its identity in any
    earlier round, removed dims included: Optuna compares a parameter with
    every distribution it ever had, and refuses only once a trial has been
    written. Choices compare by order, type and value (``True`` / ``1`` /
    ``1.0`` differ, NaN equals NaN), which Optuna does not do. ``low`` and
    ``high`` may change.
    """
    history: dict[Identity, SearchDim] = {}
    for record in rounds:
        for dim in record.declared:
            history[_identity(dim, routes)] = dim
    for dim in declared:
        recorded = history.get(_identity(dim, routes))
        if recorded is None:
            continue
        changed = _unchangeable_difference(recorded, dim)
        if changed is None:
            continue
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"{SPACE_ARG}: dimension '{dim.name}' changes its {changed} "
                f"(recorded {_describe(recorded)}, given {_describe(dim)}). A "
                "dimension's distribution kind, log scale and categorical "
                "choices cannot change within a study; start a new study "
                "(another study_name). Its range (low / high) may change."
            ),
            context={
                "surface": SPACE_ARG,
                "dim": dim.name,
                "changed": changed,
                "recorded": repr(recorded),
                "given": repr(dim),
            },
        )


# ---------------------------------------------------------------------------
# 5.2-2 and 5.3: what a new round inherits and does not expand
# ---------------------------------------------------------------------------


def carry_forward(
    declared: Sequence[SearchDim],
    previous: StoredRound,
    bounds: dict[str, dict[str, float | int]],
    routes: Mapping[str, Route],
) -> tuple[tuple[SearchDim, ...], frozenset[str]]:
    """The round's sampling base from its declaration (5.2-2).

    A dim declared exactly as in the previous round inherits that round's
    effective dim (its expansion and bounds). A changed or new dim takes its
    new range with the provider's bounds, and its name is returned as
    *changed*. A dim no longer declared is gone. Declared order is kept.
    """
    prev_declared = {_identity(d, routes): d for d in previous.declared}
    prev_effective = {_identity(d, routes): d for d in previous.effective}
    base: list[SearchDim] = []
    changed: set[str] = set()
    for dim in declared:
        key = _identity(dim, routes)
        before = prev_declared.get(key)
        if (
            before is not None
            and key in prev_effective
            and dim_signature(before) == dim_signature(dim)
        ):
            base.append(prev_effective[key])
            continue
        base.extend(attach_bounds([dim], bounds))
        changed.add(dim.name)
    return tuple(base), frozenset(changed)


def outside_range(
    dims: Sequence[SearchDim], best_params: Mapping[str, Any]
) -> frozenset[str]:
    """Numeric dims whose best value lies outside their range (5.3 bullet 2)."""
    out: set[str] = set()
    for dim in dims:
        if isinstance(dim, CategoricalDim) or dim.name not in best_params:
            continue
        value = best_params[dim.name]
        if value is not None and not dim.low <= value <= dim.high:
            out.add(dim.name)
    return frozenset(out)


# ---------------------------------------------------------------------------
# U2 (7.2): an open round's space
# ---------------------------------------------------------------------------


def _canonical_signatures(
    dims: Iterable[SearchDim], routes: Mapping[str, Route]
) -> dict[Identity, str]:
    out: dict[Identity, str] = {}
    for dim in dims:
        key = _identity(dim, routes)
        out[key] = dim_signature(dataclasses.replace(dim, name=key[1]))
    return out


def same_declared_space(
    given: Iterable[SearchDim],
    recorded: Iterable[SearchDim],
    routes: Mapping[str, Route],
) -> bool:
    """Whether two declared spaces are the same set of dims (U2).

    Matched by identity and compared through their signatures under the
    canonical name, so dict order, an omitted ``log`` and an alias spelling
    do not matter, while a range, a choice's type or a missing dim does.
    """
    return _canonical_signatures(given, routes) == _canonical_signatures(
        recorded, routes
    )


def space_summary(dims: Iterable[SearchDim], routes: Mapping[str, Route]) -> list[str]:
    """A JSON-safe, sorted summary of a space, one string per dim.

    ``category:canonical`` and the declaration (kind, range, ``log`` or the
    choices), so two spaces that differ only in a range read differently.
    """
    out = []
    for dim in dims:
        category, canonical = _identity(dim, routes)
        if isinstance(dim, CategoricalDim):
            shown = f"categorical {list(dim.choices)!r}"
        else:
            shown = f"{_kind(dim)} [{dim.low!r}, {dim.high!r}]" + (
                " log" if dim.log else ""
            )
        out.append(f"{category}:{canonical} {shown}")
    return sorted(out)
