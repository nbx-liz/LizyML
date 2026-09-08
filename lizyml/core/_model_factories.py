"""Factory functions for building splitters, inner-validation, and provider dispatch.

Extracted from Model to reduce model.py size (H-0042).
Provider dispatch added in H-0053.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy.typing as npt

from lizyml.config.schema import (
    BlockedGroupKFoldConfig,
    GroupKFoldConfig,
    GroupTimeSeriesConfig,
    HoldoutInnerValidConfig,
    KFoldConfig,
    LizyMLConfig,
    PurgedTimeSeriesConfig,
    SplitConfig,
    StratifiedGroupKFoldConfig,
    StratifiedKFoldConfig,
    TimeSeriesConfig,
)
from lizyml.core.types.task import TaskType
from lizyml.core.value_equality import values_differ
from lizyml.splitters.base import BaseSplitter
from lizyml.splitters.blocked_group_kfold import BlockedGroupKFoldSplitter
from lizyml.splitters.group_kfold import (
    GroupKFoldSplitter,
    StratifiedGroupKFoldSplitter,
)
from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter
from lizyml.splitters.kfold import KFoldSplitter, StratifiedKFoldSplitter
from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter
from lizyml.splitters.time_series import TimeSeriesSplitter
from lizyml.training.inner_valid import (
    BlockedGroupInnerValid,
    GroupHoldoutInnerValid,
    HoldoutInnerValid,
    NoInnerValid,
    TimeHoldoutInnerValid,
)

InnerValidType = (
    HoldoutInnerValid
    | GroupHoldoutInnerValid
    | TimeHoldoutInnerValid
    | BlockedGroupInnerValid
    | NoInnerValid
)


def _resolve_stratify(stratify: str | bool, task: TaskType) -> bool:
    """Resolve ``stratify: "auto"`` to a concrete boolean."""
    if isinstance(stratify, bool):
        return stratify
    if stratify == "auto":
        return task in ("binary", "multiclass")
    # Defensive fallback — schema restricts to Literal["auto"] | bool
    return str(stratify).lower() == "true"  # pragma: no cover


def _build_splitter_for_method(
    split_cfg: SplitConfig,
    n_splits: int,
    *,
    block_values: npt.NDArray[Any] | None = None,
    task: TaskType | None = None,
    seed: int | None = None,
) -> BaseSplitter:
    """Build a splitter from split config, using the given *n_splits*.

    Shared implementation for both outer CV and calibration CV splitters.
    The *n_splits* parameter is separated so that callers can override it
    (e.g. ``calibration.n_splits`` instead of ``split.n_splits``).

    ``seed`` is the fallback used when a split config's ``random_state`` is
    ``None`` (H-0080): an explicit ``random_state`` wins, otherwise the
    splitter inherits ``training.seed`` (passed by ``build_splitter`` as
    ``seed``). ``seed=None`` itself falls back to the historical default 42.
    """
    resolved_seed = 42 if seed is None else seed
    if isinstance(split_cfg, BlockedGroupKFoldConfig):
        if block_values is None:
            from lizyml.core.exceptions import ErrorCode, LizyMLError

            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message=(
                    "blocked_group_kfold requires block_values "
                    "(extracted from blocks.col by Facade)."
                ),
                context={
                    "split_method": "blocked_group_kfold",
                    "blocks_col": split_cfg.blocks.col,
                },
            )
        stratify_bool = _resolve_stratify(
            split_cfg.groups.stratify, task or "regression"
        )
        return BlockedGroupKFoldSplitter(
            block_values=block_values,
            cutoffs=split_cfg.blocks.cutoffs,
            mode=split_cfg.blocks.mode,
            train_window=split_cfg.blocks.train_window,
            n_splits=split_cfg.groups.n_splits,
            stratify=stratify_bool,
            shuffle=split_cfg.groups.shuffle,
            random_state=resolved_seed,
            min_train_rows=split_cfg.min_train_rows,
            min_valid_rows=split_cfg.min_valid_rows,
        )
    if isinstance(split_cfg, StratifiedKFoldConfig):
        return StratifiedKFoldSplitter(
            n_splits=n_splits,
            shuffle=True,
            random_state=(
                split_cfg.random_state
                if split_cfg.random_state is not None
                else resolved_seed
            ),
        )
    if isinstance(split_cfg, GroupKFoldConfig):
        return GroupKFoldSplitter(n_splits=n_splits)
    if isinstance(split_cfg, StratifiedGroupKFoldConfig):
        return StratifiedGroupKFoldSplitter(
            n_splits=n_splits,
            shuffle=split_cfg.shuffle,
            random_state=(
                split_cfg.random_state
                if split_cfg.random_state is not None
                else resolved_seed
            ),
        )
    if isinstance(split_cfg, TimeSeriesConfig):
        return TimeSeriesSplitter(
            n_splits=n_splits,
            gap=split_cfg.gap,
            max_train_size=split_cfg.train_size_max,
            max_test_size=split_cfg.test_size_max,
        )
    if isinstance(split_cfg, PurgedTimeSeriesConfig):
        return PurgedTimeSeriesSplitter(
            n_splits=n_splits,
            purge_gap=split_cfg.purge_gap,
            embargo=split_cfg.embargo,
            max_train_size=split_cfg.train_size_max,
            max_test_size=split_cfg.test_size_max,
        )
    if isinstance(split_cfg, GroupTimeSeriesConfig):
        return GroupTimeSeriesSplitter(
            n_splits=n_splits,
            gap=split_cfg.gap,
            max_train_size=split_cfg.train_size_max,
            max_test_size=split_cfg.test_size_max,
        )
    if isinstance(split_cfg, KFoldConfig):
        return KFoldSplitter(
            n_splits=n_splits,
            shuffle=split_cfg.shuffle,
            random_state=(
                split_cfg.random_state
                if split_cfg.random_state is not None
                else resolved_seed
            ),
        )
    # Loud fail — adding a new SplitConfig variant without updating this
    # dispatch must not silently produce KFold splits (#119).
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    type_name = type(split_cfg).__name__
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"Unhandled SplitConfig type: {type_name}. "
            "Update _build_splitter_for_method dispatch when adding a new variant."
        ),
        context={"split_config_type": type_name},
    )


def get_outer_n_splits(cfg: LizyMLConfig) -> int:
    """Return the outer CV n_splits regardless of split config variant (H-0073).

    ``BlockedGroupKFoldConfig`` exposes ``groups.n_splits`` (the outer
    KFold over groups) while every other variant has a top-level
    ``n_splits``. Centralised here so that callers in ``_model_factories``
    and ``_model_persistence`` use the same resolution.
    """
    if isinstance(cfg.split, BlockedGroupKFoldConfig):
        return cfg.split.groups.n_splits
    return cfg.split.n_splits


def build_splitter(
    cfg: LizyMLConfig,
    *,
    block_values: npt.NDArray[Any] | None = None,
    task: TaskType | None = None,
    seed: int | None = None,
) -> BaseSplitter:
    """Instantiate outer CV splitter from config."""
    split_cfg = cfg.split

    # Warn if classification task explicitly uses kfold (H-0013)
    if split_cfg.method == "kfold" and cfg.task in ("binary", "multiclass"):
        warnings.warn(
            f"task='{cfg.task}' with split.method='kfold' does not "
            "preserve class distribution. Consider using 'stratified_kfold' "
            "instead.",
            UserWarning,
            stacklevel=2,
        )

    n_splits = get_outer_n_splits(cfg)

    # BlockedGroupKFoldConfig has no n_splits at top level — pass block_values etc.
    if isinstance(split_cfg, BlockedGroupKFoldConfig):
        return _build_splitter_for_method(
            split_cfg,
            n_splits,
            block_values=block_values,
            task=cfg.task if task is None else task,
            seed=cfg.training.seed if seed is None else seed,
        )

    # General (non-blocked) splitters: thread training.seed as the fallback
    # for any split config whose random_state is None (H-0080).
    return _build_splitter_for_method(
        split_cfg,
        n_splits,
        seed=cfg.training.seed if seed is None else seed,
    )


def build_calibration_splitter(cfg: LizyMLConfig) -> BaseSplitter:
    """Instantiate calibration CV splitter from config (H-0044).

    .. deprecated:: H-0058
        Calibration cross-fit now reuses outer CV splits.
        This function is kept for backward compatibility only.
    """
    import warnings

    warnings.warn(
        "build_calibration_splitter is deprecated (H-0058). "
        "Calibration cross-fit now reuses outer CV splits. "
        "Will be removed in v1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    assert cfg.calibration is not None  # noqa: S101
    return _build_splitter_for_method(cfg.split, cfg.calibration.n_splits)


def _auto_inner_gap(split_cfg: Any) -> int:
    """Look-ahead gap to purge at the inner-valid boundary (H-0085 / #212).

    Propagated from the outer split so the early-stopping split gets the same
    guard: ``purge_gap + embargo`` for ``purged_time_series``, ``gap`` for
    ``time_series``. Other methods contribute no inner gap.
    """
    method = getattr(split_cfg, "method", None)
    if method == "purged_time_series":
        return int(getattr(split_cfg, "purge_gap", 0)) + int(
            getattr(split_cfg, "embargo", 0)
        )
    if method == "time_series":
        return int(getattr(split_cfg, "gap", 0))
    return 0


def _resolve_auto_inner_valid(
    split_method: str,
    ratio: float,
    seed: int,
    *,
    task: TaskType | None = None,
    gap: int = 0,
) -> (
    HoldoutInnerValid
    | GroupHoldoutInnerValid
    | TimeHoldoutInnerValid
    | BlockedGroupInnerValid
):
    """Resolve inner validation strategy based on the outer split method."""
    if split_method == "blocked_group_kfold":
        return BlockedGroupInnerValid(ratio=ratio, task=task or "regression")
    if split_method == "stratified_kfold":
        return HoldoutInnerValid(ratio=ratio, random_state=seed, stratify=True)
    if split_method in ("group_kfold", "stratified_group_kfold"):
        return GroupHoldoutInnerValid(ratio=ratio, random_state=seed)
    if split_method in ("time_series", "purged_time_series"):
        return TimeHoldoutInnerValid(ratio=ratio, gap=gap)
    if split_method == "group_time_series":
        return GroupHoldoutInnerValid(ratio=ratio, random_state=seed)
    return HoldoutInnerValid(ratio=ratio, random_state=seed, stratify=False)


def build_inner_valid(cfg: LizyMLConfig) -> InnerValidType:
    """Instantiate inner validation strategy from training config.

    When early stopping is enabled but ``inner_valid`` is not explicitly
    set, the strategy is auto-resolved based on the outer split method:

    - ``stratified_kfold`` → ``HoldoutInnerValid(stratify=True)``
    - ``group_kfold`` / ``stratified_group_kfold`` → ``GroupHoldoutInnerValid``
    - ``time_series`` → ``TimeHoldoutInnerValid``
    - ``kfold`` (or other) → ``HoldoutInnerValid(stratify=False)``
    """
    es = cfg.training.early_stopping
    if not es.enabled:
        return NoInnerValid()

    iv_cfg = es.inner_valid
    split_method = cfg.split.method
    seed = cfg.training.seed
    gap = _auto_inner_gap(cfg.split)

    # Auto-resolve: inner_valid absent or created from validation_ratio default
    if iv_cfg is None:
        return _resolve_auto_inner_valid(
            split_method, 0.1, seed, task=cfg.task, gap=gap
        )
    if not es._inner_valid_explicit:
        return _resolve_auto_inner_valid(
            split_method, iv_cfg.ratio, seed, task=cfg.task, gap=gap
        )

    # Explicit config — dispatch by concrete type
    if isinstance(iv_cfg, HoldoutInnerValidConfig):
        if split_method in ("time_series", "purged_time_series"):
            warnings.warn(
                "Explicit inner_valid method='holdout' uses a shuffled random "
                f"split, but the outer split.method='{split_method}' is "
                "time-ordered. The early-stopping validation will not respect "
                "time order and may be temporally leaked. Consider "
                "inner_valid.method='time_holdout'.",
                UserWarning,
                stacklevel=2,
            )
        return HoldoutInnerValid(
            ratio=iv_cfg.ratio,
            random_state=iv_cfg.random_state,
            stratify=iv_cfg.stratify,
        )
    if iv_cfg.method == "group_holdout":
        return GroupHoldoutInnerValid(
            ratio=iv_cfg.ratio,
            random_state=iv_cfg.random_state,
        )
    if iv_cfg.method == "time_holdout":
        return TimeHoldoutInnerValid(ratio=iv_cfg.ratio)
    return NoInnerValid()


def make_inner_valid_factory(
    cfg: LizyMLConfig,
) -> Callable[
    [float],
    (
        HoldoutInnerValid
        | GroupHoldoutInnerValid
        | TimeHoldoutInnerValid
        | BlockedGroupInnerValid
    ),
]:
    """Return a factory that produces InnerValidStrategy for a given ratio.

    Used by the Tuner when ``validation_ratio`` is a search dimension.
    """
    split_method = cfg.split.method
    seed = cfg.training.seed
    task = cfg.task
    gap = _auto_inner_gap(cfg.split)

    def factory(
        ratio: float,
    ) -> (
        HoldoutInnerValid
        | GroupHoldoutInnerValid
        | TimeHoldoutInnerValid
        | BlockedGroupInnerValid
    ):
        return _resolve_auto_inner_valid(split_method, ratio, seed, task=task, gap=gap)

    return factory


# ------------------------------------------------------------------
# EstimatorProvider dispatch (H-0053)
# ------------------------------------------------------------------


def get_provider(model_cfg: Any) -> Any:
    """Return the EstimatorProvider for the given model config.

    Dispatches on ``model_cfg.name`` to import the provider lazily.
    New algorithms only need to add an ``elif`` branch here.

    Args:
        model_cfg: A pydantic model config (e.g. ``LGBMConfig``).

    Returns:
        An ``EstimatorProvider`` instance.

    Raises:
        LizyMLError with CONFIG_INVALID for unknown model names.
    """
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    name: str = getattr(model_cfg, "name", "")
    if name == "lgbm":
        from lizyml.estimators.lgbm.provider import LGBMProvider

        return LGBMProvider()

    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=f"Unknown model name '{name}'. Supported: lgbm",
        context={"model_name": name},
    )


def check_param_names(
    provider: Any,
    named: Iterable[tuple[str, str]],
    *,
    model_name: str,
    extra_accepted: frozenset[str] = frozenset(),
) -> None:
    """Reject parameter names the estimator would silently discard (H-0093).

    LightGBM drops a name it does not know without raising -- and with the
    shipped ``verbose=-1`` it does not even warn -- so an unchecked typo yields
    a run that looks successful and in which the parameter did nothing. For a
    tuning dimension it is worse: the study explores an axis that cannot
    influence the score.

    This lives in the Facade rather than in ``config/``. ``ARCHITECTURE.md``'s
    layer DAG puts ``config/`` and ``estimators/`` both in Layer 1, and Layer 1
    may reference Layer 0 only, so the config layer cannot ask a provider what
    it accepts. The Facade is the first place the two legally meet.

    It is called at the point of use rather than at construction. A config
    handed to ``Model`` stays under the caller's reference and can be mutated
    afterwards, and ``best_model_params`` restored from an artifact are
    installed after ``__init__`` has run, so a construction-time check would
    pass over both. Firing on the way to the estimator also leaves
    ``Model.load()`` able to read back an artifact whose recorded config carries
    a name this gate would refuse.

    **What "covered" means here is enumerated, not asserted.** Three review
    rounds each falsified a prose claim about covering every route, and the
    first two were answered with another call site and another sentence. The
    population is instead a checked object: ``ESTIMATOR_ROUTES`` in
    ``tests/test_estimators/test_lightgbm_parameter_names.py`` lists every site
    in ``lizyml/`` that can put a parameter dict in front of the estimator, and
    its test fails both when a site is missing from the list and when the list
    names a site that no longer exists.

    The scan resolves LightGBM from each module's own imports and looks at the
    *producers* of a params dict as well as the adapter construction that
    consumes one. Both were fixes to false-clean scans, not refinements: the
    calibrator imports LightGBM as ``lgbm`` and was invisible while it handed
    ``calibration.params`` to a Booster, and a new caller of
    ``build_estimator_factory`` produced no detection at all. Both shapes are
    now exercised against injected sources rather than described.

    Args:
        provider: The ``EstimatorProvider`` about to receive these names.
        named: ``(surface, name)`` pairs, where *surface* is the
            user-facing location to quote back, e.g. ``model.params``.
        model_name: The estimator's name, for the message.
        extra_accepted: Names the *caller's own* surface consumes before the
            estimator sees them. The model surface has none -- its own
            namespace is the smart parameters, reported separately below --
            but the calibration surface pops four (H-0093). Passing them here
            keeps each surface's exceptions beside the code that pops them
            instead of accumulating in this function.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming every offending name and,
            for a smart parameter written where a native one belongs, the
            category it should have carried.
    """
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    accepted: frozenset[str] = provider.accepted_model_param_names() | extra_accepted
    smart: frozenset[str] = provider.smart_param_names()

    unknown = [(surface, name) for surface, name in named if name not in accepted]
    if not unknown:
        return

    lines = []
    for surface, name in unknown:
        if name in smart:
            lines.append(
                f"  {surface}: '{name}' is a LizyML smart parameter, not a "
                f"native {model_name} parameter. Declare it with "
                f"category: smart, or set it directly on the model config."
            )
        else:
            lines.append(f"  {surface}: '{name}' is not a {model_name} parameter.")
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"Unknown {model_name} parameter name(s):\n" + "\n".join(lines) + "\n"
            "An unrecognised name is discarded by the estimator without error, "
            "so it would have had no effect."
        ),
        context={
            "unknown": [{"surface": s, "name": n} for s, n in unknown],
            "model_name": model_name,
        },
    )


def model_space_names(cfg: LizyMLConfig) -> list[tuple[str, str]]:
    """Return the ``category: model`` dimensions of the tuning search space.

    ``smart`` and ``training`` dimensions are LizyML's own namespaces and are
    never handed to the estimator as parameters, so they are out of scope.
    """
    tuning = getattr(cfg, "tuning", None)
    optuna = getattr(tuning, "optuna", None) if tuning is not None else None
    space = getattr(optuna, "space", None) if optuna is not None else None
    out: list[tuple[str, str]] = []
    for dim_name, spec in (space or {}).items():
        category = spec.get("category", "model") if isinstance(spec, dict) else "model"
        if category == "model":
            out.append(("tuning.optuna.space", dim_name))
    return out


#: Native parameters that a ``training.*`` setting already controls.
#:
#: Each is one parameter under two names in two places -- LizyML's, and
#: LightGBM's -- so a config naming both has said one thing twice, and which
#: one applies is decided by machinery the caller cannot see. Measured before
#: this check (H-0094 decision 9, review round 13):
#:
#: * ``early_stopping_round``: the override reached ``lgb.train`` on every call
#:   and the callback built from ``training.early_stopping.rounds`` still
#:   decided when to stop -- ``rounds: 2`` with an override of ``10`` trained 3
#:   iterations. With the callback **off** it is not inert either: LightGBM
#:   honours the parameter itself, and LizyML has built no validation set, so
#:   the run dies with ``CONFIG_INVALID`` blaming the metric.
#: * ``seed``: the override wins, silently, over the ``training.seed`` the
#:   config states -- so a run's reproducibility control is not the one the
#:   config appears to declare.
#:
#: The values are the config paths, because the message has to name the place
#: the caller can change. The keys are canonical and expanded to every spelling
#: at check time; a literal-name check would refuse ``seed`` and admit
#: ``random_seed``, which refuses nothing.
TRAINING_MANAGED_PARAMS: dict[str, str] = {
    "early_stopping_round": "training.early_stopping.rounds",
    "seed": "training.seed",
}


def check_training_managed_space(provider: Any, cfg: LizyMLConfig) -> None:
    """Refuse a search dimension for a parameter a ``training.*`` setting controls.

    The merged-dict check below runs inside ``_merge_params``, and **trial
    parameters overlay after that**, so a ``category: model`` dimension naming
    one of these was accepted, sampled, and trained. Measured over all seven
    spellings of both entries (H-0094 decision 10, review round 14): each study
    trained real boosters, returned a ``best_model_params`` carrying the name,
    and **the following ``fit()`` then refused it** -- a study whose result its
    own next step cannot consume.

    A comment on the merged-dict call claimed it "covers every input at once".
    That was true of the three inputs that meet in `_merge_params` and false of
    the fourth, which arrives later. Checked here, before the study starts, the
    way the other two space-level refusals are.
    """
    names = [name for _, name in model_space_names(cfg)]
    if names:
        check_training_managed_overrides(
            provider,
            dict.fromkeys(names),
            cfg,
            origins=dict.fromkeys(names, "tuning.optuna.space"),
        )


def check_training_managed_overrides(
    provider: Any,
    params: dict[str, Any],
    cfg: LizyMLConfig,
    *,
    origins: dict[str, str] | None = None,
) -> None:
    """Refuse a native parameter that a ``training.*`` setting already controls.

    This is the same policy the smart-managed check applies one layer over: the
    conflict is an error, not a silent substitution. It differs in *which* way
    the silence falls, and that difference is why refusing beats picking a
    winner -- ``early_stopping_round`` loses to the config, ``seed`` beats it.
    Nothing in the config says which.

    Only an **active** setting claims its parameter: ``training.early_stopping``
    claims ``early_stopping_round`` when it is enabled, and not otherwise.

    Args:
        provider: EstimatorProvider instance.
        params: The merged model params.
        cfg: The whole config, read for the ``training`` section.
        origins: ``{name: the input it came from}``, so the message can address
            the caller's own file or call rather than always saying
            ``model.params``.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming the parameter, the spelling
            written, and the ``training.*`` setting it collides with.
    """
    if not params:
        return
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    training = getattr(cfg, "training", None)
    if training is None:
        return

    # Every spelling of each claimed parameter, derived from the provider
    # rather than listed: the protocol answers "which parameter does this name
    # identify", so inverting it over the accepted names gives the spellings
    # without this module knowing which estimator is in play.
    accepted = provider.accepted_model_param_names()
    spellings_of: dict[str, set[str]] = {}
    for name, canonical_name in provider.canonical_param_names(accepted).items():
        spellings_of.setdefault(canonical_name, set()).add(name)

    claimed: dict[str, str] = {}
    for canonical, config_path in TRAINING_MANAGED_PARAMS.items():
        if canonical == "early_stopping_round":
            stopping = getattr(training, "early_stopping", None)
            if not getattr(stopping, "enabled", False):
                continue
        elif getattr(training, "seed", None) is None:
            continue
        for spelling in spellings_of.get(canonical, {canonical}):
            claimed[spelling] = config_path

    hits = [(name, claimed[name]) for name in params if name in claimed]
    if not hits:
        return

    origins = origins or {}
    lines = [
        f"  {origins.get(name, 'model.params')}: '{name}' is already set by "
        f"'{config_path}', and the estimator treats those as one parameter."
        for name, config_path in sorted(hits)
    ]
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            "Parameter(s) already controlled by a training setting:\n"
            + "\n".join(lines)
            + "\nWhich value applies would depend on how LizyML builds the "
            "training call rather than on what you wrote. Set it in one place."
        ),
        context={
            "conflicts": [
                {
                    "surface": origins.get(name, "model.params"),
                    "name": name,
                    "config_path": config_path,
                }
                for name, config_path in sorted(hits)
            ]
        },
    )


def check_duplicate_space_dimensions(provider: Any, cfg: LizyMLConfig) -> None:
    """Refuse two ``category: model`` dimensions that name one parameter.

    ``sample_params`` writes one key per dimension, so two dimensions spelling
    one LightGBM parameter put both spellings in the same trial dict. LightGBM
    resolves them to one parameter and prefers the canonical spelling, so the
    other dimension is sampled, optimised over, and **has no effect on any
    trial**. Measured before this check, with ``learning_rate`` and ``eta`` as
    two dimensions:

    ``lgb.train`` received both on every trial and trained at the
    ``learning_rate`` value; ``best_model_params`` recorded both, so the ``fit``
    afterwards carried the dead spelling too. Optuna ranked the trials on an
    axis that did nothing (H-0094 decision 8, review round 12, named by the
    rounds 11-12 monitor and reproduced before being fixed).

    This is the same-layer rule on the layer decision 6 had not reached: the
    space is one layer, and it was meeting itself. Unlike the dict surfaces
    there is **no equal-values escape** -- two dimensions sample independently,
    so naming one parameter twice is ambiguous whatever the bounds say.

    Distinct from #279, which is a dimension colliding with a *smart parameter*,
    and from #280, which is ``model.params`` colliding with one.

    Args:
        provider: EstimatorProvider instance.
        cfg: The whole config; the space is read through ``model_space_names``
            so this and the name check cannot disagree about which dimensions
            are the estimator's.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming the dimensions and the
            parameter they share.
    """
    names = [name for _, name in model_space_names(cfg)]
    if len(names) < 2:
        return
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    canonical = provider.canonical_param_names(names)
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(canonical[name], []).append(name)

    conflicts = {
        parameter: written for parameter, written in grouped.items() if len(written) > 1
    }
    if not conflicts:
        return
    lines = [
        f"  tuning.optuna.space: {sorted(written)} are dimensions for the one "
        f"parameter '{parameter}', so only one of them can reach the estimator."
        for parameter, written in sorted(conflicts.items())
    ]
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            "Search space dimension(s) naming one parameter more than once:\n"
            + "\n".join(lines)
            + "\nThe other would be sampled and optimised over without "
            "affecting any trial."
        ),
        context={
            "conflicts": [
                {
                    "surface": "tuning.optuna.space",
                    "parameter": parameter,
                    "dimensions": sorted(written),
                }
                for parameter, written in sorted(conflicts.items())
            ]
        },
    )


#: Calibration methods whose ``params`` reach LightGBM.
#:
#: ``IsotonicCalibrator`` trains a single-feature Booster, so its params go to
#: ``lgbm.train`` and an unknown name there is discarded in silence -- the same
#: defect the model surface has (H-0093). ``platt`` and ``beta`` fit with numpy
#: and scipy and never touch LightGBM, so checking their names against
#: LightGBM's registry would refuse legitimate configs. That split is scanned
#: rather than asserted: ``test_calibration_param_names.py`` walks
#: ``lizyml/calibration/`` and fails when a module that binds LightGBM is not
#: named here.
LGBM_BACKED_CALIBRATORS: frozenset[str] = frozenset({"isotonic"})


def overlay_params(
    provider: Any, base: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    """Overlay *overlay* onto *base* by parameter identity, not by spelling.

    ``{**base, **overlay}`` keeps both spellings when the two layers name one
    parameter differently, and the estimator then picks one of them -- measured
    on LightGBM: a canonical name in the lower layer beat an alias in the
    higher one, so the override silently lost. Dropping the losing spelling
    here means the estimator never sees the ambiguity, so the outcome does not
    depend on which spelling it happens to prefer.

    Args:
        provider: EstimatorProvider instance.
        base: The lower-priority layer.
        overlay: The higher-priority layer, which wins.

    Returns:
        A new dict; neither argument is modified.
    """
    if not overlay:
        return dict(base)
    canonical = provider.canonical_param_names([*base, *overlay])
    overlaid = {canonical[name] for name in overlay}
    merged = {
        name: value
        for name, value in base.items()
        if name in overlay or canonical[name] not in overlaid
    }
    merged.update(overlay)
    return merged


def check_duplicate_identities(
    provider: Any, params: dict[str, Any], *, surface: str
) -> None:
    """Refuse one layer naming a parameter twice with different values.

    ``{"objective": "binary", "application": "binary"}`` is one parameter
    written twice. Equal values are harmless; different ones are ambiguous, and
    resolving them by dictionary order would decide the training run on
    something the caller cannot see (H-0094, review round 4).

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming the spellings and values.
    """
    if not params:
        return
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    canonical = provider.canonical_param_names(params)
    grouped: dict[str, dict[str, Any]] = {}
    for name, value in params.items():
        grouped.setdefault(canonical[name], {})[name] = value

    # Compared by equality, not by printed form. `repr` made `1` and `1.0`
    # two different values and refused a call that meant one thing twice
    # (H-0094, review round 5) -- a gate refusing valid input, which is worse
    # here than the ambiguity it exists to catch. `values_differ` is shared
    # with `_pop_by_identity`, so the two refusals cannot disagree about what
    # "the same value" means, and it survives a value whose `!=` is not a bool
    # -- a bare `!=` raised on a numpy array even when the value appeared once.
    conflicts = {
        parameter: written
        for parameter, written in grouped.items()
        if any(
            values_differ(value, next(iter(written.values())))
            for value in written.values()
        )
    }
    if not conflicts:
        return
    lines = [
        f"  {surface}: '{parameter}' is set as {written}, and the estimator "
        "treats those as one parameter."
        for parameter, written in sorted(conflicts.items())
    ]
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            "Parameter(s) set more than once under different spellings:\n"
            + "\n".join(lines)
            + "\nWhich value applies would depend on the estimator rather than "
            "on what you wrote."
        ),
        context={
            "conflicts": [
                {"surface": surface, "parameter": parameter, "written": written}
                for parameter, written in sorted(conflicts.items())
            ]
        },
    )


def check_smart_managed_overrides(
    provider: Any,
    override: dict[str, Any] | None,
    smart: dict[str, Any],
    task: Any,
    *,
    surface: str,
) -> None:
    """Refuse an override of a name a smart parameter is going to overwrite.

    Smart resolution runs after the parameter dict is merged and its result
    wins, so ``fit(params={"num_leaves": 12})`` trains at whatever
    ``auto_num_leaves`` computes and reports nothing. Measured before this
    check: ``num_leaves=12`` trained at 32, ``min_data_in_leaf=3`` at 1, and
    ``scale_pos_weight=10`` at 0.935.

    This is the policy ``LGBMConfig._validate_smart_params`` already applies to
    the same collisions in the config: the conflict is an error, not a silent
    substitution. **It applies here to the ``fit()`` override only**, because
    that is the input this change introduces.

    The other two surfaces are open, with their gaps measured rather than
    described, so this bound is not read as wider than it is:

    * ``model.params`` -- ``LGBMConfig._validate_smart_params`` compares
      literal strings and covers three of the five smart parameters, so over
      the whole surface (each smart parameter x each native name it writes x
      each spelling LightGBM accepts) **3 of 18 are refused**; of the rest, 12
      send two spellings to ``lgb.train`` and 3 are overwritten outright.
      ``config/`` cannot import ``estimators/`` under the layer rule and so
      cannot reach the alias table; where the refusal belongs is a design
      decision the maintainer holds open as **#280**, recorded in
      BLUEPRINT.md §14.4.
    * the search space has its own measured gap (H-0094, **#279**).

    An earlier wording of this docstring said the config surface "is refused at
    parse time for three of the five", which is true of the smart parameters and
    false of the surface: it counted the three canonical names and not the
    fifteen spellings and targets that pass. A bound stated wider than it holds
    is the shape this PR is about (H-0094 decision 8, review round 12).

    Args:
        provider: EstimatorProvider instance.
        override: The ``fit()`` override, or ``None``.
        smart: Merged smart params, as they will be resolved.
        task: ML task type -- a smart parameter may write a native name for one
            task and not for another.
        surface: How to name the offending input in the message.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming every managed name given.
    """
    if not override:
        return
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    managed: dict[str, tuple[str, str]] = provider.smart_managed_param_names(
        smart, task
    )
    hits = [(name, *managed[name]) for name in override if name in managed]
    if not hits:
        return

    lines = []
    for name, canonical, smart_name in hits:
        # An alias is the same parameter to the estimator, so say which one it
        # names -- otherwise the message talks about a name the user did not
        # write, or about one whose connection to theirs is invisible.
        subject = (
            f"'{name}'"
            if name == canonical
            else f"'{name}', which names '{canonical}',"
        )
        lines.append(
            f"  {surface}: {subject} is resolved from the smart parameter "
            f"'{smart_name}' and would be replaced, so setting it here would "
            f"have no effect. Disable 'model.{smart_name}' to set "
            f"'{canonical}' directly."
        )
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            "Parameter name(s) managed by a smart parameter:\n" + "\n".join(lines)
        ),
        context={
            "managed": [
                {
                    "surface": surface,
                    "name": name,
                    "canonical": canonical,
                    "smart_param": smart_name,
                }
                for name, canonical, smart_name in hits
            ]
        },
    )


def check_calibration_param_names(calibration_cfg: Any) -> None:
    """Reject ``calibration.params`` names LightGBM would silently discard.

    A fourth route into the estimator, found by review after the first three
    were gated: ``cfg.calibration.params`` is merged over the calibrator's
    defaults and handed to ``lgbm.train``, so ``{"num_leave": 7}`` trains with
    the default ``num_leaves`` and reports success.

    Only the LightGBM-backed methods are checked; see
    ``LGBM_BACKED_CALIBRATORS``.

    It is also a fourth **layer**, and the same-layer rule applies to it: one
    parameter written twice under two spellings with different values is
    refused. Until this was wired, ``{"learning_rate": 0.001, "eta": 0.5}``
    sent both to the calibrator's ``lgbm.train``, which kept the canonical one
    in silence -- the shape H-0094 decision 6 declares against, on the one layer
    that had a name check and no identity check (H-0094 decision 7, found by the
    rounds 10-11 monitor).

    Args:
        calibration_cfg: ``cfg.calibration``, or ``None`` when the run is not
            calibrated.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming every offending name, and
            naming both spellings when one parameter is written twice.
    """
    if calibration_cfg is None:
        return
    method: str = getattr(calibration_cfg, "method", "")
    if method not in LGBM_BACKED_CALIBRATORS:
        return
    params = getattr(calibration_cfg, "params", None) or {}
    if not params:
        return

    from lizyml.calibration.isotonic import CALIBRATOR_OWN_PARAM_NAMES
    from lizyml.estimators.lgbm.provider import LGBMProvider

    # The calibrator hardcodes LightGBM regardless of which estimator the model
    # uses, so the authority here is the LightGBM provider and not
    # ``get_provider(cfg.model)``.
    provider = LGBMProvider()
    check_param_names(
        provider,
        [("calibration.params", name) for name in params],
        model_name="lgbm",
        extra_accepted=CALIBRATOR_OWN_PARAM_NAMES,
    )
    check_duplicate_identities(provider, dict(params), surface="calibration.params")


def canonicalise_calibration_params(params: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``calibration.params`` names to the spellings the calibrator merges by.

    The calibrator merges the caller's parameters over its own defaults **by
    spelling** -- ``{**_ISOTONIC_DEFAULTS, **user}`` -- and those defaults are
    written in LightGBM's canonical names. So ``{"eta": 0.5}`` did not replace
    anything: the merged dict carried ``learning_rate: 0.03`` from the defaults
    *and* ``eta: 0.5`` from the caller, LightGBM resolved the two to one
    parameter and kept the canonical one, and the override was defeated in
    silence by the value it was written to override (H-0094 decision 8, review
    round 12). ``random_state`` was the same defect against the seed the facade
    supplies.

    The same-layer identity refusal does not reach this, and should not: the
    caller wrote the parameter **once**. The collision is between the caller's
    layer and the calibrator's defaults, so it is resolved by putting both in
    one spelling before they meet, rather than by teaching the calibrator about
    aliases -- ``lizyml/calibration/`` may not import ``lizyml/estimators/``.

    The calibrator's **own** parameters are left alone. ``num_boost_round`` is a
    LightGBM alias of ``num_iterations``, and canonicalising it would rename the
    key the calibrator pops for its boosting rounds; see
    ``CALIBRATOR_OWN_PARAM_NAMES``.

    Args:
        params: ``cfg.calibration.params``, as the caller wrote it.

    Returns:
        A new dict with every LightGBM-known name in its canonical spelling.
        When two of the caller's spellings canonicalise to one name they have
        already been refused unless they carry the same value, so the surviving
        entry means what both spellings meant.
    """
    if not params:
        return dict(params)

    from lizyml.calibration.isotonic import CALIBRATOR_OWN_PARAM_NAMES
    from lizyml.estimators.lgbm.provider import LGBMProvider

    canonical = LGBMProvider().canonical_param_names(params)
    return {
        name if name in CALIBRATOR_OWN_PARAM_NAMES else canonical[name]: value
        for name, value in params.items()
    }
