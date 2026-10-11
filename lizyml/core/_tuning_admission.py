"""Admission checks for ``Model.tune`` (H-0093, H-0094, H-0099, H-0122).

What a tune call must satisfy before any study write, in the order ``tune``
calls them:

* phase A -- the call's own arguments: :func:`validate_tune_args`, and for
  ``tune(space=...)`` :func:`parse_space_arg` (its shape) and
  :func:`refuse_space_on_create` (a call that creates a study takes its first
  space from the config, §3);
* phase D -- the space's names: :func:`check_space_names` on the config space
  and, retargeted (H-0122 5.2-4), on a ``space=`` declaration; then
  :func:`check_round_space` on the space the round actually samples.

Every function reads only and raises ``CONFIG_INVALID``; none writes the
study or the Model (INV-R4).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lizyml.core._model_factories import (
    check_calibration_param_names,
    check_duplicate_space_dimensions,
    check_param_names,
    check_training_managed_overrides,
    check_training_managed_space,
    model_space_names,
    overlay_params,
)
from lizyml.core._smart_combinations import (
    FIXED_LAYER,
    check_layers_over_smart_values,
)
from lizyml.core._tuning_space import SPACE_ARG
from lizyml.core._tuning_validation import validate_tuning_dimensions
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning.lineage_codec import DIM_CATEGORIES
from lizyml.tuning.search_space import parse_space
from lizyml.tuning.tuner import precheck_study_args

if TYPE_CHECKING:
    from optuna.storages import BaseStorage

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core._tuning_plan import RoundPlan
    from lizyml.core.types.search_dim import SearchDim
    from lizyml.estimators.provider import EstimatorProvider


# ---------------------------------------------------------------------------
# Phase A: the arguments
# ---------------------------------------------------------------------------


def validate_tune_args(
    cfg: LizyMLConfig,
    *,
    boundary_threshold: float,
    storage: str | BaseStorage | None,
    study_name: str | None,
    caller: str = "tune",
) -> None:
    """Validate cfg + tune-call invariants. Raises ``LizyMLError`` on violation.

    *caller* names the public method in the argument errors' ``surface``.
    """
    if cfg.tuning is None:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                "No tuning configuration found. "
                "Add a 'tuning' section to the config to enable tuning."
            ),
            context={"task": cfg.task, "model": getattr(cfg.model, "name", None)},
        )

    precheck_study_args(storage, study_name, caller=caller)

    if not 0.0 < boundary_threshold < 0.5:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"boundary_threshold must be in (0.0, 0.5), got {boundary_threshold}."
            ),
            context={"boundary_threshold": boundary_threshold},
        )


def _shape_error(message: str, **context: Any) -> LizyMLError:
    return LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"{SPACE_ARG}: {message} space= takes the form of tuning.optuna.space: "
            "{name: {type: float | int | categorical, ...}}."
        ),
        context={"surface": SPACE_ARG, **context},
    )


def parse_space_arg(
    space: Any,
) -> tuple[dict[str, dict[str, Any]], list[SearchDim]]:
    """Check the shape of ``tune(space=...)`` and parse it (phase A).

    A mapping of ``str`` names to mappings, each with a ``category`` (when
    given) of ``model`` / ``smart`` / ``training``; then ``parse_space``,
    whose own refusals stand. A missing or ill-typed ``low`` / ``high`` --
    which ``parse_space`` would raise as a bare ``KeyError`` / ``ValueError``
    -- is refused here as ``CONFIG_INVALID`` too. The config space gets no
    such check (its schema types it as a plain dict; design R-8).

    Returns:
        ``(the argument as plain dicts, its parsed dims)``. The name checks
        read the plain form, which treats only a ``dict`` spec as carrying a
        category.
    """
    if not isinstance(space, Mapping):
        raise _shape_error(
            f"expected a mapping of dimensions, got {type(space).__name__}.",
            reason="not_a_mapping",
        )
    for name, spec in space.items():
        if type(name) is not str:
            raise _shape_error(
                f"dimension name {name!r} is not a str.", reason="name_not_str"
            )
        if not isinstance(spec, Mapping):
            raise _shape_error(
                f"dimension '{name}' must be a mapping, got {type(spec).__name__}.",
                reason="dim_not_a_mapping",
                dim=name,
            )
        category = spec.get("category", "model")
        if type(category) is not str or category not in DIM_CATEGORIES:
            raise _shape_error(
                f"dimension '{name}' has category {category!r}; expected one of "
                f"{sorted(DIM_CATEGORIES)}.",
                reason="unknown_category",
                dim=name,
            )
    plain = {name: dict(spec) for name, spec in space.items()}
    try:
        return plain, parse_space(plain)
    except (KeyError, TypeError, ValueError) as exc:
        raise _shape_error(
            f"a dimension is malformed ({type(exc).__name__}: {exc}).",
            reason="malformed_dim",
        ) from exc


def refuse_space_on_create(
    given: bool,
    *,
    resume: bool,
    target_kind: str | None = None,
    state: str | None = None,
) -> None:
    """Refuse ``space=`` on a call that creates a study (§3, §7.4 E / N).

    Called twice: in phase A with ``resume`` only (``resume=False`` always
    creates a study, or is refused as row 3), so no storage is touched; and
    in phase C with the located target and its state -- a ``resume=True``
    whose storage lacks the study (U3), or a stored study in state E or N.
    A study's first space is the config's ``tuning.optuna.space``.
    """
    if not given:
        return
    if not resume:
        reason = "resume_false"
    elif target_kind in ("create_storage", "create_memory"):
        reason = "study_absent"
    elif state in ("uninitialized", "no_round"):
        reason = state
    else:
        return
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"{SPACE_ARG} declares a later round's space, but this call creates a "
            "study or starts its round 1 (resume=False, a resume=True whose "
            "storage lacks the study, or a stored study with no round yet). A "
            "study's first space is tuning.optuna.space; pass space= to a later "
            "tune(resume=True)."
        ),
        context={"surface": SPACE_ARG, "reason": reason, "creates_study": True},
    )


# ---------------------------------------------------------------------------
# Phase D: the space's names and the round's space
# ---------------------------------------------------------------------------


def check_space_names(
    provider: EstimatorProvider,
    cfg: LizyMLConfig,
    *,
    space: Mapping[str, Any] | None = None,
    surface: str = "tuning.optuna.space",
) -> None:
    """The space-level name checks, on the config space or a ``space=``.

    Run after the study's record has been checked (§5.1), so an edited config
    space gets that message first. With *space*, the same checks read the
    ``tune(space=...)`` declaration in its raw form (H-0122 5.2-4); the
    calibration check belongs to the config and runs only without it.
    """
    # H-0093: a `category: model` dimension whose name the estimator
    # does not know is sampled by Optuna, forwarded, and discarded --
    # so every trial in the study really explores the other axes, and
    # nothing says so. Checked before the study starts; trial params
    # are drawn from these names, so covering the space covers them.
    check_param_names(
        provider,
        model_space_names(cfg, space=space, surface=surface),
        model_name=cfg.model.name,
    )
    # The same-layer rule, on the layer decision 6 had not reached. Two
    # dimensions spelling one parameter both land in the trial dict, and
    # LightGBM keeps the canonical one -- so the other is sampled and
    # optimised over without affecting any trial (H-0094 decision 8).
    check_duplicate_space_dimensions(provider, cfg, space=space, surface=surface)
    # And the training-managed rule, on the same layer. `_merge_params`
    # checks the three inputs that meet there; trial parameters overlay
    # afterwards, so without this a study sampled a parameter
    # `training.*` controls, trained on it, and returned a
    # `best_model_params` the following `fit()` refused (H-0094 decision
    # 10, review round 14).
    check_training_managed_space(provider, cfg, space=space, surface=surface)
    if space is not None:
        return
    # The calibration surface is checked here too, not only on the fit path.
    # `tune()` is its own entry point: without this, a config carrying a
    # dead `calibration.params` name completes a whole study and is refused
    # only by the `fit()` that follows, having already trained. Calibration
    # itself does not run during tuning -- what this buys is the ordering
    # H-0093 decision 6 and BLUEPRINT 12.2 declare, on every entry point
    # rather than on one of them.
    check_calibration_param_names(cfg.calibration)


def check_round_space(
    cfg: LizyMLConfig,
    plan: RoundPlan,
    provider: EstimatorProvider,
    base_model_params: dict[str, Any],
    base_smart_params: dict[str, Any],
) -> None:
    """Admission checks on the space the round samples (H-0094, H-0099).

    Run on ``plan.effective``, so a ``space=`` round's dims are checked like
    the config's.
    """
    space = list(plan.effective)
    surface = SPACE_ARG if plan.source == "space_arg" else "tuning.optuna.space"
    validate_tuning_dimensions(
        provider, space, base_smart_params, cfg.task, surface=surface
    )
    check_smart_managed_layers(
        cfg, plan, provider, base_model_params, base_smart_params, surface=surface
    )
    # Resolved defaults can introduce training ownership even when Config
    # disables it. Check every native name the objective can send before
    # creating a study. Early-stopping ownership depends on presence of a
    # training override, not its sampled patience; 1 represents that state.
    training_claims = (
        {"early_stopping_rounds": 1}
        if any(
            dim.category == "training" and dim.name == "early_stopping_rounds"
            for dim in space
        )
        else None
    )
    fixed = dict(plan.fixed)
    resolved_model = overlay_params(provider, base_model_params, fixed)
    check_training_managed_overrides(
        provider, resolved_model, cfg, training_overrides=training_claims
    )
    model_dimensions = dict.fromkeys(
        dim.name for dim in space if dim.category == "model"
    )
    check_training_managed_overrides(
        provider,
        model_dimensions,
        cfg,
        origins=dict.fromkeys(model_dimensions, surface),
        training_overrides=training_claims,
    )


def check_smart_managed_layers(
    cfg: LizyMLConfig,
    plan: RoundPlan,
    provider: EstimatorProvider,
    base_model_params: dict[str, Any],
    base_smart_params: dict[str, Any],
    *,
    surface: str,
) -> None:
    """Refuse a layer name a smart dimension can make managed (H-0121 am. 1).

    ``Model._merge_params`` checks every parameter layer against the base
    smart values (H-0121 policy 1). A ``category: smart`` dimension overlays its
    sample on each trial, so a smart parameter the config leaves inactive can be
    active inside a trial, and the layer's value is then discarded for the one
    the smart parameter writes (#341). The layers a trial trains on --
    ``model.params`` and the round's fixed parameters -- are checked again
    against every combination of the values the round's smart dimensions can
    take. Read only; raises ``CONFIG_INVALID`` before any study write (INV-R4).

    Args:
        cfg: The config.
        plan: The round; its ``effective`` space and ``fixed`` layer are read.
        provider: The estimator provider.
        base_model_params: ``model.params``, as ``_merge_params`` returned it.
        base_smart_params: The effective smart values the round starts from.
        surface: Where the round's space was declared, quoted in the refusal.
    """
    check_layers_over_smart_values(
        provider,
        [("model.params", base_model_params), (FIXED_LAYER, dict(plan.fixed))],
        plan.effective,
        base_smart_params,
        cfg.task,
        space_surface=surface,
    )
