"""Factory functions for building splitters, inner-validation, and provider dispatch.

Extracted from Model to reduce model.py size (H-0042).
Provider dispatch added in H-0053.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
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


def _resolve_stratify(stratify: str | bool, task: str) -> bool:
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
    task: str | None = None,
    seed: int | None = None,
) -> BaseSplitter:
    """Build a splitter from split config, using the given *n_splits*.

    Shared implementation for both outer CV and calibration CV splitters.
    The *n_splits* parameter is separated so that callers can override it
    (e.g. ``calibration.n_splits`` instead of ``split.n_splits``).
    """
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
            random_state=42 if seed is None else seed,
            min_train_rows=split_cfg.min_train_rows,
            min_valid_rows=split_cfg.min_valid_rows,
        )
    if isinstance(split_cfg, StratifiedKFoldConfig):
        return StratifiedKFoldSplitter(
            n_splits=n_splits,
            shuffle=True,
            random_state=split_cfg.random_state,
        )
    if isinstance(split_cfg, GroupKFoldConfig):
        return GroupKFoldSplitter(n_splits=n_splits)
    if isinstance(split_cfg, StratifiedGroupKFoldConfig):
        return StratifiedGroupKFoldSplitter(
            n_splits=n_splits,
            shuffle=split_cfg.shuffle,
            random_state=split_cfg.random_state,
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
            random_state=split_cfg.random_state,
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


def build_splitter(
    cfg: LizyMLConfig,
    *,
    block_values: npt.NDArray[Any] | None = None,
    task: str | None = None,
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

    # BlockedGroupKFoldConfig has no n_splits at top level
    if isinstance(split_cfg, BlockedGroupKFoldConfig):
        return _build_splitter_for_method(
            split_cfg,
            split_cfg.groups.n_splits,
            block_values=block_values,
            task=cfg.task if task is None else task,
            seed=cfg.training.seed if seed is None else seed,
        )

    return _build_splitter_for_method(split_cfg, split_cfg.n_splits)


def build_calibration_splitter(cfg: LizyMLConfig) -> BaseSplitter:
    """Instantiate calibration CV splitter from config (H-0044).

    .. deprecated:: H-0058
        Calibration cross-fit now reuses outer CV splits.
        This function is kept for backward compatibility only.
    """
    import warnings

    warnings.warn(
        "build_calibration_splitter is deprecated (H-0058). "
        "Calibration cross-fit now reuses outer CV splits.",
        DeprecationWarning,
        stacklevel=2,
    )
    assert cfg.calibration is not None  # noqa: S101
    return _build_splitter_for_method(cfg.split, cfg.calibration.n_splits)


def _resolve_auto_inner_valid(
    split_method: str,
    ratio: float,
    seed: int,
    *,
    task: str | None = None,
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
        return TimeHoldoutInnerValid(ratio=ratio)
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

    # Auto-resolve: inner_valid absent or created from validation_ratio default
    if iv_cfg is None:
        return _resolve_auto_inner_valid(split_method, 0.1, seed, task=cfg.task)
    if not es._inner_valid_explicit:
        return _resolve_auto_inner_valid(
            split_method, iv_cfg.ratio, seed, task=cfg.task
        )

    # Explicit config — dispatch by concrete type
    if isinstance(iv_cfg, HoldoutInnerValidConfig):
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

    def factory(
        ratio: float,
    ) -> (
        HoldoutInnerValid
        | GroupHoldoutInnerValid
        | TimeHoldoutInnerValid
        | BlockedGroupInnerValid
    ):
        return _resolve_auto_inner_valid(split_method, ratio, seed, task=task)

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
