"""Factory functions for building splitters and inner-validation strategies.

Extracted from Model to reduce model.py size (H-0042).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

from lizyml.config.schema import (
    GroupKFoldConfig,
    GroupTimeSeriesConfig,
    HoldoutInnerValidConfig,
    KFoldConfig,
    LizyMLConfig,
    PurgedTimeSeriesConfig,
    SplitConfig,
    StratifiedKFoldConfig,
    TimeSeriesConfig,
)
from lizyml.splitters.base import BaseSplitter
from lizyml.splitters.group_kfold import GroupKFoldSplitter
from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter
from lizyml.splitters.kfold import KFoldSplitter, StratifiedKFoldSplitter
from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter
from lizyml.splitters.time_series import TimeSeriesSplitter
from lizyml.training.inner_valid import (
    GroupHoldoutInnerValid,
    HoldoutInnerValid,
    NoInnerValid,
    TimeHoldoutInnerValid,
)

InnerValidType = (
    HoldoutInnerValid | GroupHoldoutInnerValid | TimeHoldoutInnerValid | NoInnerValid
)


def _build_splitter_for_method(
    split_cfg: SplitConfig,
    n_splits: int,
) -> BaseSplitter:
    """Build a splitter from split config, using the given *n_splits*.

    Shared implementation for both outer CV and calibration CV splitters.
    The *n_splits* parameter is separated so that callers can override it
    (e.g. ``calibration.n_splits`` instead of ``split.n_splits``).
    """
    if isinstance(split_cfg, StratifiedKFoldConfig):
        return StratifiedKFoldSplitter(
            n_splits=n_splits,
            shuffle=True,
            random_state=split_cfg.random_state,
        )
    if isinstance(split_cfg, GroupKFoldConfig):
        return GroupKFoldSplitter(n_splits=n_splits)
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
    # Default: KFoldConfig (or any unmatched variant)
    if isinstance(split_cfg, KFoldConfig):
        return KFoldSplitter(
            n_splits=n_splits,
            shuffle=split_cfg.shuffle,
            random_state=split_cfg.random_state,
        )
    # Fallback — should not be reachable with the current union
    return KFoldSplitter(n_splits=n_splits, shuffle=True, random_state=42)


def build_splitter(cfg: LizyMLConfig) -> BaseSplitter:
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

    return _build_splitter_for_method(split_cfg, split_cfg.n_splits)


def build_calibration_splitter(cfg: LizyMLConfig) -> BaseSplitter:
    """Instantiate calibration CV splitter from config (H-0044).

    Inherits ``split.method`` and its parameters (gap, purge_gap, embargo,
    etc.) but uses ``calibration.n_splits`` for the fold count.
    """
    assert cfg.calibration is not None  # noqa: S101
    return _build_splitter_for_method(cfg.split, cfg.calibration.n_splits)


def _resolve_auto_inner_valid(
    split_method: str, ratio: float, seed: int
) -> HoldoutInnerValid | GroupHoldoutInnerValid | TimeHoldoutInnerValid:
    """Resolve inner validation strategy based on the outer split method."""
    if split_method == "stratified_kfold":
        return HoldoutInnerValid(ratio=ratio, random_state=seed, stratify=True)
    if split_method == "group_kfold":
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
    - ``group_kfold`` → ``GroupHoldoutInnerValid``
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
        return _resolve_auto_inner_valid(split_method, 0.1, seed)
    if not es._inner_valid_explicit:
        return _resolve_auto_inner_valid(split_method, iv_cfg.ratio, seed)

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
    HoldoutInnerValid | GroupHoldoutInnerValid | TimeHoldoutInnerValid,
]:
    """Return a factory that produces InnerValidStrategy for a given ratio.

    Used by the Tuner when ``validation_ratio`` is a search dimension.
    """
    split_method = cfg.split.method
    seed = cfg.training.seed

    def factory(
        ratio: float,
    ) -> HoldoutInnerValid | GroupHoldoutInnerValid | TimeHoldoutInnerValid:
        return _resolve_auto_inner_valid(split_method, ratio, seed)

    return factory
