"""Factory functions for building splitters and inner-validation strategies.

Extracted from Model to reduce model.py size (H-0042).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

from lizyml.config.schema import LizyMLConfig
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
    split_cfg: object,
    n_splits: int,
) -> BaseSplitter:
    """Build a splitter from split config, using the given *n_splits*.

    Shared implementation for both outer CV and calibration CV splitters.
    The *n_splits* parameter is separated so that callers can override it
    (e.g. ``calibration.n_splits`` instead of ``split.n_splits``).
    """
    method: str = getattr(split_cfg, "method", "kfold")
    random_state: int = getattr(split_cfg, "random_state", 42)
    shuffle: bool = getattr(split_cfg, "shuffle", True)

    if method == "stratified_kfold":
        return StratifiedKFoldSplitter(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    if method == "group_kfold":
        return GroupKFoldSplitter(n_splits=n_splits)
    if method == "time_series":
        gap = getattr(split_cfg, "gap", 0)
        train_size_max = getattr(split_cfg, "train_size_max", None)
        test_size_max = getattr(split_cfg, "test_size_max", None)
        return TimeSeriesSplitter(
            n_splits=n_splits,
            gap=gap,
            max_train_size=train_size_max,
            max_test_size=test_size_max,
        )
    if method == "purged_time_series":
        purge_gap = getattr(split_cfg, "purge_gap", 0)
        embargo: int = getattr(split_cfg, "embargo", 0)
        train_size_max = getattr(split_cfg, "train_size_max", None)
        test_size_max = getattr(split_cfg, "test_size_max", None)
        return PurgedTimeSeriesSplitter(
            n_splits=n_splits,
            purge_gap=purge_gap,
            embargo=embargo,
            max_train_size=train_size_max,
            max_test_size=test_size_max,
        )
    if method == "group_time_series":
        gap = getattr(split_cfg, "gap", 0)
        train_size_max = getattr(split_cfg, "train_size_max", None)
        test_size_max = getattr(split_cfg, "test_size_max", None)
        return GroupTimeSeriesSplitter(
            n_splits=n_splits,
            gap=gap,
            max_train_size=train_size_max,
            max_test_size=test_size_max,
        )
    # Default: kfold
    return KFoldSplitter(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


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

    # Auto-resolve when inner_valid is not explicitly set (includes
    # the case where it was auto-created from validation_ratio default)
    if iv_cfg is None or not es._inner_valid_explicit:
        ratio = iv_cfg.ratio if iv_cfg is not None else 0.1
        seed = cfg.training.seed
        return _resolve_auto_inner_valid(split_method, ratio, seed)

    # Explicit config — use getattr for fields not common to all variants
    method = iv_cfg.method
    if method == "holdout":
        return HoldoutInnerValid(
            ratio=iv_cfg.ratio,
            random_state=getattr(iv_cfg, "random_state", 42),
            stratify=getattr(iv_cfg, "stratify", False),
        )
    if method == "group_holdout":
        return GroupHoldoutInnerValid(
            ratio=iv_cfg.ratio,
            random_state=getattr(iv_cfg, "random_state", 42),
        )
    if method == "time_holdout":
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
