"""Inputs ``export_code`` gives the generated ``train.py`` to match the refit (H-0120).

The generated project must train on what the LizyML refit trained on: the same
inner-validation split, the same row weights and the same category codes. These
helpers read each of them from where the fit left it -- the persisted config and
training overlay, the fitted pipeline state, the recorded dtypes -- and describe
it in plain JSON values for ``config.json``.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from lizyml.config.schema import BlockedGroupKFoldConfig, LizyMLConfig
from lizyml.core._model_factories import resolve_inner_valid
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.training.inner_valid import (
    BaseInnerValidStrategy,
    BlockedGroupInnerValid,
    GroupHoldoutInnerValid,
    HoldoutInnerValid,
    NoInnerValid,
    TimeHoldoutInnerValid,
)


def _group_column(cfg: LizyMLConfig) -> str | None:
    """The column whose values the fit passed to the inner split as ``groups``.

    A blocked split replaces the data group column with its own (``groups.col``,
    ``dataframe_builder.prepare_for_split``); every other split passes
    ``data.group_col``.
    """
    if isinstance(cfg.split, BlockedGroupKFoldConfig):
        return cfg.split.groups.col
    return cfg.data.group_col


def describe_inner_valid(
    strategy: BaseInnerValidStrategy, cfg: LizyMLConfig
) -> dict[str, Any] | None:
    """The strategy as ``config.json["inner_valid"]``; ``None`` for no split.

    Raises:
        LizyMLError: With ``CONFIG_INVALID`` for a strategy the generated code
            does not implement, rather than exporting a split it would not make.
    """
    if isinstance(strategy, NoInnerValid):
        return None
    if isinstance(strategy, HoldoutInnerValid):
        return {
            "method": "holdout",
            "ratio": strategy.ratio,
            "random_state": strategy.random_state,
            "stratify": strategy.stratify,
        }
    if isinstance(strategy, GroupHoldoutInnerValid):
        return {
            "method": "group_holdout",
            "ratio": strategy.ratio,
            "group_col": _group_column(cfg),
        }
    if isinstance(strategy, TimeHoldoutInnerValid):
        return {"method": "time_holdout", "ratio": strategy.ratio, "gap": strategy.gap}
    if isinstance(strategy, BlockedGroupInnerValid):
        return {
            "method": "blocked_group",
            "ratio": strategy.ratio,
            "task": strategy.task,
            "group_col": _group_column(cfg),
        }
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"export_code cannot reproduce the inner-validation strategy "
            f"{type(strategy).__name__}."
        ),
        context={"strategy": type(strategy).__name__},
    )


def exported_inner_valid(
    cfg: LizyMLConfig, applied_training_params: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Rebuild the refit's strategy from the fit's inputs and describe it.

    The strategy object is not persisted, so it is built again by the function
    the fit used, from the config and the applied training overlay (H-0109).
    ``build_inner_valid`` warns about an explicit shuffled holdout over a
    time-ordered split; the fit already gave that warning, so it is silenced here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        strategy = resolve_inner_valid(cfg, applied_training_params)
    return describe_inner_valid(strategy, cfg)


def input_categories(
    df: pd.DataFrame, feature_names: list[str]
) -> dict[str, list[Any]]:
    """The features the input frame itself holds as ``category``, with categories.

    Read before anything else touches the frame: LizyML's data builder casts
    every categorical column to ``category``, so after it ``FitResult.dtypes``
    marks inferred columns and declared ones alike (H-0120 amendment 4). Only a
    declared column's categories -- order and unused ones included -- must be
    restored when a CSV loses the dtype; an inferred column is re-inferred.
    """
    return {
        col: list(df[col].cat.categories)
        for col in feature_names
        if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype)
    }


def derived_sample_weight(
    cfg: LizyMLConfig, provider: Any, best_smart_params: dict[str, Any] | None
) -> str | None:
    """The weight rule a fit with this config and tuning result applies.

    The same resolution as ``smart_params.resolve_smart_params``: ``balanced``
    from the config overlaid with the tuning result's smart parameters, ``None``
    meaning on for classification; only multiclass turns it into per-row
    weights (binary writes ``scale_pos_weight``, which the adapter carries).
    Used only when the fit's own record is unknown (H-0120 amendment 1).
    """
    smart = {**provider.extract_smart_params(cfg.model), **(best_smart_params or {})}
    balanced = smart.get("balanced")
    if balanced is None:
        balanced = cfg.task != "regression"
    return "balanced" if balanced and cfg.task == "multiclass" else None
