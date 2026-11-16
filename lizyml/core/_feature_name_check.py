"""_feature_name_check -- column names refused before fit and tune train (H-0148).

Two checks, both before any training (``Model.tune`` runs no trial and
writes no study):

- :func:`lizyml.data.dataframe_builder.check_duplicate_columns` runs on the
  loaded frame with the roles :func:`config_roles` reads, before the frame
  is split into features and roles.
- :func:`check_feature_names_for_training` asks the provider's optional
  :class:`~lizyml.estimators.provider.FeatureNameRuleProvider` capability
  which training column name the estimator library refuses; a provider
  without it is not checked.

Both refuse with ``DATA_SCHEMA_INVALID`` and a ``reason`` token.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from lizyml.config.schema import BlockedGroupKFoldConfig, LizyMLConfig
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.provider import FeatureNameRuleProvider


def config_roles(cfg: LizyMLConfig) -> dict[str, Any]:
    """The Config's column-name fields that are set, by their Config path.

    These five are the only Config fields that name a column; inner
    validation reuses the groups built from ``data.group_col`` or
    ``split.groups.col``.
    """
    roles: dict[str, Any] = {
        "data.target": cfg.data.target,
        "data.time_col": cfg.data.time_col,
        "data.group_col": cfg.data.group_col,
    }
    if isinstance(cfg.split, BlockedGroupKFoldConfig):
        roles["split.blocks.col"] = cfg.split.blocks.col
        roles["split.groups.col"] = cfg.split.groups.col
    return {where: col for where, col in roles.items() if col is not None}


def _refusal_text(refused: Any) -> str:
    if refused.reason == "json_special_character_in_feature_name":
        return (
            "holds a character the estimator refuses in a feature name "
            '(one of " , : [ ] { })'
        )
    return (
        f"becomes {refused.estimator_name!r} for the estimator, the same as the "
        f"column at position {refused.first_position}"
    )


def selected_labels(X: pd.DataFrame) -> list[Any]:
    """The column labels training sees: ``select_training_columns`` selects
    the training columns by list (``X[list(columns)]``), and pandas may
    rewrite labels there (pandas 3.0 turns ``None`` beside NaN into NaN;
    2.x keeps it), so the estimator's rule is judged on the selected labels
    of the running pandas (H-0148 owner ruling). Selected on the empty frame.
    """
    return list(X.iloc[:0][list(X.columns)].columns)


def check_feature_names_for_training(provider: Any, X: pd.DataFrame) -> None:
    """Refuse training columns the provider's estimator refuses (H-0148).

    Raises:
        LizyMLError: ``DATA_SCHEMA_INVALID`` naming the column, its position,
            the name the estimator sees and the ``reason``; a repeat also names
            the earlier column.
    """
    if not isinstance(provider, FeatureNameRuleProvider):
        return
    names = list(X.columns)
    refused = provider.refused_feature_names(selected_labels(X))
    if refused is None:
        return
    context: dict[str, Any] = {
        "column": names[refused.position],
        "position": refused.position,
        "estimator_name": refused.estimator_name,
        "reason": refused.reason,
    }
    if refused.first_position is not None:
        context["first_column"] = names[refused.first_position]
        context["first_position"] = refused.first_position
    raise LizyMLError(
        code=ErrorCode.DATA_SCHEMA_INVALID,
        user_message=(
            f"The feature column {names[refused.position]!r} "
            f"(position {refused.position}) "
            f"{_refusal_text(refused)}, so training would fail. Rename the "
            "column before fit or tune (H-0148)."
        ),
        context=context,
    )
