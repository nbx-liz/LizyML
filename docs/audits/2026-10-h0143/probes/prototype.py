"""H-0143 prototype: the JSON verdict with keyed columns judged as generated code.

``state_changes`` judges the ``json`` route by running the runtime encoder on
the JSON-projected state. For a keyed column (``identity_columns``, H-0138)
the generated ``predict.py`` looks codes up by key and stops there; the runtime
encoder then also builds ``pd.Categorical.from_codes(codes, categories=...)``,
which pandas can refuse for categories the generated code never hands it.

``prototype_state_changes`` is ``state_changes`` with one difference: on the
``json`` route, a keyed column's codes (and its mode replacement) come from
``keyed_codes`` alone -- the function the generated copy reproduces (pinned by
the H-0138 copy tests). Unkeyed columns and the ``joblib`` route are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from lizyml.features.encoders.categorical_encoder import comparable_values
from lizyml.features.encoders import state_changes as sc
from lizyml.features.encoders.category_identity import keyed_codes
from lizyml.persistence import _category_preflight as pf


def _keyed(state: Mapping[str, Any], column: Any) -> bool:
    return "category_identity_version" in state and column in (
        state.get("identity_columns") or []
    )


def _generated_codes(
    state: Mapping[str, Any], column: Any, values: list[Any], policy: str = "error"
) -> list[int]:
    known = list(state["categories"][column])
    mode = (state.get("modes") or {}).get(column)
    arrived = pd.Series(values, dtype=object)
    series, changed = comparable_values(arrived)
    codes, _ = keyed_codes(series, arrived, changed, known, mode, policy, column)  # type: ignore[arg-type]
    return [int(c) for c in codes]


def _json_column_change(
    state: Mapping[str, Any], projected: Mapping[str, Any], column: Any
) -> sc.CategoryStateChange | None:
    if not _keyed(projected, column):
        return sc.column_change(state, projected, column)
    original = sc._column_codes
    try:
        def codes(
            target: Mapping[str, Any], col: Any, values: list[Any], policy: str = "error"
        ) -> list[int]:
            if target is projected:
                return _generated_codes(target, col, values, policy)
            return original(target, col, values, policy)  # type: ignore[arg-type]

        sc._column_codes = codes  # type: ignore[assignment]
        return sc.column_change(state, projected, column)
    finally:
        sc._column_codes = original


def prototype_state_changes(state: Any) -> list[pf.SerializedChange]:
    if not isinstance(state, Mapping):
        return []
    encoder = state.get("encoder")
    if not isinstance(encoder, Mapping) or not isinstance(
        encoder.get("categories"), Mapping
    ):
        return []
    found: list[pf.SerializedChange] = []
    for serializer in pf.SERIALIZERS:
        if serializer == "joblib":
            projected = pf._through_joblib(encoder)
            changes = sc.category_state_changes(encoder, projected)
        else:
            projected = pf._encoder_through_json(encoder)
            changes = [
                c
                for c in (
                    _json_column_change(encoder, projected, col)
                    for col in encoder["categories"]
                )
                if c is not None
            ]
        found.extend(pf.SerializedChange(serializer, c) for c in changes)
    return found
