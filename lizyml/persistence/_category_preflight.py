"""_category_preflight -- does a saved category state answer as in memory (H-0136)?

Before anything is written, ``Model.export``, ``lizyml.persistence.export``
and ``export_code`` project each saved encoder state through the two
serializers a saved model uses, and compare the projection with the state
itself (``features.pipeline_base.category_state_changes``):

- ``joblib``: the state through ``joblib.dump(..., compress=3)`` and back, as
  ``fit_result.pkl`` / ``refit_model.pkl`` hold it;
- ``json``: each category and mode through the H-0133 codec and ``json``, as
  the codegen ``pipeline_state.json`` holds them.

``declared_categories`` reach a saved model only as JSON (``metadata.json``
and ``config.json``), so they go through ``json`` alone. One check serves the
three exits, so they accept the same models (H-0136 judgement 2).
"""

from __future__ import annotations

import io
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from lizyml.core.category_values import decode, encode, shown, type_name
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.features.pipeline_base import CategoryStateChange, category_state_changes

#: The serializers, in the order a column is judged.
SERIALIZERS = ("joblib", "json")


@dataclass(frozen=True)
class SerializedChange:
    """A column that answers differently after *serializer* (H-0136 P3)."""

    serializer: str
    change: CategoryStateChange


def _through_joblib(encoder: Mapping[str, Any]) -> Any:
    buffer = io.BytesIO()
    joblib.dump(dict(encoder), buffer, compress=3)
    buffer.seek(0)
    return joblib.load(buffer)


def through_json(value: Any) -> Any:
    """*value* after the H-0133 codec and a JSON round trip."""
    return decode(json.loads(json.dumps(encode(value), ensure_ascii=False)))


def _encoder_through_json(encoder: Mapping[str, Any]) -> dict[str, Any]:
    categories = encoder.get("categories") or {}
    modes = encoder.get("modes") or {}
    return {
        **encoder,
        "categories": {
            c: [through_json(v) for v in cats] for c, cats in categories.items()
        },
        "modes": {c: None if m is None else through_json(m) for c, m in modes.items()},
    }


def state_changes(state: Any) -> list[SerializedChange]:
    """Each (serializer, column) of pipeline *state* whose projection answers
    differently from the state, joblib first, columns in order."""
    if not isinstance(state, Mapping):
        return []
    encoder = state.get("encoder")
    if not isinstance(encoder, Mapping) or not isinstance(
        encoder.get("categories"), Mapping
    ):
        return []
    found: list[SerializedChange] = []
    for serializer in SERIALIZERS:
        project = _through_joblib if serializer == "joblib" else _encoder_through_json
        # Runs after the V check (H-0133): every value here pickles and encodes.
        projected = project(encoder)
        found.extend(
            SerializedChange(serializer, c)
            for c in category_state_changes(encoder, projected)
        )
    return found


def _detail(change: CategoryStateChange) -> str:
    value = change.value
    if change.kind == "raises":
        return f"restoring its categories raises {change.exception}"
    if value is None:
        return "a learned category would be unseen"
    named = f"{shown(value)} of type {type_name(value)}"
    if change.kind == "unseen":
        return f"the learned category {named} would be unseen"
    if change.kind == "codes":
        return f"the learned category {named} would get another code"
    return f"the mode {named} would get another code"


def refuse_change(
    change: CategoryStateChange, serializer: str, where: str, surface: str
) -> LizyMLError:
    """The H-0136 P3 refusal. *where* is the state name (``"refit"``,
    ``"fold 1"``, ``"last fold"``) or ``"declared categories"``."""
    shown_where = where if where == "declared categories" else f"{where} pipeline state"
    context: dict[str, Any] = {
        "column": change.column,
        "state": where,
        "serializer": serializer,
        "kind": change.kind,
    }
    if change.value is not None:
        context.update(value=shown(change.value), type=type_name(change.value))
    if change.exception is not None:
        context["exception"] = change.exception
    return LizyMLError(
        code=ErrorCode.SERIALIZATION_FAILED,
        user_message=(
            f"{surface} cannot save this model: column '{change.column}' "
            f"({shown_where}) would not answer like the fitted model once saved "
            f"({serializer}: {_detail(change)}). Model.export and export_code "
            "accept the same models, so both refuse this one. Convert the "
            "column's values to str (or int, float, bool) and fit again to "
            "save it (H-0136)."
        ),
        context=context,
    )


def warn_changes(states: list[Any], declared: Mapping[str, list[Any]] | None) -> None:
    """Warn once per column the H-0136 check would refuse in any of *states*
    or in *declared* (P4)."""
    seen: set[str] = set()
    found = [(c.serializer, c.change) for state in states for c in state_changes(state)]
    for column, cats in (declared or {}).items():
        change = declared_change(column, cats)
        if change is not None:
            found.append(("json", change))
    for serializer, change in found:
        if change.column in seen:
            continue
        seen.add(change.column)
        warnings.warn(
            f"Column '{change.column}' would not answer like this model once "
            f"saved ({serializer}: {_detail(change)}): fit and predict work, "
            "but Model.export and export_code will refuse this model. Convert "
            "the column's values to str (or int, float, bool) before fit to be "
            "able to save it (H-0136).",
            UserWarning,
            stacklevel=4,
        )


def declared_change(column: Any, categories: list[Any]) -> CategoryStateChange | None:
    """How a declared categorical dtype answers differently after JSON, or
    ``None``. A dtype pandas cannot build from the original values is not
    judged (the fit's own input would fail first)."""
    learned = list(categories)
    try:
        want = (
            pd.Series(learned, dtype=object)
            .astype(pd.CategoricalDtype(learned))
            .cat.codes
        )
    except Exception:  # noqa: BLE001 -- judged only where the original answers
        return None
    name = str(column)
    try:
        dtype = pd.CategoricalDtype([through_json(v) for v in learned])
        got = pd.Series(learned, dtype=object).astype(dtype).cat.codes
    except Exception as exc:  # noqa: BLE001 -- the change measured (H-0136 P3)
        return CategoryStateChange(name, "raises", exception=type(exc).__name__)
    for value, g, w in zip(learned, got.tolist(), want.tolist(), strict=True):
        if g != w:
            return CategoryStateChange(
                name, "unseen" if g < 0 else "codes", value=value
            )
    return None
