"""state_changes -- does an encoder state answer like another one (H-0136)?

A saved model's encoder state passes through a serializer (joblib for the
artifact, the H-0133 codec JSON for ``export_code``) and comes back as a
projected state. This module compares the two the way ``predict`` uses them:
the learned categories, as one batch in the state's order, are transformed
by a ``CategoricalEncoder`` with ``unseen_policy="error"`` under each state,
and the mode is compared by its position among each state's categories.

The reference is the original (in-memory) state and its own categories. The
promise is for rows holding a learned value; a value the original state
treats as unseen is outside it (H-0136 "残る範囲").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

ChangeKind = Literal["unseen", "raises", "codes", "mode"]


@dataclass(frozen=True)
class CategoryStateChange:
    """How one column of a projected state answers differently (H-0136 P1).

    Attributes:
        column: The column name.
        kind: ``unseen`` (a learned category is not a category of the projected
            state), ``raises`` (the projected transform raises), ``codes`` (a
            learned category gets another code) or ``mode`` (the mode gets
            another code).
        value: The learned value whose answer changes; ``None`` for ``raises``.
        exception: The exception's type name for ``raises``, else ``None``.
    """

    column: str
    kind: ChangeKind
    value: Any = None
    exception: str | None = None


def _column_codes(
    state: Mapping[str, Any], column: Any, values: list[Any]
) -> list[int]:
    """The codes *values* get from *state*'s encoder for *column* alone."""
    encoder = CategoricalEncoder(unseen_policy="error").load_state(
        {
            "unseen_policy": "error",
            "categories": {column: list(state["categories"][column])},
            "modes": {column: (state.get("modes") or {}).get(column)},
            "stand_in_columns": [
                c for c in state.get("stand_in_columns", []) if c == column
            ],
        }
    )
    frame = pd.DataFrame({column: pd.Series(values, dtype=object)})
    codes: list[int] = encoder.transform(frame)[column].cat.codes.tolist()
    return codes


def _position(categories: list[Any], value: Any) -> int | None:
    """Where *value* sits in *categories*: the same object, else the first
    that is equal with an equal hash (pandas may hold a mode of another numeric
    type than its category); ``None`` when absent.

    The mode is compared by position, not by encoding it in a batch: a
    one-row batch can be cast differently by pandas (#379), and a repeated
    row makes pandas 2.0's ``unique`` rebuild a pytz value (#379 too).
    """
    for index, category in enumerate(categories):
        if category is value:
            return index
    for index, category in enumerate(categories):
        try:
            if bool(category == value) and hash(category) == hash(value):
                return index
        except Exception:  # noqa: BLE001 -- not this one; the next may match
            continue
    return None


def _first_missing(learned: list[Any], projected: list[Any]) -> Any:
    try:
        known = set(projected)
        return next((v for v in learned if v not in known), None)
    except Exception:  # noqa: BLE001 -- only the shown value; the kind stands
        return None


def column_change(
    state: Mapping[str, Any], projected: Mapping[str, Any], column: Any
) -> CategoryStateChange | None:
    """How *column* of *projected* answers differently from *state*, or ``None``.

    A column whose original state already fails to transform its own
    categories is not judged (fit and predict fail before any save).
    """
    learned = list(state["categories"][column])
    if not learned:
        return None
    mode = (state.get("modes") or {}).get(column)
    try:
        want = _column_codes(state, column, learned)
    except Exception:  # noqa: BLE001 -- judged only where the original answers
        return None
    name = str(column)
    # The projected state's own mode: it is what replaces an unseen row
    # under unseen_policy="mode" after the save.
    projected_mode = (projected.get("modes") or {}).get(column)
    want_mode = None if mode is None else _position(learned, mode)
    got_mode = (
        None
        if projected_mode is None
        else _position(list(projected["categories"][column]), projected_mode)
    )
    try:
        got = _column_codes(projected, column, learned)
    except LizyMLError as exc:
        if exc.code != ErrorCode.DATA_SCHEMA_INVALID:
            return CategoryStateChange(name, "raises", exception=type(exc).__name__)
        # unseen_policy="error": a learned category is unseen under *projected*.
        missing = _first_missing(learned, list(projected["categories"][column]))
        return CategoryStateChange(name, "unseen", value=missing)
    except Exception as exc:  # noqa: BLE001 -- the change measured (H-0136 P3)
        return CategoryStateChange(name, "raises", exception=type(exc).__name__)
    if got != want:
        index = next(
            i for i, (g, w) in enumerate(zip(got, want, strict=True)) if g != w
        )
        return CategoryStateChange(name, "codes", value=learned[index])
    # A mode the in-memory state cannot place among its own categories (pandas
    # 2.0 rebuilds a pytz value inside ``mode()``, #379) is not judged: the
    # in-memory model is already inconsistent there.
    if want_mode is not None and got_mode != want_mode:
        return CategoryStateChange(name, "mode", value=mode)
    return None


def category_state_changes(
    state: Mapping[str, Any], projected: Mapping[str, Any]
) -> list[CategoryStateChange]:
    """Each column of encoder *state* that *projected* answers differently, in
    column order (H-0136 P1)."""
    categories = state.get("categories")
    if not isinstance(categories, Mapping):
        return []
    found: list[CategoryStateChange] = []
    for column in categories:
        change = column_change(state, projected, column)
        if change is not None:
            found.append(change)
    return found
