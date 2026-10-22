"""category_identity -- learn and look up categories by the identity key.

A column whose values (rows and declared categories) are all in V is an
*identity column* (H-0138; temporal values since H-0139). For it:

- fit keeps today's categories, order, representatives and mode whenever they
  are faithful under the key (:func:`keyed_categories`); otherwise it builds
  the categories by key and the column reaches the model as stand-in
  positions;
- transform looks every row up by its key (:func:`keyed_codes`), never through
  pandas' object Index, ``set`` or ``isin``.

A column holding a value outside V keeps today's pandas path (H-0138); this
module never sees it. Fit refuses a ``Y`` / ``M`` duration before learning
(H-0139); one arriving at transform has no key and is unseen.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.category_values import (
    cast_is_faithful,
    category_identity_key,
    is_missing,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError

UnseenPolicy = Literal["mode", "nan", "error"]


def _today(series: pd.Series) -> tuple[list[Any], Any]:
    """The categories and mode today's encoder learns (may raise)."""
    if hasattr(series, "cat"):
        cats = list(series.cat.categories)
    else:
        cats = sorted(series.dropna().unique().tolist(), key=str)
    if not cats:
        return cats, None
    mode = series.mode()
    return cats, mode.iloc[0] if len(mode) > 0 else cats[0]


def _faithful_today(series: pd.Series, cats: list[Any]) -> bool:
    """Today's categories agree with the key, pandas can hold them as they
    are, and today's lookup codes the training rows by key (executed: the
    cast and ``set_categories`` today's transform runs; it may raise)."""
    keys = [category_identity_key(c) for c in cats]
    if len(set(keys)) != len(keys):
        return False
    known = set(keys)
    if not all(
        category_identity_key(v) in known
        for v in series.to_numpy(dtype=object)
        if not is_missing(v)
    ):
        return False
    if not _holdable(cats, keys):
        return False
    if isinstance(series.dtype, pd.CategoricalDtype):
        coded = series
    else:
        coded = series.astype("category").cat.set_categories(cats)
    return cast_is_faithful(series, coded)


def _holdable(cats: list[Any], keys: list[Any]) -> bool:
    """Can a pandas ``CategoricalDtype`` hold *cats* unchanged? (executed)"""
    try:
        held = list(pd.CategoricalDtype(cats).categories)
    except (TypeError, ValueError, OverflowError):
        return False
    return [category_identity_key(c) for c in held] == keys


def _rebuilt(series: pd.Series) -> tuple[list[Any], Any]:
    """Categories by key: the first value of each key; the mode by key counts."""
    declared = isinstance(series.dtype, pd.CategoricalDtype)
    reps: dict[Any, Any] = {}
    sources = list(series.cat.categories) if declared else []
    sources += [v for v in series.to_numpy(dtype=object) if not is_missing(v)]
    for value in sources:
        reps.setdefault(category_identity_key(value), value)
    cats = list(reps.values()) if declared else sorted(reps.values(), key=str)
    if not cats:
        return cats, None
    position = {category_identity_key(c): i for i, c in enumerate(cats)}
    counts = [0] * len(cats)
    for value in series.to_numpy(dtype=object):
        if not is_missing(value):
            counts[position[category_identity_key(value)]] += 1
    best = max(range(len(cats)), key=lambda i: (counts[i], -i))
    return cats, cats[best]


def keyed_categories(series: pd.Series) -> tuple[list[Any], Any, bool]:
    """``(categories, mode, rebuilt)`` for an identity column.

    Today's answer when it is faithful under the key; otherwise the
    categories built by key (``rebuilt=True``: the column becomes a stand-in
    column).
    """
    try:
        cats, mode = _today(series)
        if _faithful_today(series, cats):
            return cats, mode, False
    except (TypeError, ValueError, OverflowError):
        # Today's own computation fails on this mix (#378, #380): build by key.
        pass
    cats, mode = _rebuilt(series)
    return cats, mode, True


def keyed_codes(
    series: pd.Series,
    arrived: pd.Series,
    changed: pd.Series,
    known: list[Any],
    mode: Any,
    policy: UnseenPolicy,
    column: Any,
) -> tuple[npt.NDArray[np.int64], list[str]]:
    """Codes of *series* (``comparable_values`` of *arrived*) looked up by key.

    *changed* marks the rows ``comparable_values`` found no category for
    (H-0127). The messages and the error are today's.
    """
    position = {category_identity_key(c): i for i, c in enumerate(known)}
    values = series.to_numpy(dtype=object)
    original = arrived.to_numpy(dtype=object)  # the messages show these
    flags = changed.to_numpy(dtype=bool)
    codes = np.full(len(values), -1, dtype=np.int64)
    hit = np.zeros(len(values), dtype=bool)
    unseen: dict[str, None] = {}
    for i, value in enumerate(values):
        if flags[i]:
            hit[i] = True
            continue
        if is_missing(value):
            continue
        pos = _lookup(position, value)
        if pos is None:
            hit[i] = True
            unseen[_text(value)] = None
        else:
            codes[i] = pos
    if isinstance(arrived.dtype, pd.CategoricalDtype):
        for declared in arrived.cat.categories:
            if _lookup(position, declared) is None:
                unseen[_text(declared)] = None
    warnings: list[str] = []
    if unseen or flags.any():
        shown = sorted(set(unseen) | {_text(v) for v in original[flags]})
        if policy == "error":
            raise LizyMLError(
                ErrorCode.DATA_SCHEMA_INVALID,
                user_message=f"Column '{column}' contains unseen categories: {shown}",
                context={"column": column, "unseen_categories": shown},
            )
        n_rows = int(hit.sum())
        present = sorted({_text(v) for v in original[hit]})
        if policy == "mode":
            codes[hit] = -1 if mode is None else position[category_identity_key(mode)]
            replaced_by = f"the training mode {mode!r}"
        else:  # "nan"
            codes[hit] = -1
            replaced_by = "a missing value"
        if n_rows:
            warnings.append(
                f"Column '{column}': {n_rows} row(s) with unseen categories "
                f"{present} replaced by {replaced_by} (unseen_policy='{policy}')"
            )
    return codes, warnings


def _text(value: Any) -> str:
    """``str(value)``; a value NumPy cannot show (a unitless ``datetime64``)
    gets a placeholder, so an unseen one never breaks the message."""
    try:
        return str(value)
    except (TypeError, ValueError):
        return f"<unprintable {type(value).__name__}>"


def _lookup(position: dict[Any, int], value: Any) -> int | None:
    try:
        return position.get(category_identity_key(value))
    except TypeError:
        # A value outside V, or a keyless temporal value (H-0139), arriving
        # in an identity column matches no category: it is unseen.
        return None
