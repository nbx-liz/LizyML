"""category_json -- the category values LightGBM can write into a saved model (H-0127).

LightGBM writes every categorical column's categories into the model string as
JSON (``_dump_pandas_categorical``): ``json.dumps`` of the list of category
lists, with a ``default`` that turns numpy integer, floating and bool scalars
into ``.item()`` and numpy arrays into ``.tolist()`` and passes every other
value on unchanged. A value ``json`` cannot encode -- ``bytes``, ``Decimal``, a
timestamp, a date, ``complex``, a ``frozenset``, or a ``tuple`` holding one --
makes the booster impossible to pickle, so ``Model.export`` cannot save the
model although fit and predict work (#338).

LightGBM receives ``list(data[col].cat.categories)`` after the encoder's
``set_categories(known)``, i.e. ``list(pd.CategoricalDtype(known).categories)``;
the check reads exactly that, inside the same two containers.

Limit: a ``tuple`` nested within a few levels of the interpreter's recursion
limit can pass this check and still fail when the booster is pickled, on
Python's own pickle recursion (not LightGBM's JSON). Such an export fails while
it writes, as before.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def type_name(value: Any) -> str:
    """The name of *value*'s type, never raising."""
    try:
        name = type.__getattribute__(type(value), "__name__")
    except Exception:  # noqa: BLE001 - a hostile metaclass must not stop fit
        return "<unknown type>"
    return name if isinstance(name, str) else "<unknown type>"


def shown(value: Any) -> str:
    """``repr(value)``, or ``<unprintable TYPE>`` when ``repr`` raises."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a hostile __repr__ must not stop fit
        return f"<unprintable {type_name(value)}>"
    return text if isinstance(text, str) else f"<unprintable {type_name(value)}>"


@dataclass(frozen=True)
class UnwritableCategory:
    """The first category value of *column* LightGBM cannot write."""

    column: str
    value: Any

    @property
    def shown(self) -> str:
        return shown(self.value)

    @property
    def type_name(self) -> str:
        return type_name(self.value)


def _numpy_default(obj: Any) -> Any:
    """LightGBM's ``_json_default_with_numpy``, refusing what it passes on."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        plain = obj.item()
        # np.longdouble.item() is a np.longdouble: LightGBM recurses on it.
        if isinstance(plain, np.generic):
            raise TypeError(type_name(obj))
        return plain
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type_name(obj))


def is_writable(value: Any) -> bool:
    """Whether LightGBM can write *value* as a category into a saved model."""
    try:
        json.dumps([[value]], default=_numpy_default)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def unwritable_categories(
    categories: Mapping[str, list[Any]],
) -> list[UnwritableCategory]:
    """Each column whose categories hold a value LightGBM cannot write, with
    the first such value, in column order."""
    found: list[UnwritableCategory] = []
    for column, known in categories.items():
        for value in list(pd.CategoricalDtype(known).categories):
            if not is_writable(value):
                found.append(UnwritableCategory(column=str(column), value=value))
                break
    return found


def encoder_categories(pipeline_state: Any) -> Mapping[str, list[Any]]:
    """The encoder categories a native pipeline state records (none if absent)."""
    if not isinstance(pipeline_state, Mapping):
        return {}
    encoder = pipeline_state.get("encoder")
    if not isinstance(encoder, Mapping):
        return {}
    categories = encoder.get("categories")
    return categories if isinstance(categories, Mapping) else {}


def warn_unsaveable_categories(pipeline_state: Any) -> None:
    """Warn once per column whose categories ``Model.export`` will refuse."""
    for item in unwritable_categories(encoder_categories(pipeline_state)):
        warnings.warn(
            f"Column '{item.column}' holds the category value {item.shown} of "
            f"type {item.type_name}, which LightGBM cannot write into a saved "
            "model: fit and predict work, but Model.export will refuse this "
            "model. Convert the column's values to str (or int, float, bool) "
            "before fit to be able to save it.",
            UserWarning,
            stacklevel=3,
        )
