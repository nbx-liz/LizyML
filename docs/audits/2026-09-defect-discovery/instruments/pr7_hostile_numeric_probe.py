"""Probe for #267: can a column reach the leakage validator's swallowing handler?

The plan's step-1 matrix found 15 of 378 cells entering the handler, all of
them a numeric ``ExtensionDtype`` (``_is_numeric = True``) whose array raises in
``__array__`` or ``isna``. That instrument was not kept, so this re-builds the
two reachable shapes and runs them through the public
``lizyml.data.validate_no_target_leakage`` against several target dtypes.

For each cell it prints whether the validator raised, returned a warning, or
returned ``[]`` -- and, by calling ``_series_perfectly_correlated`` directly,
whether the guarded call raised (i.e. whether the handler swallowed it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from lizyml.data import validate_no_target_leakage
from lizyml.data.validators import _series_perfectly_correlated


class _HostileDtype(ExtensionDtype):
    name = "hostile_numeric"
    type = float
    _is_numeric = True
    na_value = np.nan

    @classmethod
    def construct_array_type(cls) -> type[ExtensionArray]:
        return _HostileArray


class _HostileArray(ExtensionArray):
    """A numeric-declared array; ``mode`` picks which operation raises."""

    mode = "array"

    def __init__(self, values: Any) -> None:
        self._data = np.asarray(values, dtype=float)

    @classmethod
    def _from_sequence(cls, scalars: Any, *, dtype: Any = None, copy: bool = False) -> Any:
        return cls(scalars)

    @property
    def dtype(self) -> ExtensionDtype:
        return _HostileDtype()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Any) -> Any:
        out = self._data[item]
        return type(self)(out) if np.ndim(out) else out

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def isna(self) -> Any:
        if self.mode == "isna":
            raise ValueError("_HostileArray.isna")
        return np.isnan(self._data)

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        if self.mode == "array":
            raise TypeError("_HostileArray.__array__")
        return self._data

    def take(self, indices: Any, *, allow_fill: bool = False, fill_value: Any = None) -> Any:
        return type(self)(self._data.take(indices))

    def copy(self) -> Any:
        return type(self)(self._data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat: Any) -> Any:
        return cls(np.concatenate([a._data for a in to_concat]))


class _HostileIsna(_HostileArray):
    mode = "isna"


def main() -> None:
    values = np.arange(10, dtype=float)
    targets = {
        "int64": pd.Series(values.astype("int64")),
        "float64": pd.Series(values),
        "bool": pd.Series(values % 2 == 0),
        "Int64": pd.Series(values.astype("int64")).astype("Int64"),
        "complex128": pd.Series(values.astype("complex128")),
    }
    for shape, cls in (("array", _HostileArray), ("isna", _HostileIsna)):
        for tname, y in targets.items():
            col = pd.Series(cls(values))
            try:
                _series_perfectly_correlated(col, y)
                guarded = "no exception"
            except (TypeError, ValueError) as exc:
                guarded = f"{type(exc).__name__}: {exc}"
            df = pd.DataFrame({"x": col, "y": y})
            try:
                out = validate_no_target_leakage(df, "y", raise_on_violation=True)
                verdict = f"returned {out!r}"
            except Exception as exc:  # noqa: BLE001 -- the outcome is the measurement
                verdict = f"raised {type(exc).__name__}: {exc}"
            print(f"{shape:6s} target={tname:10s} guarded call: {guarded:40s} validator {verdict}")


if __name__ == "__main__":
    main()
