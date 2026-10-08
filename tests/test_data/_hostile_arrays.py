"""Numeric-declared extension arrays that raise during comparison (H-0107, #267).

A dtype with ``_is_numeric = True`` passes ``is_numeric_dtype``, so the leakage
check's comparison goes on to ``isna()`` and ``np.allclose``; these arrays
raise there. Each class raises one stored exception instance, so a test can
assert the reported ``cause`` is that very object.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pandas.api.extensions import ExtensionArray, ExtensionDtype


class HostileDtype(ExtensionDtype):
    name = "hostile_numeric"
    type = float
    _is_numeric = True
    na_value = np.nan

    @classmethod
    def construct_array_type(cls) -> type[ExtensionArray]:
        return HostileArray


class HostileArray(ExtensionArray):
    """Raises ``error`` from ``__array__`` (``np.allclose`` reaches it)."""

    error: BaseException = TypeError("HostileArray.__array__")
    raise_in = "array"

    def __init__(self, values: Any) -> None:
        self._data = np.asarray(values, dtype=float)

    @classmethod
    def _from_sequence(
        cls, scalars: Any, *, dtype: Any = None, copy: bool = False
    ) -> HostileArray:
        return cls(scalars)

    @property
    def dtype(self) -> ExtensionDtype:
        return HostileDtype()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Any) -> Any:
        out = self._data[item]
        return type(self)(out) if np.ndim(out) else out

    @property
    def nbytes(self) -> int:
        return int(self._data.nbytes)

    def isna(self) -> Any:
        if self.raise_in == "isna":
            raise self.error
        return np.isnan(self._data)

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        if self.raise_in == "array":
            raise self.error
        return self._data

    def take(
        self, indices: Any, *, allow_fill: bool = False, fill_value: Any = None
    ) -> HostileArray:
        return type(self)(self._data.take(indices))

    def copy(self) -> HostileArray:
        return type(self)(self._data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat: Any) -> HostileArray:
        return cls(np.concatenate([a._data for a in to_concat]))


class HostileIsna(HostileArray):
    """Raises ``error`` from ``isna``."""

    error = ValueError("HostileIsna.isna")
    raise_in = "isna"


class HostileOverflow(HostileArray):
    """Raises an exception the old handler did not catch (not TypeError/ValueError)."""

    error = OverflowError("HostileOverflow.__array__")
