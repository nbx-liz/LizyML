"""TargetEncoder — encode non-numeric classification targets to int codes.

Foundation-layer contract: all categories may import this type without creating
a back-dependency on Layer 1 ``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError

TaskType = Literal["regression", "binary", "multiclass"]


@dataclass(frozen=True)
class TargetEncoder:
    """Encode non-numeric classification targets to int codes.

    Numeric targets (and regression) pass through unchanged via
    ``needs_encoding=False``. For classification with a non-numeric target
    (object / str / ``pd.StringDtype`` / category-with-string-categories /
    bool), :meth:`fit` records the sorted unique class labels so consumers
    can map predicted codes back to the original labels via
    :meth:`inverse_transform`.

    Class ordering:
        Class labels are sorted **lexicographically** by their string form
        (``sorted(unique, key=str)``). This is deterministic and works for
        heterogeneous label types (e.g. mixed ``str``/``int``) but does
        **not** match natural numeric order for numeric-string labels::

            input y:   ["1", "10", "2"]
            classes_:  ("1", "10", "2")     # NOT ("1", "2", "10")
            codes:      0     1     2

        Downstream metric tables (``f1_score`` per class, etc.) inherit
        this order. If natural numeric ordering is required, cast the
        target to a numeric dtype before fitting (then the encoder becomes
        a no-op and the model uses the numeric values directly). This
        ordering rule is part of the public contract — see
        BLUEPRINT.md §7.1.

    Attributes:
        classes_: Lexicographically sorted original labels (see *Class
            ordering* above). Empty tuple when ``needs_encoding`` is False.
            ``classes_[i]`` corresponds to int code ``i``.
        needs_encoding: Whether ``transform`` / ``inverse_transform`` perform
            actual mapping. False for regression and numeric classification.
        original_dtype: String form of the original y dtype (e.g. ``"object"``,
            ``"string"``, ``"category"``, ``"int64"``). Used to restore the
            original output dtype on :meth:`inverse_transform`.
    """

    classes_: tuple[Any, ...] = field(default=())
    needs_encoding: bool = field(default=False)
    original_dtype: str = field(default="")

    @classmethod
    def no_op(cls) -> TargetEncoder:
        """Return a transparent encoder for numeric targets / migration."""
        return cls(classes_=(), needs_encoding=False, original_dtype="")

    @classmethod
    def fit(cls, y: pd.Series, task: TaskType) -> TargetEncoder:
        """Construct an encoder for the given target series.

        Args:
            y: Raw target column.
            task: ML task. ``"regression"`` always yields a no-op encoder.

        Returns:
            A frozen :class:`TargetEncoder`.

        Raises:
            LizyMLError: With ``TARGET_NOT_NUMERIC`` when ``task='regression'``
                and y is not a numeric dtype (caught before any model training).
        """
        original_dtype = str(y.dtype)

        if task == "regression":
            if not _is_numeric_target(y):
                raise LizyMLError(
                    code=ErrorCode.TARGET_NOT_NUMERIC,
                    user_message=(
                        f"task='regression' requires a numeric target column, "
                        f"but received dtype={original_dtype!r}."
                    ),
                    context={"task": task, "dtype": original_dtype},
                )
            return cls(classes_=(), needs_encoding=False, original_dtype=original_dtype)

        if _is_numeric_target(y):
            return cls(classes_=(), needs_encoding=False, original_dtype=original_dtype)

        # Non-numeric classification target — capture sorted unique labels.
        unique = pd.Series(y).dropna().unique().tolist()
        classes = tuple(sorted(unique, key=str))
        return cls(
            classes_=classes,
            needs_encoding=True,
            original_dtype=original_dtype,
        )

    def transform(self, y: pd.Series) -> pd.Series:
        """Encode y to int codes.

        No-op when ``needs_encoding`` is False (returns y unchanged).

        Raises:
            LizyMLError: With ``TARGET_UNSEEN_LABEL`` when y contains labels
                not present in :attr:`classes_`. With ``DATA_SCHEMA_INVALID``
                when y contains NaN — classification targets must be fully
                labeled before fit.
        """
        if not self.needs_encoding:
            return y

        y_series = pd.Series(y)
        nan_mask = y_series.isna()
        if nan_mask.any():
            raise LizyMLError(
                code=ErrorCode.DATA_SCHEMA_INVALID,
                user_message=(
                    f"Target column contains {int(nan_mask.sum())} NaN value(s); "
                    "classification targets must be fully labeled."
                ),
                context={"nan_count": int(nan_mask.sum())},
            )

        mapping = {c: i for i, c in enumerate(self.classes_)}
        unseen = set(y_series.unique()) - mapping.keys()
        if unseen:
            raise LizyMLError(
                code=ErrorCode.TARGET_UNSEEN_LABEL,
                user_message=(
                    f"Target column contains labels not seen during fit: "
                    f"{sorted(unseen, key=str)}"
                ),
                context={
                    "unseen": sorted(str(u) for u in unseen),
                    "known": [str(c) for c in self.classes_],
                },
            )
        encoded = y_series.map(mapping).astype(np.int64)
        encoded.index = y.index
        encoded.name = y.name
        return encoded

    def inverse_transform(
        self,
        codes: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Map int codes back to the original labels.

        No-op when ``needs_encoding`` is False (returns ``codes`` unchanged).
        Returns an ``object``-typed numpy array for non-numeric targets
        regardless of the original dtype (object / string / category).
        Pandas-extension dtype (Categorical/StringDtype) preservation is
        a non-goal for v1 — labels are recovered, not the container type.

        Raises:
            LizyMLError: With ``TARGET_UNSEEN_LABEL`` when a code is outside
                ``range(len(classes_))`` (defensive: unreachable in normal
                flow because predict outputs argmax / threshold codes only).
        """
        if not self.needs_encoding:
            return codes

        codes_arr = np.asarray(codes)
        n_classes = len(self.classes_)
        try:
            decoded = np.array(
                [self.classes_[int(c)] for c in codes_arr.ravel()],
                dtype=object,
            ).reshape(codes_arr.shape)
        except IndexError as exc:
            raise LizyMLError(
                code=ErrorCode.TARGET_UNSEEN_LABEL,
                user_message=(
                    f"Predicted code is outside [0, {n_classes}); "
                    f"classes_ has {n_classes} entries."
                ),
                context={"n_classes": n_classes},
                cause=exc,
            ) from exc

        if self.original_dtype.lower() == "string":
            return pd.array(decoded, dtype="string").to_numpy()
        return decoded


def _is_numeric_target(y: pd.Series) -> bool:
    """Numeric targets exclude bool (treated as 2-class categorical)."""
    return bool(pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y))
