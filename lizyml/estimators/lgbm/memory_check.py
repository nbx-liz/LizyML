"""memory_check -- warn before LightGBM allocates for a large num_leaves (H-0140).

LightGBM sizes its histogram cache and every tree's arrays by the configured
``num_leaves``, not by the leaves a tree can grow: on 150 rows, 131072
configured leaves need about 12 GB while each tree grows 4 to 6 leaves
(#388). This module only informs. It reads the finalized parameters the
adapter passes to ``lgb.train`` and estimates the extra CPU memory:

- the histogram cache, held while one booster trains:
  about 16 bytes x the bins of all features x ``num_leaves``;
- the trees, held by every booster kept alive at once:
  about 80 bytes x ``num_leaves`` x rounds x classes, per booster.

The estimate is approximate (measured within about 4% on small cases,
H-0140), an upper bound under early stopping, and leaves out isotonic
calibration boosters. It never changes a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.param_names import LGBM_CANONICAL_NAME

#: The warning fires at this many configured leaves, whatever the estimate.
LEAVES_THRESHOLD = 16384
#: ... or when the estimate reaches this many bytes.
BYTES_THRESHOLD = 1 << 30

#: LightGBM's own ``num_leaves`` when nothing sets it.
LIGHTGBM_DEFAULT_NUM_LEAVES = 31
#: LightGBM's own ``max_bin`` when nothing sets it.
LIGHTGBM_DEFAULT_MAX_BIN = 255

_HISTOGRAM_BYTES = 16  # per bin per configured leaf
_TREE_BYTES = 80  # per configured leaf per round per class per booster

Cause = Literal["derived", "explicit", "default"]


@dataclass(frozen=True)
class MemoryAssessment:
    """What one fit will ask LightGBM to allocate, and why.

    Attributes:
        num_leaves: The ``num_leaves`` LightGBM receives.
        cause: ``"derived"`` (``auto_num_leaves`` from ``max_depth``),
            ``"explicit"`` (set in the parameters) or ``"default"``
            (LightGBM's own 31).
        max_depth: The ``max_depth`` LightGBM receives, or ``None``.
        estimate_bytes: The approximate extra CPU memory.
        n_boosters: The boosters kept alive at once that the estimate counts.
    """

    num_leaves: int
    cause: Cause
    max_depth: int | None
    estimate_bytes: int
    n_boosters: int

    @property
    def fires(self) -> bool:
        return (
            self.num_leaves >= LEAVES_THRESHOLD
            or self.estimate_bytes >= BYTES_THRESHOLD
        )


def _canonical(params: dict[str, Any], name: str) -> Any:
    """The value of *name* under any spelling in *params*, or ``None``."""
    for key, value in params.items():
        if LGBM_CANONICAL_NAME.get(key, key) == name:
            return value
    return None


def _per_feature_caps(value: Any) -> list[int] | None:
    """``max_bin_by_feature`` as ints, in any form LightGBM accepts: a list, a
    tuple, an array, a comma-separated string or one integer. ``None`` when it
    is absent or unreadable (LightGBM then refuses it before allocating)."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return [int(value)]  # a scalar is a one-element vector to LightGBM
    items = value.split(",") if isinstance(value, str) else value
    try:
        return [int(str(item).strip()) for item in items]
    except (TypeError, ValueError):
        return None


def feature_bins(X: pd.DataFrame, params: dict[str, Any]) -> int:
    """An upper-bound approximation of the bins LightGBM builds for *X*.

    A ``category`` column counts its categories, a numeric column
    ``min(distinct values, max_bin)`` (``max_bin_by_feature`` when set), and
    any other column (object, string) one bin per row, without comparing its
    values.
    """
    max_bin = _canonical(params, "max_bin")
    default_bin = LIGHTGBM_DEFAULT_MAX_BIN if max_bin is None else int(max_bin)
    by_feature = _per_feature_caps(_canonical(params, "max_bin_by_feature"))
    total = 0
    for i, col in enumerate(X.columns):
        series = X[col]
        cap = default_bin
        if by_feature is not None and i < len(by_feature):
            cap = by_feature[i]
        if isinstance(series.dtype, pd.CategoricalDtype):
            total += len(series.cat.categories)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            total += min(int(series.nunique(dropna=True)), cap)
        else:
            total += len(series)
    return total


def estimate_bytes(
    num_leaves: int, bins: int, rounds: int, num_class: int, n_boosters: int
) -> int:
    """The approximate extra CPU memory, in bytes (see the module docstring)."""
    histogram = _HISTOGRAM_BYTES * bins * num_leaves
    trees = _TREE_BYTES * num_leaves * rounds * num_class * n_boosters
    return histogram + trees


def finalized(adapter: LGBMAdapter) -> tuple[dict[str, Any], int]:
    """The params and ``num_boost_round`` *adapter* passes to ``lgb.train``.

    ``LGBMAdapter._build_params`` without training targets, the path
    ``build_export_params`` uses, so an alias or a ``num_class`` override is
    seen as LightGBM sees it.
    """
    params, num_boost_round, _, _ = adapter._build_params(y_train=None)  # noqa: SLF001
    return params, int(num_boost_round)


def assess(
    params: dict[str, Any],
    num_boost_round: int,
    *,
    bins: int,
    auto_num_leaves: bool,
    n_boosters: int,
) -> MemoryAssessment:
    """Assess a fit from the finalized *params* and *num_boost_round*."""
    leaves = _canonical(params, "num_leaves")
    cause: Cause
    if leaves is None:
        leaves, cause = LIGHTGBM_DEFAULT_NUM_LEAVES, "default"
    else:
        cause = "derived" if auto_num_leaves else "explicit"
    num_class = _canonical(params, "num_class")
    depth = _canonical(params, "max_depth")
    return MemoryAssessment(
        num_leaves=int(leaves),
        cause=cause,
        max_depth=None if depth is None else int(depth),
        estimate_bytes=estimate_bytes(
            int(leaves),
            bins,
            num_boost_round,
            max(1, int(num_class)) if num_class is not None else 1,
            n_boosters,
        ),
        n_boosters=n_boosters,
    )


def message(assessment: MemoryAssessment) -> str:
    """The warning text: num_leaves and its cause first, the estimate after."""
    a = assessment
    if a.cause == "derived":
        why = f"derived by auto_num_leaves from max_depth={a.max_depth}"
        remedy = (
            "use a shallower max_depth, or set auto_num_leaves=False with a "
            "smaller num_leaves"
        )
    elif a.cause == "explicit":
        why = "set in the parameters"
        remedy = "use a smaller num_leaves, or fewer features, max_bin or rounds"
    else:
        why = "LightGBM's default"
        remedy = "use fewer features, max_bin or rounds"
    gib = a.estimate_bytes / (1 << 30)
    return (
        f"LightGBM will train with num_leaves={a.num_leaves} ({why}). LightGBM "
        "allocates memory for every configured leaf, not only the leaves a tree "
        f"grows: about {gib:.1f} GiB of extra memory is expected (approximate: "
        f"the histogram cache and the trees of {a.n_boosters} booster(s) kept at "
        "once; an upper bound under early stopping; isotonic calibration is not "
        f"counted). To lower it, {remedy}."
    )
