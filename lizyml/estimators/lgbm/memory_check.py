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


#: A bin cap the estimate could not read: no cap, so the estimate can only be
#: too large, never too small (H-0140 implementation review).
_UNCAPPED = None
#: Bins LightGBM may build beyond a feature's values: its default (zero) bin
#: and its missing-value bin; at most 2 were measured (H-0140 implementation
#: review, run 2), 1 for a categorical feature.
_EXTRA_BINS = 2

#: LightGBM 4.6's documented Dataset parameters, by what they do to the bins
#: this estimate counts (H-0140 implementation review, run 2, round 2).
#: ``bounded``: the bins stay within ``min(values, cap) + 2`` (caps, fewer
#: bins, the missing / zero bins). ``widening``: the bins may exceed the cap
#: (a numeric column made categorical, forced bin bounds), so a feature then
#: counts ``max(values, cap) + 2``. ``io``: reading files only.
DATASET_PARAMS: dict[str, str] = {
    "max_bin": "bounded",
    "max_bin_by_feature": "bounded",
    "min_data_in_bin": "bounded",
    "bin_construct_sample_cnt": "bounded",
    "data_random_seed": "bounded",
    "is_enable_sparse": "bounded",
    "enable_bundle": "bounded",
    "use_missing": "bounded",
    "zero_as_missing": "bounded",
    "feature_pre_filter": "bounded",
    "linear_tree": "bounded",  # bins unchanged; its tree memory is counted below
    "categorical_feature": "widening",
    "forcedbins_filename": "widening",
    "pre_partition": "io",
    "two_round": "io",
    "header": "io",
    "label_column": "io",
    "weight_column": "io",
    "group_column": "io",
    "ignore_column": "io",
    "save_binary": "io",
    "precise_float_parser": "io",
    "parser_config_file": "io",
}
#: Extra bytes ``linear_tree`` keeps per configured leaf per round, per
#: feature plus one (its coefficients; about 124 bytes measured with 10
#: features, 176 counted).
_LINEAR_TREE_BYTES = 16


def _as_int(item: Any) -> int | None:
    """One bin count as LightGBM reads it (an integer, or a finite number
    truncated as its C++ parser does), else ``None``."""
    if isinstance(item, bool):
        return None
    if isinstance(item, (int, np.integer)):
        return int(item)
    try:
        number = float(str(item).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if np.isfinite(number) else None


def _per_feature_caps(value: Any) -> list[int | None] | None:
    """``max_bin_by_feature`` per feature, ``None`` when absent.

    Any form LightGBM accepts is read alike: one number, a list, a tuple, an
    array or a comma-separated string, of integers or finite numbers. An item
    that cannot be read is ``_UNCAPPED``, and a value that is no sequence
    leaves every feature uncapped.
    """
    if value is None:
        return None
    if not isinstance(value, str) and np.ndim(value) == 0:
        return [_as_int(value)]  # a scalar is a one-element vector to LightGBM
    items = value.split(",") if isinstance(value, str) else value
    try:
        return [_as_int(item) for item in items]
    except TypeError:
        return []  # not a sequence: no feature gets a readable cap


def feature_bins(X: pd.DataFrame, params: dict[str, Any]) -> int:
    """An upper-bound approximation of the bins LightGBM builds for *X*.

    A ``category`` column counts its categories, a numeric column
    ``min(distinct values, max_bin)`` (``max_bin_by_feature`` when set), and
    any other column (object, string) one bin per row, without comparing its
    values; each feature adds LightGBM's default and missing-value bins. A
    cap that cannot be read is not applied, so a value the estimate misreads
    makes it larger, not smaller.
    """
    max_bin = _canonical(params, "max_bin")
    default_bin = LIGHTGBM_DEFAULT_MAX_BIN if max_bin is None else _as_int(max_bin)
    by_feature = _per_feature_caps(_canonical(params, "max_bin_by_feature"))
    widening = any(
        _canonical(params, name) not in (None, "", [], ())
        for name, kind in DATASET_PARAMS.items()
        if kind == "widening"
    )
    total = 0
    for i, col in enumerate(X.columns):
        series = X[col]
        cap: int | None = default_bin
        if by_feature is not None:
            cap = by_feature[i] if i < len(by_feature) else _UNCAPPED
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = len(series.cat.categories)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            distinct = int(series.nunique(dropna=True))
            if cap is None:
                values = distinct
            elif widening:
                values = max(distinct, cap)
            else:
                values = min(distinct, cap)
        else:
            values = len(series)
        total += values + _EXTRA_BINS
    return total


def estimate_bytes(
    num_leaves: int,
    bins: int,
    rounds: int,
    num_class: int,
    n_boosters: int,
    linear_features: int = 0,
) -> int:
    """The approximate extra CPU memory, in bytes (see the module docstring).

    *linear_features* is the number of features when ``linear_tree`` is on
    (each leaf then keeps a linear model), else 0.
    """
    per_leaf = _TREE_BYTES
    if linear_features:
        per_leaf += _LINEAR_TREE_BYTES * (linear_features + 1)
    histogram = _HISTOGRAM_BYTES * bins * num_leaves
    trees = per_leaf * num_leaves * rounds * num_class * n_boosters
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
    n_features: int = 0,
) -> MemoryAssessment:
    """Assess a fit from the finalized *params* and *num_boost_round*."""
    leaves = _canonical(params, "num_leaves")
    linear = _as_bool(_canonical(params, "linear_tree"))
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
            linear_features=n_features if linear else 0,
        ),
        n_boosters=n_boosters,
    )


def _as_bool(value: Any) -> bool:
    """A LightGBM boolean as LightGBM reads it (``True``, ``1``, ``"true"``)."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "+", "yes")
    return bool(value) if value is not None else False


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
