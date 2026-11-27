"""#433: automatic leaf derivation must reach its cap for every admitted depth.

``_compute_num_leaves`` multiplied the integer ``2**max_depth`` by the float
ratio, which converts it to ``float`` and raises ``OverflowError`` from depth
1024 on, before the ``min(131072, ...)`` cap applies. ``max_depth`` has no upper
bound on admission (H-0140), so fit and tune both reached it.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from lizyml.estimators.lgbm import smart_params
from lizyml.estimators.lgbm.smart_params import resolve_smart_params

_CAP = 131072
_RATIOS = (0.1, 0.25, 0.3, 0.5, 0.75, 1.0)


def _reference(max_depth: int, ratio: float) -> int:
    """The derivation written out exactly, with no float conversion of 2**d."""
    value = Fraction(ratio) * (1 << max_depth)
    return max(8, min(_CAP, math.ceil(value)))


@pytest.mark.parametrize("depth", [1024, 1025, 2000, 10**6])
@pytest.mark.parametrize("ratio", _RATIOS)
def test_a_depth_past_the_float_range_derives_the_cap(depth: int, ratio: float) -> None:
    assert smart_params._compute_num_leaves(depth, ratio) == _CAP  # noqa: SLF001


@pytest.mark.parametrize("ratio", _RATIOS)
def test_every_depth_up_to_1023_is_unchanged(ratio: float) -> None:
    """Depths the old rule could compute derive exactly what they did before."""
    for depth in range(1, 1024):
        old = max(8, min(_CAP, math.ceil((2**depth) * ratio)))
        got = smart_params._compute_num_leaves(depth, ratio)  # noqa: SLF001
        assert got == old == _reference(depth, ratio), (depth, ratio)


@pytest.mark.parametrize("depth", [None, 0, -1])
def test_no_depth_limit_still_derives_from_the_cap(depth: int | None) -> None:
    assert smart_params._compute_num_leaves(depth, 1.0) == _CAP  # noqa: SLF001
    assert smart_params._compute_num_leaves(depth, 0.1) == math.ceil(_CAP * 0.1)  # noqa: SLF001


def test_the_fit_path_resolves_a_depth_of_1024() -> None:
    """``resolve_smart_params`` is the route fit and tune share (H-0050)."""
    y = pd.Series(np.arange(20, dtype=float))
    resolved, _ = resolve_smart_params(
        {"auto_num_leaves": True, "num_leaves_ratio": 1.0},
        {"max_depth": 1024},
        n_rows=20,
        feature_names=["x"],
        y=y,
        task="regression",
    )
    assert resolved["num_leaves"] == _CAP
