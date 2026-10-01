"""``detect_boundary(threshold=...)`` changes the decision (H-0108, #268).

Existing calls pass ``threshold=0.05``, which is the default, so none showed the
argument doing anything. A best value 11% of the way into the range is not at
the edge under the default and is at the lower edge under ``threshold=0.2``.
"""

from __future__ import annotations

from lizyml.tuning.search_space import FloatDim, detect_boundary

_DIMS = [FloatDim("lr", low=0.001, high=0.1, log=False)]
_BEST = {"lr": 0.012}  # (0.012 - 0.001) / (0.1 - 0.001) = 0.111


def test_default_threshold_keeps_the_value_inside() -> None:
    (status,) = detect_boundary(_DIMS, _BEST).dims
    assert status.edge == "none"
    assert status.expanded is False


def test_a_wider_threshold_moves_it_to_the_edge() -> None:
    (status,) = detect_boundary(_DIMS, _BEST, threshold=0.2).dims
    assert status.edge == "lower"
    assert status.expanded is True
