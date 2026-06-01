"""Constructor ratio-guard tests for every inner-valid strategy (H-0178 item 5).

The ``ratio in (0, 1)`` constructor guards are the *only* enforcement point —
the config-side ratios (``schema.py``) are unbounded floats — yet only
``HoldoutInnerValid`` was previously covered. A regression weakening any other
strategy's guard would go unnoticed. This parametrizes the boundary and
out-of-range ratios across all five guarded strategies.
"""

from __future__ import annotations

import pytest

from lizyml.training.inner_valid import (
    BlockedGroupInnerValid,
    GroupHoldoutInnerValid,
    HoldoutInnerValid,
    StratifiedTimeHoldoutInnerValid,
    TimeHoldoutInnerValid,
)

# All strategies accept ``ratio`` as the first positional argument.
_GUARDED_STRATEGIES = [
    HoldoutInnerValid,
    GroupHoldoutInnerValid,
    TimeHoldoutInnerValid,
    StratifiedTimeHoldoutInnerValid,
    BlockedGroupInnerValid,
]


@pytest.mark.parametrize("strategy_cls", _GUARDED_STRATEGIES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5])
def test_ratio_out_of_open_interval_raises(strategy_cls, bad_ratio: float) -> None:
    with pytest.raises(ValueError, match=r"ratio must be in \(0, 1\)"):
        strategy_cls(bad_ratio)


@pytest.mark.parametrize("strategy_cls", _GUARDED_STRATEGIES, ids=lambda c: c.__name__)
def test_in_range_ratio_is_accepted(strategy_cls) -> None:
    # A valid ratio must not raise (guards reject only out-of-range values).
    strategy_cls(0.2)
