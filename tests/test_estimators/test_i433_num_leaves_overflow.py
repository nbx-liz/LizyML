"""#433: automatic leaf derivation must not overflow for any admitted depth.

The 131072-leaf cap is an upper bound on the derived value: a tiny ratio can
derive fewer leaves even at a large depth.

``_compute_num_leaves`` multiplied the integer ``2**max_depth`` by the float
ratio, which converts it to ``float`` and raises ``OverflowError`` from depth
1024 on, before the ``min(131072, ...)`` cap applies. ``max_depth`` has no upper
bound on admission (H-0140), so fit and tune both reached it.
"""

from __future__ import annotations

import math
import random
import tracemalloc
import warnings
from fractions import Fraction
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.estimators.lgbm import smart_params
from lizyml.estimators.lgbm.param_names import LGBM_CANONICAL_NAME
from lizyml.estimators.lgbm.smart_params import resolve_smart_params
from tests._helpers import make_config, make_regression_df

_CAP = 131072
_RATIOS = (0.1, 0.25, 0.3, 0.5, 0.75, 1.0)


def _reference(max_depth: int, ratio: float) -> int:
    """The derivation written out exactly, with no float conversion of 2**d."""
    value = Fraction(ratio) * (1 << max_depth)
    return max(8, min(_CAP, math.ceil(value)))


def _legacy(max_depth: float, ratio: float) -> int:
    """The pre-#433 expression, for inputs it could evaluate."""
    return max(8, min(_CAP, math.ceil(2**max_depth * ratio)))


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


@pytest.mark.parametrize(("depth", "expected"), [(3.0, 8), (3.5, 12), (10.25, 1218)])
def test_a_float_depth_keeps_the_old_derivation(depth: float, expected: int) -> None:
    """A float depth (a model dimension can sample one) derives what it did."""
    assert _legacy(depth, 1.0) == expected
    assert smart_params._compute_num_leaves(depth, 1.0) == expected  # noqa: SLF001


def test_every_input_the_old_product_handled_is_unchanged() -> None:
    """Fractional depths in (1023, 1024) and tiny ratios take the old expression.

    The first case is the one review round 2 found: the old product is exactly
    10.0 before ``ceil``, and a rewritten rounding path gave 11.
    """
    assert _legacy(1023.8246047210798, 6.28179385155373e-308) == 10
    rng = random.Random(433)
    cases = [(1023.8246047210798, 6.28179385155373e-308)]
    cases += [(1023 + rng.random(), rng.uniform(1e-310, 1.0)) for _ in range(2000)]
    cases += [
        (rng.uniform(0.01, 1023.99), rng.uniform(1e-300, 1.0)) for _ in range(2000)
    ]
    for depth, ratio in cases:
        got = smart_params._compute_num_leaves(depth, ratio)  # noqa: SLF001
        assert got == _legacy(depth, ratio), (depth, ratio)


@pytest.mark.parametrize("depth", [2**31 - 1, 10**18])
def test_a_huge_int_depth_does_not_build_its_power_of_two(depth: int) -> None:
    """Review round 3: ``2**2147483647`` alone needs about 273 MiB."""
    tracemalloc.start()
    try:
        got = smart_params._compute_num_leaves(depth, 1.0)  # noqa: SLF001
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert got == _CAP
    assert peak < 1 << 20  # well under 1 MiB


@pytest.mark.parametrize("depth", [1024.5, 2000.0])
def test_a_float_depth_past_the_float_range_derives_the_cap(depth: float) -> None:
    assert smart_params._compute_num_leaves(depth, 1.0) == _CAP  # noqa: SLF001


def test_the_smallest_ratio_follows_the_exact_rule_past_depth_1023() -> None:
    """The ratio domain is (0, 1]: a subnormal ratio stays below the cap."""
    tiny = 5e-324  # 2**-1074
    assert smart_params._compute_num_leaves(1024, tiny) == 8  # noqa: SLF001
    assert smart_params._compute_num_leaves(1090, tiny) == 65536  # noqa: SLF001
    assert smart_params._compute_num_leaves(1091, tiny) == _CAP  # noqa: SLF001
    for depth in (1024, 1050, 1080, 1088, 1089, 1090):
        assert smart_params._compute_num_leaves(depth, tiny) == _reference(depth, tiny)  # noqa: SLF001


def test_a_fractional_depth_with_the_smallest_ratio_is_not_rounded_early() -> None:
    """Review round 2: ratio * sqrt(2) rounded back to the smallest subnormal."""
    tiny = 5e-324
    expected = math.ceil(Fraction(tiny) * Fraction(2.0**0.5) * (1 << 1090))
    assert expected == 92682
    assert smart_params._compute_num_leaves(1090.5, tiny) == expected  # noqa: SLF001


class _Recorder:
    """Records each ``lgb.train`` call, then trains a tiny booster instead."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def __call__(
        self, params: dict[str, Any], train_set: Any, num_boost_round: int, **kw: Any
    ) -> Any:
        depth = _canonical(params, "max_depth")
        self.calls.append((depth, _canonical(params, "num_leaves")))
        small = {
            k: v
            for k, v in params.items()
            if LGBM_CANONICAL_NAME.get(k, k) not in ("num_leaves", "max_depth")
        }
        small.update(num_leaves=4, max_depth=2)
        return _REAL_TRAIN(small, train_set, num_boost_round=1, **kw)


def _canonical(params: dict[str, Any], name: str) -> Any:
    return next(v for k, v in params.items() if LGBM_CANONICAL_NAME.get(k, k) == name)


_REAL_TRAIN = lightgbm.train


@pytest.fixture
def train_stub(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(lightgbm, "train", recorder)
    return recorder


_N_SPLITS = 2


def _config(**model: Any) -> dict[str, Any]:
    cfg = make_config("regression", n_splits=_N_SPLITS, **model)
    cfg["model"]["auto_num_leaves"] = True
    return cfg


def test_fit_with_depth_1024_trains_every_booster_with_the_cap(
    train_stub: _Recorder,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the H-0140 memory warning fires here
        Model(_config(max_depth=1024)).fit(data=make_regression_df(n=80))
    # One booster per outer fold, then the refit.
    assert train_stub.calls == [(1024, _CAP)] * (_N_SPLITS + 1)


def test_tune_runs_every_trial_when_one_draws_depth_1024(train_stub: _Recorder) -> None:
    """The study used to abort at the first such trial (OverflowError)."""
    n_trials = 6
    cfg = _config()
    cfg["tuning"] = {
        "optuna": {
            "params": {"n_trials": n_trials},
            "space": {
                "max_depth": {"type": "categorical", "choices": [1024, 4]},
            },
            "space_mode": "replace",
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Model(cfg, data=make_regression_df(n=80)).tune()
    assert len(result.trials) == n_trials
    assert {t.state for t in result.trials} == {"complete"}
    depths = [t.params["max_depth"] for t in result.trials]
    assert set(depths) == {1024, 4}
    leaves = {1024: _CAP, 4: 16}
    # Trials run in order, one booster per outer fold each.
    expected = [(d, leaves[d]) for d in depths for _ in range(_N_SPLITS)]
    assert train_stub.calls == expected
