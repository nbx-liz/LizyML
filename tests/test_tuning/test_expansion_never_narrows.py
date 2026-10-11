"""#342: boundary expansion never narrows or inverts a range (H-0122 PR 2).

H-0111 lets a user range lie outside the provider bounds -- the bounds
constrain only expansion -- but the clamps in ``_expand_range`` /
``_expand_int_range`` moved an edge *inward* past the user's: a range below
``min_allowed`` was clamped up to it (the issue's reproduction, ``low > high``),
the linear ``0.0`` floor moved a negative ``low`` up, and the ``IntDim``
``max(1, ...)`` guard moved ``low`` of ``max_depth`` / ``bagging_freq`` up even
when only the upper edge expanded.

Expansion only ever widens: an edge that the bound, the floor or the guard
would move inward stays where the user put it, and an expansion that then
leaves the range unchanged is the H-0111 no-op (``expanded=False``).
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from lizyml import Model
from lizyml.core.types.search_dim import FloatDim, IntDim, SearchDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.tuning.lineage import round_key
from lizyml.tuning.search_space import (
    attach_bounds,
    detect_boundary,
    expand_dims,
    parse_space,
)
from tests.test_tuning._resume_helpers import fast_config, frame, load, url

# ---------------------------------------------------------------------------
# Unit: detect_boundary + expand_dims
# ---------------------------------------------------------------------------


def _expand(dim: SearchDim, best: Any) -> tuple[Any, SearchDim]:
    report = detect_boundary([dim], {dim.name: best}, threshold=0.05)
    (expanded,) = expand_dims([dim], report)
    return report.dims[0], expanded


def test_issue_342_reproduction_is_not_inverted() -> None:
    """The exact reproduction from #342: [3, 5], min_allowed 10, best 3."""
    space = parse_space(
        {"n_estimators": {"type": "int", "low": 3, "high": 5, "category": "model"}}
    )
    space = attach_bounds(space, LGBMProvider().parameter_bounds("regression"))
    report = detect_boundary(space, {"n_estimators": 3}, 0.05)
    (dim,) = expand_dims(space, report)

    assert isinstance(dim, IntDim)
    assert (dim.low, dim.high) == (3, 5)
    status = report.dims[0]
    # Reported as the H-0111 clamp no-op: the edge is near, the bound bit,
    # and the range did not move.
    assert status.edge == "lower"
    assert status.expanded is False
    assert (status.new_low, status.new_high) == (None, None)
    assert status.clamped_to_bound is True
    assert report.expanded_names == ()


#: ``(id, dim, best, expanded, clamped_to_bound, new range or None)``.
CASES: list[tuple[str, SearchDim, Any, bool, bool, tuple[Any, Any] | None]] = [
    (
        "int_below_min_lower_edge",
        IntDim("n_estimators", 3, 5, min_allowed=10, max_allowed=10000),
        3,
        False,
        True,
        None,
    ),
    (
        "int_below_min_upper_edge",
        # The upper-edge analogue: low sits below min_allowed and only the
        # upper edge expands; the clamp used to move low to 10 (10 > 7).
        IntDim("n_estimators", 3, 5, min_allowed=10, max_allowed=10000),
        5,
        True,
        False,
        (3, 7),
    ),
    (
        "float_below_min_lower_edge",
        FloatDim("lr", 0.001, 0.002, min_allowed=0.01, max_allowed=1.0),
        0.001,
        False,
        True,
        None,
    ),
    (
        "float_above_max_upper_edge",
        FloatDim("ff", 2.0, 3.0, min_allowed=0.0, max_allowed=1.0),
        3.0,
        False,
        True,
        None,
    ),
    (
        "float_above_max_lower_edge",
        FloatDim("ff", 2.0, 3.0, min_allowed=0.0, max_allowed=1.0),
        2.0,
        True,
        False,
        (1.0, 3.0),
    ),
    (
        "int_low_0_upper_edge",
        # bagging_freq: the max(1, ...) guard must not move low when only the
        # upper edge expands.
        IntDim("bagging_freq", 0, 4, min_allowed=0, max_allowed=100),
        4,
        True,
        False,
        (0, 8),
    ),
    (
        "float_negative_low_linear_lower_edge",
        # The linear 0.0 floor must not move a negative low up.
        FloatDim("x", -1.0, 1.0),
        -1.0,
        False,
        False,
        None,
    ),
    (
        "int_max_depth_lower_edge",
        # max_depth: -1 is meaningful; the guard would make it 1.
        IntDim("max_depth", -1, 5, min_allowed=-1, max_allowed=30),
        -1,
        False,
        False,
        None,
    ),
]


@pytest.mark.parametrize(
    ("dim", "best", "expanded", "clamped", "new_range"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_expansion_never_narrows(
    dim: SearchDim,
    best: Any,
    expanded: bool,
    clamped: bool,
    new_range: tuple[Any, Any] | None,
) -> None:
    status, out = _expand(dim, best)
    assert isinstance(out, FloatDim | IntDim)
    assert isinstance(dim, FloatDim | IntDim)
    assert out.low <= dim.low
    assert out.high >= dim.high
    assert out.low < out.high
    assert status.edge != "none"
    assert status.expanded is expanded
    assert status.clamped_to_bound is clamped
    if new_range is None:
        assert (status.new_low, status.new_high) == (None, None)
        assert (out.low, out.high) == (dim.low, dim.high)
    else:
        assert (status.new_low, status.new_high) == new_range
        assert (out.low, out.high) == new_range


_POINTS = (-3, 0, 1, 3, 10, 20, 50, 100, 200)


@pytest.mark.parametrize("kind", ["int", "float"])
@pytest.mark.parametrize("log", [False, True], ids=["linear", "log"])
def test_no_range_is_narrowed_over_a_grid(kind: str, log: bool) -> None:
    """Every range against bounds [10, 100], at both edges: never narrower.

    A closed sweep rather than a list of remembered cases: low/high take every
    position relative to the bounds (below, on, inside, above).
    """
    seen = 0
    for low, high in itertools.combinations(_POINTS, 2):
        if log and low <= 0:
            continue
        for best in (low, high):
            dim: SearchDim = (
                IntDim("d", low, high, log=log, min_allowed=10, max_allowed=100)
                if kind == "int"
                else FloatDim(
                    "d",
                    float(low),
                    float(high),
                    log=log,
                    min_allowed=10.0,
                    max_allowed=100.0,
                )
            )
            status, out = _expand(dim, best)
            assert isinstance(out, FloatDim | IntDim)
            assert out.low <= low, (low, high, best, out)
            assert out.high >= high, (low, high, best, out)
            assert out.low < out.high
            assert status.expanded is ((out.low, out.high) != (low, high))
            seen += 1
    assert seen > 20


def test_expand_dims_cannot_invert_with_a_hand_made_report() -> None:
    """The defensive invariant: a hand-made report cannot invert a range."""
    dim = IntDim("n", 3, 5)
    report = detect_boundary([dim], {"n": 5}, threshold=0.05)
    status = report.dims[0]
    bad = type(report)(
        dims=(
            type(status)(
                name="n",
                best_value=5,
                low=3,
                high=5,
                position_pct=1.0,
                edge="upper",
                expanded=True,
                new_low=9,
                new_high=8,
            ),
        ),
        expanded_names=("n",),
    )
    # Clamped outward first: low stays 3, so the result is [3, 8], not [9, 8].
    (out,) = expand_dims([dim], bad)
    assert isinstance(out, IntDim)
    assert (out.low, out.high) == (3, 8)


# ---------------------------------------------------------------------------
# Through Model.tune
# ---------------------------------------------------------------------------


def test_user_range_below_bound_is_not_inverted(tmp_path: Any) -> None:
    """Merge mode, n_estimators [3, 4] (below min 10), best at 3, expand.

    Before the fix, round 2's ``n_estimators`` became [10, 4]: every trial of
    the round FAILed inside Optuna and the round still returned success.
    """
    cfg = fast_config(mode="merge", n_trials=3)
    cfg["tuning"]["optuna"]["space"]["n_estimators"] = {
        "type": "int",
        "low": 3,
        "high": 4,
    }
    storage = url(tmp_path)
    model = Model(cfg, data=frame())
    first = model.tune(storage=storage, study_name="s", boundary_threshold=0.49)
    # Precondition (R-11): the best sits on the lower edge, where #342 bit.
    assert first.best_params["n_estimators"] == 3

    second = model.tune(resume=True, expand_boundary=True, boundary_threshold=0.49)

    study = load(storage, "s")
    record = study.user_attrs[round_key(2)]
    (n_est,) = [d for d in record["effective"] if d["name"] == "n_estimators"]
    assert (n_est["low"], n_est["high"]) == (["int", 3], ["int", 4])
    round_2 = [t for t in study.trials if t.number >= record["start_trial"]]
    assert round_2
    assert all(t.state.name == "COMPLETE" for t in round_2)
    for t in study.trials:
        assert 3 <= t.params["n_estimators"] <= 4
    row = model.boundary_table().set_index("dim").loc["n_estimators"]
    assert row["edge"] == "lower"
    assert bool(row["expanded"]) is False
    report = second.boundary_report
    assert report is not None
    (status,) = [s for s in report.dims if s.name == "n_estimators"]
    assert status.clamped_to_bound is True
