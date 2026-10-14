"""H-0124: ``step`` through expansion, the lineage codec and the lineage check.

Criteria 6 (expansion keeps edges on the grid, never narrows, stops a log
lower edge at ``step``), 7 (``step`` survives parse -> attach -> expand ->
encode / decode; a record without ``step`` reads ``None``; an unknown key is
refused; a trial whose parameters are outside its own round's space makes the
lineage ``inconsistent``).
"""

from __future__ import annotations

import copy
import json
import shutil
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.estimators.lgbm.provider import _LGBM_PARAMETER_BOUNDS
from lizyml.tuning.lineage import Route, classify_study, round_key
from lizyml.tuning.lineage_codec import (
    LineageCodecError,
    decode_dim,
    dim_signature,
    encode_dim,
    to_attr_value,
)
from lizyml.tuning.search_space import (
    attach_bounds,
    detect_boundary,
    expand_dims,
    parse_space,
)
from tests.test_tuning._lineage_fixtures import (
    FakeTrial,
    attrs_of,
    stored_end,
    stored_round,
    stored_study,
)
from tests.test_tuning._resume_helpers import fast_config, frame, load, url

BOUNDS = _LGBM_PARAMETER_BOUNDS


def _expand(dim: Any, best: float) -> tuple[Any, Any]:
    """The boundary status of *dim* for *best* and the expanded dim."""
    (bounded,) = attach_bounds([dim], BOUNDS)
    report = detect_boundary([bounded], {dim.name: best})
    (status,) = report.dims
    (expanded,) = expand_dims([bounded], report)
    return status, expanded


# ---------------------------------------------------------------------------
# Criterion 6: expansion
# ---------------------------------------------------------------------------


def test_float_log_lower_edge_snaps_down_then_stops_at_step() -> None:
    dim = FloatDim("learning_rate", 0.02, 0.1, log=True, step=0.01)
    status, expanded = _expand(dim, 0.02)
    assert status.expanded and status.new_low == 0.01  # 0.02/3 -> 0 -> step
    assert (expanded.low, expanded.high, expanded.step) == (0.01, 0.1, 0.01)


def test_int_log_lower_edge_snaps_down_then_stops_at_step() -> None:
    dim = IntDim("max_bin", 8, 64, log=True, step=4)
    status, expanded = _expand(dim, 8)
    assert status.expanded and status.new_low == 4  # 8 // 3 = 2 -> 0 -> 4
    assert (expanded.low, expanded.high, expanded.step) == (4, 64, 4)
    assert type(expanded.low) is int


def test_log_lower_edge_stops_at_step_without_a_provider_bound() -> None:
    """With no bound to clamp it, the snapped edge 0 is still not taken."""
    dim = FloatDim("x", 0.02, 0.1, log=True, step=0.01)
    status, expanded = _expand(dim, 0.02)
    assert status.expanded and not status.clamped_to_bound
    assert status.new_low == 0.01
    assert expanded.low == 0.01


def test_upper_edge_past_the_float_range_is_not_expanded() -> None:
    """3e308 is past the float range: the edge is not expanded (amendment 1).

    H-0124 stopped it at the last grid point inside the range (1.7e308); a
    grid point near the limit can leave the grid as a float, so amendment 1
    policy 10 leaves the edge where it is instead.
    """
    dim = FloatDim("x", 1e307, 1e308, log=True, step=1e307)
    status, expanded = _expand(dim, 1e308)
    assert status.edge == "upper" and not status.expanded
    assert (status.new_low, status.new_high) == (None, None)
    assert not status.clamped_to_bound
    assert (expanded.low, expanded.high, expanded.step) == (1e307, 1e308, 1e307)


def test_log_lower_edge_at_step_is_not_expanded() -> None:
    dim = FloatDim("learning_rate", 0.01, 0.1, log=True, step=0.01)
    status, expanded = _expand(dim, 0.01)
    assert status.edge == "lower"
    assert not status.expanded
    assert expanded == attach_bounds([dim], BOUNDS)[0]


def test_lower_edge_snaps_outward_not_inward() -> None:
    float_status, _ = _expand(FloatDim("x", 0.1, 1.0, log=True, step=0.01), 0.1)
    assert float_status.new_low == 0.03  # 0.1/3 = 0.0333...; inward would be 0.04
    int_status, _ = _expand(IntDim("n", 10, 100, log=True, step=2), 10)
    assert int_status.new_low == 2  # 10 // 3 = 3; inward would be 4


def test_upper_edge_clamped_to_the_bound_snaps_inward() -> None:
    dim = FloatDim("feature_fraction", 0.15, 0.6, step=0.15)
    status, expanded = _expand(dim, 0.6)
    assert status.expanded and status.clamped_to_bound
    assert status.new_high == 0.9  # 1.05 -> 1.0 (the bound) -> 0.9
    assert (expanded.low, expanded.high) == (0.15, 0.9)


def test_lower_edge_clamped_back_to_the_old_edge_is_not_expanded() -> None:
    dim = FloatDim("feature_fraction", 0.15, 0.6, step=0.15)
    status, _ = _expand(dim, 0.15)  # 0 -> 0.001 (the bound) -> 0.15
    assert status.edge == "lower"
    assert not status.expanded


def test_an_outward_snap_never_crosses_a_provider_bound() -> None:
    """0.2/3 = 0.0667 is inside the bound 0.05; its grid point 0.04 is not."""
    dim = FloatDim("validation_ratio", 0.2, 0.4, log=True, step=0.04)
    status, expanded = _expand(dim, 0.2)
    assert status.expanded and status.clamped_to_bound
    assert status.new_low == 0.08
    assert expanded.low == 0.08


def test_linear_expansion_is_exact_on_a_decimal_grid() -> None:
    """``0.2 + (0.2 - 0.1)`` is 0.30000000000000004 in float; the edge is 0.3."""
    status, expanded = _expand(FloatDim("x", 0.1, 0.2, step=0.1), 0.2)
    assert status.new_high == 0.3
    assert expanded.high == 0.3


def test_int_lower_edge_keeps_the_one_floor() -> None:
    """``[4, 8]`` step 4: the int guard holds the edge at 1, whose grid point
    above is 4 -- the old edge -- so the range is not expanded."""
    status, _ = _expand(IntDim("n", 4, 8, step=4), 4)
    assert status.edge == "lower"
    assert not status.expanded


def _on_grid(value: float, step: float) -> bool:
    ratio = Fraction(Decimal(str(value))) / Fraction(Decimal(str(step)))
    return ratio.denominator == 1


def test_expanded_edges_stay_on_the_grid() -> None:
    dims = [
        FloatDim("learning_rate", 0.05, 0.3, log=True, step=0.05),
        IntDim("num_leaves", 6, 30, step=3),
        FloatDim("feature_fraction", 0.2, 0.8, step=0.2),
    ]
    moved = 0
    for dim in dims:
        for best in (dim.low, dim.high):
            status, expanded = _expand(dim, best)
            moved += status.expanded
            assert expanded.step == dim.step
            assert _on_grid(expanded.low, dim.step), (dim, best, expanded)
            assert _on_grid(expanded.high, dim.step), (dim, best, expanded)
    assert moved >= 3


def test_dims_without_step_expand_as_before() -> None:
    status, expanded = _expand(FloatDim("x", 0.1, 0.2), 0.2)
    assert status.new_high == 0.2 + (0.2 - 0.1)
    assert expanded.step is None


# ---------------------------------------------------------------------------
# Criterion 7: step through attach / expand / the codec
# ---------------------------------------------------------------------------


def _round_trip(dim: Any) -> Any:
    return decode_dim(to_attr_value(encode_dim(dim)))


def test_step_survives_parse_attach_expand_encode_decode() -> None:
    (parsed,) = parse_space(
        {
            "learning_rate": {
                "type": "float",
                "low": 0.02,
                "high": 0.1,
                "log": True,
                "step": 0.01,
            }
        }
    )
    (bounded,) = attach_bounds([parsed], BOUNDS)
    assert bounded.step == 0.01
    report = detect_boundary([bounded], {"learning_rate": 0.02})
    (expanded,) = expand_dims([bounded], report)
    assert expanded.step == 0.01 and expanded.low == 0.01
    assert _round_trip(expanded) == expanded


@pytest.mark.parametrize(
    "dim",
    [
        FloatDim("a", 0.1, 0.3, step=0.1),
        FloatDim("a", 0.0, 3.0, step=1),
        IntDim("n", 500, 2000, step=100, min_allowed=10, max_allowed=10000),
        FloatDim("a", 0.1, 0.3),
    ],
    ids=["float", "float_int_step", "int", "none"],
)
def test_codec_round_trips_step(dim: Any) -> None:
    back = _round_trip(dim)
    assert back == dim
    assert type(back.step) is type(dim.step)


def test_a_record_without_step_reads_none() -> None:
    payload = to_attr_value(encode_dim(IntDim("n", 1, 9)))
    del payload["step"]
    assert decode_dim(payload) == IntDim("n", 1, 9)


@pytest.mark.parametrize(
    "dim",
    [IntDim("n", 1, 9), FloatDim("a", 0.1, 0.3), CategoricalDim("c", (1, 2))],
    ids=["int", "float", "categorical"],
)
def test_a_dim_record_with_an_unknown_key_is_refused(dim: Any) -> None:
    payload = to_attr_value(encode_dim(dim))
    payload["q"] = ["float", 0.1]
    with pytest.raises(LineageCodecError, match="'q'"):
        decode_dim(payload)


def test_a_categorical_record_with_step_is_refused() -> None:
    payload = to_attr_value(encode_dim(CategoricalDim("c", (1, 2))))
    payload["step"] = None
    with pytest.raises(LineageCodecError, match="'step'"):
        decode_dim(payload)


def test_codec_refuses_a_step_it_cannot_store() -> None:
    with pytest.raises(LineageCodecError):
        encode_dim(IntDim("n", 1, 9, step=1.5))  # type: ignore[arg-type]
    payload = to_attr_value(encode_dim(IntDim("n", 1, 9, step=2)))
    payload["step"] = ["float", 2.0]
    with pytest.raises(LineageCodecError):
        decode_dim(payload)


def test_signature_sees_step() -> None:
    a = FloatDim("x", 0.1, 0.3)
    assert dim_signature(a) != dim_signature(FloatDim("x", 0.1, 0.3, step=0.1))
    assert dim_signature(FloatDim("x", 0.1, 0.3, step=0.1)) != dim_signature(
        FloatDim("x", 0.1, 0.3, step=0.05)
    )
    assert json.loads(dim_signature(a))["step"] == ["none", None]


# ---------------------------------------------------------------------------
# Criterion 7: each trial's parameters are within its own round's space
# ---------------------------------------------------------------------------

X = FloatDim("x", 0.0, 1.0, step=0.2)
Y = FloatDim("y", 0.0, 1.0, step=0.25)


def _xy_attrs() -> dict[str, Any]:
    routes = (Route("x", "x", "model"), Route("y", "y", "model"))
    return attrs_of(
        stored_study(),
        (
            stored_round(1, 0, declared=(X,), effective=(X,), routes_added=routes),
            stored_round(2, 1, declared=(Y,), effective=(Y,), routes_added=()),
        ),
        {1: stored_end(), 2: stored_end()},
    )


def test_trial_params_in_their_round_space_are_consistent() -> None:
    trials = [FakeTrial(0, params={"x": 0.41}), FakeTrial(1, params={"y": 0.3})]
    assert classify_study(_xy_attrs(), trials).state == "completed"


def test_trial_param_outside_its_round_space_is_inconsistent() -> None:
    """Round 1 declares ``{x}``; trial 0 holds ``y``, which round 2 routes."""
    trials = [FakeTrial(0, params={"y": 0.41}), FakeTrial(1, params={"y": 0.3})]
    lineage = classify_study(_xy_attrs(), trials)
    assert lineage.state == "inconsistent"
    assert "'y'" in lineage.reason and "round 1" in lineage.reason


@pytest.fixture(scope="module")
def two_rounds(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("two_rounds_step")
    storage = url(root)
    Model(fast_config(), data=frame()).tune(storage=storage, study_name="s")
    Model(fast_config(), data=frame()).tune(
        resume=True, storage=storage, study_name="s"
    )
    return root / "s.db"


def _drop_from_round_1(storage: str, name: str) -> None:
    study = load(storage, "s")
    record = copy.deepcopy(study.user_attrs[round_key(1)])
    record["effective"] = [d for d in record["effective"] if d["name"] != name]
    study.set_user_attr(round_key(1), record)


def test_model_refuses_a_study_whose_trial_is_outside_its_round(
    two_rounds: Path, tmp_path: Path
) -> None:
    shutil.copy(two_rounds, tmp_path / "s.db")
    storage = url(tmp_path)
    before = load(storage, "s")
    assert classify_study(before.user_attrs, before.trials).state == "completed"
    _drop_from_round_1(storage, "num_leaves")
    study = load(storage, "s")
    assert classify_study(study.user_attrs, study.trials).state == "inconsistent"
    with pytest.raises(LizyMLError) as resumed:
        Model(fast_config(), data=frame()).tune(
            resume=True, storage=storage, study_name="s"
        )
    assert resumed.value.code is ErrorCode.CONFIG_INVALID
    assert resumed.value.context["state"] == "inconsistent"
    with pytest.raises(LizyMLError) as restored:
        Model(fast_config(), data=frame()).restore_tuning(storage, "s")
    assert restored.value.context["state"] == "inconsistent"
