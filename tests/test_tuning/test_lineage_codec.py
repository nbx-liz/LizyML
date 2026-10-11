"""Unit tests for the lineage attribute codec (H-0122, PR 1 design §2).

Every round trip goes through :func:`to_attr_value` first, because an RDB
storage hands back the ``json.loads`` of what was written, while the in-memory
storage keeps the object as written. A test that skipped the JSON step would
pass on tuples that a real storage turns into lists.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.core.types.tuning_result import BoundaryDimStatus, BoundaryReport
from lizyml.tuning.lineage_codec import (
    LineageCodecError,
    decode_dim,
    decode_dims,
    decode_report,
    decode_scalar,
    encode_dim,
    encode_dims,
    encode_report,
    encode_scalar,
    to_attr_value,
)


def _scalar_round_trip(value: Any) -> Any:
    return decode_scalar(to_attr_value(encode_scalar(value)))


def _same_scalar(a: Any, b: Any) -> bool:
    """Equal value **and** equal type; NaN equals NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return bool(a == b)


# ---------------------------------------------------------------------------
# Tagged scalars
# ---------------------------------------------------------------------------


class TestScalars:
    @pytest.mark.parametrize(
        "value",
        [
            True,
            False,
            0,
            1,
            -7,
            2**53 + 1,
            -(2**64) - 3,
            1.0,
            0.0,
            -0.0,
            1e-300,
            1e300,
            float("nan"),
            float("inf"),
            float("-inf"),
            "",
            "gbdt",
            "1",
            "nan",
            None,
        ],
    )
    def test_round_trip_keeps_value_and_type(self, value: Any) -> None:
        assert _same_scalar(_scalar_round_trip(value), value)

    def test_true_one_and_one_point_zero_stay_distinct(self) -> None:
        decoded = [_scalar_round_trip(v) for v in (True, 1, 1.0)]
        assert [type(v) for v in decoded] == [bool, int, float]
        encoded = [to_attr_value(encode_scalar(v)) for v in (True, 1, 1.0)]
        assert len({json.dumps(e) for e in encoded}) == 3

    def test_negative_zero_keeps_its_sign(self) -> None:
        assert math.copysign(1.0, _scalar_round_trip(-0.0)) == -1.0

    def test_int_beyond_2_53_is_exact(self) -> None:
        big = 2**53 + 1
        assert float(big) != big  # a float round trip would lose it
        assert _scalar_round_trip(big) == big

    def test_nan_and_inf_are_strict_json(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            text = json.dumps(encode_scalar(value), allow_nan=False)
            assert "NaN" not in text and "Infinity" not in text

    @pytest.mark.parametrize(
        "value",
        [np.float64(1.0), np.int64(1), np.bool_(True), np.str_("a"), [1], (1,), {}],
    )
    def test_non_plain_scalar_is_refused(self, value: Any) -> None:
        with pytest.raises(LineageCodecError):
            encode_scalar(value)

    def test_subclass_of_a_plain_type_is_refused(self) -> None:
        class MyFloat(float):
            pass

        with pytest.raises(LineageCodecError):
            encode_scalar(MyFloat(1.0))

    @pytest.mark.parametrize(
        "payload",
        [
            ["int", True],
            ["int", 1.0],
            ["bool", 1],
            ["float", 1],
            ["float", True],
            ["float", "NaN"],
            ["float", "1.0"],
            ["str", 1],
            ["none", 0],
            ["none", ""],
            ["complex", 1],
            ["int"],
            ["int", 1, 2],
            "int",
            None,
            {"int": 1},
            ("int", 1, 2),
        ],
    )
    def test_malformed_payload_is_refused(self, payload: Any) -> None:
        with pytest.raises(LineageCodecError):
            decode_scalar(payload)

    def test_lineage_codec_error_is_a_value_error(self) -> None:
        assert issubclass(LineageCodecError, ValueError)


# ---------------------------------------------------------------------------
# to_attr_value
# ---------------------------------------------------------------------------


class TestToAttrValue:
    def test_tuples_become_lists(self) -> None:
        assert to_attr_value({"a": (1, (2, 3))}) == {"a": [1, [2, 3]]}

    def test_is_idempotent(self) -> None:
        payload = {"a": (1, 2.5, None, True, "x"), "b": {"c": [("int", 1)]}}
        once = to_attr_value(payload)
        assert to_attr_value(once) == once

    def test_nan_is_refused(self) -> None:
        with pytest.raises(LineageCodecError):
            to_attr_value({"x": float("nan")})

    def test_non_serialisable_is_refused(self) -> None:
        with pytest.raises(LineageCodecError):
            to_attr_value({"x": object()})

    def test_non_string_key_is_refused(self) -> None:
        # json.dumps would silently turn 1 into "1", merging it with a str key.
        with pytest.raises(LineageCodecError):
            to_attr_value({1: "a"})


# ---------------------------------------------------------------------------
# SearchDim
# ---------------------------------------------------------------------------


def _dim_round_trip(dim: Any) -> Any:
    return decode_dim(to_attr_value(encode_dim(dim)))


class TestSearchDim:
    @pytest.mark.parametrize(
        "dim",
        [
            FloatDim("learning_rate", 0.01, 0.3, log=True),
            FloatDim(
                "learning_rate", 0.01, 0.3, log=True, min_allowed=1e-8, max_allowed=1.0
            ),
            FloatDim("lambda_l1", 0, 10.0),  # int literal low
            FloatDim("x", 0.0, 1.0, category="smart"),
            IntDim("num_leaves", 8, 32),
            IntDim("num_leaves", 8, 32, log=True, min_allowed=2, max_allowed=131072),
            IntDim("big", 2**53 + 1, 2**60, category="training"),
            CategoricalDim("boosting", ("gbdt", "dart")),
            CategoricalDim("c", ("gbdt", True, 1, 1.0, None, float("inf"))),
        ],
    )
    def test_round_trip_is_exact(self, dim: Any) -> None:
        decoded = _dim_round_trip(dim)
        assert decoded == dim
        assert type(decoded) is type(dim)

    def test_float_dim_int_low_keeps_its_type(self) -> None:
        decoded = _dim_round_trip(FloatDim("lambda_l1", 0, 10.0))
        assert type(decoded.low) is int
        assert type(decoded.high) is float

    def test_categorical_choice_types_survive(self) -> None:
        decoded = _dim_round_trip(CategoricalDim("c", (True, 1, 1.0, "1", None)))
        assert isinstance(decoded.choices, tuple)
        assert [type(c) for c in decoded.choices] == [
            bool,
            int,
            float,
            str,
            type(None),
        ]

    def test_categorical_nan_choice_survives(self) -> None:
        # A decoded NaN is a fresh object and ``nan != nan``, so dataclass
        # equality cannot hold here; compare the choice itself.
        decoded = _dim_round_trip(CategoricalDim("c", (1.0, float("nan"))))
        assert decoded.choices[0] == 1.0 and type(decoded.choices[0]) is float
        assert type(decoded.choices[1]) is float and math.isnan(decoded.choices[1])
        assert decoded.name == "c" and decoded.category == "model"

    def test_dims_list_round_trip(self) -> None:
        dims = [FloatDim("a", 0.1, 0.2), IntDim("b", 1, 2), CategoricalDim("c", (1,))]
        assert decode_dims(to_attr_value(encode_dims(dims))) == tuple(dims)

    def test_numpy_bound_is_refused(self) -> None:
        with pytest.raises(LineageCodecError):
            encode_dim(FloatDim("a", np.float64(0.1), 0.2))  # type: ignore[arg-type]

    def test_numpy_choice_is_refused(self) -> None:
        with pytest.raises(LineageCodecError):
            encode_dim(CategoricalDim("a", (np.int64(1),)))

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.update(kind="weird"),
            lambda p: p.pop("name"),
            lambda p: p.update(name=1),
            lambda p: p.update(category="other"),
            lambda p: p.update(log=1),
            lambda p: p.update(low=["str", "a"]),
            lambda p: p.update(low=["bool", True]),
            lambda p: p.update(min_allowed=["str", "a"]),
            lambda p: p.pop("high"),
        ],
    )
    def test_malformed_numeric_dim_is_refused(self, mutate: Any) -> None:
        payload = to_attr_value(encode_dim(FloatDim("a", 0.1, 0.2)))
        mutate(payload)
        with pytest.raises(LineageCodecError):
            decode_dim(payload)

    def test_int_dim_with_float_bound_is_refused(self) -> None:
        payload = to_attr_value(encode_dim(IntDim("a", 1, 2)))
        payload["low"] = ["float", 1.0]
        with pytest.raises(LineageCodecError):
            decode_dim(payload)

    @pytest.mark.parametrize(
        "choices", [[], "ab", [["int", True]], None, [["int", 1], "x"]]
    )
    def test_malformed_choices_are_refused(self, choices: Any) -> None:
        payload = to_attr_value(encode_dim(CategoricalDim("a", (1,))))
        payload["choices"] = choices
        with pytest.raises(LineageCodecError):
            decode_dim(payload)

    def test_dim_payload_must_be_a_dict(self) -> None:
        with pytest.raises(LineageCodecError):
            decode_dim([1, 2])
        with pytest.raises(LineageCodecError):
            decode_dims({"a": 1})


# ---------------------------------------------------------------------------
# BoundaryReport
# ---------------------------------------------------------------------------


def _report() -> BoundaryReport:
    return BoundaryReport(
        dims=(
            BoundaryDimStatus(
                name="learning_rate",
                best_value=0.29,
                low=0.01,
                high=0.3,
                position_pct=0.9655,
                edge="upper",
                expanded=True,
                new_low=None,
                new_high=0.9,
                clamped_to_bound=False,
            ),
            BoundaryDimStatus(
                name="num_leaves",
                best_value=8,
                low=8,
                high=32,
                position_pct=0.0,
                edge="lower",
                expanded=True,
                new_low=1,
                new_high=None,
                clamped_to_bound=True,
            ),
            BoundaryDimStatus(
                name="boosting",
                best_value=True,
                low=None,
                high=None,
                position_pct=None,
                edge="none",
                expanded=False,
                new_low=None,
                new_high=None,
            ),
        ),
        expanded_names=("learning_rate", "num_leaves"),
    )


class TestBoundaryReport:
    def test_round_trip_is_exact(self) -> None:
        report = _report()
        decoded = decode_report(to_attr_value(encode_report(report)))
        assert decoded == report
        assert type(decoded.dims[2].best_value) is bool
        assert type(decoded.dims[1].best_value) is int
        assert isinstance(decoded.dims, tuple)
        assert isinstance(decoded.expanded_names, tuple)

    def test_empty_report_round_trip(self) -> None:
        report = BoundaryReport(dims=(), expanded_names=())
        assert decode_report(to_attr_value(encode_report(report))) == report

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.pop("dims"),
            lambda p: p.update(expanded_names="a"),
            lambda p: p.update(expanded_names=[1]),
            lambda p: p["dims"][0].update(edge=1),
            lambda p: p["dims"][0].update(edge="sideways"),
            lambda p: p["dims"][0].update(expanded="yes"),
            lambda p: p["dims"][0].pop("clamped_to_bound"),
            lambda p: p["dims"][0].update(position_pct=["str", "x"]),
            lambda p: p["dims"][0].update(low=["bool", True]),
        ],
    )
    def test_malformed_report_is_refused(self, mutate: Any) -> None:
        payload = to_attr_value(encode_report(_report()))
        mutate(payload)
        with pytest.raises(LineageCodecError):
            decode_report(payload)
