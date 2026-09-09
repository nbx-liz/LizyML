"""Pin the deliberate literal-domain differences recorded by H-0095/H-0098."""

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.param_domain import normalise_params
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.tuning.search_space import parse_space


@pytest.mark.parametrize(
    "value",
    [np.float64(0.5), np.float32(0.5), np.int64(2), np.bool_(True), np.str_("gbdt")],
)
def test_numpy_choices_are_refused_at_the_named_entrance(value: object) -> None:
    with pytest.raises(LizyMLError) as caught:
        parse_space({"candidate": {"type": "categorical", "choices": [value]}})
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert "candidate" in str(caught.value)
    assert "index 0" in str(caught.value)
    # These same scalars are deliberately supported at the normalization surfaces.
    plain = normalise_params({"candidate": value}, surface="model.params")
    assert type(plain["candidate"]) in (float, int, bool, str)


@pytest.mark.parametrize(
    ("kind", "low", "high", "dimension", "plain_type"),
    [
        ("float", np.float64(0.1), np.float64(0.9), FloatDim, float),
        ("int", np.int64(2), np.int64(8), IntDim, int),
    ],
)
def test_numeric_bounds_are_plain_before_sampling(
    kind: str, low: object, high: object, dimension: type, plain_type: type
) -> None:
    dim = parse_space({"candidate": {"type": kind, "low": low, "high": high}})[0]
    assert isinstance(dim, dimension)
    assert isinstance(dim, (FloatDim, IntDim))
    assert type(dim.low) is plain_type
    assert type(dim.high) is plain_type


def test_plain_categorical_literals_remain_unchanged() -> None:
    values = [None, True, 2, 0.5, "gbdt"]
    dim = parse_space({"candidate": {"type": "categorical", "choices": values}})[0]
    assert isinstance(dim, CategoricalDim)
    assert all(
        actual is expected for actual, expected in zip(dim.choices, values, strict=True)
    )
