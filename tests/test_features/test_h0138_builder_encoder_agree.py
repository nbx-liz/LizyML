"""H-0138: the data builder and the encoder decide the same columns.

A column the builder keeps as object (its cast raised or was not faithful) must
be rebuilt by key in the encoder too, and become a stand-in column; the
encoder's own faithful check runs today's lookup on the training rows rather
than predicting it (implementation review round 1, major 1). The population is
every 3-value mix of the values pandas confuses, beside the sentinel ``"z"``.
"""

from __future__ import annotations

import decimal
import itertools
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.category_values import IDENTITY_OBJECT_COLUMNS, category_identity_key
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.data.dataframe_builder import _apply_categorical
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from lizyml.features.encoders.category_identity import keyed_categories
from tests.test_codegen.test_h0133_codec_copy import COPY
from tests.test_features.test_h0138_identity import NP_ONE_TYPES, _wrap

CONFUSED: list[Any] = [
    False,
    True,
    0,
    1,
    2**70,
    decimal.Decimal("1"),
    *(t(1) for t in NP_ONE_TYPES),
    *(_wrap(1, d) for d in (1, 2, 3, 8)),
]
MIXES = list(itertools.combinations(range(len(CONFUSED)), 3))


def _column(indices: tuple[int, ...]) -> pd.Series:
    return pd.Series([*(CONFUSED[i] for i in indices), "z"] * 3, dtype=object)


def test_the_collision_chain_of_round_1_is_rebuilt() -> None:
    """``False``, ``np.int8(1)``, an 8-deep tuple and ``"z"``: the cast raises."""
    series = pd.Series([False, np.int8(1), _wrap(1, 8), "z"] * 3, dtype=object)
    with pytest.raises(ValueError):
        series.astype("category")
    cats, mode, rebuilt = keyed_categories(series)
    assert rebuilt
    assert mode == _wrap(1, 8)  # the first category of the tie, by key counts
    copy_cats, copy_mode = COPY._keyed_categories(series)
    assert [category_identity_key(c) for c in copy_cats] == [
        category_identity_key(c) for c in cats
    ]
    assert category_identity_key(copy_mode) == category_identity_key(mode)


@pytest.mark.parametrize("indices", MIXES, ids=lambda i: "-".join(map(str, i)))
def test_a_column_the_builder_keeps_is_a_stand_in_column(
    indices: tuple[int, ...],
) -> None:
    X = _apply_categorical(pd.DataFrame({"c": _column(indices)}), FeatureSpec())
    marked = "c" in X.attrs.get(IDENTITY_OBJECT_COLUMNS, [])
    encoder = CategoricalEncoder().fit(X, ["c"])
    state = encoder.get_state()
    assert state["identity_columns"] == ["c"]
    if marked:
        assert state["stand_in_columns"] == ["c"]
    # Whatever was decided, the training rows get the codes of their keys.
    cats = state["categories"]["c"]
    codes = encoder.transform(X)["c"].cat.codes.tolist()
    rows = X["c"].to_numpy(dtype=object)
    assert [category_identity_key(cats[k]) for k in codes] == [
        category_identity_key(v) for v in rows
    ]
    # And the generated copy learns the same categories and mode.
    copy_cats, copy_mode = COPY._keyed_categories(X["c"])
    assert [category_identity_key(c) for c in copy_cats] == [
        category_identity_key(c) for c in cats
    ]
    assert category_identity_key(copy_mode) == category_identity_key(
        state["modes"]["c"]
    )
