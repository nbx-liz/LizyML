"""H-0133 P3: the encoder decides the stand-in columns at fit and renames them.

A column with a category the injected predicate flags hands LightGBM its
categories as positions 0..k-1; the codes are unchanged. The decision is in
the state ``get_state`` returns right after fit (CV snapshots it before any
transform), and a state without the field keeps every column raw.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from lizyml.features.encoders.categorical_encoder import CategoricalEncoder


def _unsafe(value: object) -> bool:
    return isinstance(value, (bytes, Decimal))


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "b": pd.Series([b"x", b"y", b"x", b"z"], dtype=object).astype("category"),
            "s": pd.Series(["p", "q", "p", "q"]).astype("category"),
        }
    )


def test_fit_records_the_stand_in_columns_before_any_transform() -> None:
    enc = CategoricalEncoder(needs_stand_in=_unsafe).fit(_frame(), ["b", "s"])
    assert enc.get_state()["stand_in_columns"] == ["b"]


def test_a_stand_in_column_carries_positions_with_the_same_codes() -> None:
    df = _frame()
    plain = CategoricalEncoder().fit(df, ["b", "s"]).transform(df)
    stood = CategoricalEncoder(needs_stand_in=_unsafe).fit(df, ["b", "s"]).transform(df)
    assert list(stood["b"].cat.categories) == [0, 1, 2]
    assert (stood["b"].cat.codes == plain["b"].cat.codes).all()
    # A column without a flagged value keeps its raw categories.
    assert list(stood["s"].cat.categories) == ["p", "q"]


def test_unseen_substitution_happens_on_raw_values_before_the_rename() -> None:
    df = _frame()
    enc = CategoricalEncoder(unseen_policy="mode", needs_stand_in=_unsafe).fit(
        df, ["b", "s"]
    )
    new = pd.DataFrame({"b": pd.Series([b"w", b"y"], dtype=object), "s": ["p", "q"]})
    out, warnings = enc.transform_with_warnings(new)
    assert list(out["b"].cat.codes) == [0, 1]  # b"w" -> the mode b"x" -> position 0
    assert warnings and "b'w'" in warnings[0]


def test_unseen_policy_error_still_raises_on_raw_values() -> None:
    from lizyml.core.exceptions import LizyMLError

    enc = CategoricalEncoder(unseen_policy="error", needs_stand_in=_unsafe).fit(
        _frame(), ["b"]
    )
    with pytest.raises(LizyMLError):
        enc.transform(pd.DataFrame({"b": pd.Series([b"w"], dtype=object)}))


def test_a_state_without_the_field_keeps_every_column_raw() -> None:
    df = _frame()
    state = CategoricalEncoder(needs_stand_in=_unsafe).fit(df, ["b", "s"]).get_state()
    old = {k: v for k, v in state.items() if k != "stand_in_columns"}
    out = CategoricalEncoder().load_state(old).transform(df)
    assert list(out["b"].cat.categories) == [b"x", b"y", b"z"]


def test_a_loaded_state_reproduces_the_stand_ins() -> None:
    df = _frame()
    state = CategoricalEncoder(needs_stand_in=_unsafe).fit(df, ["b", "s"]).get_state()
    out = CategoricalEncoder().load_state(state).transform(df)
    assert list(out["b"].cat.categories) == [0, 1, 2]


def test_without_a_predicate_no_column_is_replaced() -> None:
    enc = CategoricalEncoder().fit(_frame(), ["b", "s"])
    assert enc.get_state()["stand_in_columns"] == []


def test_a_refit_recomputes_the_stand_in_columns() -> None:
    enc = CategoricalEncoder(needs_stand_in=_unsafe)
    enc.fit(_frame(), ["b"])
    enc.fit(pd.DataFrame({"b": ["u", "v"]}), ["b"])
    assert enc.get_state()["stand_in_columns"] == []
