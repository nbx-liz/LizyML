"""H-0136 P1/P3 unit cases: the comparison of a state with its projection.

Hand-built encoder states reach every ``kind`` directly. Through a real fit
the mode is one of the learned categories and gets the same projection, so
``mode`` is reached only here; ``declared_change`` is reached alone here,
where a real state with the same values would be refused first.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lizyml.features.pipeline_base import CategoryStateChange, category_state_changes
from lizyml.persistence._category_preflight import declared_change, state_changes


def _state(categories: list[Any], mode: Any) -> dict[str, Any]:
    return {
        "unseen_policy": "mode",
        "categories": {"c": categories},
        "modes": {"c": mode},
        "stand_in_columns": [],
    }


def test_the_same_state_has_no_change() -> None:
    state = _state(["a", "b"], "a")
    assert category_state_changes(state, _state(["a", "b"], "a")) == []


def test_a_learned_category_missing_after_projection_is_unseen() -> None:
    change = category_state_changes(_state(["a", "b"], "a"), _state(["a", "z"], "a"))
    assert change == [CategoryStateChange("c", "unseen", value="b")]


def test_a_projection_whose_transform_raises_is_raises() -> None:
    # Every learned value is still a category, but pandas refuses categories
    # that are not unique.
    change = category_state_changes(_state([1, 2], 1), _state([1, 2, 2], 1))
    assert len(change) == 1
    assert change[0].kind == "raises"
    assert change[0].exception == "ValueError"
    assert change[0].value is None


def test_reordered_categories_change_codes() -> None:
    change = category_state_changes(_state(["a", "b"], "a"), _state(["b", "a"], "a"))
    assert change == [CategoryStateChange("c", "codes", value="a")]


def test_a_mode_with_another_code_is_mode() -> None:
    change = category_state_changes(_state(["a", "b"], "a"), _state(["a", "b"], "b"))
    assert change == [CategoryStateChange("c", "mode", value="a")]


def test_a_column_without_categories_is_not_judged() -> None:
    assert category_state_changes(_state([], None), _state([], None)) == []


def test_state_changes_reads_only_encoder_states() -> None:
    assert state_changes(None) == []
    assert state_changes({"encoder": None}) == []
    assert state_changes({"encoder": {"categories": None}}) == []


def test_declared_categories_that_collide_after_json_raise() -> None:
    change = declared_change("c", [np.uint64(4), np.timedelta64(4, "M"), "x"])
    assert change is not None
    assert (change.column, change.kind, change.exception) == (
        "c",
        "raises",
        "ValueError",
    )


def test_plain_declared_categories_have_no_change() -> None:
    assert declared_change("c", ["a", 1, 2.5]) is None


def test_declared_categories_pandas_cannot_build_are_not_judged() -> None:
    assert declared_change("c", [1, 1]) is None
