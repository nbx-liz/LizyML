"""``values_differ`` decides whether one parameter was written twice (H-0094).

Both refusals on the ``fit(params=...)`` path ask it the same question, so what
it answers for an awkward value is what they both do. Three review rounds found
defects here, each one a different way for "is this the same value?" to go
wrong, so the cases below are kept as a table of *inputs*, not of code paths:

- round 5 — comparing printed forms made ``1`` and ``1.0`` two values;
- self-review and the rounds 4-5 monitor — a bare ``!=`` raised on a numpy
  array, even for a value written once;
- round 6 — a numpy scalar fell through to the printed forms and was refused;
  a broadcast comparison called two sequences of different lengths equal; and a
  declared exception-safe fallback did not cover the comparison itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.value_equality import values_differ


class _RaisingEquality:
    """A value whose ``__eq__`` fails, which a user object legitimately may."""

    def __eq__(self, other: object) -> bool:
        raise ValueError("equality failed")

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return "_RaisingEquality()"


#: ``(label, first, second, differ)``. The label is what the case is *about*,
#: so a failure names the property rather than the literal.
CASES: list[tuple[str, object, object, bool]] = [
    ("a number equals itself", 0.5, 0.5, False),
    ("int and float of one value", 1, 1.0, False),
    ("different numbers", 1, 2, True),
    ("a library scalar and a python float", np.float64(0.5), 0.5, False),
    ("two library scalars of one value", np.array(1), np.array(1.0), False),
    ("equal arrays", np.array([1.0, 2.0]), np.array([1.0, 2.0]), False),
    ("reordered arrays", np.array([1.0, 2.0]), np.array([2.0, 1.0]), True),
    ("arrays of different length", np.array([1.0, 1.0]), np.array([1.0]), True),
    ("an empty array and a filled one", np.array([]), np.array([1.0]), True),
    ("two empty arrays", np.array([]), np.array([]), False),
    ("equal two-dimensional arrays", np.ones((2, 2)), np.ones((2, 2)), False),
    ("different two-dimensional arrays", np.ones((2, 2)), np.zeros((2, 2)), True),
    ("equal lists", [1.0, 2.0], [1.0, 2.0], False),
    ("lists of different length", [1.0], [1.0, 2.0], True),
    ("a list and an equal tuple", [1.0, 2.0], (1.0, 2.0), True),
    ("equal series", pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), False),
    ("equal strings", "binary", "binary", False),
    ("different strings", "binary", "regression", True),
    ("strings of different length", "auc", "auc_mu", True),
    ("equal dicts", {"a": 1}, {"a": 1}, False),
    ("none and none", None, None, False),
    ("none and a value", None, 0.5, True),
    ("a bool and the int it equals", True, 1, False),
    ("an equality that raises", _RaisingEquality(), _RaisingEquality(), False),
    ("nan is not itself, and that is deliberate", float("nan"), float("nan"), True),
]


@pytest.mark.parametrize(("label", "first", "second", "differ"), CASES)
def test_values_differ(label: str, first: object, second: object, differ: bool) -> None:
    assert values_differ(first, second) is differ, label


@pytest.mark.parametrize(("label", "first", "second", "differ"), CASES)
def test_values_differ_is_symmetric(
    label: str, first: object, second: object, differ: bool
) -> None:
    """Which spelling the caller wrote first must not change the answer.

    The two refusals compare in opposite orders -- the facade against the
    group's first entry, the adapter against the value it kept -- so an
    asymmetric answer would make the same call refused on one path and accepted
    on the other.
    """
    assert values_differ(second, first) is differ, label


def test_it_never_raises_on_the_case_table() -> None:
    """The docstring promises a fallback rather than an exception. Execute it.

    Round 6 found that promise false: the comparison itself ran outside the
    handler, so an ``__eq__`` that raised propagated, and the reduction caught
    only ``TypeError`` while a nested array raises ``ValueError``.
    """
    for label, first, second, _ in CASES:
        try:
            values_differ(first, second)
        except Exception as exc:  # noqa: BLE001 - that is the assertion
            pytest.fail(f"{label}: raised {type(exc).__name__}: {exc}")


def test_a_list_and_a_tuple_are_not_the_same_value() -> None:
    """Stated as its own case because it is a judgement, not an accident.

    ``[1.0, 2.0] == (1.0, 2.0)`` is ``False`` in Python, and this function does
    not second-guess that: a caller who wrote both spellings wrote two things,
    and being told so is better than one of them being chosen silently.
    """
    assert values_differ([1.0, 2.0], (1.0, 2.0))
