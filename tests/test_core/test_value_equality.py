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

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.value_equality import _defines_own_truth, values_differ


class _RaisingLength:
    """A value whose ``__len__`` fails, which a user object legitimately may.

    Found in self-review before round 7: the length step caught only
    ``TypeError``, so this propagated out of a function whose whole job is to
    answer a question safely.
    """

    def __len__(self) -> int:
        raise ValueError("len failed")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RaisingLength)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return "_RaisingLength()"


class _UnbooleanComparison:
    """A comparison result whose truth value fails.

    Round 7: the truth step caught only ``ValueError`` and ``TypeError``, the
    two an array raises, so a comparison object failing for a reason of its own
    escaped a function declared not to raise. The comparison result is supplied
    by the caller's value too, not only by numpy.
    """

    def __bool__(self) -> bool:
        raise RuntimeError("truth failed")


class _UnbooleanEquality:
    """A value whose ``__eq__`` returns something that cannot be a boolean."""

    def __eq__(self, other: object) -> Any:
        return _UnbooleanComparison()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return "_UnbooleanEquality()"


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
    ("a length that raises", _RaisingLength(), _RaisingLength(), False),
    (
        "a comparison whose truth value raises",
        _UnbooleanEquality(),
        _UnbooleanEquality(),
        False,
    ),
    (
        "frames whose comparison iterates over labels, equal",
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        False,
    ),
    (
        "frames whose comparison iterates over labels, different",
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        pd.DataFrame({"a": [9, 9], "b": [9, 9]}),
        True,
    ),
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


def test_iterating_a_comparison_does_not_always_yield_the_comparison() -> None:
    """Stated separately because it is the reduction's bound, not one case.

    Comparing two DataFrames yields a DataFrame, and iterating that yields its
    **column labels** -- strings, all truthy -- so requiring every element to be
    true called two different frames equal. Measured in self-review before round
    7. The reduction now refuses to trust string elements and falls through to
    the printed forms, which is weaker but not wrong.
    """
    left = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    right = pd.DataFrame({"a": [9, 9], "b": [9, 9]})
    assert list(left == right) == ["a", "b"], (
        "the premise of this test is that iteration yields labels; if pandas "
        "changed that, the guard needs re-reading rather than the test editing"
    )
    assert values_differ(left, right)


def test_a_list_and_a_tuple_are_not_the_same_value() -> None:
    """Stated as its own case because it is a judgement, not an accident.

    ``[1.0, 2.0] == (1.0, 2.0)`` is ``False`` in Python, and this function does
    not second-guess that: a caller who wrote both spellings wrote two things,
    and being told so is better than one of them being chosen silently.
    """
    assert values_differ([1.0, 2.0], (1.0, 2.0))


class _RaisingEverything:
    """A value with no usable equality, length, or printed form.

    The floor of the function's domain. Rounds 5, 6 and this self-review each
    found one more value shape that got past the previous enumeration, so what
    is asserted below is not another shape but the property that ends that
    series: it does not raise, whatever it is handed.
    """

    def __len__(self) -> int:
        raise RuntimeError("len failed")

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("equality failed")

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        raise RuntimeError("repr failed")


def test_a_value_is_the_same_as_itself_whatever_it_is() -> None:
    """Identity, checked first, and it is the case the callers make most often.

    A parameter written under one spelling is compared with **itself**: the
    facade compares each value in a group against the group's first, and a group
    of one is that value twice. Before identity was checked first, that went the
    long way round and could raise.
    """
    value = _RaisingEverything()
    assert values_differ(value, value) is False


def test_it_does_not_raise_on_a_value_nothing_can_analyse() -> None:
    """Total by construction, not by having enumerated enough types.

    Two distinct objects whose equality, length and printed form all fail. The
    answer is "the same" rather than "different" on purpose: the refusal this
    feeds exists to catch a parameter written twice, and refusing a value
    nothing can analyse would block a legitimate call to prevent an ambiguity
    that may not be there.
    """
    assert values_differ(_RaisingEverything(), _RaisingEverything()) is False


class _UnbooleanIterableComparison:
    """A comparison result that cannot be a boolean but *is* iterable.

    Found in self-review before round 8. Round 7 widened the truth step to catch
    every exception, which moved this shape out of the printed-form fallback and
    into the elementwise reduction -- where iterating yielded objects that are
    truthy by default, so two different values read as equal. The DataFrame case
    round 6 found, reached through a different door.
    """

    def __iter__(self) -> Any:
        return iter([object(), object()])

    def __bool__(self) -> bool:
        raise RuntimeError("truth failed")


class _ComparesToJunk:
    """Two of these are different, and only the comparison could say so."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __eq__(self, other: object) -> Any:
        return _UnbooleanIterableComparison()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"_ComparesToJunk({self.tag!r})"


#: One value per internal step, each failing at that step and no earlier, with
#: what the function must still be able to say about it.
#:
#: This is the axis, not another case. Rounds 5, 6 and 7 each added the value
#: shape that had just been found, and each time the next shape got through, so
#: what is pinned here is that **every step has somewhere to fall**: a value that
#: defeats one step is answered by a later one, and the last step is a decision
#: rather than a computation.
STEP_FAILURES: list[tuple[str, object, object, bool]] = [
    ("length raises", _RaisingLength(), _RaisingLength(), False),
    ("equality raises", _RaisingEquality(), _RaisingEquality(), False),
    ("the truth value raises", _UnbooleanEquality(), _UnbooleanEquality(), False),
    (
        "iterating the comparison yields junk, and the values differ",
        _ComparesToJunk("a"),
        _ComparesToJunk("b"),
        True,
    ),
    (
        "iterating the comparison yields junk, and the values match",
        _ComparesToJunk("a"),
        _ComparesToJunk("a"),
        False,
    ),
]


@pytest.mark.parametrize(("label", "first", "second", "differ"), STEP_FAILURES)
def test_every_step_has_somewhere_to_fall(
    label: str, first: object, second: object, differ: bool
) -> None:
    assert values_differ(first, second) is differ, label
    assert values_differ(second, first) is differ, f"{label} (reversed)"


def test_the_reduction_only_trusts_elements_that_can_state_a_truth() -> None:
    """The bound the elementwise step rests on, asserted directly.

    ``bool`` and the library scalars define ``__bool__``; a string, a list and a
    bare object do not -- their truthiness comes from length or from the default,
    neither of which answers "are these equal". Two different routes produced an
    iteration that is not the comparison (a DataFrame yielding column labels; a
    comparison object yielding anything), and this property is what excludes
    both without naming either.
    """
    assert _defines_own_truth(True)
    assert _defines_own_truth(np.True_)
    assert _defines_own_truth(np.float64(0.5))
    assert not _defines_own_truth("a")
    assert not _defines_own_truth(object())
    assert not _defines_own_truth([1])


def test_asking_whether_an_element_can_state_a_truth_never_runs_its_code() -> None:
    """The check reads the type, so a hostile ``__bool__`` is never invoked.

    ``hasattr`` would invoke it. An element whose ``__bool__`` is a property
    that raises therefore escaped the whole function, which is round 7's
    finding one step further along the same path; found before round 8.
    """

    class HostileLookup:
        @property
        def __bool__(self) -> object:
            raise RuntimeError("truth lookup failed")

    element = HostileLookup()
    assert _defines_own_truth(element) is True
    with pytest.raises(RuntimeError):
        hasattr(element, "__bool__")  # what the check used to do

    class Comparison:
        def __bool__(self) -> bool:
            raise ValueError("ambiguous")

        def __iter__(self) -> object:
            return iter([HostileLookup(), HostileLookup()])

    class Value:
        __hash__ = None  # type: ignore[assignment]

        def __eq__(self, other: object) -> object:
            return Comparison()

    assert values_differ(Value(), Value()) is True


# ---------------------------------------------------------------------------
# The bound, quantified over the population instead of over a table
# ---------------------------------------------------------------------------
# Rounds 5, 6 and 7 each shipped a declaration ("this does not raise") verified
# by hand-written cases, and each time the next round found the shape the table
# had not thought of. The rounds 6-7 monitor named the repair: quantify the
# declaration over its whole population, which is what `defect-classes.md` asks
# for. The population here is not "values" -- that is open -- but **the ways a
# caller's value can defeat each step**, and those are the three dunders this
# function reads on the value itself. The fourth thing it touches, `__bool__`,
# belongs to whatever `__eq__` returned rather than to the value, so it is
# covered by the `__eq__` return variants -- including the one whose elements
# are hostile to being *asked* whether they define it.

#: How each dunder can behave. `absent` means the type does not define it at
#: all, which is a different path from defining one that fails.
_BEHAVIOURS: dict[str, tuple[str, ...]] = {
    "__eq__": (
        "normal",
        "raises",
        "returns_unbooleanable",
        "returns_junk_iterable",
        "returns_hostile_elements",
    ),
    "__len__": ("absent", "normal", "raises"),
    "__repr__": ("normal", "raises"),
}


class _Unbooleanable:
    def __bool__(self) -> bool:
        raise RuntimeError("truth failed")


class _JunkIterable:
    def __bool__(self) -> bool:
        raise RuntimeError("truth failed")

    def __iter__(self) -> Any:
        return iter([object(), object()])


class _HostileElement:
    """An element whose ``__bool__`` raises when it is merely *looked up*.

    ``hasattr(element, "__bool__")`` invokes the property, so the step that
    decides whether an element can state a truth of its own raised through a
    function declared not to raise. Found before round 8; the check now reads
    the type's dictionaries instead.
    """

    @property
    def __bool__(self) -> Any:
        raise RuntimeError("truth lookup failed")


class _HostileElementsIterable:
    def __bool__(self) -> bool:
        raise RuntimeError("truth failed")

    def __iter__(self) -> Any:
        return iter([_HostileElement(), _HostileElement()])


def _make_awkward(eq: str, length: str, printed: str, tag: str) -> object:
    """Build a value whose dunders behave as named."""
    namespace: dict[str, Any] = {"tag": tag}

    if eq == "normal":
        namespace["__eq__"] = lambda self, other: (
            getattr(other, "tag", None) == self.tag
        )
    elif eq == "raises":
        namespace["__eq__"] = lambda self, other: (_ for _ in ()).throw(
            RuntimeError("equality failed")
        )
    elif eq == "returns_unbooleanable":
        namespace["__eq__"] = lambda self, other: _Unbooleanable()
    elif eq == "returns_junk_iterable":
        namespace["__eq__"] = lambda self, other: _JunkIterable()
    else:
        namespace["__eq__"] = lambda self, other: _HostileElementsIterable()
    namespace["__hash__"] = None

    if length == "normal":
        namespace["__len__"] = lambda self: 2
    elif length == "raises":
        namespace["__len__"] = lambda self: (_ for _ in ()).throw(
            RuntimeError("len failed")
        )

    if printed == "normal":
        namespace["__repr__"] = lambda self: f"Awkward({self.tag!r})"
    else:
        namespace["__repr__"] = lambda self: (_ for _ in ()).throw(
            RuntimeError("repr failed")
        )

    return type("Awkward", (), namespace)()


AWKWARD_COMBINATIONS: list[tuple[str, str, str]] = [
    (eq, length, printed)
    for eq in _BEHAVIOURS["__eq__"]
    for length in _BEHAVIOURS["__len__"]
    for printed in _BEHAVIOURS["__repr__"]
]


@pytest.mark.parametrize(("eq", "length", "printed"), AWKWARD_COMBINATIONS)
def test_the_no_raise_bound_holds_over_the_whole_cross_product(
    eq: str, length: str, printed: str
) -> None:
    """``values_differ`` answers, whatever the caller's value does to it.

    Every combination of the ways the three dunders this function touches can
    behave, including the ones no round has produced. A generated population
    rather than a table is the point: a table is only ever as complete as the
    last thing someone thought of, which is how rounds 5, 6 and 7 each found the
    previous round's gap.
    """
    first = _make_awkward(eq, length, printed, "a")
    second = _make_awkward(eq, length, printed, "b")
    same = _make_awkward(eq, length, printed, "a")

    for left, right in ((first, second), (first, same), (first, first)):
        for a, b in ((left, right), (right, left)):
            result = values_differ(a, b)
            assert isinstance(result, bool), (
                f"{eq}/{length}/{printed} answered {result!r}, not a bool"
            )


def test_the_cross_product_covers_every_declared_behaviour() -> None:
    """The population is derived, so a new behaviour must be declared to exist.

    Without this, shrinking ``_BEHAVIOURS`` would quietly shrink the guarantee
    while every generated case still passed.
    """
    declared = {name: len(behaviours) for name, behaviours in _BEHAVIOURS.items()}
    assert declared == {"__eq__": 5, "__len__": 3, "__repr__": 2}, declared
    assert len(AWKWARD_COMBINATIONS) == 5 * 3 * 2, AWKWARD_COMBINATIONS

    # Each `__eq__` variant must reach a different step, or the axis is wider
    # than the function is. This is the half the count does not check: a
    # behaviour added and then routed to an existing branch by `_make_awkward`
    # would grow the population without growing the coverage.
    assert len(set(_BEHAVIOURS["__eq__"])) == 5, _BEHAVIOURS["__eq__"]
    outcomes = {
        eq: type(_make_awkward(eq, "absent", "normal", "a").__eq__(object()))
        for eq in _BEHAVIOURS["__eq__"]
        if eq != "raises"
    }
    assert len(set(outcomes.values())) == len(outcomes), outcomes


def test_identical_values_are_never_reported_as_differing() -> None:
    """The half of the bound that a no-raise assertion does not cover.

    Answering is not enough: answering "different" for one object compared with
    itself would refuse a legitimate call for every awkward value at once.
    """
    for eq, length, printed in AWKWARD_COMBINATIONS:
        value = _make_awkward(eq, length, printed, "a")
        assert values_differ(value, value) is False, f"{eq}/{length}/{printed}"
