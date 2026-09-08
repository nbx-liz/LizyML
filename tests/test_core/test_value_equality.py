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
  declared exception-safe fallback did not cover the comparison itself;
- round 7 — a comparison whose ``__bool__`` failed for a reason of its own
  escaped a handler that caught only the two exceptions an array raises;
- round 8 — a ``DataFrame`` with integer column labels was called equal to a
  different one, by the second guard written to keep the elementwise step from
  reading labels. The step is gone; the cases it used to serve are decided by
  the printed forms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.value_equality import values_differ


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


class _HostileText(str):
    """A ``str`` subclass whose comparison answers whatever it likes.

    ``repr`` is allowed to return a subclass of ``str``, so comparing the two
    printed forms with ``!=`` handed the decision back to the caller's object at
    the step that exists to escape it (review round 9).
    """

    def __ne__(self, other: object) -> Any:
        return [False]

    __hash__ = str.__hash__


class _PrintsHostileText:
    """A value that cannot be compared, and prints through ``_HostileText``."""

    def __eq__(self, other: object) -> bool:
        raise ValueError("no comparison")

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return _HostileText("the same text")


class _RaisingEquality:
    """A value whose ``__eq__`` fails, which a user object legitimately may."""

    def __eq__(self, other: object) -> bool:
        raise ValueError("equality failed")

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return "_RaisingEquality()"


def _long_array(differing_at: int | None) -> Any:
    """An array long enough that ``repr`` summarises its middle away.

    The printed forms are the last thing this function has, so past numpy's
    summarisation threshold two different arrays print alike. That is the
    declared cost of deciding array-likes by ``repr``, and this builds the pair
    that reaches it.
    """
    array = np.zeros(2000)
    if differing_at is not None:
        array[differing_at] = 1.0
    return array


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
    ("a list and an equal tuple", [1.0, 2.0], (1.0, 2.0), False),
    ("an array and an equal tuple", np.array([1.0, 2.0]), (1.0, 2.0), False),
    ("an array and an equal list", np.array([1.0, 2.0]), [1.0, 2.0], False),
    ("a series and an equal tuple", pd.Series([1.0, 2.0]), (1.0, 2.0), False),
    ("a tuple and a reordered list", (1.0, 2.0), [2.0, 1.0], True),
    ("a tuple and a list of different length", (1.0,), [1.0, 2.0], True),
    ("a string and the tuple of its characters", "ab", ("a", "b"), True),
    ("a comma form and its list", "1,2", [1, 2], False),
    ("a comma form and its list under another dtype", "1,2", [1.0, 2.0], False),
    ("a joined float list and its int list", "1.0,2.0", [1, 2], False),
    ("a comma form and its tuple", "1,2", (1, 2), False),
    ("a comma form and a different list", "1,2", [5, 6], True),
    ("a comma form and a longer list", "1,2", [1, 2, 3], True),
    ("a bare word and its one-element list", "gbdt", ["gbdt"], False),
    ("a bare word and a two-element list", "auc", ["auc", "logloss"], True),
    ("a comma form of words and its list", "a,b", ["a", "b"], False),
    ("a nested comma form and its list", "[0,1],[2]", [[0, 1], [2]], True),
    ("a comma form and its array", "1.0,2.0", np.array([1.0, 2.0]), False),
    ("a comma form and an int array", "1,2", np.array([1, 2]), False),
    ("a comma form and its series", "1.0,2.0", pd.Series([1.0, 2.0]), False),
    ("a scalar and its text", 0.5, "0.5", False),
    ("an int and its text", 100, "100", False),
    ("a scalar and a different text", 0.5, "0.7", True),
    ("a scalar and a two-element comma form", 0.5, "0.5,0.5", True),
    # Excluded on purpose, not overlooked: a set has no order and every
    # sequence parameter here is positional. See `_wire_elements`.
    ("a set and the list it prints like", {1.0, 2.0}, [1.0, 2.0], True),
    # `_param_dict_to_str` skips a `None` entirely -- it means "not sent",
    # which is not the string LightGBM would read as a value.
    ("none and the text of none", None, "None", True),
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
    # Both of these were decided by the printed forms until round 11, and both
    # were wrong there: the first refused a call naming one value twice, and
    # the second called two different arrays the same. Converting to plain
    # Python answers both.
    (
        "equal numbers under different dtypes",
        np.array([1, 2]),
        np.array([1.0, 2.0]),
        False,
    ),
    (
        "a list and an equal array",
        [1.0, 2.0],
        np.array([1.0, 2.0]),
        False,
    ),
    (
        "arrays whose printed forms summarise the difference away",
        _long_array(differing_at=None),
        _long_array(differing_at=1000),
        True,
    ),
    # What is left for the printed forms: a value with no faithful conversion
    # to plain Python. This is the cost, and it is here so that something
    # reaches it.
    (
        "frames whose printed forms summarise the difference away",
        pd.DataFrame({"x": _long_array(differing_at=None)}),
        pd.DataFrame({"x": _long_array(differing_at=1000)}),
        False,
    ),
    (
        "printed forms whose own comparison is hostile",
        _PrintsHostileText(),
        _PrintsHostileText(),
        False,
    ),
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
    """Stated separately because it is why there is no elementwise step.

    Comparing two DataFrames yields a DataFrame, and iterating that yields its
    **column labels**, not its cells. Two guards were written to exclude that --
    one on the labels being strings (round 6), one on the labels defining
    ``__bool__`` (before round 8) -- and each was refuted by the next label type.
    The function no longer reads a comparison result's contents at all.
    """
    left = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    right = pd.DataFrame({"a": [9, 9], "b": [9, 9]})
    assert list(left == right) == ["a", "b"], (
        "the premise of this test is that iteration yields labels; if pandas "
        "changed that, the reasoning needs re-reading rather than the test "
        "editing"
    )
    assert values_differ(left, right)


def test_the_summarisation_cases_actually_reach_summarisation() -> None:
    """The premise of the two cases above, so neither passes for a wrong reason.

    Both pairs print identically. The arrays are nonetheless reported as
    differing, because they convert to plain Python and are compared there; the
    frames are reported as the same, because they do not convert and the printed
    forms are all that is left. If either premise stopped holding, the pair would
    print differently and the expectation would be edited to match instead of the
    reasoning being re-read.
    """
    same_printing = _long_array(differing_at=None)
    different_values = _long_array(differing_at=1000)

    assert repr(same_printing) == repr(different_values), (
        "the premise is that repr summarises the differing element away; "
        "if numpy changed that, these cases need re-reading, not re-writing"
    )
    assert not np.array_equal(same_printing, different_values), (
        "the arrays must actually differ, or the cases assert nothing"
    )

    left = pd.DataFrame({"x": same_printing})
    right = pd.DataFrame({"x": different_values})
    assert repr(left) == repr(right), "pandas no longer summarises this frame"
    assert not left.equals(right), "the frames must actually differ"
    assert not hasattr(left, "tolist"), (
        "a DataFrame gaining `tolist` would move it out of the printed-form "
        "fallback, which is the only case left that reaches that cost"
    )


def test_every_type_the_serialiser_joins_is_covered() -> None:
    """The claim quantified over the type set, not over the one type reached.

    The round-13 fix said it closed the "same value, another written form"
    class. It closed **one of the four types** its own docstring named: it asked
    `isinstance(sequence, list)`, and an ndarray and a Series are not
    `collections.abc.Sequence`, and a scalar is not a sequence at all. The
    rounds 12-13 monitor found that by execution after the claim was made
    (H-0094 decision 9).

    So the test is written the way the claim should have been checked in the
    first place: **LightGBM's own serialiser is the oracle**, and every form is
    put through it. Where two values produce the same wire string, they are one
    value and `values_differ` must say so.

    `set` is the one exclusion, asserted here as an exclusion rather than left
    to look like a gap: LightGBM does join a set, but a set has no order and
    `feature_contri` reads element *i* as feature *i*, so two sets printing
    alike did so by hash accident.
    """
    basic = pytest.importorskip("lightgbm.basic")

    # A Series is deliberately absent: LightGBM does not join it, it *refuses*
    # it. Asserted rather than assumed, because "which types the serialiser
    # accepts" is exactly the external fact this test exists to execute.
    with pytest.raises(TypeError, match="Unknown type of parameter"):
        basic._param_dict_to_str({"p": pd.Series([1.0, 2.0])})

    covered = [
        [1.0, 2.0],
        (1.0, 2.0),
        np.array([1.0, 2.0]),
        np.array([1, 2]),
        "1.0,2.0",
    ]
    for value in covered:
        for other in covered:
            same_wire = basic._param_dict_to_str(
                {"p": value}
            ) == basic._param_dict_to_str({"p": other})
            if not same_wire:
                # The wire form is not canonical -- `[1, 2]` joins to "1,2" and
                # `[1.0, 2.0]` to "1.0,2.0" -- so equal wires are sufficient for
                # sameness, not necessary. Only the sufficient direction is
                # asserted; the elementwise comparison covers the rest, and the
                # case table above pins it.
                continue
            assert not values_differ(value, other), (
                f"{value!r} and {other!r} reach LightGBM as the identical "
                "string and were reported as two values"
            )

    scalar_forms = [0.5, "0.5"]
    assert basic._param_dict_to_str({"p": 0.5}) == basic._param_dict_to_str(
        {"p": "0.5"}
    )
    assert not values_differ(*scalar_forms)

    a_set = {1.0, 2.0}
    assert basic._param_dict_to_str({"p": a_set}), "LightGBM no longer joins a set"
    assert values_differ(a_set, [1.0, 2.0]), (
        "a set is excluded on purpose; if this is ever changed, the ordering "
        "question has to be answered first"
    )


def test_the_comma_form_is_lightgbms_own_serialisation() -> None:
    """The authority for treating a text and a sequence as one value.

    Read rather than assumed: LightGBM writes every sequence parameter as
    ``",".join(...)`` regardless of its name, so the two forms are one value on
    the wire. Executing the serialiser is what makes that a fact here instead
    of a claim about a library (H-0094 decision 9, review round 13).
    """
    basic = pytest.importorskip("lightgbm.basic")

    assert basic._param_dict_to_str({"feature_contri": [1, 2]}) == "feature_contri=1,2"
    assert basic._param_dict_to_str({"feature_contri": (1, 2)}) == "feature_contri=1,2"
    assert basic._param_dict_to_str({"feature_contri": "1,2"}) == "feature_contri=1,2"
    # And the reason the comparison is elementwise rather than textual: the
    # wire form is not canonical, so two equal values join differently.
    assert basic._param_dict_to_str({"x": [1.0, 2.0]}) != basic._param_dict_to_str(
        {"x": [1, 2]}
    )
    assert not values_differ([1.0, 2.0], [1, 2])


@pytest.mark.parametrize(
    "parameter,spellings",
    [
        (
            "feature_contri",
            (
                [1.0, 2.0],
                (1.0, 2.0),
                np.array([1.0, 2.0]),
                np.array([1, 2]),
                "1,2",
            ),
        ),
        ("monotone_constraints", ([1, 0], (1, 0), np.array([1, 0]), "1,0")),
    ],
)
def test_containers_the_estimator_cannot_tell_apart_are_one_value(
    parameter: str, spellings: tuple[object, ...]
) -> None:
    """The judgement, and the execution that decides it.

    An earlier version of this file asserted the opposite -- that a list and an
    equal tuple are two values -- reasoning from Python, where
    ``[1.0, 2.0] == (1.0, 2.0)`` is ``False``. That reasoned about the wrong
    question. Nothing is chosen silently when a caller writes both spellings,
    because there is nothing to choose between: LightGBM trains the
    **byte-identical booster** from either, so a refusal here refuses a call
    that meant one thing (H-0094 decision 8, review round 12).

    This asserts that by training, not by citing it. The models are compared
    with the ``[data:`` line dropped, which carries the dataset's own name.
    """
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a", "b"])
    target = (frame["a"] + rng.normal(scale=0.2, size=120) > 0).astype(int)

    trained: list[str] = []
    for value in spellings:
        booster = lgb.train(
            {
                "objective": "binary",
                "verbose": -1,
                "seed": 1,
                "num_threads": 1,
                "deterministic": True,
                parameter: value,
            },
            lgb.Dataset(frame, target),
            num_boost_round=5,
        )
        text = booster.model_to_string()
        trained.append(
            "\n".join(
                line for line in text.splitlines() if not line.startswith("[data:")
            )
        )

    assert len(set(trained)) == 1, (
        f"LightGBM distinguishes the spellings of {parameter}; the equality "
        "below would then be wrong rather than merely unnecessary"
    )
    for other in spellings[1:]:
        assert not values_differ(spellings[0], other)
        assert not values_differ(other, spellings[0])


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


@pytest.mark.parametrize(
    ("label", "frame"),
    [
        ("string labels", pd.DataFrame({"x": [10, 20]})),
        # Integer labels are truthy, so the guard that excluded string labels
        # admitted these and two unequal frames read as equal (review round 8).
        ("integer labels", pd.DataFrame({1: [10, 20]})),
        ("a falsy integer label", pd.DataFrame({0: [10, 20]})),
    ],
)
def test_two_unequal_frames_differ_whatever_their_labels_are(
    label: str, frame: pd.DataFrame
) -> None:
    """No property of the labels can decide this, so nothing reads them.

    A ``DataFrame`` comparison yields its **column labels**, not its cells.
    Rounds 6 and 8 each found one label type that a guard on the labels let
    through; the function no longer inspects a comparison result's contents at
    all, so there is no third label type to find.
    """
    other = frame + 20

    assert values_differ(frame, other) is True, label
    assert values_differ(other, frame) is True, f"{label} (reversed)"
    assert values_differ(frame, frame.copy()) is False, f"{label} (equal copy)"


def test_a_base_exception_from_a_caller_value_is_not_swallowed() -> None:
    """The bound is "no ``Exception``", and that word is load-bearing.

    Catching ``BaseException`` would make a comparison that hangs
    uninterruptible, which is worse than the ambiguity the bound prevents. So
    the declaration is limited and this pins the limit.
    """

    class Value:
        __hash__ = None  # type: ignore[assignment]

        def __eq__(self, other: object) -> bool:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        values_differ(Value(), Value())


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

    Kept as a behaviour the caller's value can exhibit. It defeated a guard that
    asked each element with ``hasattr``, which invokes the property; that guard
    is gone along with the whole elementwise step, and this is one of the shapes
    that has nothing left to defeat.
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

    # Each `__eq__` variant must produce a distinct kind of comparison result,
    # or the axis is wider than the values it generates. This is the half the
    # count does not check: a behaviour added and then routed to an existing
    # branch by `_make_awkward` would grow the population without growing the
    # coverage. Distinct *results*, not distinct steps -- since round 8 several
    # of them are decided by the same printed-form fallback, which is the point
    # of that change rather than a gap in this one.
    assert len(set(_BEHAVIOURS["__eq__"])) == 5, _BEHAVIOURS["__eq__"]
    outcomes = {
        eq: type(_make_awkward(eq, "absent", "normal", "a").__eq__(object()))
        for eq in _BEHAVIOURS["__eq__"]
        if eq != "raises"
    }
    assert len(set(outcomes.values())) == len(outcomes), outcomes


# ---------------------------------------------------------------------------
# The cell the cross product above does not reach (round 16)
# ---------------------------------------------------------------------------
# Every case above compares an awkward value against another awkward value, so
# `text` is never a `str` and `_comma_form_matches` returns `None` before it
# touches an element. That is the cell the round-16 finding came out of: a
# **string on one side and a sequence of hostile elements on the other**, which
# is the only way to reach the two expressions that step applies to a caller's
# element. Those are `float(element)` and `str(element)`, and they are the
# population here.

# Round 17 widened it again, and the widening is the lesson: the round-16
# population covered the **element** operand only, so a `str` subclass whose
# `split` raises defeated the step from the side nothing had enumerated. The
# population is therefore both operands -- every expression in the step that
# touches something a caller owns, which is the table in the function's own
# docstring.

_TEXT_BEHAVIOURS: dict[str, tuple[str, ...]] = {
    "split": ("normal", "raises"),
    "strip": ("normal", "raises"),
}

_ELEMENT_BEHAVIOURS: dict[str, tuple[str, ...]] = {
    "__float__": ("absent", "normal", "raises"),
    # `returns_subclass` is its own behaviour because `str(element)` handing
    # back a `str` subclass puts the *derived* value under a caller override
    # too. Measured: an element whose `__str__` returns one does exactly that.
    "__str__": ("normal", "raises", "returns_subclass"),
}


def _make_awkward_text(split: str, strip: str) -> str:
    """A ``str`` subclass whose own methods behave as named."""
    namespace: dict[str, Any] = {}

    if split == "raises":
        namespace["split"] = lambda self, *a, **k: (_ for _ in ()).throw(
            RuntimeError("split failed")
        )
    if strip == "raises":
        namespace["strip"] = lambda self, *a, **k: (_ for _ in ()).throw(
            RuntimeError("strip failed")
        )

    return type("AwkwardText", (str,), namespace)("0.5")


def _make_awkward_element(to_float: str, printed: str) -> object:
    """An element whose conversions behave as named."""
    namespace: dict[str, Any] = {}

    if to_float == "normal":
        namespace["__float__"] = lambda self: 0.5
    elif to_float == "raises":
        namespace["__float__"] = lambda self: (_ for _ in ()).throw(
            RuntimeError("float failed")
        )

    if printed == "normal":
        namespace["__str__"] = lambda self: "0.5"
    elif printed == "returns_subclass":
        namespace["__str__"] = lambda self: _make_awkward_text("raises", "raises")
    else:
        namespace["__str__"] = lambda self: (_ for _ in ()).throw(
            RuntimeError("str failed")
        )

    return type("AwkwardElement", (), namespace)()


ELEMENT_COMBINATIONS: list[tuple[str, str, str, str]] = [
    (split, strip, to_float, printed)
    for split in _TEXT_BEHAVIOURS["split"]
    for strip in _TEXT_BEHAVIOURS["strip"]
    for to_float in _ELEMENT_BEHAVIOURS["__float__"]
    for printed in _ELEMENT_BEHAVIOURS["__str__"]
]


@pytest.mark.parametrize(
    ("split", "strip", "to_float", "printed"), ELEMENT_COMBINATIONS
)
def test_the_bound_holds_with_text_on_one_side_and_a_sequence_on_the_other(
    split: str, strip: str, to_float: str, printed: str
) -> None:
    """The comma-form step answers, whatever **either** operand does to it.

    Round 16: a ``float`` subclass whose ``__float__`` raises ``RuntimeError``
    came straight out of ``values_differ`` -- and out of ``fit()`` -- because
    the ``except`` named ``TypeError`` and ``ValueError``, and ``str`` one line
    down was unguarded entirely. Round 17: the same thing from the text side,
    because a ``str`` subclass is still a ``str`` and its overrides run on
    attribute access.
    """
    element = _make_awkward_element(to_float, printed)
    hostile_text = _make_awkward_text(split, strip)
    label = f"{split}/{strip}/{to_float}/{printed}"

    for sequence in ([element], (element,), [element, element]):
        for text in (hostile_text, type(hostile_text)("0.5,0.5"), "0.5"):
            for a, b in ((text, sequence), (sequence, text)):
                result = values_differ(a, b)
                assert isinstance(result, bool), (
                    f"{label} answered {result!r}, not a bool"
                )


@pytest.mark.parametrize(
    ("split", "strip"),
    [
        (split, strip)
        for split in _TEXT_BEHAVIOURS["split"]
        for strip in _TEXT_BEHAVIOURS["strip"]
    ],
)
def test_a_hostile_text_is_still_compared_rather_than_merely_survived(
    split: str, strip: str
) -> None:
    """Not raising is half the claim; the other half is admitting the pair.

    A blanket ``try`` around the subclass call would satisfy a no-raise
    assertion and still refuse ``learning_rate="0.5"`` beside ``eta=0.5`` --
    one parameter, one value, written twice. Calling the base method unbound is
    what makes the step compare rather than decline.
    """
    text = _make_awkward_text(split, strip)

    assert values_differ(text, [0.5]) is False
    assert values_differ([0.5], text) is False
    assert values_differ(text, [0.25]) is True


def test_a_float_subclass_is_reached_as_an_element_not_as_a_container() -> None:
    """The exact shape review round 16 reproduced, kept as itself.

    The generated elements above are plain objects. A ``float`` subclass is a
    different route into the same expression -- it is already a number, so
    nothing upstream converts or rejects it -- and it is the one a caller
    actually wrote.
    """

    class Rate(float):
        def __float__(self) -> float:
            raise RuntimeError("conversion unavailable")

    assert values_differ("0.5", [Rate(0.5)]) is False
    assert values_differ([Rate(0.5)], "0.5") is False
    assert values_differ("0.25", [Rate(0.5)]) is True


def test_a_str_subclass_is_reached_as_the_text_operand() -> None:
    """The exact shape review round 17 reproduced, kept as itself.

    The generated texts above are built here. This is the one a caller writes:
    a ``str`` subclass reaching ``fit(params=...)``, which is a ``str`` to
    every check upstream and runs its own ``split`` at the comparison.
    """

    class Text(str):
        def split(self, *args: Any, **kwargs: Any) -> list[str]:
            raise RuntimeError("split unavailable")

    assert values_differ(Text("0.5"), 0.5) is False
    assert values_differ(0.5, Text("0.5")) is False
    assert values_differ(Text("0.5"), [0.5]) is False
    assert values_differ(Text("0.25"), 0.5) is True


def test_the_element_cross_product_covers_every_declared_behaviour() -> None:
    """Derived like the population above, and pinned for the same reason."""
    declared = {
        name: len(values)
        for name, values in {**_TEXT_BEHAVIOURS, **_ELEMENT_BEHAVIOURS}.items()
    }
    assert declared == {
        "split": 2,
        "strip": 2,
        "__float__": 3,
        "__str__": 3,
    }, declared
    assert len(ELEMENT_COMBINATIONS) == 2 * 2 * 3 * 3, len(ELEMENT_COMBINATIONS)

    # `absent` and `raises` must be different paths, not two spellings of one.
    assert not hasattr(_make_awkward_element("absent", "normal"), "__float__")
    with pytest.raises(RuntimeError, match="float failed"):
        float(_make_awkward_element("raises", "normal"))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="str failed"):
        str(_make_awkward_element("normal", "raises"))

    # Each text behaviour must actually defeat its own method, or the axis is
    # wider than the values it generates.
    with pytest.raises(RuntimeError, match="split failed"):
        _make_awkward_text("raises", "normal").split(",")
    with pytest.raises(RuntimeError, match="strip failed"):
        _make_awkward_text("normal", "raises").strip()

    # And `returns_subclass` must put a caller override on the *derived* value,
    # which is a different escape from the element raising.
    printed = str(_make_awkward_element("absent", "returns_subclass"))
    assert type(printed) is not str, type(printed)
    with pytest.raises(RuntimeError, match="strip failed"):
        printed.strip()


def test_identical_values_are_never_reported_as_differing() -> None:
    """The half of the bound that a no-raise assertion does not cover.

    Answering is not enough: answering "different" for one object compared with
    itself would refuse a legitimate call for every awkward value at once.
    """
    for eq, length, printed in AWKWARD_COMBINATIONS:
        value = _make_awkward(eq, length, printed, "a")
        assert values_differ(value, value) is False, f"{eq}/{length}/{printed}"
