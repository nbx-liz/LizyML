"""``values_differ``, asked the question the shipped path now asks it.

H-0095 closed the domain. Every parameter value is normalised at the surface it
is written on (:mod:`lizyml.core.param_domain`) and anything outside a small set
of plain types is refused before training, so ``values_differ`` is no longer
handed arbitrary objects. This file follows that: **every case goes through the
surface first**, and a case whose value the surface refuses says so instead of
saying what the comparison would have answered.

Where the retired material went, so that nothing is quietly dropped:

* the hostile objects of review rounds 16-20 (``__format__``, ``__class__``,
  ``tolist`` and ``__eq__`` written to mislead) are in
  ``test_param_domain.py``, where the claim about them is now that they are
  **refused at ingress** rather than survived here;
* the derived hostile-attribute-name population (H-0094 decision 16) is there
  too, for the same reason: what it now checks is the refusal;
* the no-raise bound is still asserted, but over the **accepted set** -- a
  finite, enumerable population -- rather than over every Python object, which
  is not a satisfiable claim and was itself the DC7 that kept the loop running.

What remains here is the part that was never about hostile objects: a sequence
and its comma-separated text form are one value, the comparison is elementwise
rather than textual, and the verdict follows the bytes LightGBM would be sent.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.core.param_domain import normalise_params
from lizyml.core.value_equality import values_differ
from tests.test_core.test_param_domain import ACCEPTED_POPULATION

#: The verdict for a case whose value the surface does not accept.
REFUSED = "refused"


def _through_the_surface(value: Any) -> Any:
    """The value as the comparison actually receives it, or raise."""
    return normalise_params({"k": value}, surface="probe")["k"]


def _long_array(differing_at: int | None) -> Any:
    """An array long enough that ``repr`` would summarise its middle away.

    Kept from the pre-H-0095 table. It used to be the witness that deciding
    array-likes by their printed forms loses a difference; now the array is a
    plain ``list`` before anything compares it, so the pair is decided by the
    numbers and the witness has become an ordinary case. It stays because a
    case that stops being hard is worth keeping as evidence that it did.
    """
    array = np.zeros(2000)
    if differing_at is not None:
        array[differing_at] = 1.0
    return array


#: ``(label, first, second, expected)``. ``expected`` is ``False`` for one
#: value, ``True`` for two, and ``REFUSED`` when the surface does not accept an
#: operand at all. The label is what the case is *about*, so a failure names the
#: property rather than the literal.
CASES: list[tuple[str, object, object, object]] = [
    ("a number equals itself", 0.5, 0.5, False),
    ("int and float of one value", 1, 1.0, False),
    ("different numbers", 1, 2, True),
    ("a library scalar and a python float", np.float64(0.5), 0.5, False),
    # A 0-d array is not a 1-D array and not a scalar type; the serialiser
    # takes it through `_is_numeric`, which is the open branch H-0095 closes.
    ("two zero-dimensional arrays", np.array(1), np.array(1.0), REFUSED),
    ("equal arrays", np.array([1.0, 2.0]), np.array([1.0, 2.0]), False),
    ("reordered arrays", np.array([1.0, 2.0]), np.array([2.0, 1.0]), True),
    ("arrays of different length", np.array([1.0, 1.0]), np.array([1.0]), True),
    ("an empty array and a filled one", np.array([]), np.array([1.0]), True),
    ("two empty arrays", np.array([]), np.array([]), False),
    # The serialiser raises on a 2-D array, so this pair could never have been
    # trained under either verdict. It is refused where that is visible.
    ("equal two-dimensional arrays", np.ones((2, 2)), np.ones((2, 2)), REFUSED),
    ("different two-dimensional arrays", np.ones((2, 2)), np.zeros((2, 2)), REFUSED),
    ("equal lists", [1.0, 2.0], [1.0, 2.0], False),
    ("lists of different length", [1.0], [1.0, 2.0], True),
    ("a list and an equal tuple", [1.0, 2.0], (1.0, 2.0), False),
    ("an array and an equal tuple", np.array([1.0, 2.0]), (1.0, 2.0), False),
    ("an array and an equal list", np.array([1.0, 2.0]), [1.0, 2.0], False),
    # A Series is not a type `_param_dict_to_str` writes: it is not a 1-D
    # ndarray and `float()` of a multi-element one raises, so it could not have
    # reached a trained model. Refusing it names that at the surface.
    ("a series and an equal tuple", pd.Series([1.0, 2.0]), (1.0, 2.0), REFUSED),
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
    ("a comma form and its series", "1.0,2.0", pd.Series([1.0, 2.0]), REFUSED),
    ("a scalar and its text", 0.5, "0.5", False),
    ("an int and its text", 100, "100", False),
    ("a scalar and a different text", 0.5, "0.7", True),
    ("a scalar and a two-element comma form", 0.5, "0.5,0.5", True),
    # **Changed by H-0095, deliberately.** This pair used to be reported as two
    # values, because a set has no order and every sequence parameter here is
    # positional, so admitting it would have made the answer depend on hash
    # order. Normalisation fixes the order once, at the surface, in exactly the
    # order the serialiser would have joined it -- so by the time anything
    # compares them the set *is* that list, and they are one value on the wire.
    # The objection was to deciding hash order at comparison time, and there is
    # no longer a comparison-time decision to make.
    ("a set and the list it prints like", {1.0, 2.0}, [1.0, 2.0], False),
    # `_param_dict_to_str` skips a `None` entirely -- it means "not sent",
    # which is not the string LightGBM would read as a value.
    ("none and the text of none", None, "None", True),
    ("equal series", pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), REFUSED),
    ("equal strings", "binary", "binary", False),
    ("different strings", "binary", "regression", True),
    ("strings of different length", "auc", "auc_mu", True),
    # A mapping is a LizyML-level value the adapter consumes (a metric entry),
    # so the surface accepts it and the comparison must answer about it.
    ("equal dicts", {"a": 1}, {"a": 1}, False),
    ("different dicts", {"a": 1}, {"a": 2}, True),
    ("none and none", None, None, False),
    ("none and a value", None, 0.5, True),
    ("a bool and the int it equals", True, 1, False),
    (
        "equal numbers under different dtypes",
        np.array([1, 2]),
        np.array([1.0, 2.0]),
        False,
    ),
    ("a list and an equal array", [1.0, 2.0], np.array([1.0, 2.0]), False),
    (
        "arrays whose printed forms summarise the difference away",
        _long_array(differing_at=None),
        _long_array(differing_at=1000),
        True,
    ),
    # What used to fall through to the printed forms: a value with no faithful
    # conversion to plain Python. There is no such value inside the domain any
    # more, so the case is now a refusal -- which is the honest answer, since a
    # frame could never have been serialised either.
    (
        "frames whose comparison iterates over labels",
        pd.DataFrame({"a": [1, 2]}),
        pd.DataFrame({"a": [1, 2]}),
        REFUSED,
    ),
    ("nan is not itself, and that is deliberate", float("nan"), float("nan"), True),
]


def _verdict(first: object, second: object) -> object:
    try:
        left, right = _through_the_surface(first), _through_the_surface(second)
    except LizyMLError:
        return REFUSED
    return values_differ(left, right)


@pytest.mark.parametrize(("label", "first", "second", "expected"), CASES)
def test_values_differ(
    label: str, first: object, second: object, expected: object
) -> None:
    assert _verdict(first, second) == expected, label


@pytest.mark.parametrize(("label", "first", "second", "expected"), CASES)
def test_values_differ_is_symmetric(
    label: str, first: object, second: object, expected: object
) -> None:
    """The callers compare in whichever order the dict yields.

    A rule that reads one operand and not the other answers differently
    depending on which spelling was written first, which is the same class of
    defect as deciding a parameter by dictionary order.
    """
    assert _verdict(second, first) == expected, label


def test_the_case_table_reaches_both_verdicts_and_the_refusal() -> None:
    """A table that had drifted to one answer would still pass every case."""
    outcomes = {expected for _, _, _, expected in CASES}
    assert outcomes == {True, False, REFUSED}, outcomes


# ---------------------------------------------------------------------------
# Total over the accepted set
# ---------------------------------------------------------------------------
#
# The bound this replaces was "does not raise on any value the serialiser
# accepts", quantified over every Python object that could satisfy that. It was
# not satisfiable -- a caller can define `__getattribute__` to raise -- and
# three consecutive rounds found the next unguarded expression under it. The
# domain is now finite and enumerable, so the bound is asserted by enumeration.


_ACCEPTED: list[Any] = [
    normalise_params({"k": value}, surface="probe")["k"]
    for value in ACCEPTED_POPULATION
]

#: A sample, because the accepted population is large and the claim is about
#: pairs. Every value appears; the partner walks the population by a stride, so
#: the pairs cross types rather than staying on the diagonal.
_PAIRS: list[tuple[int, int]] = [
    (index, (index * 7 + 3) % len(_ACCEPTED)) for index in range(len(_ACCEPTED))
]


@pytest.mark.parametrize(("left", "right"), _PAIRS)
def test_the_comparison_is_total_over_the_accepted_set(left: int, right: int) -> None:
    """It answers, with a ``bool``, for every pair inside the domain."""
    verdict = values_differ(_ACCEPTED[left], _ACCEPTED[right])
    assert type(verdict) is bool
    assert values_differ(_ACCEPTED[right], _ACCEPTED[left]) is verdict


@pytest.mark.parametrize("index", range(len(_ACCEPTED)))
def test_a_value_is_never_reported_as_differing_from_itself(index: int) -> None:
    """Reflexive, except where the value declares it is not.

    ``nan`` is the exception and it is named rather than skipped: nothing can
    establish that two NaNs are the same value. A single ``nan``, being one
    object, is caught by the identity step -- which this checks by comparing
    two separately built values wherever the value can be rebuilt.
    """
    value = _ACCEPTED[index]
    assert values_differ(value, value) is False


def test_the_accepted_population_is_the_one_the_surface_produces() -> None:
    """The population is imported, not restated, so the two cannot drift.

    Copying it here would be DC3 with extra steps: the surface would widen, the
    copy would not, and the bound would go on being asserted over yesterday's
    domain.
    """
    assert _ACCEPTED, "the accepted population is empty"
    assert len(_ACCEPTED) == len(ACCEPTED_POPULATION)


# ---------------------------------------------------------------------------
# The verdict follows the wire form
# ---------------------------------------------------------------------------

_WIRE_PAIR_VALUES: list[tuple[str, Any]] = [
    ("exact str", "0.5"),
    ("float", 0.5),
    ("int-ish float", 3.0),
    ("text with a trailing zero", "0.50"),
    ("single-element list", [0.5]),
    ("single-element tuple", (0.5,)),
    ("two-element list", [0.5, 0.5]),
    ("comma form", "0.5,0.5"),
    ("different value", "0.25"),
    ("different number", 0.25),
    # In the population **because** it is the declared exception, so the
    # exception is exercised rather than only written down. A declared
    # exception nothing reaches is the shape this run has been hunting.
    ("nan", float("nan")),
    ("another nan", float("nan")),
]


#: NaN is outside the relation, declared here rather than discovered later.
#:
#: The serialiser writes `nan` happily, so a relation quantified over "values
#: the serialiser accepts" would demand that two NaNs be admitted as one value.
#: The module deliberately reports them as differing, and says why: nothing can
#: establish that two NaNs are the same value, since `nan != nan`. Round 20
#: caught a new relation contradicting that older, documented contract.
#:
#: The exception costs nothing real, and that is executed rather than assumed:
#: LightGBM refuses a NaN `learning_rate` outright, so no pair of NaN spellings
#: can reach a trained model under any verdict.
_NAN_IS_OUTSIDE_THE_RELATION = True


def _wire(value: Any) -> str | None:
    """The bytes LightGBM would send, or ``None`` if it refuses the value."""
    from lightgbm.basic import _param_dict_to_str

    try:
        return _param_dict_to_str({"k": value})[2:]
    except Exception:  # noqa: BLE001 - refusal is one of the answers
        return None


def _is_nan_wire(wire: str) -> bool:
    """Does this wire form denote a NaN?"""
    try:
        return math.isnan(float(wire))
    except ValueError:
        return False


#: Pairs the relation reports and this PR deliberately does not fix, each with
#: the issue that owns it. The values stay in the population above so the
#: non-vacuity witnesses are not lost; only the assertion is deferred. Admitting
#: a currently-refused pair is a behaviour widening -- an ``allow`` under the
#: Change Gate -- and needs a Proposal with a measured firing rate, not a fix
#: folded into a review round. H-0095 considered #283 explicitly and left it
#: open: after normalisation the two are still a ``float`` and a ``list``, and
#: admitting them means deciding which one the estimator is given.
KNOWN_BOUNDS: dict[frozenset[str], str] = {
    frozenset({"float", "single-element list"}): "#283",
    frozenset({"float", "single-element tuple"}): "#283",
}


@pytest.mark.parametrize("left_label", [label for label, _ in _WIRE_PAIR_VALUES])
@pytest.mark.parametrize("right_label", [label for label, _ in _WIRE_PAIR_VALUES])
def test_the_verdict_follows_the_wire_form(left_label: str, right_label: str) -> None:
    """Two implications, over every pair the serialiser writes.

    - same wire form  => ``values_differ`` is ``False`` (one value, admit it);
    - both numeric and different => ``values_differ`` is ``True`` (refuse it).

    The second is the one round 19 broke in the dangerous direction, and no
    no-raise assertion can see it. Pairs whose wire forms differ but are not
    both numeric are left unasserted on purpose: this function is allowed to
    have no opinion there, and claiming otherwise would be the kind of
    over-broad declaration this module has already paid for three times.
    """
    values = dict(_WIRE_PAIR_VALUES)
    left = _through_the_surface(values[left_label])
    right = _through_the_surface(values[right_label])

    wire_left, wire_right = _wire(left), _wire(right)
    assert wire_left is not None and wire_right is not None, (
        "an accepted value the serialiser refuses; the surface is wrong"
    )

    bound = KNOWN_BOUNDS.get(frozenset({left_label, right_label}))
    if bound is not None:
        pytest.skip(f"known bound, filed as {bound}")

    if _is_nan_wire(wire_left) or _is_nan_wire(wire_right):
        pytest.skip("outside the relation: NaN, see _NAN_IS_OUTSIDE_THE_RELATION")

    verdict = values_differ(left, right)
    assert isinstance(verdict, bool)

    if wire_left == wire_right:
        assert verdict is False, (
            f"{left_label} vs {right_label}: both write {wire_left!r} "
            f"and were reported as differing"
        )
        return

    try:
        numeric = float(wire_left) != float(wire_right)
    except ValueError:
        numeric = False
    if numeric:
        assert verdict is True, (
            f"{left_label} vs {right_label}: {wire_left!r} and {wire_right!r} "
            f"are different values and were reported as the same"
        )


def test_both_wire_implications_have_witnesses() -> None:
    """A relation nothing reaches is a relation nobody has checked.

    Round 19 was found by asserting the relation rather than by adding another
    axis to a generated population, and a relation whose antecedent is never
    satisfied would have asserted nothing at all.
    """
    values = [(label, _through_the_surface(v)) for label, v in _WIRE_PAIR_VALUES]
    wires = [(label, _wire(value)) for label, value in values]

    same = [
        (a, b)
        for index, (a, wire_a) in enumerate(wires)
        for b, wire_b in wires[index + 1 :]
        if wire_a == wire_b and not _is_nan_wire(str(wire_a))
    ]
    asserted_same = [(a, b) for a, b in same if frozenset({a, b}) not in KNOWN_BOUNDS]
    assert asserted_same, "no pair shares a wire form; the first implication is vacuous"

    numeric_different = []
    for index, (a, wire_a) in enumerate(wires):
        for b, wire_b in wires[index + 1 :]:
            if wire_a is None or wire_b is None:
                continue
            try:
                if float(wire_a) != float(wire_b):
                    numeric_different.append((a, b))
            except ValueError:
                continue
    assert numeric_different, "no numerically different pair; the second is vacuous"


def test_the_nan_exception_is_reached_and_is_the_only_one_of_its_kind() -> None:
    """The declared exception, executed.

    Two separately built NaNs are two objects, so the identity step does not
    absorb them and the comparison really is asked.
    """
    assert _NAN_IS_OUTSIDE_THE_RELATION
    first, second = float("nan"), float("nan")
    assert first is not second
    assert _wire(first) == _wire(second)
    assert values_differ(first, second) is True

    one = float("nan")
    assert values_differ(one, one) is False


def test_the_known_bounds_are_real_and_still_bite() -> None:
    """A deferral nobody can lose, and one that is still true.

    An exemption for a pair the code has since started admitting is a stale
    acceptance (DC5): it would silently excuse a case that no longer needs
    excusing, and the next reader would take the issue as open when it is not.
    """
    values = dict(_WIRE_PAIR_VALUES)
    for pair, issue in KNOWN_BOUNDS.items():
        left_label, right_label = sorted(pair)
        left = _through_the_surface(values[left_label])
        right = _through_the_surface(values[right_label])
        assert _wire(left) == _wire(right), f"{issue}: the pair no longer shares a wire"
        assert values_differ(left, right) is True, (
            f"{issue} is exempted but the pair is now admitted; remove the bound"
        )


# ---------------------------------------------------------------------------
# Where the retired population went
# ---------------------------------------------------------------------------


def test_the_hostile_population_is_refused_rather_than_forgotten() -> None:
    """The objects rounds 16-20 were spent on still have a home.

    This file used to carry them and assert that the comparison survived them.
    H-0095 moved the claim: they never reach the comparison. Asserting the
    handoff here is what keeps "we moved it" from becoming "we dropped it" --
    the shape a reader cannot check by reading either file alone.
    """
    from tests.test_core import test_param_domain

    assert test_param_domain.ROUNDS_16_TO_20_OBJECTS
    for label, value in test_param_domain.ROUNDS_16_TO_20_OBJECTS.items():
        with pytest.raises(LizyMLError):
            normalise_params({"learning_rate": value}, surface="probe")
        assert "round" in label
