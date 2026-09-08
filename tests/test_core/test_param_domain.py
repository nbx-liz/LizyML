"""The ingress normaliser, against the serialiser it is derived from.

H-0095. `values_differ` spent five consecutive review rounds (H-0094 rounds
16-20) being handed one more object whose `__format__`, `__class__`, `tolist`
or `__eq__` answered in a way it had not anticipated. Every fix was right about
the object the round named and silent about the next one, because the domain
had no boundary. `lizyml.core.param_domain` gives it one.

The central property here is **wire preservation**: for every value the
normaliser accepts, the bytes LightGBM is sent are the same before and after
normalisation. It is a closed, executable statement -- the oracle is the
serialiser itself, read rather than reconstructed -- and it is what makes
"normalise at the surface" safe to do at all. A normaliser that changed the
bytes would train a different model in silence, which is the defect class this
whole PR exists to remove.
"""

from __future__ import annotations

import contextlib
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.param_domain import (
    ACCEPTED_DESCRIPTION,
    PLAIN_SCALAR_TYPES,
    PLAIN_SEQUENCE_TYPES,
    REFUSED_SEQUENCE_TYPES,
    assert_plain_params,
    is_accepted,
    is_plain,
    normalise_params,
    normalise_value,
)


def _wire(value: Any) -> str | None:
    """The bytes LightGBM would send for this value, or ``None`` if it refuses.

    Read from the serialiser rather than reconstructed. Reconstructing it is
    what round 19 got wrong: the scalar branch writes ``f"{key}={val}"`` and the
    element branch calls ``str``, and the two disagree for a value that
    overrides one and not the other.
    """
    from lightgbm.basic import _param_dict_to_str

    try:
        return _param_dict_to_str({"k": value})[2:]
    except Exception:  # noqa: BLE001 - refusal is one of the answers
        return None


# ---------------------------------------------------------------------------
# The accepted population, built rather than listed
# ---------------------------------------------------------------------------

#: numpy scalar types, taken from numpy's own hierarchy rather than typed out.
_NUMPY_SCALAR_TYPES: list[type] = [
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.bool_,
]

#: Values chosen for the ways a float can print: exponent forms on both ends,
#: the values a reduced-precision dtype cannot hold exactly, and the three
#: non-finite ones.
_FLOAT_VALUES: list[float] = [
    0.0,
    -0.0,
    0.1,
    0.5,
    1.0,
    -1.0,
    3.14159265358979,
    1e-8,
    1e-4,
    1e3,
    123456.789,
    float("inf"),
    float("-inf"),
    float("nan"),
]
_INT_VALUES: list[int] = [0, 1, 3, 100]


def _plain_atoms() -> list[Any]:
    """Values already inside the accepted set."""
    atoms: list[Any] = [None, True, False, "", "text", "0.50", "1,2"]
    atoms += _INT_VALUES + [-7, 2**40]
    atoms += _FLOAT_VALUES
    atoms.append(pathlib.Path("models/out.txt"))
    return atoms


def _numpy_atoms() -> list[Any]:
    """One numpy scalar per (dtype, value) that the dtype can hold."""
    atoms: list[Any] = []
    for dtype in _NUMPY_SCALAR_TYPES:
        if dtype is np.bool_:
            values: list[Any] = [True, False]
        elif np.dtype(dtype).kind == "f":
            values = _FLOAT_VALUES
        else:
            values = _INT_VALUES
        for value in values:
            try:
                atoms.append(dtype(value))
            except (OverflowError, ValueError):
                continue
    return atoms


def _sequences_of(atom: Any) -> list[Any]:
    """Every sequence shape this atom can sit in, accepted or not.

    The refused shapes are here on purpose: a boundary nothing probes from the
    outside is a boundary nobody has checked. A ``set`` and a three-deep list
    are the two the module declines, and both are generated rather than named
    in the tests that assert the decline.
    """
    shapes: list[Any] = [
        [atom],
        [atom, atom],
        (atom,),
        [[atom, atom]],
        [[[atom]]],
        {"entry": atom},
        ["auc", {"entry": atom}],
    ]
    with contextlib.suppress(TypeError):  # no atom here is unhashable, but do
        shapes.append({atom})  # not assume it
    if isinstance(atom, np.generic):
        shapes.append(np.array([atom, atom], dtype=atom.dtype))
    return shapes


def _accepted_population() -> list[Any]:
    """Every value the normaliser is claimed to accept, as one flat list."""
    population: list[Any] = []
    for atom in _plain_atoms() + _numpy_atoms():
        population.append(atom)
        population.extend(_sequences_of(atom))
    population.append([])
    population.append(())
    return population


CANDIDATES: list[Any] = _accepted_population()


def _label(value: Any) -> str:
    try:
        return f"{type(value).__name__}:{value!r}"
    except Exception:  # noqa: BLE001 - a label is not worth failing over
        return type(value).__name__


def _holds_a_mapping(value: Any) -> bool:
    """Is there a ``dict`` anywhere inside this value?"""
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return any(_holds_a_mapping(member) for member in value)
    return False


def _normalise(value: Any) -> tuple[bool, Any]:
    """``(accepted, normalised)`` for one candidate, through the public entry."""
    try:
        return True, normalise_params({"k": value}, surface="probe")["k"]
    except LizyMLError:
        return False, None


#: The candidates the normaliser accepts, and the ones it refuses -- **measured**
#: rather than listed, so the boundary the tests below assert is the boundary
#: the code actually has. The two are then checked against the serialiser
#: separately: an accepted value must keep its bytes, and a refused one must be
#: one no plain stand-in of its own kind could have carried.
ACCEPTED_POPULATION: list[Any] = [v for v in CANDIDATES if _normalise(v)[0]]
REFUSED_POPULATION: list[Any] = [v for v in CANDIDATES if not _normalise(v)[0]]


@pytest.mark.parametrize(
    "value", ACCEPTED_POPULATION, ids=[_label(v) for v in ACCEPTED_POPULATION]
)
def test_normalising_does_not_change_the_bytes_the_estimator_is_sent(
    value: Any,
) -> None:
    """The central acceptance criterion of H-0095.

    Not "the normalised value is equal" -- equality is the question the whole
    PR found hard. The bytes are what the trainer reads, so the bytes are what
    has to be identical.

    A value holding a mapping is excluded, and the exclusion is asserted rather
    than assumed: such a value is never handed to the serialiser as itself (the
    adapter consumes it, and the exit assertion refuses it if it did not), so
    "the bytes it would be sent" is not a thing it has. The companion test
    below is what makes that exclusion safe.
    """
    if _holds_a_mapping(value):
        assert not is_plain(normalise_value(value)), (
            "a mapping-bearing value passed the serialiser bound; the wire "
            "exclusion below would then be hiding a real change"
        )
        pytest.skip("holds a mapping: never serialised as itself")
    before = _wire(value)
    normalised = normalise_value(value)
    assert _wire(normalised) == before, (
        f"normalising {_label(value)} changed the wire form: "
        f"{before!r} -> {_wire(normalised)!r}"
    )


@pytest.mark.parametrize(
    "value", ACCEPTED_POPULATION, ids=[_label(v) for v in ACCEPTED_POPULATION]
)
def test_every_accepted_value_normalises_into_the_closed_set(value: Any) -> None:
    """Acceptance means landing inside the set, not merely not raising.

    Against the surface predicate, and then -- for everything that is not a
    mapping -- against the narrower one the trainer is entitled to. The two are
    checked separately because they are two different claims, and collapsing
    them would weaken whichever end was asserted with the other's bound.
    """
    normalised = normalise_value(value)
    assert is_accepted(normalised), f"{_label(value)} normalised to {normalised!r}"
    if not _holds_a_mapping(normalised):
        assert is_plain(normalised), f"{_label(value)} normalised to {normalised!r}"


@pytest.mark.parametrize(
    "value", ACCEPTED_POPULATION, ids=[_label(v) for v in ACCEPTED_POPULATION]
)
def test_the_predicate_and_the_normaliser_agree(value: Any) -> None:
    """A predicate that disagreed with the function would be the SSOT drift.

    ``is_accepted`` exists so callers can ask without raising; if it answered
    ``True`` for something ``normalise_params`` refuses, or the reverse, the
    two would be two definitions of the accepted set.
    """
    normalised = normalise_value(value)
    assert is_accepted(normalised)
    if is_accepted(value):
        # `repr`, not `==`: `nan` is not equal to itself, and this is a claim
        # about the normaliser leaving an accepted value alone rather than a
        # claim about equality -- the question the whole PR found hard.
        assert repr(normalise_value(value)) == repr(value)


@pytest.mark.parametrize(
    "value", ACCEPTED_POPULATION, ids=[_label(v) for v in ACCEPTED_POPULATION]
)
def test_normalising_twice_is_normalising_once(value: Any) -> None:
    """Two surfaces may normalise the same dict; the second must be a no-op.

    `calibration.params` is normalised both where it is refused and where the
    dict that reaches the calibrator is built, so this is a property the wiring
    depends on rather than a nicety.
    """
    once = normalise_value(value)
    twice = normalise_value(once)
    assert repr(twice) == repr(once)
    assert is_accepted(twice)


def test_the_population_covers_every_shape_the_serialiser_distinguishes() -> None:
    """The population is not a handful of examples someone happened to think of.

    A declared fixture closes only the axis it is checked against, so this
    states the axes: both formatter positions, every numpy scalar type, both
    sequence origins, and the one nested form LightGBM gives meaning to.
    """
    kinds = {type(value) for value in CANDIDATES}
    assert set(PLAIN_SEQUENCE_TYPES) <= kinds, kinds
    assert dict in kinds, "no mapping in the candidate set"
    assert any(type(v) in REFUSED_SEQUENCE_TYPES for v in CANDIDATES)
    assert np.ndarray in kinds
    assert {type(None), bool, int, float, str} <= kinds
    assert any(isinstance(value, pathlib.PurePath) for value in CANDIDATES)
    dtypes = {type(value) for value in CANDIDATES if isinstance(value, np.generic)}
    assert dtypes == set(_NUMPY_SCALAR_TYPES), set(_NUMPY_SCALAR_TYPES) - dtypes
    nested = [
        value
        for value in CANDIDATES
        if type(value) is list and value and type(value[0]) is list
    ]
    assert nested, "no nested list in the population"
    non_finite = [
        value
        for value in CANDIDATES
        if type(value) is float and not math.isfinite(value)
    ]
    assert len(non_finite) == 3, non_finite


@pytest.mark.parametrize(
    "value", REFUSED_POPULATION, ids=[_label(v) for v in REFUSED_POPULATION]
)
def test_a_refusal_inside_the_candidate_set_is_forced_not_chosen(value: Any) -> None:
    """Where the normaliser refuses a value LightGBM would have taken, say why.

    A normaliser is allowed to be narrower than the serialiser, but not for no
    reason: an arbitrary refusal is DC7, a gate no valid input can pass. Exactly
    one reason is admitted here -- some element prints text that **no plain
    value of the same kind** prints, so nothing can stand in for it without
    changing the bytes. ``str(numpy.float16(1e3))`` is ``1e+03``, and no Python
    ``int`` or ``float`` prints that.

    A ``str`` would. Turning a number the caller wrote into text to get past the
    boundary is not a stand-in: it is a different value that happens to share
    the bytes, and the normalised dict is what the rest of the library then
    computes on.
    """
    assert _refusal_reason(value) is not None, (
        f"{_label(value)} is refused for none of the declared reasons"
    )


#: Why the module is allowed to be narrower than the serialiser. Two reasons,
#: closed, and every refusal inside the candidate set must name one of them.
REFUSAL_REASONS = (
    "no plain value prints the same text",
    "an unordered container in a positional parameter",
    "a list nested deeper than the serialiser gives meaning to",
)


def _refusal_reason(value: Any) -> str | None:
    """Which declared reason refuses this value, or ``None`` for none of them."""
    # Only in **element** position. A bare numpy scalar goes through the
    # serialiser scalar branch, where `.item()` preserves the bytes for every
    # dtype -- `numpy.float16(1e3)` is accepted written alone and refused
    # written inside a list, and that asymmetry is the two formatters rather
    # than an inconsistency.
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)) and any(
        isinstance(element, np.generic) and not _has_plain_stand_in(element)
        for element in _elements_of(value)
    ):
        return REFUSAL_REASONS[0]
    if type(value) in REFUSED_SEQUENCE_TYPES:
        return REFUSAL_REASONS[1]
    if _list_depth(value) > 2:
        return REFUSAL_REASONS[2]
    return None


def _list_depth(value: Any) -> int:
    """How deeply lists are nested inside this value."""
    if isinstance(value, (list, tuple, np.ndarray)):
        members = list(value)
        return 1 + max((_list_depth(member) for member in members), default=0)
    return 0


def _elements_of(value: Any) -> list[Any]:
    """Every position the serialiser would format inside ``value``."""
    if isinstance(value, np.ndarray) or type(value) in PLAIN_SEQUENCE_TYPES:
        flattened: list[Any] = []
        for member in list(value):
            flattened.extend(_elements_of(member))
        return flattened
    return [value]


def _has_plain_stand_in(element: np.generic) -> bool:
    """Does a plain number print exactly as this element does?"""
    text = str(element)
    for candidate in (element.item(), *_parsed(text)):
        if type(candidate) in (bool, int, float) and str(candidate) == text:
            return True
    return False


def _parsed(text: str) -> list[Any]:
    parsed: list[Any] = []
    for build in (int, float):
        try:
            parsed.append(build(text))
        except (TypeError, ValueError):
            continue
    return parsed


def test_the_refused_subset_is_exactly_the_declared_boundary() -> None:
    """The measured boundary and the declared one are the same set.

    Not a proportion: the candidate set deliberately contains refused shapes,
    so a proportion would only say how many of those were generated. This says
    the stronger thing -- every value the module refuses is one the declared
    reasons predict, and every value they predict is refused. A change that
    widened the refusal without a reason, or that quietly started accepting a
    shape a reason names, fails here rather than in a user config.
    """
    assert REFUSED_POPULATION, "nothing is refused; the candidate set stopped probing"
    predicted = {
        id(value) for value in CANDIDATES if _refusal_reason(value) is not None
    }
    measured = {id(value) for value in REFUSED_POPULATION}
    assert measured == predicted, (
        f"refused but not predicted: {len(measured - predicted)}; "
        f"predicted but accepted: {len(predicted - measured)}"
    )
    reasons = {_refusal_reason(value) for value in REFUSED_POPULATION}
    assert None not in reasons, "a refusal with no declared reason"
    assert reasons == set(REFUSAL_REASONS), (
        f"a declared reason nothing reaches: {set(REFUSAL_REASONS) - reasons}"
    )


# ---------------------------------------------------------------------------
# The accept set is derived from LightGBM, not copied from it
# ---------------------------------------------------------------------------


def test_the_scalar_types_are_the_ones_the_serialiser_names() -> None:
    """Derived, so that a LightGBM upgrade widening its set fails here first.

    ``_param_dict_to_str`` writes a scalar when the value is a ``str``, a
    ``Path``, or one of ``_NUMERIC_TYPES``; ``None`` is the fourth case, where
    the parameter is dropped rather than written. Anything else it accepts, it
    accepts through ``_is_numeric`` -- which is any object ``float()`` survives,
    an open set by construction and the one this module deliberately closes.
    """
    from lightgbm.basic import _NUMERIC_TYPES

    named = set(_NUMERIC_TYPES) | {str, type(None)}
    ours = set(PLAIN_SCALAR_TYPES)
    assert named <= ours, f"the serialiser names types we refuse: {named - ours}"
    extra = ours - named
    assert all(issubclass(kind, pathlib.PurePath) for kind in extra), extra


def test_the_sequence_types_are_the_ones_the_serialiser_joins() -> None:
    """The comma-joining branch, read out of the serialiser's own source.

    Both directions: nothing is treated as a sequence that the serialiser does
    not join, and the one type it joins that this refuses is refused **on the
    record** rather than by omission.
    """
    import inspect

    from lightgbm.basic import _param_dict_to_str

    source = inspect.getsource(_param_dict_to_str)
    joining = source.split("elif")[0]
    for kind in PLAIN_SEQUENCE_TYPES:
        assert kind.__name__ in joining, (
            f"{kind.__name__} is treated as a sequence here but the serialiser "
            "does not join it"
        )
    assert "_is_numpy_1d_array" in joining
    assert "set" in joining, "the serialiser no longer joins a set"
    assert set in REFUSED_SEQUENCE_TYPES


def test_a_set_is_refused_rather_than_ordered_by_hash() -> None:
    """The one type the serialiser joins and this refuses, with its reason.

    ``list({1.0, 2.0})`` writes the bytes the serialiser would have written, so
    normalising a set would preserve the wire. What it would **not** preserve
    is the reason the old comparison refused a set: every sequence parameter
    here is positional, so admitting ``{1.0, 2.0}`` beside a literal
    ``[1.0, 2.0]`` makes them one value by hash-order coincidence rather than
    by anything the caller wrote. Refusing is the narrowing half of that trade.
    """
    with pytest.raises(LizyMLError) as exc:
        normalise_params({"feature_contri": {1.0, 2.0}}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "feature_contri" in exc.value.user_message

    # The coincidence itself, so the reason is evidence rather than assertion.
    assert list({1.0, 2.0}) == [1.0, 2.0]
    assert list({3.0, 1.0, 2.0}) != [3.0, 1.0, 2.0]


def test_a_list_nested_deeper_than_the_serialiser_reads_is_refused() -> None:
    """Depth 2 is where ``_to_string`` stops, and the boundary is where it stops.

    At depth 3 the members are written by Python's own list repr, and a
    normalised member prints differently from the value the caller wrote --
    executed here rather than argued, because the argument is what a reviewer
    cannot check.
    """
    deep = [[[np.float32(0.1)]]]
    assert _wire(deep) == "[[np.float32(0.1)]]"
    assert _wire([[[0.1]]]) == "[[0.1]]"

    with pytest.raises(LizyMLError):
        normalise_params({"interaction_constraints": deep}, surface="probe")
    with pytest.raises(LizyMLError):
        normalise_params({"interaction_constraints": [[[1, 2]]]}, surface="probe")


# ---------------------------------------------------------------------------
# What is refused, and how
# ---------------------------------------------------------------------------


class _FormatsToLiar:
    """Round 20's object: `__format__` and `__str__` disagree."""

    def __format__(self, spec: str) -> str:
        return "0.5"

    def __str__(self) -> str:
        return "0.9"

    def __float__(self) -> float:
        return 0.5


class _Proxy(str):
    """Round 18's object: a `str` subclass that lies about its class."""

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # type: ignore[override]
        return float


class _Rate:
    """Round 16's object: numeric to LightGBM, nothing else."""

    def __float__(self) -> float:
        return 0.5


class _Tolist:
    """Round 19's object: a `tolist` that is not a conversion."""

    @property
    def tolist(self) -> Any:
        raise RuntimeError("looking is not free")


#: The objects the last five review rounds were spent on, by round.
ROUNDS_16_TO_20_OBJECTS: dict[str, Any] = {
    "round 16 numeric-only object": _Rate(),
    "round 17 numpy array": np.array([[1.0, 2.0], [3.0, 4.0]]),
    "round 18 class-lying str subclass": _Proxy("0.5"),
    "round 19 hostile attribute lookup": _Tolist(),
    "round 20 disagreeing formatters": _FormatsToLiar(),
}


@pytest.mark.parametrize("label", sorted(ROUNDS_16_TO_20_OBJECTS))
def test_the_objects_five_review_rounds_were_spent_on_are_refused_at_ingress(
    label: str,
) -> None:
    """Rewritten, not deleted: the same objects, now refused instead of compared.

    Each of these was the subject of one blocking finding in H-0094 rounds
    16-20, and each fix taught `values_differ` one more thing about one more
    object. They are kept because the claim being made now is about them: not
    that the comparison handles them, but that they never reach it.
    """
    with pytest.raises(LizyMLError) as exc:
        normalise_params(
            {"learning_rate": ROUNDS_16_TO_20_OBJECTS[label]}, surface="probe"
        )
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "learning_rate" in exc.value.user_message
    assert "probe" in exc.value.user_message


def test_a_refusal_names_the_surface_the_parameter_and_what_is_accepted() -> None:
    """A refusal the caller cannot act on is a refusal that costs a support round."""
    with pytest.raises(LizyMLError) as exc:
        normalise_params({"learning_rate": _Rate()}, surface="fit(params=)")

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "fit(params=)" in exc.value.user_message
    assert "learning_rate" in exc.value.user_message
    assert ACCEPTED_DESCRIPTION in exc.value.user_message
    rejected = exc.value.context["rejected"]
    assert [entry["parameter"] for entry in rejected] == ["learning_rate"]
    assert rejected[0]["type"] == "_Rate"


def test_every_rejected_parameter_is_reported_not_only_the_first() -> None:
    """One round trip per bad value is one round trip too many."""
    with pytest.raises(LizyMLError) as exc:
        normalise_params(
            {"learning_rate": _Rate(), "num_leaves": _Rate(), "seed": 1},
            surface="model.params",
        )
    reported = {entry["parameter"] for entry in exc.value.context["rejected"]}
    assert reported == {"learning_rate", "num_leaves"}


def test_a_value_with_no_plain_stand_in_is_refused_rather_than_rounded() -> None:
    """The one place the accepted set is narrower than "every numpy dtype".

    ``str(numpy.float32(1e8))`` is ``1e+08`` and no Python float prints that, so
    an element of that value has no stand-in. Converting it anyway -- with
    ``.tolist()``, the obvious choice -- sends ``100000000.0`` instead, which is
    a different number of characters for LightGBM to parse and, for
    ``float32(0.1)``, a different number. Refusing is the half of that trade
    this PR is allowed to fall on.
    """
    hostile = np.array([1e8], dtype=np.float32)
    assert _wire(hostile) == "1e+08"
    assert _wire(hostile.tolist()) == "100000000.0"

    with pytest.raises(LizyMLError) as exc:
        normalise_params({"feature_contri": hostile}, surface="model.params")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "1e+08" in exc.value.user_message


def test_a_reduced_precision_element_that_does_have_a_stand_in_is_accepted() -> None:
    """The refusal above is narrow: the same dtype is accepted where it can be."""
    accepted = np.array([0.1, 0.5], dtype=np.float32)
    assert _wire(normalise_value(accepted)) == _wire(accepted)
    assert is_plain(normalise_value(accepted))


@pytest.mark.parametrize(
    "value",
    [
        {1: "int key"},
        np.array([[1, 2], [3, 4]]),
        [(1, 2)],
        [np.array([1, 2])],
        object(),
        b"bytes",
        {"metric": object()},
    ],
    ids=[
        "non-str-mapping-key",
        "2d-array",
        "nested-tuple",
        "nested-array",
        "object",
        "bytes",
        "mapping-of-object",
    ],
)
def test_shapes_outside_the_accepted_set_are_refused(value: Any) -> None:
    """Including shapes the serialiser itself would have written unreadably.

    ``[(1, 2)]`` serialises to ``(1, 2)`` and ``[numpy.array([1, 2])]`` to
    ``[1 2]``; LightGBM parses neither, so accepting them would only move the
    failure inside the trainer, where it arrives without naming the parameter.
    """
    with pytest.raises(LizyMLError) as exc:
        normalise_params({"learning_rate": value}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID


def test_a_metric_entry_written_as_a_mapping_is_accepted_at_the_surface() -> None:
    """The two ends of the pipe have different accepted sets, on purpose.

    LightGBM has no mapping form. LizyML does: a metric entry is written as
    ``{"precision_at_k": {"k": 15}}``, or inside a list beside plain names
    (H-0065), and the adapter turns those into evaluation functions and removes
    them before serialising. Refusing a mapping at the surface would refuse a
    documented config; accepting one at ``lgb.train`` would serialise a value
    LightGBM cannot read. So the surface accepts it and the exit assertion does
    not.
    """
    entry = {"precision_at_k": {"k": np.int64(15)}}
    normalised = normalise_params({"metric": entry}, surface="probe")["metric"]
    assert normalised == {"precision_at_k": {"k": 15}}
    assert type(normalised["precision_at_k"]["k"]) is int

    in_a_list = normalise_params({"metric": ["auc", entry]}, surface="probe")["metric"]
    assert in_a_list == ["auc", {"precision_at_k": {"k": 15}}]


def test_a_mapping_that_survived_to_the_trainer_is_a_defect() -> None:
    """The other half of the sentence above, stated where it is checked."""
    with pytest.raises(LizyMLError) as exc:
        assert_plain_params({"metric": {"precision_at_k": {"k": 15}}}, where="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert not is_plain({"a": 1})
    assert not is_plain(["auc", {"a": 1}])


def test_the_one_nested_shape_lightgbm_reads_is_accepted() -> None:
    """``interaction_constraints`` is a list of lists, and it has to keep working."""
    constraints = [[0, 1], [2, 3]]
    assert _wire(constraints) == "[0,1],[2,3]"
    assert normalise_value(constraints) == constraints
    assert _wire(normalise_value(constraints)) == _wire(constraints)


def test_a_message_is_built_without_asking_the_value_to_print_itself() -> None:
    """A rejected value is one whose printing is not trusted, by definition."""

    class _Explodes:
        def __str__(self) -> str:
            raise RuntimeError("no")

        def __repr__(self) -> str:
            raise RuntimeError("no")

    with pytest.raises(LizyMLError) as exc:
        normalise_params({"learning_rate": _Explodes()}, surface="probe")
    assert "_Explodes" in exc.value.user_message


# ---------------------------------------------------------------------------
# The exit assertion
# ---------------------------------------------------------------------------


def test_the_exit_assertion_passes_everything_the_normaliser_produces() -> None:
    """The two ends agree, over the whole population rather than an example.

    Everything except the mapping-bearing values, which the trainer is entitled
    to refuse and which the adapter is supposed to have consumed. That
    exclusion is not a hole: the test below asserts each of them **is** refused,
    so the two together partition the accepted set.
    """
    normalised = [normalise_value(value) for value in ACCEPTED_POPULATION]
    serialisable = {
        f"p{index}": value
        for index, value in enumerate(normalised)
        if not _holds_a_mapping(value)
    }
    assert_plain_params(serialisable, where="probe")

    mappings = [value for value in normalised if _holds_a_mapping(value)]
    assert mappings, "no mapping in the population; the partition is one-sided"
    for index, value in enumerate(mappings):
        with pytest.raises(LizyMLError):
            assert_plain_params({f"m{index}": value}, where="probe")


def test_the_exit_assertion_refuses_a_value_that_skipped_the_surfaces() -> None:
    """What it exists for: a fifth route into training, added later, with no gate.

    Normalising at four surfaces is a claim about wiring, and a claim about
    wiring is what fails silently when someone adds a fifth (DC4). This turns
    it into a property of the value that arrives.
    """
    with pytest.raises(LizyMLError) as exc:
        assert_plain_params({"learning_rate": _Rate()}, where="lgb.train")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "lgb.train" in exc.value.user_message
    assert exc.value.context["unnormalised"] == [
        {"parameter": "learning_rate", "type": "_Rate"}
    ]


def test_the_exit_assertion_is_called_at_every_place_that_trains() -> None:
    """Derived from the source, so a third training site fails here.

    The population is every call to LightGBM's trainer inside the package. A
    site that trains without the assertion is a route the ingress does not
    cover, which is precisely the shape this assertion exists to detect -- so
    it cannot be left to a reviewer to notice.
    """
    import re

    package = pathlib.Path(__file__).resolve().parents[2] / "lizyml"
    training_sites = []
    for path in package.rglob("*.py"):
        if path.parts[-2:] == ("codegen", "templates.py"):
            # Generated code for a standalone script; it has no LizyML import.
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^\s*(?:self\._model = )?lgbm?\.train\(", text, re.M):
            # The enclosing function, not the whole file above the call: an
            # assertion in some earlier function would satisfy the looser
            # question while leaving this call unguarded.
            starts = [m.start() for m in re.finditer(r"^\s*def ", text, re.M)]
            enclosing = max((s for s in starts if s < match.start()), default=0)
            body = text[enclosing : match.start()]
            training_sites.append((path, "assert_plain_params(" in body))
    assert training_sites, "no training site found; the scan stopped working"
    missing = [str(path) for path, guarded in training_sites if not guarded]
    assert not missing, f"training without the ingress assertion: {missing}"
