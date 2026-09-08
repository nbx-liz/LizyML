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
import json
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.param_domain import (
    ACCEPTED_DESCRIPTION,
    NUMPY_SCALAR_TYPES,
    PATH_TYPES,
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


@pytest.mark.parametrize(
    "value", ACCEPTED_POPULATION, ids=[_label(v) for v in ACCEPTED_POPULATION]
)
def test_every_accepted_value_can_be_written_as_json(value: Any) -> None:
    """The requirement ``export_code`` brings, quantified over the whole set.

    The trainer is not the only consumer of a normalised value. ``export_code``
    writes the same parameters into ``config.json`` with ``json.dump``, and
    json carries neither a path nor a numpy scalar. That was found one type at
    a time -- a path trained happily and made ``export_code`` raise
    ``TypeError`` afterwards -- and one type at a time is how the next one
    would be found too. So the requirement is asserted here for every value the
    normaliser accepts, rather than for the type that happened to be reported.

    The call mirrors ``codegen/artifact_writer.py``, ``allow_nan`` included at
    its default: ``nan`` and ``inf`` are accepted parameter values and the
    writer emits them the way ``json.dump`` does.
    """
    normalised = normalise_value(value)
    json.dumps({"k": normalised}, indent=2, ensure_ascii=False)


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
    assert reasons <= set(REFUSAL_REASONS), reasons
    # The two reasons that are properties of this module rather than of the
    # numpy build. The third -- no plain value prints the same text -- depends
    # on how numpy chooses to print a reduced-precision float, and on an older
    # numpy no probed value reaches it. Requiring it unconditionally is how
    # this file went red on two CI lanes while passing here.
    version_independent = {REFUSAL_REASONS[1], REFUSAL_REASONS[2]}
    assert version_independent <= reasons, (
        f"a version-independent reason nothing reaches: {version_independent - reasons}"
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
    """Both directions, so a **widening** is visible and not only a narrowing.

    The first version of this asked whether each name we accept appears in the
    joining branch, which a serialiser that grew a new sequence type would go
    on satisfying for ever -- exactly the stale-derivation shape (DC3) the
    derivation exists to prevent, and review round 21 demonstrated it by
    feeding this test a widened source and watching it pass.

    So the names are **extracted** from the branch and compared as a set: every
    type the serialiser joins is either accepted here or refused here on the
    record, and a type that is neither fails.
    """
    import inspect
    import re

    from lightgbm.basic import _param_dict_to_str

    source = inspect.getsource(_param_dict_to_str)
    joining = source.split("elif")[0]

    match = re.search(r"isinstance\(\s*val\s*,\s*\(([^)]*)\)", joining)
    assert match, f"the joining branch is no longer an isinstance tuple: {joining}"
    joined = {name.strip() for name in match.group(1).split(",") if name.strip()}
    assert "_is_numpy_1d_array" in joining

    known = {kind.__name__ for kind in PLAIN_SEQUENCE_TYPES} | {
        kind.__name__ for kind in REFUSED_SEQUENCE_TYPES
    }
    assert joined <= known, (
        f"the serialiser joins {joined - known}, which this module neither "
        "accepts nor refuses on the record"
    )
    assert {kind.__name__ for kind in PLAIN_SEQUENCE_TYPES} <= joined, (
        "a type is treated as a sequence here that the serialiser does not join"
    )
    assert "set" in joined, "the serialiser no longer joins a set"


def test_the_numpy_scalar_types_are_derived_from_numpy() -> None:
    """The other derived set, and the one round 21 found admitting by inheritance.

    Asserted as a relation to numpy own hierarchy rather than as a list: every
    accepted type is one numpy defines under the bases a parameter value can
    come from, and the types outside those bases stay outside.
    """
    assert NUMPY_SCALAR_TYPES, "the derivation produced nothing"
    for kind in NUMPY_SCALAR_TYPES:
        assert kind.__module__.split(".")[0] == "numpy", kind
        assert issubclass(kind, (np.integer, np.floating, np.bool_, np.str_)), kind
    for kind in (np.float16, np.float32, np.float64, np.int64, np.bool_, np.str_):
        assert kind in NUMPY_SCALAR_TYPES, kind
    for kind in (np.datetime64, np.complex128, np.void, np.bytes_):
        assert kind not in NUMPY_SCALAR_TYPES, kind


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("a subclass with a lying formatter", None),
        ("numpy timedelta64", np.timedelta64(1, "ns")),
        ("numpy datetime64", np.datetime64("2020-01-01")),
        ("numpy complex128", np.complex128(1 + 2j)),
    ],
)
def test_a_numpy_value_whose_conversion_would_lose_bytes_is_refused(
    label: str, value: Any
) -> None:
    """Review round 21, both halves of it.

    ``numpy.timedelta64`` is a ``numpy.integer``, so accepting by inheritance
    admitted it, and ``.item()`` turned ``1 nanoseconds`` into ``1`` -- a fit
    that completed on bytes the caller did not write, which is the expensive
    class this whole change exists to remove. A subclass of ``numpy.float64``
    did the same through an overridden ``__format__``.

    Two defences, and this asserts both: the type must be one numpy itself
    defines, and the converted value must **write what the original writes**.
    """
    if value is None:

        class _Lying(np.float64):
            def __format__(self, spec: str) -> str:
                return "0.9"

        value = _Lying(0.1)

    with pytest.raises(LizyMLError) as exc:
        normalise_params({"learning_rate": value}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID


def test_each_defence_is_load_bearing_for_something_different() -> None:
    """Two checks, and what each one actually buys -- measured, not asserted.

    The written-form check is what refuses ``timedelta64``: it is a
    ``numpy.integer``, so the type set contains it and only the comparison of
    what it writes against what its conversion writes catches it.

    The exact-type check buys something the written-form check cannot: with it,
    **no caller-defined code runs at all** during normalisation. ``.item()``
    and ``__format__`` are numpy own implementations, because the value is one
    of numpy own types. Accepting by inheritance would run a subclass method,
    and a subclass method can do anything -- which is the whole shape rounds
    16-20 were spent on. The witness below is a subclass whose ``item`` raises:
    with the type check it is refused, and without it the exception leaves a
    function documented to raise only ``LizyMLError``.
    """
    assert np.timedelta64 in NUMPY_SCALAR_TYPES
    delta = np.timedelta64(1, "ns")
    assert format(delta, "") != format(delta.item(), "")
    with pytest.raises(LizyMLError):
        normalise_params({"learning_rate": delta}, surface="probe")

    class _Boom(np.float64):
        def item(self, *args: Any) -> Any:
            raise RuntimeError("a caller method ran")

    assert type(_Boom(0.1)) not in NUMPY_SCALAR_TYPES
    with pytest.raises(LizyMLError):
        normalise_params({"learning_rate": _Boom(0.1)}, surface="probe")


def test_a_caller_class_cannot_claim_to_be_a_numpy_type() -> None:
    """Review round 22, which defeated the first derivation twice over.

    ``__module__`` is an ordinary class attribute -- writing
    ``__module__ = "numpy"`` in a class body is enough -- so a derivation that
    filtered on it was reading the caller own claim. And the walk that produced
    it ran at import time, so whether a caller class was inside the set
    depended on whether it had been defined before this module was first
    imported.

    Measured before the repair: the serialiser wrote ``learning_rate=0.9`` for
    the caller value and every training call received ``0.1``, and the fit
    completed.

    **The derivation is re-run here rather than read from the module-level
    constant.** A witness defined inside a test body is defined after the
    import, so a test that only looked at the constant would pass against the
    order-dependent implementation -- which is exactly what the first version
    of this test did, caught by reverting the fix and watching it stay green.
    """
    from lizyml.core.param_domain import _derived_numpy_scalar_types

    class _Disguised(np.float64):
        __module__ = "numpy"

        def __format__(self, spec: str) -> str:
            return "0.1" if getattr(self, "converted", False) else "0.9"

        def item(self, *args: Any) -> float:
            self.converted = True
            return 0.1

    assert _Disguised.__module__ == "numpy"
    assert issubclass(_Disguised, np.floating)

    # Derived *now*, with the witness already defined: an implementation that
    # walks subclasses would include it, and one that reads what numpy exports
    # cannot.
    assert _Disguised not in _derived_numpy_scalar_types()
    assert type(_Disguised(0.1)) not in NUMPY_SCALAR_TYPES

    with pytest.raises(LizyMLError) as exc:
        normalise_params({"learning_rate": _Disguised(0.1)}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID


def test_the_numpy_type_set_is_what_numpy_resolves_to_itself() -> None:
    """Membership is a dtype round trip, not a namespace read.

    ``vars(numpy)`` is an ordinary module dict, so a caller who assigns into it
    before this module is first imported puts their own class in the set --
    measured by a read-only checker after round 22, with the caller ``item()``
    running inside normalisation and the serialised value changing from ``0.1``
    to ``0.9``.

    Enumeration only has to produce a superset. What decides membership is that
    numpy resolves the type back to itself, which a subclass does not.
    """
    for kind in NUMPY_SCALAR_TYPES:
        assert np.dtype(kind).type is kind, (
            f"{kind.__name__} is in the set but numpy resolves it to "
            f"{np.dtype(kind).type.__name__}"
        )
    assert np.float64 in NUMPY_SCALAR_TYPES
    assert np.str_ in NUMPY_SCALAR_TYPES


def test_a_class_put_into_the_numpy_namespace_is_still_refused() -> None:
    """The injection the namespace read admitted, closed by the round trip.

    The derivation is re-run with the class already in ``vars(numpy)``, because
    the module-level constant was computed before this test existed and would
    answer for the wrong moment.
    """
    from lizyml.core.param_domain import _derived_numpy_scalar_types

    class _Injected(np.float64):
        pass

    try:
        np.Injected = _Injected  # type: ignore[attr-defined]
        assert vars(np)["Injected"] is _Injected
        assert _Injected not in _derived_numpy_scalar_types()
        assert np.dtype(_Injected).type is np.float64
    finally:
        del np.Injected  # type: ignore[attr-defined]

    with pytest.raises(LizyMLError) as exc:
        normalise_params({"learning_rate": _Injected(0.1)}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID


def test_the_written_form_is_read_before_the_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second defence, tested with the first one deliberately switched off.

    A value that answers ``__format__`` differently after ``item()`` has run
    defeats a check that reads the written form *second*: it ends up comparing
    the conversion against the conversion. Nothing that passes the type gate
    can do that, because numpy own scalars are not stateful -- so the only way
    to test this defence is to let the witness through the first gate on
    purpose. Otherwise the ordering would be an untested claim, which is the
    shape this PR has already paid for.
    """
    import lizyml.core.param_domain as domain

    class _Stateful(np.float64):
        def __format__(self, spec: str) -> str:
            return "0.1" if getattr(self, "converted", False) else "0.9"

        def item(self, *args: Any) -> float:
            self.converted = True
            return 0.1

    monkeypatch.setattr(
        domain, "NUMPY_SCALAR_TYPES", domain.NUMPY_SCALAR_TYPES | {_Stateful}
    )
    with pytest.raises(LizyMLError) as exc:
        domain.normalise_params({"learning_rate": _Stateful(0.1)}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "0.9" in exc.value.user_message


def test_an_accepted_type_is_not_by_itself_a_writable_value() -> None:
    """Review round 24. Being in the accepted set is a claim about the type.

    Two values were accepted, passed the assertion before training, and then
    made LightGBM raise from inside:

    * a ``PurePosixPath``, because the serialiser tests ``isinstance(val,
      Path)`` and a pure path is not a ``Path``;
    * ``10 ** 5000``, because a Python ``int`` has no width and ``str`` of one
      above the interpreter decimal limit raises rather than returning digits.

    The contract is that an accepted value serialises. So the characters are
    **asked for** at the surface rather than assumed from the type.
    """
    from lightgbm.basic import _param_dict_to_str

    for value in (pathlib.PurePosixPath("f.json"), pathlib.PureWindowsPath("f.json")):
        assert not isinstance(value, pathlib.Path), (
            "this pure path is now a Path; the witness needs rewriting"
        )
        with pytest.raises(LizyMLError) as exc:
            normalise_params({"forcedsplits_filename": value}, surface="probe")
        assert exc.value.code is ErrorCode.CONFIG_INVALID

    huge = 10**5000
    with pytest.raises(ValueError):
        str(huge)
    with pytest.raises(LizyMLError) as exc:
        normalise_params({"num_leaves": huge}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    with pytest.raises(LizyMLError):
        normalise_params({"feature_contri": [huge]}, surface="probe")

    # The values either side of each boundary still pass, so the refusal is the
    # narrow one it claims to be.
    for accepted in (pathlib.Path("f.json"), 10**400):
        params = normalise_params({"forcedsplits_filename": accepted}, surface="probe")
        assert _param_dict_to_str(params)


def test_every_path_type_accepted_is_one_the_serialiser_accepts() -> None:
    """Derived from the serialiser own test, rather than from what looks right.

    ``_param_dict_to_str`` writes a scalar when ``isinstance(val, (str, Path,
    ...))``. Listing a path type here that is not a ``Path`` puts a value in
    the accepted set that cannot be written -- which is what happened.
    """
    assert PATH_TYPES, "no path type is accepted; the serialiser still names Path"
    for kind in PATH_TYPES:
        assert issubclass(kind, pathlib.Path), (
            f"{kind.__name__} is accepted here but the serialiser writes only "
            "flavours of Path"
        )
    assert type(pathlib.Path("f.json")) in PATH_TYPES
    assert not any(issubclass(kind, pathlib.PurePath) for kind in PLAIN_SCALAR_TYPES)


def test_a_path_is_carried_on_as_its_text() -> None:
    """Accepted, and converted -- because a path is not carried on as one.

    The serialiser writes a path with the scalar formatter, which for a path is
    its text, so the characters are the same either way. What differs is
    everything downstream: a path survived normalisation as a path, and
    ``export_code`` then raised ``TypeError: Object of type PosixPath is not
    JSON serializable`` on a run that had trained happily. Measured on the
    shipped path, which is why the fix is here and not in the writer.
    """

    written = pathlib.Path("models/forced.json")
    normalised = normalise_params({"forcedsplits_filename": written}, surface="probe")[
        "forcedsplits_filename"
    ]

    assert type(normalised) is str
    assert _wire(normalised) == _wire(written)
    assert is_plain(normalised)
    # The property the conversion exists for, stated where it is checked.
    assert json.dumps({"forcedsplits_filename": normalised})
    with pytest.raises(TypeError):
        json.dumps({"forcedsplits_filename": written})


def test_the_path_conversion_is_safe_because_of_the_types_admitted() -> None:
    """Why converting a path needs no check, said where it can be checked.

    ``pathlib`` defines no ``__format__``, so for these types the scalar
    formatter is ``object.__format__``, which is ``str``. The conversion
    therefore cannot change the characters -- for **these** types. A subclass
    could define one, and the exact-type gate is what keeps subclasses out.

    The first version of this test asserted that a lying subclass was refused,
    and stayed green when the conversion check was removed: the exact-type gate
    was refusing it, not the check. So the check was a guard nothing could
    reach, and it came out rather than being kept as reassurance. That is the
    sixth time in this pull request a test has passed for a reason other than
    the one it named.
    """
    for kind in PATH_TYPES:
        assert "__format__" not in vars(kind), (
            f"{kind.__name__} now defines __format__; the conversion needs a "
            "check of its own again"
        )
    assert "__format__" not in vars(pathlib.PurePath)

    written = pathlib.Path("models/forced.json")
    assert format(written, "") == str(written)

    class _Lying(type(written)):  # type: ignore[misc]
        def __format__(self, spec: str) -> str:
            return "elsewhere.json"

    assert not _is_one_of_path(_Lying("models/forced.json")), (
        "a subclass is admitted by the path gate; it is no longer exact"
    )
    with pytest.raises(LizyMLError) as exc:
        normalise_params(
            {"forcedsplits_filename": _Lying("models/forced.json")}, surface="probe"
        )
    assert exc.value.code is ErrorCode.CONFIG_INVALID


def _is_one_of_path(value: object) -> bool:
    """Does the path gate admit this value?"""
    return any(type(value) is kind for kind in PATH_TYPES)


def test_membership_is_identity_and_not_the_callers_own_equality() -> None:
    """``type(x) in <set>`` was never identity, and a metaclass proved it.

    Membership in a ``set`` or a ``tuple`` is decided by ``__hash__`` and
    ``__eq__``, and for a class those come from its metaclass, which a caller
    writes. A metaclass answering ``hash(numpy.float64)`` and comparing equal
    to it passed the gate with **no numpy base, no claimed module, and no
    dependence on import order**, and its own ``__format__`` and ``item()``
    then ran inside normalisation -- found by a read-only checker after round
    22, and the reason every gate here compares with ``is``.
    """

    class _Claiming(type):
        def __eq__(cls, other: object) -> bool:
            return other is np.float64 or cls is other

        def __hash__(cls) -> int:
            return hash(np.float64)

    class _Sneaky(metaclass=_Claiming):
        reads = 0

        def __float__(self) -> float:
            return 0.1

        def __format__(self, spec: str) -> str:
            type(self).reads += 1
            return "0.9" if type(self).reads == 1 else "0.1"

        def item(self, *args: Any) -> float:
            return 0.9

    # The claim the object makes, so the test fails loudly if a numpy or Python
    # change stops it from being able to make it.
    assert type(_Sneaky) is _Claiming
    assert type(_Sneaky()) in NUMPY_SCALAR_TYPES, (
        "the witness can no longer spoof set membership; rewrite it rather "
        "than delete it"
    )

    with pytest.raises(LizyMLError) as exc:
        normalise_params({"learning_rate": _Sneaky()}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert not is_plain(_Sneaky())
    assert not is_accepted(_Sneaky())


def test_the_element_position_admits_by_identity_too() -> None:
    """The same gate, in the position the scalar tests do not reach.

    A checker found this one untested: the element branch could be widened to
    ``isinstance`` and the whole file stayed green, while the behaviour it
    guards is real -- a subclass overriding ``__str__`` and ``item()`` inside a
    list ran its own code and changed the value.
    """

    class _Lying(np.float64):
        def __str__(self) -> str:
            return "1.0"

        def item(self, *args: Any) -> float:
            return 1.0

    written = [_Lying(2.0), 3.0]
    assert isinstance(written[0], np.floating)
    assert type(written[0]) not in NUMPY_SCALAR_TYPES

    with pytest.raises(LizyMLError) as exc:
        normalise_params({"feature_contri": written}, surface="probe")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "feature_contri" in exc.value.user_message


def test_a_numpy_array_subclass_is_refused_for_the_same_reason_a_scalar_is() -> None:
    """The array gate, closed the way the scalar gate was.

    The rounds 19-21 monitor named this while declining to verify it, and it
    was a real hole: iterating an ``ndarray`` subclass runs the subclass
    ``__iter__``, and a caller method is not obliged to answer the same twice.
    Measured -- a subclass yielding a different sequence on each call was
    normalised to one thing and the trainer would have been sent another,
    which is the silent-wire-change class this whole change exists to remove.

    So the array is admitted by exact type, and the ``1-D`` question is asked
    the way the serialiser asks it, with ``len(shape)`` rather than ``ndim``.
    """

    class _Shifting(np.ndarray):
        _calls = 0

        def __iter__(self) -> Any:
            type(self)._calls += 1
            return iter([float(type(self)._calls)] * 2)

    shifting = np.array([1.0, 2.0]).view(_Shifting)
    assert _wire(shifting) != _wire(shifting), "the witness does not shift"

    for label, value in [
        ("a subclass with a shifting __iter__", np.array([1.0, 2.0]).view(_Shifting)),
        ("numpy matrix", np.matrix([[1.0, 2.0]])),
        ("a masked array", np.ma.masked_array([1.0, 2.0], mask=[0, 1])),
    ]:
        with pytest.raises(LizyMLError) as exc:
            normalise_params({"feature_contri": value}, surface="probe")
        assert exc.value.code is ErrorCode.CONFIG_INVALID, label

    # The plain array it narrows from is untouched.
    plain = np.array([1.0, 2.0])
    assert normalise_value(plain) == [1.0, 2.0]
    assert _wire(normalise_value(plain)) == _wire(plain)


def test_an_integer_too_large_for_a_float_is_accepted_and_compared() -> None:
    """Review round 21, finding 2.

    Python integers have no width, so ``10 ** 400`` is an ordinary accepted
    value -- the serialiser writes its digits -- and the comparison converted it
    to ``float`` to ask the numeric question. ``OverflowError`` was one
    exception short of a function declared total over the accepted set.
    """
    from lizyml.core.value_equality import values_differ

    huge = 10**400
    normalised = normalise_params({"num_leaves": huge}, surface="probe")["num_leaves"]
    assert normalised == huge
    assert _wire(huge) == str(huge)

    assert values_differ(huge, "1") is True
    assert values_differ(huge, str(huge)) is False
    assert values_differ(huge, huge) is False


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

    ``_to_string`` writes a ``list`` member as its own members joined with
    commas, and writes **those** with ``str``. A third level is written by
    Python own list repr instead, which LightGBM does not read back and which
    prints its members by ``repr``.

    The refusal is unconditional; the *sharpest* justification for it is not.
    On a numpy that reprs a scalar as ``np.float32(0.1)``, normalising a
    depth-3 value changes the bytes outright -- and that is asserted where it
    is true rather than written as a literal, because a literal of it went red
    on the CI lanes carrying an older numpy.
    """
    deep_numpy = [[[np.float32(0.1)]]]
    deep_plain = [[[0.1]]]

    if repr(np.float32(0.1)) != repr(0.1):
        assert _wire(deep_numpy) != _wire(deep_plain), (
            "this numpy reprs a scalar distinctly, so normalising depth 3 must "
            "change the bytes"
        )

    for value in (deep_numpy, deep_plain):
        with pytest.raises(LizyMLError) as exc:
            normalise_params({"interaction_constraints": value}, surface="probe")
        assert exc.value.code is ErrorCode.CONFIG_INVALID


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
    """The one place the accepted set is narrower than every numpy dtype.

    Some reduced-precision floats print in a form no plain Python number
    prints. Converting one anyway -- with ``.tolist()``, the obvious choice --
    sends different characters for LightGBM to parse and, for ``float32(0.1)``,
    a different number. Refusing is the half of that trade this PR is allowed
    to fall on.

    **Which values those are depends on the numpy version**, because it is
    numpy choosing the exponent form, so the witness is *found* rather than
    written down. An assertion on the literal ``1e+08`` passed here and failed
    on the two CI lanes carrying an older numpy -- the derived-versus-copied
    distinction this file argues for everywhere else, applied to itself.
    """
    witnesses = [
        value
        for value in CANDIDATES
        if isinstance(value, np.ndarray)
        and _refusal_reason(value) == REFUSAL_REASONS[0]
    ]
    if not witnesses:
        pytest.skip(
            "this numpy prints every probed float in a form a plain number "
            "also prints, so the boundary has no witness here"
        )

    hostile = witnesses[0]
    text = str(hostile[0])
    assert str(float(text)) != text, (
        f"{text!r} does have a plain stand-in; the reason is misclassified"
    )
    assert _wire(hostile) != _wire(hostile.tolist()), (
        "tolist would not have changed the bytes for this witness"
    )

    with pytest.raises(LizyMLError) as exc:
        normalise_params({"feature_contri": hostile}, surface="model.params")
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert text in exc.value.user_message


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
