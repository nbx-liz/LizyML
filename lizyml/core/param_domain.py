"""The closed set of parameter values that may reach the estimator.

Every parameter value a caller writes -- in ``model.params``, in
``fit(params=)``, in a restored ``tuning best_model_params``, in
``calibration.params`` -- is normalised here, once, at the surface it enters
through, and anything outside the accepted set is refused before training
starts (H-0095).

Why a closed set rather than a tolerant one
-------------------------------------------
This set was originally closed to bound a comparison. A parameter written under
two spellings used to be allowed through when the two values were *equal*, and
deciding *equal* meant answering that question about any Python object: five
consecutive review rounds (H-0094, rounds 16-20) each found one more object
whose ``__format__``, ``__class__``, ``tolist`` or ``__eq__`` answered in a way
the comparison had not anticipated. The defect was not the missing guard; it was
that the domain had no boundary, so no finite set of guards could close it.

**H-0096 removed the comparison instead**, by refusing a duplicate spelling
whatever the values are. What still needs the set closed is the two consumers
that remain, neither of which is about duplicates:

* the exit assertion at the training sites (:func:`assert_plain_params`), and
* ``export_code``, which writes the same values into ``config.json`` through
  ``json.dump`` and then encodes them as UTF-8.

The second is a defect that predates all of this work: measured on
``origin/develop`` at ``ccae32b``, ``model.params={"feature_contri":
np.array([1.0, 1.0])}`` trains and then raises ``TypeError: Object of type
ndarray is not JSON serializable`` out of ``export_code``.

One structural walk derives the normalized value, unchanged status and mapping
presence together. Both boundary predicates consume those facts (H-0098).

What "accepted" is derived from
-------------------------------
LightGBM serialises a parameter dict in ``lightgbm.basic._param_dict_to_str``,
and that function is the authority on what a value *means*, because the text it
writes is the whole of what the trainer sees. It uses two different formatters:

* a **scalar** is written with ``__format__`` (``f"{key}={val}"``);
* a **sequence element** is written with ``str`` (``_to_string``), and a
  ``list`` element of a sequence is written as ``[`` + its own elements joined
  with commas + ``]``.

Those two disagree for a value that overrides one and not the other, which is
why this module distinguishes the two positions instead of normalising by type
alone.

The accepted set is therefore the subset of the serialiser's own accepted set
on which a plain Python stand-in writes the **identical bytes**. Where no plain
value writes the same bytes -- ``str(numpy.float32(1e8))`` is ``1e+08`` and no
Python float prints that -- this module refuses, rather than let training run
on bytes the caller did not write. That refusal is the deliberate half of the
trade: the defect being hunted is training silently on a different value, so
failing loudly at the surface is the acceptable side to fall on.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from lizyml.core.exceptions import ErrorCode, LizyMLError

#: Scalar types that pass through untouched.
#:
#: Matched by **exact type**, never by ``isinstance``. A subclass may override
#: ``__format__`` -- the serialiser's scalar formatter -- and print bytes its
#: base class would not, so a subclass instance can compare equal to a plain
#: value here and then train as something else (H-0094 review round 20). The
#: measured cost of refusing subclasses over this repository's own suite is
#: zero: no parameter value it constructs is one.
PLAIN_SCALAR_TYPES: tuple[type, ...] = (
    type(None),
    bool,
    int,
    float,
    str,
)

#: Path types accepted at the surface, and **converted to their text**.
#:
#: The serialiser writes a path with the scalar formatter, which for a path is
#: its text, so the bytes are the same either way -- and the text is the form
#: everything downstream can carry. A path survived normalisation as a path
#: until now, and `export_code` then raised `TypeError: Object of type
#: PosixPath is not JSON serializable` on a run that had trained happily.
#:
#: Only the flavours of `Path`. `_param_dict_to_str` tests
#: `isinstance(val, Path)`, and a `PurePosixPath` is not a `Path` -- it was
#: accepted here and the serialiser raised on it (review round 24).
#: The conversion needs no check of its own, and that is a claim about these
#: types rather than about paths in general: ``pathlib`` defines no
#: ``__format__``, so ``format(p, "")`` is ``object.__format__``, which is
#: ``str(p)``. A test asserts that for each type here. A subclass could define
#: one, and a subclass is not admitted -- the exact-type gate is what makes the
#: conversion safe, so putting a second check inside the branch would be a
#: guard nothing can reach.
PATH_TYPES: tuple[type, ...] = (pathlib.PosixPath, pathlib.WindowsPath)


def _derived_numpy_scalar_types() -> frozenset[type]:
    """The numpy scalar types a parameter value may be, derived from numpy.

    Derived rather than listed, so a numpy upgrade that adds a scalar type is
    visible here instead of falling through a test nobody revisited. Only the
    bases whose members are *values a parameter can hold* -- a number, a truth,
    a name -- are admitted, so ``datetime64``, ``complex128``, ``void`` and
    ``bytes_`` are outside by construction.

    **The deciding step is the dtype round trip, not the enumeration.** Reading
    a namespace is not enough on its own: ``vars(numpy)`` is an ordinary module
    dict, and a caller who assigns into it before this module is first imported
    puts their own class in the set -- measured. So a candidate is kept only
    when numpy own dtype machinery resolves it **back to itself**, which a
    subclass does not: ``numpy.dtype(a float64 subclass).type`` is
    ``numpy.float64``. Enumeration only has to produce a superset; the round
    trip is what closes it.

    **The bound, stated rather than implied.** This closes the domain against
    parameter *values*. It is not a sandbox against a caller who has already
    replaced part of numpy in the running process -- one who can rebind
    ``numpy.dtype`` can also rebind ``numpy.float64``, or this module. Naming
    that limit is the point: an earlier form of this docstring claimed the
    stronger thing, and the stronger thing is not attainable in Python.
    """
    candidates: set[type] = set()
    for namespace in (vars(np), getattr(np, "sctypeDict", {})):
        for exported in namespace.values():
            if isinstance(exported, type):
                candidates.add(exported)

    accepted: set[type] = set()
    for kind in candidates:
        if not issubclass(kind, (np.integer, np.floating, np.bool_, np.str_)):
            continue
        try:
            resolved = np.dtype(kind).type
        except Exception:  # noqa: BLE001 - a candidate numpy cannot resolve
            continue
        if resolved is kind:
            accepted.add(kind)
    return frozenset(accepted)


#: The numpy scalar types a parameter value may be, by **exact type**.
NUMPY_SCALAR_TYPES: frozenset[type] = _derived_numpy_scalar_types()

#: ``numpy.ndarray`` is admitted by exact type for the same reason, and it is a
#: separate statement because it was a separate hole. Iterating a subclass runs
#: the subclass ``__iter__``, and what a caller method returns is not required
#: to be the same twice: measured, a subclass yielding a different sequence on
#: each call was normalised to one thing and would have been serialised as
#: another. Reading a plain array runs numpy code only.


#: Sequence types accepted here, alongside a 1-D ndarray.
#:
#: The serialiser also joins a ``set``, and this deliberately does not accept
#: one. Every sequence parameter LightGBM takes is **positional** --
#: ``feature_contri`` reads element *i* as feature *i* -- and a set has no
#: order, so the parameter it becomes is decided by hash order. Normalising it
#: to ``list(value)`` would write the same bytes the serialiser would have
#: written, but it would also make ``{1.0, 2.0}`` and a literal ``[1.0, 2.0]``
#: one value **by hash-order coincidence**, which is the answer the old
#: comparison refused to give for the same reason. Refusing is the narrowing
#: half, and it is measured: 0 of 1518 parameter values this repository
#: constructs is a set.
PLAIN_SEQUENCE_TYPES: tuple[type, ...] = (list, tuple)

#: Joined by the serialiser and refused here, with the reason above.
REFUSED_SEQUENCE_TYPES: tuple[type, ...] = (set, frozenset)

#: What a caller is told they may write.
ACCEPTED_DESCRIPTION = (
    "None, bool, int, float, str, pathlib.Path, a numpy scalar, a list, tuple "
    "or 1-D numpy array of those, or a dict with str keys holding those "
    "(a set is refused: a sequence parameter is positional and a set has no "
    "order; a path is accepted and carried on as its text)"
)


def _is_one_of(value: Any, kinds: Any) -> bool:
    """Is ``type(value)`` **identically** one of ``kinds``?

    Not ``type(value) in kinds``. Membership in a ``set`` or a ``tuple`` is
    decided by ``__hash__`` and ``__eq__``, and for a *class* those come from
    its metaclass -- which a caller writes. A metaclass answering
    ``hash(numpy.float64)`` and ``__eq__`` true passed the check with no numpy
    base, no claimed ``__module__``, and no dependence on import order, and its
    own ``__format__`` and ``item()`` then ran inside normalisation (found by a
    read-only checker after review round 22).

    ``is`` is the only comparison in Python a caller cannot participate in, so
    it is the only one this gate can be built from.
    """
    return any(type(value) is kind for kind in kinds)


class _Unaccepted(Exception):
    """A value outside the accepted set, carried back to the surface."""

    def __init__(self, value: Any, reason: str) -> None:
        super().__init__(reason)
        self.value = value
        self.reason = reason


def _describe(value: Any) -> str:
    """Name a rejected value by type, without asking the value to print itself.

    A rejected value is by definition one whose ``__format__``/``__str__`` are
    not trusted, so the message is built from its type and never from it.
    """
    return type(value).__name__


def _written_or_refused(value: Any, write: Any) -> str:
    """The characters ``write`` produces for ``value``, or refuse the value.

    Being an accepted *type* is not the same as being a value the serialiser
    can write. A Python ``int`` has no width, and above the interpreter decimal
    limit ``str`` of one raises rather than returning digits -- so ``10 ** 5000``
    was accepted here, passed the assertion before training, and made LightGBM
    raise from inside (review round 24). Asking for the characters is the only
    way to know there are any.
    """
    try:
        written: str = write(value)
    except Exception as unwritable:  # noqa: BLE001 - the answer is the refusal
        raise _Unaccepted(
            value,
            f"the estimator cannot be sent it: {type(unwritable).__name__}",
        ) from None
    return _encodable_or_refused(value, written)


def _encodable_or_refused(value: Any, written: str) -> str:
    """Refuse characters neither consumer can encode.

    Having characters is not the same as having characters that can be sent.
    Both consumers encode UTF-8 and neither was asked: LightGBM hands the
    serialised string to ``_c_str``, and the codegen writer opens ``config.json``
    with ``encoding="utf-8"``. A lone surrogate is a ``str`` of the accepted
    type, and it passed normalisation, the assertion before training and the
    json oracle before raising ``UnicodeEncodeError`` inside each consumer
    (review round 25).

    Python is the only place in this pipeline where such a string exists, so
    this is the only place it can be refused by name rather than by traceback.
    """
    try:
        written.encode("utf-8")
    except UnicodeEncodeError:
        raise _Unaccepted(
            value, "the characters it writes cannot be encoded as UTF-8"
        ) from None
    return written


def _plain_scalar(value: Any) -> Any:
    """The plain stand-in for a value in the serialiser's *scalar* position.

    The scalar formatter is ``__format__``, and ``.item()`` on a numpy scalar
    preserves it for every dtype measured (see the wire-preservation test).
    """
    if _is_one_of(value, PLAIN_SCALAR_TYPES):
        if value is not None:
            _written_or_refused(value, lambda item: format(item, ""))
        return value
    if _is_one_of(value, PATH_TYPES):
        return _encodable_or_refused(value, str(value))
    if _is_one_of(value, NUMPY_SCALAR_TYPES):
        # Read **before** the conversion. A value asked the same question twice
        # need not answer the same way, and review round 22 built one that did
        # not: its `__format__` returned `0.9` until `item()` set a flag and
        # `0.1` afterwards, so a check that read it second compared the
        # conversion against the conversion. Nothing that reaches here can do
        # that any more -- the type set holds only types numpy exports -- but
        # reading first is free, and a check whose correctness depends on the
        # other check having worked is not a second defence.
        written = _written_or_refused(value, lambda item: format(item, ""))
        plain = value.item()
        if not _is_one_of(plain, PLAIN_SCALAR_TYPES):
            raise _Unaccepted(
                value, f"numpy scalar of dtype {value.dtype} is not plain"
            )
        # The conversion is **checked**, not trusted. `.item()` preserves the
        # scalar formatter for every dtype measured, but `numpy.timedelta64` is
        # a `numpy.integer` whose `__format__` writes `1 nanoseconds` where its
        # converted value writes `1` -- accepted and silently retrained, until
        # review round 21 measured it. Asking `format` directly is the same
        # question the serialiser asks, on this value.
        if format(plain, "") != written:
            raise _Unaccepted(
                value,
                f"it writes {written!r} and no plain value it converts to does",
            )
        return plain
    raise _Unaccepted(value, f"{_describe(value)} is not an accepted scalar")


def _plain_element(value: Any) -> Any:
    """The plain stand-in for a value in the serialiser's *element* position.

    The element formatter is ``str``, so the stand-in is the plain value whose
    ``str`` is the same text. For most numpy scalars ``.item()`` is that value;
    for the reduced-precision floats it is not, because numpy chooses an
    exponent form Python would not -- ``str(numpy.float32(1e8))`` is ``1e+08``
    and ``str(1e8)`` is ``100000000.0`` -- and there the text itself is parsed
    back. Where no plain value prints that text, this refuses.
    """
    if _is_one_of(value, PLAIN_SCALAR_TYPES):
        # `str`, because that is the element formatter, and for the same reason
        # as in scalar position: an accepted type is not by itself a value the
        # serialiser can write.
        _written_or_refused(value, str)
        return value
    if _is_one_of(value, PATH_TYPES):
        return _encodable_or_refused(value, str(value))
    if _is_one_of(value, NUMPY_SCALAR_TYPES):
        # Through the same gate as every other position. This branch read the
        # text directly, so a `numpy.str_` holding a lone surrogate normalised
        # into a plain string neither consumer can encode -- and normalising
        # that result again refused it, which broke idempotence as well
        # (review round 26).
        text = _written_or_refused(value, str)
        for candidate in _element_candidates(value, text):
            if _is_one_of(candidate, PLAIN_SCALAR_TYPES) and str(candidate) == text:
                return candidate
        raise _Unaccepted(
            value,
            f"no plain value prints as {text!r}, which is what the estimator "
            "would have been sent",
        )
    raise _Unaccepted(value, f"{_describe(value)} is not an accepted element")


def _element_candidates(value: Any, text: str) -> list[Any]:
    """Plain values that might print exactly as ``text``."""
    candidates: list[Any] = [value.item()]
    for build in (int, float):
        try:
            candidates.append(build(text))
        except (TypeError, ValueError):
            continue
    return candidates


@dataclass(frozen=True)
class _Normalization:
    value: Any
    unchanged: bool
    contains_mapping: bool


def _walk(value: Any, *, position: str = "value") -> _Normalization:
    """Normalize and derive boundary facts in the same structural dispatch.

    A top-level sequence uses member formatting. A nested list uses element
    formatting; deeper containers are refused by that scalar formatter. Mapping
    values restart at value position because adapters consume those mappings.
    """
    if position != "element" and type(value) is dict:
        mapping: dict[str, Any] = {}
        unchanged = True
        for key, member in value.items():
            if type(key) is not str:
                raise _Unaccepted(value, f"a mapping key is a {_describe(key)}")
            _encodable_or_refused(value, key)
            child = _walk(member)
            mapping[key] = child.value
            unchanged = unchanged and child.unchanged
        return _Normalization(mapping, unchanged, True)
    sequence = position == "value" and (
        type(value) is np.ndarray or _is_one_of(value, PLAIN_SEQUENCE_TYPES)
    )
    nested = position == "member" and type(value) is list
    if sequence or nested:
        if type(value) is np.ndarray and len(value.shape) != 1:
            raise _Unaccepted(
                value, f"a {len(value.shape)}-D numpy array is not a parameter value"
            )
        members: list[Any] = []
        unchanged = type(value) is list
        contains_mapping = False
        child_position = "element" if nested else "member"
        for member in value:
            child = _walk(member, position=child_position)
            members.append(child.value)
            unchanged = unchanged and child.unchanged
            contains_mapping = contains_mapping or child.contains_mapping
        return _Normalization(members, unchanged, contains_mapping)
    plain = _plain_scalar(value) if position == "value" else _plain_element(value)
    return _Normalization(plain, plain is value, False)


def normalise_value(value: Any) -> Any:
    """Return the plain stand-in, raising _Unaccepted outside the domain."""
    return _walk(value).value


def is_plain(value: Any) -> bool:
    """Whether the serializer can consume the value unchanged, without mappings."""
    try:
        result = _walk(value)
    except _Unaccepted:
        return False
    return result.unchanged and not result.contains_mapping


def is_accepted(value: Any) -> bool:
    """Whether the value is already normalized at the wider surface boundary."""
    try:
        return _walk(value).unchanged
    except _Unaccepted:
        return False


def normalise_params(params: dict[str, Any], *, surface: str) -> dict[str, Any]:
    """Return ``params`` with every value replaced by its plain stand-in.

    Args:
        params: the parameters exactly as the caller wrote them.
        surface: the input they arrived through, named in any refusal.

    Returns:
        A new dict. Keys are untouched; every value is a plain Python value.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming every rejected parameter.
    """
    normalised: dict[str, Any] = {}
    rejected: list[tuple[str, str, str]] = []
    for name, value in params.items():
        try:
            normalised[name] = normalise_value(value)
        except _Unaccepted as unaccepted:
            rejected.append((name, _describe(unaccepted.value), unaccepted.reason))
    if not rejected:
        return normalised
    lines = [
        f"  {surface}: '{name}' is a {kind} -- {reason}."
        for name, kind, reason in sorted(rejected)
    ]
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            "Parameter value(s) the estimator cannot be given as written:\n"
            + "\n".join(lines)
            + f"\nWrite {ACCEPTED_DESCRIPTION}."
        ),
        context={
            "rejected": [
                {"surface": surface, "parameter": name, "type": kind, "reason": reason}
                for name, kind, reason in sorted(rejected)
            ]
        },
    )


def assert_plain_params(params: dict[str, Any], *, where: str) -> None:
    """Refuse to hand the estimator a value that never passed a surface.

    Normalising at the four surfaces is a claim about wiring, and a claim about
    wiring is exactly what fails silently when a fifth route is added later
    (DC4). This turns it into a property: the two places that call
    ``lgb.train`` check it, so such a value stops the run and names itself
    instead of training on bytes nobody chose.

    Its set is **narrower** than what :func:`normalise_params` accepts, so a
    value here can be one that did pass a surface -- a mapping, which the
    adapter is supposed to have consumed. The message says what is true of
    every case: this value cannot be given to the estimator.

    Raises:
        LizyMLError: with ``CONFIG_INVALID``, naming every offending parameter.
    """
    offending = sorted(
        (name, _describe(value))
        for name, value in params.items()
        if not is_plain(value)
    )
    if not offending:
        return
    lines = [f"  '{name}' is a {kind}." for name, kind in offending]
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"Parameter value(s) the estimator cannot be given reached {where}:"
            "\n" + "\n".join(lines) + f"\nAccepted: {ACCEPTED_DESCRIPTION}."
        ),
        context={
            "where": where,
            "unnormalised": [
                {"parameter": name, "type": kind} for name, kind in offending
            ],
        },
    )
