"""The closed set of parameter values that may reach the estimator.

Every parameter value a caller writes -- in ``model.params``, in
``fit(params=)``, in a restored ``tuning best_model_params``, in
``calibration.params`` -- is normalised here, once, at the surface it enters
through, and anything outside the accepted set is refused before training
starts (H-0095).

Why a closed set rather than a tolerant one
-------------------------------------------
:func:`lizyml.core.value_equality.values_differ` has to answer whether two
writings of one parameter mean the same value, and it used to be asked that
question about *any* Python object. Five consecutive review rounds (H-0094,
rounds 16-20) each found one more object whose ``__format__``, ``__class__``,
``tolist`` or ``__eq__`` answered in a way the comparison had not anticipated,
and every fix was correct for the object the round named and silent about the
next one. The defect was not the missing guard; it was that the domain had no
boundary, so no finite set of guards could close it.

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
    # `Path()` instantiates a concrete flavour, so the exact types are listed
    # rather than the abstract base.
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
    pathlib.PosixPath,
    pathlib.WindowsPath,
)


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
    "order)"
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


def _plain_scalar(value: Any) -> Any:
    """The plain stand-in for a value in the serialiser's *scalar* position.

    The scalar formatter is ``__format__``, and ``.item()`` on a numpy scalar
    preserves it for every dtype measured (see the wire-preservation test).
    """
    if _is_one_of(value, PLAIN_SCALAR_TYPES):
        return value
    if _is_one_of(value, NUMPY_SCALAR_TYPES):
        # Read **before** the conversion. A value asked the same question twice
        # need not answer the same way, and review round 22 built one that did
        # not: its `__format__` returned `0.9` until `item()` set a flag and
        # `0.1` afterwards, so a check that read it second compared the
        # conversion against the conversion. Nothing that reaches here can do
        # that any more -- the type set holds only types numpy exports -- but
        # reading first is free, and a check whose correctness depends on the
        # other check having worked is not a second defence.
        written = format(value, "")
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
        return value
    if _is_one_of(value, NUMPY_SCALAR_TYPES):
        text = str(value)
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


def _plain_member(member: Any) -> Any:
    """The stand-in for one member of a top-level sequence.

    A ``list`` member is the one nested form the serialiser gives meaning to:
    ``_to_string`` writes it as its own elements joined with commas, which is
    how ``interaction_constraints`` is spelled. A ``tuple``, ``set`` or array
    in that position is written with Python's or numpy's repr instead --
    ``(1, 2)``, ``[1 2]`` -- which LightGBM cannot read, and rewriting it to a
    list would change the bytes. Both are refused rather than guessed at.
    """
    if type(member) is list:
        # One level, not recursion. `_to_string` writes a nested `list` with
        # its own bracketed form and writes **that** list's members with
        # `str`, so depth 2 is where the serialiser stops giving meaning. A
        # third level is written by Python's list repr, where a normalised
        # member prints differently from the value the caller wrote -- measured:
        # `[[[numpy.float32(0.1)]]]` writes `[[np.float32(0.1)]]` and its
        # normalised form writes `[[0.1]]`.
        return [_plain_element(inner) for inner in member]
    if type(member) is dict:
        return _plain_mapping(member)
    return _plain_element(member)


def _plain_mapping(value: dict[Any, Any]) -> dict[str, Any]:
    """The plain stand-in for a LizyML-level mapping value.

    LightGBM has no mapping form -- ``_param_dict_to_str`` raises on a ``dict``
    in scalar position and writes Python repr for one inside a sequence -- so
    nothing here is about the wire. This is about LizyML values that the adapter
    **consumes** before serialising: a metric entry is written as
    ``{"precision_at_k": {"k": 15}}`` or inside a list beside plain names
    (H-0065), and ``_build_params`` turns those into evaluation functions and
    removes them.

    So a mapping is accepted at the surface and refused at ``lgb.train``: the
    two ends have different accepted sets on purpose, and the assertion there is
    what says a mapping never survived the adapter. Keys must be exact ``str``,
    and values are normalised like any other, so the closure holds through them.
    """
    normalised: dict[str, Any] = {}
    for key, member in value.items():
        if type(key) is not str:
            raise _Unaccepted(value, f"a mapping key is a {_describe(key)}")
        normalised[key] = normalise_value(member)
    return normalised


def _plain_sequence(value: Any) -> list[Any]:
    """The plain stand-in for a sequence the serialiser joins with commas."""
    if type(value) is np.ndarray:
        # `len(shape)`, which is what `_is_numpy_1d_array` reads, rather than
        # `ndim`: two attributes that agree on a numpy array and need not agree
        # on anything else, and the serialiser reads the first.
        if len(value.shape) != 1:
            raise _Unaccepted(
                value, f"a {len(value.shape)}-D numpy array is not a parameter value"
            )
        members: list[Any] = list(value)
    else:
        members = list(value)
    return [_plain_member(member) for member in members]


def normalise_value(value: Any) -> Any:
    """Return the plain stand-in for one parameter value.

    Raises:
        _Unaccepted: when the value is outside the accepted set.
    """
    if type(value) is np.ndarray or _is_one_of(value, PLAIN_SEQUENCE_TYPES):
        return _plain_sequence(value)
    if type(value) is dict:
        return _plain_mapping(value)
    return _plain_scalar(value)


def is_plain(value: Any) -> bool:
    """Whether ``value`` is one the **serialiser** can be handed as it is.

    Narrower than what :func:`normalise_params` accepts, and deliberately: a
    mapping is a LizyML-level value the adapter consumes (a metric entry), and
    one that survives to the trainer is a defect, not a parameter.
    """
    if _is_one_of(value, PLAIN_SCALAR_TYPES):
        return True
    if type(value) is not list:
        return False
    return all(
        _is_one_of(member, PLAIN_SCALAR_TYPES)
        or (
            type(member) is list
            and all(_is_one_of(inner, PLAIN_SCALAR_TYPES) for inner in member)
        )
        for member in value
    )


def is_accepted(value: Any) -> bool:
    """Whether ``value`` is already inside the **surface** set, unchanged.

    Wider than :func:`is_plain`, by exactly the mapping: a metric entry is a
    LizyML value the adapter consumes, so it passes here and is refused at the
    trainer. Keeping the two predicates apart is what lets each say something
    true; one predicate covering both ends would have to be the looser of them,
    and the looser one is not the bound the trainer needs.
    """
    if _is_one_of(value, PLAIN_SCALAR_TYPES):
        return True
    if type(value) is dict:
        return all(
            type(key) is str and is_accepted(member) for key, member in value.items()
        )
    if type(value) is not list:
        return False
    return all(
        _is_one_of(member, PLAIN_SCALAR_TYPES)
        or (type(member) is dict and is_accepted(member))
        or (
            type(member) is list
            and all(_is_one_of(inner, PLAIN_SCALAR_TYPES) for inner in member)
        )
        for member in value
    )


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
