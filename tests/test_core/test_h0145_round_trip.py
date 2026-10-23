"""H-0145 acceptance 1-3 (#391): a frozenset that does not read back as written
leaves V, judged on the running NumPy, and the generated code agrees.

Which values collapse depends on NumPy: ``np.uint64(4)`` beside
``np.timedelta64(4, "s")`` on NumPy 1.24, a hash-colliding large ``np.uint64``
beside ``np.timedelta64(4, "ms")`` on NumPy 2.x. Every test is parametrized by
both and skips the one that reads back here; one test requires that at least
one of them is refused on every lane, so the rule runs on each.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import decimal
import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from lizyml.codegen.values import category_value
from lizyml.core import category_values as cv
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests.test_codegen.test_h0133_codec_copy import COPY
from tests.test_core.test_category_values import SAMPLES

REASON = "a frozenset whose members do not read back as written"
#: name -> (the collapsing value, a V-safe value with the same identity key).
CASES: dict[str, tuple[frozenset[Any], frozenset[Any]]] = {
    "pair": (
        frozenset({np.uint64(4), np.timedelta64(4, "s")}),
        frozenset({4, dt.timedelta(seconds=4)}),
    ),
    "big": (
        frozenset({np.uint64(9265762820973641783), np.timedelta64(4, "ms")}),
        frozenset({9265762820973641783, dt.timedelta(milliseconds=4)}),
    ),
}


@contextlib.contextmanager
def _without_h0145() -> Iterator[None]:
    """The codec's own checks with V as it was before H-0145 (the oracle)."""
    original = cv.refusal
    cv.refusal = lambda value: cv._refusal(value, 0)  # noqa: SLF001
    try:
        yield
    finally:
        cv.refusal = original


def reads_back(value: Any) -> bool:
    """The independent oracle: the public codec writes and reads it back."""
    with _without_h0145():
        try:
            cv.decode(cv.encode(value))
        except LizyMLError:
            return False
    return True


def collapsing(name: str) -> frozenset[Any]:
    value = CASES[name][0]
    if cv.refusal(value) is None:
        pytest.skip(f"{name} reads back on NumPy {np.__version__}")
    return value


def test_this_lane_refuses_at_least_one_case() -> None:
    """Without it the rule would go untested on this NumPy (DC6)."""
    refused = [name for name, (v, _) in CASES.items() if cv.refusal(v) is not None]
    assert refused, f"no H-0145 case collapses on NumPy {np.__version__}"
    for name in refused:
        assert not reads_back(CASES[name][0])


@pytest.mark.parametrize("name", list(CASES))
def test_the_key_equal_twin_stays_in_v(name: str) -> None:
    twin = CASES[name][1]
    assert cv.refusal(twin) is None
    assert reads_back(twin)


# --- acceptance 1: refused exactly when the codec cannot read it back --------------


def _matrix() -> list[tuple[str, Any]]:
    units = [
        "Y",
        "M",
        "W",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        "ps",
        "fs",
        "as",
        "2s",
        "15m",
    ]
    out: list[tuple[str, Any]] = [
        ("decimal-zero", frozenset([np.timedelta64(0, "s"), decimal.Decimal("-0")])),
        *((name, v) for name, (v, _) in CASES.items()),
        *((f"{name}-twin", t) for name, (_, t) in CASES.items()),
        ("depth-8", frozenset({_wrap(frozenset({np.uint64(4), "x"}), 7)})),
        ("controls", frozenset({1, "1", b"1", 1.5, 2j})),
    ]
    for unit, n in itertools.product(units, (0, 1, 4)):
        td = np.timedelta64(n, unit)
        u = np.uint64(n)
        out += [
            (f"pair-{unit}-{n}", frozenset({u, td})),
            (f"three-{unit}-{n}", frozenset({u, td, "x"})),
            (f"tuple-{unit}-{n}", (u, td)),
            (f"tuple-members-{unit}-{n}", frozenset({(u,), (td,)})),
            (f"set-members-{unit}-{n}", frozenset({frozenset({u}), frozenset({td})})),
            (f"nested-{unit}-{n}", (frozenset({u, td}),)),
            (f"in-tuple-in-set-{unit}-{n}", frozenset({(frozenset({u, td}),), "y"})),
        ]
    return out


def _wrap(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = (value,)
    return value


MATRIX = _matrix()


@pytest.mark.parametrize(("label", "value"), MATRIX, ids=[m[0] for m in MATRIX])
def test_refused_exactly_when_it_does_not_read_back(label: str, value: Any) -> None:
    if cv._refusal(value, 0) is not None:  # noqa: SLF001 -- outside V before H-0145
        pytest.skip("outside V for another reason")
    assert (cv.refusal(value) is None) == reads_back(value), label
    if cv.refusal(value) is not None:
        assert cv.refusal(value) == REASON


def test_samples_pairs_are_refused_exactly_when_they_do_not_read_back() -> None:
    samples = [v for vs in SAMPLES.values() for v in vs]
    checked = 0
    for a, b in itertools.combinations(samples, 2):
        for build in (frozenset, tuple):
            try:
                value = build([a, b])
            except Exception:  # noqa: BLE001 -- not constructible on this NumPy
                continue
            if cv._refusal(value, 0) is not None:  # noqa: SLF001
                continue
            checked += 1
            assert (cv.refusal(value) is None) == reads_back(value), (a, b, build)
    assert checked > 10000


# --- acceptance 2: the generated copy answers as the library -----------------------


def _generated_admits(value: Any) -> bool:
    try:
        COPY._encode_value(value)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(("label", "value"), MATRIX, ids=[m[0] for m in MATRIX])
def test_the_generated_codec_agrees(label: str, value: Any) -> None:
    admitted = cv.refusal(value) is None
    assert _generated_admits(value) == admitted, label
    if admitted:
        written = cv.encode(value)
        assert COPY._encode_value(value) == written
        assert COPY._decode_value(written) == cv.decode(written)


def test_the_generated_codec_agrees_over_samples_pairs() -> None:
    samples = [v for vs in SAMPLES.values() for v in vs]
    for a, b in itertools.combinations(samples, 2):
        try:
            value = frozenset([a, b])
        except Exception:  # noqa: BLE001 -- not constructible on this NumPy
            continue
        assert _generated_admits(value) == (cv.refusal(value) is None), (a, b)


# --- acceptance 3: the error surfaces ----------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_encode_refuses_with_the_reason(name: str) -> None:
    value = collapsing(name)
    with pytest.raises(LizyMLError) as excinfo:
        cv.encode(value)
    assert excinfo.value.code is ErrorCode.SERIALIZATION_FAILED
    assert excinfo.value.context == {"type": "frozenset", "reason": REASON}


@pytest.mark.parametrize("name", list(CASES))
def test_the_codegen_writer_refuses_with_where(name: str) -> None:
    value = collapsing(name)
    with pytest.raises(LizyMLError) as excinfo:
        category_value(value, where="categories of column 'c'")
    assert excinfo.value.code is ErrorCode.SERIALIZATION_FAILED
    assert excinfo.value.context == {
        "where": "categories of column 'c'",
        "type": "frozenset",
        "reason": REASON,
    }


@pytest.mark.parametrize("name", list(CASES))
def test_a_written_form_reading_back_outside_v_is_refused(name: str) -> None:
    value = collapsing(name)
    with _without_h0145():
        written = cv.encode(value)
    with pytest.raises(LizyMLError) as excinfo:
        cv.decode(written)
    assert excinfo.value.code is ErrorCode.DESERIALIZATION_FAILED
    with pytest.raises(ValueError):
        COPY._decode_value(written)


@pytest.mark.parametrize("name", list(CASES))
def test_the_identity_key_is_refused_like_any_value_outside_v(name: str) -> None:
    value = collapsing(name)
    with pytest.raises(TypeError, match="outside V"):
        cv.category_identity_key(value)
    assert not COPY._in_v(value)
    with pytest.raises(TypeError):
        COPY._identity_key(value)
