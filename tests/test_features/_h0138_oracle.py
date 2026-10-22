"""The inputs of the H-0138 oracles, rebuilt from their case ids.

``docs/audits/2026-10-h0138/oracle/make_oracle.py`` froze today's answers
for these inputs; the acceptance tests rebuild each input from its id the
same way and compare the implemented library's answers with the file for the
running pandas version. A case id the rebuild does not know fails the test.
"""

from __future__ import annotations

import decimal
import gzip
import itertools
import json
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd

from lizyml.core.category_values import encode
from tests.test_core.test_category_values import SAMPLES

DATA = Path(__file__).parent / "data"
NP_ONE_TYPES = (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64, np.float32, np.float64,
                np.complex64, np.complex128)  # fmt: skip


def oracle_path() -> Path:
    return DATA / f"h0138_oracle_pandas-{pd.__version__}.json.gz"


def oracle_or_none() -> dict[str, Any] | None:
    """The oracle for the running pandas version, or ``None`` when there is
    none: the test module then fails one test instead of its collection, so
    the rest of a lane still runs."""
    path = oracle_path()
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded


def oracle() -> dict[str, Any]:
    """The oracle for the running pandas version; missing -> the test fails."""
    loaded = oracle_or_none()
    assert loaded is not None, (
        f"no H-0138 oracle for pandas {pd.__version__}: {oracle_path()}"
    )
    return loaded


def overlay_path() -> Path:
    return DATA / f"h0139_overlay_pandas-{pd.__version__}.json.gz"


def overlay_or_none() -> dict[str, Any] | None:
    """H-0139's closed list of temporal cases it changes on purpose, for the
    running pandas version (``docs/audits/2026-10-h0139/oracle/make_overlay.py``),
    or ``None`` when there is none (the overlay test then fails)."""
    path = overlay_path()
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded


def model_versions_match(meta: dict[str, Any]) -> None:
    """Model cases are compared only on the oracle's numpy and lightgbm."""
    running = (np.__version__, lightgbm.__version__)
    assert running == (meta["numpy"], meta["lightgbm"]), (
        f"H-0138 model oracle was made with numpy/lightgbm "
        f"{meta['numpy']}/{meta['lightgbm']}, this lane runs {running}"
    )


def typed(value: Any) -> Any:
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": encode(value),
    }


def _wrap(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = (value,)
    return value


def probe_values(case_id: str) -> list[Any]:
    """The values of a probe case (``s3``, ``p2-7``, ``d4-a``, ``t1-8-b``)."""
    samples = [v for vs in SAMPLES.values() for v in vs]
    kind, _, rest = case_id[0], None, case_id[1:]
    if kind == "s":
        return [samples[int(rest)]]
    if kind == "p":
        i, j = (int(x) for x in rest.split("-"))
        return [samples[i], samples[j]]
    if kind == "d":
        k, order = rest.split("-")
        pair = [decimal.Decimal("1"), NP_ONE_TYPES[int(k)](1)]
        return pair if order == "a" else pair[::-1]
    if kind == "t":
        k, depth, order = rest.split("-")
        pair = [NP_ONE_TYPES[int(k)](1), _wrap(1, int(depth))]
        return pair if order == "a" else pair[::-1]
    raise AssertionError(f"unknown H-0138 probe case id {case_id!r}")


def distinct(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for v in values:
        if not any(v is d for d in out):
            out.append(v)
    return out


def ordinary(case_id: str) -> tuple[pd.Series, list[Any]]:
    """(training series, transform values) of an ordinary-input case."""
    n = 24
    letters = ["a", "b", "c", "b", "a", "a"] * (n // 6)
    cases: dict[str, tuple[pd.Series, list[Any]]] = {
        "o-object": (pd.Series(letters, dtype=object), ["a", "b", "c", "z"]),
        "o-string": (pd.Series(letters, dtype="string"), ["a", "b", "c", "z"]),
        "o-declared": (
            pd.Series(pd.Categorical(letters, categories=["c", "a", "b", "unused"])),
            ["a", "b", "c", "unused"],
        ),
        "o-int64": (pd.Series([1, 2, 3, 2] * (n // 4), dtype="int64"), [1, 2, 3, 9]),
        "o-float64": (pd.Series([0.5, 1.5, 2.5, 1.5] * (n // 4)), [0.5, 1.5, 2.5, 9.5]),
        "o-bool": (pd.Series([True, False, True] * (n // 3)), [True, False]),
        "o-missing": (
            pd.Series(["a", None, "b", "a"] * (n // 4), dtype=object),
            ["a", "b", None, "z"],
        ),
        "o-mixed": (
            pd.Series([1, "a", 2, "a"] * (n // 4), dtype=object),
            [1, 2, "a", "z"],
        ),
    }
    assert case_id in cases, f"unknown H-0138 ordinary case id {case_id!r}"
    return cases[case_id]


def all_probe_ids() -> list[str]:
    samples = [v for vs in SAMPLES.values() for v in vs]
    ids = [f"s{i}" for i in range(len(samples))]
    ids += [f"p{i}-{j}" for i, j in itertools.combinations(range(len(samples)), 2)]
    for k in range(len(NP_ONE_TYPES)):
        ids += [f"d{k}-a", f"d{k}-b"]
        ids += [f"t{k}-{d}-{o}" for d in (1, 2, 3, 8) for o in ("a", "b")]
    return ids
