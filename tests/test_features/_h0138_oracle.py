"""The inputs of the H-0138 oracles, rebuilt from their case ids.

``docs/audits/2026-10-h0138/oracle/make_oracle.py`` froze today's answers
for these inputs; the acceptance tests rebuild each input from its id the
same way and compare the implemented library's answers with the file for the
running pandas version. A case id the rebuild does not know fails the test.

H-0144: only the latest-deps CI job (:data:`MODE_ENV` set and the job named)
may fall back to the nearest older oracle of the same pandas minor; every
other lane compares its own version's file or fails.
"""

from __future__ import annotations

import decimal
import gzip
import importlib.metadata
import itertools
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from packaging.version import InvalidVersion, Version

from lizyml.core.category_values import encode
from tests.test_core.test_category_values import SAMPLES

DATA = Path(__file__).parent / "data"
NP_ONE_TYPES = (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64, np.float32, np.float64,
                np.complex64, np.complex128)  # fmt: skip


#: H-0144: the latest-deps CI job may compare a pandas without its own oracle
#: against the nearest older oracle of the same minor. Accepted only in that
#: job; anywhere else a set value fails the module instead of loosening it.
MODE_ENV = "LIZYML_H0138_ORACLE_MODE"
SAME_MINOR = "latest-same-minor"
LATEST_JOB = "quality-latest-deps"
#: Set by the test module for the answers subprocess: "0" computes no model.
MODELS_ENV = "LIZYML_H0138_MODELS"
_ORACLE = ("h0138_oracle_pandas-", ".json.gz")
#: What the oracle module compared, printed in the terminal summary (H-0144).
SUMMARY: dict[str, Any] = {}


@dataclass(frozen=True)
class Resolved:
    """Which oracle a lane compares against, decided before anything runs."""

    running: str
    mode: str | None
    version: str | None
    error: str | None

    @property
    def exact(self) -> bool:
        return self.version == self.running


def oracle_mode(environ: Mapping[str, str]) -> str | None:
    """``None`` (strict) or :data:`SAME_MINOR`; a misuse raises ``ValueError``."""
    value = environ.get(MODE_ENV)
    if value is None:
        return None
    if value != SAME_MINOR:
        raise ValueError(f"{MODE_ENV}={value!r} is not a known mode ({SAME_MINOR!r})")
    if (
        environ.get("GITHUB_ACTIONS") != "true"
        or environ.get("GITHUB_JOB") != LATEST_JOB
    ):
        raise ValueError(
            f"{MODE_ENV}={SAME_MINOR} is accepted only in the CI job {LATEST_JOB!r}"
            f" (GITHUB_ACTIONS={environ.get('GITHUB_ACTIONS')!r},"
            f" GITHUB_JOB={environ.get('GITHUB_JOB')!r})"
        )
    return value


def oracle_versions(data: Path = DATA) -> list[str]:
    """The pandas versions that have an H-0138 oracle file."""
    head, tail = _ORACLE
    return sorted(p.name[len(head) : -len(tail)] for p in data.glob(f"{head}*{tail}"))


def select_version(running: str, available: list[str], mode: str | None) -> str | None:
    """The running version's own oracle, else (same-minor mode only) the
    newest older version of the same epoch, major and minor; ``None`` when
    there is none. An unparsable version raises ``InvalidVersion``."""
    if running in available:
        return running
    if mode != SAME_MINOR:
        return None
    run = Version(running)
    best: tuple[Version, str] | None = None
    for text in available:
        cand = Version(text)
        same_minor = (cand.epoch, cand.major, cand.minor) == (
            run.epoch,
            run.major,
            run.minor,
        )
        if same_minor and cand < run and (best is None or cand > best[0]):
            best = (cand, text)
    return None if best is None else best[1]


def resolve(
    environ: Mapping[str, str] | None = None,
    running: str | None = None,
    data: Path = DATA,
) -> Resolved:
    """The oracle this lane compares against (see :func:`select_version`)."""
    env = os.environ if environ is None else environ
    run = pd.__version__ if running is None else running
    try:
        mode = oracle_mode(env)
        version = select_version(run, oracle_versions(data), mode)
    except (ValueError, InvalidVersion) as exc:
        return Resolved(run, env.get(MODE_ENV), None, str(exc))
    return Resolved(run, mode, version, None)


def oracle_path(version: str | None = None) -> Path:
    return DATA / f"{_ORACLE[0]}{version or pd.__version__}{_ORACLE[1]}"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded


def oracle_or_none(resolved: Resolved | None = None) -> dict[str, Any] | None:
    """The selected oracle, or ``None`` when there is none: the test module
    then fails one test instead of its collection, so the rest of a lane
    still runs."""
    chosen = resolve() if resolved is None else resolved
    return None if chosen.version is None else _load(oracle_path(chosen.version))


def oracle() -> dict[str, Any]:
    """The selected oracle; none (or a misused mode) -> the caller fails."""
    chosen = resolve()
    loaded = oracle_or_none(chosen)
    assert loaded is not None, (
        f"no H-0138 oracle for pandas {chosen.running} "
        f"(mode {chosen.mode}, {chosen.error or 'no candidate'}): {oracle_path()}"
    )
    return loaded


def overlay_path(version: str | None = None) -> Path:
    return DATA / f"h0139_overlay_pandas-{version or pd.__version__}.json.gz"


def overlay_or_none(resolved: Resolved | None = None) -> dict[str, Any] | None:
    """H-0139's closed list of temporal cases it changes on purpose, for the
    selected version (``docs/audits/2026-10-h0139/oracle/make_overlay.py``),
    or ``None`` when there is none (the overlay test then fails). Never the
    overlay of another version than the oracle's."""
    chosen = resolve() if resolved is None else resolved
    return None if chosen.version is None else _load(overlay_path(chosen.version))


def compare_models(meta: dict[str, Any], mode: str | None) -> bool:
    """Whether the model cases run: always on a strict lane (a version
    mismatch then fails), and in same-minor mode only on the oracle's numpy
    and lightgbm (decided before the answers subprocess computes any model)."""
    same = (np.__version__, lightgbm.__version__) == (meta["numpy"], meta["lightgbm"])
    return same or mode != SAME_MINOR


def typed(value: Any) -> Any:
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": encode(value),
    }


def pyarrow_version() -> str | None:
    """pyarrow decides how pandas holds strings, so the summary names it."""
    try:
        return importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError:
        return None


def model_versions_match(meta: dict[str, Any]) -> None:
    """Model cases are compared only on the oracle's numpy and lightgbm."""
    running = (np.__version__, lightgbm.__version__)
    assert running == (meta["numpy"], meta["lightgbm"]), (
        f"H-0138 model oracle was made with numpy/lightgbm "
        f"{meta['numpy']}/{meta['lightgbm']}, this lane runs {running}"
    )


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
