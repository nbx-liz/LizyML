"""Objective identity for tune resume (H-0122 §4).

A study's trials are only comparable while the objective they observe is the
same (INV-R1). This module builds that identity as named elements, each a
sha256 hex digest, so a mismatch can name what changed (criterion 27):

* ``data.*`` -- a summary of the frame ``tune`` actually loaded
  (:func:`summarise_frame`);
* ``config.*`` and ``metric`` -- the config parts that shape the objective
  (:func:`config_parts`).

The policy is the conservative one (Q-E): every input that may change the
objective is included, so a harmless change can be refused, but a changed
objective is never accepted. Library versions are recorded apart and only
warned about (D4, :func:`version_differences`).

Pure: no Model state is read or written here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lizyml._version import __version__
from lizyml.core._model_metrics import _DEFAULT_METRICS
from lizyml.metrics.registry import parse_metric_entry
from lizyml.tuning.lineage import objective_digest

if TYPE_CHECKING:
    from lizyml.config.schema import LizyMLConfig
    from lizyml.estimators.provider import EstimatorProvider

__all__ = [
    "CONFIG_EXCLUDED_FIELDS",
    "DATA_ELEMENTS",
    "ObjectiveIdentity",
    "VersionDifference",
    "build_identity",
    "config_parts",
    "library_versions",
    "mismatched_elements",
    "summarise_frame",
    "version_differences",
]

DATA_ELEMENTS: tuple[str, ...] = (
    "data.columns",
    "data.dtypes",
    "data.categories",
    "data.values",
    "data.categorical_codes",
    "data.object_types",
)
"""The element names :func:`summarise_frame` returns."""

CONFIG_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {"tuning", "calibration", "evaluation", "output_dir"}
)
"""Top-level config fields outside the identity (§4.2).

``evaluation`` is excluded as a whole because its only objective input, the
first metric, is the separate ``metric`` element. Every other top-level field
-- including one added later -- becomes a ``config.<field>`` element, so a new
field is included unless someone decides otherwise here.
"""

_DATA_EXCLUDED_KEYS = frozenset({"path"})
"""``data.path`` is excluded: the file's content is hashed instead (R4)."""

_LIBRARIES = ("optuna", "pandas", "scikit-learn")
"""Packages whose version is recorded besides LizyML and the provider's own."""


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _canonical(value: Any) -> Any:
    """A JSON-safe, type-tagged, order-stable form of a config value.

    ``True`` / ``1`` / ``1.0`` stay distinct; lists and tuples are one kind
    (a stored ``fixed`` comes back from JSON as lists); dict keys of mixed
    types sort without error; a numpy scalar equals its plain value; anything
    else is tagged with its type and ``repr``.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        return ["float", repr(value)]
    if isinstance(value, Mapping):
        items = [[_canonical(k), _canonical(v)] for k, v in value.items()]
        return ["dict", sorted(items, key=lambda kv: json.dumps(kv[0]))]
    if isinstance(value, list | tuple):
        return ["seq", [_canonical(v) for v in value]]
    if isinstance(value, set | frozenset):
        return ["set", sorted((_canonical(v) for v in value), key=json.dumps)]
    return ["repr", _type_name(value), repr(value)]


def _digest_value(value: Any) -> str:
    text = json.dumps(_canonical(value), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _digest_hashes(hashes: pd.Series) -> bytes:
    return hashes.to_numpy(dtype=np.uint64).tobytes()


# ---------------------------------------------------------------------------
# Data summary (§4.1 data)
# ---------------------------------------------------------------------------


def _index_info(index: pd.Index) -> list[Any]:
    """Index level dtypes and names; labels are in ``data.values``.

    The index class is left out on purpose: a ``RangeIndex`` and an ``int64``
    ``Index`` with the same labels are the same rows.
    """
    return [
        [str(d) for d in getattr(index, "dtypes", [index.dtype])],
        [[_type_name(n), str(n)] for n in index.names],
    ]


def _column_values_hash(col: pd.Series) -> bytes:
    """Row hashes of one column; R9 fallback to ``repr`` for unhashable cells."""
    try:
        return _digest_hashes(pd.util.hash_pandas_object(col, index=False))
    except TypeError:
        as_text = col.map(repr).astype(object)
        return _digest_hashes(pd.util.hash_pandas_object(as_text, index=False))


def _values_digest(df: pd.DataFrame) -> str:
    """Index labels, row order and every column's values, column by column."""
    h = hashlib.sha256()
    h.update(_digest_hashes(pd.util.hash_pandas_object(df.index)))
    for i in range(df.shape[1]):
        h.update(f"\x00col{i}\x00".encode())
        h.update(_column_values_hash(df.iloc[:, i]))
    return h.hexdigest()


def _type_names_hash(values: pd.Series | pd.Index) -> str:
    names = pd.Series(values, dtype=object).map(_type_name).astype(object)
    return hashlib.sha256(
        _digest_hashes(pd.util.hash_pandas_object(names, index=False))
    ).hexdigest()


def _categorical_parts(df: pd.DataFrame) -> tuple[list[Any], list[Any]]:
    categories: list[Any] = []
    codes: list[Any] = []
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        if not isinstance(col.dtype, pd.CategoricalDtype):
            continue
        cats = col.cat.categories
        categories.append(
            [
                i,
                bool(col.cat.ordered),
                str(cats.dtype),
                [[_type_name(v), str(v)] for v in cats],
            ]
        )
        code_hashes = pd.util.hash_pandas_object(col.cat.codes, index=False)
        codes.append([i, hashlib.sha256(_digest_hashes(code_hashes)).hexdigest()])
    return categories, codes


def _object_types(df: pd.DataFrame) -> list[Any]:
    found: list[Any] = []
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        if pd.api.types.is_object_dtype(col.dtype):
            found.append(["column", i, _type_names_hash(col)])
    for level in range(df.index.nlevels):
        labels = df.index.get_level_values(level)
        if pd.api.types.is_object_dtype(labels.dtype):
            found.append(["index", level, _type_names_hash(labels)])
    return found


def summarise_frame(df: pd.DataFrame) -> dict[str, str]:
    """Summarise the frame ``tune`` loaded, before any column is dropped.

    Every column counts, including ``features.exclude`` ones. Values, index
    labels and row order are hashed with ``pd.util.hash_pandas_object``,
    which ignores column names, categories and the type of same-looking
    values; the other elements cover exactly those gaps. Only the type-name
    extraction for object columns works per cell.
    """
    categories, codes = _categorical_parts(df)
    columns = [[_type_name(c), str(c)] for c in df.columns]
    return {
        "data.columns": _digest_value([columns, _index_info(df.index)]),
        "data.dtypes": _digest_value(
            [str(df.iloc[:, i].dtype) for i in range(df.shape[1])]
        ),
        "data.categories": _digest_value(categories),
        "data.values": _values_digest(df),
        "data.categorical_codes": _digest_value(codes),
        "data.object_types": _digest_value(_object_types(df)),
    }


# ---------------------------------------------------------------------------
# Config parts (§4.1 config, §4.2)
# ---------------------------------------------------------------------------


def config_parts(
    cfg: LizyMLConfig, fixed: Mapping[str, Any], direction: str
) -> dict[str, str]:
    """Per-part digests of the config inputs that shape the objective.

    Args:
        cfg: The validated config of the current call.
        fixed: The effective fixed parameters (``space_mode`` decides them).
        direction: The optimisation direction derived from the first metric.

    Returns:
        ``config.<field>`` for every top-level field outside
        :data:`CONFIG_EXCLUDED_FIELDS` (``data`` without ``path``), plus
        ``config.fixed`` and ``metric`` (first metric's name and arguments,
        and the direction).
    """
    dumped = cfg.model_dump(mode="python")
    parts: dict[str, str] = {}
    for name, value in dumped.items():
        if name in CONFIG_EXCLUDED_FIELDS:
            continue
        if name == "data":
            value = {k: v for k, v in value.items() if k not in _DATA_EXCLUDED_KEYS}
        parts[f"config.{name}"] = _digest_value(value)
    parts["config.fixed"] = _digest_value(dict(fixed))
    entry = (cfg.evaluation.metrics or _DEFAULT_METRICS[cfg.task])[0]
    metric_name, metric_kwargs = parse_metric_entry(entry)
    parts["metric"] = _digest_value(
        {"name": metric_name, "kwargs": metric_kwargs, "direction": direction}
    )
    return parts


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveIdentity:
    """The objective identity: named element digests and their combination."""

    digest: str
    elements: Mapping[str, str]


def build_identity(
    data: Mapping[str, str], config: Mapping[str, str]
) -> ObjectiveIdentity:
    """Combine the data summary and config parts into one identity."""
    overlap = sorted(set(data) & set(config))
    if overlap:
        raise ValueError(f"identity elements given twice: {overlap}")
    elements = {**data, **config}
    return ObjectiveIdentity(digest=objective_digest(elements), elements=elements)


def mismatched_elements(
    stored: Mapping[str, str], current: Mapping[str, str] | ObjectiveIdentity
) -> tuple[str, ...]:
    """Sorted names of elements that differ; one-sided elements count."""
    now = current.elements if isinstance(current, ObjectiveIdentity) else current
    names = set(stored) | set(now)
    return tuple(sorted(n for n in names if stored.get(n) != now.get(n)))


# ---------------------------------------------------------------------------
# Library versions (D4: recorded apart, warned about only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionDifference:
    """One package whose recorded version differs from the running one."""

    package: str
    stored: str | None
    current: str | None


def _installed_version(package: str) -> str:
    try:
        return pkg_version(package)
    except PackageNotFoundError:
        return "unknown"


def library_versions(provider: EstimatorProvider) -> dict[str, str]:
    """Versions of LizyML, Optuna, pandas, scikit-learn and the provider's deps."""
    versions = {"lizyml": __version__}
    versions.update({name: _installed_version(name) for name in _LIBRARIES})
    versions.update(provider.runtime_deps())
    return versions


def version_differences(
    stored: Mapping[str, str], current: Mapping[str, str]
) -> tuple[VersionDifference, ...]:
    """Packages whose version differs, sorted by name; one-sided ones count."""
    names = sorted(set(stored) | set(current))
    return tuple(
        VersionDifference(n, stored.get(n), current.get(n))
        for n in names
        if stored.get(n) != current.get(n)
    )
