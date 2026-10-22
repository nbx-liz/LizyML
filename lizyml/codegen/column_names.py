"""column_names -- the column-name aliases of export_code projects (H-0141).

A project ``export_code`` writes names every column by a string alias, so a
JSON key, a CSV header and a parquet column name all read back as the name
the scripts look up (#382). The alias of an admitted name is ``str(name)``;
the admitted types are exactly ``str``, ``numpy.str_``, ``int``, ``float``,
``bool``, ``None`` and ``np.float64``. The fitted model, its Config and the
saved artifacts keep their own names: only the copy written into the project
is aliased.

:class:`UnsupportedColumnName` and :func:`column_alias` are copied into the
generated scripts verbatim (``column_name_template``), so they import nothing
but ``numpy``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from lizyml.core.category_values import shown
from lizyml.core.exceptions import ErrorCode, LizyMLError

#: The marker ``config.json`` and ``pipeline_state.json`` carry (H-0141).
COLUMN_NAME_POLICY = "str_alias_v1"


class UnsupportedColumnName(ValueError):
    """A column name an export_code project cannot hold (H-0141)."""

    def __init__(self, kind: str, name: object) -> None:
        try:
            text = repr(name)
        except Exception:  # noqa: BLE001 - a hostile __repr__ must not hide the name
            text = f"<unprintable {type(name).__name__}>"
        super().__init__(f"unsupported column name ({kind}): {text}")
        self.reason = "unsupported_column_name"
        self.kind = kind


def column_alias(name: object) -> str:
    """The project's string alias of a column *name* (H-0141).

    Raises:
        UnsupportedColumnName: ``kind`` ``"type"`` for a name outside the
            admitted types, ``"empty_alias"`` for an empty alias and
            ``"invalid_alias_text"`` for a NUL or text UTF-8 cannot encode.
    """
    admitted = (str, np.str_, int, float, bool, type(None), np.float64)
    if not any(type(name) is kind for kind in admitted):
        raise UnsupportedColumnName("type", name)
    alias = str(name)
    if alias == "":
        raise UnsupportedColumnName("empty_alias", name)
    if "\x00" in alias:
        raise UnsupportedColumnName("invalid_alias_text", name)
    try:
        alias.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnsupportedColumnName("invalid_alias_text", name) from exc
    return alias


def _json_key(name: Any) -> str | None:
    """The key ``json.dumps`` writes for *name* as an object key, or None."""
    try:
        return str(next(iter(json.loads(json.dumps({name: 0})))))
    except (TypeError, ValueError):
        return None


def _same(a: Any, b: Any) -> bool:
    """Same exact type and equal (NaN equal to NaN)."""
    if type(a) is not type(b):
        return False
    if a is b:
        return True
    try:
        if a == b:
            return True
    except Exception:  # noqa: BLE001 - an uncomparable name matches nothing
        return False
    return isinstance(a, float) and a != a and b != b  # both NaN


@dataclass(frozen=True)
class _Item:
    where: str
    position: int | None
    name: Any


def _role_items(
    config_normalized: Mapping[str, Any], inner_valid: Mapping[str, Any] | None
) -> list[_Item]:
    """The Config role names, in the order the refusals are reported.

    ``inner_valid`` is the exported inner split (``exported_inner_valid``):
    its ``group_col`` is the column the refit grouped by.
    """
    data = config_normalized.get("data") or {}
    features = config_normalized.get("features") or {}
    split = config_normalized.get("split") or {}
    inner = inner_valid or {}
    items: list[_Item] = []
    for where, value in (
        ("data.target", data.get("target")),
        ("data.time_col", data.get("time_col")),
        ("data.group_col", data.get("group_col")),
    ):
        if value is not None:
            items.append(_Item(where, None, value))
    for where, values in (
        ("features.exclude", features.get("exclude") or []),
        ("features.categorical", features.get("categorical") or []),
    ):
        items.extend(_Item(where, i, v) for i, v in enumerate(values))
    for where, block in (
        ("split.blocks.col", split.get("blocks")),
        ("split.groups.col", split.get("groups")),
    ):
        if isinstance(block, Mapping) and block.get("col") is not None:
            items.append(_Item(where, None, block["col"]))
    if inner.get("group_col") is not None:
        items.append(_Item("inner_valid.group_col", None, inner["group_col"]))
    return items


def _descriptor(item: _Item, alias: str | None) -> dict[str, Any]:
    return {
        "where": item.where,
        "position": item.position,
        "type": f"{type(item.name).__module__}.{type(item.name).__qualname__}",
        "repr": shown(item.name),
        "alias": alias,
    }


def _refuse(
    reason: str, kind: str, alias: str | None, columns: list[dict[str, Any]]
) -> LizyMLError:
    names = ", ".join(c["repr"] for c in columns)
    detail = (
        f"column names {names} share the alias {alias!r}"
        if reason == "column_name_collision"
        else f"column name {names} cannot be written ({kind})"
    )
    return LizyMLError(
        code=ErrorCode.SERIALIZATION_FAILED,
        user_message=(
            f"export_code cannot write this project: {detail}. A project names "
            "every column by the string alias str(name) (H-0141); rename the "
            "column to a str and fit again."
        ),
        context={
            "surface": "export_code",
            "reason": reason,
            "kind": kind,
            "alias": alias,
            "columns": columns,
        },
    )


@dataclass(frozen=True)
class ColumnNameProjection:
    """The aliases of one project: features in order, and declared keys."""

    feature_names: tuple[Any, ...]
    aliases: tuple[str, ...]
    declared: tuple[tuple[Any, str], ...]

    def alias(self, name: Any) -> str:
        """The alias of the feature *name* (exact type and value); a name
        outside the features gets its alias by the same rule."""
        for label, alias in zip(self.feature_names, self.aliases, strict=True):
            if _same(label, name):
                return alias
        try:
            return column_alias(name)
        except UnsupportedColumnName as exc:
            item = _Item("pipeline_state", None, name)
            raise _refuse(
                exc.reason, exc.kind, None, [_descriptor(item, None)]
            ) from exc

    def declared_alias(self, key: Any) -> str:
        for k, alias in self.declared:
            if k is key or _same(k, key):
                return alias
        raise KeyError(key)  # pragma: no cover - callers pass declared keys


def alias_pipeline_state(
    state: Mapping[str, Any], projection: ColumnNameProjection
) -> dict[str, Any]:
    """A copy of a pipeline state whose column names are the project's aliases.

    Category values, modes and every other field are copied unchanged; the
    fitted state itself is never modified.
    """
    out = dict(state)
    for field in ("feature_names", "categorical_cols"):
        if isinstance(state.get(field), list):
            out[field] = [projection.alias(n) for n in state[field]]
    encoder = state.get("encoder")
    if isinstance(encoder, Mapping):
        aliased = dict(encoder)
        for field in ("categories", "modes"):
            if isinstance(encoder.get(field), Mapping):
                aliased[field] = {
                    projection.alias(k): v for k, v in encoder[field].items()
                }
        for field in ("identity_columns", "stand_in_columns"):
            if isinstance(encoder.get(field), list):
                aliased[field] = [projection.alias(c) for c in encoder[field]]
        out["encoder"] = aliased
    return out


def _resolve_declared(
    key: Any, feature_names: list[Any]
) -> tuple[int | None, str | None]:
    """``(feature position, None)`` or ``(None, kind)`` for a declared key."""
    exact = [i for i, name in enumerate(feature_names) if _same(key, name)]
    second: list[int] = []
    if type(key) is str:
        for i, name in enumerate(feature_names):
            if type(name) is np.str_:
                if str(name) == key:
                    second.append(i)
            elif type(name) is not str and _json_key(name) == key:
                second.append(i)
    if exact and second:
        return None, "ambiguous_reference"
    if exact:
        return exact[0], None
    if len(second) > 1:
        return None, "ambiguous_reference"
    if second:
        return second[0], None
    return None, "unknown_reference"


def _aliases(items: list[_Item]) -> Iterator[tuple[_Item, str]]:
    for item in items:
        try:
            yield item, column_alias(item.name)
        except UnsupportedColumnName as exc:
            raise _refuse(
                exc.reason, exc.kind, None, [_descriptor(item, None)]
            ) from exc


def project_column_names(
    feature_names: list[Any],
    config_normalized: Mapping[str, Any],
    declared_keys: list[Any],
    inner_valid: Mapping[str, Any] | None = None,
) -> ColumnNameProjection:
    """Check a project's names and return their aliases (H-0141 対応方針 2-3).

    Runs before anything is written. Reads, never changes, its inputs.

    Raises:
        LizyMLError: ``SERIALIZATION_FAILED`` with ``reason``
            ``unsupported_column_name`` or ``column_name_collision`` and the
            context H-0141 decides.
    """
    features = [_Item("feature_names", i, n) for i, n in enumerate(feature_names)]
    roles = _role_items(config_normalized, inner_valid)
    feature_aliases = [alias for _, alias in _aliases(features)]
    role_aliases = [alias for _, alias in _aliases(roles)]

    declared: list[tuple[Any, str]] = []
    for position, key in enumerate(declared_keys):
        index, kind = _resolve_declared(key, feature_names)
        if index is None:
            item = _Item("declared_categories", position, key)
            raise _refuse(
                "unsupported_column_name", str(kind), None, [_descriptor(item, None)]
            )
        declared.append((key, feature_aliases[index]))

    # Collisions: two features with one alias, or a non-string feature whose
    # alias a (string) role name also spells -- a string feature and the same
    # string role are one column.
    seen: dict[str, list[tuple[_Item, str]]] = {}
    for item, alias in zip(features, feature_aliases, strict=True):
        seen.setdefault(alias, []).append((item, alias))
    for item, alias in zip(roles, role_aliases, strict=True):
        group = seen.get(alias)
        if group is None:
            continue
        if all(type(f.name) in (str, np.str_) for f, _ in group):
            continue
        group.append((item, alias))
    for alias, group in seen.items():
        physical = [g for g in group if g[0].where == "feature_names"]
        roles_in = [g for g in group if g[0].where != "feature_names"]
        if len(physical) > 1 or (roles_in and physical):
            raise _refuse(
                "column_name_collision",
                "duplicate_alias",
                alias,
                [_descriptor(item, a) for item, a in group],
            )
    return ColumnNameProjection(
        feature_names=tuple(feature_names),
        aliases=tuple(feature_aliases),
        declared=tuple(declared),
    )
