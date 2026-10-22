"""column_name_template -- the H-0141 column-name aliases, copied into generated code.

The generated ``train.py`` / ``predict.py`` carry the library's
``UnsupportedColumnName`` and ``column_alias`` verbatim (read from
:mod:`lizyml.codegen.column_names` when the scripts are rendered, so the copy
cannot drift), plus the project-side checks: the ``column_name_policy``
marker, the marked state's shape, and the input boundary (typed frames, CSV
and parquet headers). The block follows the value codec block and uses its
``np``, ``pd`` and ``json``.
"""

from __future__ import annotations

import inspect

from lizyml.codegen import column_names

_HELPERS = '''\
import csv as _csv

_COLUMN_NAME_POLICY = "str_alias_v1"


class ColumnNameError(ValueError):
    """A column name, header or marker the project cannot read (H-0141).

    ``reason`` is the decided reason, ``names`` the names at fault and
    ``context`` the file, field and value where one applies.
    """

    def __init__(self, reason, message, names=(), **context):
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.names = list(names)
        self.context = context


def _policy_value(doc, where):
    """None (no marker) or the policy; any other value is refused."""
    if "column_name_policy" not in doc:
        return None
    value = doc["column_name_policy"]
    if type(value) is not str or value != _COLUMN_NAME_POLICY:
        raise ColumnNameError(
            "unsupported_column_name_policy",
            f"{where} names column_name_policy {value!r}; this script reads "
            f"only {_COLUMN_NAME_POLICY!r}",
            file=where, field="column_name_policy", value=repr(value),
        )
    return value


def _bad_state(where, field, message, names=()):
    return ColumnNameError(
        "invalid_column_name_state", f"{where} {field}: {message}",
        [repr(n) for n in names], file=where, field=field,
    )


def _typed_field(doc, field, where, kinds, required=True):
    """*doc[field]* when its type is one of *kinds*; else invalid_column_name_state.

    The marked-state check reads only fields whose type it has checked first,
    so a malformed document is refused with the decided reason, never with an
    error from the code that reads it.
    """
    if field not in doc:
        if required:
            raise _bad_state(where, field, "is missing")
        return None
    value = doc[field]
    if not any(type(value) is kind for kind in kinds):
        names = " or ".join("null" if k is type(None) else k.__name__ for k in kinds)
        raise _bad_state(where, field, f"is not {names}", [value])
    return value


def _names_of(doc, field, where, required=True):
    values = _typed_field(doc, field, where, (list,), required)
    if values is None:
        return None
    stray = [v for v in values if type(v) is not str]
    if stray:
        raise _bad_state(where, field, "holds names that are not str", stray)
    return values


def _role_names(config):
    """(field, value, required) for the role fields of config.json.

    A field the writer always writes but that is absent is left to the code
    that reads it (H-0120 fails such a project with a KeyError naming it);
    a field that is present must have its type.
    """
    where = "config.json"
    roles = []
    if "_target_col" in config:
        roles.append(("_target_col", config["_target_col"], True))
    split = _typed_field(config, "split", where, (dict, type(None)), False) or {}
    for key in ("time_col", "group_col"):
        roles.append((f"split.{key}", split.get(key), False))
    for block in ("blocks", "groups"):
        value = _typed_field(split, block, f"{where} split", (dict, type(None)), False)
        if value is not None:
            roles.append((f"split.{block}.col", value.get("col"), True))
    inner = _typed_field(config, "inner_valid", where, (dict, type(None)), False) or {}
    roles.append(("inner_valid.group_col", inner.get("group_col"), False))
    rule = _typed_field(config, "categorical_rule", where, (dict,), False) or {}
    rule_where = f"{where} categorical_rule"
    explicit = _typed_field(rule, "explicit", rule_where, (list,), False) or []
    roles += [("categorical_rule.explicit", c, True) for c in explicit]
    return roles


def _check_config_names(config):
    where = "config.json"
    features = _names_of(config, "feature_names", where)
    repeated = sorted({n for n in features if features.count(n) > 1})
    if repeated:
        raise _bad_state(where, "feature_names", "repeat a name", repeated)
    categorical = _names_of(config, "categorical_features", where)
    outside = [c for c in categorical if c not in features]
    if outside:
        raise _bad_state(where, "categorical_features", "are not features", outside)
    declared = (
        _typed_field(config, "declared_categories", where, (dict,), False) or {}
    )
    stray = [k for k in declared if k not in categorical]
    if stray:
        raise _bad_state(
            where, "declared_categories", "name non-categorical columns", stray
        )
    for field, value, required in _role_names(config):
        if value is None and not required:
            continue
        if type(value) is not str:
            raise _bad_state(where, field, "is not a str", [value])
    return features, categorical


def _check_state_names(state, where, features, categorical):
    if _names_of(state, "feature_names", where) != features:
        raise _bad_state(
            where, "feature_names", "differ from config.json (order included)"
        )
    if _names_of(state, "categorical_features", where) != categorical:
        raise _bad_state(where, "categorical_features", "differ from config.json")
    categories = _typed_field(state, "categories", where, (dict,))
    if set(categories) != set(categorical):
        raise _bad_state(
            where, "categories", "are not keyed by the categorical features",
            sorted(set(categories) ^ set(categorical), key=repr),
        )
    # Each column's entry holds its list of categories and its mode.
    incomplete = [
        c for c, entry in categories.items()
        if type(entry) is not dict
        or "mode" not in entry
        or type(entry.get("categories")) is not list
    ]
    if incomplete:
        raise _bad_state(
            where, "categories", "entries lack their categories or mode", incomplete
        )
    # Absent before H-0138's identity; when present, a list of categorical names.
    identity = _names_of(state, "identity_columns", where, required=False) or []
    stray = [c for c in identity if c not in categorical]
    if stray:
        raise _bad_state(
            where, "identity_columns", "name non-categorical columns", stray
        )


def check_column_name_policy(config, state=None, where="pipeline_state.json"):
    """Whether the project uses the H-0141 aliases (checked before any read).

    No marker anywhere: False (a project written before H-0141). A marker with
    another value, or a marker in one file only, is refused; a marked project
    must name its columns consistently, as H-0141 decides. Both markers are
    read before any name is checked. *state* is None only when the project
    has no state file.
    """
    in_config = _policy_value(config, "config.json")
    if state is None:
        if in_config is None:
            return False
        _check_config_names(config)
        return True
    in_state = _policy_value(state, where)
    if in_config is None and in_state is None:
        return False
    if in_config is None or in_state is None:
        raise ColumnNameError(
            "column_name_policy_mismatch",
            f"config.json has column_name_policy {in_config!r} and {where} "
            f"has {in_state!r}",
            config=repr(in_config), state=repr(in_state), file=where,
        )
    features, categorical = _check_config_names(config)
    _check_state_names(state, where, features, categorical)
    return True


def project_uses_aliases():
    """check_column_name_policy over config.json and the project's state file.

    The state is read as JSON only: no category value is decoded before the
    markers and names are checked.
    """
    path = ARTIFACTS / "pipeline_state.json"
    if not path.exists():
        return check_column_name_policy(CFG)
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    return check_column_name_policy(CFG, state, str(path))


def prepare_input(df, required=()):
    """*df* with its columns renamed to their aliases (a shallow copy).

    A project written before H-0141 (no marker) reads its input as it is.
    """
    if not project_uses_aliases():
        return df
    if isinstance(df.columns, pd.MultiIndex):
        raise ColumnNameError(
            "unsupported_column_name", "a MultiIndex header has no column names"
        )
    aliases = []
    for name in df.columns:
        try:
            aliases.append(column_alias(name))
        except UnsupportedColumnName as exc:
            raise ColumnNameError(exc.reason, str(exc), [repr(name)]) from exc
    seen = {}
    for name, alias in zip(df.columns, aliases):
        seen.setdefault(alias, []).append(name)
    for alias, names in seen.items():
        if len(names) > 1:
            raise ColumnNameError(
                "column_name_collision",
                f"input columns {names!r} share the alias {alias!r}",
                [repr(n) for n in names],
            )
    missing = [a for a in required if a not in seen]
    if missing:
        raise ColumnNameError(
            "missing_column_alias",
            f"input lacks the columns {missing}; columns present: {aliases}",
            missing,
        )
    out = df.copy(deep=False)
    out.columns = aliases
    return out


def _raw_csv_header(path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            return next(_csv.reader(f, strict=True), [])
    except (_csv.Error, UnicodeDecodeError) as exc:
        raise ColumnNameError(
            "invalid_column_header", f"{path}: the header cannot be read ({exc})"
        ) from exc


def _read_parquet_checked(path):
    # The file's own names first: pandas cannot build a frame from repeated
    # names, and would fail without a reason.
    import pyarrow.parquet as _pq

    stored = _pq.read_schema(path).names
    repeated = sorted({n for n in stored if stored.count(n) > 1})
    if repeated:
        raise ColumnNameError(
            "column_name_collision", f"{path} repeats column names", repeated
        )
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        raise ColumnNameError(
            "invalid_column_header", f"{path} has a MultiIndex header"
        )
    names = list(df.columns)
    bad = [repr(n) for n in names if type(n) is not str or n == ""]
    if bad:
        raise ColumnNameError(
            "invalid_column_header", f"{path} has non-string names", bad
        )
    if len(set(names)) != len(names):
        raise ColumnNameError(
            "column_name_collision", f"{path} repeats a column name", names
        )
    return df


def read_input(path):
    """Read a CSV or parquet input; a marked project validates its header first."""
    path = Path(path)
    marked = project_uses_aliases()
    if path.suffix == ".parquet":
        return _read_parquet_checked(path) if marked else pd.read_parquet(path)
    if not marked:
        return pd.read_csv(path)
    raw = _raw_csv_header(path)
    blank = [i for i, name in enumerate(raw) if name == ""]
    if blank:
        raise ColumnNameError(
            "invalid_column_header",
            f"{path} has blank header names at positions {blank}",
        )
    repeated = sorted({n for n in raw if raw.count(n) > 1})
    if repeated:
        raise ColumnNameError(
            "column_name_collision", f"{path} repeats header names", repeated
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns) != raw:
        raise ColumnNameError(
            "invalid_column_header",
            f"{path}: pandas read the header {list(df.columns)} from {raw}",
            raw,
        )
    return df

'''


def _copy() -> str:
    source = inspect.getsource(column_names.UnsupportedColumnName)
    source += "\n\n" + inspect.getsource(column_names.column_alias)
    return source


COLUMN_NAME_COPY = (
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "#  Column names (LizyML H-0141)\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "# A project written by LizyML >= H-0141 names every column by the string\n"
    "# alias str(name); its inputs are renamed to aliases before any lookup.\n\n"
    + _copy()
    + "\n\n"
    + _HELPERS
)
