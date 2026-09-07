"""The set of parameter names LightGBM accepts, read from LightGBM itself.

LightGBM discards a parameter it does not recognise. It does emit a warning,
but only at ``verbose >= 0``, and this library defaults to ``verbose=-1``, so in
practice an unknown name is silently inert: the booster trains, a score comes
back, and whatever the parameter was meant to do never happened.

Deciding which names are real therefore needs an authority, and the only correct
one is the library. ``LGBM_DumpParamAliases`` dumps LightGBM's own registry as
``{canonical: [alias, ...]}``. A hand-written list would go stale precisely when
LightGBM adds or removes a name -- the drift this module exists to detect
(H-0093).

The dump is done once at import. It is a single C call over an in-memory table,
but config validation runs per ``Model`` construction and there is no reason to
repeat it.
"""

from __future__ import annotations

import ctypes
import json

import lightgbm as lgb

#: Bytes reserved for the dump. LightGBM 4.6.0 produces roughly 12 KB; a
#: megabyte is chosen so that a future release growing the table does not
#: silently truncate it. ``_dump_param_aliases`` verifies the buffer was big
#: enough rather than trusting this number.
_BUFFER_BYTES = 1 << 20


def _dump_param_aliases() -> dict[str, list[str]]:
    """Return LightGBM's ``{canonical: [alias, ...]}`` table.

    Raises:
        RuntimeError: if the reserved buffer was too small, or the dump did not
            parse. Either would otherwise yield a short or empty name set, and
            an empty authority makes every check against it pass vacuously
            (DC1).
    """
    # `_LIB` is LightGBM's handle to its own shared library. It is private, and
    # there is no public Python route to the parameter registry, so its absence
    # is treated as a hard failure rather than degrading to an empty name set:
    # an empty authority would make every check against it pass vacuously (DC1).
    lib = getattr(lgb.basic, "_LIB", None)
    dump = getattr(lib, "LGBM_DumpParamAliases", None) if lib is not None else None
    if dump is None:
        raise RuntimeError(
            f"lightgbm {lgb.__version__} exposes no LGBM_DumpParamAliases via "
            "lightgbm.basic._LIB, so the accepted parameter names cannot be "
            "read from the library. Update this module to the new route "
            "rather than falling back to a hand-written list."
        )
    buf = ctypes.create_string_buffer(_BUFFER_BYTES)
    out_len = ctypes.c_int64(0)
    dump(ctypes.c_int64(_BUFFER_BYTES), ctypes.byref(out_len), ctypes.byref(buf))
    if out_len.value > _BUFFER_BYTES:
        raise RuntimeError(
            f"LightGBM's parameter table needs {out_len.value} bytes but only "
            f"{_BUFFER_BYTES} were reserved; the name set would be truncated."
        )
    table = json.loads(buf.value.decode("utf-8"))
    if not isinstance(table, dict) or not table:
        raise RuntimeError(
            f"LGBM_DumpParamAliases returned {type(table).__name__} with "
            f"{len(table) if hasattr(table, '__len__') else '?'} entries; "
            "expected a non-empty mapping of canonical names to aliases."
        )
    return table


_ALIAS_TABLE: dict[str, list[str]] = _dump_param_aliases()


def _accepted_names() -> frozenset[str]:
    names = set(_ALIAS_TABLE)
    for aliases in _ALIAS_TABLE.values():
        names.update(aliases)
    return frozenset(names)


def _canonical_by_name() -> dict[str, str]:
    """Map every accepted spelling to the canonical name it means.

    LightGBM treats an alias as the parameter itself, so any check that reasons
    about *which parameter* a name refers to has to canonicalise first.
    ``max_leaves`` is ``num_leaves``: a check comparing literal strings sees two
    different names and lets one of them through (H-0094, review round 2).
    """
    out = {name: name for name in _ALIAS_TABLE}
    for canonical, aliases in _ALIAS_TABLE.items():
        for alias in aliases:
            out[alias] = canonical
    return out


#: Every name LightGBM accepts for a training parameter, canonical or alias.
LGBM_PARAM_NAMES: frozenset[str] = _accepted_names()

#: Accepted spelling -> the canonical parameter it names. Covers the canonical
#: names themselves, which map to themselves.
LGBM_CANONICAL_NAME: dict[str, str] = _canonical_by_name()


def accepted_spellings(canonical: str) -> frozenset[str]:
    """Every spelling LightGBM accepts for *canonical*, including itself.

    Raises:
        KeyError: if *canonical* is not a name LightGBM defines. A silent empty
            set here would make an alias-aware check pass vacuously (DC1).
    """
    if canonical not in _ALIAS_TABLE:
        known = LGBM_CANONICAL_NAME.get(canonical)
        why = (
            f"it is an alias of {known!r}"
            if known is not None
            else "LightGBM does not define it"
        )
        raise KeyError(
            f"{canonical!r} is not a canonical LightGBM parameter name; {why}"
        )
    return frozenset({canonical, *_ALIAS_TABLE[canonical]})
