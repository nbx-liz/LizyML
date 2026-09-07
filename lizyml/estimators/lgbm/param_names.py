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


def _accepted_names() -> frozenset[str]:
    table = _dump_param_aliases()
    names = set(table)
    for aliases in table.values():
        names.update(aliases)
    return frozenset(names)


#: Every name LightGBM accepts for a training parameter, canonical or alias.
LGBM_PARAM_NAMES: frozenset[str] = _accepted_names()
