"""H-0135 population probe: what every artifact load reads, before any rule changes.

A pytest plugin (``-p h0135_load_census`` with this directory on
``PYTHONPATH``), active only when ``H0135_LOAD_LOG`` is set. It wraps
``lizyml.persistence.loader.load``, which both ``Model.load`` and
``lizyml.persistence.load`` call, and records the artifact's own state BEFORE
the real load runs. The real load then runs unchanged, and its outcome is
recorded too. One JSON record per call:

- ``format_version``;
- ``checksums``:
  - ``absent``;
  - ``ok``, exactly ``{"algorithm": "sha256", "files": {str: 64 lowercase hex}}``
  - ``bad_object``, an object of any other shape;
  - ``non_object``;
- ``listed``: the files ``checksums.files`` names, when ``checksums`` is ok;
- ``present``: the files in the directory;
- ``outcome``: ``ok`` or the error code the load raised.

A directory that cannot be read is recorded as ``unreadable``, never skipped.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_LOG = os.environ.get("H0135_LOAD_LOG")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _write(record: dict[str, Any]) -> None:
    record.update(pid=os.getpid(), test=os.environ.get("PYTEST_CURRENT_TEST", ""))
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _state(path: Any) -> dict[str, Any]:
    src = Path(path)
    try:
        meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- recorded, never skipped
        return {"unreadable": type(exc).__name__}
    cs = meta.get("checksums") if isinstance(meta, dict) else None
    if not isinstance(meta, dict) or "checksums" not in meta:
        kind, listed = "absent", None
    elif not isinstance(cs, dict):
        kind, listed = "non_object", None
    elif (
        # D1 exactly: two keys, sha256, and every digest 64 lowercase hex.
        type(cs) is dict
        and set(cs) == {"algorithm", "files"}
        and cs["algorithm"] == "sha256"
        and type(cs["files"]) is dict
        and all(
            type(k) is str and type(v) is str and _HEX64.fullmatch(v) is not None
            for k, v in cs["files"].items()
        )
    ):
        kind, listed = "ok", sorted(cs["files"])
    else:
        kind, listed = "bad_object", None
    entries = sorted(src.iterdir())
    return {
        "format_version": meta.get("format_version") if isinstance(meta, dict) else None,
        "checksums": kind,
        "listed": listed,
        # Every directory entry, symlinks (dangling ones too) included, with
        # its type, so a symlink policy can be read off the record.
        "present": [p.name for p in entries],
        "symlinks": [p.name for p in entries if p.is_symlink()],
    }


def pytest_configure(config: Any) -> None:
    if not _LOG:
        return
    import lizyml.persistence as persistence
    import lizyml.persistence.loader as loader

    original = loader.load

    def load(path: Any, *args: Any, **kwargs: Any) -> Any:
        record = {"kind": "load", **_state(path)}
        try:
            out = original(path, *args, **kwargs)
        except Exception as exc:
            record["outcome"] = getattr(getattr(exc, "code", None), "name", type(exc).__name__)
            _write(record)
            raise
        record["outcome"] = "ok"
        _write(record)
        return out

    loader.load = load
    persistence.load = load
