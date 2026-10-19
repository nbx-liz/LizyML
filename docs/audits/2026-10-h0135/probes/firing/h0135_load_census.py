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
  - ``ok``, an object with ``algorithm`` and ``files``, where ``files`` is an
    object of ``str`` to ``str``;
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
from pathlib import Path
from typing import Any

_LOG = os.environ.get("H0135_LOAD_LOG")


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
        set(cs) == {"algorithm", "files"}
        and isinstance(cs["files"], dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in cs["files"].items())
    ):
        kind, listed = "ok", sorted(cs["files"])
    else:
        kind, listed = "bad_object", None
    return {
        "format_version": meta.get("format_version") if isinstance(meta, dict) else None,
        "checksums": kind,
        "listed": listed,
        "present": sorted(p.name for p in src.iterdir() if p.is_file()),
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
