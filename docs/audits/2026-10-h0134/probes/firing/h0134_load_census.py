"""H-0134 population probe: every ``Model.load`` of a tuned model, with or without history.

A pytest plugin (``-p h0134_load_census`` with this directory on
``PYTHONPATH``), active only when ``H0134_LOAD_LOG`` is set. It wraps
``Model.load`` in every process pytest starts (xdist workers load ``-p``
plugins too). The real load always runs. One JSON record per load, written
after the load returns:

- ``tuned``: whether the loaded model has a tuning result;
- ``history``: whether it holds the history (H-0134 P5);
- ``format_version``: read from the artifact's ``metadata.json``.

A failure to record is written as an ``error`` record, never skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_LOG = os.environ.get("H0134_LOAD_LOG")


def _write(record: dict[str, Any]) -> None:
    record.update(pid=os.getpid(), test=os.environ.get("PYTEST_CURRENT_TEST", ""))
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def pytest_configure(config: Any) -> None:
    if not _LOG:
        return
    from lizyml import Model

    original = Model.load.__func__  # type: ignore[attr-defined]

    def load(cls: type[Model], path: Any, *args: Any, **kwargs: Any) -> Model:
        model = original(cls, path, *args, **kwargs)
        try:
            meta_path = Path(path) / "metadata.json"
            # A test that stubs persistence.load passes a path with no files;
            # its version is recorded as such, not guessed.
            version = (
                json.loads(meta_path.read_text(encoding="utf-8")).get("format_version")
                if meta_path.is_file()
                else "stubbed loader"
            )
            _write(
                {
                    "kind": "load",
                    "tuned": model._tuning_result is not None,
                    "history": bool(model._tuning_history_available),
                    "format_version": version,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- recorded, never skipped
            _write({"kind": "error", "error": repr(exc)})
        return model

    Model.load = classmethod(load)  # type: ignore[method-assign, assignment]
