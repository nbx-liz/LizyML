"""H-0141 acceptance 9: which export_code calls does the alias rule act on?

A pytest plugin (``-p i382_census`` with this directory on ``PYTHONPATH``),
active only when ``I382_CENSUS_LOG`` is set. It wraps the shared pre-write
check ``project_column_names`` at both call sites -- ``Model.export_code``
(which imports it from ``lizyml.codegen.column_names`` at call time) and
``generate_code`` (which bound it at import) -- and records, per call:

- ``aliased``: a feature name that is not exactly ``str`` (its alias is a
  new string), or a declared key mapped to a feature by the second rule;
- ``refused``: the check raised, with its reason and kind.

A call with neither is one the rule leaves untouched. The real check runs
unchanged and its result or exception is passed on.

    I382_CENSUS_LOG=<file> PYTHONPATH=docs/audits/2026-10-i382/firing \
        uv run pytest -p i382_census -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import os
from typing import Any

_LOG = os.environ.get("I382_CENSUS_LOG")


def _write(record: dict[str, Any]) -> None:
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _wrap(original: Any, site: str) -> Any:
    def check(feature_names: Any, config: Any, declared: Any, *args: Any, **kw: Any) -> Any:
        record: dict[str, Any] = {
            "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
            "site": site,
            "non_str_features": [
                type(n).__name__ for n in feature_names if type(n) is not str
            ],
        }
        try:
            projection = original(feature_names, config, declared, *args, **kw)
        except Exception as exc:
            context = getattr(exc, "context", {}) or {}
            record["refused"] = [context.get("reason"), context.get("kind")]
            _write(record)
            raise
        record["declared_remapped"] = [
            repr(key)[:40] for key, alias in projection.declared if key != alias
        ]
        _write(record)
        return projection

    return check


def pytest_configure(config: Any) -> None:
    del config
    if not _LOG:
        return
    from lizyml.codegen import column_names, generator

    generator.project_column_names = _wrap(  # type: ignore[assignment]
        column_names.project_column_names, "generate_code"
    )
    column_names.project_column_names = _wrap(  # type: ignore[assignment]
        column_names.project_column_names, "export_code"
    )
