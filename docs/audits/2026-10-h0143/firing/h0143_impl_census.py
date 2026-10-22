"""H-0143 acceptance 4: after the implementation, which calls does it change?

A pytest plugin (``-p h0143_impl_census`` with this directory on
``PYTHONPATH``), active only when ``H0143_CENSUS_LOG`` is set. It wraps
``state_changes`` and records, per call, whether the verdict before H-0143
(the json route judged by the runtime encoder, ``generated=False``) differs
from the shipped one. The shipped result is returned unchanged.

    H0143_CENSUS_LOG=<file> PYTHONPATH=docs/audits/2026-10-h0143/firing \
        uv run pytest -p h0143_impl_census -m "" -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

_LOG = os.environ.get("H0143_CENSUS_LOG")


def _before(state: Any) -> list[Any]:
    """``state_changes`` as it was before H-0143."""
    from lizyml.features.encoders.state_changes import category_state_changes
    from lizyml.persistence import _category_preflight as pf

    if not isinstance(state, Mapping):
        return []
    encoder = state.get("encoder")
    if not isinstance(encoder, Mapping) or not isinstance(
        encoder.get("categories"), Mapping
    ):
        return []
    found: list[Any] = []
    for serializer in pf.SERIALIZERS:
        project = pf._through_joblib if serializer == "joblib" else pf._encoder_through_json
        found.extend(
            pf.SerializedChange(serializer, c)
            for c in category_state_changes(encoder, project(encoder))
        )
    return found


def pytest_configure(config: Any) -> None:
    del config
    if not _LOG:
        return
    from lizyml.persistence import _category_preflight as pf
    from lizyml.persistence import exporter

    shipped = pf.state_changes

    def state_changes(state: Any) -> Any:
        result = shipped(state)
        try:
            differs = [repr(c) for c in result] != [repr(c) for c in _before(state)]
            error = None
        except Exception as exc:  # noqa: BLE001 -- recorded; the shipped result stands
            differs, error = True, type(exc).__name__
        record = {
            "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
            "changes": len(result),
            "differs": differs,
            "error": error,
        }
        with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return result

    pf.state_changes = state_changes  # type: ignore[assignment]
    exporter.state_changes = state_changes  # type: ignore[attr-defined]
