"""H-0143 firing rate: which state_changes calls of the suite does H-0143 change?

A pytest plugin (``-p h0143_census`` with this directory on ``PYTHONPATH``),
active only when ``H0143_CENSUS_LOG`` is set. It wraps
``lizyml.persistence._category_preflight.state_changes`` and records, per
call, whether the H-0143 prototype gives a different list of changes. The
real result is returned unchanged.

    H0143_CENSUS_LOG=<file> PYTHONPATH=docs/audits/2026-10-h0143/probes \
        uv run pytest -p h0143_census -m "" -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import os
from typing import Any

_LOG = os.environ.get("H0143_CENSUS_LOG")


def pytest_configure(config: Any) -> None:
    del config
    if not _LOG:
        return
    from prototype import prototype_state_changes

    from lizyml.persistence import _category_preflight as pf

    original = pf.state_changes

    def state_changes(state: Any) -> Any:
        result = original(state)
        try:
            other = prototype_state_changes(state)
            differs = [repr(c) for c in result] != [repr(c) for c in other]
            error = None
        except Exception as exc:  # noqa: BLE001 -- recorded, the real result stands
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

    from lizyml.persistence import exporter

    # The preflight's own caller reads the module attribute; exporter bound
    # the name at import.
    pf.state_changes = state_changes  # type: ignore[assignment]
    exporter.state_changes = state_changes  # type: ignore[attr-defined]
