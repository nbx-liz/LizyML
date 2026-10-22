"""#393 population probe: which categorical declarations reach LightGBM, by label type?

A pytest plugin (``-p i393_census`` with this directory on ``PYTHONPATH``),
active only when ``I393_CENSUS_LOG`` is set. It wraps ``LGBMAdapter.fit`` and
records, per call, the training frame's column labels and the categorical
declaration the adapter passes to ``lgb.Dataset``, with each label's exact
type and its position in the frame. A call whose declaration names a label
that is not a ``str`` is the population the fix changes (LightGBM reads an
``int`` entry as a position). The real call then runs unchanged.

    I393_CENSUS_LOG=<file> PYTHONPATH=docs/audits/2026-10-i393/firing \
        uv run pytest -p i393_census -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import os
from typing import Any

_LOG = os.environ.get("I393_CENSUS_LOG")


def _record(adapter: Any, X_train: Any) -> None:
    cats = adapter._categorical_features
    columns = list(X_train.columns)
    record: dict[str, Any] = {
        "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
        "pid": os.getpid(),
        "n_columns": len(columns),
        "declared": None if cats is None else len(cats),
    }
    if cats is not None:
        record["declared_types"] = sorted({type(c).__name__ for c in cats})
        record["non_str"] = [
            {"repr": repr(c)[:40], "type": type(c).__name__}
            for c in cats
            if type(c) is not str
        ]
    record["column_types"] = sorted({type(c).__name__ for c in columns})
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def pytest_configure(config: Any) -> None:
    del config
    if not _LOG:
        return
    from lizyml.estimators.lgbm.adapter import LGBMAdapter

    original = LGBMAdapter.fit

    def fit(self: Any, X_train: Any, *args: Any, **kwargs: Any) -> Any:
        _record(self, X_train)
        return original(self, X_train, *args, **kwargs)

    LGBMAdapter.fit = fit  # type: ignore[method-assign]
