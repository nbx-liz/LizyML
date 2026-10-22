"""H-0139 population probe: which path would each real categorical column take?

A pytest plugin (``-p h0139_census`` with this directory on ``PYTHONPATH``),
active only when ``H0139_CENSUS_LOG`` is set. It wraps the two places a
column's categories are first learned and records, per column per call, the
path the H-0139 mechanism would take (``h0139_rule.classify``), then runs the
real call unchanged:

- ``builder``: ``lizyml.data.dataframe_builder._apply_categorical``, the fit
  entry (``Model.fit`` / ``tune``; ``Model.predict`` does not pass through
  it), for each column it casts;
- ``encoder``: ``CategoricalEncoder.fit`` for each column it is asked to fit
  (direct encoder tests, and every fold / refit fit after the builder), of any
  dtype. Folds are separate visits, not deduplicated logical columns.

A column the prototype cannot classify is recorded with ``rule_error``, never
skipped. A record is written before the wrapped call runs, so a column whose
fit then raises is still counted.

    H0139_CENSUS_LOG=<file> PYTHONPATH=docs/audits/2026-10-h0139/firing \
        uv run pytest -p h0139_census -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import os
from typing import Any

_LOG = os.environ.get("H0139_CENSUS_LOG")


def _write(record: dict[str, Any]) -> None:
    record.update(pid=os.getpid(), test=os.environ.get("PYTEST_CURRENT_TEST", ""))
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _classify(surface: str, column: Any, series: Any) -> None:
    from h0139_rule import classify, dtype_kind

    record: dict[str, Any] = {"surface": surface, "column": str(column)}
    try:
        record["dtype"] = dtype_kind(series)
        record.update(classify(series))
    except Exception as exc:  # noqa: BLE001 -- recorded, never skipped
        record["rule_error"] = f"{type(exc).__name__}: {exc}"[:300]
    _write(record)


def _install() -> None:
    import pandas as pd

    from lizyml.data import dataframe_builder
    from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

    original_apply = dataframe_builder._apply_categorical

    def _apply_categorical(df: Any, spec: Any) -> Any:
        cat_set = set(spec.categorical)
        if spec.auto_categorical:
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                    cat_set.add(col)
        for col in cat_set:
            if col in df.columns:
                _classify("builder", col, df[col])
        return original_apply(df, spec)

    dataframe_builder._apply_categorical = _apply_categorical

    original_fit = CategoricalEncoder.fit

    def fit(self: Any, X: Any, categorical_cols: list[str]) -> Any:
        for col in categorical_cols:
            if col in X.columns:
                _classify("encoder", col, X[col])
        return original_fit(self, X, categorical_cols)

    CategoricalEncoder.fit = fit  # type: ignore[method-assign]


def pytest_configure(config: Any) -> None:
    del config
    if _LOG:
        _install()
