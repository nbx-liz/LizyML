"""H-0138 population probe: which path would each real categorical column take?

A pytest plugin (``-p h0138_census`` with this directory on ``PYTHONPATH``),
active only when ``H0138_CENSUS_LOG`` is set. It wraps the two places a
column's categories are first learned and records, per column, the path the
proposed mechanism would take (``h0138_rule.classify``), then runs the real
call unchanged:

- ``builder``: ``lizyml.data.dataframe_builder._apply_categorical``, the
  production entry (every ``Model.fit`` / ``tune`` / ``predict`` input goes
  through it), for each column it casts;
- ``encoder``: ``CategoricalEncoder.fit`` for each column that arrives as an
  object column (tests that build an encoder directly, without the builder).

A column the prototype cannot classify is recorded with ``rule_error``,
never skipped. One JSON record per column per call.

    H0138_CENSUS_LOG=<file> PYTHONPATH=docs/audits/2026-10-h0138/firing \
        uv run pytest -p h0138_census -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import os
from typing import Any

_LOG = os.environ.get("H0138_CENSUS_LOG")


def _write(record: dict[str, Any]) -> None:
    record.update(pid=os.getpid(), test=os.environ.get("PYTEST_CURRENT_TEST", ""))
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _classify(surface: str, column: Any, series: Any) -> None:
    from h0138_rule import classify

    record: dict[str, Any] = {"surface": surface, "column": str(column)}
    try:
        record["path"] = classify(series)
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
            if col in X.columns and X[col].dtype == object:
                _classify("encoder", col, X[col])
        return original_fit(self, X, categorical_cols)

    CategoricalEncoder.fit = fit  # type: ignore[method-assign]


def pytest_configure(config: Any) -> None:
    del config
    if _LOG:
        _install()
