"""H-0133 firing probe: record the category value types every encoder fit sees.

Loaded through ``PYTHONPATH`` only when ``H0133_FIRING_LOG`` is set (notebook
kernels inherit it). It wraps ``CategoricalEncoder.fit`` when the module is
imported; the real fit always runs. One record per fit and column: the sorted
type names of the learned categories and whether LightGBM could write them
(``category_json.is_writable``). The process role is read at write time.
A failure to record is written as an ``error`` record, never skipped.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys

_LOG = os.environ.get("H0133_FIRING_LOG")


def _write(record: dict) -> None:
    record.update(
        pid=os.getpid(),
        role="kernel" if any("ipykernel" in a for a in sys.argv) else "pytest",
        test=os.environ.get("PYTEST_CURRENT_TEST", ""),
    )
    with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _patch(module) -> None:  # type: ignore[no-untyped-def]
    original = module.CategoricalEncoder.fit

    def fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = original(self, *args, **kwargs)
        try:
            from lizyml.estimators.lgbm.category_json import is_writable

            for col, cats in self._categories.items():
                _write(
                    {
                        "kind": "fit",
                        "column": str(col),
                        "n": len(cats),
                        "types": sorted({type(v).__name__ for v in cats}),
                        "unwritable": sorted({type(v).__name__ for v in cats if not is_writable(v)}),
                    }
                )
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            _write({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        return out

    module.CategoricalEncoder.fit = fit


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if fullname != "lizyml.features.encoders.categorical_encoder":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):  # type: ignore[no-untyped-def]
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch  # type: ignore[method-assign]
        return spec


if _LOG:
    sys.meta_path.insert(0, _Finder())
