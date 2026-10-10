"""pytest plugin: census of category value types seen by CategoricalEncoder.fit (H-0120).

Each CategoricalEncoder.fit is one unit of the population. A fit "fires" when
any category of any column has a type outside the accepted set: Python str,
int, float, bool, or a numpy scalar whose .item() is one of those and equals
the original (the normalisation H-0120 performs before json.dump). The tally
is written to $H0120_CENSUS_OUT at session end.
"""

from __future__ import annotations

import collections
import json
import os

import numpy as np

_ACCEPTED = (str, int, float, bool)
_counts: collections.Counter[str] = collections.Counter()
_examples: list[dict[str, object]] = []


def _accepted(value: object) -> bool:
    if type(value) in _ACCEPTED:
        return True
    if isinstance(value, np.generic):
        plain = value.item()
        return type(plain) in _ACCEPTED and plain == value
    return False


def pytest_configure(config: object) -> None:
    from lizyml.features.encoders import categorical_encoder as module

    original = module.CategoricalEncoder.fit

    def fit(self, X, categorical_cols):  # type: ignore[no-untyped-def]
        result = original(self, X, categorical_cols)
        _counts["fits"] += 1
        bad = {
            col: sorted({type(v).__name__ for v in cats if not _accepted(v)})
            for col, cats in self._categories.items()
        }
        bad = {col: names for col, names in bad.items() if names}
        if bad:
            _counts["firing_fits"] += 1
            if len(_examples) < 20:
                _examples.append({"columns": bad})
        if any(self._categories.values()):
            _counts["fits_with_categories"] += 1
        return result

    module.CategoricalEncoder.fit = fit

    from lizyml.core.types import target_encoder as target_module

    original_target_fit = target_module.TargetEncoder.fit.__func__

    def target_fit(cls, y, task):  # type: ignore[no-untyped-def]
        encoder = original_target_fit(cls, y, task)
        _counts["target_fits"] += 1
        if encoder.classes_:
            _counts["target_fits_with_classes"] += 1
            names = sorted({type(v).__name__ for v in encoder.classes_ if not _accepted(v)})
            if names:
                _counts["target_firing_fits"] += 1
                if len(_examples) < 40:
                    _examples.append({"target": names})
        return encoder

    target_module.TargetEncoder.fit = classmethod(target_fit)


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    out = os.environ.get("H0120_CENSUS_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"counts": dict(_counts), "examples": _examples}, handle, indent=2)
