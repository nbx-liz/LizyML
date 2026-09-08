"""Measure the concrete types of every parameter value the suite constructs.

This is the Change Gate evidence for **H-0095** (normalise parameter values at
ingress). An accept-list at the parameter surfaces is an ``allow`` condition, so
it needs a measured firing rate rather than an argument from plausibility -- and
it needs the same measurement again afterwards, because a refusal nobody
measures after shipping is a refusal nobody knows the size of.

It measures by replaying real inputs, not by reading code. It wraps
``normalise_params`` -- the one function every parameter dict passes through at
every declared surface -- and records the concrete type of every value that
reaches it, **before** normalisation, together with whether the surface refused.

Run it as a pytest plugin over the whole suite::

    PYTHONPATH=docs/audits/2026-09-defect-discovery/instruments \\
    TYPE_CENSUS_OUT=/tmp/type_census.json \\
    uv run pytest -q --no-cov -p parameter_value_type_census

Measured **before** H-0095, at head `1403ba8` over the full suite (2898 passed,
62 skipped), by wrapping ``check_duplicate_identities``, which was the single
narrow point at the time:

===========================  =======
concrete type                 values
===========================  =======
int                             1082
float                            187
str                               78
list                              45
ndarray                           14
tuple                              8
bool                               6
NoneType                           3
**everything else**            **7**
===========================  =======

Total **1430**. Sequence elements were `float` 52, `str` 22, `int` 2 -- all
plain. The seven remaining values were `Equivalent`, `Proxy`, `Conflicting`,
`FormatsToLiar` and `Rate`: **every one a hostile object constructed by this
PR's own rounds 16-20 regression tests.** No configuration in this repository
produced a value outside the proposed accept-list.

    Firing rate: 7/1430 of every parameter value the suite constructs
    (measured by wrapping the shared identity check over the full suite);
    all 7 are adversarial objects built by review rounds 16-20.

Measured **after** H-0095, by wrapping ``normalise_params``:

    Firing rate: 14/1518 of every parameter value the suite constructs
    (7425 passed, 256 skipped); all 14 are objects this PR builds to exercise
    the refusal -- the rounds 16-20 adversarial values and the refusal-matrix
    probe. No configuration in this repository is refused.

**Bound.** This is the population *this repository* constructs. LightGBM is a
library, so the production distribution is user code and cannot be observed from
here. That bound is why H-0095's refusal is stated as loud-at-ingress rather
than as "nothing real is refused".
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
from typing import Any

import lizyml.core._model_factories as factories

_COUNTS: collections.Counter[str] = collections.Counter()
_BY_SURFACE: dict[str, collections.Counter[str]] = collections.defaultdict(
    collections.Counter
)
_ELEMENTS: collections.Counter[str] = collections.Counter()
_REFUSED: collections.Counter[str] = collections.Counter()


def _record(surface: str, params: dict[str, Any]) -> None:
    for value in params.values():
        _COUNTS[type(value).__name__] += 1
        _BY_SURFACE[surface][type(value).__name__] += 1
        if isinstance(value, (list, tuple, set)):
            for element in value:
                _ELEMENTS[type(element).__name__] += 1


def pytest_configure(config: Any) -> None:
    """Wrap the ingress normaliser, which every declared surface reaches."""
    real = factories.normalise_params

    def wrapped(params: dict[str, Any], *, surface: str) -> dict[str, Any]:
        try:
            _record(surface, params)
        except Exception:  # noqa: BLE001 - a measurement must never fail a test
            pass
        try:
            return real(params, surface=surface)
        except Exception as refusal:
            # The values the refusal **named**, not every value in the dict
            # they sat in. Counting bystanders would report a firing rate
            # larger than the one the change actually has, and the rate is the
            # Change Gate evidence -- so it has to be machine-produced from
            # the refusal itself rather than narrowed by hand afterwards.
            try:
                for entry in getattr(refusal, "context", {}).get("rejected", []):
                    _REFUSED[entry["type"]] += 1
            except Exception:  # noqa: BLE001 - a measurement must never fail a test
                pass
            raise

    factories.normalise_params = wrapped  # type: ignore[assignment]


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    out = pathlib.Path(os.environ.get("TYPE_CENSUS_OUT", "/tmp/type_census.json"))
    out.write_text(
        json.dumps(
            {
                "values_by_type": dict(_COUNTS),
                "by_surface": {k: dict(v) for k, v in _BY_SURFACE.items()},
                "sequence_elements_by_type": dict(_ELEMENTS),
                "refused_dict_values_by_type": dict(_REFUSED),
                "total_values": sum(_COUNTS.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )
