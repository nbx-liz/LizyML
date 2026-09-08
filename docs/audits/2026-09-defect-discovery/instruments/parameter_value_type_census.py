"""Measure the concrete types of every parameter value the suite constructs.

This is the Change Gate evidence for **H-0095** (normalise parameter values at
ingress). An accept-list at the parameter surfaces is an ``allow`` condition, so
it needs a measured firing rate before implementation rather than an argument
from plausibility.

It measures by replaying real inputs, not by reading code: it wraps
``check_duplicate_identities`` -- the one function every parameter dict passes
through at every declared surface, which rounds 10-12 established and pinned --
and records the concrete type of every value that reaches it.

Run it as a pytest plugin over the whole suite::

    PYTHONPATH=docs/audits/2026-09-defect-discovery/instruments \\
    TYPE_CENSUS_OUT=/tmp/type_census.json \\
    uv run pytest -q --no-cov -p parameter_value_type_census

Measured at head `1403ba8` over the full suite (2898 passed, 62 skipped):

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
plain. The seven remaining values are `Equivalent`, `Proxy`, `Conflicting`,
`FormatsToLiar` and `Rate`: **every one is a hostile object constructed by this
PR's own rounds 16-20 regression tests.** No configuration in this repository
produces a value outside the proposed accept-list.

    Firing rate: 7/1430 of every parameter value the suite constructs
    (measured by wrapping the shared identity check over the full suite);
    all 7 are adversarial objects built by review rounds 16-20.

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


def _record(surface: str, params: dict[str, Any]) -> None:
    for value in params.values():
        _COUNTS[type(value).__name__] += 1
        _BY_SURFACE[surface][type(value).__name__] += 1
        if isinstance(value, (list, tuple, set)):
            for element in value:
                _ELEMENTS[type(element).__name__] += 1


def pytest_configure(config: Any) -> None:
    """Wrap the shared identity check, which every declared surface reaches."""
    real = factories.check_duplicate_identities

    def wrapped(provider: Any, params: dict[str, Any], *, surface: str) -> None:
        try:
            _record(surface, params)
        except Exception:  # noqa: BLE001 - a measurement must never fail a test
            pass
        return real(provider, params, surface=surface)

    factories.check_duplicate_identities = wrapped  # type: ignore[assignment]

    # The facade imported it by name, so the rebind has to reach there too --
    # patching only the definition module would measure a subset and report it
    # as the whole, which is the defect class this audit exists to find.
    import lizyml.core.model as facade

    if hasattr(facade, "check_duplicate_identities"):
        facade.check_duplicate_identities = wrapped  # type: ignore[assignment]


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    out = pathlib.Path(os.environ.get("TYPE_CENSUS_OUT", "/tmp/type_census.json"))
    out.write_text(
        json.dumps(
            {
                "values_by_type": dict(_COUNTS),
                "by_surface": {k: dict(v) for k, v in _BY_SURFACE.items()},
                "sequence_elements_by_type": dict(_ELEMENTS),
                "total_values": sum(_COUNTS.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )
