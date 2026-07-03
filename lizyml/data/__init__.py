"""Data layer: dataframe building, fingerprinting, and leakage validators.

The leakage validators are exposed here as public API (H-0087) so users can run
explicit time-order / target-leakage / group-overlap checks in line with the
library's leakage-first charter::

    from lizyml.data import validate_time_series_order, validate_group_split

They are not auto-wired into :meth:`Model.fit` — call them explicitly (auto-wiring
is a behavior change deferred to a future proposal).
"""

from lizyml.data.validators import (
    validate_group_split,
    validate_no_target_leakage,
    validate_time_series_order,
)

__all__ = [
    "validate_group_split",
    "validate_no_target_leakage",
    "validate_time_series_order",
]
