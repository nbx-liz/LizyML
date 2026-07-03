"""Leakage validators are public API on ``lizyml.data`` (H-0087, #216).

The three validators were dead code (no call sites, unexported). They are now
re-exported from ``lizyml.data`` so users can run explicit leakage checks. The
empty ``lizyml.utils`` package was removed. This golden test pins both facts.
"""

from __future__ import annotations

import importlib

import pytest

import lizyml.data as data_pkg

_EXPECTED = {
    "validate_group_split",
    "validate_no_target_leakage",
    "validate_time_series_order",
}


def test_validators_in_data_all() -> None:
    assert set(data_pkg.__all__) >= _EXPECTED


def test_validators_importable_from_data() -> None:
    from lizyml.data import (  # noqa: F401
        validate_group_split,
        validate_no_target_leakage,
        validate_time_series_order,
    )

    for name in _EXPECTED:
        assert callable(getattr(data_pkg, name))


def test_utils_package_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lizyml.utils")
