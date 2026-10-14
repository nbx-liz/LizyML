"""Positive control for firing_rate_h0124_magnitude.py's wiring (run with the plugin).

Run:
  env PYTHONPATH=docs/audits/2026-10-tuning-resume/probes FIRING_RATE_H0124_OUT=<out> \
    .venv/bin/python -m pytest -q -p no:cacheprovider -p firing_rate_h0124_magnitude \
    docs/audits/2026-10-tuning-resume/probes/firing_rate_h0124_posctl.py

Each test drives one wrapped path with an input the rule flags; the output
must contain one flagged ``spec`` row and three flagged ``expand`` rows (an
``inf`` edge, a stepped edge capped today, a lower edge off the grid).
"""

from __future__ import annotations

from typing import Any

from lizyml.core.types.search_dim import FloatDim
from lizyml.tuning import search_space


def _expand(dim: FloatDim, best: float) -> Any:
    dims = search_space.attach_bounds([dim], {})
    report = search_space.detect_boundary(dims, {dim.name: best}, threshold=0.05)
    return search_space.expand_dims(dims, report)[0]


def test_spec_path_flags_a_huge_int() -> None:
    search_space.parse_space({"n": {"type": "int", "low": 0, "high": 2**60}})


def test_expand_path_flags_an_edge_past_the_float_range() -> None:
    assert _expand(FloatDim("x", 1e307, 1e308, log=True), 1e308).high == float("inf")


def test_expand_path_flags_a_stepped_edge_capped_today() -> None:
    dim = FloatDim("x", 1e307, 1e308, log=True, step=1e307)
    assert _expand(dim, 1e308).high == 1.7e308  # H-0124's cap; policy 10 drops it


def test_expand_path_flags_a_lower_edge_off_the_grid() -> None:
    dim = FloatDim("x", 1.9120500772236246e307, 2.48319537762e307, step=3e281)
    _expand(dim, 1.9120500772236246e307)
