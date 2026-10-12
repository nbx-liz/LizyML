"""Positive control for firing_rate_h0124_magnitude.py's wiring (run with the plugin).

Run:
  env PYTHONPATH=docs/audits/2026-10-tuning-resume/probes FIRING_RATE_H0124_OUT=<out> \
    .venv/bin/python -m pytest -q -p no:cacheprovider -p firing_rate_h0124_magnitude \
    docs/audits/2026-10-tuning-resume/probes/firing_rate_h0124_posctl.py

Each test drives one wrapped path with an input the rule flags, so the output
must contain a flagged ``spec`` row and a flagged ``expand`` row.
"""

from __future__ import annotations

from lizyml.core.types.search_dim import FloatDim
from lizyml.tuning import search_space


def test_spec_path_flags_a_huge_int() -> None:
    search_space.parse_space({"n": {"type": "int", "low": 0, "high": 2**60}})


def test_expand_path_flags_an_edge_past_the_float_range() -> None:
    dims = search_space.attach_bounds([FloatDim("x", 1e307, 1e308, log=True)], {})
    report = search_space.detect_boundary(dims, {"x": 1e308}, threshold=0.05)
    (expanded,) = search_space.expand_dims(dims, report)
    assert expanded.high == float("inf")  # today's behaviour, before policy 10
