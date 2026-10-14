"""Positive control for firing_rate_347.py's wiring (run with the plugin).

Run:
  env PYTHONPATH=docs/audits/2026-10-tuning-resume/probes FIRING_RATE_347_OUT=<out> \
    .venv/bin/python -m pytest -q -p no:cacheprovider -p firing_rate_347 \
    docs/audits/2026-10-tuning-resume/probes/firing_rate_347_posctl.py

The output must contain a flagged ``dim`` row (a string choice of
``auto_num_leaves``) and a flagged ``expand`` row (``num_leaves_ratio`` expanded
past 1.0 by today's provider bound).
"""

from __future__ import annotations

from lizyml.core import _tuning_admission
from lizyml.core.types.search_dim import CategoricalDim, FloatDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.tuning import search_space


def test_dim_path_flags_a_string_bool_choice() -> None:
    dim = CategoricalDim("auto_num_leaves", ("false",), category="smart")
    _tuning_admission.validate_tuning_dimensions(LGBMProvider(), [dim], {}, "binary")


def test_expand_path_flags_num_leaves_ratio_past_one() -> None:
    bounds = LGBMProvider().parameter_bounds("binary")
    dims = search_space.attach_bounds(
        [FloatDim("num_leaves_ratio", 0.5, 1.0, category="smart")], bounds
    )
    report = search_space.detect_boundary(dims, {"num_leaves_ratio": 1.0}, threshold=0.05)
    (expanded,) = search_space.expand_dims(dims, report)
    assert expanded.high > 1.0  # today's provider bound is 2.0
