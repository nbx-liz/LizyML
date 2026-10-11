"""Golden: the public names of ``lizyml.tuning`` (H-0122 §7.6).

The top-level ``lizyml.__all__`` is pinned by
``tests/test_core/test_public_exports.py`` and does not change.
"""

from __future__ import annotations

import lizyml.tuning


def test_tuning_all_is_pinned() -> None:
    assert set(lizyml.tuning.__all__) == {
        "CategoricalDim",
        "FloatDim",
        "IntDim",
        "SearchDim",
        "Tuner",
        "parse_space",
        "suggest_params",
        # H-0122 §7.6
        "RoundRecord",
        "StudyLineage",
        "read_study_lineage",
    }
    assert len(lizyml.tuning.__all__) == len(set(lizyml.tuning.__all__))


def test_every_exported_name_resolves() -> None:
    for name in lizyml.tuning.__all__:
        assert getattr(lizyml.tuning, name) is not None, name
