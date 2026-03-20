"""Tests for BlockedGroupKFoldConfig schema (H-0060).

TDD RED phase: these tests must fail until config models are implemented.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lizyml.config.schema import LizyMLConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(**split_overrides: object) -> dict:
    """Minimal LizyMLConfig dict with blocked_group_kfold split."""
    split = {
        "method": "blocked_group_kfold",
        "blocks": {
            "col": "date",
            "cutoffs": ["2025-02", "2025-03"],
        },
        "groups": {
            "col": "user_id",
            "n_splits": 3,
        },
    }
    split.update(split_overrides)
    return {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "y"},
        "split": split,
        "model": {"name": "lgbm"},
    }


# ---------------------------------------------------------------------------
# Parsing & defaults
# ---------------------------------------------------------------------------


class TestBlockedGroupKFoldConfigParsing:
    """Valid config parsing and default values."""

    def test_minimal_expanding_config(self) -> None:
        cfg = LizyMLConfig(**_base_config())
        sc = cfg.split
        assert sc.method == "blocked_group_kfold"
        assert sc.blocks.col == "date"
        assert sc.blocks.cutoffs == ["2025-02", "2025-03"]
        assert sc.blocks.mode == "expanding"
        assert sc.blocks.train_window is None

    def test_sliding_config(self) -> None:
        cfg = LizyMLConfig(
            **_base_config(
                blocks={
                    "col": "date",
                    "cutoffs": ["2025-03"],
                    "mode": "sliding",
                    "train_window": 2,
                }
            )
        )
        sc = cfg.split
        assert sc.blocks.mode == "sliding"
        assert sc.blocks.train_window == 2

    def test_groups_defaults(self) -> None:
        cfg = LizyMLConfig(**_base_config())
        sc = cfg.split
        assert sc.groups.col == "user_id"
        assert sc.groups.n_splits == 3
        assert sc.groups.stratify == "auto"
        assert sc.groups.shuffle is True

    def test_groups_explicit_values(self) -> None:
        cfg = LizyMLConfig(
            **_base_config(
                groups={
                    "col": "store_id",
                    "n_splits": 5,
                    "stratify": True,
                    "shuffle": False,
                }
            )
        )
        sc = cfg.split
        assert sc.groups.col == "store_id"
        assert sc.groups.n_splits == 5
        assert sc.groups.stratify is True
        assert sc.groups.shuffle is False

    def test_min_rows_defaults(self) -> None:
        cfg = LizyMLConfig(**_base_config())
        sc = cfg.split
        assert sc.min_train_rows == 10
        assert sc.min_valid_rows == 5

    def test_min_rows_custom(self) -> None:
        cfg = LizyMLConfig(**_base_config(min_train_rows=20, min_valid_rows=10))
        sc = cfg.split
        assert sc.min_train_rows == 20
        assert sc.min_valid_rows == 10

    def test_split_config_union_dispatches(self) -> None:
        """SplitConfig discriminated union resolves blocked_group_kfold."""
        cfg = LizyMLConfig(**_base_config())
        # Should be the new config type, not a generic dict
        assert hasattr(cfg.split, "blocks")
        assert hasattr(cfg.split, "groups")

    def test_round_trip(self) -> None:
        """model_dump → re-parse preserves all fields."""
        cfg = LizyMLConfig(**_base_config())
        dumped = cfg.model_dump()
        cfg2 = LizyMLConfig(**dumped)
        assert cfg2.split.method == "blocked_group_kfold"
        assert cfg2.split.blocks.cutoffs == ["2025-02", "2025-03"]
        assert cfg2.split.groups.n_splits == 3

    def test_stratify_string_values(self) -> None:
        """stratify accepts 'auto', 'true', 'false' as strings."""
        for val in ("auto", True, False):
            cfg = LizyMLConfig(
                **_base_config(groups={"col": "uid", "n_splits": 2, "stratify": val})
            )
            assert cfg.split.groups.stratify == val


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestBlockedGroupKFoldConfigValidation:
    """Config validation catches invalid inputs."""

    def test_same_col_raises(self) -> None:
        """blocks.col == groups.col is CONFIG_INVALID."""
        with pytest.raises(ValidationError, match="col"):
            LizyMLConfig(
                **_base_config(
                    blocks={"col": "user_id", "cutoffs": ["2025-03"]},
                    groups={"col": "user_id", "n_splits": 2},
                )
            )

    def test_sliding_without_train_window_raises(self) -> None:
        with pytest.raises(ValidationError, match="train_window"):
            LizyMLConfig(
                **_base_config(
                    blocks={"col": "date", "cutoffs": ["2025-03"], "mode": "sliding"}
                )
            )

    def test_empty_cutoffs_raises(self) -> None:
        with pytest.raises(ValidationError, match="cutoffs"):
            LizyMLConfig(**_base_config(blocks={"col": "date", "cutoffs": []}))

    def test_extra_key_in_blocks_raises(self) -> None:
        with pytest.raises(ValidationError):
            LizyMLConfig(
                **_base_config(
                    blocks={"col": "date", "cutoffs": ["2025-03"], "bad_key": 1}
                )
            )

    def test_extra_key_in_groups_raises(self) -> None:
        with pytest.raises(ValidationError):
            LizyMLConfig(
                **_base_config(groups={"col": "uid", "n_splits": 2, "bad_key": 1})
            )

    def test_expanding_with_train_window_warns(self) -> None:
        """train_window with expanding mode emits warning (value ignored)."""
        with pytest.warns(UserWarning, match="train_window"):
            LizyMLConfig(
                **_base_config(
                    blocks={
                        "col": "date",
                        "cutoffs": ["2025-03"],
                        "mode": "expanding",
                        "train_window": 2,
                    }
                )
            )
