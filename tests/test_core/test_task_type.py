"""Tests for the canonical ``TaskType`` Literal (#122, H-0075).

Pin down two contracts:

1. The runtime values match the public Config schema and BLUEPRINT §7.1.
2. Every layer that re-exports ``TaskType`` resolves to the same Literal
   (no silent drift between modules).
"""

from __future__ import annotations

import pytest

from lizyml.core.types.task import TASK_TYPES, TaskType


class TestTaskTypeContract:
    def test_known_values(self) -> None:
        assert set(TASK_TYPES) == {"regression", "binary", "multiclass"}

    def test_task_types_is_tuple(self) -> None:
        """``TASK_TYPES`` is iterable and ordered for parametrize use."""
        assert isinstance(TASK_TYPES, tuple)
        assert len(TASK_TYPES) == 3

    @pytest.mark.parametrize("task", TASK_TYPES)
    def test_each_value_round_trips(self, task: TaskType) -> None:
        assert task in TASK_TYPES


class TestTaskTypeReExports:
    """Every layer that historically defined its own ``TaskType`` Literal
    must now resolve to the same alias."""

    def test_target_encoder_re_export(self) -> None:
        from lizyml.core.types.target_encoder import TaskType as TET

        assert TET is TaskType

    def test_evaluator_re_export(self) -> None:
        from lizyml.evaluation.evaluator import TaskType as EvalT

        assert EvalT is TaskType

    def test_oof_assembly_re_export(self) -> None:
        from lizyml.training.oof_assembly import TaskType as OOFTaskType

        assert OOFTaskType is TaskType

    def test_lgbm_adapter_re_export(self) -> None:
        from lizyml.estimators.lgbm.adapter import TaskType as LGBMT

        assert LGBMT is TaskType

    def test_lgbm_defaults_uses_canonical_alias(self) -> None:
        from lizyml.estimators.lgbm.defaults import TaskType as DefT

        assert DefT is TaskType
