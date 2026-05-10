"""TaskType — canonical Literal for the supported ML tasks (#122, H-0075).

Centralised here so every layer (Foundation → Composition) imports the
same alias and a future addition (e.g. ``"ranking"``, ``"multilabel"``)
is a single-line change. The runtime values match the public Config
schema and BLUEPRINT §7.1 — never widen the union without a Proposal.
"""

from __future__ import annotations

from typing import Literal

TaskType = Literal["regression", "binary", "multiclass"]
"""The set of ML tasks that LizyML currently supports."""

TASK_TYPES: tuple[TaskType, ...] = ("regression", "binary", "multiclass")
"""Tuple form for iteration / membership checks where ``Literal`` cannot
be used directly (e.g. ``pytest.parametrize`` arguments)."""
