"""pytest plugin: census of regression fits that ask to stratify on the target (H-0120).

Each ``build_splitter`` call is one unit of the population: ``Model.fit`` and
``Model.tune`` build the outer splitter through it before any training. A call
"fires" when the task is ``regression`` and the config asks to stratify on the
target at any position H-0120 amendment 2 refuses: ``split.method``
``stratified_kfold`` / ``stratified_group_kfold``, an explicit
``split.groups.stratify: true`` (``blocked_group_kfold``), or an explicit
``training.early_stopping.inner_valid`` with ``stratify: true``. Calls are
counted before the original runs, so a call that later stops is still counted.
The tally is written to $H0120_STRATIFIED_CENSUS_OUT at session end.
"""

from __future__ import annotations

import collections
import json
import os
from typing import Any

_STRATIFIED = ("stratified_kfold", "stratified_group_kfold")
_counts: collections.Counter[str] = collections.Counter()
_examples: list[dict[str, str]] = []


def _stratify_positions(cfg: Any) -> list[str]:
    """The config positions that ask to stratify on the target (H-0120 amendment 2)."""
    positions: list[str] = []
    if cfg.split.method in _STRATIFIED:
        positions.append(f"split.method={cfg.split.method}")
    groups = getattr(cfg.split, "groups", None)
    if groups is not None and groups.stratify is True:
        positions.append("split.groups.stratify=true")
    inner_valid = cfg.training.early_stopping.inner_valid
    if (
        inner_valid is not None
        and cfg.training.early_stopping._inner_valid_explicit
        and getattr(inner_valid, "stratify", False) is True
    ):
        positions.append("inner_valid.stratify=true")
    return positions


def pytest_configure(config: object) -> None:
    from lizyml.core import _model_factories, _model_tuning, model

    original = _model_factories.build_splitter

    def build_splitter(cfg: Any, *args: Any, **kwargs: Any) -> Any:
        _counts["calls"] += 1
        _counts[f"task={cfg.task}"] += 1
        if cfg.task == "regression":
            positions = _stratify_positions(cfg)
            if positions:
                _counts["firing_calls"] += 1
                for position in positions:
                    _counts[f"firing:{position}"] += 1
                if len(_examples) < 20:
                    _examples.append(
                        {
                            "test": os.environ.get("PYTEST_CURRENT_TEST", "?"),
                            "positions": ",".join(positions),
                        }
                    )
        return original(cfg, *args, **kwargs)

    for module in (_model_factories, _model_tuning, model):
        module.build_splitter = build_splitter  # type: ignore[attr-defined]


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    out = os.environ.get("H0120_STRATIFIED_CENSUS_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"counts": dict(_counts), "examples": _examples}, handle, indent=2)
