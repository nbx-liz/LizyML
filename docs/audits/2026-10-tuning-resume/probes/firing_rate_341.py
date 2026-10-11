"""Firing rate of the #341 refusal (H-0121 amendment 1), measured on the suite.

Loaded as a pytest plugin (``-p``) with ``docs/audits/2026-10-tuning-resume/probes``
on ``PYTHONPATH``. It wraps ``check_round_space`` where ``Model.tune`` calls it
and, without changing the outcome, evaluates the predicate the amendment adds:
does some combination of the values the round's ``category: smart`` dimensions
can take make an active smart parameter manage a name that a parameter layer
(the merged base model parameters or the round's fixed parameters) sets?

Population: every tune round that reaches ``check_round_space`` (one call per
round). Writes one line per round to ``$FIRING_RATE_341_OUT``:
``<nodeid>\t<fired 0/1>\t<names>``.

Run (from the repository root):
    FIRING_RATE_341_OUT=$TMPDIR/fr341.tsv PYTHONPATH=docs/audits/2026-10-tuning-resume/probes \\
        .venv/bin/python -m pytest -q -p no:cacheprovider -p firing_rate_341 tests
"""

from __future__ import annotations

import itertools
import os
from typing import Any

import pytest

from lizyml.core import _model_tuning
from lizyml.core.types.search_dim import CategoricalDim

_current = {"nodeid": "<none>"}
_rows: list[str] = []
_original = _model_tuning.check_round_space


def _values(dim: Any) -> tuple[Any, ...]:
    if isinstance(dim, CategoricalDim):
        return tuple(dim.choices)
    return (dim.low, dim.high)


def _predicate(cfg: Any, plan: Any, provider: Any, base_model: dict[str, Any],
               base_smart: dict[str, Any]) -> list[str]:
    smart_dims = [d for d in plan.effective if d.category == "smart"]
    if not smart_dims:
        return []
    layer_names = set(base_model) | set(plan.fixed)
    hits: set[str] = set()
    for combo in itertools.product(*(_values(d) for d in smart_dims)):
        smart = {**base_smart, **{d.name: v for d, v in zip(smart_dims, combo)}}
        managed = provider.smart_managed_param_names(smart, cfg.task)
        hits.update(name for name in layer_names if name in managed)
    return sorted(hits)


def _wrapped(cfg: Any, plan: Any, provider: Any, base_model: dict[str, Any],
             base_smart: dict[str, Any]) -> None:
    names = _predicate(cfg, plan, provider, base_model, base_smart)
    _rows.append(f"{_current['nodeid']}\t{int(bool(names))}\t{','.join(names)}")
    _original(cfg, plan, provider, base_model, base_smart)


_model_tuning.check_round_space = _wrapped  # type: ignore[assignment]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("FIRING_RATE_341_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_rows) + "\n")
