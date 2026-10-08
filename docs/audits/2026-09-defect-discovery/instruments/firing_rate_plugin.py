"""Record every `LizyMLConfig` the suite builds, for Change-Gate firing rates.

Three gates proposed by Phase 3 admit or reject inputs, so CLAUDE.md's Change
Gate requires a measured firing rate before implementation:

  * #262 - reject a `tuning.optuna.space` name whose category is `model` and
    which the provider does not recognise;
  * #258 - reject an explicit `direction` that contradicts the objective
    metric's `greater_is_better`;
  * #272 - enforce `config_version` wherever a `LizyMLConfig` is built.

The population is every config the shipped test suite actually constructs.
A static sweep over dict literals would miss the ones `tests/_helpers.py`
assembles from overrides, which is most of them.

`model_validate` is the only construction route inside `lizyml/`
(`config/loader.py:297`); `__init__` is patched too because tests may call it
directly. Both record *after* a successful build, so a config the current
schema already rejects never enters the population.

The output file is opened in append mode and never truncated here -- the
driver truncates it once before the run. `instruments/trace_plugin.py`
truncated at import, which meant any second pytest invocation silently
destroyed the first one's data.
"""

from __future__ import annotations

import json
import os
from typing import Any

OUT = os.environ.get("LIZYML_FIRING_OUT", "/tmp/lizyml-discovery-plan/results/firing_rows.jsonl")


def _describe(cfg: Any) -> dict[str, Any]:
    """Project a built config down to the fields the three gates read."""
    row: dict[str, Any] = {
        "config_version": getattr(cfg, "config_version", None),
        "task": getattr(cfg, "task", None),
        "metrics": list(getattr(getattr(cfg, "evaluation", None), "metrics", []) or []),
        "direction": None,
        "direction_explicit": None,
        "space": {},
        # `model.params` is the fit-path twin of `tuning.optuna.space`: whatever
        # a user puts there is forwarded to `lgb.train`. Recorded so the gate
        # proposed for one surface can be measured on the other.
        "model_params": [],
    }
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is not None:
        row["model_params"] = sorted(getattr(model_cfg, "params", None) or {})
    tuning = getattr(cfg, "tuning", None)
    if tuning is not None:
        optuna = getattr(tuning, "optuna", None)
        if optuna is not None:
            params = getattr(optuna, "params", None)
            if params is not None:
                row["direction"] = params.direction
                # `model_fields_set` is pydantic's record of which fields the
                # input actually supplied, as opposed to which took a default.
                # The #258 gate only rejects an *explicit* contradiction, so
                # the distinction is load-bearing rather than decorative.
                row["direction_explicit"] = "direction" in params.model_fields_set
            space = getattr(optuna, "space", None) or {}
            for name, spec in space.items():
                cat = "model"
                if isinstance(spec, dict):
                    cat = spec.get("category", "model")
                row["space"][name] = cat
    return row


def _record(cfg: Any) -> None:
    try:
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_describe(cfg)) + "\n")
    except Exception:  # pragma: no cover - instrument must never fail a test
        # Deliberately broad: an instrument that breaks the suite it measures
        # produces no measurement. The failure is visible as a missing row,
        # and the row count is reconciled against the config-building test
        # count by the tally.
        pass


def pytest_configure(config: Any) -> None:  # noqa: ARG001
    from lizyml.config.schema import LizyMLConfig

    orig_validate = LizyMLConfig.model_validate.__func__  # type: ignore[attr-defined]
    orig_init = LizyMLConfig.__init__

    def patched_validate(cls: Any, *a: Any, **k: Any) -> Any:
        out = orig_validate(cls, *a, **k)
        _record(out)
        return out

    def patched_init(self: Any, *a: Any, **k: Any) -> None:
        orig_init(self, *a, **k)
        _record(self)

    LizyMLConfig.model_validate = classmethod(patched_validate)  # type: ignore[assignment]
    LizyMLConfig.__init__ = patched_init  # type: ignore[method-assign]
