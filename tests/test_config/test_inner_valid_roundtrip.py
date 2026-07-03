"""Config round-trip preserves an explicit ``inner_valid`` strategy (H-0086, #203).

``model_dump()`` always emits the computed ``validation_ratio`` field. Before
this fix, re-validating a dumped config flipped the explicitness heuristic
(``user_explicit_inner_valid = iv_in and not vr_present``) to ``False``, so the
factory silently auto-resolved the inner-valid strategy from the *outer* split —
replacing the user's explicit choice on every ``dump -> reload`` and
``export -> load -> fit``. That is leakage-relevant for time/group data.

These tests pin the round-trip contract: an explicit ``inner_valid`` stays
explicit across a dump/reload, while pure-legacy ``validation_ratio`` input keeps
its auto-resolve semantics (non-regression).
"""

from __future__ import annotations

from lizyml.config.schema import LizyMLConfig
from tests._helpers import make_config


def _explicit(cfg: LizyMLConfig) -> bool:
    return cfg.training.early_stopping._inner_valid_explicit


def test_explicit_inner_valid_survives_dump_reload() -> None:
    cfg_dict = make_config("regression")
    cfg_dict["training"]["early_stopping"] = {
        "inner_valid": {"method": "time_holdout", "ratio": 0.2}
    }
    cfg = LizyMLConfig(**cfg_dict)
    assert _explicit(cfg) is True

    reloaded = LizyMLConfig(**cfg.model_dump())
    # Before #203 this was False (auto-resolve took over on round-trip).
    assert _explicit(reloaded) is True
    assert reloaded.training.early_stopping.inner_valid.method == "time_holdout"
    assert reloaded.training.early_stopping.inner_valid.ratio == 0.2


def test_explicit_holdout_fields_survive_dump_reload() -> None:
    cfg_dict = make_config("regression")
    cfg_dict["training"]["early_stopping"] = {
        "inner_valid": {"method": "holdout", "ratio": 0.3, "random_state": 7}
    }
    cfg = LizyMLConfig(**cfg_dict)
    reloaded = LizyMLConfig(**cfg.model_dump())

    iv = reloaded.training.early_stopping.inner_valid
    assert _explicit(reloaded) is True
    assert iv.method == "holdout"
    assert iv.ratio == 0.3
    assert iv.random_state == 7


def test_legacy_validation_ratio_stays_auto_resolve() -> None:
    """Pure-legacy ``validation_ratio`` must keep ``explicit=False`` (auto-resolve)
    — the fix must not accidentally promote legacy input to explicit."""
    cfg_dict = make_config("regression")
    cfg_dict["training"]["early_stopping"] = {"validation_ratio": 0.15}
    cfg = LizyMLConfig(**cfg_dict)
    assert _explicit(cfg) is False

    reloaded = LizyMLConfig(**cfg.model_dump())
    assert _explicit(reloaded) is False


def test_default_early_stopping_is_not_explicit() -> None:
    cfg = LizyMLConfig(**make_config("regression"))
    assert _explicit(cfg) is False
    reloaded = LizyMLConfig(**cfg.model_dump())
    assert _explicit(reloaded) is False
