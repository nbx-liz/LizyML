"""Config validation hardening (#210).

Three independently-verified gaps:

1. ``config_version`` gate is bypassed by a string value (pydantic lax coercion
   then accepts it), so an unsupported version like ``"999"`` loads silently.
2. Legacy ``embargo_pct`` / ``gap`` are migrated via ``int()``, so a fractional
   ``embargo_pct=0.05`` silently collapses to ``0`` — removing the leakage guard.
3. A shuffled explicit ``inner_valid`` (holdout) combined with a time-ordered
   outer split produces a temporally leaked early-stopping split with no warning.
"""

from __future__ import annotations

import pytest

from lizyml.config.loader import load_config
from lizyml.core._model_factories import build_inner_valid
from lizyml.core.exceptions import ErrorCode, LizyMLError


def _base(split: dict, training: dict | None = None) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "y"},
        "model": {"name": "lgbm"},
        "split": split,
        "training": training or {},
    }


# ---------------------------------------------------------------------------
# 1. config_version gate bypass by string
# ---------------------------------------------------------------------------


class TestConfigVersionGate:
    def test_string_version_is_gated(self) -> None:
        raw = _base({"method": "kfold"})
        raw["config_version"] = "999"
        with pytest.raises(LizyMLError) as exc:
            load_config(raw)
        assert exc.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED

    def test_int_unsupported_version_is_gated(self) -> None:
        raw = _base({"method": "kfold"})
        raw["config_version"] = 999
        with pytest.raises(LizyMLError) as exc:
            load_config(raw)
        assert exc.value.code == ErrorCode.CONFIG_VERSION_UNSUPPORTED

    def test_supported_string_version_still_loads(self) -> None:
        raw = _base({"method": "kfold"})
        raw["config_version"] = "1"
        cfg = load_config(raw)
        assert cfg.config_version == 1


# ---------------------------------------------------------------------------
# 2. legacy embargo_pct / gap fractional truncation
# ---------------------------------------------------------------------------


class TestLegacyEmbargoFractional:
    def test_fractional_embargo_pct_rejected(self) -> None:
        raw = _base({"method": "purged_time_series", "embargo_pct": 0.05})
        with pytest.warns(DeprecationWarning), pytest.raises(LizyMLError) as exc:
            load_config(raw)
        assert exc.value.code == ErrorCode.CONFIG_INVALID

    def test_integer_valued_embargo_pct_accepted(self) -> None:
        raw = _base({"method": "purged_time_series", "embargo_pct": 3})
        with pytest.warns(DeprecationWarning):
            cfg = load_config(raw)
        assert cfg.split.embargo == 3

    def test_fractional_legacy_gap_rejected(self) -> None:
        raw = _base({"method": "purged_time_series", "gap": 1.5})
        with pytest.warns(DeprecationWarning), pytest.raises(LizyMLError) as exc:
            load_config(raw)
        assert exc.value.code == ErrorCode.CONFIG_INVALID


# ---------------------------------------------------------------------------
# 3. shuffled inner_valid under time-ordered outer split → UserWarning
# ---------------------------------------------------------------------------


class TestShuffledInnerValidWarning:
    @staticmethod
    def _cfg_time_series_holdout(method: str) -> object:
        raw = _base(
            {"method": method},
            {
                "early_stopping": {
                    "enabled": True,
                    "rounds": 10,
                    "inner_valid": {"method": "holdout", "ratio": 0.15},
                }
            },
        )
        return load_config(raw)

    def test_holdout_under_time_series_warns(self) -> None:
        cfg = self._cfg_time_series_holdout("time_series")
        with pytest.warns(UserWarning, match="(?i)shuffl|time"):
            build_inner_valid(cfg)

    def test_holdout_under_purged_time_series_warns(self) -> None:
        cfg = self._cfg_time_series_holdout("purged_time_series")
        with pytest.warns(UserWarning, match="(?i)shuffl|time"):
            build_inner_valid(cfg)

    def test_time_holdout_under_time_series_does_not_warn(self) -> None:
        raw = _base(
            {"method": "time_series"},
            {
                "early_stopping": {
                    "enabled": True,
                    "rounds": 10,
                    "inner_valid": {"method": "time_holdout", "ratio": 0.15},
                }
            },
        )
        cfg = load_config(raw)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            build_inner_valid(cfg)  # must not raise
