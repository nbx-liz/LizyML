"""Tests for H-0022 — EarlyStoppingConfig new defaults."""

from __future__ import annotations

from lizyml.config.schema import EarlyStoppingConfig


class TestEarlyStoppingDefaults:
    def test_default_enabled_true(self) -> None:
        es = EarlyStoppingConfig()
        assert es.enabled is True

    def test_default_rounds_150(self) -> None:
        es = EarlyStoppingConfig()
        assert es.rounds == 150

    def test_default_validation_ratio_01(self) -> None:
        es = EarlyStoppingConfig()
        assert es.validation_ratio == 0.1
        assert es.inner_valid is not None
        assert es.inner_valid.ratio == 0.1

    def test_explicit_override(self) -> None:
        es = EarlyStoppingConfig(enabled=False, rounds=50)
        assert es.enabled is False
        assert es.rounds == 50
