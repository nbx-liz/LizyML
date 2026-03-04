"""Tests for core exceptions, logging, seed, and import_optional."""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.seed import derive_seed, set_global_seed
from lizyml.utils.import_optional import import_optional

# ---------------------------------------------------------------------------
# LizyMLError
# ---------------------------------------------------------------------------


class TestLizyMLError:
    def test_fields_are_stored(self) -> None:
        err = LizyMLError(
            ErrorCode.CONFIG_INVALID,
            user_message="Bad config",
            debug_message="key 'x' not found",
            context={"config_path": "split.n_splits"},
        )
        assert err.code == ErrorCode.CONFIG_INVALID
        assert err.user_message == "Bad config"
        assert err.debug_message == "key 'x' not found"
        assert err.context == {"config_path": "split.n_splits"}
        assert err.cause is None

    def test_str_returns_user_message(self) -> None:
        err = LizyMLError(ErrorCode.MODEL_NOT_FIT, user_message="Model is not fit yet")
        assert str(err) == "[MODEL_NOT_FIT] Model is not fit yet"

    def test_repr_contains_code_and_message(self) -> None:
        err = LizyMLError(ErrorCode.UNSUPPORTED_TASK, user_message="Unknown task")
        r = repr(err)
        assert "UNSUPPORTED_TASK" in r
        assert "Unknown task" in r

    def test_cause_is_stored(self) -> None:
        original = ValueError("original")
        err = LizyMLError(
            ErrorCode.DATA_SCHEMA_INVALID,
            user_message="Schema error",
            cause=original,
        )
        assert err.cause is original

    def test_context_defaults_to_empty_dict(self) -> None:
        err = LizyMLError(ErrorCode.LEAKAGE_SUSPECTED, user_message="Leakage detected")
        assert err.context == {}

    def test_all_error_codes_are_defined(self) -> None:
        expected_codes = {
            "CONFIG_INVALID",
            "CONFIG_VERSION_UNSUPPORTED",
            "DATA_SCHEMA_INVALID",
            "DATA_FINGERPRINT_MISMATCH",
            "LEAKAGE_SUSPECTED",
            "LEAKAGE_CONFIRMED",
            "OPTIONAL_DEP_MISSING",
            "MODEL_NOT_FIT",
            "INCOMPATIBLE_COLUMNS",
            "UNSUPPORTED_TASK",
            "UNSUPPORTED_METRIC",
            "METRIC_REQUIRES_PROBA",
            "TUNING_FAILED",
            "CALIBRATION_NOT_SUPPORTED",
            "SERIALIZATION_FAILED",
            "DESERIALIZATION_FAILED",
        }
        defined_codes = {e.value for e in ErrorCode}
        assert expected_codes == defined_codes

    def test_is_exception_subclass(self) -> None:
        err = LizyMLError(ErrorCode.MODEL_NOT_FIT, user_message="Not fit")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            raise LizyMLError(
                ErrorCode.INCOMPATIBLE_COLUMNS,
                user_message="Column mismatch",
                context={"missing": ["col_a"], "extra": ["col_b"]},
            )
        assert exc_info.value.code == ErrorCode.INCOMPATIBLE_COLUMNS
        assert exc_info.value.context["missing"] == ["col_a"]


# ---------------------------------------------------------------------------
# import_optional
# ---------------------------------------------------------------------------


class TestImportOptional:
    def test_imports_installed_module(self) -> None:
        np_mod = import_optional("numpy")
        import numpy  # noqa: PLC0415

        assert np_mod is numpy

    def test_raises_on_missing_module(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            import_optional("_nonexistent_module_xyz_")
        err = exc_info.value
        assert err.code == ErrorCode.OPTIONAL_DEP_MISSING
        assert "_nonexistent_module_xyz_" in err.user_message
        assert err.context["module"] == "_nonexistent_module_xyz_"

    def test_custom_install_hint_in_message(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            import_optional(
                "_nonexistent_module_xyz_",
                package_name="nonexistent-pkg",
                install_hint="pip install nonexistent-pkg --extra-index-url ...",
            )
        assert "pip install nonexistent-pkg --extra-index-url" in (
            exc_info.value.user_message
        )

    def test_cause_is_import_error(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            import_optional("_nonexistent_module_xyz_")
        assert isinstance(exc_info.value.cause, ImportError)


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


class TestSeed:
    def test_set_global_seed_makes_numpy_reproducible(self) -> None:
        set_global_seed(42)
        a = np.random.rand(10)
        set_global_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_results(self) -> None:
        set_global_seed(1)
        a = np.random.rand(10)
        set_global_seed(2)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_derive_seed_is_deterministic(self) -> None:
        assert derive_seed(42, 0) == derive_seed(42, 0)
        assert derive_seed(42, 1) == derive_seed(42, 1)

    def test_derive_seed_differs_across_folds(self) -> None:
        seeds = [derive_seed(42, i) for i in range(5)]
        assert len(set(seeds)) == 5, "All fold seeds must be distinct"

    def test_derive_seed_differs_across_base_seeds(self) -> None:
        assert derive_seed(42, 0) != derive_seed(43, 0)

    def test_derive_seed_returns_non_negative_int(self) -> None:
        for fold in range(10):
            s = derive_seed(12345, fold)
            assert isinstance(s, int)
            assert s >= 0
