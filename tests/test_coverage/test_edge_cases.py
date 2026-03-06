"""Edge-case tests to improve per-file coverage to >= 95%.

Each section targets a specific module's uncovered lines.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError


# ============================================================================
# 1. splitters/holdout.py  (60% -> 95%+)
# ============================================================================


class TestHoldoutSplitter:
    def test_ratio_zero_raises(self) -> None:
        from lizyml.splitters.holdout import HoldoutSplitter

        with pytest.raises(ValueError, match="ratio must be in"):
            HoldoutSplitter(ratio=0.0)

    def test_ratio_one_raises(self) -> None:
        from lizyml.splitters.holdout import HoldoutSplitter

        with pytest.raises(ValueError, match="ratio must be in"):
            HoldoutSplitter(ratio=1.0)

    def test_ratio_over_one_raises(self) -> None:
        from lizyml.splitters.holdout import HoldoutSplitter

        with pytest.raises(ValueError, match="ratio must be in"):
            HoldoutSplitter(ratio=1.5)

    def test_split_basic(self) -> None:
        from lizyml.splitters.holdout import HoldoutSplitter

        sp = HoldoutSplitter(ratio=0.2, random_state=0)
        folds = list(sp.split(100))
        assert len(folds) == 1
        train, valid = folds[0]
        assert len(train) + len(valid) == 100
        assert len(np.intersect1d(train, valid)) == 0

    def test_split_reproducible(self) -> None:
        from lizyml.splitters.holdout import HoldoutSplitter

        sp = HoldoutSplitter(ratio=0.2, random_state=42)
        _, v1 = list(sp.split(50))[0]
        _, v2 = list(sp.split(50))[0]
        np.testing.assert_array_equal(v1, v2)


# ============================================================================
# 2. splitters/__init__.py  (65% -> 95%+)
# ============================================================================


class TestBuildSplitter:
    def _spec(self, method: str, **kw: Any) -> Any:
        from lizyml.core.specs.split_spec import SplitSpec

        defaults: dict[str, Any] = {
            "n_splits": 3, "random_state": 42, "shuffle": True, "gap": 0,
        }
        defaults.update(kw)
        return SplitSpec(method=method, **defaults)

    def test_unknown_method_raises(self) -> None:
        from lizyml.splitters import _build_splitter

        spec = self._spec("unknown_method")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown split method"):
            _build_splitter(spec)

    def test_stratified_kfold(self) -> None:
        from lizyml.splitters import StratifiedKFoldSplitter, _build_splitter

        sp = _build_splitter(self._spec("stratified_kfold"))
        assert isinstance(sp, StratifiedKFoldSplitter)

    def test_group_kfold(self) -> None:
        from lizyml.splitters import GroupKFoldSplitter, _build_splitter

        sp = _build_splitter(self._spec("group_kfold"))
        assert isinstance(sp, GroupKFoldSplitter)

    def test_time_series(self) -> None:
        from lizyml.splitters import TimeSeriesSplitter, _build_splitter

        sp = _build_splitter(self._spec("time_series"))
        assert isinstance(sp, TimeSeriesSplitter)


# ============================================================================
# 3. core/logging.py  (75% -> 100%)
# ============================================================================


class TestLogEvent:
    def test_log_event_basic(self) -> None:
        from lizyml.core.logging import log_event

        logger = logging.getLogger("test.log_event")
        with patch.object(logger, "log") as mock_log:
            log_event(logger, "fit.start", run_id="abc", fold=0)
            mock_log.assert_called_once()
            msg = mock_log.call_args[0][1]
            assert "event='fit.start'" in msg
            assert "run_id='abc'" in msg
            assert "fold=0" in msg


# ============================================================================
# 4. explain/shap_explainer.py  (81% -> 95%+)
# ============================================================================


class TestNormalizeShapOutput:
    def test_legacy_list_binary(self) -> None:
        from lizyml.explain.shap_explainer import compute_shap_values

        arr0 = np.zeros((10, 5))
        arr1 = np.ones((10, 5))
        mock_model = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [arr0, arr1]

        with patch("lizyml.explain.shap_explainer._shap") as mock_shap:
            mock_shap.TreeExplainer.return_value = mock_explainer
            result = compute_shap_values(mock_model, pd.DataFrame(np.zeros((10, 5))), "binary")
        np.testing.assert_array_equal(result, arr1)

    def test_legacy_list_multiclass(self) -> None:
        from lizyml.explain.shap_explainer import compute_shap_values

        arrs = [np.random.randn(10, 5) for _ in range(3)]
        mock_model = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = arrs

        with patch("lizyml.explain.shap_explainer._shap") as mock_shap:
            mock_shap.TreeExplainer.return_value = mock_explainer
            result = compute_shap_values(mock_model, pd.DataFrame(np.zeros((10, 5))), "multiclass")
        assert result.shape == (10, 5)
        expected = np.mean(np.abs(np.stack(arrs, axis=0)), axis=0)
        np.testing.assert_allclose(result, expected)

    def test_fallback_to_asarray(self) -> None:
        from lizyml.explain.shap_explainer import compute_shap_values

        mock_model = MagicMock()
        mock_explainer = MagicMock()
        # Return a tuple (not list, not ndarray) to trigger fallback
        mock_explainer.shap_values.return_value = ((1.0, 2.0), (3.0, 4.0))

        with patch("lizyml.explain.shap_explainer._shap") as mock_shap:
            mock_shap.TreeExplainer.return_value = mock_explainer
            result = compute_shap_values(mock_model, pd.DataFrame(np.zeros((2, 2))), "regression")
        assert isinstance(result, np.ndarray)


# ============================================================================
# 5. evaluation/thresholding.py  (85% -> 100%)
# ============================================================================


class TestOptimiseThreshold:
    def test_minimize(self) -> None:
        from lizyml.evaluation.thresholding import optimise_threshold

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])

        def zero_one_loss(y_t: Any, y_p: Any) -> float:
            return float(np.mean(y_t != y_p))

        best_thresh, best_score = optimise_threshold(
            y_true, y_proba, zero_one_loss, greater_is_better=False
        )
        assert best_score <= 0.5
        assert 0.0 <= best_thresh <= 1.0


# ============================================================================
# 6. persistence/exporter.py  (85% -> 95%+)
# ============================================================================


class TestExportError:
    def test_serialization_error(self) -> None:
        from lizyml.persistence.exporter import export

        # Build a mock fit_result with .run_meta and .metrics
        mock_fr = MagicMock()
        mock_fr.run_meta.lizyml_version = "0.1.0"
        mock_fr.run_meta.python_version = "3.11"
        mock_fr.run_meta.timestamp = "2026-01-01"
        mock_fr.run_meta.run_id = "test"
        mock_fr.metrics = {}
        mock_fr.feature_names = ["a"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lizyml.persistence.exporter.joblib") as mock_jl:
                mock_jl.dump.side_effect = IOError("disk full")
                with pytest.raises(LizyMLError) as exc_info:
                    export(
                        path=tmpdir,
                        fit_result=mock_fr,
                        refit_result=MagicMock(),
                        config={"task": "regression"},
                        task="regression",
                    )
                assert exc_info.value.code == ErrorCode.SERIALIZATION_FAILED


# ============================================================================
# 7. persistence/loader.py  (86% -> 95%+)
# ============================================================================


class TestLoadErrors:
    def test_corrupt_metadata_json(self) -> None:
        from lizyml.persistence.loader import load

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "metadata.json").write_text("NOT VALID JSON{{{", encoding="utf-8")
            with pytest.raises(LizyMLError) as exc_info:
                load(tmpdir)
            assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_corrupt_fit_result_pkl(self) -> None:
        from lizyml.persistence.exporter import FORMAT_VERSION
        from lizyml.persistence.loader import load

        with tempfile.TemporaryDirectory() as tmpdir:
            meta = {
                "format_version": FORMAT_VERSION,
                "task": "regression",
                "feature_names": ["a"],
                "config": {},
                "run_id": "test",
            }
            (Path(tmpdir) / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            (Path(tmpdir) / "fit_result.pkl").write_bytes(b"corrupt data")
            (Path(tmpdir) / "refit_model.pkl").write_bytes(b"corrupt data")
            with pytest.raises(LizyMLError) as exc_info:
                load(tmpdir)
            assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED

    def test_corrupt_analysis_context(self) -> None:
        from lizyml.persistence.exporter import FORMAT_VERSION
        from lizyml.persistence.loader import load

        with tempfile.TemporaryDirectory() as tmpdir:
            meta = {
                "format_version": FORMAT_VERSION,
                "task": "regression",
                "feature_names": ["a"],
                "config": {},
                "run_id": "test",
            }
            (Path(tmpdir) / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            # Create valid pkl for fit_result and refit but corrupt analysis_context
            import joblib

            joblib.dump({"dummy": True}, Path(tmpdir) / "fit_result.pkl")
            joblib.dump({"dummy": True}, Path(tmpdir) / "refit_model.pkl")
            (Path(tmpdir) / "analysis_context.pkl").write_bytes(b"corrupt")
            with pytest.raises(LizyMLError) as exc_info:
                load(tmpdir)
            assert exc_info.value.code == ErrorCode.DESERIALIZATION_FAILED


# ============================================================================
# 8. splitters/purged_time_series.py  (86% -> 95%+)
# ============================================================================


class TestPurgedTimeSeries:
    def test_negative_gap_raises(self) -> None:
        from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter

        with pytest.raises(ValueError, match="gap must be >= 0"):
            PurgedTimeSeriesSplitter(gap=-1)

    def test_too_few_samples_raises(self) -> None:
        from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter

        sp = PurgedTimeSeriesSplitter(n_splits=5)
        with pytest.raises(ValueError, match="too small"):
            list(sp.split(3))

    def test_large_gap_skips_folds(self) -> None:
        from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter

        # gap=50 with 100 samples and 5 splits: some folds should be skipped
        sp = PurgedTimeSeriesSplitter(n_splits=5, gap=50)
        folds = list(sp.split(100))
        assert len(folds) < 5

    def test_large_purge_skips_folds(self) -> None:
        from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter

        # purge_window so large it consumes all training data
        sp = PurgedTimeSeriesSplitter(n_splits=3, purge_window=100)
        folds = list(sp.split(40))
        assert len(folds) == 0


# ============================================================================
# 9. data/fingerprint.py  (91% -> 95%+)
# ============================================================================


class TestFingerprint:
    def test_matches_no_file_hash(self) -> None:
        from lizyml.data.fingerprint import DataFingerprint

        fp1 = DataFingerprint(row_count=10, column_hash="abc")
        fp2 = DataFingerprint(row_count=10, column_hash="abc")
        assert fp1.matches(fp2)

    def test_hash_file_nonexistent(self) -> None:
        from lizyml.data.fingerprint import _hash_file

        result = _hash_file("/nonexistent/path/to/file.csv")
        assert result is None


# ============================================================================
# 10. features/encoders/categorical_encoder.py  (92% -> 95%+)
# ============================================================================


class TestCategoricalEncoder:
    def test_fit_missing_column(self) -> None:
        from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

        enc = CategoricalEncoder()
        df = pd.DataFrame({"a": [1, 2, 3]})
        enc.fit(df, ["nonexistent"])
        assert "nonexistent" not in enc._categories

    def test_fit_object_dtype(self) -> None:
        from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

        enc = CategoricalEncoder()
        df = pd.DataFrame({"col": ["b", "a", "c", "a"]})
        enc.fit(df, ["col"])
        assert enc._categories["col"] == ["a", "b", "c"]

    def test_fit_all_na_column(self) -> None:
        from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

        enc = CategoricalEncoder()
        df = pd.DataFrame({"col": pd.Categorical([None, None, None])})
        enc.fit(df, ["col"])
        # empty categories -> mode is None
        assert enc._modes["col"] is None

    def test_transform_before_fit(self) -> None:
        from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

        enc = CategoricalEncoder()
        with pytest.raises(RuntimeError, match="must be fitted"):
            enc.transform(pd.DataFrame({"col": ["a"]}))

    def test_transform_missing_column(self) -> None:
        from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

        enc = CategoricalEncoder()
        df = pd.DataFrame({"col": pd.Categorical(["a", "b"])})
        enc.fit(df, ["col"])
        # Transform a DF that doesn't have the fitted column
        result = enc.transform(pd.DataFrame({"other": [1, 2]}))
        assert "other" in result.columns


# ============================================================================
# 11. core/exceptions.py  (92% -> 100%)
# ============================================================================


class TestLizyMLErrorRepr:
    def test_repr_with_all_fields(self) -> None:
        cause = ValueError("root")
        err = LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message="bad config",
            debug_message="detail",
            context={"key": "val"},
            cause=cause,
        )
        r = repr(err)
        assert "debug_message='detail'" in r
        assert "context={'key': 'val'}" in r
        assert "cause=" in r

    def test_repr_minimal(self) -> None:
        err = LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message="bad",
        )
        r = repr(err)
        assert "debug_message" not in r
        assert "context" not in r
        assert "cause" not in r


# ============================================================================
# 12. splitters/group_time_series.py  (92% -> 95%+)
# ============================================================================


class TestGroupTimeSeries:
    def test_negative_gap_raises(self) -> None:
        from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter

        with pytest.raises(ValueError, match="gap must be >= 0"):
            GroupTimeSeriesSplitter(gap=-1)

    def test_large_gap_skips_folds(self) -> None:
        from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter

        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        # gap=3 with 8 groups and 3 splits needs at least 7 groups
        sp = GroupTimeSeriesSplitter(n_splits=3, gap=3)
        folds = list(sp.split(len(groups), groups=groups))
        # With gap consuming groups, some folds get skipped
        assert len(folds) < 3

    def test_valid_end_clamp(self) -> None:
        from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter

        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        sp = GroupTimeSeriesSplitter(n_splits=3, gap=0)
        folds = list(sp.split(len(groups), groups=groups))
        # Last fold may have valid_end clamped to n_groups
        for train, valid in folds:
            assert len(train) > 0
            assert len(valid) > 0


# ============================================================================
# 13. data/validators.py  (93% -> 95%+)
# ============================================================================


class TestValidators:
    def test_time_series_missing_col(self) -> None:
        from lizyml.data.validators import validate_time_series_order

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validate_time_series_order(df, "nonexistent")
        assert result == []

    def test_leakage_missing_target(self) -> None:
        from lizyml.data.validators import validate_no_target_leakage

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validate_no_target_leakage(df, "nonexistent")
        assert result == []

    def test_leakage_type_error(self) -> None:
        from lizyml.data.validators import validate_no_target_leakage

        # Mixed types that cause comparison issues
        df = pd.DataFrame({
            "target": [1, 2, 3],
            "mixed": [object(), object(), object()],
        })
        # Should not raise, returns empty list
        result = validate_no_target_leakage(df, "target", raise_on_violation=False)
        assert isinstance(result, list)


# ============================================================================
# 14. config/loader.py  (94% -> 95%+)
# ============================================================================


class TestConfigLoader:
    def test_normalize_model_not_dict(self) -> None:
        from lizyml.config.loader import _normalize_model_config

        raw = {"model": "not_a_dict", "task": "regression"}
        result = _normalize_model_config(raw)
        assert result["model"] == "not_a_dict"

    def test_normalize_model_env_merge(self) -> None:
        from lizyml.config.loader import _normalize_model_config

        # model has "name" and a stray nested key (e.g. from env override)
        raw = {
            "model": {
                "name": "lgbm",
                "params": {"learning_rate": 0.01},
                "lgbm": {"params": {"n_estimators": 100}, "auto_num_leaves": True},
            }
        }
        result = _normalize_model_config(raw)
        model = result["model"]
        assert model["name"] == "lgbm"
        assert model["params"]["n_estimators"] == 100
        assert model["params"]["learning_rate"] == 0.01
        assert model["auto_num_leaves"] is True
        assert "lgbm" not in model

    def test_normalize_model_no_match(self) -> None:
        from lizyml.config.loader import _normalize_model_config

        raw = {"model": {"unknown_model": {"params": {}}}}
        result = _normalize_model_config(raw)
        assert result["model"] == {"unknown_model": {"params": {}}}

    def test_env_override_empty_path(self) -> None:
        from lizyml.config.loader import _apply_env_overrides

        with patch.dict(os.environ, {"LIZYML__": "value"}):
            result = _apply_env_overrides({"task": "regression"})
        assert result == {"task": "regression"}

    def test_env_coerce_float(self) -> None:
        from lizyml.config.loader import _coerce_env_value

        assert _coerce_env_value("1.5") == 1.5
        assert _coerce_env_value("abc") == "abc"

    def test_env_override_non_dict_node(self) -> None:
        from lizyml.config.loader import _apply_env_overrides

        with patch.dict(os.environ, {"LIZYML__TASK__NESTED": "value"}):
            result = _apply_env_overrides({"task": "regression"})
        # task is a string, can't nest into it — gracefully handled
        assert result["task"] == "regression"

    def test_load_yaml_not_dict(self) -> None:
        from lizyml.config.loader import _read_raw

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("- item1\n- item2\n")
            f.flush()
            with pytest.raises(LizyMLError) as exc_info:
                _read_raw(f.name)
            assert exc_info.value.code == ErrorCode.CONFIG_INVALID
            os.unlink(f.name)
