"""Persistence of the per-fold pipeline states and old artifacts without them (H-0114).

A ``FitResult`` written before H-0114 has no ``pipeline_state_per_fold``. Such
artifacts are rebuilt here from a current export by deleting the attribute from
the pickled ``FitResult`` (and, for the v1 shape, ``target_encoder`` too), then
rewriting the SHA-256 the loader checks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.fit_result import FitResult
from lizyml.persistence.exporter import FORMAT_VERSION

pytest.importorskip("shap")

from tests.test_explain.test_shap_importance_per_fold_pipeline import (  # noqa: E402
    _config,
    _drift_df,
)

_FIELD = "pipeline_state_per_fold"


@pytest.fixture(scope="module")
def drift_df() -> pd.DataFrame:
    return _drift_df()


@pytest.fixture(scope="module")
def fitted(drift_df: pd.DataFrame) -> Model:
    model = Model(_config("regression", "mode"))
    model.fit(data=drift_df)
    return model


def _export(model: Model, tmp_path: Path) -> Path:
    out = tmp_path / "export"
    model.export(out)
    return out


def _rewrite_fit_result(
    export_dir: Path,
    *,
    remove: tuple[str, ...],
    format_version: int | None = None,
) -> None:
    """Drop attributes from the pickled FitResult and re-sign the file."""
    pkl = export_dir / "fit_result.pkl"
    fit_result = joblib.load(pkl)
    for name in remove:
        del fit_result.__dict__[name]
    joblib.dump(fit_result, pkl, compress=3)
    meta_path = export_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["checksums"]["files"]["fit_result.pkl"] = hashlib.sha256(
        pkl.read_bytes()
    ).hexdigest()
    if format_version is not None:
        meta["format_version"] = format_version
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _drop_analysis_context(export_dir: Path) -> None:
    (export_dir / "analysis_context.pkl").unlink()
    meta_path = export_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["checksums"]["files"].pop("analysis_context.pkl", None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["target"]).iloc[:20]


def _assert_refused_for_missing_state(model: Model) -> None:
    with pytest.raises(LizyMLError) as exc:
        model.importance(kind="shap")
    err = exc.value
    assert err.code is ErrorCode.MODEL_NOT_FIT
    assert err.context == {
        "task": "regression",
        "kind": "shap",
        "method": "importance",
        "missing": _FIELD,
    }
    message = str(err.user_message)
    assert _FIELD in message
    assert "before H-0114" in message
    assert "constructed without" in message
    assert "fit()" in message


# ---------------------------------------------------------------------------
# Acceptance 5: round trip
# ---------------------------------------------------------------------------


def test_export_load_keeps_the_per_fold_states(fitted: Model, tmp_path: Path) -> None:
    before = fitted.importance(kind="shap")

    loaded = Model.load(_export(fitted, tmp_path))

    assert FORMAT_VERSION == 2
    original = fitted._get_fit_state().fit_result.pipeline_state_per_fold
    restored = loaded._get_fit_state().fit_result.pipeline_state_per_fold
    assert original is not None
    assert restored == original
    after = loaded.importance(kind="shap")
    assert after == pytest.approx(before, rel=1e-12)


# ---------------------------------------------------------------------------
# Acceptance 6: artifacts written before H-0114
# ---------------------------------------------------------------------------


def test_artifact_without_per_fold_states_loads_and_refuses_shap(
    fitted: Model, drift_df: pd.DataFrame, tmp_path: Path
) -> None:
    X = _features(drift_df)
    pred_before = fitted.predict(X).pred
    split_before = fitted.importance(kind="split")
    gain_before = fitted.importance(kind="gain")
    export_dir = _export(fitted, tmp_path)
    _rewrite_fit_result(export_dir, remove=(_FIELD,))

    loaded = Model.load(export_dir)

    internal = loaded._get_fit_state().fit_result
    assert _FIELD not in internal.__dict__
    assert internal.pipeline_state_per_fold is None
    assert loaded.fit_result.pipeline_state_per_fold is None  # __deepcopy__
    assert dataclasses.replace(internal).pipeline_state_per_fold is None
    np.testing.assert_allclose(loaded.predict(X).pred, pred_before, rtol=1e-12)
    _assert_refused_for_missing_state(loaded)
    # Everything but SHAP importance keeps working.
    assert loaded.importance(kind="split") == split_before
    assert loaded.importance(kind="gain") == pytest.approx(gain_before, rel=1e-5)


def test_split_importance_plot_still_works_without_per_fold_states(
    fitted: Model, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    export_dir = _export(fitted, tmp_path)
    _rewrite_fit_result(export_dir, remove=(_FIELD,))

    Model.load(export_dir).importance_plot(kind="split")


def test_v1_shaped_artifact_goes_through_the_migration(
    fitted: Model, drift_df: pd.DataFrame, tmp_path: Path
) -> None:
    X = _features(drift_df)
    pred_before = fitted.predict(X).pred
    export_dir = _export(fitted, tmp_path)
    _rewrite_fit_result(export_dir, remove=(_FIELD, "target_encoder"), format_version=1)

    loaded = Model.load(export_dir)

    internal = loaded._get_fit_state().fit_result
    assert internal.target_encoder.needs_encoding is False
    assert internal.pipeline_state_per_fold is None
    np.testing.assert_allclose(loaded.predict(X).pred, pred_before, rtol=1e-12)
    _assert_refused_for_missing_state(loaded)


def test_missing_state_is_reported_before_missing_analysis_context(
    fitted: Model, tmp_path: Path
) -> None:
    export_dir = _export(fitted, tmp_path)
    _rewrite_fit_result(export_dir, remove=(_FIELD,))
    _drop_analysis_context(export_dir)

    _assert_refused_for_missing_state(Model.load(export_dir))


def test_missing_analysis_context_alone_keeps_its_error(
    fitted: Model, tmp_path: Path
) -> None:
    export_dir = _export(fitted, tmp_path)
    _drop_analysis_context(export_dir)

    with pytest.raises(LizyMLError) as exc:
        Model.load(export_dir).importance(kind="shap")
    assert exc.value.code is ErrorCode.MODEL_NOT_FIT
    assert "missing" not in exc.value.context


# ---------------------------------------------------------------------------
# Acceptance 7: a FitResult constructed without the field
# ---------------------------------------------------------------------------


def test_directly_constructed_fit_result_without_the_field(fitted: Model) -> None:
    source = fitted._get_fit_state().fit_result
    kwargs: dict[str, Any] = {
        f.name: getattr(source, f.name)
        for f in dataclasses.fields(FitResult)
        if f.name != _FIELD
    }
    constructed = FitResult(**kwargs)
    assert constructed.pipeline_state_per_fold is None

    model = Model(_config("regression", "mode"))
    model.fit(data=_drift_df())
    model._fit_result = constructed

    _assert_refused_for_missing_state(model)
    assert model.importance(kind="split") == fitted.importance(kind="split")
