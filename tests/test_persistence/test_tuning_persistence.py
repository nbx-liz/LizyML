"""Persistence of tuned parameters across export/load (H-0086, #215).

Before this fix, tuned best params lived only in the in-memory ``_tuning_result``
overlay applied at fit time. ``export()`` never wrote them and ``load()`` never
restored them, so a re-``fit()`` after ``Model.load()`` silently trained with
config defaults — a silent reproducibility drift (the artifact predicts with
tuned params but retrains with defaults).

These tests pin the contract: the tuned overlay survives export → load, and a
loaded model re-``fit()``s with the tuned params. Legacy artifacts without a
``tuning`` block still load (additive, ``format_version`` unchanged).
"""

from __future__ import annotations

import json

from lizyml.core.model import Model
from tests._helpers import make_config, make_regression_df


def _tuned_model() -> Model:
    df = make_regression_df(n=200, seed=0)
    cfg = make_config("regression", n_estimators=30, n_splits=2, tuning_n_trials=3)
    model = Model(cfg)
    model.tune(data=df)
    model.fit(data=df)
    return model


def test_tuning_block_written_to_metadata(tmp_path) -> None:
    model = _tuned_model()
    out = model.export(path=tmp_path / "artifact")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

    assert "tuning" in metadata
    tuning = metadata["tuning"]
    assert set(tuning) >= {
        "best_model_params",
        "best_smart_params",
        "best_training_params",
        "best_score",
        "metric_name",
        "direction",
    }
    # format_version stays 2 (additive block).
    assert metadata["format_version"] == 2


def test_load_restores_tuning_result(tmp_path) -> None:
    model = _tuned_model()
    expected = model._tuning_result
    assert expected is not None

    out = model.export(path=tmp_path / "artifact")
    loaded = Model.load(out)

    assert loaded._tuning_result is not None
    assert loaded._tuning_result.best_model_params == expected.best_model_params
    assert loaded._tuning_result.best_smart_params == expected.best_smart_params
    assert loaded._tuning_result.best_training_params == expected.best_training_params


def test_refit_after_load_reproduces_tuned_params(tmp_path) -> None:
    """The core silent-drift fix: ``_merge_params`` overlay must match before
    and after a save/load round-trip, so a re-``fit()`` uses the tuned params.
    """
    model = _tuned_model()
    provider = model._provider
    assert provider is not None
    params_before, smart_before = model._merge_params(provider)

    out = model.export(path=tmp_path / "artifact")
    loaded = Model.load(out)

    params_after, smart_after = loaded._merge_params(loaded._provider)
    assert params_after == params_before
    assert smart_after == smart_before


def test_legacy_artifact_without_tuning_block_still_loads(tmp_path) -> None:
    """A non-tuned artifact (or a pre-#215 one) has no ``tuning`` key; load must
    succeed and leave ``_tuning_result`` as ``None`` (back-compat)."""
    df = make_regression_df(n=120, seed=0)
    model = Model(make_config("regression", n_estimators=20, n_splits=2))
    model.fit(data=df)
    out = model.export(path=tmp_path / "artifact")

    metadata_path = out / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("tuning", None)  # simulate a pre-#215 artifact
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    loaded = Model.load(out)
    assert loaded._tuning_result is None
