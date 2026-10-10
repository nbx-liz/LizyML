"""H-0120 acceptance criterion 5: negative controls.

Each fix is reverted on its own, and a case of the reproduction matrix (or of
the amendment tests) must then fail. A control that still passed would mean
the matrix does not witness that fix. The generated scripts are reverted by
editing the template text ``export_code`` writes; LizyML-side fixes are
reverted by patching the function that carries them.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.codegen import templates
from lizyml.core.exceptions import LizyMLError
from tests.test_codegen._retrain_harness import (
    assert_retrain_reproduces,
    generated_prediction,
    lizyml_prediction,
    make_config,
    make_frame,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _revert(monkeypatch: pytest.MonkeyPatch, name: str, old: str, new: str) -> None:
    """Replace *old* with *new* in a generated script's template, exactly once."""
    text = getattr(templates, name)
    assert text.count(old) == 1, f"control anchor not found once: {old!r}"
    monkeypatch.setattr(templates, name, text.replace(old, new))


def _fit(cfg: dict[str, Any], df: pd.DataFrame) -> Model:
    model = Model(cfg)
    model.fit(data=df)
    return model


def _fails(check: Callable[[], None]) -> None:
    with pytest.raises(AssertionError):
        check()


# ---------------------------------------------------------------------------
# Policy 1: the inner split and the callback condition
# ---------------------------------------------------------------------------


def test_policy_1_split_reverted_to_a_random_holdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _revert(
        monkeypatch,
        "_TRAIN_PY",
        "tr, va = inner_split(len(y), y.to_numpy(), groups, iv)",
        "perm = np.random.default_rng(CFG['seed']).permutation(len(y)); "
        "n_val = max(1, int(len(y) * iv['ratio'])); "
        "tr, va = perm[n_val:], perm[:n_val]",
    )
    df = make_frame("regression")
    model = _fit(make_config("regression", "kfold"), df)
    _fails(lambda: assert_retrain_reproduces(model, df, tmp_path / "gen"))


def test_policy_1_callback_reverted_to_always_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A validation set without a patience must not get a callback."""
    _revert(
        monkeypatch,
        "_TRAIN_PY",
        "if es_rounds is not None:",
        "es_rounds = es_rounds or 1\n        if es_rounds is not None:",
    )
    df = make_frame("binary")
    cfg = make_config("binary", "kfold", early_stopping={"enabled": False})
    cfg["tuning"] = {
        "optuna": {
            "params": {"n_trials": 1, "direction": "minimize"},
            "space": {
                "validation_ratio": {
                    "type": "categorical",
                    "choices": [0.3],
                    "category": "training",
                }
            },
            "space_mode": "replace",
        }
    }
    model = Model(cfg)
    model.tune(data=df)
    model.fit(data=df)
    _fails(lambda: assert_retrain_reproduces(model, df, tmp_path / "gen"))


# ---------------------------------------------------------------------------
# Policy 2: row order
# ---------------------------------------------------------------------------


def test_policy_2_sort_reverted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _revert(
        monkeypatch,
        "_TRAIN_PY",
        "order = training_order(df)",
        "order = np.arange(len(df))",
    )
    df = make_frame("regression")
    cfg = make_config("regression", "time_series", early_stopping={"enabled": False})
    model = _fit(cfg, df)
    _fails(lambda: assert_retrain_reproduces(model, df, tmp_path / "gen"))


# ---------------------------------------------------------------------------
# Policy 3: multiclass balanced weights
# ---------------------------------------------------------------------------


def test_policy_3_weights_reverted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _revert(
        monkeypatch,
        "_TRAIN_PY",
        'if CFG["sample_weight"] == "balanced":',
        "if False:",
    )
    df = make_frame("multiclass")
    model = _fit(make_config("multiclass", "kfold"), df)
    _fails(lambda: assert_retrain_reproduces(model, df, tmp_path / "gen"))


# ---------------------------------------------------------------------------
# Policy 5: category codes, and the declared categories
# ---------------------------------------------------------------------------

_CATEGORY_SPLITS = {"min_data_per_group": 5, "cat_smooth": 1.0, "cat_l2": 1.0}


def _state_matches_encoder(model: Model, project: Path) -> None:
    from tests.test_codegen.test_retrain_reproduction import (
        _assert_state_matches_encoder,
    )

    _assert_state_matches_encoder(model, project)


def test_policy_5_cast_reverted_to_sorted_by_str(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without the builder's cast, integer categories are numbered by str order."""
    _revert(
        monkeypatch,
        "_TRAIN_PY",
        'X_raw = cast_categoricals(df[CFG["feature_names"]])',
        'X_raw = df[CFG["feature_names"]]',
    )
    df = make_frame("binary")
    df["c"] = np.random.default_rng(5).choice([3, 1, 2, 10], len(df))
    cfg = make_config("binary", "kfold")
    cfg["model"]["params"].update(_CATEGORY_SPLITS)
    cfg["features"] = {"categorical": ["c"]}
    model = _fit(cfg, df)
    project = tmp_path / "gen"
    assert_retrain_reproduces(model, df, project)
    _fails(lambda: _state_matches_encoder(model, project))


def test_policy_5_codes_reverted_to_str_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keyed by str, "1" and 1 share a code, so predict.py drifts."""
    _revert(
        monkeypatch,
        "_PREDICT_PY",
        'series = series.astype("category").cat.set_categories(known)',
        # The pre-H-0120 lookup: a str -> code dict, later keys overwriting.
        "mapping = {str(v): i for i, v in enumerate(known)}; "
        "return series.astype(str).map(mapping).to_numpy(dtype=float)",
    )
    df = make_frame("binary")
    df["c"] = pd.Series(["1", 1, "2", 2] * (len(df) // 4), dtype=object)
    df["target"] = df["c"].map(lambda v: isinstance(v, str) != (str(v) == "2"))
    df["target"] = df["target"].astype(np.int64)
    cfg = make_config("binary", "kfold")
    cfg["model"]["params"].update(_CATEGORY_SPLITS)
    cfg["features"] = {"categorical": ["c"]}
    model = _fit(cfg, df)
    project = tmp_path / "gen"
    model.export_code(project)
    X = df.drop(columns=["target"])
    _fails(
        lambda: np.testing.assert_allclose(
            generated_prediction(project, X, "binary"),
            lizyml_prediction(model, X, "binary"),
            rtol=1e-7,
        )
    )


def test_declared_categories_restore_reverted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _revert(
        monkeypatch,
        "_TRAIN_PY",
        'declared = CFG["declared_categories"]',
        "declared = {}",
    )
    df = make_frame("binary")
    rng = np.random.default_rng(8)
    df["d"] = pd.Categorical(
        rng.choice(["v", "u"], len(df)), categories=["v", "u", "t"]
    )
    df["target"] = ((df["d"].astype(str) == "v") ^ (df["f0"] > 1.5)).astype(np.int64)
    cfg = make_config("binary", "kfold")
    cfg["model"]["params"].update(_CATEGORY_SPLITS)
    model = _fit(cfg, df)
    project = tmp_path / "gen"
    assert_retrain_reproduces(model, df, project, fmt="csv")
    _fails(lambda: _state_matches_encoder(model, project))


# ---------------------------------------------------------------------------
# Amendment 1: the weight rule from the fit's record
# ---------------------------------------------------------------------------


def test_amendment_1_record_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reading config + current tuning result instead of the record loses the
    weights a fit used before a tune chose ``balanced: False``."""
    real = Model._get_fit_state

    def without_record(self: Model) -> Any:
        return dataclasses.replace(real(self), applied_sample_weight=None)

    df = make_frame("multiclass")
    cfg = make_config("multiclass", "kfold")
    cfg["tuning"] = {
        "optuna": {
            "params": {"n_trials": 1, "direction": "minimize"},
            "space": {
                "balanced": {
                    "type": "categorical",
                    "choices": [False],
                    "category": "smart",
                }
            },
            "space_mode": "replace",
        }
    }
    model = Model(cfg)
    model.fit(data=df)
    model.tune(data=df)
    monkeypatch.setattr(Model, "_get_fit_state", without_record)
    project = tmp_path / "gen"
    model.export_code(project)
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert config["sample_weight"] is None  # the fit used "balanced"
    _fails(lambda: assert_retrain_reproduces(model, df, project, export=False))


# ---------------------------------------------------------------------------
# Amendment 2: the regression stratification check
# ---------------------------------------------------------------------------


def test_amendment_2_check_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    from lizyml.core import _model_factories

    monkeypatch.setattr(
        _model_factories, "check_regression_stratification", lambda cfg: None
    )
    df = make_frame("regression")
    df["target"] = np.clip(np.round(df["target"]), -2, 2)
    model = Model(make_config("regression", "stratified_group_kfold"))
    # Without the check, an integer-valued target is silently stratified.
    model.fit(data=df)
    with pytest.raises(LizyMLError):
        monkeypatch.undo()
        Model(make_config("regression", "stratified_group_kfold")).fit(data=df)
