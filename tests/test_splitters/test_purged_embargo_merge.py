"""``embargo`` is merged into ``purge_gap`` for ``purged_time_series`` (H-0115, #273).

The splitter subtracted ``purge_gap`` and ``embargo`` at the same position, so
the two were one knob. ``embargo`` is now a deprecated spelling whose value is
added to ``purge_gap``; the excluded rows do not change for any input that
fitted before, except ``embargo: true`` (formerly read as ``1``), which is now
refused.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.config.loader import load_config
from lizyml.core._model_factories import _auto_inner_gap
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.splitters import PurgedTimeSeriesSplitter
from lizyml.training.inner_valid import TimeHoldoutInnerValid
from tests._helpers import make_config

_DEPRECATED = r"purge_gap"


def _raw_split(**split: Any) -> dict[str, Any]:
    return {"method": "purged_time_series", "n_splits": 4, **split}


def _config(task: str = "regression", **split: Any) -> dict[str, Any]:
    raw = make_config(
        task,
        n_estimators=5,
        n_splits=4,
        split_method="purged_time_series",
        time_col="t",
    )
    raw["split"] = _raw_split(**split)
    return raw


def _load(**split: Any) -> Any:
    return load_config(_config(**split))


def _frame(task: str = "regression", n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "t": np.arange(n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    score = df["x1"] + 0.5 * df["x2"] + rng.normal(scale=0.1, size=n)
    df["target"] = (score > score.median()).astype(int) if task == "binary" else score
    return df


def _folds(splitter: PurgedTimeSeriesSplitter, n: int = 240) -> list[tuple[Any, Any]]:
    return [(tr.tolist(), va.tolist()) for tr, va in splitter.split(n)]


def _expect_config_invalid(**split: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(LizyMLError) as exc:
            _load(**split)
    assert exc.value.code is ErrorCode.CONFIG_INVALID


# ---------------------------------------------------------------------------
# Acceptance 1: the geometric fact (green before and after)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [60, 240, 1001])
@pytest.mark.parametrize("n_splits", [2, 4, 7])
@pytest.mark.parametrize("purge_gap", [0, 3, 11])
@pytest.mark.parametrize(("max_train", "max_test"), [(None, None), (25, 7)])
def test_no_fold_has_training_rows_after_its_validation_block(
    n_samples: int,
    n_splits: int,
    purge_gap: int,
    max_train: int | None,
    max_test: int | None,
) -> None:
    splitter = PurgedTimeSeriesSplitter(
        n_splits=n_splits,
        purge_gap=purge_gap,
        max_train_size=max_train,
        max_test_size=max_test,
    )
    folds = list(splitter.split(n_samples))
    assert folds
    for train_idx, valid_idx in folds:
        assert train_idx.max() < valid_idx.min()


# ---------------------------------------------------------------------------
# Acceptance 2: embargo is added to purge_gap
# ---------------------------------------------------------------------------


def test_embargo_is_added_to_purge_gap() -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        cfg = _load(purge_gap=5, embargo=2)

    assert cfg.split.purge_gap == 7
    assert not hasattr(cfg.split, "embargo")
    assert "embargo" not in cfg.model_dump()["split"]


def test_merged_folds_match_the_measured_dead_zone() -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        cfg = _load(purge_gap=5, embargo=2)
    splitter = PurgedTimeSeriesSplitter(n_splits=4, purge_gap=cfg.split.purge_gap)

    folds = list(splitter.split(240))

    # #273's measurement of purge_gap=5, embargo=2 on n=240.
    expected = [(40, 48, 95), (88, 96, 143), (136, 144, 191), (184, 192, 239)]
    assert [(tr.max(), va.min(), va.max()) for tr, va in folds] == expected
    assert _folds(splitter) == _folds(PurgedTimeSeriesSplitter(n_splits=4, purge_gap=7))


# ---------------------------------------------------------------------------
# Acceptance 3: legacy spellings and value checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["embargo", "embargo_pct", "gap"])
def test_each_spelling_adds_to_purge_gap_and_names_it(key: str) -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        cfg = _load(purge_gap=5, **{key: 2})

    assert cfg.split.purge_gap == 7


def test_explicit_zero_embargo_still_warns() -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        cfg = _load(purge_gap=5, embargo=0)

    assert cfg.split.purge_gap == 5


@pytest.mark.parametrize(
    "spellings",
    [
        {"embargo": 1, "gap": 1},
        {"embargo": 1, "embargo_pct": 1},
        {"embargo_pct": 1, "gap": 1},
    ],
)
def test_two_spellings_of_the_second_gap_are_refused(spellings: dict[str, int]) -> None:
    _expect_config_invalid(purge_gap=1, **spellings)


@pytest.mark.parametrize(
    "split",
    [
        {"embargo": True},
        {"embargo_pct": True},
        {"gap": True},
        {"purge_gap": 5, "embargo": -1},
        {"purge_gap": 5, "embargo_pct": -1},
        {"purge_gap": 5, "gap": -1},
        {"embargo": 0.5},
        {"embargo": -1},
        {"purge_gap": -1, "embargo": 2},
        {"purge_gap": -1},
        {"embargo": "1e3"},
        {"embargo": "1e400"},
        {"gap": "1e400"},
        {"embargo_pct": "1e400"},
    ],
)
def test_invalid_values_are_config_errors(split: dict[str, Any]) -> None:
    _expect_config_invalid(**split)


def test_string_values_keep_todays_integer_coercion() -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        cfg = _load(purge_gap="5", embargo="2")
    assert cfg.split.purge_gap == 7

    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        big = _load(embargo="9007199254740993")
    assert big.split.purge_gap == 9007199254740993


# ---------------------------------------------------------------------------
# Acceptance 4: the splitter's deprecated argument
# ---------------------------------------------------------------------------


def test_splitter_embargo_argument_is_added_to_purge_gap() -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        legacy = PurgedTimeSeriesSplitter(n_splits=4, purge_gap=5, embargo=2)

    assert legacy.purge_gap == 7
    assert not hasattr(legacy, "embargo")
    assert _folds(legacy) == _folds(PurgedTimeSeriesSplitter(n_splits=4, purge_gap=7))


def test_splitter_warns_on_explicit_zero_but_not_when_omitted() -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        PurgedTimeSeriesSplitter(n_splits=4, purge_gap=5, embargo=0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PurgedTimeSeriesSplitter(n_splits=4, purge_gap=5)


@pytest.mark.parametrize(("purge_gap", "embargo"), [(-1, 2), (0, -1)])
def test_splitter_refuses_negative_values_before_adding(
    purge_gap: int, embargo: int
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match=">= 0"):
            PurgedTimeSeriesSplitter(n_splits=4, purge_gap=purge_gap, embargo=embargo)


# ---------------------------------------------------------------------------
# Acceptance 5: the inner-valid gap equals the outer dead zone
# ---------------------------------------------------------------------------


def test_inner_gap_equals_the_merged_purge_gap() -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        cfg = _load(purge_gap=5, embargo=2)

    gap = _auto_inner_gap(cfg.split)

    assert gap == cfg.split.purge_gap == 7
    # #273's example: a 192-row outer train fold, ratio 0.1.
    train_idx, valid_idx = TimeHoldoutInnerValid(ratio=0.1, gap=gap).split(192)
    assert len(train_idx) == 166
    assert valid_idx.min() - train_idx.max() - 1 == 7


# ---------------------------------------------------------------------------
# Acceptance 6: end to end -- fit, calibration and tune match
# ---------------------------------------------------------------------------


def _fit(raw: dict[str, Any], task: str = "regression") -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = Model(raw)
    return model, model.fit(data=_frame(task))


def _same_splits(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return all(
        np.array_equal(x_tr, y_tr) and np.array_equal(x_va, y_va)
        for (x_tr, x_va), (y_tr, y_va) in zip(a, b, strict=True)
    )


def test_fit_with_embargo_matches_the_merged_purge_gap() -> None:
    _, legacy = _fit(_config(purge_gap=5, embargo=2))
    _, merged = _fit(_config(purge_gap=7))

    assert _same_splits(legacy.splits.outer, merged.splits.outer)
    assert _same_splits(legacy.splits.inner, merged.splits.inner)
    np.testing.assert_array_equal(legacy.oof_pred, merged.oof_pred)


def test_calibration_folds_match_the_merged_purge_gap() -> None:
    legacy_raw = _config("binary", purge_gap=5, embargo=2)
    merged_raw = _config("binary", purge_gap=7)
    for raw in (legacy_raw, merged_raw):
        raw["calibration"] = {"method": "platt"}

    _, legacy = _fit(legacy_raw, "binary")
    _, merged = _fit(merged_raw, "binary")

    assert legacy.splits.calibration is not None
    assert _same_splits(legacy.splits.calibration, merged.splits.calibration)


def test_tune_with_embargo_matches_the_merged_purge_gap() -> None:
    results = []
    for split in ({"purge_gap": 5, "embargo": 2}, {"purge_gap": 7}):
        raw = _config(**split)
        raw["tuning"] = {"optuna": {"params": {"n_trials": 2}}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = Model(raw, data=_frame())
        results.append(model.tune())
    legacy, merged = results

    assert [t.score for t in legacy.trials] == [t.score for t in merged.trials]


# ---------------------------------------------------------------------------
# Acceptance 7: the Config round trip
# ---------------------------------------------------------------------------


def test_dumped_config_reloads_without_warnings() -> None:
    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        cfg = _load(purge_gap=5, embargo=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reloaded = load_config(cfg.model_dump())

    assert reloaded.split.purge_gap == 7


# ---------------------------------------------------------------------------
# Acceptance 8: artifacts written before H-0115
# ---------------------------------------------------------------------------


def _export_with_split(tmp_path: Path, split: dict[str, Any]) -> tuple[Model, Path]:
    model, _ = _fit(_config(purge_gap=7))
    out = tmp_path / "export"
    model.export(out)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["config"]["split"] = {**meta["config"]["split"], **split}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return model, out


def test_old_artifact_with_embargo_loads_with_the_merged_gap(tmp_path: Path) -> None:
    original, out = _export_with_split(tmp_path, {"purge_gap": 5, "embargo": 2})

    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        loaded = Model.load(out)

    split_cfg = loaded._cfg.split
    assert split_cfg.purge_gap == 7
    assert not hasattr(split_cfg, "embargo")
    X = _frame().drop(columns=["target"]).iloc[:10]
    np.testing.assert_array_equal(loaded.predict(X).pred, original.predict(X).pred)

    code_dir = loaded.export_code(tmp_path / "code")
    generated = json.loads((code_dir / "config.json").read_text(encoding="utf-8"))
    generated_split = generated["split"]
    assert generated_split["purge_gap"] == 7
    assert "embargo" not in generated_split

    refit = loaded.fit(data=_frame())
    original_splits = original._get_fit_state().fit_result.splits
    assert _same_splits(refit.splits.outer, original_splits.outer)


def test_old_artifact_with_zero_embargo_loads(tmp_path: Path) -> None:
    _, out = _export_with_split(tmp_path, {"embargo": 0})

    with pytest.warns(DeprecationWarning, match=_DEPRECATED):
        loaded = Model.load(out)

    assert loaded._cfg.split.purge_gap == 7
