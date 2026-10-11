"""Unit tests for the objective identity (H-0122 §4.1-§4.3, criteria 13/26/27/28).

Each data case asserts that the element expected to see the change is among
the mismatched ones and that the overall digest changes. The exact mismatched
set is not pinned, because which *other* elements also move depends on pandas
hashing internals (int64 -> float64 moves ``data.values``, int64 -> Int64 does
not). Every case was confirmed by execution to be invisible to
``hash_pandas_object`` alone where the design says so, which is why the
auxiliary elements exist.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.config.loader import load_config
from lizyml.config.schema import LizyMLConfig
from lizyml.core._tuning_identity import (
    CONFIG_EXCLUDED_FIELDS,
    DATA_ELEMENTS,
    ObjectiveIdentity,
    VersionDifference,
    build_identity,
    config_parts,
    library_versions,
    mismatched_elements,
    provider_default_params,
    summarise_frame,
    version_differences,
)
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests._helpers import make_config

# ---------------------------------------------------------------------------
# Data summary
# ---------------------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": pd.Series([1, 2, 3, 4], dtype="int64"),
            "flt": [0.5, 1.5, np.nan, 2.5],
            "cat": pd.Categorical(["a", "b", "a", "b"]),
            "mixed_cat": pd.Categorical([1, "1", 1, "1"]),
            "obj": pd.Series([1, "x", 2, None], dtype=object),
            "excluded": [10.0, 20.0, 30.0, 40.0],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )


def _changed(base: dict[str, str], other: pd.DataFrame) -> tuple[str, ...]:
    return mismatched_elements(base, summarise_frame(other))


def _replace_col(df: pd.DataFrame, name: str, values: Any) -> pd.DataFrame:
    out = df.copy()
    out[name] = values
    return out


_DATA_CASES: dict[str, tuple[Callable[[pd.DataFrame], pd.DataFrame], str]] = {
    "column_rename": (lambda d: d.rename(columns={"num": "num2"}), "data.columns"),
    "column_order": (lambda d: d[list(reversed(d.columns))], "data.columns"),
    "column_label_type": (lambda d: d.rename(columns={"num": 1}), "data.columns"),
    "int64_float64": (lambda d: d.astype({"num": "float64"}), "data.dtypes"),
    "int64_Int64": (lambda d: d.astype({"num": "Int64"}), "data.dtypes"),
    "category_order": (
        lambda d: _replace_col(d, "cat", d["cat"].cat.reorder_categories(["b", "a"])),
        "data.categories",
    ),
    "unused_category": (
        lambda d: _replace_col(d, "cat", d["cat"].cat.add_categories(["c"])),
        "data.categories",
    ),
    "ordered": (
        lambda d: _replace_col(d, "cat", d["cat"].cat.as_ordered()),
        "data.categories",
    ),
    "mixed_category_values_swapped": (
        lambda d: _replace_col(d, "mixed_cat", pd.Categorical(["1", 1, "1", 1])),
        "data.categorical_codes",
    ),
    "object_1_vs_str": (
        lambda d: _replace_col(d, "obj", pd.Series(["1", "x", 2, None], dtype=object)),
        "data.object_types",
    ),
    "object_nan_vs_none": (
        lambda d: _replace_col(d, "obj", pd.Series([1, "x", 2, np.nan], dtype=object)),
        "data.object_types",
    ),
    "index_labels": (lambda d: d.set_axis([10, 11, 12, 13]), "data.values"),
    "index_label_type": (
        lambda d: d.set_axis(pd.Index(["0", "1", "2", "3"], dtype=object)),
        "data.columns",
    ),
    "object_index_label_type": (
        lambda d: d.set_axis(pd.Index([0, "1", 2, 3], dtype=object)),
        "data.object_types",
    ),
    "row_order": (lambda d: d.iloc[::-1], "data.values"),
    "row_order_reset_index": (
        lambda d: d.iloc[::-1].reset_index(drop=True),
        "data.values",
    ),
    "excluded_values": (
        lambda d: _replace_col(d, "excluded", [10.0, 20.0, 30.0, 41.0]),
        "data.values",
    ),
    "a_value": (
        lambda d: _replace_col(d, "flt", [0.5, 1.5, np.nan, 2.6]),
        "data.values",
    ),
    "missing_vs_value": (
        lambda d: _replace_col(d, "flt", [0.5, 1.5, 0.0, 2.5]),
        "data.values",
    ),
    "drop_row": (lambda d: d.iloc[:3], "data.values"),
}


class TestSummariseFrame:
    def test_elements_are_the_declared_set(self) -> None:
        summary = summarise_frame(_frame())
        assert set(summary) == set(DATA_ELEMENTS)
        assert all(isinstance(v, str) and len(v) == 64 for v in summary.values())

    def test_identical_copy_gives_identical_summary(self) -> None:
        # Negative control: a deep copy and a frame rebuilt from scratch.
        base = summarise_frame(_frame())
        assert summarise_frame(_frame().copy(deep=True)) == base
        assert summarise_frame(_frame()) == base

    def test_range_index_equals_the_same_int64_labels(self) -> None:
        # Negative control: the index class is not part of the identity.
        df = _frame()
        relabelled = df.set_axis(pd.Index([0, 1, 2, 3], dtype="int64"))
        assert type(df.index) is not type(relabelled.index)
        assert summarise_frame(relabelled) == summarise_frame(df)

    def test_summary_does_not_mutate_the_frame(self) -> None:
        df = _frame()
        before = df.copy(deep=True)
        summarise_frame(df)
        pd.testing.assert_frame_equal(df, before)

    @pytest.mark.parametrize("case", sorted(_DATA_CASES))
    def test_each_change_is_detected(self, case: str) -> None:
        change, element = _DATA_CASES[case]
        base_df = _frame()
        base = summarise_frame(base_df)
        other = summarise_frame(change(base_df.copy(deep=True)))
        assert element in mismatched_elements(base, other), case
        assert build_identity(base, {}).digest != build_identity(other, {}).digest

    def test_hash_pandas_object_alone_misses_the_auxiliary_cases(self) -> None:
        # DC7: the auxiliary elements exist because hash_pandas_object is blind
        # to these. If pandas ever starts seeing them, this test says so.
        def raw(d: pd.DataFrame) -> bytes:
            return pd.util.hash_pandas_object(d, index=True).to_numpy().tobytes()

        base = _frame()
        for case in (
            "column_rename",
            "int64_Int64",
            "category_order",
            "unused_category",
            "ordered",
            "mixed_category_values_swapped",
            "object_1_vs_str",
            "object_nan_vs_none",
        ):
            change, _ = _DATA_CASES[case]
            assert raw(change(base.copy(deep=True))) == raw(base), case

    def test_duplicate_column_names(self) -> None:
        df = pd.concat([_frame()[["num"]], _frame()[["num"]]], axis=1)
        base = summarise_frame(df)
        other = df.copy()
        other.iloc[0, 1] = 99
        assert "data.values" in _changed(base, other)

    def test_duplicate_categorical_names_by_position(self) -> None:
        df = pd.concat([_frame()[["cat"]], _frame()[["cat"]]], axis=1)
        base = summarise_frame(df)
        other = pd.concat(
            [
                _frame()[["cat"]],
                _frame()[["cat"]].apply(lambda c: c.cat.as_ordered()),
            ],
            axis=1,
        )
        assert "data.categories" in _changed(base, other)

    def test_multiindex_rows(self) -> None:
        idx = pd.MultiIndex.from_tuples([(1, "a"), (1, "b"), (2, "a"), (2, "b")])
        df = _frame().set_axis(idx)
        base = summarise_frame(df)
        assert summarise_frame(df.copy(deep=True)) == base
        other = df.set_axis(
            pd.MultiIndex.from_tuples([(1, "a"), (1, "b"), (2, "a"), (3, "b")])
        )
        assert "data.values" in _changed(base, other)

    def test_unhashable_object_cells_fall_back(self) -> None:
        # R9: list cells make hash_pandas_object raise TypeError.
        df = _frame()
        df["lists"] = pd.Series([[1], [2], [3], {"a": 1}], dtype=object)
        with pytest.raises(TypeError):
            pd.util.hash_pandas_object(df, index=True)
        base = summarise_frame(df)
        assert summarise_frame(df.copy(deep=True)) == base
        other = df.copy()
        other["lists"] = pd.Series([[1], [2], [4], {"a": 1}], dtype=object)
        assert "data.values" in _changed(base, other)

    def test_list_cells_vs_their_repr_strings(self) -> None:
        df = pd.DataFrame({"x": pd.Series([[1], [2]], dtype=object)})
        as_text = pd.DataFrame({"x": pd.Series(["[1]", "[2]"], dtype=object)})
        assert "data.object_types" in _changed(summarise_frame(df), as_text)

    def test_empty_and_zero_column_frames(self) -> None:
        empty = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        no_cols = pd.DataFrame(index=range(3))
        assert summarise_frame(empty) != summarise_frame(no_cols)

    def test_string_dtype_column(self) -> None:
        df = pd.DataFrame({"s": ["a", "b"]})
        base = summarise_frame(df)
        assert "data.values" in _changed(base, pd.DataFrame({"s": ["a", "c"]}))
        as_object = pd.DataFrame({"s": pd.Series(["a", "b"], dtype=object)})
        assert "data.dtypes" in _changed(base, as_object)


# ---------------------------------------------------------------------------
# Config parts
# ---------------------------------------------------------------------------

_FIXED = {"first_metric_only": True, "metric": ["huber", "mae", "mape"]}
_DEFAULTS = {"max_bin": 511, "lambda_l2": 1e-06, "objective": "huber"}


def _cfg(
    edit: Callable[[dict[str, Any]], None] | None = None, task: str = "regression"
) -> LizyMLConfig:
    raw = make_config(task, tuning_n_trials=5)
    raw["tuning"]["optuna"]["space"] = {
        "num_leaves": {"type": "int", "low": 4, "high": 8}
    }
    if edit is not None:
        edit(raw)
    return load_config(raw)


def _parts(
    cfg: LizyMLConfig,
    fixed: dict[str, Any] | None = None,
    direction: str = "minimize",
    provider_defaults: dict[str, Any] | None = None,
) -> dict[str, str]:
    return config_parts(
        cfg,
        _FIXED if fixed is None else fixed,
        direction,
        provider_defaults=_DEFAULTS if provider_defaults is None else provider_defaults,
    )


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    keys = path.split(".")

    def edit(raw: dict[str, Any]) -> None:
        node = raw
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    return edit


_INCLUDED_CASES: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "data.target": (_set("data.target", "feat_a"), "config.data"),
    "data.time_col": (_set("data.time_col", "t"), "config.data"),
    "data.group_col": (_set("data.group_col", "g"), "config.data"),
    "features.exclude": (_set("features.exclude", ["feat_b"]), "config.features"),
    "features.categorical": (
        _set("features.categorical", ["feat_b"]),
        "config.features",
    ),
    "features.auto_categorical": (
        _set("features.auto_categorical", False),
        "config.features",
    ),
    "features.unseen_policy": (
        _set("features.unseen_policy", "nan"),
        "config.features",
    ),
    "model.params": (_set("model.params", {"n_estimators": 11}), "config.model"),
    "model.params_int_vs_float": (
        _set("model.params", {"n_estimators": 10.0}),
        "config.model",
    ),
    "model.num_leaves_ratio": (_set("model.num_leaves_ratio", 0.5), "config.model"),
    "model.auto_num_leaves": (_set("model.auto_num_leaves", False), "config.model"),
    "model.balanced": (_set("model.balanced", True), "config.model"),
    "model.feature_weights": (
        _set("model.feature_weights", {"feat_a": 2.0}),
        "config.model",
    ),
    "training.seed": (_set("training.seed", 7), "config.training"),
    "training.early_stopping.rounds": (
        _set("training.early_stopping", {"rounds": 20}),
        "config.training",
    ),
    "training.inner_valid": (
        _set(
            "training.early_stopping",
            {"inner_valid": {"method": "holdout", "ratio": 0.2}},
        ),
        "config.training",
    ),
    "split.n_splits": (_set("split.n_splits", 4), "config.split"),
    "split.random_state": (_set("split.random_state", 1), "config.split"),
    "metric.first": (_set("evaluation.metrics", ["mae"]), "metric"),
    "metric.kwargs": (
        _set("evaluation.metrics", [{"huber": {"alpha": 0.5}}]),
        "metric",
    ),
}


class TestConfigParts:
    def test_parts_are_digests(self) -> None:
        parts = _parts(_cfg())
        assert {
            "config.task",
            "config.data",
            "config.features",
            "config.model",
            "config.fixed",
            "config.provider_defaults",
            "config.training",
            "config.split",
            "metric",
        } <= set(parts)
        assert all(len(v) == 64 for v in parts.values())

    def test_same_config_same_parts(self) -> None:
        assert _parts(_cfg()) == _parts(_cfg())

    @pytest.mark.parametrize("case", sorted(_INCLUDED_CASES))
    def test_included_change_moves_only_its_element(self, case: str) -> None:
        edit, element = _INCLUDED_CASES[case]
        try:
            changed = _cfg(edit)
        except Exception as exc:  # pragma: no cover - surfaced as a test bug
            pytest.fail(f"{case}: config did not validate: {exc}")
        diff = mismatched_elements(_parts(_cfg()), _parts(changed))
        assert diff == (element,), case

    def test_task_change_names_config_task(self) -> None:
        diff = mismatched_elements(_parts(_cfg()), _parts(_cfg(task="binary")))
        assert "config.task" in diff

    def test_fixed_change_moves_config_fixed_only(self) -> None:
        # Criterion 28 (unit half): a different provider default ``fixed``.
        other = {"first_metric_only": False, "metric": ["huber", "mae", "mape"]}
        diff = mismatched_elements(_parts(_cfg()), _parts(_cfg(), fixed=other))
        assert diff == ("config.fixed",)

    def test_fixed_list_and_tuple_are_the_same(self) -> None:
        # A stored fixed comes back from JSON with lists.
        as_tuple = {"first_metric_only": True, "metric": ("huber", "mae", "mape")}
        assert _parts(_cfg(), fixed=as_tuple) == _parts(_cfg())

    def test_fixed_bool_vs_int_differ(self) -> None:
        one = {"first_metric_only": 1, "metric": ["huber", "mae", "mape"]}
        assert mismatched_elements(_parts(_cfg()), _parts(_cfg(), fixed=one)) == (
            "config.fixed",
        )

    def test_direction_change_moves_metric(self) -> None:
        diff = mismatched_elements(_parts(_cfg()), _parts(_cfg(), direction="maximize"))
        assert diff == ("metric",)

    def test_default_metrics_equal_their_explicit_first_entry(self) -> None:
        explicit = _cfg(_set("evaluation.metrics", ["rmse"]))
        assert _parts(explicit) == _parts(_cfg())

    @pytest.mark.parametrize(
        "edit",
        [
            _set(
                "tuning.optuna.space",
                {"num_leaves": {"type": "int", "low": 4, "high": 9}},
            ),
            _set("tuning.optuna.space", {}),
            _set("tuning.optuna.space_mode", "replace"),
            _set("tuning.optuna.params", {"n_trials": 99, "direction": "minimize"}),
            _set("tuning.optuna.params", {"n_trials": 5, "timeout": 30.0}),
            _set("evaluation.metrics", ["rmse", "mape"]),
            _set("output_dir", "/tmp/somewhere"),
            _set("data.path", "train.csv"),
        ],
        ids=[
            "space_range",
            "space_empty",
            "space_mode",
            "n_trials",
            "timeout_and_direction_omitted",
            "second_metric",
            "output_dir",
            "data_path",
        ],
    )
    def test_excluded_change_moves_nothing(
        self, edit: Callable[[dict[str, Any]], None]
    ) -> None:
        assert _parts(_cfg(edit)) == _parts(_cfg())

    def test_calibration_is_excluded(self) -> None:
        base = load_config(make_config("binary", tuning_n_trials=5))
        calibrated = load_config(
            make_config("binary", tuning_n_trials=5, calibration="platt")
        )
        assert _parts(base) == _parts(calibrated)

    def test_tuning_absent_vs_present_is_excluded(self) -> None:
        with_tuning = load_config(make_config("regression", tuning_n_trials=5))
        without = load_config(make_config("regression"))
        assert _parts(with_tuning) == _parts(without)

    def test_every_config_field_is_classified(self) -> None:
        # DC3: a new top-level field is included unless it is listed here.
        assert (
            frozenset({"tuning", "calibration", "evaluation", "output_dir"})
            == CONFIG_EXCLUDED_FIELDS
        )
        included = set(LizyMLConfig.model_fields) - CONFIG_EXCLUDED_FIELDS
        parts = _parts(_cfg())
        assert {f"config.{name}" for name in included} <= set(parts)

    def test_timestamp_values_are_encoded_deterministically(self) -> None:
        def edit(raw: dict[str, Any]) -> None:
            raw["model"]["params"]["custom"] = pd.Timestamp("2025-01-01")

        a = _parts(_cfg(edit))
        b = _parts(_cfg(edit))
        assert a == b
        assert a["config.model"] != _parts(_cfg())["config.model"]

    def test_numpy_scalars_equal_plain_values(self) -> None:
        def as_numpy(raw: dict[str, Any]) -> None:
            raw["model"]["params"]["n_estimators"] = np.int64(10)

        assert _parts(_cfg(as_numpy)) == _parts(_cfg())

    def test_mixed_key_types_do_not_crash(self) -> None:
        def edit(raw: dict[str, Any]) -> None:
            raw["model"]["params"]["class_weight_like"] = {0: 1.0, "1": 2.0}

        def edit_str(raw: dict[str, Any]) -> None:
            raw["model"]["params"]["class_weight_like"] = {"0": 1.0, "1": 2.0}

        assert (
            _parts(_cfg(edit))["config.model"] != _parts(_cfg(edit_str))["config.model"]
        )


# ---------------------------------------------------------------------------
# Provider default parameters (spec 4.3, Q-E)
# ---------------------------------------------------------------------------


class TestProviderDefaults:
    def test_element_is_present(self) -> None:
        assert "config.provider_defaults" in _parts(_cfg())

    def test_default_change_moves_only_its_element(self) -> None:
        changed = {**_DEFAULTS, "max_bin": 15}
        diff = mismatched_elements(
            _parts(_cfg()), _parts(_cfg(), provider_defaults=changed)
        )
        assert diff == ("config.provider_defaults",)

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_lgbm_provider_reports_its_defaults(
        self, task: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lizyml.estimators.lgbm import defaults as lgbm_defaults

        provider = LGBMProvider()
        before = provider_default_params(provider, task)  # type: ignore[arg-type]
        assert before["max_bin"] == lgbm_defaults._COMMON_DEFAULTS["max_bin"]
        monkeypatch.setitem(lgbm_defaults._COMMON_DEFAULTS, "max_bin", 15)
        after = provider_default_params(provider, task)  # type: ignore[arg-type]
        assert after["max_bin"] == 15

    def test_provider_without_the_method_contributes_nothing(self) -> None:
        class ThirdParty:
            pass

        assert provider_default_params(ThirdParty(), "regression") == {}  # type: ignore[arg-type]

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_defaults_are_exactly_what_the_adapter_starts_from(self, task: str) -> None:
        """DC3: the identity reads the provider method, the trials read the
        adapter. With no user params the adapter's native params are the
        provider defaults plus ``seed`` / ``verbosity`` (``config.training``
        and a constant) and ``num_class`` (the data); ``n_estimators`` becomes
        ``num_boost_round``. A default added to the adapter alone fails here.
        """
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        adapter = LGBMAdapter(
            task=task,  # type: ignore[arg-type]
            params={},
            num_class=3 if task == "multiclass" else None,
        )
        params, num_boost_round, _, _ = adapter._build_params()
        defaults = LGBMProvider().default_params(task)  # type: ignore[arg-type]
        native = {
            k: v
            for k, v in params.items()
            if k not in {"seed", "verbosity", "num_class"}
        }
        assert native == {k: v for k, v in defaults.items() if k != "n_estimators"}
        assert num_boost_round == defaults["n_estimators"]


# ---------------------------------------------------------------------------
# Identity and versions
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_build_identity_combines_both_sides(self) -> None:
        data = summarise_frame(_frame())
        parts = _parts(_cfg())
        identity = build_identity(data, parts)
        assert isinstance(identity, ObjectiveIdentity)
        assert dict(identity.elements) == {**data, **parts}
        assert build_identity(data, parts).digest == identity.digest

    def test_overlapping_element_names_are_refused(self) -> None:
        with pytest.raises(ValueError):
            build_identity({"metric": "a" * 64}, {"metric": "b" * 64})

    def test_mismatch_names_each_kind_of_element(self) -> None:
        # Criterion 27 (unit half): data / config part / metric.
        base = build_identity(summarise_frame(_frame()), _parts(_cfg()))
        data_changed = _frame()
        data_changed.loc[0, "flt"] = 9.0
        cases = {
            "data.values": build_identity(
                summarise_frame(data_changed), _parts(_cfg())
            ),
            "config.model": build_identity(
                summarise_frame(_frame()),
                _parts(_cfg(_set("model.num_leaves_ratio", 0.5))),
            ),
            "metric": build_identity(
                summarise_frame(_frame()),
                _parts(_cfg(_set("evaluation.metrics", ["mae"]))),
            ),
        }
        for element, current in cases.items():
            assert mismatched_elements(base.elements, current) == (element,)

    def test_one_sided_elements_are_mismatched(self) -> None:
        assert mismatched_elements({"a": "1", "b": "2"}, {"a": "1", "c": "3"}) == (
            "b",
            "c",
        )

    def test_mismatched_elements_accepts_an_identity(self) -> None:
        identity = build_identity({"data.values": "x"}, {})
        assert mismatched_elements({"data.values": "x"}, identity) == ()


class TestVersions:
    def test_library_versions_names_the_five_packages(self) -> None:
        versions = library_versions(LGBMProvider())
        assert {"lizyml", "lightgbm", "optuna", "pandas", "scikit-learn"} <= set(
            versions
        )
        assert all(isinstance(v, str) and v for v in versions.values())
        assert versions["pandas"] == pd.__version__

    def test_no_difference(self) -> None:
        v = {"lizyml": "1", "lightgbm": "4"}
        assert version_differences(v, copy.deepcopy(v)) == ()

    def test_differences_are_listed_old_to_new(self) -> None:
        diffs = version_differences(
            {"lightgbm": "4.5.0", "optuna": "4.7.0", "gone": "1"},
            {"lightgbm": "4.6.0", "optuna": "4.7.0", "new": "2"},
        )
        assert diffs == (
            VersionDifference("gone", "1", None),
            VersionDifference("lightgbm", "4.5.0", "4.6.0"),
            VersionDifference("new", None, "2"),
        )
