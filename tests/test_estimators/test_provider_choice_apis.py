"""H-0079 Phase 2 — EstimatorProvider.objective_choices / metric_choices.

Functional contract tests for the two Provider Protocol additions:

- ``objective_choices(task) -> tuple[str, ...]`` returns canonical
  LightGBM objective names valid for *task*, no aliases, deterministic
  order, no duplicates.
- ``metric_choices(task) -> dict[Literal["native", "feval"],
  tuple[str, ...]]`` splits per-task valid metrics by source. Both
  tuples list canonical names only; the union is duplicate-free.

These contracts are what downstream consumers (LizyStudio Issue #461,
``default_space``, future config validators) rely on. Drift here would
re-open the Phase-1 silent-strip class of bug at the API surface.
"""

from __future__ import annotations

import pytest

from lizyml.core.types.task import TaskType
from lizyml.estimators.lgbm.provider import LGBMProvider

_TASKS: tuple[TaskType, ...] = ("regression", "binary", "multiclass")


@pytest.fixture
def provider() -> LGBMProvider:
    return LGBMProvider()


# ---------------------------------------------------------------------------
# objective_choices contract
# ---------------------------------------------------------------------------


class TestObjectiveChoicesSignature:
    """objective_choices returns a tuple[str, ...] for every supported task."""

    @pytest.mark.parametrize("task", _TASKS)
    def test_returns_tuple(self, provider: LGBMProvider, task: TaskType) -> None:
        result = provider.objective_choices(task)
        assert isinstance(result, tuple)
        assert all(isinstance(name, str) for name in result)

    @pytest.mark.parametrize("task", _TASKS)
    def test_non_empty(self, provider: LGBMProvider, task: TaskType) -> None:
        """Every supported LightGBM task must expose at least 1 objective."""
        assert len(provider.objective_choices(task)) >= 1

    @pytest.mark.parametrize("task", _TASKS)
    def test_no_duplicates(self, provider: LGBMProvider, task: TaskType) -> None:
        choices = provider.objective_choices(task)
        assert len(set(choices)) == len(choices), (
            f"Duplicate objective for task='{task}': {choices}"
        )

    @pytest.mark.parametrize("task", _TASKS)
    def test_deterministic(self, provider: LGBMProvider, task: TaskType) -> None:
        """Two calls return tuples with identical order — UI relies on this."""
        a = provider.objective_choices(task)
        b = provider.objective_choices(task)
        assert a == b


class TestObjectiveChoicesContent:
    """The returned values are the canonical LightGBM objectives per task."""

    def test_regression_canonical_set(self, provider: LGBMProvider) -> None:
        """LightGBM 4.x ships 9 canonical regression objectives."""
        result = set(provider.objective_choices("regression"))
        expected = {
            "regression",
            "regression_l1",
            "huber",
            "fair",
            "poisson",
            "quantile",
            "mape",
            "gamma",
            "tweedie",
        }
        assert result == expected

    def test_binary_canonical_set(self, provider: LGBMProvider) -> None:
        """LightGBM 4.x ships 3 canonical binary objectives."""
        result = set(provider.objective_choices("binary"))
        assert result == {"binary", "cross_entropy", "cross_entropy_lambda"}

    def test_multiclass_canonical_set(self, provider: LGBMProvider) -> None:
        """LightGBM 4.x ships 2 canonical multiclass objectives.

        ``softmax`` is an alias of ``multiclass`` and is not surfaced.
        """
        result = set(provider.objective_choices("multiclass"))
        assert result == {"multiclass", "multiclassova"}

    def test_no_aliases_returned(self, provider: LGBMProvider) -> None:
        """Surface API must not include LightGBM aliases.

        e.g. ``softmax`` ≡ ``multiclass``, ``mean_squared_error`` ≡
        ``regression``. UIs need a single canonical name per concept.
        """
        forbidden_aliases = {
            "softmax",
            "ova",
            "ovr",
            "multiclass_ova",
            "mean_squared_error",
            "l1",
            "l2",
            "mae",
            "mse",
            "regression_l2",
        }
        for task in _TASKS:
            choices = set(provider.objective_choices(task))
            collision = choices & forbidden_aliases
            assert not collision, (
                f"Task '{task}' surfaces aliases instead of canonical names: "
                f"{sorted(collision)}"
            )

    def test_aligned_with_task_compatible_objectives(
        self, provider: LGBMProvider
    ) -> None:
        """Surface tuple must equal the Phase-1 whitelist set.

        The two sources of truth (``TASK_COMPATIBLE_OBJECTIVES`` used by
        ``_build_params`` and ``objective_choices`` exposed to downstream
        UIs) MUST agree. Drift means UI shows an option that fails at fit
        time, or fit accepts a value not exposed to the UI.
        """
        from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES

        for task in _TASKS:
            surface = set(provider.objective_choices(task))
            whitelist = set(TASK_COMPATIBLE_OBJECTIVES[task])
            assert surface == whitelist, (
                f"Drift detected for task='{task}': "
                f"objective_choices={sorted(surface)} vs "
                f"TASK_COMPATIBLE_OBJECTIVES={sorted(whitelist)}"
            )


# ---------------------------------------------------------------------------
# metric_choices contract
# ---------------------------------------------------------------------------


class TestMetricChoicesSignature:
    """metric_choices returns the documented dict shape."""

    @pytest.mark.parametrize("task", _TASKS)
    def test_returns_dict_with_two_keys(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        result = provider.metric_choices(task)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"native", "feval"}

    @pytest.mark.parametrize("task", _TASKS)
    def test_values_are_tuples_of_str(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        result = provider.metric_choices(task)
        for key in ("native", "feval"):
            assert isinstance(result[key], tuple)
            assert all(isinstance(name, str) for name in result[key])

    @pytest.mark.parametrize("task", _TASKS)
    def test_non_empty_native_branch(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        """Every supported task must list at least one native metric."""
        assert len(provider.metric_choices(task)["native"]) >= 1

    @pytest.mark.parametrize("task", _TASKS)
    def test_deterministic_order(self, provider: LGBMProvider, task: TaskType) -> None:
        a = provider.metric_choices(task)
        b = provider.metric_choices(task)
        assert a["native"] == b["native"]
        assert a["feval"] == b["feval"]


class TestMetricChoicesContent:
    """Canonical metrics, no aliases, no duplicates across native/feval."""

    @pytest.mark.parametrize("task", _TASKS)
    def test_no_duplicates_within_each_branch(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        result = provider.metric_choices(task)
        for branch in ("native", "feval"):
            names = result[branch]
            assert len(set(names)) == len(names), (
                f"Duplicate metric in {branch} for task='{task}': {names}"
            )

    @pytest.mark.parametrize("task", _TASKS)
    def test_no_duplicates_across_branches(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        """A name must not appear as both native and feval (split is exclusive)."""
        result = provider.metric_choices(task)
        overlap = set(result["native"]) & set(result["feval"])
        assert not overlap, (
            f"Task '{task}': metric appears as both native and feval: {sorted(overlap)}"
        )

    @pytest.mark.parametrize("task", _TASKS)
    def test_no_aliases_in_native_branch(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        """Aliases such as l1/l2/mse/ova must not be surfaced."""
        forbidden_aliases = {
            "l1",
            "l2",
            "mse",
            "mean_squared_error",
            "mean_absolute_error",
            "regression_l1",
            "regression_l2",
            "root_mean_squared_error",
            "l2_root",
            "mean_absolute_percentage_error",
            "binary",  # alias of binary_logloss
            "xentropy",
            "xentlambda",
            "kldiv",
            "softmax",
            "multiclass",
            "ova",
            "ovr",
            "multiclass_ova",
        }
        native = set(provider.metric_choices(task)["native"])
        collision = native & forbidden_aliases
        assert not collision, (
            f"Task '{task}' surfaces alias metrics: {sorted(collision)}"
        )

    def test_regression_native_includes_core_metrics(
        self, provider: LGBMProvider
    ) -> None:
        """Sanity: rmse/mae/mape are exposed for regression."""
        native = set(provider.metric_choices("regression")["native"])
        assert {"rmse", "mae", "mape"} <= native

    def test_regression_feval_includes_lizyml_only(
        self, provider: LGBMProvider
    ) -> None:
        """Sanity: rmsle/r2/smape/wape are LizyML-only feval implementations."""
        feval = set(provider.metric_choices("regression")["feval"])
        assert {"rmsle", "r2", "smape", "wape"} <= feval

    def test_native_subset_of_validation_whitelist(
        self, provider: LGBMProvider
    ) -> None:
        """Every name in native[] must be accepted by metric_bridge.

        Otherwise UI surfaces an option that crashes at fit time when
        flowed through ``params['metric']``.
        """
        from lizyml.estimators.lgbm.metric_bridge import _LGBM_NATIVE_METRICS

        for task in _TASKS:
            native = provider.metric_choices(task)["native"]
            valid = _LGBM_NATIVE_METRICS[task]
            unsupported = set(native) - set(valid)
            assert not unsupported, (
                f"Task '{task}': metric_choices native names not in "
                f"_LGBM_NATIVE_METRICS whitelist: {sorted(unsupported)}"
            )

    def test_feval_subset_of_feval_metrics(self, provider: LGBMProvider) -> None:
        """Every name in feval[] must be a registered LizyML feval metric."""
        from lizyml.estimators.lgbm.metric_bridge import _FEVAL_METRICS

        for task in _TASKS:
            feval = provider.metric_choices(task)["feval"]
            valid = _FEVAL_METRICS[task]
            unsupported = set(feval) - set(valid)
            assert not unsupported, (
                f"Task '{task}': metric_choices feval names not in "
                f"_FEVAL_METRICS registry: {sorted(unsupported)}"
            )


# ---------------------------------------------------------------------------
# default_space integration
# ---------------------------------------------------------------------------


class TestDefaultSpaceProviderIntegration:
    """default_space honours the provider when supplied."""

    def test_default_space_uses_provider_objective_choices(
        self, provider: LGBMProvider
    ) -> None:
        """default_space(task, provider) must populate the objective dim
        from ``provider.objective_choices(task)`` so that UIs and tune
        agree on the same canonical list."""
        from lizyml.core.types.search_dim import CategoricalDim
        from lizyml.estimators.lgbm.defaults import default_space

        for task in _TASKS:
            dims = default_space(task, provider=provider)
            obj_dim = next(
                d
                for d in dims
                if isinstance(d, CategoricalDim) and d.name == "objective"
            )
            assert tuple(obj_dim.choices) == provider.objective_choices(task), (
                f"default_space({task!r}, provider) does not use "
                f"provider.objective_choices(task)"
            )

    def test_default_space_without_provider_still_works(self) -> None:
        """default_space(task) without provider preserves backward compat."""
        from lizyml.estimators.lgbm.defaults import default_space

        # Call site identical to pre-Phase-2 — must not raise.
        for task in _TASKS:
            dims = default_space(task)
            # Smoke: at least 10 dims (model + smart + training)
            assert len(dims) >= 10
