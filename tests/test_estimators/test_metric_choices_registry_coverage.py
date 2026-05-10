"""H-0079 Phase 3 — MetricRegistry coverage drift guard (Layer L4).

Layer L4 of the 7-layer regression prevention. Four different files
hold "valid metric per task" lists:

- ``lizyml.metrics.registry._TASK_METRICS``     — user-facing names that
  ``evaluator`` and ``Model.evaluate()`` accept and return.
- ``lizyml.estimators.lgbm.metric_bridge._LGBM_NATIVE_METRICS``
                                                 — names accepted by
  ``params["metric"]`` for fit-time evaluation (LightGBM 4.x).
- ``lizyml.estimators.lgbm.metric_bridge._FEVAL_METRICS``
                                                 — LizyML callables
  wired as fit-time feval functions.
- ``LGBMProvider.metric_choices(task)``         — canonical surface
  exposed to downstream UIs (introduced in Phase 2).

The contract enforced here is: **for any metric reachable as a
fit-time signal (``_LGBM_NATIVE_METRICS`` ∪ ``_FEVAL_METRICS``), the
provider's surface must list it.** Metrics reachable only via
``Model.evaluate()`` post-fit (e.g. ``auc_pr`` for multiclass — sklearn
``average_precision_score`` is computed Python-side; LightGBM has no
multiclass average_precision metric) are intentionally not surfaced
because they cannot drive early stopping / eval log.
"""

from __future__ import annotations

import pytest

from lizyml.core.types.task import TaskType
from lizyml.estimators.lgbm.metric_bridge import (
    _FEVAL_METRICS,
    _LGBM_NATIVE_METRICS,
    translate_metric,
)
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.metrics.registry import _TASK_METRICS

_TASKS: tuple[TaskType, ...] = ("regression", "binary", "multiclass")


@pytest.fixture
def provider() -> LGBMProvider:
    return LGBMProvider()


class TestMetricRegistryFitTimeCoverage:
    """Every fit-time-reachable metric must be in the provider surface.

    "Fit-time-reachable" means the metric can drive early stopping /
    eval log via ``params["metric"]`` (native) or ``feval=[...]``
    (LizyML callable). Metrics that are only computable post-fit by
    ``Model.evaluate()`` (sklearn-Python implementation) are excluded
    from the contract.
    """

    @pytest.mark.parametrize("task", _TASKS)
    def test_fit_time_registered_metrics_surfaced(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        registered = _TASK_METRICS[task]
        choices = provider.metric_choices(task)
        surface_union = set(choices["native"]) | set(choices["feval"])

        native_whitelist = _LGBM_NATIVE_METRICS[task]
        feval_whitelist = _FEVAL_METRICS[task]

        for name in registered:
            translated = translate_metric(name, task)
            fit_time_reachable = (
                translated in native_whitelist or name in feval_whitelist
            )
            if not fit_time_reachable:
                # Post-fit-only metric (e.g. auc_pr for multiclass) —
                # cannot drive early stopping; surfacing is not required.
                continue
            in_surface = translated in surface_union or name in surface_union
            assert in_surface, (
                f"Task '{task}': metric '{name}' (translated='{translated}') "
                f"is reachable at fit-time (native={translated in native_whitelist}, "
                f"feval={name in feval_whitelist}) but is missing from "
                f"provider.metric_choices(). Add it to "
                f"_LGBM_NATIVE_METRIC_CHOICES or _LGBM_FEVAL_METRIC_CHOICES."
            )


class TestMetricRegistryTranslation:
    """Every registered name has a non-empty translation."""

    @pytest.mark.parametrize("task", _TASKS)
    def test_no_registered_metrics_lost_in_translation(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        for name in _TASK_METRICS[task]:
            translated = translate_metric(name, task)
            assert translated, (
                f"Task '{task}': metric '{name}' translates to empty "
                f"value — translate_metric() lookup is broken."
            )


class TestProviderSurfaceReverseSubset:
    """Every provider-native metric is accepted by lgb.train (and every
    feval is registered in MetricRegistry).

    Reverse direction of L3 smoke-fits: catches the case where the
    provider surfaces a name that the underlying whitelists don't
    accept (UI promises something the library cannot deliver).
    """

    @pytest.mark.parametrize("task", _TASKS)
    def test_native_subset_of_lgbm_whitelist(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        native = provider.metric_choices(task)["native"]
        whitelist = _LGBM_NATIVE_METRICS[task]
        unsupported = set(native) - set(whitelist)
        assert not unsupported, (
            f"Task '{task}': provider native metrics {sorted(unsupported)} "
            f"are not in _LGBM_NATIVE_METRICS — lgb.train will reject them."
        )

    @pytest.mark.parametrize("task", _TASKS)
    def test_feval_subset_of_feval_metrics(
        self, provider: LGBMProvider, task: TaskType
    ) -> None:
        feval = provider.metric_choices(task)["feval"]
        whitelist = _FEVAL_METRICS[task]
        unsupported = set(feval) - set(whitelist)
        assert not unsupported, (
            f"Task '{task}': provider feval metrics {sorted(unsupported)} "
            f"are not in _FEVAL_METRICS — resolve_metrics will not wire them."
        )
