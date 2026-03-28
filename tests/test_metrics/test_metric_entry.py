"""Tests for MetricEntry parsing and parameterised metric instantiation (H-0065)."""

from __future__ import annotations

import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.metrics.registry import (
    MetricEntry,
    get_metric,
    get_metrics_for_task,
    parse_metric_entries,
    parse_metric_entry,
)


class TestParseMetricEntry:
    """parse_metric_entry() normalises a MetricEntry to (name, kwargs)."""

    def test_str_entry(self) -> None:
        name, kwargs = parse_metric_entry("rmse")
        assert name == "rmse"
        assert kwargs == {}

    def test_dict_entry(self) -> None:
        entry: MetricEntry = {"precision_at_k": {"k": 20}}
        name, kwargs = parse_metric_entry(entry)
        assert name == "precision_at_k"
        assert kwargs == {"k": 20}

    def test_dict_entry_empty_params(self) -> None:
        entry: MetricEntry = {"auc": {}}
        name, kwargs = parse_metric_entry(entry)
        assert name == "auc"
        assert kwargs == {}

    def test_dict_entry_multiple_keys_rejected(self) -> None:
        entry = {"auc": {}, "rmse": {}}  # type: ignore[dict-item]
        with pytest.raises(LizyMLError, match="exactly one key"):
            parse_metric_entry(entry)  # type: ignore[arg-type]

    def test_dict_entry_non_dict_value_rejected(self) -> None:
        entry = {"precision_at_k": 20}  # type: ignore[dict-item]
        with pytest.raises(LizyMLError, match="must be a dict"):
            parse_metric_entry(entry)  # type: ignore[arg-type]


class TestParseMetricEntries:
    """parse_metric_entries() handles a list of mixed entries."""

    def test_mixed_list(self) -> None:
        entries: list[MetricEntry] = ["auc", {"precision_at_k": {"k": 5}}]
        result = parse_metric_entries(entries)
        assert result == [("auc", {}), ("precision_at_k", {"k": 5})]

    def test_all_str(self) -> None:
        result = parse_metric_entries(["rmse", "mae"])
        assert result == [("rmse", {}), ("mae", {})]

    def test_empty_list(self) -> None:
        result = parse_metric_entries([])
        assert result == []


class TestGetMetricWithKwargs:
    """get_metric() supports **kwargs for parameterised metrics."""

    def test_default_k(self) -> None:
        metric = get_metric("precision_at_k")
        assert metric.k == 10  # type: ignore[attr-defined]

    def test_custom_k(self) -> None:
        metric = get_metric("precision_at_k", k=20)
        assert metric.k == 20  # type: ignore[attr-defined]

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(LizyMLError, match="Invalid parameters"):
            get_metric("precision_at_k", k=0)

    def test_no_kwargs_metric(self) -> None:
        """Metrics without __init__ params still work with no kwargs."""
        metric = get_metric("rmse")
        assert metric.name == "rmse"

    def test_unknown_kwarg_raises(self) -> None:
        """Passing unknown kwargs to a metric that doesn't accept them."""
        with pytest.raises(LizyMLError, match="Invalid parameters"):
            get_metric("rmse", unknown_param=42)

    def test_non_str_non_dict_entry_raises(self) -> None:
        """parse_metric_entry rejects non-str/non-dict types."""
        with pytest.raises(LizyMLError, match="must be a str or dict"):
            parse_metric_entry(42)  # type: ignore[arg-type]


class TestGetMetricsForTaskWithEntries:
    """get_metrics_for_task() handles MetricEntry list."""

    def test_str_entries_backward_compat(self) -> None:
        metrics = get_metrics_for_task(["auc", "logloss"], "binary")
        assert len(metrics) == 2
        assert metrics[0].name == "auc"

    def test_dict_entry_with_kwargs(self) -> None:
        entries: list[MetricEntry] = [{"precision_at_k": {"k": 25}}]
        metrics = get_metrics_for_task(entries, "binary")
        assert len(metrics) == 1
        assert metrics[0].name == "precision_at_k"
        assert metrics[0].k == 25  # type: ignore[attr-defined]

    def test_mixed_entries(self) -> None:
        entries: list[MetricEntry] = ["auc", {"precision_at_k": {"k": 5}}]
        metrics = get_metrics_for_task(entries, "binary")
        assert len(metrics) == 2
        assert metrics[1].k == 5  # type: ignore[attr-defined]

    def test_task_incompatible_dict_entry(self) -> None:
        entries: list[MetricEntry] = [{"precision_at_k": {"k": 10}}]
        with pytest.raises(LizyMLError):
            get_metrics_for_task(entries, "regression")
