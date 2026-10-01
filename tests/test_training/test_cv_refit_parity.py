"""The final refit trains under the same inputs as the CV folds (H-0103, #269).

``RefitTrainer.fit`` used to accept three of ``CVTrainer.fit``'s seven inputs.
The one with an effect was ``sample_weight``: for a multiclass ``balanced``
fit, every CV fold trained with balanced class weights and the final refit --
the model ``predict`` and ``export`` use -- trained unweighted. The other three
inputs are absent by written policy (H-0103 decisions 2-4), and this file pins
both: the weights by **value** against an expectation computed here, and the
policy by reading both signatures, so a new input on either side fails until it
is forwarded or written down.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_sample_weight

from lizyml.config.loader import load_config
from lizyml.core._model_factories import build_inner_valid
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.training.cv_trainer import CVTrainer
from lizyml.training.refit_trainer import RefitTrainer
from tests._helpers import make_binary_df, make_config

N_SPLITS = 3

#: Inputs ``CVTrainer.fit`` takes and ``RefitTrainer.fit`` deliberately does not,
#: each with the decision that says why. Read against both signatures below.
REFIT_ABSENT_BY_POLICY: dict[str, str] = {
    "time_values": "H-0103 decision 2",
    "data_fingerprint": "H-0103 decision 3",
    "run_meta": "H-0103 decision 4",
}


def _imbalanced_multiclass_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Three classes at roughly 70 / 20 / 10 percent, so balanced weights differ."""
    rng = np.random.default_rng(seed)
    target = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1])
    return pd.DataFrame(
        {
            "feat_a": target + rng.normal(scale=0.8, size=n),
            "feat_b": rng.normal(size=n),
            "target": target,
        }
    )


class _Recorded:
    """One ``lgb.Dataset`` construction: is it an eval set, and its weights."""

    def __init__(self, n_rows: int, is_valid: bool, weight: Any) -> None:
        self.n_rows = n_rows
        self.is_valid = is_valid
        self.weight = None if weight is None else np.asarray(weight, dtype=float)


@contextmanager
def _record_datasets() -> Iterator[list[_Recorded]]:
    """Record every ``lgb.Dataset`` built, in construction order."""
    seen: list[_Recorded] = []
    real_init = lgb.Dataset.__init__

    def spy(
        self: lgb.Dataset,
        data: Any,
        label: Any = None,
        reference: Any = None,
        weight: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        seen.append(_Recorded(len(data), reference is not None, weight))
        real_init(self, data, label, reference, weight, *args, **kwargs)

    lgb.Dataset.__init__ = spy  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        lgb.Dataset.__init__ = real_init  # type: ignore[method-assign]


def _training_sets(seen: list[_Recorded]) -> list[_Recorded]:
    return [r for r in seen if not r.is_valid]


#: Sentinel: leave ``model.balanced`` out of the config, so the default applies.
_OMITTED = "omitted"


def _config(
    task: str, *, balanced: bool | None | str, early_stopping: bool
) -> dict[str, Any]:
    cfg = make_config(task, n_estimators=20, n_splits=N_SPLITS)
    if balanced != _OMITTED:
        cfg["model"]["balanced"] = balanced
    cfg["training"]["early_stopping"] = {"enabled": early_stopping, "rounds": 5}
    return cfg


@pytest.mark.parametrize(
    "early_stopping", [False, True], ids=["no_inner_valid", "inner_valid"]
)
@pytest.mark.parametrize(
    "balanced",
    # ``null`` and an omitted key resolve to True for multiclass (BLUEPRINT 5.3):
    # the default configuration is the one most users run, so an implementation
    # that forwards weights only for an explicit ``true`` must fail here.
    [True, None, _OMITTED],
    ids=["true", "null", "omitted"],
)
def test_refit_trains_with_the_cv_weighting(
    balanced: bool | None | str, early_stopping: bool
) -> None:
    """Every training set -- each CV fold and the refit -- carries the balanced
    weights of exactly the rows it trains on; no eval set carries weights.

    The expectation is computed here from the target alone, not read from the
    code under test: ``compute_sample_weight("balanced", y)`` over all rows, then
    restricted to the fold's outer-train rows and, with early stopping, to the
    inner-train rows of that split.
    """
    df = _imbalanced_multiclass_df()
    raw = _config("multiclass", balanced=balanced, early_stopping=early_stopping)
    y = df["target"].to_numpy()
    full = compute_sample_weight("balanced", y)

    with _record_datasets() as seen:
        model = Model(raw)
        fit_result = model.fit(data=df)

    train_sets = _training_sets(seen)
    assert len(train_sets) == N_SPLITS + 1, (
        f"expected {N_SPLITS} CV folds and 1 refit, recorded {len(train_sets)}"
    )
    assert all(r.weight is None for r in seen if r.is_valid), (
        "an eval set carried weights; inner-valid rows must stay unweighted"
    )

    # CV folds: the reference side of the rule.
    for k, (train_idx, _valid_idx) in enumerate(fit_result.splits.outer):
        want = full[train_idx]
        if early_stopping:
            assert fit_result.splits.inner is not None
            want = want[fit_result.splits.inner[k][0]]
        got = train_sets[k].weight
        assert got is not None, f"CV fold {k} trained unweighted"
        np.testing.assert_allclose(got, want, err_msg=f"CV fold {k}")

    # Refit: the same rule over all rows.
    want_refit: npt.NDArray[np.float64] = full
    if early_stopping:
        split = build_inner_valid(load_config(raw)).split(len(y), y=y, groups=None)
        assert split is not None
        want_refit = full[split[0]]
    got_refit = train_sets[-1].weight
    assert got_refit is not None, (
        "the final refit trained unweighted while every CV fold was weighted "
        "(#269): the model predict and export use is not the one evaluated"
    )
    np.testing.assert_allclose(got_refit, want_refit, err_msg="refit")
    # Non-vacuity: the weights are not uniform, so a constant vector would fail.
    assert np.unique(np.round(want_refit, 9)).size == 3


@pytest.mark.parametrize(
    ("task", "balanced"),
    [
        pytest.param("regression", True, id="regression-balanced"),
        pytest.param("regression", False, id="regression-unbalanced"),
        pytest.param("binary", True, id="binary-balanced"),
        pytest.param("binary", False, id="binary-unbalanced"),
        pytest.param("multiclass", False, id="multiclass-unbalanced"),
    ],
)
def test_no_weight_vector_where_balanced_makes_none(task: str, balanced: bool) -> None:
    """Only multiclass ``balanced`` produces a weight vector (BLUEPRINT 5.3).

    Regression refuses ``balanced`` before training; binary turns it into
    ``scale_pos_weight``, which must reach the CV folds and the refit alike.
    """
    if task == "multiclass":
        df = _imbalanced_multiclass_df()
    elif task == "binary":
        df = make_binary_df(n=300)
    else:
        df = _imbalanced_multiclass_df().assign(target=lambda d: d["feat_a"] * 2.0)
    raw = _config(task, balanced=balanced, early_stopping=True)

    params_seen: list[dict[str, Any]] = []
    real_train = lgb.train

    def spy_train(params: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        params_seen.append(dict(params))
        return real_train(params, *args, **kwargs)

    lgb.train = spy_train  # type: ignore[assignment]
    try:
        with _record_datasets() as seen:
            if task == "regression" and balanced:
                with pytest.raises(LizyMLError) as exc:
                    Model(raw).fit(data=df)
                assert exc.value.code is ErrorCode.UNSUPPORTED_TASK
                assert not seen, "regression balanced must refuse before training"
                return
            Model(raw).fit(data=df)
    finally:
        lgb.train = real_train  # type: ignore[assignment]

    assert len(_training_sets(seen)) == N_SPLITS + 1
    assert all(r.weight is None for r in seen), "a weight vector appeared"
    assert len(params_seen) == N_SPLITS + 1
    spw = {p.get("scale_pos_weight") for p in params_seen}
    if task == "binary" and balanced:
        assert len(spw) == 1 and None not in spw, (
            f"scale_pos_weight must reach every CV fold and the refit alike: {spw}"
        )
    else:
        assert spw == {None}


def test_refit_receives_time_ordered_rows() -> None:
    """H-0103 decision 2: the refit needs no ``time_values`` because rows are
    already in time order when it receives them, even if the caller's are not.
    """
    n = 120
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="D"),
            "feat_a": rng.normal(size=n),
        }
    )
    # The target is the time rank, so time order is visible in y alone.
    df["target"] = np.arange(n, dtype=float)
    df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert not df["target"].is_monotonic_increasing  # the input really is shuffled

    raw = make_config(
        "regression",
        n_estimators=10,
        n_splits=3,
        split_method="time_series",
        time_col="t",
    )
    received: list[pd.Series] = []
    real_fit = RefitTrainer.fit

    def spy_fit(
        self: RefitTrainer, X: pd.DataFrame, y: pd.Series, *a: Any, **k: Any
    ) -> Any:
        received.append(y.copy())
        return real_fit(self, X, y, *a, **k)

    RefitTrainer.fit = spy_fit  # type: ignore[method-assign]
    try:
        Model(raw).fit(data=df)
    finally:
        RefitTrainer.fit = real_fit  # type: ignore[method-assign]

    assert len(received) == 1
    np.testing.assert_array_equal(received[0].to_numpy(), np.arange(n, dtype=float))


def _fit_inputs(cls: type) -> set[str]:
    params = inspect.signature(cls.fit).parameters
    return {name for name in params if name != "self"}


def test_trainer_inputs_differ_only_by_written_policy() -> None:
    """Every input one trainer accepts is accepted by the other or is written
    policy; and the policy names are really absent where they claim to be.
    """
    cv = _fit_inputs(CVTrainer)
    refit = _fit_inputs(RefitTrainer)

    assert refit - cv == set(), f"refit takes inputs CV does not: {refit - cv}"
    unexplained = (cv - refit) - set(REFIT_ABSENT_BY_POLICY)
    assert not unexplained, (
        f"CVTrainer.fit takes {sorted(unexplained)} and RefitTrainer.fit does not, "
        "with no written policy: forward it, or record why not (H-0103)"
    )
    for name, decision in REFIT_ABSENT_BY_POLICY.items():
        assert name in cv and name not in refit, (
            f"{name!r} is registered as absent by policy ({decision}) but "
            f"CVTrainer.fit has it: {name in cv}, RefitTrainer.fit has it: "
            f"{name in refit}; the registration is stale"
        )
