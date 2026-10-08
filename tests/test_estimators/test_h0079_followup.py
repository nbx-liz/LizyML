"""H-0079 follow-up — coverage gaps + edge case guards.

Tests added after the post-merge review of PR #160 / #161 / #162:

- G1: Codegen export with non-default objective writes the user value
  into ``config.json:lgbm_params.objective`` and the generated
  ``train.py`` reproduces the booster.
- G2: ``Model.save`` / ``Model.load`` round-trip preserves the
  user-supplied objective.
- G3: ``_check_objective_compatible`` rejects non-string and
  empty-string inputs cleanly with ``CONFIG_INVALID``.
- G4: Calibration on top of a non-default binary objective
  (``cross_entropy`` instead of the default ``binary``) produces a
  fitted calibrator and well-formed metrics.

The original H-0079 ticket only covered the strip-removal path; these
tests pin the integration boundaries that the strip used to mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lizyml.config.schema import LizyMLConfig
from lizyml.core.exceptions import LizyMLError
from lizyml.core.model import Model
from tests._helpers import make_binary_df, make_config


def _make_positive_regression_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Strictly positive regression target (works for all 9 objectives)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(0, 5, n),
        }
    )
    df["target"] = df["feat_a"] * 0.5 + df["feat_b"] + rng.uniform(0.1, 1.0, n)
    return df


# ---------------------------------------------------------------------------
# G1 — Codegen export with non-default objective
# ---------------------------------------------------------------------------


class TestG1CodegenWithNonDefaultObjective:
    """``export_code()`` must emit the user's objective into config.json
    instead of the (formerly silently-overridden) ``_TASK_OBJECTIVE``
    default. Pre-H-0079 this round-trip emitted the silent default;
    post-H-0079 the user's value flows through and must be written
    out faithfully.
    """

    def test_regression_fair_objective_in_config_json(self, tmp_path: Path) -> None:
        df = _make_positive_regression_df(n=120, seed=0)
        cfg_dict = make_config(
            "regression", n_estimators=20, n_splits=2, objective="fair"
        )
        m = Model(LizyMLConfig(**cfg_dict))
        m.fit(data=df)

        out = tmp_path / "codegen_out"
        m.export_code(out)

        with (out / "config.json").open() as f:
            cfg = json.load(f)
        assert cfg["lgbm_params"]["objective"] == "fair", (
            f"Exported config.json reports "
            f"objective='{cfg['lgbm_params']['objective']}' but the "
            f"user requested 'fair'. _build_params() round-trip "
            f"through build_export_params() may have re-applied the "
            f"silent strip."
        )

    def test_binary_cross_entropy_objective_in_config_json(
        self, tmp_path: Path
    ) -> None:
        df = make_binary_df(n=120, seed=0)
        cfg_dict = make_config(
            "binary",
            n_estimators=20,
            n_splits=2,
            split_method="stratified_kfold",
            objective="cross_entropy",
        )
        m = Model(LizyMLConfig(**cfg_dict))
        m.fit(data=df)

        out = tmp_path / "codegen_out"
        m.export_code(out)

        with (out / "config.json").open() as f:
            cfg = json.load(f)
        assert cfg["lgbm_params"]["objective"] == "cross_entropy"

    def test_exported_objective_matches_in_memory_booster(self, tmp_path: Path) -> None:
        """Internal-consistency contract: the exported config.json's
        ``objective`` must equal the in-memory booster's
        ``params["objective"]``. Pre-H-0079 these were equal too (both
        wrong); post-H-0079 they must be equal AND match the user."""
        df = _make_positive_regression_df(n=120, seed=0)
        cfg_dict = make_config(
            "regression", n_estimators=20, n_splits=2, objective="poisson"
        )
        m = Model(LizyMLConfig(**cfg_dict))
        m.fit(data=df)

        out = tmp_path / "codegen_out"
        m.export_code(out)

        with (out / "config.json").open() as f:
            cfg = json.load(f)
        booster_objective = m._refit_result.model.get_native_model().params[  # type: ignore[union-attr]
            "objective"
        ]
        assert cfg["lgbm_params"]["objective"] == booster_objective == "poisson"


# ---------------------------------------------------------------------------
# G2 — Save / Load round-trip with non-default objective
# ---------------------------------------------------------------------------


class TestG2PersistenceWithNonDefaultObjective:
    """``Model.save`` then ``Model.load`` must preserve the user's
    objective so that predict on the loaded model matches predict on
    the original.
    """

    def test_save_load_predict_equivalence_regression_fair(
        self, tmp_path: Path
    ) -> None:
        df = _make_positive_regression_df(n=120, seed=0)
        cfg_dict = make_config(
            "regression", n_estimators=20, n_splits=2, objective="fair"
        )
        m = Model(LizyMLConfig(**cfg_dict))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:30].reset_index(drop=True)
        original_pred = m.predict(X_new).pred

        out = tmp_path / "model_artifact"
        m.export(out)

        loaded = Model.load(out)
        loaded_pred = loaded.predict(X_new).pred

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_save_load_preserves_booster_objective(self, tmp_path: Path) -> None:
        df = _make_positive_regression_df(n=120, seed=0)
        cfg_dict = make_config(
            "regression", n_estimators=20, n_splits=2, objective="quantile"
        )
        m = Model(LizyMLConfig(**cfg_dict))
        m.fit(data=df)

        out = tmp_path / "model_artifact"
        m.export(out)

        loaded = Model.load(out)
        loaded_objective = loaded._refit_result.model.get_native_model().params[  # type: ignore[union-attr]
            "objective"
        ]
        assert loaded_objective == "quantile"


# ---------------------------------------------------------------------------
# G3 — ``_check_objective_compatible`` edge-case inputs
# ---------------------------------------------------------------------------


class TestG3ObjectiveCompatibilityEdgeInputs:
    """Edge inputs to the user-objective handling path must produce
    clear ``LizyMLError(CONFIG_INVALID)`` instead of cryptic
    ``TypeError`` / ``KeyError``.

    The refused edge inputs are ordinary values (empty string, dict, list,
    set, int) and the two operations H-0116 decision 2 bounds. An explicit
    ``None`` is not refused: it means "no override" and trains on the task
    default (pinned below). A value whose type lookups raise, and rendering
    the error afterwards, are outside H-0116's guarantee.
    """

    def test_empty_string_objective_raises(self) -> None:
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        adapter = LGBMAdapter(task="regression", params={"objective": ""})
        with pytest.raises(LizyMLError) as excinfo:
            adapter._build_params()
        assert excinfo.value.code.name == "CONFIG_INVALID"
        assert excinfo.value.context["objective"] == ""

    def test_dict_form_objective_raises(self) -> None:
        """Dict-form ``{"huber": {}}`` is illegal for objective (only
        valid for metric MetricEntry per H-0065). Must reject.

        This is the adapter's ``_build_params`` only. That ``Model.fit``
        refuses it before training is asserted by
        ``test_unhashable_objective_through_model_fit_is_config_invalid``.
        """
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        adapter = LGBMAdapter(task="regression", params={"objective": {"huber": {}}})
        # CONFIG_INVALID only (H-0116, #270). The test used to accept a
        # TypeError as well, which is the cryptic error this class exists to
        # rule out, so it passed while the unhashable value escaped raw.
        with pytest.raises(LizyMLError) as excinfo:
            adapter._build_params()
        assert excinfo.value.code.name == "CONFIG_INVALID"
        assert excinfo.value.context["objective"] == {"huber": {}}

    @pytest.mark.parametrize("objective", [{"huber": {}}, ["huber"], {"huber"}])
    @pytest.mark.parametrize("surface", ["config", "fit_params"])
    def test_unhashable_objective_through_model_fit_is_config_invalid(
        self, objective: object, surface: str
    ) -> None:
        """The shipped path: ``Model.fit`` refuses it before training (H-0116)."""
        from lizyml import Model
        from tests._helpers import make_config, make_regression_df
        from tests._train_spy import record_lightgbm_calls

        if surface == "config":
            model = Model(make_config("regression", objective=objective))
            fit_params = None
        else:
            model = Model(make_config("regression"))
            fit_params = {"objective": objective}

        with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as excinfo:
            model.fit(data=make_regression_df(n=80), params=fit_params)
        assert excinfo.value.code.name == "CONFIG_INVALID"
        assert "objective" in excinfo.value.user_message
        assert not seen["train_params"]

    @pytest.mark.parametrize("surface", ["adapter", "fit_params"])
    def test_an_unprintable_objective_is_still_config_invalid(
        self, surface: str
    ) -> None:
        """Building the refusal must not format the rejected value.

        A dict whose key cannot be printed escaped as that key's own error
        while the adapter built its message, before ``CONFIG_INVALID`` (#270).
        Through ``Model.fit`` the value-domain gate refuses it first; either
        way the refusal is ``CONFIG_INVALID`` and nothing trains.
        """
        from lizyml import Model
        from lizyml.estimators.lgbm.adapter import LGBMAdapter
        from tests._helpers import make_config, make_regression_df
        from tests._train_spy import record_lightgbm_calls

        class Unprintable:
            def __repr__(self) -> str:
                raise RuntimeError("repr must not run")

            __str__ = __format__ = __repr__  # type: ignore[assignment]

        objective = {Unprintable(): 1}
        with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as excinfo:
            if surface == "adapter":
                LGBMAdapter(
                    task="regression", params={"objective": objective}
                )._build_params()
            else:
                Model(make_config("regression")).fit(
                    data=make_regression_df(n=80), params={"objective": objective}
                )
        assert excinfo.value.code.name == "CONFIG_INVALID"
        assert "objective" in excinfo.value.user_message
        if surface == "adapter":
            assert "of type 'dict'" in excinfo.value.user_message
        assert not seen["train_params"]

    def test_a_hostile_string_objective_is_judged_by_its_text(self) -> None:
        """A ``str`` subclass is judged by its plain text.

        The membership test and the message use none of the overrides it
        defines (``__hash__``, ``__eq__``, ``__format__``, ``__str__``,
        ``__repr__``).
        """
        from lizyml.estimators.lgbm.param_validation import check_objective_compatible

        class Hostile(str):
            def _boom(self, *args: object) -> object:
                raise RuntimeError("a str subclass method ran")

            __hash__ = __eq__ = __format__ = __str__ = __repr__ = _boom  # type: ignore[assignment]

        with pytest.raises(LizyMLError) as excinfo:
            check_objective_compatible("regression", Hostile("not_an_objective"))
        assert excinfo.value.code.name == "CONFIG_INVALID"
        assert (
            "objective 'not_an_objective' is not compatible"
            in excinfo.value.user_message
        )
        check_objective_compatible("regression", Hostile("huber"))  # accepted

    def test_non_string_int_objective_raises(self) -> None:
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        adapter = LGBMAdapter(task="regression", params={"objective": 42})
        with pytest.raises(LizyMLError) as excinfo:
            adapter._build_params()
        assert excinfo.value.code.name == "CONFIG_INVALID"

    def test_none_objective_falls_back_to_default(self) -> None:
        """Explicit None means "no override" — same as omitting the key."""
        from lizyml.estimators.lgbm.adapter import LGBMAdapter

        adapter = LGBMAdapter(task="regression", params={"objective": None})
        params, *_ = adapter._build_params()
        # Default for regression
        assert params["objective"] == "huber"

    @pytest.mark.parametrize("surface", ["config", "fit_params"])
    def test_none_objective_through_model_fit_trains_on_the_default(
        self, surface: str
    ) -> None:
        """None is the one non-string value H-0116 does not refuse.

        Both ``Model.fit`` surfaces treat it as "no override", so
        ``lgb.train`` receives the task default.
        """
        from lizyml import Model
        from tests._helpers import make_config, make_regression_df
        from tests._train_spy import record_lightgbm_calls

        if surface == "config":
            model = Model(make_config("regression", n_estimators=5, objective=None))
            fit_params = None
        else:
            model = Model(make_config("regression", n_estimators=5))
            fit_params = {"objective": None}

        with record_lightgbm_calls() as seen:
            model.fit(data=make_regression_df(n=80), params=fit_params)
        assert seen["train_params"], "no lgb.train call was recorded"
        assert {call["objective"] for call in seen["train_params"]} == {"huber"}


# ---------------------------------------------------------------------------
# G4 — Calibration on top of non-default binary objective
# ---------------------------------------------------------------------------


class TestG4CalibrationWithNonDefaultBinaryObjective:
    """Pre-H-0079 binary calibration could only be tested with the
    default ``objective="binary"`` because user-supplied values were
    stripped. Now ``cross_entropy`` (also a probability-emitting
    binary objective) flows through; the calibration cross-fit must
    produce a fitted calibrator and well-formed metrics.
    """

    def test_cross_entropy_objective_with_platt_calibration(self) -> None:
        df = make_binary_df(n=300, seed=0)
        cfg_dict = make_config(
            "binary",
            n_estimators=20,
            n_splits=2,
            split_method="stratified_kfold",
            calibration="platt",
            objective="cross_entropy",
        )
        m = Model(LizyMLConfig(**cfg_dict))
        result = m.fit(data=df)

        # Booster trained with cross_entropy
        assert (
            m._refit_result.model.get_native_model().params["objective"]  # type: ignore[union-attr]
            == "cross_entropy"
        )

        # Calibration ran without error and produced a fitted calibrator
        assert result.calibrator is not None
        assert "calibrated" in result.metrics
        # OOF metrics must be present and finite
        oof_cal = result.metrics["calibrated"]["oof"]
        assert oof_cal, (
            "Calibrated OOF metrics must be populated when calibration "
            "runs successfully on top of cross_entropy."
        )
        for metric_name, value in oof_cal.items():
            assert np.isfinite(value), (
                f"Calibrated OOF metric '{metric_name}'={value} must be "
                f"finite when binary objective='cross_entropy'."
            )

    def test_predict_proba_within_unit_interval_after_calibration(self) -> None:
        """Calibrator output must remain in [0, 1] regardless of
        upstream objective. Calibration is applied automatically by
        ``predict()`` when a fitted calibrator is present (no opt-in
        flag needed)."""
        df = make_binary_df(n=300, seed=0)
        cfg_dict = make_config(
            "binary",
            n_estimators=20,
            n_splits=2,
            split_method="stratified_kfold",
            calibration="platt",
            objective="cross_entropy",
        )
        m = Model(LizyMLConfig(**cfg_dict))
        m.fit(data=df)

        X_new = df.drop(columns=["target"]).iloc[:50].reset_index(drop=True)
        pred = m.predict(X_new)
        # Binary predict returns probabilities for the positive class via
        # ``proba``. Allow trivial numerical tolerance at the edges.
        proba = pred.proba
        assert proba is not None
        assert (proba >= -1e-9).all()
        assert (proba <= 1 + 1e-9).all()
