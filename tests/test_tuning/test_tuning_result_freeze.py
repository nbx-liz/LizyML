"""TuningResult must deep-freeze contained dicts/lists (code-review fixes)."""

from __future__ import annotations

from lizyml.core.types.tuning_result import TrialResult, TuningResult


class TestTuningResultDeepFreeze:
    def test_best_params_immutable(self) -> None:
        source_params = {"lr": 0.1}
        tr = TuningResult(
            best_model_params=source_params,
            best_smart_params={},
            best_training_params={},
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        source_params["lr"] = 999.0
        assert tr.best_params["lr"] == 0.1

    def test_trials_list_immutable(self) -> None:
        trials = [TrialResult(number=0, params={}, score=0.5, state="complete")]
        tr = TuningResult(
            best_model_params={},
            best_smart_params={},
            best_training_params={},
            best_score=0.5,
            trials=trials,
            metric_name="rmse",
            direction="minimize",
        )
        trials.append(TrialResult(number=1, params={}, score=0.6, state="complete"))
        assert len(tr.trials) == 1
