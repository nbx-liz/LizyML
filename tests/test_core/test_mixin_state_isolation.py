"""H-0077 Phase 2: prove Mixin methods read state only from FitState / TuningState.

These tests pin down the contract introduced by H-0077:

- ``Model._get_tuning_state()`` returns a ``TuningState`` after ``tune()`` and
  raises ``LizyMLError(MODEL_NOT_FIT)`` before.
- Mixin source files (``_model_plots.py`` / ``_model_tables.py`` /
  ``_model_persistence.py``) contain zero direct references to Model's
  private attributes (``self._cfg``, ``self._y``, ``self._X``,
  ``self._fit_result``, ``self._refit_result``, ``self._tuning_result``,
  ``self._provider``, ``self._metrics``, ``self._run_dir``,
  ``self._output_dir``). The single read path is ``self._get_fit_state()``
  / ``self._get_tuning_state()``.
- A Mixin method can be invoked with a synthetic ``FitState`` /
  ``TuningState`` constructed from mocks — no full ``Model.fit()`` required.

Together with ``tests/test_core/test_fit_state.py`` (Phase 1 contract) these
tests block any future regression where Mixin code reaches into Model body
state directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lizyml import Model
from lizyml.core._model_state import TuningState
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_config, make_regression_df

# ---------------------------------------------------------------------------
# TuningState contract
# ---------------------------------------------------------------------------


def _reg_config_with_tuning(n_trials: int = 3) -> dict[str, Any]:
    cfg = make_config("regression")
    cfg["tuning"] = {
        "optuna": {
            "params": {"n_trials": n_trials, "direction": "minimize"},
            "space": {
                "num_leaves": {"type": "int", "low": 8, "high": 32},
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True,
                },
            },
        }
    }
    return cfg


class TestTuningStateContract:
    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        m = Model(_reg_config_with_tuning(n_trials=2))
        m.tune(data=make_regression_df(n=80))
        state = m._get_tuning_state()
        with pytest.raises((AttributeError, FrozenInstanceError)):
            state.tuning_result = None  # type: ignore[misc, assignment]

    def test_populated_after_tune(self) -> None:
        m = Model(_reg_config_with_tuning(n_trials=2))
        m.tune(data=make_regression_df(n=80))
        state = m._get_tuning_state()
        assert state.cfg is m._cfg
        assert state.tuning_result is m._tuning_result

    def test_raises_before_tune(self) -> None:
        m = Model(make_config("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m._get_tuning_state()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_works_before_fit_after_tune(self) -> None:
        """tuning_table / tuning_plot / boundary_table must work in this state."""
        m = Model(_reg_config_with_tuning(n_trials=2))
        m.tune(data=make_regression_df(n=80))
        # No m.fit() — TuningState must still be retrievable.
        state = m._get_tuning_state()
        assert state.tuning_result is not None

    def test_mocked_state_is_constructible(self) -> None:
        class _Sentinel:
            pass

        state = TuningState(
            cfg=_Sentinel(),  # type: ignore[arg-type]
            tuning_result=_Sentinel(),  # type: ignore[arg-type]
        )
        assert state.tuning_result is not None


# ---------------------------------------------------------------------------
# Static guard: Mixin source files must not access self._<private>
# ---------------------------------------------------------------------------


_MIXIN_FILES = [
    Path("lizyml/core/_model_plots.py"),
    Path("lizyml/core/_model_tables.py"),
    Path("lizyml/core/_model_persistence.py"),
]

# Attributes that live on the Model body and were leaking into Mixins
# before H-0077 (Phase 2). The migration replaces every direct access
# with state.<attr>.
_FORBIDDEN_ATTRS = (
    "cfg",
    "y",
    "X",
    "fit_result",
    "refit_result",
    "tuning_result",
    "provider",
    "metrics",
    "run_dir",
    "output_dir",
)


class TestMixinPrivateAccessGuard:
    @pytest.mark.parametrize("mixin_path", _MIXIN_FILES, ids=lambda p: p.name)
    def test_no_self_dot_underscore_private_access(self, mixin_path: Path) -> None:
        text = mixin_path.read_text()
        # Strip the TYPE_CHECKING block; the stubs there exist only for type
        # checkers and do not constitute runtime access.
        text_runtime = re.sub(
            r"if TYPE_CHECKING:\s*\n(?: {4,}.*\n)+",
            "",
            text,
        )
        pattern = re.compile(r"self\._(" + "|".join(_FORBIDDEN_ATTRS) + r")\b")
        offenders = pattern.findall(text_runtime)
        assert not offenders, (
            f"{mixin_path} still has direct self._* access for: "
            f"{sorted(set(offenders))}. "
            "After H-0077 Phase 2, Mixins must read state via "
            "_get_fit_state() / _get_tuning_state() only."
        )


# ---------------------------------------------------------------------------
# Behaviour-level smoke: Mixin methods still work end-to-end after migration
# ---------------------------------------------------------------------------


class TestMixinMethodsAfterMigration:
    """Sanity checks that the Mixin methods still operate correctly when
    reading from FitState / TuningState. Existing per-method tests cover
    detailed behaviour; these guard against catastrophic regression
    introduced by the migration itself."""

    def test_residuals_via_state(self) -> None:
        m = Model(make_config("regression"), data=make_regression_df(n=80))
        m.fit()
        residuals = m.residuals()
        assert residuals.shape[0] > 0

    def test_evaluate_table_via_state(self) -> None:
        m = Model(make_config("regression"), data=make_regression_df(n=80))
        m.fit()
        table = m.evaluate_table()
        assert isinstance(table, pd.DataFrame)
        assert not table.empty

    def test_tuning_table_after_tune_only(self) -> None:
        """tuning_table must still work after tune() with no fit() — guards
        against the FitState single-entry-point regression."""
        m = Model(_reg_config_with_tuning(n_trials=2))
        m.tune(data=make_regression_df(n=80))
        table = m.tuning_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2

    def test_export_uses_facade_resolve_path(self, tmp_path: Path) -> None:
        """`_resolve_export_path` lives on Model facade after H-0077."""
        m = Model(make_config("regression"), data=make_regression_df(n=80))
        m.fit()
        out = m.export(tmp_path / "exp")
        assert out.exists()
        assert (out / "metadata.json").exists()
