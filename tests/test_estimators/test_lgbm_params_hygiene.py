"""LGBMAdapter param hygiene regressions (code-review fixes).

- Cross-task objective injection must raise CONFIG_INVALID (H-0079).
- update_params must not mutate the caller's params dict.
"""

from __future__ import annotations

import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.estimators.lgbm import LGBMAdapter


class TestObjectiveOverwriteProtection:
    def test_objective_locked_after_user_params(self) -> None:
        adapter = LGBMAdapter(task="binary", params={"objective": "regression"})
        with pytest.raises(LizyMLError) as excinfo:
            adapter._build_params()
        assert excinfo.value.code.name == "CONFIG_INVALID"


class TestUpdateParamsNoMutation:
    def test_original_params_unchanged(self) -> None:
        original = {"learning_rate": 0.1}
        adapter = LGBMAdapter(task="regression", params=original)
        adapter.update_params({"max_depth": 5})
        assert "max_depth" not in original
