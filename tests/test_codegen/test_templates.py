"""Tests for codegen templates — train.py and predict.py source generation."""

from __future__ import annotations

import ast

from lizyml.codegen.templates import (
    render_predict_py,
    render_test_equivalence_py,
    render_train_py,
)


class TestRenderTrainPy:
    def test_returns_string(self) -> None:
        src = render_train_py()
        assert isinstance(src, str)
        assert len(src) > 100

    def test_valid_python(self) -> None:
        """Generated train.py must be valid Python syntax."""
        src = render_train_py()
        ast.parse(src)

    def test_contains_main_guard(self) -> None:
        src = render_train_py()
        assert 'if __name__ == "__main__"' in src

    def test_contains_key_functions(self) -> None:
        src = render_train_py()
        tree = ast.parse(src)
        func_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert "fit_pipeline" in func_names
        assert "train_lgbm" in func_names
        assert "fit_calibrator" in func_names
        assert "train" in func_names
        assert "main" in func_names

    def test_contains_imports(self) -> None:
        src = render_train_py()
        assert "import lightgbm" in src
        assert "import numpy" in src
        assert "import pandas" in src

    def test_contains_calibration_methods(self) -> None:
        src = render_train_py()
        assert "_fit_platt" in src
        assert "_fit_beta" in src
        assert "_fit_isotonic" in src

    def test_contains_oof_generation(self) -> None:
        src = render_train_py()
        assert "_generate_oof" in src
        assert "StratifiedKFold" in src


class TestRenderPredictPy:
    def test_returns_string(self) -> None:
        src = render_predict_py()
        assert isinstance(src, str)
        assert len(src) > 100

    def test_valid_python(self) -> None:
        """Generated predict.py must be valid Python syntax."""
        src = render_predict_py()
        ast.parse(src)

    def test_contains_main_guard(self) -> None:
        src = render_predict_py()
        assert 'if __name__ == "__main__"' in src

    def test_contains_key_functions(self) -> None:
        src = render_predict_py()
        tree = ast.parse(src)
        func_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert "transform" in func_names
        assert "calibrate" in func_names
        assert "predict" in func_names
        assert "main" in func_names

    def test_contains_imports(self) -> None:
        src = render_predict_py()
        assert "import lightgbm" in src
        assert "import numpy" in src
        assert "import pandas" in src

    def test_no_sklearn_dependency(self) -> None:
        """predict.py must not depend on sklearn."""
        src = render_predict_py()
        assert "sklearn" not in src
        assert "scipy" not in src

    def test_contains_calibration_apply(self) -> None:
        src = render_predict_py()
        assert "platt" in src
        assert "beta" in src
        assert "isotonic" in src

    def test_contains_task_branches(self) -> None:
        src = render_predict_py()
        assert "regression" in src
        assert "binary" in src
        assert "multiclass" in src


class TestRenderTestEquivalencePy:
    def test_returns_string(self) -> None:
        src = render_test_equivalence_py()
        assert isinstance(src, str)
        assert len(src) > 100

    def test_valid_python(self) -> None:
        src = render_test_equivalence_py()
        ast.parse(src)

    def test_contains_main_guard(self) -> None:
        src = render_test_equivalence_py()
        assert 'if __name__ == "__main__"' in src

    def test_contains_check_equivalence(self) -> None:
        src = render_test_equivalence_py()
        tree = ast.parse(src)
        func_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert "check_equivalence" in func_names
        assert "main" in func_names

    def test_contains_rtol(self) -> None:
        src = render_test_equivalence_py()
        assert "rtol" in src
        assert "1e-7" in src
