"""BLUEPRINT §2.1's layer rules hold over every lizyml import (H-0129, #360).

H-0127 put a concrete estimator import in ``persistence/`` and no test noticed:
§2.1 had no check. ``_layer_rules.check`` reads the layers from the §2.1 table
and enforces every rule there. The imports that broke a rule before the rule
had a test are listed below with their exact count; #362 decides them after
v0.19.0. A listed import that disappears, or a second import of a listed pair,
fails like any new violation, so the list can neither go stale nor grow.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.test_architecture._layer_rules import Key, check

REPO = Path(__file__).resolve().parents[2]

#: Imports that break a §2.1 rule, with how many times each occurs (#362).
EXCEPTIONS: dict[Key, int] = {
    # R2: Layer 1 imports another Layer 1 category (only provider -> config is
    # documented, BLUEPRINT §14.4).
    (
        "lizyml/estimators/provider.py",
        "lizyml.features.encoders.categorical_encoder",
        "R2",
    ): 1,
    ("lizyml/estimators/provider.py", "lizyml.features.pipeline_base", "R2"): 1,
    ("lizyml/estimators/lgbm/provider.py", "lizyml.config.schema", "R2"): 1,
    (
        "lizyml/estimators/lgbm/provider.py",
        "lizyml.features.encoders.categorical_encoder",
        "R2",
    ): 1,
    ("lizyml/estimators/lgbm/provider.py", "lizyml.features.pipeline_base", "R2"): 1,
    ("lizyml/estimators/lgbm/provider.py", "lizyml.features.pipelines_native", "R2"): 1,
    ("lizyml/estimators/lgbm/provider.py", "lizyml.metrics.registry", "R2"): 1,
    ("lizyml/estimators/lgbm/metric_bridge.py", "lizyml.metrics.base", "R2"): 1,
    ("lizyml/estimators/lgbm/metric_bridge.py", "lizyml.metrics.registry", "R2"): 1,
    # R3: Layer 2/3 imports a module that is not a declared interface.
    ("lizyml/explain/shap_explainer.py", "lizyml.features.pipelines_native", "R3"): 1,
    ("lizyml/evaluation/evaluator.py", "lizyml.metrics.registry", "R3"): 1,
    ("lizyml/plots/calibration.py", "lizyml.calibration.cross_fit", "R3"): 1,
    ("lizyml/evaluation/confusion.py", "lizyml.training.oof_assembly", "R3"): 1,
    ("lizyml/evaluation/evaluator.py", "lizyml.training.oof_assembly", "R3"): 1,
    ("lizyml/persistence/exporter.py", "lizyml.training.refit_trainer", "R3"): 1,
    # R4: the package root and core.model import each other (__version__).
    ("lizyml/__init__.py", "lizyml.core.model", "R4"): 1,
    ("lizyml/core/model.py", "lizyml", "R4"): 1,
}


def _check(root: Path) -> list[str]:
    return check(
        root / "lizyml", root / "BLUEPRINT.md", root / "pyproject.toml", EXCEPTIONS
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A copy of lizyml/, BLUEPRINT.md and pyproject.toml to mutate."""
    shutil.copytree(
        REPO / "lizyml",
        tmp_path / "lizyml",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("BLUEPRINT.md", "pyproject.toml"):
        shutil.copy(REPO / name, tmp_path / name)
    return tmp_path


def _append(tree: Path, module: str, text: str) -> None:
    path = tree / "lizyml" / module
    path.write_text(
        path.read_text(encoding="utf-8") + "\n" + text + "\n", encoding="utf-8"
    )


def _edit(tree: Path, name: str, old: str, new: str) -> None:
    path = tree / name
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{name}: {old!r} occurs {text.count(old)} times"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _rules(problems: list[str]) -> set[str]:
    return {p.split()[0] for p in problems}


# --- Criterion 1 --------------------------------------------------------------


def test_the_layer_rules_hold() -> None:
    assert _check(REPO) == []


def test_the_copy_holds_before_any_mutation(tree: Path) -> None:
    assert _check(tree) == []


# --- Criterion 2: each spelling of a concrete import from Layer 3 -------------

CONCRETE_IMPORTS = {
    "absolute": "from lizyml.estimators.lgbm.category_json import shown",
    "relative": "from ..estimators.lgbm import category_json",
    "imported name": "from lizyml.estimators import lgbm",
    "import statement": "import lizyml.estimators.lgbm.category_json",
}


@pytest.mark.parametrize("form", sorted(CONCRETE_IMPORTS))
def test_a_concrete_import_from_layer_3_fails(tree: Path, form: str) -> None:
    _append(tree, "persistence/loader.py", CONCRETE_IMPORTS[form])
    problems = _check(tree)
    assert any(
        p.startswith("R3 lizyml/persistence/loader.py -> lizyml.estimators.lgbm")
        for p in problems
    ), problems


# --- Criterion 3: one should-fail case per rule -------------------------------

RULE_MUTATIONS = {
    "R1 layer 1 imports layer 4": (
        "metrics/base.py",
        "from lizyml.core.model import Model",
        "R1",
    ),
    "R2 layer 1 imports layer 1": (
        "splitters/base.py",
        "from lizyml.metrics.base import BaseMetric",
        "R2",
    ),
    "R3 layer 2 imports layer 2": (
        "training/__init__.py",
        "from lizyml.tuning import tuner",
        "R3",
    ),
    "R3 layer 3 imports layer 3": (
        "codegen/__init__.py",
        "from lizyml.plots import _theme",
        "R3",
    ),
}


@pytest.mark.parametrize("case", sorted(RULE_MUTATIONS))
def test_each_rule_fails_on_its_violation(tree: Path, case: str) -> None:
    module, line, rule = RULE_MUTATIONS[case]
    _append(tree, module, line)
    assert rule in _rules(_check(tree))


def test_a_category_cycle_fails(tree: Path) -> None:
    _append(tree, "core/seed.py", "import lizyml.core._smart_values")
    _append(tree, "core/_smart_values.py", "import lizyml.core.seed")
    assert "R4" in _rules(_check(tree))


_IL = "import importlib\n"

DYNAMIC_FAILS = {
    "import_module of lizyml": _IL + 'importlib.import_module("lizyml.core.model")',
    "__import__ of lizyml": '__import__("lizyml.core.model")',
    "importlib.__import__": _IL + 'importlib.__import__("lizyml.core.model")',
    "non-constant name": 'import importlib\nname = "x"\nimportlib.import_module(name)',
    "module alias": 'import importlib as il\nil.import_module("lizyml.core.model")',
    "from-import": "from importlib import import_module",
    "from-import alias": "from importlib import import_module as load",
    "builtins.__import__": 'import builtins\nbuiltins.__import__("lizyml.core.model")',
    "uncalled reference": "import importlib\nf = importlib.import_module",
    "getattr string": 'import importlib\ng = getattr(importlib, "import_module")',
    "relative name": _IL + 'importlib.import_module(".core.model", "lizyml")',
    "__import__ level 1": '__import__("core.model", globals(), None, ["x"], 1)',
    "__import__ level keyword": '__import__("torch", level=1)',
}

DYNAMIC_PASSES = {
    "import_module of torch": 'import importlib\nimportlib.import_module("torch")',
    "alias of torch": 'import importlib as il\nil.import_module("torch")',
    "__import__ of torch": '__import__("torch")',
    "__import__ level 0": '__import__("torch", globals(), None, [], 0)',
}


@pytest.mark.parametrize("case", sorted(DYNAMIC_FAILS))
def test_a_dynamic_import_outside_the_rule_fails(tree: Path, case: str) -> None:
    _append(tree, "core/seed.py", DYNAMIC_FAILS[case])
    assert "R5" in _rules(_check(tree))


@pytest.mark.parametrize("case", sorted(DYNAMIC_PASSES))
def test_a_dynamic_import_of_another_package_passes(tree: Path, case: str) -> None:
    _append(tree, "core/seed.py", DYNAMIC_PASSES[case])
    assert _check(tree) == []


def test_a_module_without_a_row_fails(tree: Path) -> None:
    (tree / "lizyml" / "newpkg").mkdir()
    (tree / "lizyml" / "newpkg" / "__init__.py").write_text("", encoding="utf-8")
    assert "R0 lizyml.newpkg has no row in the §2.1 table" in _check(tree)


def test_an_unparseable_module_fails(tree: Path) -> None:
    _append(tree, "core/seed.py", "def (")
    with pytest.raises(SyntaxError):
        _check(tree)


def test_a_removed_row_fails(tree: Path) -> None:
    _edit(tree, "BLUEPRINT.md", ", `codegen/`", "")
    assert any(p.startswith("R0 lizyml.codegen") for p in _check(tree))


def test_a_row_naming_no_module_fails(tree: Path) -> None:
    _edit(tree, "BLUEPRINT.md", ", `codegen/`", ", `codegen/`, `ghost/`")
    assert "R0 the §2.1 table names lizyml.ghost, which no module is" in _check(tree)


def test_a_table_cell_outside_the_grammar_fails(tree: Path) -> None:
    _edit(tree, "BLUEPRINT.md", ", `codegen/`", ", `codegen/`（生成）")
    with pytest.raises(ValueError, match="backticked paths only"):
        _check(tree)


# --- Criterion 3b: the generated _version.py ----------------------------------


def test_a_tree_without_the_generated_module_holds(tree: Path) -> None:
    (tree / "lizyml" / "_version.py").unlink(missing_ok=True)
    assert _check(tree) == []


def test_a_version_file_outside_the_table_fails(tree: Path) -> None:
    (tree / "lizyml" / "_version.py").unlink(missing_ok=True)
    _edit(
        tree,
        "pyproject.toml",
        'version-file = "lizyml/_version.py"',
        'version-file = "lizyml/_ver2.py"',
    )
    problems = _check(tree)
    assert "R0 the generated lizyml._ver2 has no row in the §2.1 table" in problems


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        (
            'version-file = "lizyml/_version.py"',
            'version-file = "lizyml/_version.py',
            ".",
        ),
        ('version-file = "lizyml/_version.py"', "", "version-file is missing"),
        ('version-file = "lizyml/_version.py"', "version-file = 3", "must be a string"),
    ],
    ids=["malformed TOML", "missing key", "not a string"],
)
def test_a_broken_version_file_declaration_fails(
    tree: Path, old: str, new: str, match: str
) -> None:
    """Checked on every run, even when a stale _version.py is still present."""
    (tree / "lizyml" / "_version.py").write_text(
        "__version__ = '0'\n", encoding="utf-8"
    )
    _edit(tree, "pyproject.toml", old, new)
    with pytest.raises(ValueError, match=match):
        _check(tree)


# --- Criterion 4: the exception list cannot go stale or grow ------------------


def test_a_fixed_exception_must_leave_the_list(tree: Path) -> None:
    _edit(
        tree,
        "lizyml/evaluation/confusion.py",
        "from lizyml.training.oof_assembly import compute_oof_valid_mask\n",
        "",
    )
    assert (
        "R3 lizyml/evaluation/confusion.py -> lizyml.training.oof_assembly: "
        "0 import(s), 1 allowed by the exception list"
    ) in _check(tree)


def test_a_second_import_of_a_listed_pair_fails(tree: Path) -> None:
    _append(tree, "evaluation/confusion.py", "import lizyml.training.oof_assembly")
    assert (
        "R3 lizyml/evaluation/confusion.py -> lizyml.training.oof_assembly: "
        "2 import(s), 1 allowed by the exception list"
    ) in _check(tree)
