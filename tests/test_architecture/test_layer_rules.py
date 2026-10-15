"""BLUEPRINT §2.1's layer rules hold over every lizyml import (H-0129, #360).

H-0127 put a concrete estimator import in ``persistence/`` and no test noticed:
§2.1 had no check. ``_layer_rules.check`` reads the layers from the §2.1 table
and enforces every rule there. H-0130 (#362) closed the imports that broke a
rule before the rule had a test, and added the same-layer allowance table that
R2 reads. An import that breaks a rule can still be listed below with its exact
count until it is decided; a listed import that disappears, or a second import
of a listed pair, fails like any new violation, so the list can neither go stale
nor grow.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.test_architecture._layer_rules import Key, check

REPO = Path(__file__).resolve().parents[2]

#: Imports that break a §2.1 rule, with how many times each occurs. H-0130
#: closed the seventeen #362 left here: the same-layer allowance table admits
#: estimators -> features / metrics, and the rest were moved or re-declared.
EXCEPTIONS: dict[Key, int] = {}


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


#: H-0130 amendment 2: lower layers cannot name a Config class, because
#: ``config/`` is neither an interface nor an allowed same-layer target. This
#: pins the import edge only; selecting by ``cfg.name`` or another Config value
#: imports nothing and is kept by review (BLUEPRINT §2.1).
CONFIG_IMPORTS = {
    "layer 2": ("evaluation/evaluator.py", "R3"),
    "layer 1": ("estimators/provider.py", "R2"),
}


@pytest.mark.parametrize("case", sorted(CONFIG_IMPORTS))
def test_a_config_class_import_below_the_facade_fails(tree: Path, case: str) -> None:
    module, rule = CONFIG_IMPORTS[case]
    _append(tree, module, "from lizyml.config.schema import LGBMConfig")
    assert (
        f"{rule} lizyml/{module} -> lizyml.config.schema: "
        "1 import(s), 0 allowed by the exception list"
    ) in _check(tree)


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
    "__import__ level unpacked": '__import__("torch", **{"level": 1})',
    "__import__ star args": 'a = ("model", globals(), None, ["x"], 1)\n__import__(*a)',
    "import_module star args": _IL + 'importlib.import_module("torch", *("x",))',
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


@pytest.mark.parametrize(
    ("new", "match"),
    [
        (", `codegen/`（生成）", "joined by ', '"),
        ("; `codegen/`", "joined by ', '"),
        (", `/codegen/`", "is not a package/ or a module.py path"),
        (", `codegen//`", "is not a package/ or a module.py path"),
        (", `codegen`", "is not a package/ or a module.py path"),
    ],
    ids=["annotation", "semicolon", "leading slash", "double slash", "no suffix"],
)
def test_a_table_cell_outside_the_grammar_fails(
    tree: Path, new: str, match: str
) -> None:
    _edit(tree, "BLUEPRINT.md", ", `codegen/`", new)
    with pytest.raises(ValueError, match=match):
        _check(tree)


def test_a_malformed_separator_row_fails(tree: Path) -> None:
    header = "| 下の層から参照してよいモジュール（IF） |\n"
    _edit(
        tree,
        "BLUEPRINT.md",
        header + "|---|---|---|---|---|",
        header + "| x | x | x | x | x |",
    )
    with pytest.raises(ValueError, match="malformed separator row"):
        _check(tree)


@pytest.mark.parametrize(
    "extra",
    [
        {("lizyml/ghost.py", "lizyml.ghost", "R9"): 1},
        {("lizyml/ghost.py", "lizyml.ghost", "R3"): 0},
        {("lizyml/ghost.py", "lizyml.ghost", "R3"): True},
    ],
    ids=["unknown rule", "zero count", "bool count"],
)
def test_a_malformed_exception_entry_fails(extra: dict[Key, int]) -> None:
    problems = check(
        REPO / "lizyml",
        REPO / "BLUEPRINT.md",
        REPO / "pyproject.toml",
        {**EXCEPTIONS, **extra},
    )
    assert any(p.startswith("exception list:") for p in problems), problems


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


# --- Criterion 4 (H-0129) / H-0130 criterion 3: the exception list ------------
# The real list is empty since H-0130, so the mechanism is exercised on a copy
# with one injected violation and a list naming it.

_INJECTED = "import lizyml.training.oof_assembly"
_INJECTED_KEY: Key = (
    "lizyml/evaluation/confusion.py",
    "lizyml.training.oof_assembly",
    "R3",
)


def _check_with(root: Path, exceptions: dict[Key, int]) -> list[str]:
    return check(
        root / "lizyml", root / "BLUEPRINT.md", root / "pyproject.toml", exceptions
    )


def test_the_exception_list_is_empty() -> None:
    assert EXCEPTIONS == {}


def test_a_listed_violation_passes(tree: Path) -> None:
    _append(tree, "evaluation/confusion.py", _INJECTED)
    assert _check(tree) != []
    assert _check_with(tree, {_INJECTED_KEY: 1}) == []


def test_a_fixed_exception_must_leave_the_list(tree: Path) -> None:
    assert (
        "R3 lizyml/evaluation/confusion.py -> lizyml.training.oof_assembly: "
        "0 import(s), 1 allowed by the exception list"
    ) in _check_with(tree, {_INJECTED_KEY: 1})


def test_a_second_import_of_a_listed_pair_fails(tree: Path) -> None:
    _append(tree, "evaluation/confusion.py", _INJECTED)
    _append(tree, "evaluation/confusion.py", _INJECTED)
    assert (
        "R3 lizyml/evaluation/confusion.py -> lizyml.training.oof_assembly: "
        "2 import(s), 1 allowed by the exception list"
    ) in _check_with(tree, {_INJECTED_KEY: 1})


# --- H-0130 criterion 2: the same-layer allowance table -----------------------

LEAD = "同じ層の中で許す参照（H-0130）:"
HEADER = "| 参照する側 | 参照される側 | 根拠 |"
FEATURES_ROW = (
    "| `estimators/` | `features/` | §2.2 の provider が pipeline の factory を組む"
    "（H-0053）。`UnseenPolicy` を受け取り、具象の pipeline を作る |"
)
METRICS_ROW = (
    "| `estimators/` | `metrics/` | LightGBM の feval への橋渡し"
    "（`estimators/lgbm/metric_bridge.py`）と、provider の指標の解決 |"
)
AFTER_BLOCK = "\n\n表の読み方と検査（H-0129、H-0130）:"


def test_the_blueprint_holds_the_allowance_block_these_tests_mutate() -> None:
    text = (REPO / "BLUEPRINT.md").read_text(encoding="utf-8")
    block = f"{LEAD}\n\n{HEADER}\n|---|---|---|\n{FEATURES_ROW}\n{METRICS_ROW}"
    assert text.count(block + AFTER_BLOCK) == 1


def test_an_undeclared_layer_1_pair_fails(tree: Path) -> None:
    _append(tree, "estimators/base.py", "import lizyml.calibration.base")
    assert (
        "R2 lizyml/estimators/base.py -> lizyml.calibration.base: "
        "1 import(s), 0 allowed by the exception list"
    ) in _check(tree)


def test_the_reverse_of_an_allowed_pair_fails(tree: Path) -> None:
    _append(tree, "features/pipeline_base.py", "import lizyml.estimators.base")
    rules = _rules(_check(tree))
    assert {"R2", "R4"} <= rules


def test_removing_an_allowance_row_fails_its_imports(tree: Path) -> None:
    _edit(tree, "BLUEPRINT.md", "\n" + METRICS_ROW, "")
    problems = [p for p in _check(tree) if p.startswith("R2 ")]
    assert sorted(p.split(":")[0] for p in problems) == [
        "R2 lizyml/estimators/lgbm/metric_bridge.py -> lizyml.metrics.base",
        "R2 lizyml/estimators/lgbm/metric_bridge.py -> lizyml.metrics.registry",
        "R2 lizyml/estimators/lgbm/provider.py -> lizyml.metrics.registry",
    ]


def test_an_allowance_no_import_uses_fails(tree: Path) -> None:
    _edit(
        tree,
        "BLUEPRINT.md",
        METRICS_ROW,
        METRICS_ROW + "\n| `estimators/` | `splitters/` | unused |",
    )
    assert (
        "R2 the §2.1 allowance lizyml.estimators -> lizyml.splitters "
        "is used by no import"
    ) in _check(tree)


def test_any_module_of_an_allowed_category_passes(tree: Path) -> None:
    _append(tree, "estimators/base.py", "import lizyml.metrics.regression")
    assert _check(tree) == []


SHORT_FEATURES = "| `estimators/` | `features/` |"

#: One representative mutation per position of the full-match grammar
#: (H-0130 policy 6): (old, new, expected message).
ALLOWANCE_MUTATIONS = {
    "lead missing": (LEAD + "\n", "", "0 lines equal"),
    "lead twice": (AFTER_BLOCK, "\n\n" + LEAD + AFTER_BLOCK, "2 lines equal"),
    "lead trailing space": (LEAD + "\n", LEAD + " \n", "0 lines equal"),
    "blank missing": (LEAD + "\n\n", LEAD + "\n", "line 1 after the lead"),
    "blank twice": (LEAD + "\n\n", LEAD + "\n\n\n", "line 2 after the lead"),
    "header changed": (HEADER, HEADER.replace("根拠", "理由"), "line 2 after the lead"),
    "separator two cells": (
        "|---|---|---|\n" + FEATURES_ROW,
        "|---|---|\n" + FEATURES_ROW,
        "line 3 after the lead",
    ),
    "separator spaced": (
        "|---|---|---|\n" + FEATURES_ROW,
        "| --- | --- | --- |\n" + FEATURES_ROW,
        "line 3 after the lead",
    ),
    "separator aligned": (
        "|---|---|---|\n" + FEATURES_ROW,
        "|:---|---|---|\n" + FEATURES_ROW,
        "line 3 after the lead",
    ),
    "source without slash": (
        SHORT_FEATURES,
        "| `estimators` | `features/` |",
        "malformed row",
    ),
    "target with two paths": (
        SHORT_FEATURES,
        "| `estimators/` | `features/`, `metrics/` |",
        "malformed row",
    ),
    "empty reason": (FEATURES_ROW, SHORT_FEATURES + "  |", "malformed row"),
    "two cells": (FEATURES_ROW, SHORT_FEATURES, "malformed row"),
    "four cells": (FEATURES_ROW, FEATURES_ROW + " x |", "malformed row"),
    "row trailing space": (FEATURES_ROW, FEATURES_ROW + " ", "malformed row"),
    "no rows": ("\n" + FEATURES_ROW + "\n" + METRICS_ROW, "", "no rows"),
    "stray row before": (
        "\n\n" + LEAD,
        "\n\n| stray |\n\n" + LEAD,
        "table row outside both tables",
    ),
    "stray row after": (
        AFTER_BLOCK,
        "\n\n| stray |" + AFTER_BLOCK,
        "table row outside both tables",
    ),
    "not a layer 1 category": (
        SHORT_FEATURES,
        "| `estimators/` | `training/` |",
        "lizyml.training is not a Layer 1 category",
    ),
    "same category": (
        SHORT_FEATURES,
        "| `estimators/` | `estimators/` |",
        "allowed to itself",
    ),
    "pair twice": (
        METRICS_ROW,
        METRICS_ROW + "\n" + FEATURES_ROW,
        "lizyml.estimators -> lizyml.features twice",
    ),
}


@pytest.mark.parametrize("case", sorted(ALLOWANCE_MUTATIONS))
def test_an_allowance_block_outside_the_grammar_fails(tree: Path, case: str) -> None:
    old, new, match = ALLOWANCE_MUTATIONS[case]
    _edit(tree, "BLUEPRINT.md", old, new)
    with pytest.raises(ValueError, match=match):
        _check(tree)
