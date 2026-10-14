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

from tests.test_architecture._dispatch_rules import DispatchKey
from tests.test_architecture._layer_rules import Key, check

REPO = Path(__file__).resolve().parents[2]

#: Imports that break a §2.1 rule, with how many times each occurs. H-0130
#: closed the seventeen #362 left here: the same-layer allowance table admits
#: estimators -> features / metrics, and the rest were moved or re-declared.
EXCEPTIONS: dict[Key, int] = {}

#: R6 (H-0130 amendment 1): the type dispatches Layers 1-3 may make on a
#: concrete lizyml class, and the ones whose target the checker cannot resolve,
#: with exact counts. BLUEPRINT §2.1 states the same exception in prose.
DISPATCH_ALLOWED: dict[DispatchKey, int] = {
    # The provider checks its own adapter type (§2.1's first exception):
    # params_summary (feval display names) and build_export_params.
    (
        "lizyml/estimators/lgbm/provider.py",
        "lizyml.estimators.lgbm.adapter.LGBMAdapter",
    ): 2,
    # Loop variables over tuples of builtin types: (int,) / (int, float), and
    # tuning.search_space._ALLOWED_CHOICE_TYPES.
    ("lizyml/tuning/lineage_codec.py", "unresolved:t"): 2,
    ("lizyml/tuning/search_space.py", "unresolved:allowed"): 1,
}


def _check(root: Path) -> list[str]:
    return check(
        root / "lizyml",
        root / "BLUEPRINT.md",
        root / "pyproject.toml",
        EXCEPTIONS,
        dispatch_allowed=DISPATCH_ALLOWED,
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
        dispatch_allowed=DISPATCH_ALLOWED,
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
        root / "lizyml",
        root / "BLUEPRINT.md",
        root / "pyproject.toml",
        exceptions,
        dispatch_allowed=DISPATCH_ALLOWED,
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


# --- H-0130 amendment 1: R6, concrete assembly and type dispatch -------------

_ADAPTER = "from lizyml.estimators.lgbm.adapter import LGBMAdapter\n"
_CONCRETE = "lizyml.estimators.lgbm.adapter.LGBMAdapter"

#: One representative per position of R6's closed dispatch grammar, each added
#: to estimators/base.py (same category, so no import rule fires).
DISPATCH_MUTATIONS = {
    "isinstance": _ADAPTER + "isinstance(x, LGBMAdapter)",
    "issubclass": _ADAPTER + "issubclass(x, LGBMAdapter)",
    "type is": _ADAPTER + "type(x) is LGBMAdapter",
    "reversed ==": _ADAPTER + "LGBMAdapter == type(x)",
    "is not": _ADAPTER + "type(x) is not LGBMAdapter",
    "case class": _ADAPTER + "match x:\n    case LGBMAdapter():\n        pass",
    "tuple": _ADAPTER + "isinstance(x, (int, LGBMAdapter))",
    "union": _ADAPTER + "isinstance(x, int | LGBMAdapter)",
    "module constant": _ADAPTER + "_KINDS = (LGBMAdapter,)\nisinstance(x, _KINDS)",
    "attribute through a relative import": (
        "from .lgbm import adapter as a\nisinstance(x, a.LGBMAdapter)"
    ),
    "re-exported name": (
        "from lizyml.estimators.lgbm import LGBMAdapter\nisinstance(x, LGBMAdapter)"
    ),
}


@pytest.mark.parametrize("case", sorted(DISPATCH_MUTATIONS))
def test_a_concrete_type_dispatch_outside_the_list_fails(tree: Path, case: str) -> None:
    _append(tree, "estimators/base.py", DISPATCH_MUTATIONS[case])
    assert (
        f"R6 lizyml/estimators/base.py -> {_CONCRETE}: "
        "1 dispatch(es), 0 allowed by the dispatch list"
    ) in _check(tree)


def test_an_unresolvable_dispatch_target_fails(tree: Path) -> None:
    _append(tree, "estimators/base.py", "def f(kind):\n    return isinstance(1, kind)")
    assert (
        "R6 lizyml/estimators/base.py -> unresolved:kind: "
        "1 dispatch(es), 0 allowed by the dispatch list"
    ) in _check(tree)


def test_a_second_dispatch_of_a_listed_pair_fails(tree: Path) -> None:
    _append(tree, "estimators/lgbm/provider.py", "isinstance(x, LGBMAdapter)")
    assert (
        f"R6 lizyml/estimators/lgbm/provider.py -> {_CONCRETE}: "
        "3 dispatch(es), 2 allowed by the dispatch list"
    ) in _check(tree)


def test_a_dispatch_list_entry_no_code_uses_fails() -> None:
    problems = check(
        REPO / "lizyml",
        REPO / "BLUEPRINT.md",
        REPO / "pyproject.toml",
        EXCEPTIONS,
        dispatch_allowed={**DISPATCH_ALLOWED, ("lizyml/ghost.py", "unresolved:x"): 1},
    )
    assert (
        "R6 lizyml/ghost.py -> unresolved:x: 0 dispatch(es), "
        "1 allowed by the dispatch list"
    ) in problems


@pytest.mark.parametrize("count", [0, True], ids=["zero", "bool"])
def test_a_malformed_dispatch_list_entry_fails(count: int) -> None:
    problems = check(
        REPO / "lizyml",
        REPO / "BLUEPRINT.md",
        REPO / "pyproject.toml",
        EXCEPTIONS,
        dispatch_allowed={
            **DISPATCH_ALLOWED,
            ("lizyml/ghost.py", "unresolved:x"): count,
        },
    )
    assert any(p.startswith("dispatch list:") for p in problems), problems


_REGISTRY = "from lizyml.core.registries import MetricRegistry\n"

REGISTRY_MUTATIONS = {
    "get": (_REGISTRY + 'MetricRegistry.get("rmse")', "MetricRegistry.get"),
    "private attribute": (_REGISTRY + "MetricRegistry._items", "MetricRegistry._items"),
}


@pytest.mark.parametrize("case", sorted(REGISTRY_MUTATIONS))
def test_registry_access_outside_the_two_registries_fails(
    tree: Path, case: str
) -> None:
    text, used = REGISTRY_MUTATIONS[case]
    _append(tree, "splitters/base.py", text)
    assert any(
        p.startswith("R6 lizyml/splitters/base.py:")
        and f"uses lizyml.core.registries.{used} outside" in p
        for p in _check(tree)
    )


def test_a_new_registry_outside_core_registries_fails(tree: Path) -> None:
    _append(
        tree,
        "splitters/base.py",
        'from lizyml.core.registries import _Registry\n_Registry("splitter")',
    )
    assert any(
        p.startswith("R6 lizyml/splitters/base.py:")
        and "creates a registry outside core/registries.py" in p
        for p in _check(tree)
    )


R6_PASSES = {
    "a Layer 0 class from Layer 2": (
        "tuning/search_space.py",
        "from lizyml.core.types.search_dim import FloatDim\nisinstance(x, FloatDim)",
    ),
    "an interface class from Layer 3": (
        "persistence/loader.py",
        "from lizyml.estimators.base import BaseEstimatorAdapter\n"
        "isinstance(x, BaseEstimatorAdapter)",
    ),
    "a concrete class from the Facade": (
        "core/seed.py",
        _ADAPTER + "isinstance(x, LGBMAdapter)",
    ),
    "builtin and foreign types": (
        "estimators/base.py",
        "import numpy as np\nisinstance(x, (int, np.ndarray)) or type(x) is type(None)",
    ),
    "registry register and keys": (
        "splitters/base.py",
        _REGISTRY + "MetricRegistry.keys()\nMetricRegistry.register",
    ),
}


@pytest.mark.parametrize("case", sorted(R6_PASSES))
def test_r6_passes_what_the_rule_allows(tree: Path, case: str) -> None:
    module, text = R6_PASSES[case]
    _append(tree, module, text)
    assert _check(tree) == []
