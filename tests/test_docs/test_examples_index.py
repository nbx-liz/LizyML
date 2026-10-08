"""``docs/examples.md`` describes the notebooks it lists (DC3).

The index named methods its notebooks never called and listed the wrong extras
for seven of eight notebooks. These tests parse each notebook's code cells with
``ast`` (comments and strings are not calls) and check the index against them:
the listed notebooks are exactly the notebooks on disk, every method the index
names for a notebook is called in it, and the extras it lists are exactly the
ones the notebook's imports and calls need. Anything the derivation cannot
decide -- a non-literal ``kind=``, an import outside the standard library, the
base install and the extras -- fails the test instead of passing it.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs" / "examples.md"
NOTEBOOKS = ROOT / "notebooks"

#: Top-level modules the base install provides (``pyproject.toml``
#: ``dependencies``, and scipy through scikit-learn).
BASE_MODULES = frozenset(
    {
        "lizyml",
        "numpy",
        "pandas",
        "sklearn",
        "lightgbm",
        "yaml",
        "joblib",
        "pydantic",
        "scipy",
    }
)
#: Extra name for each top-level module an extra provides.
EXTRA_MODULES = {"plotly": "plots", "shap": "explain", "optuna": "tuning"}


def _sections() -> dict[str, str]:
    parts = re.split(
        r"^### `(tutorial_\w+\.ipynb)`\n", INDEX.read_text(encoding="utf-8"), flags=re.M
    )
    names, bodies = parts[1::2], parts[2::2]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    assert not duplicated, f"notebooks listed twice: {duplicated}"
    return {
        name: body.split("\n---")[0] for name, body in zip(names, bodies, strict=True)
    }


def _tree(name: str) -> ast.Module:
    nb = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    body: list[ast.stmt] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        # IPython magics and shell escapes are not Python.
        source = "\n".join(
            "" if line.lstrip().startswith(("%", "!")) else line
            for line in source.splitlines()
        )
        body.extend(ast.parse(source, filename=name).body)
    return ast.Module(body=body, type_ignores=[])


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _is_model_call(node: ast.AST) -> bool:
    """``Model(...)`` or ``Model.load(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "Model") or (
        isinstance(func, ast.Attribute)
        and func.attr == "load"
        and isinstance(func.value, ast.Name)
        and func.value.id == "Model"
    )


def _outside_scopes(node: ast.AST) -> list[ast.AST]:
    """``node`` and its descendants, not descending into a function or class body."""
    found = [node]
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SCOPES):
            continue
        found.extend(_outside_scopes(child))
    return found


def _model_calls(name: str, tree: ast.Module) -> set[str]:
    """Methods called on a LizyML model, read in execution order.

    A name counts as a model from the top-level ``x = Model(...)`` (or
    ``Model.load(...)``) that binds it, for the statements after it. Calls
    inside a function, lambda or class body are not counted: which object their
    receiver is cannot be told from the text. Anything that would make the
    receiver undecidable fails instead of passing: a ``Model(...)`` bound
    anywhere but a top-level assignment to plain names, and a model name that is
    rebound, deleted, imported over or declared ``global``.
    """
    bound: set[str] = set()
    called: set[str] = set()
    for statement in tree.body:
        nodes = _outside_scopes(statement)
        for node in nodes:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in bound
            ):
                called.add(node.func.attr)
        fresh = (
            isinstance(statement, ast.Assign)
            and _is_model_call(statement.value)
            and all(isinstance(t, ast.Name) for t in statement.targets)
        )
        for node in ast.walk(statement):
            if _is_model_call(node) and not fresh:
                parent_assigns = [
                    n
                    for n in ast.walk(statement)
                    if isinstance(n, ast.Assign | ast.AnnAssign | ast.NamedExpr)
                    and getattr(n, "value", None) is node
                ]
                assert not parent_assigns, (
                    f"{name}: a model is bound outside a top-level `x = Model(...)`; "
                    "the receiver check cannot follow it"
                )
            if isinstance(node, ast.Global):
                rebound = sorted(set(node.names) & bound)
                assert not rebound, f"{name}: model name declared global: {rebound}"
        targets = (
            {t.id for t in statement.targets if isinstance(t, ast.Name)}
            if fresh
            else set()
        )
        for node in nodes:
            stored = None
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
                stored = node.id
            elif isinstance(node, ast.alias):
                stored = (node.asname or node.name).split(".")[0]
            if stored in bound and stored not in targets:
                raise AssertionError(f"{name}: model name {stored!r} is rebound")
        bound |= targets
    assert bound, f"{name}: binds no `Model(...)`; the receiver check has no anchor"
    return called


def _literal(
    name: str, call: ast.Call, keyword: str, position: int | None = None
) -> object:
    """The literal value passed as ``keyword`` (or at ``position``), else None.

    A value that is not a literal, or a ``**`` mapping that could carry the
    keyword, fails: the extras it implies cannot be decided.
    """
    for kw in call.keywords:
        assert kw.arg is not None, (
            f"{name}: a `**` argument may carry `{keyword}=`; pass it literally"
        )
    if position is not None and len(call.args) > position:
        arg = call.args[position]
        assert isinstance(arg, ast.Constant), (
            f"{name}: positional `{keyword}` is not a literal; write it literally"
        )
        return arg.value
    for kw in call.keywords:
        if kw.arg == keyword:
            assert isinstance(kw.value, ast.Constant), (
                f"{name}: `{keyword}=` is not a literal, so the extras it needs "
                "cannot be decided; write the value literally"
            )
            return kw.value.value
    return None


def _needed_extras(name: str, tree: ast.Module) -> set[str]:
    needed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            modules = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in modules:
                top = module.split(".")[0]
                if top in EXTRA_MODULES:
                    needed.add(EXTRA_MODULES[top])
                else:
                    assert (
                        top in BASE_MODULES
                        or top in sys.stdlib_module_names
                        or top == "__future__"
                    ), f"{name}: imports {top!r}, which no install declares"
        elif isinstance(node, ast.Call):
            func = node.func
            called = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if "plot" in called:
                needed.add("plots")
            if called == "tune":
                needed.add("tuning")
            # `importance(kind)` and `importance_plot(kind)` take `kind` first.
            position = 0 if called in {"importance", "importance_plot"} else None
            if _literal(name, node, "kind", position) == "shap":
                needed.add("explain")
            if _literal(name, node, "return_shap") is True:
                needed.add("explain")
        elif isinstance(node, ast.Constant) and node.value == "pytest":
            raise AssertionError(f"{name}: runs pytest, a dev-only dependency")
    return needed


def _listed_extras(body: str) -> set[str]:
    line = re.search(r"\*\*Extras required:\*\* (.*)", body)
    assert line, "a notebook entry has no 'Extras required' line"
    if line.group(1).startswith("none"):
        return set()
    extras = re.search(r"lizyml\[([a-z,]+)\]", line.group(1))
    assert extras, f"unparseable extras line: {line.group(1)!r}"
    return set(extras.group(1).split(","))


SECTIONS = _sections()


def test_the_index_lists_exactly_the_notebooks_on_disk() -> None:
    on_disk = {p.name for p in NOTEBOOKS.glob("tutorial_*.ipynb")}
    assert on_disk, "no notebooks found; the glob lost its anchor"
    assert set(SECTIONS) == on_disk


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_every_method_the_index_names_is_called_in_the_notebook(name: str) -> None:
    named = set(re.findall(r"`(\w+)\(", SECTIONS[name]))
    assert named, f"{name}: the entry names no method"
    missing = sorted(named - _model_calls(name, _tree(name)))
    assert not missing, f"{name}: named but never called on the model: {missing}"


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_the_listed_extras_are_the_ones_the_notebook_needs(name: str) -> None:
    assert _listed_extras(SECTIONS[name]) == _needed_extras(name, _tree(name))
