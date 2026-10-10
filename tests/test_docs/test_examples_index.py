"""The notebook index is a closed contract (H-0119, #334).

``docs/examples.md`` promises two things per notebook: the ``Model`` methods it
demonstrates and the extras it needs. Each notebook declares both in
``metadata.lizyml.index``; its ``index-example`` cells hold the examples in a
closed grammar; the method-to-extra registry (``lizyml/_extras.py``) derives
the extras from them; and ``scripts/examples_index.py`` generates the region
at the top of ``docs/examples.md``. These tests call the same functions as
``scripts/examples_index.py --check``.

This replaces PR #335's static ``ast`` reading of whole notebooks, whose review
kept finding forms it could not decide. The counterexamples from that review
are replayed below (static) and in ``tests/test_notebooks/test_index_recording.py``
(runtime recording); each must fail one of the two.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lizyml_examples_index", ROOT / "scripts" / "examples_index.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


ix = _load()
SURFACE = ix.model_surface()

INDEX = {
    "models": ["model"],
    "methods": ["fit", "importance_plot"],
    "extras": ["explain", "plots"],
}


def _code(source: str, *, tagged: bool = True, tags: Any = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if tags is not None:
        metadata["tags"] = tags
    elif tagged:
        metadata["tags"] = [ix.TAG]
    return {
        "cell_type": "code",
        "metadata": metadata,
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def _nb(*cells: dict[str, Any], index: Any = None) -> dict[str, Any]:
    return {
        "cells": list(cells),
        "metadata": {
            "lizyml": {"index": copy.deepcopy(INDEX if index is None else index)}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


GOOD = _nb(
    _code("model = Model(config)", tagged=False),
    _code("model.fit(data=df)"),
    _code('fig = model.importance_plot(kind="shap")'),
    _code("fig.show()", tagged=False),
)


def _declaration(index: Any) -> Any:
    return ix.parse_declaration({"lizyml": {"index": index}}, SURFACE)


def _fails(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ix.ContractError, match=match):
        call()


# --- 1. The declaration (H-0119 section 2) -----------------------------------


def test_a_valid_declaration_passes() -> None:
    declaration = _declaration(INDEX)
    assert declaration.models == ("model",)
    assert declaration.methods == ("fit", "importance_plot")
    assert declaration.extras == ("explain", "plots")


def test_empty_extras_pass() -> None:
    assert _declaration({**INDEX, "extras": []}).extras == ()


def test_other_keys_beside_the_index_pass() -> None:
    """Only ``metadata.lizyml.index`` is closed (review round 1, finding 1)."""
    metadata = {
        "kernelspec": {"name": "python3"},
        "lizyml": {"index": INDEX, "other": 1, "notes": {"any": ["thing"]}},
    }
    assert ix.parse_declaration(metadata, SURFACE) == _declaration(INDEX)


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({}, "no metadata.lizyml"),
        ({"lizyml": []}, "metadata.lizyml must be an object"),
        ({"lizyml": {"other": 1}}, "no metadata.lizyml.index"),
        ({"lizyml": {"index": [INDEX]}}, "must be a JSON object"),
    ],
)
def test_the_metadata_container_is_closed(metadata: Any, match: str) -> None:
    _fails(lambda: ix.parse_declaration(metadata, SURFACE), match)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        # Missing and extra keys.
        ({"models": None}, "exactly the keys"),
        ({"note": ["x"]}, "exactly the keys"),
        # Types.
        ({"methods": "fit"}, "array of strings"),
        ({"models": [1]}, "array of strings"),
        ({"extras": [["plots"]]}, "array of strings"),
        # Duplicates and order.
        ({"methods": ["fit", "fit"]}, "duplicate"),
        ({"methods": ["importance_plot", "fit"]}, "sorted"),
        ({"extras": ["plots", "explain"]}, "sorted"),
        # Empty.
        ({"models": []}, "must not be empty"),
        ({"methods": []}, "must not be empty"),
        # Identifiers.
        ({"models": ["1model"]}, "not a Python identifier"),
        ({"models": ["my-model"]}, "not a Python identifier"),
        ({"models": ["class"]}, "keyword"),
        # Methods.
        ({"methods": ["_get_fit_state"]}, "public"),
        ({"methods": ["train"]}, "not a public Model method"),
        ({"methods": ["fit_result"]}, "property, not an instance method"),
        ({"methods": ["load"]}, "classmethod, not an instance method"),
        # Extras.
        ({"extras": ["calibration"]}, "unknown extra"),
        ({"extras": ["shap"]}, "unknown extra"),
    ],
)
def test_each_declaration_rule_fails_when_broken(
    change: dict[str, Any], match: str
) -> None:
    index = {**INDEX, **change}
    index = {k: v for k, v in index.items() if v is not None}
    _fails(lambda: _declaration(index), match)


# --- 2. The tagged-cell grammar (H-0119 section 3) ---------------------------

PERMISSIVE = {
    "models": ["model"],
    "methods": sorted(SURFACE.methods),
    "extras": [],
}


def _statements(*cells: dict[str, Any], index: Any = None) -> list[Any]:
    nb = _nb(*cells, index=PERMISSIVE if index is None else index)
    return ix.tagged_statements(nb, _declaration(nb["metadata"]["lizyml"]["index"]))


@pytest.mark.parametrize(
    "source",
    [
        # The two statements.
        "model.fit(data=df)",
        "result = model.predict(X_new)",
        # Values: constants.
        "model.fit(data=None)",
        'model.importance("gain")',
        "model.confusion_matrix(0.25)",
        # Positional `kind` at position 0, for importance and importance_plot.
        'model.importance_plot("shap", 5)',
        'model.importance("shap")',
        # Values: names, attribute chains, subscripts.
        "model.fit(df)",
        "model.predict(data.frames.test)",
        'model.fit(data=frames["train"])',
        "model.fit(frames[0])",
        "model.fit(frames[key.name])",
        'model.fit(frames["a"][0])',
        "model.fit(frames.parts[0])",
        # Values: unary minus of a constant.
        "model.importance_plot(top_n=-1)",
        "model.confusion_matrix(threshold=-0.5)",
        # Values: list / tuple / set / dict of values.
        'model.plot_learning_curve(metrics=["rmse", name])',
        "model.fit(data=(a, 1))",
        "model.fit(data={1, b.c})",
        'model.fit(params={"num_leaves": 7, key: [1, -2]})',
        "model.evaluate(metrics=[])",
        # Comments and several statements in one cell.
        "# fit, then explain\nmodel.fit(df)\nfig = model.importance_plot(kind='shap')",
        # A statement spanning lines.
        "model.fit(\n    data=df,\n)",
    ],
)
def test_each_allowed_form_passes(source: str) -> None:
    assert _statements(_code(source))


@pytest.mark.parametrize(
    ("source", "match"),
    [
        # Other statements.
        ("import lizyml", "statement"),
        ("x = 1", "statement"),
        ("pass", "statement"),
        ("if True:\n    model.fit(df)", "statement"),
        ("for _ in [1]:\n    model.fit(df)", "statement"),
        ("while False:\n    model.fit(df)", "statement"),
        ("try:\n    model.fit(df)\nexcept Exception:\n    pass", "statement"),
        ("with ctx:\n    model.fit(df)", "statement"),
        ("def f():\n    model.fit(df)", "statement"),
        ("class C:\n    x = model.fit(df)", "statement"),
        ("del df", "statement"),
        ("n += model.fit(df)", "statement"),
        ("n: int = model.fit(df)", "statement"),
        ("a = b = model.fit(df)", "exactly one target"),
        ("a, b = model.fit(df)", "simple name"),
        ("x.y = model.fit(df)", "simple name"),
        ("x[0] = model.fit(df)", "simple name"),
        # Other expressions as the statement.
        ("model.evaluate_table().round(4)", "R.m"),
        ("model.plot_learning_curve().show()", "R.m"),
        ("print(model.fit(df))", "R.m"),
        ("model.fit", "R.m"),
        ("Model(config)", "R.m"),
        ("fit(df)", "R.m"),
        ("self.model.fit(df)", "R.m"),
        ('"model.fit(df)"', "R.m"),
        # Other expressions as values.
        ("model.fit(data=load())", "not a value"),
        ("model.fit(data=lambda: df)", "not a value"),
        ("model.fit(data=[d for d in dfs])", "not a value"),
        ("model.fit(data=a or b)", "not a value"),
        ("model.fit(data=a if c else b)", "not a value"),
        ("model.fit(data=(d := df))", "not a value"),
        ("model.fit(data=await df)", "not a value"),
        ("model.fit(data=a + b)", "not a value"),
        ("model.fit(data=not a)", "not a value"),
        ("model.fit(data=-a)", "not a value"),
        ("model.fit(data=a < b)", "not a value"),
        ('model.fit(data=f"{a}")', "not a value"),
        ("model.fit(data=a[1:2])", "not a value"),
        ("model.fit(data=load()[0])", "not a value"),
        ('model.fit(data=a["x"].b)', "not a value"),
        ("model.fit(data=[*a])", "not a value"),
        ("model.fit(params={**a})", "not a value"),
        # `*` / `**` in the call.
        ("model.fit(*args)", r"\*"),
        ("model.fit(**kwargs)", r"\*\*"),
        # The receiver, the method and the target.
        ("other.fit(df)", "not a declared model"),
        ("model.train(df)", "not a declared method"),
        ("model.fit_result(df)", "property, not an instance method"),
        # A classmethod is not an instance method (review round 1, finding 4).
        ("loaded = model.load(path)", "classmethod, not an instance method"),
        ("model = model.fit(df)", "names a declared model"),
        # The extras-related arguments are constants.
        ("model.importance(kind=k)", "constant"),
        ("model.importance(k)", "constant"),
        ("model.predict(X, return_shap=flag)", "constant"),
        ("model.importance_plot(kind=-1)", "constant"),
        # The call matches the method's signature.
        ("model.fit(1, 2, 3)", "signature"),
        ("model.predict(X, True)", "signature"),
        ("model.fit(dat=df)", "signature"),
        ('model.importance_plot("split", kind="shap")', "signature"),
        # Magic lines and empty cells.
        ("%time model.fit(df)", "magic"),
        ("!ls\nmodel.fit(df)", "magic"),
        ("model.fit(df)\n  %matplotlib inline", "magic"),
        ("", "no statement"),
        ("# model.fit(df)", "no statement"),
        # Not Python.
        ("model.fit(", "does not parse"),
        # `ast.parse` accepts a repeated keyword (only `compile` refuses it, on
        # 3.11-3.13); binding it as a dict would keep the last value silently.
        ('model.importance(kind="split", kind="gain")', "keyword argument repeated"),
    ],
)
def test_each_rejected_form_fails(source: str, match: str) -> None:
    _fails(lambda: _statements(_code(source)), match)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('model.importance(kind="SHAP")', set()),
        ("model.importance(kind=None)", set()),
        ('model.importance("shap")', {"explain"}),
        ("model.predict(X, return_shap=1)", {"explain"}),
        ('model.predict(X, return_shap="")', set()),
        ('model.importance_plot(kind="SHAP")', {"plots"}),
        ("model.importance_plot(kind=True)", {"plots"}),
    ],
)
def test_conditions_follow_the_runtime_predicates(
    source: str, expected: set[str]
) -> None:
    (statement,) = _statements(_code(source))
    derived = ix.REGISTRY.extras_for(statement.method, statement.conditions)
    assert derived == frozenset(expected)


def test_a_repeated_keyword_fails_in_the_binder() -> None:
    """The binder itself refuses a repeated keyword (review round 1, finding 2)."""
    call = ast.Call(
        func=ast.Attribute(ast.Name("model"), "importance"),
        args=[],
        keywords=[
            ast.keyword("kind", ast.Constant("split")),
            ast.keyword("kind", ast.Constant("shap")),
        ],
    )
    signature = SURFACE.signatures["importance"]
    _fails(lambda: ix.bind_call(call, "importance", signature, "here"), "repeated")


def test_the_tags_of_a_cell_are_a_list_of_strings() -> None:
    _fails(lambda: _statements(_code("model.fit(df)", tags="index-example")), "tags")
    _fails(lambda: _statements(_code("model.fit(df)", tags=[1])), "tags")


def test_a_tagged_cell_is_a_code_cell() -> None:
    cell = {"cell_type": "markdown", "metadata": {"tags": [ix.TAG]}, "source": "x"}
    _fails(lambda: _statements(cell), "code cell")


def test_an_untagged_cell_with_other_tags_and_metadata_passes() -> None:
    other = _code("print(model)", tags=["hide-input"])
    other["metadata"].update({"collapsed": True, "jupyter": {"source_hidden": True}})
    statements = _statements(other, _code("model.fit(df)"))
    assert [s.method for s in statements] == ["fit"]


def test_a_cell_whose_tags_are_undecidable_fails_even_untagged() -> None:
    """Whether such a cell is tagged cannot be decided, so it fails."""
    bad = _code("print(model)", tagged=False)
    bad["metadata"] = ["tags"]
    _fails(lambda: _statements(bad, _code("model.fit(df)")), "metadata")
    _fails(
        lambda: _statements(_code("x", tags="hide-input"), _code("model.fit(df)")),
        "tags",
    )


def test_untagged_cells_are_not_constrained() -> None:
    statements = _statements(
        _code("%time x = [model.tune() for _ in []]\nprint(1)", tagged=False),
        _code("model.fit(df)"),
    )
    assert [(s.receiver, s.method) for s in statements] == [("model", "fit")]


def test_a_cell_source_may_be_a_list_of_lines() -> None:
    cell = _code("")
    cell["source"] = ["model.fit(df)\n", "model.evaluate_table()"]
    assert [s.method for s in _statements(cell)] == ["fit", "evaluate_table"]


def test_statements_carry_their_conditions_and_source() -> None:
    (first, second) = _statements(
        _code('fig = model.importance_plot("shap", top_n=3)\nmodel.fit(df)')
    )
    assert first.conditions == {"kind": "shap"}
    assert first.target == "fig"
    assert first.source == 'fig = model.importance_plot("shap", top_n=3)'
    assert second.conditions == {}
    assert (second.cell, second.number) == (0, 1)


# --- The notebook as a whole (sections 2-4) ----------------------------------


def test_a_valid_notebook_passes() -> None:
    assert ix.check_notebook(GOOD, SURFACE) == _declaration(INDEX)


def _good_with(**changes: Any) -> dict[str, Any]:
    nb = copy.deepcopy(GOOD)
    nb["metadata"]["lizyml"]["index"].update(changes)
    return nb


@pytest.mark.parametrize(
    ("nb", "match"),
    [
        (_good_with(methods=["fit", "importance_plot", "predict"]), "methods"),
        (_good_with(methods=["fit"]), "not a declared method"),
        (_good_with(models=["fig", "model"]), "names a declared model"),
        (_good_with(models=["model", "other"]), "models"),
        (_good_with(extras=["plots"]), "extras"),
        (_good_with(extras=["explain", "plots", "tuning"]), "extras"),
        (_nb(_code("model.fit(df)", tagged=False)), "no index-example cell"),
    ],
)
def test_the_notebook_must_match_its_declaration(
    nb: dict[str, Any], match: str
) -> None:
    _fails(lambda: ix.check_notebook(nb, SURFACE), match)


def test_extras_are_derived_with_defaults() -> None:
    nb = _nb(
        _code("model.importance_plot()\nmodel.predict(X)\nmodel.importance()"),
        index={
            "models": ["model"],
            "methods": ["importance", "importance_plot", "predict"],
            "extras": ["plots"],
        },
    )
    assert ix.check_notebook(nb, SURFACE).extras == ("plots",)


# --- 3. Replays of the counterexamples PR #335's review found ----------------
# Each of these passed (or could pass) the replaced `ast` check. Runtime
# replays (a receiver rebound outside the tagged cells, a look-alike receiver)
# are in tests/test_notebooks/test_index_recording.py.

REPLAYS = {
    "a call after `and`": "model.fit(df) and model.importance_plot(kind='shap')",
    "a call after `or`": "model.fit(df) or model.importance_plot(kind='shap')",
    "an empty comprehension": "[model.importance_plot(kind='shap') for _ in []]",
    "a conditional expression": "model.importance_plot(kind='shap') if False else 0",
    "an unreachable branch": "if False:\n    model.importance_plot(kind='shap')",
    "a reassignment in the tagged cell": "model = Model(config)",
    "a `def` rebinding the model": "def model():\n    pass",
    "a `class` rebinding the model": "class model:\n    pass",
    "a magic line": "%time model.fit(df)",
    "a call only in a comment": "# model.importance_plot(kind='shap')",
    "a look-alike receiver": "fake.importance_plot(kind='shap')",
}


@pytest.mark.parametrize("case", sorted(REPLAYS))
def test_counterexample_replays_fail(case: str) -> None:
    nb = _nb(_code("model.fit(df)"), _code(REPLAYS[case]))
    with pytest.raises(ix.ContractError):
        ix.check_notebook(nb, SURFACE)


def test_a_call_only_in_a_comment_beside_a_real_one_fails() -> None:
    nb = _nb(_code("model.fit(df)\n# fig = model.importance_plot(kind='shap')"))
    _fails(lambda: ix.check_notebook(nb, SURFACE), "methods")


# --- 4. docs/examples.md (section 5) ------------------------------------------
#
# The generated region is the first K = N + 8 lines of docs/examples.md. The
# checker compares physical lines only; it reads nothing after the region and
# promises nothing about how the file renders (H-0119 decision 1).


def _decl(**index: Any) -> Any:
    return _declaration({**INDEX, **index})


DECLS = {
    "a.ipynb": _decl(),
    "b.ipynb": _decl(methods=["fit"], extras=[]),
}
ROWS = [
    "| `a.ipynb` | `fit()`, `importance_plot()` "
    "| `pip install 'lizyml[explain,plots]'` |",
    "| `b.ipynb` | `fit()` | none (base install) |",
]
SUFFIX = "\n## Notes\n\nHandwritten text.\n"


def _comment(rows: list[str]) -> str:
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return (
        "<!-- Generated from each notebook's metadata.lizyml.index by "
        "scripts/examples_index.py. Do not edit this region by hand. "
        f"rows={len(rows)} sha256={digest} -->"
    )


REGION = [
    "# Notebook Index",
    "",
    _comment(ROWS),
    "",
    "| Notebook | Demonstrates | Extras required |",
    "|---|---|---|",
    *ROWS,
    "",
    "<!-- index:end -->",
]
DOC = "\n".join(REGION) + "\n" + SUFFIX


def _errors(text: str, decls: Any = None) -> str:
    errors = ix.check_index(text, DECLS if decls is None else decls)
    assert errors, "the mutation was not detected"
    return "\n".join(errors)


def test_render_index() -> None:
    assert ix.render_index(DECLS) == REGION


def test_the_digest_matches_the_known_vector() -> None:
    rows = [
        "| `a.ipynb` | `fit()` | none (base install) |",
        "| `b.ipynb` | `predict()` | café |",
    ]
    assert ix.rows_digest(rows) == (
        "9fa91b360618931dd5717f9cf60c166f0058cc92734b624f9c49800d3676ab92"
    )


def test_rows_follow_code_point_order() -> None:
    decls = {name: _decl() for name in ["b.ipynb", "a.ipynb", "_x.ipynb", "B.ipynb"]}
    names = [row.split("`")[1] for row in ix.render_index(decls)[6:-2]]
    assert names == ["B.ipynb", "_x.ipynb", "a.ipynb", "b.ipynb"]


def test_a_matching_index_passes() -> None:
    assert ix.check_index(DOC, DECLS) == []


def _region_mutations() -> dict[str, list[str]]:
    """Every way to delete, add, reorder or change one line of the region."""
    cases: dict[str, list[str]] = {}
    for i in range(len(REGION)):
        cases[f"delete line {i + 1}"] = REGION[:i] + REGION[i + 1 :]
        cases[f"change line {i + 1}"] = REGION[:i] + [REGION[i] + "x"] + REGION[i + 1 :]
        cases[f"add a line before {i + 1}"] = REGION[:i] + [""] + REGION[i:]
    cases["swap the rows"] = REGION[:6] + [REGION[7], REGION[6]] + REGION[8:]
    return cases


@pytest.mark.parametrize("case", sorted(_region_mutations()))
def test_a_changed_region_line_fails(case: str) -> None:
    _errors("\n".join(_region_mutations()[case]) + "\n" + SUFFIX)


def test_the_end_marker_only_in_the_handwritten_part_fails() -> None:
    region = REGION[:-1]
    _errors("\n".join(region) + "\n" + SUFFIX + "\n<!-- index:end -->\n")


@pytest.mark.parametrize(
    "decls",
    [
        {**DECLS, "a.ipynb": _decl(methods=["fit"])},
        {**DECLS, "c.ipynb": _decl()},
        {"a.ipynb": DECLS["a.ipynb"]},
    ],
    ids=["changed declaration", "added notebook", "removed notebook"],
)
def test_a_declaration_or_census_change_fails(decls: Any) -> None:
    _errors(DOC, decls)


@pytest.mark.parametrize("name", ["café.ipynb", "a b.ipynb", "a+b.ipynb", "a|b.ipynb"])
def test_a_notebook_name_outside_the_name_characters_fails(name: str) -> None:
    decls = {**DECLS, name: _decl()}
    assert "name" in _errors(DOC, decls)
    with pytest.raises(ix.ContractError, match="name"):
        ix.rewrite_index(DOC, decls)


def test_no_notebook_fails() -> None:
    assert "no notebooks" in _errors(DOC, {})
    with pytest.raises(ix.ContractError, match="no notebooks"):
        ix.rewrite_index(DOC, {})


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "```\nunclosed fence\n",
        "<!-- index:end -->\n| `x.ipynb` | `y()` | z |\n",
        " \n---\n### `ghost.ipynb`\n",
        "\r\n\r\n",
    ],
)
def test_nothing_after_the_region_is_read(suffix: str) -> None:
    assert ix.check_index("\n".join(REGION) + "\n" + suffix, DECLS) == []


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"], ids=["LF", "CRLF", "CR"])
def test_every_line_ending_reads_the_same(ending: str) -> None:
    assert ix.check_index(DOC.replace("\n", ending), DECLS) == []
    assert ix.check_index(DOC.replace("\n", ending), {"a.ipynb": DECLS["a.ipynb"]})


# --- --write (section 5) -------------------------------------------------------

STALE_DECLS = {**DECLS, "b.ipynb": _decl(methods=["fit", "predict"], extras=[])}


@pytest.mark.parametrize(
    "decls",
    [STALE_DECLS, {**DECLS, "c.ipynb": _decl()}, {"a.ipynb": DECLS["a.ipynb"]}],
    ids=["same count", "added notebook", "removed notebook"],
)
def test_write_repairs_an_intact_old_region(decls: Any) -> None:
    rewritten = ix.rewrite_index(DOC, decls)
    assert rewritten == "\n".join(ix.render_index(decls)) + "\n" + SUFFIX
    assert ix.check_index(rewritten, decls) == []


@pytest.mark.parametrize(
    "suffix",
    [
        "| `x.ipynb` | `y()` | z |\n\n<!-- index:end -->\n",
        "\n\n<!-- index:end -->\n<!-- index:end -->\r\n| row\r",
        "",
    ],
)
def test_write_keeps_everything_after_the_old_region(suffix: str) -> None:
    rewritten = ix.rewrite_index("\n".join(REGION) + "\n" + suffix, STALE_DECLS)
    assert rewritten == "\n".join(ix.render_index(STALE_DECLS)) + "\n" + suffix


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"], ids=["LF", "CRLF", "CR"])
def test_write_uses_the_first_line_ending_for_the_region(ending: str) -> None:
    mixed = ending.join(REGION) + ending + "a\nb\r\nc\rd"
    rewritten = ix.rewrite_index(mixed, STALE_DECLS)
    region = ending.join(ix.render_index(STALE_DECLS)) + ending
    assert rewritten == region + "a\nb\r\nc\rd"


# The base file of acceptance criterion 4: a two-row old region whose rows are
# the known digest vector, then a blank line and a handwritten heading.
VECTOR_ROWS = [
    "| `a.ipynb` | `fit()` | none (base install) |",
    "| `b.ipynb` | `predict()` | café |",
]
VECTOR_D = "9fa91b360618931dd5717f9cf60c166f0058cc92734b624f9c49800d3676ab92"
COUNTER_ROW = "| `c.ipynb` | `x()` | none (base install) |"


def _base() -> list[str]:
    return [
        "# Notebook Index",
        "",
        _comment(VECTOR_ROWS),
        "",
        "| Notebook | Demonstrates | Extras required |",
        "|---|---|---|",
        *VECTOR_ROWS,
        "",
        "<!-- index:end -->",
        "",
        "## Notes",
    ]


def _header(
    lines: list[str], rows: str | None = None, digest: str | None = None
) -> None:
    head, _, tail = lines[2].partition(" rows=")
    old_rows, _, rest = tail.partition(" sha256=")
    old_digest = rest.removesuffix(" -->")
    lines[2] = (
        f"{head} rows={old_rows if rows is None else rows} "
        f"sha256={old_digest if digest is None else digest} -->"
    )


def _counterexample(lines: list[str]) -> None:
    lines[8:10] = [COUNTER_ROW, "", "<!-- index:end -->"]


def _row_without_prefix(lines: list[str]) -> None:
    lines[7] = "x" + lines[7]
    _header(lines, digest=ix.rows_digest(lines[6:8]))


# Acceptance criterion 4: each mutation of the base file and the one reason
# --write must report for it.
REFUSALS: dict[str, tuple[Callable[[list[str]], object], str]] = {
    "rows=0 with the empty digest": (
        lambda lines: _header(lines, rows="0", digest=hashlib.sha256(b"").hexdigest()),
        "header-format",
    ),
    "rows=02": (lambda lines: _header(lines, rows="02"), "header-format"),
    "uppercase D": (
        lambda lines: _header(lines, digest=VECTOR_D.upper()),
        "header-format",
    ),
    "63-character D": (
        lambda lines: _header(lines, digest=VECTOR_D[:-1]),
        "header-format",
    ),
    "non-hex D": (
        lambda lines: _header(lines, digest="g" + VECTOR_D[1:]),
        "header-format",
    ),
    "changed wrapper text": (
        lambda lines: lines.__setitem__(2, lines[2].replace("Do not edit", "Do edit")),
        "header-format",
    ),
    "changed fixed line 5": (
        lambda lines: lines.__setitem__(4, lines[4].replace("Demonstrates", "Methods")),
        "fixed-lines",
    ),
    "row without the prefix, D recomputed": (_row_without_prefix, "row-prefix"),
    "N=3": (lambda lines: _header(lines, rows="3"), "row-prefix"),
    "N=1": (lambda lines: _header(lines, rows="1"), "boundary"),
    "deleted blank line": (lambda lines: lines.__delitem__(8), "boundary"),
    "changed end marker": (
        lambda lines: lines.__setitem__(9, "<!-- end -->"),
        "boundary",
    ),
    "review counterexample": (_counterexample, "boundary"),
    "review counterexample with N=3": (
        lambda lines: (_counterexample(lines), _header(lines, rows="3")),
        "digest-mismatch",
    ),
    "another valid D": (
        lambda lines: _header(lines, digest="0" * 64),
        "digest-mismatch",
    ),
    "changed row, same D": (
        lambda lines: lines.__setitem__(6, lines[6].replace("fit()", "fit2()")),
        "digest-mismatch",
    ),
}


def test_every_refusal_reason_has_a_case() -> None:
    assert sorted({reason for _, reason in REFUSALS.values()}) == sorted(ix.REASONS)


def test_write_accepts_the_base_file() -> None:
    text = "\n".join(_base()) + "\n"
    assert ix.rewrite_index(text, DECLS) == "\n".join(REGION) + "\n\n## Notes\n"


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_write_refuses_with_one_reason(case: str) -> None:
    mutate, reason = REFUSALS[case]
    lines = _base()
    mutate(lines)
    with pytest.raises(ix.RegionError) as caught:
        ix.rewrite_index("\n".join(lines) + "\n", DECLS)
    assert caught.value.reason == reason
    assert "version control" in str(caught.value)


def test_a_recomputed_count_and_digest_redefine_the_region() -> None:
    # The integrity boundary: N and D recomputed together claim the
    # handwritten row as part of the region, and --write replaces it.
    lines = _base()
    _counterexample(lines)
    _header(lines, rows="3", digest=ix.rows_digest([*VECTOR_ROWS, COUNTER_ROW]))
    rewritten = ix.rewrite_index("\n".join(lines) + "\n", DECLS)
    assert rewritten == "\n".join(REGION) + "\n\n## Notes\n"


# --- The repository ------------------------------------------------------------


def test_the_repository_passes_the_check() -> None:
    assert ix.check(ROOT) == []


def test_every_notebook_on_disk_is_checked() -> None:
    on_disk = sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))
    assert on_disk, "no notebooks found; the glob lost its anchor"
    assert [p.name for p in ix.notebook_paths(ROOT)] == on_disk
    assert len(on_disk) == 8


def test_changing_one_notebook_declaration_fails_the_repository_check(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "examples.md").write_text(
        (ROOT / "docs" / "examples.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    for path in ix.notebook_paths(ROOT):
        (tmp_path / "notebooks" / path.name).write_bytes(path.read_bytes())
    assert ix.check(tmp_path) == []
    target = tmp_path / "notebooks" / "tutorial_calibration.ipynb"
    nb = json.loads(target.read_text(encoding="utf-8"))
    nb["metadata"]["lizyml"]["index"]["extras"] = []
    target.write_text(json.dumps(nb), encoding="utf-8")
    assert ix.check(tmp_path)


def _repo_copy(tmp_path: pathlib.Path, doc: bytes) -> pathlib.Path:
    """A copy of the notebooks with ``doc`` as docs/examples.md; returns its path."""
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "docs").mkdir()
    for path in ix.notebook_paths(ROOT):
        (tmp_path / "notebooks" / path.name).write_bytes(path.read_bytes())
    target = tmp_path / "docs" / "examples.md"
    target.write_bytes(doc)
    return target


def _stale(doc: bytes, ending: bytes) -> bytes:
    """``doc`` with an intact but outdated region: one row edited, D recomputed."""
    separator = ending.decode()
    lines = doc.decode("utf-8").split(separator)
    count = int(lines[2].split(" rows=")[1].split(" ")[0])
    lines[6] = lines[6].replace(" | `", " | `stale_", 1)
    head = lines[2].split(" sha256=")[0]
    lines[2] = f"{head} sha256={ix.rows_digest(lines[6 : 6 + count])} -->"
    return separator.join(lines).encode("utf-8")


def _left_alone(target: pathlib.Path, before: bytes) -> None:
    assert target.read_bytes() == before
    assert sorted(p.name for p in target.parent.iterdir()) == ["examples.md"]


def test_write_repairs_a_stale_region_and_keeps_the_file_line_endings(
    tmp_path: pathlib.Path,
) -> None:
    good = (ROOT / "docs" / "examples.md").read_bytes().replace(b"\n", b"\r\n")
    stale = _stale(good, b"\r\n")
    assert stale != good
    target = _repo_copy(tmp_path, stale)
    assert ix.check(tmp_path), "the stale region was not detected"
    ix.write(tmp_path)
    assert target.read_bytes() == good
    assert ix.check(tmp_path) == []


def test_write_leaves_a_refused_file_unchanged(tmp_path: pathlib.Path) -> None:
    broken = (
        (ROOT / "docs" / "examples.md")
        .read_bytes()
        .replace(b"<!-- index:end -->\n", b"", 1)
    )
    target = _repo_copy(tmp_path, broken)
    with pytest.raises(ix.RegionError):
        ix.write(tmp_path)
    _left_alone(target, broken)


def test_write_leaves_the_file_unchanged_when_writing_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = _stale((ROOT / "docs" / "examples.md").read_bytes(), b"\n")
    target = _repo_copy(tmp_path, stale)

    class Failing:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> Failing:
            return self

        def __exit__(self, *exc: object) -> None:
            self.handle.close()

        def write(self, text: str) -> int:
            self.handle.write(text[: len(text) // 2])
            raise OSError("disk full")

    monkeypatch.setattr(
        ix,
        "open",
        # The wrapper owns the handle and closes it in __exit__.
        lambda *a, **k: Failing(open(*a, **k)),  # noqa: SIM115
        raising=False,
    )
    with pytest.raises(OSError, match="disk full"):
        ix.write(tmp_path)
    _left_alone(target, stale)


def test_write_leaves_the_file_unchanged_when_the_replace_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = _stale((ROOT / "docs" / "examples.md").read_bytes(), b"\n")
    target = _repo_copy(tmp_path, stale)

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(ix.os, "replace", failing_replace)
    with pytest.raises(OSError, match="replace failed"):
        ix.write(tmp_path)
    _left_alone(target, stale)


def test_write_replaces_with_a_finished_sibling_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = (ROOT / "docs" / "examples.md").read_bytes()
    target = _repo_copy(tmp_path, _stale(good, b"\n"))
    calls: list[tuple[pathlib.Path, bytes, bool]] = []
    opened: list[Any] = []
    real_replace = os.replace

    def recording_open(*args: Any, **kwargs: Any) -> Any:
        handle = open(*args, **kwargs)  # noqa: SIM115 - the module closes it
        opened.append(handle)
        return handle

    def spy(src: str, dst: str) -> None:
        closed = bool(opened) and all(handle.closed for handle in opened)
        calls.append((pathlib.Path(src), pathlib.Path(src).read_bytes(), closed))
        real_replace(src, dst)

    monkeypatch.setattr(ix, "open", recording_open, raising=False)
    monkeypatch.setattr(ix.os, "replace", spy)
    ix.write(tmp_path)
    ((src, content, closed),) = calls
    assert src.parent == target.parent
    assert closed, "the temporary file was still open at the replace"
    assert content == good, "the temporary file was not finished before the replace"
    assert target.read_bytes() == good


def test_write_accepts_the_base_file_on_disk(tmp_path: pathlib.Path) -> None:
    target = _repo_copy(tmp_path, ("\n".join(_base()) + "\n").encode("utf-8"))
    ix.write(tmp_path)
    declarations, errors = ix._declarations(tmp_path)
    assert not errors
    region = "\n".join(ix.render_index(declarations))
    assert target.read_bytes() == (region + "\n\n## Notes\n").encode("utf-8")


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_write_refuses_on_disk_without_touching_the_file(
    tmp_path: pathlib.Path, case: str
) -> None:
    mutate, reason = REFUSALS[case]
    lines = _base()
    mutate(lines)
    before = ("\n".join(lines) + "\n").encode("utf-8")
    target = _repo_copy(tmp_path, before)
    with pytest.raises(ix.RegionError) as caught:
        ix.write(tmp_path)
    assert caught.value.reason == reason
    _left_alone(target, before)


def _bad_name(notebooks: pathlib.Path) -> None:
    first = sorted(notebooks.iterdir())[0]
    (notebooks / "café.ipynb").write_bytes(first.read_bytes())


def _bad_declaration(notebooks: pathlib.Path) -> None:
    target = notebooks / "tutorial_calibration.ipynb"
    nb = json.loads(target.read_text(encoding="utf-8"))
    nb["metadata"]["lizyml"]["index"]["extras"] = ["unknown"]
    target.write_text(json.dumps(nb), encoding="utf-8")


def _no_notebooks(notebooks: pathlib.Path) -> None:
    for path in notebooks.iterdir():
        path.unlink()


@pytest.mark.parametrize(
    "spoil",
    [_bad_name, _bad_declaration, _no_notebooks],
    ids=["name", "declaration", "no notebooks"],
)
def test_write_checks_the_notebooks_before_touching_the_file(
    tmp_path: pathlib.Path, spoil: Callable[[pathlib.Path], None]
) -> None:
    before = (ROOT / "docs" / "examples.md").read_bytes()
    target = _repo_copy(tmp_path, before)
    spoil(tmp_path / "notebooks")
    with pytest.raises(ix.ContractError):
        ix.write(tmp_path)
    _left_alone(target, before)


def test_no_notebook_name_selects_another_with_pytest_k() -> None:
    """CI selects one notebook with ``-k <stem>``; no stem may contain another."""
    stems = [p.stem for p in ix.notebook_paths(ROOT)]
    clashes = [(a, b) for a in stems for b in stems if a != b and a in b]
    assert not clashes


def test_ci_helpers_report_the_declared_extras() -> None:
    assert ix.uv_extra_flags_for(ROOT, "tutorial_codegen_export") == []
    assert ix.uv_extra_flags_for(ROOT, "tutorial_regression_tuning_lgbm") == [
        "--extra",
        "plots",
        "--extra",
        "tuning",
    ]
    assert ix.uv_extra_flags_without(ROOT, "plots") == [
        "--extra",
        "explain",
        "--extra",
        "tuning",
    ]
    with pytest.raises(ix.ContractError):
        ix.uv_extra_flags_without(ROOT, "calibration")
    with pytest.raises(ix.ContractError):
        ix.uv_extra_flags_for(ROOT, "tutorial_missing")
    matrix = ix.ci_matrix(ROOT)
    assert matrix["removed"] == ["explain", "plots", "tuning"]
    assert matrix["notebook"] == [p.stem for p in ix.notebook_paths(ROOT)]
