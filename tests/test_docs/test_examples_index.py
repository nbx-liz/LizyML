"""The notebook index is a closed contract (H-0119, #334).

``docs/examples.md`` promises two things per notebook: the ``Model`` methods it
demonstrates and the extras it needs. Each notebook declares both in
``metadata.lizyml.index``; its ``index-example`` cells hold the examples in a
closed grammar; the method-to-extra registry (``lizyml/_extras.py``) derives
the extras from them; and ``scripts/examples_index.py`` generates the index's
machine-readable blocks. These tests call the same functions as
``scripts/examples_index.py --check``.

This replaces PR #335's static ``ast`` reading of whole notebooks, whose review
kept finding forms it could not decide. The counterexamples from that review
are replayed below (static) and in ``tests/test_notebooks/test_index_recording.py``
(runtime recording); each must fail one of the two.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
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


def _decl(**index: Any) -> Any:
    return _declaration({**INDEX, **index})


DECLS = {
    "a.ipynb": _decl(),
    "b.ipynb": _decl(methods=["fit"], extras=[]),
}


def _block(name: str) -> str:
    return "\n".join(ix.render_block(name, DECLS[name]))


def _doc(*sections: str, tail: str = "") -> str:
    return "# Notebook Index\n\nIntro.\n\n" + "\n".join(sections) + tail


def _section(name: str, block: str | None = None) -> str:
    body = _block(name) if block is None else block
    return f"### `{name}`\n\nProse with `fit()`.\n\n{body}\n\n---\n"


DOC = _doc(
    _section("a.ipynb"), _section("b.ipynb"), tail="\n## Other\n\n```bash\n# x\n```\n"
)


def test_render_block() -> None:
    assert ix.render_block("a.ipynb", DECLS["a.ipynb"]) == [
        "<!-- index:begin a.ipynb -->",
        "**Demonstrates:** `fit()`, `importance_plot()`",
        "",
        "**Extras required:** `pip install 'lizyml[explain,plots]'`",
        "<!-- index:end -->",
    ]
    assert ix.render_block("b.ipynb", DECLS["b.ipynb"])[3] == (
        "**Extras required:** none (base install)"
    )


def test_a_matching_index_passes() -> None:
    assert ix.check_index(DOC, DECLS) == []


def _errors(text: str, decls: Any = None) -> str:
    errors = ix.check_index(text, DECLS if decls is None else decls)
    assert errors, "the mutation was not detected"
    return "\n".join(errors)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        # Headings.
        (_doc(_section("a.ipynb")), "missing"),
        (_doc(_section("a.ipynb"), _section("b.ipynb"), _section("b.ipynb")), "twice"),
        (DOC.replace("### `b.ipynb`", "## `b.ipynb`"), "heading"),
        (DOC.replace("### `b.ipynb`", "### b.ipynb"), "heading"),
        (DOC.replace("### `b.ipynb`", "### `b.ipynb` (new)"), "heading"),
        (DOC + "\n### `c.ipynb`\n", "not a notebook"),
        # Markers.
        (
            DOC.replace(
                "<!-- index:end -->", "<!-- index:end -->\n<!-- index:end -->", 1
            ),
            "end",
        ),
        (DOC.replace("<!-- index:end -->\n", "", 1), "not closed"),
        (
            DOC.replace("<!-- index:begin b.ipynb -->", "<!-- index:begin a.ipynb -->"),
            "section",
        ),
        (
            DOC.replace("<!-- index:begin b.ipynb -->", "<!--index:begin b.ipynb-->"),
            "marker",
        ),
        (
            _doc(
                _section("a.ipynb", block=_block("a.ipynb") + "\n" + _block("a.ipynb")),
                _section("b.ipynb"),
            ),
            "more than one",
        ),
        (
            _doc(_section("a.ipynb", block="no block"), _section("b.ipynb")),
            "no generated block",
        ),
        ("<!-- index:begin a.ipynb -->\n<!-- index:end -->\n" + DOC, "outside"),
        (DOC + "\n<!-- index:begin b.ipynb -->\n<!-- index:end -->\n", "outside"),
        # Content.
        (
            DOC.replace(
                "`pip install 'lizyml[explain,plots]'`", "`pip install 'lizyml[plots]'`"
            ),
            "differs",
        ),
        (
            DOC.replace(
                "**Demonstrates:** `fit()`\n\n", "**Demonstrates:** `fit()`\n", 1
            ),
            "differs",
        ),
    ],
)
def test_index_mutations_fail(text: str, match: str) -> None:
    assert match in _errors(text) or pytest.fail(_errors(text))


@pytest.mark.parametrize(
    "fence",
    [
        # An inner, shorter fence must not close the outer one (review round 2).
        "````md\n```\n{section}```\n````\n",
        "~~~~md\n~~~\n{section}~~~\n~~~~\n",
        # A fence of the other character never closes it either.
        "```md\n~~~\n{section}~~~\n```\n",
    ],
)
def test_a_section_inside_a_code_fence_is_not_a_section(fence: str) -> None:
    text = _doc(
        _section("a.ipynb"), tail="\n" + fence.format(section=_section("b.ipynb"))
    )
    assert "missing" in _errors(text)


def test_a_heading_inside_a_nested_fence_is_ignored() -> None:
    example = "\n````md\n```bash\n# x\n```\n### `c.ipynb`\n````\n"
    assert ix.check_index(DOC + example, DECLS) == []


@pytest.mark.parametrize(
    "extra",
    [
        # CommonMark renders each of these as a heading, so a parser that only
        # knows column-0 ATX headings would miss it (review run 2, round 1).
        " ### `a.ipynb`\n",
        "   ## Other\n",
        "> ### `a.ipynb`\n",
        "- ### `a.ipynb`\n",
        "1. ## Other\n",
        "missing.ipynb\n--------------\n",
        "Other\n=====\n",
        "Other\n-\n",
        "Other\n--\n",
        # Container continuations and quoted setext (review run 2, round 2).
        "- item\n\n    ### `ghost.ipynb`\n",
        "- > item\n    > ### `ghost.ipynb`\n",
        "> ghost.ipynb\n> ===\n",
        # HTML blocks, indented code and other shapes outside the line grammar.
        "<div>\n\n### `ghost.ipynb`\n\n</div>\n",
        "    ### `ghost.ipynb`\n",
        "| a | b |\n",
        " ```\n### `ghost.ipynb`\n```\n",
        "Text.\n---\n",
        "#hashtag\n",
        # Not blank: only spaces and tabs make a blank line (run 2, round 3).
        "  \n",
        "\t　\n",
        # Not a fence in CommonMark (a backtick in the info string), and a
        # line starting with three backticks or tildes is reserved for fences.
        "```bad`info\n",
        # Stated in H-0119 section 5 (review run 3, round 2): a line of only
        # hyphens other than a thematic break after a blank line, and a
        # thematic break that does not follow a blank line.
        "--\n",
        "--  \n",
        "Text.\n***\n",
        "Text.\n___\n",
    ],
)
def test_a_line_outside_the_index_grammar_fails(extra: str) -> None:
    assert "grammar" in _errors(DOC + "\n" + extra)


def test_an_indented_heading_before_a_block_fails() -> None:
    text = DOC.replace(
        "<!-- index:begin b.ipynb -->", " ## Other\n\n<!-- index:begin b.ipynb -->"
    )
    assert "grammar" in _errors(text)


def test_a_fence_closed_by_an_indented_closer_fails() -> None:
    # CommonMark closes the fence here, so the parser must not stay fenced.
    assert "column 0" in _errors(DOC + "\n```\ncode\n  ```\n")


@pytest.mark.parametrize(
    "prose",
    [
        "**Bold** text and `code()` and _emphasis_.\n",
        "[ref]: https://example.com\n",
        "2026 is a year.\n",
        "Text.\n\n***\n\n___\n",
        # CommonMark list markers use ASCII digits only (review run 3, round 1).
        "١. Arabic-Indic digit.\n",
        "１２. Fullwidth digits.\n",
    ],
)
def test_ordinary_prose_is_in_the_grammar(prose: str) -> None:
    assert ix.check_index(DOC + "\n" + prose, DECLS) == []


#: Every character Python treats as whitespace that CommonMark does not: only
#: a space and a tab make a line blank, indent it, or end a heading marker or
#: a fence closer (review run 2, round 3). A carriage return is excluded:
#: CommonMark reads it as a line ending (review run 3, round 1).
_NON_COMMONMARK_SPACES = [
    chr(c) for c in range(0x110000) if chr(c).isspace() and chr(c) not in " \t\n\r"
]


@pytest.mark.parametrize("ending", ["\r\n", "\r"], ids=["CRLF", "CR"])
def test_carriage_return_line_endings_read_as_line_feeds(ending: str) -> None:
    assert ix.check_index(DOC.replace("\n", ending), DECLS) == []
    # Each CommonMark reading: the fence closes, and "####" is its own line.
    closed = _doc(_section("a.ipynb"), "```\ncode\n```" + ending, _section("b.ipynb"))
    assert ix.check_index(closed, DECLS) == []
    assert ix.check_index(DOC + "\n#### x" + ending + "After.\n", DECLS) == []
    assert "grammar" in _errors(DOC + "\nText." + ending + "---\n")


_STALE = DOC.replace("`fit()`, `importance_plot()`", "`fit()`")


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"], ids=["LF", "CRLF", "CR"])
def test_write_keeps_every_line_ending(ending: str) -> None:
    # --write changes the generated blocks only (review run 3, round 2).
    assert ix.rewrite_index(_STALE.replace("\n", ending), DECLS) == DOC.replace(
        "\n", ending
    )


def test_write_keeps_mixed_line_endings_outside_the_blocks() -> None:
    head, _, tail = _STALE.partition("<!-- index:begin a.ipynb -->")
    mixed = head.replace("\n", "\r\n") + "<!-- index:begin a.ipynb -->" + tail
    rewritten = ix.rewrite_index(mixed, DECLS)
    assert rewritten.startswith(head.replace("\n", "\r\n"))
    assert rewritten.endswith(DOC.partition("<!-- index:begin a.ipynb -->")[2])


@pytest.mark.parametrize(
    "space", _NON_COMMONMARK_SPACES, ids=lambda s: f"U+{ord(s):04X}"
)
def test_a_line_of_other_whitespace_is_not_blank(space: str) -> None:
    # CommonMark reads such a line as paragraph text, so the break underlines
    # it as a setext heading.
    assert "grammar" in _errors(DOC + "\n" + space + "\n---\n")


@pytest.mark.parametrize(
    "space", _NON_COMMONMARK_SPACES, ids=lambda s: f"U+{ord(s):04X}"
)
def test_other_whitespace_after_a_fence_closer_keeps_the_fence_open(space: str) -> None:
    # CommonMark keeps the fence open, so section b and its block render as code.
    fence = "```\ncode\n```" + space + "\n"
    text = _doc(_section("a.ipynb"), fence, _section("b.ipynb"))
    assert "fence" in _errors(text)


@pytest.mark.parametrize(
    "space", _NON_COMMONMARK_SPACES, ids=lambda s: f"U+{ord(s):04X}"
)
def test_other_whitespace_after_hashes_is_not_a_heading(space: str) -> None:
    assert "grammar" in _errors(DOC + "\n####" + space + "x\n")


@pytest.mark.parametrize(
    "text",
    [
        "This prose mentions index:end safely.\n",
        "Inline `<!-- index:begin a.ipynb -->` in prose.\n",
        "```\nliteral index:end token\n```\n",
        "```text\n# index:begin and index:end\n```\n",
    ],
)
def test_marker_text_outside_a_marker_candidate_passes(text: str) -> None:
    # Only a line that starts with `<!--` is a marker candidate.
    assert ix.check_index(DOC + "\n" + text, DECLS) == []


_NEAR_MISS_MARKERS = [
    "<!--index:begin b.ipynb-->",
    "<!-- index:begin b.ipynb-->",
    "<!-- index:begin b.ipynb -->",
    " <!-- index:begin b.ipynb -->",
    "<!-- index:end --> ",
    "<!-- index:begin b.ipynb --> trailing",
]


@pytest.mark.parametrize("marker", _NEAR_MISS_MARKERS)
def test_a_near_miss_marker_outside_a_fence_fails(marker: str) -> None:
    assert "malformed index marker" in _errors(DOC + "\n" + marker + "\n")


@pytest.mark.parametrize(
    "marker",
    ["<!-- index:begin b.ipynb -->", "<!-- index:end -->", "  <!-- index:end -->"]
    + _NEAR_MISS_MARKERS,
)
def test_a_marker_candidate_inside_a_fence_fails(marker: str) -> None:
    assert "inside a code fence" in _errors(DOC + "\n```\n" + marker + "\n```\n")


def test_a_thematic_break_after_a_blank_line_is_not_a_heading() -> None:
    assert ix.check_index(DOC + "\nText.\n\n---\n\n***\n", DECLS) == []


def test_a_thematic_break_on_the_first_line_is_not_a_heading() -> None:
    # Nothing precedes it to underline, as after a blank line (H-0119 section 5).
    assert ix.check_index("---\n" + DOC, DECLS) == []


@pytest.mark.parametrize("name", ["café.ipynb", "a b.ipynb", "a+b.ipynb"])
def test_a_notebook_name_outside_the_name_characters_fails(name: str) -> None:
    # Names are ASCII letters, digits, "_", "." and "-" only (H-0119 section 5).
    heading = DOC + f"\n### `{name}`\n"
    assert "### `<name>.ipynb`" in _errors(heading)
    marker = DOC.replace("<!-- index:begin b.ipynb -->", f"<!-- index:begin {name} -->")
    assert "malformed index marker" in _errors(marker)


def test_a_hash_inside_a_fence_is_not_a_heading() -> None:
    assert ix.check_index(DOC + "\n```bash\n  # comment\n> # x\n```\n", DECLS) == []


def test_a_generated_block_inside_a_code_fence_fails() -> None:
    # The block would render as a code sample, not as the section's content
    # (implementation review round 3).
    fenced = "```md\n" + _block("b.ipynb") + "\n```"
    text = _doc(_section("a.ipynb"), _section("b.ipynb", block=fenced))
    assert "fence" in _errors(text)


def test_an_unclosed_fence_fails() -> None:
    assert "fence" in _errors(DOC + "\n````md\n```\n")


def test_changing_a_declaration_fails_the_index() -> None:
    changed = {**DECLS, "b.ipynb": _decl(methods=["fit", "predict"], extras=[])}
    assert "differs" in _errors(DOC, changed)


def test_markers_and_headings_inside_fences() -> None:
    fenced = DOC + "\n```\n### `b.ipynb`\n```\n"
    assert ix.check_index(fenced, DECLS) == [], "a fenced heading is not a heading"
    marker = DOC + "\n```\n<!-- index:end -->\n```\n"
    assert "marker" in _errors(marker), "a marker counts even in a fence"


def test_write_rewrites_only_the_blocks() -> None:
    stale = DOC.replace("`fit()`, `importance_plot()`", "`fit()`")
    assert ix.rewrite_index(stale, DECLS) == DOC
    with pytest.raises(ix.ContractError, match="missing"):
        ix.rewrite_index(_doc(_section("a.ipynb")), DECLS)


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


def test_write_keeps_the_file_line_endings(tmp_path: pathlib.Path) -> None:
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "docs").mkdir()
    for path in ix.notebook_paths(ROOT):
        (tmp_path / "notebooks" / path.name).write_bytes(path.read_bytes())
    good = (ROOT / "docs" / "examples.md").read_bytes().replace(b"\n", b"\r\n")
    target = tmp_path / "docs" / "examples.md"
    target.write_bytes(
        good.replace(
            b"**Demonstrates:** `evaluate_table()`, ", b"**Demonstrates:** ", 1
        )
    )
    assert ix.check(tmp_path), "the stale block was not detected"
    ix.write(tmp_path)
    assert target.read_bytes() == good


def test_write_leaves_a_malformed_file_unchanged(tmp_path: pathlib.Path) -> None:
    # The rewrite is validated before the file is opened for writing
    # (review run 3, round 3).
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "docs").mkdir()
    for path in ix.notebook_paths(ROOT):
        (tmp_path / "notebooks" / path.name).write_bytes(path.read_bytes())
    broken = (
        (ROOT / "docs" / "examples.md")
        .read_bytes()
        .replace(b"<!-- index:end -->\n", b"", 1)
    )
    target = tmp_path / "docs" / "examples.md"
    target.write_bytes(broken)
    with pytest.raises(ix.ContractError, match="not closed"):
        ix.write(tmp_path)
    assert target.read_bytes() == broken


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
