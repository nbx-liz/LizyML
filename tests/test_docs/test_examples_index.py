"""``docs/examples.md`` describes the notebooks it lists (DC3).

The index named methods its notebooks never called and told readers that seven
notebooks needed no extras while they plot (``plots``) or compute SHAP
(``explain``). These tests read each notebook's code cells and check the index
against them: the listed notebooks are exactly the notebooks on disk, every
method the index names for a notebook is called in it, and the extras it lists
are the ones the notebook's calls need.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs" / "examples.md"
NOTEBOOKS = ROOT / "notebooks"


def _sections() -> dict[str, str]:
    parts = re.split(
        r"^### `(tutorial_\w+\.ipynb)`\n", INDEX.read_text(encoding="utf-8"), flags=re.M
    )
    names, bodies = parts[1::2], parts[2::2]
    return {
        name: body.split("\n---")[0] for name, body in zip(names, bodies, strict=True)
    }


def _code(name: str) -> str:
    nb = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )


def _needed_extras(code: str) -> set[str]:
    needed = set()
    if re.search(r"\.\w*plot\w*\(", code):
        needed.add("plots")
    if re.search(r"""kind\s*=\s*["']shap["']|return_shap\s*=\s*True""", code):
        needed.add("explain")
    if re.search(r"\.tune\(", code):
        needed.add("tuning")
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
    code = _code(name)
    named = re.findall(r"`(\w+)\(", SECTIONS[name])
    assert named, f"{name}: the entry names no method"
    missing = sorted({m for m in named if not re.search(rf"\b{m}\(", code)})
    assert not missing, f"{name}: named but never called: {missing}"


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_the_listed_extras_are_the_ones_the_notebook_needs(name: str) -> None:
    assert _listed_extras(SECTIONS[name]) == _needed_extras(_code(name))
