"""Check or regenerate the machine-readable part of ``docs/examples.md`` (H-0119).

Each notebook declares, in ``metadata.lizyml.index``, the ``Model`` methods it
demonstrates and the extras it needs. Its ``index-example`` cells hold the
examples in a closed grammar (``R.m(...)`` or ``N = R.m(...)`` with plain
values as arguments); anything outside the grammar fails. The extras are
derived from those calls through ``lizyml/_extras.py``, and each section of
``docs/examples.md`` carries one generated block that must equal the
declaration.

Usage::

    python scripts/examples_index.py --check
    python scripts/examples_index.py --write
    python scripts/examples_index.py --ci-matrix
    python scripts/examples_index.py --uv-flags-for NOTEBOOK_STEM
    python scripts/examples_index.py --uv-flags-without EXTRA

The three ``--ci-*`` / ``--uv-*`` modes read only the standard library and the
registry file, so CI can run them before installing the package.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import keyword
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TAG = "index-example"
KEYS = ("models", "methods", "extras")


class ContractError(Exception):
    """A notebook or the index breaks the H-0119 contract."""


def _registry(root: Path = ROOT) -> ModuleType:
    """``lizyml/_extras.py``, loaded by path (it imports only the stdlib)."""
    path = root / "lizyml" / "_extras.py"
    if not path.is_file():  # a temporary root in tests has no package
        path = ROOT / "lizyml" / "_extras.py"
    spec = importlib.util.spec_from_file_location("lizyml_extras_registry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


REGISTRY = _registry()


@dataclass(frozen=True)
class Declaration:
    models: tuple[str, ...]
    methods: tuple[str, ...]
    extras: tuple[str, ...]


@dataclass(frozen=True)
class ModelSurface:
    """The public instance methods of ``Model`` and the signatures calls bind to.

    ``others`` names the public members that are not instance methods (a
    classmethod, staticmethod or property) by their kind, so a declaration or
    an example that uses one fails with a clear message.
    """

    methods: frozenset[str]
    signatures: Mapping[str, inspect.Signature]
    others: Mapping[str, str]


@dataclass(frozen=True)
class Statement:
    """One statement of an ``index-example`` cell."""

    cell: int
    number: int
    receiver: str
    method: str
    target: str | None
    conditions: Mapping[str, object]
    source: str


def model_surface() -> ModelSurface:
    """Public instance methods of ``Model``, inherited mixin methods included.

    An instance method is a plain function found on ``Model``'s MRO. The
    receiver check binds each example to a ``Model`` instance, so a
    classmethod, staticmethod or property cannot be an example (H-0119 2.).
    """
    from lizyml import Model

    signatures: dict[str, inspect.Signature] = {}
    others: dict[str, str] = {}
    for name, _ in inspect.getmembers(Model):
        if name.startswith("_"):
            continue
        raw = inspect.getattr_static(Model, name)
        if inspect.isfunction(raw):
            parameters = list(inspect.signature(raw).parameters.values())[1:]
            signatures[name] = inspect.signature(raw).replace(parameters=parameters)
        elif isinstance(raw, classmethod | staticmethod | property):
            others[name] = type(raw).__name__
    return ModelSurface(frozenset(signatures), signatures, others)


def _not_instance_method(name: str, surface: ModelSurface) -> str | None:
    kind = surface.others.get(name)
    return None if kind is None else f"{name!r} is a {kind}, not an instance method"


# --- The declaration (section 2) ----------------------------------------------


def _string_array(index: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = index[key]
    if type(value) is not list or any(type(v) is not str for v in value):
        raise ContractError(f"{key}: must be a JSON array of strings, got {value!r}")
    if len(set(value)) != len(value):
        raise ContractError(f"{key}: has a duplicate entry: {value!r}")
    if value != sorted(value):
        raise ContractError(f"{key}: must be sorted ascending: {value!r}")
    return tuple(value)


def parse_declaration(metadata: Any, surface: ModelSurface | None) -> Declaration:
    """Validate ``metadata.lizyml.index``.

    Only ``metadata.lizyml.index`` is closed; other keys under
    ``metadata.lizyml`` are not part of the contract.

    Args:
        metadata: The notebook's ``metadata`` object.
        surface: ``Model``'s public instance methods. ``None`` skips only the
            membership check of ``methods``, for the CI helpers that run before
            the package is installed (the full check runs in the test suite and
            before each notebook execution).
    """
    if type(metadata) is not dict or "lizyml" not in metadata:
        raise ContractError("no metadata.lizyml.index declaration")
    container = metadata["lizyml"]
    if type(container) is not dict:
        raise ContractError("metadata.lizyml must be an object")
    if "index" not in container:
        raise ContractError("no metadata.lizyml.index declaration")
    index = container["index"]
    if type(index) is not dict:
        raise ContractError("metadata.lizyml.index must be a JSON object")
    if set(index) != set(KEYS):
        raise ContractError(
            f"index must have exactly the keys {list(KEYS)}, got {sorted(index)}"
        )
    models, names, extras = (_string_array(index, key) for key in KEYS)
    for key, values in (("models", models), ("methods", names)):
        if not values:
            raise ContractError(f"{key}: must not be empty")
    for name in models:
        if not name.isidentifier():
            raise ContractError(f"models: {name!r} is not a Python identifier")
        if keyword.iskeyword(name):
            raise ContractError(f"models: {name!r} is a keyword")
    for name in names:
        if name.startswith("_"):
            raise ContractError(f"methods: {name!r} is not public")
        if surface is None:
            continue
        problem = _not_instance_method(name, surface)
        if problem is not None:
            raise ContractError(f"methods: {problem}")
        if name not in surface.methods:
            raise ContractError(f"methods: {name!r} is not a public Model method")
    for extra in extras:
        if extra not in REGISTRY.EXTRA_PACKAGES:
            raise ContractError(f"extras: unknown extra {extra!r}")
    return Declaration(models, names, extras)


# --- The tagged-cell grammar (section 3) ----------------------------------------


def _is_value(node: ast.expr) -> bool:
    """Constants, names, ``a.b.c``, ``v[v]``, ``-constant`` and containers of values."""
    if isinstance(node, ast.Constant | ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        root: ast.expr = node
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name)
    if isinstance(node, ast.Subscript):
        return _is_value(node.value) and _is_value(node.slice)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_is_value(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_value(k) for k in node.keys) and all(
            _is_value(v) for v in node.values
        )
    return False


def _tags(cell: Mapping[str, Any]) -> list[str]:
    """The cell's tags.

    Any cell, tagged or not, fails when its ``metadata`` is not an object or its
    ``tags`` is not a list of strings: whether the cell is an ``index-example``
    cell is then undecidable. Other metadata and other tags are not constrained.
    """
    metadata = cell.get("metadata", {})
    if type(metadata) is not dict:
        raise ContractError("a cell's metadata is not an object")
    tags = metadata.get("tags", [])
    if type(tags) is not list or any(type(t) is not str for t in tags):
        raise ContractError(f"a cell's tags must be a list of strings, got {tags!r}")
    return tags


def _source(cell: Mapping[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    if not isinstance(source, str):
        raise ContractError("a cell's source is neither a string nor a list of lines")
    return source


def bind_call(
    call: ast.Call, method: str, signature: inspect.Signature, where: str
) -> dict[str, object]:
    """Bind the call to ``signature`` and read its extras-related arguments.

    A repeated keyword fails before binding (binding a dict would keep only the
    last value). Each extras-related argument the call passes must be a
    constant; an omitted one is left out, so the registry uses the default.
    """
    names = [kw.arg for kw in call.keywords]
    repeated = sorted({n for n in names if n is not None and names.count(n) > 1})
    if repeated:
        raise ContractError(f"{where}: keyword argument repeated: {repeated}")
    try:
        signature.bind(
            *call.args, **{kw.arg: kw.value for kw in call.keywords if kw.arg}
        )
    except TypeError as exc:
        raise ContractError(
            f"{where}: the call does not match the signature {method}{signature}: {exc}"
        ) from exc
    found: dict[str, object] = {}
    parameters = list(signature.parameters)
    for (owner, argument), spec in REGISTRY.CONDITION_ARGUMENTS.items():
        if owner != method:
            continue
        node: ast.expr | None = None
        for kw in call.keywords:
            if kw.arg == argument:
                node = kw.value
        position = parameters.index(argument)
        if node is None and position < len(call.args):
            if spec.position != position:
                raise ContractError(f"{where}: pass `{argument}` by keyword")
            node = call.args[position]
        if node is None:
            continue
        if not isinstance(node, ast.Constant):
            raise ContractError(f"{where}: `{argument}` must be a constant")
        found[argument] = node.value
    return found


def _statement(
    node: ast.stmt,
    text: str,
    where: str,
    declaration: Declaration,
    surface: ModelSurface,
    cell: int,
    number: int,
) -> Statement:
    target: str | None = None
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise ContractError(f"{where}: an assignment must have exactly one target")
        name = node.targets[0]
        if not isinstance(name, ast.Name):
            raise ContractError(f"{where}: the assignment target must be a simple name")
        if name.id in declaration.models:
            raise ContractError(
                f"{where}: the target {name.id!r} names a declared model"
            )
        target, call = name.id, node.value
    elif isinstance(node, ast.Expr):
        call = node.value
    else:
        raise ContractError(
            f"{where}: {type(node).__name__} is not an allowed statement"
        )
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
    ):
        raise ContractError(f"{where}: the statement is not of the form R.m(...)")
    receiver, method = call.func.value.id, call.func.attr
    if receiver not in declaration.models:
        raise ContractError(f"{where}: {receiver!r} is not a declared model")
    problem = _not_instance_method(method, surface)
    if problem is not None:
        raise ContractError(f"{where}: {problem}")
    if method not in declaration.methods or method not in surface.methods:
        raise ContractError(f"{where}: {method!r} is not a declared method")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise ContractError(f"{where}: `*` unpacking is not allowed")
    if any(kw.arg is None for kw in call.keywords):
        raise ContractError(f"{where}: `**` unpacking is not allowed")
    for value in [*call.args, *(kw.value for kw in call.keywords)]:
        if not _is_value(value):
            raise ContractError(f"{where}: {ast.unparse(value)!r} is not a value")
    conditions = bind_call(call, method, surface.signatures[method], where)
    segment = ast.get_source_segment(text, node)
    assert segment is not None
    return Statement(cell, number, receiver, method, target, conditions, segment)


def tagged_statements(
    nb: Mapping[str, Any], declaration: Declaration, surface: ModelSurface | None = None
) -> list[Statement]:
    """Every statement of the notebook's ``index-example`` cells, validated."""
    surface = model_surface() if surface is None else surface
    found: list[Statement] = []
    for position, cell in enumerate(nb["cells"]):
        if TAG not in _tags(cell):
            continue
        where = f"cell {position}"
        if cell.get("cell_type") != "code":
            raise ContractError(f"{where}: an {TAG} cell must be a code cell")
        text = _source(cell)
        if any(line.lstrip().startswith(("%", "!")) for line in text.splitlines()):
            raise ContractError(f"{where}: a magic or shell line is not Python")
        try:
            body = ast.parse(text).body
        except SyntaxError as exc:
            raise ContractError(f"{where}: does not parse: {exc}") from exc
        if not body:
            raise ContractError(f"{where}: has no statement")
        for number, node in enumerate(body):
            found.append(
                _statement(
                    node,
                    text,
                    f"{where}, statement {number}",
                    declaration,
                    surface,
                    position,
                    number,
                )
            )
    return found


def check_notebook(
    nb: Mapping[str, Any], surface: ModelSurface | None = None
) -> Declaration:
    """All notebook-side rules of sections 2-4; returns the declaration."""
    surface = model_surface() if surface is None else surface
    declaration = parse_declaration(nb.get("metadata"), surface)
    statements = tagged_statements(nb, declaration, surface)
    if not statements:
        raise ContractError(f"has no {TAG} cell")
    used = sorted({s.method for s in statements})
    if used != list(declaration.methods):
        raise ContractError(
            f"methods: declared {list(declaration.methods)}, examples call {used}"
        )
    receivers = sorted({s.receiver for s in statements})
    if receivers != list(declaration.models):
        raise ContractError(
            f"models: declared {list(declaration.models)}, examples use {receivers}"
        )
    derived: set[str] = set()
    for s in statements:
        derived |= REGISTRY.extras_for(s.method, s.conditions)
    if sorted(derived) != list(declaration.extras):
        raise ContractError(
            f"extras: declared {list(declaration.extras)}, "
            f"the examples need {sorted(derived)}"
        )
    return declaration


# --- docs/examples.md (section 5) ----------------------------------------------

_HEADING = re.compile(r"^#{1,6}\s")
_NB_HEADING = re.compile(r"^### `([A-Za-z0-9_.-]+\.ipynb)`$")
_BEGIN = re.compile(r"^<!-- index:begin ([A-Za-z0-9_.-]+\.ipynb) -->$")
_END = "<!-- index:end -->"
#: A CommonMark fence opener: up to three spaces, then three or more backticks
#: (no backtick in the info string) or three or more tildes.
_FENCE_OPEN = re.compile(r"^ {0,3}(?:(`{3,})[^`]*|(~{3,}).*)$")


_LIST_MARKER = re.compile(r"(?:[-+*]|\d{1,9}[.)])[ \t]+")
_ATX = re.compile(r"#{1,6}(?:[ \t]|$)")
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")


def _hidden_heading(line: str) -> bool:
    """Whether ``line`` is an ATX heading in any form but the canonical one.

    The canonical form starts at column 0. CommonMark also renders a heading
    indented by one to three spaces, or inside a block quote or list item;
    those are refused rather than parsed, so that no heading escapes the
    section and census rules. A line indented by four or more spaces with no
    container is an indented code block, not a heading.
    """
    if _HEADING.match(line):
        return False
    rest = line
    contained = False
    while True:
        stripped = rest.lstrip(" \t")
        indent = rest[: len(rest) - len(stripped)]
        if not contained and (indent.count(" ") >= 4 or "\t" in indent):
            return False
        if stripped.startswith(">"):
            rest, contained = stripped[1:], True
            continue
        marker = _LIST_MARKER.match(stripped)
        if marker:
            rest, contained = stripped[marker.end() :], True
            continue
        return bool(_ATX.match(stripped))


def _closes(line: str, fence: str) -> bool:
    """Whether ``line`` closes a fence opened with ``fence``.

    The closer uses the same character, at least as many of it, and nothing
    after it but spaces, so a shorter inner fence or one of the other
    character leaves the outer fence open.
    """
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return False
    run = len(stripped) - len(stripped.lstrip(fence[0]))
    return run >= len(fence) and not stripped[run:].strip()


def render_block(name: str, declaration: Declaration) -> list[str]:
    methods = ", ".join(f"`{m}()`" for m in declaration.methods)
    extras = (
        f"`pip install 'lizyml[{','.join(declaration.extras)}]'`"
        if declaration.extras
        else "none (base install)"
    )
    return [
        f"<!-- index:begin {name} -->",
        f"**Demonstrates:** {methods}",
        "",
        f"**Extras required:** {extras}",
        _END,
    ]


def _blocks(
    lines: Sequence[str], names: Sequence[str]
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Locate each section's generated block; return it and the structure errors."""
    errors: list[str] = []
    sections: list[str] = []
    blocks: dict[str, tuple[int, int]] = {}
    section: str | None = None
    begin: int | None = None
    fence: str | None = None
    fence_line = 0
    # Whether the previous line could be continued by a setext underline:
    # non-blank text outside a fence that is not a heading, marker or fence.
    paragraph = False
    for i, line in enumerate(lines):
        where = f"docs/examples.md:{i + 1}"
        was_fenced = fence is not None
        if fence is not None:
            if _closes(line, fence):
                fence = None
            opened = False
        else:
            opener = _FENCE_OPEN.match(line)
            opened = opener is not None
            if opener is not None:
                fence = opener.group(1) or opener.group(2)
                fence_line = i
        outside = not was_fenced and not opened
        if outside and _hidden_heading(line):
            errors.append(
                f"{where}: a heading must start at column 0, outside quotes and "
                "lists (an indented, quoted or listed heading is not parsed)"
            )
        if outside and paragraph and _SETEXT_UNDERLINE.match(line):
            errors.append(
                f"{where}: a setext heading is not parsed; use a ### heading, or "
                "put a blank line before a thematic break"
            )
        paragraph = (
            outside
            and bool(line.strip())
            and not _HEADING.match(line)
            and not _SETEXT_UNDERLINE.match(line)
            and "index:begin" not in line
            and "index:end" not in line
        )
        if fence is None and not opened and _HEADING.match(line):
            if begin is not None:
                errors.append(
                    f"{where}: the block opened at line {begin + 1} is not closed"
                )
                begin = None
            match = _NB_HEADING.match(line)
            if match:
                section = match.group(1)
                sections.append(section)
            elif ".ipynb" in line:
                errors.append(f"{where}: a notebook heading must be ### `<name>.ipynb`")
                section = None
            elif len(line) - len(line.lstrip("#")) <= 3:
                section = None
        if ("index:begin" in line or "index:end" in line) and (
            fence is not None or opened
        ):
            # A marker in a fence would render as a code sample, not as the
            # section's content, so it fails rather than counting.
            errors.append(f"{where}: an index marker inside a code fence")
            continue
        if "index:begin" in line or "index:end" in line:
            match = _BEGIN.match(line)
            if match is None and line != _END:
                errors.append(f"{where}: malformed index marker {line!r}")
            elif match is not None:
                if begin is not None:
                    errors.append(f"{where}: a begin marker inside an open block")
                elif section is None:
                    errors.append(f"{where}: a marker outside a notebook section")
                elif match.group(1) != section:
                    errors.append(
                        f"{where}: the marker names {match.group(1)} "
                        f"in the {section} section"
                    )
                elif section in blocks:
                    errors.append(
                        f"{where}: {section} has more than one generated block"
                    )
                else:
                    begin = i
            elif begin is None:
                errors.append(
                    f"{where}: an end marker without a begin marker"
                    if section
                    else f"{where}: a marker outside a notebook section"
                )
            else:
                assert section is not None
                blocks[section] = (begin, i)
                begin = None
    if begin is not None:
        errors.append(f"docs/examples.md:{begin + 1}: the block is not closed")
    for name in sorted({s for s in sections if sections.count(s) > 1}):
        errors.append(f"{name}: listed twice")
    for name in sorted(set(names) - set(sections)):
        errors.append(f"{name}: missing from docs/examples.md")
    for name in sorted(set(sections) - set(names)):
        errors.append(f"{name}: listed in docs/examples.md but not a notebook")
    for name in sorted(set(sections) & set(names) - set(blocks)):
        errors.append(f"{name}: the section has no generated block")
    if fence is not None:
        errors.append(
            f"docs/examples.md:{fence_line + 1}: "
            "the code fence opened here is not closed"
        )
    return blocks, errors


def check_index(text: str, declarations: Mapping[str, Declaration]) -> list[str]:
    lines = text.split("\n")
    blocks, errors = _blocks(lines, list(declarations))
    if errors:
        return errors
    for name, (begin, end) in sorted(blocks.items()):
        if lines[begin : end + 1] != render_block(name, declarations[name]):
            errors.append(
                f"{name}: the generated block differs; "
                "run scripts/examples_index.py --write"
            )
    return errors


def rewrite_index(text: str, declarations: Mapping[str, Declaration]) -> str:
    lines = text.split("\n")
    blocks, errors = _blocks(lines, list(declarations))
    if errors:
        raise ContractError("\n".join(errors))
    for name, (begin, end) in sorted(blocks.items(), key=lambda kv: -kv[1][0]):
        lines[begin : end + 1] = render_block(name, declarations[name])
    return "\n".join(lines)


# --- The repository --------------------------------------------------------------


def notebook_paths(root: Path = ROOT) -> list[Path]:
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    if not paths:
        raise ContractError(f"no notebooks under {root / 'notebooks'}")
    return paths


def read_notebook(path: Path) -> dict[str, Any]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    if type(nb) is not dict or type(nb.get("cells")) is not list:
        raise ContractError(f"{path.name}: not a notebook")
    return nb


def _declarations(root: Path) -> tuple[dict[str, Declaration], list[str]]:
    surface = model_surface()
    declarations: dict[str, Declaration] = {}
    errors: list[str] = []
    for path in notebook_paths(root):
        try:
            declarations[path.name] = check_notebook(read_notebook(path), surface)
        except ContractError as exc:
            errors.append(f"{path.name}: {exc}")
    return declarations, errors


def check(root: Path = ROOT) -> list[str]:
    """Every error ``--check`` reports (empty when the contract holds)."""
    declarations, errors = _declarations(root)
    if errors:
        return errors
    text = (root / "docs" / "examples.md").read_text(encoding="utf-8")
    return check_index(text, declarations)


def write(root: Path = ROOT) -> None:
    declarations, errors = _declarations(root)
    if errors:
        raise ContractError("\n".join(errors))
    path = root / "docs" / "examples.md"
    path.write_text(
        rewrite_index(path.read_text(encoding="utf-8"), declarations), encoding="utf-8"
    )


# --- CI helpers (standard library only) ----------------------------------------------


def _declared_extras(root: Path, stem: str) -> tuple[str, ...]:
    path = root / "notebooks" / f"{stem}.ipynb"
    if path not in notebook_paths(root):
        raise ContractError(f"{stem}: no such notebook")
    return parse_declaration(read_notebook(path).get("metadata"), None).extras


def _flags(extras: Sequence[str]) -> list[str]:
    return [part for extra in sorted(extras) for part in ("--extra", extra)]


def uv_extra_flags_for(root: Path, stem: str) -> list[str]:
    """``uv sync`` flags installing exactly the extras ``stem`` declares."""
    return _flags(_declared_extras(root, stem))


def uv_extra_flags_without(root: Path, extra: str) -> list[str]:
    """``uv sync`` flags installing every registry extra but ``extra``."""
    if extra not in REGISTRY.EXTRA_PACKAGES:
        raise ContractError(f"unknown extra {extra!r}")
    return _flags([e for e in REGISTRY.EXTRA_PACKAGES if e != extra])


def ci_matrix(root: Path = ROOT) -> dict[str, list[str]]:
    return {
        "notebook": [p.stem for p in notebook_paths(root)],
        "removed": sorted(REGISTRY.EXTRA_PACKAGES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--ci-matrix", action="store_true")
    mode.add_argument("--uv-flags-for", metavar="NOTEBOOK_STEM")
    mode.add_argument("--uv-flags-without", metavar="EXTRA")
    args = parser.parse_args(argv)
    try:
        if args.check:
            errors = check()
            for error in errors:
                print(error, file=sys.stderr)
            return 1 if errors else 0
        if args.write:
            write()
        elif args.ci_matrix:
            print(json.dumps(ci_matrix()))
        elif args.uv_flags_for:
            print(" ".join(uv_extra_flags_for(ROOT, args.uv_flags_for)))
        else:
            print(" ".join(uv_extra_flags_without(ROOT, args.uv_flags_without)))
    except ContractError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
