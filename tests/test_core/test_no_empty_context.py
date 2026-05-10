"""Lint-style regression tests for ``LizyMLError(context=...)`` quality (#118).

Two rules are enforced over ``lizyml/``:

1. **No literal ``context={}``**. Empty dicts strip the user of useful
   debugging info — see the project rule "Never swallow errors silently".
2. **No all-``None`` ``context``** (e.g. ``context={"x": None, "y": None}``).
   A dict with only ``None`` values carries the same information as an empty
   one — the keys exist but every value was missing at the time the error
   was raised.

Both rules use AST parsing so they correctly handle multi-line context
dicts and miss-spaced literals that a naive regex would skip.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SOURCE_ROOT = Path(__file__).parent.parent.parent / "lizyml"


def _iter_python_files() -> list[Path]:
    return [p for p in _SOURCE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def _is_lizyml_error_call(node: ast.AST) -> bool:
    """Return True if *node* is a call expression whose callable is
    ``LizyMLError`` (or anything ending with ``.LizyMLError``)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "LizyMLError":
        return True
    return isinstance(func, ast.Attribute) and func.attr == "LizyMLError"


def _extract_context_kwarg(call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "context":
            return kw.value
    return None


def _is_all_none_dict(node: ast.expr) -> bool:
    """Return True if *node* is a dict literal whose every value is the
    literal ``None``. Empty dicts return False here (handled separately)."""
    if not isinstance(node, ast.Dict):
        return False
    if not node.values:
        return False
    return all(isinstance(v, ast.Constant) and v.value is None for v in node.values)


def _is_empty_dict(node: ast.expr) -> bool:
    return isinstance(node, ast.Dict) and not node.values


def _scan(predicate: object) -> list[str]:
    """Walk every production file and collect `path:line: snippet` for every
    ``LizyMLError(context=…)`` whose context expression matches *predicate*."""
    offenders: list[str] = []
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - parse errors caught elsewhere
            continue
        for node in ast.walk(tree):
            if not _is_lizyml_error_call(node):
                continue
            ctx = _extract_context_kwarg(node)
            if ctx is None:
                continue
            if predicate(ctx):  # type: ignore[operator]
                rel = path.relative_to(_SOURCE_ROOT.parent)
                offenders.append(f"{rel}:{node.lineno}")
    return offenders


class TestNoEmptyContext:
    def test_source_root_exists(self) -> None:
        """Sanity: the scan target must exist."""
        assert _SOURCE_ROOT.is_dir(), f"missing source root: {_SOURCE_ROOT}"

    def test_no_empty_context_in_production(self) -> None:
        """Every ``LizyMLError`` must carry a non-empty ``context``."""
        offenders = _scan(_is_empty_dict)
        if offenders:
            pytest.fail(
                "Empty context={} found in production code (#118 rule). "
                "Every LizyMLError must include at least one diagnostic key:\n"
                + "\n".join(offenders)
            )

    def test_no_all_none_context_in_production(self) -> None:
        """``context={"x": None}`` is information-equivalent to ``{}`` and is
        forbidden too. At least one value must be non-``None`` (a literal
        ``None`` is fine when it is one of multiple keys)."""
        offenders = _scan(_is_all_none_dict)
        if offenders:
            pytest.fail(
                "All-None context found in production code (#118 rule). "
                "Provide at least one non-None diagnostic value:\n"
                + "\n".join(offenders)
            )
