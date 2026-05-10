"""Lint-style regression test: deprecation warnings must name a removal version.

See #120, #121, H-0076.

Every ``warnings.warn(..., DeprecationWarning)`` and
``warnings.warn(..., UserWarning)`` site that talks about deprecated config
surfaces must include a ``Will be removed in vX.Y`` suffix so users know
how urgently to migrate. The single source of truth is
``docs/DEPRECATIONS.md``.

This test scans the warning *messages* themselves (string literals) in
``lizyml/`` rather than relying on grepping call sites, so multi-line
messages and f-strings are covered. Sites that are deliberately not
deprecation-related (no "deprecated" / "deprecat" / "will be removed"
language) are skipped.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SOURCE_ROOT = Path(__file__).parent.parent.parent / "lizyml"
_REMOVAL_PATTERN = re.compile(r"Will be removed in v\d+\.\d+")
_DEPRECATION_KEYWORDS = ("deprecated", "deprecat", "will be removed")


def _iter_python_files() -> list[Path]:
    return [p for p in _SOURCE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def _is_warnings_warn(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "warn"
        and isinstance(func.value, ast.Name)
        and func.value.id == "warnings"
    )


def _flatten_string_arg(node: ast.expr) -> str | None:
    """Return the literal text of a warning message arg, or None if dynamic.

    Supports ``"a" "b"`` implicit-concat, ``"a" + "b"``, and constants only.
    f-strings with non-constant parts are concatenated by their literal
    fragments — good enough for our keyword + suffix detection.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f-string
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                parts.append("?")  # placeholder for FormattedValue
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _flatten_string_arg(node.left)
        right = _flatten_string_arg(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _scan_warning_messages() -> list[tuple[Path, int, str]]:
    """Return (path, line, message) for every ``warnings.warn`` call site
    whose first arg is a string literal we can resolve."""
    found: list[tuple[Path, int, str]] = []
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_warnings_warn(node):
                continue
            if not node.args:
                continue
            msg = _flatten_string_arg(node.args[0])
            if msg is None:
                continue
            found.append((path, node.lineno, msg))
    return found


class TestDeprecationRegistry:
    def test_source_root_exists(self) -> None:
        assert _SOURCE_ROOT.is_dir(), f"missing source root: {_SOURCE_ROOT}"

    def test_deprecations_doc_exists(self) -> None:
        """The central registry referenced by warnings must be present."""
        path = _SOURCE_ROOT.parent / "docs" / "DEPRECATIONS.md"
        assert path.is_file(), f"missing {path}"

    def test_every_deprecation_warning_names_a_removal_version(self) -> None:
        """Each deprecation-flavoured ``warnings.warn`` must include
        ``Will be removed in vX.Y``."""
        offenders: list[str] = []
        for path, lineno, msg in _scan_warning_messages():
            lower = msg.lower()
            if not any(kw in lower for kw in _DEPRECATION_KEYWORDS):
                continue
            if not _REMOVAL_PATTERN.search(msg):
                rel = path.relative_to(_SOURCE_ROOT.parent)
                offenders.append(
                    f"{rel}:{lineno}: missing 'Will be removed in vX.Y'\n"
                    f"  message preview: {msg[:120]!r}"
                )
        if offenders:
            pytest.fail(
                "Deprecation warning(s) missing removal target (H-0076 / "
                "see docs/DEPRECATIONS.md):\n" + "\n".join(offenders)
            )
