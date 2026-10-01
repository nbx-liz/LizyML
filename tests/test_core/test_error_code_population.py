"""Every declared ``ErrorCode`` member is raised by production code (H-0106, #263).

An error code is a claim that a condition is detected. A member that no
``raise`` produces is a documented check that does not exist. This is the
cheap static half; ``test_error_code_raising.py`` executes each condition, so
an unreachable ``if False: raise`` that satisfies this scan still fails there.
"""

from __future__ import annotations

import ast
from pathlib import Path

import lizyml
from lizyml.core.exceptions import ErrorCode

_PACKAGE = Path(lizyml.__file__).resolve().parent


def _raised_members() -> set[str]:
    """``ErrorCode.X`` attribute references inside ``raise`` statements.

    Only ``ast.Raise`` subtrees are walked, so a mention in a comment, a
    docstring or a non-raising expression does not count.
    """
    raised: set[str] = set()
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "ErrorCode"
                ):
                    raised.add(sub.attr)
    return raised


def test_the_scan_finds_the_population() -> None:
    """A scan that resolved the wrong directory finds nothing and would report
    every member as unraised; that is a broken instrument, not a finding."""
    assert _PACKAGE.name == "lizyml"
    assert len(_raised_members()) >= 10


def test_every_member_is_raised_in_production_code() -> None:
    raised = _raised_members()
    declared = {m.name for m in ErrorCode}
    unraised = sorted(declared - raised)
    undeclared = sorted(raised - declared)
    assert not unraised, f"ErrorCode members that no production site raises: {unraised}"
    assert not undeclared, f"raise sites naming no ErrorCode member: {undeclared}"
