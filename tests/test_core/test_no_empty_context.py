"""Lint-style regression test: forbid bare ``context={}`` in production code (#118).

The rule:
    Every ``LizyMLError(...)`` must include at least one diagnostic key in
    ``context``. Empty dicts strip the user of useful debugging info — see
    the project rule \"Never swallow errors silently\".

This test scans ``lizyml/`` for the literal ``context={}`` and fails if any
sites are found. Tests and SKILL.md examples are exempt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SOURCE_ROOT = Path(__file__).parent.parent.parent / "lizyml"
_PATTERN = re.compile(r"context\s*=\s*\{\s*\}")


def _iter_python_files() -> list[Path]:
    return [p for p in _SOURCE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


class TestNoEmptyContext:
    def test_source_root_exists(self) -> None:
        """Sanity: the scan target must exist."""
        assert _SOURCE_ROOT.is_dir(), f"missing source root: {_SOURCE_ROOT}"

    def test_no_empty_context_in_production(self) -> None:
        """Every ``LizyMLError`` must carry a non-empty ``context``."""
        offenders: list[str] = []
        for path in _iter_python_files():
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if _PATTERN.search(line):
                    rel = path.relative_to(_SOURCE_ROOT.parent)
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

        if offenders:
            pytest.fail(
                "Empty context={} found in production code (#118 rule). "
                "Every LizyMLError must include at least one diagnostic key:\n"
                + "\n".join(offenders)
            )
