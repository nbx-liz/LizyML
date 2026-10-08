"""The error-code references list exactly the declared members (H-0106, DC3).

``ErrorCode`` is the source of truth; ``docs/api.md`` and ``BLUEPRINT.md``
§16.2 are derived copies. Before H-0106 the API reference documented a member
nothing raised and omitted three that are raised.
"""

from __future__ import annotations

import re
from pathlib import Path

from lizyml.core.exceptions import ErrorCode

_ROOT = Path(__file__).resolve().parents[2]
_MEMBERS = {m.value for m in ErrorCode}


def _section(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i : text.index(end, i + len(start))]


def test_api_reference_lists_every_member() -> None:
    text = (_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    table = _section(text, "| Code | When raised |", "\n\n")
    documented = set(re.findall(r"^\| `([A-Z_]+)` \|", table, flags=re.MULTILINE))
    assert documented == _MEMBERS


def test_blueprint_lists_every_member() -> None:
    text = (_ROOT / "BLUEPRINT.md").read_text(encoding="utf-8")
    section = _section(text, "## 16.2", "\n# 17.")
    documented = set(re.findall(r"^- `([A-Z_]+)`$", section, flags=re.MULTILINE))
    assert documented == _MEMBERS
