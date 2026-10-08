"""The documents state the embargo merge (H-0115, #273, acceptance criterion 10).

Each check reads the shipped document and pins the statement H-0115 made it
carry, so a revert to the old two-knob wording fails here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin + len(start))]


# ---------------------------------------------------------------------------
# docs/config-reference.md
# ---------------------------------------------------------------------------


def test_config_reference_defaults_have_no_embargo() -> None:
    text = _read("docs/config-reference.md")
    row = next(
        line
        for line in text.splitlines()
        if line.startswith("| `purged_time_series` |")
    )
    assert "embargo" not in row


def test_config_reference_draws_no_embargo_after_validation() -> None:
    guide = _section(
        _read("docs/config-reference.md"),
        "#### 2) `purged_time_series`",
        "#### 3) `group_time_series`",
    )
    diagram = _section(guide, "```text", "```\n\n")
    assert "embargo" not in diagram
    assert "[purge_gap]" in diagram
    assert "deprecated" in guide
    assert "added to `purge_gap`" in guide


def test_config_reference_comparison_has_one_exclusion_key() -> None:
    text = _read("docs/config-reference.md")
    table = _section(text, "Quick comparison:", "#### 1)")
    header = next(line for line in table.splitlines() if line.startswith("| method |"))
    assert header.startswith("| method | boundary key |")
    assert "extra exclusion key" not in header
    assert "embargo" not in table


# ---------------------------------------------------------------------------
# docs/DEPRECATIONS.md
# ---------------------------------------------------------------------------


def _deprecation_rows() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in _read("docs/DEPRECATIONS.md").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`"):
            rows[cells[0]] = cells
    return rows


def test_deprecations_map_every_spelling_to_purge_gap() -> None:
    rows = _deprecation_rows()
    expected = {
        "`purged_time_series.embargo`": "H-0115",
        "`purged_time_series.embargo_pct`": "H-0040",
        "`purged_time_series.gap`": "H-0038",
        "`purged_time_series.purge_window`": "H-0038",
        "`PurgedTimeSeriesSplitter(embargo=...)`": "H-0115",
    }
    for key, source in expected.items():
        target, removal, since = rows[key][1], rows[key][2], rows[key][3]
        assert target.startswith("`purge_gap`"), key
        assert "v1.0" in removal, key
        assert since.startswith(source), key
        assert "H-0021" not in since, key


def test_deprecations_note_the_v1_load_path() -> None:
    text = _read("docs/DEPRECATIONS.md")
    note = _section(text, "### `purged_time_series.embargo` merged", "\n### ")
    assert "Removal note for v1.0" in note
    assert "Model.load()" in note


# ---------------------------------------------------------------------------
# Tutorial notebook
# ---------------------------------------------------------------------------


def test_tutorial_notebook_does_not_use_embargo() -> None:
    nb = json.loads(_read("notebooks/tutorial_time_series_lgbm.ipynb"))
    sources = "".join("".join(cell["source"]) for cell in nb["cells"])
    assert not re.search(r"\bembargo\b", sources)
    assert '"purge_gap": 74' in sources


# ---------------------------------------------------------------------------
# BLUEPRINT.md
# ---------------------------------------------------------------------------


def test_blueprint_defaults_table_has_no_embargo() -> None:
    row = next(
        line
        for line in _read("BLUEPRINT.md").splitlines()
        if line.startswith("| `purged_time_series` | `n_splits=5`")
    )
    assert "embargo" not in row


def test_blueprint_states_the_forward_chaining_geometry() -> None:
    text = _read("BLUEPRINT.md")
    section = _section(text, "## 10.2 ", "## 10.3 ")
    assert "前向き連鎖" in section
    assert "学習の最大 index は検証の最小 index より小さい" in section
    assert "embargo" in section and "purge_gap` に統合した" in section


def test_blueprint_legacy_key_rule_adds_to_purge_gap() -> None:
    text = _read("BLUEPRINT.md")
    assert "値を `purge_gap` に**加算**する（H-0115）" in text
    assert "| `PurgedTimeSeriesSplitter.embargo` | api |" in text


def test_blueprint_inner_gap_rule_names_purge_gap() -> None:
    section = _section(_read("BLUEPRINT.md"), "### 10.3.1", "## 10.4")
    rule = next(line for line in section.splitlines() if "境界 gap である" in line)
    assert "`purged_time_series` では `purge_gap`（" in rule
