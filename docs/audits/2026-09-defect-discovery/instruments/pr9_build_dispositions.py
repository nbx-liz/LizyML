#!/usr/bin/env python3
"""PR 9: bootstrap docs/proposal_dispositions.toml from the three scout results.

One-time generator. After PR 9 the TOML file is the maintained source and is
edited by hand (a new proposal adds its own row); this script is kept only so
the PR's first version is reproducible from ``results/``.

Inputs (``results/``):
  pr9_inventory_A.json, pr9_inventory_B.json  -- clause audit of 42 entries
  pr9_dispositions_C.json                     -- dispositions of the other 68
Plus ``EXTRA_ANCHORS`` below: anchors for decisions the PR folds into BLUEPRINT
that the scouts named as ``proposed_anchor`` under a different spelling, or that
scout C flagged as decided but unstated.

For the 42 audited entries the disposition is ``specified``. Anchors are the
scout's verified ``anchors`` plus each owed clause's ``proposed_anchor``, kept
only when it occurs by full token in the proposal's own HISTORY entry (the
check's rule). Proposed anchors that fail that rule are printed, not dropped
silently, so they can be replaced by hand.

Usage (from the repository root)::

    .venv/bin/python docs/audits/2026-09-defect-discovery/instruments/pr9_build_dispositions.py \
        > docs/proposal_dispositions.toml
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).resolve().parent.parent / "results"
sys.path.insert(0, str(ROOT))

from tests.test_docs._history_grammar import entry_texts  # noqa: E402
from tests.test_docs.test_proposal_blueprint_coverage import has_token  # noqa: E402

OWED = ("missed", "contradicted")
# Line numbers drift with every BLUEPRINT edit (DC3); reasons keep the prose only.
_LINE_REF = re.compile(r"\s*\((?:BLUEPRINT|BP)[ :]?\d[\d,\s\-–]*\)")

EXTRA_ANCHORS: dict[str, list[str]] = {}

# Scout C's reasons that cite a BLUEPRINT line number in prose the regex cannot strip.
REASON_OVERRIDES = {
    "H-0075": "Internal refactor (TaskType annotations, dispatch dicts); the entry states "
    "public API, return meanings and TaskType values are unchanged. Unifying the "
    "unknown-task path to UNSUPPORTED_TASK is internal cleanup of a code BLUEPRINT "
    "already lists in section 16.2.",
}

NAMES = {
    "CHECKSUM_ALGORITHM": {"disposition": "documented", "where": "BLUEPRINT.md"},
    "SUPPORTED_CONFIG_VERSIONS": {"disposition": "documented", "where": "BLUEPRINT.md"},
    "TASK_TYPES": {
        "disposition": "internal",
        "reason": "A module constant of lizyml/core/types/task.py that lizyml does not "
        "re-export: lizyml/__init__.py __all__ lists TaskType, not TASK_TYPES. The "
        "three task values are documented through TaskType.",
    },
    "DEFAULT_TEMPLATE": {
        "disposition": "internal",
        "reason": "Defined in the private module lizyml/plots/_theme.py.",
    },
    "DEFAULT_HEIGHT": {
        "disposition": "internal",
        "reason": "Defined in the private module lizyml/plots/_theme.py.",
    },
    "DEFAULT_WIDTH": {
        "disposition": "internal",
        "reason": "Defined in the private module lizyml/plots/_theme.py.",
    },
}


def toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def main() -> int:
    entries = entry_texts((ROOT / "HISTORY.md").read_text(encoding="utf-8"))
    rows: dict[str, dict] = {}
    rejected: list[str] = []
    for batch in ("A", "B"):
        data = json.loads((RESULTS / f"pr9_inventory_{batch}.json").read_text("utf-8"))
        for pid, entry in data["entries"].items():
            anchors: list[str] = []
            for anchor in entry.get("anchors", []):
                if anchor in anchors:
                    continue
                # The scouts verified their anchors against BLUEPRINT only.
                if has_token(anchor, entries[pid]):
                    anchors.append(anchor)
                else:
                    rejected.append(f"{pid}: stated anchor {anchor!r} not in HISTORY entry")
            for clause in entry["clauses"]:
                proposed = clause.get("proposed_anchor")
                if clause["verdict"] not in OWED or not proposed or proposed in anchors:
                    continue
                if has_token(proposed, entries[pid]):
                    anchors.append(proposed)
                else:
                    rejected.append(f"{pid}: proposed anchor {proposed!r} not in HISTORY entry")
            anchors += [a for a in EXTRA_ANCHORS.get(pid, []) if a not in anchors]
            if not anchors:
                rejected.append(f"{pid}: NO anchor left")
            rows[pid] = {"disposition": "specified", "anchors": anchors}
    data = json.loads((RESULTS / "pr9_dispositions_C.json").read_text("utf-8"))
    for pid, entry in data["entries"].items():
        if pid in rows:
            raise SystemExit(f"{pid} is in both the audit and batch C")
        row = {"disposition": entry["disposition"]}
        if entry["disposition"] == "specified":
            anchors = list(entry["anchors"])
            anchors += [a for a in EXTRA_ANCHORS.get(pid, []) if a not in anchors]
            row["anchors"] = anchors
        if entry["disposition"] == "superseded":
            row["superseded_by"] = entry["superseded_by"]
        if entry["disposition"] != "specified":
            row["reason"] = REASON_OVERRIDES.get(
                pid, _LINE_REF.sub("", entry["reason"]).strip()
            )
        rows[pid] = row
    # This PR's own proposal: BLUEPRINT names the dispositions file it introduces.
    rows["H-0110"] = {"disposition": "specified", "anchors": ["proposal_dispositions.toml"]}
    missing = sorted(set(entries) - set(rows))
    if missing:
        raise SystemExit(f"no disposition for {missing}")

    out = [
        "# BLUEPRINT.md disposition of every HISTORY.md proposal (H-0110, #271).",
        "#",
        "# Checked by tests/test_docs/test_proposal_blueprint_coverage.py; its module",
        "# docstring defines the grammar. A new proposal adds its row here in the same PR.",
        "#",
        "#   specified      anchors = [...]  each anchor occurs by full token in BLUEPRINT.md",
        "#                                   AND in the proposal's own HISTORY.md entry",
        "#   no_obligation  reason           nothing on the CLAUDE.md section 3 surface in force",
        "#   superseded     superseded_by, reason",
        "#   pending        reason           not decided yet",
        "",
    ]
    for pid in sorted(rows):
        row = rows[pid]
        out.append(f'[proposals."{pid}"]')
        out.append(f"disposition = {toml_str(row['disposition'])}")
        if "anchors" in row:
            out.append("anchors = [" + ", ".join(toml_str(a) for a in row["anchors"]) + "]")
        for key in ("superseded_by", "reason"):
            if key in row:
                out.append(f"{key} = {toml_str(row[key])}")
        out.append("")
    out.append("# Public names #271 found documented nowhere: documented (full token in")
    out.append("# `where`) or internal with a reason.")
    out.append("")
    for name, row in NAMES.items():
        out.append(f'[names."{name}"]')
        for key, value in row.items():
            out.append(f"{key} = {toml_str(value)}")
        out.append("")
    sys.stdout.write("\n".join(out))
    for line in rejected:
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
