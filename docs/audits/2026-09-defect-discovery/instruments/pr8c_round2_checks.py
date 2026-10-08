"""PR 8c measurement after design review round 2: provenance and timing.

1. Each reintroduction mutation's `old` text: is every one of its lines among the
   `+` lines the fixing PR's merge added to that file? A mutation that removes
   text the fix did not add would not be reintroducing that fix's defect.
2. Each pinned closure comment: written no earlier than every fixing PR merged.

    .venv/bin/python .../pr8c_round2_checks.py
"""

from __future__ import annotations

import json
import subprocess

MUTATIONS = {
    "264": (278, "lizyml/core/model.py",
            "provider, override=params, validate_values=True"),
    "288": (278, "lizyml/core/_model_factories.py",
            "    if not overlay:\n        return dict(base)\n    canonical = "),
}
PINNED = {
    "258": ([292], 5613263234), "259": ([305], 5924845155), "260": ([305], 5924845419),
    "261": ([275], 5922951205), "262": ([275, 300], 5923365294), "263": ([310], 5926847668),
    "264": ([278], 5599706420), "265": ([274, 300], 5923365543), "266": ([274], 5922951570),
    "267": ([312], 5927643516), "268": ([314], 5928972201), "269": ([302], 5923745761),
    "272": ([310], 5926848025), "277": ([296], 5677178764), "279": ([292], 5613266788),
    "281": ([316], 5930404069), "282": ([292], 5613270618), "284": ([291], 5606053798),
    "285": ([290], 5668402175), "286": ([290], None), "287": ([291], 5606056205),
    "288": ([278], 5599706837), "306": ([308], 5925321349),
}


def gh_json(*args: str) -> object:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main() -> int:
    print("== mutation provenance")
    for issue, (pr, path, old) in MUTATIONS.items():
        oid = gh_json("pr", "view", str(pr), "--json", "mergeCommit")["mergeCommit"]["oid"]
        diff = subprocess.run(["git", "diff", f"{oid}^1", oid, "--", path],
                              capture_output=True, text=True, check=True).stdout
        added = {ln[1:].strip() for ln in diff.splitlines()
                 if ln.startswith("+") and not ln.startswith("+++")}
        lines = [ln.strip() for ln in old.splitlines() if ln.strip()]
        covered = [ln for ln in lines if ln in added]
        print(f"#{issue}: {len(covered)}/{len(lines)} lines of `old` were added by #{pr} "
              f"({path}); not added: {[ln for ln in lines if ln not in added]}")
    print("== pinned closure comment written after every fixing merge")
    for issue, (prs, cid) in PINNED.items():
        if cid is None:
            print(f"#{issue}: no pinned comment (not-planned)")
            continue
        c = gh_json("api", f"repos/nbx-liz/LizyML/issues/comments/{cid}")
        created = c["created_at"]
        merged = {p: gh_json("pr", "view", str(p), "--json", "mergedAt")["mergedAt"]
                  for p in prs}
        late = [p for p, m in merged.items() if m > created]
        print(f"#{issue}: comment {created}; merges {merged}; "
              f"{'OK' if not late else 'PREDATES ' + str(late)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
