"""PR 8c measurement after design review round 3: is #288's mutation evidence for #288?

#288 reported a coverage gap, not a defect: two of the identity merge seams had no
test writing an alias at that layer. Its fix is test-only (commit 22b11b3 in PR
#278). So the evidence a mutation can give is that the tests 22b11b3 added catch a
spelling-based merge which the tests before it did not. At the after tree, with
and without the manifest's #288 `red_mutation`, run the row's test file as shipped
and as it was at 22b11b3^ (before the coverage tests).

    .venv/bin/python .../pr8c_round3_probe.py <after-ref> <scratch-dir>
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import phase3_gap as g  # noqa: E402

COMMIT = "22b11b3"


def main(after_ref: str, scratch: str) -> int:
    repo = pathlib.Path.cwd()
    runner = g.Runner(repo, repo / ".venv/bin/python", pathlib.Path(scratch))
    after = runner.worktree(runner.rev(after_ref))
    row = g.load_manifest()["issues"]["288"]
    MUTATIONS["manifest (overlay_params merges by spelling)"] = row["red_mutation"]
    (test,) = row["tests"]
    shipped = (after / test).read_text(encoding="utf-8")
    older = runner.git("show", f"{COMMIT}^:{test}")
    print(f"{test}: shipped {len(shipped.splitlines())} lines, "
          f"at {COMMIT}^ {len(older.splitlines())} lines")
    try:
        for label, text in (("shipped", shipped), (f"at {COMMIT}^", older)):
            (after / test).write_text(text, encoding="utf-8")
            _, _, plain = runner.run_tests(after, [test])
            base = {c.node for c in plain if c.outcome == "failed"}
            print(f"{label} tests, unmutated: {dict(g.outcomes(plain))}")
            for name, mutation in MUTATIONS.items():
                with g.mutated(after, mutation):
                    _, _, mutated = runner.run_tests(after, [test])
                new = sorted(c.node.split("::", 1)[1] for c in mutated
                             if c.outcome == "failed" and c.node not in base)
                print(f"    {name}: {dict(g.outcomes(mutated))}; newly failed {len(new)}: {new}")
    finally:
        (after / test).write_text(shipped, encoding="utf-8")
    return 0


def _spelling_merge(file: str, target: str, base: str, layer: str) -> dict[str, str]:
    return {"file": file, "old": f"{target} = overlay_params(provider, {base}, {layer})",
            "new": f"{target} = {{**{base}, **{layer}}}",
            "fix_text": f"overlay_params(provider, {base}, {layer})", "why": "probe"}


MUTATIONS = {
    "manifest (overlay_params merges by spelling)": None,
    "tuning-result seam only (model.py)": _spelling_merge(
        "lizyml/core/model.py", "model_params", "model_params", "best_model_params"),
    "provider-fixed seam only (model.py)": _spelling_merge(
        "lizyml/core/model.py", "model_params", "model_params", "fixed"),
    "provider-fixed seam only (_model_tuning.py, tune)": {
        **_spelling_merge("lizyml/core/_model_tuning.py", "resolved_model",
                          "base_model_params", "fixed")},
    "trial-overlay seam only (_model_tuning.py)": _spelling_merge(
        "lizyml/core/_model_tuning.py", "merged_model", "merged_model", "model_p"),
}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
