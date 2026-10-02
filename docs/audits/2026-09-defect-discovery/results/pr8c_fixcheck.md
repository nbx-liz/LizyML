FIX CHECK: **CONFIRMED**

Checked `1d2fbc7d9c358379048e5f3d5754ab2deeb9d3e3`. Repository remains clean; all writes were confined to temporary scratch.

| Item | Confirmed? | Evidence (what I executed) |
|---|---|---|
| Symlink staging bypass | Yes | Exercised `files_to_stage` → `_p2_before_tree` → `staged`. Before-file, before-parent, after-file, and after-parent symlinks raised `ManifestError` with **zero copy calls**. Before trees and `lizyml/a.py` remained unchanged. Ordinary staging copied three files, ran a passing test, and restored the before tree. [Probe results](/tmp/pr8c-fixcheck-HlwslCz4/results.json) |
| Test count | Yes | Fresh collection: **128** instrument tests; **148** across `tests/test_docs`. Instrument execution: **128 passed**. [Execution log](/tmp/pr8c-fixcheck-HlwslCz4/unit.txt) |
| Test-module description | Yes | Subprocess tracing recorded **20 nested pytest launches across 12 historical counterexample cases**, including two per historical staging case. Round-7/8 checks launched none, matching the revised docstring. [Trace](/tmp/pr8c-fixcheck-HlwslCz4/subprocess-trace.json) |

Claims:

| Claim | TRUE/FALSE/PARTLY | Evidence |
|---|---|---|
| Criterion 4b: confinement before copying; restoration | TRUE | Executed refusal probes, ordinary staging, and existing confinement/restoration tests passed. |
| Section 5: exhausted budget; one narrow check, without comprehensive post-fix review | TRUE | Matches the maintainer’s instruction in the capsule and this check’s scope. |
| No tracked symlinks at the six specified commits | TRUE | `git ls-tree -r` found zero mode-`120000` entries at HEAD, `33a3f6e`, `5fb8a80`, `1d7c4e2`, `ccae32b`, and `fc9d820`. |
| Measurement 16: corrected counts and execution description | TRUE | Collection and tracing above. |
| Trial 7 equals completion output | TRUE | Byte-identical comparison. |
| Trial 7 differs from trial 6 only in timing | PARTLY | Result bodies match exactly; wrapper **run labels and starting load** also differ. |
| Criterion 10: nine mutations caught | TRUE for recorded output | Saved report contains nine caught cases and the 9/9 summary; campaign was not rerun during this narrow check. |

Outside scope (reported, not part of the verdict): none.