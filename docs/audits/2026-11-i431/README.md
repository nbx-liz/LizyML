# #431: lightgbm 4.6.0 -> 4.7.0 and the H-0138 oracles

The lock bump turned Quality (3.10), (3.11) and (3.12) red: the required lanes
compare the pandas 2.3.3 and 3.0.1 oracles in strict mode, and those were made
with lightgbm 4.6.0 (`model_versions_match`, H-0144 対応方針 2). The 2.0.0
oracle (lowest-direct lane, lightgbm 4.4.0) and the 3.1.0 oracle (already
lightgbm 4.7.0) are not touched.

All runs below were on V (aarch64, Ubuntu), through `heavy.sh`, with
`PYTHONHASHSEED=0`, in a `b297de4` worktree carrying develop's
`docs/audits/2026-10-h0138/{oracle,firing}` (CONTRIBUTING.md step 1). The
environments were synced from the lock: develop's (lightgbm 4.6.0) for the
control, this branch's (lightgbm 4.7.0) for the new files. Only lightgbm
differs between them: Python 3.12.3 / pandas 3.0.1 / numpy 2.4.2 and Python
3.10.22 / pandas 2.3.3 / numpy 2.2.6, pyarrow 25.0.1 in both.

| Step | Script | Result |
|---|---|---|
| Control: regenerate under 4.6.0, twice | `regen_431.sh` | both runs byte-identical to the committed 3.0.1 (`62d88cf7…`) and 2.3.3 (`b4e5a8db…`) files, so the rig reproduces the shipped oracles |
| Regenerate under 4.7.0, twice | `regen_431.sh` | each pair byte-identical: 3.0.1 `88c74048…`, 2.3.3 `06291850…` |
| Case diff, committed vs 4.7.0 | `diff_oracle.py`, `diff_model.py` | `meta.lightgbm` and `model_sha256` of all 74 model cases differ; every encoder case, every temporal case, and each model case's `pred_hex`, `pandas_categorical` and `feature_position` are equal |
| Why the hash moved | `model_text.py`, `model_text.diff` | lightgbm 4.7.0's `model_to_string()` adds one parameter line, `[gpu_device_id_list: ]`; the trees are the same |
| All 74 models | `strip_probe.sh` | hashing the 4.7.0 model text with that one line removed reproduces the committed 4.6.0 `model_sha256` of every case, in both files (0 cases differ); the unpatched run above is the negative control (74 differ) |
| Overlays under 4.7.0, twice | `overlay_431.sh` | 3.0.1 (`1cc072d9…`) and 2.3.3 (`64ec8ce4…`) byte-identical to the committed overlays |
| Tests on this head | `pytest tests/test_features -m ""` | 3.12: 8824 passed; 3.10: 9052 passed; both print `mode=strict … exact=True … model_cases=74, models=compared` |

So lightgbm 4.7.0 does not change any oracle model: the new files differ from
the old ones only in the version they record and in the hash of a model text
that gained one empty parameter line. `provenance.json` is rewritten with
`write_provenance.py` for all four versions; the 3.0.1 and 2.3.3 entries
(oracle and overlay) now record lightgbm 4.7.0 and pyarrow 25.0.1, the
environment that regenerated them.

The scripts keep the session's absolute paths; they are a record of what ran,
not a reusable tool.
