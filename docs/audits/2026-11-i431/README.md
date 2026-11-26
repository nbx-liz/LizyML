# #431: lightgbm 4.6.0 -> 4.7.0 and the H-0138 oracles

The lock bump turned Quality (3.10), (3.11) and (3.12) red: the required lanes
compare the pandas 2.3.3 and 3.0.1 oracles in strict mode, and those were made
with lightgbm 4.6.0 (`model_versions_match`, H-0144 対応方針 2). The 2.0.0
oracle (lowest-direct lane, lightgbm 4.4.0) and the 3.1.0 oracle (already
lightgbm 4.7.0) are not touched.

All runs below were on V (aarch64, Ubuntu), through `heavy.sh`, with
`PYTHONHASHSEED=0`. The environments were synced from the lock: develop's
(lightgbm 4.6.0) for the control, this branch's (lightgbm 4.7.0) for the new
files. Only lightgbm differs between them: Python 3.12.3 / pandas 3.0.1 /
numpy 2.4.2 and Python 3.10.22 / pandas 2.3.3 / numpy 2.2.6, pyarrow 25.0.1
in both.

Two source trees were used, as CONTRIBUTING.md prescribes:

- **The oracles** (`regen_431.sh`, `strip_probe.sh`) were generated in a
  `b297de4` worktree carrying develop's `docs/audits/2026-10-h0138/{oracle,firing}`
  (step 1). `make_oracle.py` imports `Model` and the encoder from that tree,
  so the oracle is made by the code before H-0138, never by the code it checks.
- **The overlays** (`overlay_431.sh`) and **the tests** ran in the PR worktree
  (`eec71dd` plus the new oracles), as step 2 and step 5 do. `make_overlay.py`
  serves the frozen codec and refuses any other `lizyml` module.

| Step | Script / output | Result |
|---|---|---|
| Control: regenerate under 4.6.0, twice | `regen_431.sh` | both runs byte-identical to the committed 3.0.1 (`62d88cf7…`) and 2.3.3 (`b4e5a8db…`) files, so the rig reproduces the shipped oracles |
| Regenerate under 4.7.0, twice | `regen_431.sh` | each pair byte-identical: 3.0.1 `88c74048…`, 2.3.3 `06291850…` |
| Full comparison, committed (`eec71dd`) vs this head | `full_compare.py`, `full_compare.out` | after masking `meta.lightgbm` and each model case's `model.model_sha256`, both files are equal to the committed ones in every other field of every case and every temporal case, types included and floats compared by `float.hex` (so `-0.0` and `0.0` differ); the hash changed in 74 of 74 model cases |
| The comparison catches other changes | `mutation_check.py`, `mutation_check.out` | copies of each committed file with one change each (an `encoder` field and the `dtype` of the last model case, a `pred_hex`, a temporal case, the sign of a float zero in a model case's `encoder` and in its `model.pandas_categorical`, `meta.numpy`) all fail `full_compare.py`; the unmutated copy passes. The round-2 version of `full_compare.py` (`bb28606`, plain `==`) passed both zero-sign mutants, which is the defect the round-2 review found |
| Why the hash moved | `model_text.py`, `model_text.diff` | lightgbm 4.7.0's `model_to_string()` adds one parameter line, `[gpu_device_id_list: ]`, to a small sample model; the trees are the same |
| All 74 models | `strip_probe.sh`, `full_compare.out` | the generator, patched to require that line exactly once as a whole line in every model text and to hash the text without it, matches the committed 4.6.0 files in all 74 hashes and in every other unmasked field; the only remaining difference is `meta.lightgbm` (4.6.0 -> 4.7.0). The unpatched files above are the negative control (74 of 74 change). An EXIT trap restores the generator |
| Overlays under 4.7.0, twice | `overlay_431.sh` | 3.0.1 (`1cc072d9…`) and 2.3.3 (`64ec8ce4…`) byte-identical to the committed overlays |
| Tests on this head | `pytest tests/test_features -m ""` | 3.12: 8824 passed; 3.10: 9052 passed; both print `mode=strict … exact=True … model_cases=74, models=compared` |

`diff_oracle.py` and `diff_model.py` were the first, exploratory look; they
report only some fields, and `full_compare.py` is the check the claims rest on.

So lightgbm 4.7.0 does not change any oracle answer: the new files differ from
the old ones only in the lightgbm version they record and in the hash of a
model text that gained one empty parameter line. `provenance.json` is
rewritten with `write_provenance.py` for all four versions; the 3.0.1 and 2.3.3
entries (oracle and overlay) now record lightgbm 4.7.0 and pyarrow 25.0.1, the
environment that regenerated them. CONTRIBUTING.md now describes this case (a
numpy or lightgbm move on a pandas version that already has files).

The scripts keep the session's absolute paths; they are a record of what ran,
not a reusable tool.
