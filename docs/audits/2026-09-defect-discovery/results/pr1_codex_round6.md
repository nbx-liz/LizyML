# PR 1 — Codex review, round 6 (2026-09-07)

Rounds 1–5 are in the sibling files. Round 5 closed the loop on a
pre-registration and handed PR 1 to the maintainer as D5. The relational monitor
over rounds 4–5 (`results/pr1_monitor_round45.md`) then judged that stop **sound
in content, defective in procedure** — self-certified by the party owning the
deliverable — and recommended `take-stop-condition`.

**The maintainer read both and directed one further round, with the review
target narrowed to the remedies for findings already raised.** That instruction
overrides the run's own pre-registration, and it repairs the procedural defect
the monitor named: the party outside the loop decided, not the maker. The
monitor's disposition is therefore `redirect`, not `continue` — the scope
narrows rather than the loop continuing.

---

## Verdict

```
VERDICT: APPROVE
```

The first `APPROVE` on PR 1, after 4, 4, 2, 1, 1 blocking findings. No blocking
finding in the four surfaces under review, and no `OUT OF SCOPE` section.

---

## What this round reviewed

Four surfaces, all of them remedies, and nothing else:

- **A.** Round 5's remedy — `check_calibration_param_names` at the `tune()`
  entry point (`_model_tuning.py:211`), and the ordering test parametrized over
  `TRAINING_ENTRY_POINTS`.
- **B.** The entry-point axis closed after round 5 (`5cf143b`) —
  `NON_TRAINING_ENTRY_POINTS`, the partition against `dir(Model)`, and the
  execution of each non-training claim.
- **C.** A DC1 defect the maker found inside B and fixed before the round
  (`7c77278`), disclosed in the prompt rather than left to be found.
- **D.** H-0093 decisions 4 and 6, checked for DC5 only.

Explicitly out of scope: everything rounds 1–5 already reviewed, the audit prose
in `DECISIONS-PENDING.md`, and round 3's carried-forward exclusion of call shapes
that appear nowhere in `lizyml/`.

---

## The defect found before the round (C)

`test_non_training_entry_points_really_do_not_train` called each classified
method bare, as `getattr(model, name)()`. **3 of the 21 have a required
argument**, so the call raised `TypeError` at argument binding: the body never
executed, and `spy.calls == []` held because nothing ran, not because nothing
trained. Two of the three are `load` and `export_code` — the methods rounds 1
and 2 found real production defects in.

This is DC1 inside the remedy for the axis-closure finding, and it is the same
shape the PR spent rounds 3–5 on: an assertion that is green because it did not
look. It was found by reading the remedy against its own docstring claim ("a
method that cannot be invoked without arguments still cannot have trained,
because the spy would have seen it" — false; the spy sees nothing when nothing
runs).

The fix asserts the *binding* before the call, so a classified method with a
required argument fails loudly rather than passing vacuously, and
`NON_TRAINING_ENTRY_POINT_ARGS` supplies what the three need. RED verified per
entry; all three reach the body (`MODEL_NOT_FIT`, `MODEL_NOT_FIT`,
`DESERIALIZATION_FAILED`).

---

## What the reviewer ran

Each item below is the reviewer's own execution, not the maker's:

- **Targeted suite** — `pytest tests/test_calibration/test_calibration_param_names.py`:
  **50 passed**.
- **A, the tuning gate** — disabled *only* the tuning-path calibration check in
  memory: the `fit` cell passed and the `tune` cell failed with `DID NOT RAISE`,
  which is the shape of the round-5 miss. With the check enabled, **resume,
  repeated tuning, and attachment to an existing study** all refused an invalid
  `calibration.params` with **zero Booster calls** — the three re-entry paths the
  prompt asked about.
- **B, the partition** — injected ordinary, inherited, `classmethod`,
  `staticmethod`, and class-creation methods onto `Model`; **each failed as
  unclassified**. Nonexistent names and overlapping classifications failed too.
  So the axis is closed against the ways a new public callable can arrive, not
  only against the ordinary one.
- **C, the binding remedy** — removed each of the three argument entries
  independently; every affected cell failed with the vacuity message.
  `sys.settrace` confirmed **all 21 method bodies execute**, including the three
  disclosed cases with their expected error codes.
- **Spy wiring** — identity-checked that the adapter and the calibrator share
  the patched LightGBM module; a valid calibrated fit recorded **8 Booster
  calls**, so the spy that asserts zero can see non-zero.
- **D, the declarations** — H-0093 decisions 4 and 6 read against the measured
  remedies: no blocking mismatch.

Repository files were unchanged by the review.

---

## The one bound the reviewer named

Not blocking, and recorded rather than argued with: the non-training execution
evidence covers **calls on an unfitted `Model`** and does not establish behaviour
across every model state. The test's docstring now states that bound instead of
the stronger claim it carried, which is the same disposition round 3 took on the
AST scan — narrow the claim to what was measured rather than grow the apparatus
until the claim is true.

---

## State handed to the maintainer

Blocking findings per round: **4, 4, 2, 1, 1, 0**. Six rounds, five of them
finding something real; four live production defects and two specification
statements false of the code, every one reproduced before it was accepted.

The last two rounds each found the previous round's fix incomplete, which is why
round 5 stopped; round 6 exists because the maintainer, outside the loop,
narrowed it to the remedies and asked for one more pass. It found none, having
executed the three re-entry paths and the five ways a public callable can arrive
that the maker had not.

Per the round-6 pre-registration, written before the verdict was seen: `APPROVE`
merges PR 1 under the standing run policy (Codex APPROVE plus CI green). There
is no round 7.
