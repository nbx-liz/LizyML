# PR 1 — Codex review, round 4 (2026-09-07)

Rounds 1–3 are in the sibling files. Before this round a **relational** monitor
comparing rounds 2–3 returned `CONVERGING` / `redirect`
(`scratchpad/monitor-pr1-r3.md`). Its disposition was acted on before the prompt
was written: the route scan's declared claim was **narrowed** rather than the
scan grown further, and the round was aimed at decision 6 — the calibration
gate, which no reviewer had seen.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

One blocking finding, on exactly the part the round was aimed at.

### 1 — the calibration check fires after the whole outer CV, not before training

`BLUEPRINT.md:1018` and H-0093 both state the check fires before training
starts. The call sat in `_run_calibration`, which runs after the outer CV
completes. Codex configured an invalid `num_leave` and recorded `lightgbm.train`:

```
train_calls_before_rejection 2
bad_forwarded 0
```

So a config that could never produce a usable model paid for a full training
first, and the canonical specification was false of the code.

`test_calibration_param_names.py` did not catch it because it asserted only that
the bad name never reached LightGBM — true under the defective placement too.

---

## Disposition

Accepted. Two changes.

**The call moved to `_merge_params`'s point of return** in `_fit_impl`, so both
parameter surfaces are now checked at the same place, before any component is
built or any split taken. This keeps decision 4's rationale intact: the check
still reads the config at use time, after any post-construction mutation, and
still leaves `Model.load()` able to read a legacy artifact.

**The ordering is now its own assertion.** `test_unknown_calibration_param_is_
refused_before_any_training` asserts `spy.calls == []` alongside the
name-absence check. The spy patches the `lightgbm` module object, which the
adapter and the calibrator both resolve at call time, so it counts every Booster
either would train.

RED verified by restoring the defect and running the one test:

```
AssertionError: 3 Booster(s) were trained before the config was refused
```

(three rather than Codex's two — a different fold count in this config; the
direction is what matters). The file was restored afterwards and the check
re-run green.

H-0093 decision 6 records the finding and the ordering criterion; decision 4's
call-site table now names the real point; `BLUEPRINT.md` §12.2 states that no
Booster is trained for a refused config.

---

## Checked and clean (round 4)

Reported by the reviewer, each with what was run:

- 177 passed across the five files under review.
- An unknown calibration name never reaches the calibrator's own `lgbm.train`.
- `_ISOTONIC_DEFAULTS`, the three self-consumed names, and the `seed`
  non-exemption all agree with execution.
- The `platt` / `beta` exclusion matches the implementation.
- The LightGBM-backed calibrator scan and `LGBM_BACKED_CALIBRATORS` both give
  `["isotonic"]`.
- The route scan's six measured routes match `ESTIMATOR_ROUTES` exactly.
- `Model.load()`, `predict` and the metrics assembly reach an already-fitted
  calibrator; none is a new route for user-supplied names.
- `0/3` names its population, and the 875-vs-824 difference is explained.
- The `fit(params=...)` → PR 2 and `platt` / `beta` → D4 boundaries match the
  code.
