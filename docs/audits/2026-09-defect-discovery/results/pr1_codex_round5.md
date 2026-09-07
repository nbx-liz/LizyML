# PR 1 — Codex review, round 5 (2026-09-07)

Rounds 1–4 are in the sibling files. Before this round a **relational** monitor
comparing rounds 3–4 returned `CONVERGING` / `continue`
(`scratchpad/monitor-pr1-r4.md`), measuring zero AST-apparatus growth in round 4
and confirming that round 3's claim-narrowing had been disclosed to the round-4
reviewer rather than applied quietly.

That monitor also raised a flag, which the main context adopted as a **binding
pre-registration** and wrote into the round-5 prompt: round 4 had found a defect
in round 3's remedy, which is legitimate once; a round-5 finding located in
*round 4's* remedy would be a remedy-of-remedy chain, and the loop ends there.

It happened. **This is the last round.**

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

One blocking finding, located exactly where the pre-registration said it would
end the loop.

### 1 — `tune()` is a second entry point, and the check is not on it

`lizyml/core/_model_tuning.py:195` validates `model.params` and the tuning
search space and never asks about `calibration.params`; round 4's remedy put
that call on the fit path only. Codex ran a one-trial binary `tune()` followed
by `fit()`, spying on the shared `lightgbm.train`:

```
tune_result=completed  calls_after_tune=2
fit_result=raised ErrorCode.CONFIG_INVALID  total_calls=2  bad_forwarded=0
```

So a config `fit()` refuses completes an entire study first, contradicting
`BLUEPRINT.md:1018` and H-0093 decisions 3 and 6. Round 4's ordering test stayed
green because it exercises direct `fit()` only.

---

## Disposition

The loop is **closed**, per the pre-registration. No round 6 was opened, and the
prompt for round 5 told the reviewer as much.

The finding was still fixed rather than left standing, because handing the
maintainer a PR carrying a reviewer-confirmed defect is worse than handing over a
fixed one. Fixing a confirmed defect is not continuing the loop; opening another
round would have been.

**The fix is shaped by the fact that this is the second miss of the same kind.**
The check was missed at an entry point once by living at construction time, and
again by living on the fit path while `tune()` has its own. So the remedy is not
"call it in one more place":

- `check_calibration_param_names(cfg.calibration)` is called in the tuning path
  beside the existing `check_param_names`;
- `TRAINING_ENTRY_POINTS` enumerates the public methods that train (`fit`,
  `tune`), and `test_no_entry_point_trains_before_refusing` is parametrized over
  it, asserting for each that refusal precedes training. `predict` and
  `export_code` are deliberately absent: neither trains, so neither can train
  before a refusal.

RED verified by removing the tuning-path call and running the parametrized test:
`1 failed, 1 passed` — the `tune` cell red, the `fit` cell green, which is the
shape of the miss itself. The file was restored and the check re-run green.

H-0093 decision 6 records the finding, the entry-point enumeration, and the
"twice missed" reason; decision 4's call-site table now names both entry points;
the acceptance criterion asserts per entry point.

---

## Checked and clean (round 5)

Reported by the reviewer, each with what was run:

- 177 passed across the five files under review.
- Direct invalid calibrated `fit()` refuses with **zero** Booster calls.
- Post-construction mutation of `calibration.params` refuses with zero calls.
- A valid isotonic fit succeeds.
- No calibration, empty isotonic params, and `platt` / `beta` all reach training
  untouched.
- **The ordering spy is correctly wired**: `lightgbm`, `adapter.lgb` and
  `isotonic.lgbm` are the same module object, so `spy.calls == []` observes both
  the model's Boosters and the calibrator's.
- Legacy-artifact loading, restored `best_model_params` rejection on refit,
  model-config mutation, export gating, search-space gating, the route inventory
  and the real-LightGBM name checks all passed their focused tests.
- No additional blocking DC2–DC7 issue in the round-4-touched surface.

## State handed to the maintainer

Blocking findings per round: **4, 4, 2, 1, 1**. Every one reproduced before being
accepted; four were live production defects and two were specification statements
false of the code. The last two rounds each found the previous round's fix
incomplete on an entry point, which is why the loop stops here rather than on a
verdict token. See `DECISIONS-PENDING.md` D5.
