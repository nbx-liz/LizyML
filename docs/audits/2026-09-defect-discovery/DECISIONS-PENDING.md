# Phase 3 — decisions awaiting the maintainer

One place to read every judgement call made during the autonomous run, so they
can be confirmed or reversed in a single pass rather than one interruption at a
time.

**Nothing here blocks the run.** Each item states what was decided, why, what it
would take to reverse, and how urgent the reversal window is. Items are appended
as they arise; the run continues past them under the stated assumption.

Legend for **Reversal cost**:

- `cheap` — a follow-up PR, no rework of merged code
- `moderate` — a follow-up PR that revisits merged code or a merged Proposal
- `expensive` — reversing would invalidate work in several merged PRs

Legend for **Window**:

- `open` — can be decided any time, including after the run
- `before-close` — must be decided before the owning issue is closed
- `before-release` — must be decided before the next version ships

---

## Confirmed already (no action needed, listed for continuity)

| # | Decision | Confirmed |
|---|---|---|
| C1 | Run scope is 40 entries / 129 clauses / 77 edits | 2026-09-06 kickoff |
| C2 | Merge gate for PR 1+ is Codex APPROVE + CI green, no per-PR approval | 2026-09-06 kickoff |
| C3 | H-0024 handled in Phase 3: direction by PR 3, space-merge by a new PR 3b | 2026-09-06 kickoff |
| C4 | PR 0 (#274) shape approved and merged | 2026-09-07 |
| C5 | Phase 3 completion instrument stays deferred; one reconciliation pass immediately before PR 9 | 2026-09-07 |
| C6 | PR 0's implementation change (`BlockedGroupInnerValid` regression fallback) approved | 2026-09-07 |

---

## Open items

### D1 — `feature_weights` becomes effective, changing fits for anyone who set it

**PR** 1 · **Proposal** H-0093 · **Reversal cost** `moderate` · **Window** `before-release`

`model.feature_weights` has been a silent no-op since it shipped: LightGBM 4.6.0
defines `feature_contri`, and `feature_weights` is not a name or an alias, so the
emitted key was discarded. Measured, same data and seed, suppressing `f0`:

```
baseline            f0 gain 2704.69   order ['f1','f0','f2','f3','f4']
feature_weights=…   f0 gain 2704.69   order ['f1','f0','f2','f3','f4']   (identical)
feature_contri=…    f0 gain    0.00   order ['f1','f2','f3','f4','f0']
```

**Decided:** emit `feature_contri`, keep the Config field name `feature_weights`.

**What this means for users:** anyone who set `feature_weights` gets a different
model after this PR. Their tuned `best_params` and saved artifacts were obtained
from a model the weights never touched. No migration is required and nothing
fails to load, but re-tuning is advisable.

**To reverse:** drop the emitted-key change and instead deprecate the Config
field. That would mean deciding the feature is not worth having, since keeping it
under the old key means keeping it inert.

**Also worth knowing:** `BLUEPRINT.md:1425` declares "`feature_weights` →
importance ordering changes" as an invariant to be verified. It was false of the
shipped code for the entire life of the feature, and the test named for it
asserted only that two column names were present in an importance dict — true
whether or not the weights applied.

### D2 — the name gate fires on the way to the estimator, not at config parse

**PR** 1 · **Proposal** H-0093 · **Reversal cost** `cheap` · **Window** `open`

The plan specified rejecting unknown parameter names in `config/schema.py` at
config-parse time. That is not reachable: `ARCHITECTURE.md`'s layer DAG puts
`config/` and `estimators/` both in Layer 1, and Layer 1 may reference Layer 0
only, so the config layer cannot consult the provider.

**Decided:** the check lives in `core/_model_factories.py` (Layer 4), the first
point where config and provider legally meet, and is called at each **point of
use** — after `_merge_params`, before the tuning study, and before `export_code`
writes. `ErrorCode.CONFIG_INVALID` is unchanged and it still fires before any
training. The calibration surface (D3) is checked at the head of each entry point
that trains — `fit` and `tune` — after review round 4 measured that checking it
where the calibrator is built ran the entire outer CV first, and round 5 measured
that putting it on the fit path alone let `tune()` complete a whole study.

It was originally called from `Model.__init__`, and review found two routes that
choice could not reach, both measured: a config mutated after `Model(cfg)`
returns (the caller keeps the reference), and `best_model_params` restored from
an artifact, which are installed after `__init__` has run.

**Consequence:** constructing a bare `LizyMLConfig` with a misspelled parameter
name does not raise, and neither does constructing a `Model` from it. Calling
`fit` / `tune` / `export_code` does.

**Not validated on `load()`** — a saved artifact records a fit that happened, and
refusing to load one because it carries a misspelled key helps nobody. Moving to
point of use is what makes that possible: the artifact reads back, and the run
that would use the dead name is what fails.

### D3 — PR 1 also gates `calibration.params`, which the plan did not scope

**PR** 1 · **Proposal** H-0093 decision 6 · **Reversal cost** `cheap` ·
**Window** `open`

Review round 3 found a fourth route already in the tree: `IsotonicCalibrator`
merges `calibration.params` over its defaults and hands the result to
`lgbm.train`, so an unknown name there is discarded exactly as on `model.params`.
Measured: `IsotonicCalibrator({"not_a_lightgbm_parameter": 7})` forwards the name
to LightGBM and trains without complaint.

**Decided:** gate it in this PR rather than deferring. PR 1's acceptance
criterion is that names LightGBM would discard are refused; shipping a route the
PR itself enumerated as ungated would make that declaration false (DC5).

**Firing rate** `0/3 of configs carrying calibration.params` — 875 configs
recorded over the shipped suite, 94 carrying a calibration block, 3 carrying
params, none rejected. Distinct keys `{num_boost_round, seed}`.

**Consequence:** a config with an unknown name under `calibration.params` and
`method: isotonic` now fails at `fit`. Measured occurrences in the shipped
corpus: zero. Only LightGBM-backed methods are checked, and which those are is
scanned from each calibrator's imports rather than declared in prose.

**To reverse:** delete `check_calibration_param_names` and its call. The other
three surfaces are unaffected.

### D4 — `platt` / `beta` ignore `calibration.params` entirely (not fixed here)

**PR** — · **Proposal** — · **Reversal cost** — · **Window** `open`

Not a decision so much as a finding that needs one. `PlattCalibrator.__init__`
takes `params` and never reads it — `LogisticRegression(C=1.0, solver="lbfgs",
max_iter=200)` is hardcoded — and `BetaCalibrator.__init__` drops it the same
way. So `calibration.params` under `method: platt` or `method: beta` is
silently inert — the same user-visible shape as the defect PR 1 is closing, but
a different mechanism (nothing reads it, rather than LightGBM discarding it).

**Not changed in PR 1.** Deciding what those params should mean is a design
question, not a name check, and PR 1's gate deliberately does not touch
non-LightGBM calibrators.

**Needs a decision:** file it as an issue for this run's later PRs, or accept
`params` as meaningless for `platt` / `beta` and say so in BLUEPRINT §12.


### D5 — PR 1's review loop closed without an APPROVE; the run is paused on it

**PR** 1 · **Proposal** H-0093 · **Reversal cost** — · **Window** `before-close`

**This item now needs a decision. The run has stopped opening rounds on PR 1.**

Blocking findings per round: **4, 4, 2, 1, 1**. Every one was reproduced before being
accepted and every one was real; three were live production defects
(`Model.load()` rejecting legacy artifacts, `export_code` ungated,
`calibration.params` ungated) and one was a specification statement that was
false of the code (the calibration check ran after the whole outer CV).

The count converged and severity fell, but rounds 4 and 5 each found the
*previous round's fix* incomplete on an entry point — round 4 that the check ran
after the outer CV, round 5 that `tune()` has its own entry point the fix did not
reach. Both were pre-registered as the stop condition before round 5 ran, by the
relational monitor and in this item, so the loop closed on that condition rather
than on a verdict token.

**What is worth knowing:** the PR grew past its planned scope twice, both times
because review found a route the plan had not enumerated — `export_code` in
round 2 and `calibration.params` in round 3 (D3). PR 1's scope in the plan was
written from the issue text rather than from a scan of the tree, and that is the
reason, not reviewer thoroughness. The later PRs in this run have scopes written
the same way.

**Round 5's finding was fixed** (the tuning path now carries the same check, and
`TRAINING_ENTRY_POINTS` parametrizes the ordering assertion over `fit` and
`tune`, RED-verified). Fixing a reviewer-confirmed defect is not continuing the
loop; opening a round 6 would have been, and none was opened. So the tree is in
the best state this run can put it in without another review round.

**What the maintainer decides.** The standing merge gate is Codex `APPROVE` + CI
green, and PR 1 does not have the first half. Three options, no recommendation
implied by their order:

1. **Merge on the round-5 record.** Every round-5 finding is closed and the
   reviewer's own clean list covers 10 items including the ones a merge would
   depend on. CI has not run yet — the branch is not pushed.
2. **Authorise a round 6**, waiving the stop condition explicitly. The trend
   (4, 4, 2, 1, 1) suggests it would find at most one more thing, and the last
   two were both entry-point completeness rather than new defect classes.
3. **Split PR 1.** The model-surface gate (`model.params`, tuning space,
   `export_code`) has been clean since round 2; the calibration surface (D3) is
   what rounds 3, 4 and 5 have been about. Shipping the first and moving the
   second to its own PR would close the reviewed part now.

**Also worth knowing, and the reason this is not only a PR-1 question:** the PR
grew past its planned scope twice, both times because review found a route the
plan had not enumerated — `export_code` in round 2 and `calibration.params` in
round 3 (D3). PR 1's scope in the plan was written from the issue text rather
than from a scan of the tree. **The later PRs in this run have scopes written
the same way**, so the same overrun is likely unless their scopes are re-derived
by scanning first. That is a decision about the plan, not about PR 1.
