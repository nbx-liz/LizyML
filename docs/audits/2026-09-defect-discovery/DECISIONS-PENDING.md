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

> **RESOLVED 2026-09-07 — by a fourth option the three below did not name.**
>
> The maintainer read this item and the rounds 4-5 monitor and directed **one
> further round, with the review target narrowed to the remedies for findings
> already raised**. That is neither "merge on the round-5 record" (option 1) nor
> "open a full round 6" (option 2): it repairs the procedural defect the monitor
> named — the stop had been self-certified by the party owning the deliverable —
> without re-opening surfaces five rounds had already covered.
>
> Round 6 returned **`APPROVE`**, the first on PR 1, with no blocking finding and
> no out-of-scope finding. Record: `results/pr1_codex_round6.md`.
>
> Before the round, the maker found and fixed a **DC1 defect inside the remedy
> itself**: 3 of the 21 classified methods have a required argument, so calling
> them bare raised at argument binding and the spy observed an empty list because
> nothing had executed. Two of the three were `load` and `export_code`. Disclosed
> in the round-6 prompt rather than left to be found; the reviewer then verified
> the fix by removing each argument entry and by tracing that all 21 bodies run.
>
> The reviewer also executed what the maker had not: the three `tune()` re-entry
> paths (resume, repeat, existing study), all refusing with zero Boosters, and
> five ways a new public callable can arrive on `Model` (ordinary, inherited,
> `classmethod`, `staticmethod`, class-creation), all failing as unclassified.
>
> **The merge gate is now met**: external review `APPROVE` + CI green. The
> analysis below is kept as the record of the state that produced the decision.

## State

PR **#275**, draft, pushed, 6 commits on `fix/phase3-pr1-lgbm-parameter-names`.
**CI 12/12 pass**, `MERGEABLE` / `CLEAN`. Full suite 2171 passed; `ruff check .`,
`ruff format --check .`, `mypy lizyml/` clean.

The standing merge gate is external review `APPROVE` + CI green. **Only the
first half is missing.**

## What happened

Blocking findings per round: **4, 4, 2, 1, 1**. Every one was reproduced before
being accepted and every one was real. Four were live production defects
(`Model.load()` refusing legacy artifacts, `fit(params=)` inert, `export_code`
ungated, `calibration.params` ungated) and two were specification statements
false of the code.

The reviewer never rejected the design. Every finding was "this is not covered",
and every one was covered. Severity fell monotonically:

| Round | Finding | Reach |
|---|---|---|
| 3 | `calibration.params` reached LightGBM unchecked | user config, live defect |
| 4 | the check ran after the whole outer CV | correct refusal, but a full training was paid first, and the spec said otherwise |
| 5 | the check covered `fit` but not `tune` | correct refusal, one entry point uncovered |

Rounds 4 and 5 each found the **previous round's remedy** incomplete. That
recursion has no fixed point — "round N's fix is unreviewed" is true of every
round including the last — so it cannot itself be the stopping rule. A stop
condition fixed *before* the outcome was known is what breaks it, and one was:
the rounds 3-4 monitor raised the flag, and it was written into round 5's prompt
before round 5 ran.

## The loop audit

`policy:loop-monitor` owns the question of whether a review loop should still be
running. Five monitors ran, each in a fresh read-only context:

| Monitor | Observed | Verdict | Recommendation |
|---|---|---|---|
| absolute | round 1 | `DELIVERABLE-FOCUSED` | `continue` |
| relational | 1-2 | `CONVERGING` | `redirect` |
| relational | 2-3 | `CONVERGING` | `redirect` |
| relational | 3-4 | `CONVERGING` | `continue` |
| relational | 4-5 | `CONVERGING` | **`take-stop-condition`** |

**The rounds 4-5 monitor was run late, and that is a procedural defect worth
recording.** The loop was closed and escalated first; the monitor the policy
places outside the loop was spawned only after the maintainer asked what it had
said. Its finding on that point, verbatim in substance: the stop was *sound in
content, defective in procedure* — the trigger genuinely fired (round 4 anchored
the check at one caller of a shared helper rather than in the helper, and
`tune()` is the second caller), but it was self-certified by the party owning
the deliverable. It adds that it would have raised the same flag unprompted.

Its grounds for stopping: two consecutive rounds of **zero periphery growth**
(the AST apparatus is unchanged since round 4 — 588 lines, neither scan file
touched), findings shrinking and production-real, and the one class still
producing findings now being caught by the maker without a review round.

## Known open, and fixable without a round

`TRAINING_ENTRY_POINTS` (`tests/test_calibration/test_calibration_param_names.py:151`)
is hand-written — `fit` and `tune`, 2 of `Model`'s 23 public callables — and is
checked against nothing. Its contents are correct today (`predict`, `evaluate`
and `export_code` do not train), so there is **no live defect**, but the axis is
open: a future public method that trains would not fail this test. It is the
same declared-fixture shape rounds 3-5 were about, and it can be closed by
deriving the set instead of listing it.

Both the main context and the monitor found it independently, after the loop
closed. That cuts toward stopping rather than toward a sixth round.

## What the maintainer decides

Three options; the order implies no recommendation.

1. **Merge on the round-5 record.** Every round-5 finding is closed, CI is
   12/12 green, and the reviewer's own clean list covers 10 items including the
   ones a merge depends on. Fastest path to PR 2.
2. **Authorise a round 6**, waiving the stop condition explicitly. The trend
   (4, 4, 2, 1, 1) and the fact that the last two were entry-point completeness
   rather than new defect classes suggest at most one more finding — but that is
   an estimate, not a guarantee.
3. **Split PR 1.** The model surface (`model.params`, tuning space,
   `export_code`) has been clean since round 2; rounds 3, 4 and 5 are all about
   the calibration surface (D3). Shipping the first closes the reviewed part
   now, and **unblocks PR 2**, which builds on the model surface.

Also open: how to handle the two gate issues this loop produced —
nbx-liz/claude-code-config#327 (the mechanized close-the-grammar review format
never fired) and #276 (the discovery audit stated PR 1's population in prose).

## What was missing from the run policy

The kickoff gate assumed `APPROVE` would arrive. Nothing said what to do when
real findings keep arriving and it does not. That gap is why this item exists,
and the next long-run kickoff should settle a round bound, or an equivalent
stop condition, alongside the merge gate.

### D6 — the plan's populations were rechecked; only PR 1's was prose

**PR** — · **Proposal** — · **Reversal cost** — · **Window** `open`

Filed as its own item because it is a finding about the *plan*, not about PR 1,
and because an earlier version of D5 asserted the opposite without checking.

PR 1 grew past its planned scope twice, both times because review found a route
the plan had not enumerated — `export_code` in round 2 and `calibration.params`
in round 3 (D3). D5 originally concluded that the later PRs were scoped the same
way and would overrun likewise. **That was written without measuring it, and it
is wrong.** Every population the plan declares, recomputed at `1d7c4e2`:

| PR | declared | measured | |
|---|---|---|---|
| PR 3 | 22 `(task, metric)` pairs in `_TASK_METRICS` | 22 | ✓ |
| PR 4 | `CVTrainer.fit` 7, `RefitTrainer.fit` 3, union 7 | 7 / 3 / 7 | ✓ |
| PR 5 | 3 `UnseenPolicy` values | 3 (`mode`, `nan`, `error`) | ✓ |
| PR 6 | 20 `ErrorCode` members | 20 | ✓ |
| PR 8 | 74 defaulted / keyword-only `__init__` params | 74 | ✓ |
| PR 9 | 92 proposals | 94 | explained below |
| PR 2 | enumerated from `Model`'s public signatures | 23 callables, 21 params | ✓ |

PR 9's difference is exactly the two proposals this run added — H-0092 in PR 0
and H-0093 in PR 1 — the population-grows-with-the-run effect already scheduled
for the reconciliation pass before PR 9 (C5).

**PR 1's route population was the one thing stated in prose rather than derived
by scanning, and it is the only one that failed.** That is checkable before the
fact: a population given as a sentence rather than as a derivation is the one to
distrust.

**No decision about the plan is needed on this account.** The item is here so
the correction is on the record, and because it is the evidence behind #276, the
gate issue against the discovery audit.

Recomputation script:
`docs/audits/2026-09-defect-discovery/instruments/plan_population_recheck.py`.


### D7 — PR 2 stopped at round 5 on a pre-registered condition

**PR** 2 (#278) · **Proposal** H-0094 · **Reversal cost** — · **Window** `before-close`

> **RESOLVED 2026-09-07 — option 2, one further round scoped to the remedies.**
>
> The same choice the maintainer made on PR 1's D5, where a scoped round
> returned the first `APPROVE` after five rounds without one.
>
> The rounds 4-5 monitor, spawned before that round and told the situation
> plainly, judged it **warranted on its merits rather than for the missing
> token**: asked whether a scoped round could still find anything, it answered
> by finding something — the round-5 remedy raised `ValueError` on an
> array-valued parameter, on the production entrypoint, with no duplicate
> spelling needed. **The maker had found the same defect in self-review minutes
> earlier**; two independent contexts converged on it without knowing of each
> other. Fixed before the round, and disclosed in its prompt:
> `results/pr2_monitor_round45.md`.
>
> The analysis below is kept as the record of the state that produced the
> decision.
>
> ---
>
> **The scoped round ran and returned `REQUEST_CHANGES` with three findings, all
> in the pre-round-6 fix** (`results/pr2_codex_round6.md`): a library scalar
> refused as a different value, broadcasting hiding two sequences of different
> lengths, and a declared exception-safe fallback that did not cover the
> comparison itself. All three fixed and RED-verified per finding; the
> pre-registration said no round 7 either way, and none was opened.
>
> **So PR 2 is back here, and the shape has changed.** Rounds 1-4 found defects
> that had shipped. Rounds 5 and 6 found defects the previous remedy wrote, and
> round 6's were all inside a 40-line helper written to fix round 5's. The
> subject has narrowed from "does the override reach the model" to "how do you
> compare two values", and the loop is no longer finding anything that predates
> it.
>
> Blocking per round: **1, 1, 2, 2, 1, 3**. Full suite **2334 passed**.
>
> **The options are the same three, with one changed weight.** Option 2 has now
> been tried: it cost one round and found three real defects in the newest code,
> which is an argument both ways — the round paid for itself, and it also shows
> the newest code is where the risk now lives. What the record cannot tell you
> is whether a *seventh* round would find three more in the fix for these three;
> the honest answer is that rounds 5 and 6 both did.
>
> The recommendation is now **option 1 (merge on this record)**, changed from
> the previous "option 1, with option 2 as the maintainer's call": the scoped
> round has been taken, the deliverable's own path has had six independent
> passes, and the remaining findings are in a helper whose whole job is one
> comparison, tested against 25 labelled inputs in both directions.
>
> ---
>
> **Round 7 (2026-09-07): the maintainer directed one more round; the monitor
> before it returned `DRIFTING`, the first non-converging verdict in this run.**
>
> The monitor rejected the series being reported: **6 of 10 findings had been
> written by the loop**, and **no round since round 4 had found a defect that
> predates the PR**. It showed the next generation already at HEAD, read-only,
> and concluded that an open value domain cannot be exhausted by review rounds —
> what closes it is a bound. It redirected the round to the one shipped surface
> six rounds never touched: persistence and export of an overridden fit.
>
> Adopted in full. Before the round: the value helper was bounded by
> construction (identity first, and a floor that answers "the same" when nothing
> can analyse the pair); two further self-authored defects were found and fixed
> (`__len__` raising anything but `TypeError`; a DataFrame comparison iterating
> over column labels, so two different frames read as equal); and the
> persistence axis was measured and pinned — it is **correct**: the override
> reaches the artifact and the generated project, and does not survive a load.
>
> **Round 7 found two more, both in code written in rounds 5-7** — a truth-value
> conversion that still let ordinary exceptions escape a function declared not
> to raise, and an export test that passed with `export` replaced by a no-op.
> Both fixed and RED-verified. `results/pr2_codex_round7.md`.
>
> **The pre-registered stop condition tripped, and the reviewer closed the loop
> itself**: "Both findings concern rounds 5–7, so the stated stop condition
> closes this review loop; this verdict does not initiate round 8."
>
> Blocking per round: **1, 1, 2, 2, 1, 3, 2**. Full suite **2349 passed**.
>
> **The recommendation is unchanged and now has four rounds of evidence behind
> it: option 1.** The loop has not found a defect predating this PR since round
> 4. Rounds 5, 6 and 7 each found defects in what the previous round wrote, and
> round 7's second finding was in a test written in the same pass as the round
> it was meant to close. Another round is expected to find something — in the
> code that round produces. That is not a reason to keep going; it is the
> monitor's DRIFTING verdict stated as a prediction.
>
> If the maintainer prefers a further round anyway, the honest framing is that
> it buys another generation of the same, not a closer approach to a clean
> verdict. **Reopening is a decision only the maintainer can make; this run will
> not open round 8.**
>
> ---
>
> **RESCINDED 2026-09-07 by the maintainer. The stop condition above no longer
> applies to this PR.** The instruction is to run PR 2 on the standard applied
> to PR 1 — **until the external reviewer returns `APPROVE`** — with the stated
> reasoning that not obtaining `APPROVE` is itself evidence that real problems
> remain in the fix code. The record supports that premise: all twelve findings
> were reproduced.
>
> The rounds 6-7 relational monitor (`results/pr2_monitor_round67.md`) returned
> **`CONVERGING` / `redirect`**, reversing its predecessor's `DRIFTING` with a
> measurement, and answered the reachability question directly: PR 1's round 6
> partitioned an **enumerable** population, PR 2 was enumerating an **open** one,
> and round 7's remedy closed PR 2's domain by construction — so round 8 can be
> PR-1-round-6-shaped. It also named why three consecutive rounds each found
> something: each remedy ships a *declaration* verified by a hand-written table,
> and the next round finds the gap between declaration and verification. **On
> that method, "run until APPROVE" manufactures its own next finding.**
>
> Its prescription — quantify each declaration over its whole population, the
> DC7 durable repair — was adopted in full before round 8: two instruments, plus
> three self-found defects fixed ahead of the round. The third, found while
> checking the instruments, is round 7's finding one step further along the same
> path: `hasattr(element, "__bool__")` **invokes** the attribute, so an element
> whose `__bool__` is a property that raises escaped a function declared not to
> raise. The check now reads the type's dictionaries, which runs no caller code
> and is the more accurate question anyway. Every expression in `values_differ`
> that touches a caller's value is now inside a `try` or reads the type without
> invoking it — a property of the shape rather than of the cases anyone
> remembered.
>
> Also corrected in the instruments themselves: the exporting-test population is
> now matched on AST **calls** rather than source substrings (a marker also
> matches a docstring — DC2 inside an instrument built to catch DC1), it is
> defined by the tests that **write** rather than the ones that read (round 7's
> own bad test read nothing, so a reader-side rule would have missed it), and a
> test whose signature this instrument cannot supply is now refused loudly
> instead of having its `TypeError` absorbed as "the test noticed" — DC1 in the
> instrument written to catch DC1.
>
> Full suite **2390 passed**. Round 8 is scope-limited to the round-7 remedies
> and these instruments, with rounds 1-6 surfaces out of scope and no new region
> admitted. **It is not pre-registered as the last round**; the standard is
> `APPROVE`.
>
> ---
>
> **Round 8 ran and returned `REQUEST_CHANGES` with four findings — the most of
> any round in this run** (`results/pr2_codex_round8.md`). All four reproduced,
> all four fixed. **Two of them were in the instruments the rounds 6-7 monitor
> prescribed**, and one was a production correctness defect: a `DataFrame` with
> integer column labels was reported equal to a different one, which is round
> 6's defect through the second guard written to exclude it.
>
> That finding was answered by **removing the elementwise step** rather than
> patching it a third time. It was the only step whose correctness depended on
> what iterating an arbitrary object yields; every version of it was a hypothesis
> about object structure that the next round refuted. All 104 cases in the file
> passed unchanged after the removal, so nothing in the table had depended on it,
> and the cost — two arrays with equal numbers under different dtypes are now
> reported as differing — is stated in the docstring rather than discovered later.
>
> Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4**. Full suite **2403 passed**.
>
> The monitor's prediction held for an eighth round, including for the apparatus
> built to end the pattern. The rounds 7-8 relational monitor runs before round
> 9 and must be given the `4` unsoftened, with one question: **after these fixes,
> is any declaration in scope still verified by a table rather than by a
> construction or a stated limit?**
>
> ---
>
> **The rounds 7-8 monitor answered `DRIFTING` with a measured denominator**
> (`results/pr2_monitor_round78.md`): the forwarding path — the declared
> deliverable — **has not changed since before round 6 was reviewed**, while the
> apparatus now runs 2144 lines against 146 lines of helper. It named three
> declarations still carried by hand-written tables; all three were reproduced
> and repaired **in place**, and the repair of the third immediately exposed a
> fourth instance (a constant tuple folded into the code object, so that case had
> never compared anything either). Its constraint on round 9 was adopted: a
> finding in existing apparatus is repaired in place or that apparatus is
> deleted; no new module, generator or scanner as a remedy.
>
> **Round 9 returned `REQUEST_CHANGES` with three findings**
> (`results/pr2_codex_round9.md`), **two of them in the same two constructs round
> 8 found**. One was a production defect again: `repr` may return a `str`
> subclass, so comparing the printed forms with `!=` handed the decision back to
> the caller's object at the step that exists to escape it — the fallback
> returned a list and both refusals fired on a pair that printed identically.
>
> The exporting-test scanner was **deleted** rather than taught a fourth
> spelling. After `writer = model.export` and `getattr(model, "export")` came
> `getattr(model, name)`, and after that `operator.methodcaller`,
> `functools.partial` and `Model.__dict__`: "every test that exports" is a claim
> no AST scan can deliver. It is replaced by four names, hand-maintained, saying
> so, with the cost stated — nothing detects a new exporting test that is not
> added. **That is the second contraction in two rounds**, and both removals took
> out a claim no implementation could keep.
>
> Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3**. Full suite **2402 passed**.
>
> The rounds 8-9 relational monitor runs before round 10, and is to be told
> plainly: apparatus share 2/3, both in the constructs round 8 had already found,
> and the remedy deleted one of them. Its question: **does anything left in scope
> make a claim over an open population?**
>
> ---
>
> **The rounds 8-9 monitor answered `CONVERGING`** (`results/pr2_monitor_round89.md`),
> the first such verdict reached over a measurement rather than an argument: the
> periphery **shrank 64 lines** after growing 431, findings fell 4 → 3, and both
> apparatus findings landed in constructs now deleted or structurally repaired.
> It found the last two claims a source scan could not keep — the
> `SMART_PARAM_TARGETS` closure and the writer set — and both were reproduced and
> closed by replacing a parse with an observation: run the resolvers, and ask the
> class.
>
> **Round 10 returned `REQUEST_CHANGES` with two findings**
> (`results/pr2_codex_round10.md`), and **neither was a production defect**. Both
> were claims the apparatus made about itself that exceeded what it does: the
> smart-parameter observation supplied `num_leaves_ratio` without the branch that
> reads it and overwrote its own declared `feature_weights`, so either activation
> could go inert with both tests green; and `_probe` called "failed after
> reaching a writer" evidence of artifact inspection, which ordering alone cannot
> establish.
>
> The remedy was subtraction plus one property: prerequisites declared, declared
> values actually used, the expected refusal asserted instead of absorbed, both
> declarations cut back to a *bounded set of executions rather than a closed
> input domain*, and the verdict renamed `failed-after-writing` for what is
> observed. One new assertion — that every activation changes what the resolvers
> produce — which is the property both faults violated at once.
>
> Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2**. Full suite **2409
> passed**.
>
> The rounds 9-10 relational monitor runs before round 11.
>
> ---
>
> **The rounds 9-10 monitor returned `DRIFTING` and refused the question it was
> asked** (`results/pr2_monitor_round910.md`), which was the most useful thing
> any monitor in this run has done. Asked whether round 10's clean production
> result meant the deliverable was finished, it answered that **the record cannot
> distinguish that from "not yet reached", because rounds 6, 8, 9 and 10 were
> each scoped to the previous round's remedies** — and that round 7, the one
> round pointed elsewhere, found production defects. It named the one observation
> that settles it and redirected round 11 to take it: unscoped over the
> deliverable path.
>
> It also stated what the scoping does to the maintainer's premise: *"no APPROVE
> means real problems remain in the fix code" is not tested by remedy-scoped
> rounds; findings in apparatus written two commits ago are not evidence about
> the fix.*
>
> **Round 11 ran unscoped and returned three findings, all in production**
> (`results/pr2_codex_round11.md`), two of them on the merge path itself:
>
> 1. **tuning evaluated different parameters from the ones it selected** — the
>    trial merge was the fourth seam and the only one still merging by spelling,
>    so trials trained at the config's value while the study recorded the
>    trial's, and the fit afterwards used the recorded one;
> 2. **the same-layer duplicate refusal had no caller for `model.params`** —
>    declared in decision 6, wired for `fit(params=)` only, so a config with both
>    `learning_rate` and `eta` sent both to `lgb.train`;
> 3. **equal arrays under two spellings were refused** — the cost round 8 wrote
>    into a docstring, executed on the production entrypoint. Acknowledging a
>    cost does not satisfy the requirement not to refuse valid input.
>
> All three fixed, recorded as **H-0094 decision 7**, with the Change Gate
> measurement for the new refusal: `Firing rate: 0/811 of pre-existing configs
> carrying model.params`.
>
> Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3**. Full suite **2417
> passed**.
>
> **This is the measurement the run was missing.** Four consecutive
> remedy-scoped rounds found nothing in production; one unscoped round found
> three. The maintainer's premise is confirmed on the deliverable path, and the
> narrowing of rounds 8-10 was itself the drift the loop monitor exists to catch.
> **Round 12 stays unscoped** — on this record, only an unscoped `APPROVE` means
> anything under the maintainer's standard.

## State

PR **#278**, draft, pushed. **CI 12/12 pass** on the previous head and re-running
on this one. Full suite **2267 passed**; `ruff check .`,
`ruff format --check .`, `mypy lizyml/` clean.

The merge gate is external review `APPROVE` + CI green. **Only the first half is
missing**, as on PR 1.

## What happened

Blocking findings per round: **1, 1, 2, 2, 1**. All seven reproduced, all seven
on the path `fit(params=)` → `lgb.train`, and the records carry each one with
what was run: `results/pr2_codex_round[1-5].md`.

Four were live production defects that had shipped — the override overwritten by
smart resolution, the merge keeping two spellings, an objective alias skipping
the task-compatibility check, and the config surface matching literal names.
Three were introduced by a previous round's remedy, which is why the loop was
watched: rounds 1→2, 3→4 and 4→5 each found a defect the previous stage wrote.

**The stop is pre-registered, not chosen after the fact.** The rounds 3-4 monitor
supplied the condition, phrased on authorship rather than on surface because the
previous pre-registration had measured the wrong axis, and the main context
adopted it verbatim before round 5's verdict was seen. Round 5's finding was
round-4-authored, and the reviewer independently said the same:

> The adopted stop condition is triggered: fix this defect, then hand PR 2 to
> the maintainer rather than proceeding to round 6.

Both were done: the finding is fixed and RED-verified, and no round 6 was opened.

## The options

**1. Merge on the round-5 record.** The gate's substance — a reviewer who
executed everything it reports, a fix for every finding, CI green, and a stop
condition set from outside the loop and honoured — is met; the `APPROVE` token
is not. This is the same disposition PR 1 took at D5 before the maintainer
directed a scoped round.

- **PRO:** every finding is fixed and RED-verified; the deliverable's own path
  has had five independent passes over it; three surfaces found along the way
  are filed rather than absorbed (#277, #279, #280); the remaining risk is on
  surfaces this PR does not change.
- **CON:** no `APPROVE` token, and three of the seven findings were
  remedy-introduced, which is the pattern that argues the next round would find
  something too.

**2. One further round, scoped to the remedies** — what the maintainer chose on
PR 1, where it returned the first `APPROVE` after six rounds.

- **PRO:** the round-5 remedy is small and self-contained (one comparison and
  its tests), so a scoped round is cheap; it repairs the one procedural gap —
  the last fix is unreviewed.
- **CON:** it reopens a loop that has been stopped on a condition set from
  outside it, and the pre-registration said the loop ends here.

**3. Split the PR** — ship the forwarding and the name checks, defer the
identity work to its own PR.

- **PRO:** the identity work grew past the plan's file list and carries the
  user-visible behaviour changes.
- **CON:** the forwarding **does not work** without the identity merge — that
  was measured, not argued. Splitting ships an override that an alias defeats,
  which is the defect #264 reports.

## Recommendation

**Option 1**, with option 2 as the maintainer's call if the missing token
matters more than the missing round. The reason is the same one that decided
D5: the loop was stopped by the party outside it, on a condition fixed before
the outcome was known, and the thing it named was fixed rather than argued away.

Option 3 is not viable on the measurement.

### Round 12 (2026-09-08) — unscoped, and the population closed

`REQUEST_CHANGES`, 2 findings, both `[P2]`, both reproduced before anything was
changed. A third was found afterwards by the main context, by enumerating every
place one parameter dict meets another in `lizyml/`.

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2 (+1)**. Full suite
**2445 passed**.

**What discriminates this round from round 5.** Neither reviewer finding is in
code round 11 wrote. Finding 1 lands in the round-11 fix *file* but predates the
step round 11 added — before the `tolist` conversion existed, the same pair went
to the printed forms and also differed. Finding 2 is a downstream merge this PR
gated but did not write, and the reviewer said so unprompted. So this is not the
authorship pattern the maintainer rescinded; it is the deliverable surfacing
defects that were already in it.

**The third finding is #280, and is not implemented.** The smart-managed
refusal is wired to `fit(params=)` only. Measured over the whole config surface
(smart parameter x native name it writes x spelling LightGBM accepts): **3 of 18
refused, 12 send two spellings to `lgb.train`, 3 are overwritten outright**.
That defect is already recorded in BLUEPRINT §14.4 and already filed as #280,
open for the maintainer because `config/` cannot reach the alias table under the
layer rule. What was fixed here is the part that is this PR's own: the
docstring claimed "refused at parse time for three of the five", which is true
of the smart parameters and false of the surface. The measurement went to
BLUEPRINT and to the docstring.

**One more of the same class is recorded and not fixed**, for the same reason:
`calibration.params = {"min_data_in_leaf": 7}` trains at `ceil(n x 0.01)` under
every spelling, because the calibrator writes that key unconditionally from its
always-present ratio default. Identical before and after this change.

**The seam population is now closed, not sampled.** 24 dict-merge expressions in
`lizyml/`; 12 where dicts from different sources meet; every one executed and
recorded in H-0094 decision 8. This class produced a finding in most rounds, so
enumerating it was cheaper than meeting it again.

**Round 13 stays unscoped**, for the reason round 12 did.

#### The rounds 11-12 monitor, and what changed because of it

`CONVERGING` / `continue`. It measured the periphery-to-deliverable ratio as
flat (~29:1 in both rounds, docstrings stripped) rather than accreting, and it
**tested** the authorship question instead of adopting this context's reading —
executing `values_differ` from before and after round 11 and confirming round
12's finding 1 predates the step round 11 wrote.

It also raised three objections to the seam table. All three were acted on
before round 13, following the rounds 10-11 precedent that probing a named layer
beats leaving it for the next round to find:

1. **The scan could not see the construct its own findings lived in** —
   `d[k] = v` was not in the declared set, and two of round 12's three defects
   live there. The set is widened; candidates 24 → 48.
2. **The seam it named is real, reproduced, and fixed.** Two `category: model`
   dimensions spelling one LightGBM parameter both reach `lgb.train`; the
   non-canonical one is sampled and optimised over **without affecting any
   trial**, and `best_model_params` records the dead value. Third layer with no
   caller for the same-layer rule. `Firing rate: 0/69` of pre-existing spaces.
3. **The instrument was not shipped**, so the table could not be regenerated —
   DC3 by this repository's own rule. Shipped, with its limits stated.

Blocking per round now reads **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2 (+2 found by
the main context)**. Full suite **2447 passed**.

Its prediction is recorded rather than adopted: the record predicts at least one
finding in round 13 and does not predict `APPROVE`. The seam it called unprobed
is no longer unprobed.

### Round 13 (2026-09-08) — unscoped

`REQUEST_CHANGES`, 3 findings, all production, all reproduced before anything
was changed. Plus one non-blocking finding this round **invited**: the prompt
asked the reviewer to break the seam-enumeration claim, and it did.

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3**. Full suite
**2474 passed**.

**None of the three is in code round 12 wrote.** Findings 1 and 3 are
pre-existing readers this PR exposed by making aliases and forwarding work;
finding 2 is in PR-authored code but the case predates round 12's step. The
authorship pattern the maintainer rescinded still does not fire.

**What the record now shows about the shape of the loop.** Round 13 is the
**third consecutive round finding the next equivalence class in one function**
— round 11: dtype; round 12: container; round 13: text grammar. That is the
open-grammar shape DC1's own note warns about, and it is the thing to watch
rather than the round count. The fix is written to close it rather than chase
it: LightGBM's own serialiser was read (`_param_dict_to_str` joins **every**
sequence the same way, whatever the parameter is called), the rule applied is
that uniform one, the test executes the serialiser, and the one form not covered
— nested grammar — is stated in the code and pinned by a case rather than left
to be found.

**The other two findings opened a construct the seam scan does not cover**: not
"one dict meets another" but "a user-spelled dict is read under one spelling"
(finding 1), and "LizyML and LightGBM each own a name for one parameter"
(finding 3). Both populations were enumerated and executed — 4 literal reads, of
which 1 was live; 2 `training.*` controls, both broken, in opposite directions.

**The seam-enumeration claim is retitled, not repaired.** "The population is
enumerated" was asserted twice and falsified twice within a round of being made.
The instrument now says it generates *candidates*, that the table asserts only
what was executed, and that calling a scan over an open space a closure is the
DC5 this run keeps finding in other people's declarations.

**Round 14 stays unscoped**, for the reason rounds 12 and 13 did.

#### The rounds 12-13 monitor — the first `redirect` of this run

`CONVERGING` / **`redirect`**, and it earned the redirect by falsifying a claim
round 13 had just made. Decision 9-2 said it had closed the equivalence-class
grammar rather than chasing it; the monitor executed the other three of the four
types LightGBM joins and found them still refused, with the serialiser producing
the byte-identical wire string for each pair. Re-executed here before adopting:
3 of 4 pairs refused.

Adopted in full before round 14 opened. `_wire_elements` now answers the
question for every type, the test uses the serialiser itself as the oracle over
the whole type set, and the two exclusions (`set`, `None`) are written as
judgements with their reasons and pinned by cases rather than left looking like
the gap they were.

The periphery measurement is the reason the verdict is still `CONVERGING`:
production grew (+72 to +94 code-only lines) while the periphery more than
halved (1262 to 516). It declined `take-stop-condition` because round 13's
findings were not in round 12's code, which matches this context's own reading.

**The operational lesson, from two instances one round apart**: the seam claim
and the grammar claim were both asserted as closed and both falsified within a
round. Asserting from measurement is right; **when the claim is "applied to the
whole set", the execution has to cover the whole set.**

Full suite **2498 passed**. Round 14 stays unscoped.

### Round 14 (2026-09-08) — unscoped

`REQUEST_CHANGES`, **1 finding**, `[P1]`, production, reproduced before it was
changed. Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1**. Full
suite **2500 passed**.

**The lowest count since round 5, and the first single-finding round since
then.** But the count is not the interesting part. The finding was produced by a
question the loop had not been asking until the rounds 12-13 monitor supplied
it: *where this diff claims a set, check whether it executed over that set.* The
round-14 prompt put that question to the reviewer explicitly, and the one finding
is an instance of it — a comment claiming a check "covers every input at once"
when it covered three inputs of four.

**The finding is the third instance of one shape**: a rule declared for every
layer and wired to some. Round 11 found it for `check_duplicate_identities`; the
decision-8 addendum found it for the search space; round 14 found it for
`check_training_managed_overrides`, again on the search space, because trial
parameters overlay after `_merge_params` runs. Each was reproduced, each was
wired, each firing rate measured 0 over the pre-existing population.

**A second thing worth recording about this round.** The reviewer's "checked and
clean" section independently re-executed round 13's first fix (export metadata
across `metric` / `metrics` / `metric_types`) rather than taking the record's
word, and stated plainly what its read-only constraint prevented it from
establishing — disk export, generated-project execution, and a real `load()`
followed by fit. Those were run here. That is the first round where the clean
section carried its own executed evidence at that level.

**Round 15 stays unscoped**, for the reason rounds 12-14 did. Before it, the
rounds 14-15 monitor gets the pattern unsoftened: three set-claims falsified in
three rounds (seam constructs, serialiser types, "every input at once"), each
within a round of being made, and each found by a different party — a monitor,
a monitor, then the reviewer once the question was put to it.

#### The rounds 13-14 monitor — and a correction to the round-14 entry above

`CONVERGING` / `redirect`, the second consecutive redirect, and the second
consecutive monitor to find something this context had got wrong.

**The correction, first, because the entry above understated it.** `git show
92e3d51` confirms round 13's fix commit authored both
`check_training_managed_overrides` and the comment claiming it "covers every
input at once". So **round 14's finding is in code round 13's fix wrote — D7's
authorship condition plainly fired, for the first time this run.** The round-14
entry above called it "the third instance of one shape" and did not say that.

The condition stays **rescinded** by the maintainer, so it does not stop the
loop. But recording it unsoftened when it fires is this run's own standard, and
that was not done. It is recorded now.

**Question (a) — the one population that was prose.** The round-13 literal-read
enumeration said "a grep returns four candidates" with the pattern nowhere
recorded, so it could not be re-run, while the write-direction scan has been
shipped with a positive control since H-0093. Both declarations the monitor
asked for are now shipped, and writing them was not a formality:

- the 5x5 refusal grid caught **three cells marked wired with no executed
  input**;
- the read-direction scan caught **two undeclared reads and two stale
  allow-list entries** — because the list had been written by reading rather
  than by running the scan, which is the same error that produced the round-13
  enumeration it replaces.

**Question (b) — converging, or mining a blind spot.** Its argument: the class
is a finite grid, rounds 11, 12 and 14 each drained one cell, the two cells left
are already open with issues and measured rates (#279, #280), so the loop is
draining a nearly-full grid rather than mining a blind spot. Adopted, and turned
into an executable table rather than left as an argument.

**One part of the redirect declined.** It recommended opening round 15 *scoped*
to the two new declarations. Rounds 6, 8, 9 and 10 were each scoped to the
previous round's remedies and each found nothing in production — a result this
run established was produced by the scope. The precedent for a monitor's named
work is to handle it **before** the round and leave the round unscoped, which is
what turned two previous monitors' named layers into fixed defects instead of
next-round findings. **Round 15 is unscoped.**

Full suite **2529 passed**.

### Round 15 (2026-09-08) — unscoped

`REQUEST_CHANGES`, 2 blocking + 1 non-blocking, all reproduced before anything
was changed. Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1, 2**.
Full suite **2533 passed**.

**The prompt asked the reviewer to attack the two new executable declarations,
and it did.** Finding 2 falsified one of the refusal grid's own `n/a`
rationales: `tuning best_model_params x check_duplicate_identities` claimed the
overlay was "overlaid by identity into a checked dict", and `overlay_params`
checks the overlay against the layer **below** it, not against itself. A
restored `best_model_params` naming one parameter twice sent both spellings to
`lgb.train`.

That is the declaration doing its job. The harness requiring every `wired` cell
to have an executed input is what forced the correction to be real rather than a
reworded reason.

**Finding 1 is round 13's check failing from a second direction**: it read
`cfg.training.early_stopping.enabled` while the trainer takes its patience from
a tuning result when one supplies it. One definition now serves both.

**The non-blocking finding was fixed rather than deferred.** `params_table()`
listed nothing for a parameter written under an alias, though the booster
trained at the overridden value. Reporting only — but it misreports the run on
the one path this change exists to make work, and it is the same literal-read
construct that cost the round-13 export defect. The read scan shipped last round
did not catch it, for a reason its own docstring declares.

**What the reviewer did that no previous round had done**: it inspected the
exception traceback for all 12 grid fixtures to confirm each reached its named
checker rather than failing for another reason, and it executed a real
`export()` / `load()` round trip — a path no reviewer had executed since round 7.

**Round 16 stays unscoped.**

#### The rounds 14-15 monitor — the diagnosis the verdict could not carry

`CONVERGING` / `redirect`, the third consecutive redirect, and the most
substantive monitor of the run. Both of its enumerations were executed here
before anything was acted on: one was a defect, one was clean.

**Its diagnosis, adopted as the finding of record.** It said the binary verdict
misses the actual reason fifteen rounds have produced no `APPROVE`: *the maker
ships a universal declaration each round without executing it over its set, and
the next round falsifies it* — four consecutive now (the seam scan, the
value-equality class, the grid cell, and the one it held). That is a statement
about this context's practice rather than about the code, and it is correct.

**Enumeration 1, a defect.** Decision 11 called
`effective_early_stopping_rounds` "the single definition" and said the trainer
and the gate "cannot disagree". The question has **four** readers and the claim
had been executed over two: `export_code` and `params_table` both read the
config alone. Executed — config patience 7, tuned 2, the run trained at 2, and
both reported 7. `export_code` generates a project meant to reproduce the
training, so it was generating one that trains a **different model**. All four
now share the definition.

The monitor also stated the consequence: that claim is in round 15's own fix
commit, so a round-16 finding there would fire D7 on round 15. Probing before
the round is what prevents that.

**Enumeration 2, clean.** It named this PR's blind-spot class — *a parameter
that reaches `lgb.train` correctly and is then outranked by a channel that is
not the params dict* — and said how to look: assert on what the booster did, not
on `booster.params`. Executed: `num_boost_round` is honoured under all seven
spellings with an empty dict, `categorical_feature` is honoured in index form
and fails **loudly** in `name:` form. Both are now pinned by tests, which is the
durable half of a clean result.

It declined to claim a third candidate because it could not rule out that the
mechanism was dead — restraint that is why its two actual claims were worth
executing.

**Its recommendation was adopted in full, including the scoping.** For the first
time a monitor's redirect recommended leaving the next round **unscoped** rather
than narrowing it, reaching that from the run's own record: probe-before-round
has twice turned a named surface into a fix instead of a finding, while scoping
a round has four times produced nothing.

Full suite **2535 passed**. **Round 16 stays unscoped.**

---

### D8 — PR 2 stopped at round 20 on a pre-registered condition, five rounds into one function

**Status: RESOLVED 2026-09-08 — option F (ingress normalisation), Proposal H-0095.
One item remains open: whether to merge PR #278 as it stands. See the end of this entry.**

#### What happened

Rounds 16 through 20 each returned a blocking finding, and **every one of them
was in `lizyml/core/value_equality.py`, in the code the immediately preceding
round's fix had written.** D7's authorship condition has fired five consecutive
times.

Before round 20 ran, this context registered the trigger and told the maintainer
about it; the rounds 18-19 monitor was shown it and endorsed it:

> The pre-registered round-20 trigger is appropriate: another same-function
> authorship failure after this methodological intervention would directly
> undermine the reason for continuing.

Round 20 found two. The condition is met and this is the decision point.

#### What is not in question

- **Every finding across all twenty rounds was real and reproduced**, including
  round 20's. None was apparatus, and none has been waved away.
- **The original defect is fixed and executed.** `Model.fit(params=...)` was
  documented as overriding `model.params` and forwarded nowhere; it now reaches
  `lgb.train` on every path, pinned end to end.
- **The merge gate's second half has been satisfied since round 15**: CI green,
  full suite 2898 passed, `ruff`, `ruff format`, `mypy` clean, the shipped
  lifecycle grid exits 0.
- **The last two monitors returned `CONVERGING` / `continue`**, and the second
  judged the recent work to be *"a specification being written down"* rather than
  a loop that cannot terminate — while refusing to call it domain closure.

#### What is in question

The last five rounds have been spent on one Layer 0 function whose job is to
decide whether two spellings name one parameter value. Each round's fix has been
correct about the case it was shown and incomplete about a neighbouring one, and
the neighbours are supplied by an **open** domain: what an arbitrary Python
object can do to `format`, `str`, `__eq__`, `__len__`, `__class__`, `tolist`.

Round 19 changed the method — an oracle relation against LightGBM's own
serialiser, and a hostile population derived from Python's dunder list rather
than written down. Round 20 still found two, and one of them was the new
relation contradicting an older contract of the same module.

#### The options

**A. Continue scoped.** Open round 21 against the round-20 remedy, as rounds 18,
19 and 20 were opened. Precedent: rounds 18 and 19 each reduced the blocking
count and each produced a substantive clean result alongside its finding.
Against: this is the fifth iteration, and the trigger for stopping was set
precisely so that it is not extended by default.

**B. Narrow this function's contract** — the option named when the trigger was
registered, and the one the monitor called *"appropriate, because it revisits the
obligation generating the repairs."* Reverse the round-13 admission: refuse a
mixed text/sequence pair unless the two wire forms are identical, rather than
comparing elementwise. Fewer admissions, a much smaller surface for a hostile
object to act on. **The monitor's caveat is adopted and repeated here: this
trades compatibility for a narrower comparison contract, and calling it a
"near-zero attack surface" was this context's phrase and is not proven.** It
would also need its own Change Gate treatment, because it removes an admission
that round 13 added deliberately after measuring it.

**C. Take the stop condition and merge without `APPROVE`.** The declared
deliverable is delivered and verified; the residual is a hardening exercise on
one function against adversarial inputs no measured configuration produces. File
the remaining hostile-input classes as issues, close PR 2, proceed to PR 3.
Against: the maintainer's standing instruction is that the absence of `APPROVE`
is itself evidence of remaining problems, and rounds 16-20 do not contradict
that — the findings were real.

**D. Something else.** Every previous stop condition in this run was resolved
with an option nobody had listed, twice — round 5 (a scoped round) and round 17
(hold and ask). That is the empirically most likely outcome, and the reason these
three are written as a starting point rather than a menu.

#### What is ready either way

Round 20's two findings are already fixed, RED-verified and committed, because
stopping the loop is a decision about the next round and not a reason to ship a
known defect.

#### 決定（2026-09-08）: **F — 入口で正規化して領域を閉じる。提案は H-0095。**

管理者の判断は **F**。選択肢 A-E のいずれでもなく、**Codex の評価が最後に付け足した
「誰も挙げていなかった選択肢」**である。この run で停止条件が発火したのは 3 度目で、
**3 度とも、事前に列挙した選択肢の外が選ばれた**（round 5 = 範囲限定ラウンド、
round 17 = 保留して確認、round 20 = 入口正規化）。

##### 評価の経緯

全選択肢を Codex（`gpt-6-astra`, effort **medium** — レビュー 20 ラウンドは全て
effort `low` で走っていた）に評価させた。結果は **E > D > B > A > C**、推奨は E。

**その E を、こちらで実行して却下した。**

```
pair                      wire A     wire B    wire一致  現状
[1, 2]      vs '1,2'      1,2        1,2       True     admit
[1.0, 2.0]  vs '1,2'      1.0,2.0    1,2       False    admit  <- E なら誤拒否
(1.0, 2.0)  vs [1, 2]     1.0,2.0    1,2       False    admit  <- E なら誤拒否
0.5         vs '0.50'     0.5        0.50      False    admit  <- E なら誤拒否
```

E の中核は「wire form の一致を同一性の定義にする」ことだが、**wire form は正準形ではない**。
`_comma_form_matches` の docstring が既にそう書いており、**E は round 13 finding 2 の修正を、
その理由が書いてある行ごと元に戻す**。Codex は「E は B の互換性トレードオフを継承する」と
一般論では書いていたが、実測するとその範囲は `0.5` と `"0.50"` にまで及ぶ。

**評価そのものは高い価値があった。** Codex はこちらの事実誤りを 2 件訂正し
（provider protocol は 8 でなく **18** メソッド、`check_duplicate_identities` の呼び出しは
3 でなく **4** 箇所）、こちらが挙げていなかったリスクを 1 件挙げ（`_param_dict_to_str` は
private、`pyproject.toml` は `lightgbm>=4.0` を許す）、こちらの診断の言い過ぎも 1 件突いた
（「直近の指摘は全て serialiser 再実装の不一致」は round 20 の 2 件目＝NaN 契約の矛盾を
含まないので文字通りには完全でない）。**そして最後に F を出した。**

##### F を選ぶ根拠

| | E | **F** |
|---|---|---|
| 領域を閉じるか | 閉じない（serialiser 内でユーザーのメソッドが走る） | **閉じる** |
| round 13 の誤拒否 | **再導入する**（実測済み） | しない |
| private API 依存 | `_param_dict_to_str`、バージョン幅未検証 | 不要 |
| 公開 protocol 変更 | 18 → 19 | 不要 |
| 実装場所 | `estimators/` へ移設 | **`core/` のまま** |
| 比較時と学習時で値が変わる危険 | 残る | **消える**（入口で凍結） |

**前提の訂正が 1 つ効いた。** 「Layer 0 = 標準ライブラリのみ」は**アーキテクチャの規則では
なく、`value_equality.py` が自ら課したもの**である。`ARCHITECTURE.md` の「依存ゼロ」は
*内部レイヤ*依存ゼロの意味で、Layer 0 の他モジュールは numpy も pandas も import している。
**numpy を型として名指すことは最初から許されていた** — duck typing は必要に迫られたもの
ではなく、それが開いた領域の原因だった。

##### 実測（変更ゲート）

```
Firing rate: 7/1430 of every parameter value the suite constructs
             (measured at head 1403ba8 over the full suite)
```

1423 件は受理集合の内側。**残る 7 件は rounds 16-20 が自分で構築した敵対オブジェクトのみ。**
計測器は `instruments/parameter_value_type_census.py` として出荷済み。

##### 残っていた 1 件の判断 — **決着済（2026-09-08）**

「PR #278 を今の状態でマージするか、H-0095 が着地するまで draft のままか」は
**管理者の指示で解消した: 「F を実装後、両方あわせて Approve を取得してください」。**
PR は draft のまま、**#264 本体と H-0095 を 1 本の PR として round 21 に出す**。
別々にマージしない。

##### 実装の記録（2026-09-08）

F を実装した。commits は `d6fa4de` (提案 Accepted) / `48a0801` (入口の正規化) /
`4161837` (wire 保存と閉じた領域の固定) / `835398e` (`value_equality.py` の縮小) /
`0f94389` (実装が訂正した提案の記録) / `e478ef5` (比較テストの書き換え) と、
自己レビューで見つけた 5 件の修正。

**実装が提案を 5 か所訂正した。** すべて実行して確かめた:

1. `1-D ndarray` → `.tolist()` は wire を変える（float16/float32、27 通り中 8 件）。
   要素位置のフォーマッタは `str` なので、変換も位置ごとに分ける。
2. `str` サブクラス → 厳密な `str` も wire を変える。**厳密な型一致**で受理し、
   サブクラスは拒否する。
3. 入れ子リスト（`interaction_constraints`）と **mapping**（metric entry、H-0065）を
   受理表に追加。mapping を落としていたら**出荷済みの設定形式を入口で拒否**していた。
   入口と出口で受理集合が異なるのが正しい。
4. `set` は**拒否**する。旧 `values_differ` が set を別値としていた理由
   （列は位置依存 / set に順序が無い）は入口正規化でも消えない — `list(set)` が
   リテラルのリストと一致するかはハッシュ順の偶然である。
5. リストの入れ子は**深さ 2 まで**。3 段目は Python の list repr で書かれ、
   正規化が wire を変える（`[[[np.float32(0.1)]]]`）。母集団が深さ 3 を生成して
   いなかったので性質テストが見られず、自己レビューで見つけた。

受け入れ基準 7（**#283**）は「**H-0095 では解決しない**」と決めた。正規化後も
`float` と `list` であり、admit するには「学習器にどちらを渡すか」を決める必要がある
＝ 別の Proposal（`allow`）。

計測: `Firing rate: 14/1518`（実装後、`normalise_params` を包んで実測。14 件すべて
この PR 自身のテストが構築した値）。フルスイート **7425 passed / 256 skipped**。

---

## D9 — 事前登録した停止条件が発火した（2026-09-08、PR 2 round 22 後）

### 何が起きたか

rounds 20-22 の関係監視は `CONVERGING` / `continue` を出したうえで、**自分の判断を
反証する条件を 2 本、事前に宣言した**:

- **(a) authorship** — `1147c98` / `051b2fa`（round 21 の修正）が書いたコードに
  blocking が出たら。
- **(b) DC1** — 4 surface のいずれかで受理された値が、`lgb.train` に届く bytes を
  呼び出し元の値が書くものと違えたら。

**round 22 の指摘は両方に当たる。** `_derived_numpy_scalar_types`（`1147c98` が書いた）が
`__module__`（呼び出し元が書ける属性）を信じており、`class Disguised(np.float64):
__module__ = "numpy"` が 4 surface すべてで `0.9` と書いて `0.1` で学習した。

**加えて round 22 は verdict を返していない** — provider 側のコンテンツフィルタで
実行が中断された。指摘はログに残っていた再現である。

### 形の評価（判断材料）

**「同じサイクルの引っ越し」に見える側:**
- rounds 16-20 = 比較が値を理解し損ねる → rounds 21-22 = **閉包の機構そのものが
  なりすまし可能**。**毎ラウンド、前ラウンドが閉じたつもりの門が 1 つ破られている。**
- 「これで終わり」という主張は 2 回して 2 回とも外れた（round 21 前、round 22 前）。

**「境界が動いて直った」に見える側:**
- 修正の方向が毎回**弱い判定 → 強い判定**へ動いている:
  `isinstance` → 厳密型一致 → **`vars(numpy)` による同一性**。
  最後のものには**呼び出し元が自称できる余地が無い**（numpy 自体を書き換える以外に）。
- rounds 16-20 は「もう 1 つ guard を足す」修正だったが、21-22 は**判定の根拠を
  変える**修正である。
- **round 16 以降の指摘はすべて、意図的に敵対的なオブジェクトを必要とする。**
  出荷済みの欠陥（#264 本体）は round 15 までに直り、固定されている。
  計測: **受理集合の外は 14/1518、全てこの PR 自身のテストが構築した値**。

### 選択肢

- **A: 範囲を絞って round 23 を開く** — 対象を `param_domain.py` の型導出と 2 段の
  防御に限定する。round 5 / round 18 でユーザー自身が選んだ第 4 の選択肢と同じ形。
  clean なら「閉包が安定した」証拠になり、続く非限定ラウンドで APPROVE への道が開く。
- **B: 非限定の round 23 を開く** — round 22 は verdict を返していないので、
  「round 22 をやり直す」に近い。
- **C: `APPROVE` を要求せずマージする** — #264 本体は直っており CI 緑。残る指摘は
  すべて敵対的オブジェクト由来。**選択肢 C の形なので明示承認が要る。**
- **D: 受入基準を「敵対的オブジェクトを除く」に変える** — 成果物の作者が受入基準を
  書き換えることになるので**推奨しない**（D8 で同じ理由により却下した形）。

### こちらの推奨

**A（範囲限定の round 23）。** 理由: 停止条件は「ループを止めろ」ではなく
「ループの外側の当事者に判断を渡せ」であり、この run で 2 度、範囲限定ラウンドが
正解だった。今回の指摘は 100% が「閉包の機構」に集中しているので、そこだけ開けば
5 ラウンド分の面を開き直さずに決着がつく。

**ただし停止の可否を決めるのはループの外側**なので、明示的な判断なしに進めない。

### 決定（2026-09-08）: **A — 範囲を絞って round 23 を開く**

管理者の判断は **A**。**この run で 3 度目の「範囲限定ラウンド」**であり、
round 5・round 18 と同じ形である（どちらもユーザー自身が出した第 4 の選択肢だった）。

round 23 の対象は **`lizyml/core/param_domain.py` の型導出と 2 段の防御**に限定する:

1. `_derived_numpy_scalar_types()` が `vars(numpy)` を読むこと（同一性であって自称でない、
   import 順に依存しない）
2. `type(value) in NUMPY_SCALAR_TYPES` / `type(value) is np.ndarray` の厳密型一致
3. `format(value, "")` を `.item()` の**前**に読み、変換後と比較すること
4. 上記 3 つの RED 検証が本当に赤くなること（この run で「テストが別の理由で緑」は 5 回）

**`APPROVE` はこのラウンドでは成果物全体の判定ではない** — clean なら閉包が安定した
証拠とし、続けて非限定ラウンドを回す。

### D9 の続き — 範囲限定ラウンドは clean にならなかった（2026-09-08 追記）

D9 は「**clean なら**閉包が安定した証拠とし、続けて非限定ラウンドを回す」と書いたが、
**clean でなかった場合の分岐を書いていなかった。**

実際に起きたこと: **round 23 は自分のスコープ項目 3 つすべてを反証した**（`in` は
同一性でない / `vars(numpy)` は書き込み可能 / 要素位置の門が未検証）。それでも
**round 24 を非限定で走らせた**。判断としては妥当だったと考える —— 3 件は同一の
機構に集中しており、修正の方向は「弱い判定 → 強い判定」で一貫していて、round 24 は
実際にその機構以外の欠陥（path と巨大 int）を出した ——
**が、その判断はどこにも記録していなかった。** rounds 23-25 の監視がこれを指摘した。
**指摘は正しく、ここに記録する。**

**そして D9 の判断材料が 1 つ古くなった。** 「round 16 以降の指摘はすべて意図的に
構築された入力を必要とする」は **round 24 で偽になった** —— `PurePosixPath` と
`10**5000` は普通に書ける値である。`Firing rate: 14/1518` は敵対的オブジェクトに
ついての測定であって、**受理集合の外にある普通の値については何も言っていない**。

---

## D10 — ループをどう終わらせるか（2026-09-08、round 24 後）

### 何が問われたか

24 ラウンド回って `APPROVE` が出ておらず、指摘ゼロのラウンドが 1 度も無い。
選択肢は 4 つ提示した: (1) 非限定 round 25 を回す / (2) いったん止めて H-0095 の
受理集合と消費者を提案として書き直してから再開 / (3) `APPROVE` 無しでマージ /
(4) PR 分割。

### 決定（2026-09-08）: **2 — 止めて契約を書き直してから再開する**

**根拠は「レビューが答えている問いが間違っている」ことである。** 現在の受入基準は
実質「この検証を破る値は存在するか」で、これは Python のあらゆるオブジェクトを渡る
全称命題であり、**開いた領域に対する「反例なし」は有限のレビューでは示せない**。
ラウンドを足しても、答えの出ない問いに対する試行回数が増えるだけである。

契約を「受理集合はこれ / 消費者はこれ / 敵対的呼び出し元はスコープ外」として確定
すれば、レビューは**破れるかどうか**ではなく**契約通りかどうか**を判定でき、
`APPROVE` が出せる対象になる。

### やること（3 つ、pr2_NEXT.md の推奨経路）

1. **受理集合を明示的に列挙する。** 今は 19 件の補正の副産物として存在しており、
   提案本体の受理表は 2 か所が既に古い。位置（scalar / element / member / mapping）
   × 厳密型の matrix として書き、**各行に固定しているテストを名指す**。
2. **消費者を全部数える。** 4 surface + `lgb.train` 2 サイト + `export_code`。
   ただし**走査を閉包と呼ばない**（`parameter_merge_seams.py` の docstring が自ら
   そう書いており、この run が他人の宣言に見つけ続けている DC5 がそれである）。
   閉じられるのは**要件のリスト**のほうである: 正規化後の値が満たすべき性質を
   列挙し、それぞれを**受理母集団全体の上で実行するオラクル**にする。消費者は
   各要件の根拠として名指す。
3. **敵対的な呼び出し元を明示的にスコープ外と宣言する。** プロセス内で numpy を
   差し替えられる相手を Python で防げないことは実証済みで、達成不能な宣言を書くのは
   DC7 である（この run で 3 度書き直している）。

### B / C / D は保留のまま

B（Codex 不可時に `policy:fresh-checker` をマージゲートとして認めるか）、
C（停止条件）、D（#283 の方針）は round 25 を開く直前にまとめて出す。
**C は round 25 を開くときに必要になるのであって、今は必要でない。**

### D10 の続き — B / C / D の回答（2026-09-08）

契約の書き直しが済んだ時点で 3 件まとめて聞き、以下の回答を得た。

- **B: 認めない。** マージゲートは **Codex `APPROVE`** であり、`policy:fresh-checker` の
  clean はこれを代替しない。Codex がコンテンツフィルタで落ちたときは**依頼の書き方を
  変えて Codex を通す**（round 24 が実際にそれで完走した）。fresh-checker は
  引き続き**発見の手段**として使ってよいが、ゲートではない。
- **C: authorship 条件（D7 と同じ）。** **round N の修正が書いたコードの欠陥が
  round N+1 で出たら停止**し、判断を仰ぐ。ラウンド上限は置かない。
- **D: 現方針を維持。** #283 は H-0095 では解決しない。open のままとし、
  `KNOWN_BOUNDS` の免除も残す。admit は振る舞いの拡大（`allow`）なので
  firing rate 付きの別 Proposal が要る。

---

## D11 — round 25 で停止条件（C）が発火した（2026-09-08）

### 何が起きたか

**新しい契約に対する最初のラウンドは `REQUEST_CHANGES` 3 件**（記録:
`results/pr2_codex_round25.md`）。**3 件とも監視が事前に定義した「契約の内側」の形**で、
「§1/§3 の外側の値が検査を破る」形（＝書き直しが効いていない形）は 0 件だった。

1. **UTF-8 エンコード可能性が要件に無い**（`contract` + 実装の穴）。孤立サロゲート
   `"\ud800"` が正規化・出口の表明・`json.dumps` オラクルを通り、`_c_str` と
   `write_artifacts` の両方で `UnicodeEncodeError`。**実行して再現済み。**
2. **述語が正規化と食い違う**（`deliverable-path`）。`10**5000` は
   `is_accepted` / `is_plain` / 出口の表明を通り、正規化は `CONFIG_INVALID`。
3. **母集団オラクルが型集合を覆っていない**（`periphery` + `contract`）。fixture の
   numpy 型はベタ書き 12 型で、導出集合から 5 型が抜けている。加えて §1 の値領域を
   「有限」と書いたのは誤りで、無限領域の**有限標本**である。

### 停止条件の判定

**指摘 2 は round 24 の修正 `68d4972` が作った乖離**である（`_written_or_refused` を
追加して正規化だけを狭め、`is_plain` / `is_accepted` は触っていない。`git show` で確認）。
**よって C が発火した。**

指摘 1 と 3 は authorship 再帰ではない（それぞれ H-0095 当初実装と、ラウンド間の
書き直しに由来する）。

### 判断材料として明示しておくこと

- **指摘 1 を修正すると受理集合がまた変わる。** 前回監視は反証条件を生かしている ——
  「**どの消費者のためであれ受理集合が再び変わったら DRIFTING**」。UTF-8 は消費者由来の
  要件なので、修正は `export_code` のときと同じ形になる。**ラウンドを続けるなら、
  次の監視はこれを DRIFTING と判定する可能性が高い。**
- **3 件はいずれも安く直せる**（述語に writability 検査を通す / UTF-8 要件とオラクルを
  足す / fixture の型軸を受理型集合から導出する）。難しいのは**直すこと**ではなく
  **どこで止めるか**である。
- **#264 本体（rounds 1-15）は約 14 ラウンド clean** で、`model.py` の最終変更は
  H-0095 の入口コミットである。

### 選択肢

1. **3 件を直して round 26 を範囲限定で回す**（レビュー対象＝指摘の修正のみ）。
   PR 1 で APPROVE に到達した形。
2. **3 件を直して round 26 を非限定で回す。**
3. **3 件を直して `APPROVE` を求めずマージする**（CI 緑・スイート緑を根拠に）。
4. **指摘 2 だけ直し、1 と 3 は Issue 化して繰り延べる**（ただし 1 は出荷済みの
   欠陥なので、繰り延べは弱い）。
5. **PR を分割する** — #264 本体を先にマージし、H-0095 を別 PR にする（D10 で一度却下）。

### D11 の続き — 反証条件を 1 つに固定する（2026-09-08）

rounds 24-26 の監視が**監視自身の欠陥**を返した: **同じ反証条件を 3 つの監視が
3 通りに書いている**（2325「2 つの `lgb.train` の外側の消費者」/ 2425「どの消費者であれ」/
2426「ラウンド開始時点の §2 の表に無い消費者」）。**毎ラウンド書き直される反証条件は
停止規則ではない。** よってここで固定する。

**固定した反証条件（以後の監視はこれを継承すること）:**

> **ラウンド開始時点の H-0095 §2 の表に無い sink を名指す指摘が出たら DRIFTING。**
> 表に載っている消費者の要件が実測で不十分だった、または §2/§3 が標本の上で不完全
> だった、という指摘は**契約の内側**であり、`REQUEST_CHANGES` でも CONVERGING。

この文言を採る根拠は監視が実行して確かめた: 指摘 1 の 2 つの sink
（LightGBM の `_c_str`、`artifact_writer` の UTF-8 書き込み）は、**round 25 開始時点で
既に §2 に載っていた消費者**に属する。`65dfdbe` の HISTORY 差分でも、増えたのは
**既存消費者に対する要件の行**であって**消費者の行は 0** である。
2425 の字義どおりの読み（「どの消費者であれ」）は自滅的でもある —— `export_code` を
採用した以上、その後のどんな精緻化も永遠に発火し、採用が無意味になる。

**残る唯一の開いた軸**は、§2 自身が「閉じられるのは要件のリストであって消費者のリスト
ではない」と宣言していることである。drift が入るとすればここだけである。

**round 26 の take-stop トリガ（監視が付けたもの、採用する）**: round 26 が
**`65dfdbe` が書いたコードの欠陥**を出したら、クラス単位の修復のあとの 2 連続
authorship 再帰であり、**round 27 を開かず止めて判断を仰ぐ**。

---

## D12 — round 26 で停止条件が 2 つとも発火した（2026-09-08）

### 何が起きたか

範囲限定の round 26（対象＝`65dfdbe` の 1 コミット）は `REQUEST_CHANGES` 2 件
（記録: `results/pr2_codex_round26.md`）。

1. **numpy の文字列要素が UTF-8 検査を迂回する。** `[np.str_("\ud800")]` が
   `["\ud800"]` に正規化され、**2 回目の正規化で拒否される**（冪等性違反）。
   検査漏れ自体は `65dfdbe` より前からあるが、**1 回目と 2 回目で答えが変わる不整合は
   `65dfdbe` が作った**。レビュアーはサロゲート 2048 個 × 12 構成を列挙している。
2. **`repr` の一致は「正規化が値を変えなかった」ことを示さない。**
   `np.printoptions(legacy="1.25")` の下で `np.int64(1)` と `1` は同じ `repr` になり、
   **変換されていない numpy 値が `is_accepted` / `is_plain` / 出口の表明を通る**。
   **`65dfdbe` が書いたコードの欠陥である。**

### 停止条件

**2 つとも発火した。**

- **C**（ユーザー決定の authorship 条件）: 指摘 2 が `65dfdbe` の書いたコードの欠陥。
- **rounds 24-26 監視の take-stop トリガ**: 「`65dfdbe` の欠陥が出たら round 27 を
  開かず止める」。同じ指摘で発火。

D11 で固定した**反証条件（§2 の表に無い sink）は発火していない** —— 2 件とも
`param_domain.py` の中で、形は契約の内側のままである。

### 判断材料

- **どちらも安く直せる。** 指摘 1 は numpy 要素のテキストを `_written_or_refused` に
  通すだけ。指摘 2 は「変わっていない」の判定を `repr` から**型の再帰的な保存**へ
  変えるだけ。
- **ただし指摘 2 は「こちらの修正が持ち込んだ」2 例目である**（round 25 指摘 2 →
  round 26 指摘 2）。**クラス単位の修復を入れた直後に、その修復自身が新しい偽陽性を
  作った**という事実が、監視が止めるべきだと言った理由である。
- **`repr` を選んだのは実装判断の誤りである。** 表示テキストは numpy の print option で
  変えられる＝**呼び出し元が参加できる比較**であり、この PR が 24 ラウンドかけて
  排除してきたもの（`is` を使う理由）と同じ穴を、値の比較側で開けた。
- **#264 本体（rounds 1-15）は依然 clean** で、`model.py` の最終変更は H-0095 の入口。

### 選択肢

1. **2 件を直して round 27 を範囲限定で回す**（同じ形を続ける）。
2. **2 件を直して `APPROVE` を求めずマージする。**
3. **PR を分割する** —— #264 本体を先にマージし、H-0095 を別 PR に切り出す。
   H-0095 は 12 ラウンド連続で「直すと次が出る」状態にある。
4. **H-0095 を revert し、#264 本体だけを残す** —— 領域閉包そのものを別提案に戻す。

---

## D13 — D12 の判断材料が誤っていたので取り直す（2026-09-08）

### 訂正

D12 では「2 件はどちらも安く直せる」「**毎回別クラス**」という前提で選択肢を出した。
rounds 25-27 の監視（`results/pr2_monitor_round2527.md`）が**その片方を実測で覆した**。

- `68d4972`（round 24 の修正）は `_written_or_refused` を `_plain_element` の
  **素のスカラー分岐**に入れ、その 1 行下の **numpy 分岐**を残した。
- `65dfdbe`（round 25 の修正）は `_encodable_or_refused` を scalar / path /
  mapping キーに通し、**同じ分岐をまた飛ばした**。
- **round 26 の指摘 1 がその分岐である。**

つまり **round 25 指摘 2 と round 26 指摘 1 は同じクラスの 2 連続**であり、
**そのクラスに当てた修復の形は「もう 1 か所を通す」を 2 回**である。
`7ee10fe` の「最後の位置」という言い方は**到達範囲の主張であって計測ではない**。
監視の推奨は **take-stop-condition**（＝この訂正の上で判断を取り直すこと）。

### 監視が同時に付記したこと

`_is_unchanged` は `normalise_value` の dispatch と `_holds_a_mapping` に続く
**3 つ目の構造列挙**で、**DC3 の形が再び存在する**。監視は 32 形＋冪等性を probe して
**今日は 3 者が一致している**ことを確認しているが、一致を保つ機構は無い。

### drift ではない（念のため）

D11 で固定した反証条件は発火していない。指摘は deliverable path に留まり、
`param_domain.py` は 3 つの修正すべてで変わっている。周辺の増加は計測上ゼロ。

### 選択肢

1. **「位置」を計測にする** —— 位置の母集団を**ソースから導出**し、
   **各位置で各門（writability / encodability / `_is_unchanged`）が適用されている**ことを
   テストで固定する。今回のクラス（「門を足したが 1 か所を通していない」）を
   **位置ごとの修正ではなく性質として閉じる**唯一の形。そのうえで round 27。
   —— 構造列挙が 3 つある問題（DC3）も同じ形で閉じられる。
2. **そのまま round 27 を範囲限定で回す**（D12 の決定を維持する）。
3. **PR を分割する** —— #264 本体を先にマージし、H-0095 を別 PR に切り出す。
4. **`APPROVE` を求めずマージする。**

### 追記 — 原因解析（2026-09-09、ユーザー指示）

D13 の 4 択を出す前に、**なぜ 26 ラウンドで `APPROVE` が出ないのか**を解析した。
全文は `results/pr2_why_no_approve.md`、生の計測は
`results/pr2_duplicate_tolerance_measurement.txt`、計測器は
`instruments/duplicate_tolerance_firing_rate.py`（出荷済み、実行して確認済み）。

**原因は 2 つあり、独立している。**

- **A（設計）**: round 5 で決めた「同一層の重複綴りは**値が等しければ許す**」が、
  任意の Python 値についての**全域な等価判定**を要求している。`values_differ` の
  呼び出し元は今も 2 か所だけで、どちらもまさにこの問い。この 1 行の仕様が
  `value_equality.py` + `param_domain.py` = **728 行**、production commit
  **51 件中 30 件**、**rounds 18-26 の 9 連続**を生んでいる。
  H-0095 はこの述語の入力を有界にするために存在する（`value_equality.py:9`）ので、
  **独立した層ではなく A の下流**である。
- **B（手続き）**: round 21 以降、問いが「**どんな値でも門を破れないか**」という
  **全称命題**になった。反例でしか答えられないので **`APPROVE` の出口が無い**。
  **A を除いても B は残る。**

**新規実測 — A の許容分岐は出荷済み母集団で 0 回発火する。**
4 surface すべての呼び出し点を包んでスイート全体（`9731 passed`）を計測した:

```
one parameter under two spellings: 51
  REFUSED  (different values): 14
  TOLERATED (equal values):    37
帰属: 37/37 が tests/test_core/test_fit_params_override.py（本 PR が追加した file）
      pre-existing のヒット: 0
```

round 11 の `0/811`、round 12 の `0/1009` / `0/22`、round 13 の `0/916` / `0/928`、
round 14 の `0/70`、H-0093 の `0/736` / `0/52` / `0/3` と整合する。
**「等しければ許す」は DC6 の形をした許容**であり、それが 30 commit を運んでいる。

**同じ PR に厳しい側の前例がある**: `7f50c59` は探索空間 surface で
**値を見ずに重複綴りを拒否**している。round 5 の許容は制約ではなく選択だった。

### 追加された選択肢

5. **仕様を狭める** —— **同一層の重複綴りは値によらず拒否する**（`7f50c59` と同じ規則を
   残り 4 surface へ）。`values_differ` の 2 つの呼び出し元が消え、`param_domain.py` の
   存在理由（比較の領域を閉じること）も消える。**レビューの問いが有限になる。**
   —— **H-0094 round 5 の決定の改訂であり Change Gate 案件。** 上の実測が
   firing rate 証拠になる。bound: 計測できたのは本リポジトリが構築する母集団だけで、
   ユーザーが将来書く config は測れない（＝振る舞いの変更ではある）。

### 追記 2 — 先行事例の調査（2026-09-09、ユーザー指示）

「原因 A / B は現場でよく遭遇する問題のはずなので、解決策を調査せよ」との指示。
全文 `results/pr2_prior_art.md`、計測器 3 本を `instruments/` に出荷
（`lgbm_duplicate_alias_behaviour.py` / `wire_form_as_equality.py` /
`duplicate_key_prior_art.py`）。

**前回の計測を 1 件訂正する。** 「LightGBM は重複別名に黙っている」は誤りで、
`verbose: -1` がログを抑止していただけだった。**verbosity 既定で fd 捕捉すると
全ケースで警告する。**

- **LightGBM は値を比較しない。等しくても違っても、重複そのものを警告する。**
- **優先順位は決定的で dict 順に依存しない**（`eta` はどちらの順でも `shrinkage_rate` に勝つ）。
- ゆえに round 4 の拒否理由「どの値が効くかはライブラリで決まる」は**半分しか正しくない**。
  決定的であり、LightGBM 自身がそう言う。

**調査した 9 処理系のうち、意味的な値の等価で分岐するのは C プリプロセッサだけ**であり、
その C ですら**トークン列の同一性**（構文的）で判定する。
Python の呼び出しは等しくても `TypeError`、pydantic は alias が無言で勝つ、
Go yaml.v3 / Ruby Psych はエラー、PyYAML / PostgreSQL / dict / json は後勝ち。
**`values_differ` の形（任意 Python オブジェクトの意味的等価）は先行事例に無い。**

**「sink に委譲すれば閉じる」は否定された（実行）**: `_param_dict_to_str` の
`_is_numeric` は `try: float(obj)` なので **`__float__` を持つ任意のクラスが通る** ——
rounds 21-23 が見つけた穴と同じ形が sink 自身にある。さらに `set`(hash 順) /
入れ子の深さ / サロゲート str / `None` 無言脱落 の 4 つの誤受理を実測。
**委譲＋有限の拒否リスト**にはできるが「構成上閉じる」とは言えない。

**wire 比較**（C のトークン同一性に相当）は numpy / ndarray / tuple / カンマ文字列を
**すべて無料で解く**が、**round 5 の緊張は解消しない**（`1` vs `1.0` は拒否に戻る）。
境界が意味的から構文的に移るだけであり、採るなら「設計上の誤拒否として文書化」する形。

**原因 B の現場解は 3 つ**: **B-1** 範囲を宣言してその中で網羅（bounded verification /
small scope hypothesis、受入は「scope k の内側に反例なし」）/ **B-2** parse, don't
validate（網羅性をテストでなく型に持たせる。DC3 の 3 走査問題はこれの裏返し）/
**B-3** 規模と時間で切る（**1 回 400 行以下**が広く使われる閾値。**本 PR の production
差分 1,916 行は約 5 倍**）。

**上位文書との関係** —— **この段落は誤りだったので訂正する（2026-09-09）**。
§5.3 しか見ずに「BLUEPRINT は別名重複に触れていない」と書いたが、
**`BLUEPRINT.md` §14.4（1291 行目）が明記している**:
「同じ層で 1 パラメーターが複数綴り・異なる値で指定されたら `CONFIG_INVALID` と
すること（**同値は通す**）」。**許容規則は上位文書に載っており、改訂には
BLUEPRINT の更新が要る**（H-0096 で実施）。§5.3 が固定しているのはスマート
パラメーターと `params` の競合だけ、というほうは正しい。

### 選択肢の再提示（A の選択 × B の選択、4 経路）

**経路 1 — 縮小してから通す**: A = 同一層の重複綴りを**値によらず拒否**（LightGBM 自身が
重複を警告する事実・Python 呼び出しの `TypeError`・実測 firing rate 0 が支持）。
`values_differ` と 2 呼び出し元が消え、`param_domain` は消費者行 7（比較の全域性）を失う。
残る消費者は `export_code` の `json.dump` + UTF-8 だけで、これは**出口側で解ける**
（消費者行 3「正規化は wire を変えない」＝学習には影響しないことが契約で保証済み）。
B = **B-1**（round 27 は有限の 2 問だけを問う）。**Change Gate 必要。37 件の許容テストを
拒否テストへ書き換える。**

**経路 2 — 分割してから縮小**: #264 本体（rounds 1-15 で決着、以後 11 ラウンド clean）を
先にマージし、H-0095 は経路 1 の設計で新規 PR。**400 行閾値の 5 倍という規模の証拠が
これを支持する。** 切り出した転送側は単独検証が要る。

**経路 3 — 現状維持で位置を計測**: 機構を残し、各位置で各門が適用されることを性質として
固定（元の選択肢 1）。B = B-1。**原因 A は残るので、領域の拡大↔閉包の綱引きは続く。**

**経路 4 — `APPROVE` 無しでマージ**: 原因 B の分析（全称命題に出口が無い）は支持しうるが、
原因 A の 728 行はそのまま出荷される。D10 で再確認した承認ゲートを覆す。

### 決定 — 経路 1「縮小してから通す」（2026-09-09、ユーザー）

**ユーザーが 4 経路の提示に対して経路 1 を選択した。** 本節はその処分の記録であり、
H-0096 の「決定日 2026-09-09」が指しているのはこの行である。

経緯: D13 は当初 4 択（位置を計測 / round 27 続行 / 分割 / `APPROVE` 無しマージ）で
提示したが、ユーザーは決めずに「原因を解析せよ」と差し戻した。解析
（`results/pr2_why_no_approve.md`）が原因 A / B を分離し、追記 1 で選択肢 5
「仕様を狭める」を追加した。ユーザーは再び決めずに「原因 A / B は現場でよく遭遇する
問題のはずなので解決策を調査せよ」と差し戻し、調査（`results/pr2_prior_art.md`）の後、
追記 2 で 4 経路に再編したところで**経路 1 を選択した**。

**この順序が記録として重要である。** 仕様の縮小は、レビューを抜けるために maker が
提案して自ら採ったものではない。**maker は 2 度決定を求めて 2 度差し戻され、
根本原因の測定と先行事例の調査を経たうえで、principal が選んだ。**
ループ監視（`results/pr2_monitor_round2627.md`）が `escalate` を推奨した理由が
まさにこの区別であり、本節はそれに対する応答である。

**決定の内容**（H-0096 として実装済み）:

- **A**: 同一層の重複綴りは**値によらず拒否**する。`value_equality.py` を削除。
- **B**: round 27 の問いを**有限形**に置き換える（宣言母集団の上で各消費者要件が
  成立するか / 母集団の外は拒否されるか）。
- **範囲外**: `param_domain.py` の再設計は [#284](https://github.com/nbx-liz/LizyML/issues/284)。
  縮小の実寸は **157 行**であって 728 行ではない（H-0096 に明記）。

**未決のまま残るもの**: D1-D4 / D6、gate issue #327 / #276。

---

## D14 — round 28 で停止条件 C が発火した（2026-09-09）

### 何が起きたか

範囲限定の round 28（対象＝`5715ee2`）は `REQUEST_CHANGES` 1 件
（記録: `results/pr2_codex_round28.md`）。位置は **`periphery`** ——
`5715ee2` が書いた回帰テストが、「印字できない値」を
`sys.get_int_max_str_digits()` に頼って作っていた。`PYTHONINTMAXSTRDIGITS=0` は
上限そのものを無効化するので、テストはどちらのヘルパーにも到達せずに落ちる。

**「拒否は値を読まない」ことを主張するテストが、値が読めないことを
インタプリタの設定に頼っていた。** 修正済み（描画が raise するオブジェクトへ置換、
3 通りの設定で検証、RED も確認）。フルスイート 7710 passed、CI 用ゲート clean。

### 停止条件

**C が発火した。** そして rounds 26-27 の監視が予告していたとおりである:

> リセットはここで使い切っている。round 28 が `5715ee2` の書いたコードの欠陥を
> 出せば C は再度発火し、**次はリセットの根拠が無い。**

### 前回 C が発火したとき（D12/D13）との違い

| | rounds 25-26 | rounds 27-28 |
|---|---|---|
| 指摘の位置 | **どちらも `deliverable-path`** | 27 = 拒否の**報告**経路、28 = **テスト** |
| クラス | **同一クラスが 2 連続**（門を足して 1 か所を漏らす） | **別クラス**（値の描画依存 / テストの環境依存） |
| 因果 | round N の修復が round N+1 の欠陥を**作っていた** | round 28 が指したのは round 27 の修復に**付随したテスト** |
| 規則そのものへの指摘 | —— | **0 件**。レビュアーは 6 つの値の形で両ヘルパーを実行し、12 の例外すべてが `str` / `repr` / traceback 整形に耐えることを確認 |

### そのほかの判断材料

- **`APPROVE` 未取得のまま 28 ラウンド。**
- **production への指摘は round 27 の 1 件が最後**（round 28 は 0 件）。
- **未解決の非 blocking な観測が 1 件**: adapter 側の拒否メッセージは surface を
  名指していない（facade 側は名指す）。`_pop_by_identity` が surface を引数に
  取らないため、直すには署名を変える。`5715ee2` より前からの性質。
- **起票済みで未着手**: #284（`param_domain` の三重走査、DC3）、
  #285（`LGBMAdapter` の 6 か所目、DC4）。
- 副次: 修正中に auto-lint がローカル `import sys` を消していたことに起因して
  1 テストを壊し、フルスイートで検出・復旧した（`results/pr2_codex_round28.md`）。

### 選択肢

1. **範囲限定の round 29**（対象 = `6b14b99` のみ）。D5 でユーザーが選んだ
   「第 4 の選択肢」と同じ形。
2. **`APPROVE` を求めずマージする。** D10 で再確認した承認ゲートを覆す。
3. **PR を分割する** —— #264 本体を先にマージし、H-0095 / H-0096 を別 PR へ。
4. **停止条件のほうを変える** —— 例えば「`deliverable-path` の指摘が 2 ラウンド
   連続で 0 なら `APPROVE` 相当とみなす」。これはゲートの改訂であり、
   **maker が決めてよいものではない**。

   **⚠️ この選択肢に付けた前提は誤りだったので訂正する（2026-09-09）。**
   初出では「rounds 27-28 はすでにその状態にある」と書いたが、
   **round 27 の指摘は `deliverable-path` である** ——
   `results/pr2_codex_round27.md` 自身がそう記録している。拒否メッセージの構築は
   production コードであり、**約束した例外の代わりに別の例外を投げることは
   観測可能な振る舞いの違反**である。正しくは **「1 件 → 0 件」**であって
   「0 件が 2 連続」ではない。**この選択肢の前提は成り立っていない。**
   （D14 状況評価が実測で指摘した。`results/pr2_d14_assessment.md`）

### 状況評価（2026-09-09、Codex `gpt-6-astra` effort low）

ユーザー指示により、決める前に状況そのものを評価させた。全文は
`results/pr2_d14_assessment.md`。**推奨は「完了基準を明示したうえで、管理者が
承認する受け入れレビューを 1 回」**（上の選択肢 1-4 のどれとも少し違う第 5 の形）。

**こちらの記述を 3 点訂正された**（すべて確認済み）: 上記の round 27 のタグ /
D14 の追記が未コミットで作業ツリーが clean でなかったこと /
「28 ラウンド」は完走した Codex verdict の数ではないこと（round 22 は verdict 無し、
round 23 は別の checker）。

**「毎回の修正が次のラウンドの対象を供給する」という主張も過大と評価された**:
修正は新しい**材料**を供給するが**欠陥**を供給するとは限らず、
指摘が 2 回続いたことは終わらない過程を立証しない。
rounds 16-17 も等価判定だけの作業ではなかった。

**評価者が指定した「止めることが正当化される条件」**: 評価した head と契約を凍結し、
適用可能な証拠とその限界を特定し、**#284 / #285 と surface 名の不一致を明示的に処分**し、
互換性の帰結を文書化し、承認か明示的な例外を得る。そして**さらに指摘が出た場合に
何が受け入れを妨げるかを、あらかじめ宣言する**。
**発火した停止条件は自動的なラウンドを止めることを正当化するが、マージは許可しない。**

### 決定（2026-09-09）

**選択肢 A —— 完了基準を明示したうえで、管理者が承認する受け入れレビューを 1 回**
（状況評価の推奨。D14 に最初に書いた 4 択のどれとも違う第 5 の形）。

**先に完了基準を書き、管理者が承認してからレビューを走らせる。**
完了基準は `results/pr2_acceptance_criteria.md` —— 凍結 head / 受け入れ基準 → 証拠の
対応表 / 証拠の限界 / #284・#285・surface 名不一致の明示的処分 / 互換性の帰結 /
レビューが答える 3 つの問い / **指摘が出た場合の 4 バケット規則** / 承認の順序。

**採らなかった選択肢と理由**:

- **1（範囲限定の round 29 のみ）** —— clean が返っても PR 全体の承認にはならず、
  同じ位置に戻る。A はその戻りを、「範囲限定の検証結果が全体のゲートをどう支持するか」を
  **事前に定義する**ことで塞ぐ。
- **2（`APPROVE` を求めずマージ）** —— 評価者いわく「緑のチェックはこの選択を支持するが
  単独では正当化しない」。A を経てから改めて判断できる。
- **3（PR を分割）** —— 過去のラウンドの clean さは分割後に自動的には引き継がれず、
  export が依存する修正を落とすリスクもある。
- **4（停止条件のほうを変える）** —— **前提が成り立たない**（round 27 は
  `deliverable-path`）。

**完了基準を書く過程で 2 件の drift を見つけ、レビューを開く前に閉じた**
（どちらも DC3、振る舞いは無変更）:

1. H-0096 の**受け入れ基準 5** が「拒否メッセージは値を名指す」のまま残っていた。
   提案節は round 27 のあとに訂正済みだったが、基準側は訂正されていなかった。
   round 28 前に監視が指摘したのと同じクラスの 2 例目。
2. `test_two_spellings_of_one_value_in_calibration_params_are_accepted` は
   **本体が拒否を主張しているのに名前が受理を主張**していた（H-0096 で意味が反転した
   ときに docstring だけが直っていた）。`..._are_refused` へ改名、アサーションは無変更。

### 追記: 非 blocking と分類した 2 件を、本 PR 内で直した（2026-09-09）

受け入れレビュー（round 29）は 2 件を返し、**§6 の事前宣言はどちらも非 blocking と
分類した** —— #288（B4、カバレッジの穴）と #287（B3、探索空間の合成。fail-closed で
Booster 0 本、向きは「有効な入力を拒む」）。

**管理者はそのうえで、両方を本 PR 内で直すことを指示した。** 理由は
「後回しにして解決しやすくなるものは無い」であり、実際どちらも設計判断を要さないか
（#288 = テストのみ）、要しても小さかった（#287 = 既存の拒否経路にサブクラスの穴を
閉じるだけで、新しい振る舞いを足さない）。この環境では**未処理の繰り延べが
post-closure audit の rework になる実績**がある（PR #535→#536→#537、8 ラウンド）。

**これは §6 の完了経路の実行ではなく、§6 への例外である。**
§6 は「非 blocking な B3 は起票して終わり」「B4 は再レビューしない」と規定しており、
B3 を PR 内で直す手順は書かれていない。rounds 28-29 の監視がこの点を指摘し、
**「§6 の B1 が定める手順に入る」というこちらの記述は誤り**だと訂正した。
完了基準 §1 と §3 に例外として記録した。

**凍結 head は意図的に破られた。** production の差分は `6b14b99` から 1 ファイル
（`lizyml/tuning/search_space.py`）。よって **round 30 を、選ばれた修正コミットに
限定して**開く（`22b11b3` / `77e02a8` / `9267c9a`）。

**監視が指定した「ループが名前を変えて再開した」ことの証拠**（そのまま持ち越す）:

> - 非 blocking な指摘が**繰り返し**必須の前提条件になる
> - **round 30 が選ばれた変更を超えて広がる**、またはさらなるラウンドを自動的に生む
> - 認識済みの周辺問題が開いたままでは明示的な受け入れ判断が下せない
>
> **「管理者が承認した」は権限を立証する。有界な follow-through だけが収束を立証する。**

**残る未処分は 3 件**（#284 / #285 / #286）。#287 は「4 surface が numpy を受理するのに
探索空間は拒否する」という不整合のために **OPEN のまま**にしてある（すり抜けは直したが
不整合は閉じていない）。#288 は develop に乗るまで OPEN。

### 受け入れ宣言（2026-09-09）—— **D14 の終端**

**管理者は PR #278 を受け入れた。マージのタイミングは別途判断する。**
draft は外した（ready for review）。**マージはしていない。**

#### 受け入れの根拠（完了基準 §6 の承認条件に対する充足）

| 条件 | 状態 |
|---|---|
| (a) 最後の修復が clean | **満たす** —— round 30 が `APPROVE`、**production の欠陥 0 件** |
| (b) 対応表の全行が `satisfied` | **文字どおりには満たさない。** round 29 が 8 行のポインタ不足を指摘し訂正した。**基準そのものの不成立は 1 件も報告されていない**（レビュアー自身が「証拠のポインタが不完全であること自体は production の欠陥を立証しない」と前置きしている） |
| (c) 4 surface すべてで相互作用が通る | **文字どおりには満たさない。** 3/4。4 つ目は §5(c) の**こちらの誤記**（正規化の契約が持つ 4 つ目は tuning の**結果**であって探索空間ではない）に由来し、見つかった穴は #287 として起票、すり抜けは修正済み |
| §3 の処分を管理者が受け入れる | **満たす（本節）** |
| CI | **11/11 SUCCESS**（non-blocking レーン含む） |
| ローカルゲート | フルスイート **7721 passed / 230 skipped**、ruff / ruff-format / mypy clean |

**(b) と (c) が文字どおりには満たされていないことを承知のうえでの受け入れである。**
どちらも「基準が head で成立していない」ではなく、「**基準を指す証拠の書き方**が
不正確だった」「**基準の範囲についてのこちらの記述**が不正確だった」であり、
いずれも訂正した。§6 の事前宣言に照らせば B4（文書）に当たる。

#### OPEN のまま develop に乗るもの

- **#284** —— `param_domain` の三重走査（DC3）。保守性の負債。直すと受理集合が動く
- **#285** —— 6 か所目が重複綴りを黙って選ぶ（DC4）。**公開 surface から到達不能**（実測・テスト固定済み）
- **#286** —— adapter の拒否メッセージが surface を名指さない。公開コンストラクタに provenance を通すデータ契約変更が要る
- **#287** —— **意図的に OPEN**。すり抜け（`np.float64` / `np.str_`）は直したが、
  「4 surface が numpy を受理するのに探索空間は拒否する」という不整合は閉じていない
- **#288** —— merge で close される

#### この run で確定した手続き上の事実（次の PR へ持ち越す）

- **長いレビューループを止めるときは、完了基準を先に文書化して承認を得てから
  受け入れレビューを 1 回。** 対応表を作ること自体が検査になる（この run では
  作る過程で DC3 の drift が 2 件出た）。
- **発火した停止条件は自動ラウンドの停止を正当化するが、マージは許可しない。**
- **監視の勧告への reconcile を無条件に先に書かない** —— 監視が次の一手に影響を
  与える能力を落とす。適用条件を限定すること（完了基準 §8）。
- **「管理者が承認した」は権限を立証する。有界な follow-through だけが収束を立証する。**

**D14 はこれで閉じる。** 残るのはマージのタイミングの判断のみ。
