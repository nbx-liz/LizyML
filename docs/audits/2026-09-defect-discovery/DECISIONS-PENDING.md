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
