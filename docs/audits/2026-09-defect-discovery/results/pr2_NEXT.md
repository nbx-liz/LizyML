# PR 2 — the next action, standing alone

Written so that a session starting cold can act without reading the run.

## Where this is

PR **#278**, draft, branch `fix/phase3-pr2-fit-params-forwarding`. H-0094 /
issue #264: `Model.fit(params=...)` was accepted, documented as overriding
`model.params`, and forwarded nowhere.

**Thirteen review rounds have run. No `APPROVE` yet.** Blocking findings per
round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3**, plus two the main context found
itself in round 12 by enumeration. Every one was reproduced before
it was fixed and RED-verified after.

The maintainer's instruction is the whole gate: **run PR 2 to the same standard
as PR 1 — until the external reviewer returns `APPROVE`.** The absence of
`APPROVE` is itself treated as evidence that problems remain in the fix code,
and rounds 11 and 12 confirmed that by execution. The merge gate is
`APPROVE` + CI green, already decided; do not re-ask
(`memory/feedback_phase3_run_policy.md`).

## The one thing to know about scope

Rounds 6, 8, 9 and 10 were each narrowed to the previous round's remedies, and
each found nothing in production. That result was **produced by the scope, not
by the code**: a round aimed at freshly written test apparatus finds
test-apparatus defects. Round 11 was widened and found three production
defects; round 12 was widened and found two more, plus one the main context
found by enumeration.

**Every further round is unscoped.** Do not narrow one to "review the fixes",
whatever the previous round returned.

## The next action

1. **The rounds 12-13 monitor has already run**, returning `CONVERGING` /
   `redirect` — the first redirect of this run. Its redirect was verified by
   execution and adopted in full before round 14: the round-13 grammar fix
   reached one of the four types LightGBM joins, and the other three were
   refused although the serialiser produced the byte-identical wire string. See
   `results/pr2_monitor_round1213.md`. **Round 14 can open directly.**

   Before round 15, spawn the rounds 14-15 monitor with the numbers unsoftened,
   plus the pattern two monitors have now established: a claim of the form
   "applied to the whole set" has been falsified within one round of being made,
   twice — the seam construct set, then the serialiser type set. Ask it what
   else in this diff claims a set it has not executed over.

   A monitor's output is a finding to reconcile, never a verdict to adopt
   (`policy:main-context-ownership`).

2. **Then round 14, unscoped**, on the whole diff except `docs/`. Write the
   prompt to `scratchpad/codex-pr2-review-prompt-r14.md` with the metadata block
   the `review-loop-monitor-guard.sh` hook validates (`Review-kind` on line 1;
   round 3+ requires the relational monitor fields).

3. Codex invocation, and the two rules around it:

   ```
   CODEX_HOME=<writable copy> codex exec --sandbox read-only \
     -C /home/rem/repos/LizyML --color never - < prompt > log 2>&1
   ```

   `setup_codex_home.py` makes the copy; **`cleanup_codex_home.py` deletes it
   after every run** — it holds `auth.json`. `codex-home/`, `~/.codex`, `~/.ssh`
   and `~/.aws` are outside every reviewer's read scope.

4. Reproduce every finding before fixing it, RED-verify every regression test,
   and measure a firing rate by replaying real configs rather than estimating
   it.

## State at the time of writing

- Head: the round-12 fixes, on `fix/phase3-pr2-fit-params-forwarding`.
- Full suite **2498 passed**; `ruff check .`, `ruff format --check .`,
  `mypy lizyml/` clean.
- Round 13's record: `results/pr2_codex_round13.md`. Decisions:
  `HISTORY.md` H-0094, decisions 1-9. Monitor:
  `results/pr2_monitor_round1213.md`. Open question log:
  `DECISIONS-PENDING.md` D7.

## What is deliberately not in this PR

- **#280** — the smart-managed refusal is wired to `fit(params=)` only; the
  config surface is 3 of 18 refused, measured in round 12 and recorded in
  BLUEPRINT §14.4. `config/` cannot import `estimators/`, so where the refusal
  belongs is a design decision the maintainer holds.
- **#279** — a `category: model` dimension colliding with a **smart parameter**,
  54/67 measured. (Two dimensions colliding with **each other** is a different
  seam and *is* fixed here — H-0094 decision 8's addendum.)
- **#277** — `calibration.params` accepted and ignored for `platt` / `beta`.
- The calibration layer's `min_data_in_leaf` case, which belongs to #280's class
  and is recorded in round 12's note.

## After this PR

PR 3 (#258 tuning direction), PR 3b (H-0024 space merge — must resolve
`HISTORY.md:1615` against `:1616`), PR 4-9. One reconciliation pass immediately
before PR 9 for the deferred Phase 3 completion-measurement tooling
(`phase3_gap.py` + manifest).
