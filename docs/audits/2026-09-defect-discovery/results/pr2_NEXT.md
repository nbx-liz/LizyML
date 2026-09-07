# PR 2 — the next action, standing alone

Written so that a session starting cold can act without reading the run.

## Where this is

PR **#278**, draft, branch `fix/phase3-pr2-fit-params-forwarding`. H-0094 /
issue #264: `Model.fit(params=...)` was accepted, documented as overriding
`model.params`, and forwarded nowhere.

**Twelve review rounds have run. No `APPROVE` yet.** Blocking findings per
round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2**. Every one was reproduced before
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

1. **Run the rounds 12-13 relational monitor first.** It is mandatory before
   round 13 (`policy:loop-monitor`), read-only, fresh context, via
   `templates/review-loop-monitor-capsule.md`. Give it the numbers unsoftened
   and the two things round 12 established:
   - neither reviewer finding was in code round 11 wrote, so this is not the
     authorship pattern the maintainer rescinded;
   - the parameter-merge seam population is now **enumerated and closed** (24
     expressions, 12 cross-source, each executed — H-0094 decision 8). Ask it
     whether that table is the closed population, or whether it can name a merge
     the AST scan's hint-word filter would miss.

   Its output is a finding to reconcile, never a verdict to adopt
   (`policy:main-context-ownership`).

2. **Then round 13, unscoped**, on the whole diff except `docs/`. Write the
   prompt to `scratchpad/codex-pr2-review-prompt-r13.md` with the metadata block
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
- Full suite **2445 passed**; `ruff check .`, `ruff format --check .`,
  `mypy lizyml/` clean.
- Round 12's record: `results/pr2_codex_round12.md`. Decisions:
  `HISTORY.md` H-0094, decisions 1-8. Open question log:
  `DECISIONS-PENDING.md` D7.

## What is deliberately not in this PR

- **#280** — the smart-managed refusal is wired to `fit(params=)` only; the
  config surface is 3 of 18 refused, measured in round 12 and recorded in
  BLUEPRINT §14.4. `config/` cannot import `estimators/`, so where the refusal
  belongs is a design decision the maintainer holds.
- **#279** — the same collision inside a `category: model` tuning space, 54/67
  measured.
- **#277** — `calibration.params` accepted and ignored for `platt` / `beta`.
- The calibration layer's `min_data_in_leaf` case, which belongs to #280's class
  and is recorded in round 12's note.

## After this PR

PR 3 (#258 tuning direction), PR 3b (H-0024 space merge — must resolve
`HISTORY.md:1615` against `:1616`), PR 4-9. One reconciliation pass immediately
before PR 9 for the deferred Phase 3 completion-measurement tooling
(`phase3_gap.py` + manifest).
