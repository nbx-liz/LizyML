Review-kind: monitor
Monitor-mode: relational
Observed-rounds: 4-5

# Relational loop monitor — PR 1 Codex gate, after round 5 (run late)

Carrier: fresh general-purpose subagent, read-only. Spawned after the main
context had already closed the loop and escalated, which `policy:loop-monitor`
requires to happen before.

VERDICT: CONVERGING — recommendation `take-stop-condition`

## Round-by-round

- R1 — 4 blocking, all deliverable; gate moved `Model.__init__` → point of use.
- R2 — 4 blocking; `export_code` added, `fit(params=)` retracted to PR 2.
- R3 — 2 blocking; 1 live ungated route (`isotonic.py:105`), 1 apparatus.
- R4 — 1 blocking; check ran after the whole outer CV, falsifying
  `BLUEPRINT.md:1018` (`results/pr1_codex_round4.md`).
- R5 — 1 blocking; `tune()` is a second entry point round 4's remedy missed,
  `_model_tuning.py:196` (`results/pr1_codex_round5.md`).

## Convergence

**Deliverable moved; periphery did not, for the second consecutive round.**
Round 5's production delta is `_model_tuning.py` +21 (two gate calls plus the
comment stating why), one parametrized ordering assertion, and H-0093 decisions
4/6. AST apparatus unchanged: `tests/_ast_scan.py` (10:29) and
`test_lightgbm_parameter_names.py` (10:42) both predate the round-4 log (10:46)
and the round-5 log (10:59) — 588 lines per monitor r4, neither file touched since.

Findings 4, 4, 2, 1, 1, severity falling monotonically: R3 an unknown name
reaching LightGBM ungated, R4 the gate firing correctly but late, R5 the gate
absent on one of two entry points. All reproduced by execution on production
paths. None is the always-findable class.

The counter-signal is real: R4 and R5 are the same shape twice — an incomplete
enumeration of entry points — and the axis is **still** open at head.
`TRAINING_ENTRY_POINTS` (`test_calibration_param_names.py:151`) is hand-written,
referenced from nowhere but its own parametrize, and names 2 of `Model`'s 23
public callables.

**One periphery growth is outside the diff, in the escalation record.** D5 in
`DECISIONS-PENDING.md` accretes by append rather than edit: lines 168 and 196
are near-duplicate paragraphs, the retracted "later PRs written the same way"
claim is still present above its own retraction at line 201, and option 1 still
reads "CI has not run yet — the branch is not pushed" against a pushed PR #275
with CI 12/12. The maintainer decides from this document. For reconciliation.

## On the stop

**Sound in content, defective in procedure — and the content is what matters.**

Verified against the round-5 prompt itself, not the maker's record: the
pre-registration is at `codex-pr1-review-prompt-r5.md:8` and lines 26-33, which
name round 4's remedy as the round's centre and state there is no round 6 either
way. Written 10:55, after monitor r4 (10:54), before the round-5 log (10:59).

The trigger genuinely fired. Round 4 anchored the check to `_merge_params`'s
point of return — but in `_fit_impl` only, at *one caller* of a shared helper
rather than in the helper. `tune()` calls the same `self._merge_params(provider)`
at `_model_tuning.py:196`, so the second caller was left uncovered. That is a
placement defect in round 4's remedy, not a sibling gap.

The flag itself is right. Rounds 4 and 5 each reviewed the previous round's
remedy, and the argument that opened round 5 ("round 4's fix is unreviewed")
applies identically to round 5's fix. That recursion has no fixed point, so it
cannot be the stopping rule; a pre-registration fixed before the outcome is seen
is what breaks it.

The procedure was wrong: the stop was self-certified by the party owning the
deliverable, skipping the monitor the policy places outside the loop. The defect
is in *who* ratified the flag, not *what* it says — I would have raised it
unprompted. Named so the gate bug is visible, not to reopen the loop.

The `TRAINING_ENTRY_POINTS` gap cuts *for* stopping. The main context found and
characterized it **without a round**, on the class the loop spent rounds 3-5
teaching. Round 6 would likely find something real; it is no longer the cheapest
instrument for finding it.

## Recommendation

`take-stop-condition` — two rounds of zero periphery growth with shrinking,
production-real findings is convergence, and the one class still producing
findings is now caught by the maker without a review round, so D5 belongs with
the maintainer rather than in a sixth round. Offered for reconciliation, not
adoption: the open `TRAINING_ENTRY_POINTS` axis and D5's stale text are both
fixable without a round, and both bear on choosing among D5's three options.
