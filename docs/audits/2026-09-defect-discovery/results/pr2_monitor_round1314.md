# PR 2 — relational monitor, rounds 13-14 (2026-09-08)

Spawned before round 15.

```
VERDICT: CONVERGING
RECOMMENDATION: redirect
```

The second consecutive `redirect`, and the second consecutive monitor to find
something this context had got wrong. Both of its named items were verified by
execution here before being acted on.

## What it measured

Raw `git diff --stat`, no docstring stripping (its own note: direction
comparable with the previous monitor, scale not):

| | production | tests | docs |
|---|---|---|---|
| round 12 | +72 / −8 | | ~1262 (tests+docs) |
| round 13 | +94 / −1 | | |
| round 14 | +45 / −2 | +35 | +208 |

The deliverable is shrinking and the periphery fell from ~1262 to ~208. All
three round-14 production files are on the declared path, and the fix **added** a
check at the space layer without moving the merged-dict one — no relocation.

## The correction this context had to make

> **D7 fired.** `git show 92e3d51` confirms both the un-wired check and the
> falsified comment were authored by round 13's fix. Round 14 found a defect in
> code round 13's fix wrote — the plain reading of D7's authorship condition,
> for the first time this run. The round-14 note's "third instance of one shape"
> is a reframing, not a refutation.

**Verified and accepted.** `git show 92e3d51` was re-run here: that commit
introduced `check_training_managed_overrides` **and** the comment claiming it
"covers every input at once". So round 14's single finding is in code round 13's
fix wrote, and the round-14 record described it only as the third instance of a
recurring shape without saying that.

The record is corrected. The condition remains **rescinded** by the maintainer,
so it does not stop the loop — but the run's own standard is that it is recorded
unsoftened when it fires, and it was not. The monitor was right to say so, and
right that this is the first time it has plainly fired.

## Question (a) — a declaration claiming more than it executed

It named **the round-13 literal-read population**: `pr2_codex_round13.md` says a
grep "returns four candidates", the grep's pattern is nowhere recorded so the
enumeration cannot be re-run, and the write-direction scan is shipped and
quantified with a positive control while the read direction has no counterpart.
Every other population in this PR got an executable test; this one was prose.

It also named `adapter.py:455-458` — `user_params.setdefault("seed",
user_params.pop("random_state"))` — as a literal read handling `random_state`
and not `random_seed`, in no enumeration in the record, and explicitly declined
to grade its liveness.

**Graded here, by execution, and the answer is less than it looks.**
`random_seed=7` reached `lgb.train` under its own name and LightGBM honoured it:
the booster is byte-identical to one trained with `seed=7` and differs from
`seed=99`. **No value was lost, so this is a consistency fix, not a defect fix**,
and it is written down that way. On the facade path the branch is unreachable
anyway — `training.seed` is always set (default 42; explicit null refused), so
the training-managed refusal claims every spelling of `seed` first.

The tidy-up also produced a regression worth recording: the first version used
`_pop_by_identity`, which **refuses** two spellings with different values, and
that broke `test_seed_takes_priority_over_random_state` — an accepted decision
that `seed` wins over `random_state`. Changing a refusal as a side effect of a
naming tidy-up is not this commit's business, so the fix was narrowed to keep
that priority exactly.

## Question (b) — converging, or mining a blind spot

> One class in three coats, and the class is a **finite grid**: 5 layers × 5
> checks. Rounds 11, 12 and 14 each found one empty cell. The remaining empty
> cells are `check_smart_managed_overrides` on the space and on `model.params`,
> both already declared open with measured gaps and issued (#279, #280). So the
> loop is not mining a blind spot; it is draining a nearly-full finite grid, one
> cell per round, and each fix filled a cell rather than moving one.

Adopted, and made executable rather than left as an argument — which is the
whole point of its recommendation.

## Its recommendation, adopted in substance and declined in one part

> `redirect` — ship the layer×check matrix and the read-direction literal scan
> as executable declarations, **then open round 15 scoped to them**.

**Both declarations are shipped**, before round 15:

- `tests/test_core/test_refusal_matrix.py` — the 5×5 grid, with every `wired`
  cell **executed** (the refusal fires, names the layer, and nothing trained
  first), every `open` cell required to name an issue, the table required to
  stay rectangular, and a check that no refusal exists in `_model_factories`
  without a column. Writing it immediately caught three cells marked wired with
  no executed input; all three now have one.
- `tests/test_estimators/test_literal_parameter_reads.py` — the read-direction
  scan, with an allow-list carrying a reason per entry, a staleness check in
  both directions, and seven hostile sources including two negative controls.
  Running it caught two undeclared reads and two stale entries in the list this
  context had written from reading rather than from running.

**The scoping of round 15 is declined, and this is the one place the monitor's
recommendation is not adopted.** Rounds 6, 8, 9 and 10 were each scoped to the
previous round's remedies and each found nothing in production — a result this
run established was produced by the scope, not by the code. The established
precedent for a monitor's named work is the opposite of scoping: probe it
*before* the round and leave the round unscoped, which is what turned the
rounds 10-11 and 11-12 monitors' named layers into fixed defects rather than
next-round findings. Round 15 is unscoped and the new declarations are in it,
along with everything else.

Full suite **2529 passed**.
