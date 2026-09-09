# PR 2 — relational monitor, rounds 9-10 (2026-09-08)

Spawned before round 11. Given the numbers unsoftened — blocking per round
**1, 1, 2, 2, 1, 3, 2, 4, 3, 2** — and asked, besides convergence, the question
round 10 raised by being the first round to find **no production defect**: is
that the deliverable being finished, or the ordinary next step of a review that
has not yet reached something?

```
VERDICT: DRIFTING
RECOMMENDATION: redirect
```

## It answered the question by refusing it, and gave the reason

> **The record does not distinguish the two readings, and cannot, because of how
> the rounds were scoped.**
>
> Rounds 6, 8, 9 and 10 were each scoped to the previous round's remedies. The
> one exception is round 7, pointed at an unreviewed shipped surface
> (persistence/export) — **and it found production defects.** A round aimed at
> freshly written apparatus finds apparatus defects at whatever rate new code
> carries them; it yields "no production defect" whether or not the deliverable
> is finished. **Round 10's result is mechanically produced by its scope, not by
> the deliverable's state.**

And it named the observation that would settle it:

> one round over the deliverable path **unrestricted** (the whole diff minus
> `docs/`), asked the deliverable question. No remedy-scoped round can produce
> that observation.

## On the maintainer's premise

Stated plainly rather than softened, and it is the most consequential line in the
report:

> "no `APPROVE` means real problems remain **in the fix code**" is not tested by
> remedy-scoped rounds. Twenty-one reproduced findings is a true count, but
> findings in apparatus written two commits ago are not evidence about the fix.

## What it measured

Verified independently here before adopting:

- **The deliverable has not changed since before round 6 was reviewed.**
  `adapter.py` and `_model_factories.py` last touched at `9b53aa5`, `model.py` at
  `c0066dd`, `provider.py` at `848e4ac`. Round 10 changed nothing executable —
  its one production edit is the comment above `SMART_PARAM_TARGETS`.
- **The contraction did not hold.** Periphery 2096 → 2244 lines across the
  window, net **+84**, after monitor 89 measured −64.
- **Authorship is decisive.** Round 10's finding 1 lives in code introduced by
  `d44668f` (monitor 89's own prescribed repair); finding 2 lives in `_probe`,
  last rewritten by `f92a20d`, round 9's remedy. **Both findings are in code
  written after round 9's review** — the authorship pattern D7 first registered
  as a stop signal. The chain it names: *monitor prescribes apparatus → next
  round finds defects in it → remedy grows apparatus.*

Counter-evidence it kept: the standing constraint held (no new module, generator
or scanner — `import ast` in the test module is now **0**, confirmed here),
findings fell 3 → 2, and both remedies were subtractive in claim.

## What the record does support

> `value_equality.py` and both callers were executed clean in round 10 (32
> `CASES`, no wrong answer beyond documented printed-form limits) — real
> evidence for that Layer-0 piece. The merge path itself has been reviewed once
> since it last changed (round 6) and was not flagged, **which is not the same as
> declared clean.**

## Its recommendation, adopted

> **`redirect`** — round 11 should be **unscoped over the deliverable path**
> rather than pointed at round 10's remedies; that single observation separates
> "finished" from "not yet reached", whereas another remedy-scoped round will
> find defects in round-10's new code by the same mechanism and settle nothing.

Adopted in full. Round 11 is deliberately a **widening**, the first since round 7
— the whole diff minus `docs/`, asked the deliverable question. It is the round
that tests the maintainer's premise rather than assuming it, and its result is
informative either way: a production finding says the fix code still has
problems, and a clean pass over the unrestricted deliverable path is the first
evidence in this run that it does not.
