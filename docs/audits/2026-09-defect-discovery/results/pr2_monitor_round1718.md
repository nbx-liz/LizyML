# PR 2 — relational monitor, rounds 17-18 (2026-09-08)

Spawned before round 19, at head `82ce532`.

```
VERDICT: CONVERGING
RECOMMENDATION: continue
```

**The first `CONVERGING` in three monitors, and the first `continue` since the
rounds 13-14 pass.** Its two predecessors both returned `DRIFTING` / `escalate`.

## What it measured

`git diff --numstat`, production = `lizyml/`, tests = `tests/`, audit instruments
excluded:

| interval | production | tests |
|---|---:|---:|
| round 17 (`04f3930..42148e2`) | +68/−21 | +223/−16 |
| round 18 (`42148e2..82ce532`) | +56/−13 | +140/−23 |
| whole change (`origin/develop..82ce532`) | +1582/−53 | +4468/−36 |

## It executed the claim it was asked to distrust

The prompt named the risk directly: narrowing a bound so the failing case falls
outside it is a legitimate repair when the narrowed bound is the one that
matters, and **DC5 dressed as a fix** when it is not. It was told the evidence
offered was that `_param_dict_to_str` raises on the excluded value, and told to
check that rather than accept it.

It ran the installed serialiser itself, and then went further than asked — it
attempted `lgb.train` directly with the excluded object under both spellings:

> The raising-`__class__` object produced `RuntimeError: class unavailable`; the
> proxy and ordinary `0.5` both produced `learning_rate=0.5`. Direct `lgb.train`
> attempts with the excluded object failed under both `learning_rate` and `eta`.
> **The declaration change is legitimate for the demonstrated exclusion, rather
> than DC5 disguised as repair.**

## And it corrected the record

> The record's "never reaches `lgb.train`" wording is imprecise: serialization
> occurs inside LightGBM, so the function can be entered. The supported,
> relevant claim is that this object **cannot complete training** through that
> path.

Correct, and adopted. The wording is fixed in `HISTORY.md` decision 15, in the
round-18 record, in `CHANGELOG.md`, and in both places `value_equality.py` said
it. This is the second time a monitor has caught this context overstating a
claim in its own favour, and both corrections are on the record rather than
quietly applied.

It also bounded what its own execution establishes, unprompted: not that every
serialiser-accepted object trains, not that a universal no-exception function is
logically impossible, and not that the oracle relation closes all Python
behaviour — the test still samples a finite population, and the oracle improves
the assertion rather than the completeness of the population.

## On the scoped round

> The scoped round changed behavior, not merely output size: it independently
> challenged a remedy and obtained bounded stability evidence. The third
> consecutive authorship failure remains adverse evidence; the drop from two
> blockers to one alone would be weak. **But the predecessors' escalation has now
> produced a maintainer-selected intervention with a distinguishable result.
> Repeating their reading unchanged would overlook that result.**

That is the answer to the question the maintainer's decision posed. It is also
the correct handling of its own predecessors: not a reversal, but new evidence
they did not have.

On the deliverable it is even-handed in both directions — neither round adds
another *forwarding* connection, and yet failed attempts now preserve the trained
model's state and a legitimate duplicate spelling now passes comparison toward
training, both of which are correctness on the touched parameter path. It notes
that round 18 added no production change outside `value_equality.py` and that the
reporting instrument stopped expanding:

> The evidence no longer shows every repaired part generating another repair.

## The main context's reconciliation

Adopted in full, including the correction and including the recommendation.
Nothing here needed declining, which is itself the first time in three monitors.

**Round 19 is scoped to the round-18 remedy**, per the framework the maintainer
set when they chose the scoped round: clean means the fixes have stabilised and
the following round goes unscoped; a finding means the loop is still cycling. The
monitor's finite check is that scope stated precisely — the 108 behaviour
combinations × three sequence forms × three text forms × two operand orders, plus
the oracle witness and the shipped-path proxy case.

Full suite **2666 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean; the shipped lifecycle grid exits 0.
