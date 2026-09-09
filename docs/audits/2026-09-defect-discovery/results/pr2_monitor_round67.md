# PR 2 — relational monitor, rounds 6-7 (2026-09-07)

Spawned after the maintainer **rejected the stop** and directed the loop to run
until the external reviewer returns `APPROVE`, on the standard applied to PR 1.
Their stated reasoning, given to the monitor verbatim: *not obtaining APPROVE is
itself evidence that real problems remain in the fix code.*

```
VERDICT: CONVERGING
RECOMMENDATION: redirect
```

## It reversed its predecessor's reading, with a measurement

The rounds 5-6 monitor returned `DRIFTING` and said "what closes it is a bound,
not a round". This one found that **round 7's remedy is that bound**, and
verified it independently rather than taking the record's word:

> a generated cross product of `__eq__` / `__len__` / `__repr__` / `__bool__`
> failure modes, **72 cases, 0 raises**. R5→R6 enumerated more value shapes;
> R6→R7 stopped enumerating. That is the one thing true after R7 that was not
> true after R6, and it is a change in kind.

## Is `APPROVE` reachable? — yes, and it said why PR 1 got there

> PR 1's round 6 and PR 2's round 6 were both maintainer-narrowed sixth rounds;
> PR 1 got APPROVE, PR 2 got three findings. The difference is the domain each
> remedy closed: PR 1 partitioned an **enumerable** population (`dir(Model)`),
> so a round could verify completeness; PR 2 was enumerating an **open** one
> (any Python object), where each round buys one more exotic value. R7 closed
> PR 2's domain by construction. Round 8 can now be PR-1-round-6-shaped.

## On the maintainer's premise

It agreed, and then said what the record adds to it:

> The maintainer's inference — no APPROVE means real problems remain — is true
> on this record: every one of the twelve findings was reproduced. But the
> record also says where they come from. Each remedy ships a *declaration*
> verified by a hand-written table, and the next round finds the gap between
> declaration and verification. **On the current method, "run until APPROVE"
> manufactures its own next finding; it terminates only if the method changes.**

## What it prescribed, and what was done

> Quantify each declaration over its whole population — DC7's durable repair,
> verbatim — and ship exactly two instruments … This is apparatus growth, and it
> is the **last** growth: after a bound, no further case needs adding.

**Instrument 1 — the no-raise bound over a generated population.** 24
combinations of the three dunders `values_differ` touches, each asserted in both
directions and against itself, plus a test that the population is derived (a
shrunk behaviour table fails) and one asserting identical values are never
reported as differing — the half a no-raise assertion does not cover.

**Instrument 2 — every artifact-reading test must fail when nothing is
written.** The population is found by reading the test module, not listed. **It
found a second instance of round 7's defect on its first run**: with
`Model.export` replaced by a no-op, `test_export_code_generates_the_overridden_value`
still passed — because it reads `export_code`'s output, which that substitution
never touched. The instrument now patches both writers, and the scan covers the
generated project. An instrument that names the wrong test is the same defect as
a test that checks nothing.

**Before the round, one more self-authored defect was found and fixed.** Round
7's widening of the truth step moved an unbooleanable comparison out of the
printed-form fallback and into the elementwise reduction, where iterating
yielded objects that are truthy by default — so two different values read as
equal. The DataFrame case round 6 found, through a different door. The string
special-case was replaced by the property it was standing in for: **an element
that defines no `__bool__` of its own is not a comparison outcome**, since its
truthiness comes from length or from the default. That excludes column labels
and bare objects without naming either.

## Where it pointed round 8

> the R7 remedies plus the two instruments above, with rounds 1-6 surfaces
> explicitly out of scope and **no new region admitted**. The persistence/export
> redirect already measured the last unmeasured shipped surface and found
> production correct — record that so nobody reopens it as fresh territory.

Adopted. Full suite **2383 passed**.
