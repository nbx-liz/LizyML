# PR 2 — relational monitor, rounds 16-17 (2026-09-08)

Spawned before round 18, at head `65333eb`.

```
VERDICT: DRIFTING
RECOMMENDATION: escalate
```

**The second consecutive `DRIFTING` and the second consecutive `escalate`.**

## What it measured

`git diff --numstat`, production = `lizyml/`, tests = `tests/`:

| interval | production | tests | other |
|---|---:|---:|---:|
| round 16 (`bca3844..200f571`) | +134/−35 | +316/−0 | +248/−1 |
| round 17 (`200f571..65333eb`) | +68/−21 | +269/−16 | +749/−1 |
| whole change (`origin/develop..65333eb`) | +1539/−53 | +4351/−36 | +6425/−3 |

## Its reasoning

> The stronger evidence is consecutive authorship failures **after preventive
> enumeration** — not the absence of `APPROVE` alone.

It was given the previous monitor's recommendation, told that it was adopted and its
enumeration executed, and told that round 17 then found two more defects in the code
that enumeration had just been run over. Its reading:

> The prevention remedy remains unproven. The grid grew from 48 to 64 cells after the
> reviewer identified omitted failure transitions. The record also reports a regression
> test and RED harness requiring correction within round 17. This demonstrates useful
> detection and correction, but **not a stable boundary that prevents remedies from
> generating the next findings.**

On the deliverable it was even-handed, and declined the overreach the previous monitor
was partly declined on:

> The deliverable gains protection for legitimate duplicate spellings and more faithful
> reporting of the retained trained model. Those are connected to "correctly, on every
> path this change touches"; generated training is a substantive output, so **dismissing
> all reporting work as unrelated would be unjustified.**

But it holds the structural point: neither round adds another *forwarding* connection,
and each remedy enlarges what must stay correct before the original deliverable can
clear its gate.

## The question it was asked, and its answer

The prompt asked it directly: the previous monitor's recommendation was adopted and the
next round still found two defects, so **if you recommend the same thing, say what would
make it work this time, or say that you cannot.**

> **I cannot show that repeating the previous escalation or expanding its grid will work
> this time.**

That is the most important sentence in this file. It is not a claim that the work is
wrong; it is a refusal to predict that continuing the current method converges — and the
same restraint that made three earlier monitors worth acting on.

## The main context's reconciliation

`policy:main-context-ownership` — a monitor's output is a finding to reconcile, never a
verdict to adopt. What is adopted, and what is not:

**Adopted.** The premise behind the standing instruction has changed shape, and this
should be said plainly rather than absorbed. That instruction rests on: *no `APPROVE`
means problems remain in the fix code.* For rounds 1-15 that was true in the strong
sense — the defects were in **shipped** code the fix had not yet reached. For rounds 16
and 17 it is true only in a weaker sense: the defects are in **the fix's own code**,
each one produced by the previous round's remedy. Round 17 was entirely authorship.
Those are different situations, and continuing is a decision that turns on which one is
believed to be operating.

**Not adopted as a stop.** The monitor does not close the loop, and neither does this
file. Every finding in rounds 16 and 17 was real, reproduced, and on the shipped path;
none was apparatus. The work delivered in each round stands on its own merits.

**Escalated, and this time blocking.** The previous `escalate` was surfaced without
blocking, because a single monitor's recommendation does not outweigh an explicit
standing instruction. Two consecutive escalations, the second declining to predict that
the method converges, is new evidence rather than a repetition — and it lands exactly on
the judgment the maintainer reserved at round 5, when a stop condition last fired and
the answer was an option nobody had listed. Round 18 is therefore held for that decision
rather than opened.

Full suite **2592 passed**; `ruff check .`, `ruff format --check .`, `mypy lizyml/`
clean. CI green. The merge gate's second half has been satisfied since round 15; only
`APPROVE` is missing.
