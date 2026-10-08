# PR 2 — relational monitor, rounds 18-19 (2026-09-08)

Spawned before round 20, at head `17b2c7c`.

```
VERDICT: CONVERGING
RECOMMENDATION: continue
```

The second consecutive `CONVERGING` / `continue`.

## What it measured

`git diff --numstat`, production = `lizyml/`, tests = `tests/`:

| interval | production | tests | other |
|---|---:|---:|---:|
| round 18 (`42148e2..0b45250`) | +60/−13 | +140/−23 | +334/−1 |
| round 19 (`0b45250..17b2c7c`) | +52/−14 | +327/−6 | +220/−1 |
| whole change (`origin/develop..17b2c7c`) | +1624/−53 | +4789/−36 | +7067/−3 |

Both rounds changed **only `value_equality.py`** in production, and the lifecycle
instrument did not change at all.

## The question it was asked

Four consecutive authorship rounds in one function is either a loop that cannot
terminate, or a function whose real specification is only now being written down.
The prompt asked which, and what would distinguish them.

> **The evidence favors specification being written down.** Round 19 changes what
> the tests ask: exception safety cannot distinguish correct admission from silent
> conflict, whereas the new implications can. Their immediate discovery of
> additional cases is evidence of increased discrimination. That is different in
> kind from adding another guard.

And then it refused to let that settle the matter, which is the more useful half:

> Deriving attribute names does not derive all behaviors or interactions. The
> hostile generator applies one hostile attribute at a time; the pair population
> remains finite. The recorded initially-green mutation check — until
> formatter-disagreement examples were added — demonstrates that stronger
> assertions still depend on representative inputs. **This is methodological
> progress, not domain closure.**

**Its discriminator, adopted as round 20's actual question:**

> Whether the strengthened assertions and declared bounds **survive the next
> independent scoped challenge without another repair to the preceding remedy.**
> Repeated additions of exceptions, behavioral examples, or declarations after
> each challenge would instead support continued cycling at a higher level.
> **Falling blocker counts alone distinguish neither.**

That last sentence is aimed at this context, and it is right: 2, 2, 2, 2, 1, 1 is
not evidence of anything on its own.

## On the pre-registered escalation point

The prompt disclosed the trigger this context had set itself — a fifth defect in
this function, after both relations and the derived population, goes to the
maintainer with the round-13 reversal on the table — and asked whether it is the
right trigger and the right option.

> The pre-registered round-20 trigger is appropriate: another same-function
> authorship failure after this methodological intervention would directly
> undermine the reason for continuing. Reversing the round-13 admission is an
> appropriate maintainer option because **it revisits the obligation generating
> the repairs.**

And it corrected the framing, which is adopted:

> Identical-wire admission would trade compatibility for a narrower comparison
> contract; **"near-zero attack surface" remains an unproven characterization.**

That phrase was this context's, in the round-19 handoff. It is struck. **This is
the third monitor in a row to catch this context stating something more strongly
than it had executed**, and all three corrections are on the record rather than
quietly applied.

## Also noted, and accepted

- Neither round adds a *forwarding* connection — but the callers in
  `_model_factories.py:893` and `adapter.py:62` connect this work to the declared
  deliverable, and the false admission it fixed is the serious kind: *"training
  could silently select between different requested values."*
- Filing #283 constrained further behavioural expansion, **at the cost of the
  broad admission declaration now carrying an explicit exception**. That is a real
  trade and it is stated in the test as a `KNOWN_BOUNDS` entry with a staleness
  check, not hidden.
- It declined to claim the #283 widening is dead in practice: constructed
  witnesses establish activation in tests, not a production firing rate. Which is
  exactly why #283 asks for one before implementation.

## The main context's reconciliation

Adopted in full. **Round 20 is scoped to the round-19 remedy**, and its question
is the monitor's discriminator rather than a generic re-review: do the
strengthened assertions and the derived population survive an independent
challenge *without another repair to the preceding remedy*.

The pre-registered escalation stands unchanged: a fifth same-function authorship
finding goes to the maintainer, with the round-13 reversal named as an option and
described as the monitor described it — a narrower comparison contract traded
against compatibility, not a proven reduction in attack surface.

Full suite **2895 passed, 6 skipped**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean; the lifecycle grid exits 0.
