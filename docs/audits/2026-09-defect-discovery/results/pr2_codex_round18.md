# PR 2 — Codex review, round 18 (2026-09-08)

**Scoped to the round-17 remedies**, at head `42148e2`. The first scoped round
since round 10, opened on the maintainer's decision after two consecutive
monitors returned `DRIFTING` / `escalate`.

```
VERDICT: REQUEST_CHANGES
```

**One** blocking finding — down from two — reproduced here before anything was
changed, fixed, RED-verified. Plus one non-blocking finding that turned out to
be the more important of the two.

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1, 2, 2, 2, 1**.

## The result the scoped round existed to obtain

**Round 17's state-publication fix held under attack.** This is the first time in
three rounds that a remedy came back clean when a reviewer went at it directly,
and it is the evidence the maintainer opened this round for:

> Injected failures at CV training, calibration, evaluation, and full-data
> refit. All six grouped fit fields retained their previous object identities.
> Failed tuning at `_run_tune_round` and `_assemble_tuning_result` retained all
> 12 inspected fit/tuning fields. Instrumented attribute access during ordinary
> successful fit and tune calls; neither read the six grouped fields during
> execution.

It also ran the shipped instrument and reproduced its numbers exactly — 64
cells, 54 agree, 2 known bounds, 8 `n/a`, exit 0 — and mutation-tested the new
regression cases, confirming each detects the production line it claims to pin.

## D7's authorship condition fired a third consecutive round, on `d83af2b`

The blocking finding is again in the immediately preceding round's fix.

## Finding 1 — the unbound-call repair refuses a value that trains

`lizyml/core/value_equality.py::_comma_form_matches`. **DC7, DC5.**

Round 17 called `str.split` unbound so a `str` subclass override could not run.
But **`isinstance` reads `__class__` and the unbound descriptor reads `type()`**,
and a *proxy* separates them — an object that is not a `str` and answers
`__class__` with one. Reproduced:

```
each trains, identical booster : True
the pair                       : TypeError: descriptor 'split' for 'str' objects
                                 doesn't apply to a 'Proxy' object
```

Round 17's finding was that the function **raised** where it should compare.
This one is its mirror: the function **refuses** where it should compare. Both
halves of the bound are load-bearing, and the round-17 repair bought the first
by giving up the second.

## The root cause is the declaration, not the guards

Three consecutive rounds have found the next unenumerated expression in this one
function. The reviewer supplied the piece that explains why, as a non-blocking
note: **a raising `__class__` defeats the module's declared bound at
`isinstance`, and it did so at `04f3930` too.** So the declaration —

> *This function does not raise an `Exception`.*

— is quantified over **every Python object**, and no implementation can satisfy
it: a caller can make `__getattribute__` or `__class__` raise and then any
expression fails. That is **DC7 on the declaration itself**, and it is why each
round could find a fresh dunder.

### Asked the module's own named authority

The docstrings already name `_param_dict_to_str` as the authority on value
equivalence. Executed against it rather than reasoned about:

```
Proxy (isinstance str, type not)   -> 'learning_rate=0.5'
Rate (raising __class__)           -> RuntimeError: class unavailable
Text (plain str subclass)          -> 'learning_rate=0.5'
plain 0.5                          -> 'learning_rate=0.5'

type(str(Text('0.5')))          : str
type(str(SelfPrinting('0.5')))  : SelfPrinting
str.split accepts a subclass    : ['0.5']
```

Two decisions follow, and neither is a guess:

- **The proxy must be admitted.** The serialiser emits `learning_rate=0.5` for
  it, so it is a value LightGBM trains on. `text` is now normalised through
  `str()` — which is what the serialiser then does to it — before the unbound
  call. `str()` on a plain `str` subclass returns an exact `str`; on one that
  returns a subclass, the unbound call accepts it anyway.
- **The raising `__class__` is outside the bound, and no guard would change
  that.** The serialiser raises the caller's own exception on it, so the value
  **cannot complete training** under any spelling, so there is no pair to admit.
  (An earlier draft of this line said "never reaches `lgb.train`". The rounds
  17-18 monitor executed the path and corrected it: serialisation happens
  *inside* LightGBM, so `lgb.train` is entered and then fails. The supported
  claim is the one stated here.)

**The bound is restated relative to the serialiser:**

> *Does not raise on any value the serialiser accepts, and admits any pair the
> serialiser would treat as one value.*

That is a closed, executable population. The old one was open, which is what
three rounds were spending themselves on. The expression table on
`_comma_form_matches` now carries `isinstance` as a row whose closure is
**"not closed here"**, with the reason.

## The test is now a relation, not an enumeration

`isinstance(result, bool)` over a generated cross product is an enumeration, and
enumerations are what kept going stale. The generated population now carries a
`__class__` axis (`normal` / `proxy` / `raises`) on the text operand — 108
combinations — and asserts the **relation to the oracle**:

> `values_differ` raises only where `_param_dict_to_str` raises.

with a companion test that neither side of the relation is vacuous, since a
relation whose antecedent never holds is satisfied by any implementation — the
DC6 shape, in the test written to escape an enumeration.

The declared-behaviour guard caught the new axis on its first run, as designed.

## RED verification

```
reverted F1 proxy normalisation -> 26 failed, 415 passed
restored control run            -> 441 passed
```

Full suite **2666 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean; the lifecycle grid exits 0.

## Bounds

The reviewer stated, and this record repeats: **this is a verdict on the
round-17 remedies, not on the whole change.** An unscoped round follows a clean
scoped one.
