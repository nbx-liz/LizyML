# PR 2 — Codex review, round 20 (2026-09-08)

**Scoped to the round-19 remedy**, at head `17b2c7c`. Its question was the
discriminator the rounds 18-19 monitor named: *do the strengthened assertions
and declared bounds survive an independent challenge without another repair to
the preceding remedy?*

```
VERDICT: REQUEST_CHANGES
```

**The answer is no.** Two blocking findings, both in the round-19 remedy, both
reproduced here before anything was changed, both fixed and RED-verified.

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1, 2, 2, 2, 1, 1, 2**.

## The pre-registered escalation point has fired

> **D7's authorship condition has now fired five consecutive rounds, all five in
> `lizyml/core/value_equality.py`.** Rounds 16, 17, 18, 19 and 20.

This context registered the trigger before round 20 ran, and the rounds 18-19
monitor endorsed it: *another same-function authorship failure after the
methodological intervention would directly undermine the reason for continuing.*
It has happened. **The decision goes to the maintainer, and round 21 is not
opened.** The options are in `DECISIONS-PENDING.md`.

The findings are still real, still reproduced, and are fixed here — stopping the
loop is a decision about the next round, not a reason to ship a known defect.

## Finding 1 — the formatter's result kept its overridable behaviour

`lizyml/core/value_equality.py::_as_wire_text`. **DC1.**

Round 19 normalised through `format(value, "")` and **returned the result as it
came**. `format` may hand back a `str` **subclass**, and that subclass's
overrides then decide the comparison. Both directions reproduced:

```
wire forms    : k=0.25 | k=0.50     <- two values
values_differ : False False         <- reported as one (DC1)

wire forms    : k=0.5  | k=0.5      <- one value
values_differ : True True           <- reported as two
```

The first is the serious one, and it reaches training: the reviewer executed
both spellings separately and got boosters at `0.25` and `0.5`.

**The repair is the module's own idiom, applied one step later than before.**
`str.__str__(...)` unbound, on the formatter's result. Executed rather than
assumed, against a subclass overriding `__str__`, `__eq__` and `__len__`:

```
str(x)                   -> 'lied'   exact=False  len=99
str.__str__(x)           -> '0.25'   exact=True   len=4
```

`str()` is not enough — it dispatches to `__str__`, which the same object may
override. Only the base method returns the characters LightGBM will send.

## Finding 2 — the new relation contradicted an older, deliberate contract

`tests/test_core/test_value_equality.py`. **DC7, DC5.**

The serialiser writes `nan` happily, so *"same wire form ⟹ admit"* demands that
two NaNs be admitted as one value. The module **deliberately** reports them as
differing and has said why since round 5: `nan != nan`, so nothing can establish
they are the same value.

Round 19 wrote a relation quantified over the serialiser without checking it
against the contract it was quantified over. That is the same failure the
relation was introduced to end, one level up.

NaN is now a **declared** exception with its reason, and the cost is executed
rather than argued: `_fit(learning_rate=nan)` is refused by LightGBM
(`Check failed: (learning_rate) > (0.0)`), so no verdict on a NaN pair can
change what any model trains on. Both NaNs are in the population **because**
they are the exception, so the exception is exercised rather than written down.

## A third test that passed for the wrong reason

Recorded because it is the third in this PR, and each time only RED verification
caught it.

The shipped-path pin for finding 1 first wrote the other spelling as a **float**
(`eta: 0.50`). A number routes through the comma-form step, which refuses
correctly even with the fix reverted — so the test passed while pinning nothing.
The other spelling is now text, matching the reviewer's own reproduction, and
the revert takes both tests red.

## Reported and not fixed

- **`values_differ([], "")` is `True`** although both write `k=`. Pre-existing —
  identical at `0b45250` — and outside the scoped remedy. Non-blocking.
- **The derivation check does not detect a *newly added* string lookup.** The
  reviewer patched the module source in memory to add one and the companion test
  still passed: it verifies that declared names are still present, not that
  every actual lookup is declared. No missing lookup exists today. Non-blocking,
  and a real weakness in a check this context wrote.

## What the reviewer verified clean

- Both `KNOWN_BOUNDS` entries still disagree in both directions; **74**
  non-exempt same-wire witnesses and **78** different-numeric ordered pairs
  survive, so neither implication is vacuous.
- The derived population enumerates **63** names — 57 serialiser-accepted, 4
  refused, 2 construction skips (`__init__`, `__new__`), and it constructed
  callable overrides for both skips and found no live defect.
- The `format`/`str` split is applied on the correct side of each formatter.

## RED verification

```
reverted F1 exact-str coercion -> 2 failed, 671 passed
restored control run           -> 673 passed
```

Full suite **2898 passed, 62 skipped**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean; the lifecycle grid exits 0.

## Bounds

Scoped to the round-19 remedy; not a verdict on the whole change. The reviewer
could not read issue #283's body (no GitHub auth in its sandbox), so agreement
between the `KNOWN_BOUNDS` exemption and that issue is verified here rather than
by it.
