# PR 2 — Codex review, round 17 (2026-09-08)

Unscoped, at head `04f3930`.

```
VERDICT: REQUEST_CHANGES
```

Two blocking findings, both reproduced here at that head before anything was
changed, both fixed, both RED-verified.

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1, 2, 2, 2**.

## D7's authorship condition fired again, on `a7ac071` — and this round is 100% authorship

Both findings land in code round 16's fix wrote. Not one of the two is older.

> **D7 authorship condition — a defect found in round N+1 inside the code round
> N's fix wrote — fired for the second consecutive round. The commit is
> `a7ac071`, decision 13's fix.**

This shape has occurred before: rounds 5-6 were the same, and that is what
triggered the D5 stop-condition question then. The recurrence is stated here as
the record; the disposition belongs to the maintainer under the standing
instruction, and the rounds 15-16 monitor's `escalate` has already been surfaced.

**What the two findings have in common is one sentence.** Each fix declared an
invariant and executed it only over the cases the *finding* named:

- decision 13's invariant — *`applied_training_params` describes `fit_result`* —
  was executed over five **success** lifecycles and zero failure transitions;
- decision 13's other invariant — *no expression on a caller's value sits outside
  a guard* — was executed over the **element** operand and not the text one.

The lesson is not "probe harder". It is that the set to execute over is the
**invariant's own quantifier**, and the maker is the party least able to see
where that quantifier was silently narrowed to the example that prompted it.

## Finding 1 — a rejected fit overwrites the retained model's state

`lizyml/core/model.py`. **DC3, DC5.**

`fit()` published each piece of state at the point it happened to be available,
which is before the fit has succeeded. A **refused** call therefore rewrote the
state of the model that was kept:

```
the fit trained with ratio : 0.2
before tune                : (0.2, 0.2)
after tune                 : (0.2, 0.2)
a rejected fit             : LizyMLError
same trained adapter       : True
after the rejected fit     : (0.45, 0.45)
```

The adapters are untouched and the reports describe the attempt that failed.

### The set, enumerated and executed before the fix

Every `self._*` that `fit()` writes before it succeeds, under both failure
points — refused by a gate, and raised mid-training:

| written | rejected fit, before | after |
|---|---|---|
| `_applied_training_params` | **overwritten** | kept |
| `_X` / `_y` | **overwritten** — 200 rows became a 90-row frame with a column the retained model never saw | kept |
| `_metrics` | published before the refit could fail | committed with the result |
| `_provider` | rewritten, but from `cfg`, which cannot change between calls | unchanged |
| `_run_dir` | kept | kept |

`_applied_training_params` is this PR's code. **`_X` / `_y` is pre-existing** —
it is what SHAP and the diagnostics read — and it is folded in because the repair
is the same line move, and nothing in `fit()` reads either mid-flight.

Executing the same question one method over found `tune()` doing it too:
`self._X, self._y = X, y` sat before the study, so a study that raised left the
retained fit describing rows it had never seen. Same repair, adjacent method.

**The fix is to commit with the result.** Every value a reporting surface reads
about "the fit" is now published in one group at the end of `fit()`, with nothing
between the assignments that can raise, and the same at the end of `tune()`.

## Finding 2 — the text operand can defeat the comma-form step

`lizyml/core/value_equality.py::_comma_form_matches`. **DC7, DC5.**

Round 16 guarded `float(element)` and `str(element)`. `text.split(",")` was never
guarded, and a `str` subclass is still a `str`, so its override runs:

```
{'learning_rate': Text('0.5')}              TRAINED
{'eta': 0.5}                                TRAINED
{'learning_rate': Text('0.5'), 'eta': 0.5}  RuntimeError: split unavailable
```

**The repair is not another `try`.** A `try` around a subclass method call would
satisfy a no-raise assertion and still *refuse* a legitimate pair — one
parameter, one value, written twice — which is the other half of the defect this
module exists to prevent. The module's own idiom is to call the base method
unbound, so the override cannot run at all: `str.split(text, ",")` returns exact
`str` parts, and everything derived from them is ours again.

Two facts were executed before writing it, not assumed:

```
str.split unbound      : ['0.5'] ['str']
element str() subclass : Text          <- str(element) CAN return a subclass
```

The second is why `printed.strip()` needed the same treatment: an element whose
`__str__` returns a `str` subclass puts a caller override on the *derived* value.

The function's docstring now carries the population as a table — expression ×
owner × how it is closed — rather than a sentence, and the module's declared
bound says "inside a `try`, **or** through the base type's method unbound",
because the second closure was already in use at `_printed_forms_differ` and the
bound did not mention it.

## The instrument was extended, because the reviewer named its gap

> "The lifecycle grid omits failed-fit transitions, so its 48 cells cannot
> establish this property."

Correct, and it is the same failure one level up: all six lifecycles were success
orderings. `report_lifecycle_grid.py` now runs **64 cells** over eight
lifecycles, the two new ones being `fit -> tune -> refused fit` and
`fit -> tune -> failed training`.

```
cells: 64 (40 from decision 13's five lifecycles, 24 from the three added
           because a declared set is only as good as its declaration)
  agrees: 54    known-bound: 2    n/a: 8    DISAGREES: 0
```

## A regression test that passed for the wrong reason

Worth recording, because it is the hunted class found inside this round's own
remedy. The first version of the `_X` / `_y` claim ran `fit -> tune -> rejected
fit`. **`tune()` also assigns `_X` / `_y`**, so the assertion held against a
build where `fit()` never assigned them at all — the RED verification caught it
by not going red. The claim is now its own test with no intervening `tune()`, and
the tune case is separate.

The RED harness had the same shape of error: deleting a moved assignment is not
the "before" state, because the tests then read whatever the previous successful
call left. Both revert cases now move the assignment back to where it was.

## RED verification

Each fix reverted independently, against both regression files:

```
F2 text operand (split)                 -> 21 failed, 346 passed
F2 text operand (strip)                 ->  8 failed, 359 passed
F1 commit point (fit _X/_y)             ->  3 failed, 364 passed
F1 commit point (overlay)               ->  3 failed, 364 passed
F1 commit point (tune _X/_y)            ->  1 failed, 366 passed
F1 commit point (fit _X/_y, moved back) ->  2 failed, 365 passed
restored control run                    -> 367 passed
```

Full suite **2592 passed**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean.
