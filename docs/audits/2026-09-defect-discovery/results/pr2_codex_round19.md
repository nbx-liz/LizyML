# PR 2 — Codex review, round 19 (2026-09-08)

**Scoped to the round-18 remedy**, at head `0b45250`. The second scoped round,
under the framework the maintainer set: clean means the fixes have stabilised
and the next round goes unscoped; a finding means the loop is still cycling.

```
VERDICT: REQUEST_CHANGES
```

**One** blocking finding, plus one non-blocking that had to be fixed anyway.
Both reproduced here before anything was changed, both RED-verified.

Blocking per round: **1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3, 1, 2, 2, 2, 1, 1**.

## D7's authorship condition fired a fourth consecutive round, on `fdf5cc2`

## Finding 1 — the normalisation used the wrong formatter

`lizyml/core/value_equality.py`. **DC1, DC5, DC7.**

Round 18 normalised a `str`-like operand through `str()`. The serialiser writes
a **scalar** parameter with `f"{key}={val}"` — which dispatches to `__format__`
— and writes sequence **elements** through `_to_string`, which calls `str`.
Those two disagree for a value that overrides one and not the other, and the
consequence ran in both directions:

```
F1a  wire form of the pair  : a=0.5 | a=0.5      <- one value
     values_differ          : True True          <- REFUSED
     each trains, equal     : True
     the pair               : LizyMLError

F1b  wire form of the pair  : a=0.25 | a=0.5     <- two values
     values_differ          : False              <- ADMITTED (DC1)
```

F1b is the serious half: `learning_rate` at `0.25` beside `eta` at `0.5` was
reported as one value, and the booster trained on whichever LightGBM kept.

### Executed the correspondence rather than reasoning about it

Eight values × both formatters, against the installed serialiser:

```
scalar branch -- f-string, i.e. format(val, '')
  OK exact str        wire='0.5'   format='0.5'   str='0.5'
  OK __str__ only     wire='0.25'  format='0.25'  str='0.25'
  OK __format__ only  wire='0.75'  format='0.75'  str='0.5'
  OK both overridden  wire='0.5'   format='0.5'   str='0.25'
  OK proxy            wire='0.5'   format='0.5'   str='0.5'

element branch -- _to_string uses str(x), NOT format
  OK all eight        _to_string == str
```

`format` matches the scalar wire form in all eight; `str` matches the element
wire form in all eight. **The two sides use different functions because
LightGBM does**, so `str(element)` further down is correct as it stands and
must not be "tidied" to match.

## Finding 2 — the `tolist` lookup, reported non-blocking, fixed anyway

`getattr(value, "tolist", None)` swallows `AttributeError` and nothing else, so
a `tolist` **property** that raises came straight out of the function. The
reviewer marked it non-blocking because it predates the round-18 remedy.

It is fixed here because **it falsifies the bound this PR declared one round
ago**: the serialiser accepts that value (`learning_rate=0.5`) and it trains, so
it is inside the bound, unlike the raising `__class__` that decision 15 excluded.

## The real repair: the missing half of the relation

Round 18 replaced an enumeration with an oracle relation, and **built one half
of it**: *raises only where the serialiser raises*. It asserted nothing about
the other half — *admits iff the serialiser joins*. Both of round 19's
directions pass a no-raise test by construction.

Adding `__format__` as another axis would have caught this one instance. The
missing relation catches it under **any** axis:

- same wire form ⟹ `values_differ` is `False`
- both wire forms numeric and different ⟹ `values_differ` is `True`

**It found three more the moment it existed**, in its first run:

| what | disposition |
|---|---|
| `"0.5"` vs a proxy, both orders (8 cells) | **fixed** — decision 15's own gap |
| scalar vs single-element sequence (4 cells) | **filed as [#283](https://github.com/nbx-liz/LizyML/issues/283)** |

The proxy gap is instructive: round 18 normalised inside `_comma_form_matches`,
which only runs when the *other* operand is a sequence, so a proxy compared
against ordinary text never reached it. Normalisation now happens **once, at the
entry**, for both operands.

The scalar/single-sequence case is not fixed here because admitting a
currently-refused pair is a behaviour widening — an `allow` under the Change
Gate — and needs a Proposal with a measured firing rate, not a fix folded into a
review round. Its values stay in the population with a `KNOWN_BOUNDS` entry
naming the issue, so the relation keeps its non-vacuity witnesses, and a
staleness check fails the run if the exemption stops being needed.

## A population derived instead of declared

Four rounds were spent adding an axis the previous round had not thought of:
`__float__`, `__str__`, `split`, `__class__`, then `__format__` and `tolist`.
Every one was chosen by a person, which is why every one was incomplete.

`DERIVED_HOSTILE_NAMES` is now built from **Python's own dunder list** on the
types this module handles, plus the attribute names the module looks up by
string — one hostile variant each, run through the relation, and a companion
test asserting the derivation is a derivation (including that each explicit
lookup still appears in the module source, or the copy has gone stale).

It found nothing new, which is the point of running it.

## A RED verification that did not go red

Worth recording. The first RED run reverted `format` back to `str` and **the
suite stayed green**: the reviewer's own reproduction values were never pinned,
so the population contained no value whose `format` and `str` disagree. Fixed by
adding both of them, in the generated population and end to end on the shipped
path.

## RED verification

```
reverted F1 format vs str            -> 41 failed, 629 passed
reverted F1b normalise at the entry  -> 46 failed, 624 passed
reverted F2 guarded tolist lookup    ->  1 failed, 669 passed
restored control run                 -> 670 passed
```

Full suite **2895 passed, 6 skipped**; `ruff check .`, `ruff format --check .`,
`mypy lizyml/` clean; the lifecycle grid exits 0.

## Bounds

Scoped to the round-18 remedy; not a verdict on the whole change.
