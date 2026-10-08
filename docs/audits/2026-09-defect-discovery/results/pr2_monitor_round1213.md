# PR 2 — relational monitor, rounds 12-13 (2026-09-08)

Spawned before round 14. Given the numbers unsoftened — blocking per round
**1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2, 3** — and asked two specific questions
besides the verdict.

```
VERDICT: CONVERGING
RECOMMENDATION: redirect
```

**The first `redirect` of this run**, and it earned it: it falsified a claim
round 13 had just made, by execution.

## What it measured

| | code-only production | tests + docs added |
|---|---|---|
| round 12 | +72 / −8 | 1262 |
| round 13 | +94 / −1 | 516 |

The deliverable grew while the periphery **more than halved**. All four
round-13 production files sit on the declared path. It noted its docstring
stripping is stricter than the previous monitor's, so the direction is
comparable and the scale is not.

## Question 1 — is `values_differ` closing a grammar or chasing it?

Round 13's fix claimed to close it: LightGBM's serialiser joins **every**
`list`, `tuple`, `set` and 1-D ndarray with `","`, so applying that uniform rule
closes the class rather than adding one more special case.

**The monitor found the claim false, and this context verified it before acting
on it.** `_comma_form_matches` asked `isinstance(sequence, list)`, and
`_as_plain_sequence` normalises only `collections.abc.Sequence` — which an
ndarray is not, and which a scalar is not at all. So the rule reached **one of
the four types its own docstring named**.

Re-executed here, with LightGBM's serialiser as the oracle:

```
                        wire form A     wire form B     same wire   refused?
ndarray vs its text     'p=1.0,2.0'     'p=1.0,2.0'     True        REFUSED
scalar vs its text      'p=0.5'         'p=0.5'         True        REFUSED
set vs its list         'p=1.0,2.0'     'p=1.0,2.0'     True        REFUSED
list vs its text        'p=1.0,2.0'     'p=1.0,2.0'     True        trained
```

Three of four pairs reach LightGBM as the **byte-identical string** and were
refused. The claim that round 13 closed the class was wrong.

**Adopted, with one difference recorded plainly.** The monitor declined to press
the `set` case, and this context agrees but goes further than "not pressed":
`set` and `frozenset` are now **excluded on purpose**, because a set has no
order and every sequence parameter here is positional — two sets printing alike
did so by hash accident. The exclusion is asserted by a case rather than left
looking like the gap it was. `None` is excluded for a different stated reason:
`_param_dict_to_str` skips it entirely, so it means "not sent", not `"None"`.

`_wire_elements` now answers "what would LightGBM join for this value" for every
type, and the test uses **the serialiser itself as the oracle** over the whole
type set — which is how the round-13 claim should have been checked when it was
made. Executing it also established a fact that had been assumed: LightGBM does
not join a `pd.Series`, it **refuses** one with `TypeError`. That is pinned.

## Question 2 — was retitling the seam claim honest, or a retreat?

> Honest correction. The claim was falsifiable, was falsified twice within a
> round of being made, and the instrument now states what it asserts (executed
> rows) versus what it does not (closure), with `HINTS` named so "the scan
> missed it" stays checkable. The alternative — making it true by a different
> method — would need a type analysis over an open Python codebase, and no such
> method exists here.

It also noted that HISTORY's round-12 heading still reads 継ぎ目の全数列挙 above
its 24-construct set, with the retraction appended in the round-13 section. That
is the append-only convention rather than a defect, and it is left as it is.

## Its recommendation, adopted

> `redirect` — close the normalisation over **every** type the serialiser joins,
> with the test executed across that whole type set rather than the one type
> reached, because a round-14 `values_differ` finding is now a measurement I
> have already taken rather than a prediction.

Adopted in full, before round 14 opens. It also declined
`take-stop-condition` on the grounds that round 13's findings were not in round
12's code, so D7's authorship condition did not fire — which matches this
context's own reading.

Full suite **2498 passed**.
