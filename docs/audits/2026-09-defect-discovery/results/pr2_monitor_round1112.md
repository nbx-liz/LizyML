# PR 2 — relational monitor, rounds 11-12 (2026-09-08)

Spawned before round 13. Given the numbers unsoftened — blocking per round
**1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3, 2 (+1 found by the main context)** — and asked
two things: whether the loop is converging, and whether round 12's seam
enumeration is the closed population or whether it could name a merge the scan
would miss.

```
VERDICT: CONVERGING
RECOMMENDATION: continue
```

## What it measured

The deliverable moved in both rounds, which is what ruled out `DRIFTING`. It
measured the ratio rather than asserting it, stripping docstrings and comments
by AST:

| | code-only production | tests | docs |
|---|---|---|---|
| round 11 | +25 / −4 | +237 / −14 | +499 / −3 |
| round 12 | +29 / −9 | +247 / −8 | +605 / −66 |

Periphery-to-deliverable ≈ 29:1 in **both** rounds — flat, not accreting.

**It tested the authorship question rather than adopting this context's
reading.** It executed `values_differ` from `f74fc06` (pre-round-11) and
`5fd6169` (post) and found both return `True` for `np.array([1., 2.])` vs
`(1., 2.)`. So round 12's finding 1 is in a *file* round 11 changed and predates
the step round 11 wrote. The D7 authorship pattern the maintainer rescinded does
not fire.

## Its three objections to the seam table, and what was done with each

Taken as findings to reconcile, not as a verdict. All three were acted on.

**1. The scan could not see the construct its own findings lived in. Correct,
and the more serious of the three.** The declared construct set was `{**a, **b}`,
`.update`, `.setdefault`, `|` and two named helpers. Round 12's finding 2 is
`merged["verbose"] = -1` and finding 3 is `resolved["num_leaves"] = …` — both
`d[k] = v`, which was not declared. The table listed them under the *nearest*
declared construct, so they were found by reading adjacent code and the
population was closed by proximity.

The construct set now includes `d[k] = v`, `dict(a, **b)` and `f(**x)`.
Candidates: 24 → **48**.

**2. The seam it named is real, and is fixed.** Not adopted on its word —
executed first:

```
space = {learning_rate: [0.001, 0.01], eta: [0.4, 0.5]}
-> every trial sent both spellings to lgb.train and trained at learning_rate
-> best_model_params: {'learning_rate': 0.0064, 'eta': 0.4545}
```

`sample_params` writes one key per dimension, so two dimensions spelling one
LightGBM parameter both land in the trial dict. **The `eta` dimension was
sampled, optimised over, and had no effect on any trial** — the study ranked
trials on an axis that did nothing, and `best_model_params` recorded the dead
value for the `fit` afterwards to carry (DC1 + DC6). `check_duplicate_identities`
had three callers, none over the space: the third layer decision 6 declares and
had not reached.

`check_duplicate_space_dimensions` is wired before the study starts. There is no
same-value exemption here, unlike the dict surfaces — two dimensions sample
independently.

```
Firing rate: 0/69 of pre-existing configs carrying a category:model search space
             (1/70 including this change's own regression test)
```

**3. The instrument was not shipped.** Correct, and DC3 by this repository's own
rule: the decision-8 table was derived from a scan that lived only in the
scratchpad, so it could not be regenerated. Shipped as
`instruments/parameter_merge_seams.py`, with what it *cannot* do stated in the
module docstring — the hint-word filter is a heuristic over identifier text, not
a type analysis, and `HINTS` is a named constant so "the scan missed it" is
checkable.

## The precedent this follows

The rounds 10-11 monitor named two layers as candidate scope; probing them
before the round turned one into a fixed defect with its firing rate and the
other into a pinned assertion, rather than into round 12's findings. The same is
done here: the named seam was executed and fixed in this round.

## Its prediction, recorded unsoftened and not adopted

> Every unscoped round (1-5, 7, 11, 12) has named a `lizyml/` file. Round 13
> reviews 29 never-reviewed code-only production lines plus one unprobed seam.
> The record predicts at least one finding; it does not predict `APPROVE`.

The unprobed seam is no longer unprobed. Full suite **2447 passed**.
