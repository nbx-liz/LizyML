# PR 2 — relational monitor, rounds 10-11 (2026-09-08)

Spawned before round 12. Given the numbers unsoftened — blocking per round
**1, 1, 2, 2, 1, 3, 2, 4, 3, 2, 3** — and asked, besides convergence, a concrete
question: **is there a surface of this change no round has examined?**

```
VERDICT: CONVERGING
RECOMMENDATION: continue
```

## What it measured, and how it read the verdict

> The deliverable moved for the first time in the window. Round 11 changed
> executable production code in `_model_tuning.py`, `model.py` and
> `value_equality.py`, two of them on the merge path the previous monitor had
> flagged as *"reviewed once since it last changed, not declared clean."*

And it refused to let the verdict be read as more than it is:

> That is convergence **in direction**, not in count: findings went 2 → 3, and
> the periphery still grew faster in absolute lines (+157 vs +66). Read
> `CONVERGING` as "the loop is back on the deliverable," **not "near
> `APPROVE`."**

It also verified round 10's claim rather than taking it: filtered of comments,
round 10's production diff is empty.

## One class it measured closed

> `grep -rn 'overlay_params(' lizyml/` returns exactly five call sites
> (`model.py:468,471,502`; `_model_tuning.py:457,458`), all identity-aware. The
> seam enumeration decision 5 opened is now closed at **4/4** — real evidence
> that the merge-path class is finite.

## And what it predicted for round 12

> Each of rounds 1-5, 7 and 11 — **every non-remedy-scoped round** — names at
> least one `lizyml/` file in its findings. Round 12 reviews +66 production lines
> written yesterday and never reviewed. The record does not predict `APPROVE` in
> round 12; it predicts at least one further finding, and if that finding lands
> *inside* the round-11 fix code, that is the D7 authorship pattern the
> maintainer rescinded — worth watching, not deciding.

## The unexamined surfaces — it named two, and both were probed here

Not adopted on its word. Each was executed before anything changed.

**1. `calibration.params` had a name check and no identity check. Real, and
fixed.** The same-layer rule is declared for "every layer", and
`check_duplicate_identities` had exactly two call sites. Measured:

```
calibration.params = {"learning_rate": 0.001, "eta": 0.5}
-> trained; the calibrator's lgb.train received BOTH spellings
```

The fourth route into the estimator — the one H-0093 named as *"a fourth route
that no config-side gate sees"* — was the one layer with a name check and no
identity check. Wired inside `check_calibration_param_names`, at the same entry
and with the same provider. Both directions tested, and:

```
Firing rate: 0/22 of pre-existing configs carrying calibration.params
```

**2. The smart layer merges by spelling. Probed, and it is correct.** The reason
every other layer merges by identity is that the estimator resolves aliases.
Measured over the provider's own list: **no smart parameter name has a single
LightGBM alias** — they are LizyML's own names and the library has never heard
of them, so there is no second spelling for one to arrive under. Recorded as an
assertion rather than left as an assumption, because "no aliases" is exactly the
claim that goes stale when a name is added.

**3. Its "stale, not unexamined" note is kept as-is**: export → load →
refit/tune was bounded to round 7's evidence, and that path has since gained the
`_merge_params` refusal and the trial reorder. A loaded artifact whose config
carries two spellings will now be refused on a **re-fit**; `load` and `predict`
do not go through `_merge_params` and are unaffected. Round 12 is unscoped, so
it reaches this path.

## Its recommendation, adopted

> **`continue`** — round 12 unscoped is already decided and is the right
> instrument; the two named layers are candidate scope for it, not a redirect.

Adopted, with one difference recorded plainly: the two named layers were not left
as candidate scope. One was a reproduced defect and is fixed with its regression
tests and its firing rate; the other was probed, found correct, and pinned.
Leaving a known instance of a class for the next round to find is how the
previous monitor said the next finding gets manufactured.

Full suite **2420 passed**.
