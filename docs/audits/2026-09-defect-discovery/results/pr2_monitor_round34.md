# PR 2 — relational monitor, rounds 3-4 (2026-09-07)

Read-only, fresh context. It was given three facts the main context believed cut
**against** continuing — that round 4's findings were introduced by round 3's
remedy, that the pre-registration had not fired, and that the remedy had now
touched seven files — and asked to judge them.

```
VERDICT: CONVERGING
RECOMMENDATION: continue
```

## It corrected the main context's framing

The capsule stated that both round-4 findings were introduced by the round-3
remedy. **Half of that is wrong**, and the monitor checked rather than accepted
it: it read the pre-round-3 adapter (`git show 31a25a6:…/adapter.py`) and found
that the base params dict *always* set the canonical `objective`, so LightGBM's
canonical preference masked any alias-spelled objective.

> R4 finding 1 — `_check_objective_compatible` seeing one spelling —
> **predates R3 and was unmasked by it**, not introduced. Revert-attribution
> cannot tell those apart: reverting the shadow-drop restores the mask.

Only finding 2, the `KeyError`, was authored by round 3.

So the remedy-of-remedy links are R1→R2 and R3→R4 — real, but **not
consecutive**, and PR 1's terminal shape needed consecutive links. Round 3 was a
fresh on-deliverable defect that had been shipping since inception.

## What it measured

Per stage (prod / test / spec / loop records): R1 173/193/53/122 · R2
99/82/14/109 · archive 0/0/2/118 · R3 90/123/29/124 · monitor 0/0/0/93 · R4
110/174/20/116. Cumulative against `origin/develop`: prod +471, tests +907, loop
records +680 — 1.4× production, "a cost signal, not apparatus inside the
artifact".

**Not drift by the capsule's own definition**: drift requires the deliverable
declared clean while it never changes. Production on `fit(params=)` → `lgb.train`
changed in every remedy stage and no finding left that path.

**The population on this path looks exhausted.** Every seam that discriminates
on a parameter name is now identity-based — the merge, the smart-managed
refusal, the default shadow-drop, the special-handling pop — and the two
remaining literal reads (`random_state`, `verbose`) normalise into `user_params`
and then feed the identity-aware drop.

**It verified the closure mechanism itself** rather than taking the record's
word: the test that reads `adapter.py` and asserts the `_pop_by_identity`
population equals the tested one. "A fourth specially handled parameter fails
that test instead of becoming round 5's finding."

## On the round-4 pre-registration

> It was aimed at the wrong axis. It measured *surface* — adjacent vs on-path —
> because the rounds-2/3 monitor saw the reviewer broadening. What actually
> materialised was *authorship*: defects in code the previous remedy wrote. A
> surface tripwire cannot see that, and correctly did not fire.

## Its flag, and the main context's disposition

> R4 shipped two user-visible contract changes (the round count now honours any
> spelling; a new `CONFIG_INVALID` on `fit(params=)` for conflicting spellings).
> That is scope growth #4 on a plan whose file list is already recorded stale.
> The accretion in this PR is contract surface, not apparatus — the main context
> owns whether that still fits H-0094 and #264.

**Disposition: they fit, and the record is made complete rather than argued.**
Both are consequences of the identity rule H-0094 decisions 5 and 6 declare: a
parameter is one parameter however it is spelled. Neither can be dropped without
reintroducing the defect — the round-count extraction *is* special handling
matched by a literal name, and admitting a conflicting pair means picking a
value by dictionary order. What was missing is that only one of the two was in
the CHANGELOG; both are now, since both change what an existing call does.

## The next stop condition, adopted before round 5

Phrased on **authorship** rather than surface, at the monitor's recommendation:

> **round 5 reports a blocking defect in code round 4 itself wrote** —
> `_pop_by_identity` or `check_duplicate_identities`. That would be two
> consecutive remedy-introduced links, which is PR 1's terminal shape. A round-5
> finding that is unmasked pre-existing behaviour is progress and does not trip
> it.

Adopted verbatim, before round 5's verdict is seen. If it trips, PR 2 goes to
the maintainer rather than to a round 6.
