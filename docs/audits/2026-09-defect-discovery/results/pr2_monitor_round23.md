# PR 2 — relational monitor, rounds 2-3 (2026-09-07)

Read-only, fresh context, inheriting neither the maker's rationale nor the
reviewer's conclusions. It was given two facts that cut in opposite directions:
that round 3 produced two findings after one each and reached a file no earlier
stage had touched, and that its second finding was one the maker had already
filed and declined to fix.

```
VERDICT: CONVERGING
RECOMMENDATION: continue
```

## What it measured

| Stage | prod | test | spec docs | loop records |
|---|---|---|---|---|
| R1 remedy `0396c08..48b3780` | +177 | +194 | 56 | 122 |
| R2 remedy `..13dc7dd` | +123 | +94 | 16 | 109 |
| archive `..31a25a6` | 0 | 0 | 4 | 120 |
| R3 remedy `..db483bb` | +99 | +123 | 29 | 124 |

**On-deliverable blocking count is 1, 1, 1.** Round 3's second finding is the
reviewer reaching into an adjacent surface, not a third defect in the centre.

## The three judgements it was asked for

**Round 3 was not a remedy's remedy.** R2 was — the drift shape. R3's finding
sits on `_merge_params`, the production entrypoint, and the maker's own
measurement widened it beyond the report: `_COMMON_DEFAULTS` injects
`learning_rate` on every fit, so a config setting any defaulted parameter under
an alias has been inert since the library shipped.

**`adapter.py` is the deliverable's actual location, not scope growth.** Same
single path `fit(params=)` → `lgb.train`, second seam on it, and the two-seam
necessity was measured rather than assumed.

**#280 is discipline, not avoidance**, and it named the distinguishing fact:
the maker fixed the **DC5 half at its own cost** (`31a25a6`, the declarations
that falsely claimed those collisions were refused) before the verdict arrived.
"Avoidance would have deferred the false declaration too."

## What it flagged

**The plan's file list is stale, and this is Phase 3's third scope overrun.**
The user-visible config behaviour change is outside the plan's `Files` list. It
is recorded in H-0094 decision 5 and the CHANGELOG rather than absorbed
silently, but "the main context owns whether the plan text is now stale."

→ Adopted: `phase3-plan.md`'s PR 2 section now carries the actual scope, round
by round, with why each move happened, and names the pattern — the plan's file
list was written from the defect's symptom, and the fix lives where the
behaviour is actually decided. Same class as #276.

**Loop records (471) now exceed shipped production (377)**, crossing the parity
the rounds-1/2 monitor flagged at 249/297. Its own reading: a cost signal, not
drift, since these are records under `docs/audits/` rather than apparatus inside
the artifact.

## The tripwire, adopted as a pre-registration for round 4

> the *reviewer* is broadening (finding #2 on a new surface; "Checked and clean"
> now sweeps 140/307 registry entries, 66 name-gate tests, an export test that
> could not set up). If round 4's findings are all adjacent-surface with no
> change to `_merge_params`/`_build_params`, that is drift and the disposition
> changes.

The main context adopts this **before round 4's verdict is seen**: a round-4
finding that produces no production change on the fit-params path takes the stop
condition, and PR 2 goes to the maintainer rather than to a round 5. A finding
that does change that path is the loop working, and is fixed.
