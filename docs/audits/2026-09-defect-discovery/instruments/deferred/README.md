# Deferred — the Phase 3 completion instrument

**Do not run these. They do not work as archived.** They are here because the
alternative was losing them a fourth time, not because they are ready.

`phase3-plan.md` §8 specifies how Phase 3 completion is measured: not by counting
merged PRs and not by one named test per issue, but by a per-issue population
manifest (`phase3_manifest.json`) that `phase3_gap.py` checks six propositions
against, with `check_derivations.py` asserting every declared population is
derived from the shipped code, and 29 unit tests in `test_phase3_gap.py`.

PR 0's Files list named all four. They were built in `/tmp/lizyml-discovery-plan/`,
which was lost, and were recovered here from the session transcript by
`../recover_from_transcript.py`. The recovery replays `Write` and `Edit` tool
calls; **changes a script made to another file cannot be replayed**, and the
manifest was edited by scripts. What came back is therefore an earlier state of
the manifest than the one the plan describes.

## What is verified

- `phase3_gap.py` is current: it carries the `plan_pr` / `github_pr` split and
  the `require_github_pr` refusal that §8 says round 5 introduced.
- Two lost manifest edits were reapplied by re-running their own recovered
  scripts (`fix_manifest_pr.py`, `fix_263.py`), so the archived manifest has the
  `plan_pr` / `github_pr` shape and #263's after-SHA population of 19.
- `check_derivations.py` runs. After a third repair — #271's `derived_from` was
  over-escaped (`\\s` where `\s` was meant) and printed 0, exactly the defect §8
  says the checker caught on its first run — it reports **14 of 15 derivations
  agreeing**.

## What is stale, and why this is a design decision rather than an edit

1. **#271's population is not constant across the run.** It declares 92
   `HISTORY.md` proposals, measured at `5712f41`. Every Phase 3 PR adds a
   Proposal, so the derivation already reports **93** with H-0092 in the tree and
   will keep climbing. Proposition 4 compares a *collected parametrisation*
   against a *declared literal*, so a population that grows with the run cannot
   satisfy it. Either the row derives its cardinality at the merge SHA, or
   proposition 4 stops comparing against a literal — that is a change to the
   instrument's contract.
2. **#265's row names a test that does not exist.**
   `tests/test_training/test_inner_valid_gap_propagation.py::test_gap_propagation_per_resolution_path`
   was never written. The pin landed as
   `tests/test_training/test_inner_valid_purge_embargo.py::TestExplicitInnerValidDoesNotInheritGap`,
   which follows the plan's own §1 step 3 — extend the population's permanent
   test rather than add a single-case file — but the manifest was not updated.
3. **#266's row names a node that does not exist and counts the wrong thing.**
   It declares `test_documents_state_the_shipped_version` (the test is
   `test_declared_version_matches_code`) and a population of 2, meaning the two
   constants; the parametrisation is over *sites*, of which there are 7. The
   population and the thing that enumerates it are not the same set.
4. **Eight of the 29 unit tests are missing.** §8 says `test_phase3_gap.py`
   ships 29 tests; the recovered file collects **21**
   (`pytest --collect-only docs/audits/.../deferred/test_phase3_gap.py`). The
   eight that are gone were added by an unrecovered edit, and nothing here says
   which propositions they covered — so the instrument's own test coverage is
   not merely unshipped, it is unknown.
5. Rows for PRs 1–9 name tests that do not exist yet **by design** — those are
   the future PRs' deliverables and are not defects here.

Shipped as-is, the instrument would report a correctly repaired issue as
incomplete: a gate that no real input can pass, which is DC7 — in the tool whose
own review rounds 5–7 were spent removing DC7. That is why it is deferred rather
than patched into PR 0.

`phase3-plan.md` §8 still narrates the instrument as it behaved in the lost
scratchpad — "both exist and both run", 29 passing tests, 15 of 15 derivations.
Those sentences are false of everything in this repository; §8 carries a
superseding banner saying so, and this file is the current statement.

## The uncovered guarantee

**`phase3-plan.md` §8 — end-of-run completion measurement — has no shipped
instrument.** Until one lands, "Phase 3 is complete" rests on per-PR judgement,
which is the DC5 shape §8 exists to prevent. The remaining work is one pass over
`phase3_manifest.json` reconciling every row against the tests that actually
land, plus the proposition-4 decision in item 1. It belongs to a later PR in
this run, before the last issue is closed.

`test_phase3_gap.py` is not under `tests/`, so pytest does not collect it; the
paths in all four files still point at the lost `/tmp/lizyml-discovery-plan/`
scratchpad and must be repointed when the tool is finished.
