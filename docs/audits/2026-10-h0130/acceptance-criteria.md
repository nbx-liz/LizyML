# H-0130 proposal — completion criteria for the acceptance review

The H-0130 proposal review run (`h0130-proposal-review`, three rounds) ended without APPROVE.

- Round 1 found defects in the first draft.
- Rounds 2 and 3 found defects in the text the previous revision had added.

A fresh-context relational monitor read all three rounds and returned DRIFTING. Its findings:

- The core policy has been stable since round 1, and no round flagged it again: the seventeen dispositions, the allowance table, the `RefitResultView` Protocol and the `BaseCalibrationResult` marker.
- The findings kept recurring in two open enumerations: documentation sentences, and the should-fail forms of the allowance grammar.
- Round 3's findings also landed on a probe that round 2 had added.

The maintainer decided on 2026-10-14 to close both enumerations with explicit bounds and finish with **one acceptance review** against the criteria below. It is not another full review. This is the same procedure as H-0129 and D14.

## Frozen subject

- The H-0130 entry in `HISTORY.md` (from `## H-0130` to the end of the file), at the head given to the reviewer. That head is revision 3: the round 3 text plus the remedies for round 3's five findings.
- No library code changed.
- Evidence:
  - `docs/audits/2026-10-h0130/probes/census_h0130.py` and its `.out`.
  - GitHub issue #362.
  - The three review rounds of run `h0130-proposal-review`.

## Completion criteria

The proposal is accepted when all five criteria hold. Each row points to the text that must satisfy it. Line numbers refer to the frozen head and may drift by a few lines.

| # | Criterion | Where it is answered |
|---|---|---|
| C1 | The maintainer's decisions are specified without contradiction. **A**: `estimators/` -> `features/` and `estimators/` -> `metrics/` are declared in a same-layer allowance table that the test reads, with no restructuring. **B option 3**: the classes stay in place (pickles unchanged), with a `RefitResultView` Protocol in `training/results.py` and a `BaseCalibrationResult` marker in `calibration/base.py`. The seven no-decision fixes are included. | 管理者の判断 (HISTORY.md ~13728); 方針 1-8 (~13774-13872) |
| C2 | Every acceptance criterion states a behaviour and a should-fail (or should-pass) case. The should-fail case fails on `e8b4c4e` or on a mutated copy, and passes after the change. | 受け入れ基準 1-7 (~13969-14030) |
| C3 | Every factual statement matches the code or the probe output at `e8b4c4e` / the frozen head. This covers counts, `file:line` positions, the claims about the old loader, pickle and dataclasses, and the claim that H-0088's declared edges are resolved. | 実測 (~13747); 規則が縛る位置 (~13911); 互換性 (~13942); Firing rate (~13964); 方針 9's table (~13873) |
| C4 | **The two bounds are closed.** (a) The allowance grammar is defined by a per-line full match (lead line, one empty line, literal header, literal separator, a row regex, the end of the block), and every `\|` line in §2.1 belongs to one of the two tables. Criterion 2(e) holds one representative mutation per position, not a list of forms. (b) The documentation obligation is a closed set read in full: `BLUEPRINT.md`, `ARCHITECTURE.md`, `docs/add-estimator-guide.md` in full, and `README.md`'s `## Architecture` section. A stated reason excludes the other Markdown files. The search probe is removed. | 方針 6 読み方 and the document bullet (~13797-13842); 方針 9 (~13873); 受け入れ基準 2(e), 7 |
| C5 | Round 3's five findings are resolved. (1) The census misses `BLUEPRINT.md:57` and the H-0054/H-0088 tables: they now appear in the known-places table, and C4(b) closes the rest. (2) The registry caller count is removed. (3) The probe comment now says eight, matching its `MOVED` mapping. (4) The fail-open documentation probe is removed. (5) The grammar mutations are covered by C4(a). | as in C4; 方針 6 (the registry bullet); `census_h0130.py:31-34` |

## Questions the review answers

1. Do C1-C5 hold? For each criterion, answer yes or no with evidence.
2. Within the closed document set, is there a sentence that the implementation could leave unchanged and still contradict H-0130's rules, because the proposal states something that makes it look correct? Answer by reading the set in full. A sentence that is merely not yet listed in 方針 9's known-places table is **not** a finding, because the table is declared non-exhaustive and the implementation must read the whole set.

## Bucket rule for findings

This rule is exhaustive. Every finding goes into exactly one bucket.

- **Blocking**:
  - a false factual statement (C3);
  - a contradiction with a maintainer decision (C1);
  - an acceptance criterion without a should-fail case (C2);
  - a bound that is not closed (C4);
  - a round 3 finding not resolved (C5);
  - a design that cannot be implemented as written.
- **Minor**: everything else. This includes:
  - "add one more representative mutation";
  - "list one more known document place";
  - wording.

  Minor findings are settled in the implementation review against real tests and the real documents, not by another proposal round.

The verdict is `APPROVE` when no finding is blocking.

## Order of approval

1. The reviewer returns findings and a verdict.
2. I fix any blocking finding and report every finding to the maintainer.
3. **The maintainer declares acceptance**, not the reviewer. After that, H-0130's status becomes Accepted and the implementation starts on the same branch.

## Limits of the evidence

- `census_h0130.py` is a static `ast` scan of `lizyml/` using the H-0129 checker. It does not execute imports.
- The proposed tests and document rewrites do not exist yet. C2 and C4 are judged on the stated text. The implementation review checks the real tests, and reads the closed document set in full.
