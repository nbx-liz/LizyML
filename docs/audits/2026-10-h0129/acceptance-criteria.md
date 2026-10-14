# H-0129 proposal — completion criteria for the acceptance review

The H-0129 proposal review run (`h0129-proposal-review`, three rounds) ended without APPROVE. Round 1 found
defects in the first draft, and rounds 2 and 3 found defects in the text the previous revision added. The
maintainer decided on 2026-10-14 to finish with **one acceptance review** against the criteria below, not
another full review. This follows the same procedure as D14 and H-0128.

## Frozen subject

- The H-0129 entry in `HISTORY.md` (from `## H-0129` to the end of the file) at the head given to the
  reviewer, which is revision 3. That is the round 3 text plus the remedies for round 3's two findings.
- No library code changed. Evidence: `docs/audits/2026-10-h0129/probes/` (`layer_rules.py`, `mutations.py`
  and their `.out` files), GitHub issues #360 and #362.

## Completion criteria

The proposal is accepted when all four hold. Each row points to the text that must satisfy it.

| # | Criterion | Where it is answered |
|---|---|---|
| C1 | The maintainer's decisions are specified without contradiction. Option B: `core/param_domain`, `core/specs` in Layer 0. Option 2: every §2.1 rule is enforced, and the existing violations form an exact-count exception list pointing to #362. `persistence.export` keeps its check and signature. | 管理者の判断 (HISTORY.md:13581-13584); 方針 3-4 (:13608-13612); 方針 5-7 (:13613-13633) |
| C2 | Every acceptance criterion states a behaviour and a should-fail (or should-pass) case. The should-fail case fails on `050949b` or on a mutated copy, and passes after the change. | 受け入れ基準 1-8, 3b, 5b (:13661-13671) |
| C3 | Every factual statement matches the code or the probe output at `050949b` / the frozen head: counts, `file:line` positions, "imports only", "0 occurrences", and the Python and CI facts. | 実測 (:13586-13601); 規則が縛る位置 (:13636-13640); 互換性 (:13642-13646); Firing rate (:13656-13659) |
| C4 | The round 3 findings are resolved. (a) R5 refuses relative dynamic imports (`importlib.import_module(".x", "lizyml")`, `__import__(..., level != 0)`). (b) `pyproject.toml` is read with `tomllib` on 3.11+ and `tomli` on 3.10, and the test runs in the 3.10 lane. | 方針 7 R5 (:13630) and 方針 5 (:13618); 受け入れ基準 3, 3b |

## Questions the review answers

1. Do C1-C4 hold? For each criterion, the answer is yes or no with evidence.
2. Is any rule of BLUEPRINT §2.1 left unenforced, or any rule enforced that §2.1 does not state, except where the proposal declares it as a bound?

## Bucket rule for findings

This rule is exhaustive. Every finding goes into exactly one bucket.

- **Blocking**: a false factual statement (C3); a contradiction with a maintainer decision (C1); a round 3
  finding not resolved (C4); a §2.1 rule left unenforced without a declared bound; a design that cannot be
  implemented as written.
- **Minor**: everything else. This includes "make this test input sharper", "add one more should-fail
  spelling", and wording. Minor findings are settled in the implementation review against real tests, not
  by another proposal round.

The verdict is `APPROVE` when no finding is blocking.

## Order of approval

1. The reviewer returns findings and a verdict.
2. I fix any blocking finding and report every finding to the maintainer.
3. **The maintainer declares acceptance**, not the reviewer. After that, H-0129's status becomes Accepted
   and the implementation (M3) starts.

## Limits of the evidence

- The probes are static `ast` scans of `lizyml/`. They do not execute imports.
- No test of the proposed layer test exists yet. C2 is judged on the stated cases, and the implementation
  review checks the real tests.
