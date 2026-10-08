# 2026-09-04 — #271 re-triage, content-by-content

Supersedes `r13_triage_verdicts.py`'s 52 fold-in / 5 no-obligation, which Codex
round 1 found unsound in 7 of 10 sampled entries. Corrected again by Codex
round 2, which reversed 2 of 14.

## Method

The old triage screened the **proposal** side by keyword and never looked at what
BLUEPRINT contains. This one reads every entry against BLUEPRINT's content.

Instrument: `r15_extract.py` builds a per-entry evidence packet — the identifiers
each proposal names, the BLUEPRINT lines they land on by full-token match, the
proposal's own `- Related: BLUEPRINT.md §x` pointer, and any later proposal
citing it. It emits **no verdict**: an identifier hit is not evidence of
specification, because a hit can be a *deprecation* of the very thing the
proposal decided (H-0044). Every one of the 57 got a read.

`r16_tally.py` reports and writes nothing. It re-parses `HISTORY.md` and
`BLUEPRINT.md` on every run rather than trusting the cached evidence file (DC3),
and it **fails** on an unparseable proposal marker, on population drift against
the cache, or on any entry without exactly one verdict (DC1).

## Criterion

An absent proposal owes BLUEPRINT an update iff BLUEPRINT, read today, would
leave a reader ignorant of a decision on the `CLAUDE.md` §3 surface that the
proposal made **and that is still in force** — all three of §3's invariants:
contract, leakage, and responsibility separation.

| status | meaning | obligation |
|---|---|---|
| `SPECIFIED` | BLUEPRINT states the substance; the id's absence is immaterial | no |
| `SUPERSEDED` | a later proposal replaced it; folding it in would reinstate a retired rule | no |
| `NO_SURFACE` | decides nothing on the §3 surface | no |
| `PARTIAL` | part stated, a named part omitted or now stale | yes, bounded |
| `MISSING` | none of it stated | yes |

## Result

<!-- GENERATED:tables -->
<!-- /GENERATED:tables -->

## What review changed

| round | what it reversed |
|---|---|
| Codex r1 | the whole method. `id absent` is not `content missing`; 7 of 10 sampled verdicts unsound (6 `DESCRIBED_WITHOUT_ID`, 1 superseded). The old triage was discarded. |
| Codex r2 | H-0003 `SPECIFIED` → `PARTIAL` (live artifact filenames absent from BLUEPRINT — the same evidence shape H-0011 was already called `PARTIAL` on); H-0086 `MISSING` → `PARTIAL` (`BLUEPRINT.md:1500` does state part of the re-export decision). Plus four instrument defects: the tally ignored its own `unparseable_markers` (DC1), validated a cached snapshot instead of re-deriving from source (DC3), and let three proposal blocks run into the next entry's heading (DC2); the issue body cited `loader.py:26`, an import line, as the verification. |

Both r2 reclassifications were independently re-confirmed here before adoption.

## What this changes about #271 itself

- **57 is not the work.** It is the count of ids absent from a document with no
  stated obligation to cite ids. 33 of the 57 are fully specified without one.
- **The canonical instance holds.** H-0083 is real, and eight more like it.
- **The proposed id-presence gate would false-fail all 33 `SPECIFIED` entries**
  and false-pass on a bare `(H-0083)` citation. It is the wrong repair.
- Two body facts drifted: the eight undocumented public names are **six**, and
  seven proposals (H-0054..H-0060) carry **both** delimiter spellings.
- The original body's claim that none of the 57 was decided-but-unimplemented is
  **withdrawn**, not restated: nothing in this pass measured implementation
  completeness.

## New findings this pass produced

Not in any filed issue:

1. **BLUEPRINT §14.4's Protocol is three methods short of the real one** —
   `parameter_bounds`, `objective_choices`, `metric_choices`
   (`estimators/provider.py:157`, `:171`, `:190`). §2.2 presents §14.4 as the
   multi-algorithm extension IF.
2. **BLUEPRINT:1024's Metric IF is one property short** — `needs_simplex`.
3. **BLUEPRINT:143 contradicts :1077** on `roc_curve_plot`'s task support.
4. **BLUEPRINT:310-313 states a `random_state` default the schema does not have.**
5. **BLUEPRINT:1063's calibrated-metrics enumeration is closed and wrong.**
