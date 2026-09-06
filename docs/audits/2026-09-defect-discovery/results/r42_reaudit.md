# The no-obligation re-audit

H-0001's reversal exposed a defect in a verdict, not in an entry: it checked
that a section covering the topic **exists**, not that every clause the proposal
decided is **stated**. 33 of the 38 remaining no-obligation entries had been
judged on the same reasoning, and none had been re-checked since the original
batch reads. This pass re-checked all 38.

## Method

Three read-only Codex batches, split by the question each status raises rather
than by size:

| batch | population | question |
|---|---|---|
| A | 16 SPECIFIED | is every clause stated, or does a section merely cover the topic? |
| B | 16 SPECIFIED | same |
| C | 1 SUPERSEDED + 5 NO_SURFACE | is the exemption real? |

Each enumerated its proposals' HISTORY clauses **first**, then searched BLUEPRINT
per clause, and had to name the BLUEPRINT line for every clause it called
`stated`. H-0001 was handed to A and B as the calibration case.

`r35_verify_no_obligation.py` checks the union arithmetically;
`r36_control_no_obligation.py` proves it refuses — baseline passes, and all six
perturbations are caught: a skipped entry, a `stated` clause with no line, a
range outside the block, an out-of-population proposal, a `superseded` clause
naming nobody, and one naming a proposal that does not exist. The merged
enumeration **PASSES**, and `--control H-0020` against that real file still
fires.

## Result

**24 of 38 entries reversed.** 365 clauses enumerated: 198 stated, 78 missed,
52 not_section_3, 37 superseded.

| | before | after |
|---|---|---|
| obligation entries | 19 | **42** |
| no obligation | 38 | **15** |
| clauses | 65 | **137** |
| distinct edits | 47 | **79** |
| firing rate | 38/57 | **15/57** |

The 15 survivors: H-0000, H-0010, H-0012, H-0013, H-0021, H-0022, H-0023,
H-0027, H-0029, H-0031, H-0033, H-0037, H-0044, H-0045, H-0046. Every `stated`
citation on all 15 was read line by line; all hold.

## Four adjustments made in the main context, not by a batch

A subagent returns findings, not verdicts (`policy:main-context-ownership`).

1. **Seven of A's 59 missed clauses reclassified off-surface.** Four say a type
   "is a dataclass" — implementation form, not observable by a consumer. Two are
   pure visual arrangement (a 45-degree reference line, a horizontal layout).
   One is warning wording. CLAUDE.md section 3 fixes 「形」と「意味」, not
   presentation. A plot clause naming *which data* it reads (H-0009's "QQ uses
   OOS residuals only") is leakage semantics and was kept.
   H-0013's only missed clause was in that set, so it returns to SPECIFIED.

2. **H-0028 reversed against batch B's own verdict.** B marked "tuning_plot
   before tune raises MODEL_NOT_FIT" as stated at BLUEPRINT:147. That line says
   「`tune()` 後。」 and names no error code, while :149 and :150 spell
   `MODEL_NOT_FIT` out for `params_table()` and `fit_result`. The code does
   raise it (`_model_plots.py:242`). **r35 checks that a `stated` clause names a
   line, not that the line states the clause** — this is that gap, found by
   reading all 18 survivor citations by hand. It is the one check in this scheme
   that cannot be mechanized.

3. **H-0037 moved NO_SURFACE → SPECIFIED.** No obligation either way, but its
   own text claimed to change no specification while it decided the public
   `balanced` default; the record should not carry that claim.

4. **Nine new edit groups.** H-0002's clauses are two table rewrites, not 23
   edits; H-0020's five defaults are one table; likewise H-0035, H-0014, H-0009,
   H-0007, H-0004.

## The asymmetry between A and B, resolved

A found 59 missed to B's 15 on equal-sized batches. That is section coverage,
not method drift:

- B's entries are public-API additions, and BLUEPRINT:134-151 is an API table
  that **does** state preconditions. B enumerated 6 error-code clauses and found
  all 6 stated with lines (only :147 was wrong, above).
- A's entries land on BLUEPRINT section 7, a **typeless bullet list**. H-0002 is
  the proposal that fixed every field of all three Result types; a large
  obligation there is the expected result, not inflation.

## What C found that reading had not

H-0025 and H-0067 both claimed, in their own text, to change no specification.
Both were exempted on that claim and nobody had checked it:

- **H-0025** — ratio constraints are stated at BLUEPRINT:246-247 and :332-333,
  but nothing connects an out-of-range value to `CONFIG_INVALID`, which
  HISTORY:1692-1694 requires and `config/schema.py:258-265` implements.
- **H-0067** — the mixed / all-NaN calibration fallback
  (`calibration/cross_fit.py:127-140`) changes what `calibrated_oof` **means**,
  and BLUEPRINT states none of it. A leakage-surface omission.

H-0044's supersession is real: BLUEPRINT:662 states H-0058's outer-split reuse
explicitly and carries no live version of the earlier rule.

## Findings outside the enumeration

- **`yourlib_version`** at BLUEPRINT:475 **and** :1276 — a template placeholder
  never renamed. The field is `lizyml_version` (`core/types/artifacts.py:73`),
  and HISTORY:377 already says so. `deps_version` at :475 is likewise
  `deps_versions` in code. DC3.
- **BLUEPRINT's RunMeta list omits `run_id` and `timestamp`**, both decided at
  HISTORY:313-314 — inside H-0002's block, so A's attribution is right.
- **BLUEPRINT:602 contradicts :647** on whether `purge_gap`/`embargo` propagates
  into an auto-resolved `TimeHoldoutInnerValid`. Both are current normative
  wording. Carrier is H-0085, outside this population.
- **H-0024's two contracts were never implemented** — see
  [r39_h0024_repro.md](r39_h0024_repro.md). Both reproduce; neither is a
  documentation gap.
