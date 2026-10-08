---

# Remediation — what the independent audit changed

The Phase 1 results above were reviewed by an independent pass
(`reviews/exec-r1.json`, 23 counts recomputed, 1 mismatch). It returned
`REQUEST_CHANGES` with 8 blocking and 3 major findings, and it was right on
every substantive point. This section records what was wrong, what was done, and
what the numbers are now.

**Measured completion, then and now** (`gap.py`, counted from the recorded rows,
not asserted):

| | before | after |
|---|---|---|
| members whose verdict the declared procedure produced | 2,655 / 3,260 = **81.4%** | 3,286 / 3,286 = **100%** |
| members undecided | 608 | 0 |
| steps meeting all three exit conditions | 0 / 8 | 8 / 8 |

The population grew by 26 because R7 closed two boundary surfaces D1a had
declared and never checked.

## The five overclaims, and what replaced them

### O1 — "zero unclassified" while D4 held 60 `UNCLASSIFIED`

Exit condition 1 is "zero unclassified members remain in the declared
population". `UNCLASSIFIED` is a D4 verdict, but it is the verdict for *not
decided*, so the step had not met the condition and the report said it had.

**Fixed by driving the stages instead of hoping for them.** One config per task
invoked `HoldoutInnerValid` and, for binary, the isotonic calibrator — nothing
else. The inner-valid strategy is selected by `split.method` and the calibrator
by `calibration.method`, so a matrix of up to 11 variants per task now reaches
all 13. Abstract bases (`BaseInnerValidStrategy`, `BaseCalibratorAdapter`) carry
an `ABSENT-BY-DESIGN` citation: an abstract class is not a stage that runs, and
recording it `UNCLASSIFIED` reports the AST sweep's own inclusion rule as a gap
in the library. **`UNCLASSIFIED` 60 → 0; `REACHES` 27 → 79.**

### O2 — the D3b control was a static inference

`discovery-plan.md:816` binds the step: *"Execution required: yes. An
`UNGUARDED-DEFECT` verdict means running the conflicting combination and
observing the wrong outcome."* The first pass read `greater_is_better` and
regex-searched for a validator. That predicts what a tuning run would do.

**All 44 cells now run as real `Model.tune()` jobs** — 6 trials each, a search
space wide enough that trials differ in quality — and the verdict is read off
*which trial the search selected*. The control, observed:

```
(binary, auc, minimize)
  trial scores : [0.4704, 0.4802, 0.5170, 0.5339, 0.5372, 0.5372]
  best_score   : 0.4704      <- the lowest AUC of the six
```

22 `AGREES` / 22 `UNGUARDED-DEFECT`, 0 `CANNOT-TELL`. The 10-of-22 figure at the
default direction is unchanged — but it is now evidence rather than a prediction,
and #258's comment was corrected to say so.

### O3 — D3's 461 `OK` rows were produced by counting words

Sets 1–4 decided 302 rows by how often a name appears in source, spec or tests.
That reported `LGBMConfig.feature_weights` — the field #261 shows has no effect
at all — as `OK`. The dangerous direction.

**Each set redone against the column the plan declares:** config by mutating the
field and comparing the trained model; contract fields by AST assignment and AST
attribute access in tests; the public API against the recorded execution trace of
all 2051 test items; knobs against the owning config class's schema plus the
recorded binding observation. The control now fires: `feature_weights` → `DEAD`.

**The new instrument failed its own control three times before it worked**, and
each failure is worth recording because each is a class this plan hunts:

1. A `dict[str, float] | None` field was handed `0.25`; the `ValidationError`
   was read as "the field has a consumer". A rejection proves the *validator*
   reads it, not that it affects output.
2. Fixed, the comparison used `model_to_string()` — whose text carries
   `[feature_pre_filter: 0]`, the flag `feature_weights` flips as a side effect
   while leaving every tree and prediction identical. The digest reported the
   control as a field that *does* change the model, contradicting #261's own
   measurement. Now the comparison is behavioural: predictions, importances,
   metrics, calibrated OOF, tuning outcome.
3. Fixed again, the baseline was 6 trees on 200 rows, where early stopping never
   fires. That produced **16 `DEAD` rows**, including
   `EarlyStoppingConfig.enabled` — a field that plainly changes behaviour. A
   second baseline (400 rounds, 5 folds, early stopping live) **flipped 7 of the
   16 back to `OK`**; the remaining 9 were adjudicated individually.

`config` ends at 16 `OK` / 1 `DEAD` / 1 `WRONG` / 76 `CANNOT-TELL`. The
`CANNOT-TELL` count is high and is the honest number: a field the baseline cannot
reach, or for which no meaningful alternative can be derived, has not been shown
sound.

### O4 — D1a's permanent test covered one of two declared boundaries

The plan defines the boundary as the `lgb.train` params **and** the `lgb.Dataset`
keyword arguments, across runtime **and** generated code. The delivered test
spied `adapter.lgb.train` alone.

**Both surfaces are now checked**: 5 distinct runtime `Dataset` keywords over 54
constructions, and 9 generated-code call sites read from the template strings'
own ASTs — 26 further cells, **0 UNKNOWN**. The permanent test grew two cases and
still fails on exactly the `feature_weights` defect and nothing else.

### O5 — 182 hollow candidates enumerated, not confirmed

Disclosure is not the exit criterion. **All 184 are now settled** by disabling
their producers — one mutation per producer set rather than 182 by hand — with
the kill mechanism itself controlled: a real end-to-end test fails under every
mode, so a zero count is about the candidates and not the instrument.
**179 confirmed hollow, 5 downgraded.** Recorded on #270.

## The three major findings

- **#267 was filed without entering the path it claimed.** 17 constructed column
  types, including an object whose `__eq__` always raises, were pushed through
  both the helper and the public entry point; **none entered the handler**. The
  issue was rewritten and downgraded from `severity: high`, and the D5 row became
  `CANNOT-TELL` — the verdict that exists for exactly this.
- **D2 read values off unrelated text.** `- Legacy calibration path:
  oof_raw_scores=None` produced an `AGREE` for `DataConfig.path`. Rejecting a
  match with another `identifier=` between the anchor and the literal moved
  **15 rows** out of `AGREE`.
- **The D1b gate counted references, not raises.** It now reads `ast.Raise`
  subtrees only. Both readings return the same three members on this head, so the
  change is invisible today and load-bearing later.

## One new defect, found by the repair

**#272 — `config_version` is enforced on one entry path and not the other.**
`BLUEPRINT.md:276` states `1` only; `_check_config_version` lives in
`config/loader.py:193` and runs only via `load_config`; the schema declares a
bare `config_version: int`. `Model.__init__` routes a dict or path through the
loader and a `LizyMLConfig` object straight through. Verified by execution before
filing: `Model(LizyMLConfig.model_validate({...'config_version': 2}))` is
accepted, `Model(raw_dict)` raises `CONFIG_VERSION_UNSUPPORTED`.

## A note on this file's own instrument

`gap.py` first reported 100% with several steps hard-coded complete and the
exit-criterion table written as literal `True`s. That is the stale-acceptance
shape (DC5) the plan exists to refuse — a closing declaration exceeding what the
artifacts show. It was rewritten to read every number from the recorded result
files; the figures above come from that version.

## Process

Every heavy run went through `run-exclusive.sh`, a `flock` wrapper. It exists
because "one CPU-heavy job at a time" was written down after the first incident
(three traced suites at once, load 90 on 32 cores, misdiagnosed as external) and
then broken anyway on the second (load 63.5). Per the mechanize-on-recurrence
rule, the second occurrence is where documentation has to be replaced by
something that executes.
