Review-kind: critique
Review-round: 1

# Critique: adding `step` to LizyML search dimensions, with "plan A" for `log` + `step`

You are a read-only analyst. The subject is a **design proposal not yet written into HISTORY**. Judge it against the LizyML code at this head (`/home/rem/repos/LizyML`, code identical to develop `c65ef4d`) and against Optuna 4.7.0 as installed in `.venv` (run code to verify claims; `.venv/bin/python`). Each answer is a finding with a `file:line` location and its evidence. Verdict `CHANGES_REQUESTED` if the design has a defect or a gap that needs a decision before it can be specified; `APPROVE` if it can be written as a Proposal as is. Read the findings, not the verdict, as the analysis.

## Background (facts already established)

- LizyML's `tuning.optuna.space` (and, since H-0122, `tune(space=...)`) has never supported `step`: `lizyml/tuning/search_space.py` `parse_space` silently drops unknown keys, including `step`. Documented format: `type`, `low`, `high`, `log`, `choices`, `category` (`docs/config-reference.md`). A separate Proposal (H-0123, issue #344, in HISTORY.md, not yet accepted) will refuse unknown keys.
- Optuna supports `step` natively for `suggest_int` / `suggest_float` when `log=False` (executed: `[500, 2000]` step 100 samples only multiples; `[0.5, 1.0]` step 0.05 likewise; a range not divisible by `step` is silently truncated to the last grid point with a UserWarning). With `log=True`, Optuna refuses `step` (float: any step; int: step != 1) with `ValueError` at suggest time. Upstream history (optuna PR #1201, #1329, #1387): log+step was tentatively implemented for ints in 2020 following automl ConfigSpace, then disallowed for consistency because float log+step was never implemented and the intended semantics were disputed.
- Downstream LizyStudio puts its UI stepper increment into the space as `step` (e.g. `learning_rate` log range with `step: 0.001`, `n_estimators` with `step: 100`), and its BLUEPRINT intends step-wise search. LizyML currently ignores it. The maintainer judges that users need log + step.

## The proposal under critique

1. **Format.** `float` and `int` dimensions accept an optional `step` (> 0; for `int`, a positive integer). Default: no step (continuous float / every integer). The default spaces get no step.
2. **Without `log`.** Pass `step` to Optuna natively (`suggest_float(step=)`, `suggest_int(step=)`). Refuse, before any study write, a range where `high - low` is not a multiple of `step` (instead of Optuna's silent truncation), naming the nearest valid `high`.
3. **With `log` ("plan A", the Hyperopt `qloguniform` / ConfigSpace `q` semantics).** Optuna samples the dimension log-uniformly **without** step (`FloatDistribution(log=True)` / `IntDistribution(log=True)`); LizyML then rounds the sampled value `v` to the nearest grid point `low + k * step` (k integer, clipped to `[low, high]`; `high - low` must be a multiple of `step` as in 2) and trains with the rounded value.
   - The rounded value is the value of record: it is stored per trial (e.g. a trial user attribute), and every consumer that today reads `trial.params` uses it — best-trial routing to `best_model_params` / `best_smart_params` / `best_training_params` (`lizyml/tuning/rounds.py`, `lizyml/tuning/lineage.py` `split_by_routes`), `fit()` after tune (INV-R2: `config ⊕ fixed ⊕ trained values`), `TuningResult.trials`, `tuning_table` / `tuning_plot`, `boundary_table`, H-0122's `restore_tuning`.
   - Optuna keeps the raw `v` as the observation; TPE learns on the continuous log scale.
   - Several raw samples can round to the same grid point, so the same configuration may be evaluated more than once.
4. **H-0122 interaction.** `step` is part of a dimension's declaration: recorded in the round lineage (`lizyml/tuning/lineage_codec.py` dims), compared by U2 on an open round, and a change of `step` on resume is refused like a change of distribution kind or `log` (spec 5.2-3), because old observations were taken under another quantisation. Not part of the objective identity (the search space is excluded, spec 4.2).
5. **Boundary expansion (H-0068 / H-0078 / H-0111 / #342).** Expanded edges are aligned outward to the step grid anchored at the original `low`; detection uses the value of record (rounded), not the raw sample.

## Questions

1. Is plan A sound as specified? In particular: anchoring the grid at `low` versus at 0 (Hyperopt anchors at multiples of `q`); rounding versus flooring; clipping at the edges and the resulting non-uniform mass at `low`/`high`; int dimensions with `log=True` and `step > 1`; float rounding error (e.g. `0.1 + 3 * 0.1`) in the value of record and in the divisibility check (Optuna uses `decimal` for its own adjustment — check how).
2. Does storing the rounded value as the value of record leave any consumer reading the raw `trial.params` — enumerate them in the code (tuner, rounds, lineage, restore, plots/tables, persistence/export of tuning results, LizyStudio-facing `TuningResult.trials`)? What breaks if one is missed (INV-R2)?
3. Duplicate evaluations: is re-evaluating the same grid point acceptable, or should LizyML reuse the earlier score (and how would that interact with TPE, determinism, H-0122 budgets and `n_trials`)?
4. The divisibility refusal in 2 versus Optuna's truncation: is refusing the right call, and how exactly should "multiple of `step`" be decided for floats?
5. Resume rules in 4 and expansion in 5: anything inconsistent with the H-0122 lineage (`lizyml:round:<k>` records, `same_declared_space`, `refuse_unchangeable`) or with #342's "never narrow or invert"?
6. Anything else that must be decided before this can be written as a Proposal, and any simpler alternative that meets the same need.

Bound the analysis to this design and the code it touches; do not review unrelated code.
