# Tuning and resume: design intent versus implementation (2026-10-11)

Prepared for the decision on H-0121 (#299) and a possible new Proposal on resume compatibility. Head: `lizyml/` identical to `develop` `6273d39`. Measurements: scratch probes (regression, n=150, 3-fold, `space_mode: replace`) run 2026-10-11; split and `model.params` changes were not executed.

## Conclusion

- No Proposal states that `resume` continues **the same objective**. H-0068 acceptance criterion 2 ("best_score does not get worse", HISTORY.md:5333) silently relies on it.
- The implementation checks only the optimisation **direction** against an existing study (`lizyml/tuning/tuner.py`, `study.directions`; H-0099 decision 2).
- Executed: a resume after changing the data or the metric (same direction) continues silently, mixing scores of different objectives, and selects the best across all trials. Reattaching a persistent study with a different config let a best from a different space reach `fit()`.

## 1. Design intent (records)

| Feature | Record | Intent | Stated premise about the objective |
|---|---|---|---|
| `tune` | BLUEPRINT §6.1:438, §6.2 | Each trial routes dims with `split_by_category` (model / smart / training), trains through `_build_train_components` like `fit`, returns the OOF score; the best is applied to the next `fit()` (config < tune best < `fit(params=)`). | — |
| Resume + boundary expansion | H-0068 (HISTORY.md:5071) | Keep the previous study and add trials so TPE uses past trials; enqueue the previous best; widen a dim stuck at an edge on one side (default space: expand; user space: not). Resume before any tune: `TUNING_FAILED`. | Only "selection bias from repeated evaluation on the same folds" (:5092). Nothing about keeping data/config unchanged. |
| Persistent storage | H-0072 (:5663) | Crash recovery for 24h+ LizyStudio jobs; reattach with `load_if_exists=True`, also when `resume=False` (:5732); journal keeps no round metadata (:5740); INV-6 progress/round independent of storage (:5772). | None. |
| Bounds | H-0078 | Clamp expanded ranges to provider bounds. | — |
| Space merge/replace | H-0102 (:9967) | "Resume reuses stored dimensions, bounds, expansion and fixed-default policy; later Config mutations must not reintroduce defaults" (:9990). "Start a fresh study when changing space semantics; old trials are not observations of the new space" (:10004). | Closest statement, but only about the space, and as user guidance, not enforcement. |
| Admission | H-0093, H-0099 (:9880-9900) | Model-category names checked before the study; direction derived from the first metric (d1) and checked against an existing study (d2); the resolved space checked before the study, including resume (d3); training dims limited to two names (d4). "Existing studies optimized in the wrong direction require a new study name" (:9900). | Direction only. |
| Persistence of results | H-0086 (:6934), H-0094 d13-14, H-0109, H-0120 amendment 1 | `best_*`, score, metric, direction saved in metadata and restored as a minimal `TuningResult`; study/trial restore left as a follow-up (never filed; `exporter.py:66-72`). Reporting surfaces answer for the fitted model; tune does not replace adapters; failed calls change no state. | — |

## 2. Implementation

- **Space** (`lizyml/core/_model_tuning.py:395-445`): on resume with a stored `_space`, the stored space and fixed params are used as they are; the config's space is not read. A fresh tune builds the space from the config (merge/replace) and attaches bounds.
- **Base params**: `_merge_params(include_tuning_result=resume)` (:241) uses the config's current model/smart values, overlaid with the previous best only on resume. A resume therefore combines **the stored space** with **the current config outside the space**.
- **Admission** (:208-282): runs each time against the current config.
- **Boundary expansion**: only on resume, judged on the previous best across all trials (:462-483).
- **Round execution** (:578-629): on resume, `prior_trials`, `best_score_before` and the enqueue are set and `study=self._study` is passed, so `storage` / `study_name` are not used.
- **Tuner** (`tuner.py`): checks direction only; best = `study.best_params` / `best_value` across all trials. `split_by_category` (`search_space.py:218`) routes a name absent from the current space to `model`.
- **After `Model.load`**: `_study` is not restored; `tune(resume=True)` raises `TUNING_FAILED` (executed).

## 3. Gaps

| Item | Design | Implementation (executed unless noted) | Effect | Issue |
|---|---|---|---|---|
| Resume after changing the data | Undefined (H-0068 c2 implicitly assumes comparability) | P1: round-2 scores ~553, the round-1 best 5.38 stays best; `fit` applies those params to the new data | Silent wrong | none |
| Resume after changing the metric (same direction) | Direction only (H-0099 d2) | P2: rmse and mae trials mixed; reported as `metric_name="mae"` (contradicts BLUEPRINT §7.2 "optimised metric") | Silent wrong | none |
| Resume after changing split / `model.params` / smart config | Undefined | Same mechanism (not executed); `auto_num_leaves` confirmed in P7 | Silent wrong (inferred) | none |
| Reattaching a persistent study (`resume=False`, other config) | Crash recovery; direction only | P4b: an old mae trial chosen as the rmse best; `learning_rate` / `n_estimators`, absent from the current space, entered `best_model_params` and `fit()` applied them | Silent wrong | none |
| Rounds/progress on that path | H-0072 §4: after load `prior_trials` is correct (:5739); INV-6 | `prior_trials=0`; `current_trial` 4,5,6 exceeds `total_trials=3`; 6 trials recorded as `rounds=[(1,3)]` | Display | none |
| Config space edited before resume | Ignored (H-0102, by design) | P3: ignored, with no notice to the user (`test_space_merge.py:183-212` pins it) | No notice | — |
| `resume=True` with a different `storage` | Continue the existing `_study` (§11.5.1) | P5: continues in memory; nothing written to the new URL | Silently ignored (by design) | none |
| Resume after `Model.load` | Out of scope (H-0086) | `TUNING_FAILED` | Loud | #315 (display side) |
| `auto_num_leaves` true → false, then resume | Undefined | P7: `num_leaves_ratio` keeps being sampled, affects nothing, and is recorded | Silent wrong | #299 |
| BLUEPRINT §11.3 fixed-params table lists `auto_num_leaves: True` | Stated as fixed | Not in `default_fixed_params` (`defaults.py:151`); follows the config | Documentation (one root of #299) | #299 |

## 4. Undefined by the design

- What a resumed or reattached study must be bound to (data fingerprint, metric, split, model/smart config, library versions). H-0079 only acknowledges that a version change can change the metric on re-tune (HISTORY.md:6578).
- The meaning of a resume after changing config outside the space (today: current config + stored space).
- Handling of parameters in a reattached study that are absent from the current space.
- Whether the best is chosen across all trials or within a round, and any guarantee that scores across rounds are comparable.
- Persisting round and space history for reattachment (H-0072 option E, never filed) and restoring the study after load (H-0086 follow-up, never filed).
- The enqueue when a dimension is removed from the space (the previous best still carries its value).

## 5. Relation to H-0121 (facts)

#299 has two symptoms, which belong to different places:

- **(i) Undeclared smart names**: fresh-tune admission only, unrelated to resume. `validate_tuning_dimensions` (`_tuning_validation.py:52-96`) does not ask the provider for smart names; pinned by strict xfails (`tests/test_tuning/test_search_space_name_validation.py:70, 268`).
- **(ii) A smart dimension whose consumer is inactive**: occurs in a fresh tune (`auto_num_leaves: false` with the default space) and in a resume after the config changed (P7). On resume, admission runs with the current config's smart values against the stored space, so the resume case is "the same admission check applied to an old space", not the objective-identity check itself.
- **Overlap**: `auto_num_leaves` is also one of the axes that change the objective. If a Proposal "refuse a resume whose objective changed" covers smart/model config changes, the resume case of H-0121 policy 4 (acceptance criterion 6, "change the config to `auto_num_leaves: false` and resume") is refused before it, and that branch of policy 4 receives no input. Both Proposals currently touch this case.

## 6. Options for the decision

The probes are in `probes/` (`probe_resume.py`: P1-P3, P5-P7; `probe_p4b.py`: P4b).

**Q1. Scope of H-0121 (#299).**

- **A1. Fresh-tune admission only.** Undeclared smart names refused; a user-written smart dimension with an inactive consumer refused; a default-space one dropped. No resume rule in H-0121. The resume case (P7) is handled by Q2's Proposal (refused as an objective change). Also fixes the BLUEPRINT §11.3 table.
- **A2. A1 plus a narrow resume refusal.** On resume, a stored dimension that is now inactive is refused (`CONFIG_INVALID`, start a new study) regardless of origin, before boundary analysis, enqueue and optimisation. Closes P7 even if Q2 is deferred; overlaps with Q2 once it lands.

**Q2. Resume and reattach compatibility (new issue and Proposal).**

- **B1. Objective identity, refuse on mismatch.** Store a fingerprint of the objective-relevant inputs with the study (in memory, and in the study's user attributes for persistent storage): data fingerprint, target/features, task, config outside the search space (model/smart params, split and seed, early stopping and inner validation, unseen policy), the full first-metric specification. On resume or reattach, refuse with `CONFIG_INVALID` on mismatch and point to a new study (and a new study name for storage). Covers P1, P2, P4b and P7.
- **B2. B1, plus refuse a resume when the configured space differs from the stored one** (today silently ignored, P3), instead of only documenting it.
- **B3. File the issue only, after v0.19.0.**

Smaller item found on the way (display only): `prior_trials` / round counting on a reattached study (P4b); it does not affect results.
