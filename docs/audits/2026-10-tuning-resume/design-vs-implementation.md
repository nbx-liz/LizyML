# Tuning and resume: design intent versus implementation (2026-10-11)

Prepared for the decision on H-0121 (#299) and a possible new Proposal on resume compatibility. Head: `lizyml/` identical to `develop` `6273d39`. Measurements: the probes in `probes/` (regression, n=150, 3-fold, `space_mode: replace`), run 2026-10-11 against a checkout whose `lizyml/` equals `6273d39`; they import the repository they are checked out in. Split and `model.params` changes were not executed. Revised 2026-10-11 after a Codex critique (citations, a storage conflict, the options).

## Conclusion

- No Proposal states that `resume` continues **the same objective**. H-0068 acceptance criterion 2 ("best_score does not get worse", HISTORY.md:5326) silently relies on it.
- The implementation checks only the optimisation **direction** against an existing study (`lizyml/tuning/tuner.py`, `study.directions`; H-0099 decision 2).
- Executed: a resume after changing the data or the metric (same direction) continues silently, mixing scores of different objectives, and selects the best across all trials. Reattaching a persistent study with a different config let a best from a different space reach `fit()`.

## 1. Design intent (records)

| Feature | Record | Intent | Stated premise about the objective |
|---|---|---|---|
| `tune` | BLUEPRINT §6.1:438, §6.2 | Each trial routes dims with `split_by_category` (model / smart / training), trains through `_build_train_components` like `fit`, returns the OOF score; the best is applied to the next `fit()` (config < tune best < `fit(params=)`). | — |
| Resume + boundary expansion | H-0068 (HISTORY.md:5071) | Keep the previous study and add trials so TPE uses past trials; enqueue the previous best; widen a dim stuck at an edge on one side (default space: expand; user space: not). Resume before any tune: `TUNING_FAILED`. | Only "selection bias from repeated evaluation on the same folds" (:5092). Nothing about keeping data/config unchanged. |
| Persistent storage | H-0072 (:5663) | Crash recovery for 24h+ LizyStudio jobs; reattach with `load_if_exists=True`, including a second call with `resume=False` and `_study is None` (:5804; BLUEPRINT.md:1101); journal keeps no round metadata (:5740); INV-6 progress/round independent of storage (:5772). | None. |
| Bounds | H-0078 | Clamp expanded ranges to provider bounds. | — |
| Space merge/replace | H-0102 (:9967) | "Resume reuses stored dimensions, bounds, expansion and fixed-default policy; later Config mutations must not reintroduce defaults" (:9963-9964). "Start a fresh study when changing space semantics; old trials are not observations of the new space" (:9981-9982). | Closest statement, but only about the space, and as user guidance, not enforcement. |
| Admission | H-0093, H-0099 (:9880-9900) | Model-category names checked before the study; direction derived from the first metric (d1) and checked against an existing study (d2); the resolved space checked before the study, including resume (d3); training dims limited to two names (d4). "Existing studies optimized in the wrong direction require a new study name" (:9900). | Direction only. |
| Persistence of results | H-0086 (:6934), H-0094 d13-14, H-0109, H-0120 amendment 1 | `best_*`, score, metric, direction saved in metadata and restored as a minimal `TuningResult`; study/trial restore left as a follow-up (`exporter.py:66-72`); the decision whether to persist trials and reporting state is owned by #315 (HISTORY.md:10684-10688, :10733), while reconstructing the Optuna study itself is unfiled. Reporting surfaces answer for the fitted model; tune does not replace adapters; failed calls change no state. | — |

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
| `resume=True` with `storage` and no in-memory study (e.g. after `Model.load`) | H-0072 says it reattaches with `load_if_exists=True` (HISTORY.md:5735-5736); BLUEPRINT.md:1101-1102 later documents `resume=False` for crash reattachment (the records conflict) | `_validate_tune_inputs` refuses every `resume=True` without `_study` before `Tuner` sees `storage` (`_model_tuning.py:375-383`) | Loud (design conflict) | none |
| `auto_num_leaves` true → false, then resume | Undefined | P7: `num_leaves_ratio` keeps being sampled, affects nothing, and is recorded | Silent wrong | #299 |
| BLUEPRINT §11.3 fixed-params table lists `auto_num_leaves: True` | Stated as fixed | Not in `default_fixed_params` (`defaults.py:151`); follows the config | Documentation (one root of #299) | #299 |

## 4. Undefined by the design

- What a resumed or reattached study must be bound to (data fingerprint, metric, split, model/smart config, library versions). H-0079 only acknowledges that a behaviour change can change re-tune results (HISTORY.md:6577).
- The meaning of a resume after changing config outside the space (today: current config + stored space).
- Handling of parameters in a reattached study that are absent from the current space.
- Whether the best is chosen across all trials or within a round, and any guarantee that scores across rounds are comparable.
- Persisting round and space history for reattachment (H-0072 option E, never filed) and restoring the Optuna study after load (unfiled; persisting trials for the reporting surfaces is #315).
- The enqueue when a dimension is removed from the space (the previous best still carries its value).

## 5. Relation to H-0121 (facts)

#299 has two symptoms, which belong to different places:

- **(i) Undeclared smart names**: fresh-tune admission only, unrelated to resume. `validate_tuning_dimensions` (`_tuning_validation.py:52-96`) does not ask the provider for smart names; pinned by strict xfails (`tests/test_tuning/test_search_space_name_validation.py:70, 268`).
- **(ii) A smart dimension whose consumer is inactive**: occurs in a fresh tune (`auto_num_leaves: false` with the default space) and in a resume after the config changed (P7). On resume, admission runs with the current config's smart values against the stored space, so the resume case is "the same admission check applied to an old space", not the objective-identity check itself.
- **Overlap**: `auto_num_leaves` is also one of the axes that change the objective. If a Proposal "refuse a resume whose objective changed" covers smart/model config changes, the resume case of H-0121 policy 4 (acceptance criterion 6, "change the config to `auto_num_leaves: false` and resume") is refused before it, and that branch of policy 4 receives no input. Both Proposals currently touch this case.

## 6. Options for the decision

The probes are in `probes/` (`probe_resume.py`: P1-P3, P5-P7; `probe_p4b.py`: P4b).

**Q1. Scope of H-0121 (#299).**

- **A0. The current H-0121 policy 4 (as amended in review round 1).** Records each dimension's origin; on resume drops an inactive default-origin dimension and refuses an inactive user-origin one. **Unsound by itself**: dropping a dimension from the current list does not remove it from the old Optuna trials; `Tuner` still selects `study.best_params` across all trials (`tuner.py:255-291`), and `split_by_category` routes a best parameter absent from the current dims to `model` (`search_space.py:213-224`).
- **A1. Fresh-tune admission only.** Undeclared smart names refused; a user-written smart dimension with an inactive consumer refused; a default-space one dropped. No resume rule in H-0121. Sound only together with a Q2 policy that refuses an incompatible resume (including studies without a recorded identity) before v0.19.0.
- **A2. A1 plus a narrow resume refusal.** On resume, a stored dimension that is now inactive is refused (`CONFIG_INVALID`, start a new study) regardless of origin, before boundary analysis, enqueue and optimisation. Prevents an inert dimension on an in-process resume, but leaves data/metric/config mixing and `resume=False` reattachment open; a temporary safeguard only if Q2 is deferred.

**Q2. Resume and reattach compatibility (new issue and Proposal).**

- **B1. Objective identity, refuse on mismatch.** Store a canonical identity of the objective with the study (in memory, and in the study's user attributes for persistent storage), check it before any enqueue or study mutation, and refuse with `CONFIG_INVALID` on mismatch (new study, and a new study name for storage). A study without an identity (created before this change) is refused (fail closed). The identity must contain:
  - the data: a hash of the values and order of the effective X, y and group/time/block inputs, or of the realised splits (not `data.fingerprint.compute`, which hashes only row count, column names and dtypes when `file_path` is `None`, `lizyml/data/fingerprint.py:16-42`);
  - task, provider/model identity, the canonical effective fixed/base model, smart and training inputs, the feature-pipeline configuration, split and seed, early stopping and inner validation, the full first-metric specification, and each dimension's name-to-category/consumer routing;
  - an objective-implementation compatibility identifier that changes with relevant LizyML / LightGBM / splitter / metric behaviour (not an indiscriminate list of every library version).
  - It must not contain: search bounds or distributions, `n_trials`, timeout, sampler controls, later evaluation metrics, storage, progress, timestamps or run IDs.
- **B2. B1 plus space identity.** Retain a canonical snapshot of the declared space policy (`space_mode`, the resolved dimension identities, categories and distributions, origins and the fixed-default policy) separately from the operational `_space`, and refuse a resume whose configured space differs from it, while allowing the recorded boundary-expansion lineage (`_space` is replaced by the expanded space, `_model_tuning.py:473-496, 348`, so the comparison must not be against `_space`). Closes P3 as well. `n_trials` and timeout stay changeable.
- **B3. Clean fork on mismatch.** Instead of refusing, create a new study automatically. No score mixing, but study-name proliferation and unclear provenance of results; needs explicit result and lifecycle semantics.
- **B4. File the issue only, after v0.19.0.** Algorithmically unsound and incompatible with the goal of fixing silent-wrong defects before v0.19.0.

Also to decide in the Q2 Proposal: the supported API for reattaching persistent storage (the H-0072 versus BLUEPRINT conflict above).

Smaller item found on the way (display only): `prior_trials` / round counting on a reattached study (P4b); it does not affect results.
