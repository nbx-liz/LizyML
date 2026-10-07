Review-kind: review
Review-round: 1

# Row-by-row check: #270 UNIT / NEGATIVE verdicts at develop 1d41b66 (follow-up part 2)

You are a read-only critic. The repository at the reviewed head is your working directory. Do not run the full test suite.

## Why this run exists

A previous run was asked to check every row of a larger table, but its transcript shows it opened fewer than half of them. This run covers only the rows it did not open. **Open and read every row's test function below, and its fixtures/helpers where the assertion depends on them.** A row you did not read must not be reported as correct.

## Context

Each test in the table **still passes when the producer set named in its `kill mode` column is patched to raise** (measured). A source-reading triage labelled each row:

- `UNIT` — read honestly, the claim is about the unit the test actually calls; the claim does not need the killed producer.
- `NEGATIVE` — the claim is a refusal / absence / "before training"; passing with the producer disabled is exactly the claim.

Kill modes: `train` = `lightgbm.train`; `api` = `Model`'s public members; `metric` = `Model` public members ∪ `lizyml.metrics.*` / `lizyml.evaluation.*`; `split` = training/calibration stage entry points ∪ concrete splitters' `split`; `all` = union.

## The rule for a wrong verdict

1. `UNIT` is wrong if a reader of the name/docstring would believe a library-level effect is covered (training changes, a parameter reaching the booster, a public `Model` method's behaviour, a real fit's FitResult, a load/export round trip), the body only observes an upstream value, and no other test asserts that same effect for the same task/parameter/branch at the boundary → `HOLLOW` (or `OVERCLAIM` if such a sibling exists — name it and confirm you read it).
2. `NEGATIVE` is wrong if the test can pass vacuously: no refusal asserted, the accepted refusal is broader than the claimed gate (e.g. `pytest.raises((LizyMLError, TypeError))`), or the inputs would also trip a different gate than the one the name/docstring credits (e.g. a hostile value written beside a second spelling, when a duplicate-spelling gate refuses that pair regardless) → `WEAK`.
3. A conditional assertion that passes when the claimed thing is absent, or a name/docstring contradiction → `WEAK`.

## Output — one finding per row, no exceptions

Return the JSON your output contract requires, with **exactly one finding per table row**, in table order:

- correct row: `severity` `"minor"`, `title` = the test node id, `location` = the `path:line` of the assertion you read, `detail` = `"CORRECT: "` + what the assertion checks, in your own words (≤ 25 words).
- wrong row: `severity` `"major"`, `title` = the test node id, `location` = `path:line`, `detail` = the new category (`HOLLOW` / `OVERCLAIM` / `WEAK`), the reason, and for `OVERCLAIM` the sibling `path::name`.

Verdict: `CHANGES_REQUESTED` if any row is wrong, else `APPROVE`. The number of findings must equal the number of table rows.

## Table: `test | kill mode | category | claim | observes | evidence`

tests/test_core/test_param_domain.py::test_a_caller_class_cannot_claim_to_be_a_numpy_type | train | NEGATIVE | Caller class claiming numpy __module__ is refused | not in derived numpy types; normalise_params raises CONFIG_INVALID | tests/test_core/test_param_domain.py:681
tests/test_core/test_param_domain.py::test_an_accepted_type_is_not_by_itself_a_writable_value | train | NEGATIVE | Non-serialisable values of accepted types are refused | normalise_params raises for pure paths and 10**5000; neighbours pass | tests/test_core/test_param_domain.py:805
tests/test_core/test_param_domain.py::test_the_exit_assertion_refuses_a_value_that_skipped_the_surfaces | train | NEGATIVE | Exit assertion refuses an unnormalised value | assert_plain_params raises CONFIG_INVALID with context unnormalised | tests/test_core/test_param_domain.py:1464
tests/test_core/test_param_domain.py::test_the_exit_assertion_is_called_at_every_place_that_trains | train | UNIT | Every lgb.train call site in package calls assert_plain_params | regex source scan of lizyml/ for guarded training sites | tests/test_core/test_param_domain.py:1480
tests/test_core/test_param_domain_cycles.py::test_cyclic_mapping_is_a_controlled_training_refusal | train | NEGATIVE | Cyclic mapping param is a controlled refusal | is_plain false; assert_plain_params raises CONFIG_INVALID | tests/test_core/test_param_domain_cycles.py:10
tests/test_core/test_refusal_matrix.py::test_each_wired_cell_refuses_before_training | train | NEGATIVE | Each wired refusal-matrix cell refuses before training | fit/tune raises CONFIG_INVALID naming layer; no train calls | tests/test_core/test_refusal_matrix.py:321
tests/test_core/test_refusal_matrix.py::test_a_restored_tuning_result_is_refused_too | train | NEGATIVE | Restored tuning best_model_params refused on re-fit | fit raises CONFIG_INVALID; no train calls recorded | tests/test_core/test_refusal_matrix.py:387
tests/test_core/test_seed_propagation.py::TestBuildSplitterSeedResolution::test_default_inherits_training_seed_backward_compat | train | UNIT | Default random_state inherits training.seed in outer splitter | build_splitter(...).split folds equal for seed=42 vs random_state=42 | tests/test_core/test_seed_propagation.py:29
tests/test_core/test_seed_propagation.py::TestBuildSplitterSeedResolution::test_training_seed_propagates_to_splitter | train | UNIT | training.seed propagates to outer splitter | build_splitter(...).split folds differ for seed 42 vs 123 | tests/test_core/test_seed_propagation.py:40
tests/test_core/test_seed_propagation.py::TestBuildSplitterSeedResolution::test_explicit_random_state_overrides_training_seed | train | UNIT | Explicit random_state overrides training.seed | build_splitter(...).split folds equal across seeds with random_state=7 | tests/test_core/test_seed_propagation.py:50
tests/test_core/test_splitter_dispatch.py::TestUnhandledSplitConfigRaises::test_unknown_split_config_raises_lizyml_error | split | NEGATIVE | Unknown split config refused by splitter dispatch | _build_splitter_for_method raises CONFIG_INVALID | tests/test_core/test_splitter_dispatch.py:24
tests/test_core/test_task_type.py::TestTaskTypeReExports::test_oof_assembly_re_export | all | UNIT | oof_assembly re-exports the canonical TaskType | identity OOFTaskType is TaskType | tests/test_core/test_task_type.py:45
tests/test_data/test_data_layer.py::TestValidators::test_clean_group_split_passes | split | UNIT | validate_group_split passes a clean group split | validators.validate_group_split returns [] | tests/test_data/test_data_layer.py:338
tests/test_e2e/test_public_api.py::TestPublicImports::test_fit_result_importable | train | UNIT | FitResult is importable | import succeeds and FitResult is not None | tests/test_e2e/test_public_api.py:21
tests/test_estimators/test_lgbm_metric_bridge.py::TestValidateLgbmMetrics::test_auc_rejected_for_multiclass_native | all | NEGATIVE | auc rejected as multiclass native metric pre-fit | validate_lgbm_metrics raises CONFIG_INVALID | tests/test_estimators/test_lgbm_metric_bridge.py:92
tests/test_estimators/test_lgbm_metric_bridge.py::TestResolveMetrics::test_lizyml_name_translated_before_split | split | UNIT | logloss translated to binary_logloss and classified native | resolve_metrics returns native [binary_logloss], fevals [] | tests/test_estimators/test_lgbm_metric_bridge.py:170
tests/test_estimators/test_lightgbm_parameter_names.py::test_model_params_names_are_lightgbm_names | train | NEGATIVE | Unknown model.params key refused at fit | fit raises CONFIG_INVALID naming the key | tests/test_estimators/test_lightgbm_parameter_names.py:201
tests/test_estimators/test_lightgbm_parameter_names.py::test_every_route_into_the_estimator_is_inventoried | train | UNIT | Every estimator route in package is inventoried | AST scan found routes == ESTIMATOR_ROUTES, non-empty | tests/test_estimators/test_lightgbm_parameter_names.py:702
tests/test_estimators/test_literal_parameter_reads.py::test_every_literal_parameter_read_is_declared | train | UNIT | Every literal parameter read in package is declared | source scan reads minus _ALLOWED is empty | tests/test_estimators/test_literal_parameter_reads.py:160
tests/test_estimators/test_metric_choices_registry_coverage.py::TestMetricRegistryFitTimeCoverage::test_fit_time_registered_metrics_surfaced | all | UNIT | Fit-time reachable registered metrics appear in metric_choices | provider.metric_choices vs whitelists membership | tests/test_estimators/test_metric_choices_registry_coverage.py:58
tests/test_estimators/test_param_behavioral_effect.py::TestSmartParamsBehavior::test_feature_weights_resolves_to_an_ordered_list | train | UNIT | feature_weights resolves to positional feature_contri list | resolve_smart_params output feature_contri ordering | tests/test_estimators/test_param_behavioral_effect.py:411
tests/test_estimators/test_provider_choice_apis.py::TestObjectiveChoicesContent::test_aligned_with_task_compatible_objectives | train | UNIT | objective_choices equals TASK_COMPATIBLE_OBJECTIVES per task | set equality of two registries | tests/test_estimators/test_provider_choice_apis.py:124
tests/test_estimators/test_provider_choice_apis.py::TestMetricChoicesContent::test_no_duplicates_across_branches | split | UNIT | metric_choices native and feval branches disjoint | set intersection of provider.metric_choices branches empty | tests/test_estimators/test_provider_choice_apis.py:201
tests/test_estimators/test_provider_choice_apis.py::TestMetricChoicesContent::test_native_subset_of_validation_whitelist | all | UNIT | metric_choices native names are in validation whitelist | subset check against _LGBM_NATIVE_METRICS | tests/test_estimators/test_provider_choice_apis.py:257
tests/test_notebooks/test_notebook_cells.py::test_time_series_has_split_summary | split | UNIT | Time-series notebook calls split_summary() | substring in notebook code cells | tests/test_notebooks/test_notebook_cells.py:81
tests/test_notebooks/test_notebook_cells.py::test_time_series_has_time_series_method | split | UNIT | Time-series notebook uses time_series split method | substring time_series in notebook code cells | tests/test_notebooks/test_notebook_cells.py:89
tests/test_persistence/test_export_load_errors.py::TestLoadErrors::test_corrupt_fit_result_pkl | train | NEGATIVE | Loading corrupt fit_result.pkl raises deserialization error | loader.load raises DESERIALIZATION_FAILED | tests/test_persistence/test_export_load_errors.py:55
tests/test_plots/test_helpers.py::TestCollectIsData::test_empty_folds_return_empty_arrays | split | UNIT | collect_is_data returns empty arrays for empty folds | pred/actual size 0, dtype matches y | tests/test_plots/test_helpers.py:44
tests/test_calibration/test_h0058_outer_reuse.py::TestBuildCalibrationSplitterDeprecated::test_build_calibration_splitter_warns | api | UNIT | deprecated build_calibration_splitter emits DeprecationWarning | pytest.warns(DeprecationWarning) around build_calibration_splitter(cfg) | tests/test_calibration/test_h0058_outer_reuse.py:179
tests/test_calibration/test_isotonic_calibration.py::TestIsotonicCalibrator::test_monotone | api | UNIT | IsotonicCalibrator predictions are monotone in score | IsotonicCalibrator.fit/predict on sorted scores; np.diff >= -1e-10 | tests/test_calibration/test_isotonic_calibration.py:53
tests/test_calibration/test_isotonic_calibration.py::TestIsotonicCalibrator::test_reproducibility_with_seed | api | UNIT | IsotonicCalibrator with same seed gives identical predictions | two IsotonicCalibrator(seed=123) fits produce equal predictions | tests/test_calibration/test_isotonic_calibration.py:139


---
Runner contract (lizy-runner, slice 1). These lines are appended by the runner.

- You are reviewing commit 1d41b662563a52c4c6aec1398f1a3abe5287f08f. Your working directory is a digest-fixed,
  read-only copy of that commit's tree; it is not the live repository.
- The tree contains no submodules.
- Your final message must be exactly one JSON object and nothing else: no code
  fences, no prose before or after it. Its shape is closed:
  {"verdict": "APPROVE" or "CHANGES_REQUESTED", "reviewed_head": "1d41b662563a52c4c6aec1398f1a3abe5287f08f",
   "findings": [{"severity": "critical" or "major" or "minor", "title": "...",
                 "location": "...", "detail": "..."}]}
- Use "APPROVE" only when every finding is "minor". Unknown fields make the
  result invalid.
