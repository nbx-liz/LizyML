Review-kind: review
Review-round: 1

# Row-by-row check: #270 UNIT / NEGATIVE verdicts at develop 1d41b66 (follow-up part 1)

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

tests/test_calibration/test_calibration_failed_optimisation.py::test_refit_failing_before_minimize_does_not_keep_the_earlier_fit | train | NEGATIVE | Refit failing before minimize leaves calibrator unfitted | refit raises; predict/export_params then raise CALIBRATION_NOT_FITTED | tests/test_calibration/test_calibration_failed_optimisation.py:81
tests/test_calibration/test_calibration_failed_optimisation.py::test_default_settings_still_fit | train | UNIT | Platt/Beta default settings still fit | calibrator.fit().predict() probabilities in (0,1) | tests/test_calibration/test_calibration_failed_optimisation.py:112
tests/test_calibration/test_calibration_failed_optimisation.py::test_cross_fit_names_c_final | train | NEGATIVE | cross_fit_calibrate failure in C_final names stage c_final | raises CALIBRATION_FAILED, context stage=c_final, no fold key | tests/test_calibration/test_calibration_failed_optimisation.py:145
tests/test_calibration/test_calibration_param_contract.py::test_beta_default_fit_emits_no_warning | train | UNIT | BetaCalibrator default fit emits no warning | BetaCalibrator().fit under warnings-as-errors does not raise | tests/test_calibration/test_calibration_param_contract.py:218
tests/test_calibration/test_calibrator_not_fitted.py::test_predict_before_fit_raises_lizyml_error | all | NEGATIVE | Each calibrator predict before fit raises LizyMLError | raises CALIBRATION_NOT_FITTED with context calibrator tag | tests/test_calibration/test_calibrator_not_fitted.py:30
tests/test_calibration/test_h0058_outer_reuse.py::TestNSplitsDeprecation::test_explicit_n_splits_warns | split | UNIT | Explicit calibration n_splits emits deprecation warning | pytest.warns UserWarning on CalibrationConfig construction | tests/test_calibration/test_h0058_outer_reuse.py:35
tests/test_calibration/test_h0058_outer_reuse.py::TestNSplitsDeprecation::test_default_n_splits_no_warning | split | UNIT | Omitting calibration n_splits emits no warning | CalibrationConfig(method=platt) under warnings-as-errors | tests/test_calibration/test_h0058_outer_reuse.py:40
tests/test_calibration/test_h0058_outer_reuse.py::TestNSplitsDeprecation::test_model_dump_roundtrip_no_warning | split | UNIT | CalibrationConfig model_dump round-trip emits no warning | n_splits in dump; re-parse under warnings-as-errors | tests/test_calibration/test_h0058_outer_reuse.py:46
tests/test_calibration/test_isotonic_calibration.py::TestIsotonicCalibrator::test_a_forced_parameter_is_forced_under_its_canonical_name | all | UNIT | Isotonic calibrator forces monotone_constraints/verbosity under canonical name | IsotonicCalibrator._lgbm_params[forced] != 99; name canonical | tests/test_calibration/test_isotonic_calibration.py:93
tests/test_calibration/test_platt_mle.py::test_target_smoothing_can_be_turned_off_and_changes_the_fit | train | UNIT | Platt target_smoothing=False changes fitted coefficients | PlattCalibrator.export_params a,b differ with/without smoothing | tests/test_calibration/test_platt_mle.py:84
tests/test_calibration/test_platt_mle.py::test_default_fit_emits_no_warning | train | UNIT | PlattCalibrator default fit emits no warning | PlattCalibrator().fit under warnings-as-errors | tests/test_calibration/test_platt_mle.py:94
tests/test_codegen/test_templates.py::TestRenderTrainPy::test_contains_oof_generation | split | UNIT | Rendered train.py source contains OOF generation | substring _generate_oof and StratifiedKFold in render_train_py() | tests/test_codegen/test_templates.py:77
tests/test_config/test_schema.py::TestEnvOverride::test_override_training_seed | train | UNIT | Env var LIZYML__training__seed overrides config seed | load_config(...).training.seed == 999 | tests/test_config/test_schema.py:268
tests/test_config/test_stratified_default.py::TestStratifiedDefault::test_binary_no_split_defaults_to_stratified | split | UNIT | Binary config without split defaults to stratified_kfold | load_config(...).split.method == stratified_kfold | tests/test_config/test_stratified_default.py:16
tests/test_config/test_stratified_default.py::TestStratifiedDefault::test_multiclass_no_split_defaults_to_stratified | split | UNIT | Multiclass config without split defaults to stratified_kfold | load_config(...).split.method == stratified_kfold | tests/test_config/test_stratified_default.py:26
tests/test_config/test_stratified_default.py::TestStratifiedDefault::test_regression_no_split_defaults_to_kfold | split | UNIT | Regression config without split defaults to kfold | load_config(...).split.method == kfold | tests/test_config/test_stratified_default.py:36
tests/test_core/test_calibration_params_reach.py::test_fit_refuses_before_training | train | NEGATIVE | Invalid calibration.params refused by fit before any training | raises CONFIG_INVALID naming calibration.params; no lgb.train params recorded | tests/test_core/test_calibration_params_reach.py:164
tests/test_core/test_exceptions.py::TestSeed::test_derive_seed_differs_across_folds | split | UNIT | derive_seed gives distinct seeds across fold indices | len(set(derive_seed(42, i) for i in 0..4)) == 5 | tests/test_core/test_exceptions.py:118
tests/test_core/test_fit_params_override.py::test_two_spellings_in_the_config_are_refused_before_training | train | NEGATIVE | Two config spellings of one param refused before training | raises CONFIG_INVALID naming model.params; no train calls recorded | tests/test_core/test_fit_params_override.py:233
tests/test_core/test_fit_params_override.py::test_two_search_dimensions_naming_one_parameter_are_refused | train | NEGATIVE | Two search dimensions naming one parameter refused before study | tune raises CONFIG_INVALID naming tuning.optuna.space; no train calls | tests/test_core/test_fit_params_override.py:558
tests/test_core/test_fit_params_override.py::test_a_search_dimension_a_training_setting_controls_is_refused | train | NEGATIVE | Search dimension controlled by a training setting refused before study | every spelling: tune raises CONFIG_INVALID; no train calls | tests/test_core/test_fit_params_override.py:777
tests/test_core/test_fit_params_override.py::test_every_calibration_alias_is_canonical_before_the_defaults_merge | all | UNIT | Every calibration alias canonicalised before calibrator defaults merge | canonicalise_calibration_params output over all default spellings | tests/test_core/test_fit_params_override.py:956
tests/test_core/test_fit_params_override.py::test_two_spellings_of_one_value_in_the_config_are_refused | train | NEGATIVE | Config duplicate spelling with equal values refused before training | fit raises CONFIG_INVALID; no train calls recorded | tests/test_core/test_fit_params_override.py:1025
tests/test_core/test_fit_params_override.py::test_an_unknown_name_in_fit_params_is_refused_before_training | train | NEGATIVE | Unknown name in fit(params=) refused before training | fit raises CONFIG_INVALID; no train calls recorded | tests/test_core/test_fit_params_override.py:1070
tests/test_core/test_fit_params_override.py::test_the_refusal_names_the_fit_params_surface | train | NEGATIVE | Unknown fit(params=) name refusal names the fit(params=) surface | raises; message and context surface == fit(params=) | tests/test_core/test_fit_params_override.py:1090
tests/test_core/test_fit_params_override.py::test_a_smart_name_in_fit_params_is_reported_as_smart | train | NEGATIVE | Smart name in fit(params=) refused and reported as smart | raises; message contains smart parameter and fit(params=) | tests/test_core/test_fit_params_override.py:1112
tests/test_core/test_fit_params_override.py::test_each_input_is_named_by_its_own_surface | train | NEGATIVE | Unknown-name refusal attributes each name to its own surface | raises; context unknown maps three names to three surfaces | tests/test_core/test_fit_params_override.py:1125
tests/test_core/test_fit_params_override.py::test_the_alias_review_round_2_measured_is_refused | train | NEGATIVE | max_leaves alias in fit(params=) refused before training | raises CONFIG_INVALID; no train calls; context managed names num_leaves | tests/test_core/test_fit_params_override.py:1347
tests/test_core/test_fit_params_override.py::test_an_objective_alias_gets_the_same_task_check | train | NEGATIVE | Cross-task objective via alias application is refused | raises CONFIG_INVALID mentioning regression; no train calls | tests/test_core/test_fit_params_override.py:1956
tests/test_core/test_fit_params_override.py::test_two_spellings_with_different_values_are_refused | train | NEGATIVE | Two spellings with different values refused, naming both | raises CONFIG_INVALID naming objective and application; no train calls | tests/test_core/test_fit_params_override.py:2010
tests/test_core/test_fit_params_override.py::test_the_adapter_refuses_a_duplicate_spelling_carrying_equal_values | all | NEGATIVE | Adapter _pop_by_identity refuses duplicate spelling with equal values | _pop_by_identity raises CONFIG_INVALID naming both spellings | tests/test_core/test_fit_params_override.py:3456
tests/test_core/test_fit_params_override.py::test_a_duplicate_spelling_carrying_equal_values_is_refused_before_training | train | NEGATIVE | Duplicate spelling with equal values refused before training end to end | fit raises CONFIG_INVALID; no train calls recorded | tests/test_core/test_fit_params_override.py:3472


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
