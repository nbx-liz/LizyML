Review-kind: review
Review-round: 1

# Review: the implementation of H-0148 (#409) — feature names LightGBM refuses, and duplicate column names, refused before fit and tune train

Review the implementation at this head against Proposal H-0148 (HISTORY.md `## H-0148`, Accepted after `h0148-proposal-review` round 3 APPROVE at `4b17976`). The implementation is `git diff 4b17976 HEAD`. Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the implementation does what the Proposal says and nothing it forbids; `CHANGES_REQUESTED` otherwise.

## Checks (run what your environment allows)

1. **LightGBM's rule** (`lizyml/estimators/lgbm/feature_names.py` `lightgbm_refused_feature_name`): `str()`, the NUL cut, the seven JSON characters checked before the space rewrite, the repeat against earlier rewritten names, in column order; the first refused name is returned with the fields of `RefusedFeatureName`. Is there any name list on which it disagrees with LightGBM 4.6.0's `Dataset::set_feature_names` as reached from `lgb.Dataset` with a pandas frame?
2. **Capability** (`lizyml/estimators/provider.py` `FeatureNameRuleProvider`, `RefusedFeatureName`; `LGBMProvider.refused_feature_names`): `runtime_checkable`, separate from `EstimatorProvider`, and core imports no concrete estimator.
3. **Where it runs**: `Model.fit` (`lizyml/core/model.py`) calls `check_feature_names_for_training` right after `get_provider` and before any training; `Model.tune` (`lizyml/core/_model_tuning.py`) right after Phase B's `get_provider`, before Phase C reads the study and before any trial or study write. Is there any path through `fit` or `tune` that reaches LightGBM with an unchecked name (a different `X`, a later column change)?
4. **Duplicate columns** (`lizyml/data/dataframe_builder.py` `check_duplicate_columns`, `lizyml/core/_feature_name_check.py` `config_roles`, called from `_prepare_training_data` right after the frame is loaded): role columns refused regardless of `features.exclude` and the split, features refused, non-role excluded duplicates admitted; context `column`, `positions`, `role`, `reason`. Are the five role fields exactly the Config's column-name fields? Does label comparison behave for non-string labels (`0`, `1.5`, `True`, NaN)?
5. **Contexts and messages**: exactly the keys the Proposal names, `first_column`/`first_position` only for repeats.
6. **Tests** (`tests/test_estimators/test_h0148_feature_names.py`): does each acceptance criterion have a test that fails without the change, and does acceptance 4 compare against the real LightGBM `lgb.Dataset(...).construct()` for every listed case?
7. **Docs**: BLUEPRINT §8.2 and §14.4, CHANGELOG `[Unreleased]`, `docs/proposal_dispositions.toml`, and HISTORY's 実装の記録 against the code.

## Environment

The workspace is a digest-fixed, read-only copy of this head. Run single test files with the workspace's Python if it has the dependencies; never the full suite. The full suite with slow tests ran locally on this head (24043 passed, 264 skipped). Say what could not run.

## Stop criterion

The Proposal's own レビューの停止基準 applies: findings must be contract defects (a name LightGBM refuses reaching training, a name it accepts refused, fit and tune disagreeing, tune running a trial, a missing `reason` or context) or false statements in the diff. Non-colliding space names, H-0147's save check, predict frames and estimators other than LightGBM are not findings.
