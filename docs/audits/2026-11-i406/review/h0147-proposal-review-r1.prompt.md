Review-kind: review
Review-round: 1

# Review: Proposal H-0147 (HISTORY.md `## H-0147`) — refuse an empty feature name before saving, warn at fit, and add `reason` to the two category-save refusals (#406)

Review the Proposal text at this head, not an implementation (there is none yet). Answer as findings with `file:line` locations and executed or quoted evidence. `APPROVE` when the Proposal is correct, complete and implementable as written; `CHANGES_REQUESTED` otherwise. Also check the facts stated in issue #406 (restated in the Proposal's 目的 and 実測) against the code at this head and the evidence under `docs/audits/2026-11-i406/`.

## Decided (do not reopen)

The owner's rulings in 管理者の判断 (2026-11-16): the empty name is refused by the check the three save surfaces share, with `SERIALIZATION_FAILED`, and `Model.fit` only warns; no fit refusal and no renaming of features for LightGBM; `reason` is a snake_case token and the prose stays in `user_message`; names LightGBM rejects at fit are #409; v0.21.2 follows.

## Checks (run what your environment allows)

1. **Cause and population.** Is the 実測 right that only an empty name reaching the booster saves and then cannot load? Read `lizyml/estimators/lgbm/adapter.py` (`lgb.train(..., keep_training_booster=True)`), how the boosters are pickled, and `docs/audits/2026-11-i406/{roundtrip-names,empty-name-cases,booster-names}.out`. Is there a name the booster holds that the model text loses and the probes miss (for example after LightGBM's space rewrite, or a name only some fold boosters see)? Is the Proposal's statement of the rule in 対応方針 1 true of LightGBM's model-text writer and reader?
2. **Where the check sits.** 対応方針 2 puts it first inside `check_category_values` (`lizyml/persistence/exporter.py`), asking each saved state's adapter (`_saved_pipeline_states`). Do `Model.export` (`lizyml/core/_model_persistence.py`), public `lizyml.persistence.export` and `export_code` all reach it before anything is written, including before `_resolve_export_path` can create a run directory? Is the claim that `export_code`'s existing H-0141 refusal stays first and unchanged true (`project_column_names` order)? Is asking the adapter, rather than reading `refit_result.feature_names`, consistent with H-0129 and §2.1 layering?
3. **Fit warning.** 対応方針 3: does placing it beside `warn_saved_state_changes` in `lizyml/core/model.py` judge with the same answer export uses, and is "one warning per name" well defined across refit and folds? Could the warning fire for a model that export accepts, or miss one export refuses?
4. **`reason` values.** 対応方針 4: are `_refuse_value` and the adapter branch the only two raises that lack `reason` among the category-save refusals of `check_category_values`? Do the tokens conflict with existing `reason` values in `lizyml/`? Which existing tests assert these contexts as exact dicts (`tests/test_persistence/test_unwritable_categories.py` and others) and does the Proposal say they change?
5. **Compatibility.** Are the four 互換性 statements true, in particular that already-saved models stay unloadable and that the new `BaseEstimatorAdapter` method with a `[]` default needs no change from other adapters?
6. **Change Gate fields.** The `Firing rate:` line follows the form and its count (5/17) matches the three `.out` files; `Gate-category: none` is admissible.
7. **Acceptance criteria.** Is each criterion testable as written, and does criterion 1 fail on this head? Does criterion 5 actually exercise the declared rule against the installed LightGBM (DC7), and criterion 7 reach both raises by execution rather than by source reading?

## Environment

The workspace is a digest-fixed, read-only copy of this head. You may run single test files or the probes with the workspace's Python if it has the dependencies; never the full suite. Say what could not run.

## Stop criterion

The Proposal's own レビューの停止基準 applies: findings must be defects inside the contract (an unloadable model still passing export, a loadable model refused, the three surfaces disagreeing, the fit warning and the export refusal judged by different rules, a missing `reason`) or false factual claims. #409's names, rescuing artifacts saved by 0.21.1, and rules for estimators other than LightGBM are not findings.
