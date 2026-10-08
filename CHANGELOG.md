# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Documentation

- **`docs/examples.md` describes what each notebook does and needs.** The index named methods several notebooks never call (`evaluate()` and `export()` in the regression tutorial, OOF coverage in the binary and time-series tutorials, `oof_per_fold` in the multiclass tutorial, a Beta run in the calibration tutorial) and listed the wrong extras for seven of the eight notebooks: four entries marked `none (base install)` plot (`lizyml[plots]`), three of them also compute SHAP (`lizyml[explain]`), and the tuning, SHAP and calibration entries left out `lizyml[plots]`. The binary tutorial's text said Platt scaling where its config uses isotonic calibration, and the calibration tutorial said Beta needs `lizyml[calibration]`, while scikit-learn already installs scipy. A test now checks the index against each notebook's code.
- **The codegen tutorial runs its equivalence check.** It ran the generated `test_equivalence.py` through `pytest`, which found no tests in it, and printed the failure instead of stopping; its text described a comparison with OOF predictions at `atol=1e-5`. It now writes `Model.predict()` reference predictions and runs `python test_equivalence.py <data> --reference <file>`, which stops the notebook on a mismatch.

## [0.18.0] - 2026-10-08

Phase 3 of the defect-discovery audit (`docs/audits/2026-09-defect-discovery/`): parameters reach LightGBM under one spelling or are refused, refusals name their source, and checks that answered "clean" without looking now fail loudly. Read *Removed* and *Changed* before upgrading; `docs/migration.md` lists what to change.

### Removed -- breaking

- **Breaking: `ErrorCode.DATA_FINGERPRINT_MISMATCH` is removed** ([#263](https://github.com/nbx-liz/LizyML/issues/263)). Nothing ever raised it: no prediction-time comparison against the recorded fingerprint is satisfiable (row counts differ between valid batches, `file_hash` is `None` for every `fit(df)`, `column_hash` depends on column order). Code that references the member now gets `AttributeError`. `DataFingerprint` is still recorded as provenance. Missing columns raise `DATA_SCHEMA_INVALID`, and a column numeric at fit that arrives with a non-numeric dtype raises `INCOMPATIBLE_COLUMNS` (a numeric arrival of another width, such as `int64` for a `float64` column, still predicts).

- **`feature_weights` in `model.params` is no longer accepted; use `feature_contri`** ([#261](https://github.com/nbx-liz/LizyML/issues/261)). This is the LightGBM parameter dict, not the smart Config field `model.feature_weights`, which stays and now takes effect (see *Changed*). `feature_weights` is not a LightGBM parameter — it is neither a canonical name nor an alias in LightGBM's own `LGBM_DumpParamAliases` — so **it has never had any effect** since it shipped. Measured with the same data and seed, a fit with `feature_weights` set produced an importance ordering and gain identical to one without it, while `feature_contri` changed both. A config carrying `feature_weights` in `model.params` now raises `CONFIG_INVALID` instead of training a model in which it did nothing.

  ```yaml
  model:
    params:
      feature_weights: [0.0, 1.0, 1.0]   # before: accepted, inert
      feature_contri:  [0.0, 1.0, 1.0]   # after:  the same intent, and it applies
  ```

  A run that used `feature_weights` and looked fine was training without per-feature weighting. Renaming the key will change the model it produces — that is the point.

### Changed -- action may be required

- **Changed -- results change: `model.feature_weights` now takes effect** (H-0093, [#261](https://github.com/nbx-liz/LizyML/issues/261)). The smart parameter emitted its weights to LightGBM under the key `feature_weights`, which LightGBM does not define and discarded, so it never had an effect (measured: a byte-identical model). It now emits `feature_contri` (with `feature_pre_filter: false`), and the Config field keeps its name. A config that sets `model.feature_weights` trains a different model from before.

- **One parameter written under two spellings is now refused whatever the two values are** (H-0096, [#264](https://github.com/nbx-liz/LizyML/issues/264)). Until now the pair was allowed through when the two values were *equal*, so `model.params: {"learning_rate": 0.5, "eta": 0.5}` trained and `{"learning_rate": 0.5, "eta": 0.9}` was refused. Both are refused now, on every surface -- `model.params`, `fit(params=...)`, `calibration.params`, a restored tuning result, and the estimator adapter. **Write the parameter once, under one spelling.** The refusal names every spelling you used and the surface it came from. A parameter written under a **single** spelling is untouched, alias or not. Measured over everything this repository's suite constructs: **no pre-existing config or test reaches the behaviour that changed** -- all 37 occurrences of the tolerated case are tests written to exercise the tolerance itself. Restored artifacts are the one place a duplicate can arrive without you writing it: `tune()` has refused duplicate search dimensions since earlier in this release, so only a `best_model_params` written before that can trip this, and re-fitting it is now refused rather than resolved by the library (`Model.load()` still reads the artifact). Why the tolerance went rather than being repaired: deciding whether two arbitrary Python values are "the same" has no closed domain, and it was the source of nine consecutive rounds of review findings. LightGBM itself does not make that judgement either -- it warns about the duplicate whether or not the values agree, and resolves it by a fixed precedence.

- **Parameter values are normalised at the surface they are written on, and values outside the accepted set are refused before training** (H-0095, [#264](https://github.com/nbx-liz/LizyML/issues/264)). A parameter value written in `model.params`, `fit(params=...)`, `calibration.params`, or restored from a tuning artifact must now be `None`, `bool`, `int`, `float`, `str`, `pathlib.Path`, a numpy scalar, a list/tuple/1-D numpy array of those (whose elements may be lists of those, one level deep), or a dict with str keys holding those. Anything else raises `CONFIG_INVALID` naming the parameter and the surface. LightGBM itself is more permissive -- it accepts any object `float()` survives -- so this is deliberately narrower: an object that answers `__format__`, `__str__` or `__eq__` differently in different places trained a model nobody chose. (The set was first closed so that two spellings of a value could be compared; H-0096 removed that comparison, and what still needs it closed is the assertion before training and `export_code`, which writes the same values through `json.dump`.) Values are normalised to plain Python without changing the bytes LightGBM receives; the one exception is a reduced-precision numpy float whose printed form no plain number reproduces (`numpy.float16(1e3)` prints as `1e+03`), which is refused rather than rounded. **Measured over every parameter value this repository's suite constructs: 14 of 1518 are refused, and all 14 are objects the suite builds to exercise the refusal.** Ordinary configs are unaffected. A `numpy` scalar or array you pass still works and now arrives at LightGBM as a plain value, which is visible in `params_table()` -- but only for the numpy types a parameter value can be (integers, floats, booleans, strings). `numpy.timedelta64` is refused, because it is a `numpy.integer` whose printed form is `1 nanoseconds` while the integer it converts to is `1`: it used to train on the second. A **subclass** of `numpy.ndarray` is refused for the same reason: iterating one runs the subclass method, and what a caller method returns is not required to be the same twice. Plain arrays and plain numpy scalars are unaffected. A `pathlib.PurePosixPath` or `PureWindowsPath` is refused -- LightGBM writes flavours of `Path`, and a pure path is not one -- and so is an integer too long for Python to convert to a string (above 4300 digits by default), which LightGBM could not have written either. A `pathlib.Path` you pass is accepted and carried on as its text: LightGBM is sent the same characters either way, and a path that stayed a path made `export_code()` raise `TypeError: Object of type PosixPath is not JSON serializable` after a run that had trained successfully. A metric entry written as a mapping (`metric={"precision_at_k": {"k": 15}}`) is unaffected: the adapter consumes it before serialisation, so it is accepted at the surface and refused only if it somehow survives to the trainer.

- **A `set` is now refused as a parameter value** (H-0095, [#264](https://github.com/nbx-liz/LizyML/issues/264)). LightGBM serialises one, but every sequence parameter it takes is positional -- `feature_contri` reads element *i* as feature *i* -- and a set has no order, so which parameter it becomes is decided by hash order. Writing one now raises `CONFIG_INVALID` at the surface instead of training on an order nobody chose. Measured: **0 of 1518** parameter values this repository's suite constructs is a set. Write a list.

- Identify the winning input and parameter spelling when merged LightGBM
  objective or metric values are invalid, before training starts.

- Reject duplicate seed or verbosity spellings in direct `LGBMAdapter` calls.
  Migration: supply each parameter once; canonical-wins behavior is removed.
  Single-spelling aliases remain supported.

- **Tuning optimises in the metric's direction, and search dimensions nothing consumes are refused** (H-0099, [#258](https://github.com/nbx-liz/LizyML/issues/258), [#279](https://github.com/nbx-liz/LizyML/issues/279), [#282](https://github.com/nbx-liz/LizyML/issues/282)). An omitted or `null` `tuning.optuna.params.direction` is now taken from the first evaluation metric's `greater_is_better`; before, 10 of the 22 task x metric pairs inferred the wrong orientation, so the study could report its worst trial as `best_score`. An explicit `minimize` / `maximize` that contradicts the metric, or an existing study created in the other direction, raises `CONFIG_INVALID` before the study starts. A `category: model` dimension that an active smart parameter would overwrite (for example `num_leaves` or an alias of it under `auto_num_leaves`), and a `category: training` dimension other than `early_stopping_rounds` / `validation_ratio`, are refused instead of being sampled and then discarded. **Migration:** omit `direction` (or set it to `null`), disable the smart parameter to tune its native one, remove unsupported training dimensions, and use a new study name for a study that was optimised in the wrong direction.

- **A partial tuning space keeps the provider's other default dimensions** (H-0102). `tuning.optuna.space_mode` is new and defaults to `merge`: each dimension you write replaces the default dimension for the same parameter (matched by parameter identity, so `eta` replaces `learning_rate`), and the rest of the default space is kept. Before, naming one parameter dropped every other default dimension. **Migration:** set `space_mode: replace` to keep the previous behaviour; a partial space may now search, and train, more dimensions. An empty or omitted space still uses the defaults, automatic boundary expansion on a partial space needs `tune(expand_boundary=True)`, and a changed space needs a new study.

- **Changed -- results change for `platt`**: Platt scaling is now fitted as Platt defined it -- slope and intercept by maximum likelihood with smoothed targets and no penalty -- instead of `LogisticRegression(C=1.0)`, which added an L2 penalty and used 0/1 targets. Calibrated outputs of new `platt` fits change; the measured difference is small (none at n=2000, expected calibration error 0.114 to 0.107 at n=100). Artifacts saved earlier load and predict exactly as before.

- **Migration**: a LogisticRegression name such as `C` in `calibration.params` for `platt` is now refused; it never had an effect.

- **Changed -- results change for multiclass `balanced`: the final refit trains with the CV folds' class weights** (H-0103, [#269](https://github.com/nbx-liz/LizyML/issues/269)). With a multiclass task and `balanced` active (including the default `balanced: null`), each CV fold model trained with balanced sample weights and the final refit trained without them, so the model you predicted and exported with differed from the models whose OOF metrics you were shown. The refit now receives the same weights. Predictions and exports of such fits change; CV folds and OOF metrics do not; regression, binary and `balanced: false` are unchanged, and saved artifacts predict as before. `RefitTrainer.fit` takes a new optional `sample_weight` argument. The generated `train.py` still retrains without the weights ([#301](https://github.com/nbx-liz/LizyML/issues/301)).

- **Changed -- results may change: metrics computed by the LightGBM feval receive the model's probabilities once** (H-0105, [#306](https://github.com/nbx-liz/LizyML/issues/306)). The feval re-applied the sigmoid (binary) or softmax (multiclass) to values that were already probabilities, so binary `accuracy` / `f1` / `brier` / `ece` and multiclass `brier` were computed on squashed values, and early stopping on `accuracy` or `f1` selected iteration 1. A config that lists one of these in `model.params.metric` with early stopping may now select a different iteration and train a different model; the `history` values are correct either way. Saved artifacts predict as before, and a newly exported `train.py` uses the corrected feval.

- **A third-party `EstimatorProvider` must implement four new methods** (H-0093, H-0094). `accepted_model_param_names()`, `smart_param_names()`, `smart_managed_param_names()` and `canonical_param_names()` are now part of the provider protocol. `fit()` and `tune()` use them for the parameter-name check, the smart-parameter ownership check and merging by parameter identity, whenever the parameters those checks read are present. The built-in `LGBMProvider` implements them.

- **`predict()` raises `INCOMPATIBLE_COLUMNS` when a column that was numeric (or bool) at fit arrives with a non-numeric dtype** (string, object, category, datetime, pyarrow numeric, ...). Before, these predictions failed with a raw LightGBM or numpy error that named no LizyML condition. `context["columns"]` lists each such column with its fit and predict dtypes. Every arrival this refuses failed before, so no prediction that used to succeed through the built-in pipeline is refused (33 dtypes measured). Columns that were categorical at fit are not checked.

- **The six probability metrics refuse values that are not probabilities with `METRIC_REQUIRES_PROBA`.** The six are `logloss`, `auc`, `auc_pr`, `brier`, `ece` and `precision_at_k`. Refused values are non-numeric, non-finite, outside [0, 1], or 1-D for more than two classes. Called directly with scores, `auc` / `auc_pr` / `ece` / `precision_at_k` used to compute silently and `logloss` / `brier` raised a raw scikit-learn error. **Changed -- action may be required:** a `Model.fit` with `objective: cross_entropy_lambda` whose metrics are limited to `auc` / `auc_pr` / `ece` / `precision_at_k` used to succeed and report outputs above 1 as probabilities. When its outputs exceed 1 (300 rounds on #307's data) it now raises `METRIC_REQUIRES_PROBA`; a short fit whose outputs stay below 1 still succeeds ([#307](https://github.com/nbx-liz/LizyML/issues/307) tracks the objective itself).

- **`config_version` is checked on every path into `Model`** ([#272](https://github.com/nbx-liz/LizyML/issues/272)). Before, a `LizyMLConfig` built with `model_validate`, `model_construct`, assignment or `model_copy(update=...)`, a version set through `LIZYML__config_version`, `config_version: false` (stored as `0`), and the string `"2.0"` (coerced to version 2) were all accepted with an unsupported version. A fractional float keeps its old outcome (`2.5` is unsupported, `1.5` is invalid); `inf` now raises `CONFIG_INVALID` instead of a raw `OverflowError`. `SUPPORTED_CONFIG_VERSIONS` is defined once, in `lizyml/config/version.py`, and stays importable from `lizyml.config.loader`.

- **Changed — action may be required: the leakage checks refuse a frame without the column they are named for** ([#311](https://github.com/nbx-liz/LizyML/issues/311)). `validate_no_target_leakage(df, target)` used to return `[]` when `target` was not a column of `df`, and `validate_time_series_order(df, time_col)` did the same for `time_col`. That is the same answer as a frame that was checked and found clean, so a misspelt name got a clean result from a check that compared nothing. Both now raise `LizyMLError(DATA_SCHEMA_INVALID)` before checking anything, whatever `raise_on_violation` is. `context` holds `target` / `time_col`, `missing_columns` and `available_columns`. Calls whose column is present behave as before. In the test suite, 2 of 51 calls passed a missing column, and both were the tests that pinned the old `[]`.

- **Changed — action may be required: a Platt or Beta calibration whose optimiser did not converge now fails** ([#297](https://github.com/nbx-liz/LizyML/issues/297)). Both calibrators used `scipy.optimize.minimize`'s coefficients without checking `success`. An optimisation that stopped at an iteration or evaluation limit, or after a failed line search, therefore shipped its intermediate or initial coefficients as a fitted calibrator, with no warning. Such a fit now raises `LizyMLError(CALIBRATION_FAILED)`, a new `ErrorCode`. `context` holds `calibrator`, `method`, scipy's `message` / `status` and `nit`. `Model.fit` adds `stage` (`"cross_fit"` with the `fold`, or `"c_final"`). The generated `train.py` raises `RuntimeError` in the same case. Measured over the test suite, no default-setting fit reaches this path (0 of 397 calls). A `calibration.params` `options` limit such as `{"maxiter": 1}` does.

- **Changed — action may be required: SHAP importance explains each CV fold model with that fold's own feature pipeline** ([#303](https://github.com/nbx-liz/LizyML/issues/303)). `importance(kind="shap")` used to encode every training row with the **last** fold's pipeline, although each fold model was trained, and predicted its OOF rows, on data its own pipeline encoded. With `features.auto_categorical: false`, a string column learns a different category set per fold, so the importance was computed on encodings the fold models never saw (measured: the `cat` importance moved by about 3-6 %), and under a sliding window `unseen_policy: "error"` made `importance(kind="shap")` raise `DATA_SCHEMA_INVALID` after `fit` had succeeded. `FitResult` now records each fold's pipeline state in a new field, `pipeline_state_per_fold`, and SHAP importance uses it, so it sees the same encoding and substitution as the OOF predictions. `auto_categorical: true` (the default), `features.categorical` and numeric-only data are unaffected. `pipeline_state` keeps its meaning (the last fold's state). `format_version` stays 2. Artifacts written before this load and predict as before, but `importance(kind="shap")` on them raises `MODEL_NOT_FIT` with `context["missing"] == "pipeline_state_per_fold"`: call `fit()` again to get the states (re-exporting does not create them). `compute_shap_importance` takes the states as a new last, optional argument; calls without it behave as before.

- **A non-string `objective` other than `None` is refused with `CONFIG_INVALID`** ([#270](https://github.com/nbx-liz/LizyML/issues/270)). A dict or list `objective`, written in `model.params`, in `fit(params=)` or to the LightGBM adapter directly, raised a raw `TypeError: unhashable type` from the task-compatibility check instead of the `CONFIG_INVALID` that H-0079 promises for an incompatible objective. It now raises `LizyMLError(CONFIG_INVALID)` before training, with the same `context` as an incompatible string and a message naming the value's type, so building the refusal neither formats nor hashes the value (a `str` subclass is judged by its text). The error keeps the value as written in `context`, so `repr(error)` still shows it through its own `__repr__`, and a value whose type lookups raise (a hostile `__class__` or metaclass) is outside this guarantee. Plain string objectives behave as before, and an explicit `None` still means "no override" and trains on the task default.

### Deprecated

- **Deprecated — action recommended: `purged_time_series` has one exclusion knob, `purge_gap`; `embargo` is merged into it** ([#273](https://github.com/nbx-liz/LizyML/issues/273)). The splitter subtracted `purge_gap` and `embargo` at the same position (the end of each training block), so they were one knob, and since training always comes before validation there is no position for an embargo after the validation block, which is what the term means. `embargo` (and the older `embargo_pct` / `gap`) is still accepted until v1.0, with a `DeprecationWarning` even when `0`, and its value is **added** to `purge_gap`: write `{purge_gap: 7}` instead of `{purge_gap: 5, embargo: 2}`. Folds, the inner-validation gap and calibration folds are unchanged for every input that fitted before, except `embargo: true`, which is now refused (below). At most one of `embargo` / `embargo_pct` / `gap` may be given. `model_dump()` no longer has an `embargo` key, and `PurgedTimeSeriesConfig` / `PurgedTimeSeriesSplitter` no longer have an `embargo` attribute; `PurgedTimeSeriesSplitter(embargo=...)` is deprecated the same way. Values now refused: `embargo: true` (was read as `1`); a negative `purge_gap` or `embargo` is a `CONFIG_INVALID` at config validation instead of a `ValueError` at fit; an overflowing `embargo_pct` / `gap` such as `"1e400"` is a `CONFIG_INVALID` instead of a leaked `OverflowError`. Models exported before this load with a `DeprecationWarning` and the same splits. `docs/config-reference.md` no longer draws an embargo after the validation block, and `docs/DEPRECATIONS.md` names H-0038 / H-0040 as the source of the older keys.

### Added

- **`features.unseen_policy` chooses what happens to a category not seen at fit, and substitutions at prediction are reported** (H-0104, [#259](https://github.com/nbx-liz/LizyML/issues/259), [#260](https://github.com/nbx-liz/LizyML/issues/260)). `"mode"` (the default, results unchanged), `"nan"` or `"error"`. At `predict()` an unseen category is now listed in `PredictionResult.warnings` instead of being replaced silently; substitutions on CV validation folds during `fit()` are not reported. Custom `BaseFeaturePipeline` subclasses no longer break at predict time. `EstimatorProvider.build_pipeline_factory` takes a new keyword argument, `unseen_policy`, which a third-party provider must accept. A newly exported `predict.py` reproduces the policy and keeps missing values missing instead of treating them as unseen. With `"error"`, `fit` can raise `DATA_SCHEMA_INVALID` when `features.auto_categorical` is false and a string column not listed in `features.categorical` has a value in a validation fold that its training rows lack (a declared categorical column learns its categories from all rows, so none is unseen within `fit`).

- **`calibration.params` now takes effect for `platt` and `beta`** ([#277](https://github.com/nbx-liz/LizyML/issues/277)). Both calibrators accepted the parameters and discarded them. They now accept `x0`, `method`, `bounds`, `tol` and `options` (and `target_smoothing` for `platt`), and refuse anything else with `CONFIG_INVALID` naming `calibration.params` before training starts, in `fit()` and in `tune()`. Measured: no config in this repository's suite passed params to either calibrator apart from the test that pinned the old accept-and-ignore behaviour.

- **`export_code()` rebuilds the calibrator with the same settings**: `config.json` carries `calibration_params`, and the generated `train.py` applies them for all three calibrators. Previously a retrain ignored them even for `isotonic`. The generated `requirements.txt` lists `scipy` for `platt` as well as `beta`.

- **A loaded model reports the inner-validation ratio its fit used** ([#281](https://github.com/nbx-liz/LizyML/issues/281)). After `tune -> fit -> export -> load`, `params_table()` and `export_code()` reported the configured `validation_ratio` instead of the tuned one the fit trained with, so the generated `train.py` held out a different fraction than the run did. `export()` now records the training overlay the fit applied as `applied_training_params` in `metadata.json` (`{}` when the fit applied none), holding the values as training converted them (the patience an integer, the ratio a float), and `Model.load()` restores it. `format_version` stays 2. Artifacts written before this load as before and fall back to the configured ratio; re-exporting one does not invent a record. A record no fit could have written (not an object, a name other than `early_stopping_rounds` / `validation_ratio`, a patience that is not an integer, a ratio that is not a finite float in (0, 1)) is refused on load with `DESERIALIZATION_FAILED`.

### Fixed

- **`Model.fit(params=...)` now reaches the trained model** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). The argument was accepted and documented as overriding `model.params`, and was never forwarded: the overlay that would apply it had no caller, so an override was discarded in silence and the booster trained on the config value. Two fits differing only in `params` produced byte-identical boosters. They now differ, at the documented priority (config defaults < tune best < `fit()` args). **If you passed `params=` before, your model was trained on the config values, not yours.** The override applies to that call only and does not mutate the config you handed in. A native name an active smart parameter resolves (`num_leaves` under `auto_num_leaves`, `min_data_in_leaf` / `min_data_in_bin` under their ratios, `scale_pos_weight` under `balanced`, `feature_contri` / `feature_pre_filter` under `feature_weights`) is **refused** with `CONFIG_INVALID` naming the smart parameter, rather than accepted and then replaced -- smart resolution runs downstream of the merge and wins. Disable the smart parameter to set the native one directly. The same collision inside a `category: model` tuning space is refused before the study starts since H-0099 ([#279](https://github.com/nbx-liz/LizyML/issues/279)).

- **Parameter names LightGBM does not define are refused instead of silently discarded** (H-0093, [#261](https://github.com/nbx-liz/LizyML/issues/261), [#262](https://github.com/nbx-liz/LizyML/issues/262)). LightGBM drops an unknown key without raising, and LizyML ships `verbose=-1`, so a misspelled or invented name produced a run that looked successful and in which the parameter did nothing. `model.params`, `tuning.optuna.space` (`category: model`), `calibration.params` for LightGBM-backed calibrators, the params `export_code()` generates, and now `fit(params=...)` are all checked against LightGBM's own registry before training starts, and an unknown name raises `CONFIG_INVALID` naming the surface it came from.

- **A parameter written under a LightGBM alias now takes effect** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). LightGBM accepts aliases (`eta` for `learning_rate`, `max_leaves` for `num_leaves`, …) and prefers the canonical spelling when both reach it. LizyML merges its own defaults, the config, the tuning result and the `fit()` override by dictionary key, so a parameter written as an alias survived beside the canonical one and was then ignored: `model.params: {"eta": 0.07}` trained at `0.001`, the default. The layers are now merged by parameter identity and only one spelling reaches LightGBM. **A config that set a parameter under an alias was silently training with the default; it will now train with your value**, which changes the model that config produces.

- **The boosting round count is honoured under any of its names, and one parameter written twice with different values is refused** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). Only the literal `n_estimators` was turned into LightGBM's `num_boost_round`; `num_iterations`, `num_round` and the rest stayed in the parameter dict and reached `lgb.train` beside a different round count. They now all set the rounds. Separately, `fit(params={"objective": "binary", "application": "cross_entropy"})` — one parameter under two spellings — now raises `CONFIG_INVALID` naming both, instead of letting the library decide which applies. **Since H-0096 this does not depend on the two values: a duplicate spelling is refused whether or not they agree.**

- **Tuning now evaluates the parameters it then selects** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). Trial parameters were merged into the config by dictionary key while every other layer was merged by parameter identity. With a config setting `learning_rate` and a search dimension named `eta`, **the trials trained at the config's value while the study recorded the trial's**, and the fit afterwards used the recorded one — so tuning selected a model it had never evaluated, and the reported best score belonged to a different model. Trial parameters are now overlaid by identity like every other layer. **A study whose search space named a parameter under an alias of one your config also set produced a meaningless best value; re-run it.**

- **Two spellings of one parameter are refused instead of both reaching LightGBM, on every layer** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). The rule that one parameter written twice under two spellings with different values raises `CONFIG_INVALID` was applied to `fit(params=...)` only. A config carrying both `learning_rate: 0.001` and `eta: 0.5` — in `model.params` or in `calibration.params` — sent both to `lgb.train` and the library silently kept the canonical one. Both layers are now refused, naming both spellings and the layer they came from. **Since H-0096 the values are not read at all** — see the H-0096 entry under *Changed -- action may be required*. Measured over every config this repository's suite constructs: **0 of 813** pre-existing configs with `model.params` and **0 of 22** with `calibration.params` are affected.

- **`export_code()` keeps a custom metric written under an alias** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). Training reads the metric by parameter identity, so `fit(params={"metrics": "brier"})` evaluated Brier correctly — but the exporter read the literal name `metric`, so the generated project carried `metric="None"` and **no evaluation function at all**. The generated `train_lgbm` did not merely lose the metric; it refused to run (`For early stopping, at least one dataset and eval metric is required`). The exporter now resolves the metric the way training does. **If you exported a project after setting `metrics` or `metric_types`, regenerate it.**

- **A parameter `training.*` already controls is refused instead of silently disagreeing** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). `training.early_stopping.rounds` and `training.seed` are LizyML's names for LightGBM's `early_stopping_round` and `seed`, and setting the native name in `model.params` or `fit(params=...)` disagreed with them **in opposite directions**: the early-stopping override reached `lgb.train` on every call while the configured patience still decided when to stop, and `seed` silently beat `training.seed`, so the run's reproducibility control was not the one the config declared. Both are now `CONFIG_INVALID`, under every spelling (`n_iter_no_change`, `random_state`, …), naming the `training.*` setting to change instead. Measured: **0 of 916** and **0 of 928** pre-existing configs are affected.

- **`params_table()` and `export_code()` report the early-stopping patience the run actually used** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). A tuning result supplies `early_stopping_rounds` and the trainer takes it, but both of these read the configured value instead. Measured: config 7, tuned 2, the run trained at 2, and both reported 7 -- so **`export_code()` generated a project that would train a different model**. All four readers of that setting now share one definition. **If you exported code after a tune that changed the patience, regenerate it.**

- **A restored tuning result naming one parameter twice is refused before re-training** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). `best_model_params` restored from an artifact was overlaid without checking its own internal duplicates, so a result carrying both `learning_rate` and `eta` sent both to `lgb.train` and the library kept the canonical one. `Model.load()` still reads such an artifact -- it is the record of a fit that happened -- but the re-fit now raises `CONFIG_INVALID`. Current `tune()` cannot produce such a result, since two dimensions naming one parameter are refused before the study starts.

- **`params_table()` reports a parameter written under an alias** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). After `fit(params={"eta": 0.5})` the booster trained at `learning_rate=0.5` and the table listed neither name, because it read a fixed list of canonical names out of the booster by literal spelling. It now resolves each by parameter identity.

- **A search dimension for a parameter `training.*` controls is refused before the study starts** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). A `category: model` dimension named `seed` or `early_stopping_round` was sampled, trained on, and recorded in `best_model_params` -- and then the `fit()` that follows refused it, so a completed study produced a result its own next step could not consume. Measured over all seven spellings of both parameters: each study trained real boosters before the refusal arrived. The conflict is now raised before the study starts, naming `tuning.optuna.space` and the training setting. Measured: **0 of 70** pre-existing `category: model` spaces are affected.

- **Two search dimensions naming one parameter are refused instead of one being optimised over for nothing** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). A `category: model` space with both `learning_rate` and `eta` sampled both on every trial and put both in the parameter dict, so LightGBM kept the canonical one and **the other dimension had no effect on any trial** — Optuna ranked the trials on an axis that did nothing, and `best_model_params` recorded the dead value, which the `fit` afterwards then carried. Two dimensions spelling one LightGBM parameter now raise `CONFIG_INVALID` before the study starts, naming both. As for the parameter dicts since H-0096, there is no same-value exemption: two dimensions sample independently. Measured: **0 of 69** pre-existing `category: model` spaces this repository's suite constructs are affected. (A dimension colliding with an active smart parameter, [#279](https://github.com/nbx-liz/LizyML/issues/279), is refused as well since H-0099.)

- **An alias in `calibration.params` now overrides the calibrator's default instead of being defeated by it** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). The isotonic calibrator merges your parameters over its own defaults by dictionary key, and those defaults use LightGBM's canonical names. So `calibration: {params: {eta: 0.5}}` passed every check — you wrote the parameter once — and then arrived at `lgb.train` beside the default `learning_rate: 0.03`, which LightGBM preferred: the calibrator trained at `0.03`. `random_state` was defeated the same way. Names are now canonicalised before the merge. **A calibration config that set a parameter under an alias was training at the calibrator's default; it will now train with your value.** Relatedly, the calibrator's own `verbose = -1` was written under an alias and could be overridden by writing `verbosity`; it is now forced under the canonical name and holds.

- **`params_table()` and `export_code()` describe the fit that happened, not the newest study** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). `tune()` replaces the tuning result without replacing the fitted models, so after `fit()` then `tune()` both surfaces reported the new study's early-stopping patience while the fitted models still carried the configured one — and `export_code`, which generates a project meant to reproduce the training, generated one that would train a different model. The patience is now read from the trained model itself, so it also survives `export()` / `load()`. The inner-validation ratio had the same shape in the other direction: a `category: training` `validation_ratio` dimension was applied by the trainer and neither surface reported it. Both now come from what the fit applied. A model restored with `Model.load()` reports the applied ratio too since H-0109 (under *Added*).

- **A `fit()` or `tune()` that fails no longer rewrites what the retained model reports** (H-0094, [#264](https://github.com/nbx-liz/LizyML/issues/264)). Both published their state where each value became available, which is before the call has succeeded. So a refused or failed call left the model that was kept describing the attempt that failed: `params_table()` and `export_code()` took the failed call's inner-validation ratio, and `_X` / `_y` — the data SHAP and the diagnostics read — became a frame no trained model had ever seen. Everything a report reads about a fit is now committed together with the fit result, and the same at the end of `tune()`.

- **`validate_no_target_leakage` no longer skips a column it cannot compare** ([#267](https://github.com/nbx-liz/LizyML/issues/267)). A column whose comparison with the target raised `TypeError` / `ValueError` was skipped, and the check returned the remaining columns' result -- `[]` if nothing else leaked, the same answer as a frame that was fully checked. Any other exception escaped raw, without the column name. Now any comparison failure raises `LizyMLError(DATA_SCHEMA_INVALID)` with `context["column"]` / `context["target"]` and the original error as `cause`, whatever `raise_on_violation` is. The reachable input is a numeric-declared pandas extension array whose `__array__` or `isna` raises. Measured: no ordinary column dtype reaches it (0 of 462 dtype x target cells, 0 of 13 calls in the test suite). Columns are still checked in order, so an earlier leaking column still raises `LEAKAGE_SUSPECTED` first.

- **`tune(resume=True)` no longer reports an expansion that leaves a range unchanged** ([#318](https://github.com/nbx-liz/LizyML/issues/318)). A dimension whose best value sat at an edge that was already on a clamp (`min_allowed` / `max_allowed`, the linear `0.0` floor, the `IntDim` `max(1, ...)` guard) was reported as expanded every round even though its range did not move. H-0078 decided such an expansion is re-judged as not expanded. It now reports `expanded=False` with `new_low` / `new_high` `None`, and it is left out of `BoundaryReport.expanded_names`, `RoundSummary.expanded_dims`, the progress callback's `expanded_dims` and the `expanded` column of `boundary_table()`. `clamped_to_bound` stays `True` when a `min_allowed` / `max_allowed` clamp caused it, and stays `False` for the `0.0` floor and the `max(1, ...)` guard, which never set it. `IntDim` expansion is computed in integer arithmetic, so the comparison stays exact above `2**53`. The search space itself is unchanged. In the default LightGBM space, `feature_fraction` and `bagging_fraction` start at their upper bound of 1.0.

### Documentation

- **BLUEPRINT states the contracts the code keeps.** It now folds in every decided proposal, and `docs/proposal_dispositions.toml` records each proposal's disposition, checked by a test (H-0110, [#271](https://github.com/nbx-liz/LizyML/issues/271)). It states one rule for the inner-validation gap: only an automatically resolved inner validation inherits the outer split's gap, and `TimeHoldoutInnerValid.gap` is set only by that resolution (H-0092, H-0101, [#265](https://github.com/nbx-liz/LizyML/issues/265)). `ARCHITECTURE.md` states `format_version` 2 ([#266](https://github.com/nbx-liz/LizyML/issues/266)). The `FitResult` and `PredictionResult` field lists in BLUEPRINT §7 and `docs/api.md` match the dataclasses, and a test keeps them in step ([#326](https://github.com/nbx-liz/LizyML/issues/326)).

- **BLUEPRINT §5.5 states where every defaulted constructor setting gets its value** ([#268](https://github.com/nbx-liz/LizyML/issues/268)). #268 reported 25 such settings as unreachable from Config; executing each path shows most were reachable under a different name. The six `max_train_size` / `max_test_size` come from `split.train_size_max` / `split.test_size_max`, `PrecisionAtK.k` / `ECE.n_bins` / `HuberLoss.delta` from dict-form metric entries in `evaluation.metrics`, and `early_stopping_rounds` from `training.early_stopping.rounds`. Of all 74, 60 receive a Config key's value, 4 come from a public argument, 3 are derived, 2 are fixed by the library (`LGBMAdapter.verbose_eval = -1`, `StratifiedKFoldSplitter.shuffle = True`) and 5 are internal. The 14 that no Config key sets are listed with their source in §5.5, and a test keeps the list complete. No behaviour changes.

- BLUEPRINT and HISTORY now state what the code does for five more behaviours. The legacy `embargo_pct` key rejects fractional values instead of converting them with `int()`. Under `output_dir`, `fit()` / `tune()` write `run.log` only (a pathless `export()` still writes to `{run_dir}/export`), and plots are never saved. Metrics carry no `supports_task` property; the registry decides which task a metric supports. There is no `migrations/` package. The error codes raised are `UNSUPPORTED_METRIC`, `EVALUATION_FAILED` and `CONFIG_INVALID`. HISTORY Status lines for seven implemented proposals are corrected ([#319](https://github.com/nbx-liz/LizyML/issues/319)). No other behaviour changes.

- `docs/api.md` and BLUEPRINT §16.2 now list exactly the `ErrorCode` members, and a test keeps both in step with the enum.

### Internal

- **Releases are tagged only from a merge commit of develop** (H-0117). `auto-release.yml` checks that a release PR's head is `develop` in this repository, that its merge commit has two parents and that its title is exactly `release: vX.Y.Z`, then tags the merge commit idempotently, so a re-run after a failed later step can finish. `scripts/release.py` no longer commits `CHANGELOG.md` on develop or pushes develop. `CONTRIBUTING.md` is the tracked source of the workflow rules.

- Derive parameter-domain predicates during normalization, removing duplicate
  structural walks without changing accepted values or estimator wire bytes.

- Pin the deliberate categorical-choice restriction and numeric-bound conversion
  across all search dimension types; numpy choices remain entrance errors.

- Tests only: 16 tests are repaired. Eleven whose names claimed an effect they did not observe now assert it where it happens, or name the unit they test. Five that observed their outcome without isolating it (they could pass vacuously, credited the wrong gate, or carried a name contradicting their assertion) are tightened or renamed. The re-measured population and the instruments that produced it are under `docs/audits/2026-09-defect-discovery/` (#270). No behaviour changes.

- A test now reads every public reporting surface before export and after load (440 cells). Two differences remain and are declared: `importance("gain")` moves by up to about 5e-6 relative because LightGBM writes split gains with six significant digits, and the tuning table, plot and boundary table lose the trial history H-0086 does not persist ([#315](https://github.com/nbx-liz/LizyML/issues/315)).

### Known issues

- `model.params` dict-form entries for a LightGBM-native metric drop their parameters ([#313](https://github.com/nbx-liz/LizyML/issues/313)).
- A loaded model's tuning table, plot and boundary table show no trials, because the trial history is not persisted ([#315](https://github.com/nbx-liz/LizyML/issues/315)).
- The generated `train.py` retrains a multiclass `balanced` model without class weights ([#301](https://github.com/nbx-liz/LizyML/issues/301)), and the generated pipeline keys categories by their string form ([#304](https://github.com/nbx-liz/LizyML/issues/304)).
- `objective: cross_entropy_lambda` outputs are not probabilities ([#307](https://github.com/nbx-liz/LizyML/issues/307)).

## [0.17.1] - 2026-07-04

Internal / test-quality patch — no user-facing API or behavior change ([#218](https://github.com/nbx-liz/LizyML/issues/218), [#247](https://github.com/nbx-liz/LizyML/pull/247)).

### Internal

- **Extracted pure helpers to remove white-box test mocking** ([#218](https://github.com/nbx-liz/LizyML/issues/218)). `explain/shap_explainer.py` gains `_normalize_shap_output(raw, task)` and `data/validators.py` gains `_series_perfectly_correlated(col, y)`, isolating the SHAP-output normalization and the target-leakage NaN-ordering guard so both are unit-testable with plain inputs. Pure refactor — public API / `FitResult` / `Artifacts` / `format_version` unchanged.

### Tests

- **Redistributed the date-batched / coverage-anchored test files into per-module `tests/test_<module>/` directories** ([#218](https://github.com/nbx-liz/LizyML/issues/218)). The `test_bugfix_batch_2026_04*.py`, `test_code_review_fixes.py`, and `test_coverage/test_edge_cases.py` grab-bags were split by the module under test (empty `test_coverage/` package removed). The SHAP-normalization and leakage-ordering suites now assert against the extracted helpers with plain inputs instead of patching library internals (`MagicMock` explainer / `np.allclose` monkeypatch), and a mismatched-NaN-count regression case pins the leakage guard's short-circuit ordering.

## [0.17.0] - 2026-07-04

Full-package review remediation release: leakage-safe codegen retrain, metrics
transparency, and facade decomposition (issues [#203](https://github.com/nbx-liz/LizyML/issues/203)–[#218](https://github.com/nbx-liz/LizyML/issues/218), [#228](https://github.com/nbx-liz/LizyML/issues/228), [#237](https://github.com/nbx-liz/LizyML/issues/237); H-0085–H-0091).

### Added

- **Leakage validators are now public API on `lizyml.data`** (H-0087, [#216](https://github.com/nbx-liz/LizyML/issues/216)). `validate_time_series_order`, `validate_no_target_leakage`, and `validate_group_split` (previously dead code with no call sites) are re-exported from `lizyml.data` and documented, so users can run explicit time-order / target-leakage / group-overlap checks in line with the leakage-first charter. They are not auto-wired into `Model.fit` (that behavior change is deferred to a future proposal). The empty, unused `lizyml.utils` package was removed. Additive; no behavior change.
- **Contract types and the unified exception are now importable from the top-level package** (H-0086, [#213](https://github.com/nbx-liz/LizyML/issues/213)). `FitResult`, `PredictionResult`, `TuningResult`, `LizyMLError`, `ErrorCode`, `load_config`, and `TaskType` are re-exported from `lizyml` (previously only reachable via the private-looking `lizyml.core.*` paths), and `DataFingerprint` is re-exported from `lizyml.core.types`. Users can now write `from lizyml import FitResult, LizyMLError` for type annotations and `except LizyMLError` handling. Purely additive; a golden test pins the top-level `__all__`.
- **Calibrated OOF metrics now surface an uncalibrated-fallback row count** (H-0089, [#218](https://github.com/nbx-liz/LizyML/issues/218)). When a cross-fit calibration fold has no usable training scores (e.g. a TimeSeriesCV first period whose rows are all uncovered) or a single-class training slice, its validation rows fall back to the **uncalibrated** OOF probabilities (documented H-0058 behavior). Previously this blend was silent. `CalibrationResult` now records `fallback_fold_flags` (per-fold) and `n_fallback_rows` (total), and `metrics["calibrated"]` carries a `fallback_row_count` alongside `oof` / `oof_per_fold`. Additive — the common fully-calibrated case reports `0`, existing calibrated metric values are unchanged, and `format_version` stays `2` (pre-#218 `CalibrationResult` objects read back as `0` via `getattr`).
- **Tuned parameters are now persisted and restored across `export()` / `Model.load()`** (H-0086, [#215](https://github.com/nbx-liz/LizyML/issues/215)). `export()` records the tuned-param overlay (`best_model_params` / `best_smart_params` / `best_training_params` + score / metric / direction) under a `tuning` block in `metadata.json`, and `Model.load()` restores it into `_tuning_result`. A re-`fit()` after `load()` now reproduces the tuned params instead of silently reverting to config defaults. Additive and back-compatible — non-tuned and pre-#215 artifacts have no `tuning` block and load with `_tuning_result = None`; `format_version` stays `2`. (Restoring the full optuna study for a complete `tune(resume=True)` from a loaded model remains a follow-up.)

### Changed (potentially breaking)

- **Unified the inner-valid (early-stopping) pipeline fit boundary** (H-0085, [#208](https://github.com/nbx-liz/LizyML/issues/208)). `RefitTrainer` now fits the feature pipeline **once on the full dataset** — the same outer-train boundary `CVTrainer` already uses — instead of fitting on the inner-train slice and then refitting a second pipeline on all data. This removes the double fit and aligns `best_iteration` selection with the CV folds. OOF predictions are unaffected (the y-free `NativeFeaturePipeline` never let outer-valid rows into the fit); however, a model's `best_iteration` — and therefore the refit model — can change for configs that use early stopping. Resolves a BLUEPRINT self-contradiction (§6.2 / §10.3.2). `format_version` unchanged.
- **`purge_gap` / `embargo` / `gap` now propagate to the early-stopping (inner-valid) boundary** (H-0085, [#212](https://github.com/nbx-liz/LizyML/issues/212)). For an auto-resolved inner-valid split, `purged_time_series` purges `purge_gap + embargo` rows and `time_series` purges `gap` rows between inner-train and inner-valid, instead of placing them directly adjacent (zero gap). This closes a look-ahead leak that biased `best_iteration` on every fold when the target is constructed from future windows. OOF predictions are unaffected. Because the early-stopping split changes, `best_iteration` — and therefore the trained model — can change for `time_series` / `purged_time_series` configs with a non-zero gap/purge/embargo and early stopping enabled. `format_version` unchanged.

### Changed (potentially breaking)

- **Generated `train.py` now reproduces the model's time/group split** (H-0090, [#228](https://github.com/nbx-liz/LizyML/issues/228)). Following up on #206 (which only warned), the exported retrain script rebuilds the calibration OOF folds from `config.json["split"]` using the model's actual `split.method` — `time_series` / `group_kfold` / `stratified_group_kfold` via sklearn, and `purged_time_series` / `group_time_series` / `blocked_group_kfold` via a numpy port of LizyML's splitters — sorting the data the same way (by `time_col` / `blocks.col`) before splitting. Retraining from the script is now leakage-safe (no shuffle across the temporal/group boundary), verified by a codegen test asserting fold-index equality with LizyML for every method. The calibrator is now fit on covered rows only (uncovered first-period rows have NaN OOF), matching LizyML's cross-fit `C_final`. The #206 `UserWarning` / `train.py` banner is removed. `config.json` gains an additive `split` block (older exports without it fall back to the previous shuffled K-fold); `predict.py` / `model.txt` are unchanged.

### Fixed

- **Bare `ValueError` / `RuntimeError` in user-reachable paths are now unified `LizyMLError`s** ([#214](https://github.com/nbx-liz/LizyML/issues/214)). The NaN-in-covered-OOF guard in `Evaluator` (reachable from `Model.fit()`) now raises `LizyMLError(EVALUATION_FAILED)` with `nan_count` / `nan_indices` context; the calibrators' not-fitted guards (`platt` / `isotonic` / `beta`, reachable via `fit_result.calibrator`) raise `LizyMLError(CALIBRATION_NOT_FITTED)` with a `calibrator` tag; and the LightGBM feval-construction guards raise `LizyMLError(EVALUATION_FAILED` / `CONFIG_INVALID)`. `except LizyMLError` now catches all of them. Two additive `ErrorCode` members (`EVALUATION_FAILED`, `CALIBRATION_NOT_FITTED`) were added. Callers that caught the bare `ValueError` / `RuntimeError` should switch to `LizyMLError`.
- **An explicit `inner_valid` strategy now survives a config round-trip** (H-0086, [#203](https://github.com/nbx-liz/LizyML/issues/203)). `model_dump()` always re-emits the computed `validation_ratio`, which previously flipped the explicitness heuristic on reload so an explicit `inner_valid` (e.g. `time_holdout` / `group_holdout`) was silently replaced by split-derived auto-resolution on every `dump -> reload` and `export -> load -> fit` — leakage-relevant for time/group data. `EarlyStoppingConfig` now emits a round-trip-safe `inner_valid_explicit` marker (popped on re-validation like `validation_ratio`) that preserves the user's explicit choice. Additive: pure-legacy `validation_ratio` input keeps its auto-resolve semantics, and configs dumped before this change fall back to the previous heuristic. No new settable input field.
- **`fit()` and `load()` no longer expose the internal `FitResult` by reference** (H-0086, [#204](https://github.com/nbx-liz/LizyML/issues/204)). `fit()` — the primary access path — now returns a selective deep copy (the same isolation the `fit_result` property already applied under H-0082), and `Model.load()` deep-copies the restored metrics dict. Mutating a returned result can no longer corrupt internal state or contaminate a later `export()`'s `metadata.json`. Trained estimators (`models` / `calibrator` / `pipeline_state`) stay shared by reference as before; the return type is unchanged.
- **Config validation hardening** (H-0085, [#210](https://github.com/nbx-liz/LizyML/issues/210)):
  - **`config_version` string bypass** — a string value (e.g. `"999"`) previously skipped the supported-version gate and was then lax-coerced by pydantic, loading an unsupported version silently. The gate now coerces to `int` before the check, so unsupported versions are rejected as `CONFIG_VERSION_UNSUPPORTED` regardless of input type.
  - **Legacy `embargo_pct` / `gap` truncation** — a fractional legacy value (e.g. `embargo_pct=0.05`) was migrated via `int()` and silently collapsed to `0`, removing the leakage guard. Fractional legacy values are now rejected with `CONFIG_INVALID` and guidance to supply an integer observation count; integer-valued inputs still migrate.
  - **Shuffled `inner_valid` under a time-ordered outer split** — an explicit `inner_valid.method="holdout"` (shuffled) combined with a `time_series` / `purged_time_series` outer split now emits a `UserWarning` (the temporally-leaked early-stopping split is otherwise silent). Behavior is unchanged — the explicit choice is still honored.
- **Exported `predict.py` now matches `Model.predict()` on unseen categories** ([#205](https://github.com/nbx-liz/LizyML/issues/205)). The runtime `CategoricalEncoder` defaults to `unseen_policy="mode"` (unseen values → most frequent training category), but the exported `predict.py` mapped unseen categories to NaN and the artifact conversion dropped `unseen_policy` / `modes` — a silent prediction divergence in production. `pipeline_state.json` now carries `unseen_policy` and a per-column `unseen_codes` (the training mode's integer code), and `predict.py` applies the policy. This also fixes a latent crash: `predict.py` now feeds LightGBM a numpy array (matching features positionally) instead of a DataFrame, so a model with any categorical feature no longer raises `train and valid dataset categorical_feature do not match` — previously untested because the real-subprocess equivalence data had no categorical column.
- **`export_code()` warns when the generated `train.py` cannot reproduce the model's split** ([#206](https://github.com/nbx-liz/LizyML/issues/206)). For `time_series` / `purged_time_series` / `group_kfold` / `stratified_group_kfold` / `group_time_series` / `blocked_group_kfold` models, the generated `train.py` retrains and calibrates with **shuffled** random K-fold CV — silently leaking across the temporal/group boundary on retrain. `export_code()` now emits a `UserWarning` and injects a banner into `train.py` documenting the limitation (the exported `model.txt` / `predict.py` are unaffected — only re-training from the script is). Reproducing the time/group split in the generated retrain is tracked as follow-up work.
- **A NaN in a numeric / regression target is now rejected** (H-0085, [#207](https://github.com/nbx-liz/LizyML/issues/207)). Previously the `Model.fit` contract was undefined for a NaN numeric target (only string classification targets were validated), so it was silently accepted and could corrupt training. It now raises `LizyMLError(DATA_SCHEMA_INVALID)` with a `nan_count` context, symmetric with the existing string-target check.
- **Generated `requirements.txt` now pins `scipy` only when the model uses beta calibration** ([#218](https://github.com/nbx-liz/LizyML/issues/218)). `scipy` is imported by the generated `train.py` solely inside `_fit_beta` (the predict-time beta application is pure numpy), so a non-beta export previously over-pinned an unused dependency, contradicting the README note that `scipy` is needed only "when the model uses beta calibration". `render_requirements_txt()` now emits `scipy` conditionally on `calibration_method == "beta"`. Generated `predict.py` / `model.txt` are unaffected.

### Internal

- **`tune()` orchestration extracted into a writer-exempt mixin** (H-0091, [#237](https://github.com/nbx-liz/LizyML/issues/237)). The ~430-line tuning orchestration (`tune` + `_validate_tune_inputs` / `_resolve_search_space` / `_maybe_expand_boundary` / `_build_tune_objective` / `_run_tune_round` / `_assemble_tuning_result` / `_get_tuning_state`) moved from `core/model.py` to a new `core/_model_tuning.py` (`ModelTuningMixin`), bringing the facade from 1257 to **789 lines** (< the 800 guidance). Resolves the H-0077 conflict deferred from #209: the read-only-mixin invariant applies only to the *diagnostic* mixins (plots/tables/persistence, enforced by the `_MIXIN_FILES` guard); `ModelTuningMixin` is a *writer* that runs during the mutating `tune()` lifecycle and is deliberately outside that guard. `_DEFAULT_METRICS` moved to the shared `core/_model_metrics.py`. Pure refactor — no public-API / `FitResult` / `format_version` change; the full tuning + retune suite is unchanged.
- **Fail-closed leakage tests** (H-0085, [#207](https://github.com/nbx-liz/LizyML/issues/207)) — replaced a tautological OOF leakage assertion with a trap that records each fold estimator's training rows and asserts disjointness from its validation rows; added traps for per-fold calibration boundaries (calibrator fit on the train slice, not the scored valid rows) and inner-index containment (relative to each outer train fold).
- **Plot optional-dependency guard de-duplicated** ([#218](https://github.com/nbx-liz/LizyML/issues/218)). The per-module `_require_plotly` / `_check_plotly` guards (two names, three message strings) are replaced by a single `lizyml/plots/_deps.py::require_plotly(...)`, now applied across **all** plot modules (`learning_curve` / `oof_distribution` / `classification` / `residuals` previously kept an inline `if _plotly is None` guard). Same `OPTIONAL_DEP_MISSING` behavior. The near-duplicate `_collect_is_data` (classification) / `_build_is_data` (residuals) IF-data collectors are hoisted into a single `lizyml/plots/_helpers.py::collect_is_data(...)`.
- **`fit()` after `tune()` now logs an optimistic-bias warning** ([#218](https://github.com/nbx-liz/LizyML/issues/218)). A `fit()` following `tune()` reuses the identical deterministic CV splits used to select the params, so its reported OOF metrics are optimistically biased (documented user-side policy, BLUEPRINT §11.6, previously unenforced at runtime). The facade now emits one `event='fit.post_tune'` warning per such fit. Log-only; no contract change.
- **Test hygiene — white-box regression test made observable** ([#218](https://github.com/nbx-liz/LizyML/issues/218)). The target-leakage NaN-order regression test no longer patches `np.allclose` to inspect internal call arguments; it asserts the observable return value instead. Stale per-file coverage-delta anchors (`# N. module (X% -> Y%)`) in `test_edge_cases.py` were replaced with plain module headers. Broader redistribution of the date-batched bugfix files into per-module test directories remains tracked in #218.
- **Facade slimming — extracted accumulated logic out of `core/model.py`** ([#209](https://github.com/nbx-liz/LizyML/issues/209)). Round-summary assembly and per-trial round renumbering moved to `lizyml/tuning/rounds.py` (`assemble_round_result`); split-driven data ordering/extraction moved to `lizyml/data/dataframe_builder.py` (`prepare_for_split` / `sort_components`). `core/model.py` dropped from 1355 to 1244 lines. Pure refactor — no behavior change, no public-API/`format_version` change. Relocating the ~471-line `tune()` orchestration into a mixin is tracked as a follow-up (it needs an invariant decision vs the H-0077 read-only-mixin rule).
- **Generated `test_equivalence.py` now imports `predict.py`** ([#217](https://github.com/nbx-liz/LizyML/issues/217)) — the codegen equivalence checker previously inlined a second copy of the transform / calibration logic that had already diverged from `predict.py` (e.g. missing column-drift validation), so it could pass while validating a different code path than users run. It now calls `predict.predict()` directly, and a template guard test asserts no second prediction implementation is emitted.

## [0.16.1] - 2026-06-30

### Fixed

- **Generated `train.py` / `predict.py` / `test_equivalence.py` open JSON artifacts as UTF-8** ([#192](https://github.com/nbx-liz/LizyML/issues/192)) — every `open()` in the exported scripts now pins `encoding="utf-8"`. A Windows (`cp1252`) or `C`-locale end-user running the generated scripts over data with a non-ASCII categorical value previously hit a `UnicodeEncodeError` / `UnicodeDecodeError` when writing/reading `pipeline_state.json`. (Completes [#180](https://github.com/nbx-liz/LizyML/issues/180), which fixed only the LizyML-side codegen writes.)

## [0.16.0] - 2026-06-01

Quality-audit remediation release: resolves the v0.15.0 comprehensive audit
(issues [#167](https://github.com/nbx-liz/LizyML/issues/167)–[#180](https://github.com/nbx-liz/LizyML/issues/180), all except the v1.0 deprecation-removal tracker #148).

### Changed (potentially breaking)

- **`training.seed` now propagates to the outer splitter and the isotonic calibrator** (H-0080, [#169](https://github.com/nbx-liz/LizyML/issues/169)). `split.random_state` defaults to a sentinel `None` meaning "inherit `training.seed`", and the loader no longer hard-codes `42`. For configs that set `training.seed` to a non-`42` value **without** an explicit `split.random_state`, CV fold composition — and therefore OOF predictions, metrics, and saved split indices — now reflects `training.seed` (previously folds were silently fixed at `42`). Explicit `split.random_state` is still honored and unchanged; all-default configs are unaffected (the effective seed stays `42`).

### Added

- **Artifact integrity binding (SHA-256)** (H-0083, [#179](https://github.com/nbx-liz/LizyML/issues/179)). `export()` records the SHA-256 of each `.pkl` (`fit_result` / `refit_model` / `analysis_context`) in `metadata.json` under a `checksums` field; `Model.load()` verifies the digest before `joblib.load` and raises `DESERIALIZATION_FAILED` on a mismatch, detecting tampering or corruption. Additive and back-compatible — artifacts without the field still load and `format_version` stays `2`. (Does not make pickle safe against a trusted-but-malicious producer; the trusted-source contract is unchanged.)

### Fixed

- **multiclass OvR metrics (AUC / AUCPR / Brier) no longer fail on a class-missing CV fold** ([#167](https://github.com/nbx-liz/LizyML/issues/167)) — they now macro-average only over the classes present in `y_true` instead of raising an unwrapped `ValueError` / silently degrading.
- **cross-fit calibration tolerates a single-class training fold** ([#168](https://github.com/nbx-liz/LizyML/issues/168)) — it falls back gracefully instead of raising an unwrapped `ValueError`.
- **`evaluate(None)` and the `fit_result` property return independent copies** (H-0082, [#174](https://github.com/nbx-liz/LizyML/issues/174)) — they no longer hand out live internal references, so mutating a returned metrics dict / `FitResult` can no longer corrupt internal state or contaminate a later `export()`. Trained estimators reachable via `fit_result` (`models` / `calibrator` / `pipeline_state`) are shared by reference (read-only by convention) to preserve LightGBM Booster fidelity.
- **`export_code()` writes generated files as UTF-8** ([#180](https://github.com/nbx-liz/LizyML/issues/180)) — fixes a `UnicodeEncodeError` when exporting on Windows, where the default `cp1252` codec could not encode non-ASCII template characters.

### Internal

- **Facade `predict()` extracted** ([#172](https://github.com/nbx-liz/LizyML/issues/172)) — estimator/calibration/SHAP branching moved to `core/_model_predict.py`; `Model.predict()` is now assembly-only.
- **`FitState` / `TuningState` moved out of Layer-0** (H-0084, [#171](https://github.com/nbx-liz/LizyML/issues/171)) — relocated from `core/types/` to facade-adjacent `core/_model_state.py`, restoring the Layer-0 "dependency-free" invariant (the sole DAG back-edge removed). Internal types; no public-API change.
- **Forward-compat CI lane + OS portability matrix** ([#180](https://github.com/nbx-liz/LizyML/issues/180)) — a non-blocking latest-deps lane (`uv sync --upgrade`) surfaces upstream breakage early, and an ubuntu/windows/macos smoke matrix catches path/newline portability issues. The no-upper-bound dependency policy is documented in CONTRIBUTING and BLUEPRINT §18.2.
- **Branch coverage enabled** ([#173](https://github.com/nbx-liz/LizyML/issues/173)) — `branch = true`; `--cov-fail-under` re-baselined to a branch-inclusive `96` (measured ~97%).
- **Forward-looking test hardening** ([#178](https://github.com/nbx-liz/LizyML/issues/178)) — codegen real-subprocess equivalence, a feature-pipeline leakage trap, reproducibility bit-equality + seed-sensitivity, README-block execution, inner-valid ratio guards across all strategies, and notebook coverage. Executing every tutorial in CI surfaced two previously-unreferenced, broken notebooks — the `calibration` and `SHAP` tutorials now run end-to-end (fixed outdated config/result-access against the current API); notebook execution also skips gracefully on a remote-dataset network outage instead of failing the release gate.
- **Documentation reconciliation** ([#170](https://github.com/nbx-liz/LizyML/issues/170), [#175](https://github.com/nbx-liz/LizyML/issues/175), [#176](https://github.com/nbx-liz/LizyML/issues/176), [#177](https://github.com/nbx-liz/LizyML/issues/177)) — corrected the FAQ custom-objective example, `api.md` `tune()` storage/study_name, `migration.md`, several docstrings, documented `oof_coverage`, scoped the bit-identical reproducibility guarantee to a fixed `(num_threads, CPU)` environment, and corrected the erroneous `simulate` CHANGELOG entry.

## [0.15.0] - 2026-05-10

### Changed (potentially breaking)

- **`LGBMConfig.params["objective"]` is now respected when task-compatible** (H-0079 Phase 1, [#159](https://github.com/nbx-liz/LizyML/issues/159)) — pre-0.15 `LGBMAdapter._build_params()` silently stripped any user/Optuna-supplied `objective` and force-set `_TASK_OBJECTIVE[task]`, so `default_space("regression")` trials sampling `"fair"` actually trained with `"huber"`. From this release: same-task values flow through to `lgb.train` (e.g. `objective="fair"` for regression now actually uses Fair loss); cross-task values raise `LizyMLError(CONFIG_INVALID)` instead of being silently demoted. Users who relied on the silent strip to suppress accidental cross-task injection see no behaviour change at the contract level (still rejected), but **same-task non-default objectives may produce different metrics than pre-0.15 runs**. Re-running tune over `default_space` may yield a different `best_params` because `"fair"` is now genuinely evaluated.
- **`LGBMAdapter._build_params()` enforces an objective invariant assertion** (H-0079 L5) — at the end of `_build_params()`, an `assert` validates that any user-supplied `objective` survives the build. Active in dev / test / CI; suppressed under `python -O`. Fail-fast guard against future regressions to the silent-strip pattern.

### Added

- **`TASK_COMPATIBLE_OBJECTIVES` whitelist in `lizyml.estimators.lgbm.defaults`** (H-0079 Phase 1) — public mapping `dict[str, frozenset[str]]` enumerating the canonical LightGBM objective names valid per task (regression: 9, binary: 3, multiclass: 2). Used by `_build_params()` for cross-task validation and exposed for downstream integrations until the Provider-level API ships in Phase 2.
- **`EstimatorProvider.objective_choices(task) -> tuple[str, ...]`** (H-0079 Phase 2, [#159](https://github.com/nbx-liz/LizyML/issues/159)) — new Protocol method returning canonical objective names valid for *task* in deterministic order. `LGBMProvider.objective_choices(task)` ships ordered tuples (regression: 9, binary: 3, multiclass: 2) sourced from the same whitelist as `TASK_COMPATIBLE_OBJECTIVES`. Drift between the two surfaces is caught by a load-time invariant. Used by `default_space()` and downstream UIs (LizyStudio) so the canonical list lives in one place.
- **`EstimatorProvider.metric_choices(task) -> dict[Literal["native", "feval"], tuple[str, ...]]`** (H-0079 Phase 2, [#159](https://github.com/nbx-liz/LizyML/issues/159)) — new Protocol method returning per-task valid metrics split by source. `"native"` lists LightGBM-evaluated metrics composed straight into `params["metric"]`; `"feval"` lists LizyML custom metrics wired as feval callables. Canonical names only (aliases like `l1` / `l2` / `mse` are still accepted at config-input time but not surfaced).
- **`MetricChoices` type alias in `lizyml.estimators.provider`** (H-0079 Phase 2) — `dict[Literal["native", "feval"], tuple[str, ...]]`. Forward-compatible: future estimators may add new keys (e.g. `"sklearn"` for scikit-learn-backed metrics) without breaking existing consumers.
- **`default_space(task, provider=None)` accepts an optional `EstimatorProvider`** (H-0079 Phase 2) — when supplied, the `objective` `CategoricalDim` is built from `provider.objective_choices(task)` so default-space and user-supplied provider stay aligned. Existing call sites unchanged (provider defaults to `None` → conservative tune-safe subset).

### Fixed

- **`metric_bridge._LGBM_NATIVE_METRICS["multiclass"]` incorrectly listed `auc`** (H-0079 Phase 3, surfaced by L4 drift test) — LightGBM 4.x raises `LightGBMError("Multiclass objective and metrics don't match")` when `auc` reaches multiclass `params["metric"]`. The whitelist accepted the name pre-validation, so users got a cryptic LightGBM-side error instead of a clear LizyML rejection. The whitelist now omits `auc` for multiclass; users requesting AUC on multiclass should use `Model.evaluate(metrics=["auc"])` (sklearn OvR, computed Python-side post-fit) or `auc_mu` for fit-time evaluation.

### Internal

- **`_OBJECTIVE_CHOICES` retired** (H-0079 Phase 3) — replaced by a conservative `_DEFAULT_TUNE_OBJECTIVES` table in `lizyml/estimators/lgbm/defaults.py` whose values intentionally exclude `gamma` / `poisson` / `tweedie` / `mape` (target-distribution-restricted). The full canonical set is exposed via `LGBMProvider().objective_choices(task)` for downstream UIs and explicit user-supplied search spaces.
- **`_LGBM_OBJECTIVE_CHOICES` self-validates against `TASK_COMPATIBLE_OBJECTIVES`** at module load time (H-0079 Phase 2/3) so the two sources of truth cannot drift.
- **L4 MetricRegistry coverage drift test** (`tests/test_estimators/test_metric_choices_registry_coverage.py`, H-0079 Phase 3) — every fit-time-reachable metric in `MetricRegistry._TASK_METRICS` is now asserted to appear in `LGBMProvider.metric_choices()` after alias translation. Caught the multiclass `auc` omission above.
- **`_validate_metric_consistency()` load-time guard** (H-0079 follow-up, [#164](https://github.com/nbx-liz/LizyML/pull/164)) — parallel to `_validate_objective_consistency()`. Asserts at module load that every name surfaced via `_LGBM_NATIVE_METRIC_CHOICES` / `_LGBM_FEVAL_METRIC_CHOICES` is reachable via `metric_bridge` whitelists. Drift would offer a metric to a downstream UI that the library would later reject; fail-fast prevents that.
- **H-0079 follow-up coverage tests** (`tests/test_estimators/test_h0079_followup.py`, [#164](https://github.com/nbx-liz/LizyML/pull/164)) — 11 tests pinning previously-untested integration boundaries: codegen export with non-default objective (3), save/load round-trip with non-default objective (2), `_check_objective_compatible` edge inputs (4), and Platt calibration on top of binary `cross_entropy` objective (2).
- **`LGBMProvider.build_export_params` docstring** (H-0079 follow-up) gains a `Note:` block documenting the intentional same-package private call into `LGBMAdapter._build_params()` so future refactors update both call sites together.

## [0.14.0] - 2026-05-10

### Added

- **`EstimatorProvider.parameter_bounds(task)`** (H-0078, [#152](https://github.com/nbx-liz/LizyML/issues/152)) — new Protocol method returning per-parameter meaningful bounds (`{"min": ..., "max": ...}`) for boundary expansion. `LGBMProvider.parameter_bounds(task)` ships a static map for 15 LightGBM parameters (e.g. `learning_rate ∈ [1e-8, 1.0]`, `feature_fraction ∈ [1e-3, 1.0]`, `validation_ratio ∈ [0.05, 0.5]`, `max_depth ∈ [-1, 30]`). Used by `Model.tune` and downstream UIs (LizyStudio) to constrain user input. Third-party providers may return `{}` for unbounded behaviour.
- **`SearchDim.min_allowed` / `max_allowed`** (H-0078) — optional bounds on `FloatDim` / `IntDim` (default `None`). Boundary expansion clamps to these limits when set.
- **`BoundaryDimStatus.clamped_to_bound: bool`** (H-0078) — flags dims whose expansion hit the parameter-meaningful bound, so downstream UIs can badge "max reached" dims. Defaults `False`.
- **`attach_bounds(dims, bounds)` helper in `lizyml.tuning.search_space`** (H-0078) — injects `min_allowed` / `max_allowed` onto matching dims by name. Called from `Model._resolve_search_space` so default-space and user-supplied dims both pick up provider bounds automatically.

### Changed

- **`parse_space()` rejects degenerate / inverted ranges and log-with-non-positive-low** (H-0078, [#152](https://github.com/nbx-liz/LizyML/issues/152)) — `low >= high` and `log=True ∧ low <= 0` now raise `LizyMLError(CONFIG_INVALID)` at parse time instead of letting Optuna raise a generic error mid-trial. Strictly better failure mode (earlier and clearer).
- **`expand_dims` propagates `min_allowed` / `max_allowed`** (H-0078) — re-tune over multiple rounds preserves provider-supplied bounds, preventing the original `learning_rate` drift (`0.1 → 0.3 → 0.9 → 2.7`) reported in #152. 5- and 10-round regression tests guard the contract.

### Internal

- **`_expand_range` is bounds-aware** (H-0078) — keyword-only `min_allowed` / `max_allowed` arguments + 3-tuple return `(low, high, clamped)`. Internal signature change; only `detect_boundary` is a caller.

## [0.13.0] - 2026-05-10

### Added

- **`EstimatorProvider.build_export_params()`** (H-0073, [#109](https://github.com/nbx-liz/LizyML/issues/109), [#126](https://github.com/nbx-liz/LizyML/issues/126)) — codegen-relevant booster params and feval metadata are now retrieved through the `EstimatorProvider` Protocol, so `Model.export_code()` is fully estimator-agnostic. Adding a new estimator no longer requires editing `lizyml/core/_model_persistence.py`. The new method also unifies `BlockedGroupKFold` `n_splits` resolution between persistence and factories.
- **`FitState` / `TuningState` frozen dataclasses + `Model._get_fit_state()` / `_get_tuning_state()`** (H-0074 Phase 1 + H-0077 Phase 2, [#112](https://github.com/nbx-liz/LizyML/issues/112)) — `ModelPlotsMixin` / `ModelTablesMixin` / `ModelPersistenceMixin` now read state exclusively through these snapshots. Direct `self._<private>` access is forbidden inside Mixin bodies and enforced by a static guard test (`tests/test_core/test_mixin_state_isolation.py`). Mixins become unit-testable with synthetic state. Public API unchanged.
- **`docs/DEPRECATIONS.md` central deprecation registry** (H-0076, [#120](https://github.com/nbx-liz/LizyML/issues/120), [#121](https://github.com/nbx-liz/LizyML/issues/121)) — single source of truth for every deprecated public surface and its removal target version. Every `DeprecationWarning` LizyML raises now contains "Will be removed in vX.Y." (currently "v1.0"); `tests/test_core/test_deprecation_registry.py` enforces this contract in CI.

### Changed

- **`TaskType` Literal centralised + propagated to all dispatch sites** (H-0075, [#122](https://github.com/nbx-liz/LizyML/issues/122)) — `lizyml/core/types/task.py` exposes `TaskType = Literal["regression", "binary", "multiclass"]`. Every branch on task now uses an exhaustive dispatch table, eliminating string-comparison divergence between modules. Public API unchanged; internal type-safety only.
- **`DeprecationWarning` messages now state the removal target version** (H-0076) — users see "Will be removed in v1.0." in every deprecation message so migration deadlines are explicit.
- **Inner-valid membership checks vectorised with numpy** (H-0065 follow-up, [#135](https://github.com/nbx-liz/LizyML/issues/135)) — measurable speedup on large CV folds; behaviour unchanged.
- **`Model.tune()` decomposed into 5 testable helpers** (H-0040 follow-up, [#114](https://github.com/nbx-liz/LizyML/issues/114)) — orchestrator + per-step helpers, no behaviour change.
- **`LizyMLError.context` enriched at 23 sites** ([#118](https://github.com/nbx-liz/LizyML/issues/118)) — fold index / config path / method name now consistently carried; regression guard added.
- **`storage` parameter type tightened to `str | BaseStorage | None`** ([#136](https://github.com/nbx-liz/LizyML/issues/136)) — was `Any`; aligns with H-0072 docstring.
- **Plot theme deduplicated via `apply_default_layout`** ([#134](https://github.com/nbx-liz/LizyML/issues/134)) — `lizyml/plots/_theme.py` is now the single source for default layout settings shared across every plot module.
- **`StratifiedTimeHoldoutInnerValid` tail-holdout fallback inlined** ([#133](https://github.com/nbx-liz/LizyML/issues/133)) — readability improvement, behaviour unchanged.
- **`TargetEncoder` lexicographic class ordering documented with example** ([#132](https://github.com/nbx-liz/LizyML/issues/132)).

### Fixed

- **Tuning progress callback warning now includes exception type and message** ([#128](https://github.com/nbx-liz/LizyML/issues/128)) — debugging callback failures no longer requires re-running with logging tweaks.
- **`FloatDim` linear lower expansion clamped at zero** ([#129](https://github.com/nbx-liz/LizyML/issues/129)) — boundary-detection re-tune (H-0068) no longer drives lower bounds below zero on naturally non-negative parameters.
- **Legacy top-level `validation_ratio` input emits `DeprecationWarning`** ([#130](https://github.com/nbx-liz/LizyML/issues/130)) — previously a YAML with only `early_stopping.validation_ratio` (and no `inner_valid:` block) was silent. Now the deprecation contract is enforced on input, not just output.
- **Unhandled `SplitConfig` variants raise `LizyMLError(CONFIG_INVALID)`** ([#131](https://github.com/nbx-liz/LizyML/issues/131)) — previously a silent fallback could mask schema regressions.
- **Code-review HIGH issues + `#124` test gap closed** ([#141](https://github.com/nbx-liz/LizyML/issues/141)) — Sprint 1+2 follow-up batch.
- **MEDIUM / LOW code-review batch** ([#142](https://github.com/nbx-liz/LizyML/issues/142)) — accumulated cleanup landed in one PR.

### Internal

- **`TrainComponents` rebuild path extracted to `_sort_and_rebuild_components()`** ([#137](https://github.com/nbx-liz/LizyML/issues/137)) — refactor only.
- **Mixin source files contain zero direct `self._<private>` access** (H-0077 Phase 2, enforced by `tests/test_core/test_mixin_state_isolation.py`).

## [0.12.0] - 2026-05-06

### Added

- **Resumable tuning via Optuna persistent storage** (H-0072, [#105](https://github.com/nbx-liz/LizyML/issues/105)) — `Tuner` and `Model.tune()` now accept `storage` (Optuna URL such as `sqlite:///path/to.db` or a `BaseStorage` instance) and `study_name`. When set, trial state is persisted to disk after each trial completes; re-invoking `Model.tune(storage=..., study_name=...)` with the same identifiers re-attaches via `load_if_exists=True` so completed trials are not re-run. `storage=None` (default) preserves the in-memory behavior with no disk I/O. Designed for long-running tune jobs that must survive process kill, server restart, or network outage. No new dependencies (uses Optuna's built-in storage backends).

## [0.11.0] - 2026-05-05

### Added

- **sMAPE / WAPE — zero-tolerant percentage-style regression metrics** (H-0071, [#101](https://github.com/nbx-liz/LizyML/issues/101)) — `lizyml.metrics.SMAPE` and `lizyml.metrics.WAPE` are now available for `task=regression` and registered in `MetricRegistry` under `"smape"` / `"wape"`. Both close the gap left by MAPE on datasets where `y_true` may be `0` (sales / demand / count regressions). Wired into the LightGBM metric bridge so `params={"metric": ["smape", "wape"]}` produces feval-driven entries in `eval_history` / learning curves, and into the codegen exporter so `Model.export_code()` reproduces the same values offline. Authoritative formulas and edge-case conventions are documented in [`docs/config-reference.md` § Metric formula reference](docs/config-reference.md#metric-formula-reference).
- **Metric formula reference** — `docs/config-reference.md` now documents authoritative formulas, ranges, and edge-case conventions for every regression and classification metric LizyML ships, plus a MAPE / sMAPE / WAPE selection guide.

## [0.10.0] - 2026-05-04

### Added

- **Auto-encode non-numeric classification targets** (H-0070, [#98](https://github.com/nbx-liz/LizyML/issues/98)) — `task ∈ {binary, multiclass}` now accepts non-numeric `y` (object / `pd.StringDtype` / category / bool). LizyML applies a `TargetEncoder` automatically and `Model.predict()` returns predictions in the **original label dtype** (e.g. `"Adelie"` instead of `2`). The new `FitResult.target_encoder` carries `classes_` so consumers (incl. `export_code()`-generated `train.py` / `predict.py`) can map int codes back to the original labels. Calibration / tuning paths work transparently.
- **New error codes**: `TARGET_NOT_NUMERIC`, `TARGET_UNSEEN_LABEL`.

### Changed

- **`task=regression` × non-numeric `y` now raises `TARGET_NOT_NUMERIC` before model training starts** (H-0070) — previously fit failed with an unclear error from the LightGBM layer.
- **Codegen `predict.py` output dtype**: when the original target was non-numeric, generated predictions now decode int codes back to the original labels via a `target_encoder.classes` array baked into `config.json`.
- **Persistence `FORMAT_VERSION` bumped to 2** (H-0070) — `Model.load()` accepts both `format_version=1` (old) and `2` (current). v1 artifacts are migrated in-memory by injecting a no-op `TargetEncoder`, so existing saved models continue to load without user action.

## [0.9.1] - 2026-05-02

### Fixed

- **`Model.load()` fails for non-holdout `inner_valid`** (H-0069, [#95](https://github.com/nbx-liz/LizyML/issues/95)) — Saving a model fit with `inner_valid.method ∈ {group_holdout, time_holdout}` produced an artifact that could not be re-loaded (`CONFIG_INVALID`). Root cause: `validation_ratio` and `inner_valid.ratio` were two mutable fields whose only synchronization was a one-way validator branch that ignored `group_holdout` / `time_holdout`. `model_dump()` always emitted both keys, so the round-trip silently broke.

### Changed

- **`EarlyStoppingConfig.validation_ratio` is now a read-only computed field** (H-0069) — `validation_ratio` mirrors `inner_valid.ratio` automatically, eliminating the dual-write inconsistency at its source. Existing YAML inputs (`validation_ratio: 0.1` only, or `inner_valid: {...}` only) are fully backward compatible. Existing `model.lizyml` artifacts load without migration. Side effect: codegen `export_code()` now uses the correct holdout fraction when `inner_valid.ratio` differs from the default 0.1 (previously a silent ratio mismatch).

## [0.9.0] - 2026-04-12

### Added

- **Re-tune: Study Resume + Boundary Expansion** (H-0068)
  - `Model.tune(resume=True)` resumes from the previous Optuna Study with additional trials; TPE sampler reuses knowledge from prior trials and previous best params are enqueued as a warm-start trial
  - Automatic boundary detection identifies dimensions where best params are near the search space edge
  - Asymmetric space expansion extends promising directions only (linear: 2× range, log: 3× in log space)
  - `TuningResult.rounds` tracks per-round history (`RoundSummary` with scores, expanded dims, space snapshots)
  - `TuningResult.boundary_report` provides dimension-by-dimension boundary analysis (`BoundaryReport` / `BoundaryDimStatus`)
  - `Model.boundary_table()` returns boundary detection results as a DataFrame
  - `TuneProgressInfo` gains `round`, `cumulative_trials`, `expanded_dims` for real-time progress monitoring
  - `TrialResult.round` indicates which re-tune round each trial belongs to
  - `tuning_table()` includes `round` and `state` columns
  - `plot_tuning_history()` shows round boundary separators with expanded dimension annotations
  - New public types exported from `lizyml`: `BoundaryReport`, `BoundaryDimStatus`, `RoundSummary`
  - Fully backward compatible: `tune()` with no new parameters behaves identically to previous versions

## [0.8.1] - 2026-04-11

### Fixed

- **ECE formula corrected** (H-0067) — ECE per-bin accuracy now uses `mean(y_true)` (fraction of positives) instead of binarized-prediction accuracy. The old formula systematically overestimated ECE for well-calibrated models. Same fix applied to codegen templates.
- **Confusion matrix NaN exclusion** (H-0067) — `confusion_matrix_table()` now applies `compute_oof_valid_mask()` to exclude structurally uncovered rows (e.g., TimeSeriesCV first period) from the OOS matrix. Previously, NaN predictions were silently treated as class 0.
- **Leakage validator eval order** (H-0067) — `validate_no_target_leakage()` now checks NaN positions (`isna().equals()`) before `np.allclose()`, preventing a silent `ValueError` swallow when columns have NaN at different positions.
- **Isotonic calibrator log suppression** (H-0067) — Changed `lgbm.log_evaluation(period=0)` to `period=-1` for well-defined behavior in LightGBM 4.x.
- **RefitTrainer pipeline leakage boundary** (H-0067) — Pipeline is now fitted on inner-train rows only (consistent with CVTrainer). A second pipeline is fitted on all data for the final `pipeline_state` used at inference. `categorical_features` sourced from the full-data pipeline.
- **Cross-fit calibration NaN guard** (H-0067) — `cross_fit_calibrate()` now guards against NaN in validation indices. Finite rows go to `cal.predict()`, NaN rows fall back to uncalibrated OOF predictions.
- **Calibrated metrics include oof_per_fold** (H-0067) — `metrics["calibrated"]` now includes `oof_per_fold` in addition to `oof`. IF metrics remain excluded (leakage risk).
- **Inner validation empty train guard** (H-0067) — `HoldoutInnerValid` and `TimeHoldoutInnerValid` now raise `ValueError` when `n_valid >= n_samples` instead of producing an empty training set.

## [0.8.0] - 2026-04-03

### Added

- **Codegen feval metric support** (H-0066) — `export_code()` now preserves feval metrics (f1, brier, ece, precision_at_k, accuracy, rmsle, r2) in generated code. Previously, feval metrics were silently dropped during code generation, causing `metric="None"` in config.json and incorrect early stopping behavior.
  - `config.json` gains a `feval_metrics` field with metric metadata (name, params, greater_is_better, needs_proba)
  - `train.py` template includes pure numpy/sklearn feval implementations and a `build_feval_from_config()` factory
  - Backward compatible: empty `feval_metrics` produces identical output to previous versions
- **New estimator implementation guide** — `docs/add-estimator-guide.md` documents all requirements for adding non-LightGBM estimators: adapter, provider, config, metric bridge, codegen, and test checklist

## [0.7.3] - 2026-04-02

### Fixed

- **Tune → Fit exact identity** — Unified tune objective and fit code paths so both go through `_build_train_components(training_overrides=...)`. Previously, the tune objective rebuilt the estimator factory with pre-smart-resolution params (`merged_model`) when `early_stopping_rounds` was in the search space, causing `num_leaves` and `scale_pos_weight` to be missing. Tune and fit now produce bit-for-bit identical OOF scores.

## [0.7.2] - 2026-04-02

### Fixed

- **Tune → Fit parameter identity** (#76) — `default_fixed_params()` was leaking `auto_num_leaves` (a smart param) into model params during tuning. Additionally, `best_training_params` (`early_stopping_rounds`, `validation_ratio`) from tuning were not applied during subsequent `fit()`. Both issues caused score divergence between tune best trial and fit OOF. After fix, tune and fit produce identical model params and near-identical OOF scores (within LightGBM floating-point tolerance).

## [0.7.1] - 2026-04-02

### Fixed

- **Categorical search space validation** — `parse_space()` now rejects non-scalar choices (e.g. nested lists from YAML `- [auc, binary_logloss]`) with a clear `CONFIG_INVALID` error and a hint for the correct YAML format. Previously, such values passed through to Optuna's `suggest_categorical()` causing repeated warnings during tuning.

## [0.7.0] - 2026-03-28

### Added

- **Parameterised MetricEntry** (H-0065) — `precision_at_k` `k` parameter is now user-configurable via dict form in both `evaluation.metrics` and `model.lgbm.params.metric`
  - `metrics: ["auc", {precision_at_k: {k: 20}}]` sets top-K% cutoff
  - Evaluation and Model Params support independent `k` values
  - `params_summary()` displays feval metric parameters (e.g. `precision_at_k (k=20)`)
  - Learning curve subplot titles show parameterised metric names
  - Plain string `"precision_at_k"` continues to use default `k=10` (backward compatible)
  - Invalid metric parameters now raise `LizyMLError(CONFIG_INVALID)` instead of raw `ValueError`

## [0.6.1] - 2026-03-28

### Fixed

- **`r2` metric with early stopping** — `r2` was listed as a LightGBM native metric but is not implemented in LightGBM 4.6.0 (only in unreleased master). Passing `metric: "r2"` silently produced empty eval results, breaking early stopping. Moved `r2` from native whitelist to feval (custom function) so it works correctly with early stopping and learning curves.

## [0.6.0] - 2026-03-28

### Added

- **Metric bridge** (`metric_bridge.py`) — unified metric handling for LightGBM training (H-0064, #57, #58, #59)
  - LizyML metric names auto-translate to LightGBM equivalents (`logloss` → `binary_logloss`, `auc_pr` → `average_precision`)
  - Per-task whitelist validation before `lgb.train()` with clear error messages
  - Custom feval support for LizyML-only metrics: `rmsle`, `f1`, `brier`, `ece`, `precision_at_k`, `accuracy`
  - Native + feval metrics can be mixed (e.g. `params={"metric": ["auc", "f1"]}`)
- 64 new tests: metric mapping, whitelist validation, feval numerical correctness, behavioral training tests
- `docs/config-reference.md`: training metric reference, metric details table, two-system explanation

### Changed

- `_build_params()` now returns `(params, num_boost_round, feval_list)` — 3-element tuple
- `fit()` passes feval callables to `lgb.train(feval=...)` when custom metrics are specified
- Invalid metric names are now rejected at `_build_params()` time (pre-validation) instead of relying on LightGBM post-hoc detection

## [0.5.0] - 2026-03-28

### Added

- `LGBMConfig.params.metric` — user-specified LightGBM evaluation metric override with task-default fallback (H-0061, #50, #51)
- `plot_learning_curve(*, metrics=None)` — filter displayed subplots by metric name (H-0062, #52)
- `Model.plot_learning_curve(*, metrics=None)` — pass-through for metrics filter
- `params_summary()` now includes `metric` in output rows
- Silent invalid metric detection: `UserWarning` when user metric produces no eval results
- 70 new tests: metric override, learning curve filter, Config propagation + behavioral effect (H-0063)

### Fixed

- Variable shadowing bug in `plot_learning_curve()` — loop variable `metrics` overwrote the function parameter (pre-existing, exposed by H-0062)
- Empty string metric (`[""]`) now correctly falls back to task default

### Changed

- `_build_params()` no longer strips user-specified `metric` from params dict
- Error handling split: `LightGBMError` (metric keyword) and `ValueError` (eval metric) caught separately for precise diagnostics

### Removed

- Duplicate `_FIXED_METRIC` dict in `defaults.py` — consolidated into `_TASK_METRIC`

## [0.4.2] - 2026-03-21

### Added

- `CONTRIBUTING.md` — development workflow, quality gates, and spec-first process
- `SECURITY.md` — vulnerability reporting policy
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `Makefile` — unified development commands (`make ci`, `make test`, etc.)
- `.editorconfig` — cross-editor formatting consistency
- `.github/dependabot.yml` — automated dependency updates (pip + GitHub Actions)
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist template
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates

## [0.4.1] - 2026-03-21

### Changed

- Rewrote README: 620 → 190 lines with badges, installation, quick start, and architecture diagram
- Extracted Config Reference to `docs/config-reference.md` (384 lines)

### Added

- `scripts/release.py` — automated release script (CHANGELOG validation, commit, push, PR creation)
- `.github/workflows/auto-release.yml` — auto-tag and GitHub Release on merge to main
- GitHub Releases for all past versions (v0.1.0–v0.4.0)
- `CONTRIBUTING.md` — development workflow, quality gates, and spec-first process
- `SECURITY.md` — vulnerability reporting policy
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `Makefile` — unified development commands (`make ci`, `make test`, etc.)
- `.editorconfig` — cross-editor formatting consistency
- `.github/dependabot.yml` — automated dependency updates (pip + GitHub Actions)
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist template
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates

## [0.4.0] - 2026-03-21

### Added

- `blocked_group_kfold` split method — 2-axis cross-validation combining period-block splitting with group KFold (H-0060)
- `BlockedGroupKFoldSplitter` — new splitter with expanding/sliding window modes and cutoff-based period boundaries
- `BlockedGroupKFoldConfig` with nested `blocks` (col/cutoffs/mode/train_window) and `groups` (col/n_splits/stratify/shuffle) sections
- `BlockedGroupInnerValid` — group-isolated, time-ordered, stratified inner validation for early stopping
- `StratifiedTimeHoldoutInnerValid` — per-class tail selection fallback for inner validation when fewer than 4 groups
- 62 new tests (config, splitter, inner valid, factory, E2E) with 100% splitter coverage

## [0.3.0] - 2026-03-20

### Added

- `Model.export_code(path)` — generate LizyML-independent training and prediction scripts (H-0059)
- `lizyml/codegen/` package — config_writer, artifact_writer, templates, generator modules
- Exported output: `train.py`, `predict.py`, `test_equivalence.py`, `config.json`, `requirements.txt`, `artifacts/`
- Supports all task types (regression, binary, multiclass) and all calibrators (Platt, Beta, Isotonic)
- `BaseCalibratorAdapter.export_params()` — abstract method for calibrator parameter export
- `PlattCalibrator.export_params()`, `BetaCalibrator.export_params()`, `IsotonicCalibrator.export_params()` + `save_model_text()`
- `LGBMAdapter.save_model_text()` — export Booster to human-readable text format
- `NativeFeaturePipeline.export_state_json()` — export pipeline state to JSON
- 73 new codegen tests including E2E equivalence verification (5 patterns)

## [0.2.0] - 2026-03-17

### Added

- `StratifiedGroupKFoldSplitter` — new split method combining stratification with group boundaries (`split.method: "stratified_group_kfold"`) (H-0055)
- `metrics["raw"]["oof_coverage"]` — float (0.0–1.0) indicating the fraction of rows covered by OOF validation folds; `evaluate_table().attrs["oof_coverage"]` for programmatic access (H-0057)
- `compute_oof_valid_mask()` — derives OOF coverage mask from split indices, not NaN detection; NaN in covered rows raises `ValueError` for bug detection (H-0057)

### Changed

- Calibration cross-fit now reuses outer CV split indices directly instead of generating independent splits (H-0058)
- `CalibrationConfig.n_splits` is deprecated — non-default values emit `UserWarning` and are ignored (H-0058)
- `build_calibration_splitter()` is deprecated with `DeprecationWarning` (H-0058)
- OOF metrics are computed on covered rows only; TimeSeriesCV first-period rows (never validated) are excluded instead of propagating NaN (H-0057)
- `cross_fit_calibrate()` handles NaN training rows with identity fallback to `oof_pred` probabilities (H-0058)

### Internal

- 5-layer DAG architecture migration: dead code removal (H-0051), layer dependency purification (H-0052), `EstimatorProvider` protocol introduction (H-0053)
- `TrainComponents` frozen dataclass, `resolve_smart_params` dict unification, `TuningResult` 3-way category split (H-0050)
- `EstimatorProvider` extensibility: `params_summary`, `set_categorical_features`, provider-level factory dispatch (H-0054)
- Systematic test reinforcement: 92 new tests across 5 categories — config propagation, facade orchestration, provider invariants, artifact compatibility, tuning reproducibility, dtype boundaries, pairwise parameters (H-0056)

## [0.1.5] - 2026-03-15

### Fixed

- Calibration cross-fit OOF array now NaN-initialized instead of `np.empty` — prevents silent garbage values for time-series splitters
- `GroupTimeSeriesSplitter` last fold now extends to include all trailing groups (previously silently dropped)
- `ECE` metric last bin is now right-inclusive (`y_pred == 1.0` no longer excluded)
- `RMSLE` raises `LizyMLError` for negative predictions/targets instead of producing NaN
- `FitResult` post-construction mutation replaced with `dataclasses.replace()`
- `_prepare_training_data` no longer mutates `DataFrameComponents` in-place
- `evaluate()` bare `assert` replaced with proper `LizyMLError`
- `_filter_metrics` removes empty branches after filtering
- Task-locked `objective`/`metric` can no longer be overridden by user search space params
- `LGBMAdapter.update_params` creates new dict instead of mutating in-place
- `compute_shap_importance` handles empty models list without `ZeroDivisionError`
- QQ plots raise `LizyMLError(OPTIONAL_DEP_MISSING)` instead of bare `ImportError` when scipy is missing
- Tuner trial failures now logged via warning callback; catch tuple narrowed from `Exception` to specific types
- `TuningResult`/`TrialResult` deep-copy mutable `dict`/`list` fields in `__post_init__`
- `HoldoutInnerValid` `n_valid` uses `ceil` to match `HoldoutSplitter` rounding
- All timestamps now include UTC timezone info
- `params_table` guards against empty models list

### Changed

- CI test matrix now includes Python 3.11
- Added `[tool.coverage.run]` and `[tool.coverage.report]` configuration to `pyproject.toml`
- `PredictionResult.proba` docstring corrected for multiclass shape
- `cross_fit_calibrate` docstring notes raw score (logit) support

## [0.1.4] - 2026-03-14

### Fixed

- Multiclass OVA (`multiclassova`) predictions now correctly pass `roc_auc_score` validation; row-wise normalization applied only to simplex-required metrics (AUC, LogLoss) (H-0049)

### Added

- `BaseMetric.needs_simplex` property (default `False`) to distinguish metrics requiring probability distributions (sum=1) from per-class OvR metrics (H-0049)
- `AUC` and `LogLoss` override `needs_simplex=True`; per-class metrics (`AUCPR`, `Brier`) keep raw predictions (H-0049)

## [0.1.3] - 2026-03-14

### Added

- `IsotonicCalibrator` migrated to LightGBM native Booster API with early stopping and internal validation split (H-0047)
- `TuneProgressInfo` / `TuneProgressCallback` for `Model.tune(progress_callback=fn)` (H-0048)

### Fixed

- Remove double-sigmoid in `IsotonicCalibrator.predict()` — `Booster.predict()` already returns probabilities (H-0047)

## [0.1.2] - 2026-03-10

### Changed

- Calibration cross-fit splits now inherit `split.method` and its parameters (group/time/purge/embargo boundaries); only fold count is overridden by `calibration.n_splits` (H-0044)
- `evaluate()` now returns `raw.oof_per_fold` metrics computed on each outer fold's valid indices; `evaluate_table()` fold columns changed from IF to OOF-per-fold (H-0045)
- Calibration split failure now raises `LizyMLError(CONFIG_INVALID)` with `split_method`, `calibration_n_splits`, `n_samples`, and `n_groups` (when applicable) in context
- BLUEPRINT §13.4: IF/OOF classification for diagnostic vs generalization monitoring APIs (H-0046)

### Added

- Contract tests for `purged_time_series` calibration splits (purge_gap + embargo boundary verification)
- Contract tests for `group_time_series` calibration splits (group disjointness + temporal ordering)
- Golden test coverage for `oof_per_fold` in metrics structure
- README and notebook documentation for calibration split.method inheritance contract

## [0.1.1] - 2026-03-08

### Changed

- Decompose Model facade into mixins: ModelPlotsMixin, ModelTablesMixin, ModelPersistenceMixin, factory functions (H-0042)
- Consolidate test helpers into `tests/_helpers.py`; remove ~40 duplicated definitions (H-0043)
- Enhance pytest parametrize usage for common task-agnostic tests (H-0043)
- CI now runs on develop branch PRs; slow tests excluded for develop, included for main (H-0043)
- Default `pytest` run skips slow tests via `addopts`; use `-m ""` for all tests (H-0043)
- Add `--cov-fail-under=95` coverage threshold to CI (H-0043)

## [0.1.0] - 2026-03-07

### Added

- Config-driven ML pipeline for regression, binary, and multiclass classification
- LightGBM estimator adapter using native Booster API
- Cross-validation training with OOF/IF predictions
- Inner validation (early stopping) support
- Feature pipeline with leakage prevention
- Splitters: KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, Holdout
- Calibration: Platt, Isotonic, Beta (cross-fit, OOF-only)
- Evaluation with pre-computed metrics (raw + calibrated)
- SHAP explanations (optional dependency)
- Optuna-based tuning with unified search space (optional dependency)
- Plotly-based visualizations: learning curve, importance, OOF distribution, residuals (optional dependency)
- Export/load with format_version=1 and metadata
- ~~Simulate (bootstrap prediction distributions)~~ — listed in error; this
  feature was never shipped (no `simulate` API has existed in any release).
  The CHANGELOG/code reconciliation was completed in
  [#177](https://github.com/nbx-liz/LizyML/issues/177); implementing the feature
  is tracked separately in [#194](https://github.com/nbx-liz/LizyML/issues/194).
- YAML/JSON config loading with pydantic validation
