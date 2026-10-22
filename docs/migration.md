# Version Migration Guide

This document covers breaking changes and migration steps for each LizyML
release from v0.5.0 onward.

---

## Unreleased

### A memory warning for a large `num_leaves`; `max_depth=0` is no limit; no `max_depth` ceiling in expansion (H-0140)

- **New `UserWarning` before training.** `Model.fit` and `Model.tune` warn when
  the `num_leaves` LightGBM receives is at least 16384 or the estimated extra
  memory is at least 1 GiB ([#388](https://github.com/nbx-liz/LizyML/issues/388)).
  Results do not change. A default-space tune can warn on its deepest trials
  (`max_depth` 11-12 with many rounds and folds), once per `tune()` call. To
  stop instead of training, turn the warning into an error, for example
  `warnings.filterwarnings("error", message="LightGBM will train with num_leaves")`.
- **`max_depth=0` with `auto_num_leaves` now derives like `-1`.** It derived 8
  leaves before; LightGBM treats `max_depth <= 0` as no limit. A new fit with
  `max_depth=0` gets 131072 leaves with the default ratio; set a positive
  `max_depth`, or `auto_num_leaves=False` with an explicit `num_leaves`, to keep
  a small tree. Saved models keep the parameters they were fitted with.
- **`parameter_bounds(task)["max_depth"]` has no `max`.** Boundary expansion may
  widen `max_depth` beyond 30 in a new study or a changed dimension; a recorded
  study keeps its recorded range. Code that reads `parameter_bounds` (for
  example a UI range editor) must accept an entry with only `min` or only `max`.

### `export_code` projects name every column by a string alias (H-0141)

A project written by `export_code` now names each column by `str(name)`
([#382](https://github.com/nbx-liz/LizyML/issues/382)): a column named `7` is
`"7"` in `config.json`, `pipeline_state.json` and the input headers the
scripts read. Before, a project with int-named columns could not predict from
the frame it was fitted on, or from any CSV. What to do:

- **CSV and parquet inputs** to the generated `train.py`, `predict.py` and
  `test_equivalence.py` must carry the aliases as headers (`7`, `7.5`,
  `True`, `None`); a blank, repeated or pandas-renamed header is refused. A
  typed DataFrame passed to `predict.predict` is renamed for you.
- **Names `export_code` now refuses** with `SERIALIZATION_FAILED`, before
  writing anything: NumPy scalar names other than `np.str_` and `np.float64`
  (they failed with a bare `TypeError` before), other non-admitted types, an
  empty name, and two columns or a non-string column and a Config role (target,
  time, group, excluded, categorical, block or group column) sharing one alias
  (`7` and `"7"`). Rename the column to a `str` and fit again.
- **Projects written before this release** carry no `column_name_policy`
  marker and keep their old behaviour, also when retrained. Write the project
  again from the model (or from the model `Model.load` reads) to get the
  aliases.

String-named projects keep their names, order, categories and model files;
both JSON files gain `column_name_policy: "str_alias_v1"`. The fitted model,
its Config and the saved artifact format are unchanged.

### Int-named categorical columns reach LightGBM by position (H-0142)

A categorical column whose name is an `int` was handed to LightGBM by name,
and LightGBM read the name as a column position
([#393](https://github.com/nbx-liz/LizyML/issues/393)): a column named `1`
made column 1 categorical, a column named `7` in a smaller frame made none.
The adapter now passes positions. Fit again any model with int-named
categorical columns whose name differed from its position; its predictions
change because the right column is now categorical. String-named columns,
and an int name that equals its position, train the same booster as before.
Saved models are unchanged.

### Date and time categories are matched by instant; timedelta64 years and months are refused at fit (H-0139)

The category key covers date and time values
([#379](https://github.com/nbx-liz/LizyML/issues/379)): two time-zone-aware
`datetime` / `Timestamp` values are one category when they are the same UTC
instant, whatever the time zone (UTC, `ZoneInfo`, pytz, a fixed offset) or fold;
naive values are one when their wall-clock values are exactly equal (sub-ns
`np.datetime64` units included); durations are one when their lengths are equal.
Naive and aware values stay apart, and so do a `date` and a midnight `datetime`.

- **Columns that work today do not change**, with these deliberate exceptions
  in newly fitted models: a `datetime` column holding both folds of one wall
  time in one zone (two instants Python's `==` called equal) now has two
  categories; sub-ns `np.datetime64` values are no longer coded as another
  value (pandas 2 replaced them with the mode, pandas 3 gave them the epoch's
  code); pytz's standard-time value and the same instant as a `ZoneInfo`
  `fold=1` value are now one category.
- **#379 is fixed.** On pandas 2, a column of pytz `Timestamp`s in one zone
  keeps every training value at predict, and `Model.export` / `export_code`
  save it (H-0136 refused it before).
- **New refusal at fit.** A categorical column holding a `np.timedelta64` in
  `Y` or `M` units (in a row, inside a tuple / frozenset, or as a declared
  category) raises `DATA_SCHEMA_INVALID` with context
  `{"column", "units", "reason": "calendar_timedelta64"}`; such a value has no
  fixed length. Convert it to days or to a string before fit. A unitless
  `np.datetime64` (which NumPy cannot print) is refused the same way, with
  `"reason": "unitless_datetime64"`. Either value arriving at predict is
  treated as unseen.
- **No format change.** Artifacts stay `format_version` 5 and
  `category_identity_version` stays `1`. Saved models (any format) predict
  with the lookup they were saved with; fit again to use the temporal key.

### Category values are matched by one structural key; artifacts are `format_version` 5 (H-0138)

Categorical columns whose values pandas confused
([#378](https://github.com/nbx-liz/LizyML/issues/378),
[#380](https://github.com/nbx-liz/LizyML/issues/380)) now fit and predict:
which values are one category is decided by
`lizyml.core.category_values.category_identity_key`. Numbers equal as numbers
(`0`, `-0.0`, `False`; `Decimal("1")` and `np.int8(1)`) are one category, a
string or bytes value and its NumPy twin are one, and tuples and frozensets
compare item by item.

- **Columns that work today do not change.** Where pandas' answer agrees with
  the key, the categories, their order, the mode, the codes and the trained
  model are the same as before. Only columns pandas could not cast or coded
  wrongly change: they are built by key and reach LightGBM as position
  stand-ins. Date and time values follow the same key since H-0139 (below).
- **Older LizyML cannot read v5 artifacts.** `Model.load` in v0.20.0 or earlier
  raises `DESERIALIZATION_FAILED` for `format_version` 5. An older version would
  match categories through pandas and give the fixed columns their old codes.
- **Artifacts saved before (`format_version` 1-4) predict as they did.** Their
  encoder states carry no identity marker, so a loaded model keeps pandas'
  lookup, and `export_code` on it generates code that keeps it too. Fit the
  model again to use the key.
- **Generated code.** `config.json` has a new key, `category_identity_version`
  (`1`, or `null` for a model loaded from a v1-4 artifact), and the generated
  `pipeline_state.json` carries `category_identity_version` and
  `identity_columns`. `predict.py` refuses an identity version it does not know.

## v0.20.0

### Models a saved copy would answer differently for are refused (H-0136)

`Model.export`, `lizyml.persistence.export` and `export_code` now run one
check before writing anything. Each saved category state is passed through
joblib and through the tagged JSON `export_code` writes. The model is refused
with `SERIALIZATION_FAILED` when the learned categories, as one batch in the
state's order, would get other codes or become unseen, or an unseen row the
mode replaces would get another code. The context has `serializer` and `kind`. `fit` warns about such a
column once the fit has finished.

- **Refused now, saved before:** a column holding `np.uint64(4)` beside
  `np.timedelta64(4, 'M')` (and a few other NumPy pairs whose equality is not
  transitive), when the JSON copy changes the answer for the categories in the
  order the state holds them. v0.19.0's `Model.export` saved such a model;
  artifacts saved then still load and predict, but exporting the loaded model
  again is refused. v0.19.0's `export_code` wrote a project whose `predict.py`
  failed for it.
- **Refused, never released as accepted:** a pytz `Timestamp` at the repeated
  hour of a daylight-saving change (pickle restores it with another `fold`).
  v0.19.0 refused every `Timestamp` category at `Model.export`.
- **What to do:** convert the column's values to `str` (or `int`, `float`,
  `bool`) before `fit`.
- **Not covered:** a value the fitted model itself treats as unseen, and two
  in-memory defects found on the way (#379, #380).

### `Model.load` reads only the files `checksums` lists (H-0135)

`metadata.json`'s `checksums` now decides what `Model.load` (and
`lizyml.persistence.load`) reads
([#373](https://github.com/nbx-liz/LizyML/issues/373),
[#374](https://github.com/nbx-liz/LizyML/issues/374)). Artifacts written by
LizyML load as before.

- **An `analysis_context.pkl` that `checksums` does not list is not read.**
  Before, exporting a model without a context into a directory that held an
  earlier export left the old `analysis_context.pkl` there, and the loaded
  model answered `residuals()` and the other diagnostic methods with the
  earlier model's data. Now that file is ignored (logged at INFO), and those
  methods raise `MODEL_NOT_FIT`. Exporting into such a directory also removes
  the earlier export's `analysis_context.pkl` and `tuning_history.json` first;
  other files in the directory are kept.
- **A malformed `checksums` is refused**, in every format: it must be exactly
  `{"algorithm": "sha256", "files": {<name>: <64 lowercase hex digits>}}` and
  list `fit_result.pkl` and `refit_model.pkl`. Before, an empty list skipped
  verification and a non-empty one raised `AttributeError`.
- **`checksums` is required from `format_version` 3.** Artifacts of format 1-2
  without it (written before v0.16.0) still load unverified. A format 3 or 4
  artifact whose `checksums` was removed by hand is refused.
- For the refusals above, `LizyMLError(DESERIALIZATION_FAILED)` carries `context["reason"]`:
  `checksums_malformed`, `checksums_algorithm`, `checksums_missing`,
  `required_file_unlisted`, `declared_file_missing` or `not_a_file` (a
  directory or other non-file under a payload's name).
- **`format_version` must be a JSON integer.** `true` and `2.0` used to pass as
  1 and 2; they are refused now, as is a `metadata.json` that is not an object.
  These two errors carry no `reason` key, like the other version and metadata
  errors.

### Saved artifacts are `format_version` 4 and keep the tuning history (H-0134)

A tuned model now keeps its trials, rounds and boundary report through
`Model.export` -> `Model.load`
([#315](https://github.com/nbx-liz/LizyML/issues/315)): `tuning_table()`,
`tuning_plot()` and `boundary_table()` on the loaded model return what they
returned before the export. The history is written to `tuning_history.json`
(strict JSON, listed in `checksums`) and declared in `metadata.json` under
`tuning.history`.

- **Older LizyML cannot read v4 artifacts.** `Model.load` in v0.19.0 or earlier
  raises `DESERIALIZATION_FAILED` for `format_version` 4, including for models
  that were never tuned. An older version would ignore the history file and
  show the trials as an empty table. Load v4 artifacts with v0.20.0 or later.
- **Artifacts saved before v0.20.0 (`format_version` 1-3) have no history.** On
  a tuned model loaded from one, the three methods now raise
  `LizyMLError(MODEL_NOT_FIT)` with `context["missing"] == "tuning_history"`,
  instead of returning an empty DataFrame (`(0, 0)`) or a figure with no
  traces. The tuned parameters are restored as before and `fit()` uses them.
  If the study's storage still exists, `restore_tuning(storage=...,
  study_name=...)` rebuilds the history. Exporting such a model again keeps it
  without a history.
- **`tune(resume=True)` after `Model.load` still needs the study's storage.**
  The saved history is a record for the reports; it is not a study to resume.
- **Size**: about 0.5 KB per trial for a 10-dimension search space. Every trial
  is kept.
- **`lizyml.persistence.load`** returns, for a tuned model,
  `metadata["tuning"]["history"]` as the decoded history (a `TuningHistory`),
  or `None` when the artifact has none.

### Category columns with JSON-unsafe values can be saved (H-0133)

Category columns can now hold values that JSON cannot represent -- `bytes`,
`Decimal`, dates, datetimes, `Timestamp`, timedeltas, `complex`, and tuples or
frozensets of these -- and the model can still be saved with `Model.export` and
`export_code` ([#357](https://github.com/nbx-liz/LizyML/issues/357)). To make
that possible, a column holding a value LightGBM cannot write is handed to
LightGBM with its categories renamed to their positions `0..k-1`; the codes, and
so the predictions, are unchanged.

- **Older LizyML cannot read these artifacts.** H-0133 raised `format_version`
  to 3 (and H-0134 to 4); `Model.load` in v0.19.0 or earlier refuses them with
  `DESERIALIZATION_FAILED`. This is deliberate: an older version would read a
  renamed column as raw categories and predict wrongly without an error.
- **v1 and v2 artifacts load and predict as before.** Nothing needs migrating:
  a v2 artifact could not hold the values that need renaming (its export failed).
- **Using the LightGBM booster directly.** For a renamed column only,
  `get_native_model()` returns a booster whose `pandas_categorical` holds the
  positions. Passing it a frame with the raw values of that column predicts
  wrongly; go through `Model.predict`, which applies the pipeline. A column
  whose values are all `str`, `int`, `float`, `bool` (or numpy scalars of
  these), or tuples of them, is not renamed.
- **`export_code` accepts the same feature-category values as `Model.export`**
  (since H-0136 through one shared check; see the H-0136 section above).
  `pipeline_state.json` and `config.json` write values other than `str`, `int`,
  `float` and `bool` as tagged JSON objects (`{"$lizyml": ..., ...}`) that the
  generated scripts decode; tuples are no longer refused. Target labels keep
  the `str` / `int` / `float` / `bool` set.
- **`declared_categories` in `metadata.json`** is written for every column the
  input held as `category` dtype (tagged when needed), instead of the whole
  record being left out when one value was not plain.
- **`lizyml.persistence.export(categories_checked=True)` is deprecated.** The
  argument is ignored and warns: the public export always checks the category
  values first. It will be removed in v1.0.
- **Still refused, before anything is written:** a category value outside the
  saveable set -- a NaN-like value inside a tuple, `np.longdouble`, a subclass of
  a saveable type, a user-defined type, a tuple nested more than 8 deep, or a
  timezone that cannot be restored equal. `fit` warns about such a column.

---

## v0.19.0

Tuning with `resume=True`, `storage` or `study_name` changes (H-0122). Some
configs and search spaces that v0.18.0 silently read as something else are
now refused: smart-parameter collisions and smart dimensions (H-0121), the
search-space format and `step` (H-0123, H-0124), regression stratification
(H-0120), and a few objective and metric values (H-0125). Each has a section
below.
A `tune()` that starts a new in-memory study, and an in-memory `tune()` ->
`tune(resume=True)` that keeps the data and the config unchanged between rounds,
need no edits; their results can still change (see below). Saved artifacts keep
`format_version` 2.

### Studies created before v0.19.0

A study created before v0.19.0 cannot be continued. It carries no LizyML record
of the objective its trials observe, so `tune(resume=True)` raises
`CONFIG_INVALID` for it. Start again under a new `study_name`; the old study's
trials stay readable through Optuna. Finish paused jobs (for example a paused
LizyStudio job) before upgrading, since their studies cannot be resumed
afterwards.

### Calls that now raise `LizyMLError(CONFIG_INVALID)`

| You wrote | Write instead |
|-----------|---------------|
| `tune(storage=..., study_name=...)` to re-attach to a stored study after a crash | `tune(resume=True, storage=..., study_name=...)`: it reattaches, and creates the study when the storage lacks it, so one call both starts and resumes a job |
| `tune(resume=False, ...)` with a `study_name` the storage already holds | `resume=True` to continue it, or a new `study_name` for a new study |
| `study_name` without `storage` | Drop `study_name` for an in-memory study, or pass `storage` as well |
| `tune(resume=True, storage=..., study_name=...)` on a Model that holds a different study | Omit `storage` and `study_name` to continue the held study, or use a new `Model` to reattach to another study |
| Different data, config (outside the search space), split or first metric, then `tune(resume=True)` | A new study (`resume=False`, or a new `study_name`); the error's context names what changed |
| An edited `tuning.optuna.space` (or `model._cfg`), then `tune(resume=True)` | `tune(resume=True, space={...})` |
| An edited `tuning.optuna.space_mode`, then `tune(resume=True)` | A new study |
| `space=` that changes a dimension's distribution kind, `log`, choices or category | A new study; only ranges (`low` / `high`) can change within a study |
| `space=` on the first `tune()` of a study | Declare round 1's space in `tuning.optuna.space` |
| An `n_trials`, `expand_boundary` or `space=` other than the recorded one when finishing a round a crash interrupted | Omit them (or pass the recorded values); the round finishes with its recorded budget |

### Results that change

- **A resumed round no longer re-evaluates the previous best.** Before, each
  resumed round started by enqueueing the previous best parameters and training
  them again, so one of its `n_trials` repeated a known point. Every trial is now
  a new TPE sample, so with the same seed a resumed round's trials, and possibly
  its best, differ from v0.18.0. `best_params` and `best_score` are still the
  study's best trial, which may come from an earlier round.
- A new study starts at round 1; `resume=False` no longer keeps counting the
  previous study's rounds. A reattached Model continues the study's round
  numbers.
- A round that ends with no COMPLETE trial of its own raises
  `TUNING_FAILED`, is recorded `failed`, and is left out of
  `TuningResult.rounds` (H-0128, below), so `rounds` can skip round numbers
  and can start at round 2 with `best_score_before=None`. Code that assumes
  `rounds[0].round == 1`, or that `len(rounds)` counts every round, needs the
  round numbers instead.
- `RoundSummary.n_trials` is the number of trials the round ran, and
  `TuneProgressInfo.current_trial` counts within the round, continuing at K+1
  when a round with K finished trials is resumed.

### Changing the search space on resume: `tune(space=...)`

`tuning.optuna.space` declares a study's first round only. Change the range of
a later round with `space=`, in the same form, resolved by the study's
`space_mode`:

```python
model.tune()  # round 1: tuning.optuna.space
model.tune(
    resume=True,
    space={"learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True}},
)
```

With `space_mode: merge`, `space=` is laid over the provider's default
dimensions: a default dimension you overrode in the config but leave out of
`space=` returns to its default declaration, and a dimension only the config
declared (not a provider default) is no longer searched unless `space=`
declares it again. With `replace`, `space=` is the whole space and a
dimension left out is no longer searched. A dimension that is no longer
searched trains on its config value in later trials. Trials of the old range stay in the
study, and the best of all trials is still the result.

### Continuing after a crash

```python
model = Model(config, data=df)
model.tune(resume=True, storage="sqlite:///optuna.db", study_name="job-1")
```

Run the same call again after a crash. A round the crash interrupted is
finished with its recorded space and remaining budget; a closed round is
followed by a new one. If the process died after the round closed but before
the result was received, `model.restore_tuning("sqlite:///optuna.db",
"job-1")` returns that result without adding trials.
`lizyml.tuning.read_study_lineage(storage, study_name)` reports which case a
stored study is in without changing it.

### Third-party providers

`EstimatorProvider.default_params(task)` is a new, optional method: the
defaults the provider lays under the user's parameters. They enter the
objective identity, so a changed default refuses a resume. A provider without
the method keeps working, and its defaults are not checked.

`EstimatorProvider.smart_param_owners()` is a new, required method (H-0121):
for each smart parameter that is read only while another one is true, that
owner. `tune` calls it, so a provider without it fails there; one without
smart parameters returns `{}`.

`EstimatorProvider.smart_value_rules()` is a new, required method (H-0121
amendment 2): a `SmartValueRule` for every name `smart_param_names()` returns,
stating the values that smart parameter can hold. `tune` checks every
`category: smart` dimension against it before the study, and `restore_tuning`
and `Model.load` check recorded values against it, so a provider without it
fails there. A provider without smart parameters returns `{}`.

`BaseEstimatorAdapter.unsaveable_categories(pipeline_state)` is a new, optional
method (H-0129): the category values the estimator cannot write when its model
is saved. The default, `[]`, is the old behaviour for third-party adapters:
`Model.export` no longer checks their models against LightGBM's JSON limit.

---

### Smart parameters and smart dimensions (H-0121)

A value for a parameter a smart parameter manages is now refused, in any
LightGBM spelling, by `fit` and `tune` before training
([#280](https://github.com/nbx-liz/LizyML/issues/280)). It used to be
overwritten by the smart value, or sent to LightGBM next to it. The literal
names `num_leaves`, `min_data_in_leaf` and `min_data_in_bin` were refused when
the config was built; they are now refused at `fit` / `tune`.

| You wrote (in `model.params`, `fit(params=)` or a tuned `best_model_params`) | Write instead |
|-----------|---------------|
| `num_leaves` or an alias such as `max_leaves`, with `auto_num_leaves` on (the default) | `model.auto_num_leaves: false` to set the leaves yourself |
| `min_data_in_leaf` (any alias), with `min_data_in_leaf_ratio` set (the default is `0.01`) | `model.min_data_in_leaf_ratio: null` |
| `min_data_in_bin` (any alias), with `min_data_in_bin_ratio` set (the default is `0.01`) | `model.min_data_in_bin_ratio: null` |
| `feature_contri` or `feature_pre_filter`, with `feature_weights` set | One of the two |
| `scale_pos_weight` on a binary task, with `balanced` on (`null`, the default, is on for binary) | `model.balanced: false` |

Search dimensions with `category: smart`
([#299](https://github.com/nbx-liz/LizyML/issues/299),
[#341](https://github.com/nbx-liz/LizyML/issues/341),
[#347](https://github.com/nbx-liz/LizyML/issues/347)), refused before the study:

| You wrote | Write instead |
|-----------|---------------|
| A smart dimension whose name is not a smart parameter, such as `num_leaves` | `category: model` for a LightGBM parameter. If a smart parameter manages it, turn that one off as well: for `num_leaves`, `model.auto_num_leaves: false` (the first table) |
| A `num_leaves_ratio` dimension with `auto_num_leaves: false` and no `auto_num_leaves` dimension that can be true | Remove it, or turn `auto_num_leaves` on |
| `model.params`, the round's fixed parameters or a `category: model` dimension that one of the round's smart dimensions could manage (for example `max_leaves` while the space holds `auto_num_leaves: [true]`) | Drop the parameter or the dimension |
| A smart dimension value of the wrong type or range: `"false"`, `0` / `1` for a bool, a bool for a number, NaN, a ratio outside its range, `balanced: true` on regression | The values listed in docs/config-reference.md |

- The default space's `num_leaves_ratio`, when `auto_num_leaves` cannot be
  true, is dropped from the tune instead of searched to no effect (logged at
  INFO, `event='tune.drop_dims'`).
- The boundary expansion of `num_leaves_ratio` stops at `1.0` instead of `2.0`.

### Search space format and `step` (H-0123, H-0124)

`tuning.optuna.space` and `tune(space=...)` are checked by one rule before a
study is created ([#344](https://github.com/nbx-liz/LizyML/issues/344)); the
refusal's `context` names the dimension, the key and a `reason`. These used to
be read as a different search:

| You wrote | Write instead |
|-----------|---------------|
| `log: "false"` (a string) | `log: false` |
| A key outside the format (a misspelt `lo`, `low` / `high` on a categorical dimension) | The keys in docs/config-reference.md |
| An `int` bound that is not a whole number, such as `1.7` | A whole number (`"16.0"` and `"1e2"` are accepted as 16 and 100) |
| `choices` that is a string, set, dict, range or array | A non-empty list |
| A `category` other than `model`, `smart` or `training` | One of the three |
| A bound that is NaN, infinite, a bool, `None` or a non-number string | A finite number |
| A `step` whose `low` or `high` is not a multiple of it ([#350](https://github.com/nbx-liz/LizyML/issues/350)) | Bounds on the grid; the message suggests one |
| A `step` of zero or below | A positive step (a whole number of at least 1 on an `int` dimension) |

A `step` is now honoured: the trials sample the multiples of `step` between
`low` and `high`. v0.18.0 ignored it, so a stepped dimension's trials, and the
best parameters, differ from v0.18.0 with the same seed.

### Regression splits and generated code (H-0120)

- A regression config with `split.method: stratified_kfold` /
  `stratified_group_kfold`, `split.groups.stratify: true` or an explicit
  `inner_valid` with `stratify: true` raises `CONFIG_INVALID` before training.
  Use `kfold`, `group_kfold` or `stratify: false`.
- The `train.py` of `export_code` reproduces the refit again
  ([#301](https://github.com/nbx-liz/LizyML/issues/301),
  [#304](https://github.com/nbx-liz/LizyML/issues/304)). An existing generated
  project retrains as before; export again to get the new `train.py`.

---

### Binary objectives, native metric params and tune values (H-0125)

| You wrote | Write instead |
|-----------|---------------|
| `objective: cross_entropy_lambda` on a binary task (any spelling of the key) | `binary` or `cross_entropy`. Its output is LightGBM's lambda, not a probability ([#307](https://github.com/nbx-liz/LizyML/issues/307)); the metric `cross_entropy_lambda` stays valid |
| A parameter on a LightGBM-native metric in `model.params["metric"]`, e.g. `{"huber": {"delta": 9.0}}` | The bare name (`"huber"`), LightGBM's own parameter (`alpha` for `huber`, shared with the objective), or the parameterised LizyML metric in `evaluation.metrics` ([#313](https://github.com/nbx-liz/LizyML/issues/313)) |
| An `objective` / `metric` search dimension that is `int` / `float`, or has a choice fit would refuse | A categorical dimension whose every choice fit accepts; `Model.tune` now refuses it before the study instead of failing those trials |

- `Model.load` refuses (`DESERIALIZATION_FAILED`) an artifact whose boosters
  were trained with `cross_entropy_lambda`; fit again with `binary` or
  `cross_entropy` and export again. Other artifacts load as before.
- An artifact that recorded a parameter on a native metric still loads and
  predicts; a re-fit or `export_code` from it raises `CONFIG_INVALID`. Remove
  the parameter from the config.
- The learning curve (`FitResult.history`, `plot_learning_curve`) reports
  LightGBM's metrics: its `huber` (threshold `alpha`) and `mape`
  (`|y-p| / max(1,|y|)`, a ratio) differ from `FitResult.metrics` entries of
  the same name ([#354](https://github.com/nbx-liz/LizyML/issues/354)).

### Minimum dependency versions (H-0126)

| Installed | Needed |
|-----------|--------|
| `pydantic` 2.0 | 2.1 or later (2.0 could not import LizyML) |
| `lightgbm` 4.0-4.3 | 4.4 or later (4.0-4.3 raise on `early_stopping_round=0` without a validation set) |
| `optuna` 3.0, with the `tuning` extra | 3.1 or later (3.0 conflicts with the `calibration` extra's `scipy>=1.10`) |

`pip install -U lizyml` raises them. The generated project's `requirements.txt`
asks for `lightgbm>=4.4` too.

### Failed resumed rounds and LightGBM trial errors (H-0128)

- `tune(resume=True)` raises `TUNING_FAILED` when none of the round's own
  trials completed, even if earlier rounds did; before, it returned the
  previous best. Code that resumes in a loop should catch it and call
  `tune(resume=True, space=...)` with a corrected space. The Model keeps its
  previous tuning result.
- A LightGBM error inside a trial (an invalid parameter value in a range) is
  a failed trial: the round runs its budget instead of stopping at the first
  trial with `Optuna study failed`.
- An open round with no COMPLETE trial accepts a different `space=`: it is
  closed `failed` (`UserWarning`) and a new round starts. Before, that call
  raised `CONFIG_INVALID`.
- `restore_tuning` on a study whose last round failed still raises
  `TUNING_FAILED`; it does not restore earlier rounds.

### Category values (H-0127)

- A column categorical at fit with integer categories now predicts from
  `float16` and `longdouble` values instead of raising a raw pandas error. A
  `longdouble` value float64 changes (for example `3 + 3 eps`) is unseen and
  follows `features.unseen_policy`; it never takes the neighbouring category.
- `Model.export` of a model whose categorical column holds a value LightGBM
  cannot write (`bytes`, `Decimal`, dates and timestamps, `Timedelta`,
  `complex`, `frozenset`, a `tuple` holding one) still raises
  `SERIALIZATION_FAILED`, now before anything is written and naming the
  column, value and type. `fit` emits a `UserWarning` for such a column; a
  test that turns warnings into errors stops there. Convert the values to
  `str` before `fit` to be able to save the model.

## v0.18.0

Most configs need no edits. Go through the sections below and act on the items
that apply; `CHANGELOG.md` (`[0.18.0]`) has the details and the measurements
behind each one. Saved artifacts keep `format_version` 2 and load and predict as
before.

### Configs that now raise `LizyMLError(CONFIG_INVALID)`

| You wrote | Write instead |
|-----------|---------------|
| `feature_weights` in `model.params` | `feature_contri` (`feature_weights` is not a LightGBM parameter and never had an effect) |
| One parameter under two spellings, e.g. `learning_rate` and `eta`, even with equal values | One spelling |
| A name LightGBM does not define | The LightGBM name |
| A `set`, an arbitrary object, `numpy.timedelta64`, an `ndarray` subclass or a pure path as a value | `None`, `bool`, `int`, `float`, `str`, `pathlib.Path`, a numpy scalar, a list / tuple / 1-D array of those, or a dict with str keys holding those |
| `seed` or `early_stopping_round` (any spelling) in `model.params` or `fit(params=...)` | `training.seed`, `training.early_stopping.rounds` |
| A native parameter an active smart parameter resolves, e.g. `num_leaves` with `auto_num_leaves` on | Turn the smart parameter off to set the native one |
| `tuning.optuna.params.direction` that contradicts the metric | Omit it, or set `null` (it is taken from the metric) |
| A `category: model` search dimension an active smart parameter overwrites, two dimensions for one parameter, or a `category: training` dimension other than `early_stopping_rounds` / `validation_ratio` | Remove it, or turn the smart parameter off |
| `calibration.params` for `platt` / `beta` with names other than `x0`, `method`, `bounds`, `tol`, `options` (and `target_smoothing` for `platt`), e.g. `C`, or a value those names do not accept (an unknown `method` or solver option, an `x0` of the wrong length, malformed `bounds`, a non-positive `tol`, a non-boolean `target_smoothing`) | Remove them, or fix the value (before, all of these were ignored) |
| A dict or list `objective` | A string |
| `embargo: true` in a `purged_time_series` split | An integer in `purge_gap` |

### Code that needs changes

- `ErrorCode.DATA_FINGERPRINT_MISMATCH` is removed; a reference to it raises
  `AttributeError`. Nothing ever raised it.
- `validate_no_target_leakage` and `validate_time_series_order` raise
  `LizyMLError(DATA_SCHEMA_INVALID)` when the column they are named for is
  missing, and `validate_no_target_leakage` does the same for a column it cannot
  compare, whatever `raise_on_violation` is. Before, a missing column got `[]`
  and an uncomparable column was skipped.
- Error codes you may now receive: `CALIBRATION_FAILED` is new (a Platt or Beta
  optimiser did not converge). `INCOMPATIBLE_COLUMNS` (a column numeric at fit
  arrives non-numeric at `predict`) and `METRIC_REQUIRES_PROBA` (a probability
  metric received values that are not probabilities) were declared before but
  never raised.
- A `config_version` other than `1` raises `CONFIG_VERSION_UNSUPPORTED` on every
  path into `Model`, including `model_validate`, `model_construct`, assignment
  and `LIZYML__config_version`; before, several of those paths accepted it.
- `PredictionResult.warnings` lists categories that were not seen at fit.
- A third-party `EstimatorProvider` must implement `accepted_model_param_names()`,
  `smart_param_names()`, `smart_managed_param_names()` and
  `canonical_param_names()`, and accept `build_pipeline_factory(unseen_policy=...)`.
- `PurgedTimeSeriesConfig` and `PurgedTimeSeriesSplitter` no longer have an
  `embargo` attribute, and `model_dump()` has no `embargo` key: read
  `purge_gap`, which includes it. `PurgedTimeSeriesSplitter(embargo=...)` is
  deprecated like the config key.

### Deprecated: `embargo` in `purged_time_series`

`embargo`, `embargo_pct` and `gap` still work until v1.0, with a
`DeprecationWarning`, and are **added** to `purge_gap`. The folds do not change:

```yaml
split:
  method: purged_time_series
  purge_gap: 5
  embargo: 2      # before
```

```yaml
split:
  method: purged_time_series
  purge_gap: 7    # after: the same folds
```

### Results that change without a config edit

Re-run, and re-check metrics, if any of these applies to you:

- You set `model.feature_weights`: it now takes effect (it is sent to LightGBM
  as `feature_contri`). Before, LightGBM discarded it.
- You passed `fit(params=...)`: it is now applied. Before, the model trained on
  the config values.
- You wrote a parameter under a LightGBM alias (`eta`, `max_leaves`, ...) in
  `model.params` or `calibration.params`: it now takes effect. Before, the
  default or the calibrator's default won.
- You use `platt` calibration: it is fitted as Platt defined it (no L2 penalty,
  smoothed targets). The measured change is small.
- Multiclass with `balanced` active (the default `balanced: null` included): the
  final refit trains with the same class weights as the CV folds, so predictions
  and exports change.
- Early stopping on a metric LizyML computes itself (binary `accuracy`, `f1`,
  `brier`, `ece`; multiclass `brier`): the metric now sees real probabilities,
  so a different iteration may be selected.
- A partial `tuning.optuna.space`: the other default dimensions are kept now
  (`space_mode: merge`). Set `space_mode: replace` for the previous behaviour.
- `tune()` without an explicit `direction`: the direction now follows the
  metric. A study tuned the wrong way needs a new study name.
- SHAP importance with `features.auto_categorical: false`: each fold model is
  explained with its own pipeline.

### Artifacts and exported code

- `importance(kind="shap")` on a model exported before v0.18.0 raises
  `MODEL_NOT_FIT`; call `fit()` again to record the per-fold pipeline states.
- Re-fitting a restored tuning result that names one parameter twice raises
  `CONFIG_INVALID`.
- Regenerate an `export_code()` project if you exported after setting `metrics`
  / `metric_types`, or after a `tune()` that changed the early-stopping
  patience; the old project trains a different model.

---

## v0.15.0

### `LGBMConfig.params["objective"]` is now respected when task-compatible

**Impact:** Tuning / training with a non-default LightGBM `objective` (H-0079).

Before v0.15.0 a user/Optuna-supplied `objective` was silently stripped and
replaced with the task default. From v0.15.0:

- A **same-task** objective string flows through to `lgb.train` — e.g.
  `objective: "fair"` on a regression task now actually trains with Fair loss.
  Re-running `tune()` over `default_space` may therefore yield different
  `best_params` / metrics than pre-0.15 runs.
- A **cross-task** objective now raises `LizyMLError(CONFIG_INVALID)` instead of
  being silently demoted (contract unchanged — still rejected).
- A **callable** objective is rejected with `CONFIG_INVALID` (unsupported).

No config edits are required; review any explicit non-default `objective`.

---

## v0.10.0

### Persistence `format_version` bumped to 2

**Impact:** Saved `model.lizyml` artifacts (H-0070).

Auto-encoding of non-numeric classification targets added a `TargetEncoder` to
the artifact, bumping `FORMAT_VERSION` from 1 to 2. `Model.load()` accepts
**both** versions: v1 artifacts are migrated in memory by injecting a no-op
`TargetEncoder`, so existing saved models keep loading with no user action. New
saves are written as v2.

---

## v0.8.x

### ECE formula corrected

**Impact:** Binary calibration metrics only.

The Expected Calibration Error (ECE) computation was updated to use
`fraction_of_positives` (the true positive rate per bin) instead of the
old formula that was using `mean_predicted_value` for the accuracy term.
Existing saved metrics will show numerically different ECE values after
recomputing with v0.8.x.

**Action:** Re-run `fit()` to get updated ECE values. No config changes required.

---

### `confusion_matrix_table` excludes NaN rows

**Impact:** Binary and multiclass tasks with missing target values.

Rows where the target is `NaN` are now excluded from confusion matrix
computation. Previously they were included and could produce incorrect
totals. The change aligns `confusion_matrix()` output with `evaluate()`
which already excluded NaN rows.

**Action:** No config change required. If you were relying on the old row
count for validation, update your assertions.

---

### Calibrated metrics now include `oof_per_fold`

**Impact:** `evaluate()` output structure for binary tasks with calibration.

The `"calibrated"` key now contains `oof_per_fold` in addition to `oof`:

```python
# v0.8.x and later
result["calibrated"] == {
    "oof": {"roc_auc": ..., ...},
    "oof_per_fold": [{"roc_auc": ...}, ...],
}
```

Code that previously assumed `"calibrated"` had only one key will still
work — the additional key is additive.

---

### `RefitTrainer` pipeline boundary change

**Impact:** Prediction pipeline behaviour.

The pipeline `fit` boundary in `RefitTrainer` now matches the CVTrainer
boundary exactly: the pipeline is fitted on the full training set `X_train`
with no inner validation split. This ensures that feature statistics (e.g.
target encodings) use all available data uniformly.

**Action:** No config change required. Predictions may differ slightly from
v0.7.x artifacts when using target encoding transformers.

---

### Inner validation raises on empty training split

**Impact:** Configurations with very small datasets or high `validation_ratio`.

`InnerValid` now raises `ValueError` (with a descriptive message) instead of
silently training on zero rows when the inner training split is empty.

**Action:** Reduce `early_stopping.validation_ratio` or disable early stopping
for small datasets.

---

## v0.8.0

### Codegen feval support (H-0066)

`export_code()` now includes custom `feval` metric functions in the generated
`train.py` when the config specifies custom evaluation metrics. The generated
code is self-contained and no longer requires LizyML to be installed.

**Action:** Re-export with `export_code()` to get feval-aware generated code.

---

### New estimator implementation guide

A step-by-step guide for adding new estimator adapters is available at
`docs/add-estimator-guide.md`. No API changes.

---

## v0.7.x

### Tune-fit identity fixes (v0.7.2–v0.7.3)

**Impact:** Any workflow using `tune()` followed by `fit()`.

Before v0.7.2, `tune()` and `fit()` used different data preparation
paths, causing inconsistent splits and preprocessing. The best params from
`tune()` were also not always forwarded correctly.

After v0.7.3:
- Both use the unified `_build_train_components` path.
- `TuningResult.best_params` is automatically applied on the next `fit()`.
- Split seed is consistent between tuning and fitting.

**Action:** No code changes required. Update to v0.7.3+ to get correct
tune-then-fit behaviour.

---

### OOF coverage (H-0057, v0.7.x)

`evaluate()` now returns `oof_coverage` under `raw`:

```python
result["raw"]["oof_coverage"]  # float in [0, 1]
```

OOF metrics are computed only on covered rows (rows that appeared in a
validation fold). Previously all rows were included, which could bias
metrics for time-series splitters where early rows are never held out.

`evaluate_table()` exposes coverage via `df.attrs["oof_coverage"]`.

**Action:** No config change required. Metrics may differ numerically from
v0.6.x when coverage < 1.0 (time-series or purged splits).

---

### Outer split reuse for calibration (H-0058, v0.7.x)

Calibration now reuses the outer CV splits instead of creating a separate
set of calibration splits. This makes calibration deterministic and removes
the `calibration.n_splits` parameter.

```yaml
# v0.6.x (still accepted but deprecated)
calibration:
  method: isotonic
  n_splits: 5   # deprecated — emits UserWarning

# v0.7.x and later
calibration:
  method: isotonic
  # n_splits is no longer needed
```

`build_calibration_splitter()` is deprecated and emits `DeprecationWarning`.

**Action:** Remove `calibration.n_splits` from your config. If you call
`build_calibration_splitter()` directly, migrate to passing the outer splits
from `FitResult.splits.outer`.

---

## v0.6.0

### Blocked Group KFold (H-0060)

A new `BlockedGroupKFoldConfig` split type was added for 2-axis
cross-validation (time blocks × group folds):

```yaml
split:
  type: blocked_group_kfold
  blocks:
    col: date
    cutoffs: ["2023-01-01", "2023-07-01"]
    mode: expanding   # or "sliding"
  groups:
    col: customer_id
    n_splits: 5
    stratify: true
```

**Action:** No breaking changes. Existing configs using `time_series` or
`kfold` are unaffected.

---

## v0.5.0

### EstimatorProvider and 5-layer architecture (H-0051/H-0052/H-0053)

v0.5.0 introduced a major internal refactor to a 5-layer category
architecture. Public API is unchanged, but several internal modules were
reorganized:

- `lizyml/estimators/lgbm.py` → `lizyml/estimators/lgbm/` sub-package
  (`adapter.py`, `smart_params.py`, `defaults.py`, `provider.py`).
- `EstimatorProvider` protocol introduced (`lizyml/estimators/provider.py`).
- Default search space and fixed params moved from `tuning/search_space.py`
  to `estimators/lgbm/defaults.py`.
- `DataFingerprint` moved from a utility module to
  `lizyml/core/types/artifacts.py`.

**Action:** If you import from internal LizyML modules (not recommended),
update your import paths. All public API imports via `from lizyml import Model`
are unaffected.

---

## Checking Your Version

```python
import lizyml
print(lizyml.__version__)
```

Or via uv:

```bash
uv run python -c "import lizyml; print(lizyml.__version__)"
```
