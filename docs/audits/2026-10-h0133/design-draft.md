# H-0133 design draft (#357): make JSON-unsafe category values saveable

Working draft for the Codex design critique, before the Proposal. Base: develop `918c489`.

## Facts (measured or read)

- F1. LizyML's `CategoricalEncoder` hands LightGBM a pandas category column whose categories are the raw learned values (`categorical_encoder.py:213-214`, `set_categories(known)`). LightGBM stores them as `pandas_categorical` and JSON-encodes them in `model_to_string` (pickle, `save_model`). JSON-unsafe values fail with "Circular reference detected" (#338, H-0127).
- F2. LightGBM trains and predicts on the codes. Renaming a column's categories to their positions `0..k-1` gives identical predictions (max abs difference 0.0) and the booster then pickles and saves, for str, bytes, Decimal, Timestamp and date (`probes/stand_in_probe.out`, LightGBM 4.6.0, pandas 3.0.1).
- F3. Mixing forms is silently wrong. A positions booster given a raw-categories frame, and a raw booster given a positions frame, both predict without error and differ from the correct prediction (all values become missing; `stand_in_probe.out`). So the saved form must be recorded, and an older LizyML must not load the new form.
- F4. Persistence is joblib (pickle) of FitResult and RefitResult. The encoder state, including the raw values, is pickled, not JSON (`exporter.py:249-254`). `FORMAT_VERSION = 2`, and the loader supports {1, 2} and refuses others (`loader.py:35`, `:160-172`). BLUEPRINT §15.2: removing a field or changing its meaning requires a bump.
- F5. `export_code` writes `pipeline_state.json` and `config.json` (`declared_categories`) through `codegen/values.py`, which accepts only str, int, float, bool and numpy scalars of those (`values.py:21`, `:38-50`). Generated train.py and predict.py read category lists from JSON and match by value (`templates.py:181-229`, `:1320-1368`). Generated train.py feeds float64 codes to LightGBM, so its booster has an empty `pandas_categorical`. `export_code` writes `model.txt` from the runtime booster (`artifact_writer.py:89`), which today carries the raw values.
- F6. Parity gap: `Model.export` accepts what `json.dumps` with LightGBM's numpy default can write (including tuples, which come back as lists), while `export_code` refuses tuples. The divergence is pinned by `tests/test_persistence/test_h0120_records.py:300-315`.
- F7. The transformed frame does not reach users: SHAP returns importance numbers only (`shap_explainer.py:178-198`). Renaming changes the stored `booster.pandas_categorical` and the `pandas_categorical:` line in `model.txt` from names to positions.

## Draft decisions

- D1. Where: in `CategoricalEncoder.transform`. Matching against the learned raw values is unchanged (unseen detection, `unseen_policy`, NaN handling, `comparable_values`). After `set_categories(known)`, rename the categories to positions `0..k-1`. The codes are the same, so predictions are the same (F2). Raw values stay in the encoder state.
- D2 (fork). Scope:
  - A. every categorical column, always;
  - B. only columns with at least one value LightGBM cannot write.
  Draft recommendation: A. It is one path, every existing categorical test runs it, there is no activation condition (no DC6 risk), and there is no collision question. Cost: the booster's `pandas_categorical` and `model.txt` show positions instead of names for every new model.
- D3. Compatibility: the encoder state records its form (for example `category_form: "positions"`). A state without the field is the raw form and keeps today's behaviour, so v2 artifacts load and predict exactly as today. `FORMAT_VERSION` becomes 3, so an older LizyML refuses new artifacts (F3). Nothing to migrate: v2 artifacts cannot hold unsafe values (export failed), and they stay raw.
- D4 (fork). The value set. A closed set V is accepted by both `Model.export` and `export_code`, pinned by one test over one population:
  - the plain set: str, int, float, bool, and numpy scalars of those;
  - bytes, Decimal, `pd.Timestamp` (naive and tz-aware), `datetime.datetime`, `datetime.date`, `np.datetime64`, `pd.Timedelta`, `datetime.timedelta`, complex;
  - tuple and frozenset whose elements are in V.
  Anything else stays refused by both, with SERIALIZATION_FAILED naming the column, value and type, even when pickle could hold it. Alternative: let `Model.export` accept any picklable hashable value. That breaks the parity goal of #357.
- D5. Generated code: values outside the plain set are written as tagged JSON objects, `{"$lizyml": "<type>", ...}`. Examples: bytes as base64; Decimal as its string; Timestamp as ISO plus unit and tz; tuple and frozenset as lists of tagged items. A fixed decoder in train.py and predict.py rebuilds objects of the same type and value, so set membership and equality match the runtime. Tags cannot collide with category values: a category is hashable, so never a dict, and plain values stay JSON scalars. Generated train.py's `fit_pipeline` writes the same tags. Its booster is unchanged (it feeds codes).
- D6. #338: the fit warning and the export refusal stay only for values outside V. `np.longdouble` stays outside V, because JSON cannot keep it exactly.
- D7. Unchanged: `unseen_policy` (mode substitution uses the raw mode and then the positions), NaN and None, `declared_categories` semantics, the tuning identity (it hashes the input frame).

## Acceptance (draft)

1. For every type in V, `fit -> Model.export -> Model.load -> predict` equals the in-memory model, and the same holds through `export_code` with generated `predict.py`. This also covers mixed columns and tz-aware Timestamps.
2. One population pins that `Model.export` and `export_code` accept and refuse the same values.
3. A v2 artifact (an existing fixture) loads and predicts as before, and an artifact with format_version 3 is refused by the v2 loader (simulated).
4. `unseen_policy` behaves as before for every policy.
5. Negative controls:
   - skipping the rename makes export of an unsafe column fail;
   - dropping the form field makes the v2 fixture predict wrongly, and the test catches it;
   - a decoder that returns lists for tuples is caught.

## Firing rate

D2-B and D4 each add an activation or allow condition. These are measured over the full suite and the notebook tier before the Proposal: the category value types of every encoder fit (`probes/firing/`).
