# H-0138: カテゴリの値の同一性を決めている箇所（develop `b297de4`）

H-0138（#378・#380）の設計の前に、カテゴリの特徴量の列で「どの値が同じか」を決めている箇所を、実行時のライブラリと生成コードの両方から洗い出したもの。読み取りのみの調査（2026-10-21）。Codex の設計批評は run `v021-category-critique`（`b39608d`）にある。

## 実行時（`lizyml/`）

- `core/category_values.py`
  - `:93-95` `_is`: 型の完全一致（`type(v) is T`）で `ADMITTED_TYPES`（`:64-80`）と照合する。
  - `:147-162`: 自分自身と等しくない値（NaN、Decimal NaN、complex NaN、NaT）と、戻せないタイムゾーンを拒否する。
  - `encode` `:222-223`: NumPy の普通のスカラーを `.item()` で Python の値にする。`np.bytes_` → `bytes`（`:224`）、NumPy の complex → `complex`（`:228`）。
  - NumPy と pandas の時刻の型は保つ: `np.datetime64`・`np.timedelta64` は単位と数（`:248-256`）、`pd.Timestamp` は単位と tz（`:231-237`）、`pd.Timedelta` は単位（`:257-262`）。
  - `frozenset` の要素は `json.dumps` で並べる（`:274`）。
  - `decode` `:439` は `_encode` が書く形だけを受け付ける（`!=` で比べる）。
- `data/dataframe_builder.py:104-111` `_apply_categorical`: `astype("category")`。pandas が値を factorize（hash と ==）し、カテゴリを並べる（`[2,'a',1]` は `[1,2,'a']`）。fit のときだけで、predict では行わない。
- `core/_codegen_inputs.py:112-115` `input_categories`: `list(df[col].cat.categories)`（宣言の順）。
- `features/pipelines_native.py:67-71`: `.cat` か文字列の dtype の列を選ぶ。`:156-166` `export_state_json` は codec の `encode` を通す。
- `features/encoders/categorical_encoder.py`
  - `:120`: `cat.categories` の順を保つ。pandas Index を回すと、int32 の Index は Python の `int` に、datetime64 の Index は `Timestamp` になる。object の Index は `np.int32` をそのまま持つ。
  - `:122`: それ以外は `sorted(dropna().unique().tolist(), key=str)`。
  - `:125-126`: 最頻値は `series.mode().iloc[0]`（pandas の数え方。同数はカテゴリの順か並べた順）。
  - `:133`: `pd.CategoricalDtype(cats).categories`（もう一度正規化する）を `needs_stand_in` に渡す。
  - `:188`・`:22-72` `comparable_values`: float16 と longdouble を float64 で比べる。
  - `:191-194`: unseen の判定は Python の `set` の差（hash と ==）。
  - `:213`: `isin`（pandas の hashtable）。
  - `:220`: 最頻値を object の Series で埋める。
  - `:233-234`: `astype("category").cat.set_categories(known)`（pandas Index の `get_indexer`）。
  - `:237`: 置き換えの列は `rename_categories(range(k))`。
- `features/encoders/state_changes.py:68-80`: 読み込み直したエンコーダの符号（上の規則）。`:90-93` は `set` を作る。`:83-87` は最頻値を置き換えの符号で判定する。
- `persistence/_category_preflight.py:46-50`: joblib の往復。`:53-67`: codec と JSON の往復。`:160-184` `declared_change`: `astype(CategoricalDtype(...)).cat.codes`。
- `persistence/exporter.py:73-101`: カテゴリと最頻値の `refusal`。`:526-529`: 宣言したカテゴリを encode する。`:531`: `json.dumps(default=str)`（V の検査の後でだけ走る str への逃げ道）。`persistence/loader.py:131-173`: 宣言したカテゴリを decode する。
- `estimators/lgbm/category_json.py:58-75` `is_writable` / `needs_stand_in`: `json.dumps(default=_numpy_default)`。`:78-88` `stand_in_columns`。`:99` `CategoricalDtype(known).categories`。`adapter.py:158-170` は置き換えの列を飛ばす。
- LightGBM 自身（`lightgbm/basic.py`）
  - `:848`: `pandas_categorical = list(cat.categories)` を持つ。
  - `:853-854`: predict（`adapter.py:331` から）で `list(cats) != list(stored)` を Python の list の `==` で比べ、`set_categories(stored)` を呼ぶ。
  - `:879`: model.txt と pickle に `json.dumps(default=_json_default_with_numpy)` で書く（3 つ目の保存処理）。
- `core/_model_predict.py:54-57`: `load_state(refit_result.pipeline_state)` の後で `transform_with_warnings`。`features/column_check.py:67`: カテゴリの列は dtype の検査を飛ばす。
- 近いが別の箇所: `core/_tuning_identity.py:137-150` は、カテゴリを `[type_name, str(v)]` で指紋にする。

## 生成コード（`codegen/templates.py`）

- `:8-216`: codec の写し。`_encode_value`（`:90-133`）と `_decode_value`（`:205-215`）。
- train.py
  - `:278-295` `cast_categoricals`: データの builder と同じ。
  - `:298-344` `fit_pipeline`: エンコーダの fit と同じ（`:318-326`）。
  - `:356-394`: `comparable_values` と同じ。
  - `:397-444` `encode_column`: transform と同じ。NaN を含む float の符号を返し、置き換えはしない。
  - `:482-497` `apply_declared_categories`: decode してから `CategoricalDtype`。
  - `:621-646`: LightGBM は数の符号を受け取る。
- predict.py
  - `:1482-1493`: 状態を decode する。
  - `:1546-1593`: `encode_column` と同じ。
  - `:1675`: NumPy の配列を渡すので、LightGBM の `pandas_categorical` の照合を通らない。
- `codegen/artifact_writer.py:14-50` と `codegen/values.py:66-92`: `category_values`（`refusal` と `encode`）を通す。`codegen/config_writer.py:106-110`: 宣言したカテゴリ。

## 新しい同一性を当てる箇所（実行時と生成コードをそろえるため）

1. エンコーダの fit: `categorical_encoder.py:119-135`、写しは `templates.py:318-326`。
2. `_apply_categorical` の cast: `dataframe_builder.py:104-111`、写しは `templates.py:285-294`。
3. unseen の判定、最頻値の埋め、`set_categories`: `categorical_encoder.py:190-237`、写しは `templates.py:408-441` と `:1557-1590`。
4. `comparable_values`（3 つの写し）。
5. codec の encode と decode、その写し、decode の正規形の検査。
6. H-0136 の写し（`state_changes.py`、`_category_preflight.py`）。
7. `CategoricalDtype(...).categories`: `categorical_encoder.py:133`、`category_json.py:99`、`_category_preflight.py:167/175`。
8. `apply_declared_categories` と `input_categories`。
9. LightGBM 自身の list の `==` と `set_categories` の照合。実行時だけで、生成コードは通らない。

## 今の保存状態

- `FORMAT_VERSION = 4`（`exporter.py:68`）。版 3 で `stand_in_columns`、版 4 で tune の履歴を足した。`loader.py:58` は {1,2,3,4} を受け付ける。`categorical_encoder.py:255-257` は、`stand_in_columns` が無ければ `[]` とする。
- joblib（`fit_result.pkl` の `pipeline_state` / `pipeline_state_per_fold`、`refit_model.pkl` の `pipeline_state`）は、codec を通さない生のオブジェクトを持つ:
  `{feature_names, categorical_cols, encoder:{unseen_policy, categories:{col:[values]}, modes:{col:value|None}, stand_in_columns:[col]}, transformer:{}}`。
  pickle された booster も `pandas_categorical` を持つ（置き換えの列は位置の番号）。
- `metadata.json`: `declared_categories:{col:[tagged]}`。
- `pipeline_state.json`（codegen）は版のキーも `stand_in_columns` も持たない:
  `{feature_names, categorical_features, categories:{col:{categories:[tagged], mode:tagged|null}}, unseen_policy}`。`train.py` は自分の fit でこのファイルを書き直す。
- `config.json`: `declared_categories`（tagged）、`categorical_rule{explicit,auto}`、`categorical_features`、`unseen_policy`。
- 3 つ目の JSON の形: `export_state_json` は、joblib のキーの木を tagged の値で書く。

## 今の振る舞いを固定しているテスト（`tests/` の下）

- `test_core/test_category_values.py`: `SAMPLES`、`PYTHON_TYPE_RETURNS`、`test_every_admitted_value_round_trips_equal_with_an_equal_hash`、単位・fold・frozenset の順のテスト、派生クラスの拒否。
- `test_codegen/test_h0133_codec_copy.py`: `COPY`（写しがライブラリと同じに読み書き・拒否する）。
- `test_persistence/test_h0136_grid.py::test_the_verdict_matches_the_real_serializers`（`SAMPLES` の組）。`test_h0136_state_changes.py`、`test_h0136_acceptance.py`（全て）。
- `test_persistence/test_h0133_acceptance.py`: `test_every_admitted_type_saves_and_predicts_the_same`、置き換えと LightGBM のテスト（`:548-561`）。
- `test_persistence/test_unwritable_categories.py`: `VALUES`、`WRITABLE`、`test_the_predicate_agrees_with_the_real_writer`。
- `test_persistence/test_artifact_compat.py::test_a_v2_shaped_artifact_loads_and_predicts_as_before`（`stand_in_columns` が無いことを確かめる）、`_artifact_versions.py`。
- `test_persistence/test_h0120_records.py`（宣言の記録の往復、tagged の値）。
- `test_features/`: `test_category_stand_in.py`（全て）、`test_categorical_float_arrivals.py`、`test_unseen_policy.py`、`test_feature_pipeline.py`、`test_categorical_encoder_edge.py`。
- `test_codegen/`: `test_unseen_policy_codegen.py`（`test_retrain_picks_the_same_mode_as_the_runtime`、`test_export_keeps_the_mode_code_for_a_float32_category` を含む）、`test_float_arrival_codegen.py`、`test_predict_subprocess_equivalence.py::test_unseen_category_matches_model`、`test_retrain_reproduction.py`（`test_categories_parquet`、`test_mixed_type_column_predict_py_matches`、`test_csv_within_the_stated_condition`、`test_new_data_keeps_a_new_category_of_an_inferred_column`、`test_generated_cast_follows_the_builder_rule`）、`test_artifact_writer.py::test_convert_exports_unseen_policy_and_mode`、`test_export_params.py`（`export_state_json`）。
- `test_tuning/test_objective_identity.py`（カテゴリの指紋）。

## H-0139 で時刻の型に広げた箇所（branch `feat/h0139-temporal-category-identity`）

H-0139（#379）は、上の箇所のうち次を変えた。上の行番号は `b297de4` のときのもので、ここは H-0139 の実装の head の位置である。ほかの箇所は変えていない（HISTORY.md `## H-0139` の対応方針 6）。

- `core/category_temporal.py`（新しい）: `temporal_key`（`:95`）と `calendar_units`（`:52`）。
- `core/category_values.py`: `category_identity_key`（`:531`）は時刻の値もキーにする（`_key` `:671` が `temporal_key` に渡す）。`check_calendar_durations`（`:553`）が年・月の値を拒否する。`identity_eligible`（`:640`）は時刻の値を除かず、年・月の値を除く。`holds_temporal`（`:497`）は観察のためだけに残す。
- `data/dataframe_builder.py:122` と `features/encoders/categorical_encoder.py:137`: fit で学ぶ前に `check_calendar_durations` を呼ぶ。
- `features/encoders/category_identity.py:180` `_lookup`: 年・月の値（キーが無い）は unseen になる。
- `persistence/_category_preflight.py`: `declared_change`（`:174`）が refit のエンコーダの状態を受け取り、版 1 で `identity_columns` にある列の時刻の宣言は `_declared_change_by_key`（`:225`）でキーの並びを比べる（`_keyed_temporal` `:212`）。`warn_changes`（`:145`）がそれを渡す。`persistence/exporter.py` の `_refit_encoder`（`:226`）が refit の状態を探し、`:155` と `:291` で渡す。
- 生成コード: `codegen/identity_template.py` の `_temporal_key`（`:131`）・`_calendar_units`（`:79`）・`_check_calendar_durations`（`:93`）・`_key`（`:201`）・`_identity_key`（`:216`）・`_identity_eligible`（`:233`）。`codegen/templates.py` の `cast_categoricals`（`:307`）・`fit_pipeline`（`:368`）・`apply_declared_categories`（`:564`）が学ぶ前に年・月を拒否する。
