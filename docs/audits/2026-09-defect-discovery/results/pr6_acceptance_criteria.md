# PR 6 — 完了基準（レビューを開く前に書いた、2026-10-01）

計画 Revision 6 §12.5。提案は **H-0106**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 証拠列のテスト名は実装前に宣言した
名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#263**: `ErrorCode` の宣言されたメンバーはすべて、本番コードのどこかで、到達できる条件で出る。
  `DATA_FINGERPRINT_MISMATCH` は削除、`INCOMPATIBLE_COLUMNS` / `METRIC_REQUIRES_PROBA` は実装。
- **#272**: `config_version` の検査が、`LizyMLConfig` が `Model` に入るすべての経路で効く（インスタンス、
  `model_construct`、代入、`model_copy`、環境変数の上書き、`False`）。
- **対象外**: `cross_entropy_lambda` の扱いそのもの（#307）、自作 `BaseMetric` サブクラスの確率検査、
  `NativeFeaturePipeline` 単体利用での dtype 検査、`Model` に渡した後のインスタンスの書き換え、
  #271 の残り 7 名（H-0106 の「対象外」に理由）。
- **ラウンド予算 8。** round 2 の前に absolute monitor、round 3 以降は relational monitor。
- **指摘の分類**: B1 = 受け入れ基準を満たさない欠陥、B2 = 基準の欠落（基準を足す）、B3 = 文書の事実誤認
  （管理者の要望により blocking）、B4 = 非 blocking の改善提案。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `1abf7fb`（PR 5b = #308） |
| 契約 | `HISTORY.md` **H-0106**、H-0104（facade の列検査）、#210 |
| 上位文書 | `BLUEPRINT.md` §4 の Config 表（`config_version`）/ §16.2（例外コード） |
| 実測 | `results/pr6_measurements.txt`（スクリプトは `instruments/pr6_*.py`） |

## 2. 修正前の実測（RED の根拠、`1abf7fb`）

| 何を | 結果 |
|---|---|
| `ast.Raise` に現れない `ErrorCode` | 3/20: `DATA_FINGERPRINT_MISMATCH` / `INCOMPATIBLE_COLUMNS` / `METRIC_REQUIRES_PROBA` |
| float64 で学習した列に 33 種の dtype で予測 | 18 種は成功、15 種は生の例外（LightGBM の `ValueError` 2 種、numpy の `DTypePromotionError`）。numpy 型の規則との不一致 0/33 |
| `needs_proba` の指標に binary の logit | `auc` / `auc_pr` / `ece` / `precision_at_k` は黙って計算、`brier` / `logloss` は scikit-learn の `ValueError` |
| `cross_entropy_lambda` で指標を 1 つに限った `Model.fit`（#307 のデータ） | `auc` / `auc_pr` / `ece` / `precision_at_k` は成功し proba 最大 3.148。`logloss` / `brier` は `ValueError` |
| `config_version` | `model_validate(v=2)` / `model_construct` / 代入 / `model_copy` / 環境変数 `2` が受理。`False` も全経路で受理（検証を通る経路では `0` に変換、`model_construct` / 代入 / `model_copy` では `bool` の `False` のまま。設計レビュー round 1 が訂正） |
| firing rate（フルスイート、8108 passed、計測器は round 1 の指摘で修正済み） | dtype 0/112 `run_predict`、読み戻せない記録 0/23 dtype、確率 61/7053（60 は `test_feval_probabilities.py` の `cross_entropy_lambda`、1 は `test_metric_entry_integration.py::test_feval_returns_display_name` の合成 logit。round 2 が帰属を訂正）、版 0/1253 完了した `Model.__init__`・0/1375 完了した `LizyMLConfig` の検証 |

## 3. 受け入れ基準 → 証拠の対応表

「種別」は RED（修正前に失敗する）かガード（今日の挙動を固定する）か。RED の行は、修正前のコードに
対して実行して失敗を確認してから実装に進む。

| # | 基準（H-0106） | 種別 | 証拠 |
|---|---|---|---|
| 1a | `lizyml/` の `ast.Raise` の部分木に現れる `ErrorCode.X` の集合が `set(ErrorCode)` と一致（両方向。初版は包含しか確かめていなかった、コードレビュー round 1） | RED | `tests/test_core/test_error_code_population.py::test_every_member_is_raised_in_production_code` |
| 1b | 走査が 10 メンバー以上を見つける（空の走査を「全メンバー未発生」と読まない） | ガード | `::test_the_scan_finds_the_population` |
| 1c | キーが `set(ErrorCode)` と等しい dict の各メンバーについて、条件を作ると `code` と context のキーが出る | RED | `tests/test_core/test_error_code_raising.py::test_the_conditions_cover_the_enum` / `::test_member_is_raised[*]` |
| 1d | `DATA_FINGERPRINT_MISMATCH` が enum に無い。既存の一覧テストを更新（削除しない） | 更新（一覧は実装と同じコミットで直したので、修正前に RED を実行したのではない。enum の削除は 1a / 1c が RED で捉える） | `tests/test_core/test_exceptions.py::test_all_error_codes_are_defined` |
| 2a | 学習時 float64 の列に 33 種の dtype で予測: 規則が受理 ⇒ 成功、拒否 ⇒ `INCOMPATIBLE_COLUMNS` と context `{"columns": [{"column", "fit_dtype", "predict_dtype"}]}` | RED | `tests/test_features/test_column_dtype_check.py::test_numeric_at_fit_arrival_matrix[*]` |
| 2b | 学習時に数値の各 dtype（int64 / Int64 / bool / boolean / float32 / Sparse 等）で、`str` の到着が拒否される | RED | `::test_every_numeric_fit_dtype_is_checked[*]` |
| 2c | 学習時 `category`（元が `str` / `object` / `category`）の列は dtype の規則で検査されず、33 種の到着で予測できる。例外は整数の category に `float16` / `longdouble` が届く 4 セルで、encoder の中の pandas の生の例外（RED テストの段で発見、#309、strict な xfail） | ガード | `::test_categorical_at_fit_is_not_dtype_checked[*]` |
| 2d | 不足列は dtype より先に `DATA_SCHEMA_INVALID` | ガード | `::test_missing_column_is_reported_before_dtype` |
| 2e | 違反列は 1 回の例外ですべて、学習時の列順で報告 | RED | `::test_every_offending_column_is_reported` |
| 2f | 自作 pipeline（列を検査しない）でも同じ例外 | RED | `::test_custom_pipeline_gets_the_same_check` |
| 2g | `Model.load()` 後も同じ例外 | RED | `::test_check_survives_save_and_load` |
| 2h | fit できる 23 種の dtype の `FitResult.dtypes` の文字列がすべて `pandas_dtype` で読み戻せる | ガード | `::test_every_fittable_dtype_records_a_parseable_string[*]` |
| 2i | 記録が読み戻せない列は検査されず、予測が今日と同じに進む（免除の固定。設計レビュー round 1 の blocking 7） | ガード | `::test_unreadable_recorded_dtype_is_exempt` |
| 2j | 予測時に文字列を数値へ変換する自作 pipeline でも、数値で学習した列に文字列が届けば `INCOMPATIBLE_COLUMNS`（意図した制約の固定。blocking 1） | RED | `::test_converting_custom_pipeline_is_refused` |
| 3a | `needs_proba` の全指標（登録から読む）で、NaN / inf / 負 / 1 超 / 数値でない / 3 クラス以上で 1 次元が `METRIC_REQUIRES_PROBA`、context に `metric` と `reason` | RED | `tests/test_metrics/test_metric_requires_proba.py::test_non_probabilities_are_refused[*]` |
| 3b | 同じ全指標で、binary の正当な確率・0/1 のハードラベル・数値の object 配列は通り、確率の値は lizyml を使わず手で計算した期待値と一致する（初版は有限性しか確かめていなかった、コードレビュー round 1） | ガード | `::test_probabilities_and_hard_labels_pass[*]` |
| 3f | multiclass に対応する 4 指標（`logloss` / `auc` / `auc_pr` / `brier`）は 2 次元の正当な確率で通り、2 次元で [0, 1] の外は拒否 | RED（拒否側） | `::test_multiclass_matrices[*]` |
| 3g | 0/1 以外の 2 値ラベル（`[3, 7]`）の binary は 1 次元の規則で拒否されない | ガード | `::test_two_class_labels_other_than_zero_one_pass[*]` |
| 3c | 検査の対象が登録から読んだ 6 指標と一致（手書きの一覧ではない） | ガード | `::test_the_population_is_every_needs_proba_metric` |
| 3d | `cross_entropy_lambda`、指標 `auc` の `Model.fit`（#307 のデータ）が `METRIC_REQUIRES_PROBA` | RED | `::test_cross_entropy_lambda_fit_reports_the_metric` |
| 3e | feval のテストの「同じ失敗」分岐が新しい例外でも成り立つ | ガード | `tests/test_estimators/test_feval_probabilities.py`（変更なしで通る） |
| 3h | feval の表示名のテストが、LightGBM が実際に渡す確率を入力にして通る（H-0105 の誤った前提の合成 logit を確率に替える。主張は変えない） | 書き直し | `tests/test_metrics/test_metric_entry_integration.py::test_feval_returns_display_name` |
| 4a | 入口（`load_config(dict)` / `Model(dict)` / `model_validate` / `Model(model_validate(...))` / `model_construct` / 代入 / `model_copy(update=)` / 環境変数）× 版（`1` / `2`）: `1` は受理、`2` は `CONFIG_VERSION_UNSUPPORTED` | RED（`2` の新しい入口） | `tests/test_config/test_config_version_entry_paths.py::test_entry_path_by_version[*]` |
| 4b | `config_version: False` がすべての入口で `CONFIG_VERSION_UNSUPPORTED`（dict、`model_validate`、環境変数 `"false"`、`bool` を保持する `Model(model_construct)` / `Model(代入)` / `Model(model_copy)`）。検証を通らない 3 経路では、構築・代入・コピーは成功し `Model` が受け取る時点で拒否される（blocking 3） | RED | `::test_false_is_not_version_zero[*]`（`load_config` / `Model(dict)` / `model_validate` / `Model(model_validate)` / 環境変数。クローズレビューの指摘で 2 経路を追加）と `::test_false_is_refused_when_model_receives_it[*]`（検証を通らない 3 経路） |
| 4e | `True` と `"1"` はすべての入口で受理される | ガード | `::test_true_and_string_one_are_version_one[*]` |
| 4c | loader が拒否する値の context は利用者の綴り（`"2"` / `"2.0"` / `" 2 "` / `2.5` / `-1.5` / `0.5`）。小数の float は今日と同じく切り捨てて判定 | RED（`"2.0"` は今日受理、小数は初版の実装で `CONFIG_INVALID`。コードレビュー round 1） | `::test_loader_context_keeps_the_raw_value[*]` |
| 4f | 版でない値（`1.5` / `"1.5"` / `"x"` / `None` / `inf` / `nan`）は loader で `CONFIG_INVALID`。検証を通らないインスタンスの `1.5` / `"x"` / `None` は `CONFIG_VERSION_UNSUPPORTED`（切り捨てない） | ガード（`inf` は今日生の `OverflowError`） | `::test_values_that_are_not_versions_are_config_invalid[*]` / `::test_unvalidated_instance_with_a_non_version_is_refused[*]` |
| 4d | `lizyml.config.loader.SUPPORTED_CONFIG_VERSIONS is lizyml.config.version.SUPPORTED_CONFIG_VERSIONS` | RED（モジュールが無い） | `::test_supported_versions_is_one_object` |
| 5a | `docs/api.md` の例外コード表 = `set(ErrorCode)` | RED | `tests/test_docs/test_error_code_docs.py::test_api_reference_lists_every_member` |
| 5b | `BLUEPRINT.md` §16.2 の一覧 = `set(ErrorCode)` | RED | `::test_blueprint_lists_every_member` |
| 6 | `docs/DEPRECATIONS.md` に削除の行、`CHANGELOG.md` に H-0106、`BLUEPRINT.md` の `config_version` 行に定義の場所、`PLAN.md` から削除 | — | diff |
| 7 | 計画 `phase3-plan.md` §PR 6 の `METRIC_REQUIRES_PROBA` の段落とファイル一覧、§7 の firing rate を実際の設計に合わせる | — | diff |
| 8 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | — | PR 本文 |

RED の確認（`1c2eebc`、production コードは `1abf7fb` のまま）: 新しい 6 ファイルを実行し、RED と宣言した行は
すべて失敗した。1a（未発生 3 メンバー）、1c（enum の網羅 + 2 メンバー）、2a（15 セル）、2b（18）、2e、2f（2）、
2g、2j、3a（42）、3d、3f（4）、4a（新しい入口の版 2: 6）、4b（6）、4d、5a、5b。ガードと宣言した行は修正前も
通った（2c は #309 の 4 セルを除く。この 4 セルは RED の段で見つかり、strict な xfail にした）。

修正後（`a170494`）: 新しい 6 ファイル + `test_exceptions.py` + feval の 2 ファイルで 666 passed / 4 xfailed。
フルスイート 8408 passed。失敗は環境起因の `test_version_matches_package_metadata` の 1 件（インストール済みの
パッケージメタデータが古い。CI では通る）。ruff / ruff format / mypy `lizyml/` は通過。
