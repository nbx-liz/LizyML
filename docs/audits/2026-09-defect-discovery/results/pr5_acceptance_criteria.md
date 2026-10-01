# PR 5 — 完了基準（レビューを開く前に書いた、2026-10-01）

計画 Revision 6 §12.5。提案は **H-0104**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 証拠列のテスト名は実装前に宣言した
名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#259**: `BaseFeaturePipeline` の宣言どおりに実装した pipeline が推論経路を通るようにする。推論時の
  列検査を facade に置き、自作 pipeline でもすり抜けられないようにする（PR 6 の `INCOMPATIBLE_COLUMNS`
  はここに足す）。
- **#260**: 未知カテゴリの置換を `PredictionResult.warnings` に報告し、`unseen_policy` を Config に出す。
- **対象外**: fit 中（CV の検証 fold）の置換の報告（H-0104 決定 8、外れる保証を明記）、dtype の不一致
  （PR 6）、公開の provider 登録手段（存在しない。自作 pipeline のテストは provider の factory に差し込む）。
- **ラウンド予算 8。** round 2 の前に absolute monitor、round 3 以降は relational monitor。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `2d3bc54`（PR 4 = #302） |
| 契約 | `HISTORY.md` **H-0104**、H-0054 / H-0085、#205 |
| 上位文書 | `BLUEPRINT.md` §5.4（`features`）/ §7.3（`warnings`）/ §9.2（列ズレ方針） |

## 2. 修正前の実測（RED の根拠）

| 何を | 結果（`2d3bc54`） |
|---|---|
| 4 メソッドだけの `BaseFeaturePipeline` サブクラス | `hasattr(p, "transform_with_warnings")` は `False`。推論経路は `_model_predict.py:46` で無条件に呼ぶ（#259 の再現） |
| 既定方針で推論時に未知カテゴリ | 置換は起きるが `warnings == []`（#260 の再現） |
| Config に `features.unseen_policy` | `extra="forbid"` で `CONFIG_INVALID` |
| 既定を `"error"` に差し替えてフルスイート | CV 中の拒否は、それを起こすために作られたテスト 1 件だけ（自然な設定での発火 0 件） |

## 3. 受け入れ基準 → 証拠の対応表

設計レビュー round 1（REQUEST_CHANGES）を受けて改訂した。「種別」は RED（修正前に失敗した）か
ガード（今日の挙動を固定する回帰ガード）か。RED の行は、修正前のコードに対して実行して失敗を確認した。

| # | 基準（H-0104） | 種別 | 証拠 |
|---|---|---|---|
| 1a | 4 メソッドだけの pipeline（`categorical_cols` を持たない状態）で `fit` → `predict` が通る | RED | `tests/test_features/test_pipeline_conformance.py::test_minimal_pipeline_survives_fit_and_predict` |
| 1b | 同 pipeline で SHAP（`predict(return_shap=True)` と SHAP 重要度）が通る | RED | `::test_minimal_pipeline_survives_shap` |
| 2a | 自作 pipeline（列を検査しない）でも、推論時の不足列は `DATA_SCHEMA_INVALID` | RED | `::test_facade_refuses_missing_columns_for_any_pipeline` |
| 2b | 余剰列は警告ちょうど 1 件（自作 / `NativeFeaturePipeline` の両方） | RED（custom）/ ガード（native） | `::test_extra_columns_warn_exactly_once[custom]` / `[native]` |
| 2c | 自作 pipeline が自分で出した警告は変えずに届き、facade の列警告と並ぶ | RED | `::test_a_pipeline_reports_its_own_warnings_unchanged` |
| 3a | `UnseenPolicy` の全値（`typing.get_args`）を Config から指定し、推論時に `"mode"` = 警告 + 最頻値と同じ予測、`"nan"` = 警告 + 欠損と同じ予測、`"error"` = `DATA_SCHEMA_INVALID`。最頻値に置換した予測と欠損にした予測が異なることを同じテストで確かめる（識別できるデータ） | RED | `tests/test_features/test_unseen_policy.py::test_every_policy_is_observable_end_to_end[*]` |
| 3b | 既定（キー省略）は `"mode"` で、置換が警告として報告される（#260 の回帰テスト） | RED | `::test_default_policy_reports_the_substitution` |
| 3c | 未知カテゴリが無ければ警告は空 | ガード | `::test_no_warning_without_unseen_categories` |
| 3d | Config の値の集合と `UnseenPolicy` が一致する（層の規約上 2 か所に書くため） | RED | `::test_config_literal_is_the_encoder_type` |
| 3e | `tune` が作るすべての pipeline に指定した方針が載る | RED | `::test_tune_applies_the_configured_policy[*]` |
| 3f | 予測は保存済みの pipeline 状態の方針に従い、現在の Config の方針には従わない（生きたモデルと `load()` 後の両方、識別できるデータで。コードレビュー round 1 の blocking 2） | RED（変異） | `::test_predict_follows_the_saved_policy_not_the_current_config`。「予測直前に Config の方針で上書きする」変異で失敗することを確認 |
| 4 | 指定した方針が refit の pipeline 状態に載り、`Model.load()` 後の `predict` でも方針と警告が保たれる | RED | `::test_policy_survives_save_and_load` |
| 5a | `"error"` で検証 fold にだけ現れる値: `auto_categorical: false` では `fit` が `DATA_SCHEMA_INVALID`、`true` では通る（決定 7） | RED | `::test_error_policy_applies_to_cv_valid_folds` |
| 5b | スライディング窓で最後の fold に属さない行の値: `"error"` は SHAP 重要度が `DATA_SCHEMA_INVALID`、`"mode"` は通る（決定 8） | RED | `::test_shap_importance_applies_the_stored_policy_outside_the_last_fold` |
| 6a | エクスポートした `config.json` と `pipeline_state.json` に、fit が適用した方針が載る | RED | `tests/test_codegen/test_unseen_policy_codegen.py::test_export_carries_the_policy_the_fit_applied[*]` |
| 6b | 生成 `predict.py` が `"mode"` / `"nan"` の置換をログに出し、`"error"` は拒否し、欠損は欠損のまま | RED | `::test_generated_predict_logs_substitutions[*]` |
| 6c | 生成 `predict.py` は `"error"` でも欠損値を拒否しない | RED | `::test_missing_values_are_not_refused_under_error` |
| 6d | 生成 `train.py` の `fit_pipeline` が書き直す pipeline 状態に方針と最頻値コードが残り、生成 `predict.py` の変換がそれに従う（`train.py` 全体の再学習は実行していない） | RED | `::test_retrain_keeps_the_policy[*]` |
| 6e | 同値の最頻値（数値 `[2, 10]`、カテゴリ順 `["b", "a"]`）で、生成 `fit_pipeline` が実行時の encoder と同じ値を選ぶ（コードレビュー round 1 の blocking 1） | RED（変異） | `::test_retrain_picks_the_same_mode_as_the_runtime[*]`。文字列化してから最頻値を取る旧実装で 2 件とも失敗することを確認 |
| 7 | `BLUEPRINT.md` §5.4 / §9.2、`docs/config-reference.md`、`ARCHITECTURE.md` が基底クラスと一致する | — | diff |
| 8 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | — | PR 本文 |

RED の確認方法: 1a-3e と 4-5 は `develop` `2d3bc54` の production コードで実行して失敗（`AttributeError` /
`CONFIG_INVALID` / 警告が空 / `DID NOT RAISE`）。6a-6d は Config と features の変更を入れたまま
`lizyml/codegen/` だけを `2d3bc54` に戻して実行し、9 件すべて失敗した。
