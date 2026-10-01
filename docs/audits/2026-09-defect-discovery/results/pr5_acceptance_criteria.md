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

| # | 基準（H-0104） | 証拠 |
|---|---|---|
| 1a | 4 メソッドだけの pipeline（`categorical_cols` を持たない状態）で `fit` → `predict` が通る | `tests/test_features/test_pipeline_conformance.py::test_minimal_pipeline_survives_fit_and_predict` |
| 1b | 同 pipeline で SHAP（`predict(return_shap=True)` と SHAP 重要度）が通る | `::test_minimal_pipeline_survives_shap` |
| 1c | 1a / 1b は修正前に RED（`AttributeError`） | 実装前のコミットで実行した結果を PR 本文に記録 |
| 2a | 自作 pipeline（列を検査しない）でも、推論時の不足列は `DATA_SCHEMA_INVALID` | `::test_facade_refuses_missing_columns_for_any_pipeline` |
| 2b | 余剰列は警告ちょうど 1 件（自作 pipeline と `NativeFeaturePipeline` の両方で、2 件にならない） | `::test_extra_columns_warn_exactly_once[custom]` / `[native]` |
| 3a | `UnseenPolicy` の全値（`typing.get_args` で読む）を Config から指定し、推論時の観測結果が `"mode"` = 警告 + 最頻値と同じ予測、`"nan"` = 警告 + 欠損と同じ予測、`"error"` = `DATA_SCHEMA_INVALID` | `tests/test_features/test_unseen_policy.py::test_every_policy_is_observable_end_to_end[*]` |
| 3b | 既定（キー省略）は `"mode"` で、置換が警告として報告される（#260 の回帰テスト） | `::test_default_policy_reports_the_substitution` |
| 4 | 指定した方針が refit の pipeline 状態に載り、`Model.load()` 後の `predict` でも方針と警告が保たれる | `::test_policy_survives_save_and_load` |
| 5 | `"error"` で検証 fold にだけ現れるカテゴリがあると `fit` が `DATA_SCHEMA_INVALID`（決定 7） | `::test_error_policy_applies_to_cv_valid_folds` |
| 6 | 生成 `predict.py`: 状態に設定した方針が載り、`"mode"` / `"nan"` の置換でログが出る | `tests/test_codegen/test_unseen_policy_codegen.py::test_generated_predict_logs_substitutions[*]` |
| 7 | `BLUEPRINT.md` §5.4 / §9.2、`docs/config-reference.md`、`ARCHITECTURE.md` が基底クラスと一致する | diff |
| 8 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | PR 本文 |
