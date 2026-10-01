# PR 8b — 完了基準（レビューを開く前に書く、2026-10-01）

計画 Revision 6 §12.5。提案は **H-0109**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 証拠列のテスト名は実装前に宣言した
名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#281**: `tune → fit → export → load` の後、`params_table()` と `export_code()` は fit が使った
  inner valid の比率（tuned 0.45）ではなく config の比率（0.2）を答える。artifact が「どの training overlay を
  fit が適用したか」を記録していないため。**fit が適用した overlay を `metadata.json` に記録し、`load()` で
  復元する。**
- 計画 §3 の恒久検査「報告面が答える値はすべて `load()` を越えて残る」を、19 の公開報告面 × 4 構成 ×
  5 lifecycle の全数で実行する。残らない値は 3 種類あり、#281 だけをこの PR で直す。
  - **#281**（`validation_ratio`、24 セル）: この PR で直す。
  - **gain 重要度**（20 セル）: LightGBM のモデルテキストが `split_gain` を有効数字 6 桁で書くため
    （相対誤差の上限 5e-6）。直さずに書いて、許容差で比べる。予測と OOF は bit 単位で一致する。
  - **tuning の 3 面**（36 セル）: H-0086 が trial の履歴を保存しないと決めた結果。空の表と空の図で
    「trial が無い」ように見えることは書かれていない → **#315** に切り出した（DC1、判断が要る）。
- **対象外**: #315 の判断。`format_version` は 2 のまま（追加のキーだけで、H-0083 `checksums` /
  H-0086 `tuning` と同じ形）。
- **ラウンド予算 8。** round 2 の前に absolute monitor、round 3 以降は relational monitor。
- **指摘の分類**: B1 = 受け入れ基準を満たさない欠陥、B2 = 基準の欠落、B3 = 文書の事実誤認（管理者の要望により
  blocking）、B4 = 非 blocking の改善提案。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `036cd18`（PR 8 = #314） |
| 契約 | `HISTORY.md` **H-0109**、#281、#315、H-0094 決定 13（置き換える bound）、H-0086（`tuning` ブロック） |
| 実測 | `results/pr8b_measurements.txt`（各計測器の出力は同じディレクトリの `pr8b_*.txt`） |

## 2. 修正前の実測（`036cd18`）

| 何を | 結果 |
|---|---|
| 報告面の全数（`pr8b_load_census.py`） | 380 セル中 80 セルが `load()` 後に違う: `params_table` 12 / `export_code` 12（#281）、`importance_gain` 20、`tuning_table` 16 / `tuning_plot` 16 / `boundary_table` 4（#315） |
| #281 のセル | fit が tuning result を消費した lifecycle（`tune_fit` / `tune_fit_reexport` / `tune_resume_fit`）× 4 構成 × 2 面 = 24。`fit_tune` は一致する（その fit は overlay を使っておらず、config に落ちるのが正しい） |
| gain の原因（`pr8b_gain_precision.py`） | `split_gain=` は有効数字 6 桁。pickle した adapter の gain = テキスト往復の gain（3 タスク）。観測した最大相対誤差 1.040e-06、形式から決まる上限 5e-6。予測と OOF は bit 単位で一致 |
| tuning の 3 面（`pr8b_tuning_surfaces_after_load.py`） | `tuning_table` (4, 14) → (0, 0)、`tuning_plot` 2 trace → 0 trace、`boundary_table` 表 → `MODEL_NOT_FIT`。#315 |
| 修正前の `metadata.json` のキー（`pr8b_metadata_keys.py`） | `checksums config feature_names format_version lizyml_version metrics python_version run_id task timestamp`（tuning 後は `tuning` も）。`tuning.best_training_params` は `tune_fit` と `fit_tune` で同じ `{'early_stopping_rounds': 219, 'validation_ratio': 0.45}` → tuning ブロックからは、どの fit が消費したかが分からない |

## 3. 受け入れ基準 → 証拠の対応表

| # | 基準（H-0109） | 種別 | 証拠 |
|---|---|---|---|
| 1a | `export()` は `metadata.json` に `applied_training_params` を書く: `fit` → `{}`、`tune → fit` → tuning result の `best_training_params` と等しい dict、`fit → tune` → `{}`（tuning ブロックは overlay を持つのに） | RED（キーが無い） | `tests/test_persistence/test_applied_training_params.py::test_export_records_the_overlay_the_fit_applied[*]` |
| 1b | 書かれた値の型は JSON の往復で変わらない（`early_stopping_rounds` は int、`validation_ratio` は float。既定の空間の IntDim と、利用者の categorical の両方） | RED | `::test_recorded_values_keep_their_types` |
| 2 | `tune → fit → export → load` の後、`params_table()` と `export_code()` が tuned の比率（0.45）を答える。3 タスク | RED | `::test_a_loaded_model_reports_the_ratio_its_fit_applied[*]` |
| 3 | `load → export → load` で値が残る（記録を持つ artifact の再 export は同じ記録を書く） | RED | `::test_reexport_carries_the_record_forward` |
| 4a | キーの無い artifact（修正前の export。キーを消した metadata は修正前の export と同じキー集合であることを §2 で測った）は今までどおり読め、比率は config に落ちる（H-0094 決定 13 の bound は、記録の無い artifact に限って残る） | ガード | `::test_an_artifact_without_the_record_falls_back_to_the_config` |
| 4b | キーの無い artifact を load して再 export しても、キーを書かない（「不明」を `{}`「overlay なし」として記録しない） | ガード（修正前も通る。`None` と `{}` を潰す実装で失敗する） | `::test_an_unknown_record_is_not_rewritten_as_empty` |
| 5 | 不正な記録は `load()` で `DESERIALIZATION_FAILED`（キー名を context に）: dict でない / 知らないキー / bool / 文字列 / 非有限の値 | RED（今は黙って無視） | `::test_a_malformed_record_is_refused_on_load[*]` |
| 6 | `load()` の後に `fit()` すると、記録はその fit が適用した overlay に置き換わる（復元した tuning result を消費する） | ガード | `::test_a_fit_after_load_records_its_own_overlay` |
| 7a | 恒久検査: 19 面 × 4 構成 × 5 lifecycle で、`load()` 前後の読みが一致する。例外は宣言した 2 つだけ: gain 重要度は相対 5e-6 以内、tuning の 3 面は #315 のセル（`tuning_table` / `tuning_plot` は tuning した lifecycle、`boundary_table` は `tune_resume_fit`）。**違うセルの集合 = 宣言した例外の集合**（#315 を直すとテストがそれを知らせる） | RED（#281 の 24 セル） | `tests/test_persistence/test_reporting_surfaces_survive_load.py::test_every_reading_survives_load_except_the_declared` |
| 7b | 空振りの防止: 19 面それぞれに、修正前のモデルが読みを返したセルが 1 つ以上ある。tuned と config の比率が違うセルが 12 以上ある | ガード | `::test_every_surface_was_exercised` |
| 8 | `report_lifecycle_grid.py` の `known-bound` の 2 セルが `agrees` になり、計測器の主張を書き換える | RED（計測器の assert） | 計測器の出力（`results/pr8b_lifecycle_grid_after.txt`） |
| 9 | 既存のテスト `test_a_loaded_model_reports_the_patience_its_adapters_carry`（`fit → tune → export → load`）は、比率が config のままであることを引き続き確かめる（その fit は overlay を使っていない）。docstring の「比率は残らない」を直す | ガード | `tests/test_core/test_fit_params_override.py::test_a_loaded_model_reports_the_patience_its_adapters_carry` |
| 10 | 文書: BLUEPRINT §7.4 / §15.1 に記録を足し、l.1455 の bound を書き換える。`exporter.py` の配置の docstring、bound を書いた 4 か所のコメント（`_model_state.py` / `model.py` / `_model_tables.py` / `_model_persistence.py`）、CHANGELOG、計画 §3 の行 8b | — | diff |
| 11 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | — | PR 本文 |

RED と宣言した行（1a、1b、2、3、5、7a、8）は修正前に失敗し、修正後に通る。4b は修正前も通る
（修正前は何も書かない）。4b の役目は「不明」と「overlay なし」を同じ `{}` に潰す実装を落とすことで、
レビューでその変異を確かめる。
