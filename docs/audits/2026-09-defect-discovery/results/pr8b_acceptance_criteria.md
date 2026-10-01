# PR 8b — 完了基準（レビューを開く前に書き、設計レビュー round 1 で改訂した、2026-10-01）

計画 Revision 6 §12.5。提案は **H-0109**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 証拠列のテスト名は実装前に宣言した
名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#281**: `tune → fit → export → load` の後、`params_table()` と `export_code()` は fit が使った
  inner valid の比率（tuned 0.45）ではなく config の比率（0.2）を答える。artifact が「どの training overlay を
  fit が適用したか」を記録していないため。**fit が適用した overlay を `metadata.json` に記録し、`load()` で
  復元する。**
- 計画 §3 の恒久検査「報告面が答える値はすべて `load()` を越えて残る」を、`Model` の公開名 24 個（報告面
  22 面と、理由を書いて除く 4 個）× 4 構成 × 5 lifecycle で実行する。各面は既定の引数で呼ぶ（bound）。
  残らない値は 3 種類あり、#281 だけをこの PR で直す。
  - **#281**（`validation_ratio`、24 セル）: この PR で直す。
  - **gain 重要度**（20 セル）: LightGBM のモデルテキストが `split_gain` を有効数字 6 桁で書き、binary32 で
    読み戻すため（相対誤差の上限 5.0596e-6）。直さずに書いて、許容差で比べる。
  - **tuning の 3 面**（36 セル）: H-0086 が trial の履歴を保存しないと決めた結果。→ **#315** に切り出した
    （DC1、判断が要る）。
- 恒久検査は「すべての値が残る」の証明ではなく、上の母集団と bound の中で「違うセル = 宣言した例外」を
  固定する回帰の契約である。
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
| 契約 | `HISTORY.md` **H-0109**、#281、#315、H-0094 決定 13（置き換える bound）と決定 14（fit の commit）、H-0086（`tuning` ブロック） |
| 実測 | `results/pr8b_measurements.txt`（各計測器の出力は同じディレクトリの `pr8b_*.txt`） |

## 2. 修正前の実測（`036cd18`）

| 何を | 結果 |
|---|---|
| 報告面の全数（`pr8b_load_census.py`、定義は `tests/test_persistence/_load_census.py`） | 440 セル中 80 セルが `load()` 後に違う: `params_table` 12 / `export_code` 12（#281）、`importance_gain` 20、`tuning_table` 16 / `tuning_plot` 16 / `boundary_table` 4（#315）。`evaluate` / `evaluate_table` / `fit_result` / `predict` を含む他の 16 面は全セル一致 |
| #281 のセル | fit が tuning result を消費した lifecycle（`tune_fit` / `tune_fit_reexport` / `tune_resume_fit`）× 4 構成 × 2 面 = 24。`fit_tune` は一致する（その fit は overlay を使っておらず、config に落ちるのが正しい） |
| gain の原因（`pr8b_gain_precision.py`） | `split_gain=` は有効数字 6 桁。pickle した adapter の gain = テキスト往復の gain（3 タスク）。実モデルの最大相対誤差 1.040e-06。`%g` → binary32 の往復を 1e-38〜1e38 で掃いた最悪 5.054398e-06（設計レビューの反例と同じ値）で、上限 5.059605e-06 の内側。split gain はすべて正（最小 0.0144）。予測と OOF は測った fixture で bit 単位で一致 |
| tuning の 3 面（`pr8b_tuning_surfaces_after_load.py`） | `tuning_table` (4, 14) → (0, 0)、`tuning_plot` 2 trace → 0 trace、`boundary_table` 表 → `MODEL_NOT_FIT`。一度も tuning していないモデルでは 3 面とも `MODEL_NOT_FIT`。#315 |
| 修正前の `metadata.json` のキー（`pr8b_metadata_keys.py`） | `checksums config feature_names format_version lizyml_version metrics python_version run_id task timestamp`（tuning 後は `tuning` も）。`tuning.best_training_params` は `tune_fit` と `fit_tune` で同じ `{'early_stopping_rounds': 219, 'validation_ratio': 0.45}` → tuning ブロックからは、どの fit が消費したかが分からない |

## 3. 受け入れ基準 → 証拠の対応表

| # | 基準（H-0109） | 種別 | 証拠 |
|---|---|---|---|
| 1a | `export()` は `metadata.json` に `applied_training_params` を書く: `fit` → `{}`、`tune → fit` → tuning result の `best_training_params` と等しい dict、`fit → tune` → `{}`（tuning ブロックは overlay を持つのに） | RED（キーが無い） | `tests/test_persistence/test_applied_training_params.py::test_export_records_the_overlay_the_fit_applied[*]` |
| 1b | 書かれた値の型は JSON の往復で変わらない（`early_stopping_rounds` は int、`validation_ratio` は float。既定の空間の IntDim と、利用者の categorical の両方） | RED | `::test_recorded_values_keep_their_types[*]` |
| 1c | 修正後の export の最上位のキー = 修正前に測ったキー集合 ∪ {`applied_training_params`}（キーを消した artifact を修正前の artifact として使う根拠） | RED | `::test_the_record_is_the_only_new_key[*]` |
| 2 | `tune → fit → export → load` の後、`params_table()` と `export_code()` が tuned の比率（0.45）を答える。3 タスク | RED | `::test_a_loaded_model_reports_the_ratio_its_fit_applied[*]` |
| 3 | `load → export → load` で値が残る | RED | `::test_reexport_carries_the_record_forward` |
| 4a | キーの無い artifact は今までどおり読め、比率は config に落ちる（H-0094 決定 13 の bound は、記録の無い artifact に限って残る）。状態は `None`（不明） | RED（今は `{}`。比率の落ち方は修正前も同じ） | `::test_an_artifact_without_the_record_falls_back_to_the_config` |
| 4b | キーの無い artifact を load して再 export しても、キーを書かない（「不明」を `{}`「overlay なし」として記録しない） | ガード（修正前も通る。`None` と `{}` を潰す実装で失敗する） | `::test_an_unknown_record_is_not_rewritten_as_empty` |
| 5 | 不正な記録は `load()` で `DESERIALIZATION_FAILED`（メッセージにキー名）: dict でない / 知らないキー / bool / 文字列 / null / NaN / inf / `validation_ratio` が 0 や 1 | RED（今は黙って無視） | `::test_a_malformed_record_is_refused_on_load[*]` |
| 6a | `load()` の後に成功した `fit()` は、記録をその fit が適用した overlay に置き換える（tuned の artifact からは復元した tuning result を消費し、記録の無い未 tuning の artifact からは `{}`）。再 export がそれを書く | RED（再 export がキーを書かない） | `::test_a_fit_after_load_records_its_own_overlay[*]` |
| 6b | `load()` の後の `tune()`、拒否された `fit()`、学習中に失敗した `fit()` は、記録（既知でも不明でも）を変えず、再 export もそれを保つ（既知 → 同じ dict、不明 → キーなし）。比率の報告も変わらない | RED（状態が `{}`） | `::test_a_loaded_record_survives_calls_that_do_not_replace_the_fit[*]`（3 × 2 セル） |
| 7a | 恒久検査: 22 面 × 4 構成 × 5 lifecycle で、`load()` 前後の読みが一致する。例外は宣言した 2 つだけ: gain 重要度は相対 5.1e-6 以内、tuning の 3 面は #315 のセル。**違うセルの集合 = 宣言した例外の集合**（#315 を直すとテストがそれを知らせる） | RED（#281 の 24 セル） | `tests/test_persistence/test_reporting_surfaces_survive_load.py::test_every_reading_survives_load_except_the_declared` |
| 7b | 母集団: `INVENTORY` のキー = `Model` の公開名、読む面の集合 = `SURFACES` | ガード（新しい公開メソッドで失敗する） | `::test_the_inventory_classifies_every_public_name` |
| 7c | 空振りの防止: 22 面それぞれに、修正前のモデルが読みを返したセルがある。tuned と config の比率が違うセルが 12 | ガード | `::test_every_surface_was_exercised` |
| 8 | `report_lifecycle_grid.py` の `known-bound` の 2 セルが `agrees` になり、計測器の主張を書き換える | RED（計測器の assert） | 計測器の出力（`results/pr8b_lifecycle_grid_after.txt`） |
| 9 | 既存のテスト `test_a_loaded_model_reports_the_patience_its_adapters_carry`（`fit → tune → export → load`）は、比率が config のままであることを引き続き確かめる（その fit は overlay を使っていない）。docstring の「比率は残らない」を直す | ガード | `tests/test_core/test_fit_params_override.py::test_a_loaded_model_reports_the_patience_its_adapters_carry` |
| 10 | 文書: BLUEPRINT §7.4 / §15.1 に記録を足し、l.1455 の bound を書き換える。`exporter.py` の配置の docstring、bound を書いた 4 か所（`_model_state.py` / `model.py` / `_model_tables.py` / `_model_persistence.py`）、CHANGELOG、計画 §3 の行 8b | — | diff |
| 11 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | — | PR 本文 |

RED と宣言した行（1a、1b、1c、2、3、4a、5、6a、6b、7a、8）は修正前に失敗し、修正後に通る。4b は修正前も
通る（修正前は何も書かない）。4b の役目は「不明」と「overlay なし」を同じ `{}` に潰す実装を落とすことで、
レビューでその変異を確かめる。

RED の確認（テストは未コミット、production コードは `036cd18` のまま）: 2 ファイルで 31 failed / 3 passed。
通ったのは 4b、7b、7c（ガードと宣言した行）。6b の失敗はすべて状態の assert（既知 → `{}` == `None`、
不明 → `{}` is `None`）で、拒否と失敗の補助関数は想定どおり例外を出している。7a の失敗は宣言外のセル
（#281 の `params_table` / `export_code`）だけで、宣言した #315 のセルが一致する失敗は無い。
