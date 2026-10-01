# PR 7 — 完了基準（レビューを開く前に書いた、2026-10-01）

計画 Revision 6 §12.5。提案は **H-0107**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 証拠列のテスト名は実装前に宣言した
名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#267**: `validate_no_target_leakage` が、目的変数と比べられなかった列を黙って飛ばし、検査済みで漏洩なしと
  同じ `[]` を返す。比べられなかった列を列名付きの `DATA_SCHEMA_INVALID` で報告する。
- **対象外**: 目的変数が frame に無いとき `[]` を返す振る舞い（既存テストが固定。#311）、他の 2 つの検査（例外を捕まえる箇所が無い）、`Model.fit` からの自動実行（漏洩検査は明示的に
  呼ぶ公開 API、H-0087）。
- **ラウンド予算 8。** round 2 の前に absolute monitor、round 3 以降は relational monitor。
- **指摘の分類**: B1 = 受け入れ基準を満たさない欠陥、B2 = 基準の欠落、B3 = 文書の事実誤認（管理者の要望により
  blocking）、B4 = 非 blocking の改善提案。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `97381db`（PR 6 = #310） |
| 契約 | `HISTORY.md` **H-0107**、H-0087、#267 |
| 実測 | `results/pr7_measurements.txt`（スクリプトは `instruments/pr7_*.py`） |

## 2. 修正前の実測（RED の根拠、`97381db`）

| 何を | 結果 |
|---|---|
| 数値を名乗り比較で例外を出す拡張配列 × 目的変数 5 種（2 つの形） | 10/10 で比べる呼び出しが `TypeError` / `ValueError`、検査は `[]` を返す |
| 普通の列（33 dtype の列 × 元の順 / 逆順 × 目的変数 7 種） | 0/462 が例外 |
| フルスイートの比べる関数の呼び出し | 0/13 が例外（検査経由 9、直接 4。8424 passed） |

## 3. 受け入れ基準 → 証拠の対応表

| # | 基準（H-0107） | 種別 | 証拠 |
|---|---|---|---|
| 1 | 3 つの形（`TypeError` / `ValueError` / `OverflowError`）× 目的変数 5 種 × `raise_on_violation` 2 値で `DATA_SCHEMA_INVALID`、`context["column"]` / `context["target"]`、`cause` が元の例外と同じオブジェクト | RED | `tests/test_data/test_leakage_validator_unchecked_column.py::test_unchecked_column_is_reported[*]` |
| 2 | 2 つの順 × 2 値: 比較できない列が先なら `DATA_SCHEMA_INVALID`。漏洩列が先なら `True` は `LEAKAGE_SUSPECTED`（即時の報告は今日のまま）、`False` は `DATA_SCHEMA_INVALID`。どの場合も戻り値は返らない | RED | `::test_unchecked_column_stops_the_check[*]` |
| 3 | 普通の列: 漏洩列は `LEAKAGE_SUSPECTED` と `context["leaking_column"]`（包み直されない）、`False` なら警告 1 件、漏洩なしは `[]` | ガード | `::test_ordinary_columns_are_unchanged[*]` |
| 4 | `validators.py` に `except ...: pass` と「Non-comparable」のコメントが無い | RED | `::test_no_silent_skip_remains` |
| 5 | 既存テストを削除しない。`docs/api.md` の漏洩検査の節と `DATA_SCHEMA_INVALID` の行、`CHANGELOG.md` | — | diff |
| 6 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | — | PR 本文 |
