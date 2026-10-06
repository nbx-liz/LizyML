# PR 9 — 完了基準（レビューを開く前に書く、2026-10-06）

計画 §4 PR 9（Revision 5 の範囲）。提案は **H-0110**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 証拠列のテスト名は実装前に宣言した
名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#271**: HISTORY.md で決まり、実装され、BLUEPRINT.md に畳み込まれなかった決定がある。代表例は H-0083
  （artifact の `.pkl` ごとの SHA-256 を `metadata.json` に書き、load 時に検証する）で、BLUEPRINT は
  `checksum` に 1 度も触れていない（`13fb9d7` で 0 件）。
- **この PR がすること**:
  1. 2026-09-06 の母集団（40 entries / 129 clauses / 77 edits）を、`13fb9d7` の BLUEPRINT に対して
     clause ごとに**洗い直す**（§2）。理由: 129 のうち 65 は記録が残っておらず復元できない
     （`instruments/extract_missed_clauses.py` の docstring）。残る 64 も 2026-09-06 の BLUEPRINT に対する
     判定で、その後 Phase 3 の PR が BLUEPRINT を編集した。HISTORY は 92 → 110 件に増えた。
  2. 洗い直しで `missed` / `contradicted` と判定された clause を BLUEPRINT に書く（節ごとのコミット）。
  3. 恒久検査 `tests/test_docs/test_proposal_blueprint_coverage.py` を足す。110 件すべての提案に、
     `docs/proposal_dispositions.toml` の処分が 1 つずつある（Status を問わない。管理者決定 2026-10-06）。
  4. #271 が挙げた未文書化の公開名 6 個を、文書化するか internal と記録する。
- **しないこと（宣言した境界）**:
  - 洗い直しの対象外の 68 件（BLUEPRINT がすでに id を引用している提案と、2026-09-05/06 の再監査で
    義務なしとされた提案）は、clause ごとには監査しない。処分の anchor（BLUEPRINT と HISTORY の両方に
    全単語一致で現れる語）を確かめるだけである。したがって恒久検査は「すべての決定が BLUEPRINT に
    書かれている」の証明ではない。**「すべての提案に、内容で確かめられる処分がある」**の検査である。
  - anchor が決定を述べている行を指すかは、計測器では読めない。処分ファイルのレビューで確かめる
    宣言である（#271 の再監査 調整 2 と同じ、機械化できない 1 点）。
  - HISTORY の古い Status 行（H-0009〜H-0012、H-0056 は実装済みで `proposed`）は直さない。Issue にする。
  - コードは変えない。BLUEPRINT がコードと食い違う場合は、コードが提案どおりなら BLUEPRINT を直し、
    コードが提案と違う場合（`not_in_force`）は直さずに Issue にする。
- **ラウンド予算**: 設計レビュー 3 回 + 受け入れレビュー 4 回（管理者決定 2026-10-06）。尽きたら最後の
  ラウンドの修正だけを狭く事実確認し、それでも APPROVE が出なければ管理者に上げる。停止は authorship
  （直前の修正が書いた箇所の欠陥か）で判断する。round 2 の前に absolute monitor、round 3 以降は
  relational monitor。
- **指摘の分類**: B1 = 受け入れ基準を満たさない欠陥、B2 = 基準の欠落、B3 = 文書の事実誤認（管理者の
  要望により blocking）、B4 = 非 blocking の改善提案。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `13fb9d7`（PR 8c = #317） |
| 契約 | #271 の本文と 2026-09-06 のコメント、`HISTORY.md` **H-0110**、計画 §4 PR 9（Revision 5） |
| 実測 | `results/pr9_census_13fb9d7.txt`、`results/pr9_clause_inventory.md`（§2） |

## 2. 母集団（実測、`13fb9d7`。`results/pr9_measurements.txt`）

| 母集団 | 件数 | 出どころ |
|---|---|---|
| HISTORY の提案 | 110（H-0110 を足して 111） | `pr9_census_13fb9d7.txt` |
| clause を洗い直した entry | 42（#271 の 40 + H-0105 / H-0107） | `pr9_inventory_{A,B}.json` |
| その clause | 307（stated 111 / missed 113 / contradicted 14 / superseded 17 / not_in_force 6 / off_surface 46） | `pr9_clause_inventory.md` |
| 書き足しが要る entry | 41 + 調査 C が報告した 5 = 46 | 同上、`pr9_dispositions_C.json` |
| 畳み込む clause | 127 + 5 件の決定 + H-0110 決定 4 の 2 点 + 方針 8 の 1 行 | 同上 |
| 処分のみ決めた entry | 68（specified 64 / superseded 2 / no_obligation 2） | `pr9_dispositions_C.json` |
| 免除の処分 | 4/110 | 処分ファイル |
| 未文書化の公開名 | 6（#271 の 8 から、文書化済みの 2 を除く） | #271 本文 |

正解の分かっている clause 3 件をラベルなしで混ぜ、3/3 が正しい行で `stated` と返った（item 3）。

## 3. 受け入れ基準

| # | 基準 | 証拠 |
|---|---|---|
| A1 | 処分の行 = HISTORY の id（両方向） | `test_every_proposal_has_exactly_one_row`、`test_rows_must_equal_the_register_in_both_directions` |
| A2 | HISTORY の id ごとに 1 件で、件数は HISTORY から導出し、全件が通る | `test_proposal_disposition_holds[H-xxxx]`（111 件） |
| A3 | 文法の拒否: 上位文字列 5 例、境界のある一致 4 例、不正な行 15 種、`[names]` の不正 4 種 | `test_a_superstring_is_not_a_token_match`、`test_a_bounded_occurrence_is_a_token_match`、`test_a_malformed_row_is_refused`、`test_a_malformed_name_row_is_refused` |
| A4 | RED: 未編集の BLUEPRINT で 20 件が失敗する（書き足しが要る entry 18 件 + H-0110 自身の行 + `CHECKSUM_ALGORITHM`）。書き足しの要らない既存の提案の行は失敗しない | `pr9_measurements.txt` item 6、RED コミットでのテスト実行 |
| A5 | 書き足しが要る 46 件すべてが、`13fb9d7` に無かった anchor を持つ | `instruments/pr9_discriminating_anchors.py --base 13fb9d7` exit 0 |
| A6 | 畳み込む 127 clause、5 件の決定、H-0110 決定 4 の 2 点（重なる clause はその行で示す）、方針 8 の 1 行のそれぞれに、それを述べる BLUEPRINT の行がある | `results/pr9_fold_map.md`（レビューで確かめる） |
| A7 | `[names]` の行 = #271 の 6 名（両方向）で、6 名すべてが成り立つ | `test_public_name_dispositions_hold` |
| A8 | HISTORY の id の検査は文法の移設の前後で同じ結果 | `test_history_ids.py` 5 passed（移設の前後） |
| A9 | phase3 manifest の #271 行: `population_test` が 111 件を集め（`derived_from` で HISTORY から導出）、`red_mutation` が H-0083 の checksum の記述を消して失敗する | `instruments/phase3_gap.py`（PR 9b で `closure_comment` を埋めて exit 0） |
| A10 | 本 PR で直さないもの（§5）を Issue にする | Issue 番号を PR 本文に書く |

## 4. 受理/拒否の表（Accept/reject matrix）と領域の閉じ方（Domain closure）

Accept/reject matrix:

| 入力 | 受理 | 拒否（失敗。読み飛ばさない） |
|---|---|---|
| HISTORY の entry | `## ` 見出し（fence の外）で、id を `## H-xxxx`（直後が `:` か行末）か `` - ID: `H-xxxx` `` で宣言する。見出しと 1 行の `- ID:` が同じ id を名指す形は 1 つの宣言として受理する（実測: H-0054〜H-0060 の 7 件） | id が 0 / 2 個の entry、`- ID:` 行が 2 行以上の entry（同じ id でも。`grammar_violations`、設計レビュー round 2）、ID 行のように始まるが受理する綴りでない行（大文字小文字や空白の違いを含む。round 3）、2 つの entry が宣言する id、`## H-0042-extra` のような接尾辞つきの見出し（0 id として失敗）（H-0101 の文法、`_history_grammar.py`） |
| fence | 3 個以上の `` ` `` か `~` で開き（字下げは任意）、**同じ文字で同じ長さ以上**の行だけで閉じる | 閉じないまま終わるファイル（`grammar_violations`）。`` ``` `` の中の `~~~` は閉じない（設計レビュー round 1） |
| 処分ファイルの最上位 | `[proposals]` と `[names]` の 2 表だけ | それ以外の表、TOML として読めないファイル（`tomllib` の例外） |
| `[proposals."H-xxxx"]` | `disposition` が 4 種のいずれかで、その種の必須 key がそろい、任意 key 以外が無い | 知らない disposition、必須 key の欠落、知らない key、空の文字列、表でない値 |
| `anchors` | 1 個以上の、重複の無い空でない文字列で、それぞれ BLUEPRINT と提案自身の entry の両方に全単語一致 | 空のリスト、重複、提案 id を含む文字列、どちらかの文書に無い語 |
| `superseded_by` | 自分以外の、HISTORY が宣言する id | 自分自身、HISTORY に無い id |
| `[names."NAME"]` | `documented` + `where`（`BLUEPRINT.md` / `docs/api.md`）で全単語一致、または `internal` + `reason` | 他の文書、全単語一致しない名前、知らない disposition |
| 全単語一致 | 前後に `[A-Za-z0-9_]` が無い出現 | 上位文字列の中の出現（`checksum_algorithm` の中の `checksum`） |

Domain closure: 処分の行の集合は HISTORY の id の集合と両方向で等しく（A1）、提案ごとのテストはその id で parametrize する（A2）ので、母集団は HISTORY が閉じる。HISTORY の文法は H-0101 の 2 つの綴りに閉じ、それ以外は失敗する。処分の種は 4 つに閉じ、key の集合も種ごとに閉じる。anchor の一致の境界は `[A-Za-z0-9_]` の文字クラスで閉じる（日本語の語は境界の外にあるので、その前後は常に境界である）。

## 5. 計測器に読めず、宣言とレビューに残るもの

1. **anchor が決定を述べる行を指すか。** 計測器は語の存在しか読めない。`results/pr9_fold_map.md` が行を示し、レビューが読む。#271 の再監査 調整 2 と同じ、機械化できない 1 点。
2. **処分のみ決めた 68 件の clause は監査していない。** 恒久検査は「すべての決定が書かれている」ではなく「すべての提案に内容で確かめられる処分がある」の検査である。
3. **調査の判定そのもの**（`stated` の 111 件、`off_surface` の 46 件、`superseded` の 17 件）は全数では再検査していない。正解 3 件の対照が 3/3 で、`stated` には行と引用がある。設計レビュー round 1 が 30 clause を抜き取って 29 件に同意し、残る 1 件（H-0002 の dataclass）と同じ形の H-0082 の 1 件を `off_surface` から `missed` に直した（`results/pr9_measurements.txt` item 7）。
4. **畳み込んだ後に BLUEPRINT の文が消えても、anchor の語が別の場所に残れば検査は通る。** 恒久検査が守るのは語の存在であって文ではない。
5. **本 PR で直さず Issue にするもの**: BLUEPRINT か提案が書いていてコードがしない 6 件（H-0110「本 PR で直さず Issue にするもの」）、HISTORY の Status 行のずれ 5 件以上、H-0080 の entry に入った無関係な節。
