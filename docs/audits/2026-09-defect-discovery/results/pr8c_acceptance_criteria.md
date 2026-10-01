# PR 8c — 完了基準（レビューを開く前に書いた、2026-10-01）

計画 Revision 6 §12.5。**Proposal は無い**（計画 §3 の行 8c が `—`。公開 API・保存形式・Config に触れない、
監査の計測器と `tests/test_docs` の単体テストだけの PR）。ただし計画 §8 は「修理が満たすべき仕様」として
残っているので、仕様の変更は §8 に **Revision 7** として書く。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。**

---

## 0. これは何で、何ではないか

- 計画 §8 の完了測定器（`phase3_gap.py` + `phase3_manifest.json` + 単体テスト）を **出荷する**。PR 0 が
  `instruments/deferred/` に回収したまま保留したもので、README が「このまま出すと、正しく直った issue を
  未完と報告する（DC7）」と書いている。
- 実測（`results/pr8c_measurements.txt`）で、保留の理由は README の 5 項目より多かった。**archive のまま
  動かすと全行が誤判定になる** 原因が 2 つ（命題 6 の `closedByPullRequestsReferences` が全 25 件で空、
  worktree に生成ファイル `lizyml/_version.py` が無く `import lizyml` が失敗）。加えて manifest の行は
  ほぼ全部が古い（テスト名・母集団・行の集合）。**manifest は修理せず、各 PR の完了基準とマージされた
  テストから作り直す。**
- **対象外**: 計測した結果として見つかる各 issue の残作業の修正（見つかれば起票する）。#271 自体（PR 9）。
  CI でこの計測器を回すこと（worktree と pytest を回す run tool のまま。§8 の決定どおり）。
- **ラウンド予算 8。** round 2 の前に absolute monitor、round 3 以降は relational monitor。
- **指摘の分類**: B1 = 受け入れ基準を満たさない欠陥、B2 = 基準の欠落、B3 = 文書の事実誤認（管理者の要望により
  blocking）、B4 = 非 blocking の改善提案。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `33a3f6e`（PR 8b = #316）、ブランチ `chore/phase3-pr8c-completion-instrument` |
| 契約 | 計画 §8（命題 1-6）、§3 の表（行の集合）、`instruments/deferred/README.md`（保留の理由） |
| 実測 | `results/pr8c_measurements.txt`（各計測器の出力は同じディレクトリの `pr8c_*.txt`） |

## 2. 作り直す manifest（25 行。行の集合は §3 の Fixes 列と Refs 列から導出）

`github_prs` はその issue を直した PR（複数可）。before の木 = 最も古い `github_prs` のマージコミットの第 1 親。
母集団: 「node」はその node の収集数を宣言数と比べる行、「note」はパラメータ化でない理由を書く行。
`derived` はコードから数を読む snippet があり、develop で実行した値（`results/pr8c_derivations.txt`）。

| issue | 計画 PR | github_prs | disposition | テスト | 母集団 | 数 | derived |
|---|---|---|---|---|---|---|---|
| 258 | 3 | 292 | regression | `test_tuning/test_direction_reconciliation.py` | node `::test_inferred_direction_selects_correct_extremum` | 22 | 22 |
| 259 | 5 | 305 | regression | `test_features/test_pipeline_conformance.py` | note（抽象メソッド 4 を 1 つの最小 subclass が実装する） | 4 | 4 |
| 260 | 5 | 305 | regression | `test_features/test_unseen_policy.py`, `test_codegen/test_unseen_policy_codegen.py` | node `::test_every_policy_is_observable_end_to_end` | 3 | 3 |
| 261 | 1 | 275 | regression | `test_estimators/test_lightgbm_parameter_names.py` | node `::test_train_param_names_are_lightgbm_names`（タスクごとに smart を全部有効にする） | 3 | 3 |
| 262 | 1, 3d | 275, 300 | regression | `test_tuning/test_search_space_name_validation.py` | node `::test_search_space_name_is_gated`（6 セルは strict xfail = #299） | 27 | 27 |
| 263 | 6 | 310 | regression | `test_core/test_error_code_raising.py`, `test_core/test_error_code_population.py`, `test_docs/test_error_code_docs.py` | node `::test_member_is_raised` | 19 | 19 |
| 264 | 2 | 278 | regression | `test_core/test_fit_params_override.py` | note（優先順位 3 段をそれぞれ名前のあるテストが確かめる） | 3 | — |
| 265 | 0, 3d | 274, 300 | decision-only | `test_training/test_inner_valid_purge_embargo.py` | note（pin。修正前も後も green） | 1 | — |
| 266 | 0 | 274 | regression | `test_docs/test_declared_versions.py` | node `::test_declared_version_matches_code`（文書の記載箇所。id に行番号を含む） | 7 | — |
| 267 | 7 | 312 | regression | `test_data/test_leakage_validator_unchecked_column.py` | node `::test_unchecked_column_is_reported`（3 形 × 5 dtype × 2、テストが宣言） | 30 | — |
| 268 | 8 | 314 | regression | `test_config/test_knob_reachability.py` | note（74 個の census を registry が分類しきる 1 node） | 74 | — |
| 269 | 4 | 302 | regression | `test_training/test_cv_refit_parity.py` | note（2 つの `fit` の引数の和集合を 1 node が確かめる） | 7 | 7 |
| 270 | 1, 2（Refs） | 275, 278 | partial | `test_estimators/test_param_behavioral_effect.py`, `test_core/test_config_propagation.py` | note（スイート全体の性質。179 中 2 を修理） | 179 | — |
| 271 | 9 | （なし） | regression | `test_docs/test_proposal_blueprint_coverage.py`（未作成） | note（PR 9 がテストと一緒にこの行の母集団を書く） | — | — |
| 272 | 6 | 310 | regression | `test_config/test_config_version_entry_paths.py` | node `::test_entry_path_by_version`（8 入口 × 2 版、テストが宣言） | 16 | — |
| 277 | 3c | 296 | regression | `test_core/test_calibration_params_reach.py`, `test_calibration/test_calibration_param_contract.py`, `test_calibration/test_platt_mle.py`, `test_codegen/test_calibration_params_codegen.py` | note（登録された calibrator ごとの契約を 1 node が確かめる） | 3 | 3 |
| 279 | 3 | 292 | regression | `test_tuning/test_dimension_consumption.py` | node `::test_every_smart_owned_native_spelling_refused_before_study` | 18 | — |
| 281 | 8b | 316 | regression | `test_persistence/test_reporting_surfaces_survive_load.py`, `test_persistence/test_applied_training_params.py` | note（22 面 × 4 構成 × 5 lifecycle = 440 セルを 1 node が回す） | 440 | — |
| 282 | 3 | 292 | regression | `test_tuning/test_dimension_consumption.py` | node `::test_unconsumed_training_dimension_refused`（テストが宣言した 3 名） | 3 | — |
| 284 | 2c | 291 | decision-only | `test_core/test_param_domain_cycles.py`, `test_core/test_param_domain.py` | note（構造の統合。修正前も同じ受理集合） | 1 | — |
| 285 | 2b | 290 | regression | `test_core/test_param_refusal_origin.py` | node `::test_direct_adapter_refuses_duplicate_spellings` | 4 | — |
| 286 | 2b | （なし） | not-planned | （なし） | — | — | — |
| 287 | 2c | 291 | decision-only | `test_tuning/test_literal_domain_contract.py` | node `::test_numpy_choices_are_refused_at_the_named_entrance` | 5 | — |
| 288 | 2 | 278 | regression | `test_core/test_fit_params_override.py` | note（4 つの identity overlay の継ぎ目と、それを数える 1 node） | 4 | — |
| 306 | 5b | 308 | regression | `test_estimators/test_feval_probabilities.py` | node `::test_feval_value_is_the_metric_of_lightgbm_predictions` | 57 | 57 |

disposition を regression 以外にした行の理由（manifest の `justification` に書く）:

- **#265 decision-only**: 欠陥は BLUEPRINT が両立しない 2 つの規則を書いていたこと。実装は採用した側に既に
  従っており、pin は修正前も後も green（実測: マージ直前 `3abb6c4` で 9 passed）。
- **#284 decision-only**: 受理集合を 3 つの構造の walk で述べていたものを 1 つにまとめた統合。修正前も受理集合は
  同じ（実測: マージ直前 `2436a66` で 2 passed）。循環 mapping のテストが落ちたのは PR の最初の候補に対してで、
  基点ではない（`results/pr2c_acceptance_criteria.md`）。
- **#287 decision-only**: `results/pr2c_acceptance_criteria.md`「The original #287 symptom is historical and already
  fixed at this base」。PR は残った方針を記録してテストにした（実測: マージ直前で 8 passed）。
- **#270 partial**: 179 件の hollow test のうち 2 件を PR 1 / PR 2 で修理。残りは別スコープで、issue は開いたまま。
- **#286 not-planned**: PR 2b で「計画しない」として閉じた（stateReason `NOT_PLANNED`）。テストは無い。

## 3. 計測器の契約（計画 §8 の Revision 7）

| 命題 | archive の定義 | Revision 7 | 理由（実測） |
|---|---|---|---|
| 行の集合 | manifest に書いた 15 行 | manifest のキー = 計画 §3 の Fixes 列と Refs 列にある issue の集合。違えば manifest エラー | §3 は 25 件を割り当てている。副産物 10 件を測らずに完了と言えば DC5（測定 1） |
| p1 | テストが after にある | 同じ | — |
| p2 | 全行共通の `5712f41` でテストが FAILED を 1 つ以上出す | before = 最も古い `github_prs` のマージコミットの第 1 親。FAILED が 1 つ以上、**または** collection error がすべて「`lizyml` の名前が無い」ImportError / ModuleNotFoundError で、その名前が after では import できる（`red-by-missing-api` と報告に明記）。それ以外の collection error は赤と数えない | 5712f41 では 9 行が collection error。マージ直前なら 3 行（#264 / #288 / #277。名前は修正が足したもの）だけ（測定 5） |
| p2 の準備 | 行のテストファイルだけを before の木に写す | 加えて after の `tests/` にある補助モジュール（`_*.py`）と `__init__.py` を写し、終わったら元に戻す | 補助モジュール 5 個と `tests/test_docs/__init__.py` が無いと、欠陥と無関係に collection error になる（測定 5） |
| p3 | after で通る | 同じ（strict xfail は pytest が exit 0 を返すので通る扱い） | #262 の 6 セル（測定 4） |
| p4 | `population_test` の収集数 = 宣言数 | 同じ。加えて `population` を省いて `derived_from` だけを持てる（収集数 = p5 の値）。`population_note` の行は名前を挙げたテストが 1 つ以上収集されること | #271 の母集団は run の間に増える（測定 6） |
| p5 | 全行に `derived_from` が必須 | 任意。ある行は値 = 宣言数（または収集数）。無い行は「テストが宣言した集合」であることを `population_note` か行の `population_source` に書く | 6 行の母集団はテストが宣言したリストで、コードから導出できない。必須にすると満たせない宣言（DC7）（測定 4） |
| p6 | `closedByPullRequestsReferences` に `github_pr` が含まれる | issue が CLOSED で stateReason が COMPLETED（not-planned 行は NOT_PLANNED）、かつ各 `github_prs` が MERGED、マージコミットが after の祖先、本文が `#N` を桁境界つきで引用、issue の closedAt ≥ PR の mergedAt。partial 行は issue が OPEN | 全 25 件で `closedBy` が空（DC7）。手で閉じた run の記録として読めるのはこの 4 条件（測定 2） |
| 実行環境 | `.venv/bin/python` を `cwd=<worktree>` で | 加えて各 worktree に `lizyml/_version.py` を写す。interpreter は `absolute()` で `resolve()` しない。derivation は `-c` でだけ実行する。`PYTHONDONTWRITEBYTECODE=1` | 測定 3 |
| 判定 | COMPLETE / PARTIAL / INCOMPLETE / UNKNOWN | 加えて NOT-PLANNED。`github_prs` が空の regression 行は INCOMPLETE（「直した PR がまだ無い」）で、manifest エラーにしない | #271 は PR 9 の前で、行が空であることが正しい状態 |

## 4. 受け入れ基準 → 証拠の対応表

| # | 基準 | 種別 | 証拠 |
|---|---|---|---|
| 1 | 出荷物: `instruments/phase3_gap.py`、`instruments/phase3_manifest.json`（§2 の 25 行）、`tests/test_docs/test_phase3_gap.py`。`deferred/` の 3 ファイル（`phase3_gap.py` / `phase3_manifest.json` / `test_phase3_gap.py`）は移すか置き換え、`check_derivations.py` は p5 が本体に入るので消す。`deferred/README.md` は「PR 8c で出荷した」と書き、archive の manifest が `.gitignore` の `*.json` のため一度もコミットされていなかったこと（測定 7）を訂正する。`.gitignore` に `!docs/audits/**/*.json` を足し（`git add -f` は pre-commit hook が拒否する）、出荷する manifest が `git ls-files` に載っていることを確かめる | — | diff、`git ls-files` |
| 2a | manifest の文法: 必須キー、disposition の値、regression 以外の justification、`population_test` か `population_note`、`github_prs` の型（正の整数のリスト）、数の型（正の整数）。違反はすべて `ManifestError` | 単体（偽の manifest） | `tests/test_docs/test_phase3_gap.py::test_a_malformed_row_is_refused[*]` |
| 2b | 出荷した manifest は文法を通り、キーの集合 = 計画 §3 から導出した集合（25） | 単体（実ファイル） | `::test_the_shipped_manifest_covers_the_plan_issue_set` |
| 2c | §3 の表の読み取りが閉じている: 列数が合わない行、Fixes / Refs 列が無い表は `ManifestError` | 単体 | `::test_the_plan_table_reader_refuses_a_malformed_table[*]` |
| 3 | 収集行の文法が閉じている: node id でない `::` 行と重複 id は `ManifestError`。パラメータ id に `::` や空白を含む行を数え損なわない | 単体 | `::test_node_ids_are_parsed_with_a_closed_grammar[*]` |
| 4a | p2: FAILED が 1 つ以上なら赤 | 単体（偽の runner） | `::test_p2_red_by_a_failed_node` |
| 4b | p2: collection error が `lizyml` の欠けた名前だけで、after では import できるなら `red-by-missing-api` | 単体 | `::test_p2_red_by_a_missing_lizyml_name` |
| 4c | p2: その名前が after でも import できない、`lizyml` 以外の名前、ImportError 以外の collection error、FAILED も error も無い（全部通る）ときは赤でない | 単体 | `::test_p2_not_red[*]` |
| 4d | p2: before は最も古い `github_prs` のマージの第 1 親。写したファイルは実行後に元に戻る（before に元からあったファイルは元の中身に、無かったファイルは消す） | 単体 | `::test_p2_runs_at_the_first_parent_of_the_earliest_fix`, `::test_p2_restores_the_before_tree` |
| 5 | p4 / p5: 収集数 ≠ 宣言数、derived ≠ 宣言数、`population` を省いた行で収集数 ≠ derived、derivation が整数を出さない／失敗する → それぞれ INCOMPLETE か UNKNOWN（どちらも完了に数えない） | 単体 | `::test_population_mismatch_is_not_complete[*]` |
| 6a | p6: 4 条件のどれか 1 つを外した偽の GitHub 応答（PR が未マージ、祖先でない、本文が引用しない、マージ前に閉じた、stateReason が違う）でそれぞれ COMPLETE にならない。`#26` は `#263` を引用したことにならない（DC2） | 単体 | `::test_p6_requires_every_condition[*]`, `::test_issue_citation_is_digit_bounded` |
| 6b | partial 行は issue が OPEN のときだけ PARTIAL、not-planned 行は NOT_PLANNED で閉じたときだけ NOT-PLANNED | 単体 | `::test_partial_and_not_planned_rows[*]` |
| 7 | runner: 各 worktree に `_version.py` を写す、interpreter を resolve しない、derivation を `-c` で実行する、`PYTHONDONTWRITEBYTECODE=1`、worktree の HEAD が要求した SHA と違えば `ManifestError` | 単体（subprocess を差し替え） | `::test_runner_prepares_each_worktree`, `::test_runner_refuses_a_worktree_at_the_wrong_commit` |
| 8 | 判定の算術: INCOMPLETE と UNKNOWN が 1 つでもあれば exit 1、無ければ 0。PARTIAL と NOT-PLANNED は完了を妨げない | 単体 | `::test_exit_code_counts_incomplete_and_unknown` |
| 9 | **実データでの実行**: `phase3_gap.py --after origin/develop`（PR 8c をマージする直前の develop）で、UNKNOWN 0、#271 だけ INCOMPLETE、#270 PARTIAL、#286 NOT-PLANNED、他の 22 行は COMPLETE。exit 1（#271 のため）。出力を `results/pr8c_completion_<sha>.txt` に保存する | 実行 | 計測器の出力 |
| 10 | 変異の確認: (a) manifest の 1 行の数を 1 ずらす → その行が INCOMPLETE、(b) `github_prs` を関係のない PR に替える → p6 で INCOMPLETE、(c) `_version.py` を写さない → UNKNOWN か INCOMPLETE が出て exit 1。どれも COMPLETE のまま通らない | 実行 | 計測器の出力（`results/pr8c_mutations.txt`） |
| 11 | 文書: 計画 §8 に Revision 7 の banner（§3 の表の内容）、§3 の行 8c、`MANIFEST.md` の「覆われていない保証」、`deferred/README.md`、`results/pr2_NEXT.md`。**完了の判定は PR 9 の後にこの計測器を回すことで行う**と §3 の行 8c に書く | — | diff |
| 12 | 品質ゲート: ruff / ruff format（`tests/`。`docs/audits/**` は ruff の対象外）/ mypy `lizyml/`（変更なし）/ フルスイート / CI | — | PR 本文 |

## 5. 証拠の限界

- 計測器は「出荷されたテストが、宣言した母集団を数え、直す前に赤で後に緑で、issue が直した PR の後に閉じた」
  ことを確かめる。テストが issue の主張を正しく捉えているかは確かめない（それは各 PR のレビューと close
  レビューが見た）。
- `red-by-missing-api` は「修正が足した API が無いと、テストが走りすらしない」ことしか示さない。#264 / #288 /
  #277 の振る舞いの RED は各 PR の RED コミットで確かめられたが、squash merge でその SHA は develop に残らない。
- 母集団の数がテスト自身の宣言による行（#264 #265 #266 #267 #268 #272 #279 #281 #282 #284 #285 #287 #288）は、
  コードが変わって母集団が増えても manifest は気づかない。気づくのはテスト側の census（例: #268 の
  `test_registry_classifies_exactly_the_census`）がある行だけ。
