# PR 8c — 完了基準（レビューを開く前に書き、設計レビュー round 1 / 2 で改訂した、2026-10-01）

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
| 262 | 1, 3d | 275, 300 | regression | `test_tuning/test_search_space_name_validation.py` | node `::test_search_space_name_is_gated`（6 セルは strict xfail = #299。ファイル全体の xfailed 9 を node id で `expected_nonpass` に宣言） | 27 | 27 |
| 263 | 6 | 310 | regression | `test_core/test_error_code_raising.py`, `test_core/test_error_code_population.py`, `test_docs/test_error_code_docs.py` | node `::test_member_is_raised` | 19 | 19 |
| 264 | 2 | 278 | regression（`red_mutation`） | `test_core/test_fit_params_override.py` | note（優先順位 3 段をそれぞれ名前のあるテストが確かめる） | 3 | — |
| 265 | 0, 3d | 274, 300 | decision-only | `test_training/test_inner_valid_purge_embargo.py` | note（pin。修正前も後も green） | 1 | — |
| 266 | 0 | 274 | regression | `test_docs/test_declared_versions.py` | node `::test_declared_version_matches_code`（文書を走査して見つけた記載箇所。id に行番号を含む） | 7 | — |
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
| 284 | 2c | 291 | decision-only | `test_core/test_param_domain_cycles.py`, `test_core/test_param_domain.py` | note（構造の統合。修正前も同じ受理集合。`test_param_domain.py` の skipped 230 を理由つきで `expected_nonpass` に宣言） | 1 | — |
| 285 | 2b | 290 | regression | `test_core/test_param_refusal_origin.py` | node `::test_direct_adapter_refuses_duplicate_spellings` | 4 | — |
| 286 | 2b | （なし） | not-planned | （なし） | — | — | — |
| 287 | 2c | 291 | decision-only | `test_tuning/test_literal_domain_contract.py` | node `::test_numpy_choices_are_refused_at_the_named_entrance` | 5 | — |
| 288 | 2 | 278 | regression（`red_mutation`） | `test_core/test_fit_params_override.py` | note（4 つの identity overlay の継ぎ目と、それを数える 1 node） | 4 | — |
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

## 3. 計測器の契約（計画 §8 の Revision 7。設計レビュー round 1 / 2 で p2 / p3 / p6 を改訂）

| 命題 | archive の定義 | Revision 7 | 理由（実測） |
|---|---|---|---|
| 行の集合 | manifest に書いた 15 行 | manifest のキー = 計画 §3 の Fixes 列と Refs 列にある issue の集合。違えば manifest エラー | §3 は 25 件を割り当てている。副産物 10 件を測らずに完了と言えば DC5（測定 1） |
| p1 | テストが after にある | 同じ | — |
| p2 の判定 | FAILED が 1 つ以上 | **FAILED の node が 1 つ以上** のときだけ赤。collection error は赤と数えない。赤は次の (a) か (b) のどちらかで示す | round 1 指摘 1、round 2 指摘 2 |
| p2 (a) before の木 | 全行共通の `5712f41` に、行のテストファイルだけを写す | before = 最も古い `github_prs` のマージコミットの第 1 親。行のテスト、after の `tests/` の補助モジュール（`_*.py`）と `__init__.py`、**after にあって before に無い `lizyml/` のファイルだけ** を写し、実行後に元に戻す。ただし before の `lizyml/` のどれかがその新しいモジュールを **import する**（`try` の中も含む。相対 import を解決し、`import_module` に渡る文字列も数える。ast で読む）ときは、写すと直す前の振る舞いが変わりうるので (a) は不成立 | 5712f41 では 9 行が collection error、マージ直前なら 4 行（測定 5）。#277 は新しいモジュールを写すと 102 failed（測定 8）。round 2 指摘 4 の反例（`try: from lizyml.new import Y except ImportError: Y = 0`）はこの検査で拒否される。測った全 23 行で、新しいモジュールを import する before のコードは 0 件（測定 10） |
| p2 (b) 欠陥を戻す変異 | — | 行が `red_mutation`（`file` / `old` / `new` / `fix_text` / `why`）を宣言する。計測器は次をすべて確かめる: `old` が after のそのファイルにちょうど 1 回ある、`fix_text` は `old` に含まれ `new` に含まれない、`fix_text` は修正 PR のマージが追加した行（`git diff <merge>^1 <merge> -- <file>` の `+` 行）の部分文字列、変異を当てた after でテストに FAILED があり collection error が無い。判定は `COMPLETE-RED-BY-MUTATION` として COMPLETE と別に数える | #264 / #288 は修正が既存の `_model_factories.py` に足した名前を import しており、どの before の木でも走らない。#278 は修正のコミットがテストのコミットより先（測定 8）。変異で #264 は 96 failed、#288 は 13 failed（測定 10）。round 2 指摘 2 の反例（修正が足した名前の未使用 import と自明な assert）は、変異を当てても通るので赤でない。round 1 / 2 の waiver はこれで置き換えた |
| p3 | after で通る（exit 0） | 行のテストの各 node を JUnit で読み、node id を復元する。passed でない node はすべて、行の `expected_nonpass` のちょうど 1 項目（`test` = node id か関数の id、`outcome` = skipped / xfailed、`reason` = メッセージの一部、`count` = 正確な数）に当たる。各項目の数はちょうど一致。JUnit の件数 = 収集した node の数。pytest の exit が 0 | round 1 の反例（skip した `assert False`）と round 2 の反例（xfail を別の node に移しても数は同じ）は通らない。#262 は 9 つの node id で xfailed（#299）、#284 は 1 関数の skipped 230（理由 `holds a mapping`）を宣言する（測定 9、10） |
| p4 | `population_test` の収集数 = 宣言数 | 同じ。`population` を省き `derived_from` だけでもよい（収集数 = p5 の値）。`population_note` の行は、名前を挙げたテストが 1 つ以上収集され p3 を満たすこと。note 行の数は **宣言値** として報告し、収集数とは区別する | 母集団がテスト外から決まる行は、固定値でなく導出で持つほうが古い数字を残さない（測定 6） |
| p5 | 全行に `derived_from` が必須 | 任意。ある行は値 = 宣言数（または収集数） | 5 行の母集団はテストが宣言したリストで、コードから導出できない。必須にすると満たせない宣言（DC7）（測定 4） |
| p6 | `closedByPullRequestsReferences` に `github_pr` が含まれる | issue が CLOSED で stateReason が COMPLETED（not-planned 行は NOT_PLANNED）、各 `github_prs` が MERGED、マージコミットが after の祖先、本文が `#N` を引用（桁境界つき、`owner/repo#N` は数えない）。加えて、行が **固定した close コメント**（`closure_comment` = コメントの id）がその issue にあり、書かれた時刻が各 PR の mergedAt 以後かつ closedAt 以前で、各 PR を `#PR` か題名そのもので名指す。partial 行は issue が OPEN | 全 25 件で `closedBy` が空（DC7、測定 2）。「最後のコメント」を読む round 1 の案は、修正の宣言の後に謝辞が付いた正しい記録を落とした（round 2 指摘 3）ので、コメントを固定する。23 件の固定コメントはすべて各 PR のマージ後に書かれている（測定 10）。**コメントが修正を肯定していること** は計測器には読めない。manifest の宣言で、本 PR のレビューが 23 件を読んで事実確認する（§5） |
| 実行環境 | `.venv/bin/python` を `cwd=<worktree>` で | 加えて各 worktree に `lizyml/_version.py` を写す。interpreter は `absolute()` で `resolve()` しない。derivation は `-c` でだけ実行する。`PYTHONDONTWRITEBYTECODE=1` | 測定 3 |
| 判定 | COMPLETE / PARTIAL / INCOMPLETE / UNKNOWN | 加えて `COMPLETE-RED-BY-MUTATION` と `NOT-PLANNED`。最終行は種別ごとに数を出し、COMPLETE と合算しない。exit は INCOMPLETE と UNKNOWN が 0 のときだけ 0 | round 1 非 blocking 3 |

命題が適用されない行（round 1 非 blocking 1）:

| 行 | p1 | p2 | p3 | p4 | p5 | p6 | 判定 |
|---|---|---|---|---|---|---|---|
| not-planned（#286） | — | — | — | — | — | NOT_PLANNED で閉じている（固定コメントなし） | NOT-PLANNED |
| partial（#270） | ○ | — | ○ | note | — | OPEN | PARTIAL |
| decision-only（#265 #284 #287） | ○ | — | ○ | ○ | 任意 | ○ | COMPLETE |
| `github_prs` が空の regression（#271） | ○（無ければ不成立） | — | — | — | — | — | INCOMPLETE（「直した PR がまだ無い」。manifest の文法は通す） |

## 4. 受け入れ基準 → 証拠の対応表

証拠のテストはすべて `tests/test_docs/test_phase3_gap.py`。

| # | 基準 | 種別 | 証拠 |
|---|---|---|---|
| 1 | 出荷物: `instruments/phase3_gap.py`、`instruments/phase3_manifest.json`（§2 の 25 行）、`tests/test_docs/test_phase3_gap.py`。`deferred/` の 3 ファイル（`phase3_gap.py` / `phase3_manifest.json` / `test_phase3_gap.py`）は置き換え、`check_derivations.py` は p5 が本体に入るので消す。`deferred/README.md` は「PR 8c で出荷した」と書き、archive の manifest が `.gitignore` の `*.json` のため一度もコミットされていなかったこと（測定 7）を訂正する。`.gitignore` に `!docs/audits/**/*.json` を足し（`git add -f` は pre-commit hook が拒否する）、出荷する manifest が `git ls-files` に載っていることを確かめる | — | diff、`git ls-files` |
| 2a | manifest の文法: 必須キー、disposition の値、regression 以外の justification、`closure_comment`（PR のある閉じた行）、`population_test` か `population_note`、`github_prs` の型、数の型、`expected_nonpass` の項目（outcome は skipped / xfailed だけ、count は正の整数）、`red_mutation` の項目（`fix_text` は `old` にあり `new` に無い、regression 行だけ）。違反はすべて `ManifestError` | 単体 | `::test_a_malformed_row_is_refused[*]`（24 ケース）, `::test_a_well_formed_row_is_accepted` |
| 2b | 出荷した manifest は文法を通り、キーの集合 = 計画 §3 から導出した集合（25） | 単体（実ファイル） | `::test_the_shipped_manifest_covers_the_plan_issue_set`, `::test_the_shipped_manifest_is_valid_json_with_every_row_validated` |
| 2c | §3 の表の読み取りが閉じている | 単体 | `::test_the_plan_table_reader_reads_both_columns`, `::test_the_plan_table_reader_refuses_a_malformed_table[*]` |
| 3 | 収集行の文法が閉じている: node id でない `::` 行と重複 id は `ManifestError`。パラメータ id に `::` や空白を含む行を数え損なわない | 単体 | `::test_node_ids_are_counted[*]`, `::test_node_ids_are_parsed_with_a_closed_grammar[*]` |
| 4a | p2 (a): FAILED が 1 つ以上なら赤。全部通る、collection error だけ、は赤でない。before は最も古い修正のマージの第 1 親 | 単体 | `::test_p2_red_by_a_failed_node`, `::test_p2_not_red[*]`, `::test_p2_runs_at_the_first_parent_of_the_earliest_fix` |
| 4b | p2 (a) の準備: before に無い `lizyml/` のファイルだけを写し、before にあるものは写さない。テスト側の補助を写す。実行後に before が元に戻る | 単体（一時ディレクトリの木） | `::test_p2_stages_only_new_package_files`, `::test_p2_restores_the_before_tree` |
| 4c | p2 (a) の参照検査: 新しいモジュールを import する before のコード（`try` の中、相対 import、親パッケージからの相対、`import_module` の文字列）を見つける。同名の無関係な import（`importlib.metadata.version`）とコメントの言及は数えない。見つかれば (a) は不成立。round 2 の反例（guarded import）を実際に拒否する | 単体 | `::test_new_module_references_are_resolved_statically[*]`, `::test_a_new_module_the_before_code_imports_is_refused` |
| 4d | round 1 の反例: 新しいモジュールの未使用 import と自明な assert は、写した木で通り、赤でない（実 pytest） | 単体（実 pytest） | `::test_an_unused_import_of_a_new_module_is_not_red` |
| 4e | p2 (b): 変異でテストが落ちれば `COMPLETE-RED-BY-MUTATION`。after のファイルは元に戻る。テストが通る、`fix_text` が修正 PR の追加行に無い、変異で collection が壊れる → INCOMPLETE。`old` が 1 回でない → `ManifestError`（UNKNOWN） | 単体 | `::test_a_mutation_that_reddens_the_tests_is_its_own_verdict`, `::test_a_mutation_is_not_red_evidence_when[*]`, `::test_a_mutation_must_match_exactly_once` |
| 4f | round 2 の反例: 修正が足した名前の未使用 import と自明な assert は、変異を当てても通り、赤でない（実 pytest） | 単体（実 pytest） | `::test_an_unrelated_test_is_not_red_under_a_mutation` |
| 4g | p3: 宣言の無い skip、別の node に移った xfail、別の理由の xfail、宣言より多い／少ない skip、失敗、収集したのに報告されない node → いずれも COMPLETE にならない。宣言どおりの xfail は通る。round 1 の反例（`@pytest.mark.skip` の `assert False`）は実 pytest で skipped と読まれ、説明されない | 単体（偽と実の両方） | `::test_p3_accounts_for_every_node[*]`, `::test_a_skip_in_a_real_run_is_seen`, `::test_junit_cases_carry_node_ids` |
| 5 | p4 / p5: 収集数 ≠ 宣言数、derived ≠ 宣言数、`population` を省いた行で収集数 ≠ derived → INCOMPLETE。derivation の失敗 → UNKNOWN。note 行の数は「declared」と表示 | 単体 | `::test_population_mismatch_is_not_complete[*]`, `::test_a_derived_only_row_compares_the_collection_with_the_derivation`, `::test_a_failing_derivation_is_unknown`, `::test_note_row_counts_are_reported_as_declared` |
| 6a | p6: 各条件を 1 つ外した偽の GitHub 応答（issue が開いている、stateReason が違う、PR 未マージ、祖先でない、本文が引用しない、本文が上位の番号だけ引用、固定コメントが PR のマージ前、固定コメントが close の後、固定コメントが issue に無い、固定コメントが PR を名指さない）でそれぞれ COMPLETE にならない | 単体 | `::test_p6_requires_every_condition[*]` |
| 6b | p6: 固定コメントの後に謝辞が付いても通る（round 2 の反例の逆側）。題名そのもので名指せば通り、題名の一部や 12 文字未満の題名では名指しにならない。`#26` は `#263` を、`owner/repo#263` は `#263` を引用しない | 単体 | `::test_p6_reads_the_pinned_comment_not_the_last_one`, `::test_p6_accepts_the_pr_named_by_its_exact_title`, `::test_titles_must_match_exactly[*]`, `::test_citations_are_digit_bounded[*]` |
| 6c | partial 行は issue が OPEN のときだけ PARTIAL、not-planned 行は NOT_PLANNED で閉じたときだけ NOT-PLANNED、decision-only 行は p2 を走らせない、`github_prs` が空の regression 行と after にテストが無い行は INCOMPLETE | 単体 | `::test_a_partial_row_needs_its_issue_open`, `::test_a_not_planned_row`, `::test_a_decision_only_row_skips_p2`, `::test_a_regression_row_without_a_fixing_pr_is_incomplete`, `::test_a_missing_test_file_is_incomplete` |
| 7 | runner: 各 worktree に `_version.py` を写す、interpreter を resolve しない、derivation を `-c` で cwd = worktree で実行する、`PYTHONDONTWRITEBYTECODE=1`、worktree の HEAD が要求した SHA と違えば `ManifestError` | 単体（subprocess を差し替え） | `::test_runner_prepares_each_worktree`, `::test_runner_refuses_a_worktree_at_the_wrong_commit` |
| 8 | 判定の算術と報告: INCOMPLETE と UNKNOWN が 1 つでもあれば exit 1。最終行は 6 種別を別々に数える | 単体 | `::test_exit_code_counts_incomplete_and_unknown[*]`, `::test_the_summary_keeps_each_verdict_separate` |
| 9 | **実データでの実行**: `phase3_gap.py --after origin/develop`（PR 8c をマージする直前の develop）で、UNKNOWN 0、#271 だけ INCOMPLETE、#270 PARTIAL、#286 NOT-PLANNED、#264 / #288 COMPLETE-RED-BY-MUTATION、他の 20 行は COMPLETE。exit 1（#271 のため）。出力を `results/pr8c_completion_<sha>.txt` に保存する | 実行（目標。実装後に観測する） | 計測器の出力 |
| 10 | 変異の確認（実データ、manifest か木を一時的に変えて実行）: (a) node 行の数を 1 ずらす → INCOMPLETE、(b) note 行の宣言数を derived と食い違わせる → INCOMPLETE、(c) `github_prs` を関係のない PR に替える → p6 で INCOMPLETE、(d) after の母集団の node に skip を付ける → p3 で INCOMPLETE、(e) #262 の xfail 宣言の 1 つを別の node id に替える → p3 で INCOMPLETE、(f) #264 の `red_mutation` を消す → INCOMPLETE、(g) #264 の `fix_text` を #278 が足していない文字列に替える → INCOMPLETE、(h) `closure_comment` を別のコメントの id に替える → INCOMPLETE、(i) `_version.py` を写さない → UNKNOWN か INCOMPLETE で exit 1。どれも COMPLETE のまま通らない | 実行 | 計測器の出力（`results/pr8c_mutations.txt`） |
| 11 | 文書: 計画 §8 に Revision 7 の banner（§3 の表の内容）、§3 の行 8c、`MANIFEST.md` の「覆われていない保証」、`deferred/README.md`、`results/pr2_NEXT.md`。**完了の判定は PR 9 の後にこの計測器を回すことで行う**と §3 の行 8c に書く | — | diff |
| 12 | 品質ゲート: ruff / ruff format（`tests/`。`docs/audits/**` は ruff の対象外）/ mypy `lizyml/`（変更なし）/ フルスイート / CI | — | PR 本文 |

## 5. 証拠の限界（round 3 は、ここに書いた範囲の外の反例を blocking にしない）

- **固定コメントが修正を肯定していること** は計測器には読めない。計測器が確かめるのは、そのコメントが issue に
  あり、時刻が各 PR のマージ以後かつ close 以前で、各 PR を番号か題名で名指すことまで。肯定であることは
  manifest の宣言で、本 PR のレビューが 23 件を読んで事実確認する。自然文の否定（「#296 は直していない」）を
  読み分ける規則は作らない（開いた文法になる。DC1 の注意書きのとおり）。
- **`red_mutation` が欠陥そのものを戻していること** も宣言である。計測器が確かめるのは、消すテキストが修正 PR の
  追加した行にあること、変異後にテストが落ち、collection は壊れないことまで。変異が広すぎて関係の無い理由で
  テストを落としている可能性は、`why` と変異の差分をレビューが読んで確かめる。
- **テスト側の補助モジュール**（`tests/_*.py`）は after のものを before に写す。p2 は「after のテストが before の
  システムに対して落ちるか」を問うので、補助はテストの一部として扱う。測った before の木で `tests/_helpers.py` は
  after と同一（測定 10）。
- **新しいモジュールを写す規則** が見ない経路: ast に現れない import（`exec` で組み立てた文字列、`__import__` に
  渡す連結した文字列）。測った 3 つの before の木に `pkgutil` / `import_module` / `iter_modules` /
  `entry_points` は無い（測定 8）。
- 2 つの PR を持つ行（#262、#265）の p2 は、最初の PR の前と後の全体の比較で、各 PR が単独で赤から緑にしたことは
  示さない。
- 母集団がテスト自身の宣言による行（#264 #265 #267 #268 #272 #279 #281 #282 #284 #285 #287 #288）は、コードが
  変わって母集団が増えても manifest は気づかない。気づくのはテスト側の census（例: #268 の
  `test_registry_classifies_exactly_the_census`）がある行だけ。#266 は文書を走査して数えるので、記載が増えれば
  収集数が変わり p4 が気づく。
- 計測器はテストが issue の主張を正しく捉えているかを確かめない（それは各 PR のレビューと close レビューが見た）。
