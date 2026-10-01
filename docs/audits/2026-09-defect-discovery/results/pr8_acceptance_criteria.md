# PR 8 — 完了基準（レビューを開く前に書いた、2026-10-01）

計画 Revision 6 §12.5。提案は **H-0108**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 証拠列のテスト名は実装前に宣言した
名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#268**: 公開クラスの既定値付き構成値（74 個）がどこから値を得るかを全数で分類し、Config から来ない
  ものの出どころを BLUEPRINT に書く。**#268 と計画の分類（名前の照合）は誤りで、計画の「9 個を公開」は
  置き換える**（9 個ともすでに Config から届く。H-0108 の目的に実測）。
- **production コードは変えない。** Config に新しいキーを足さない。
- **対象外**: 届かない 2 個（`verbose_eval`、`StratifiedKFoldSplitter.shuffle`）を公開すること（方針として
  書く。H-0108 代替案 1）。
- **ラウンド予算 8。** round 2 の前に absolute monitor、round 3 以降は relational monitor。
- **指摘の分類**: B1 = 受け入れ基準を満たさない欠陥、B2 = 基準の欠落、B3 = 文書の事実誤認（管理者の要望により
  blocking）、B4 = 非 blocking の改善提案。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `91a698b`（PR 7 = #312） |
| 契約 | `HISTORY.md` **H-0108**、#268 |
| 実測 | `results/pr8_measurements.txt`（`instruments/pr8_write_measurements.py` で再生成） |

## 2. 修正前の実測（`91a698b`）

| 何を | 結果 |
|---|---|
| 母集団（AST） | 74 個（#268 の数と同じ） |
| #268 が「届かない」とした 25 個 | 実行すると config 12（分割器 6、指標 3、`early_stopping_rounds`、`unseen_policy` 2）、api 3（`Tuner` の 3 つ）、derived 4、policy 1（`verbose_eval`）、internal 5 |
| #268 に無い届かない構成値 | `StratifiedKFoldSplitter.shuffle`（構築箇所が常に `True`） |
| 全体の分類 | config 52 / api 5 / derived 10 / policy 2 / internal 5 |
| #268 の「使われていない 4 オプション」 | 2 つ（`detect_boundary(threshold=)`、`Model(data=)`）は今はテストで使われている。`Model.importance_plot(top_n=)` と `Model.plot_learning_curve(metrics=)` は使われていない |

## 3. 受け入れ基準 → 証拠の対応表

| # | 基準（H-0108） | 種別 | 証拠 |
|---|---|---|---|
| 1 | 台帳のキー = AST の母集団、母集団は 60 個以上、種類は 5 つのどれか | ガード（新しい構成値で失敗する） | `tests/test_config/test_knob_reachability.py::test_registry_classifies_exactly_the_census` |
| 2a | 台帳の config 行の集合 = 実行セルの集合 | ガード | `::test_every_config_row_has_an_executed_cell` |
| 2b | 52 セル: Config に書いた既定でない値がコンストラクタに届く（`LGBMAdapter.params` / `IsotonicCalibrator.params` は書いた項目を含む） | ガード | `::test_config_value_reaches_the_constructor[*]` |
| 3 | api 5 行が本物の呼び出しで届く | ガード | `::test_api_rows_reach_the_constructor` |
| 4 | `BLUEPRINT.md` §5.5 の表の行と種類 = 台帳の config 以外の 22 行 | RED（§5.5 が無い） | `::test_rows_outside_config_are_stated_in_blueprint` |
| 5a | derived: `task` の 3 行が設定の task に従う | ガード | `::test_derived_task_rows_follow_the_config[*]` |
| 5b | derived: `TimeHoldoutInnerValid.gap` は自動解決で outer の gap、`time_holdout` 明示で 0 | ガード | `::test_time_holdout_gap_is_the_outer_gap_only_when_resolved_automatically` |
| 5c | policy の 2 行はライブラリが既定から変えない | ガード | `::test_policy_rows_are_never_passed_by_the_library` |
| 6 | `Model.importance_plot(top_n=1)` は 1 特徴、`Model.plot_learning_curve(metrics=["rmse"])` は rmse だけ | ガード（テストの欠落を埋める） | `tests/test_plots/test_model_plot_options.py::*` |
| 7 | 計画 §3 の行 7 / 8、§PR 8 の訂正、§6 の #268 の行、§7 の firing rate、`pr2_NEXT.md`、#268 への訂正コメント | — | diff / Issue コメント |
| 8 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | — | PR 本文 |

このPRの RED は 4 だけである。他の行は、今日すでに成り立っている性質を初めて固定するガード（#268 の主張が
誤りだったので、直す対象が無い）。これらのガードが空振りしないことは、レビューで変異（台帳から 1 行消す、
Config の値を書き換える、§5.5 の行を消す）を当てて確かめる。
