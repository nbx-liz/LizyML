# PR 8 — 完了基準（レビューを開く前に書き、設計レビュー round 1 で改訂した、2026-10-01）

計画 Revision 6 §12.5。提案は **H-0108**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 証拠列のテスト名は実装前に宣言した
名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#268**: 公開クラスの既定値付き構成値（74 個）がどこから値を得るかを全数で分類し、Config のキーが設定
  しないものの出どころを BLUEPRINT に書く。**#268 と計画の分類（名前の照合）は誤りで、計画の「9 個を公開」は
  置き換える**（9 個ともすでに Config から届く。H-0108 の目的に実測）。
- **production コードは変えない。** Config に新しいキーを足さない。
- **対象外**: policy の 2 個（`verbose_eval`、`StratifiedKFoldSplitter.shuffle`）を公開すること（方針として
  書く。H-0108 代替案 1）。`model.params` の `metric` で組み込みの指標の dict の引数が捨てられること（#313）。
- **ラウンド予算 8。** round 2 の前に absolute monitor、round 3 以降は relational monitor。
- **指摘の分類**: B1 = 受け入れ基準を満たさない欠陥、B2 = 基準の欠落、B3 = 文書の事実誤認（管理者の要望により
  blocking）、B4 = 非 blocking の改善提案。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `91a698b`（PR 7 = #312） |
| 契約 | `HISTORY.md` **H-0108**、#268、#313 |
| 実測 | `results/pr8_measurements.txt`（`instruments/pr8_write_measurements.py` で再生成） |

## 2. 修正前の実測（`91a698b`）

| 何を | 結果 |
|---|---|
| 母集団（AST） | 74 個（#268 の数と同じ） |
| #268 が「届かない」とした 25 個 | 実行すると、Config のキーの値が届くもの 12（分割器 6、指標 3、`early_stopping_rounds`、`unseen_policy` 2）、公開の引数 3（`Tuner` の 3 つ）、その他は台帳で config 1（`TimeHoldoutInnerValid.gap`）/ derived 3 / policy 1（`verbose_eval`）/ internal 5 に分類 |
| #268 に無い、変えられない構成値 | `StratifiedKFoldSplitter.shuffle`（構築箇所が定数 `True` を渡す） |
| 全体の分類 | config 60 / api 4 / derived 3 / policy 2 / internal 5（設計レビュー round 2 の指摘で `TimeHoldoutInnerValid.gap` と `StratifiedTimeHoldoutInnerValid.ratio` を derived から config へ） |
| #268 の「使われていない 4 オプション」 | `Model(data=)` は今は使われている。`detect_boundary(threshold=)` は既定値と同じ `0.05` でしか呼ばれていない。`Model.importance_plot(top_n=)` と `Model.plot_learning_curve(metrics=)` は使われていない |
| 設計レビュー round 1 で発見 | `model.params` の `metric` の `{"huber": {"delta": ...}}` は組み込みの huber になり `delta` が捨てられる（#313） |

## 3. 受け入れ基準 → 証拠の対応表

| # | 基準（H-0108） | 種別 | 証拠 |
|---|---|---|---|
| 1 | 台帳のキー = AST の母集団、母集団は 60 個以上、種類は 5 つのどれか | ガード（新しい構成値で失敗する） | `tests/test_config/test_knob_reachability.py::test_registry_classifies_exactly_the_census` |
| 2a | 台帳の config 行の集合 = 実行セルの集合 ∪ `Model.output_dir` | ガード | `::test_every_config_row_has_an_executed_cell` |
| 2b | 各セル（出どころが違う経路ごと: 明示 / 自動解決の inner valid、inner valid の gap（`time_series` / `purged_time_series` の自動解決）、`BlockedGroupInnerValid` の分類 / 回帰の代替経路の ratio、`split.random_state` が無いときの `training.seed`、指標の `evaluation.metrics` / feval、task の regression / binary）で、その経路が渡すべき値がコンストラクタに届く（`LGBMAdapter.params` / `IsotonicCalibrator.params` は書いた項目を含む）。多くのセルは既定でない値を設定し、既定値と同じ値になる分岐（回帰の task、自動解決の `stratify=False`）も確かめる。どの行にも既定でない値のセルが少なくとも 1 つある | ガード | `::test_config_value_reaches_the_constructor[*]`、`::test_every_config_row_has_a_non_default_witness` |
| 2c | `Model.output_dir` は Config の `output_dir` から、引数を渡せば引数から | ガード | `::test_output_dir_comes_from_config_unless_given` |
| 3 | api 4 行が本物の呼び出しで届く | ガード | `::test_api_rows_reach_the_constructor` |
| 4 | `BLUEPRINT.md` §5.5 が §5 の中（`# 5.` の下、`# 6.` の前）にあり、表の行と種類 = 台帳の config 以外の 14 行 | RED（§5.5 が無い） | `::test_rows_outside_config_are_stated_in_blueprint` |
| 5a | derived: クラス数は multiclass だけ渡し、regression と binary は `None` | ガード | `::test_derived_class_counts_are_set_only_for_multiclass` |
| 5b | derived: `collect_raw_scores` は fit では較正の有無、tune では（較正を設定しても）`False` | ガード | `::test_derived_collect_raw_scores_follows_calibration_on_fit_only` |
| 5c | `TimeHoldoutInnerValid.gap` は自動解決で `split.gap` / `purge_gap + embargo`（2b のセル）、`time_holdout` 明示で 0 | ガード | `::test_explicit_time_holdout_gets_no_gap` |
| 5d | policy の 2 行は Config を変えても固定値のまま | ガード | `::test_policy_rows_hold_their_fixed_value` |
| 6a | `Model.importance_plot(top_n=1)` は 1 特徴、`Model.plot_learning_curve(metrics=["rmse"])` は rmse だけ | ガード（テストの欠落を埋める） | `tests/test_plots/test_model_plot_options.py::*` |
| 6b | `detect_boundary` の `threshold` が判定を変える（同じ値が既定では端でなく、`0.2` では端） | ガード（テストの欠落を埋める） | `tests/test_tuning/test_detect_boundary_threshold.py::*` |
| 7 | 計画 §3 の行 7 / 8、§PR 8 の訂正、§6 の #268 の行、§7 の firing rate、`pr2_NEXT.md`、#268 への訂正コメント | — | diff / Issue コメント |
| 8 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | — | PR 本文 |

この PR の RED は 4 だけである。他の行は、今日すでに成り立っている性質を初めて固定するガード（#268 の主張が
誤りだったので、直す対象が無い）。設計レビュー round 1 は、自動解決の経路の seed を定数に変える変異が初版の
全テストを通ることを示した。改訂後のセルはその変異で失敗する（2b の `inner_auto_holdout` /
`group_kfold_auto` / `stratified_kfold_auto` のセル）。

RED の確認（`e64810e`、production コードは `91a698b` のまま、BLUEPRINT に §5.5 なし）: `tests/test_config/test_knob_reachability.py` は 84 passed / 1 failed で、失敗は RED と宣言した 4（`test_rows_outside_config_are_stated_in_blueprint`）だけ。他の行はガードで、修正前も通る。設計レビューは変異（自動解決の seed を定数に、自動解決の ratio を変える、§5.5 の行を消す・種類を変える、母集団を空にする、tune の較正セルに True を入れる）でガードが失敗することを確かめた。

修正後（BLUEPRINT §5.5 を入れたコミット）: フルスイート 8550 passed。失敗は環境起因の `test_version_matches_package_metadata` の 1 件。ruff / ruff format / mypy `lizyml/` は通過。
