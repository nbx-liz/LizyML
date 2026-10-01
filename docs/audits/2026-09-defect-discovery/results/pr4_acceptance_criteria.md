# PR 4 — 完了基準（レビューを開く前に書いた、2026-10-01）

計画 Revision 6 §12.5。提案は **H-0103**。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。** 下表の証拠列のテスト名は、
実装前に宣言した名前である。実装後に名前が変わった場合はこの表を先に直す。

---

## 0. これは何で、何ではないか

- **#269 の決着**: `RefitTrainer.fit` に欠けている 4 入力のうち `sample_weight` を渡し、残る 3 つ
  （`time_values` / `data_fingerprint` / `run_meta`）は理由つきの方針として固定する。
- **生成コード（`export_code` の `train.py`）は対象外**。同じ規則の 2 つ目の位置であり、#301 に
  繰り延べた（H-0103「規則が縛る位置」の 2。繰り延べで外れる保証を明記済み）。
- **fingerprint を誰も照合しないこと**は #263 / #272（PR 6）の範囲。
- **ラウンド予算 8。** 上限であって目標ではない。round 2 の前に absolute monitor、round 3 以降は
  relational monitor を回す。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| 基点 | `develop` `96171da`（PR 3d = #300） |
| 契約 | `HISTORY.md` **H-0103**、H-0050 / H-0085 / H-0036 |
| 上位文書 | `BLUEPRINT.md` §5.3（`balanced`）/ §8 手順 8 / §10.3（refit の inner valid） |

## 2. 修正前の実測（RED の根拠）

`develop` `96171da`、クラス不均衡な multiclass 639 行、3 fold、early stopping は既定どおり有効。
学習用 `lgb.Dataset`（`reference` なし）を spy して `(inner-train の行数, 重みの有無)` を記録した:

| `balanced` | CV fold 1-3 | refit |
|---|---|---|
| true | (383, 重み付き) × 3 | **(575, 重みなし)** |
| false | (383, 重みなし) × 3 | (575, 重みなし) |

初版は「575 行、early stopping なし」と書いていた。575 は全行数ではなく refit の inner-train の行数で、
early stopping は有効だった（設計レビュー round 1 の非 blocking 1 を受けて再計測し訂正）。
設計レビュー round 1 は `balanced: null` でも同じ欠陥を実行で再現した。

## 3. 受け入れ基準 → 証拠の対応表

| # | 基準（H-0103） | 証拠 |
|---|---|---|
| 1a | multiclass、`balanced` が `true` / `null` / 省略（既定）の 3 通り × early stopping なし: refit の学習 Dataset の重みが `compute_sample_weight("balanced", y)` と**値で**一致（全行） | `tests/test_training/test_cv_refit_parity.py::test_refit_trains_with_the_cv_weighting[*-no_inner_valid]`（3 件） |
| 1b | 同 3 通り × early stopping あり: refit の重みが inner-train 行の重みと値で一致し、eval set には重みが無い | `::test_refit_trains_with_the_cv_weighting[*-inner_valid]`（3 件） |
| 1c | 同じ fit の CV 各 fold の重みも同じ規則で一致する（基準側が変わっていない） | `::test_refit_trains_with_the_cv_weighting`（両 parametrize で CV fold 分も照合） |
| 1d | 1a / 1b は修正前に RED | 実装前のコミットで実行した結果を PR 本文に記録 |
| 2a | regression + `balanced: true` は `UNSUPPORTED_TASK`、学習 0 回 | `::test_no_weight_vector_where_balanced_makes_none[regression-balanced]` |
| 2b | regression + `balanced: false`: 重み配列も `scale_pos_weight` もなしで学習 | `::test_no_weight_vector_where_balanced_makes_none[regression-unbalanced]` |
| 2c | binary + `balanced: true`: どの学習 Dataset にも重み配列が無く、CV と refit の params に同じ `scale_pos_weight` が載る | `::test_no_weight_vector_where_balanced_makes_none[binary-balanced]` |
| 2d | `balanced: false`（binary / multiclass）: 重み配列も `scale_pos_weight` もなし | `::test_no_weight_vector_where_balanced_makes_none[binary-unbalanced]` / `[multiclass-unbalanced]` |
| 3 | 時間順 split に行をシャッフルして渡すと、`RefitTrainer.fit` に届く `y` が時間順（決定 2 の根拠） | `::test_refit_receives_time_ordered_rows` |
| 4a | `inspect.signature` から読んだ 2 つの `fit` の入力差が、すべて受理済みか H-0103 の方針登録済み | `::test_trainer_inputs_differ_only_by_written_policy` |
| 4b | 方針登録した名前が実際に `RefitTrainer.fit` に無く `CVTrainer.fit` に有る（登録が古びたら落ちる） | 同上 |
| 5 | `BLUEPRINT.md` §5.3 / §8 手順 8 / §10.3 が refit の重み付けと 3 つの方針を述べる | diff |
| 6 | 品質ゲート: ruff / ruff format / mypy `lizyml/` / フルスイート / CI | PR 本文 |
