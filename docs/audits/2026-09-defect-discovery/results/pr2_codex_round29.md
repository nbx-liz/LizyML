# PR 2 — round 29（受け入れレビュー、2026-09-09）

D14 = A が承認した**1 回の受け入れレビュー**。完了基準は
`pr2_acceptance_criteria.md`（レビューを走らせる前に書き、管理者が承認した）。
依頼は `prompt-templates/pr2-review-round29-acceptance.md`。
Codex `gpt-6-astra` effort medium、read-only、head `c13ce170`。
production は `6b14b99` で凍結（`git diff 6b14b99 HEAD -- lizyml/` が空、レビュアーが確認）。

**`APPROVE` / `REQUEST_CHANGES` は求めていない。** 返ってきたのは基準ごとの
`satisfied` / `not-satisfied` である。

---

## (a) 最後の修復 —— **satisfied**

> 置き換えたオブジェクトは `__str__` / `__repr__` / `__format__` から決定的に raise し、
> 構築にインタプリタの変換上限を要しない。テストは両方の拒否ヘルパーに到達し、
> `CONFIG_INVALID` を確認し、描画された両方の綴りを確認し、`repr(error)` を描画して
> **context も描画している**。名前が言うとおりの性質を主張している。
> **`PYTHONINTMAXSTRDIGITS` を未設定・`0`・`640` の 3 通りで実行して成功した。**

---

## (b) 対応表 —— 8 行が `not-satisfied`

レビュアーの前置き:

> **これは各行が名指した証拠についての判定である。証拠のポインタが不完全であること自体は
> production の欠陥を立証しない。**

| 基準 | H-0094 | H-0095 | H-0096 |
|---|---|---|---|
| 1 | satisfied | satisfied | **not-satisfied** |
| 2 | satisfied | **not-satisfied** | satisfied |
| 3 | satisfied | satisfied | **not-satisfied** |
| 4 | satisfied | **not-satisfied** | **not-satisfied** |
| 5 | satisfied | satisfied | **not-satisfied** |
| 6 | **not-satisfied** | satisfied（superseded） | satisfied |
| 7 | satisfied（superseded/rewrite） | satisfied | satisfied |
| 8 | **not-satisfied** | satisfied | — |
| 9 | satisfied | satisfied | — |

**8 行すべてが「ポインタが足りない」であって「基準が成立していない」ではない。**
レビュアーは 8 行のうち 6 行について、**正しい証拠がどこにあるかを名指している**。

| 行 | 何が足りなかったか | レビュアーが名指した正しいテスト |
|---|---|---|
| **H-0096.1** | `test_two_spellings_are_refused_whatever_the_values` は `normalise_and_check` を**直接呼んでおり 4 surface を実行していない**。しかも 4 つ目のラベルが **`tuning best_model_params`** で `tuning.optuna.space` ではない | 探索空間のカバレッジは他のテストでも埋まっていない |
| **H-0096.3** | `test_a_single_value_of_any_shape_passes_the_duplicate_refusal` は**ヘルパーを呼ぶだけで学習しない**ので、単独の列 / カンマ文字列が学習に届く対照になっていない | `test_a_sequence_or_its_comma_text_still_trains_when_written_alone` |
| **H-0096.4** | `test_no_production_module_imports_the_deleted_comparison` は**ファイルの内容**を走査する。**パス名の不在を主張していない —— そのパスに空ファイルがあれば通る**（実際にはモジュールは不在） | （テスト側の補強で閉じる） |
| **H-0096.5** | `test_the_refusal_names_the_fit_params_surface` は**不明名**の拒否のテストで、重複綴りではない。印字不能値のテストは両綴りと context を主張するが、**重複メッセージの surface は主張していない** | 行 1 のヘルパー格子が実際にはその主張を供給している |
| **H-0095.2** | numpy の 2 テストは numpy 由来・階層・dtype 同一性を見るが、**LightGBM に一度も問い合わせていない**。`--check` は HISTORY とモジュールを比べるので**一致した転写でも通る** | `test_the_scalar_types_are_the_ones_the_serialiser_names` / `test_the_sequence_types_are_the_ones_the_serialiser_joins` |
| **H-0095.4** | `test_the_exit_assertion_is_called_at_every_place_that_trains` は**ソーステキストを走査**する。もう一方はヘルパーを別ラベルで呼ぶだけ。**どちらも入口から学習器までを実行していない** | `test_the_normalised_value_is_the_one_the_estimator_is_given`（ただし対象は H-0095 の 4 surface = tuning **結果**であって探索空間ではない） |
| **H-0094.6** | `test_the_estimator_never_sees_two_spellings_of_one_parameter` は config + `fit(params=)` の別名しか通っていない。**tuning 結果 / provider fixed / trial の継ぎ目を通っていない** | —— |
| **H-0094.8** | 3 つのテストは boosting 回数の転送と互換な canonical `objective` の学習を立証するが、**非互換な `application` を渡して学習前拒否を主張しているものが無い** | `test_an_objective_alias_gets_the_same_task_check` |

### superseded 行について（明示的に確認された）

> **許容テストの系列と入力は拒否ケースとして生き残っている** —— 容れ物の違い、numpy 値、
> 等しい通常値、calibration の重複を含む。adapter のケースも拒否のカバレッジを受けている。
> **書き換えと称して削除された許容ケースは 1 件も見つからなかった。**
> 比較モジュール自身のテストを削除したことは、明示的に superseded となった消費者要件に従う。

---

## (c) 相互作用検査 —— 4 surface のうち **3 つは holds、1 つは合成が破れる**

| surface | 判定 |
|---|---|
| **`model.params`** | **holds.** 単独の `eta=np.float64(0.5)` が素の float になり実学習に `0.5` で届き、adapter 既定に勝つ。等値・非等値の重複綴りはどちらも `CONFIG_INVALID`、学習呼び出し 0 |
| **`fit(params=)`** | **holds.** 同じ値が config `0.001` と tuning 結果 `0.25` の上に届く。正規化は `k=0.5` を保存。重複は両方とも学習前に拒否 |
| **`calibration.params`** | **転送・wire 保存・拒否は holds。ただし「1 度だけ」は文字どおりには成立しない** —— 正規化が **2 回**呼ばれる（検証側は結果を捨て、calibrator 構築側がもう一度正規化する）。**H-0095 の確定した消費者表はこの経路を明示的に許し、冪等性を要求している**ので、**未開示の振る舞いの継ぎ目ではなく B4 の文言不一致** |
| **`tuning.optuna.space`** | **合成が破れる。** カテゴリ次元 `eta` に `choices=[np.float64(0.5)]` を与えると、**サンプルされた値が入口の正規化を迂回して** adapter の出口表明に `float64` のまま到達する。表明は `CONFIG_INVALID` を上げ、tuning は **`TUNING_FAILED`** として露出する。**Booster は 0 本。** 素の float の対照は成功し config に勝つ。等値・非等値の重複次元は正しく学習前に拒否される。継ぎ目は `_model_tuning.py:469` の**正規化されていない trial overlay**（探索空間のサンプリングと学習の間）。**#284-#286 とは別物** |

### こちらで再現した（起票前の実行確認）

```
numpy 2.4.2
numpy-float64:          LizyMLError code=TUNING_FAILED
                        (adapter.py:247 assert_plain_params -> param_domain.py:575
                         CONFIG_INVALID: 'eta' is a float64)
plain-float (control):  tuned OK, best_model_params={'eta': 0.5}
```

再現スクリプトは `instruments/space_choice_normalisation.py` として出荷した。

---

## そのほか（レビュアーが B4 と分類）

- 対応表の網羅性の主張が、**H-0094 の no-op / 非永続の基準**（`None` / `{}` /
  上書きが呼び出しをまたがない / 利用者の config を書き換えない）を落としている。
- H-0095 の確定した表が**古いテスト名** `test_an_integer_too_large_for_a_float_is_accepted_and_compared`
  を残している（現在は `..._is_accepted` で終わる）。

---

## レビュアーが述べた bound

読んだ: 完了基準（最初に）、該当する上位契約の節、名指されたテストと補助ヘルパー、
変換の履歴 diff、4 つの production 経路。作業ツリー clean、head `c13ce170`、
`6b14b99` からの production diff が空であることを確認した。

**実行した**: 領域テストと選んだ受け入れテスト（**5,290 passed / 230 skipped**）、
override テスト（**205 passed / 13 deselected**）、契約の `--check`、
インタプリタ設定の検査、**実学習の相互作用プローブ**。何も変更していない。

**していない**: フルスイートの再実行、export 書き出しテスト、lint / mypy、
過去の firing rate 計測、リモートの CI / PR チェック、PR diff 全体のレビュー、
任意の値や platform の網羅。

---

## 主コンテキストの処分（完了基準 §6 の表に従う）

| 指摘 | バケット | 処分 |
|---|---|---|
| (a) satisfied | —— | 受け入れ |
| (b) 8 行のポインタ不足 | **B4**（文書） | **対応表を訂正**。レビュアーが名指した正しいテストへ差し替え、名指しの無い 2 行は「証拠が無い」と明記する。**再レビューはしない** |
| (b) H-0096.4 の走査がパス名を見ていない | **B4**（テスト） | **テストを補強**（パスの不在も主張する）。振る舞いは変わらない |
| (c) `calibration.params` の「1 度だけ」 | **B4**（文言） | 完了基準の文言を H-0095 の確定した消費者表に合わせる |
| (c) `tuning.optuna.space` の合成 | **B3、非 blocking** | **起票する。** §6 の B3 例外（宣言した拒否をすり抜けて学習に届く = DC1 の向き）**には当たらない** —— **fail-closed で Booster 0 本**、向きは逆（有効な入力を拒む） |

**`tuning.optuna.space` を B3 と判定した根拠を明示する。** §2 が正規化について名指す
4 surface は H-0095 の確定した契約のもので、その 4 つ目は
**`tuning best_model_params`（tuning の結果）であって探索空間ではない**。
H-0096 基準 1 は `tuning.optuna.space` を名指すが、それは**重複綴りの拒否**についてで、
それは成立している（レビュアーが実行して確認）。
**したがって探索空間の正規化は §2 のどの基準にも含まれていない。**

**完了基準 §5(c) の書き方がこちらの誤りだった** —— 4 つ目の surface を
`tuning.optuna.space` と書いたが、正規化の契約が持っているのは tuning **結果**である。
その誤記のおかげで契約の穴が 1 つ見つかった、というのが実際に起きたことである。
**誤記が blocking な指摘を作り出すことも、実在する穴を隠すことも、どちらも避ける** ——
穴は起票し、判定は事前宣言した規則どおり非 blocking とする。

**受け入れの宣言は管理者が行う。** レビュアーは行っていないし、本節も行っていない。
