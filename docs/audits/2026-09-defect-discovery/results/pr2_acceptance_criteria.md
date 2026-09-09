# PR 2 — 受け入れレビューの完了基準（D14 = 選択肢 A、2026-09-09）

**この文書はレビューを走らせる前に書き、管理者が承認する。** 走らせたあとに
書き足したものは完了基準ではない。

D14 の決定は **A =「完了基準を明示したうえで、管理者が承認する受け入れレビューを 1 回」**
（状況評価 `pr2_d14_assessment.md` の推奨）。評価者が指定した「止めることが正当化される
条件」を、この文書が 1 つずつ満たす。

---

## 0. これは何で、何ではないか

**これは受け入れレビューである。** 問いは 3 つだけで、下の §5 に書いてある。

**これは 29 回目の欠陥探索ではない。** diff 全体に対する非限定の敵対レビューは
D14 の選択肢 B であり、採用しなかった。**「まだ `APPROVE` が出ていない」ことは
調査対象を与えない**（評価者の指定）。

**レビュアーは受け入れを宣言しない。** 返すのは基準ごとの「満たす / 満たさない」であって
`APPROVE` ではない。**最終的な受け入れは管理者が行う。**

---

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| production コード | **`6b14b99`** |
| テスト | `6b14b99` + 下記デルタ 1 件 |
| 契約 | `HISTORY.md` の **H-0094**（決定 1-8 と受け入れ基準）、**H-0095 §「契約の確定」**（提案節ではない）、**H-0096** |
| 上位文書 | `BLUEPRINT.md` §5.3 / §14.4 |

**production は `6b14b99` から 1 行も動かさない。** 検査:

```
git diff 6b14b99 <head> -- lizyml/    # 空であること
```

**宣言するデルタ（この文書と同時に入れた、振る舞いを変えない 2 件）**:

1. `HISTORY.md` H-0096 受け入れ基準 5 —— 「メッセージが**値**を名指す」と書いたままだった。
   round 27 がそれを欠陥として差し戻し、提案節（9419 行付近）は訂正済みだったが、
   **受け入れ基準の側が訂正されずに残っていた**。SSOT ↔ 実装の drift（**DC3**）で、
   round 28 前に監視が指摘したものと同じクラスの 2 例目。訂正した。
2. `tests/.../test_fit_params_override.py` —— `test_two_spellings_of_one_value_in_
   calibration_params_are_accepted` は**本体が拒否を主張しているのに名前が受理を主張**
   していた（H-0096 で意味が反転したときに docstring だけ直っていた）。
   `..._are_refused` に改名。**アサーションは 1 行も変えていない。**

この 2 件は「表を作る過程で見つかった穴を、レビューを開く前に塞いだ」ものである
（評価者の「適用可能な証拠とその限界を特定する」に対応）。

---

## 2. 受け入れ基準 → 証拠の対応表

**ポインタの無い行は、レビューを開く前に埋めるか、明示的に処分する。**
埋まっていない行が 1 つでもあれば、この文書は未完成であり、レビューは走らせない。

### H-0096（同一層の重複綴りを値によらず拒否）

| # | 基準 | 証拠（`tests/test_core/test_fit_params_override.py`） |
|---|---|---|
| 1 | 等しい値の重複綴りが **5 か所すべて**で `CONFIG_INVALID` | `test_two_spellings_are_refused_whatever_the_values`（**⚠ round 29 訂正**: これは `normalise_and_check` を**直接呼ぶ**ので 4 surface を実行しておらず、4 つ目のラベルも `tuning best_model_params` で `tuning.optuna.space` ではない）／`test_the_adapter_refuses_a_duplicate_spelling_carrying_equal_values`（5 か所目）／`test_equal_values_of_any_shape_are_refused_under_two_spellings`／`test_two_spellings_of_one_value_in_calibration_params_are_refused`／`test_two_search_dimensions_naming_one_parameter_are_refused`（探索空間 surface を実行）。**振る舞いは round 29 が 4 surface で実行して成立を確認した** |
| 2 | 異なる値の重複綴りは従来どおり拒否（回帰させない） | `test_two_spellings_with_different_values_are_refused`／`test_arrays_that_differ_are_still_refused`／`test_a_genuine_conflict_between_plain_values_is_still_refused` |
| 3 | 綴りが 1 つなら学習に届く（対照） | `test_a_single_spelling_still_reaches_training`／`test_a_single_spelling_in_the_config_reaches_training`／`test_a_single_spelling_reaches_the_calibrator`／**`test_a_sequence_or_its_comma_text_still_trains_when_written_alone`**（round 29 が名指した正しいポインタ —— 列 / カンマ文字列が単独で**学習に届く**ことを主張する）。**⚠ `test_a_single_value_of_any_shape_passes_the_duplicate_refusal` はヘルパーを呼ぶだけで学習しない**ので、この行の証拠にはならない |
| 4 | `value_equality.py` が存在せず production に import が 0 | `test_no_production_module_imports_the_deleted_comparison`（走査）。**round 29 の指摘で補強済み** —— 元の走査は**内容**しか見ておらず、そのパスに空ファイルがあれば通った。**パス名の不在も主張する**ようにし、空ファイルを置いて RED を確認した |
| 5 | メッセージが surface と全綴りを名指し、**値は名指さない**（message からも context からも） | `test_neither_refusal_needs_the_values_to_be_printable`（`str`/`repr`/`__format__` が raise する値。両綴りと **context の描画**まで主張する）／`test_two_spellings_in_the_config_are_refused_before_training`（surface 名を主張）。**⚠ round 29 訂正**: `test_the_refusal_names_the_fit_params_surface` は**不明名**の拒否のテストで、重複綴りのものではない。**⚠ adapter 側は surface を名指さない** → §3（#286） |
| 6 | `param_domain.py` の振る舞い無変更 | `git diff` が `lizyml/core/param_domain.py` と `tests/test_core/test_param_domain.py` の両方で空（H-0096 実装コミット `9d33737` 以降） |
| 7 | フルスイート緑、`export_code` + ndarray の既存修復が保たれる | `test_export_code_writes_a_config_after_a_numpy_parameter`（`origin/develop` の `ccae32b` で `TypeError` を実測、本ブランチでは出ない） |

### H-0095（入口正規化、正は §「契約の確定」）

| # | 基準 | 証拠 |
|---|---|---|
| 1 | 正規化が wire form を保存する | `test_normalising_does_not_change_the_bytes_the_estimator_is_sent` |
| 2 | 受理集合を LightGBM から**導出**する（写さない） | **`test_the_scalar_types_are_the_ones_the_serialiser_names`／`test_the_sequence_types_are_the_ones_the_serialiser_joins`**（round 29 が名指した正しいポインタ —— **LightGBM のシリアライザに問い合わせている**）。**⚠ round 29 訂正**: numpy の 2 テストは numpy 由来・階層・dtype 同一性を見るだけで **LightGBM に一度も問い合わせない**し、`--check` は HISTORY とモジュールを比べるので**一致した転写でも通る** |
| 3 | 拒否は入口で、学習前に、`CONFIG_INVALID` | `test_the_refused_subset_is_exactly_the_declared_boundary`／`test_a_refusal_inside_the_candidate_set_is_forced_not_chosen`、および HISTORY §「契約の確定」1 の表が受理集合の**行ごとに**名指す `test_param_domain.py` の各テスト |
| 4 | 4 surface すべてで正規化が効く（実行で確認）。**この 4 つは `model.params` / `fit(params=)` / `calibration.params` / tuning の結果 `best_model_params` であって、探索空間は入らない** | **`test_the_normalised_value_is_the_one_the_estimator_is_given`**（round 29 が名指した正しいポインタ —— 入口から学習器までを**実行する**）。**⚠ round 29 訂正**: `test_the_exit_assertion_is_called_at_every_place_that_trains` は**ソーステキストを走査**し、`test_two_spellings_are_refused_whatever_the_values` はヘルパーを別ラベルで呼ぶだけで、**どちらも入口から学習器までを実行していない** |
| 5 | rounds 16-20 の敵対オブジェクトが入口で拒否される（**テストは削除せず書き換え**） | `test_a_hostile_value_is_refused_beside_a_second_spelling_too`／`test_one_sequence_written_in_two_containers_is_refused` |
| 6 | ~~`values_differ` が閉じた型集合の上で全域~~ | **superseded（H-0096 で消費者が消滅）。** 消費者表の 5 行目も superseded 済み |
| 7 | #283 をこの提案で解決するか明示的に決める | HISTORY §「受け入れ基準 7 の決定: #283 は H-0095 では解決しない」 |
| 8 | 出口の表明（4 surface の配線ではなく到達点で主張） | `test_the_exit_assertion_passes_everything_the_normaliser_produces` |
| 9 | 消費者ごとの要件（`export_code` の `json.dump` / UTF-8 / **冪等（`calibration.params` は正規化を 2 回通る —— 確定した消費者表はそれを明示的に許し、冪等性を要求する。round 29 が実測して確認）** / 述語の両向き一致） | `test_every_accepted_value_can_be_written_as_json`／`test_a_string_neither_consumer_can_encode_is_refused`／`test_normalising_twice_is_normalising_once`／`test_the_predicate_and_the_normaliser_agree`（HISTORY §「契約の確定」2 の消費者表が全 6 行を名指す） |

### H-0094（`fit(params=)` の転送、#264 本体）

| # | 基準 | 証拠 | 備考 |
|---|---|---|---|
| 1 | 学習済み Booster が変わる／`lgb.train` が受け取った値が上書き値のみ | `test_fit_params_changes_the_trained_booster`／`test_the_override_reaches_lgb_train_itself` | 修正前は両者バイト同一で RED |
| 2 | 優先順位の 2 段（`fit` > tune > config） | `test_fit_params_outrank_the_tuning_result`／`test_the_tuning_result_still_outranks_the_config`／`test_tuning_evaluates_the_parameters_it_then_selects`（決定 7 の trial マージ） | |
| 3 | 境界が開かない（不明名は `CONFIG_INVALID`、Booster 0 本） | `test_an_unknown_name_in_fit_params_is_refused_before_training` | |
| 4 | 出所の名指し（3 入力の同時判定） | `test_the_refusal_names_the_fit_params_surface`／`test_a_smart_name_in_fit_params_is_reported_as_smart`／`test_each_input_is_named_by_its_own_surface` | |
| 5 | 決定 4（管理名の拒否、18 綴り × 2 方向） | `test_a_managed_name_is_refused_rather_than_replaced`（方向 a）／`test_the_same_name_applies_once_its_smart_parameter_is_off`（方向 b）／`test_every_alias_of_a_managed_name_is_covered`（母集団の導出）／`test_an_unmanaged_name_is_never_refused`（対照） | |
| 6 | 決定 5（同一性マージ、4 継ぎ目） | `test_the_estimator_never_sees_two_spellings_of_one_parameter`（config + `fit(params=)` の 2 継ぎ目のみ）／`test_a_calibration_alias_reaches_the_calibrator` | **⚠ round 29 訂正: 残り 3 継ぎ目（tuning 結果 / provider fixed / trial）を実行するテストをこの表は名指せていない。** round 29 も代わりのポインタを出していない —— **証拠が足りない行である**。#288 として起票 |
| 7 | 決定 6 のうち **値の等価性に関する条項**（(c) 同値なら通る、round 5/6/8/11/12 の比較順序） | —— | **superseded by H-0096。** 該当条項は「値によらず拒否」に置き換わり、37 件の許容ケースは拒否ケースへ**書き換え済み**（削除ではない） |
| 8 | 決定 6 のうち残り（別名の同一性、boosting 回数の別名、`application` の task 不一致） | `test_a_boosting_round_alias_sets_the_rounds`／`test_one_spelling_of_objective_still_trains`／`test_a_parameter_passed_as_a_keyword_is_honoured_under_every_spelling`／**`test_an_objective_alias_gets_the_same_task_check`**（round 29 が名指した正しいポインタ —— **非互換な `application` を渡して学習前拒否を主張する**） | ⚠ round 29 訂正: 前 3 者だけでは task 不一致の条項をカバーしていなかった |
| 9 | 決定 7（tuning trial の同一性マージ、4 つ目の層 `calibration.params`） | `test_tuning_evaluates_the_parameters_it_then_selects`（trial マージ）／`test_two_spellings_in_calibration_params_are_refused_before_training`／`test_every_calibration_alias_is_canonical_before_the_defaults_merge`／`test_no_smart_parameter_name_has_an_estimator_alias`（スマート層は綴りマージのままでよい、の実測） | |

**H-0094 の受け入れ基準のうち、値の比較順序を定めた条項（round 5/6/8/11/12 由来）は
すべて H-0096 に置き換わっている。** これはこの PR の中で契約が改訂されたということであり、
レビュアーには「置き換わったこと自体が正しいか」ではなく
「**置き換え後の契約が head で成立しているか**」を問う（改訂の可否は D13 で決着済み）。

---

## 3. 明示的な処分（評価者が指定した必須項目）

| 項目 | クラス | 処分 | 理由 |
|---|---|---|---|
| **#284** —— `param_domain` が 1 つの境界を 3 つの構造走査で述べ、一致を保つ機構が無い | DC3 | **本 PR の外。OPEN のまま追跡する。** | 境界そのものは `test_the_refused_subset_is_exactly_the_declared_boundary` が固定しており、**述べ方が 3 通りあること**が問題。振る舞いの欠陥ではなく保守性の負債。直すには `param_domain` の再構成が要り、それは H-0095 の受理集合を動かす = 新しい Change Gate 案件 |
| **#285** —— `LGBMAdapter._build_params` の 6 か所目が `seed`/`verbosity` の重複綴りを黙って選ぶ | DC4 | **本 PR の外。OPEN のまま追跡する。** | **facade からは到達不能であることを実測し、テストで固定済み**（`test_the_sixth_site_is_unreachable_from_every_surface`）。到達するのは `LGBMAdapter` の直接構築のみで、これは公開 API ではない。修正は adapter の署名変更を伴う |
| **adapter の拒否メッセージが surface を名指さない**（facade 側は名指す） | 観測（非 blocking） | **本 PR の外。[#286](https://github.com/nbx-liz/LizyML/issues/286) として起票済み。** | `_pop_by_identity` が surface を引数に取らないため、直すには署名変更 = H-0096 の影響範囲を超える。**`5715ee2` より前からの性質**で、本 PR が作ったものではない。H-0096 受け入れ基準 5 に注記済み |
| **H-0096 受け入れ基準 5 の drift** | DC3 | **この文書と同時に訂正済み**（§1 デルタ 1） | —— |
| **テスト名が本体と反対を主張** | DC3 | **この文書と同時に改名済み**（§1 デルタ 2） | —— |

**この 3 件（#284 / #285 / #286）を残したまま出荷することを、管理者が明示的に受け入れる。**
どれも「振る舞いの欠陥」ではなく、順に「保守性の負債」「公開経路から到達不能」
「メッセージの情報量」である。

### round 29 が追加した処分（2026-09-09、§6 の表に従って分類した）

| 項目 | クラス | バケット | 処分 |
|---|---|---|---|
| **[#287](https://github.com/nbx-liz/LizyML/issues/287)** —— 探索空間からサンプルされた値が入口の正規化を迂回し、study の内側で拒否される | DC7 | **B3、非 blocking** | **本 PR の外。起票済み。** **fail-closed で Booster 0 本**、向きは「有効な入力を拒む」であって §6 の B3 例外（宣言した拒否をすり抜けて学習に届く = DC1 の向き）**ではない**。§2 が正規化について名指す 4 surface に探索空間は入らない（§5(c) の訂正を参照） |
| **[#288](https://github.com/nbx-liz/LizyML/issues/288)** —— 同一性マージの 4 継ぎ目のうち 2 つ（tuning 結果 / provider fixed）に、その層で別名を書くテストが無い | DC5 | **B4（テスト）** | **本 PR の外。起票済み。** 再現された欠陥ではなく**カバレッジの穴**。レビュアーも production の欠陥は報告していない |
| 対応表の 8 行でポインタが不足 | DC3 | **B4（文書）** | **§2 を訂正済み**（レビュアーが名指した正しいテストへ差し替え、代わりが無い行は #288 へ） |
| `test_no_production_module_imports_the_deleted_comparison` が**パス名の不在を主張していない** | DC1 の向き（走査が「見なかった」を「clean」と報告しうる） | **B4（テスト）** | **補強済み。** 空ファイルを置いて RED、消して GREEN を確認 |
| `calibration.params` の「1 度だけ」が文字どおりには成立しない（正規化が 2 回走る） | 文言 | **B4** | **§2 の H-0095 行 9 を訂正済み。** 確定した消費者表はこの経路を明示的に許し**冪等性を要求**しており、レビュアーは冪等が成立していることを実測した |

**この 2 件（#287 / #288）も残したまま出荷することを、管理者が明示的に受け入れる。**

---

## 4. 証拠の限界（何を証明していないか）

- **継承した証拠は相互作用を見落としうる**（評価者の指定した A のリスク）。
  rounds 1-28 の各ラウンドは**その時点の head** に対して回っている。
  → §5 の問い (c) が、この限界に直接答えるために置いてある。
- **受理集合は「この numpy・この platform での」集合である。**
  `longdouble` / `longlong` / `ulonglong` は C の型への別名で、platform によって解決先が
  変わる。`--check` が別環境で落ちるのは drift ではなく環境差。
- **型集合は値集合より広い。** 型が受理されることは、その型の**すべての値**が書けることを
  意味しない（`timedelta64`、`longdouble` が実測された 2 例）。
- **入力領域は無限である。** 網羅テストは存在しない。**それは理由ある受け入れを妨げない**
  （評価者の判断）。この PR が主張するのは「閉じた受理集合の外は入口で拒否する」であって
  「あらゆる Python オブジェクトに対して正しい」ではない。
- **状況評価そのものの bound**: 評価者は production コードのレビュー、指摘の再現、
  チェックの実行、リモートの issue / CI / draft 状態の確認、先行事例の外部主張の検証を
  **いずれも行っていない**（`pr2_d14_assessment.md` 末尾）。

---

## 5. 受け入れレビューが答える問い（これだけ）

**(a) 最後の修復。** `6b14b99` は主張どおりのことをしているか。
（round 28 の指摘 = 回帰テストがインタプリタ設定に依存 —— を、依存を持ち込まずに直したか）

**(b) 対応表。** §2 の各行のポインタは、**head において**その基準を実際に成立させているか。
名前が存在するかではなく、**そのテストがその基準を主張しているか**を見る。
成立させていない行があれば、その行を名指す。

**(c) 相互作用（1 つだけ、範囲を区切って）。** H-0094 の転送 ×
H-0095 の正規化 × H-0096 の拒否を、**4 surface それぞれで端から端まで** 1 度ずつ。
「継承した証拠が相互作用を見落としている」という A のリスクに、これが答える。

**⚠ 訂正（round 29 の結果を受けて、2026-09-09）。** ここで 4 つ目の surface を
`tuning.optuna.space` と書いたのは**こちらの誤記**である。**正規化の契約が持っている
4 つ目は tuning の結果（`best_model_params`）であって探索空間ではない。**
H-0096 の重複拒否は探索空間も名指しており、そちらは成立している。
**誤記のおかげで契約の穴が 1 つ見つかった** —— 探索空間からサンプルされた値は入口の
正規化を迂回する（#287、fail-closed で Booster 0 本）。誤記が blocking な指摘を
作り出すことも、実在する穴を隠すことも避けるため、**穴は起票し、判定は §6 の
事前宣言どおり非 blocking とした**（§3 参照）。

**verdict の形**: 基準ごとに `satisfied` / `not-satisfied`（+ 名指した行）。
**`APPROVE` / `REQUEST_CHANGES` ではない。** 受け入れの宣言はレビュアーの仕事ではない。

---

## 6. 指摘が出た場合にどうするか（**事前宣言**、評価者の必須項目）

レビュアーが返しうる指摘は、**次の 4 つのいずれかに落ちる**。
どれにも落ちない指摘が出たら、それはこの文書の不備であり、**管理者に戻す**。

| バケット | 条件 | 受け入れを妨げるか | 何が起きるか |
|---|---|---|---|
| **B1** | §2 の基準に**反する** —— 凍結 head で基準が成立していない | **妨げる** | 修正 1 件 + その修正だけを対象にした限定検証 1 回。**その検証がさらに指摘を返したら自動継続はせず、管理者に戻す** |
| **B2** | §3 で処分した既知項目（#284 / #285 / #286）の**別の現れ方** | **妨げない** | 該当 issue に追記して終わり。ラウンドを開かない |
| **B3** | §2 の基準の外にある**新しい production の欠陥** | **原則妨げない。ただし例外 1 つ** —— 「基準が拒否すると宣言した値が学習に届く」形（4 surface のいずれかでの **DC1**）なら**妨げる** | 妨げない場合は起票して終わり。妨げる場合は B1 と同じ手順 |
| **B4** | テスト / 文書 / 命名 | **妨げない** | 直すか起票するか。**再レビューはしない** |

**B1 と B3 の例外だけが受け入れを妨げる。** それ以外は「出荷して追跡する」。
この線引きを**レビューを走らせる前に**引いたことが、選択肢 A が選択肢 B に化けないための
唯一の仕組みである。

**範囲限定の clean な結果は、それ自体では PR 全体の承認にならない。**
承認になるのは「(a) が clean、かつ (b) の全行が `satisfied`、かつ (c) が
4 surface すべてで通る、かつ §3 の処分を管理者が受け入れる」が揃ったときだけである。

---

## 7. 互換性の帰結（出荷したら何が変わるか）

- **`fit(params=)` が実際に効くようになる。** これまで黙って捨てられていた。
  上書きに依存していなかった利用者に影響は無い。上書きを書いていた利用者は、
  **これまでと違う値で学習する**（それが #264 の修正である）。
- **同一層に 1 パラメーターを 2 綴りで書いた config は、値が等しくても
  `CONFIG_INVALID` で拒否される**（H-0096）。実測した firing rate は
  **pre-existing 0/37**（37/37 が本 PR 自身のテスト由来）なので、出荷済み config は
  1 件も壊れない。LightGBM 自身も等しい値の重複を警告する（`pr2_prior_art.md`）。
- **受理集合の外の値は学習前に `CONFIG_INVALID` になる**（H-0095）。
  これまでは学習器か `export_code` の中で別の例外になっていた。
  **`export_code` + ndarray は `origin/develop` では `TypeError` で落ちていたのが直る。**
- **config だけにエイリアスがある場合も上書きが効くようになる**（H-0094 決定 5(e)、
  CHANGELOG 記載済み）。
- `format_version` は上げていない —— **保存形式は変わっていない。**

---

## 8. 承認欄

この文書に管理者が同意したら、次の順で進む。

1. relational monitor（rounds 27-28 を観測、`pr2_monitor_round2829.md`）→ 主コンテキストが reconcile
2. 受け入れレビューを 1 回（`prompt-templates/pr2-review-round29-acceptance.md`）
3. §6 の表に従って処分
4. **管理者が受け入れを宣言**（レビュアーではない）→ draft を外して develop へ

**監視は `take-stop-condition` を勧告する可能性が高い**（rounds 26-27 の監視が
「リセットは使い切った」と予告し、round 28 で条件 C が実際に発火したため）。

**訂正（rounds 27-28 の監視の指摘による、2026-09-09）**: 初版はこの節の reconcile を
**無条件に**先に書いていた。監視はそれを「この監視が次の一手に影響を与える能力を落とし、
別種のラウンドであることが主張だけに依存する危険がある」と指摘した。**正しい指摘なので、
適用条件を限定する。**

- **`CONVERGING` または `INCONCLUSIVE` + `take-stop-condition`** → 下の reconcile を適用し、
  受け入れレビューへ進む。「**自動的な**ラウンドを止める」勧告と、
  完了基準に沿った受け入れレビューを 1 回開くことは両立する。
- **`DRIFTING`**、または**完了基準そのものが不健全だという指摘** →
  **受け入れレビューを開かず、管理者に戻す。** そのとき採る形は D14 の選択肢 2-5 である。

適用される場合の reconcile:

> 監視の指摘は記録する。**このラウンドは D14 = A が承認した受け入れレビューであって、
> ループの継続ではない。** 完了基準は事前に書かれ（本文書）、管理者が承認している。
> 発火した停止条件は**自動的なラウンド**を止めることを正当化しており、それはすでに
> 起きている —— round 29 は自動的なラウンドではない。

**発火した停止条件はマージを許可しない。** マージを許可するのは §6 の条件と管理者の宣言である。
