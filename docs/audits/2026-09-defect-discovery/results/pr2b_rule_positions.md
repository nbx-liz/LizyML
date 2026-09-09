# PR 2b — 規則が縛る位置の導出（2026-09-09）

計画 Revision 6 §12.4 が定めた手続きの**初回適用**。
Proposal を書く前に、**H-0094 / H-0095 / H-0096 が宣言した規則が縛る位置を
ソースから導出**し、各位置の準拠状況を測った。

計測器: `instruments/refusal_surface_positions.py`（出荷済み）。

---

## 結果の要点

**#286 は 1 件ではなく、少なくとも 3 件だった。**
レビューが 1 件見つけて起票し、残り 2 件は誰も見ていなかった。
**導出しなければ、この 2 件は後続ラウンドが 1 件ずつ発見して issue になっていた** ——
これが §12.1 で測った生成器そのものである。

---

## R4 —— 「拒否は出所（surface）を名指す」（H-0094 決定 3）

### 導出（機械的）

`lizyml/` の全 `.py` を AST で走査し、`LizyMLError` を `CONFIG_INVALID` で raise する
関数を集め、**その raise が surface を名指すか**で分類した。

```
CONFIG_INVALID raising functions: 33

EVERY RAISE NAMES A SURFACE (6)
  lizyml/core/_model_factories.py:429  check_param_names
  lizyml/core/_model_factories.py:646  check_training_managed_overrides
  lizyml/core/_model_factories.py:739  check_duplicate_space_dimensions
  lizyml/core/_model_factories.py:860  check_duplicate_identities
  lizyml/core/_model_factories.py:965  check_smart_managed_overrides
  lizyml/core/param_domain.py:508      normalise_params

SOME RAISES NAME A SURFACE AND SOME DO NOT (0)

NO RAISE NAMES A SURFACE (27)
```

**計測器の初版には false negative があった。** 判定を「シグネチャに `surface` 引数が
あるか」で書いたため、**`check_param_names` を非準拠側に分類していた** ——
この関数は surface を `named: Iterable[tuple[str, str]]` のタプルで受け取って名指す。
**走査が「見なかった」を「準拠していない」と報告する形**で、DC1 の鏡像である。
フラグされた位置を 1 つ読んで気づき、判定を **raise が surface を名指すか**に変えた。
初版なら準拠 3、訂正後は **6**。

### 27 の非名指しのうち、規則が実際に縛るもの（実測）

**到達可能性は AST では導けない**（単一入口の拒否は出所が自明なので名指す必要がない）。
そこでパラメーター経路の候補について、**同じ入力を 2 つの surface から入れて
メッセージを比較した**（`scratchpad/probe_surface_reach.py`）。

| 位置 | `model.params` と `fit(params=)` | 判定 |
|---|---|---|
| `_model_factories.py:429 check_param_names` | **異なる** —— `model.params:` / `fit(params=):` を行頭に置く | **準拠** |
| `adapter.py:86 _check_objective_compatible` | **完全に同一**、surface 無し | **非準拠。#286 と同クラス** |
| `metric_bridge.py` の metric 検証 | **完全に同一**、surface 無し | **非準拠。#286 と同クラス** |
| `adapter.py:28 _pop_by_identity` | 同上（**#286** として起票済み） | **非準拠** |

実測されたメッセージ:

```
objective incompatible: {'application': 'regression'}
  model.params : objective 'regression' is not compatible with task 'binary'. ...
  fit(params=) : objective 'regression' is not compatible with task 'binary'. ...
  -> SAME MESSAGE, names a surface: False

unknown metric: {'metric': 'not_a_metric_at_all'}
  model.params : Metric 'not_a_metric_at_all' is not a valid LightGBM metric ...
  fit(params=) : Metric 'not_a_metric_at_all' is not a valid LightGBM metric ...
  -> SAME MESSAGE, names a surface: False
```

対照（規則が働いている例）:

```
smart param bad: {'num_leaves_ratio': -1.0}
  model.params : "  model.params: 'num_leaves_ratio' is a LizyML smart parameter, ..."
  fit(params=) : "  fit(params=): 'num_leaves_ratio' is a LizyML smart parameter, ..."
  -> differs   （出所ごとに違う住所を答えている）
```

**この probe も 1 度誤った** —— 初版はメッセージの**先頭行だけ**を比較しており、
`check_param_names` が surface を**次行以降**に置くため「同一」と報告した。
全文比較に直した。**部分を見て全体を判定する**形は、この run で 2 度目である。

### 残り 23 の処分

**規則の対象外**と判定する。単一入口からのみ到達し、出所が自明である ——
`config/loader.py`（設定ファイル）、`plots/*`（プロット引数）、
`data/dataframe_builder.py`（データ）、`training/inner_valid.py`（split 設定）、
`tuning/search_space.py`（探索空間のみ）、`calibration/isotonic.py`（calibrator 構築）。

**`param_domain.py:550 assert_plain_params` は意図的に対象外**である ——
これは**出口**の表明で、`where="lgb.train"` のように**到達先**を名指す。
入口の出所ではなく、どの sink に届いたかが答えるべき情報である。

---

## R1 —— 「パラメーター層は同一性でマージする」（H-0094 決定 5 / 7）

位置 = `overlay_params` の呼び出し点。**PR 2 で AST から導出済み**
（`test_every_identity_overlay_seam_has_a_test`）。層名 4 つ ——
`fixed` / `best_model_params` / `override` / `model_p` —— すべて準拠、各々にテストがある。

**導出の bound は round 30 が実測した**: 2 モジュール内の**素の呼び出しの相異なる層名**
しか集めない。新しい層名で導入された継ぎ目は捕まえるが、既存名の再利用・修飾された
呼び出し・dict マージ・第 3 のモジュールは捕まえない。

---

## R2 —— 「同一層の重複綴りは値によらず拒否」（H-0096）

| 位置 | 状態 |
|---|---|
| `normalise_and_check` の 4 呼び出し点（`model.params` / `fit(params=)` / `calibration.params` / `tuning best_model_params`） | 準拠 |
| `tuning.optuna.space`（`_model_factories.py:804`） | 準拠 |
| `_pop_by_identity` の 3 呼び出し点（`num_iterations` / `objective` / `metric`） | 準拠 |
| **`_build_params` の `seed` / `verbosity`（6 か所目）** | **非準拠 —— 黙って選ぶ（#285）** |

**#285 には既存の決定との衝突がある。** `_build_params` の当該箇所は
「`seed` が `random_state` に優先する」という**受理済みの決定**を実装しており
（`test_lgbm_defaults.py` が固定）、ヘルパー経由にすると**その決定を撤回する**ことになる。
公開 surface からは到達不能であることは実測・テスト固定済み。
**Proposal で「撤回するか、到達不能を根拠に据え置くか」を決める。**

---

## R3 —— 「入口で 1 度だけ正規化し、出口で表明する」（H-0095）

| 位置 | 状態 |
|---|---|
| 入口 4 surface（`normalise_and_check`） | 準拠 |
| 出口 2 サイト（`assert_plain_params`）—— ソースから導出済み | 準拠 |
| **`tuning.optuna.space`** | **正規化しない。** すり抜けは PR 2 で塞いだ（型の同一性で拒否）が、**4 surface が numpy を受理するのに探索空間は拒否する不整合が残る（#287）** |

---

## PR 2b の Proposal に載せる位置ブロック（草案）

```
Rule positions (R4, surface naming): 33 CONFIG_INVALID raising functions,
  derived by AST over lizyml/ (instruments/refusal_surface_positions.py)
  complying     : 6 name a surface in every raise
  fixed here    : 3 -- _pop_by_identity (#286),
                       _check_objective_compatible,
                       metric_bridge metric validation
                  (all three measured: identical message from model.params and
                   fit(params=), naming neither)
  dispositioned : 23 reachable from one entrance only, plus assert_plain_params
                  which names its sink by design
```

**2 件は起票していない。** §12.4 の手続きどおり、**Proposal の中の位置として処分する** ——
issue にして後で拾うのではなく。

---

## この導出で自分が 2 度間違えた点

1. **計測器の判定をシグネチャで書いた** → `check_param_names` を非準拠と誤報告。
   **走査は「何を見たか」ではなく「何を主張したいか」で書くこと。**
2. **probe がメッセージの先頭行だけを比較した** → 準拠している位置を「同一」と誤報告。
   **部分を見て全体を判定しない。**

どちらも**フラグされた位置を実際に読んで**気づいた。
**導出は出発点であって結論ではない。**
