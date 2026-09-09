> Resume correction (2026-09-09): the historical statement below that no
> seed-priority test exists is false. `test_lgbm_defaults.py` contained
> `test_seed_takes_priority_over_random_state` since `6619d7eb` (2026-03-07).
> H-0097 Revision 2 explicitly changes that behavior. Six duplicate-input facade
> probes were reproduced; their reachability result is bounded to those cases.
> The implementation now validates merged objective/metric values with per-key
> origins and shares the adapter validators. See the revised acceptance criteria.
> This is a local candidate, not a committed or accepted change.

# 次の一手 — 2026-09-09（PR 2b、実装前に中断。**判断が 1 つ開いている**）

このファイルだけ読めば次の作業に入れるように書いてある。
**前版（PR 2 マージ待ちの版）はこの版に置き換わる。**

---

## ⛔ 最初にやること —— **H-0097 の書き直し方針を決める。コードはまだ 1 行も書いていない。**

**PR 2b は実装前に止まっている。** 規則の位置を導出したところ、
**提案 H-0097 の前提が 3 つ実測で覆った**（下記）。**書き直しが要る。**

### 開いている判断

**規則が実際に縛るのは 2 位置**（`_check_objective_compatible` と metric 検証）。
どちらも「入口が見ない問い ＝ 値が task と両立するか」を扱い、
`model.params` と `fit(params=)` から**バイト同一のメッセージ**を返して住所を名乗らない。

| | 内容 | 動く公開面 |
|---|---|---|
| **(A)** | 出所を**パラメーターごとに** adapter まで通す | **`EstimatorProvider.build_estimator_factory`（公開 Protocol）+ `LGBMAdapter.__init__`** |
| **(B)** ← 推奨 | **2 つの検査を入口へ移す** | **公開面は動かない** |

**(B) を推す理由**: 入口は既に provider と task を持ち、**H-0095 は「値の検査を入口で行う」設計を確立している**ので (B) はその延長になる。
**H-0097 が (B) を「1 つの境界に宣言を 2 つ持つ」として棄却したのは取り違えだった** ——
入口の `check_param_names` は「**既知の名前か**」を見ており、「その**値**が task と両立するか」は別の問いである。

**(B) を採る場合に残る小さい判断**: adapter 側の検査を**残すか**（多重防御）**消すか**（単一宣言）。

---

## 実測で覆った 3 つの前提（`results/pr2b_rule_positions.md` に全文）

### 1. **#286 は欠陥ではない** —— 公開経路から到達不能

`_pop_by_identity` を spy でくるみ、3 パラメーター × 2 surface で測った:

```
objective  via model.params   -> entrance  names surface: True
objective  via fit(params=)   -> entrance  names surface: True
metric     via model.params   -> entrance  names surface: True
metric     via fit(params=)   -> entrance  names surface: True
rounds     via model.params   -> entrance  names surface: True
rounds     via fit(params=)   -> entrance  names surface: True
Direct construction, the only caller left:  adapter fired: True
```

**6/6 で入口（`check_duplicate_identities`）が先に拒否し、住所を名指す。**
adapter の重複拒否が発火するのは**直接構築だけ**で、そこに名指すべき出所は無い。
**#285 と同じ形。** → **#286 の処分（close するか再スコープするか）が未決。**
測定は issue にコメントとして記録済み。

**教訓**: **起票済みという事実は、その位置が規則の対象であることを意味しない。**
起票時に到達可能性を測っていれば #286 は立たなかった。

### 2. **H-0097 の決定 1（単一の `surface` を渡す）は偽になる**

マージ後の dict は複数入口から来る（config に `learning_rate`、`fit(params=)` に `eta`）。
出所は**パラメーターごと**で、`_merge_params` は `origins` を持つが
**`return model_params, smart_params` で捨てている**。

### 3. **#283 は H-0096 で解消済み** → **close 済み**（superseded）

`values_differ` ごと削除されているので再現しない。対照（同じ値を 2 綴り）も拒否されるので、
**値を見ていない** = D13 で決めた振る舞いであって欠陥ではない。

---

## 状態（2026-09-09、すべて実測）

| 項目 | 状態 |
|---|---|
| `develop` | **`53f6cbf`** |
| ブランチ | **`fix/phase3-pr2b-parameter-domain-residue`**、head **`8dcf5bd`** + 本コミット |
| PR | **未作成**（実装前なので開いていない） |
| **production の変更** | **`adapter.py` のコメント 1 か所のみ**（記録に無い決定を主張していた箇所の訂正）。**振る舞いは無変更** |
| PR #278 | **MERGED**（`0920c2a`）。#264 / #288 close |
| PR #289 | **MERGED**（計画 Revision 6） |
| ruff / mypy | clean |

---

## PR 2b の残りスコープ（更新後）

| | 内容 | 状態 |
|---|---|---|
| `_check_objective_compatible` の住所 | **未起票。H-0097 の位置として処分** | **(A)/(B) の判断待ち** |
| metric 検証の住所 | 同上 | 同上 |
| **#285** | 6 か所目を `_pop_by_identity` 経由に | **小。決定の撤回ではない**（実測で確認） |
| ~~#286~~ | **到達不能。欠陥ではない** | 処分未決 |
| ~~#283~~ | **close 済み** | —— |

**#284 / #287 は PR 2c**（計画 §3 で分割済み）。

---

## 再開したら読むもの（この順）

1. **本ファイル**（開いている判断）
2. `results/pr2b_rule_positions.md` —— 4 規則の位置の導出と、**覆った 3 前提の実測全文**
3. `HISTORY.md` の **H-0097** —— **書き直しが必要な提案**
4. `results/pr2b_acceptance_criteria.md` —— 完了基準。**§2 の証拠列は空欄**（実装時に埋める。
   **空欄が 1 行でも残ったらレビューを開かない**）
5. `phase3-plan.md` **§3**（順序）と **§12**（Revision 6 の根拠）

---

## Phase 3 の順序（`phase3-plan.md` §3 が正）

`0`✅ → `1`✅ → `2`✅ → **`2b`（ここ）** → `2c`(#284,#287) → `3`(+#279,#282) → `3b`(H-0024)
→ `3c`(#277) → `4` → `5` → `6` → `7` → `8` → `8b`(#281) → `8c`(完了測定器) → `9`

---

## この run で確定した手続き（すべての PR に適用）

- **完了基準は PR を開くときに書く。対応表を作ること自体が検査である。**
- **規則を宣言する Proposal は、規則が縛る位置をソースから導出して列挙する**（§12.4）。
  **初回適用で、起票済み issue 1 件の前提と、提案自身の前提 2 つを覆した。**
- **ラウンド予算 8 を事前宣言する**（§12.6）。副産物は約 0.37 件/ラウンド。
- **発火した停止条件は自動ラウンドの停止を正当化するが、マージは許可しない。**
- **監視の勧告への reconcile を無条件に先に書かない。** 適用条件を限定すること。
- **繰り延べで解決しやすくなるものは無い。** 設計判断を要するかどうかで分けること。

---

## この run で自分が間違えた点（繰り返さない）

1. **BLUEPRINT を 1 節だけ見て「上位文書と非衝突」と判断した**（§5.3 だけ見て §14.4 を見落とし）。
2. **`verbose: -1` を渡したまま「LightGBM は黙っている」と結論した**（交絡）。
3. **「28 ラウンド」を完走 verdict 数のように書いた。**
4. **round 27 の指摘を `periphery` と誤記し、それを前提に選択肢を組んだ。**
5. **§6 を「B1 が定める手順」と述べた** —— 実際は**承認された例外**。監視が訂正した。
6. **「タプルの `in` はハッシュと等価による探索」と書いた** —— 偽。ハッシュを引くのは `set`。
7. **`0/54` から「動いている config は存在しない」と書いた** —— 測定の範囲を超える。
8. **監視の capsule に記録のパスを接頭辞なしで書いた** → `INCONCLUSIVE`。
   **capsule のパスはリポジトリルートからのフルパスで書くこと。**
9. **`import` を消す前に grep しなかった。**
10. **`run-exclusive.sh` の第 1 引数がラベルであることを忘れた**（exit 127 が `tail` 越しに 0 に見えた）。
11. **計測器の判定をシグネチャで書いた** → 準拠している位置を非準拠と誤報告（DC1 の鏡像）。
12. **probe がメッセージの先頭行だけを比較した** → 準拠している位置を「同一」と誤報告。
13. **起票済みの #286 を「規則が縛る位置」として数えた** → 到達可能性を測っていなかった。
14. **`adapter.py` のコメントの過大な主張をそのまま報告に引き継いだ**（#285 が
    「受理済み決定の撤回」だと述べた）。**コード内コメントも一次資料ではない。**

---

## 環境メモ（踏むと時間を失う）

- `uv` は読み取り専用の既定キャッシュで落ちる → **`UV_CACHE_DIR="$TMPDIR/uv-cache"`**。
  **git-manager にこの指示を毎回渡すこと**
- **コマンドガードが引用符・アポストロフィを解析できずに拒否する** →
  Python は**スクリプトファイルにして実行**、コミットメッセージにアポストロフィを入れない
- **`gh --body-file` は絶対パスで渡すこと**
- **`gh issue close` に `--body-file` は無い** → `gh issue comment` してから `close --reason`
- **`develop` へのマージは GitHub の自動 close を発火させない**（既定ブランチが `main`）
- **git-manager の push が 3 度失敗したが、こちらで `git push` を叩くと毎回通った**
  （`~/.config` 読み取り拒否と `localhost:3128` プロキシの一時失敗）
- **Codex は `instruments/setup_codex_home.py` で `CODEX_HOME` の書き込み可能コピーを作る。**
  レビュー 1 ラウンドは effort medium、監視・状況評価は effort low
- フルスイートは `instruments/run-exclusive.sh <label> <command...>` 経由（**第 1 引数はラベル**）
- **`import` を消す前に必ず grep すること**
