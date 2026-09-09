# 次の一手 — 2026-09-09（PR 2 は受け入れ済み、マージ待ち）

このファイルだけ読めば次の作業に入れるように書いてある。
**前版（D14 が開いていた版）はこの版に置き換わる。**

---

## ⛔ 最初にやること —— **PR #278 のマージ判断だけが残っている**

**PR #278 は管理者が受け入れた。** draft は外してある（ready for review）。
**マージはしていない —— タイミングは管理者が別途判断する。**

判断を得たら:

```
gh pr checks 278                 # 再確認（前回 11/11 SUCCESS）
gh pr merge 278 --squash         # feature → develop は Squash merge
```

マージすると **#264 と #288 が close** される。**#284 / #285 / #286 / #287 は
OPEN のまま残る**（意図的。下記）。

---

## 状態（2026-09-09、すべて実測）

| 項目 | 状態 |
|---|---|
| ブランチ | `fix/phase3-pr2-fit-params-forwarding` |
| head | **`4c9204f`**（このコミット後に監査記録が 1 件増える）、作業ツリー clean |
| PR **#278** | **OPEN、ready for review**（draft ではない） |
| CI | **11/11 SUCCESS**（`4c9204f`、non-blocking レーン含む） |
| フルスイート | **7721 passed / 230 skipped** |
| ruff / ruff-format / mypy | clean |
| **Codex `APPROVE`** | **round 30 で取得**（選ばれた修復について） |
| ラウンド数 | **30** |

---

## この PR が含んでいるもの

- **H-0094**（#264）—— `fit(params=)` が学習に届く。パラメーター層は綴りではなく
  **同一性**で 5 つの継ぎ目をマージする。
- **H-0095** —— 4 つの入口で 1 度だけ正規化し、受理集合の外は学習前に拒否。
  **#264 とは独立の既存欠陥も直している**（`export_code` の
  `TypeError: Object of type ndarray is not JSON serializable`）。
- **H-0096** —— 同一層の重複綴りを**値によらず拒否**。`value_equality.py`（157 行）削除。
- **#288** —— 同一性マージの継ぎ目ごとにテスト。母集団は AST から導出。
- **#287** —— 探索空間の `choices` を**型の同一性**で判定。

---

## OPEN のまま残るもの（管理者が明示的に引き受けた）

| | 性質 | 直すのに要るもの |
|---|---|---|
| **#284** | `param_domain` が 1 境界を 3 走査で述べる（DC3）。保守性の負債 | 受理集合の表現の再構成 = Change Gate |
| **#285** | 6 か所目が重複綴りを黙って選ぶ（DC4）。**公開 surface から到達不能**（テストで固定済み） | adapter の署名変更 |
| **#286** | adapter の拒否メッセージが surface を名指さない | **公開コンストラクタに provenance を通すデータ契約変更** = Change Gate |
| **#287** | **意図的に OPEN。** すり抜けは直した。「4 surface が numpy を受理するのに探索空間は拒否する」不整合が残る | 設計判断: parse 時に正規化するか、探索空間を厳しいままにするか |

そのほか起票済み: #277 / #279 / #280 / #281 / #282 / #283。

---

## この run で確定した手続き（次の PR へ持ち越す）

- **長いレビューループを止めるときは、完了基準を先に文書化して管理者の承認を得てから、
  受け入れレビューを 1 回。** 完了基準は `results/pr2_acceptance_criteria.md` が実物。
  凍結 head / 受け入れ基準 → 証拠の対応表 / 証拠の限界 / 未処分項目の明示的処分 /
  互換性の帰結 / レビューが答える問い / **指摘が出た場合の網羅的なバケット規則** /
  承認の順序、を含む。
- **対応表を作ること自体が検査である** —— この run では作る過程で DC3 の drift が 2 件出た。
  **ポインタの無い行が 1 つでもあればレビューを走らせない。**
- **発火した停止条件は自動ラウンドの停止を正当化するが、マージは許可しない。**
- **監視の勧告への reconcile を無条件に先に書かない** —— 監視が次の一手に影響を与える
  能力を落とす。適用条件を限定すること。
- **繰り延べで解決しやすくなるものは無い。** 設計判断を要するかどうかで分けること。
  テストだけの項目を繰り延べる利益はゼロ。

---

## この run で自分が間違えた点（繰り返さない）

1. **BLUEPRINT を 1 節だけ見て「上位文書と非衝突」と判断した** —— §5.3 だけ見て
   §14.4 を見落とし、そこに反対のことが書いてあった。
2. **`verbose: -1` を渡したまま「LightGBM は黙っている」と結論した** —— 交絡。
   fd レベルで既定 verbosity で取り直したら警告していた。
3. **「28 ラウンド」を完走 verdict 数のように書いた** —— 系列のラベルにすぎない。
4. **round 27 の指摘を `periphery` と誤記し、それを前提に選択肢を組んだ。**
5. **§6 を「B1 が定める手順」と述べた** —— §6 は B3 を PR 内で直す手順を書いていない。
   行っていたのは**承認された例外**。監視が訂正した。
6. **「タプルの `in` はハッシュと等価による探索」と書いた** —— 偽。ハッシュを引くのは
   `set`。round 30 が訂正し、実測で確認した。
7. **`0/54` から「動いている config は存在しない」と書いた** —— 測定が立証する範囲を超える。
8. **監視の capsule に記録のパスを接頭辞なしで書いた** —— 監視が 5 つの成果物を
   「見つからない」と報告して `INCONCLUSIVE` になった。**capsule のパスは
   リポジトリルートからのフルパスで書くこと。**
9. **`import` を消す前に grep しなかった** —— メモリ `gotcha_import_removal_needs_grep.md`。

---

## 記録の所在

- **完了基準（受け入れの契約）**: `results/pr2_acceptance_criteria.md`
- ラウンド: `results/pr2_codex_round[1-30].md`
- ループ監視: `results/pr2_monitor_round*.md`。**直近は `pr2_monitor_round2930.md`**
  ⚠️ `pr2_monitor_round23.md` は rounds 2-3 の監査であって round 23 のものではない
- 状況評価: `results/pr2_situation_assessment.md`（round 26）、`results/pr2_d14_assessment.md`（D14）
- 原因解析と先行事例: `results/pr2_why_no_approve.md` / `results/pr2_prior_art.md`
- 測定: `results/pr2_duplicate_tolerance_measurement.txt` /
  `results/pr2_space_choice_measurement.txt`
- 判断: `DECISIONS-PENDING.md` の **D7-D14**（**D14 は受け入れ宣言で閉じた**）
- 提案: `HISTORY.md` の **H-0094** / **H-0095**（正は「契約の確定」節 + 末尾の
  探索空間の決定）/ **H-0096**

---

## 環境メモ（踏むと時間を失う）

- `uv` は読み取り専用の既定キャッシュで落ちる → **`UV_CACHE_DIR="$TMPDIR/uv-cache"`**。
  **git-manager にこの指示を毎回渡すこと**
- **コマンドガードが引用符・アポストロフィを解析できずに拒否する** →
  Python は**スクリプトファイルにして実行**、コミットメッセージにアポストロフィを入れない
- **`gh --body-file` は絶対パスで渡すこと**（`$TMPDIR` 展開がガード越しで壊れた）
- **Codex は `instruments/setup_codex_home.py` で `CODEX_HOME` の書き込み可能コピーを
  作ってから実行**。フォアグラウンドで `timeout` を長めに。
  レビュー 1 ラウンドは effort medium、監視・状況評価は effort low
- **git-manager の push が失敗しても、こちらで `git push` を叩くと通ることがある**
  （プロキシと `~/.config` 読み取りの一時的な失敗が 2 度起きた）
- フルスイートは `instruments/run-exclusive.sh <label> <command...>` 経由
  （**第 1 引数はラベル**。間違えると `run: command not found` で exit 127 になり、
  `tail` 越しには exit 0 に見える）
- **`import` を消す前に必ず grep すること**

---

## この run のあと

**PR 3（#258 tuning direction）**、**PR 3b**（H-0024 space merge、
`HISTORY.md:1615` と `:1616` の矛盾を解消すること）、**PR 4-9**。
**PR 9 の直前に繰り延べ 1 件**: Phase 3 完了測定ツール（`phase3_gap.py` + manifest）は
`instruments/deferred/` に未出荷で archive。
