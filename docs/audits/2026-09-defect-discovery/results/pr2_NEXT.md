# 次の一手 — 2026-10-06（Phase 3 は測定で完了。PR 9b で記録する）

このファイルだけ読めば次の作業に入れるように書いてある。
**前版（2026-10-06、「PR 9 の PR を作成」）はこの版に置き換わる。** 前版は git 履歴に残っている。

---

## 状態 —— **Phase 3 完了（測定済み）**

PR 9（#320、H-0110）は `21db071` でマージ済み、#271 は close レビュー（APPROVE）を経て close 済み（close コメント 6010100067）。PR 9b は manifest の #271 行に `github_prs: [320]` と `closure_comment` を入れ、#266 行の母集団を定数 7 から `derived_from`（`SITES` の件数）に変えた。PR 9 が BLUEPRINT §15.2 に `format_version` の記述を 2 つ足したため、宣言の走査が 8 件を集め、定数 7 の行が INCOMPLETE になったからである（DC3: 派生した件数を定数で持っていた）。

`phase3_gap.py --after origin/develop`（`21db071`、PR 9b の manifest）: COMPLETE 19 / COMPLETE-RED-BY-MUTATION 4 / PARTIAL 1（#270、計画どおり open）/ NOT-PLANNED 1 / INCOMPLETE 0 / UNKNOWN 0、exit 0（`results/phase3_completion_21db071.txt`）。

残る派生: #270（177 件の hollow test、別の母集団修復として起票予定）、#318（コードがしない 6 件の判断）、#319（HISTORY の記録のずれ）、#309 / #311 / #313 / #315、#299 / #301 / #303 / #304 / #307 / #295 / #297 / #298 / #273 / #276 / #280。develop は Phase 3 の変更を未リリースで多数保持している（CHANGELOG の H-0102 記載漏れあり）。
以下の PR 9 以前の節は経緯として残す。


## 最初にやること —— **PR 9（#271、H-0110）がマージされていれば PR 9b**

PR 9 は決定済みの提案を BLUEPRINT.md に畳み込み、全提案の処分を内容で検査する（`docs/proposal_dispositions.toml`、`tests/test_docs/test_proposal_blueprint_coverage.py`）。マージされたかは `gh pr list --state merged --head docs/phase3-pr9-blueprint-fold` で確かめる。マージ後: (1) close レビューを経て #271 に close コメントを書いて close する。(2) PR 9b で `instruments/phase3_manifest.json` の #271 行に `github_prs: [320]` と `closure_comment` の id を同時に入れる（`phase3_gap.py` は `github_prs` があると `closure_comment` を必須にするので、PR 9 では `github_prs` を空のままにした）。(3) `run-exclusive.sh` 経由で `phase3_gap.py --after origin/develop` を回し、exit 0（INCOMPLETE と UNKNOWN が 0）で Phase 3 の完了を判定する。PR 9 で畳み込まなかったものは #318 / #319。完了基準は `results/pr9_acceptance_criteria.md`。
以下の PR 8c 以前の節は経緯として残す。


## 最初にやること —— **PR 8c がマージされていれば PR 9（#271）**

PR 8c は計画 §8 の完了測定器を出荷する（`instruments/phase3_gap.py`、`instruments/phase3_manifest.json`、`tests/test_docs/test_phase3_gap.py`）。マージされたかは `gh pr list --state merged --head chore/phase3-pr8c-completion-instrument` で確かめる。次は PR 9: 決定済みの Proposal を BLUEPRINT.md に畳み込み、#271 のテスト（`tests/test_docs/test_proposal_blueprint_coverage.py`）を足す。PR 9 の後、`run-exclusive.sh` 経由で `phase3_gap.py --after origin/develop` を回し、exit 0（INCOMPLETE と UNKNOWN が 0）で Phase 3 の完了を判定する。manifest の #271 の行（`github_prs` が空）は PR 9 の番号で埋める。

### PR 8c（完了測定器、この版を書いた時点で PR 作成中）

PR 8b（#316、H-0109）はマージ済み（`develop` = `33a3f6e`）、#281 は close 済み、#315 を起票。PR 8c は計画 §8 の完了測定器を出荷する（`instruments/deferred/` に残っていた 3 ファイルを、実際にマージされたテストに合わせて作り直す。archive の manifest は `.gitignore` の `*.json` のため一度もコミットされていなかった）。完了基準は `results/pr8c_acceptance_criteria.md`、実測は `results/pr8c_measurements.txt`。以下の PR 8b 以前の節は経緯として残す。

### （済）PR 8b（#281）

PR 8（#314、H-0108）はマージ済み（`develop` = `036cd18`）、#268 は close 済み。PR 8b は H-0109（fit が適用した training overlay を artifact に記録し、`load()` 後も報告面が同じ値を答える）。完了基準は `results/pr8b_acceptance_criteria.md`、実測は`results/pr8b_measurements.txt`。以下の PR 8 以前の節は経緯として残す。

### （済）PR 8（#268）

PR 7（#312、H-0107）はマージ済み（`develop` = `91a698b`）、#267 は close 済み、#311 を起票。PR 8 は H-0108。**#268 と計画の前提（名前の照合で「到達できない 25 / 22」）は誤りだった**: 実行すると「公開する 9 個」は全部すでに届いていた。74 個を出どころで分類し（config 60 / api 4 / derived 3 / policy 2 / internal 5）、Config のキーが設定しない 14 行を BLUEPRINT §5.5 に書く。設計レビューで #313（`model.params` の組み込みの指標の dict の引数が捨てられる）を起票。完了基準は `results/pr8_acceptance_criteria.md`。以下の PR 7 以前の節は経緯として残す。

### （済）PR 7（#267）

PR 6（#310、H-0106）はマージ済み（`develop` = `97381db`）、#263 / #272 は close 済み。PR 7 は H-0107（漏洩検査が比べられない列を黙って飛ばさない）。完了基準は `results/pr7_acceptance_criteria.md`。**Issue を close する前のレビュー（クローズレビュー）は独立したゲートとして、マージ後・close コメント投稿前に回す**（PR 6 では、コードレビューを通った誇張 4 件と「マージ済み」の早すぎる記述を見つけた）。以下の PR 6 以前の節は経緯として残す。

### （済）PR 6（#263 / #272）

PR 5（#305、H-0104）と、計画外に挿入した PR 5b（#308、H-0105、feval の二重変換 #306）はマージ済み（`develop` = `1abf7fb`）。**外部レビューでは Issue 本文と Proposal の事実確認も依頼する**（管理者の要望、2026-10-01）。以下の PR 4 の節は経緯として残す。

### （済）PR 4（#269）

PR 3d（#300）はマージ済み（`develop` = `96171da`）。
`origin/develop` から `fix/phase3-pr4-refit-input-parity` を切り直す。

`RefitTrainer.fit` は `CVTrainer.fit` より 4 つ少ない入力しか受け取らない
（`sample_weight` / `time_values` / `data_fingerprint` / `run_meta`）。
**multiclass の `balanced` では CV fold は重み付きで学習し、最終 refit は重みなしで学習する。**

**着手前に実測すること（計画 §4 PR 4 の前提が 1 つ怪しい）**: `CVTrainer` も inner valid の
分割に `time_values` を渡していない（fold の時間範囲を記録するだけ、`training/cv_trainer.py`
の `_split_inner`）。#269 の「refit は時間列を見られないので時間順の inner split ができない」
という理由は成り立たない可能性がある。Proposal を書く前に確かめること。

### 開く前にやること（計画 Revision 6 §12.4-12.6）

1. **完了基準を先に書く**（雛形: `results/pr2_acceptance_criteria.md`）。
   **証拠のポインタが無い行が 1 つでもあればレビューを開かない。**
2. **Proposal に「規則が縛る位置」を列挙する** —— ソースから導出、実装前。
   **導出の bound も併記する。**
3. **ラウンド予算 8 を宣言する**（上限であって目標ではない）。
4. **HISTORY の ID は `next-id.sh` で採り、マージ直前にもう一度確かめる。** 並行ブランチが
   同じ番号を取った実例がある（H-0100 が PR 3b と PR 3c で重複、PR 3d で解消）。

---

## 状態（2026-10-01、すべて実測）

| 項目 | 状態 |
|---|---|
| `develop` | **`33a3f6e`**（#316 = PR 8b） |
| Phase 3 | **18 本中 16 本完了**（0 / 1 / 2 / 2b / 2c / 3 / 3b / 3c / 3d / 4 / 5 / 5b / 6 / 7 / 8 / 8b）、8c 進行中 |
| 2026-10-01 に close | **#261**（#275 で修正）/ **#266**（#274 で修正）。どちらも独立した close レビューで確認 |
| close レビューで残作業ありとした 2 件 | **#262** / **#265** → PR 3d（#300）で処理し、2026-10-01 に close |
| 新規起票 | **#299**（`category: smart` の次元は消費されなくても受理される。#262 の行列を dict で検査して発見） |

---

## Phase 3 の順序（`phase3-plan.md` §3 が正）

`0`✅ → `1`✅ → `2`✅ → `2b`✅ → `2c`✅ → `3`✅ → `3b`✅ → `3c`✅ → `3d`✅
→ `4`✅ → `5`✅ → `5b`✅ → `6`✅ → `7`✅ → `8`✅ → `8b`✅
→ **`8c`(完了測定器) ← ここ** → `9`(#271)

**未処分の副産物**: **#315**（load 後のモデルの tuning 面が、保存しない試行履歴を空の study として示す、判断待ち）、#309 / #311 / #313（PR 6-8 で起票）、**#299**（smart 次元、Change Gate 案件）、**#295**（pydantic の下限、DC7）、
**#297**（platt/beta が最適化の失敗を学習済みとして通す、仕様判断待ち）、**#298**（PR 3c の
完了基準で証拠が主張より弱い行）、**#280**（着手前に再検証が必要）、**#273**（`embargo` の意味、
仕様判断待ち）、**#276**（計画書 §12.3 が答えている）、#270（177 件の hollow test、別スコープ）。

---

## 再開したら読むもの（この順）

1. **本ファイル**
2. `phase3-plan.md` **§3**（順序と状態）と **§12**（Revision 6 の根拠と手続き）
3. `results/pr2_acceptance_criteria.md` —— 完了基準の雛形
4. PR 4 に関係する既存記録: #269 の本文、`HISTORY.md` の H-0036（ratio params を inner-train の大きさで解決）と H-0085（refit の pipeline fit 境界）

---

## この run で確定した手続き

- **完了基準は PR を開くときに書く。対応表を作ること自体が検査である。**
- **規則を宣言する Proposal は、規則が縛る位置をソースから導出して列挙する**（§12.4）。
- **ラウンド予算 8 を事前宣言する**（§12.6）。副産物は約 0.37 件/ラウンド。
- **発火した停止条件は自動ラウンドの停止を正当化するが、マージは許可しない。**
- **監視の勧告への reconcile を無条件に先に書かない。**
- **繰り延べで解決しやすくなるものは無い。** 設計判断を要するかどうかで分ける。

---

## この run で自分が間違えた点（繰り返さない）

1. BLUEPRINT を 1 節だけ見て「上位文書と非衝突」と判断した（§5.3 だけ見て §14.4 を見落とし）。
2. `verbose: -1` を渡したまま「LightGBM は黙っている」と結論した（交絡）。
3. 「28 ラウンド」を完走 verdict 数のように書いた。
4. round 27 の指摘を `periphery` と誤記し、それを前提に選択肢を組んだ。
5. §6 を「B1 が定める手順」と述べた —— 実際は承認された例外。監視が訂正した。
6. 「タプルの `in` はハッシュと等価による探索」と書いた —— 偽。ハッシュを引くのは `set`。
7. `0/54` から「動いている config は存在しない」と書いた —— 測定の範囲を超える。
8. 監視の capsule に記録のパスを接頭辞なしで書いた → `INCONCLUSIVE`。
9. `import` を消す前に grep しなかった。
10. `run-exclusive.sh` の第 1 引数がラベルであることを忘れた。
11. 計測器の判定をシグネチャで書いた → 準拠位置を非準拠と誤報告。
12. probe がメッセージの先頭行だけを比較した → 準拠位置を「同一」と誤報告。
13. 起票済みの #286 を「規則が縛る位置」として数えた → 到達可能性を測っていなかった。
14. `adapter.py` のコメントの主張をそのまま報告に引き継いだ。
15. **「2 綴りのどちらが勝つかを固定したテストは存在しない」と報告した —— 偽。**
    `test_seed_takes_priority_over_random_state` は 2026-03-07 から存在した。
    確認に使った `grep ... | head -8` が、97 行目のそのテストを出力から切り落としていた。
    **別セッションが見つけて訂正した。** 12 と同じ「部分を見て全体を判定する」誤りで、
    この run で 3 度目。**否定の主張（「存在しない」）は、出力を切り詰めずに確認すること。**

---

## 環境メモ（踏むと時間を失う）

- `uv` は読み取り専用の既定キャッシュで落ちる → `UV_CACHE_DIR="$TMPDIR/uv-cache"`。
  git-manager にこの指示を毎回渡すこと
- コマンドガードが引用符・アポストロフィを解析できずに拒否する →
  Python はスクリプトファイルにして実行、コミットメッセージにアポストロフィを入れない
- `gh --body-file` は絶対パスで渡す。`gh issue close` に `--body-file` は無い →
  `gh issue comment` してから `close --reason`
- **`develop` へのマージは GitHub の自動 close を発火させない**（既定ブランチが `main`）
- **`gh` が `gh auth login` を求めて失敗することがある**（`~/.config` 読み取り拒否の一時失敗）。
  同じコマンドの再実行で通る。git-manager の push 失敗も同様で、こちらで `git push` すると通る
- Codex は `instruments/setup_codex_home.py` で `CODEX_HOME` の書き込み可能コピーを作る。
  レビュー 1 ラウンドは effort medium、監視・状況評価は effort low
- フルスイートは `instruments/run-exclusive.sh <label> <command...>` 経由（第 1 引数はラベル）
