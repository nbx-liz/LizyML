# 次の一手 — 2026-09-09（compaction 前の単独ハンドオフ）

このファイルだけ読めば次の作業に入れるように書いてある。
**前版（2026-09-08、D13 が開いていた版）はこの版に置き換わる。**

---

## ⛔ 最初にやること — **D14 が開いたまま、管理者の判断待ち**

**判断が来るまでコードを触らないこと。**

### 問われていること

**round 28 で停止条件 C が発火した**あと、どう進めるか。
選択肢は `DECISIONS-PENDING.md` の **D14**、状況評価は
`results/pr2_d14_assessment.md`。

**Codex の推奨は、D14 に書いた 4 択のどれとも少し違う第 5 の形**:

> **完了基準を明示したうえで、管理者が承認する受け入れレビューを 1 回。**
> 最後の修復の検証と、蓄積した証拠が PR の契約を支持するかの判定を合わせて行う。
> **その関係を事前に定義すること —— テストだけの clean なレビューが、黙って
> PR 全体の承認に化けてはならない。**

**この選択肢を採る場合、「受け入れレビューの完了基準」を先に文書化してから走らせる。**
評価者が指定した条件は下の「止めることが正当化される条件」。

---

## 状態（2026-09-09、すべて実測）

| 項目 | 状態 |
|---|---|
| ブランチ | `fix/phase3-pr2-fit-params-forwarding` |
| head | **`e2e0250`**、`origin` と一致、**作業ツリー clean** |
| PR **#278** | **draft**、OPEN |
| CI | `e2e0250` で 7 SUCCESS / 5 実行中（Quality レーン）。**直前の `6b14b99` までは全緑** |
| フルスイート | **7710 passed / 230 skipped** |
| ruff / ruff-format / mypy | clean |
| **Codex `APPROVE`** | **未取得（28 ラウンド）** |

**走っているバックグラウンド処理**: CI watch のみ（`scratchpad/ci_final.txt`）。

---

## このセッションでやったこと（時系列、すべてコミット済み）

1. **D13 の判断を仰ぐ前に、ユーザー指示で原因解析**
   （`results/pr2_why_no_approve.md`、commit `7382054`）。
   原因 **A**（設計: round 5 の「同値は許す」が任意 Python 値の全域等価判定を要求）と
   **B**（手続き: 問いが全称命題で `APPROVE` の出口が無い）を分離。
   **A の許容分岐の firing rate を実測 = 37/37 が本 PR 自身のテスト由来、pre-existing 0。**
2. **ユーザー指示で先行事例を調査**（`results/pr2_prior_art.md`、commit `bc175a6`）。
   **LightGBM は値を比較せず、等しくても重複そのものを警告する**（前回の「黙っている」は
   `verbose: -1` によるログ抑止の誤読、訂正済み）。9 処理系のうち値で分岐するのは
   C プリプロセッサだけで、それもトークン列の同一性。
   **否定した案 2 件**: sink 委譲は閉じない（`_is_numeric` が `try: float(obj)`）、
   wire 比較は `1` vs `1.0` を拒否に戻すので round 5 の緊張を解かない。
3. **ユーザーが D13 = 経路 1「縮小してから通す」を選択** → **H-0096**
   （提案 `3fba763` / 上位文書整合 `559fa8e` / 実装 `9d33737`）。
   **同一層の重複綴りを値によらず拒否**し、`value_equality.py`（157 行）と
   その 457 行のテストを削除。`BLUEPRINT.md` §14.4 改訂。
4. **round 27**（H-0096 に対する最初のラウンド）: `REQUEST_CHANGES` 1 件、
   **`deliverable-path`** —— 拒否の**報告**が値の印字可能性に依存していた。
   **報告されていない兄弟（`check_duplicate_identities`）にも同じ欠陥**を見つけ、
   両方まとめて修正（`5715ee2`）。値は message からも context からも外した。
5. **round 28**（`5715ee2` に限定）: `REQUEST_CHANGES` 1 件、**`periphery`** ——
   回帰テストが `sys.get_int_max_str_digits()` に依存していた
   （`PYTHONINTMAXSTRDIGITS=0` で無効化される）。修正 `6b14b99`。
   **production への指摘は 0 件**、レビュアーは 6 つの値の形で両ヘルパーの成立を確認。
   → **停止条件 C 発火 → D14**。
6. **ユーザー指示で D14 の状況評価**（`results/pr2_d14_assessment.md`、commit `e2e0250`）。

---

## 外部評価がこちらの記述を訂正した 3 点（繰り返さないこと）

1. **round 27 の指摘は `deliverable-path` である。** D14 選択肢 4 に
   「rounds 27-28 は `deliverable-path` 0 件が 2 連続」と書いたのは**誤り**。
   **正しくは「1 件 → 0 件」**。選択肢 4 の前提は成り立っていない。
2. **「毎回の修正が次のラウンドの対象を供給する」は過大。**
   修正は新しい**材料**を供給するが**欠陥**を供給するとは限らず、
   指摘が 2 回続いたことは終わらない過程を立証しない。
   **承認が数学的に不可能だという帰結は出ない。**
   rounds 16-17 も等価判定だけの作業ではなかった（境界は round 18 が正しい）。
3. **「28 ラウンド」は完走した Codex verdict の数ではない**
   （round 22 は verdict 無し、round 23 は別の checker）。

---

## 「止めること」が正当化される条件（評価者の指定、そのまま）

> 評価した head と契約を凍結する。適用可能な証拠とその限界を特定する。
> **#284 / #285 と surface 名の不一致を明示的に処分する。**
> 互換性の帰結を文書化する。承認か、管理者による明示的な例外を得る。
> **さらに指摘が出た場合に何が受け入れを妨げるのか、そのとき何が起きるのかを、
> あらかじめ宣言する。**
>
> **発火した停止条件は自動的なラウンドを止めることを正当化する。
> それ自体はマージを許可しない。**

---

## 未処分のまま残っているもの（どの選択肢でも消えない）

- **#284** —— `param_domain` が 1 つの境界を **3 つの構造走査**で述べ、
  一致を保つ機構が無い（DC3）。round 25/26 はこの境界を言い直すたびに指摘が出た。
- **#285** —— `LGBMAdapter._build_params` の**6 か所目**が、他の 5 か所が拒否する
  重複綴り（`seed` / `verbosity`）を**黙って選ぶ**（DC4）。
  facade からは到達不能（実測、テストで固定済み）、直接構築では到達する。
- **adapter の拒否メッセージが surface を名指さない**（facade 側は名指す）。
  `_pop_by_identity` が surface を引数に取らないので、直すには署名変更。

**そのほかの積み残し（起票済み）**: #277 / #279 / #280 / #281 / #282 / #283。

---

## 記録の所在

- ラウンド: `results/pr2_codex_round[1-28].md`
- ループ監査: `results/pr2_monitor_round*.md`。**直近は `pr2_monitor_round2728.md`**
  （その前が `2627`、さらに前が `2527`）。
  ⚠️ `pr2_monitor_round23.md` は rounds 2-3 の監査であって round 23 のものではない。
- 状況評価: `results/pr2_situation_assessment.md`（round 26 時点）、
  **`results/pr2_d14_assessment.md`（現在の正）**
- 原因解析と先行事例: `results/pr2_why_no_approve.md` / `results/pr2_prior_art.md`
- 判断: `DECISIONS-PENDING.md` の **D7-D14**（**D14 が開いている**。
  D13 には `### 決定` 節がある）
- 提案: `HISTORY.md` の **H-0094** / **H-0095**（正は「契約の確定」節）/ **H-0096**
- レビュー依頼の雛形: `prompt-templates/pr2-review-round24-completed.md`（完走した初版）、
  `pr2-review-round27.md`（H-0096 に対する版）、`pr2-review-round28.md`（範囲限定の版）

---

## 環境メモ（踏むと時間を失う）

- `uv` は読み取り専用の既定キャッシュで落ちる → **`UV_CACHE_DIR="$TMPDIR/uv-cache"`**。
  pre-commit hook も同じで、**git-manager にこの指示を毎回渡すこと**
- **コマンドガードが引用符・アポストロフィを解析できずに拒否する** →
  **ヒアドキュメントに引用符を入れない / スクリプトファイルにして実行 /
  コミットメッセージにアポストロフィを入れない**。python のインラインも同様に落ちる
- **Codex は `instruments/setup_codex_home.py` で `CODEX_HOME` の書き込み可能コピーを
  作ってから実行**（`cleanup_codex_home.py` で消す）。**フォアグラウンドで `timeout` を
  長めに**（バックグラウンドだと低メモリ判定で kill される）。
  レビュー 1 ラウンドは effort medium で 10 分以内、状況評価は effort low
- **`import` を消す前に必ず grep すること。** このセッションで 1 度踏んだ:
  トップレベルに `import sys` を足したら auto-lint が同じ関数内のローカル
  `import sys` を冗長として削除し、その後トップレベルを消して未定義になった。
  **pyright の「not accessed」ヒントだけで消さない**
- フルスイートは `instruments/run-exclusive.sh` 経由（約 1.5-2 分）
- `gh pr checks <N> --watch` を `run_in_background` で使うとき、
  **そのシェルに `$TMPDIR` は無い** —— 出力先は絶対パスで書くこと
- **監視を回している間はファイルを編集しないこと**

---

## この PR が実際に含んでいるもの

- **#264 / H-0094**（rounds 1-15 で決着、以後 13 ラウンド clean）:
  `fit(params=)` が学習に届く。パラメーター層は綴りではなく**同一性**でマージ。
- **H-0095**: `lizyml/core/param_domain.py`。4 surface の入口で 1 度だけ正規化し、
  受理集合の外は学習前に `CONFIG_INVALID`。**571 行は残る** ——
  比較の消費者は消えたが、**`export_code` の `json.dump` + UTF-8 は本 PR とは独立の
  既存欠陥を直している**（`origin/develop` の `ccae32b` で
  `model.params={"feature_contri": np.array([1.0,1.0])}` が
  `TypeError: Object of type ndarray is not JSON serializable` を出すことを実測）。
- **H-0096**: 同一層の重複綴りを**値によらず拒否**。`value_equality.py` 削除。
  **削ったのは 157 行であって 728 行ではない**（D13 提示時の説明より小さい）。

---

## この run のあと

PR 3（#258 tuning direction）、PR 3b（H-0024 space merge、
`HISTORY.md:1615` と `:1616` の矛盾を解消すること）、PR 4-9。
**PR 9 の直前に繰り延べ 1 件**: Phase 3 完了測定ツール（`phase3_gap.py` + manifest）は
`instruments/deferred/` に未出荷で archive。
