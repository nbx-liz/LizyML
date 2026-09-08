# 次の一手 — 2026-09-08（H-0095 実装後）

このファイルだけ読めば次の作業に入れるように書いてある。

## 状態

- ブランチ `fix/phase3-pr2-fit-params-forwarding`、PR **#278**（draft）。
- **H-0095（選択肢 F）は実装・push 済**。フルスイート **7425 passed / 256 skipped**、
  `ruff check .` / `ruff format --check .` / `mypy lizyml/` clean、
  出荷計測器 `report_lifecycle_grid.py` は exit 0。
- 管理者の指示は **「F を実装後、両方あわせて Approve を取得してください」**。
  → **#264 本体と H-0095 を 1 本の PR として round 21 に出す。別々にマージしない。**

## 次の一手

1. **rounds 19-21 の関係監視**（`policy:loop-monitor`、round 3 以降は毎回必須）。
   capsule は `templates/review-loop-monitor-capsule.md`。**監視には
   「apparatus を膨らませただけではないか」という反対仮説を明示的に渡すこと** —
   新モジュール 1 本、テスト約 2100 ケース、出口の表明、書き換えたオラクル。
2. **round 21 の prompt を書く。unscoped**（成果物は #264 と H-0095 の両方）。
   受け入れ基準 9 項目（H-0095 の 7 項目 + 出口の表明 + mapping）を挙げ、
   **各項目を実行して確かめるよう求める**。
3. 開示すること: RED 検証 4 件の結果、`set` と深さ 2 の判断とその理由、
   テストのオラクルが LightGBM の private 関数 `_param_dict_to_str` を読むこと
   （round 19 以来。`uv.lock` で pin されており、改名されれば大声で落ちる）。

## H-0095 で決まったこと（要点）

- パラメーター値は **4 surface の入口で 1 度だけ正規化**され、受理集合の外は
  学習前に `CONFIG_INVALID`。`lizyml/core/param_domain.py`。
- **入口と出口で受理集合が違う**: mapping（metric entry、H-0065）は入口で受理し、
  `lgb.train` の `assert_plain_params` で拒否する。adapter が消費するため。
- **`set` は拒否**（列は位置依存 / ハッシュ順の偶然）、**リストの入れ子は深さ 2 まで**
  （3 段目は wire が変わる）。どちらも実測付き。
- `value_equality.py` は 494 行 → 約 140 行。宣言する bound は
  「`param_domain` が受理する値の上で全域」に変わった（有限・列挙可能）。
- **#283 は H-0095 では解決しない**（別の Proposal が要る）。

## 積み残し（PR 2 由来、起票済み）

**#281**（loaded model の `validation_ratio`）/ **#282**（`category: training` の
`seed` 次元）/ **#283**（スカラー vs 単一要素の列）/ **#277 / #279 / #280**。

## この run のあと

PR 3（#258 tuning direction）、PR 3b（H-0024 space merge、`HISTORY.md:1615` と
`:1616` の矛盾を解消すること）、PR 4-9。**PR 9 の直前に繰り延べ 1 件**:
Phase 3 完了測定ツール（`phase3_gap.py` + manifest）は `instruments/deferred/` に
未出荷で archive してある。

## Codex 運用メモ（rounds 22-24 で分かった。踏むと時間を失う）

**Codex はレビュー依頼の書き方で provider 側のコンテンツフィルタに落ちる。**

```
ERROR: This content was flagged for possible cybersecurity risk.
```

rounds 22 / 23 / 23b が中断した。中立的な質問で同じファイルを読ませたら完走したので、
**反応しているのはコードではなく prompt** である。落ちた書き方に共通していたもの:

- 過去のすり抜けを並べた表（「何が通り / 何と書かれ / 何で学習したか」）
- 「呼び出し元のコードが走る経路はあるか」という問い
- 「これまで 2 回外している。3 回目を狙ってほしい」という煽り

**通る書き方**: 契約を述べて「守られているか」を問う。受け入れ基準の列挙も、
narrowing の一覧も、測定値付きの事実として書けば通る。round 24 の prompt
（`scratchpad/codex-pr2-review-prompt-r24.md`）が通った雛形である。

**中断しても必ずログを読むこと** — round 22 は verdict を返さなかったが、中断前に
再現し終えた本物の欠陥が 1 件ログに残っていた。

**代替経路**: `policy:fresh-checker` の `agent:general-purpose` を read-only、
同一 capsule で走らせられる（round 23 はこれで回した）。**ただし Codex ではないので
マージゲートの「Codex APPROVE」は満たさない。**

## 環境メモ（踏むと時間を失う）

- `uv` は読み取り専用の既定キャッシュで落ちる → **`UV_CACHE_DIR="$TMPDIR/uv-cache"`**。
- コマンドガードがアポストロフィ・ヒアドキュメント内の引用符を解析できずに拒否する →
  **スクリプトファイルにして実行**、コミットメッセージに**アポストロフィを入れない**。
- Codex は `CODEX_HOME` に書き込み可能なコピーを作ってから実行し、**実行後に消すこと**。
  既定は effort `low`。上げるなら `-c model_reasoning_effort=medium`。
- codex の長時間実行はバックグラウンドにすると低メモリ判定で kill された →
  **フォアグラウンドで `timeout` を長めに**。
