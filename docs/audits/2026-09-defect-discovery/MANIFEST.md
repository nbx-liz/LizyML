# 2026-09 系統的欠陥発見 — アーカイブ manifest

Issue #258–#273 を生んだ調査の作業一式。**この作業は `/tmp` で行われ、2 度失われた**
（2026-09-03 / 2026-09-05）。ここに置くのはその再発防止であり、GitHub Issue 群と並ぶ
2 つ目の恒久記録である。

## 由来 — 何が原本で、何が復元物か

**ここにあるファイルの大半は原本ではなく、セッション transcript から復元したものである。**
`Write` / `Edit` ツールで作られたファイルは全文が transcript に残るため完全に復元できる。
一方、**スクリプトが `write_text()` でディスクへ直接書いた生成物は transcript に本文が無く、
復元できない。**

| 状態 | 対象 |
|---|---|
| 原本と同一（復元済み） | `phase3-plan.md`, `RESUME.md`, `results/*.md`, `instruments/*` |
| **失われた（復元不能）** | `results/r16_verdicts.json`（最終版）, `results/r38_merged.json`, `reviews/noob-{A,B,C}.json` |
| 部分的に再構成 | 再監査の `missed` clause 78 件（`instruments/extract_missed_clauses.py` が transcript から再生する） |

失われた 3 ファイルは再監査の verdict データそのもので、再生成には Codex 3 バッチの
再実行が要る。結論・数値・entry ID は `results/r42_reaudit.md` に残っている。

## 中身

| パス | 内容 |
|---|---|
| `phase3-plan.md` | Phase 3（修復）の計画。PR 0–9 の分割、各 PR の RED 条件・母集団・exit。Codex 7 ラウンドを経ている |
| `RESUME.md` | 調査の再開手順 |
| `results/FINDINGS*.md` | Phase 1（発見）の所見。D1–D6 の各探索と修復方針 |
| `results/r42_reaudit.md` | no-obligation 38 件の再監査（2026-09-05）。#271 の母集団が 18→42 に動いた経緯 |
| `results/r39_h0024_repro.md` | H-0024 が決めた 2 契約が未実装である件の実行再現 |
| `results/r16_summary.md` ほか | 各探索の中間集計 |
| `instruments/kill_producers.py` | 変異注入。テストが実際に落ちるかを測る |
| `instruments/trace_plugin.py` | 実行経路トレース。到達可能性の測定に使う |
| `instruments/firing_rate_plugin.py` | 条件の発火率測定（`change-gate.md` の Firing rate 用） |
| `instruments/run-exclusive.sh` | CPU 競合を避けて重いジョブを直列化する |
| `instruments/recover_from_transcript.py` | transcript から `Write`/`Edit` を再生してファイルを復元する。**この archive 自体がこれで作られた** |
| `instruments/extract_missed_clauses.py` | 再監査の `missed` clause 78 件を transcript から再構成する |

`instruments/` はリポジトリのテストにはならない道具である。CI は動かさない。

## 除外したもの

- **Codex の認証情報（`codex-home/`）** — `auth.json` を含むため、復元対象からも
  この archive からも意図的に除外している。
- 一度きりの scratch スクリプト（`d2_inspect.py`, `bump_rev10.py`, `batch*.json` など、
  約 170 ファイル）。再現に必要なら `instruments/recover_from_transcript.py` で
  transcript から取り出せる。

## この調査が今どこにあるか

Phase 1（発見）と Phase 2（起票）は完了。Phase 3（修復）は `phase3-plan.md` の PR 0 から
着手した（H-0092 がその最初の Proposal）。**`phase3-plan.md` の PR 9 の数値は旧版で、
現在の母集団は 40 entries / 129 clauses / 77 edits**（#271 の 2026-09-06 コメント、および
`phase3-plan.md` 冒頭の Revision 5 バナー参照）。

PR 0 のレビュー記録は `results/pr0_codex_round1.md` / `pr0_codex_round2.md`。
各ラウンドの verdict と、それに対する main context の処置を併記している。

### 未充足の保証（PR 0 の繰り延べ）

`phase3-plan.md` §8 が定める**Phase 3 完了の測定手段は、まだ出荷されていない**。
`phase3_gap.py` / `phase3_manifest.json` / その単体テストは PR 0 の Files に挙がって
いたが、scratchpad 消失後に transcript から復元した manifest が 4 箇所古く、うち 1 箇所は
編集ではなく道具側の設計判断を要する（#271 の母集団は本 run が Proposal を足すたびに増える）。
このまま出荷すると、正しく修復された issue を incomplete と報告する DC7 になるため、
`instruments/deferred/` に README 付きで置いて繰り延べた。**それまで「Phase 3 完了」は
PR ごとの判断に依存する**（§8 が防ぐために存在する DC5 の形）。残作業は manifest の
全行を実際に着地したテストと突き合わせる 1 パスと、proposition 4 の判断。
