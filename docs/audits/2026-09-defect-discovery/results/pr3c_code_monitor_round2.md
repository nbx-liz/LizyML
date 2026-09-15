# PR 3c コードレビュー — ループ監視（relational、rounds 1-2、2026-09-15）

運び手: fresh context・read-only のサブエージェント（general-purpose）。Codex がメモリ監視で 3 回連続停止したため。
capsule は `templates/review-loop-monitor-capsule.md` を relational で埋めたもの（spawn プロンプトに直書き）。

## VERDICT: **CONVERGING** ／ Recommendation: **`continue`**

- round 1（`f67f8c7`）: blocking 2 / should-change 1 / B4 8 行。production 修正 約 70 行。
- round 2（`761c05e`、round 1 の修正に限定）: blocking 1（A）/ should-change 1（C）/ note 1（B）。production 修正 約 35 行。
- 成果物側: 指摘はすべて H-0100 決定 4 の契約（反映するか学習前に拒否する、実行時と生成コード）の上。blocking 件数も修正量も減少。clean とされた部分が引き戻されていない。§2 は広がっていない。
- 周辺: 記録・手続き（prompt template、monitor 記録、#297/#298 起票、運び手の変更）。テストや仕組みの増築ではない。
- 観察（採点ではない）:
  - **A は authorship 型**（round N の修正が書いたコードの欠陥が round N+1 で出る。PR 2 を長引かせた形）。
  - 修正のたびに新しい検査面が生まれる（round 2 の修正は scipy 警告文の先頭一致と float 収まり判定を実行時と生成の両側に入れる）。
  - C の方向が全称命題（「どの入力でも拒否されるか」）に広がると PR 2 の原因 B と同じ構造。現時点ではその形ではない。
  - B は限定範囲の外から出た。

## 主コンテキストの処分: **`continue`（採用）**

監視の推奨どおり、**round 3 の前に停止条件を宣言する**:

> **round 3 で、round 2 の修正（`1e51c97`）が書いたコードの欠陥が見つかった場合は、round 4 を開かない。** 修正はせず、管理者に「受け入れ／分割／範囲限定でもう 1 回」を戻す。
> round 3 の範囲外の指摘は 1 行の Out of scope として受け取り、処分は §4 のバケットで決める（再レビューの対象にしない）。
> round 3 の問いは「round 2 の 3 件の修正がそれぞれの指摘を閉じたか、壊したものは無いか」に限り、全称命題の形にしない。
