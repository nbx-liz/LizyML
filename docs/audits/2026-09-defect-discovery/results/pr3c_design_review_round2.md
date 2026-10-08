# PR 3c 設計レビュー round 2（2026-09-15）

依頼: `prompt-templates/pr3c-design-review-round2.md`。対象: `results/pr3c_design.md` 改訂 2。
Codex `gpt-6-astra` effort medium、read-only、branch `fix/phase3-pr3c-calibration-params` @ `5ac725e`。
事前の監視: `results/pr3c_monitor_round1.md`（DELIVERABLE-FOCUSED / redirect → 主コンテキストは continue）。

## VERDICT: **ADHERES-WITH-CHANGES**（blocking なし）

- **round 1 の 5 件はすべて正しく反映されている**（§3.2 / §3.4 / §3.3 / §5 / §2・§3.5）。不完全な適用は無い。
- **Platt の定式化は正しい**: モデル・正則化なしの同時推定・平滑化目標値・初期値は、研究ノートと
  インストール済み sklearn に一致。`a = −A`, `b = −B` は `sigmoid(a·s + b)` と正しく対応。
- **format_version 2 の据え置きは条件付きで妥当。** `__setstate__` は pickle が当該クラスを復元するとき
  必ず呼ばれるので、`c_final` 以外の入れ物（list / dict / 別の結果オブジェクト）でも働く。

## 指摘と主コンテキストの確認

| # | 重さ | 指摘 | 主コンテキストの確認 | 処分 |
|---|---|---|---|---|
| 1 | should | **`tol` と `options` の優先順位が未定義。** 既定 `options` に `gtol`/`ftol` を常に入れると、scipy は `tol` を `setdefault` で渡すので**利用者の `tol` が効かない** | **確認**: `scipy/optimize/_minimize.py` の既定トレランス設定は `options.setdefault('ftol'/'gtol', tol)` | **採用** |
| 2 | should | **最適化手法と引数の組み合わせの契約が無い。** 例: `method="BFGS"` は `bounds` を使わない。名前と長さの検査だけでは、承認済みの上書きが警告つきで無視されうる | 妥当（scipy の手法ごとの `bounds` 対応は既知） | **採用** |
| 3 | should | **縮尺変更時の `x0` / `bounds` の座標系が未定義。** `F = s/k` 上の slope は `k` 倍になるので、初期 slope と slope の bounds も `k` 倍する必要がある。終了時だけ戻しても制約付き問題は同じにならない。「結果不変」は数学的等価であって有限精度の一致ではない | 妥当（sklearn の縮尺処理を読んで確認済み） | **採用** |
| 4 | should | **環境の記述が誤り。** `uv.lock` は Python `< 3.11` に sklearn 1.7.2 / scipy 1.15.3、`>= 3.11` に 1.8.0 / 1.17.1 を選ぶので、3.11 の `.venv` は**ずれていない**。また lowest-direct レーンは `--frozen` なので、レーン名だけでは 1.3 / 1.10 で動いた証拠にならない | **確認**: `uv.lock` に scikit-learn 1.7.2 と 1.8.0 の 2 エントリ（Python のバージョン marker で分岐）。CI は `uv sync --frozen --dev --resolution lowest-direct` | **採用。改訂 2 と研究ノートの記述は私の誤り** |
| 5 | should | **README の更新が漏れている。** README は「beta のときだけ scipy」と約束している。「sklearn は scipy>=1.10 を必須にしている」は確認した sklearn 1.8.0 のメタデータに限定すべき | **確認**: `README.md` の "plus `scipy` when the model uses beta calibration" | **採用** |
| 6 | note | **互換の根拠と保証範囲を精密にせよ。** version 据え置きの根拠は「移行が version 非依存だから」ではなく「公開 Artifact 契約と旧モデルの推論を保つ内部状態の変換だから」。式が同じでも浮動小数点の完全一致は示せない —— 旧 sklearn と同等の sigmoid 計算を指定し、通常値と極端なスコアで旧予測と比較する。新状態の再読込、未学習（`_model=None`）の扱いも明記。**H-0030 より前（確率入力）の artifact は係数の移行だけでは意味が解決しない**ので保証範囲から外す | 妥当 | **採用** |

## レビュアーが述べた bound

改訂 2、round 1、研究ノート、監視の記録、指定規約の該当節、calibration skill、元コミットの差分、
現行の校正・保存・生成経路、lock / CI / インストール済み sklearn・scipy のソースを読んだ。
層規約・OOF-only・outer split・covered 行との追加の矛盾は認めなかった。
**Platt 論文本文、最低版環境、リポジトリ全体は未確認**。`AGENTS.md` はリンク先欠落で読めなかった。
変更・ネットワーク・依存同期・テスト実行はしていない。
