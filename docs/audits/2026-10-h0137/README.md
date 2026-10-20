# H-0137 の測定（#385）

HISTORY.md の H-0137「実測」の各主張と、それを出したファイルの対応。どれもリポジトリの根から実行する。通信が要るものは CI では走らせない。

| 主張 | スクリプト | 出力 | 通信 |
|---|---|---|---|
| カーネルの中で、ガードが拒否する 15 個の入口に届く呼び出しは 8 本とも 0 件。各 notebook が `data/` から読み込みで開くファイルは manifest の記録と同じ。`socket.*` の監査イベント（`_socket` を直接呼ぶものも含む）も記録する | `socket_census.py`（`heavy.sh` で実行） | `socket_census.json` | 不要 |
| `bind`・`connect`・`sendto`・`sendmsg` はホスト名を監査イベントより前に解決する（監査のフックでは拒否できない）。各関数・メソッドが出す監査イベント | `socket_audit_events.py` | `socket_audit_events.out` | 不要 |
| 上の記録の仕組みは、今の取得方法の通信を捕まえる（陽性対照） | `socket_census_control.py` | `socket_census_control.out` | 要 |
| 4 つのデータで、同梱したファイルから作るデータフレームが、取得元から作るものと一致する。作り直したファイルの sha256 は 2 回とも manifest と同じ | `scripts/vendor_notebook_data.py --out <dir>` を 2 回 | `rebuild_run1.out`、`rebuild_run2.out` | 要 |
| sdist に `notebooks/data/` が入り、wheel には入らない。大きさ | `uv build` を develop と branch で | `dist_contents.out` | 不要 |
| ライセンスと帰属の表示 | WebFetch で出典のページを開いた | `licences.md` | 要 |
| ガードのテストが、ガードを壊した版（mutant）を捕まえる | `mutate_guard.sh` | `mutate_guard.out` | 不要 |
