# H-0137 の測定（#385）

HISTORY.md の H-0137「実測」の各主張と、それを出したファイルの対応。どれもリポジトリの根から実行する。通信が要るものは CI では走らせない。

| 主張 | スクリプト | 出力 | 通信 |
|---|---|---|---|
| カーネルの中で、8 本とも `socket` の入口への呼び出し（ガードが拒否する `getaddrinfo` と `connect` を含む 15 個）も `socket.*` の監査イベントも 0 件。各 notebook が `data/` から読み込みで開いたファイルも記録した（round 1〜3 の設計の名残。今の設計は照合しない） | `socket_census.py`（`heavy.sh` で実行） | `socket_census.json` | 不要 |
| 今の取得方法（`fetch_openml`、URL を読む `pd.read_csv`）は、`getaddrinfo` と `connect` を通る（陽性対照） | `socket_census_control.py` | `socket_census_control.out` | 要 |
| 4 つのデータで、同梱したファイルから作るデータフレームが、取得元から作るものと一致する。作り直したファイルの sha256 は 2 回とも manifest と同じ | `scripts/vendor_notebook_data.py --out <dir>` を 2 回 | `rebuild_run1.out`、`rebuild_run2.out` | 要 |
| sdist に `notebooks/data/` が入り、wheel には入らない。大きさ | `dist_contents.sh <base>`（develop と branch を clean な worktree から build） | `dist_contents.out` | 不要 |
| ライセンスと帰属の表示 | WebFetch で出典のページを開いた | `licences.md` | 要 |
