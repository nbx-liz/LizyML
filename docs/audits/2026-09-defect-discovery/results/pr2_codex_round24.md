# PR 2 — Codex review round 24（2026-09-08、unscoped）

head `5a96daa` / `gpt-6-astra`, effort `low` / **REQUEST_CHANGES**
/ blocking **2**（どちらも `deliverable-path`、どちらも実行して再現）。

## Codex が完走した — 原因はコードではなく prompt の書き方だった

rounds 22 / 23 / 23b は provider 側のコンテンツフィルタで中断していた。
**中立的な質問（「このモジュールは何をするか 3 文で」）で `param_domain.py` を
読ませたところ問題なく完走した**ので、反応していたのは**ファイルの中身ではなく
レビュー依頼の書き方**だと分かった。round 23b にはまだ

- 過去のすり抜けを並べた表（「何が通り / 何と書かれ / 何で学習したか」）、
- 「呼び出し元のコードが走る経路」という問い、
- 「これまで 2 回外している。3 回目を狙ってほしい」という煽り

が残っていた。round 24 はこれらを外し、**契約の検証**として書いた:

> 受理した値は、正規化の前後で同じ文字列にシリアライズされる。保証できない値は
> surface で拒否する。これは守られているか。

**完走した。** 教訓は運用メモに残す（`pr2_NEXT.md`）。

## finding 1 — [deliverable-path] 受理した型が「書ける値」とは限らない

2 つの値が surface を通り、学習前の表明も通り、**LightGBM の内側で raise した**:

```
PurePosixPath("f.json")   surface 受理 | 表明 受理 | シリアライザ TypeError
10**5000                  surface 受理 | 表明 受理 | シリアライザ ValueError
```

- **`PurePosixPath` / `PureWindowsPath`**: シリアライザは
  `isinstance(val, (str, Path, ...))` で判定する。**pure path は `Path` ではない。**
  受理集合にこれらを入れていたのがそのまま誤りだった。
- **`10**5000`**: Python の `int` に幅は無いが、**インタプリタの十進変換上限
  （既定 4300 桁）を超えると `str()` は桁を返さず raise する**。
  round 21 で `float()` の `OverflowError` を直したが、`str()` の `ValueError` は
  別物だった。

**修正 — 型からの推定をやめ、文字列を実際に要求する。**
`_written_or_refused(value, write)` を scalar 位置（`format`）と element 位置
（`str`）の両方に置き、書けない値は surface で `CONFIG_INVALID`。
path は `Path` のフレーバーだけに絞り、**「受理する path 型はすべて
`issubclass(kind, pathlib.Path)` であること」をテストで固定**（シリアライザ自身の
判定から導出）。

## finding 2 — [deliverable-path] `values_differ` がまだ全域でなかった

`_comma_form_matches` の `str(element)` が例外ハンドラの外にあり、
`values_differ("", 10**5000)` が `ValueError` を投げた。
入口で拒否されるようになったので到達しなくなるが、**「全域である」と宣言する関数が
もう一方の門が効いていることに依存してはならない**ので、こちらも囲った。

## レビュアーの受け入れ基準の結果

1 と 6 が finding で反証。**2 / 3 / 4 / 5 / 7 / 8 / 9 は合格**。
特に 8 は**レビュアーが AST で学習サイトを列挙**して 2 件（adapter:243、
isotonic:139）を確認し、どちらにも表明が先行することを確かめている。

## 修正後

RED 検証 2 件とも赤を確認（pure path を戻す / writability 検査を両位置から外す）。
`ruff` / `mypy` clean、フルスイート **7441 passed / 256 skipped**、
lifecycle grid exit 0。

## レビュアーの環境上の注記（欠陥ではない）

一時ファイルが作れず 13 件が setup error。フルスイート・lint・CI 12 lane・
生成スクリプトは未実行と明示している。
