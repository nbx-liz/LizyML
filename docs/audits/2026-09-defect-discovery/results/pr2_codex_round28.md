# PR 2 — review round 28（2026-09-09、Codex 完走、`5715ee2` に範囲限定）

依頼文: `prompt-templates/pr2-review-round28.md`。

## VERDICT: `REQUEST_CHANGES`（**1 件**、`periphery`）

指摘数の推移: 25 = 3、26 = 2、27 = 1、**28 = 1**。
ただしラウンドごとに範囲が違うので減少率の測定ではない。

## 指摘 —— 回帰テストがインタプリタの設定に依存していた

`tests/test_core/test_fit_params_override.py` の
`test_neither_refusal_needs_the_values_to_be_printable`（**`5715ee2` が書いた**）が
`10 ** (sys.get_int_max_str_digits() + 1)` で「印字できない値」を作っていた。
**`PYTHONINTMAXSTRDIGITS=0` は変換上限そのものを無効化する**ので
`sys.get_int_max_str_digits()` は `0` を返し、値は `10` になる。
テストはどちらのヘルパーにも到達せずに落ちる。

レビュアーの再現をこちらでも実行して確認した:

```
PYTHONINTMAXSTRDIGITS=0 pytest ...::test_neither_refusal_needs_the_values_to_be_printable
>       with pytest.raises(ValueError):
E       Failed: DID NOT RAISE <class 'ValueError'>
```

**「拒否は値を読まない」ことを主張するテストが、値が読めないことをインタプリタの
設定に頼っていた。** 修正は依存そのものを外すこと ——
`__str__` / `__repr__` / `__format__` がいずれも raise するオブジェクトを使う。
`PYTHONINTMAXSTRDIGITS` を unset / `0` / `640` の 3 通りで実行して確認した。

RED 検証: 新しい値を修復前のメッセージ書式（`f"{dict(supplied)}"`）に通すと
`RuntimeError` になることを実行して確認（出荷形の `sorted(supplied)` は通る）。

## レビュアーが確認した production 側 —— C1-C3 は成立

**指摘は production に 1 件も無い。** レビュアーは 6 つの値の形で両ヘルパーを実行した:
描画が raise するオブジェクト / 描画が raise する `int` サブクラス / 入れ子の list と
dict / 循環参照する list / 上限超えの `int`。**12 の例外すべてが `str` / `repr` /
traceback 整形に耐えた。** 単一綴りの場合は値が保たれることも確認。

- 値はどちらのヘルパーの context にも残っていない。ヘルパー内・`LizyMLError` の構築と
  描画・呼び出し元と tuning のエラー処理に**4 か所目は見つからなかった**。
- production 108 / test 181 / script 2 ファイルを走査し、削除した context キーの
  消費者は残っていない（**外部の消費者は未検証、context のスキーマは変わっている**）。

## レビュアーの非 blocking な観測（記録のみ）

**adapter 側の拒否メッセージは surface を名指していない**（facade 側は名指す）。
`5715ee2` より前からの性質である。`_pop_by_identity` は surface を引数に取らないので、
名指すには署名を変えることになる。**本ラウンドでは直していない。**

## 停止条件の判定: **C が発火した**

条件 C =「round N の修正が書いたコードの欠陥が round N+1 で出たら停止」。
**round 28 の指摘は `5715ee2`（round 27 の修正）が書いたテストの欠陥である。**

そして rounds 26-27 の監視（`results/pr2_monitor_round2728.md`）が明示していた:

> **リセットはここで使い切っている。** round 28 が `5715ee2` の書いたコードの
> 欠陥を出せば C は再度発火し、**次はリセットの根拠が無い。**

**その通りになった。round 29 は開かず、判断を仰ぐ。**

## 判断材料

- **指摘の位置が 2 ラウンド続けて deliverable の外にある。** round 27 は拒否の
  **報告**経路、round 28 は**テスト**。**H-0096 の規則そのもの（拒否の判定）に
  対する指摘は 0 件**であり、レビュアーは 6 つの値の形で成立を確認している。
- **形は round 25-26 と違う。** あのときは「門を足して 1 か所を漏らす」が 2 連続し、
  修復が次の欠陥を作っていた。今回の 2 件は**別クラス**（値の描画依存 / テストの
  環境依存）で、round 27 の修復は round 28 の指摘を作っていない ——
  round 28 が指したのは round 27 の修復に**付随したテスト**である。
- **どちらも安く直せた。** production 33 行と、テスト 1 か所の値の作り方。
- **未取得のまま 28 ラウンド。**

## この修正で起きた副次の事故（記録）

修正の途中で `import sys` を削除して 1 テストを壊した。原因は
**トップレベルに `import sys` を足したことで auto-lint が同じテスト関数内の
ローカル `import sys`（`9d33737` の 2723 行目）を冗長として削除しており**、
その後トップレベル側を消したので未定義になったこと。
フルスイートで検出して復旧済み（7710 passed）。

`~/.claude` のメモリは「未使用 import を ruff --fix が消す」形を機械化済みだが、
**これは「重複 import を足すと既存のローカル import が消える」という別の形**であり、
手で消したので既存の機械化（`hooks/auto-lint.sh` の F401 warning）では防げなかった。

## 修正後

- フルスイート **7710 passed / 230 skipped**
- `ruff check` / `ruff format --check` / `mypy lizyml/` clean

## レビュアーが述べた bound

作業ツリーで 5,506 passed / 230 skipped / **13 setup error**（read-only 環境が
pytest の一時ディレクトリを作れないため）。
**やっていないこと**: フルスイートの再実行、lint と mypy、artifact / export の
完全なテスト、値の網羅的な列挙（有限の母集団である旨を明記）。
