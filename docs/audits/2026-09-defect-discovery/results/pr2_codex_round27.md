# PR 2 — review round 27（2026-09-09、Codex 完走、H-0096 に対する最初のラウンド）

依頼文: `prompt-templates/pr2-review-round27.md`（head `5e23617`）。
**H-0096 の変更に対して走った最初のラウンド**であり、rounds 1-26 が
レビューしていた規則そのものが差し替わっている。

## VERDICT: `REQUEST_CHANGES`（**1 件**、P2、`deliverable-path`）

指摘数の推移: round 25 = 3、round 26 = 2、**round 27 = 1**。
ただしラウンドごとにレビュー範囲が違うので、これは減少率の測定ではない。

## 指摘 —— 拒否の**報告**が値の印字可能性に依存していた

`lizyml/estimators/lgbm/adapter.py` の `_pop_by_identity` が、
`LizyMLError` を組み立てる際に `dict(supplied)` をメッセージへ書式化していた。
Python の `int` は `sys.get_int_max_str_digits()`（既定 4300）桁を超えると
**十進テキストを持たない**ので `str()` が raise する。結果、約束していた
`CONFIG_INVALID` が**素の `ValueError`** になる。

レビュアーが `5e23617` で実行した再現:

```
int_max_str_digits: 4300
ValueError None
Exceeds the limit (4300 digits) for integer string conversion
```

**規則の判定そのものは値に依存していない**（綴りの数だけを見る）。
依存していたのは**報告**のほうであり、H-0096 が「値を読まない」と宣言した以上、
報告も値に依存してはならない。

レビュアーは bound を正しく述べている: **これは出荷経路の迂回ではない。**
4 つの facade surface は同じ組で `CONFIG_INVALID` を返す —— `param_domain` が
「文字を作れない値」を先に拒否するためである（round 24 の帰結）。
再現は adapter ヘルパーの直接呼び出しによる。

## こちらが実行して見つけた、報告されていない同じ欠陥

**`check_duplicate_identities` にも同じ形があった。** 同じ値で実行して確認:

```
--- reported: _pop_by_identity            ValueError code=None
--- the sibling: check_duplicate_identities  ValueError code=None
```

**片方だけ直すことはしなかった。** 「門を足したが 1 か所を通していない」は
round 25 指摘 2 → round 26 指摘 1 で 2 連続した形であり、D13 はそれを理由に
取り直された判断である。同じ形を、今度は修復の側で作ることになる。

**3 か所目もある**: `LizyMLError.__repr__` は `context` を `!r` で描画するので、
context に生の値を置くこと自体が同じクラスの一歩後ろの位置になる。

## 修正

**両方の refusal から値の描画を外し、綴りだけを名指す。**
綴りは `str` のキーであり必ず印字できる。値は message からも context からも外した。

- `adapter.py`: `sorted(supplied)` を message に、`context={"parameter", "spellings"}`。
- `_model_factories.py`: `sorted(written)` を各行に、`context[...]["spellings"]`。

既存テスト 1 件（`test_two_spellings_of_an_ordinary_parameter_are_refused_too`）が
`context["conflicts"][0]["written"]` に値を期待していたので、`spellings` を見るように
更新した。**値を context に残す選択はしなかった** —— `__repr__` が `!r` で描画する以上、
それは印字できない値を下流に渡すことだからである。

## RED 検証

`test_neither_refusal_needs_the_values_to_be_printable` を先に書き、
**報告どおり `ValueError` で赤になることを確認**してから修正した。
テストは**両方の呼び出し点を 1 つのテストで**表明し、`str()` と `repr()` の
両方が描画できることまで見る。

## 修正後

- フルスイート **7708 passed / 230 skipped**
- `ruff check` / `ruff format --check` / `mypy lizyml/` clean

## 停止条件の判定: **発火していない**

条件 C は「round N の修正が書いたコードの欠陥が round N+1 で出たら停止」である。
**round 27 は H-0096 に対する最初のラウンド**であり、直前のラウンドの修正を
レビューしたものではない。指摘は H-0096 が書いたコードの内側だが、それは
**この変更が初めてレビューされた**という意味であって、修復の再帰ではない。

rounds 24-26 の監視が固定した反証条件（ラウンド開始時点の §2 の表に無い sink を
名指したら DRIFTING）も発火していない。指摘は `deliverable-path` に留まっている。

## レビュアーが述べた bound

パラメーター領域スイート 5,263 passed / 230 skipped（候補 1,328 = 受理 929 /
拒否 399）。両ヘルパーは 368 の別名ペアと、3 綴り以上を含む 46 グループを
すべて拒否。R3 は `param_domain.py` の実行可能 AST が不変であることで確認。
R4 は 109 の production ファイルに参照 0 件。

**やっていないこと（レビュアー自身の申告）**: フルスイートの再実行、
export / artifact のラウンドトリップ、lint と mypy、
述べたテスト母集団を超える値の列挙。
