# PR 2 — Codex review round 21（2026-09-08、unscoped）

head `8f63a02` / `gpt-6-astra`, effort `low` / **REQUEST_CHANGES**
/ blocking **3**（`deliverable-path` 2、`periphery` 1）。

**このラウンドは #264（H-0094）と H-0095 を合わせた unscoped レビュー**（管理者指示
「F を実装後、両方あわせて Approve を取得してください」）。事前に rounds 19-21 の
関係監視を実施（`pr2_monitor_round1921.md`、CONVERGING / continue）。

監視が**事前に宣言した反証条件**を prompt の最重要問いとして載せた:

> 導出が生成しなかった *形* であって、それを受理すると `lgb.train` に届く bytes が
> 変わるもの。記録層の指摘や正当な形の過剰拒否は反証しない。

**finding 1 はまさにそれで、監視の CONVERGING は反証された。**

---

## finding 1 — [deliverable-path] numpy を継承で受理していた（DC1）

`param_domain.py` の numpy 分岐は `isinstance(value, np.generic)` で、`.item()` を
**信頼**していた。2 通りに破れる:

```
値                                    caller が書く wire   学習に届く wire
np.float64 のサブクラス（__format__ が嘘）   0.9              0.1
np.timedelta64(1, "ns")                   1 nanoseconds    1
```

**どちらも fit は完了する。** 出口の表明は変換後の素の値を見るので検出できない。
**これは過剰拒否ではなく、静かな受理と学習入力の変化**である。

`np.timedelta64` が入ってしまう理由は非自明で、実行して分かった:
**`np.timedelta64` は `np.integer` のサブクラスである。** 型集合を numpy の抽象基底から
導出しても、これは中に入る。

### 修正 — 2 段の防御、それぞれ別のものを買う

1. **`NUMPY_SCALAR_TYPES` を numpy 自身の階層から導出し、厳密な型一致で受理する。**
   `np.integer` / `np.floating` / `np.bool_` / `np.str_` の下の具象型のうち
   **numpy 自身が定義したもの**だけ。`datetime64` / `complex128` / `void` / `bytes_`
   は構成上外。**これが買うのは「正規化中に呼び出し元のコードが 1 行も走らない」こと** —
   `.item()` も `__format__` も numpy 自身の実装になる。rounds 16-20 が費やされた軸が
   構成上消える。
2. **書かれる形を検査する。** `format(plain, "") != format(value, "")` なら拒否。
   `timedelta64` を捕まえるのはこちらで、**型集合の中にいるので 1 だけでは捕まらない**。

RED 検証で 2 段が**別々に**効いていることを確認した（1 を戻すと `item` が例外を投げる
サブクラスが `LizyMLError` 以外を投げて出てくる／2 を戻すと `timedelta64` が通る）。

## finding 2 — [deliverable-path] `values_differ` が全域でなかった（DC7、宣言違反）

`_comma_form_matches` は `float(part) == float(element)` を `(TypeError, ValueError)`
だけで囲っていた。**Python の `int` に幅は無い**ので `10**400` は受理集合の内側の
ごく普通の値（シリアライザは桁をそのまま書く）だが、`float()` は `OverflowError` を投げる。

```
normalise_params({"learning_rate": 10**400, "eta": "1"}) → 受理
normalise_and_check(...)                                  → OverflowError
```

**「受理集合の上で全域」という宣言そのものを反証する。** 宣言は正しく、コードが
例外 1 つ足りなかった。`OverflowError` を捕捉するよう修正。

## finding 3 — [periphery] 導出テストが「広がり」を検出できなかった（DC3）

`test_the_sequence_types_are_the_ones_the_serialiser_joins` は
「こちらが受理する各名前が join 分岐に**現れるか**」しか見ておらず、
**シリアライザが新しい列型を得ても永遠に通り続ける**。レビュアーは in-memory で
広げたソースを食わせて実証した。「Both directions」という docstring と受け入れ基準 2 が
未達だった。

修正: join 分岐の `isinstance` タプルから**名前を抽出して集合として比較する**。
実測（read-only の mutation）:

```
frozenset を追加 → 通る（正しい: frozenset は REFUSED_SEQUENCE_TYPES に記録済み）
deque を追加     → 落ちる
array を追加     → 落ちる
```

---

## 受け入れ基準に対するレビュアーの結果

1 wire 保存 / 2 導出 / 6 全域 が finding で反証、3 / 4 / 5 / 7 / 8 / 9 は合格。

## レビュアーの環境上の注記（欠陥ではない）

一時ディレクトリが使えず 12 件が setup error、lifecycle grid も同じ理由で止まった。
こちらの環境ではフルスイート **7432 passed / 256 skipped**、grid は exit 0。

## 修正後

`ruff check .` / `ruff format --check .` / `mypy lizyml/` clean、フルスイート
**7432 passed / 256 skipped**、lifecycle grid exit 0。RED 検証 3 件すべて赤を確認。
