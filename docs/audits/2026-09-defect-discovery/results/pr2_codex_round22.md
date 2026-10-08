# PR 2 — Codex review round 22（2026-09-08、unscoped、**verdict 未取得**）

head `261457b` / `gpt-6-astra`, effort `low` /
**実行は provider 側のコンテンツフィルタで中断され、verdict は返らなかった**。

```
ERROR: This content was flagged for possible cybersecurity risk.
```

敵対的オブジェクトを構築して欠陥を再現するという**このレビュー自体の手法**が
誤検知されたものと見られる。**ただし中断前に、レビュアーは 1 件を再現し終えており、
ログにその再現コードと出力が残っている。**

## 中断前に再現された 1 件 — [deliverable-path] 呼び出し元が numpy を自称できた（DC1）

**監視が事前に宣言した反証条件 (b)（silent wire change）そのもの。**

round 21 の修正は numpy の受理を「numpy 自身の階層から導出した集合の厳密型一致」に
変えたが、その導出は

- `np.integer` などの `__subclasses__()` を**import 時に**歩き、
- `kind.__module__.split(".")[0] == "numpy"` で絞っていた。

**`__module__` はクラス本体に書けるただの属性である。** レビュアーはこう書いた:

```python
class Disguised(np.float64):
    __module__ = "numpy"
    def __format__(self, spec):
        return "0.1" if getattr(self, "converted", False) else "0.9"
    def item(self):
        self.converted = True
        return 0.1
```

実測（レビュアーのログ、および 4 surface すべてで再現）:

```
                              caller が書く wire   学習に届く wire
model.params                    learning_rate=0.9   0.1, 0.1, 0.1
fit(params=)                    learning_rate=0.9   0.1, 0.1, 0.1
tuning best_model_params        learning_rate=0.9   0.1, 0.1, 0.1
calibration.params              learning_rate=0.9   0.001, 0.001, 0.1, ...
```

**fit は完了する。** 2 段の防御が両方破られていた:

1. **型集合**は `__module__`（呼び出し元が書く）を信じ、しかも `__subclasses__()` の
   走査が **import 時**なので、**そのクラスが lizyml の import より前に定義されたか
   どうかで答えが変わる**（レビュアーの再現は前に定義していた）。
2. **書かれる形の検査**は `format` を 2 度呼ぶが、この値の `__format__` は
   `item()` が立てたフラグで答えを変えるので、**変換後の値を変換後の値と比べていた**。

### 修正

1. **型集合を `vars(numpy)` から読む。** 「numpy がその名前で export している型か」は
   **同一性**の問いであり、呼び出し元が主張できない。import 順にも依存しない。
2. **`format(value, "")` を `.item()` の前に読む。** 型集合が正しければ状態を持つ値は
   そもそも入らないが、**「もう一方の検査が効いていることに正しさが依存する検査」は
   2 段目ではない**。

### RED 検証で 2 度目のやり直しをした

最初に書いたテストは**どちらの revert でも緑のままだった**:

- witness をテスト本体で定義すると **import より後**になるので、
  `__subclasses__()` 実装でも集合に入らない → 緑。
- 状態を持つ値は型集合を通れないので、順序の検査に届かない → 緑。

書き直した:

- **導出関数を witness 定義後に呼び直す**（`_derived_numpy_scalar_types()`）。
  走査実装なら入るので赤くなる。
- **型集合を monkeypatch で意図的に緩めて**、書かれる形の検査だけを単独で試す。

両方 RED を確認。**この run で「テストが別の理由で緑だった」のは 5 回目**である。

## 状態

`ruff` / `mypy` clean、フルスイート **7436 passed / 256 skipped**、
lifecycle grid exit 0。round 23 が必要（verdict 未取得のため）。

## 運用上の注記

**Codex の実行が provider 側フィルタで落ちうる。** 敵対的オブジェクトを構築する
レビューでは再発しうるので、**ログを必ず読むこと** — 今回は verdict が無いだけで、
中断前の再現は本物だった。
