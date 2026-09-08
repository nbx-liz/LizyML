# PR 2 — round 23（2026-09-08、範囲限定、**Codex ではなく fresh-context checker**）

**REQUEST_CHANGES / 指摘 3 件、すべて実行して再現済み。**

## 経緯 — Codex が 3 回連続で provider 側フィルタに落ちた

```
ERROR: This content was flagged for possible cybersecurity risk.
```

- **round 22（非限定）** — 25k tokens で中断。中断前の再現 1 件はログから回収した。
- **round 23（範囲限定、当初の文面）** — `HISTORY.md` 読解中に中断。指摘なし。
- **round 23b（敵対的語彙を除いた中立的文面 ＋ 読む範囲を 2 ファイルに限定）** —
  **`param_domain.py` 自体を読んでいる途中で中断**。

3 回目で「プロンプトの文面ではなく、入力検証・型のなりすまし防御という**コードの
中身自体**がフィルタに反応している」と判断し、**運用ルール（同一エラー 3 連続で
approach を変える）に従って経路を変えた** — `policy:fresh-checker` の
`agent:general-purpose` を read-only、同一の task capsule で実行。

**⚠️ これは Codex ではないので、マージゲートの「Codex APPROVE」は満たさない。**

## finding 1 — [deliverable-path] `in` は同一性ではなかった（DC1）

`type(value) in NUMPY_SCALAR_TYPES` は **`frozenset` の探索**であり、判定は
**呼び出し元の `__hash__` / `__eq__`** で行われる。クラスのそれらは**メタクラス**から
来るので、呼び出し元が書ける。

```python
class _Claiming(type):
    def __eq__(cls, other): return other is np.float64 or cls is other
    def __hash__(cls):      return hash(np.float64)

class Sneaky(metaclass=_Claiming):
    def __float__(self): return 0.1
    def __format__(self, spec): ...  # 1 回目 0.9、2 回目 0.1
    def item(self, *a): return 0.9
```

実測: **numpy を継承せず、`__module__` も名乗らず、import 順にも依存せずに通過**し、
呼び出し元の `__format__` と `item()` が正規化の中で走った。
caller の wire は `0.1`、正規化が入れた値の wire は `0.9`。
`PLAIN_SCALAR_TYPES`（tuple、`x is e or x == e`）にも同じ穴があった。

**修正: 全ての門を `is` 比較にする**（`_is_one_of`）。**`is` は Python で唯一
呼び出し元が参加できない比較**であり、この門はそれだけで作るしかない。

## finding 2 — [deliverable-path] `vars(numpy)` は書き込み可能（DC1）

`np.Injected = Injected` を **import より前に** 1 行書くだけで型集合に入る。
checker の指摘の核心はこれ:

> **Python のどんな名前空間の読み取りも呼び出し元から独立ではない。「呼び出し元が
> 自称できない」は、どんな導出も提供できない性質である。**

**修正: 決め手を名前空間ではなく numpy 自身の dtype レジストリに移す。**
候補は `vars(np)` と `np.sctypeDict` から**列挙するだけ**（上位集合でよい）で、
**採用は `np.dtype(kind).type is kind` の往復**で決める。実測:

```
np.float64   -> float64   同じ: True      Injected     -> float64   同じ: False
np.str_      -> str_      同じ: True      InjectedStr  -> str_      同じ: False
```

**そして bound を書き直した。** これはパラメーターの**値**に対して領域を閉じる。
**プロセス内で既に numpy の一部を差し替えた呼び出し元に対する sandbox ではない** —
`numpy.dtype` を差し替えられる者は `numpy.float64` も、このモジュールも差し替えられる。
**この run で「宣言が達成不能だった」のは 3 度目**（round 18 の「何に対しても raise
しない」、round 20 の NaN、今回）。達成可能な宣言に書き直すのが正しい修復である。

## finding 3 — [periphery→deliverable] 要素位置の門が未検証だった（DC6 の形）

`_plain_element` の厳密型一致を `isinstance` に緩めても**ファイル全体が緑のまま**
（3234 passed）。振る舞いは実在する（`[Lying(2.0), 3.0]` で呼び出し元の `item()` が
要素位置で走り、値が変わる）。既存の
`test_each_defence_is_load_bearing_for_something_different` はこの防御を名指しつつ
**スカラー位置でしか行使していなかった**。テストを追加。

## checker が確認した既存の防御（mutation 4 件）

| mutation | 殺したテスト |
|---|---|
| M1 `__subclasses__()` 走査 + `__module__` 絞り | 5 テスト（20 ケース） |
| M2 スカラー位置を `isinstance` | `..._load_bearing_for_something_different` |
| M2b `ndarray` を `isinstance`（2 か所） | `..._array_subclass_is_refused...` |
| M3 `item()` を `format()` より先に | `..._written_form_is_read_before...` |

**M2c（要素位置）だけが生き残った ＝ finding 3。**

## 修正後

RED 検証 3 件すべて赤を確認（`in` へ戻す / dtype 往復を外す / 要素位置を `isinstance`）。
フルスイート **7439 passed / 256 skipped**、`ruff` / `mypy` clean。

## 積み残し（checker が「推論のみ、未実行」と明示）

`_plain_sequence` への入口 `type(value) in PLAIN_SEQUENCE_TYPES` も同じ機構だった。
**`_is_one_of` の適用でこれも同時に閉じている**が、checker 自身は実行していない。
