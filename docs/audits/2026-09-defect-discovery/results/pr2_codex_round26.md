# PR 2 — review round 26（2026-09-08、Codex 完走、範囲限定）

`gpt-6-astra`, effort medium。head `e52f062`（レビュー対象は `65dfdbe` の 1 コミット）。
依頼は `prompt-templates/pr2-review-round26.md`。

## VERDICT: REQUEST_CHANGES（2 件、どちらも P2 / `deliverable-path`）

**round 25 の UTF-8 指摘は部分的にしか閉じておらず、述語の書き直しは新しい偽陽性を
持ち込んだ。**

### 1. numpy の文字列**要素**が UTF-8 検査を迂回する

`_plain_element` の numpy 分岐は `text = str(value)` を直接使い、**素の候補を返す前に
encode 可能性を見ていない**。したがって `[np.str_("\ud800")]` は
`["\ud800"]` に**正規化されてしまい**、どちらの消費者も encode できない。
さらにその結果をもう一度正規化すると `LizyMLError` になり、**冪等性の要件に違反する**。

**authorship**: 検査していない numpy 分岐は**このコミットより前からある**（修正の
漏れ）。ただし **1 回目と 2 回目で答えが変わる不整合は、このコミットが作った**
（親コミットは両方の pass で受理していた）。レビュアーが実行して確認している。

**レビュアーの網羅**: **サロゲート符号位置 2048 個 × 構成 12 通り**を列挙した。
素の文字列（scalar / element）、path（scalar / element）、mapping のキーと値、
numpy スカラー位置では**全件拒否**。**list / tuple / 入れ子 list / ndarray /
mapping 内の list に numpy 文字列が入った形は全件通過。**

**処方**: numpy 要素のテキストも `_written_or_refused(value, str)` を通すこと。

### 2. `repr` の一致は「正規化が値を変えなかった」ことを示さない

numpy が公式に持つ `np.printoptions(legacy="1.25")` の下では、`np.int64(1)` と
正規化後の Python `int` は**どちらも `repr` が `"1"`** になる。結果として
`is_accepted` / `is_plain` / 出口の表明が**変換されていない numpy 値を受理する**。
それは素の Python 値ではなく `json.dumps` で書けない。
`np.float32` / `np.bool_` / `np.int64` を含む list でも再現済み。

**authorship**: **このコミットが書いたコードの欠陥**。親の述語は同じ print option の
下で正しく `False` を返す。

**処方**: 拒否の権威は正規化関数のままにしつつ、「変わっていない」の判定を
**表示テキストではなく型を再帰的に保存しているか**で行うこと。

## 範囲限定の他の項目は通過

- fixture の numpy 軸は正しく導出されている: **17 型 / 候補 1318 / 受理 929 / 拒否 389**。
- 拒否分類を**全候補について評価**: **理由の無い拒否 0 / 受理なのに理由が付く値 0 /
  到達しない宣言理由 0**。6 理由の内訳は **67, 129, 133, 9, 9, 42**。
- `nan` / `-0.0` / `bool` vs `int` / それらを含むコンテナは正しい。tuple と path は
  「変換が要る」と正しく判定される。
- 有限標本の記述は round 25 の契約訂正に答えている。生成契約の `--check` も通過。
- 対象 6 ファイルのテスト: **7525 passed, 278 skipped**。

**レビュアーが述べた bound**: フルスイート・lint・mypy・CI 13 レーンは再実行していない。
別の numpy 版 / platform は試していない。PR の残りは再監査していない。
無限の値領域は列挙していない。反例は LightGBM の encode ヘルパと json の UTF-8 encode を
実際に通しているが、学習と export の完全な実行ではない。

## 停止条件の判定: **2 つとも発火した**

- **C（ユーザーが決めた authorship 条件）**: 「round N の修正が書いたコードの欠陥が
  round N+1 で出たら停止」。**指摘 2 がそれである**（`65dfdbe` が書いた `is_accepted`）。
- **rounds 24-26 監視が付けた take-stop トリガ**: 「round 26 が `65dfdbe` が書いた
  コードの欠陥を出したら、クラス単位の修復のあとの 2 連続 authorship 再帰なので
  round 27 を開かず止める」。**同じ指摘で発火。**

**監視の反証条件（D11 で固定した文言）は発火していない** —— 2 件とも
`param_domain.py` の中であり、§2 の表に無い sink を名指す指摘は 0 件。
形は「契約の内側」のままである。

したがって **round 27 を開かず、判断を仰ぐ。**
