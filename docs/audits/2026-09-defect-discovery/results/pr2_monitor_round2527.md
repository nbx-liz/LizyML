# PR 2 — rounds 25-27 relational monitor（2026-09-08）

`policy:loop-monitor`。read-only、fresh context。観測範囲 **25-27**（27 側は
round 26 の修正 `7ee10fe`。round 27 はまだ走っていない）。head `7ee10fe`。

主文脈は中心の問いを明示して渡した ——「**こちらの修正が欠陥を書いた**」が 2 連続で
起きており、前回監視は round 25 の修正を「クラス単位の修復」と呼んで**そのクラスは
閉じると予測した**が、予測は外れた。同じクラスの再発か別クラスかを判定させた。

## VERDICT: CONVERGING（周辺の増加は計測上ゼロ。ただし**横ばいであって閉じていない**）
## recommendation: **take-stop-condition**

## ラウンドごと（authorship 付き）

- **R25**（head `0c3eff8`、非限定）: blocking 3、すべて `param_domain.py` とそのテスト。
  f1 ← `48a0801` + `8743c7c` / **f2 ← `68d4972`（round 24 の修正）** / f3 ← `4161837`。
  監視が実行して確認: `git show 68d4972 -- lizyml/core/param_domain.py` の中に
  `is_accepted` / `is_plain` の変更は **0 件**、同じコミットが正規化だけを狭めている。
- **R26**（head `e52f062`、`65dfdbe` に範囲限定）: blocking 2、どちらも `param_domain.py`。
  **f1 ← 2 つの修正が続けて飛ばした分岐**（下記）/ **f2 ← `65dfdbe`**
  （`git log -S 'repr(normalised) == repr(value)'` は `65dfdbe` … `7ee10fe` のみを返す）。
- **R26 の修正**（`7ee10fe`）: production 1 ファイル `+41/−7`、テスト `+31`。
  新モジュール・新 instrument・新ゲートはゼロ。`0c3eff8..7ee10fe` の窓が足したのは
  ラウンドの記録だけ（verdict 2 / 監視 1 / prompt template 1）。**apparatus 増加ゼロ。**

## 中心の問いへの答え —— **D12 の数え方が 1 件足りず、形が変わる**

R25f2 と R26f2 は**別クラス**である（DC3 の宣言二重化 vs 呼び出し元が設定できる proxy）。
その意味で前回監視の「クラス単位の修復」は**自分が名指したクラスについては成立した** ——
`is_accepted` はもう領域を言い直していない。

**しかし監視は 3 例目を数えた。**

- `68d4972` は `_written_or_refused` を `_plain_element` の**素のスカラー分岐**に入れ、
  その 1 行下の **numpy 分岐**（`text = str(value)`）を残した。
- `65dfdbe` は `_encodable_or_refused` を scalar / path / mapping キーの位置に通し、
  **同じ分岐をまた飛ばした**（`git show 65dfdbe | grep '^[+-].*_or_refused'` は 5 hit、
  うちその分岐は 0）。
- **R26f1 がその分岐である。**

したがって **R25f2「門を足したが述語をそこに通していない」と R26f1「門を足したが
numpy 要素位置に通していない」は 1 つのクラスの 2 連続**であり、
**そのクラスに対して適用した修復の形は「もう 1 か所を通す」を 2 回**である。
`7ee10fe` のコミット題名（*the last position*）は**到達範囲の主張であって計測ではない。**

capsule の用語での横ばい: blocking 2 / 3 / 2、authorship 再帰は f1 を数えると
0/2 → 1/3 → **2/2**、production コミット 51 本のうち 42-51 の 10 本はすべて H-0095 の修正。
**CONVERGING の予測は 2 回出て 2 回とも closure に至っていない。**

**drift ではない**: D11 で固定した反証条件は発火せず、指摘は deliverable path に留まり、
`param_domain.py` は 3 つの修正すべてで変わっている（「clean と宣言され一度も変わらない」
escape の形ではない）。

## 監視が付記したこと（レビューではない）

`_is_unchanged` は `normalise_value` の dispatch と `_holds_a_mapping` に続く
**3 つ目の構造列挙**であり、**DC3 の形が再び存在する**。
監視は 32 形（intern されない新規スカラー、ndarray、tuple、入れ子 list、dict、
あらゆる位置の numpy）＋冪等性を probe し、**今日の時点では 3 者は一致している**。

## round 27 のための識別子

- **R27 が「既存の門（writability / encodability / `_is_unchanged`）が適用されていない
  位置」を名指したら → 3 例目**。位置ごとの修復はクラス単位ではない → **take-stop**。
- 新しい proxy、または型集合の穴 → 別クラス → continue。
- D11 の反証条件の文言は**そのまま継承**すること。

## 主文脈の処置

**take-stop を採用する。** D12 の選択肢は「安く直せる / 毎回別クラス」という前提で
提示したが、**その前提の片方は誤りだった** —— 2 件のうち 1 件は、修復の形が 2 回続けて
同じ位置を落としたクラスの 2 例目である。**誤った入力で下した判断なので、
訂正した入力で取り直す。**
