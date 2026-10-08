# PR 2 — 原因 A / B は既知の問題である。開発現場の解答を調べた（2026-09-09）

ユーザー指示で、`results/pr2_why_no_approve.md` が特定した 2 原因について
**現場で同じ問題がどう解かれているか**を調べた。**主張はすべて実行して確かめた**か、
出典を挙げる。DC7 の規律に従い、「宣言された外部の振る舞い」は引用ではなく実行で確かめた。

---

## 原因 A の先行事例 —— 「1 つのパラメーターが 2 綴りで来た」ときの解答

### まず LightGBM 自身がどう答えるか（実行、`.venv` の lightgbm 4.x）

**前回の計測は誤りだった。** `verbose: -1` を渡していたので C++ のログが抑止されており、
「LightGBM は黙っている」と読んでしまった。**verbosity を既定に戻して fd レベルで
捕捉すると、LightGBM は全ケースで警告する。**

```
canonical + alias, different   -> learning_rate=0.5
  [LightGBM] [Warning] learning_rate is set=0.5, eta=0.9 will be ignored. Current value: learning_rate=0.5
canonical + alias, EQUAL       -> learning_rate=0.5
  [LightGBM] [Warning] learning_rate is set=0.5, eta=0.5 will be ignored. Current value: learning_rate=0.5
two aliases, different         -> learning_rate=0.2
  [LightGBM] [Warning] learning_rate is set with eta=0.2, shrinkage_rate=0.3 will be ignored. ...
two aliases, REVERSED order    -> learning_rate=0.2
  [LightGBM] [Warning] learning_rate is set with shrinkage_rate=0.3, will be overridden by eta=0.2. ...
single spelling (control)      -> 警告なし
```

**3 つの事実が出た。**

1. **LightGBM は値を比較しない。** 等しくても違っても、**重複そのもの**を警告する。
2. **優先順位は決定的で、dict の順序に依存しない。** `eta` と `shrinkage_rate` は
   どちらの順で書いても `eta` が勝つ。正規名は常に別名に勝つ
   （`basic.py:650` `_choose_param_value` と C++ Config の両方）。
3. したがって **H-0094 round 4 の拒否理由**「どの値が効くかは*書いたもの*ではなく
   *ライブラリ*で決まる」は**半分しか正しくない**。ライブラリで決まるのは事実だが、
   **決定的で、しかも LightGBM は自分でそう言う**。「不可視」ではない。

### 他の処理系はどうか（実行）

| 処理系 | 同一キーが 2 度来たとき | 値を比較するか |
|---|---|---|
| **Python の呼び出し** `f(**{a:1}, **{a:1})` | **`TypeError: got multiple values for keyword argument`** | **しない**（等しくても落ちる） |
| Python の dict リテラル / `json.loads` | 後勝ち、無言 | しない |
| **pydantic**（本プロジェクトの検証ライブラリ）alias + `populate_by_name` | **alias が勝つ、無言**（値が違っても同じ） | **しない** |
| LightGBM | 正規名 > 別名の固定順、**警告あり** | **しない** |
| PostgreSQL `guc.c` | 後勝ち（`ALTER SYSTEM` はこの方針に依存） | しない |
| Go `gopkg.in/yaml.v3` / Ruby Psych | **エラー** | しない |
| PyYAML / js-yaml | 後勝ち、無言 | しない |
| **C プリプロセッサ** `#define` の再定義 | **トークン列が同一なら黙認、違えば診断** | **する（ただし構文的に）** |

**調べた範囲で、意味的な値の等価で分岐するのは C プリプロセッサだけ**であり、
その C ですら**値ではなくトークン列の同一性**で判定する ——
「識別子リスト・トークン列・空白の出現位置が同一なら同じ定義」。
委員会がこれを許した理由は「**独立したヘッダがそれぞれの理解を書けるように**」であり、
**診断は定義が食い違うときだけ**出る（[C Rationale 3.8](https://www.lysator.liu.se/c/rat/c8.html) /
[GCC cpp](https://gcc.gnu.org/onlinedocs/gcc-3.0.1/cpp_3.html)）。

**これが本 PR との差である。** C は**既にトークン化された有限の表現**の上で同一性を見る。
LizyML の `values_differ` は**任意の Python オブジェクト**の上で意味的な等価を見ようとした。
同じ「等しければ黙認」でも、**判定の対象が開いているか閉じているか**が違う。

### 「sink に委譲すれば領域は閉じる」は成り立たない（実行して否定）

「型の判定を LightGBM のシリアライザに任せれば領域は構成上閉じる」という案は**成立しない**。
`lightgbm/basic.py:543` `_param_dict_to_str` をソースから読むと、受理集合はこうなっている:

```python
if isinstance(val, (list, tuple, set)) or _is_numpy_1d_array(val):   # 列
elif isinstance(val, (str, Path, _NUMERIC_TYPES)) or _is_numeric(val):  # スカラー
elif val is not None: raise TypeError(...)                            # 拒否
                                                                       # None は無言で脱落
```

`_NUMERIC_TYPES = (int, float, bool)`、そして **`_is_numeric(obj)` は
`try: float(obj)` である**（`basic.py:316`）。つまり **`__float__` を定義した
任意の呼び出し元クラスが通る** —— rounds 21-23 が「呼び出し元が numpy を自称できた」
として見つけた穴と**同じ形が sink 自身にある**。委譲は領域を閉じない。

実測した sink の誤受理（本 PR が既に個別に拒否しているもの）:

```
set          -> 'p=1,2'          （hash 順。列は位置依存なので不定）
nested list  -> 'p=[1,2],[3,4]'  （LightGBM が読む深さより深くても書けてしまう）
surrogate str-> 'p=\ud800'       （このあと UTF-8 encode で落ちる = round 25 指摘 1）
None         -> 無言で脱落
dict/object/2-D ndarray -> TypeError（ここは正しく拒否）
```

**したがって委譲は「型の質問」を安く解くが、`set` の順序・入れ子の深さ・UTF-8・
`__float__` の 4 点は依然として自前の有限な拒否リストが要る。** 委譲＋4 点、
という**閉じた文法**にはできるが、「構成上閉じる」とは言えない。

### 「wire を比較する」案の位置づけ

`wire(a) != wire(b)`（C のトークン同一性に相当）を等価判定に使うと、
rounds 6-26 が 30 commit かけた numpy / ndarray / tuple / 列とカンマ文字列は**すべて無料で解ける**（実測）:

```
numpy スカラー vs python   p=1        p=1        同じ
ndarray vs list           p=1.0,2.0  p=1.0,2.0  同じ
list vs tuple             p=1.0,2.0  p=1.0,2.0  同じ
list vs カンマ文字列       p=1.0,2.0  p=1.0,2.0  同じ
round 5 の int vs float   p=1        p=1.0      違う  ← 拒否に戻る
bool vs int               p=True     p=1        違う  ← 拒否に戻る
```

**round 5 の緊張は解消しない。境界が意味的から構文的に移るだけ**である。
C も `#define X 1` と `#define X 1.0` を診断するが、**規則が 1 文で言えるので誰も欠陥と呼ばない**。
採るなら「**設計上の誤拒否であり、文書化する**」という形になる。
なお上記のとおり wire 比較も `_is_numeric` の穴を継承するので、型の門は別途要る。

---

## 原因 B の先行事例 —— 「全称命題を問うレビュー」の終わらせ方

現場の解答は 3 つあり、**どれも「安全性を証明する」ことをやめている**。

### B-1. 範囲を宣言して、その中で網羅する（bounded verification / small scope hypothesis）

Alloy・有界モデル検査の標準的な受入形は「**scope k の内側に反例が無い**」であって
「反例が無い」ではない。小スコープ仮説は「**大半の欠陥は小さなスコープの網羅で出る**」
という経験則で、**受入基準にスコープを書き込む**ことで検査が終わるようにしている
（[Evaluating the Small Scope Hypothesis](http://users.csc.calpoly.edu/~gfisher/work/specl/documentation/related-work/testing/evaluating_small_scope_hypoth.pdf) /
[Bounded Exhaustive Search of Alloy Specification Repairs](https://arxiv.org/pdf/2103.00327)）。
**外部評価が推した形はこれである。**

### B-2. 検証をやめて型にする（parse, don't validate / illegal states unrepresentable）

境界で 1 度だけ parse して、以降は型が保証する。網羅性は**テストではなくコンパイラ**
（`match` の網羅検査 / `assert_never`）が見る。**全称命題が有限の分岐列挙に変わる**
（[Parse, don't validate](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/) /
[Make Illegal States Unrepresentable](https://deviq.com/principles/make-illegal-states-unrepresentable/)）。
**本 PR の構造走査が 3 つある問題（DC3）は、これが 1 つの dispatch になっていないことの帰結**である。

### B-3. 規模と時間で切る（review size / timebox）

レビューの実務では **1 回 400 行以下**（理想 300 行以下）、**60 分以内**が広く使われる閾値で、
それを超えると欠陥発見能力が落ちるとされる。大きい変更は**レビュー前に分割する**のが定石
（[Code Review Best Practices](https://codeant.ai/blogs/code-review-process-guide) /
[A Roadmap on Modern Code Review](https://arxiv.org/pdf/2405.18216)）。

**本 PR の production 差分は 1,916 行 = 閾値の約 5 倍**、テストは 5,571 行である。
**「26 ラウンドで終わらない」は、この規模の変更に対する正常な反応**でもある。

---

## 上位文書との関係

**⚠️ 初版のこの節は誤りだった（2026-09-09 訂正）。** §5.3 だけを見て
「BLUEPRINT は別名重複に触れていない / 改訂は不要」と書いたが、
**`BLUEPRINT.md` §14.4（1291 行目）が明記している**:

> 同じ層で 1 パラメーターが複数綴り・異なる値で指定されたら `CONFIG_INVALID` と
> すること（**同値は通す**）

**許容規則は上位文書に載っている。** したがって改訂には **BLUEPRINT の更新が必要**で
あり、H-0096 がそれを行う。§5.3 が固定しているのはスマートパラメーターと `params` の
競合だけ（round 1 が先例として引いたのはそちら）、というほうは正しい。

この訂正は、**上位文書は該当節だけでなく全体を検索して確かめる**という手続きの教訓でもある。

## この調査が言えないこと

先行事例は「どう決めたか」の分布であって、**LizyML にとっての正解ではない**。
特に「後勝ち・無言」を採る系（pydantic / PyYAML / PostgreSQL）が多数派だが、
**#264 はまさに「黙って上書き」が欠陥として報告された issue** なので、
無言の系はこのプロジェクトの入力にならない。**採れるのは「拒否」か「警告して決定的に選ぶ」**である。
