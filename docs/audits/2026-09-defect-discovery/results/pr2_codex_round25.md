# PR 2 — review round 25（2026-09-08、Codex 完走）

`gpt-6-astra`, effort medium。head `0c3eff8`（作業ツリー clean、read-only）。
依頼は `prompt-templates/pr2-review-round25.md`（契約検証として書き、
**契約そのものを拒否してよい**と明記した版）。

## VERDICT: REQUEST_CHANGES（3 件、すべて P2）

**3 件とも監視が事前に定義した「契約の内側」の形である** ——
「§2 の要件が受理母集団の上で成り立っていない」「§2/§3 が不完全」。
「§1/§3 の外側の値が検査を破る」形（＝書き直しが効いていない形）は **0 件**。

### 1. `contract` — 要件に **UTF-8 エンコード可能性**が抜けている

`_param_dict_to_str` の出力が学習器の bytes、json 化可能が export の要件、と書いたが、
**どちらの消費者ももう 1 段ある**: LightGBM の `_c_str` は UTF-8 に encode し、
`artifact_writer.py:88` は UTF-8 のファイルに書く。

**実測**: 孤立サロゲート `"\ud800"` は**正規化・出口の表明・出荷済みの `json.dumps`
オラクルをすべて通り**、その後 `_c_str` と `write_artifacts` の両方で
`UnicodeEncodeError` になる（レビュアーは filesystem を mock して実際に
`write_artifacts` を通して再現した）。

分類: **ある消費者に対して述べた要件が、その消費者には足りていない**。
`param_domain.py` の受理の穴（`deliverable-path`）と、
`test_param_domain.py:303` のオラクルの穴（`periphery`）を同時に露出させている。

### 2. `deliverable-path` — 述語が書ける値の境界で正規化と食い違う

`is_plain` / `is_accepted` は**厳密な Python `int` を無条件に受理**し、正規化が使う
writability 検査を持たない。実測（`10**5000`）:

| 経路 | 答え |
|---|---|
| `is_accepted` | `True` |
| `is_plain` | `True` |
| 出口の表明 | 通る |
| `normalise_params` | `CONFIG_INVALID` |

**§2 の「述語が正規化関数と一致すること」に対する反例。**
既存の一致テストは**正規化が受理した値だけを走査する**ので、この偽陽性を見られない。

### 3. `periphery` / `contract` — 母集団オラクルが宣言した型母集団を覆っていない

numpy の母集団ビルダは `test_param_domain.py:65` の**ベタ書き 12 型**を使っており、
この環境ではモジュールの導出型集合から
**`longdouble` / `longlong` / `str_` / `timedelta64` / `ulonglong` の 5 型が抜けている**
（`candidates 1014 / accepted 796`）。抜けた型のいくつかは受理される値を持ち、
**生成ブロック自身が `longdouble` の要素位置での受理を明示している**。

分類: **「オラクルは受理母集団全体の上で実行している」という主張が証拠を超えている。**
生成ブロックの `--check` は通るが、それは**その型集合を母集団オラクルに接続しない**。

加えて: 契約の最終段落が §1 の値領域を「**有限**」と書いているが、
**文字列・整数・コンテナの中身は無制限なので領域は無限**である。
有限の fixture はその**標本**であって領域そのものではない。

**処方**: fixture の型軸を**受理型集合から導出**し、両位置で probe し、
「有限標本の上での網羅」と「すべての受理値についての主張」を書き分けること。

## レビュアーが実行した検証

- 生成契約 `--check`: **pass**
- param_domain / value_equality / fit_params_override / refusal_matrix /
  isotonic calibration のテスト: **6053 passed, 256 skipped**
- fixture の 796 個の正規化値の**順序対 633,616 通りを全列挙**し、
  `values_differ` が raise せず `bool` を返すことを確認

**レビュアーが述べた bound**: フルスイート・lint・mypy・CI 13 レーンは再実行していない。
パラメーター値の網羅列挙・消費者の完全性の証明・繰り延べ 6 件の再監査はしていない。

## 停止条件（C）の判定: **発火した**

C は「**round N の修正が書いたコードの欠陥が round N+1 で出たら停止**」。

| 指摘 | 由来 | authorship 再帰か |
|---|---|---|
| 1. UTF-8 | `PLAIN_SCALAR_TYPES` は `48a0801`（H-0095 当初）、json オラクルは `8743c7c`（ラウンド間の書き直し） | ✗ |
| **2. 述語の不一致** | **`68d4972` = round 24 の修正が `_written_or_refused` を追加して正規化だけを狭め、`is_plain` / `is_accepted` を触っていない**（`git show 68d4972` で確認） | **✓ 発火** |
| 3. 母集団ビルダ | `4161837`（H-0095 のテスト実装） | ✗ |

**指摘 2 は round 24 の修正が作った乖離である。** 修正前は正規化も `10**5000` を
受理していたので、**述語と正規化が食い違う状態そのものが round 24 の修正の産物**である。

したがって **C により、ここで止めて判断を仰ぐ**。
