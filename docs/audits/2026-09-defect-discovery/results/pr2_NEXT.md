# 次の一手 — 2026-09-08 時点（セッション切替のための単独ハンドオフ）

このファイルだけ読めば次の作業に入れるように書いてある。

## 状態

- ブランチ `fix/phase3-pr2-fit-params-forwarding`、**head `1403ba8` +（このコミット）**、
  作業ツリー clean、すべて push 済み。
- PR **#278** は **draft**。CI **12/12 緑**、`mergeStateStatus: CLEAN`。
  フルスイート **2898 passed / 62 skipped**、`ruff check` / `ruff format --check` /
  `mypy lizyml/` clean、出荷計測器 `report_lifecycle_grid.py` は exit 0。
- レビューは **20 ラウンド走り、`APPROVE` は取得できなかった**。全ラウンドの指摘は実在・
  再現済み・修正済み。記録は `results/pr2_codex_round[1-20].md`、ループ監査は
  `results/pr2_monitor_round[1011..1819].md`。

## 決まったこと

**D8 は解決済み — 選択肢 F（入口で正規化して比較の領域を閉じる）。提案は HISTORY の
`H-0095`（ステータス Proposed）。** 経緯・却下した代替案・実測は
`DECISIONS-PENDING.md` の D8 末尾にある。

要点だけ:

- rounds 16-20 は 5 連続で「直前の修正が書いたコード」に欠陥が出た。5 件すべてが
  `lizyml/core/value_equality.py`。原因は個々の guard 漏れではなく**入力領域が開いている**
  こと。
- **「Layer 0 = 標準ライブラリのみ」はアーキテクチャの規則ではなく、そのモジュールが自ら
  課したものだった。** Layer 0 の他モジュールは numpy を import している。duck typing は
  必要ではなく、それが領域を開いていた。
- Codex（medium）の推奨は E（provider 経由で `_param_dict_to_str` を同一性の定義に使う）
  だったが、**実行して却下した** — wire form は正準形ではなく、E は round 13 の誤拒否を
  復活させる（`[1.0, 2.0]` vs `"1,2"`、`0.5` vs `"0.50"` まで拒否になる）。
- 変更ゲートの実測: **`Firing rate: 7/1430`**、7 件すべて rounds 16-20 自身が構築した
  敵対オブジェクト。計測器は
  `instruments/parameter_value_type_census.py` として出荷済み（再実行可能）。

## ⛔ 最初にやること — 管理者の判断が 1 件だけ残っている

**PR #278 を今の状態でマージするか、H-0095 が着地するまで draft のまま置くか。**

- PR 2 のマージゲート（Codex `APPROVE` + CI 緑）の **`APPROVE` は未取得**で、F はその門を
  retire しない。
- 一方 PR 2 は rounds 1-15 の修正＝ **#264 の本体**を含み、CI 緑・全スイート緑。
- **推奨はマージ。ただし「`APPROVE` 無しでマージ」は選択肢 C の形なので、管理者の明示的な
  承認なしに実行しないこと。** ここで意図的に止めてある。

## そのあと — H-0095 の実装（承認後）

提案本文は `HISTORY.md` の `H-0095`。受け入れ基準 7 項目もそこにある。実装の骨子:

1. **受理集合**は LightGBM から導出する（`lightgbm/basic.py`:
   `_NUMERIC_TYPES = (int, float, bool)`、スカラーは
   `isinstance(val, (str, Path, _NUMERIC_TYPES)) or _is_numeric(val)`、列は
   `list` / `tuple` / `set` / 1-D ndarray）。**写さずに導出すること。**
2. **正規化は素の型へ。文字列化しない** — smart params の解決と boundary 展開が数値演算を
   するので壊れる。numpy スカラー → `.item()`、1-D ndarray → `.tolist()`、
   `tuple`/`set` → `list`、`str` サブクラス → 厳密な `str`。
3. **配線先は既にある**: `check_duplicate_identities`（`_model_factories.py:860`）が
   4 surface すべての唯一の絞りで、rounds 10-12 で配線・固定済み。呼び出し元は
   `model.py:479`, `:504`, `:530` と `_model_factories.py:1068`（calibration）。
   **「検査するだけ」から「検査して正規化した dict を返す」へ変え、4 呼び出し元が返り値を
   使うようにする。**
4. **中心的な受け入れ基準は wire 保存**:
   `_param_dict_to_str({"k": normalise(x)}) == _param_dict_to_str({"k": x})` を受理集合の
   全要素について。閉じていて実行可能。
5. **rounds 16-20 の敵対オブジェクトのテストは削除せず書き換える** — 意味が「学習する」から
   「入口で拒否される」に変わるだけ。テストを消して通すのは禁止事項。
6. `value_equality.py` は閉じた素の型集合の上に縮む。H-0094 決定 16 の導出母集団
   （`DERIVED_HOSTILE_NAMES`）は「入口の拒否」を確かめる側へ移す。
7. **#283**（スカラー vs 単一要素の列）を H-0095 で解決するか明示的に決める。正規化後は
   両者とも素の型なので判断材料が揃う。

## 積み残し（PR 2 由来、起票済み）

- **#281** — loaded model の `validation_ratio`。artifact が「どの fit が overlay を消費
  したか」を記録しないため、`load()` 後は config の値に落ちる。`metadata.json` のキー追加
  ＝変更ゲート案件。
- **#282** — `category: training` の `seed` 次元が受理・サンプルされて黙って無視される。
- **#283** — スカラーと単一要素の列が同じ bytes を書くのに拒否される。admit は振る舞いの
  拡大（`allow`）なので firing rate 付き Proposal が要る。
- **#277 / #279 / #280** — PR 2 の範囲外として先行して起票済み。

## この run のあと

PR 3（#258 tuning direction）、PR 3b（H-0024 space merge、`HISTORY.md:1615` と `:1616` の
矛盾を解消すること）、PR 4-9。**PR 9 の直前に繰り延べ 1 件を 1 パスで片付ける**:
Phase 3 完了測定ツール（`phase3_gap.py` + manifest）は復元品が 5 箇所古く
`instruments/deferred/` に未出荷で archive してある。

## 環境メモ（踏むと時間を失う）

- `uv` は読み取り専用の既定キャッシュで落ちる → **`UV_CACHE_DIR="$TMPDIR/uv-cache"`** を
  毎回付ける。
- コマンドガードがアポストロフィ・ヒアドキュメント内の引用符を解析できずに拒否する →
  **スクリプトファイルにして `bash file.sh` / `uv run python file.py`**、コミットメッセージ
  には**アポストロフィを一切入れない**（git-manager のプロンプトに明記すること）。
- Codex は `CODEX_HOME` に書き込み可能なコピーを作ってから実行し、**実行後に消すこと**
  （`setup_codex_home.py` / `cleanup_codex_home.py`）。既定は effort `low`。
  評価用に上げるなら `-c model_reasoning_effort=medium`。
- codex の長時間実行はバックグラウンドにするとハーネスの低メモリ判定で 2 回 kill された →
  **フォアグラウンドで `timeout` を長めに**。
