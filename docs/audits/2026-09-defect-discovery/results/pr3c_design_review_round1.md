# PR 3c 設計レビュー round 1（2026-09-15）

依頼: `prompt-templates/pr3c-design-review-round1.md`。対象: `results/pr3c_design.md`（初版）。
Codex `gpt-6-astra` effort medium、read-only、branch `fix/phase3-pr3c-calibration-params` @ `5ac725e`。
管理者の指示: **外部レビューで元の設計が遵守されていることをダブルチェックする。**

## VERDICT: **ADHERES-WITH-CHANGES**

> 手法ごとのパラメーター上書きを反映することは `b465240` と管理者の決定に沿う。
> isotonic の生成コード再現は**必要なスコープ**である。ただし実装前に、method を区別した
> 正規名化、消費者までの配線の明示、受け入れ基準の強化が要る。

## 指摘（重い順）と主コンテキストの確認

| # | 重さ | 指摘 | 主コンテキストの確認 | 処分 |
|---|---|---|---|---|
| 1 | **blocking** | 値の正規化と LightGBM の名前正規名化を分けよ。現状 `canonicalise_calibration_params` は全 method に無条件で適用され、platt の `random_state` が `seed` に書き換わる | **実コードで確認**（`lizyml/core/model.py:833`、method 分岐の前で呼ばれる）。 | **採用** —— 値の正規化は 3 手法、LightGBM 正規名化は isotonic のみ、platt/beta の名前はそのまま |
| 2 | should-change | 生成コードへの経路を全部名指せ（`_model_persistence.export_code` → `generator.generate_code` → `config_writer.build_config` → `config.json` → 生成 `fit_calibrator` → `_CAL_FITTERS[method]`）。isotonic は正規名化済みの上書きと実効 seed、少数行（< 20）で early stopping を切る振る舞いまで再現せよ | 経路は既読のコードと一致 | **採用** |
| 3 | should-change | 「fit 前」を「両入口で学習前」と明示せよ。`fit()` と `tune()` の両方でマージ直後に検査し、calibrator 構築や `_run_calibration` に移さない。入口と順序のテストを platt/beta へ広げよ | 呼び出しは `model.py:235` と `_model_tuning.py:236` で既存 | **採用** |
| 4 | should-change | 実行時と生成コードの一致だけでは効き目を示さない（両方が params を無視していても一致する）。既定と異なる値の観測可能な効果、cross-fit 全 calibrator と C_final への到達、同じ生スコア/ラベルでの生成 fitter の一致、生成コードでの再学習の実行を求めよ。既存の subprocess 一致テストは export 済みの学習済み artifact を使うので `_fit_*` への転送は示せない | 妥当。受け入れ基準に入れる | **採用** |
| 5 | should-change | 元から受け継ぐ要件と新しい方針を分けて記録せよ。元資料が定めるのは「手法ごとの上書き」までで、LogisticRegression の全引数の開放や platt の強制値ではない。H-0100 の決定として書き、BLUEPRINT §12.2 の暫定段落（「受理して無視・#277 未決」）を置き換え、LightGBM 固有の名前検査だけが isotonic 限定であることを明記せよ。`export_params` の `"method"` キーを保つ。beta で除外する引数の理由は「管理者が承認した範囲」で足りる（すべてが callable ではない） | 妥当 | **採用** |

## 未決点への回答

1. **platt の強制値**: `fit_intercept=True` の強制を推奨する（ただし**新しい方針**として明示する）。`False` は `b=0` に制限された sigmoid になるが、元資料はそれを明示的には禁じていない。solver/penalty を丸ごと固定せず、出力の sigmoid 形を保つ組み合わせを受理する。`warm_start` / `n_jobs` / `verbose` / `class_weight` に元資料上の強制根拠はない。
   **beta**: 尤度・生スコアから確率への変換・3 係数の形は固定。`x0` / `bounds` は 3 次元であることを検証する。係数の符号制約を足したり、承認済みの上書きを削ったりしない。
2. **`bounds` / `options` と H-0095**: `bounds` は **2 要素リストのリスト（長さ 3）**なら受理集合に入る。**タプルのリストは同じ形ではない**（列の member に許されるのは list で、tuple ではない）。`options` は str キーの dict で値が正規化できれば入る。この区別を文書化し、受理集合を黙って広げない。
3. **名前検査を facade に置くこと**: 妥当（BLUEPRINT §2.1、H-0093 / H-0094）。calibrator は `lizyml/estimators/` を import せずに自分の受理名を宣言できる。

**isotonic の生成コード再現はスコープの膨張ではなく必須**（H-0059 が「同一設定での calibrator 再構築」を約束している）。

## 既存の食い違い（本 PR 以前から）

BLUEPRINT §12.2 と H-0047 は「isotonic の Booster predict は raw score を返すので sigmoid を適用する」と書くが、実装は適用していない。**主コンテキストが確認**: `lizyml/calibration/isotonic.py:186-188` は「`objective="binary"` の `Booster.predict()` は確率を返す」とコメントし、sigmoid を掛けない。LightGBM の挙動としては実装が正しく、文書が古い。**コードを上位として黙って扱わず、H-0100 で記録して BLUEPRINT を訂正する**（doc-hierarchy の運用規則）。

## レビュアーが述べた bound

設計、`b465240` の該当 diff、指定した BLUEPRINT / HISTORY の節、層規約、calibration skill、指定したコード経路、関連する persistence / tuning / テストの抜粋を読んだ。`AGENTS.md` は壊れたシンボリックリンクで読めなかった。無関係なコミット、リポジトリ全体は読んでいない。変更・ネットワーク・テスト実行・数値実験・依存バージョンの互換性確認はしていない。
