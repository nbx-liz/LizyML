# PR 3c — 完了基準（レビューを開く前に書いた、2026-09-15）

計画 Revision 6 §12.5。提案は **H-0100**、設計は `results/pr3c_design.md`（改訂 3）。
**証拠のポインタが無い行が 1 つでもあれば、レビューを開かない。**

---

## 0. これは何で、何ではないか

- **#277 の決着**: `calibration.params` を 3 手法すべてで反映する（管理者決定: 拒否ではなく反映）。
- **Platt を原典の方法で推定する**（管理者決定: 本 PR で既定値を原典に寄せる）。
- **ラウンド予算 8。** 8 で一度止めて、受け入れ／範囲限定でもう 1 回／分割 を判断する。
- **分割点**（H-0100 で事前宣言）: 「platt の既定値変更・自前 MLE・旧 artifact 移行」は分けてコミットできる塊。予算に達したか、この塊だけに検証の問題が残ったら PR 3c-2 に切り出す。
- **レビュアーは受け入れを宣言しない。** 受け入れは管理者の宣言。

## 1. 凍結する head と契約

| 対象 | 凍結先 |
|---|---|
| production | 実装コミット `1181e05`（テスト `33339cb`） |
| 契約 | `HISTORY.md` **H-0100**、H-0030 / H-0031 / H-0047 / H-0058 / H-0059 / H-0090 / H-0093 / H-0094 決定 8 / H-0095 |
| 上位文書 | `BLUEPRINT.md` §12.2 / §15.4、`CLAUDE.md` §3（保存互換性） |

## 2. 受け入れ基準 → 証拠の対応表

| # | 基準（H-0100） | 証拠 |
|---|---|---|
| 1 | 既定の platt が scikit-learn の `_sigmoid_calibration` と許容誤差内で一致（参照はテストでのみ使う） | `tests/test_calibration/test_platt_mle.py::test_default_matches_platt_as_sklearn_implements_it`（4 ケース、うち 1 つは縮尺あり） |
| 2 | offset のずれで intercept が推定され、intercept を 0 に固定した fit より損失が小さい | `test_platt_mle.py::test_the_intercept_is_estimated_and_it_matters` |
| 3a | platt の params が観測可能な効果を持つ（単体） | `test_platt_mle.py::test_target_smoothing_can_be_turned_off_and_changes_the_fit` |
| 3b | platt / beta の params が facade 経由の fit で効く | `tests/test_core/test_calibration_params_reach.py::test_platt_params_change_the_fitted_calibrator`、`::test_beta_params_change_the_fitted_calibrator`、`tests/test_calibration/test_calibration_param_contract.py::test_beta_bounds_take_effect` |
| 3c | cross-fit の全 fold と C_final に届く | `test_calibration_params_reach.py::test_every_calibrator_built_is_given_the_params`（3 fold + C_final = 4 回、すべて同じ params） |
| 4a | `tol` が既定の options に負けない | `test_calibration_param_contract.py::test_a_written_tol_is_not_defeated_by_the_default_options` |
| 4b | 書いた `options` が書いた `tol` にキー単位で勝ち、既定にマージされる | `::test_written_options_beat_written_tol_key_by_key`、`::test_written_options_merge_over_the_defaults` |
| 4c | 既定の options は手法ごと（L-BFGS-B の `ftol` を BFGS に渡さない） | `::test_defaults_belong_to_their_method` |
| 4d | 表の 8 手法すべてで警告なしに fit し、既定と同じ解に達する | `::test_every_method_in_the_table_fits_platt` |
| 5a | 大きなスコアで `bounds` が書いた座標で効く | `test_platt_mle.py::test_bounds_on_the_slope_hold_in_the_written_coordinates_for_large_scores` |
| 5b | 縮尺しても同じ問題になる（`x0` と `bounds` の座標変換） | `test_platt_mle.py::test_rescaling_is_the_same_problem` |
| 6a | 受理範囲の外（未知名、`x0` / `bounds` の長さ、タプルの組、表外の手法、BFGS/CG + bounds、未知の option、型違い、beta への `target_smoothing`）が `CONFIG_INVALID`、出所 `calibration.params` | `test_calibration_param_contract.py::test_platt_refuses`（18 ケース）、`::test_beta_refuses`（6 ケース）。ケースは `PLATT_REFUSED` / `BETA_REFUSED` に置き、生成 fitter の検査（8f）と共有する。round 1 で `method` がリスト / dict の場合（`TypeError` で抜けていた）、round 2 で float に収まらない整数（`OverflowError` で抜けていた）を追加 |
| 6b | 受理範囲の中は通る | `::test_platt_accepts_its_surface`（6）、`::test_beta_accepts_its_surface`（5） |
| 6c | fit でも tune でも、Booster / study が学習される前に拒否 | `test_calibration_params_reach.py::test_fit_refuses_before_training`（6）、`::test_tune_refuses_before_any_study`（6、出所 `calibration.params` の文言も主張） |
| 6d | LightGBM の登録表ではなく calibrator の宣言で拒否される | `tests/test_calibration/test_calibration_param_names.py::test_calibrators_that_do_not_use_lightgbm_are_checked_by_their_own_contract`（旧挙動を固定していたテストを書き直した。削除していない） |
| 7 | platt / beta に LightGBM 正規名化が掛からず、isotonic には掛かる（H-0094 決定 8 の回帰なし） | `test_calibration_params_reach.py::test_platt_and_beta_are_not_given_lightgbm_canonicalisation`（呼び出しを spy で主張）、`::test_isotonic_is_given_lightgbm_canonicalisation`、既存 `tests/test_core/test_fit_params_override.py` の calibration 別名テスト群 |
| 8a | `config.json` が前処理後の実効値を持つ | `tests/test_codegen/test_calibration_params_codegen.py::test_config_json_carries_the_effective_calibration_params` |
| 8b | 生成 fitter が実行時と一致（platt 3 / beta 3） | `::test_generated_fitter_matches_the_runtime_calibrator`（6 ケース。round 2 で開始点が bounds 外になる Powell / Nelder-Mead を追加） |
| 8c | 生成 isotonic が params を反映し、実行時と同じ予測 | `::test_generated_isotonic_fitter_honours_its_params` |
| 8d | 一致だけでなく、生成 fitter が params で変わる | `::test_generated_platt_fitter_is_changed_by_its_params` |
| 8e | 生成コードで再学習が実際に走り、params が効く | `::test_generated_retrain_uses_the_params` |
| 8f | 生成 fitter が実行時と同じものを拒否する（`config.json` は編集可能、H-0059）。round 1 で追加 | `test_calibration_params_codegen.py::test_generated_fitter_refuses_what_the_runtime_refuses`（実行時の拒否ケースのうち JSON で表せる 23 ケースすべて）、`::test_generated_project_refuses_calibration_params_that_are_not_a_mapping`（6）、`::test_generated_project_reads_a_missing_calibration_params_as_empty` |
| 9a | 旧 platt calibrator が通常・極端なスコアで同じ predict を返す | `test_platt_mle.py::test_a_legacy_calibrator_predicts_as_it_did`（±1e3 まで、atol 1e-12） |
| 9b | 未学習の旧状態と、新状態の再読込 | `::test_an_unfitted_legacy_calibrator_stays_unfitted`、`::test_a_current_calibrator_survives_a_pickle_round_trip` |
| 10a | 既定の platt / beta の fit で警告なし | `test_platt_mle.py::test_default_fit_emits_no_warning`、`test_calibration_param_contract.py::test_beta_default_fit_emits_no_warning` |
| 10b | 出力形式と predict の式は不変 | `test_platt_mle.py::test_export_form_and_predict_agree`、既存 `tests/test_codegen/test_export_params.py` |
| 10c | 登録された calibrator すべてが受理契約を宣言し、生成 fitter を持つ（位置の導出） | `test_calibration_params_reach.py::test_every_registered_calibrator_declares_its_params_contract`、`::test_every_registered_calibrator_has_a_generated_fitter` |
| 10d | README と生成 requirements の scipy の記述が一致 | `test_calibration_params_codegen.py::test_requirements_list_scipy_when_the_generated_code_imports_it`（platt / beta）、`::test_readme_names_both_calibrators_that_need_scipy`、`tests/test_codegen/test_templates.py::TestRenderRequirementsTxt` |
| 11 | OOF-only・outer split 再利用・covered 行は不変 | 既存 `tests/test_calibration/test_calibration.py::TestCrossFitCalibrate`、`test_h0058_outer_reuse.py`、`test_cross_fit_nan_guard.py`（無変更で通ること） |
| 12 | 最低依存（scikit-learn 1.3 / scipy 1.10）で同じテストが通り、警告なし | **充足（2026-09-15）**。`instruments/lowest_deps_calibration_check.sh` 相当を Python 3.11 で実行し、解決結果は scikit-learn 1.3.0 / scipy 1.10.0 / numpy 1.24.0 / pandas 2.0.0 / lightgbm 4.0.0。対象 6 ファイルで **122 passed**（`-W error::DeprecationWarning -W error::FutureWarning`）。残る 36 件の警告は LizyML 自身の `UserWarning`（binary + kfold の助言）で、依存ライブラリ由来ではない。**例外 1 点**: 宣言下限の pydantic 2.0 では `import lizyml.config.schema` が `TypeError`（discriminator）で失敗するため、pydantic だけ lock の 2.12.5 に上げた。これは PR 3c と無関係の既存欠陥（DC7）で、#295 に起票済み。Python 3.10 は uv の Python ストアが読み取り専用で作れず未確認（下限の検査対象は Python の minor ではない） |

**12 行目が埋まるまでレビューを開かない。**

## 3. 明示的な処分

| 項目 | 処分 | 理由 |
|---|---|---|
| scikit-learn の全バージョンについての scipy 下限 | 確認しない | インストール済み 1.8.0 のメタデータのみ確認。最低依存の確認（行 12）で実際の組み合わせを動かす |
| scipy 1.10 での手法表と未知 option 警告の挙動 | 行 12 で確認 | 1.17.1 でのみ実測済み |
| H-0030 より前の artifact | 保証範囲外 | 係数の移行だけでは入力の意味が解決しない（H-0100 決定 5） |
| BLUEPRINT §12.2 / H-0047 の isotonic sigmoid 記述 | BLUEPRINT を訂正、H-0047 本文は記録として残す | 実装が正しく文書が古い（H-0100 決定 6） |

## 4. 指摘が出た場合（事前宣言）

| バケット | 条件 | 受け入れを妨げるか | 何が起きるか |
|---|---|---|---|
| **B1** | §2 の基準に反する | 妨げる | 修正 1 件 + その修正だけの限定検証 1 回。さらに出たら管理者に戻す |
| **B2** | 分割点の塊（既定値変更・MLE・移行）だけに残る検証の問題 | 妨げる | 管理者に分割（PR 3c-2）を提案 |
| **B3** | §2 の外の新しい production 欠陥。例外: params が学習前に拒否されずに捨てられる形（DC1 / DC4）なら妨げる | 原則妨げない | 起票。妨げる場合は B1 と同じ |
| **B4** | テスト / 文書 / 命名 | 妨げない | 直すか起票。再レビューしない |

どのバケットにも落ちない指摘は、この文書の不備として管理者に戻す。

## 5. レビューの記録

### round 1（2026-09-15、head `f67f8c7`、`REQUEST_CHANGES`）

記録: `results/pr3c_code_review_round1.md`。3 件とも主コンテキストで再現した。

| 指摘 | バケット | 処分 |
|---|---|---|
| 1. `method` がリスト / dict のとき `CONFIG_INVALID` でなく `TypeError` | **B1**（6a / 6c） | 修正: 表の照合の前に文字列であることを確認。拒否ケースを 6a と fit / tune の検査に追加 |
| 2. 生成 fitter が実行時の受理契約を持たず、編集した `config.json` の未知名などを黙って捨てる | **B3 の例外**（学習前に拒否されずに捨てられる形、DC1 / DC3） | 修正: 生成 platt / beta が実行時と同じものを拒否し、scipy の未知 option 警告を拒否にする。行 8f を追加 |
| 3. 最適化の失敗（`success=False`）が初期値のまま学習済みとして通る | **B3**（§2 に失敗時の扱いが無い、DC1） | **#297 に起票**。scikit-learn の `_sigmoid_calibration` も `success` を見ていない（1.8.0 で確認） |
| 証拠が主張より弱い行 | **B4** | 6a / 6c / 8b は上の修正のテストで補強。3c / 4a-4c / 5b / 8a / 9a-9b は **#298 に起票**（既知の production 欠陥ではない） |

### round 2（2026-09-15、head `761c05e`、round 1 の修正に限定、`REQUEST_CHANGES`）

記録: `results/pr3c_code_review_round2.md`。運び手は Codex からサブエージェントに変更（Codex がメモリ監視で 3 回連続停止）。

| 指摘 | 性質 | 状態 |
|---|---|---|
| A. 生成 `_run_minimize` がすべての `OptimizeWarning` を拒否にし、Powell / Nelder-Mead + bounds の正当な設定で生成 `train.py` が止まる | **round 1 の修正が書いたコードの欠陥**（回帰） | 修正: `Unknown solver options` の警告だけを拒否にし、他は再送出（文言は scipy 1.10.0 / 1.17.1 で実測一致）。8b に Powell / Nelder-Mead + 範囲外の開始点の 2 ケースを追加 |
| B. 生成 `fit_calibrator` の `or {}` が偽値の非 dict を既定値で黙って走らせる | round 1 の差分の外（本 PR 由来） | 修正: `_calibration_params` が非 mapping を拒否、キー無し（H-0100 以前の export）は `{}`。テスト 7 件 |
| C. 巨大整数の `tol` / `x0` / `bounds` が `OverflowError` で抜ける | round 1 の指摘 1 と同じ形の残り | 修正: 実行時と生成の数値判定が float に収まらない整数を拒否。6a（共有ケース 3 件、生成側にも流れる）と 6c に追加 |

**§4 B1 の「限定検証 1 回。さらに出たら管理者に戻す」に該当 → 管理者に戻した。管理者の決定（2026-09-15）: 3 件を修正し、修正 3 件だけを対象に round 3。**

検証: 関連 423 passed、フルスイート 7999 passed / 0 failed（環境由来のバージョン検査 1 件を除外）、ruff / format / mypy clean。**最低依存（scikit-learn 1.3.0 / scipy 1.10.0）で修正後のコードを再実行: 168 passed**（行 12 の実行は修正前だったため）。

### round 3（2026-09-15、head `5fd59f1`、round 2 の修正に限定、`APPROVE`）

記録: `results/pr3c_code_review_round3.md`。監視（relational、rounds 1-2）は `CONVERGING` / `continue`（`results/pr3c_code_monitor_round2.md`）。**事前宣言した停止条件（round 2 の修正が書いたコードの欠陥）は発火しなかった。**

- 修正 A / B / C はそれぞれ指摘を閉じ、壊したものは無い。A は scipy 1.17.1 と 1.10.0 の両方で実行時と生成の 31 ケースが一致。
- note 1 件（未知 option を拒否するとき同じ呼び出しの他の警告を再送出しない。結果は変わらない）と範囲外 3 件（生成 platt の bounds×scale の inf 化、`np.int64` の拒否、options probe が bounds なし）は **B4 / 観察として記録のみ**。いずれも fit か拒否かの結果を変えず、本 PR の修正で入ったものではない。
- ラウンド使用: 予算 8 のうち 3。**受け入れは管理者の宣言（§0）。**
