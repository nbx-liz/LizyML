# PR 3c 設計案 —— `calibration.params` を 3 手法すべてで反映する（#277）

Status: **改訂 3（round 1 / round 2 設計レビュー、Platt intercept 調査、管理者決定を反映）**。実装していない。
Base: `develop` = `5ac725e`。

## 0. 管理者の決定（2026-09-15）

1. **反映する**（#277 の選択肢 2）。**元の機能設計を重視する。**
2. beta の上書き範囲は `x0 / method / bounds / tol / options`。
3. **platt の既定を原典の Platt（正則化なし・目標値平滑化）に寄せる。本 PR で行う。**
4. **platt は自前の Platt MLE で実装する**（`scipy.optimize.minimize`、beta と同じ形）。
   上書き面は `x0 / method / bounds / tol / options` に**目標値平滑化の on/off** を足す。
5. `liblinear` は拒否する —— 決定 4 で LogisticRegression を使わなくなるため問題そのものが消える。

記録: `results/pr3c_design_review_round1.md`（5 件採用）、`results/pr3c_platt_intercept_research.md`、
`results/pr3c_monitor_round1.md`（redirect を不採用、分割点を宣言）、`results/pr3c_design_review_round2.md`（6 件採用）。

**分割点（事前宣言）**: 「platt の既定値変更・自前 MLE・旧 artifact 移行」は、
「3 手法への params の反映・前処理の分離・入口検査・生成コードの再現」から**分けてコミットできる塊**として扱う。
ラウンド予算 8 に達したとき、またはこの塊だけに検証の問題が残ったときは **PR 3c-2 に切り出す**。

---

## 1. 現状（実測・読解済み）

| 位置 | `calibration.params` の扱い |
|---|---|
| 実行時 `PlattCalibrator` (`lizyml/calibration/platt.py`) | 受け取るが**保持も参照もしない**。`LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)` を直書き（**L2 正則化あり・目標値 0/1** —— 原典 Platt から逸脱） |
| 実行時 `BetaCalibrator` (`lizyml/calibration/beta.py`) | 同上。`minimize(x0=[1,1,0], method="L-BFGS-B")` を直書き |
| 実行時 `IsotonicCalibrator` | **反映済み**（H-0047、H-0093、H-0094 決定 8） |
| facade の前処理 (`lizyml/core/model.py:833`) | **`canonicalise_calibration_params` を method を問わず適用** |
| 生成コード `_fit_platt` / `_fit_beta` / `_fit_isotonic` (`lizyml/codegen/templates.py`) | **3 つとも既定値を直書き**。`config.json` に calibration params が無い。`_fit_isotonic` に `"verbose": -1` が残る |
| 生成 `requirements.txt` と README | 基本は `lightgbm / numpy / pandas / scikit-learn`。**scipy は beta のときだけ**（`templates.render_requirements_txt` の注記、`README.md` の "plus `scipy` when the model uses beta calibration"） |
| fit 前の名前検査 `check_calibration_param_names` | `LGBM_BACKED_CALIBRATORS = {"isotonic"}` のみ。呼び出しは `model.py:235`（fit）と `_model_tuning.py:236`（tune） |
| 永続化 (`lizyml/persistence/exporter.py`) | `fit_result.pkl` に calibrator オブジェクトごと joblib で pickle。`FORMAT_VERSION = 2`。`loader._migrate_fit_result` は format_version で分岐 |
| 依存 | `scikit-learn>=1.3` が必須、`calibration` extra が `scipy>=1.10`。**インストール済みの sklearn 1.8.0 のメタデータは `scipy>=1.10.0` を必須にしている**（sklearn の全バージョンについての下限は確認していない） |
| lock と CI | `uv.lock` は Python `< 3.11` に sklearn 1.7.2 / scipy 1.15.3、`>= 3.11` に 1.8.0 / 1.17.1。CI の lowest-direct レーンは `uv sync --frozen --dev --resolution lowest-direct` で、**`--frozen` のため sklearn 1.3 / scipy 1.10 で動いた証拠にはならない** |

## 2. 元の設計（根拠）と、この PR が足す方針

### 2.1 元資料から受け継ぐもの

1. **`b465240`（Phase 23-G）**: `CalibrationConfig.params` は「method-specific overrides」、`get_calibrator(name, params=None)`。
2. **BLUEPRINT §12.2 / H-0047**: Isotonic の型（既定値の表 → 上書き → 強制 → fit 前検査）。
3. **BLUEPRINT §12.2 が名指す「Platt Scaling」の定義**（Platt 1999）: `P(y=1|f) = 1/(1 + exp(A·f + B))`、
   A と B を同時に最尤推定、目標値は平滑化、正則化項なし。**intercept（B）はモデルの定義に含まれる。**
4. **H-0031**: Beta は 3 係数モデルを `scipy.optimize.minimize` で最適化、共通 IF。
5. **H-0059 / BLUEPRINT §15.4**: 生成コードで「同一設定のまま calibrator 再構築」。
6. **BLUEPRINT §18.1.2**: 「Calibration 実効性: `calibration.params` → calibrator パラメータ到達」。
7. **変えない上位規則**: H-0030、H-0058、calibration skill、`export_params` の `"method"` と `{a, b}` / `{a, b, c}`、層規約、`CLAUDE.md` §3 の保存互換性。

### 2.2 この PR が新しく決める方針（H-0100 の決定として記録）

- **platt の既定値を原典の Platt に変える**（Phase 13 `2ac4331` の `LogisticRegression(C=1.0)` からの意図的な変更）。
  **利益を過大に書かない**: 研究ノートの実測では n=2000 で差なし、n=100 で ECE 0.114 → 0.107 程度。変更の主な根拠は「BLUEPRINT が名指す手法の定義に合わせる」ことである。
- **platt の実装を自前の Platt MLE に置き換える**。
- **上書き面**: platt は `x0` / `method` / `bounds` / `tol` / `options` / `target_smoothing`、beta は `x0` / `method` / `bounds` / `tol` / `options`（管理者承認の範囲）。
- **最適化の手法ごとの契約**（§3.2）と、**旧 artifact の platt calibrator の移行**（§3.7）。

## 3. 設計

### 3.1 手法ごとのモデル・既定値・上書き

| 手法 | モデル（固定） | 既定値 | 上書きできる名前 |
|---|---|---|---|
| **platt** | `p = sigmoid(a·s + b)`（Platt の `A = −a`, `B = −b`）。**a・b とも常に推定**（intercept は外せない） | `target_smoothing=True`、`method="L-BFGS-B"`、`x0` は Platt / sklearn の初期値（`a=0`, `b=−log((N− + 1)/(N+ + 1))`）、L-BFGS-B の `options` は sklearn `_sigmoid_calibration` と同じ `gtol=1e-6`, `ftol=64·eps` | `x0`, `method`, `bounds`, `tol`, `options`, `target_smoothing` |
| **beta** | `sigmoid(a·log s + b·log(1 − s) + c)`（H-0031） | `x0=[1.0, 1.0, 0.0]`、`method="L-BFGS-B"`、`options` は scipy の既定 | `x0`, `method`, `bounds`, `tol`, `options` |
| **isotonic** | 現行どおり | 現行どおり | 現行どおり |

- **座標系（round 2 指摘 3）**: 利用者が書く `x0` と `bounds` は**`export_params` と同じ係数**で表す —— platt は `(a, b)`、beta は `(a, b, c)`。`calibrator.json` に出る値と同じ座標なので、利用者が読む値と書く値が一致する。
- **出力形式は変えない**: `export_params` は `{"method": "platt", "a", "b"}` / `{"method": "beta", "a", "b", "c"}`。`predict` の式も同じ。
- **`bounds` の形**: 長さ（platt 2 / beta 3）の、2 要素**リスト**のリスト。H-0095 の受理集合で列の member に許されるのは list であり、**タプルのリストは拒否**。受理集合を広げない。

### 3.2 最適化の手法ごとの契約（round 2 指摘 1・2）

**受理する `method` は閉じた表で持つ**（scipy のバージョンによって変わらないよう、scipy 1.10 で使える手法に限る）。

| `method` | `bounds` | 勾配 |
|---|---|---|
| `L-BFGS-B`（既定） / `TNC` / `SLSQP` / `trust-constr` | 使える | 解析的な勾配を渡す |
| `Powell` / `Nelder-Mead` | 使える | 使わない |
| `BFGS` / `CG` | **使えない** → `bounds` と一緒に書くと**学習前に拒否** | 解析的な勾配を渡す |

- ヘッセ行列を要求する手法（`Newton-CG` など）は表に入れない（受理しない）。
- **実測（scipy 1.17.1、`/tmp/claude-1000/method_table_check.py`）**: 表の 8 手法すべてが 2 係数のロジスティック問題で
  収束し、同じ解に達した（`trust-constr` はヘッセ行列なしで動作）。`BFGS` に `bounds` を渡すと
  `RuntimeWarning: Method BFGS cannot handle bounds.` が出て**bounds は無視される**（学習前に拒否する根拠）。
  未知の option は `OptimizeWarning: Unknown solver options: ...` になる（入口検査で拒否に変える根拠）。
  **scipy 1.10 での同じ挙動は未確認**で、§7 の最低依存の確認で検証する。
- **`tol` と `options` の優先順位**: **利用者の `options` のキー ＞ 利用者の `tol` ＞ LizyML の既定**。
  実装は「LizyML の既定の `options` を、利用者の `tol` が対応するキー（scipy が `tol` から設定するキー）については入れない」形にする。
  こうすると scipy の `setdefault` によって `tol` が効き、利用者が `options` に同じキーを書けばそれが勝つ。
  `options` は LizyML の既定にキー単位で上書きする（丸ごと置き換えない）。
- **`options` のキーの妥当性は、学習前に実物の scipy で確かめる**: 入口の検査で、受理した `method` と `options` を使って
  **小さな既知の問題に `scipy.optimize.minimize` を 1 回かけ、scipy が出す「未知の option」警告を拒否に変える**。
  名前の表を写すのではなく実物を実行するので、scipy のバージョン差にも追随する（DC7 の再発防止の形）。

### 3.3 大きなスコアの縮尺（round 2 指摘 3）

sklearn と同じく、**`max(|s|) ≥ 30` のときは全スコアを `k = max(|s|)` で割って**最適化する。
このとき**slope に当たる係数を座標変換**する:

- platt: 縮尺後の問題の slope は `a′ = k·a`。**利用者の `x0` の slope と、slope の `bounds` を `k` 倍**してから最適化し、
  結果の slope を `k` で割って戻す。intercept `b` は変換しない。
- beta: `log s` と `log(1 − s)` はスコアを確率にしてから取るので、スコアの縮尺の影響を受けない。**beta では縮尺しない**。
- 「結果が変わらない」は**数学的に同じ問題になる**という意味であって、浮動小数点での完全一致ではない。

### 3.4 前処理を method で分ける（round 1 指摘 1）

1. **値の正規化**（H-0095 `normalise_params`, `surface="calibration.params"`）—— 3 手法すべて。
2. **LightGBM の名前正規名化**（`canonicalise_calibration_params`）—— **isotonic のみ**。platt / beta の名前は書き換えない。

### 3.5 名前・形・組み合わせの検査（round 1 指摘 3）

- 各 calibrator module が**受理名・形・手法の契約**を宣言する（`lizyml/estimators/` を import しない）。
- facade の `check_calibration_param_names` が method ごとの宣言で検査し、違反は `CONFIG_INVALID`（出所 `calibration.params`）。**LightGBM 固有の名前検査は isotonic 限定のまま**。
- **位置は変えない**: `fit()`（`model.py:235`）と `tune()`（`_model_tuning.py:236`）で、マージ直後・**Booster も study も学習する前**。

### 3.6 生成コードで再現する（round 1 指摘 2）

経路: `_model_persistence.export_code` → `codegen/generator.generate_code` → `codegen/config_writer.build_config` → `config.json` の `calibration_params` → 生成 `train.py::fit_calibrator` → `_CAL_FITTERS[method]`。

- 書き出す値は §3.4 の前処理を通した実効値（isotonic は正規名化済みの上書きと実効 seed）。
- 生成 `_fit_platt` / `_fit_beta` / `_fit_isotonic` が実行時と**同じモデル・既定値・上書き・手法の契約・縮尺**を使う。
- `_fit_isotonic` は `"verbosity": -1` に揃え、自前キーの上書きと 20 行未満で early stopping を切る振る舞い（H-0047）まで再現する。
- **scipy の明示**: calibration が platt または beta のとき、生成 `requirements.txt` に scipy を載せる。
  **`render_requirements_txt` の注記と `README.md` の依存の記述を同時に直す**（round 2 指摘 5）。

### 3.7 旧 artifact の互換（round 2 指摘 6）

- `fit_result.pkl` の旧 `PlattCalibrator` は `_model: LogisticRegression` を持つ。
- **`PlattCalibrator.__setstate__` で旧状態を変換する**: `_model.coef_[0, 0]` → `a`、`_model.intercept_[0]` → `b`。
  pickle は当該クラスを復元するときに必ずこれを呼ぶので、calibrator がどの入れ物にあっても働く。
- **旧 `_model=None`（未学習）**は新しい未学習状態に変換する。**新しい状態の再読込**はそのまま通す。
- **predict の数値**: 旧 predict は `LogisticRegression.predict_proba`（sklearn の expit）で計算していた。
  新 predict は**数値的に安定な sigmoid（`scipy.special.expit`）**で計算し、**通常のスコアと極端なスコアで旧 predict と比較する**。
  一致は浮動小数点の許容誤差の範囲で主張する（完全一致は主張しない）。
- **`FORMAT_VERSION` を 2 のままにする根拠**: 公開の Artifact 契約（ディレクトリ構成・metadata・`Model.load()` の振る舞い）が変わらず、
  旧モデルの推論が保たれる**内部状態の変換**だからである（`CLAUDE.md` §3 の「破壊的変更」に当たらない）。
- **保証範囲の外**: H-0030 より前（確率を入力にしていた時期）の artifact。係数の移行だけでは入力の意味が解決しない。

## 4. 規則が縛る位置（Revision 6 §12.4）

規則: **「`calibration.params` は、それを消費する calibrator に届くか、学習開始前に拒否される」。**

- calibrator 3（`CalibratorRegistry`）× 消費者 2（実行時の cross-fit 各 fold と C_final / 生成コード `_CAL_FITTERS`）= **6**、入口 2（fit / tune）、移行 1。
- 登録表・`_CAL_FITTERS`・入口の呼び出し点から導出し、**登録された calibrator すべてに受理契約の宣言と生成 fitter があること**をテストで固定する。
- **bound**: 「3 × 2」は網羅すべきカテゴリであって、値が運ばれていることの証明ではない（証明は §5）。登録表を経由しない直接構築、`calibration.params` 以外の経路は含まない。

## 5. 受け入れ基準（テスト観点、完了基準の下書き）

1. **原典どおり**: 既定の platt が sklearn の `_sigmoid_calibration` と最適化許容誤差内で一致（参照としてテストでのみ使う）。
2. **intercept**: offset のずれがあるスコアで `b ≠ 0` に推定され、較正誤差が intercept なしより小さい。
3. **効き目**: 既定と異なる値（platt `target_smoothing=False`、beta `bounds`、isotonic `learning_rate`）で出力が既定と観測可能に異なる。
4. **`tol` の効き目**（round 2 指摘 1）: `tol` だけを変えると実効の停止条件が変わる。`options` に同じキーを書くと `options` が勝つ。
5. **手法の契約**（round 2 指摘 2）: 表の各手法で fit できる。`BFGS` / `CG` と `bounds` の組み合わせ、表に無い手法、scipy が知らない `options` キーが**学習前に** `CONFIG_INVALID`。
6. **縮尺**（round 2 指摘 3）: `max(|s|) ≥ 30` のスコアで、非ゼロの `x0` と有限の `bounds` を与えたとき、縮尺しない解き方と許容誤差内で同じ `(a, b)` になる。
7. **到達**: cross-fit の全 fold の calibrator と C_final が上書き後の値で構築される。
8. **生成コードの fitter**: 同じ生スコアとラベルで生成 `_fit_*` と実行時の出力が一致（3 手法）。**生成コードで再学習を実際に走らせる**。
9. **拒否**: 未知名、長さ違いの `x0` / `bounds`、タプルの `bounds`、`target_smoothing` の型違いが、**fit でも tune でも Booster と study が学習される前**に `CONFIG_INVALID`、出所 `calibration.params`。
10. **前処理の分離**: platt / beta の名前は正規名化されない。isotonic の別名は従来どおり（H-0094 決定 8 の回帰なし）。
11. **互換**（round 2 指摘 6）: 旧 `PlattCalibrator` を pickle した artifact の predict が、**通常のスコアと極端なスコアで**旧 predict と許容誤差内で一致。未学習の旧状態、新状態の再読込も通る。`FORMAT_VERSION` は 2。
12. **不変**: OOF-only・outer split・covered 行、`export_params` の形、`predict` の式、isotonic の `monotone_constraints` 強制。
13. **警告なし**: 既定の platt / beta の fit で sklearn / scipy の警告が出ない。
14. **位置の導出**: 登録された calibrator すべてに受理契約と生成 fitter がある。
15. **文書**: `README.md` と生成 `requirements.txt` の scipy の記述が一致する。

## 6. 互換性

- **platt の既定が変わる** → 新しい fit の calibrated 結果が変わる。CHANGELOG と H-0100 に数値ごと記載。旧 artifact は §3.7 のとおり変わらない。
- **これまで無視されていた `platt` / `beta` の params が効く**。LogisticRegression の引数名（`C` 等）は未知名として拒否（これまでも効いていなかった）。
- 生成 `config.json` にキーが増える。生成 `requirements.txt` と README に platt でも scipy が載る。生成済みのコードは影響を受けない。
- `format_version` は 2 のまま。公開 API（`CalibrationConfig`、`get_calibrator` のシグネチャ）は変えない。

## 7. 手続き

- 提案: **H-0100**（§0 の決定、§2.2 の新方針、§3.5 の BLUEPRINT 更新、§3.7 の移行、isotonic sigmoid 記述の訂正を含む）。
- BLUEPRINT: §12.2 に Platt / Beta の表（モデル・既定値・上書き・手法の契約）、暫定段落の置き換え、isotonic の sigmoid 記述の訂正。§15.4 に `calibration_params` の再現。
- 計画: `phase3-plan.md` §3 の 3c 行と §12.7 の「wire or refuse」を管理者の決定に合わせて更新（監視 round 1 の手当て）。
- **実装前に計測**: 名前検査は `allow` 条件なので、スイートを再生して `platt` / `beta` に空でない params を渡す config の数を記録する（grep では 0 件）。
- **最低依存での確認**（round 2 指摘 4）: lowest-direct レーンは `--frozen` なので、**`scikit-learn==1.3.*` と `scipy==1.10.*` を実際に入れた隔離環境**で calibration のテストを走らせ、**解決されたバージョンを記録**する。
- **完了基準は PR を開く前に書く**（§5 を対応表の形にする）。**ラウンド予算 8。**
