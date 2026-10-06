# PR 9 fold map (H-0110, #271)

One row per owed clause of `pr9_inventory_{A,B}.json` (`missed` / `contradicted`), per item
C1-C5 (scout C flags, H-0110 decision 6), per D1-D3 (H-0110 decision 4 and 8), and the two
clauses the design review reclassified from `off_surface` to `missed` (`H-0002#1`, `H-0082#2`).
`status`: `folded` = this fold wrote it; `already_stated` = BLUEPRINT §15.1-15.3 as folded
by the main context; `not_folded` = not written, with the evidence.
The quote column is a verbatim substring of `BLUEPRINT.md` (a script checks it; it is shown
inside a code span; a quote that itself contains backticks is a double-backtick span with
one padding space on each side, which is not part of the quote). Clause ids are
`<proposal>#<clause index in the inventory>`.

| id | clause (short) | status | exact quote from the new BLUEPRINT | introduced anchor token | note |
|---|---|---|---|---|---|
| H-0005#4 | evaluate_table() before fit -> MODEL_NOT_FIT | folded | `` `Model.evaluate_table()` を `fit()` の前に呼ぶと `MODEL_NOT_FIT`（H-0005） `` | `Model.evaluate_table()` | §4.1 |
| H-0005#2 | calibrated adds cal_oof only (contradicted) | folded | `` calibrated がある場合は `cal_oof` 列だけを追加する（fold 別の calibrated 列は無い、H-0005） `` | - | §13.2; `cal_fold_0`...`cal_fold_N-1` removed (entered BLUEPRINT in 4be4a30 without a proposal; code adds only `cal_oof`, table_formatter.py:45-49). Folded per the main-context correction |
| H-0006#1 | residuals()/residuals_plot() on classification -> UNSUPPORTED_TASK | folded | `` `Model.residuals()` / `Model.residuals_plot()` は binary / multiclass で `UNSUPPORTED_TASK`（H-0006） `` | `Model.residuals()` | §4.1 |
| H-0007#0 | importance(kind='shap') per-fold valid rows, fold-mean of mean(abs SHAP) | folded | `` fold ごとにその fold の validation 行（`valid_idx`）で SHAP を計算し `` | `Model.importance(kind="shap")` | §13.3; shap_explainer.py:165-174 |
| H-0007#1 | importance(kind='shap') returns dict[str, float] over every feature | folded | `` 全特徴量をキーに持つ `dict[str, float]` を返す `` | `Model.importance(kind="shap")` | §4.1 (holds for every kind) |
| H-0007#5 | missing shap -> OPTIONAL_DEP_MISSING | folded | `` shap が未導入なら `OPTIONAL_DEP_MISSING` `` | - | §13.3 |
| H-0008#1 | public plot methods return plotly.graph_objects.Figure | folded | `` 公開の plot メソッドはすべて `plotly.graph_objects.Figure` を返す（H-0008） `` | `plotly.graph_objects.Figure` | §13.3 |
| H-0009#4 | QQ plot uses OOS residuals only | folded | `QQ パネルは OOS 残差だけを使う（IS は使わない、H-0009）` | `plot_residuals()` | §13.3 |
| H-0009#9 | invalid kind raises an error | folded | `` 未知の `kind` は `CONFIG_INVALID`（H-0009 は `INVALID_CONFIG` と書いたが `` | `plot_residuals()` | HISTORY says INVALID_CONFIG, which is not an ErrorCode; code raises CONFIG_INVALID (residuals.py:271-279). Code wins |
| H-0011#0 | evaluate_table column order if_mean, oof, fold_*, cal_oof | folded | `` 列の順は `if_mean, oof, fold_0...fold_N-1, cal_oof` で固定する（H-0011） `` | `if_mean, oof, fold_0...fold_N-1, cal_oof` | §13.2 |
| H-0014#1 | precision_at_k needs_proba=True | folded | `` `needs_proba=True`（確率で上位を選ぶ）、`greater_is_better=True` `` | `k=10` | §13.1 |
| H-0014#2 | precision_at_k greater_is_better=True | folded | `` `needs_proba=True`（確率で上位を選ぶ）、`greater_is_better=True` `` | `k=10` | §13.1 |
| H-0014#3 | k defaults to 10 (top 10%), per-entry configurable | folded | `` 既定は `k=10`（上位 10%、`1 <= k <= 100`）で、MetricEntry ごとに指定できる `` | `k=10` | §13.1 (cross-ref §13.1.1) |
| H-0014#4 | precision_at_k for regression/multiclass -> UNSUPPORTED_METRIC | folded | `` `precision_at_k` を回帰か multiclass で（H-0014） `` | `k=10` | §13.1 task-compatibility bullet |
| H-0016#1 | confusion DataFrames: rows true, columns predicted, integer-indexed | folded | `行 = 真のラベル、列 = 予測ラベル、index / columns は整数（H-0016）` | `model.confusion_matrix()` | §13.3 |
| H-0016#5 | confusion_matrix() for regression -> UNSUPPORTED_TASK | folded | `` `model.confusion_matrix()` は regression で `UNSUPPORTED_TASK`（H-0016） `` | `model.confusion_matrix()` | §4.1 |
| H-0017#4 | calibration plots with calibration disabled -> CALIBRATION_NOT_SUPPORTED | folded | `` Calibration 未有効時に呼び出した場合は `CALIBRATION_NOT_SUPPORTED`（H-0017） `` | `Calibration 未有効時` | §4.1; phrase anchor from the H-0017 acceptance criteria |
| H-0017#5 | calibration plots on non-binary -> UNSUPPORTED_TASK | folded | `` `calibration_plot()` / `probability_histogram_plot()` は binary 以外の task で `UNSUPPORTED_TASK` `` | `Calibration 未有効時` | §4.1 |
| H-0018#2 | multiclass-OvR metrics for regression -> UNSUPPORTED_METRIC | folded | `` multiclass 対応の `auc` / `auc_pr` / `brier` を回帰で（H-0018） `` | `average_precision_score`, `brier_score_loss` | §13.1; OvR definition also written |
| H-0019#0 | roc_curve_plot supports multiclass (contradicted 'binary 専用') | folded | `` `model.roc_curve_plot()`（binary/multiclass。binary は IS/OOS の ROC Curve を重ね描き `` | `plot_roc_curve` | §4.1 'binary 専用' replaced; §13.3 already described multiclass |
| H-0019#5 | roc_curve_plot() for regression -> UNSUPPORTED_TASK | folded | `` `roc_curve_plot()` も regression で `UNSUPPORTED_TASK`（H-0019） `` | `plot_roc_curve` | §4.1 and §13.3 |
| H-0028#1 | tuning_plot() before tune() -> MODEL_NOT_FIT | folded | `` `Model.tuning_plot()` を `tune()` の前に呼ぶと `MODEL_NOT_FIT`（H-0028） `` | `Model.tuning_plot()` | §4.1 |
| H-0035#1 | params_table index parameter, sole column value | folded | `` 形: index は `parameter`、単一列は `value`（H-0035） `` | `単一列` | §4.1; phrase anchor |
| H-0035#3 | params_table includes early_stopping_rounds / validation_ratio | folded | `` training 設定（`early_stopping_rounds` / `validation_ratio`。その fit が使った値 `` | `単一列` | §4.1 |
| H-0035#4 | params_table fold-0 native objective/metric/learning_rate/.../max_bin | folded | `` fold 0 の学習済み booster から読んだ解決済みネイティブパラメーター（`objective` / `metric` / `learning_rate` `` | `解決された絶対値` | §4.1 (inventory listed under §14.4) |
| H-0035#5 | params_table fold-0 feature_fraction/.../num_iterations | folded | `` `bagging_freq` / `lambda_l1` / `lambda_l2` / `num_iterations`。booster に無い名前は行を作らない `` | `解決された絶対値` | §4.1 |
| H-0082#0 | evaluate(metrics=None) returns a deep copy | folded | `` `evaluate(metrics=None)` は内部の metrics dict の deep copy（`copy.deepcopy`）を返し `` | `deepcopy` | §4.1 |
| H-0082#1 | fit_result is a selective deep copy; estimators shared read-only; identity not kept | folded | `` 学習済みの `models` / `calibrator` / `pipeline_state` は参照を共有し、慣例として read-only とする `` | `__deepcopy__` | §7.1 |
| H-0082#2 | FitResult stays a non-frozen dataclass (design review round 1) | folded | `` `FitResult` は frozen でない `dataclass` のままにする `` | `__deepcopy__` | §7.1; reclassified from off_surface by the main context |
| H-0086#0 | fit() returns a selective deep copy | folded | `` 返すのは内部の `FitResult` ではなく、その選択的 deep copy（`FitResult.__deepcopy__`、§7.1）である `` | `inner_valid_explicit` | §6.2 |
| H-0086#1 | after load() internal metrics are a deep copy of fit_result.metrics | folded | `` `load()` は内部の metrics を `fit_result.metrics` の deep copy として持ち `` | `inner_valid_explicit` | §4.2 |
| H-0086#2 | metadata.json additive tuning block | already_stated | `` tune 済みのモデルだけ: `tuning` ブロック（`best_model_params` / `best_smart_params` `` | - | §15.1 (main context) |
| H-0086#3 | load() restores a minimal TuningResult from the tuning block | already_stated | `` `load()` はこれを tuning result として復元し、load 後の再 `fit()` が tuned params を再現する（H-0086） `` | - | §15.1 (main context); restated in §4.2 |
| H-0086#5 | model_dump emits computed inner_valid_explicit; validation pops it as source of truth | folded | `` `LizyMLConfig.model_dump()` は computed field の `inner_valid_explicit` を書き `` | `inner_valid_explicit` | §10.3.1 |
| H-0086#6 | lizyml top-level __all__ | folded | `` 公開面 (`__all__`: Model, FitResult, PredictionResult, TuningResult, `` | `load_config` | §19 tree (lizyml/__init__.py:19-35) |
| H-0001#5 | model is a discriminated union keyed by name | folded | `` `name` を判別キーとする discriminated union（`Field(discriminator="name")` `` | `discriminator`, `Literal["lgbm"]` | §5.4 top-level table |
| H-0001#7 | group_col required for group splits | folded | `` `group_kfold` / `stratified_group_kfold` / `group_time_series` では必須 `` | `ValidationError` | §5.4 data table; enforced only as a raw ValueError in the splitters (group_kfold.py:31,68; group_time_series.py:47) |
| H-0001#23 | invalid Config -> LizyMLError(CONFIG_INVALID) wrapping ValidationError | folded | `` 不正な Config は pydantic の `ValidationError` を包んだ `LizyMLError(CONFIG_INVALID)` になる `` | `ValidationError` | §5.1 |
| H-0025#3 | out-of-range min_data_in_leaf_ratio -> CONFIG_INVALID | folded | `` smart ratio の範囲バリデーション（`min_data_in_leaf_ratio` / `min_data_in_bin_ratio` は `(0,1)` の開区間 `` | `範囲バリデーション`, `(0,1)`, `早期エラー` | §5.1 |
| H-0025#4 | out-of-range min_data_in_bin_ratio -> CONFIG_INVALID | folded | `` smart ratio の範囲バリデーション（`min_data_in_leaf_ratio` / `min_data_in_bin_ratio` は `(0,1)` の開区間 `` | `範囲バリデーション`, `(0,1)`, `早期エラー` | §5.1 |
| H-0032#3 | purged/group time-series aliases | folded | `` `purged-time-series` / `purgedtimeseries`（→ `purged_time_series`） `` | `purged-time-series` | §5.1 |
| H-0034#1 | output_dir is a flat top-level Config key | folded | `` 実装は最上位のキー `output_dir` である（H-0034） `` | `output`, `ログファイルが出力先に保存される` | §5.4 row + §17; HISTORY said an `output` section, code has a flat key (schema.py:600) |
| H-0038#2 | purge_window accepted with warning -> purge_gap | folded | `` `purge_window` → `purge_gap`（H-0038） `` | `purge_window` | §5.4 notes |
| H-0038#3 | purged gap accepted with warning -> embargo | folded | `` `gap` → `embargo`（H-0038 は `embargo_pct` に写すと決めたが `` | `purge_window` | §5.4 notes; code maps gap -> embargo since H-0040 (schema.py:171-178) |
| H-0039#1 | output_dir precedence constructor > config > unset | folded | `` `output_dir` の優先順位は constructor > config > 未指定 `` | `constructor > config > 未指定`, `Model(..., output_dir=...)` | §17 (and §5.4 row) |
| H-0039#2 | fit/tune/export create run dirs; export() without path -> {run_dir}/export | folded | `` `export()` を path 無しで呼ぶと、直前の run のディレクトリがあれば `{run_dir}/export` に `` | `constructor > config > 未指定` | §17 |
| H-0040#7 | embargo_pct emits a deprecation warning | folded | `` `purged_time_series` の旧キーは移行期間のみ警告付きで受理し（`DeprecationWarning`、v1.0 で削除、H-0076） `` | `警告付きで受理` | §5.4 notes; the §10.2 `int()` sentence (not_in_force) left untouched |
| H-0055#1 | StratifiedGroupKFold aliases | folded | `` `stratified-group-kfold` / `stratifiedgroupkfold`（→ `stratified_group_kfold`、H-0055） `` | `stratified-group-kfold` | §5.1 |
| H-0076#0 | deprecated surfaces scheduled for removal in v1.0 | folded | `` 非推奨の面（`validation_ratio` の入力、`calibration.n_splits`、`purged_time_series` の `` | `build_calibration_splitter`, `docs/DEPRECATIONS.md` | §5.1 |
| H-0076#1 | docs/DEPRECATIONS.md is the single registry | already_stated | `` 非推奨の面とその削除目標（v1.0）の唯一の登録簿は `docs/DEPRECATIONS.md` である `` | `docs/DEPRECATIONS.md` | §15.2 (main context) |
| H-0080#0 | random_state int or None, default None, for kfold/stratified/stratified_group (contradicted) | folded | `` `n_splits=5`, `random_state=null`, `shuffle=True` `` | `KFoldConfig`, `StratifiedKFoldConfig`, `StratifiedGroupKFoldConfig` | §5.4 split table (42 -> null on 3 rows) |
| H-0080#1 | None inherits training.seed at build; explicit wins; not written back | folded | `` `null`（既定）なら splitter を構築するときに `training.seed` を継承し、明示した値はそれに勝つ `` | `KFoldConfig` | §5.4 notes |
| H-0071#5 | codegen reproduces smape / wape feval | folded | `f1, brier, ece, precision_at_k, accuracy, rmsle, r2, smape, wape` | `smape`, `wape` | §6.6 (templates.py:219-241) |
| H-0002#1 | FitResult / PredictionResult / SplitIndices / RunMeta are dataclasses (design review round 1) | folded | `` `FitResult` / `PredictionResult` / `SplitIndices` / `RunMeta` は `dataclass` であり `` | `lizyml_version` | §7.1; reclassified from off_surface by the main context |
| H-0002#2 | oof_pred is np.ndarray (contradicted) | folded | `` `oof_pred`（`np.ndarray`。shape は regression / binary で `(n_samples,)`、multiclass で `(n_samples, n_classes)`） `` | `lizyml_version` | §7.1; `pd.Series` dropped |
| H-0002#3 | oof_pred shape | folded | `` `oof_pred`（`np.ndarray`。shape は regression / binary で `(n_samples,)`、multiclass で `(n_samples, n_classes)`） `` | `lizyml_version` | §7.1 |
| H-0002#4 | if_pred_per_fold length n_splits, each predicts the full training fold | folded | `` 長さは `n_splits`、各要素はその fold の学習行（`train_idx`）全体に対する予測 `` | `lizyml_version` | §7.1 |
| H-0002#6 | raw/if_per_fold len == n_splits | folded | `` `oof_per_fold` / `if_per_fold` の長さは `n_splits`（fold の順） `` | `lizyml_version` | §7.1 |
| H-0002#10 | feature_names ordered list; dtypes name->dtype str; categorical_features names | folded | `` `feature_names`: 学習に使った特徴量名の順序付き `list[str]` `` | `lizyml_version` | §7.1 |
| H-0002#18 | RunMeta field names lizyml_version / deps_versions (contradicted) | folded | `` `lizyml_version / python_version / deps_versions / config_normalized / config_version / run_id / timestamp` `` | `lizyml_version` | §7.1; same edit as D2 |
| H-0002#19 | run_id UUID string, timestamp ISO-8601 string | folded | `` `run_id`（`str`、UUID 文字列）、`timestamp`（`str`、ISO 8601、UTC） `` | `lizyml_version` | §7.1 (model.py:802 uses UTC isoformat) |
| H-0002#21 | PredictionResult.pred np.ndarray (n_samples,) | folded | `` `pred`（`np.ndarray`、shape `(n_samples,)`。回帰: 予測値、分類: クラスラベル `` | `lizyml_version` | §7.3; 'class / proba' narrowed (pred is always labels for classification) |
| H-0002#22 | binary proba positive-class probability (n_samples,) | folded | `` `proba`（binary: 正例の確率、shape `(n_samples,)`。校正が有効なら `C_final` を通した値 `` | `lizyml_version` | §7.3 |
| H-0002#24 | shap_values None unless requested; (n_samples, n_features) | folded | `` `shap_values`（要求時のみ（`return_shap=True`）、それ以外は `None`。shape は task を問わず `(n_samples, n_features)` に統一し `` | `lizyml_version` | §7.3 |
| H-0002#25 | used_features lists the features used for prediction | folded | `` `used_features`（予測に使った特徴量名。refit が学習した `feature_names`。列ズレ検知用） `` | `lizyml_version` | §7.3 |
| H-0107#0 | columns checked in order; first leak raises immediately; return only after all compared | folded | `` `raise_on_violation=True` では最初に見つかった漏洩列で `LEAKAGE_SUSPECTED` を即座に送出する `` | `raise_on_violation` | §8.2; target-absent -> [] left undecided (#311) |
| H-0107#1 | comparison exception -> DATA_SCHEMA_INVALID {column, target}, cause | folded | `` 比較が例外を出した列は、`raise_on_violation` に関わらず `LizyMLError(DATA_SCHEMA_INVALID)` `` | `raise_on_violation` | §8.2 |
| H-0107#2 | catch scope comparison only, any Exception; LEAKAGE_SUSPECTED outside | folded | `捕まえる範囲は比較の呼び出し 1 つだけで、例外の型は問わない` | `raise_on_violation` | §8.2 |
| H-0067#5 | mixed finite/NaN val scores: calibrate finite, fallback NaN rows | folded | `有限の行だけを校正し、NaN の行には fallback（未校正の OOF 確率）を書く` | `calibration/cross_fit.py` | §10.5; dead in practice under H-0058 outer-split reuse (DC6) - stated as a guard |
| H-0067#6 | all-NaN val scores: fallback for every row | folded | `validation 行がすべて NaN の fold では全行に fallback を書く` | `calibration/cross_fit.py` | §10.5 |
| H-0078#0 | parse_space rejects low >= high | folded | `` `low >= high` は `CONFIG_INVALID` `` | `parse_space` | §11.1 |
| H-0078#1 | parse_space rejects log=True with low <= 0 | folded | `` `log=True` で `low <= 0` も `CONFIG_INVALID` `` | `parse_space` | §11.1 |
| H-0078#2 | Protocol parameter_bounds(task) (contradicted) | folded | `def parameter_bounds(` | `parameter_bounds` | §14.4 Protocol block |
| H-0078#3 | LGBMProvider.parameter_bounds 15-parameter table | folded | `` `LGBMProvider` は task によらない 15 パラメーターの表 `` | `parameter_bounds` | §14.4 |
| H-0078#4a | expansion clamps new_low/new_high to provider bounds | folded | `` 拡張後の `new_low` / `new_high` は次元の `min_allowed` / `max_allowed`（§11.2）でクランプする `` | `min_allowed` | §11.5 |
| H-0078#4b | both sides blocked -> no expansion (expanded=False re-judged) | not_folded | - | - | not in force: detect_boundary sets expanded=should_expand and appends the name to expanded_names whenever an edge is hit, with no re-judgement after the clamp (search_space.py:378-416); HISTORY:6417 promises the re-judgement. Not written |
| H-0078#5 | BoundaryDimStatus.clamped_to_bound: bool = False (contradicted) | folded | `clamped_to_bound: bool = False   # H-0078` | `clamped_to_bound` | §11.5 code block |
| H-0078#6 | FloatDim/IntDim optional min_allowed/max_allowed | folded | `` `FloatDim` / `IntDim` は任意の `min_allowed` / `max_allowed`（既定 `None`）を持つ `` | `min_allowed` | §11.2 |
| H-0078#7 | Model.tune attaches parameter_bounds to default and user spaces | folded | `` `Model.tune` は既定の空間にも利用者の空間にも `provider.parameter_bounds(task)`（§14.4）を付ける `` | `parameter_bounds` | §11.2 (and §14.4) |
| H-0030#3 | calibrator predict consumes raw scores, returns calibrated probabilities | folded | `` `BaseCalibratorAdapter.predict()` は生スコアを受け取り校正済み確率を返す（H-0030） `` | `BaseCalibratorAdapter.predict()` | §12.1 |
| H-0089#3 | fallback rows get uncalibrated raw OOF and stay in calibrated_oof/metrics | folded | `` fallback の行は `calibrated_oof` と calibrated metrics に含まれたまま残り（値は H-0089 で変わらない） `` | `n_fallback_rows` | §12.1 |
| H-0080#3 | isotonic seed inherits training.seed (contradicted seed=42) | folded | `` `seed` は `calibration.params` に無ければ `training.seed` を継承する `` | `KFoldConfig` | §12.2 Isotonic |
| H-0089#0 | CalibrationResult.fallback_fold_flags: list[bool] | folded | `` `fallback_fold_flags: list[bool]`（fold ごと、split の順 `` | `fallback_fold_flags` | §12.3 |
| H-0089#1 | CalibrationResult.n_fallback_rows: int = 0 | folded | `` `n_fallback_rows: int = 0`（未校正の fallback を受けた validation 行の総数 `` | `n_fallback_rows` | §12.3 |
| H-0004#1 | MAPE with zero y_true -> UNSUPPORTED_METRIC | folded | `` `mape`: `y_true` に 0 を含むと `UNSUPPORTED_METRIC`（H-0004） `` | `HuberLoss`, `delta=1.0` | §13.1 |
| H-0004#2 | HuberLoss delta default 1.0, configurable; 'huber' means delta=1.0 | folded | `` 文字列 `"huber"` は `delta=1.0` を意味する（H-0004） `` | `HuberLoss`, `delta=1.0` | §13.1 (cross-ref §13.1.1) |
| H-0004#3 | Huber quadratic below delta, linear above | folded | `` `huber`: 誤差 `e` について `` | `HuberLoss` | §13.1; the formula follows the quote (it contains pipe characters, so it is not quoted here) |
| H-0004#4 | mape/huber for non-regression -> UNSUPPORTED_METRIC | folded | `` `mape` / `huber` を回帰以外で（H-0004） `` | `HuberLoss` | §13.1 |
| H-0049#0 | multiclassova predict_proba rows need not sum to 1 | folded | `` `objective: multiclassova` ではクラスごとの独立 sigmoid の出力なので、行和は 1 とは限らない `` | `needs_simplex` | §14.1 |
| H-0049#1 | evaluator row-normalises needs_simplex multiclass predictions, zero-row guard | folded | `` `needs_proba` かつ `needs_simplex` の指標には、multiclass の 2 次元予測を行和で正規化して渡す（全 0 の行は割らない） `` | `needs_simplex`, `_pred_for_metric()` | §13.1 |
| H-0049#2 | BaseMetric.needs_simplex default False; AUC/LogLoss True | folded | `` `BaseMetric.needs_simplex` は既定 `False` の具体プロパティで `` | `needs_simplex`, `BaseMetric.needs_simplex` | §13.1; `supports_task` (not_in_force) left in place |
| H-0071#0 | smape definition, range [0,200], 0/0 rows contribute 0 | folded | `` 範囲 `[0, 200]`。`y = ŷ = 0` の行は 0 として数える（H-0071） `` | `smape` | §13.1; formula precedes the quote |
| H-0071#1 | wape definition; UNSUPPORTED_METRIC only when sum(abs y_true)==0 | folded | `` `UNSUPPORTED_METRIC` になるのは `sum( `` | `wape` | §13.1; quote stops before the pipe characters of the formula |
| H-0071#2 | smape/wape greater_is_better=False, needs_proba=False | folded | `` `smape` / `wape` はどちらも `greater_is_better=False`、`needs_proba=False` `` | `smape`, `wape` | §13.1 |
| H-0071#3 | _TASK_METRICS['regression'] includes smape and wape | folded | `` `_TASK_METRICS["regression"]` = `rmse` / `mae` / `r2` / `rmsle` / `mape` / `huber` / `smape` / `wape`（H-0071） `` | `smape`, `wape` | §13.1 |
| H-0089#2 | metrics['calibrated'].fallback_row_count | folded | `` `fallback_row_count`（= `CalibrationResult.n_fallback_rows`、fallback が無ければ `0`） `` | `fallback_row_count` | §13.2 |
| H-0079#0 | task-compatible objective passes through; incompatible -> CONFIG_INVALID (contradicted) | folded | `` 含まれればそのまま `lgb.train` に渡し、含まれなければ `CONFIG_INVALID` とする（H-0079） `` | `TASK_COMPATIBLE_OBJECTIVES` | §14.2 and §18.1.2 task-locked bullet (old 'タスクから固定' corrected) |
| H-0079#1 | TASK_COMPATIBLE_OBJECTIVES 9/3/2 | folded | `LightGBM の canonical 名で regression 9 / binary 3 / multiclass 2` | `TASK_COMPATIBLE_OBJECTIVES` | §14.2 |
| H-0079#2 | Protocol objective_choices(task) -> tuple (contradicted) | folded | `def objective_choices(self, task: TaskType) -> tuple[str, ...]` | `objective_choices` | §14.4 Protocol block + bullet |
| H-0079#3 | Protocol metric_choices(task) -> {native, feval} (contradicted) | folded | `def metric_choices(self, task: TaskType) -> MetricChoices` | `metric_choices` | §14.4 Protocol block + bullet |
| H-0079#5 | multiclass native whitelist excludes auc | folded | `` multiclass のネイティブ whitelist に `auc` は無い `` | `metric_choices` | §14.3 Metric Bridge |
| H-0071#4 | smape/wape are regression feval metrics driving early stopping | folded | `` `smape` / `wape` は回帰の feval 指標なので、`params.metric` に書けば early stopping と学習曲線を駆動する（H-0071） `` | `smape`, `wape` | §14.3 (table also gains r2, which the code has) |
| H-0105#0 | feval does not re-transform LightGBM predictions (contradicted) | folded | `**feval は LightGBM が渡す予測を再変換しない**（H-0105）` | `_pred_for_metric` | §14.3; sigmoid/softmax column and bullets replaced |
| H-0105#1 | feval hands values under the evaluator's rule | folded | `` feval は evaluator と同じ規則（`evaluation.evaluator._pred_for_metric`、§13.1）で metric に値を渡す `` | `evaluation.evaluator._pred_for_metric` | §14.3 |
| H-0105#2 | non-2-D multiclass feval input -> EVALUATION_FAILED (contradicted reshape) | folded | `` multiclass の feval 入力が 2 次元 `(n, num_class)` でなければ reshape せず `EVALUATION_FAILED` を送出する `` | `_pred_for_metric` | §14.3 |
| H-0003#0 | artifact directory: metadata.json + fit_result.pkl + refit_model.pkl, joblib | already_stated | `` 中身は `metadata.json`、`fit_result.pkl`（`FitResult`）、`refit_model.pkl`（`RefitResult`） `` | `fit_result.pkl` | §15.3 (main context) |
| H-0003#1 | metadata.json keys | already_stated | `` 常に書く: `format_version` / `lizyml_version` / `python_version` / `timestamp` / `run_id` `` | `fit_result.pkl` | §15.1 (main context); list also includes v2 `checksums` |
| H-0003#3 | adding fields is minor; removal/type change bumps format_version | already_stated | `` **フィールドの追加は後方互換の変更で、`format_version` を上げない。** `` | - | §15.2 (main context) |
| H-0003#5 | joblib .pkl; load only from trusted sources | already_stated | `` `Model.load()` は信頼できる出どころの artifact だけに使う（H-0003、上記の脅威モデル） `` | - | §15.3 (main context) |
| H-0003#6 | metadata validated on load; missing required -> DESERIALIZATION_FAILED | already_stated | `` 必須キー（`_REQUIRED_METADATA_KEYS` = `format_version` / `task` / `feature_names` / `config` / `run_id`）が欠けていれば `` | - | §15.2 (main context) |
| H-0026#3 | analysis_context stored as optional analysis_context.pkl | already_stated | `` 任意の `analysis_context.pkl`（load 後の診断 API が使う `y_true` と `X_for_explain`。無ければ load 時に `None`、H-0026） `` | `再 export を促す` | §15.3 (main context) |
| H-0026#4 | legacy artifacts without analysis_context usable for predict/evaluate | already_stated | `` `analysis_context.pkl` を持たない artifact（H-0026 以前）でも `predict()` と `evaluate()` は使える `` | `再 export を促す` | §15.2 (main context) |
| H-0026#5 | diagnostics on legacy artifacts fail with MODEL_NOT_FIT + re-export message | already_stated | `` 必要なデータが無いと `MODEL_NOT_FIT` で明示的に失敗し、最新版での再 export を促す（H-0026） `` | `再 export を促す` | §15.2 (main context); phrase anchor |
| H-0083#0 | metadata.json checksums {algorithm, files} | already_stated | `` `checksums` は `{"algorithm": "sha256", "files": `` | `checksums`, `sha256` | §15.1 (main context) |
| H-0083#1 | load verifies digests; unknown algorithm/mismatch -> DESERIALIZATION_FAILED | already_stated | `` `algorithm` が `sha256` でない、または digest が一致しないときは pickle を実行する前に `DESERIALIZATION_FAILED` `` | `checksums` | §15.2 (main context) |
| H-0083#2 | artifacts without checksums load unverified | already_stated | `` `checksums` を持たない artifact（H-0083 以前）と、`files` に載っていないファイルは検査せずに読む `` | `checksums` | §15.2 (main context) |
| H-0083#4 | bytes read once, verified in memory (no TOCTOU) | already_stated | `ファイルを再 open しないので、検査と復元の間の TOCTOU が無い` | `joblib.load(io.BytesIO(...))` | §15.2 (main context) |
| H-0083#5 | threat model: metadata unsigned; checksums not a pickle safety guarantee | already_stated | `` `metadata.json` 自体は署名しない `` | `checksums` | §15.2 (main context) |
| H-0090#0 | train.py rebuilds calibration-OOF folds from config.json['split'] per split.method | folded | `` 生成 `train.py` は校正用 OOF の CV fold をこのブロックから作り、`split.method` を再現する `` | `config.json["split"]`, `argsort()` | §15.4 (+ §6.6 pointer) |
| H-0090#1 | sort by time_col / blocks.col with argsort, map back | folded | `` `blocked_group_kfold` では `blocks.col` で pandas の `argsort()` により並べてから分割し、fold を元の行順に戻す `` | `argsort()` | §15.4 |
| H-0090#2 | config.json split block with resolved values | folded | `` method 固有のパラメーターを解決済みの値で書く（`_build_split_metadata(cfg)` `` | `_build_split_metadata(cfg)`, `config.json["split"]` | §15.4 |
| H-0090#3 | generated calibrator fitted on covered rows only | folded | `生成 calibrator は covered な（OOF が NaN でない）行だけで学習する` | `config.json["split"]` | §15.4 |
| H-0090#4 | exports without split block fall back to legacy shuffled K-fold | folded | `` `split` ブロックを持たない export（H-0090 以前）は、従来の task 別のシャッフル K-fold `` | `config.json["split"]` | §15.4 |
| H-0105#3 | generated train.py feval does not re-transform; non-2-D -> ValueError | folded | `` 生成 `train.py` の feval も LightGBM が渡す予測を再変換しない（§14.3、H-0105） `` | `_pred_for_metric` | §15.4 |
| H-0091#0 | tune lives in _model_tuning.py; model.py no longer holds tune (contradicted) | folded | `` `model.py` は `tune` を持たない（H-0091） `` | `ModelTuningMixin` | 付録 B 実装メモ; `tune` removed from model.py's lifecycle list |
| H-0091#1 | mixins split into read-only diagnostic and writer orchestrator | folded | `` **診断用の read-only mixin**（`_model_plots` / `_model_tables` / `_model_persistence`）と `` | `ModelTuningMixin`, `_MIXIN_FILES` | 付録 B |
| H-0091#2 | INV-2: exactly one writer mixin, excluded from the read-only guard | folded | `` の read-only の静的検査（`_MIXIN_FILES`）から除外される（INV-2） `` | `_MIXIN_FILES` | 付録 B |
| H-0091#3 | §19 tree lists _model_tuning | folded | `_model_tuning.py            ModelTuningMixin (tune orchestration、唯一の writer mixin、H-0091)` | `ModelTuningMixin` | §19 |
| C1 H-0057 | NaN in a covered OOF row is refused | folded | `` `Evaluator.evaluate()` は `LizyMLError(EVALUATION_FAILED)` `` | `Evaluator.evaluate()` | §13.2; HISTORY (3713, 3744) says ValueError, code raises LizyMLError(EVALUATION_FAILED) (evaluator.py:140-157). Code wins; discrepancy stated in BLUEPRINT |
| C2 H-0085 | NaN in a numeric target -> DATA_SCHEMA_INVALID | folded | `` 数値 target に NaN があれば `Model.fit` の入口で `LizyMLError(DATA_SCHEMA_INVALID)`（context `nan_count`） `` | `nan_count`, `LizyMLError(DATA_SCHEMA_INVALID)` | §8.2; target_encoder.py:123-135 (classification targets refused the same way, also stated) |
| C3 H-0087 | three public validators via lizyml.data, not wired into Model.fit | folded | `` `lizyml.data` が `validate_time_series_order` / `validate_no_target_leakage` / `validate_group_split` を公開する `` | `validate_time_series_order`, `validate_group_split`, `lizyml.data` | §8.2; lizyml/data/__init__.py:1-28 |
| C3b H-0087 | not auto-wired into Model.fit | folded | `` **`Model.fit` には自動配線しない** `` | `validate_time_series_order` | §8.2 |
| C4 H-0104 | transform_with_warnings concrete default | folded | `` `BaseFeaturePipeline.transform_with_warnings(X)` は具体メソッドで、既定の実装は `(self.transform(X), [])` である `` | `transform_with_warnings` | §9.1; pipeline_base.py:50-70 |
| C4b H-0104 | get_state categorical_cols optional key | folded | `` `get_state()` の `"categorical_cols"` キーは任意で `` | `categorical_cols` | §9.1; pipeline_base.py:76-81 |
| C5 H-0106 | INCOMPATIBLE_COLUMNS condition | folded | `` 学習時に数値だった列（`FitResult.dtypes` の記録）が予測時に数値でない dtype で届いたら `INCOMPATIBLE_COLUMNS` `` | `FitResult.dtypes`, `fit_dtype`, `predict_dtype` | §9.2; column_check.py:88-108 |
| C5b H-0106 | METRIC_REQUIRES_PROBA condition | folded | `` を受け取ると `METRIC_REQUIRES_PROBA` を送出する（H-0106） `` | `cross_entropy_lambda` | §13.1; classification.py:45-86 |
| D1 H-0110 | multiclass PredictionResult.proba is (n, k) | folded | `` multiclass: shape `(n_samples, n_classes)`。regression: `None` `` | - | §7.3; resolves the H-0002 binary-only not_in_force clause (_model_predict.py:87-89) |
| D2 H-0110 | rename yourlib_version -> lizyml_version (every occurrence) | folded | `` `lizyml_version / python_version / deps_versions / config_normalized / config_version / run_id / timestamp` `` | - | §7.1 (this fold) and §15.1 (main context); overlaps H-0002#18. 0 occurrences of yourlib remain |
| D2b H-0110 | rename deps_version -> deps_versions (every occurrence) | folded | `` `deps_versions`（`dict[str, str]`） `` | - | §7.1; §14.4 already said `RunMeta.deps_versions`; overlaps H-0002#18 |
| D2c H-0110 | rename YourLibError -> LizyMLError (every occurrence) | folded | `LizyMLError(code, user_message, *, debug_message=None, cause=None, context=None)` | - | §16.1; signature also gains keyword-only marker and `context` (exceptions.py:45-53) |
| D3 H-0110 | proposal_dispositions.toml governance line | folded | `` `docs/proposal_dispositions.toml` が HISTORY の全提案について `` | `proposal_dispositions.toml` | §18.1.1 |

Counts: already_stated 16, folded 124, not_folded 1 (rows 141).

## Introduced anchors per proposal

Each token occurs by full token in the proposal's own HISTORY entry, is absent from
BLUEPRINT.md at `13fb9d7`, and is present now (`has_token` from
`tests/test_docs/test_proposal_blueprint_coverage.py`).

- H-0001: `discriminator`, `Literal["lgbm"]`, `ValidationError`
- H-0002: `lizyml_version`
- H-0003: `fit_result.pkl`
- H-0004: `HuberLoss`, `delta=1.0`
- H-0005: `Model.evaluate_table()`
- H-0006: `Model.residuals()`
- H-0007: `Model.importance(kind="shap")`
- H-0008: `plotly.graph_objects.Figure`
- H-0009: `plot_residuals()`
- H-0011: `if_mean, oof, fold_0...fold_N-1, cal_oof`
- H-0014: `k=10`
- H-0016: `model.confusion_matrix()`
- H-0017: `Calibration 未有効時`
- H-0018: `average_precision_score`, `brier_score_loss`
- H-0019: `plot_roc_curve`
- H-0025: `範囲バリデーション`, `(0,1)`, `早期エラー`
- H-0026: `再 export を促す`
- H-0028: `Model.tuning_plot()`
- H-0030: `BaseCalibratorAdapter.predict()`
- H-0032: `purged-time-series`
- H-0034: `output`, `ログファイルが出力先に保存される`
- H-0035: `単一列`, `解決された絶対値`
- H-0038: `purge_window`
- H-0039: `constructor > config > 未指定`, `Model(..., output_dir=...)`
- H-0040: `警告付きで受理`
- H-0049: `needs_simplex`, `_pred_for_metric()`, `BaseMetric.needs_simplex`
- H-0055: `stratified-group-kfold`
- H-0057: `Evaluator.evaluate()`
- H-0067: `calibration/cross_fit.py`
- H-0071: `smape`, `wape`
- H-0076: `build_calibration_splitter`, `docs/DEPRECATIONS.md`
- H-0078: `parse_space`, `parameter_bounds`, `min_allowed`, `clamped_to_bound`
- H-0079: `TASK_COMPATIBLE_OBJECTIVES`, `objective_choices`, `metric_choices`
- H-0080: `KFoldConfig`, `StratifiedKFoldConfig`, `StratifiedGroupKFoldConfig`
- H-0082: `deepcopy`, `__deepcopy__`
- H-0083: `checksums`, `sha256`, `joblib.load(io.BytesIO(...))`
- H-0085: `nan_count`, `LizyMLError(DATA_SCHEMA_INVALID)`
- H-0086: `inner_valid_explicit`, `load_config`
- H-0087: `validate_time_series_order`, `validate_group_split`, `lizyml.data`
- H-0089: `n_fallback_rows`, `fallback_fold_flags`, `fallback_row_count`
- H-0090: `config.json["split"]`, `argsort()`, `_build_split_metadata(cfg)`
- H-0091: `ModelTuningMixin`, `_MIXIN_FILES`
- H-0104: `transform_with_warnings`, `categorical_cols`
- H-0105: `_pred_for_metric`, `evaluation.evaluator._pred_for_metric`
- H-0106: `FitResult.dtypes`, `fit_dtype`, `predict_dtype`, `cross_entropy_lambda`
- H-0107: `raise_on_violation`
- H-0110: `proposal_dispositions.toml`
