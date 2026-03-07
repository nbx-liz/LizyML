# 0. ステータスとスコープ

## 0.1 ステータス

- 本ドキュメントは実装の単一の正とする（仕様変更は `HISTORY.md` の提案プロセスを経る）。
- 「仕様未確定は仮実装しない」を厳守する。

## 0.2 スコープ（当面）

- 最初は LightGBM を最優先でサポートする。
- 将来拡張として `sklearn` / DNN（Torch）を想定し、IF と境界を先に固定する。

## 0.3 非スコープ（当面）

- 分散学習基盤（Ray / Dask 等）への本格対応。
- Auto Feature Engineering の大型実装（ただし拡張点は確保する）。

# 1. 目的

複数の分析ライブラリを使って、以下の分析機能を Config 駆動で統一的に実行する。

- 最適化: `tune`（例: Optuna）
- 学習: `fit`（CV / Refit / EarlyStopping）
- 評価: `evaluate`（IF / OOF、校正前後の比較）
- 推論: `predict`（列ズレ検知、説明可能性オプション）
- 配布: `export`（Model Artifact、互換性管理）

# 2. 設計原則

- 再現性を最優先する。
- `seed / split / params / versions / data schema / split indices / data fingerprint` を必ず保存する。
- 仕様未確定は仮実装しない。
- 独自推測実装を禁止し、必ず提案プロセス（`HISTORY.md`）を経る。
- 「Facade は組み立てのみ」とする。
- `Model` はロジックを持たず、部品を接続して実行する。
- IF を固定し、実装の自由度を確保する。
- `Splitter / FeaturePipeline / EstimatorAdapter / Tuner / Calibrator / Metric / Explainer` を分離する。

# 3. 要件（機能・品質）

## 3.1 品質要件

- 保守・可読性が高い。
- `1クラス1ファイル / 単一責任 / 重複排除 / 神クラス禁止` を守る。
- 例外処理を統一する。
- ユーザー向けメッセージと開発者向けデバッグ情報を分離する。
- Optional dependency を明確化する。
- Torch 等は optional とし、未導入時エラーも統一する。

## 3.2 機能要件（ユーザー価値）

- 少ないコード量でモデル構築・評価できる。
- 学習過程、特徴量重要度、残差分布などを可視化できる。
- 評価指標を複数サポートし、ユーザーが選択できる。
- CV 時に IF と OOF の両方を返す。
- 保存・読込を提供し、互換性管理と破壊的変更を前提に扱う。
- 新規データ予測・評価で列ズレ検知とスキーマ強制 / 警告ポリシーを持つ。
- 特徴量加工・目的変数加工（`FeaturePipeline`）を扱える。
- CV と HPO（Optuna 等）を扱える。
- Binary のスコアキャリブレーションを提供する。
- `Platt / Beta / Isotonic` を扱う（`Isotonic` は LGBM の単調制約を利用）。
- 校正のためのデータ分割・cross-fit を行う（OOF のみ利用、リーク禁止）。
- 特徴量指定の手間を減らす。
- `target` 指定後、その他を自動で feature 選択する。
- `exclude` を指定可能にする。
- 非数値データの categorical 自動扱い（LGBM 前提）と明示指定をサポートする。

## 3.3 追加の必須要件（抜けやすい実務要件）

- Config の入口を整備する。
- `YAML / JSON / dict`、CLI / 環境変数 override、Config versioning、正規化（表記揺れ / alias）に対応する。
- split indices を保存する（外側 CV / inner valid / 校正のすべて）。
- data fingerprint を保存する（ファイルパスだけに依存しない）。
- `FeaturePipeline` の状態を永続化する（学習時の統計量・カテゴリ辞書等）。
- 列ズレ時の方針を仕様化する（余剰列 / 不足列 / unseen category）。
- `tuning x CV` のリーク回避方針を仕様化する（同一 CV での最適化から評価の楽観化を防ぐ）。
- パッケージ配布時の build 定義と配布メタデータを固定する（PyPI に公開できる最小要件を満たす）。
- インストール直後の import 導線と README の利用例を一致させる（公開 API と利用例の乖離を禁止する）。

# 4. 公開 API（案）

## 4.1 Model（学習・評価・推論の Facade）

```python
model = Model(config=config)
tuning_result = model.tune()       # TuningResult（best_params / best_score / trials）
tuning_df = model.tuning_table()   # 全 trial の DataFrame（trial / score / params）
fit_result = model.fit()
eval_result = model.evaluate()
pred_result = model.predict(X_test, return_shap=True)
model.export("path/to/export_dir")
```

補足:
- `fit()` の default は、最も評価が良かったパラメーターで学習する。
- 必要に応じて、最終学習に使うパラメーターを明示指定できるようにする。
- `tune()` は `TuningResult` を返す。`TuningResult` は `best_params`（最良パラメーター）、`best_score`（最良スコア）、`trials`（全 trial の `TrialResult` リスト）を持つ。
- `tuning_table()` は `TuningResult.trials` を `pd.DataFrame` に変換して返す（列: `trial`, メトリクス名, 探索パラメーター名）。`tune()` 未実行時は `MODEL_NOT_FIT`。
- 学習後は、以下の補助 API を提供する。
  - `model.importance(kind="split|gain|shap")`（特徴量重要度。`shap` は optional dependency）
  - `model.importance_plot(kind="split|gain|shap", top_n=20)`（特徴量重要度の可視化、Plotly）
  - `model.residuals()`（回帰専用。OOF 残差 `y - oof_pred` を `np.ndarray` で返す）
  - `model.residuals_plot(kind="scatter|histogram|qq|all")`（回帰専用。残差可視化、Plotly。IS/OOS 比較対応。デフォルト `kind="all"` で scatter + histogram + QQ の 3 パネル。scatter は Actual vs Predicted（x=predicted, y=actual）。IS サンプルは OOS 数に合わせてダウンサンプリング）
  - `model.evaluate_table()`（評価結果を `pd.DataFrame` で返す）
  - `model.roc_curve_plot()`（binary 専用。IS/OOS の ROC Curve を重ね描き、Plotly）
  - `model.confusion_matrix(threshold=0.5)`（binary/multiclass。IS/OOS の Confusion Matrix を `{"is": DataFrame, "oos": DataFrame}` で返す）
  - `model.calibration_plot()`（binary + calibration 有効時。Raw/Calibrated の Reliability Diagram、Plotly）
  - `model.probability_histogram_plot()`（binary + calibration 有効時。Raw/Calibrated の確率分布ヒストグラム、Plotly）
  - `model.tuning_plot()`（`tune()` 後。trial ごとのスコア推移と最良スコア推移を重ね描き、Plotly。完了/枝刈り/失敗を色分け）
  - `model.split_summary()`（fold ごとの分割情報を `pd.DataFrame` で返す。時系列分割時は期間情報を含む）
  - `model.params_table()`（解決済みパラメーターテーブル。Config smart params + resolved booster params + fold ごとの `best_iteration` を単一 `pd.DataFrame` で返す。`fit()` 未実行時は `MODEL_NOT_FIT`）
  - `model.fit_result`（read-only プロパティ。`fit()` 後の `FitResult` を返す。`fit()` 未実行時は `MODEL_NOT_FIT`）
- `residuals()` / `residuals_plot()` / `importance(kind="shap")` / `roc_curve_plot()` / `confusion_matrix()` / `calibration_plot()` / `probability_histogram_plot()` は、`fit()` 後と `Model.load()` 後の両方で利用可能とする。
- `Model.load()` 後の上記 API は、Artifact に含める `analysis_context`（`y_true`, `X_for_explain`）を参照して動作させる。

## 4.2 `Model.load()`（Artifact 読込）

`export` で生成される `Model Artifact` をロードし、推論だけでなく学習時の評価情報や設定も参照できるようにする。

```python
loaded_model = Model.load("export_dir")
eval_result = loaded_model.evaluate()
pred_result = loaded_model.predict(X_new)
```

# 5. Config 設計

## 5.1 方針

- `pydantic`（`extra="forbid"`）で typo を確実にエラー化する。
- `config_version / schema_version` を必須にする。
- Config loader で以下を統一する。
  - 読込: `dict / JSON / YAML`
  - override: CLI / 環境変数（例: `LIZYML__model__lgbm__params__learning_rate=0.05`）
  - 正規化: 表記揺れの吸収（例: `k-fold` と `kfold`）、deprecated key の警告 / 拒否方針

## 5.2 Config 例（dict）

```python
config = {
    "config_version": 1,
    "task": "regression",
    "data": {"path": "data.csv", "target": "y"},
    "features": {
        "exclude": ["id"],
        "auto_categorical": True,
        "categorical": ["cat_feature1", "cat_feature2"],
    },
    "split": {"method": "kfold", "n_splits": 5, "random_state": 1120},
    "model": {
        "lgbm": {
            "params": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
            },
            # スマートパラメーター（§5.3 参照）
            "auto_num_leaves": True,       # max_depth から num_leaves を自動算出
            "num_leaves_ratio": 0.8,       # 基準値に対する割合
            "min_data_in_leaf_ratio": 0.01, # 学習データ行数に対する割合
            "min_data_in_bin_ratio": 0.01,  # 学習データ行数に対する割合
            # "feature_weights": {"important_feat": 2.0},  # 特徴量重み辞書
            # "balanced": None,            # None=タスク依存自動（regression→False, 分類→True）
        }
    },
    "training": {
        "early_stopping": {
            "enabled": True,
            "validation_ratio": 0.1,  # inner_valid.ratio のエイリアス
            # inner_valid 未指定時は外側 split.method に応じて自動解決
            # 明示指定例:
            # "inner_valid": {"method": "holdout", "ratio": 0.1, "stratify": True}
            # "inner_valid": {"method": "group_holdout", "ratio": 0.1}
            # "inner_valid": {"method": "time_holdout", "ratio": 0.1}
        }
    },
    "tuning": {
        "optuna": {
            "params": {
                "n_trials": 50,
                "direction": "minimize",
            },
            # space が空 or 未指定の場合はタスク別デフォルト空間を自動適用（§11.3 参照）
            "space": {},
        }
    },
    "evaluation": {"metrics": ["rmse", "mae"]},
}
```

## 5.3 LGBMConfig 拡張パラメーター

`LGBMConfig` に以下のスマートパラメーターを提供する。これらは `fit()` 時に学習データに基づいて LightGBM ネイティブパラメーターに解決される。`params` の直接指定とは独立して機能し、`params` で同一パラメーターが指定されている場合は競合エラーとする。

### auto_num_leaves（葉の数の自動算出）

- `auto_num_leaves: bool = True`: 有効時、`max_depth` から `num_leaves` を自動算出する。
- `num_leaves_ratio: float = 1.0`（`0 < ratio ≤ 1`）: 基準値に対する割合。
- 算出ロジック:
  - `params.max_depth` が未指定または負値（制限なし）→ 基準値 = `131072`
  - `params.max_depth` が指定されている → 基準値 = `2 ^ max_depth`
  - `num_leaves = clamp(ceil(基準値 × num_leaves_ratio), 8, 131072)`
- 制約: `auto_num_leaves=True` 時に `params.num_leaves` の直接指定は `CONFIG_INVALID`。

### データサイズ相対比率パラメーター

学習データの行数に対する割合で指定し、CV の各 fold 内で inner validation 分割後の実学習データ行数（`n_rows_inner_train`）を基準に絶対値に変換する（H-0036）。

- `min_data_in_leaf_ratio: float | None = 0.01`（`0 < ratio < 1`）→ `min_data_in_leaf = max(1, ceil(n_rows_inner_train × ratio))`
- `min_data_in_bin_ratio: float | None = 0.01`（`0 < ratio < 1`）→ `min_data_in_bin = max(1, ceil(n_rows_inner_train × ratio))`
- `n_rows_inner_train` の定義: outer fold の学習データから inner validation（early stopping 用）を分割した後の行数。early stopping が無効（inner validation 分割なし）の場合は outer fold の学習データ行数を使用する。
- fold ごとに `n_rows_inner_train` が異なる場合、各 fold で個別に解決する。
- 制約: ratio 指定と対応する絶対値パラメーター（`params.min_data_in_leaf` 等）の同時指定は `CONFIG_INVALID`。

### feature_weights（特徴量重みの辞書指定）

- `feature_weights: dict[str, float] | None`: 特徴量名をキーとした重み辞書。
- 未指定特徴量は `1.0` で自動補完される。
- 学習データの特徴量順に並び替えたリストに変換し、LightGBM に渡す。
- 副作用: `feature_pre_filter = False` を強制する。
- 制約: 重み `> 0` 必須。学習データに存在しない未知の特徴量名は `CONFIG_INVALID`。

### balanced（クラス重み自動均衡化）

- `balanced: bool | None = None`: 学習データのクラス比率から自動的に重みを算出する。
  - `None`（デフォルト）: タスク依存で自動解決（regression→`False`, binary/multiclass→`True`）。
  - `True`: binary は `scale_pos_weight = neg_count / pos_count` を設定。multiclass は `sample_weight` でクラス逆頻度重み付け。
  - `False`: 重み均衡化を無効にする。
  - regression で `True` を指定した場合は `UNSUPPORTED_TASK`。

## 5.4 Config Reference（全キー一覧）

`config_version=1` で利用可能な全 Config キーの型・デフォルト・制約を以下にまとめる。

### トップレベル

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `config_version` | `int` | Yes | - | `1` のみサポート |
| `task` | `"regression" \| "binary" \| "multiclass"` | Yes | - | |
| `data` | `object` | Yes | - | |
| `features` | `object` | No | `{}` | |
| `split` | `object` | No | タスク依存 | binary/multiclass→stratified_kfold, regression→kfold |
| `model` | `object` | Yes | - | LightGBM のみ |
| `training` | `object` | No | `{}` | seed=42, early stopping 有効 |
| `tuning` | `object \| null` | No | `null` | `tune()` 呼び出し時のみ必要 |
| `evaluation` | `object` | No | `{}` | |
| `calibration` | `object \| null` | No | `null` | binary 専用 |

### data

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `path` | `str \| null` | No | `null` | CSV/Parquet パス |
| `target` | `str` | Yes | - | 目的変数列名 |
| `time_col` | `str \| null` | No | `null` | 時系列列名（`time_series` / `purged_time_series` / `group_time_series` では必須） |
| `group_col` | `str \| null` | No | `null` | グループ列名 |

### features

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `exclude` | `list[str]` | No | `[]` | 除外列 |
| `auto_categorical` | `bool` | No | `True` | 自動カテゴリ検出 |
| `categorical` | `list[str]` | No | `[]` | 明示カテゴリ指定 |

### split

`split.method` は以下のいずれか: `kfold` / `stratified_kfold` / `group_kfold` / `time_series` / `purged_time_series` / `group_time_series`。

| method | 固有キー |
|---|---|
| `kfold` | `n_splits=5`, `random_state=42`, `shuffle=True` |
| `stratified_kfold` | `n_splits=5`, `random_state=42` |
| `group_kfold` | `n_splits=5` |
| `time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |
| `purged_time_series` | `n_splits=5`, `purge_gap=0`, `embargo=0`, `train_size_max=null`, `test_size_max=null` |
| `group_time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |

注記:
- `time_series` / `purged_time_series` / `group_time_series` は共通で `data.time_col` 必須。
- 3 メソッドは共通で `train_size_max` / `test_size_max` を受け取り、学習窓・検証窓の上限を制御する。
- `purged_time_series` の旧キー `embargo_pct` は移行期間のみ後方互換として扱い、`embargo` に正規化する。

### model（LightGBM）

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `params` | `dict[str, Any]` | No | `{}` | LightGBM パラメーター |
| `auto_num_leaves` | `bool` | No | `True` | §5.3 参照 |
| `num_leaves_ratio` | `float` | No | `1.0` | `0 < ratio ≤ 1` |
| `min_data_in_leaf_ratio` | `float \| null` | No | `0.01` | `0 < ratio < 1` |
| `min_data_in_bin_ratio` | `float \| null` | No | `0.01` | `0 < ratio < 1` |
| `feature_weights` | `dict[str, float] \| null` | No | `null` | 重み > 0 必須 |
| `balanced` | `bool \| null` | No | `null` | `null`=タスク依存自動（regression→false, binary/multiclass→true）。分類専用。 |

### training

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `seed` | `int` | No | `42` | グローバルシード |
| `early_stopping.enabled` | `bool` | No | `True` | |
| `early_stopping.rounds` | `int` | No | `150` | |
| `early_stopping.validation_ratio` | `float \| null` | No | `0.1` | inner_valid.ratio のエイリアス |
| `early_stopping.inner_valid` | `object \| null` | No | `null`（自動解決） | |

### tuning

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `optuna.params.n_trials` | `int` | No | `50` | |
| `optuna.params.direction` | `"minimize" \| "maximize"` | No | `"minimize"` | |
| `optuna.params.timeout` | `float \| null` | No | `null` | |
| `optuna.space` | `dict[str, Any]` | No | `{}` | 空ならデフォルト空間 |

### evaluation

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `metrics` | `list[str]` | No | `[]` | ランタイムデフォルトあり |

### calibration

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `method` | `"platt" \| "isotonic" \| "beta"` | No | `"platt"` | |
| `n_splits` | `int` | No | `5` | cross-fit fold 数 |

# 6. 実行フロー（概念）

## 6.1 `tune`

1. Config validate -> `ProblemSpec` 生成
2. `DataSource` 読込 -> DF
3. `FeaturePipeline` 選択 -> `fit_transform`
4. `Splitter` で外側 CV index 生成（保存対象）
5. `Tuner`（Optuna）で `objective=CV` 平均（推奨）
6. `TuningResult`（`best_params` / `best_score` / 全 trial 履歴）を返す

`tuning` と最終評価のリーク回避方針は 10 章を参照。

## 6.2 `fit`（CV）

1. 外側 CV 各 fold で `train / valid` を作る。
   - `split.method` が `time_series` / `purged_time_series` / `group_time_series` の場合、`data.time_col` を基準に昇順へ並べた上で分割する。
2. `InnerValidStrategy` により early stopping 用の `(X_train, y_train), (X_valid, y_valid)` を統一生成する。
3. `EstimatorAdapter.fit()` を実行する。
4. OOF / IF を生成する（ロジックは `evaluation/oof.py` に隔離）。
5. 必要なら `Calibrator` を cross-fit 学習する（OOF 予測のみ使用）。
6. `FitResult` を返し、Artifacts を保持する。

補足:

- `fit()` は default で最良パラメーターを使用する。
- 他のパラメーターセットを指定して学習できる拡張点も残す。

## 6.3 `evaluate`

`FitResult` を入力に、指定メトリクスで以下を返す。

- `oof`
- `if_mean`
- `if_per_fold`
- 校正前後（binary）を同一集合で比較

## 6.4 `predict`

1. 入力 DF の列を schema と照合する（列ズレ検知）。
2. `FeaturePipeline.transform` を適用する（状態は Artifacts）。
3. fold アンサンブル or refit モデルで予測する。
4. 校正器を適用する（binary）。
5. `PredictionResult` を返す（要求時のみ SHAP など付与）。

## 6.5 `export`

- `Model Artifact` を `export_dir` に保存する。
- `FeaturePipeline state / schema / models / calibrator / metrics / history / config / versions / format_version` を含める。
- load 後診断 API 用に `analysis_context`（`y_true`, `X_for_explain`）を含める。
- `Model.load()` で復元可能にし、復元後に予測と評価情報参照の両方を行えるようにする。

# 7. Artifacts（戻り値と保存対象の固定）

## 7.1 FitResult（固定スキーマ）

- `oof_pred`（`np.ndarray / pd.Series`）
- `if_pred_per_fold`（`list[np.ndarray]`）
- `metrics`（階層固定）
  - 例: `{"raw": {"oof": {...}, "if_mean": {...}, "if_per_fold": [...]}, "calibrated": {...}}`
- `models`
  - fold ごとのモデル
  - 任意: refit モデル（全データ学習）
- `history`
  - fold ごとの eval history / best_iteration
- `feature_names / dtypes / categorical_features`
- `splits`
  - 外側 CV indices（必須）
  - inner valid indices（有効時必須）
  - calibration CV indices（有効時必須）
  - `time_range`（時系列分割時。fold ごとの train/valid の期間情報 `list[dict] | None`）
- `data_fingerprint`
  - `row_count / column_hash / optional: file_hash` 等
- `pipeline_state`（`FeaturePipeline` の状態、必須）
- `calibrator`（有効時）
- `run_meta`
  - `yourlib_version / python_version / deps_version / config_normalized / config_version`

## 7.2 TuningResult（固定スキーマ）

- `best_params`（`dict[str, Any]`）: 最良のハイパーパラメーター
- `best_score`（`float`）: 最良の OOF メトリクス値
- `trials`（`list[TrialResult]`）: 全 trial の結果（番号順）
  - `TrialResult`: `number` / `params` / `score` / `state`（`"complete"` / `"pruned"` / `"fail"`）
- `metric_name`（`str`）: 最適化メトリクス名
- `direction`（`str`）: `"minimize"` / `"maximize"`

## 7.3 PredictionResult（固定スキーマ）

- `pred`（回帰: float、分類: class / proba）
- `proba`（binary の場合）
- `shap_values`（要求時のみ、形は統一）
- `used_features`（列ズレ検知用）
- `warnings`（補正が走った場合の通知）

補足:

- 回帰では `pred` を主とする。
- 分類では `pred` に加えて `proba` を返せるようにする。

## 7.4 Exported Model Artifacts

- `FeaturePipeline state`
- `schema`（`feature_names, dtypes, categorical handling`）
- `model`（fold ensemble / refit）
- `calibrator`（`C_final`）
- `metrics / history / fit summary`
- `analysis_context`（`y_true`, `X_for_explain`。load 後に診断 API を実行するための最小データ）
- `config_normalized`
- `format_version / versions`

目的:

- `Model.load(path)` で復元し、予測だけでなく「そのモデルの精度がどうだったか」および残差/SHAP/分類・校正可視化を後から確認できるようにする。

# 8. データと検証（`data/`）

## 8.1 DataSource

- `CSV / Parquet / DataFrame` を「読むだけ」に限定する。
- 入口で `DataFrameBuilder` が `target / time / group` を分離する。

## 8.2 Validators（危険検知）

- 時系列: ソート、未来情報混入疑い、shuffle 禁止
- group: group 跨ぎ、分割条件の不整合
- leakage: target リーク疑い（例: target と完全一致の列、時間逆転など）

## 8.3 Data fingerprint（必須）

- `row_count`
- `column_hash`（列名 + dtype + 順序から作る）
- optional: `file_hash`（読み込んだファイルのハッシュ）

# 9. FeaturePipeline（`features/`）

## 9.1 必須要件

- `fit(X, y) / transform(X) / fit_transform(X, y)` の IF を固定する。
- 状態（state）の永続化を必須にする。
- OneHot のカテゴリ辞書、欠損補完統計量、target transform パラメータ等を保持する。

## 9.2 列ズレ方針（仕様として固定）

- 余剰列: デフォルト無視（警告） or エラー（オプション）
- 不足列: デフォルトエラー（安全側）
- unseen category:
  - OneHot: unknown 用カテゴリ or all-zero（ポリシー選択）
  - LGBM native categorical: 扱いを固定（未知カテゴリの扱い・dtype 強制）

# 10. Split（`splitters/`）と InnerValidStrategy（`training/`）

## 10.1 Splitter の責務

- 「index を返すだけ」に徹底する。
- 外側 CV / early stopping / calibration で共通利用する。

## 10.2 Outer CV（例）

- `KFold`
- `StratifiedKFold`（binary/multiclass のデフォルト）
- `GroupKFold`
- `TimeSeriesSplit`
- `PurgedTimeSeries`
- `GroupTimeSeries`

注記:
- `task` が `binary` または `multiclass` かつ `split.method` が未指定の場合、`StratifiedKFold` をデフォルトとする。分類タスクで `method: "kfold"` を明示指定した場合は警告を出す。回帰タスクのデフォルトは `KFold` のまま。
- `time_series` / `purged_time_series` / `group_time_series` は共通で `data.time_col` を基準に昇順へ並べてから分割する。
- `time_series` / `group_time_series` は `gap`、`purged_time_series` は `purge_gap` を持つ（いずれも train と valid の間のギャップ）。
- `PurgedTimeSeries` は `embargo`（train と valid の間に設ける追加除外 Obs 数、`int`、`gap` / `purge_gap` と同じ単位）を持つ。`embargo_pct` は移行期間のみ後方互換キーとする（`int()` で変換）。
- 3 メソッドは共通で `train_size_max` / `test_size_max` を持つ。
- `GroupTimeSeries` は group 列の出現順と `time_col` 順を整合させて時系列的にグループを分割する。

## 10.3 InnerValidStrategy（early stopping 用）

- CV fold 内でさらに `train / valid` を作る概念を分離する。
- 実装（holdout ベース。将来 `InnerKFoldValid` も拡張可能とする）:
  - `HoldoutInnerValid(ratio, stratify=False, random_state)`: ランダム holdout。`stratify=True` で `y` に基づく層化抽出。
  - `GroupHoldoutInnerValid(ratio, random_state)`: group 単位の holdout。group overlap を禁止する。
  - `TimeHoldoutInnerValid(ratio)`: 時系列順を維持し、末尾 ratio 割合を validation に割り当てる（shuffle 禁止）。
- 時系列は内側も時系列順を厳守する（shuffle 禁止）。
- Config で `inner_valid.method` を指定する。`inner_valid` 未指定かつ `early_stopping.enabled=True` の場合、外側 CV の method に応じて自動解決する。

| 外側 split.method | inner_valid のデフォルト |
|---|---|
| `stratified_kfold` | `holdout(stratify=True)` |
| `group_kfold` | `group_holdout` |
| `time_series` | `time_holdout` |
| `purged_time_series` | `time_holdout` |
| `group_time_series` | `group_holdout` |
| `kfold`（または CV 未使用） | `holdout(stratify=False)` |

## 10.4 split indices の保存（必須）

- 外側 CV: fold ごとの `train_idx / valid_idx`
- inner valid: fold 内の `inner_train_idx / inner_valid_idx`
- calibration CV: 校正用の `train_idx / valid_idx`

# 11. Tuning（`tuning/`）

## 11.1 SearchSpace 表現の統一

- Optuna に依存しない space 表現（離散・連続・対数・カテゴリ）を使う。

## 11.2 SearchDim カテゴリ

SearchDim にカテゴリ属性を持たせ、Tuner がパラメーターの適用先を区別する。

- `model`: `LGBMAdapter.params` に直接渡す（既存 SearchDim の挙動）
- `smart`: `LGBMConfig` のスマートパラメーター（`num_leaves_ratio` 等）として `resolve_smart_params()` に渡す
- `training`: trial ごとに `EarlyStoppingConfig` / `InnerValidStrategy` を再構築する

## 11.3 デフォルト Tuning Space

`tuning.optuna.space` が空（`{}`）の場合、タスク別のデフォルト探索空間を自動適用する。ユーザーが `space` を指定した場合はユーザー指定を使用する。

### 探索次元

| パラメーター | 型 | 範囲 | カテゴリ |
|---|---|---|---|
| `objective` | categorical | regression: `[huber, fair]`, binary: `[binary]`, multiclass: `[multiclass, multiclassova]` | model |
| `n_estimators` | int | `[600, 2500]` | model |
| `learning_rate` | float (log) | `[0.0001, 0.1]` | model |
| `max_depth` | int | `[3, 12]` | model |
| `feature_fraction` | float | `[0.5, 1.0]` | model |
| `bagging_fraction` | float | `[0.5, 1.0]` | model |
| `num_leaves_ratio` | float | `[0.5, 1.0]` | smart |
| `min_data_in_leaf_ratio` | float | `[0.01, 0.2]` | smart |
| `early_stopping_rounds` | int | `[40, 240]` | training |
| `validation_ratio` | float | `[0.1, 0.3]` | training |

### 固定パラメーター（探索しない）

| パラメーター | 値 |
|---|---|
| `auto_num_leaves` | `True` |
| `first_metric_only` | `True` |
| `metric` | regression: `[huber, mae, mape]`, binary: `[auc, binary_logloss]`, multiclass: `[auc_mu, multi_logloss]` |

注記:
- `brier` / `precision_at_k` は LightGBM ネイティブ未対応のため除外。
- Binary の `objective` は `[binary]` のみ（選択肢 1 つで実質固定）。

## 11.4 リーク回避方針（必須で明文化）

- 最適化に使った CV で最終性能を主張しない。

推奨パターン（選択式）:

1. `holdout`（固定検証セット）で最終評価
2. `nested CV`（外側評価、内側最適化）
3. `CV + 追加のテストセット`（OOF は参考値、テストを主指標）

デフォルトは 1 または 3 を推奨する（実装コストを抑えつつ安全側）。

# 12. Calibration（binary）

## 12.1 MUST（リーク禁止）

- 校正器学習は、必ず Base モデルの OOF 生スコア（raw score / logits。sigmoid/softmax 適用前）のみを使う。
- `EstimatorAdapter.predict_raw(X)` で生スコアを取得する（§14.1 参照）。
- 校正性能評価は、校正器も OOF（cross-fit）で生成した値で行う。
- 校正器は元の特徴量 `X` を使わない（入力は `s_oof`（生スコア）と `y` のみ）。
- 推論時は保存された `C_final` を使用する。
- Calibration が未指定の場合は従来どおり `predict_proba`（確率値）を OOF/IF 予測に使用する。Calibration 有効時のみ生スコアベースの校正パスに入る。

## 12.2 方法

- Platt Scaling
- Beta Calibration
- Isotonic Regression（LGBM の単調制約利用）

## 12.3 評価（推奨）

- `LogLoss`（必須推奨）
- `Brier score`（必須推奨）
- `ECE`（binning 定義を仕様化）
- `ROC-AUC / PR-AUC`（ランキング監視）

## 12.4 MUST NOT

- 同一行を学習に含む予測で校正器学習する（リーク）。
- `C_final` で `s_oof` を変換した値を評価に使う（楽観評価）。
- 校正器が `X` を利用する。

# 13. Metrics / Evaluation / Plots

## 13.1 Metrics

- Metric IF
- `needs_proba / greater_is_better / supports_task`
- 回帰: `rmse / mae / r2 ...`
- 分類（binary）: `logloss / auc / auc_pr / f1 / accuracy / brier / ece / precision_at_k ...`
- 分類（multiclass）: `logloss / auc(OvR) / auc_pr(OvR) / f1(macro) / accuracy / brier(OvR) ...`
- multiclass の `auc / auc_pr / brier` は One-vs-Rest 展開 + macro 平均で計算する。メトリクス名は binary と共通（`__call__` 内で `y_pred.ndim` により分岐）。

## 13.2 評価出力（固定）

- IF / OOF と fold 別を必ず返す。
- 校正前後も同一フォーマットで返す（binary）。
- `evaluate_table()` は `evaluate()` が返す固定構造 dict を `pd.DataFrame` に変換する純粋フォーマッタ。ロジックは `evaluation/table_formatter.py` に配置する。
  - 行 = メトリクス名、列 = `if_mean`, `oof`, `fold_0`...`fold_N-1`。
  - calibrated がある場合は `cal_oof` 列を追加。

## 13.3 可視化

全プロットを Plotly ベースに統一する。Plotly は optional dependency（`pip install 'lizyml[plots]'`）。未インストール時は `OPTIONAL_DEP_MISSING` を返す。

実装済み:
- `importance_plot(kind="split|gain")`: fold 平均の特徴量重要度（横棒グラフ）
- `importance_plot(kind="shap")`: fold 平均の mean(|SHAP|)（横棒グラフ）。shap optional dependency も必要。
- `plot_learning_curve()`: fold ごとの train/valid loss 推移（折れ線グラフ）
- `plot_oof_distribution()`: OOF 予測値の分布（ヒストグラム）
- `residuals_plot(kind="scatter|histogram|qq|all")`: 回帰専用。IS/OOS 比較対応。`kind` で表示プロットを選択。デフォルト `kind="all"` で scatter + histogram + QQ の 3 パネル。scatter は Actual vs Predicted（x=predicted, y=actual, y=x 参照線）。IS サンプルは OOS 数に合わせてダウンサンプリング（`_downsample_is()`、seed=0 で再現可能）。

追加で用意したい可視化（一部実装済み）:
- binary/multiclass: `roc_curve_plot()`（binary: IS/OOS の 2 本の ROC Curve 重ね描き。multiclass: IS/OOS を subplot 横並びにし、クラスごとの OvR ROC Curve を描画。各クラスの AUC 値を凡例に表示、macro 平均 AUC も表示）
- binary/multiclass: `confusion_matrix(threshold=0.5)`（IS/OOS の Confusion Matrix テーブル。`{"is": DataFrame, "oos": DataFrame}` を返す。binary は threshold、multiclass は argmax でクラスラベル変換）
- calibration: `calibration_plot()`（Raw/Calibrated の Reliability Diagram。bin 数デフォルト 10。理想線 y=x を参照線として描画。データソースは cross-fit 由来の `calibrated_oof`、`c_final` は使用しない）
- calibration: `probability_histogram_plot()`（Raw/Calibrated の確率分布ヒストグラム重ね描き。校正前後の分布シフトを視覚的に確認）
- tuning: `tuning_plot()`（trial ごとのスコア推移。X 軸 = trial 番号、Y 軸 = スコア。完了/枝刈り/失敗を色分け。最良スコア推移ラインを重ね描き）
- 時系列: `split_summary()`（fold ごとの分割サイズ。時系列分割時は `train_start / train_end / valid_start / valid_end` の期間情報を含む `pd.DataFrame` を返す）
- 未実装: `PR Curve / threshold最適化レポート`

# 14. Estimators（`estimators/`）

## 14.1 EstimatorAdapter IF

```python
fit(X_train, y_train, X_valid=None, y_valid=None, **kwargs)
predict(X)
predict_proba(X)  # 分類（sigmoid/softmax 適用後の確率値）
predict_raw(X)    # 分類（sigmoid/softmax 適用前の生スコア / logits。Calibration 用）
importance(kind="split|gain|shap")
get_native_model()  # export用途
```

## 14.2 LGBM adapter の責務

- `objective / metric` 整合
- categorical の扱い統一
- early stopping の設定吸収
- SHAP（内蔵寄り）対応

### 14.2.1 Booster API の使用（H-0041）

`LGBMAdapter` は LightGBM の **Booster API**（`lgb.train()`）を使用する。sklearn wrapper（`LGBMRegressor` / `LGBMClassifier`）は使用しない。

理由:
- sklearn wrapper 内部の `model_to_string()` → `model_from_string()` ラウンドトリップに起因する間欠バグ（microsoft/LightGBM#7186）を回避する。
- Booster API は `keep_training_booster=True` により上記ラウンドトリップを回避でき、直接的な制御が可能。

制約:
- `fit()` は `lgb.Dataset` を構築し、`lgb.train()` で学習する。
- `predict()` / `predict_proba()` / `predict_raw()` は `Booster.predict()` を使用する。
  - `predict_proba()` の shape 契約（binary: `(n, 2)`, multiclass: `(n, k)`）は維持する。
- `get_native_model()` は `lgb.Booster` を返す。
- パラメーター名は Booster API の名前空間に準拠する（`n_estimators` → `num_boost_round` 引数、`random_state` → `seed` パラメーター等の変換を adapter 内で吸収する）。
- 学習履歴は `evals_result` dict から取得する（sklearn の `evals_result_` 属性ではない）。

## 14.3 LightGBM デフォルトパラメータープロファイル

`LGBMAdapter` はタスク別のデフォルトパラメーターを提供する。`LGBMConfig.params` で明示指定した値はデフォルトを上書きする。

### タスク別デフォルト

| | regression | binary | multiclass |
|---|---|---|---|
| objective | `huber` | `binary` | `multiclass` |
| metric | `[huber, mae, mape]` | `[auc, binary_logloss]` | `[auc_mu, multi_logloss]` |

注記:
- regression の objective を `huber` とする（外れ値に対してロバスト）。
- `brier` / `precision_at_k` は LightGBM ネイティブ未対応。将来のカスタム feval 拡張点とする。

### 共通デフォルト

| パラメーター | デフォルト値 | 備考 |
|---|---|---|
| `boosting` | `gbdt` | |
| `first_metric_only` | `False` | |
| `num_boost_round` | `1500` | `lgb.train()` の引数として渡す |
| `learning_rate` | `0.001` | 低学習率で early stopping に依存 |
| `max_depth` | `5` | |
| `max_bin` | `511` | |
| `feature_fraction` | `0.7` | |
| `bagging_fraction` | `0.7` | |
| `bagging_freq` | `10` | |
| `lambda_l1` | `0.0` | |
| `lambda_l2` | `0.000001` | |

### Training デフォルト

| パラメーター | デフォルト値 |
|---|---|
| `early_stopping.enabled` | `True` |
| `early_stopping.rounds` | `150` |
| `early_stopping.validation_ratio` | `0.1` |

# 15. Persistence / Export（`persistence/`）

## 15.1 保存の基本方針

- `format_version` を必須にする。
- 保存対象:
  - `yourlib_version`
  - `python_version`
  - 依存 versions（`lgbm / sklearn / optuna ...`）
  - `config_normalized`
  - `schema`（`feature_names / dtypes / categorical policy`）
  - split indices
  - `data_fingerprint`
  - `pipeline_state`
  - `models, calibrator`

## 15.2 互換性ポリシー（必須）

- `format_version` が読めない場合は明示的に拒否する（黙って壊れた復元をしない）。
- 将来 migration を実装できる前提で serializer に拡張点を残す。

## 15.3 `export`（`Model Artifact`）

- `Model Artifact` を 1 ディレクトリにまとめる。
- `Model.load()` で復元し、推論と評価情報参照に加えて診断 API（残差/SHAP/分類・校正可視化）も利用可能にする。

## 15.4 パッケージ配布（PyPI）

- `pyproject.toml` に `PEP 517/518` 準拠の `[build-system]` を必須で定義し、`sdist / wheel` を同一ソースから生成できるようにする。
- `[project]` メタデータは最低限以下を必須とする。
  - `name / version / description / readme / requires-python`
  - `license`
  - `authors or maintainers`
  - `classifiers`
  - `urls`（少なくとも `Homepage` と `Repository`）
- `README.md` は PyPI の long description として成立する内容にし、公開済みでない API や未実装の import 例を載せない。
- `README.md` のサンプルコードは「インストール直後に動く import」を基準にし、トップレベル公開面（`package/__init__.py`）と必ず一致させる。
- optional dependency は配布利用者向けの install 契約と、開発者向けの依存を分離する。
  - 配布利用者向け: `[project.optional-dependencies]`
  - 開発者向け: dependency groups
- 型ヒントを配布対象に含める場合は `py.typed` を同梱し、配布物と型情報の不整合を禁止する。
- バージョン定義の正を 1 箇所に固定し、配布メタデータと import 後に参照できるバージョン文字列を乖離させない。

# 16. 例外設計（`core/exceptions.py`）

## 16.1 統一例外

```python
YourLibError(code, user_message, debug_message=None, cause=None)
```

## 16.2 例外コード（例）

- `CONFIG_INVALID`
- `CONFIG_VERSION_UNSUPPORTED`
- `DATA_SCHEMA_INVALID`
- `DATA_FINGERPRINT_MISMATCH`
- `LEAKAGE_SUSPECTED`
- `LEAKAGE_CONFIRMED`
- `OPTIONAL_DEP_MISSING`
- `MODEL_NOT_FIT`
- `INCOMPATIBLE_COLUMNS`
- `UNSUPPORTED_TASK`
- `UNSUPPORTED_METRIC`
- `METRIC_REQUIRES_PROBA`
- `TUNING_FAILED`
- `CALIBRATION_NOT_SUPPORTED`
- `SERIALIZATION_FAILED`
- `DESERIALIZATION_FAILED`

# 17. Logging / Run 管理（`core/logging.py`）

- `run_id` を生成し、出力先（`logs / artifacts / plots`）を統一する。
- 重要イベントを構造化ログで出す（config hash, data fingerprint, split hash 等）。
- エラー時は `code` を必ずログに残す。
- `output_dir` オプション（Config or コンストラクタ引数）指定時、`{output_dir}/{run_id}/` にログ・plot 保存先を統一する。
- `output_dir` 未指定時は現行動作（ログは標準出力、plot は返却のみ）を維持する。

# 18. テスト / CI（必須）

## 18.1 テスト戦略

- Golden test: `FitResult / PredictionResult` スキーマ固定を検証する。
- 再現性テスト: 同一 config で split indices と主要指標が一致する。
- 列ズレテスト: 余剰 / 不足 / unseen category のポリシー通り動く。
- optional dependency テスト: 未導入時の例外コード / メッセージが崩れない。
- Public API surface テスト: `from lizyml import Model` 等のトップレベル公開面が壊れていないことを検証する。
- バージョン一致テスト: `lizyml.__version__` と配布メタデータのバージョンが一致することを検証する。
- README サンプルコードテスト: `README.md` に記載された最短利用例が `SyntaxError` / `ImportError` なく実行可能であることを検証する（データ依存部分はモック可）。

## 18.2 CI（推奨）

- type check（`mypy / pyright`）
- lint / format（`ruff` 等）
- unit tests（`pytest`）
- 最低限の統合テスト（LGBM 小規模データ）
- 配布前検証として `sdist / wheel` の build を CI で必ず実行する。
- 配布メタデータ検証（例: `twine check` 相当）を CI に含める。
- install smoke test を行い、配布物からの import と README の最短利用例が破綻していないことを確認する。
- 複数 Python バージョン（最低限 `requires-python` の下限と最新安定版）でテストを実行する。
- 依存の下限バージョンでのテストを CI に含める（`uv` の resolution 機能で `lowest-direct` を使用）。

# 19. ディレクトリ構成（更新案）

```text
LizyML/
  pyproject.toml
  README.md
  BLUEPRINT.md
  HISTORY.md
  LICENSE
  LizyML/
    __init__.py

    config/
      loader.py              # YAML/JSON/dict、override、正規化、version管理
      schema.py              # pydantic schema（extra=forbid）

    calibration/
      base.py
      platt.py
      isotonic.py
      beta_calibration.py

    core/
      model.py               # Facade（組み立てと Artifact load）
      types.py               # 型の再export / 集約点（薄く保つ）
      registries.py
      exceptions.py
      logging.py
      seed.py
      types/
        fit_result.py
        predict_result.py
        artifacts.py         # FitArtifacts / PredictArtifacts 等
      specs/
        problem_spec.py
        feature_spec.py
        split_spec.py
        training_spec.py
        tuning_spec.py
        calibration_spec.py
        export_spec.py       # exportの方針（形式/互換性など）

    data/
      datasource.py
      dataframe_builder.py
      validators.py
      fingerprint.py         # data fingerprint 算出

    features/
      pipeline_base.py
      pipelines_native.py
      pipelines_sklearn.py
      pipelines_dnn.py
      encoders/
        categorical_encoder.py  # 必要時のカテゴリ処理部品
      transformers/
        target_transformer.py
        feature_transformer.py

    splitters/
      base.py
      kfold.py
      group_kfold.py
      time_series.py
      purged_time_series.py
      group_time_series.py

    estimators/
      base.py
      lgbm.py
      sklearn.py
      dnn_base.py
      dnn_torch.py

    training/
      cv_trainer.py
      refit_trainer.py
      tuning_trainer.py
      inner_valid.py         # InnerValidStrategy 群

    tuning/
      base.py
      search_space.py
      optuna_tuner.py

    metrics/
      base.py
      regression.py
      classification.py
      registry.py

    evaluation/
      oof.py
      evaluator.py
      thresholding.py        # binary閾値最適化（任意）

    explain/
      base.py
      shap.py
      lgbm_contrib.py
      integrated_gradients.py

    plots/
      learning_curve.py
      importance.py
      residuals.py
      calibration.py         # reliability diagram 等
      classification.py      # ROC/PR/confusion 等

    persistence/
      serializer.py
      model_store.py

    utils/
      import_optional.py
      array.py
      pandas.py
      time.py
```

# 20. 既知の将来拡張（設計で塞がない）

- multi-class calibration（別仕様）
- ranking タスク（`objective / metric` の拡張）
- `export` の追加形式（`Booster text / ONNX / TorchScript` 等）
- 大規模データ（out-of-core、カテゴリ辞書の扱い）

# 付録 A: ユースケース（例）

```python
# Config設定
config = {
    "config_version": 1,
    "task": "regression",
    "data": {"path": "data.csv", "target": "y"},
    "split": {"method": "kfold", "n_splits": 5, "random_state": 1120},
    "model": {
        "lgbm": {
            "params": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
            }
        }
    },
    "tuning": {
        "optuna": {
            "params": {
                "n_trials": 50,
                "direction": "minimize",
            },
            # space 未指定でデフォルト空間を自動適用（§11.3 参照）
            "space": {},
        }
    },
    "evaluation": {"metrics": ["rmse", "mae"]},
}

model = Model(config=config)

tuning_result = model.tune()
model.tuning_table()  # 全 trial の DataFrame 表示
fit_result = model.fit()

importance = model.importance()
model.plot_learning_curve()
model.importance_plot(kind="shap")

eval_result = model.evaluate(metrics=["rmse", "mae"])

residuals = model.residuals()
preds = model.predict(X_test)
preds_shap = model.predict(X_test, return_shap=True)

model.export("export_dir")
loaded_model = Model.load("export_dir")
loaded_model.evaluate()
loaded_model.predict(X_new)
loaded_model.residuals()
loaded_model.residuals_plot()
loaded_model.importance(kind="shap")
loaded_model.roc_curve_plot()
loaded_model.confusion_matrix()
loaded_model.calibration_plot()
loaded_model.probability_histogram_plot()
```

# 付録 B: Facade の責務補足

## Model が担うこと

- Config を validate して `ProblemSpec` に変換する。
- `DataSource` から DF を読む。
- `FeaturePipeline / Splitter / EstimatorAdapter / Tuner / Calibrator` を registry 経由で選ぶ。
- `Trainer`（または `CVRunner`）へ処理を渡して実行する。
- 得られた `FitResult / Artifacts` を保持する。
- 保存済み `Model Artifact` を `Model.load(path)` で復元する。

## Model に置かないこと

- OOF / IF 生成ロジック（`evaluation/oof.py`）
- metric 計算（`evaluation/evaluator.py`）
- LGBM 固有処理（`estimators/lgbm.py`）
- plot 実装本体（`plots/*`）
- 保存形式の詳細（`persistence/*`）

## 実装メモ

- `core/model.py` は組み立て専用とし、ロジックを持たせない。
- 依存関係の切り離しが必要な箇所では Lazy Import を許容する。
