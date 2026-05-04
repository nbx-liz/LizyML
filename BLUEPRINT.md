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

## 2.1 5 層カテゴリアーキテクチャ（H-0051/H-0052/H-0053）

モジュール間の依存を 5 層の DAG（非巡回有向グラフ）で管理する。詳細は `ARCHITECTURE.md` を参照。

| Layer | 名称 | 依存先 | 含まれるカテゴリ |
|---|---|---|---|
| 0 | Foundation | なし | `core/exceptions`, `core/logging`, `core/types/` |
| 1 | Leaf | Foundation のみ | `config/`, `data/`, `splitters/`, `features/`, `estimators/`, `metrics/`, `calibration/` |
| 2 | Composition | Foundation + Layer 1 の IF | `training/`, `evaluation/`, `tuning/` |
| 3 | Optional | Foundation + Layer 1/2 の IF | `explain/`, `plots/`, `persistence/` |
| 4 | Facade | 全 Layer | `core/model.py`, `core/_model_*.py`, `core/_model_factories.py` |

依存ルール:
- 各カテゴリは自分より**上の Layer にのみ**依存する（下方向のみ）。
- Layer 2 は Layer 1 の**抽象 IF のみ**を参照する（具象クラスを import しない）。
- 具象クラスの組み立て・型ディスパッチは **Layer 4（Facade）のみ**が行う。
- カテゴリ間の**循環依存は禁止**する。

## 2.2 EstimatorProvider（マルチアルゴリズム拡張 IF）（H-0053）

新しいアルゴリズムの追加を `model.py` 変更ゼロで行えるようにするため、各 estimator モジュールが `EstimatorProvider` protocol を実装する。

`EstimatorProvider` が提供するもの:
- Config → model params / smart params の抽出
- Smart param の解決（data-size dependent な変換）
- Per-fold ratio resolver の構築
- Estimator factory の構築
- Pipeline factory の構築
- デフォルト tuning space の提供

新アルゴリズム追加時の手順:
1. `estimators/<name>/` に adapter + provider + config を作成
2. `config/schema.py` の `ModelConfig` union に追加
3. Facade の provider dispatch に追加
4. `model.py` の変更: ゼロ

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
tuning_result = model.tune()       # TuningResult（best_model_params / best_smart_params / best_training_params / best_score / trials）
tuning_df = model.tuning_table()   # 全 trial の DataFrame（trial / score / params）
fit_result = model.fit()
eval_result = model.evaluate()
pred_result = model.predict(X_test, return_shap=True)
model.export("path/to/export_dir")
model.export_code("path/to/codegen_dir")  # LizyML 非依存の学習・推論コード生成（H-0059）
```

補足:
- `fit()` の default は、最も評価が良かったパラメーターで学習する。
- 必要に応じて、最終学習に使うパラメーターを明示指定できるようにする。
- `tune()` は `TuningResult` を返す。`TuningResult` は `best_model_params` / `best_smart_params` / `best_training_params`（カテゴリ別最良パラメーター）、`best_score`（最良スコア）、`trials`（全 trial の `TrialResult` リスト）、`metric_name`、`direction` を持つ。`best_params` プロパティは3カテゴリの flat view を返す（H-0050）。
- `tune()` は `progress_callback: TuneProgressCallback | None = None` を受け取り、各 trial 完了時に `TuneProgressInfo`（`current_trial / total_trials / elapsed_seconds / best_score / latest_score / latest_state`）をコールバックに渡す（H-0048）。コールバック内例外は catch して warning に変換し、tuning を中断させない。
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

`split.method` は以下のいずれか: `kfold` / `stratified_kfold` / `group_kfold` / `stratified_group_kfold` / `time_series` / `purged_time_series` / `group_time_series` / `blocked_group_kfold`。

| method | 固有キー |
|---|---|
| `kfold` | `n_splits=5`, `random_state=42`, `shuffle=True` |
| `stratified_kfold` | `n_splits=5`, `random_state=42` |
| `group_kfold` | `n_splits=5` |
| `stratified_group_kfold` | `n_splits=5`, `random_state=42`, `shuffle=True` |
| `time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |
| `purged_time_series` | `n_splits=5`, `purge_gap=0`, `embargo=0`, `train_size_max=null`, `test_size_max=null` |
| `group_time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |
| `blocked_group_kfold` | `blocks={col, cutoffs, mode, train_window}`, `groups={col, n_splits, stratify, shuffle}`, `min_train_rows=10`, `min_valid_rows=5` |

注記:
- `time_series` / `purged_time_series` / `group_time_series` は共通で `data.time_col` 必須。
- 3 メソッドは共通で `train_size_max` / `test_size_max` を受け取り、学習窓・検証窓の上限を制御する。
- `purged_time_series` の旧キー `embargo_pct` は移行期間のみ後方互換として扱い、`embargo` に正規化する。
- `blocked_group_kfold` は2軸交差検証（期間 × グループ）。`blocks.col` で期間を `cutoffs` で区切り、`groups.col` で KFold する。詳細は §10.6 参照。

### model（LightGBM）

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `params` | `dict[str, Any]` | No | `{}` | LightGBM パラメーター。`metric` キーで evaluation metric を指定可能。LizyML 名（`logloss`, `auc_pr` 等）も自動変換される（§14.3 参照、H-0061/H-0064） |
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
| `early_stopping.validation_ratio` | `float` (read-only) | — | `inner_valid.ratio` から派生 | `inner_valid.ratio` の computed alias（H-0069）。入力は `inner_valid` 経由を推奨。legacy YAML での入力は受理 |
| `early_stopping.inner_valid` | `object \| null` | No | `null`（自動解決） | inner valid の唯一の正規表現 |

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
| `n_splits` | `int` | No | `5` | **deprecated (H-0058)**: 無視される。calibration cross-fit は outer CV splits を再利用する。指定時は `UserWarning` を出力。 |

# 6. 実行フロー（概念）

## 6.1 `tune`

1. Config validate → データ読込・前処理
2. Config から smart params のデフォルト値を抽出する（`extract_smart_params`）
3. `Splitter` で外側 CV index 生成
4. 各 trial で:
   a. Optuna がパラメーターを提案し、`split_by_category` で model / smart / training に分類する
   b. Config defaults と trial params をマージし、`_build_train_components()` で CVTrainer の構成要素を構築する（fit と同じコードパス）
   c. CVTrainer で CV 実行 → OOF スコアを返す
5. `TuningResult`（`best_model_params` / `best_smart_params` / `best_training_params` / `best_score` / 全 trial 履歴）を返す
6. `Tuner` の責務は Optuna study の管理のみ。objective クロージャは `Model` 側で構築する

`tuning` と最終評価のリーク回避方針は 10 章を参照。

## 6.2 `fit`（CV）

1. Config defaults + tune 結果 + 引数 override をマージし、`_build_train_components()` で `TrainComponents`（`estimator_factory` / `sample_weight` / `ratio_resolver` / `inner_valid`）を構築する（tune と同じコードパス）。パラメータ優先順位: `Config defaults < tune best < fit() 引数`。
2. 外側 CV 各 fold で `train / valid` を作る。
   - `split.method` が `time_series` / `purged_time_series` / `group_time_series` の場合、`data.time_col` を基準に昇順へ並べた上で分割する。
3. `InnerValidStrategy` により early stopping 用の `inner_train / inner_valid` を生成する。
   - 分割対象は outer fold の `train` 部分のみとする。
   - `inner_train_idx / inner_valid_idx` は、その outer fold の `train` 部分に対する 0-based 相対 index として扱う。
   - `early_stopping.enabled=False` の場合は inner split を作らない。
4. `FeaturePipeline.fit()` は outer fold の `train` 全体に対して行う。
   - inner valid は estimator の early stopping 用 evaluation set であり、pipeline の fit 境界は outer train からさらに狭めない。
5. `EstimatorAdapter.fit()` を実行する。
   - inner split がある場合は `inner_train` を学習データ、`inner_valid` を eval set として渡す。
   - inner split がない場合は outer fold の `train` 全体で学習する。
6. OOF / IF を生成する（ロジックは `training/oof_assembly.py` に隔離）。
   - OOF は outer fold の `valid` 行に対してのみ生成する。
7. 必要なら `Calibrator` を cross-fit 学習する（OOF 予測のみ使用）。
8. 全データ Refit を実行する（同一の `TrainComponents` を使用し、CV との一貫性を構造的に保証する）。
   - `CVTrainer` と `RefitTrainer` は同じ `InnerValidStrategy` を共有する。
9. `FitResult` を返し、Artifacts を保持する。

補足:

- `fit()` は default で最良パラメーターを使用する。
- 他のパラメーターセットを指定して学習できる拡張点も残す。

## 6.3 `evaluate`

`FitResult` を入力に、指定メトリクスで以下を返す。

- `oof`
- `oof_per_fold`
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

## 6.6 `export_code`（Codegen Export, H-0059）

LizyML 非依存の学習・推論コードを自動生成する。

- **出力構造**: `config.json` + `train.py` + `predict.py` + `artifacts/` + `requirements.txt` + `test_equivalence.py`
- **train.py**: Feature pipeline fit → LightGBM refit（全データ学習）→ OOF 生成（軽量 CV）→ Calibrator fit
- **predict.py**: Feature transform → LightGBM predict → Calibration apply
- **config.json**: ハイパーパラメータ・特徴量定義・校正設定を集約。コード編集なしでパラメータ変更可能
- 生成コードは `import lizyml` を含まない。依存は `lightgbm` / `numpy` / `pandas` / `scikit-learn`（学習時のみ）
- `test_equivalence.py` で `Model.predict()` と codegen 出力の一致を `rtol=1e-7` で検証
- 初期実装は LightGBM のみ対応。将来の EstimatorProvider 拡張で他アルゴリズムにも対応可能
- Calibrator 保存形式: Platt → JSON (a, b)、Beta → JSON (a, b, c)、Isotonic → Booster テキスト
- **feval metric 対応（H-0066）**: ユーザー指定の feval metric（f1, brier, ece, precision_at_k, accuracy, rmsle, r2）を `config.json` の `feval_metrics` フィールドに記録し、`train.py` 内に pure numpy で再実装する。feval metric 未使用時は `feval_metrics: []` で後方互換を維持

# 7. Artifacts（戻り値と保存対象の固定）

## 7.1 FitResult（固定スキーマ）

- `oof_pred`（`np.ndarray / pd.Series`）
- `if_pred_per_fold`（`list[np.ndarray]`）
- `metrics`（階層固定）
  - 例: `{"raw": {"oof": {...}, "oof_per_fold": [...], "if_mean": {...}, "if_per_fold": [...]}, "calibrated": {...}}`
- `models`
  - fold ごとのモデル
  - 任意: refit モデル（全データ学習）
- `history`
  - fold ごとの eval history / best_iteration
- `feature_names / dtypes / categorical_features`
- `splits`
  - 外側 CV indices（必須、元データ基準の absolute index）
  - inner valid indices（有効時必須。各 outer fold train に対する 0-based 相対 index）
  - calibration CV indices（有効時必須）
  - `time_range`（時系列分割時。fold ごとの train/valid の期間情報 `list[dict] | None`）
- `data_fingerprint`
  - `row_count / column_hash / optional: file_hash` 等
- `pipeline_state`（`FeaturePipeline` の状態、必須）
- `calibrator`（有効時）
- `run_meta`
  - `yourlib_version / python_version / deps_version / config_normalized / config_version`
- `target_encoder`（H-0070, format_version=2）
  - `TargetEncoder(classes_: tuple[Any, ...], needs_encoding: bool, original_dtype: str)`
  - 数値 y / regression では `needs_encoding=False` の no-op
  - 非数値 classification y では `classes_` に sorted 元ラベルを保持
  - `predict()` / codegen / persistence migration で利用

## 7.2 TuningResult（固定スキーマ）

- `best_model_params`（`dict[str, Any]`）: 最良の model カテゴリパラメーター（`learning_rate` 等）
- `best_smart_params`（`dict[str, Any]`）: 最良の smart カテゴリパラメーター（`num_leaves_ratio` 等）
- `best_training_params`（`dict[str, Any]`）: 最良の training カテゴリパラメーター（`early_stopping_rounds` 等）
- `best_score`（`float`）: 最良の OOF メトリクス値
- `trials`（`list[TrialResult]`）: 全 trial の結果（番号順）
  - `TrialResult`: `number` / `params` / `score` / `state`（`"complete"` / `"pruned"` / `"fail"`）
- `metric_name`（`str`）: 最適化メトリクス名
- `direction`（`str`）: `"minimize"` / `"maximize"`
- `best_params`（computed property）: `{**best_model_params, **best_smart_params, **best_training_params}` の flat view

## 7.3 PredictionResult（固定スキーマ）

- `pred`（回帰: float、分類: class / proba）
- `proba`（binary の場合）
- `shap_values`（要求時のみ、形は統一）
- `used_features`（列ズレ検知用）
- `warnings`（補正が走った場合の通知）

補足:

- 回帰では `pred` を主とする。
- 分類では `pred` に加えて `proba` を返せるようにする。
- 分類で y が非数値（object/str/StringDtype/category/bool）の場合、`pred` の dtype は **fit 時の元 y dtype と一致**する（H-0070, INV-2）。`proba` の列順は `FitResult.target_encoder.classes_` と一致する（INV-3）。

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
- `DataFrameBuilder.build()` は `ProblemSpec.task` を見て `TargetEncoder` を fit/apply し、`DataFrameComponents.target_encoder` に格納する（H-0070）。
  - 数値 y / regression は no-op（`needs_encoding=False`）
  - 非数値 classification y は int code に encode（下流の training/estimators/calibration は常に int y を見る）
  - regression × 非数値 y → `TARGET_NOT_NUMERIC` を fit 開始前に raise

## 8.2 Validators（危険検知）

- 時系列: ソート、未来情報混入疑い、shuffle 禁止
- group: group 跨ぎ、分割条件の不整合
- leakage: target リーク疑い（例: target と完全一致の列、時間逆転など）
- target dtype: regression × 非数値 y は `TARGET_NOT_NUMERIC`（H-0070）

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
- 外側 CV / calibration で共通利用する。
- early stopping 用の内側分割は `training/inner_valid.py` の `InnerValidStrategy` が担当し、splitter とは責務を分離する。
- calibration cross-fit は outer CV splits をそのまま再利用する（H-0058）。`calibration.n_splits` は deprecated（指定時 `UserWarning`、値は無視）。

## 10.2 Outer CV（例）

- `KFold`
- `StratifiedKFold`（binary/multiclass のデフォルト）
- `GroupKFold`
- `TimeSeriesSplit`
- `PurgedTimeSeries`
- `GroupTimeSeries`
- `BlockedGroupKFold`（2軸交差検証: 期間 × グループ、H-0060）

注記:
- `task` が `binary` または `multiclass` かつ `split.method` が未指定の場合、`StratifiedKFold` をデフォルトとする。分類タスクで `method: "kfold"` を明示指定した場合は警告を出す。回帰タスクのデフォルトは `KFold` のまま。
- `time_series` / `purged_time_series` / `group_time_series` は共通で `data.time_col` を基準に昇順へ並べてから分割する。
- `time_series` / `group_time_series` は `gap`、`purged_time_series` は `purge_gap` を持つ（いずれも train と valid の間のギャップ）。
- `PurgedTimeSeries` は `embargo`（train と valid の間に設ける追加除外 Obs 数、`int`、`gap` / `purge_gap` と同じ単位）を持つ。`embargo_pct` は移行期間のみ後方互換キーとする（`int()` で変換）。
- 3 メソッドは共通で `train_size_max` / `test_size_max` を持つ。
- `GroupTimeSeries` は group 列の出現順と `time_col` 順を整合させて時系列的にグループを分割する。

## 10.3 InnerValidStrategy（early stopping 用）

- CV fold 内でさらに `train / valid` を作る概念を分離する。
- `InnerValidStrategy` は `Model._build_train_components()` で解決し、`CVTrainer` と `RefitTrainer` に同一インスタンスを渡す。
- `early_stopping.enabled=False` の場合は `NoInnerValid` を使い、inner split を生成しない。

### 10.3.1 設定の解決規則

- `training.early_stopping.inner_valid` を明示指定した場合は、その method / ratio / random_state をそのまま使う。外側 `split.method` は参照しない。
- `training.early_stopping.validation_ratio` は legacy 入力ショートハンドであり、method 指定ではない。`inner_valid` を明示指定していない場合、ratio は `validation_ratio` から取り、method は外側 `split.method` から自動解決する。H-0069 以降、`validation_ratio` は `inner_valid.ratio` から派生する read-only の computed field。出力 (`model_dump()`) には常に同値の `validation_ratio` が含まれる。
- `validation_ratio` と `inner_valid` を同時に明示指定した場合、ratio が一致しなければ `CONFIG_INVALID`、一致すれば許容する（round-trip 互換）。一致しない場合の検知は維持される。
- 自動解決時に inner valid が継承する outer CV 設定は `split.method` のみとする。`n_splits` / `shuffle` / `random_state` / `gap` / `purge_gap` / `embargo` / `train_size_max` / `test_size_max` は inner valid に伝搬しない。
- 自動解決時の seed は `training.seed` を使う。outer split の `random_state` は inner valid に伝搬しない。

| 外側 split.method | inner_valid のデフォルト |
|---|---|
| `stratified_kfold` | `holdout(stratify=True)` |
| `group_kfold` | `group_holdout` |
| `stratified_group_kfold` | `group_holdout` |
| `time_series` | `time_holdout` |
| `purged_time_series` | `time_holdout` |
| `group_time_series` | `group_holdout` |
| `blocked_group_kfold` | `blocked_group_inner_valid`（§10.6.2 参照） |
| `kfold`（または CV 未使用） | `holdout(stratify=False)` |

補足:

- `split.method` 未指定時は outer CV 側のデフォルトが先に確定し、その method を使って inner valid を自動解決する。
  - `binary` / `multiclass`: `stratified_kfold` → `holdout(stratify=True)`
  - `regression`: `kfold` → `holdout(stratify=False)`

### 10.3.2 CV / Refit への適用位置

- `CVTrainer` では各 outer fold の `train` 部分に対してのみ inner split を作る。outer fold の `valid` 部分は inner valid の対象に含めない。
- `FitResult.splits.inner` に保存する `inner_train_idx / inner_valid_idx` は、各 outer fold の `train` 部分に対する 0-based 相対 index とする。inner valid が無効な場合は `FitResult.splits.inner = None` とする。
- `FeaturePipeline.fit` は outer fold の `train` 全体に対して行う。inner valid は estimator の early stopping 用 evaluation set であり、FeaturePipeline の fit 境界は outer train のままとする。
- estimator は inner valid が有効な場合 `inner_train` のみで学習し、`inner_valid` を eval set として early stopping を行う。OOF の割当先は引き続き outer fold の `valid` のみとする。
- `RefitTrainer` でも同じ `InnerValidStrategy` を全データに適用して final model の early stopping 用 split を作る。
  - inner valid がある場合、pipeline は **inner-train のみ**で fit する（CVTrainer と一致する leakage 境界）。estimator は inner-train で学習、inner-valid で early stopping。
  - 最終的な `pipeline_state`（推論用）は、別途全データで fit した pipeline から取得する。`categorical_features` も全データ fit 由来の pipeline から取得する。
  - inner valid が無い場合（`NoInnerValid`）は、pipeline は全データで 1 回のみ fit する（二重 fit を回避）。
  - `time_series` / `purged_time_series` / `group_time_series` では、`Model._prepare_training_data()` により時系列昇順へ並べ替えた後の全データに対して inner valid を切る。

### 10.3.3 各 strategy の分割規則

- `HoldoutInnerValid(ratio, stratify=False, random_state)`:
  - `stratify=False`: outer fold train 行を乱択し、`ceil(n_rows * ratio)` 行を validation に割り当てる。
  - `stratify=True`: `y` に基づく stratified holdout を行う。
  - `n_valid >= n_samples` の場合は `ValueError` を発出する（空の train set 防止）。
- `GroupHoldoutInnerValid(ratio, random_state)`:
  - group overlap を禁止する。
  - validation には、入力順（group の first appearance 順）の末尾 `max(1, floor(n_unique_groups * ratio))` 個の group を割り当てる。
  - shuffle は行わないため、`group_time_series` では time-sort 後の入力順に従って末尾 group が validation になる。
- `TimeHoldoutInnerValid(ratio)`:
  - 行順を保持したまま、末尾 `max(1, floor(n_rows * ratio))` 行を validation に割り当てる。
  - `purged_time_series` で outer CV が purge / embargo を持っていても、inner valid 自体は追加の purge / embargo を持たない。
  - `n_valid >= n_samples` の場合は `ValueError` を発出する（空の train set 防止）。
- `BlockedGroupInnerValid(ratio)`:
  - `blocked_group_kfold` 専用。グループ分離 + 時間順序 + 層化（分類時）を同時に満たす。
  - 詳細は §10.6.2 を参照。
- `StratifiedTimeHoldoutInnerValid(ratio)`:
  - `BlockedGroupInnerValid` のフォールバック（グループ数 < 4 時）。
  - 各クラス内で時間順序を保持し、末尾 `ratio` 分を validation に割り当てる。全クラス最低1行を保証する。
  - 回帰タスクでは `TimeHoldoutInnerValid` と同等。

## 10.4 split indices の保存（必須）

- 外側 CV: fold ごとの `train_idx / valid_idx`
- inner valid: fold 内の `inner_train_idx / inner_valid_idx`
- calibration CV: 校正用の `train_idx / valid_idx`
  - calibration split は outer split と同一値を保存する（H-0058）。冗長だが後方互換性と明示性のためフィールドは残す。

## 10.5 Calibration CV の分割規約（必須）

- calibration cross-fit は outer CV の split indices (`fit_result.splits.outer`) をそのまま再利用する（H-0058）。
- `calibration.n_splits` は **deprecated**（指定時 `UserWarning` を出力し、値は無視する）。
- calibration の入力は `(oof_scores, y)` のみで X は使わない（§12.1）。outer splits を再利用しても同一行リークは発生しない（各行の OOF score はその行を含まないモデルが生成したものであり、cross-fit 構造がさらにリークを防ぐ）。
- これにより calibrated OOF の coverage は raw OOF の coverage と構造的に一致する。

## 10.6 blocked_group_kfold（2軸交差検証、H-0060）

期間軸（blocks）とグループ軸（groups）の直積で交差検証を行う。各 fold = (時間分割 t) × (ユーザー分割 u) として生成される。

### 10.6.1 Config 構造

```yaml
split:
  method: blocked_group_kfold
  blocks:
    col: date                         # 期間を定義するカラム（ソート可能な型）
    cutoffs: ["2025-02", "2025-03"]   # 境界値リスト（valid 期間の開始点）
    mode: sliding                     # expanding | sliding
    train_window: 2                   # sliding 時: train に使う期間数
  groups:
    col: user_id                      # グループ分割するカラム
    n_splits: 3                       # グループの分割数
    stratify: auto                    # auto | true | false
    shuffle: true                     # グループ分割時のシャッフル
  min_train_rows: 10                  # fold スキップ閾値
  min_valid_rows: 5
```

**blocks**: `cutoffs: [C₁, C₂, ..., Cₙ]` から `n+1` 個の期間を生成する（P₀: `col < C₁`, P₁: `C₁ ≤ col < C₂`, ..., Pₙ: `col ≥ Cₙ`）。`expanding` は train が累積、`sliding` は直前 `train_window` 期間のみ train に使用。

**groups**: 全ユーザーを `n_splits` 分割し KFold する。`stratify: auto` は binary/multiclass で代表ラベル（多数決クラス）による層化を適用する。

**fold 生成**: 各時間 fold t のデータから全ユーザーを取得し、`n_splits` 分割。各ユーザー fold u に対して:
- Train = train 期間の行 ∩ train_users の行
- Valid = valid 期間の行 ∩ valid_users の行
- 除外 = train 期間 × valid_users + valid 期間 × train_users

合計 fold 数 = `len(cutoffs) × groups.n_splits − skip数`。`min_train_rows` / `min_valid_rows` 未満の fold はスキップ + 警告。

**バリデーション**: `blocks.col == groups.col` → `CONFIG_INVALID`。`mode: sliding` で `train_window` 未指定 → `CONFIG_INVALID`。`cutoffs` 空 → `CONFIG_INVALID`。

**Facade 責務**: `blocks.col` でデータをソートし、`blocks.col` の値を splitter コンストラクタに注入する。`BaseSplitter.split()` のシグネチャは変更しない。

### 10.6.2 Inner Valid: BlockedGroupInnerValid

outer fold の train データ（特定期間 × 特定ユーザー）に対して、グループ分離 + 時間順序 + 層化（分類時）を同時に満たす inner valid を提供する。

**アルゴリズム:**

1. outer fold train 内のユニークグループを取得
2. 各グループの代表ラベルを算出（多数決クラス）※分類時のみ
3. 各グループの最終出現時刻でソート
4. 分類時: 各クラス内で末尾 `ratio` 分のグループを inner valid に割り当て（各クラス最低1グループ保証）。回帰時: 単純に末尾 `ratio` 分のグループを割り当て
5. グループ単位で完全分離: inner train = train グループの全行、inner valid = valid グループの全行

**フォールバック**: `n_unique_groups < 4` の場合、`StratifiedTimeHoldoutInnerValid`（各クラスの末尾行から `ratio` 分）に切り替え、警告を出す。

**明示指定**: `training.early_stopping.inner_valid` を明示指定した場合は auto 解決を上書きする。

# 11. Tuning（`tuning/`）

## 11.1 SearchSpace 表現の統一

- Optuna に依存しない space 表現（離散・連続・対数・カテゴリ）を使う。

## 11.2 SearchDim カテゴリ

SearchDim にカテゴリ属性を持たせ、Tuner がパラメーターの適用先を区別する。

- `model`: `LGBMAdapter.params` に直接渡す（既存 SearchDim の挙動）
- `smart`: スマートパラメーター（`num_leaves_ratio` 等）として `resolve_smart_params()` に渡す。fit / tune で同一の dict ベース `resolve_smart_params()` を使用する（H-0050）
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

## 11.4 Progress Callback（H-0048）

`tune()` 実行時に外部ツール（Widget 等）が進捗情報をリアルタイムに取得するためのコールバック機構を提供する。

### TuneProgressInfo（frozen dataclass）

| フィールド | 型 | 説明 |
|---|---|---|
| `current_trial` | `int` | 現在の trial 番号（1-indexed） |
| `total_trials` | `int` | 全 trial 数 |
| `elapsed_seconds` | `float` | 経過時間（秒） |
| `best_score` | `float \| None` | これまでの最良スコア（complete trial なしの場合 `None`） |
| `latest_score` | `float \| None` | 直近 trial のスコア（fail/pruned の場合 `None`） |
| `latest_state` | `str` | `"complete"` / `"pruned"` / `"fail"` |
| `round` | `int` | 現在のラウンド番号（1-indexed）。H-0068 で追加 |
| `cumulative_trials` | `int` | 全ラウンド通算の試行数。H-0068 で追加 |
| `expanded_dims` | `tuple[str, ...]` | このラウンドで拡張された次元名。H-0068 で追加 |

### TuneProgressCallback

```python
TuneProgressCallback = Callable[[TuneProgressInfo], None]
```

### 使用例

```python
def on_progress(info: TuneProgressInfo) -> None:
    print(f"[Round {info.round}] Trial {info.current_trial}/{info.total_trials} "
          f"(cumulative {info.cumulative_trials}) "
          f"score={info.latest_score} best={info.best_score}")

result = model.tune(progress_callback=on_progress)
```

### 制約

- `progress_callback` はデフォルト `None`（後方互換）。
- コールバック内で例外が発生した場合は catch して warning に変換し、tuning を中断させない。
- Optuna の `study.optimize(callbacks=[...])` を活用し、各 trial 完了時に通知する。
- `TuneProgressInfo` と `TuneProgressCallback` は `lizyml/__init__.py` の公開面に含める。

## 11.5 Re-tune: Study Resume + 境界検知拡張（H-0068）

初回 tuning 後に追加探索を行い、さらなる精度向上を目指す。

### Model.tune() の拡張パラメーター

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `resume` | `False` | `True`: 前回 Study を再利用して追加試行 |
| `n_trials` | `None` | 追加試行数（`None` → config 値） |
| `expand_boundary` | `None` | 境界拡張。`None`: デフォルト空間→`True`、ユーザー空間→`False` |
| `boundary_threshold` | `0.05` | 端判定閾値（0.0〜1.0） |

### 境界検知ルール

- **linear 空間**: `(best - low) / (high - low) < threshold` → 下限近傍
- **log 空間**: 対数空間で同一計算
- **categorical**: 拡張不可（ログ通知のみ）

### 非対称拡張ルール

- linear: 端方向に `(high - low)` を追加（range 2 倍）
- log: 端方向に対数空間で 3 倍に拡張
- `IntDim`: `max(1, new_low)` で下限ガード
- 反対側の端は据え置き

### RoundSummary / BoundaryReport

```python
@dataclass(frozen=True)
class RoundSummary:
    round: int                        # 1-indexed
    n_trials: int
    best_score_before: float | None
    best_score_after: float
    expanded_dims: tuple[str, ...]
    space_snapshot: tuple[SearchDim, ...]

@dataclass(frozen=True)
class BoundaryDimStatus:
    name: str
    best_value: float | int | str | None
    low: float | int | None
    high: float | int | None
    position_pct: float | None
    edge: str                    # "lower" | "upper" | "none"
    expanded: bool
    new_low: float | int | None
    new_high: float | int | None

@dataclass(frozen=True)
class BoundaryReport:
    dims: tuple[BoundaryDimStatus, ...]
    expanded_names: tuple[str, ...]
```

### TuningResult 拡張

```python
# 追加フィールド
rounds: tuple[RoundSummary, ...]        # デフォルト: (RoundSummary(round=1, ...),)
boundary_report: BoundaryReport | None  # resume 時のみ設定
```

### TrialResult 拡張

```python
round: int  # 追加: どのラウンドの試行か (1-indexed)。デフォルト 1
```

### tuning_table() 拡張

`round` 列と `state` 列を追加。

### boundary_table() 新設

`BoundaryReport` を DataFrame に変換。列: `dim`, `best`, `low`, `high`, `position`, `edge`, `expanded`, `new_low`, `new_high`。

### plot_tuning_history() 拡張

- ラウンド境界に縦の破線
- ラウンドごとのアノテーション（拡張次元名）
- best score 累積線はラウンドをまたいで連続

### Widget / Studio 連携

LizyML Core は callback + 結果型でデータを提供し、Widget/Studio が消費する設計。

| 消費者 | 情報源 | 用途 |
|---|---|---|
| Widget（リアルタイム） | `TuneProgressInfo.round`, `.cumulative_trials`, `.expanded_dims` | 進捗バー、拡張パネル |
| Studio（ダッシュボード） | `TuningResult.rounds`, `.boundary_report` | Round History、Search Space Evolution、収束判定 |

収束判定（`expanded_dims` 空 + 改善微小 → fit 推奨）は Widget/Studio 側の責務。Core は判断材料のみ提供する。

### 制約

- `resume=False` は現在と同一動作（完全後方互換）
- `resume=True` で未 tune → `LizyMLError(TUNING_FAILED)`
- Tuner は study オブジェクトの受け取り・返却に対応するが、study の永続化（RDB storage 等）は対象外

## 11.6 リーク回避方針（必須で明文化）

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
- 校正 cross-fit は outer CV splits をそのまま再利用する（§10.5, H-0058）。これにより raw OOF と calibrated OOF の coverage が構造的に一致する。
- 校正器は元の特徴量 `X` を使わない（入力は `s_oof`（生スコア）と `y` のみ）。
- 推論時は保存された `C_final` を使用する。
- Calibration が未指定の場合は従来どおり `predict_proba`（確率値）を OOF/IF 予測に使用する。Calibration 有効時のみ生スコアベースの校正パスに入る。

## 12.2 方法

- Platt Scaling
- Beta Calibration
- Isotonic Regression（LGBM の単調制約利用）

### Isotonic Regression 詳細（H-0047）

`IsotonicCalibrator` は LightGBM Booster API（`lgb.train()`）を使用し、単一特徴（raw score）に対する単調非減少写像を学習する。

#### デフォルトパラメーター

| パラメーター | デフォルト | 備考 |
|---|---|---|
| `objective` | `binary` | |
| `metric` | `binary_logloss` | |
| `monotone_constraints` | `[1]` | **常に強制（上書き不可）** |
| `monotone_constraints_method` | `advanced` | |
| `num_leaves` | `7` | 1次元補正器なので控えめ |
| `max_depth` | `3` | |
| `min_data_in_leaf_ratio` | `0.01` | fit 時に `max(1, ceil(n_train * ratio))` に解決 |
| `learning_rate` | `0.03` | 過学習しにくい低学習率 |
| `lambda_l2` | `5.0` | |
| `min_gain_to_split` | `0.0` | |
| `feature_fraction` | `1.0` | 1特徴なのでランダム化不要 |
| `bagging_fraction` | `1.0` | 同上 |
| `bagging_freq` | `0` | 同上 |
| `num_boost_round` | `1000` | `lgb.train()` の引数 |

#### Early Stopping

- `patience=100`（`lgb.early_stopping(stopping_rounds=100)` コールバック）。
- validation データ: calibration 学習データから 10% をランダムサンプリング（`validation_ratio=0.1`, `seed=42` デフォルト）。
- calibration データが少数（< 20 行）の場合は Early Stopping を無効化し、全データで学習する。

#### ユーザー上書き

- `calibration.params` で上記デフォルト（`monotone_constraints` 以外）を上書き可能。
- `validation_ratio` と `seed` も `calibration.params` 経由で指定可能。

#### Booster API 固有の注意

- `objective="binary"` の `Booster.predict()` は raw score を返すため、predict 時に sigmoid 適用 + `np.clip(0, 1)` で確率に変換する。

## 12.3 評価（推奨）

- `LogLoss`（必須推奨）
- `Brier score`（必須推奨）
- `ECE`（equal-width binning, M=10。各 bin の accuracy = `mean(y_true[mask])`（正例割合）、confidence = `mean(y_pred[mask])`。ECE = Σ (|bin| / N) × |accuracy − confidence|）
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

### 13.1.1 パラメータ付き MetricEntry（H-0065）

パラメータを持つメトリクス（`precision_at_k` の `k` 等）は、`str | dict[str, dict[str, Any]]` 形式（`MetricEntry`）で指定する。

```python
MetricEntry = str | dict[str, dict[str, Any]]
```

- `str` 指定: デフォルトパラメータで動作（後方互換）
- `dict` 指定: `{metric_name: {param: value}}`。キー数は 1。

`EvaluationConfig.metrics` と `model.lgbm.params.metric` の両方で使用可能。各設定箇所で独立した値を指定できる。

```yaml
evaluation:
  metrics: [auc, {precision_at_k: {k: 20}}]
model:
  lgbm:
    params:
      metric: [{precision_at_k: {k: 5}}]
```

`BaseMetric.name` プロパティは変更しない（`"precision_at_k"` のまま）。`k` の可視化は Plot 凡例と `params_summary()` に限定する。

## 13.2 評価出力（固定）

- IF / OOF と fold 別を必ず返す。
- 校正前後も同一フォーマットで返す（binary）。
- `evaluate_table()` は `evaluate()` が返す固定構造 dict を `pd.DataFrame` に変換する純粋フォーマッタ。ロジックは `evaluation/table_formatter.py` に配置する。
  - 行 = メトリクス名。
  - `oof`: OOF 集約値（**covered 行ベース**。split で valid に一度も含まれない行は除外。KFold では全行=covered、TimeSeriesCV では先頭行が non-covered）。
  - `fold_0`...`fold_N-1`: 各 outer fold の OOF（valid_idx）値。
  - `if_mean`: IF（train_idx）指標の fold 平均（参考値として保持）。
  - calibrated がある場合は `cal_oof` 列と `cal_fold_0`...`cal_fold_N-1` 列を追加。calibrated ブランチの metrics 構造は `{"oof": {...}, "oof_per_fold": [...]}`。IF metrics は leakage リスクのため含めない。`oof_coverage` は raw と構造的に一致する（H-0058: outer splits 再利用）ため、`calibrated` に別途含めない。
  - `df.attrs["oof_coverage"]`: float (0.0–1.0)。covered 行の割合。KFold では常に `1.0`。TimeSeriesCV では `< 1.0` になりうる。

## 13.3 可視化

全プロットを Plotly ベースに統一する。Plotly は optional dependency（`pip install 'lizyml[plots]'`）。未インストール時は `OPTIONAL_DEP_MISSING` を返す。

実装済み:
- `importance_plot(kind="split|gain")`: fold 平均の特徴量重要度（横棒グラフ）
- `importance_plot(kind="shap")`: fold 平均の mean(|SHAP|)（横棒グラフ）。shap optional dependency も必要。
- `plot_learning_curve(*, metrics=None)`: fold ごとの train/valid loss 推移（折れ線グラフ）。`metrics: list[str] | None` で表示 metric をフィルタ可能（H-0062）。`None` で全 metric、指定時は `/` 以降の metric 名で一致するもののみ表示。一致なしで `LizyMLError`。
- `plot_oof_distribution()`: OOF 予測値の分布（ヒストグラム）
- `residuals_plot(kind="scatter|histogram|qq|all")`: 回帰専用。IS/OOS 比較対応。`kind` で表示プロットを選択。デフォルト `kind="all"` で scatter + histogram + QQ の 3 パネル。scatter は Actual vs Predicted（x=predicted, y=actual, y=x 参照線）。IS サンプルは OOS 数に合わせてダウンサンプリング（`_downsample_is()`、seed=0 で再現可能）。

追加で用意したい可視化（一部実装済み）:
- binary/multiclass: `roc_curve_plot()`（binary: IS/OOS の 2 本の ROC Curve 重ね描き。multiclass: IS/OOS を subplot 横並びにし、クラスごとの OvR ROC Curve を描画。各クラスの AUC 値を凡例に表示、macro 平均 AUC も表示）
- binary/multiclass: `confusion_matrix(threshold=0.5)`（IS/OOS の Confusion Matrix テーブル。`{"is": DataFrame, "oos": DataFrame}` を返す。binary は threshold、multiclass は argmax でクラスラベル変換。OOS は `compute_oof_valid_mask()` でカバー済み行のみを対象とする — NaN の構造的未カバー行は除外）
- calibration: `calibration_plot()`（Raw/Calibrated の Reliability Diagram。bin 数デフォルト 10。理想線 y=x を参照線として描画。データソースは cross-fit 由来の `calibrated_oof`、`c_final` は使用しない）
- calibration: `probability_histogram_plot()`（Raw/Calibrated の確率分布ヒストグラム重ね描き。校正前後の分布シフトを視覚的に確認）
- tuning: `tuning_plot()`（trial ごとのスコア推移。X 軸 = trial 番号、Y 軸 = スコア。完了/枝刈り/失敗を色分け。最良スコア推移ラインを重ね描き）
- 時系列: `split_summary()`（fold ごとの分割サイズ。時系列分割時は `train_start / train_end / valid_start / valid_end` の期間情報を含む `pd.DataFrame` を返す）
- 未実装: `PR Curve / threshold最適化レポート`

## 13.4 評価・可視化 API の目的分類

各 API のデータソースと主目的を以下のとおり分類する。IS(In-Sample) = IF(train_idx) の集約値、OOS(Out-of-Sample) = OOF(valid_idx) の値。

| API | データソース | 主目的 | カテゴリ |
|-----|------------|--------|---------|
| `evaluate()` | OOF + IF | 汎化性能の定量評価 | 汎化監視 |
| `evaluate_table()` | OOF(fold列) + IF(if_mean列) | 汎化性能の比較表 | 汎化監視 |
| `roc_curve_plot()` | IS + OOS | 過学習検知（IS/OOS 比較） | 診断 |
| `confusion_matrix()` | IS + OOS | 予測分布の確認（IS/OOS 比較） | 診断 |
| `residuals_plot()` | IS + OOS | 残差パターンの確認（IS/OOS 比較） | 診断 |
| `plot_learning_curve()` | train/valid loss | 学習収束・過学習の検知 | 学習過程監視 |
| `plot_oof_distribution()` | OOF | 予測分布の全体像 | 汎化監視 |
| `calibration_plot()` | OOF(cross-fit) | 校正効果の確認 | 汎化監視 |
| `probability_histogram_plot()` | OOF(cross-fit) | 確率分布シフトの確認 | 汎化監視 |
| `importance_plot()` | fold 平均 | 特徴量寄与の把握 | 汎化監視 |

- **汎化監視**: モデルの汎化性能を評価する API。OOF（valid_idx）を主データソースとする。
- **診断**: 過学習・予測パターンを検知する API。IS（train_idx）と OOS（valid_idx）の比較を提供する。
- **学習過程監視**: 学習の進行状況を確認する API。学習履歴を使用する。

# 14. Estimators（`estimators/`）

## 14.1 EstimatorAdapter IF

> Layer 1（Leaf）に属する。Foundation のみに依存し、`config/` や他の Leaf カテゴリに依存しない。

```python
fit(X_train, y_train, X_valid=None, y_valid=None, **kwargs)
predict(X)
predict_proba(X)  # 分類（sigmoid/softmax 適用後の確率値）
predict_raw(X)    # 分類（sigmoid/softmax 適用前の生スコア / logits。Calibration 用）
importance(kind="split|gain|shap")
get_native_model()  # export用途
set_categorical_features(cols: list[str] | None) -> None  # デフォルト no-op (H-0054)
```

- `set_categorical_features()` は `fit()` 呼び出し前に CVTrainer が呼ぶ。categorical feature の扱いはアダプタの責務であり、`cv_trainer.py` に estimator 固有の kwarg を漏洩させない。

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
- `LGBMConfig.params` に `metric` を指定した場合、ユーザー指定値を優先する。未指定時は上記タスク別デフォルトにフォールバックする（H-0061）。

#### Metric Bridge（H-0064）

`metric_bridge.py` が metric 指定に対して以下の処理を行う:

1. **名前マッピング**: LizyML 評価用名 → LightGBM 学習用名に自動変換

| LizyML 名 | LightGBM 名 | タスク |
|-----------|-------------|-------|
| `logloss` | `binary_logloss` / `multi_logloss` | binary / multiclass |
| `auc_pr` | `average_precision` | binary / multiclass |

2. **ホワイトリストバリデーション**: マッピング後の名前をタスク別ホワイトリストで事前検証。無効な metric 名は `LizyMLError(CONFIG_INVALID)` で即座に拒否する（LightGBM 呼び出し前）。

3. **feval カスタム関数**: LightGBM ネイティブ未対応の metric は `lgb.train(feval=...)` 経由でカスタム評価関数として注入する。

| feval Metric | Regression | Binary | Multiclass | y_pred 変換 |
|-------------|:---:|:---:|:---:|------------|
| `rmsle` | ✅ | | | そのまま |
| `f1` | | ✅ | ✅ | sigmoid / softmax → 閾値 / argmax |
| `brier` | | ✅ | ✅ | sigmoid / softmax |
| `ece` | | ✅ | | sigmoid |
| `precision_at_k` | | ✅ | | sigmoid |
| `accuracy` | | ✅ | ✅ | sigmoid / softmax → 閾値 / argmax |

- binary: `y_pred` は raw logits → `sigmoid` で確率に変換
- multiclass: `y_pred` は flatten `(n * k,)` → `reshape(-1, k)` + `softmax` で確率に変換
- native metric と feval metric の混在指定が可能（例: `["auc", "f1"]`）
- feval-only 指定時も early stopping が正常に機能する

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

## 14.4 EstimatorProvider protocol（H-0053）

各 estimator モジュールが実装する protocol。`model.py`（Facade）が estimator 固有の知識なしに TrainComponents を構築するための統一 IF。

```python
class EstimatorProvider(Protocol):
    def extract_model_params(self, model_cfg: Any) -> dict[str, Any]: ...
    def extract_smart_params(self, model_cfg: Any) -> dict[str, Any]: ...
    def resolve_smart_params(
        self, smart: dict, effective: dict, n_rows: int,
        feature_names: list[str], y: Series, task: str,
    ) -> tuple[dict[str, Any], ndarray | None]: ...
    def build_ratio_resolver(
        self, smart: dict,
    ) -> Callable[[int], dict[str, Any]] | None: ...
    def build_estimator_factory(
        self, task: str, params: dict, n_classes: int | None,
        early_stopping_rounds: int | None, seed: int,
    ) -> Callable[[], BaseEstimatorAdapter]: ...
    def build_pipeline_factory(self) -> Callable[[], BaseFeaturePipeline]: ...
    def default_space(self, task: str) -> list[SearchDim]: ...
    def default_fixed_params(self, task: str) -> dict[str, Any]: ...
    def runtime_deps(self) -> dict[str, str]: ...
    def params_summary(
        self, model: BaseEstimatorAdapter, model_cfg: Any,
    ) -> list[dict[str, Any]]: ...
```

制約:
- `EstimatorProvider` は `config/` の具象型（`LGBMConfig` 等）を参照してよい（provider は Facade 層から呼ばれるため、Leaf → Leaf の依存にはならない）。
- `model_cfg` 引数は `Any` 型で受け取るが、各 provider 内部で `isinstance` チェックして具象型にキャストする。
- `runtime_deps()` はアルゴリズム固有の依存パッケージ名とバージョンを返す（例: `{"lightgbm": "4.5.0"}`）。`RunMeta.deps_versions` に使用。
- `params_summary()` は `params_table()` 用のパラメータ行を返す。smart params + native model params（`metric` を含む、H-0061）の両方を含む。
- `build_pipeline_factory` は estimator 固有の FeaturePipeline が必要な場合（例: EntityEmbedding のカテゴリ埋め込み）に対応する。デフォルトは `NativeFeaturePipeline` を返す。

ディレクトリ構成（estimator ごとにサブパッケージ化）:

```text
estimators/
├── base.py              BaseEstimatorAdapter（IF + set_categorical_features）
├── provider.py          EstimatorProvider protocol 定義
├── lgbm/
│   ├── __init__.py      LGBMAdapter, LGBMProvider を re-export
│   ├── adapter.py       LGBMAdapter（現在の lgbm.py から）
│   ├── provider.py      LGBMProvider（EstimatorProvider 実装）
│   ├── smart_params.py  resolve_smart_params / resolve_ratio_params
│   └── defaults.py      _COMMON_DEFAULTS / task defaults / default_space
└── <future>/            EntityEmbedding 等（同構造で追加）
```

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
- 現行 `FORMAT_VERSION = 2`（H-0070）。`{1, 2}` の両方を loader が受理し、v1 artifact には no-op `TargetEncoder` を in-memory で注入して contract を整合させる（INV-5）。

## 15.3 `export`（`Model Artifact`）

- `Model Artifact` を 1 ディレクトリにまとめる。
- `Model.load()` で復元し、推論と評価情報参照に加えて診断 API（残差/SHAP/分類・校正可視化）も利用可能にする。

## 15.4 `export_code`（Codegen Export, H-0059）

- LizyML 非依存の学習・推論コードを生成する。`export`（§15.3）とは独立した出力形式。
- `format_version` とは無関係（pickle を使用せず、テキスト/JSON のみ）。
- 出力ディレクトリ構造:

```
{path}/
├── config.json             # 全設定（ハイパーパラメータ / 特徴量 / 校正）
├── train.py                # 学習（pipeline fit → refit → calibration）
├── predict.py              # 推論（transform → predict → calibrate）
├── requirements.txt        # 最小依存
├── test_equivalence.py     # LizyML との一致検証
└── artifacts/              # train.py が生成
    ├── model.txt           # LightGBM Booster テキスト
    ├── pipeline_state.json # 学習済み Pipeline 状態
    ├── calibrator.json     # Calibrator パラメータ
    └── calibrator_model.txt # Isotonic Booster（該当時のみ）
```

- `artifacts/` の初期内容は `export_code()` 実行時に元の FitResult/RefitResult から生成される
- `train.py` で新データから再学習すると `artifacts/` が上書きされる
- **feval metric 対応（H-0066）**: `config.json` に `feval_metrics` フィールドを追加。各要素は `{"name": str, "params": dict, "greater_is_better": bool, "needs_proba": bool}` 形式。`train.py` が起動時にこのメタ情報から feval callable を再構築し、`lgb.train()` の `feval` パラメータに渡す

## 15.5 パッケージ配布（PyPI）

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

テストは以下の 10 カテゴリで構成する。各カテゴリは独立したテスト目的を持ち、組み合わせで回帰耐性を確保する。

### 18.1.1 基本テストカテゴリ

- **Golden test（契約固定）**: `FitResult / PredictionResult / RunMeta / SplitIndices` のフィールド名・型・構造を固定し、意図しない破壊的変更を検知する。
- **再現性テスト**: 同一 config + seed で `oof_pred`, `predict`, `metrics`, `split indices` が bit 一致する。`tune()` も同一 seed で `best_params` / `best_score` / trial 順序が一致する。
- **リーク防止テスト**: OOF が held-out データのみから生成されること、calibration が cross-fit で分離されていること、feature pipeline が train fold のみで fit されることを検証する。
- **列ズレテスト**: 余剰 / 不足 / unseen category のポリシー通り動く。カテゴリ順序ずれ（学習時と推論時で同一カテゴリだが出現順が異なる）もカバーする。
- **例外テスト**: 全 `ErrorCode` に対して少なくとも 1 テストが存在し、`context` dict の必須キーを検証する。
- **optional dependency テスト**: 未導入時の例外コード / メッセージが崩れない。全 optional dependency（optuna, shap, plotly, scipy）について "missing" パスを検証する。
- **Public API surface テスト**: `from lizyml import Model` 等のトップレベル公開面が壊れていないことを検証する。
- **バージョン一致テスト**: `lizyml.__version__` と配布メタデータのバージョンが一致することを検証する。
- **README サンプルコードテスト**: `README.md` に記載された最短利用例が `SyntaxError` / `ImportError` なく実行可能であることを検証する（データ依存部分はモック可）。

### 18.1.2 Config 伝搬・実効性テスト（H-0056 カテゴリ A + H-0063）

Config の各フィールドが最終的なコンポーネント（Booster params, split indices, pipeline state 等）に正しく到達し、**実際の動作に反映されている**ことを、**モックなしの observable outcome** で検証する。

**伝搬テスト**（値が到達すること）:

- Config → Booster params: `learning_rate`, `max_depth`, `seed`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l1`, `lambda_l2`, `max_bin`, `boosting`, `first_metric_only`, `metric`（H-0061）, 任意パラメータ透過 等が Booster の `params` dict に到達。
- Config → early_stopping: `rounds` が adapter の `early_stopping_rounds` に到達。`enabled=False` で `None` に。
- Config → features: `exclude` で列が除外される。`categorical` でカテゴリ認識される。
- Config → evaluation: `metrics` リストが FitResult.metrics のキーに反映される。
- Config → split: `n_splits` が fold 数に反映。`random_state` で fold が決定的に再現。`group_col` で group 制約が機能。
- Config → smart params: `auto_num_leaves` + `num_leaves_ratio` + `max_depth` の計算結果が Booster に到達。
- Config → task-locked: `objective` がタスクから固定。`num_class` が multiclass で自動注入。`verbosity` が `-1` 固定。

**実効性テスト**（値が動作に反映されること）:

- **2 値比較パターン**: 各 Booster パラメータについて、異なる値で fit → 予測が変わることを検証する。対象: `learning_rate`, `max_depth`, `n_estimators`, `max_bin`, `lambda_l1`, `lambda_l2`, `bagging_fraction`, `feature_fraction`, `boosting`, `metric`, `num_leaves`, `min_data_in_leaf`。
- **Smart Params 動作反映**: `feature_weights` → importance 順序変化、`balanced` → 不均衡データの予測分布変化、`scale_pos_weight` → 予測分布変化。
- **Training 実効性**: `early_stopping.random_state` → 同一 seed で同一 inner split、`validation_ratio` → inner valid サイズ比例。
- **Feature 実効性**: `auto_categorical` → string 列の自動検出。
- **Calibration 実効性**: `calibration.params` → calibrator パラメータ到達。

### 18.1.3 Facade オーケストレーションテスト（H-0056 カテゴリ A 関連）

`Model.fit()` が各コンポーネントを正しい順序・正しい引数で呼ぶことを検証する。

- CVTrainer と RefitTrainer が同一の `pipeline_factory` / `estimator_factory` / `ratio_resolver` を受け取る。
- Evaluator が Config 指定のメトリクスリストを受け取る。
- Calibration が `cfg.calibration is not None` かつ `task="binary"` の場合のみ実行される。non-binary で `CALIBRATION_NOT_SUPPORTED` を返す。
- `get_provider()` が model name で正しい provider を返す。未知の name で `CONFIG_INVALID`。
- `_merge_params` の優先順位: Config defaults < tune best < fit() args。

### 18.1.4 Artifact 互換テスト（H-0056 カテゴリ A）

Artifact の保存・復元が `format_version` 管理のもとで安全に動作することを検証する。

- **Frozen artifact fixture**: `tests/fixtures/` に CI 生成の artifact スナップショットを格納し、`Model.load()` → `predict()` の結果が既知の期待値と一致することを検証する。将来の `format_version` bump 時に migration テストの基盤となる。
- **Legacy calibration path**: `oof_raw_scores=None` の旧形式 artifact が probability 入力で calibrate される経路を検証する。
- **format_version rejection**: 未知の version（`99`, `0` 等）で `DESERIALIZATION_FAILED`。
- **metadata 部分欠損**: 必須フィールドを 1 つずつ削除し、各欠損で正しいエラーメッセージを検証。
- **Booster string roundtrip**: `model_to_string()` → `model_from_string()` 往復で predict 結果が一致。

### 18.1.5 Provider/Adapter 共通 Invariant チェック（H-0056 カテゴリ B）

scikit-learn の `check_estimator` に相当する、EstimatorProvider / BaseEstimatorAdapter の自動適合性テストスイート。新 provider 追加時に自動で全チェックが走る。

- **Protocol 適合**: `extract_model_params`, `extract_smart_params`, `build_estimator_factory`, `build_pipeline_factory`, `build_ratio_resolver`, `resolve_smart_params`, `runtime_deps`, `default_space`, `default_fixed_params`, `params_summary` の戻り値型チェック。
- **Factory → fit → predict 往復**: 全タスク型 × 全 provider で fit → predict が完走し出力 shape が正しい。
- **Pickle 往復**: fit 済み adapter を pickle → unpickle し predict 結果が一致。
- **Importance**: fit 後に `importance("split")` / `importance("gain")` が feature_names と同じキーの dict を返す。
- **データ多様性**: `dense_float_2col`, `dense_float_20col`, `mixed_dtype`（float + int + category）, `with_missing`（NaN 列）, `single_feature`（1列）, `high_cardinality_cat`（100+ unique）を横断。

### 18.1.6 Tuning 再現性・失敗マトリクス（H-0056 カテゴリ C）

Optuna の seed 固定ポリシーに準拠し、tuning 結果の再現性と失敗系パスを網羅する。

- **再現性**: 同一 seed で `best_params`, `best_score`, trial 順序が一致。
- **全 trial 失敗**: objective が常に例外 → `TUNING_FAILED` + 正しい context。
- **部分 trial 失敗**: 一部 trial のみ失敗 → 成功 trial の best が正しく返る。
- **NaN/inf 返却**: objective の異常値に対する挙動。
- **Search space と Config の衝突**: space の param が Config の同名 param を上書きすることの検証。

### 18.1.7 入力ソース・dtype・境界値の E2E（H-0056 カテゴリ D）

LightGBM/XGBoost が `all_x_types` / `all_y_types` で実施しているコンテナ型・dtype 差分テストに相当する。

- **入力ソース多様性**: CSV / Parquet 経由で fit → predict → export → load が完走する。
- **dtype 横断**: `float32`, `float64`, nullable `Int64`, `Float64`, `string` dtype で fit が正常動作するか、明確なエラーを返す。
- **境界値**: 0行 DataFrame, 1行 DataFrame, 重複列名, `inf`/`-inf` 含有列で明確なエラーメッセージ。
- **カテゴリ順序ずれ**: 学習時と推論時で同一カテゴリの出現順が異なる場合の挙動。

### 18.1.8 パラメータ組み合わせの Pairwise テスト（H-0056 カテゴリ E）

パラメータの相互作用バグを効率的に検出するため、全直積ではなく **Pairwise（2因子間カバレッジ）** で ~20-30 ケースを生成する。

- **因子**: `task` × `split_method` × `calibration` × `early_stopping` × `n_estimators`
- **検証**: 有効な組み合わせは fit 完走。無効な組み合わせは明確なエラー。
- **重要な相互作用の個別テスト**: `calibration + group_kfold`, `balanced + multiclass`, `feature_weights + auto_num_leaves`, `tuning + calibration`, `n_estimators=1 + early_stopping`, `exclude + categorical` 等。

### テスト基盤方針（H-0043）

- **共通ヘルパーの集約**: データ生成ヘルパー（`make_regression_df()`, `make_binary_df()`, `make_multiclass_df()`, `make_config()` 等）は `tests/_helpers.py` に集約する。各テストファイルでのローカル重複定義を排除する。データ多様性 fixture（`dense_float_20col`, `mixed_dtype`, `with_missing` 等）も同ファイルに追加する。
- **parametrize の活用**: タスク横断テスト（regression/binary/multiclass）は `@pytest.mark.parametrize` で統合し、テストロジックの重複を削減する。Provider 適合性テストは `@pytest.mark.parametrize("check", ALL_CHECKS)` で自動展開する。
- **slow テストの分離**: `@pytest.mark.slow` 付きテスト（notebook 実行等）はローカル開発時にデフォルトスキップする（`addopts = "-m 'not slow'"`）。CI の main PR では全テストを実行し、develop PR では slow を除外する。
- **カバレッジ閾値**: CI で `--cov-fail-under=95` を設定し、カバレッジ回帰を防止する。
- **Frozen artifact の管理**: `tests/fixtures/` に格納する artifact スナップショットは、`format_version` bump 時に新旧両方を保持し migration テストに使用する。生成スクリプトを `tests/fixtures/generate_fixtures.py` に置く。

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
- `develop` および `main` ブランチへの PR で CI を実行する。`develop` PR では slow テストを除外し、`main` PR では全テストを実行する（H-0043）。

# 19. ディレクトリ構成

5 層カテゴリアーキテクチャ（§2.1）に基づく。各ディレクトリの所属 Layer を明示する。

```text
lizyml/
│
├── __init__.py                     公開面 (Model, FitResult, PredictionResult, ...)
│
├── core/                           ── Layer 0: Foundation ──
│   ├── exceptions.py               LizyMLError + ErrorCode
│   ├── logging.py                  logger + run_id + output_dir
│   ├── registries.py               MetricRegistry, CalibratorRegistry
│   └── types/
│       ├── fit_result.py           FitResult
│       ├── predict_result.py       PredictionResult
│       ├── tuning_result.py        TuningResult, TrialResult
│       ├── artifacts.py            RunMeta, SplitIndices, DataFingerprint
│       └── search_dim.py           SearchDim, FloatDim, IntDim, CategoricalDim, DimCategory
│
│                                   ── Layer 0/4: Facade (core/ 内の特殊位置) ──
│   ├── model.py                    Model facade (組み立てと委譲のみ)
│   ├── _model_factories.py         splitter / inner_valid / estimator provider 構築
│   ├── _model_plots.py             ModelPlotsMixin
│   ├── _model_tables.py            ModelTablesMixin (EstimatorProvider 経由)
│   ├── _model_metrics.py           _has_metric_content, _filter_metrics
│   ├── _model_persistence.py       ModelPersistenceMixin
│   ├── train_components.py         TrainComponents (frozen dataclass)
│   ├── seed.py                     seed 固定ユーティリティ
│   └── specs/
│       ├── problem_spec.py         ProblemSpec (data/ が使用)
│       └── feature_spec.py         FeatureSpec (data/ が使用)
│
├── config/                         ── Layer 1: Config ──
│   ├── schema.py                   pydantic schemas (extra="forbid")
│   └── loader.py                   YAML/JSON/dict → LizyMLConfig
│
├── data/                           ── Layer 1: Data ──
│   ├── datasource.py               CSV / Parquet / DataFrame
│   ├── dataframe_builder.py        X/y/groups 分離 + categorical
│   └── fingerprint.py              DataFingerprint 計算 (compute 関数)
│
├── splitters/                      ── Layer 1: Splitting ──
│   ├── base.py                     BaseSplitter
│   ├── kfold.py                    KFoldSplitter, StratifiedKFoldSplitter
│   ├── group_kfold.py              GroupKFoldSplitter, StratifiedGroupKFoldSplitter
│   ├── time_series.py              TimeSeriesSplitter
│   ├── purged_time_series.py       PurgedTimeSeriesSplitter
│   └── group_time_series.py        GroupTimeSeriesSplitter
│
├── features/                       ── Layer 1: Features ──
│   ├── pipeline_base.py            BaseFeaturePipeline
│   ├── pipelines_native.py         NativeFeaturePipeline
│   ├── encoders/
│   │   └── categorical_encoder.py  カテゴリ処理部品
│   └── transformers/
│       └── feature_transformer.py  特徴量変換 (passthrough 拡張点)
│
├── estimators/                     ── Layer 1: Estimators ──
│   ├── base.py                     BaseEstimatorAdapter
│   ├── provider.py                 EstimatorProvider protocol (§14.4)
│   └── lgbm/                       LightGBM 実装 (サブパッケージ)
│       ├── __init__.py             LGBMAdapter, LGBMProvider を re-export
│       ├── adapter.py              LGBMAdapter
│       ├── provider.py             LGBMProvider (EstimatorProvider 実装)
│       ├── smart_params.py         resolve_smart_params / resolve_ratio_params
│       └── defaults.py             _COMMON_DEFAULTS / default_space / default_fixed_params
│
├── metrics/                        ── Layer 1: Metrics ──
│   ├── base.py                     BaseMetric
│   ├── registry.py                 MetricRegistry helpers + task validation
│   ├── regression.py               RMSE, MAE, R2, RMSLE, MAPE, Huber
│   └── classification.py           LogLoss, AUC, AUCPR, F1, Accuracy, Brier, ECE, PrecisionAtK
│
├── calibration/                    ── Layer 1: Calibration ──
│   ├── base.py                     BaseCalibratorAdapter
│   ├── cross_fit.py                cross_fit_calibrate + CalibrationResult
│   ├── registry.py                 get_calibrator
│   ├── platt.py                    PlattCalibrator
│   ├── isotonic.py                 IsotonicCalibrator
│   └── beta.py                     BetaCalibrator
│
├── training/                       ── Layer 2: Training ──
│   ├── cv_trainer.py               CVTrainer (outer CV loop)
│   ├── refit_trainer.py            RefitTrainer + RefitResult
│   ├── inner_valid.py              BaseInnerValidStrategy + 6 concrete
│   └── oof_assembly.py             fill_oof / get_fold_pred / init_oof
│
├── evaluation/                     ── Layer 2: Evaluation ──
│   ├── evaluator.py                Evaluator (raw metrics のみ)
│   ├── table_formatter.py          evaluate_table 整形
│   ├── confusion.py                confusion_matrix_table
│   └── thresholding.py             threshold 最適化ユーティリティ
│
├── tuning/                         ── Layer 2: Tuning ──
│   ├── tuner.py                    Tuner (Optuna study management)
│   └── search_space.py             SearchDim, parse/suggest/split_by_category
│
├── explain/                        ── Layer 3: Explain (optional) ──
│   └── shap_explainer.py           compute_shap_values / compute_shap_importance
│
├── plots/                          ── Layer 3: Plots (optional) ──
│   ├── importance.py               feature importance bar chart
│   ├── learning_curve.py           training/validation loss curve
│   ├── oof_distribution.py         OOF prediction distribution
│   ├── residuals.py                scatter / histogram / QQ
│   ├── classification.py           ROC curve
│   ├── calibration.py              reliability diagram + probability histogram
│   └── tuning.py                   tuning history plot
│
└── persistence/                    ── Layer 3: Persistence ──
    ├── exporter.py                 export() + AnalysisContext + FORMAT_VERSION
    └── loader.py                   load() + format_version validation
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

- OOF / IF 生成ロジック（`training/oof_assembly.py`）
- metric 計算（`evaluation/evaluator.py`）
- estimator 固有処理（`estimators/<name>/` — EstimatorProvider 経由で委譲）
- plot 実装本体（`plots/*`）
- 保存形式の詳細（`persistence/*`）

## 実装メモ

- `core/model.py` は組み立て専用とし、ロジックを持たせない。
- `Model` クラスは mixin で構成する（H-0042）。plot 系は `_model_plots.py`、table/accessor 系は `_model_tables.py`、persistence 系は `_model_persistence.py` に分割し、`model.py` には core lifecycle（`__init__`, `fit`, `predict`, `evaluate`, `tune`）とプライベートヘルパーのみを残す。
- mixin は `_` プレフィックスの非公開モジュールとし、`Model` の import パス（`lizyml.core.model.Model`）は変更しない。
- 依存関係の切り離しが必要な箇所では Lazy Import を許容する。
