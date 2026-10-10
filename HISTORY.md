# HISTORY.md

仕様変更の提案・決定・廃止の履歴。1変更につき1エントリ。

---

## 2026-03-04: PyPI 配布要件の明文化

- ID: `H-0000`
- Status: `accepted`
- Scope: `Config | Packaging`
- Related: `BLUEPRINT.md §15.4, §18.2`

### Context

PyPI 公開を前提にした場合、build 定義・配布メタデータ・README・optional dependency・CI 検証の要件が明文化されていないとリリース品質がぶれる。

### Proposal

- `pyproject.toml` に PEP 517/518 準拠の `[build-system]` を定義し、`sdist / wheel` を生成できるようにする。
- `[project]` に name / version / description / readme / requires-python / license / authors / classifiers / urls を必須で記載する。
- optional dependency を `[project.optional-dependencies]`（配布利用者向け）と `[dependency-groups]`（開発者向け）に分離する。
- `README.md` の import 例を実際のパッケージ名と一致させる。
- `py.typed` を同梱して PEP 561 に準拠する。

### Impact

- `pyproject.toml` / `README.md` の変更のみ。公開 API の shape は変更しない。

### Compatibility

- 破壊的変更なし。配布契約とドキュメント契約を追加するのみ。

### Alternatives Considered

- 実装時に都度判断し仕様に書かない → 担当者ごとの判断に依存してリリース品質がぶれるため却下。

### Acceptance Criteria

- `uv build` で sdist / wheel が生成できる。
- `twine check` が PASSED になる。
- `lizyml/py.typed` が存在する。
- README の import 例がパッケージ名と一致している。
- BLUEPRINT §15.4 / §18.2 に要件が追加されている。

### Decision

- Date: `2026-03-04`
- Result: `accepted`
- Notes: BLUEPRINT §15.4 / §18.1 / §18.2 に反映済み。`fix/phase-0-pypi-compliance` ブランチで実施。

---

## 2026-03-04: Config Schema の全フィールド確定

- ID: `H-0001`
- Status: `accepted`
- Scope: `Config`
- Related: `BLUEPRINT.md §5, §3.3`

### Context

Phase 2 でpydantic v2 スキーマを実装する前に、LizyMLConfig の全フィールドとバリデーション方針を仕様として固定する必要がある。未確定のままスキーマを実装すると、後から Config のキーや型を変更するたびに破壊的変更が生じる。

### Proposal

`LizyMLConfig`（トップレベル）の全フィールドと各 sub-config を以下の通り確定する。

#### トップレベル

```python
class LizyMLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config_version: int                      # 必須。将来の config schema 変更を追跡
    task: Literal["regression", "binary", "multiclass"]
    data: DataConfig
    features: FeaturesConfig
    split: SplitConfig
    model: Annotated[ModelConfig, Field(discriminator="name")]  # lgbm / (将来 sklearn 等)
    training: TrainingConfig
    tuning: Optional[TuningConfig] = None
    evaluation: EvaluationConfig
    calibration: Optional[CalibrationConfig] = None
```

#### DataConfig

```python
class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str | None = None          # CSV / Parquet ファイルパス（DataFrame 渡し時は None）
    target: str                      # 目的変数列名
    time_col: str | None = None      # 時系列列名（時系列分割時に必須）
    group_col: str | None = None     # グループ列名（グループ分割時に必須）
```

#### FeaturesConfig

```python
class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exclude: list[str] = []          # 学習から除外する列
    auto_categorical: bool = True    # 非数値列を自動でカテゴリ扱いにする
    categorical: list[str] = []      # 明示的にカテゴリ指定する列
```

#### SplitConfig（discriminated union）

```python
class KFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["kfold"]
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True

class StratifiedKFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["stratified_kfold"]
    n_splits: int = 5
    random_state: int = 42

class GroupKFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["group_kfold"]
    n_splits: int = 5

class TimeSeriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["time_series"]
    n_splits: int = 5
    gap: int = 0

SplitConfig = Annotated[
    KFoldConfig | StratifiedKFoldConfig | GroupKFoldConfig | TimeSeriesConfig,
    Field(discriminator="method"),
]
```

#### ModelConfig（discriminated union）

```python
class LGBMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["lgbm"]
    params: dict[str, Any] = {}

ModelConfig = Annotated[LGBMConfig, Field(discriminator="name")]
```

#### TrainingConfig

```python
class HoldoutInnerValidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["holdout"]
    ratio: float = 0.1
    random_state: int = 42

class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    rounds: int = 50
    inner_valid: HoldoutInnerValidConfig | None = None
    validation_ratio: float | None = None  # inner_valid.ratio のエイリアス (H-0010)

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 42
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
```

#### TuningConfig

```python
class OptunaParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_trials: int = 50
    direction: Literal["minimize", "maximize"] = "minimize"
    timeout: float | None = None

class OptunaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    params: OptunaParamsConfig = OptunaParamsConfig()
    space: dict[str, Any] = {}

class TuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optuna: OptunaConfig = OptunaConfig()
```

#### EvaluationConfig

```python
class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics: list[str] = []          # 例: ["rmse", "mae"]
```

#### CalibrationConfig

```python
class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["platt", "isotonic", "beta"] = "platt"
    n_splits: int = 5                # calibration cross-fit の fold 数
```

#### バリデーション方針

- 全 sub-config に `extra="forbid"` を適用してタイポを必ずエラー化する。
- `config_version` は必須とし、将来のスキーマ変更時に `CONFIG_VERSION_UNSUPPORTED` で拒否できるようにする。
- 不正な Config は `LizyMLError(CONFIG_INVALID)` として統一的に扱う（pydantic の ValidationError をラップ）。
- loader 層で alias 正規化（例: `k-fold` → `kfold`）を行い、スキーマ validate 前に適用する。
- 環境変数 override: `LIZYML__` prefix、`__` でネスト区切り（例: `LIZYML__model__lgbm__params__learning_rate=0.01`）。

### Impact

- `lizyml/config/schema.py` の新規実装。
- `lizyml/config/loader.py` の新規実装。
- `lizyml/core/specs/` 以下の Spec クラス群の新規実装。
- `lizyml/core/registries.py` の新規実装。
- `tests/test_config/` の新規テスト群。

### Compatibility

- 新規実装につき既存コードへの破壊的影響なし。
- 将来 Config スキーマを変更する際は `config_version` を上げ、migration 方針を本 HISTORY.md に記録する。

### Alternatives Considered

- marshmallow / cerberus 等の他バリデーションライブラリ → pydantic v2 は `extra="forbid"` とdiscriminated union により typo 検知と型安全性が高いため採用。
- Config を dict のまま扱う → 未知キーを検知できず契約が壊れるため却下。

### Acceptance Criteria

- `LizyMLConfig` が正常 dict から生成できる。
- 未知キー混入時に `CONFIG_INVALID` が返る。
- `config_version` 欠落時に ValidationError が返る。
- YAML / JSON / dict 各形式からのロードが成功する。
- 環境変数 override が動作する。
- alias 正規化（`k-fold` → `kfold`）が機能する。
- Config → 各 Spec 変換の網羅テストが通過する。

---

## 2026-03-04: FitResult / PredictionResult / Artifacts の全フィールド確定

- ID: `H-0002`
- Status: `accepted`
- Scope: `Artifacts`
- Related: `BLUEPRINT.md §7`

### Context

Phase 4 でデータクラスを実装する前に、FitResult / PredictionResult / Artifacts の全フィールドと shape・意味・階層を仕様として固定する。スキーマ確定前に実装すると後からの変更が破壊的変更となり format_version を上げる必要が生じる。

### Proposal

#### FitResult

```python
@dataclass
class FitResult:
    oof_pred: np.ndarray          # shape: (n_samples,) regression/binary, (n_samples, n_classes) multiclass
    if_pred_per_fold: list[np.ndarray]  # len == n_splits, 各要素は train fold 全体の予測
    metrics: dict                  # {"raw": {"oof": {...}, "if_mean": {...}, "if_per_fold": [...]},
                                   #  "calibrated": {...}}  # binary + calibrator 有効時のみ
    models: list[Any]              # fold ごとのモデル（EstimatorAdapter 内包）
    history: list[dict]            # per-fold: {"eval_history": ..., "best_iteration": int}
    feature_names: list[str]       # 学習に使用した特徴量名（順序固定）
    dtypes: dict[str, str]         # 特徴量名 → dtype 文字列
    categorical_features: list[str]  # カテゴリ特徴量名
    splits: SplitIndices           # 外側 CV / inner valid / calibration の全 indices
    data_fingerprint: DataFingerprint  # データ同一性の検証用
    pipeline_state: Any            # FeaturePipeline の保存状態
    calibrator: Any | None         # binary + calibration 有効時のみ
    run_meta: RunMeta              # バージョン・Config 情報
```

#### PredictionResult

```python
@dataclass
class PredictionResult:
    pred: np.ndarray               # shape: (n_samples,)
    proba: np.ndarray | None       # binary のみ、shape: (n_samples,)
    shap_values: np.ndarray | None # 要求時のみ、shape: (n_samples, n_features)
    used_features: list[str]       # 実際に使用した特徴量名
    warnings: list[str]            # 列ズレ等の補正通知
```

#### SplitIndices

```python
@dataclass
class SplitIndices:
    outer: list[tuple[np.ndarray, np.ndarray]]  # fold ごとの (train_idx, valid_idx)
    inner: list[tuple[np.ndarray, np.ndarray]] | None  # fold ごとの inner valid
    calibration: list[tuple[np.ndarray, np.ndarray]] | None  # calibration CV indices
```

#### RunMeta

```python
@dataclass
class RunMeta:
    lizyml_version: str
    python_version: str
    deps_versions: dict[str, str]   # {"lightgbm": "4.x.x", "pydantic": "2.x.x", ...}
    config_normalized: dict          # ロード時に正規化済みの Config dict
    config_version: int
    run_id: str                      # UUID
    timestamp: str                   # ISO 8601
```

#### metrics の階層（固定）

```python
{
    "raw": {
        "oof": {"rmse": float, "mae": float, ...},
        "if_mean": {"rmse": float, ...},
        "if_per_fold": [{"rmse": float, ...}, ...]   # len == n_splits
    },
    "calibrated": {  # binary + calibrator 有効時のみ存在
        "oof": {...},
        "if_mean": {...},
        "if_per_fold": [...]
    }
}
```

### Impact

- `lizyml/core/types/fit_result.py` の新規実装。
- `lizyml/core/types/predict_result.py` の新規実装。
- `lizyml/core/types/artifacts.py` の新規実装。
- `lizyml/core/types.py` の re-export。
- `tests/test_core/test_contracts.py` のゴールデンテスト。

### Compatibility

- 新規実装につき既存コードへの破壊的影響なし。
- 将来フィールドを追加する場合は format_version を上げ、本 HISTORY.md に migration を記録する。

### Alternatives Considered

- pydantic モデルで FitResult を定義する → np.ndarray 等を含む大型 dataclass には dataclass が適切。pydantic は Config 層に限定する。
- 動的 dict で返す → 型安全性がなく、ゴールデンテストでスキーマを固定できないため却下。

### Acceptance Criteria

- `FitResult` / `PredictionResult` のフィールド名・型が定義通りであることをゴールデンテストで固定する。
- `metrics` の階層 `raw/oof`, `raw/if_mean`, `raw/if_per_fold` が必ず存在することを検証する。
- スキーマ変更時にゴールデンテストが意図的に落ちることを確認する（テスト自体の有効性の検証）。

---

## 2026-03-04: Persistence / Export フォーマット仕様の確定

- ID: `H-0003`
- Status: `accepted`
- Scope: `Artifacts | Export`
- Related: `BLUEPRINT.md §14, §15.4`

### Context

Phase 14 で `Model.export()` / `Model.load()` を実装する前に、保存フォーマット・`format_version` の意味・将来の破壊的変更に対する migration 方針を仕様として固定する。未確定のまま実装すると、フォーマット変更のたびに無方針の破壊的変更が発生する。

### Proposal

#### ディレクトリ構造

```
{path}/
  metadata.json          # format_version, lizyml_version, timestamp, config, metrics, run_id
  fit_result.pkl         # FitResult dataclass (joblib 圧縮)
  refit_model.pkl        # RefitResult dataclass (joblib 圧縮)
```

#### metadata.json スキーマ（v1）

```json
{
  "format_version": 1,
  "lizyml_version": "0.1.0",
  "python_version": "3.11.x",
  "timestamp": "2026-03-04T12:00:00",
  "run_id": "uuid4",
  "config": { ... },
  "metrics": { ... },
  "feature_names": ["feat_a", "feat_b"],
  "task": "regression"
}
```

#### format_version の取り扱い

- `format_version = 1` を初版とする。
- フィールドの追加はマイナー変更（後方互換）とし、format_version を上げない。
- フィールドの削除・型変更・意味変更は破壊的変更とし、format_version を上げる。
- ロード時に `format_version` が未知の場合は `DESERIALIZATION_FAILED` を返す。

#### セキュリティ方針

- `.pkl` ファイルは joblib で保存・復元。
- `Model.load()` のドキュメントに「信頼できる出所からのみロードすること」を明記する。
- `metadata.json` のバリデーション（format_version / task / feature_names）をロード時に必ず実行する。

### Impact

- `lizyml/persistence/exporter.py`: `export(model, path)` の新規実装。
- `lizyml/persistence/loader.py`: `load(path) -> Model` の新規実装。
- `lizyml/core/model.py`: `export()` / `load()` の NotImplementedError を実装に置き換え。
- `tests/test_persistence/test_persistence.py`: export → load → predict E2E テスト。

### Compatibility

- 新規実装につき既存コードへの破壊的影響なし。
- 将来 format_version を上げる場合は本 HISTORY.md に migration エントリを追記する。

### Alternatives Considered

- 単一 `.pkl` に全情報を保存 → metadata.json を分離しておくことで version 確認・human-readable なメタ参照が可能になるため分離を採用。
- ONNX や PMML 形式 → LizyML 固有の FitResult / Artifacts の完全な復元には向かないため却下（将来の軽量 export フォーマットとして追加検討）。

### Acceptance Criteria

- `model.export(path)` でディレクトリが生成され、`metadata.json` / `fit_result.pkl` / `refit_model.pkl` が存在する。
- `Model.load(path)` でロードし、`predict()` が元モデルと同じ結果を返す。
- `format_version` が未知の場合に `DESERIALIZATION_FAILED` が返る。
- `metadata.json` に必須フィールド不足の場合に `DESERIALIZATION_FAILED` が返る。

### Decision

- Date: `2026-03-04`
- Result: `accepted`
- Notes: Phase 14 の実装前提として受け入れ。`format_version=1` を初版とする。

### Migration

- `format_version=1` から `format_version=2` への移行が必要になった場合、`lizyml/persistence/migrations/v1_to_v2.py` を追加し、ロード時に自動マイグレーションを試みる（または明示的エラーで移行を促す）。（H-0111 注記: `migrations/` パッケージは作られなかった。v1 の artifact は `lizyml/persistence/loader.py` の中でメモリ上で v2 に引き上げる（H-0070、BLUEPRINT §15.2）。#318）

---

## 2026-03-04: 回帰メトリクス MAPE・Huber Loss の追加

- ID: `H-0004`
- Status: `accepted`
- Scope: `Metrics`
- Related: `BLUEPRINT.md §7`

### Context

Tutorial Notebook でよく使われる回帰メトリクス（MAPE・Huber Loss）が未実装のため、チュートリアルでの利用および実務での利用に制限がある。

### Proposal

- `lizyml/metrics/regression.py` に `MAPE`・`HuberLoss` クラスを追加する。
- `MAPE`: 分母がゼロの場合は `UNSUPPORTED_METRIC` エラーを返す。
- `HuberLoss`: `delta=1.0` をデフォルトとし、コンストラクタで設定可能にする。Config 文字列では `"huber"` で `delta=1.0` として登録する。
- `lizyml/metrics/registry.py` の `_TASK_METRICS["regression"]` に `"mape"`, `"huber"` を追加する。
- 既存メトリクスへの影響なし（追加のみ）。

### Impact

- `lizyml/metrics/regression.py`: MAPE・HuberLoss クラス追加。
- `lizyml/metrics/__init__.py`: エクスポート追加。
- `lizyml/metrics/registry.py`: `_TASK_METRICS["regression"]` 更新。
- `tests/metrics/test_regression_metrics.py`: 新規テストファイル。

### Compatibility

- 既存の `"rmse"`, `"mae"`, `"r2"`, `"rmsle"` への影響なし。
- `format_version` 変更不要。

### Alternatives Considered

- SMAPE（対称 MAPE）を代わりに実装する → MAPE の方が一般的なため MAPE を優先し、SMAPE は将来の拡張候補とする。

### Acceptance Criteria

- `evaluate(metrics=["mape", "huber"])` が回帰タスクで正常に動作する。
- MAPE: y_true にゼロが含まれる場合に `LizyMLError(UNSUPPORTED_METRIC)` が返る。
- HuberLoss: 誤差が delta 以下の場合に二乗損失、超える場合に線形損失となることをテストで確認する。

### Decision

- Date: `2026-03-04`
- Result: `accepted`
- Notes: Tutorial Notebook の要件として受け入れ。

---

## 2026-03-04: model.evaluate_table() の追加

- ID: `H-0005`
- Status: `accepted`
- Scope: `Evaluation | Public API`
- Related: `BLUEPRINT.md §4.1, §13.2`

### Context

Notebook で評価結果を確認する際、`evaluate()` が返す nested dict を手作業で DataFrame 化する必要があり、「ユーザーにコードを書かせない」思想に反する。

### Proposal

- `Model.evaluate_table()` を追加し、`evaluate()` の dict を `pd.DataFrame` に整形して返す。
- 行 = メトリクス名、列 = `oof`, `if_mean`, `fold_0`...`fold_N-1`。calibrated がある場合は `cal_oof` 列を追加。
- ロジックは `lizyml/evaluation/table_formatter.py` に配置（Model にロジックを置かない原則を遵守）。

### Impact

- `lizyml/evaluation/table_formatter.py`: 新規。
- `lizyml/core/model.py`: `evaluate_table()` メソッド追加。
- `tests/test_evaluation/test_table_formatter.py`: 新規テスト。

### Compatibility

- FitResult / PredictionResult / Artifacts / format_version 変更なし。非破壊的追加。

### Alternatives Considered

- `evaluate()` の返り値自体を DataFrame にする → 既存契約の破壊になるため却下。

### Acceptance Criteria

- `model.evaluate_table()` が fit 後に DataFrame を返す。
- 行 = メトリクス名、列に oof / if_mean / fold 別が含まれる。
- calibrated 有りの場合 cal_oof 列が追加される。
- fit 前に呼ぶと MODEL_NOT_FIT。

### Decision

- Date: `2026-03-04`
- Result: `accepted`
- Notes: Notebook の UX 改善として受け入れ。

---

## 2026-03-04: model.residuals() / model.residuals_plot() の追加

- ID: `H-0006`
- Status: `accepted`
- Scope: `Public API | Plots`
- Related: `BLUEPRINT.md §4.1, §13.3`

### Context

BLUEPRINT §4.1 で `residuals()` / `residuals_plot()` が計画されていたが未実装。回帰タスクの残差分析はモデル診断の基本であり、Notebook でワンコールで可視化できる必要がある。

### Proposal

- `Model.residuals()`: 回帰タスク専用。`y - oof_pred` を `np.ndarray` で返す。
- `Model.residuals_plot()`: ヒストグラム + QQ plot の 2 パネルを Plotly で表示。
- `fit()` 中に `self._y` を一時保持（export/persistence には含めない）。
- `Model.load()` 後は y が不在のため呼び出し不可（MODEL_NOT_FIT エラー）。
- binary/multiclass では `UNSUPPORTED_TASK` を返す。
- プロット実装は `lizyml/plots/residuals.py` に配置。

### Impact

- `lizyml/core/model.py`: `_y` フィールド追加、`residuals()` / `residuals_plot()` メソッド追加。
- `lizyml/plots/residuals.py`: 新規。
- `tests/test_plots/test_residuals.py`: 新規テスト。

### Compatibility

- FitResult / format_version 変更なし。`_y` は Model の一時状態であり Artifacts に含めない。

### Alternatives Considered

- FitResult に y_true を保存する → Artifacts 契約の変更になるため却下。y はユーザーデータであり、モデル成果物ではない。
- load 後も利用可能にするため y を export に含める → データ漏洩リスクがあるため却下。

### Acceptance Criteria

- `model.residuals()` が回帰タスクで `(n_samples,)` の ndarray を返す。
- `model.residuals_plot()` が Plotly Figure を返す（ヒストグラム + QQ plot）。
- binary/multiclass で UNSUPPORTED_TASK。
- load 後に呼ぶと MODEL_NOT_FIT。

### Decision

- Date: `2026-03-04`
- Result: `accepted`
- Notes: 回帰タスクの基本診断機能として受け入れ。

---

## 2026-03-04: model.importance(kind="shap") / model.importance_plot(kind="shap") の追加

- ID: `H-0007`
- Status: `accepted`
- Scope: `Public API | Explain`
- Related: `BLUEPRINT.md §4.1, §14.1`

### Context

BLUEPRINT §4.1 で `importance(kind="shap")` が計画されていたが未実装。SHAP ベースの特徴量重要度は split/gain よりモデル非依存な指標であり、Notebook でワンコールで可視化できる必要がある。

### Proposal

- `Model.importance(kind="shap")`: fold ごとの validation データで SHAP を計算し、mean(|SHAP|) を fold 平均して `dict[str, float]` で返す。
- `Model.importance_plot(kind="shap")`: 上記 dict を Plotly 横棒グラフで表示。
- `fit()` 中に `self._X` を一時保持（export/persistence には含めない）。
- `Model.load()` 後は X が不在のため呼び出し不可（MODEL_NOT_FIT エラー）。
- `compute_shap_importance()` を `lizyml/explain/shap_explainer.py` に追加。
- `plot_importance_from_dict()` を `lizyml/plots/importance.py` に追加。
- shap は optional dependency（既存パターン踏襲）。

### Impact

- `lizyml/core/model.py`: `_X` フィールド追加、`importance()` / `importance_plot()` の kind="shap" 対応。
- `lizyml/explain/shap_explainer.py`: `compute_shap_importance()` 追加。
- `lizyml/plots/importance.py`: `plot_importance_from_dict()` 追加。
- `tests/test_explain/`: SHAP importance テスト追加。

### Compatibility

- FitResult / format_version 変更なし。`_X` は Model の一時状態。

### Alternatives Considered

- refit モデル + 全データで SHAP を計算 → CV の fold 構造を無視するため却下。fold 別 validation データで計算する方が CV philosophy に整合する。

### Acceptance Criteria

- `model.importance(kind="shap")` が `dict[str, float]` を返し、全 feature を含む。
- `model.importance_plot(kind="shap")` が Plotly Figure を返す。
- load 後に呼ぶと MODEL_NOT_FIT。
- shap 未インストール時に OPTIONAL_DEP_MISSING。

### Decision

- Date: `2026-03-04`
- Result: `accepted`
- Notes: SHAP 重要度の可視化機能として受け入れ。

---

## 2026-03-04: 全プロットの Plotly 移行

- ID: `H-0008`
- Status: `accepted`
- Scope: `Plots | Optional Dependency`
- Related: `BLUEPRINT.md §13.3`

### Context

matplotlib ベースのプロットは静的で Notebook 上での視認性・操作性に劣る。Plotly に移行することでインタラクティブなプロットを提供し、UX を向上させる。

### Proposal

- `pyproject.toml` の optional dependency `plots` グループを `matplotlib>=3.7` → `plotly>=5.0` に変更。
- `dependency-groups` (dev) も同様に変更。
- 既存 3 ファイル（`importance.py`, `learning_curve.py`, `oof_distribution.py`）を Plotly に書き換え。
- 新規ファイル（`residuals.py`）は最初から Plotly で実装。
- optional dep sentinel を `_mpl` → `_plotly` に変更。
- 返り値型を `matplotlib.figure.Figure` → `plotly.graph_objects.Figure` に変更。

### Impact

- `pyproject.toml`: optional dependency 変更。
- `lizyml/plots/importance.py`: Plotly 移行。
- `lizyml/plots/learning_curve.py`: Plotly 移行。
- `lizyml/plots/oof_distribution.py`: Plotly 移行。
- `tests/test_plots/test_plots.py`: Plotly Figure アサーションに更新。
- mypy overrides: `matplotlib.*` → `plotly.*`。

### Compatibility

- plot メソッドの返り値型が変わる破壊的変更。ただし plots は optional 機能であり、0.x バージョンのため許容する。

### Alternatives Considered

- デュアルサポート（matplotlib + plotly 両対応）→ 保守コストが倍増するため却下。
- 新機能のみ Plotly → ライブラリ内で可視化の一貫性が失われるため却下。

### Acceptance Criteria

- 全プロットメソッドが Plotly Figure を返す。
- plotly 未インストール時に OPTIONAL_DEP_MISSING。
- 既存テストが Plotly Figure アサーションで通過。

### Decision

- Date: `2026-03-04`
- Result: `accepted`
- Notes: UX 向上のため全面移行を受け入れ。

---

## 2026-03-04: residuals_plot() の拡張（散布図追加・kind 引数・IS/OOS 比較）

- ID: `H-0009`
- Status: `accepted`（実装済み。H-0111 で Status を訂正、#319）
- Scope: `Public API | Plots`
- Related: `BLUEPRINT.md §4.1, §13.3`

### Context

H-0006 で `residuals_plot()` を実装したが、以下の不足がある。

1. Actual vs Predicted 散布図が未実装。
2. 常に 2 パネル（histogram + QQ）が表示され、個別選択できない。
3. In-Sample（IF）と Out-of-Sample（OOF）の傾向比較ができない。

### Proposal

- `residuals_plot(kind=...)` に `kind` 引数を追加する。
  - `"scatter"`: Actual vs Predicted 散布図（x=predicted, y=actual）。IS と OOS を色分けオーバーレイ。y=x の完全予測参照線。
  - `"histogram"`: 残差ヒストグラム。IS と OOS を色分けオーバーレイ。mean/std アノテーション（OOS のみ）。
  - `"qq"`: QQ plot（OOS 残差のみ）。45 度参照線。
  - `"all"`: 上記 3 つを横並びサブプロットで表示（デフォルト）。
- 内部関数 `plot_residuals()` のシグネチャを変更し、`FitResult` + `y_true` を受け取る形式に統一する（他の plot 関数と同じパターン）。
- IS データは `fit_result.if_pred_per_fold[i]` + `fit_result.splits.outer[i][0]`（train_idx）から組み立てる。
- `kind` の値が不正な場合は `LizyMLError(INVALID_CONFIG)` を返す。（H-0111 注記: `INVALID_CONFIG` という `ErrorCode` は無く、実装は `CONFIG_INVALID` を送出する。#319）

### Impact

- `lizyml/plots/residuals.py`: シグネチャ変更 + 3 プロット実装。
- `lizyml/core/model.py`: `residuals_plot(kind=...)` 引数追加。
- `tests/test_plots/test_residuals.py`: 新シグネチャ対応 + kind 別テスト追加。

### Compatibility

- `Model.residuals_plot()` のデフォルト `kind="all"` により、引数なし呼び出しは引き続き動作する。ただしパネル構成が 2 パネル（histogram + QQ）→ 3 パネル（scatter + histogram + QQ）に変わる。
- 内部関数 `plot_residuals()` のシグネチャは破壊的変更だが、内部 API のため影響は限定的。

### Alternatives Considered

- `residuals_plot()` とは別に `residuals_scatter()` を追加する → API が増えすぎるため却下。`importance_plot(kind=...)` と同じパターンに統一する。
- IS/OOS 比較を別メソッドにする → 同一グラフ上でのオーバーレイが最も直感的なため、`kind` で制御する方式を採用。

### Acceptance Criteria

- `model.residuals_plot(kind="scatter")` が Actual vs Predicted の Plotly Figure を返し、IS/OOS 両方のトレースと y=x 参照線を含む。
- `model.residuals_plot(kind="histogram")` が IS/OOS オーバーレイのヒストグラムを返す。
- `model.residuals_plot(kind="qq")` が QQ plot を返す。
- `model.residuals_plot(kind="all")` が 3 サブプロットの Figure を返す。
- `model.residuals_plot()` がデフォルトで `kind="all"` として動作する。
- 不正な kind 値で `INVALID_CONFIG` エラーが返る。（H-0111 注記: `CONFIG_INVALID`。#319）

---

## 2026-03-04: EarlyStoppingConfig に validation_ratio エイリアス追加

- ID: `H-0010`
- Status: `superseded`（実装後、H-0069 が置き換えた。H-0111 で Status を訂正、#319）
- Scope: `Config`
- Related: `BLUEPRINT.md §5.2, HISTORY.md H-0001`

### Context

現在の early stopping 設定は `early_stopping.inner_valid.ratio` で指定するが、ネストが深く冗長。`validation_ratio` エイリアスを追加して簡略化する。

### Proposal

- `EarlyStoppingConfig` に `validation_ratio: float | None = None` フィールドを追加する。
- `validation_ratio` 指定時、内部で `HoldoutInnerValidConfig(method="holdout", ratio=validation_ratio)` を自動生成する。
- `inner_valid` と `validation_ratio` の両方を指定した場合はバリデーションエラー。
- 既存の `inner_valid` 指定は引き続き動作する（後方互換）。

Config 例（新しい簡略記法）:
```python
"early_stopping": {"enabled": True, "rounds": 50, "validation_ratio": 0.1}
```

### Impact

- `lizyml/config/schema.py`: `EarlyStoppingConfig` に `validation_ratio` フィールド + `model_validator` 追加。
- テスト: validation_ratio ショートハンド・競合エラー・後方互換のテスト追加。

### Compatibility

- 非破壊的追加。既存の `inner_valid` 指定は変更なく動作する。

### Alternatives Considered

- `inner_valid` を廃止して `validation_ratio` に完全置換 → 将来 `InnerKFoldValid` 等の拡張余地がなくなるため却下。エイリアスとして共存させる。

### Acceptance Criteria

- `validation_ratio=0.2` 指定で `inner_valid.ratio == 0.2` になる。
- `inner_valid` と `validation_ratio` の両方指定でバリデーションエラー。
- 既存の `inner_valid` 形式が引き続き動作する。

---

## 2026-03-04: evaluate_table() の列順変更

- ID: `H-0011`
- Status: `accepted`（実装済み。H-0111 で Status を訂正、#319）
- Scope: `Evaluation | Public API`
- Related: `BLUEPRINT.md §13.2, HISTORY.md H-0005`

### Context

現在の `evaluate_table()` の列順は `oof, if_mean, fold_0...fold_N-1, cal_oof` だが、実務では IF（学習時の性能）を先に確認し、次に OOF（汎化性能）を比較するフローが自然。列順を `if_mean, oof, fold_0...fold_N-1, cal_oof` に変更する。

### Proposal

- `lizyml/evaluation/table_formatter.py` の `format_metrics_table()` で列の挿入順を `if_mean` → `oof` → `fold_0...fold_N-1` → `cal_oof` に変更する。

### Impact

- `lizyml/evaluation/table_formatter.py`: 列構築順の変更。
- `tests/test_evaluation/test_table_formatter.py`: 列順アサーションの更新。
- `BLUEPRINT.md §13.2`: 仕様記載の列順更新。

### Compatibility

- `evaluate_table()` の返り値は `pd.DataFrame` であり、列名でアクセスする限り影響なし。列の「位置」に依存するコードのみ影響する（通常ない）。

### Alternatives Considered

- 列順をユーザーが Config で指定できるようにする → 過剰な柔軟性のため却下。固定列順で十分。

### Acceptance Criteria

- `evaluate_table()` の列順が `if_mean, oof, fold_0...fold_N-1, cal_oof` になる。
- 既存テストが新しい列順で通過する。

---

## 2026-03-04: residuals_plot() の IS/OOS サンプル数バランシング

- ID: `H-0012`
- Status: `accepted`（実装済み。H-0111 で Status を訂正、#319）
- Scope: `Plots`
- Related: `BLUEPRINT.md §13.3, HISTORY.md H-0009`

### Context

K-fold CV（例: 5-fold）では IS サンプル数が OOS の約 4 倍になる。`residuals_plot(kind="scatter")` や `kind="histogram"` で IS/OOS を重ね描きすると、IS の点がOOS を覆い隠してグラフが見にくくなる。

### Proposal

- `lizyml/plots/residuals.py` 内部の IS データ描画時に、IS サンプル数が OOS サンプル数を超える場合、ランダムサンプリングで OOS と同数に間引く。
- サンプリングは `np.random.default_rng(seed=0)` で再現可能にする。
- バランシングは scatter と histogram の両方に適用する（QQ は OOS のみなので対象外）。
- 実装は `_build_is_data()` ヘルパーの後段、描画直前で行う（`_downsample_is()` ヘルパーを新設）。

### Impact

- `lizyml/plots/residuals.py`: `_downsample_is()` ヘルパー追加。`_add_scatter_traces()` / `_add_histogram_traces()` 呼び出し前に適用。

### Compatibility

- 既存テストの IS/OOS トレース存在チェックは変更不要（ダウンサンプリング後も IS トレースは描画される）。

### Alternatives Considered

- ユーザーに `max_is_samples` パラメータを公開する → 過剰な柔軟性のため却下。内部で OOS 数に合わせる方式で十分。
- opacity のみで対応する → サンプル数が大きく異なる場合は opacity だけでは不十分。

### Acceptance Criteria

- IS サンプル数 > OOS サンプル数の場合、IS が OOS と同数にダウンサンプリングされる。
- IS サンプル数 <= OOS サンプル数の場合、ダウンサンプリングは行われない。
- ダウンサンプリングは seed=0 で再現可能。

---

## 2026-03-05: Binary/Multiclass で StratifiedKFold をデフォルト化 + KFold 警告

- ID: `H-0013`
- Status: `implemented`
- Scope: `Config | Split`
- Related: `BLUEPRINT.md §5.2, §10.2`

### Context

現在、全タスク（regression/binary/multiclass）で `kfold` がデフォルトの split method。分類タスクではクラス比率を保持する `stratified_kfold` がベストプラクティスであり、ユーザーが明示指定を忘れると不均衡な fold 分割が発生する。

### Proposal

- Config loader の正規化で、`task` が `binary` または `multiclass` かつ `split.method` が未指定の場合、`stratified_kfold` をデフォルトにする。
- ユーザーが分類タスクで `method: "kfold"` を明示指定した場合、`warnings.warn()` で「StratifiedKFold の使用を推奨する」旨の警告を出す。
- 回帰タスクの挙動は変更しない（`kfold` のまま）。

### Impact

- `lizyml/config/loader.py`: 正規化ロジック追加。
- `lizyml/core/model.py`: `_build_splitter()` で `task` を参照してデフォルト判定。
- BLUEPRINT §5.2 の Config 例、§10.2 の Outer CV リストに注記追加。

### Compatibility

- 既存の `method: "kfold"` 明示指定は引き続き動作する（警告付き）。
- `method` 未指定で分類タスクを使っていたユーザーは、暗黙的に `stratified_kfold` に切り替わる（split indices が変わる）。

### Alternatives Considered

- `method` 未指定時はエラーにする → 既存ユーザーの breaking change になるため却下。
- 警告なしでデフォルトを変えるだけ → KFold を意図的に選んだユーザーへの情報がないため却下。

### Acceptance Criteria

- `task="binary"` かつ `split.method` 未指定 → StratifiedKFold が使われる。
- `task="binary"` かつ `split.method="kfold"` → 警告が出る + KFold が使われる。
- `task="regression"` かつ `split.method` 未指定 → KFold が使われる（変更なし）。
- `task="multiclass"` でも同様に StratifiedKFold がデフォルト。

---

## 2026-03-05: Precision at K メトリクス追加

- ID: `H-0014`
- Status: `implemented`
- Scope: `Metrics`
- Related: `BLUEPRINT.md §13.1`

### Context

Binary 分類で「上位 K% をポジティブと予測したときの精度」を評価する `Precision at K` は、不均衡データでのモデル評価に有用。現在未登録。

### Proposal

- `lizyml/metrics/classification.py` に `PrecisionAtKMetric` を追加する。
  - 名前: `precision_at_k`
  - `needs_proba: True`（確率ベースで上位 K% を算出）
  - `greater_is_better: True`
  - `supports_task: ["binary"]`
  - デフォルト `k=10`（上位 10%）。`k` はメトリクス設定で指定可能。
- `TASK_METRICS["binary"]` に登録する。

### Impact

- `lizyml/metrics/classification.py`: クラス追加。
- `lizyml/metrics/registry.py`: TASK_METRICS 更新。

### Compatibility

- 新規追加のみ。既存メトリクスの挙動は変更しない。

### Alternatives Considered

- `k` を固定値（10%）のみにする → 柔軟性が低いため、パラメータ化を採用。
- `Recall at K` も同時追加する → スコープを最小限にするため今回は見送り。

### Acceptance Criteria

- `precision_at_k` が `evaluate()` の結果に含まれる（binary タスク）。
- `k` パラメータで上位 K% のカットオフを変更できる。
- regression/multiclass タスクで指定した場合、`UNSUPPORTED_METRIC` エラー。

---

## 2026-03-05: ROC Curve プロット追加（IS/OOS 対応）

- ID: `H-0015`
- Status: `implemented`
- Scope: `Plots | Public API`
- Related: `BLUEPRINT.md §13.3`

### Context

Binary 分類の ROC Curve は基本的な評価可視化であり、BLUEPRINT §13.3 で「未実装」として明記されている。IS（In-Sample）と OOS（Out-of-Sample）の比較は過学習の判定に有用。

### Proposal

- `lizyml/plots/classification.py` を新規作成する。
- `plot_roc_curve(fit_result, y_true)` を追加する。
  - IS/OOS 両方の ROC Curve を重ね描きする。
  - IS: `if_pred_per_fold` + `splits.outer` の train_idx から算出。
  - OOS: `oof_pred` から算出。
  - AUC 値を凡例に表示する。
  - Plotly Figure を返す。
- `Model.roc_curve_plot()` を Facade メソッドとして追加する。

### Impact

- `lizyml/plots/classification.py`: 新規ファイル。
- `lizyml/plots/__init__.py`: export 追加。
- `lizyml/core/model.py`: `roc_curve_plot()` メソッド追加。

### Compatibility

- 新規追加のみ。既存 API に変更なし。

### Alternatives Considered

- fold ごとの ROC を個別に描画する → 煩雑になるため、IS/OOS 集約の 2 本線を採用。
- PR Curve も同時追加する → スコープを最小限にするため今回は見送り。

### Acceptance Criteria

- `model.roc_curve_plot()` が Plotly Figure を返す。
- IS と OOS の 2 本の ROC Curve が描画される。
- AUC 値が凡例に表示される。
- binary タスク以外で呼び出した場合は `LizyMLError` を返す。
- `y_true` は `fit()` 時に一時保持した値を使用する（`residuals_plot` と同じパターン）。

---

## 2026-03-05: Confusion Matrix テーブル追加（IS/OOS 対応）

- ID: `H-0016`
- Status: `implemented`
- Scope: `Evaluation | Public API`
- Related: `BLUEPRINT.md §13.3`

### Context

Binary/Multiclass 分類の Confusion Matrix はモデル評価の基本。BLUEPRINT §13.3 で「未実装」として明記されている。IS/OOS の比較でモデルの過学習を判定したい。出力は可視化（プロット）ではなくテーブル（DataFrame）とする。

### Proposal

- `lizyml/evaluation/confusion.py` を新規作成する。
- `confusion_matrix_table(fit_result, y_true, *, threshold=0.5) -> dict[str, pd.DataFrame]` を追加する。
  - 戻り値: `{"is": pd.DataFrame, "oos": pd.DataFrame}`
  - DataFrame は sklearn の `confusion_matrix` 相当の行列形式。
  - IS: `if_pred_per_fold` + `splits.outer` の train_idx から集約。
  - OOS: `oof_pred` から算出。
  - binary: `threshold` で確率→クラスラベル変換。
  - multiclass: argmax でクラスラベル変換。
- `Model.confusion_matrix()` を Facade メソッドとして追加する。

### Impact

- `lizyml/evaluation/confusion.py`: 新規ファイル。
- `lizyml/core/model.py`: `confusion_matrix()` メソッド追加。

### Compatibility

- 新規追加のみ。既存 API に変更なし。

### Alternatives Considered

- Plotly ヒートマップで可視化する → ユーザー要件がテーブル出力のため、DataFrame を採用。
- IS/OOS を 1 つの DataFrame にまとめる → 可読性が落ちるため dict で分離。

### Acceptance Criteria

- `model.confusion_matrix()` が `{"is": DataFrame, "oos": DataFrame}` を返す。
- binary タスクで `threshold` パラメータが機能する。
- multiclass タスクでも動作する。
- regression タスクで呼び出した場合は `LizyMLError` を返す。

---

## 2026-03-05: Calibration Curve + Predicted Probability Histogram 追加

- ID: `H-0017`
- Status: `implemented`
- Scope: `Plots | Public API`
- Related: `BLUEPRINT.md §12.3, §13.3`

### Context

Binary 分類の Calibration 有効時に、校正の効果を可視化する手段がない。BLUEPRINT §13.3 で「reliability diagram / ECE」として計画されている。Calibration Curve（Reliability Diagram）で校正精度を確認し、Predicted Probability Histogram で Raw/Calibrated の分布変化を比較したい。

### Proposal

- `lizyml/plots/calibration.py` を新規作成する。
- `plot_calibration_curve(fit_result, y_true) -> plotly.graph_objects.Figure` を追加する。
  - Raw OOF（`fit_result.oof_pred`）と Calibrated OOF（`fit_result.calibrator.calibrated_oof`）の 2 本の Reliability Diagram を描画。
  - 理想線（y=x）を参照線として描画。
  - bin 数はデフォルト 10（`sklearn.calibration.calibration_curve` 相当）。
- `plot_probability_histogram(fit_result) -> plotly.graph_objects.Figure` を追加する。
  - Raw OOF と Calibrated OOF の確率分布ヒストグラムを重ね描き。
  - 校正前後の分布シフトを視覚的に確認できるようにする。
- `Model.calibration_plot()` および `Model.probability_histogram_plot()` を Facade メソッドとして追加する。

### Impact

- `lizyml/plots/calibration.py`: 新規ファイル。
- `lizyml/plots/__init__.py`: export 追加。
- `lizyml/core/model.py`: 2 メソッド追加。

### Compatibility

- 新規追加のみ。既存 API に変更なし。

### Alternatives Considered

- Calibration Curve と Histogram を 1 つの Figure にサブプロットで統合する → 個別に使いたいケースがあるため、別関数を採用。
- ECE 値もプロットに埋め込む → 将来追加可能だが、初期実装はシンプルに保つ。

### Acceptance Criteria

- `model.calibration_plot()` が Plotly Figure を返す。
- Raw と Calibrated の 2 本の Reliability Diagram + 理想線が描画される。
- `model.probability_histogram_plot()` が Plotly Figure を返す。
- Raw と Calibrated の 2 つのヒストグラムが重ね描きされる。
- Calibration 未有効時に呼び出した場合は `LizyMLError` を返す。
- binary タスク以外で呼び出した場合は `LizyMLError` を返す。
- データソースは OOF（cross-fit 由来の `calibrated_oof`）であり、`c_final` は使用しない。

---

## 2026-03-05: Multiclass メトリクス拡張（AUC OvR / Average Precision OvR / Brier OvR）

- ID: `H-0018`
- Status: `implemented`
- Scope: `Metrics | Public API`
- Related: `BLUEPRINT.md §13.1`

### Context

Multiclass 分類タスクの `TASK_METRICS["multiclass"]` は現在 `logloss / f1 / accuracy` の 3 種のみ。AUC（OvR）、Average Precision（OvR）、Brier（OvR）は multiclass でも One-vs-Rest 展開で計算可能であり、Binary Notebook と対称的な評価を行うために必要。

### Proposal

既存の `AUCMetric` / `AUCPRMetric` / `BrierMetric` を multiclass 対応に拡張し、`TASK_METRICS["multiclass"]` に登録する。

- **AUC（OvR）**: `y_pred` が 2D `(n_samples, n_classes)` の場合、`roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')` を呼ぶ。
- **Average Precision（OvR）**: `y_true` を One-Hot 展開し、クラスごとに `average_precision_score` を計算して macro 平均。
- **Brier（OvR）**: `y_true` を One-Hot 展開し、クラスごとに `brier_score_loss` を計算して macro 平均。
- 各メトリクスの `__call__` で `y_pred.ndim` を分岐条件とし、1D（binary）はそのまま、2D（multiclass）は OvR ロジックに分岐する。
- `_require_1d_same_len` ガードは multiclass 経路ではスキップする（2D は長さ比較で `y_pred.shape[0] == len(y_true)` を使う）。

### Impact

- `lizyml/metrics/classification.py`: `AUCMetric.__call__` / `AUCPRMetric.__call__` / `BrierMetric.__call__` に multiclass 分岐を追加。
- `lizyml/metrics/registry.py`: `TASK_METRICS["multiclass"]` に `auc`, `auc_pr`, `brier` を追加。
- `lizyml/metrics/classification.py`: `supports_task` に `"multiclass"` を追加（各クラス）。

### Compatibility

- 既存の binary 経路は変更なし（`y_pred.ndim == 1` の場合は従来ロジック）。
- multiclass で新たにこれらメトリクスが利用可能になる（追加のみ）。

### Alternatives Considered

- 別名メトリクス（`auc_ovr` / `brier_ovr`）として新規追加する → メトリクス名が増え Config が煩雑になるため、同名で multiclass 対応する方式を採用。
- `weighted` 平均をデフォルトにする → `macro` の方が class imbalance に対して公平な評価のため、`macro` を採用。

### Acceptance Criteria

- `task="multiclass"` で `evaluate(metrics=["auc", "auc_pr", "brier"])` が値を返す。
- multiclass AUC は `roc_auc_score(..., multi_class='ovr', average='macro')` と一致する。
- multiclass Average Precision はクラスごとの `average_precision_score` の macro 平均と一致する。
- multiclass Brier はクラスごとの `brier_score_loss` の macro 平均と一致する。
- binary タスクの既存動作が変わらない。
- regression タスクで指定した場合は `UNSUPPORTED_METRIC` エラー。

---

## 2026-03-05: ROC Curve の Multiclass OvR 拡張

- ID: `H-0019`
- Status: `implemented`
- Scope: `Plots | Public API`
- Related: `BLUEPRINT.md §13.3, HISTORY.md H-0015`

### Context

H-0015 で提案した ROC Curve プロットは binary 限定。Multiclass 分類では One-vs-Rest（OvR）方式でクラスごとの ROC Curve を描画するのが標準的な手法。Binary Notebook と対称的な可視化を Multiclass Notebook でも提供したい。

### Proposal

H-0015 の `plot_roc_curve(fit_result, y_true)` を multiclass 対応に拡張する。

- `task="multiclass"` の場合、クラスごとに OvR の ROC Curve を描画する。
  - IS: `if_pred_per_fold`（2D）+ `splits.outer` の train_idx から集約し、クラスごとの OvR を算出。
  - OOS: `oof_pred`（2D）からクラスごとの OvR を算出。
- レイアウト: IS と OOS を Plotly subplots で横並びにし、各 subplot にクラスごとの ROC 曲線を描画する。
- 各クラスの AUC 値を凡例に表示する。
- macro 平均 AUC もタイトルまたは凡例に表示する。
- `task="binary"` の場合は H-0015 の従来動作（IS/OOS の 2 本）を維持する。

### Impact

- `lizyml/plots/classification.py`: `plot_roc_curve` の multiclass 分岐を追加。
- H-0015 の binary 実装と同一関数内で分岐する。

### Compatibility

- binary の既存動作は変更なし。
- multiclass は新規追加のみ。

### Alternatives Considered

- binary と multiclass で関数を分ける（`plot_roc_curve_ovr`）→ Facade API が増えるため、同一関数で task 分岐する方式を採用。
- micro 平均の ROC も描画する → 初期実装はシンプルに保ち、macro 平均 + クラス別のみ。

### Acceptance Criteria

- `task="multiclass"` で `model.roc_curve_plot()` が Plotly Figure を返す。
- IS と OOS の 2 つの subplot にクラスごとの OvR ROC Curve が描画される。
- 各クラスの AUC 値が凡例に表示される。
- macro 平均 AUC が表示される。
- `task="binary"` では H-0015 の従来動作が維持される。
- `task="regression"` で呼び出した場合は `LizyMLError` を返す。

---

## 2026-03-05: InnerValid の split method 設定対応（stratified / group / time-aware holdout）

- ID: `H-0020`
- Status: `implemented`
- Scope: `Config | Training | Split`
- Related: `BLUEPRINT.md §5.2, §10.3`

### Context

現在の `EarlyStoppingConfig.inner_valid` は `HoldoutInnerValidConfig(method="holdout")` のみで、ランダム分割しかサポートしない。`HoldoutInnerValid.split()` は `y` と `groups` を引数に受け取るが無視しており、stratified / group / time-aware な内側分割ができない。

BLUEPRINT §10.3 では `HoldoutInnerValid(ratio, stratify, group, time, random_state)` が計画されているが未実装。分類タスクで Stratified、group_col がある場合に group-aware、time_col がある場合に time-aware な inner split が必要。

### Proposal

#### Config 変更

`HoldoutInnerValidConfig` に `stratify` パラメータを追加し、`InnerValidConfig` を discriminated union に拡張する。

```python
class HoldoutInnerValidConfig(BaseModel):
    method: Literal["holdout"]
    ratio: float = 0.1
    stratify: bool = False  # 新規追加
    random_state: int = 42

class GroupHoldoutInnerValidConfig(BaseModel):
    method: Literal["group_holdout"]
    ratio: float = 0.1
    random_state: int = 42

class TimeHoldoutInnerValidConfig(BaseModel):
    method: Literal["time_holdout"]
    ratio: float = 0.1

InnerValidConfig = HoldoutInnerValidConfig | GroupHoldoutInnerValidConfig | TimeHoldoutInnerValidConfig
```

`EarlyStoppingConfig.inner_valid` の型を `InnerValidConfig | None` に変更する。

#### デフォルト解決ルール

`inner_valid` が未指定（`None`）かつ `enabled=True` の場合、`Model.fit()` 時に外側 CV の method に応じて自動解決する。

| 外側 split.method | inner_valid のデフォルト |
|---|---|
| `stratified_kfold` | `holdout(stratify=True)` |
| `group_kfold` | `group_holdout` |
| `time_series` | `time_holdout` |
| `kfold`（またはCV未使用） | `holdout(stratify=False)` |

この解決は Config loader ではなく `Model._build_inner_valid()` で行う（外側 split の情報が必要なため）。

#### InnerValid 実装

- `HoldoutInnerValid`: `stratify=True` の場合、`sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=ratio)` を使い `y` に基づく層化抽出を行う。`stratify=False` は現行のランダム分割を維持。
- `GroupHoldoutInnerValid`: `groups` をユニークグループ単位で分割する。validation には末尾グループを使用し、group overlap を防ぐ。
- `TimeHoldoutInnerValid`: 時系列順を維持し、末尾 `ratio` 割合を validation に割り当てる（shuffle なし）。BLUEPRINT §10.3 の「時系列は内側も時系列順を厳守」に準拠。

#### CVTrainer の変更

`cv_trainer.py` で `inner_valid.split()` に `y` と `groups` を適切に渡す。現在すでに `y=y_train.to_numpy()` を渡しているが、`groups` は渡していないため追加する。

### Impact

- `lizyml/config/schema.py`: `InnerValidConfig` discriminated union、`GroupHoldoutInnerValidConfig`、`TimeHoldoutInnerValidConfig` 追加。`HoldoutInnerValidConfig` に `stratify` フィールド追加。
- `lizyml/training/inner_valid.py`: `StratifiedHoldoutInnerValid`（または `HoldoutInnerValid` に stratify 分岐追加）、`GroupHoldoutInnerValid`、`TimeHoldoutInnerValid` 追加。
- `lizyml/core/model.py`: `_build_inner_valid()` にデフォルト解決ロジック追加。
- `lizyml/training/cv_trainer.py`: `inner_valid.split()` 呼び出しに `groups` を渡す。

### Compatibility

- 既存の `inner_valid: {method: "holdout", ratio: 0.1}` は動作が変わらない（`stratify` のデフォルトは `False`）。
- `validation_ratio` ショートハンドも引き続き動作する（デフォルト解決で自動判定）。
- `inner_valid` 未指定のデフォルト挙動が変わる: 現在は常にランダム holdout → 今後は外側 CV 方式に追従。ただし `kfold` の場合はランダム holdout のままで既存挙動と一致。

### Alternatives Considered

- Config loader で外側 split.method を参照してデフォルトを解決する → loader 時点では `task` 情報しかなく `split` と `inner_valid` の関連性を解決できないため、`_build_inner_valid()` での解決を採用。
- `inner_valid.method` を外側と完全同名にする（`stratified_kfold` など）→ 内側は常に 1 分割の holdout であり KFold ではないため、名前の混乱を避けて `holdout` / `group_holdout` / `time_holdout` を採用。

### Acceptance Criteria

- `split.method="stratified_kfold"` かつ `inner_valid` 未指定 → inner split が stratified holdout になる。
- `split.method="group_kfold"` かつ `inner_valid` 未指定 → inner split が group holdout になる（group overlap なし）。
- `split.method="time_series"` かつ `inner_valid` 未指定 → inner split が time holdout になる（末尾を validation、shuffle なし）。
- `split.method="kfold"` かつ `inner_valid` 未指定 → inner split がランダム holdout になる（既存挙動維持）。
- `inner_valid` を明示指定した場合は外側 split.method に関わらずその設定が優先される。
- `validation_ratio` ショートハンドが引き続き動作する。
- `time_holdout` で shuffle が行われないことをテストで検証する。
- `group_holdout` で group overlap が発生しないことをテストで検証する。

---

## 2026-03-05: LGBMConfig スマートパラメーター追加（auto_num_leaves / ratio パラメーター / feature_weights / balanced）

- ID: `H-0021`
- Status: `implemented`
- Scope: `Config | EstimatorAdapter`
- Decision Date: 2026-03-05
- Related: `BLUEPRINT.md §5.3, §14.2`

### Context

現在の `LGBMConfig.params` は `dict[str, Any]` の生パラメーターのみで、データサイズやタスクに依存するパラメーターをユーザーが手動計算する必要がある。Config の簡潔さを損ない、設定ミスの原因になる。

### Proposal

`LGBMConfig` に以下のスマートパラメーターフィールドを追加し、`fit()` 時に学習データの情報に基づいて LightGBM ネイティブパラメーターに解決する。

#### 1. auto_num_leaves（葉の数の自動算出）

- `auto_num_leaves: bool = True`
- `num_leaves_ratio: float = 1.0`（`0 < ratio ≤ 1`）
- 算出ロジック:
  - `params.max_depth` が未指定または負値（制限なし）→ 基準値 = `131072`
  - `params.max_depth` が指定されている → 基準値 = `2 ^ max_depth`
  - `num_leaves = clamp(ceil(基準値 × num_leaves_ratio), 8, 131072)`
- 制約: `auto_num_leaves=True` 時に `params.num_leaves` を直接指定した場合は `CONFIG_INVALID`。

#### 2. データサイズ相対比率パラメーター

学習データの行数に対する割合で指定し、fit 時に絶対値に変換する。

- `min_data_in_leaf_ratio: float | None = None`（`0 < ratio < 1`）→ `min_data_in_leaf = max(1, ceil(n_rows × ratio))`
- `min_data_in_bin_ratio: float | None = None`（`0 < ratio < 1`）→ `min_data_in_bin = max(1, ceil(n_rows × ratio))`
- 制約: ratio 指定と対応する絶対値パラメーター（`params.min_data_in_leaf` 等）の同時指定は `CONFIG_INVALID`。

#### 3. feature_weights（特徴量重みの辞書指定）

- `feature_weights: dict[str, float] | None = None`
- 未指定特徴量は `1.0` で自動補完。
- 学習データの特徴量順に並び替えたリストに変換し、LightGBM に渡す。
- 副作用: `feature_pre_filter = False` を強制する。
- 制約: 重み `> 0` 必須。学習データに存在しない未知の特徴量名は `CONFIG_INVALID`。

#### 4. balanced（クラス重み自動均衡化）

- `balanced: bool = False`
- `True` 時、学習データのクラス比率から自動的に重みを算出する。
  - binary: `scale_pos_weight = neg_count / pos_count` を設定。
  - multiclass: `sample_weight` でクラス逆頻度重み付け。
  - regression: `UNSUPPORTED_TASK` エラー。

### Impact

- `lizyml/config/schema.py`: `LGBMConfig` に 6 フィールド追加 + `model_validator` でバリデーション。
- `lizyml/estimators/lgbm.py`: `resolve_smart_params(n_rows, feature_names, y)` — fit 時にスマートパラメーターを LightGBM ネイティブパラメーターに解決するロジック追加。
- `lizyml/core/model.py`: `fit()` で `n_rows` / `feature_names` / `y` を解決関数に渡す。

### Compatibility

- 既存の `LGBMConfig(params={...})` は影響なし（新フィールドはすべてデフォルト付き）。
- `auto_num_leaves` のデフォルトが `True` のため、`params.num_leaves` を直接指定しているユーザーは `auto_num_leaves=False` の追加が必要（バリデーションエラーで通知）。
- `format_version` 変更不要（Config の拡張のみ）。

### Alternatives Considered

- `TrainingConfig` に配置 → LightGBM 固有のため `LGBMConfig` が適切。将来 sklearn adapter 等で同様の概念があれば各 adapter config に追加する。
- `params` dict の中にネストする → pydantic バリデーションが効かないため却下。
- `num_leaves_ratio` を `num_leaves` の型を `int | float` にして判定する → 暗黙的で分かりにくいため、明示的な `auto_num_leaves` フラグを採用。

### Acceptance Criteria

- `auto_num_leaves=True`, `max_depth=5` → `num_leaves = ceil(32 × ratio)`, `clamp(8, 131072)` が適用される。
- `auto_num_leaves=True` + `params.num_leaves` 指定 → `CONFIG_INVALID`。
- `auto_num_leaves=False` + `params.num_leaves=64` → そのまま `64` が使われる。
- `min_data_in_leaf_ratio=0.01`, `n_rows=10000` → `min_data_in_leaf=100`。
- `min_data_in_leaf_ratio` + `params.min_data_in_leaf` 同時指定 → `CONFIG_INVALID`。
- `feature_weights={"a": 2.0}` + features=`[a, b, c]` → `[2.0, 1.0, 1.0]`, `feature_pre_filter=False`。
- `feature_weights={"unknown": 1.0}` → `CONFIG_INVALID`。
- `balanced=True`, binary → `scale_pos_weight` が正しく設定される。
- `balanced=True`, regression → `UNSUPPORTED_TASK`。

---

## 2026-03-05: LightGBM タスク別デフォルトパラメータープロファイル

- ID: `H-0022`
- Status: `implemented`
- Scope: `Config | EstimatorAdapter`
- Decision Date: 2026-03-05
- Related: `BLUEPRINT.md §14.3, §5.2`

### Context

現在 `LGBMAdapter._build_params()` は `objective` / `metric` / `verbose` / `random_state` のみをデフォルト設定し、`learning_rate` / `max_depth` 等は LightGBM ライブラリの内部デフォルトに依存している。実務で頻繁に使うパラメーターの推奨デフォルト値を明示的に設定し、ユーザーが最小限の Config でも妥当な精度のモデルを得られるようにする。

### Proposal

#### タスク別 objective / metric デフォルト

| | regression | binary | multiclass |
|---|---|---|---|
| objective | `huber` | `binary` | `multiclass` |
| metric | `[huber, mae, mape]` | `[auc, binary_logloss]` | `[auc_mu, multi_logloss]` |

注記:
- regression の objective を `regression`（L2）から `huber` に変更。外れ値に対してロバスト。
- `brier` は LightGBM ネイティブ未対応のため、binary metric デフォルトから除外。カスタム feval 対応は将来の拡張点とする。
- `precision_at_k` も LightGBM ネイティブ未対応。将来のカスタム feval 対応として保留。

#### 共通デフォルト

| パラメーター | デフォルト値 | 備考 |
|---|---|---|
| `boosting` | `gbdt` | |
| `first_metric_only` | `False` | |
| `n_estimators` | `1500` | sklearn API 相当の `num_boost_round` |
| `learning_rate` | `0.001` | 低学習率で early stopping に依存 |
| `max_depth` | `5` | |
| `max_bin` | `511` | |
| `feature_fraction` | `0.7` | |
| `bagging_fraction` | `0.7` | |
| `bagging_freq` | `10` | |
| `lambda_l1` | `0.0` | |
| `lambda_l2` | `0.000001` | |

#### Training デフォルト変更

| パラメーター | 現在のデフォルト | 新デフォルト |
|---|---|---|
| `early_stopping.enabled` | `False` | `True` |
| `early_stopping.rounds` | `50` | `150` |

`validation_ratio` のデフォルトは `0.1`（`EarlyStoppingConfig.validation_ratio` のデフォルトとして設定。`early_stopping.enabled=True` 時に `inner_valid` 未指定の場合に自動適用）。

### Impact

- `lizyml/estimators/lgbm.py`: `_build_params()` のデフォルト値拡張、`_TASK_OBJECTIVE` / `_TASK_METRIC` マッピング更新。
- `lizyml/config/schema.py`: `EarlyStoppingConfig` のデフォルト値変更（`enabled=True`, `rounds=150`, `validation_ratio=0.1`）。
- 既存テスト: seed 固定テスト・再現性テストの期待値が変わる可能性あり（デフォルト objective / パラメーター変更のため）。

### Compatibility

- `LGBMConfig.params` で明示指定した値はデフォルトを上書きするため、パラメーターを指定しているユーザーは影響なし。
- デフォルト値のみ使用しているユーザーは挙動が変わる（`0.x` バージョンのため許容）。
- regression の `objective` が `regression` → `huber` に変わるため、既存の回帰モデルの出力が変わる。
- `early_stopping.enabled` が `True` になるため、未指定ユーザーは early stopping が有効になる。

### Alternatives Considered

- デフォルト値を変更せず、推奨設定を Config テンプレートとしてドキュメントで提供 → ユーザーが毎回コピーする手間がかかるため却下。
- profile 方式（`"conservative"` / `"aggressive"` 等の名前付きプロファイル）→ 過度な抽象化のため却下。単一のバランスの取れたデフォルトを提供する。
- `huber` ではなく `regression`（L2）を維持し、外れ値対応はユーザー責任とする → 実務では外れ値がある場合が多く、`huber` の方がロバストなデフォルトとして適切。

### Acceptance Criteria

- Config 未指定時に `learning_rate=0.001`, `max_depth=5`, `max_bin=511` 等がデフォルト適用される。
- `params` で明示指定した値がデフォルトを上書きする。
- regression タスクで `objective=huber` がデフォルトになる。
- binary タスクで `metric=[auc, binary_logloss]` がデフォルトになる。
- multiclass タスクで `metric=[auc_mu, multi_logloss]` がデフォルトになる。
- `early_stopping.enabled` のデフォルトが `True` になる。
- `early_stopping.rounds` のデフォルトが `150` になる。
- `early_stopping.validation_ratio` のデフォルトが `0.1` になる。
- 既存テストがデフォルト変更に伴い適切に更新されている。

---

## 2026-03-05: TuningResult 型導入と tuning_table() API 追加

- ID: `H-0023`
- Status: `accepted`
- Scope: `Public API | Tuning`
- Related: `BLUEPRINT.md §6.1, §4.1, §7.1`

### Context

現在 `Tuner.tune()` は `dict(study.best_params)` のみを返し、Optuna Study オブジェクト（全 trial の探索履歴）を破棄している。`Model.tune()` も同様に `dict[str, Any]` を返す。

Tuning Notebook で「探索したパラメーターと各パラメーターでの評価」を一覧表示するには、全 trial の履歴が必要だが、現在の実装では取得手段がない。

### Proposal

#### 1. TuningResult 型の導入

`lizyml/core/types/tuning_result.py` に `TuningResult` dataclass を新設する。

```python
@dataclass(frozen=True)
class TrialResult:
    number: int               # trial 番号（0-indexed）
    params: dict[str, Any]    # 探索パラメーター
    score: float              # OOF メトリクス値
    state: str                # "complete" | "pruned" | "fail"

@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    trials: list[TrialResult]  # 全 trial 履歴（番号順）
    metric_name: str           # 最適化メトリクス名
    direction: str             # "minimize" | "maximize"
```

#### 2. Tuner.tune() の戻り値変更

`Tuner.tune()` の戻り値を `dict[str, Any]` → `TuningResult` に変更する。Optuna Study の `study.trials` から全 trial 情報を収集して `TuningResult` を構築する。

#### 3. Model.tune() の戻り値変更

`Model.tune()` の戻り値を `dict[str, Any]` → `TuningResult` に変更する。内部で `self._best_params = result.best_params` を維持し、`fit()` 連携は既存通り。`TuningResult` を `self._tuning_result` として保持する。

#### 4. Model.tuning_table() の追加

`Model.tuning_table() -> pd.DataFrame` を追加する。`TuningResult.trials` を DataFrame に変換する。

- 列: `trial`, `score`, + 各探索パラメーター名
- 行: trial 番号順
- `score` 列名は `TuningResult.metric_name` を使用する（例: `rmse`）
- `tune()` 未実行時は `MODEL_NOT_FIT` エラー

### Impact

- `lizyml/core/types/tuning_result.py`: 新規ファイル（`TuningResult`, `TrialResult`）。
- `lizyml/tuning/tuner.py`: `tune()` 戻り値を `TuningResult` に変更。
- `lizyml/core/model.py`: `tune()` 戻り値変更 + `tuning_table()` メソッド追加 + `_tuning_result` 保持。
- `tests/test_tuning/`: `tune()` の戻り値アサーション更新、`tuning_table()` テスト追加。

### Compatibility

- `tune()` の戻り値型が `dict` → `TuningResult` に変わる破壊的変更。ただし `0.x` バージョンのため許容。
- `TuningResult.best_params` で従来の dict アクセスパターンは維持可能。
- `fit()` 連携は内部で `best_params` を参照するため影響なし。

### Alternatives Considered

- `tune()` の戻り値は dict のまま、別途 `study` を保持して `tuning_table()` で変換する → API として `TuningResult` の方が明確で、study への依存を外部に漏らさない。
- Optuna の `study.trials_dataframe()` をそのまま返す → Optuna 依存が公開 API に漏れるため却下。自前で変換する。
- `tuning_table()` を `TuningResult` のメソッドにする → `Model` の Facade パターンに合わせ、`Model.tuning_table()` として提供する。

### Acceptance Criteria

- `model.tune()` が `TuningResult` を返す。
- `TuningResult.best_params` が `dict[str, Any]` で最良パラメーターを返す。
- `TuningResult.best_score` が最良スコアを返す。
- `TuningResult.trials` が全 trial の `TrialResult` リストを返す（番号順）。
- `model.tuning_table()` が `pd.DataFrame` を返す。
- DataFrame の列が `trial`, メトリクス名, 探索パラメーター名を含む。
- `tune()` 未実行時に `tuning_table()` を呼ぶと `MODEL_NOT_FIT` エラー。
- `fit()` が `tune()` 後に `best_params` を正しく使用する（既存動作維持）。

### Decision

- Date: `2026-03-05`
- Result: `accepted`
- Notes: `feat/phase-20-classification-enhancements` ブランチで実施。`TuningResult` / `TrialResult` を `lizyml/core/types/tuning_result.py` に追加。`Tuner.tune()` と `Model.tune()` の戻り値を `TuningResult` に変更。`Model.tuning_table()` メソッドを追加。

---

## 2026-03-05: デフォルト Tuning Space の導入（タスク別デフォルト探索空間 + Tuner 拡張）

- ID: `H-0024`
- Status: `accepted`
- Scope: `Config | Tuning | Public API`
- Related: `BLUEPRINT.md §11.1, §5.2, §14.3`

### Context

現在 `Model.tune()` は `tuning.optuna.space` が必須で、ユーザーが毎回 SearchSpace を手動定義する必要がある。実務では LightGBM のハイパーパラメーターの探索範囲はタスク種別によりほぼ定型化されており、デフォルトの探索空間を提供すればユーザーの手間を大幅に削減できる。

また、現在の Tuner は `LGBMConfig.params` の model パラメーターのみ探索可能で、スマートパラメーター（H-0021）や training パラメーター（`early_stopping_rounds` / `validation_ratio`）は trial 間で固定されている。これらも探索対象に含めることで、より効果的なハイパーパラメーター最適化が可能になる。

### Proposal

#### 1. デフォルト Tuning Space の定義

`tuning.optuna.space` が空（`{}`）の場合、タスク別のデフォルト探索空間を自動適用する。

##### 探索次元（SearchDim）

| パラメーター | 型 | 範囲 | カテゴリ | 備考 |
|---|---|---|---|---|
| `objective` | categorical | regression: `[huber, fair]`, binary: `[binary]`, multiclass: `[multiclass, multiclassova]` | model | タスク別選択肢 |
| `n_estimators` | int | `[600, 2500]` | model | `num_boost_round` 相当 |
| `learning_rate` | float (log) | `[0.0001, 0.1]` | model | 対数スケール |
| `max_depth` | int | `[3, 12]` | model | |
| `feature_fraction` | float | `[0.5, 1.0]` | model | |
| `bagging_fraction` | float | `[0.5, 1.0]` | model | |
| `num_leaves_ratio` | float | `[0.5, 1.0]` | smart | `auto_num_leaves=True` 前提 |
| `min_data_in_leaf_ratio` | float | `[0.01, 0.2]` | smart | データサイズ相対 |
| `early_stopping_rounds` | int | `[40, 240]` | training | `EarlyStoppingConfig.rounds` |
| `validation_ratio` | float | `[0.1, 0.3]` | training | `EarlyStoppingConfig.validation_ratio` |

##### 固定パラメーター（探索しない）

| パラメーター | 値 | 備考 |
|---|---|---|
| `auto_num_leaves` | `True` | `num_leaves_ratio` で間接制御 |
| `first_metric_only` | `True` | 早期停止の判定を主メトリクスのみにする |
| `metric` | regression: `[huber, mae, mape]`, binary: `[auc, binary_logloss]`, multiclass: `[auc_mu, multi_logloss]` | H-0022 のデフォルトと同一 |

注記:
- `brier` は LightGBM ネイティブ未対応のため Binary metric から除外。
- `precision_at_k` も LightGBM ネイティブ未対応のため除外。
- Binary の objective は `binary` のみ（選択肢が 1 つのため実質固定）。

##### 最適化メトリクスと方向

| タスク | `metric_name`（OOF 評価） | `direction` |
|---|---|---|
| regression | Config の `evaluation.metrics[0]` またはデフォルト `rmse` | `minimize` |
| binary | Config の `evaluation.metrics[0]` またはデフォルト `auc` | メトリクスの `greater_is_better` に従う |
| multiclass | Config の `evaluation.metrics[0]` またはデフォルト `logloss` | メトリクスの `greater_is_better` に従う |

#### 2. SearchDim のカテゴリ拡張

`SearchDim` にカテゴリ属性を追加し、Tuner がパラメーターの適用先を区別できるようにする。

- `model`: `LGBMAdapter.params` に渡す（現行通り）
- `smart`: `LGBMConfig` のスマートパラメーターとして `resolve_smart_params()` に渡す
- `training`: trial ごとに `EarlyStoppingConfig` / `InnerValidStrategy` を再構築

#### 3. Tuner の拡張

- `estimator_factory` のシグネチャを拡張し、smart params と training params を受け取れるようにする。
- `validation_ratio` が探索対象の場合、trial ごとに `InnerValidStrategy` を再構築する（`inner_valid_factory` パターン）。
- `early_stopping_rounds` が探索対象の場合、trial ごとに `LGBMAdapter` の `early_stopping_rounds` を変更する。

#### 4. Config の挙動

- `tuning.optuna.space` が空 `{}` → デフォルト空間を自動適用。
- `tuning.optuna.space` が指定されている → ユーザー指定を使用（現行通り）。
- デフォルト空間の個別次元を上書きしたい場合は、`space` に該当キーを指定する（デフォルトとマージ）。

### Impact

- `lizyml/tuning/search_space.py`: `default_space(task)` 関数追加、`SearchDim` にカテゴリ属性追加。
- `lizyml/tuning/tuner.py`: smart params / training params の per-trial 適用ロジック追加、`inner_valid_factory` パターン導入。
- `lizyml/core/model.py`: `tune()` でデフォルト空間の自動適用、拡張 `estimator_factory` / `inner_valid_factory` の構築。
- `lizyml/config/schema.py`: `OptunaConfig.space` が空の場合のデフォルト挙動を文書化。

### Compatibility

- 既存の `tuning.optuna.space` 指定は変更なく動作する。
- `space` 未指定時の挙動が変わる: 現在は空 space でエラーまたは探索なし → 今後はデフォルト空間が適用される。`0.x` のため許容。
- `Tuner` の内部 API（`estimator_factory` シグネチャ）が変わるが、内部 API のため影響は限定的。

### Alternatives Considered

- デフォルト空間を Config テンプレートとしてドキュメントで提供 → ユーザーが毎回コピーする手間がかかるため却下。
- training params を探索対象に含めない → `early_stopping_rounds` と `validation_ratio` は精度に大きく影響するため、デフォルトに含める。
- `brier` をカスタム feval で LightGBM に渡す → 実装コストが高く、将来の拡張点とする。

### Acceptance Criteria

- `tuning.optuna.space` が空の場合、タスク別デフォルト空間が自動適用される。
- regression の objective が `[huber, fair]` から探索される。
- multiclass の objective が `[multiclass, multiclassova]` から探索される。
- `learning_rate` が対数スケールで `[0.0001, 0.1]` の範囲で探索される。
- `num_leaves_ratio` が `[0.5, 1.0]` の範囲で探索され、`auto_num_leaves=True` で解決される。
- `early_stopping_rounds` が trial ごとに変更される。
- `validation_ratio` が trial ごとに `InnerValidStrategy` を再構築する。
- ユーザー指定の `space` がデフォルトを上書きする。
- `first_metric_only=True` と `metric` がデフォルトで固定適用される。
- Binary の metric に `brier` が含まれない（ネイティブ未対応）。
- 全テスト・lint・mypy 通過。

### Decision

- Date: `2026-03-05`
- Result: `accepted`
- Notes: `feat/phase-20-classification-enhancements` ブランチで実施。`SearchDim` に `category` 属性追加。`default_space(task)` を10次元（model/smart/training）に拡張。`default_fixed_params(task)` と `split_by_category()` を追加。Tuner を拡張し smart/training params の per-trial 適用を実装。`resolve_smart_params_from_dict()` を追加。

---

## 2026-03-05: Phase 20/21 監査乖離の是正タスク追加

- ID: `H-0025`
- Status: `accepted`
- Scope: `Public API | Config | Training | Notebook`
- Related: `BLUEPRINT.md §4.4, §5.3, §10.3, §13.3`

### 目的

Phase 20/21 の Requirements Audit で検出された部分的乖離を、仕様変更ではなく「既存仕様への整合修正」として計画化し、次タスクで確実に是正する。

対象の乖離は以下の 4 点。

1. `Model.load()` 後の `probability_histogram_plot()` が実行可能で、他の「学習時ターゲット必須API」と境界不整合。
2. `GroupHoldoutInnerValid` の validation group 選定が「shuffle 後末尾」であり、仕様の「末尾 group 割当」と不一致。
3. `LGBMConfig` の `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` に `(0,1)` 範囲検証が未実装。
4. Notebook の LightGBM パラメーター確認セルで、スマートパラメーター表示項目が仕様要求を完全網羅していない。

### Proposal

#### 1. load 後 API 境界の統一

- `Model.probability_histogram_plot()` でも `self._y is None` を検知し、`MODEL_NOT_FIT` を返す。
- `roc_curve_plot()` / `confusion_matrix()` / `calibration_plot()` と同じ境界に揃える。

#### 2. GroupHoldout の割当方針を仕様準拠化

- `GroupHoldoutInnerValid` を「入力順の末尾 group を validation」に変更する。
- group overlap 禁止は維持する。
- 時系列/順序データでの再現可能な挙動を優先する。

#### 3. smart ratio の範囲バリデーション追加

- `min_data_in_leaf_ratio`: `0 < ratio < 1`
- `min_data_in_bin_ratio`: `0 < ratio < 1`
- 範囲外は `CONFIG_INVALID` とする。

#### 4. Notebook 確認セルの網羅化

- `tutorial_regression_lgbm.ipynb` に `min_data_in_bin_ratio`, `feature_weights`, `balanced` の表示を追加。
- `tutorial_binary_lgbm.ipynb` / `tutorial_multiclass_lgbm.ipynb` にも同等の確認セルを揃える。

### 影響範囲

- `lizyml/core/model.py`
- `lizyml/training/inner_valid.py`
- `lizyml/config/schema.py`
- `tests/test_*`（load後境界、group holdout、ratio検証）
- `notebooks/tutorial_regression_lgbm.ipynb`
- `notebooks/tutorial_binary_lgbm.ipynb`
- `notebooks/tutorial_multiclass_lgbm.ipynb`

### 互換性

- 公開メソッド追加/削除はない。既存 API surface は維持。
- `probability_histogram_plot()` の load 後挙動のみ厳格化（仕様準拠）。
- `GroupHoldoutInnerValid` の group 選定規則が変わるため、同一 seed でも inner split が変わる可能性がある（仕様準拠の挙動変更）。
- Config の ratio 範囲外指定は新たに早期エラーとなる。

### 代替案

- 現行挙動を仕様側に合わせて変更する: 監査で仕様準拠を優先する方針のため採用しない。
- `GroupHoldoutInnerValid` に `shuffle_groups` フラグを追加し両対応する: Config/API の複雑化を避けるため採用しない。
- Notebook は regression のみ更新する: 21-C で binary/multiclass への横展開方針があるため採用しない。

### 受け入れ基準

- `Model.load()` 後の `probability_histogram_plot()` が `MODEL_NOT_FIT` を返す。
- `GroupHoldoutInnerValid` が入力順末尾 group を validation に割り当て、group overlap が発生しない。
- `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` の `<=0` または `>=1` が `CONFIG_INVALID` になる。
- 3つの Notebook でスマートパラメーター確認セルが同等方針で揃う。
- 追加/更新テストが通過する。

---

## 2026-03-05: `Model.load()` 後に診断APIを利用可能にする仕様変更

- ID: `H-0026`
- Status: `accepted`
- Scope: `Public API | Persistence`
- Related: `BLUEPRINT.md §4.1, §6.5, §7.4, §15.3`
- Supersedes: `H-0025` の「1. load 後 API 境界の統一」

### 目的

`Model.load()` 後の利用体験を「推論・評価参照のみ」から「診断APIも含む」に拡張し、学習実行環境がない場面でも残差分析・SHAP 重要度・分類/校正可視化を再利用できるようにする。

対象 API:

- `residuals()`
- `residuals_plot()`
- `importance(kind="shap")`
- `roc_curve_plot()`
- `confusion_matrix()`
- `calibration_plot()`
- `probability_histogram_plot()`

### Proposal

1. `Model.load()` 後でも上記 API を利用可能とする（`fit()` 後と同等の利用境界）。
2. Exported Model Artifacts に load 後診断APIで必要な最小データを `analysis_context` として含める。
   - `y_true`（学習時ターゲット）
   - `X_for_explain`（SHAP重要度算出に必要な特徴量データ）
3. `Model.load()` は `analysis_context` を復元し、診断APIが追加データ入力なしで動作するようにする。

### 影響範囲

- `BLUEPRINT.md`（公開API境界、export/load、artifacts 契約）
- `lizyml/persistence/*`（保存/読込対象）
- `lizyml/core/model.py`（load 後 API ガード）
- `tests/test_plots/*`, `tests/test_explain/*`（load 後境界テスト）

### 互換性

- 公開 API は拡張のみで、既存メソッドの削除はない。
- 既存 artifact（`analysis_context` 未保持）については migration 方針を定義し、少なくとも以下を保証する。
  - `predict()` / `evaluate()` は従来どおり利用可能。
  - 追加された load 後診断 API は、必要データがない場合に明示的エラーを返すか、再 export を促す。

### 代替案

- 現行どおり load 後は診断 API を禁止する: ユースケース拡張の目的を満たせないため採用しない。
- 診断 API ごとに外部から `y_true`/`X` を都度受け取る: Facade 利用性が低下し API 一貫性を損なうため採用しない。

### 受け入れ基準

- `Model.load()` 後に対象 7 API が呼び出し可能である。
- `export` 成果物に `analysis_context` が含まれる。
- load 後診断 API の回帰・分類・校正系テストが通過する。
- 既存 artifact 互換方針がドキュメント化され、テストで担保される。

### Decision

- Date: `2026-03-05`
- Result: `accepted`
- Notes: API 境界を「fit 後のみ」から「fit 後 + load 後」に拡張する方針を採用。

---

## 2026-03-06: Config Reference の BLUEPRINT 反映と README デフォルト値修正

- ID: `H-0027`
- Status: `accepted`
- Scope: `Config`
- Related: `BLUEPRINT.md §5.4`

### 目的

README に記載されている Config Reference（全キー・デフォルト値・バリデーション制約の一覧表）を BLUEPRINT に正式な仕様として反映する。併せて、スキーマ実装のデフォルト値を README（仕様の正）に合わせて修正する（`min_data_in_leaf_ratio: None`→`0.01`, `min_data_in_bin_ratio: None`→`0.01`, `balanced: False`→`None`（タスク依存自動解決: regression→False, binary/multiclass→True））。

### Proposal

1. BLUEPRINT §5.4 として「Config Reference（全キー一覧）」セクションを追加し、README の Config Reference の内容を仕様として固定する。
2. スキーマ実装（`schema.py`）のデフォルト値を README に合わせて修正する。

### 影響範囲

- BLUEPRINT.md §5.4（新規セクション追加）
- `lizyml/config/schema.py`（デフォルト値の修正）

### 互換性

- デフォルト値の変更により、既存の Config で明示指定していないユーザーの動作が変わる。ただし README を参照しているユーザーにとっては期待通りの動作となる。

### 代替案

- README にのみ記載し BLUEPRINT に反映しない: 仕様の正が分散するため却下。

### 受け入れ基準

- BLUEPRINT §5.4 に全 Config キーの型・デフォルト・制約が記載されている。
- README のデフォルト値がスキーマ実装と一致している。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: 仕様の明文化。`balanced` のデフォルトは `None`（タスク依存自動解決: regression→False, binary/multiclass→True）に変更。`min_data_in_leaf_ratio=0.01`, `min_data_in_bin_ratio=0.01` をデフォルトに設定。

---

## 2026-03-06: Tuning 探索状況の可視化 (`tuning_plot`)

- ID: `H-0028`
- Status: `accepted`
- Scope: `Public API | Plots`
- Related: `BLUEPRINT.md §4.1, §13.3`

### 目的

`tune()` 実行後に探索状況を可視化する `model.tuning_plot()` を公開 API に追加する。Optuna の最適化履歴（trial ごとのスコア推移）を Plotly で描画する。

### Proposal

1. `Model.tuning_plot()` を追加する。`tune()` 未実行時は `MODEL_NOT_FIT`。
2. X 軸 = trial 番号、Y 軸 = スコア値。完了/枝刈り/失敗を色分けする。最良スコアの推移ラインも重ね描きする。
3. 実装は `plots/tuning.py` に配置し、Model には委譲のみ。
4. Plotly optional dependency。

### 影響範囲

- `BLUEPRINT.md §4.1`（公開 API 追加）
- `BLUEPRINT.md §13.3`（可視化追加）
- `lizyml/plots/tuning.py`（新規）
- `lizyml/core/model.py`（委譲メソッド追加）

### 互換性

- 追加のみ。破壊的変更なし。

### 代替案

- Optuna の built-in visualization を直接使う: Optuna 依存を公開 API に露出させるため却下。

### 受け入れ基準

- `model.tuning_plot()` が Plotly Figure を返す。
- 完了/枝刈り/失敗の trial が区別される。
- 最良スコア推移ラインが描画される。
- `tune()` 未実行時に `MODEL_NOT_FIT`。
- Plotly 未インストール時に `OPTIONAL_DEP_MISSING`。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: Phase 22 追加開発で実装。

---

## 2026-03-06: `Model.fit_result` プロパティの追加

- ID: `H-0029`
- Status: `accepted`
- Scope: `Public API`
- Related: `BLUEPRINT.md §4.1`

### 目的

`fit()` 後の `FitResult` をユーザーが直接参照できる read-only プロパティ `model.fit_result` を追加する。これにより、Notebook 等で学習結果の詳細（models, history, splits 等）を直接確認できる。

### Proposal

1. `Model.fit_result` プロパティを追加する（`@property`、read-only）。
2. `fit()` 未実行時は `MODEL_NOT_FIT`。
3. Model クラス内に新しいロジックは追加しない（既存の `self._fit_result` を返すだけ）。

### 影響範囲

- `BLUEPRINT.md §4.1`（公開 API 追加）
- `lizyml/core/model.py`（プロパティ追加のみ）

### 互換性

- 追加のみ。破壊的変更なし。

### 代替案

- `fit()` の戻り値だけで十分とする: `tune()` → `fit()` の流れで戻り値を使わない場合にアクセスできなくなるため却下。

### 受け入れ基準

- `model.fit_result` が `FitResult` を返す。
- `fit()` 未実行時に `MODEL_NOT_FIT`。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: Phase 22 追加開発で実装。

---

## 2026-03-06: Calibration に生スコア（logits）を渡す仕様の明確化

- ID: `H-0030`
- Status: `accepted`
- Scope: `Calibration`
- Related: `BLUEPRINT.md §12.1`

### 目的

現在の BLUEPRINT §12.1 では校正器の入力を「OOF スコア」と記載しているが、確率値（predict_proba の出力）なのか生スコア（logits）なのかが曖昧。LightGBM の binary タスクでは predict_proba が sigmoid 適用後の確率を返すため、現状は確率値が渡されている。しかし校正の理論的正しさの観点から、校正器には生スコア（raw score / logits。sigmoid/softmax 適用前）を渡すべきである。

### Proposal

1. BLUEPRINT §12.1 を更新し、校正器への入力は「Base モデルの OOF 生スコア（raw score / logits）」であることを明示する。
2. `EstimatorAdapter` に `predict_raw(X)` メソッドを追加し、sigmoid/softmax 適用前の生スコアを返す手段を提供する。
3. `BaseCalibratorAdapter.fit()` の入力を確率値から生スコアに変更する。
4. `BaseCalibratorAdapter.predict()` は生スコアを受け取り、校正済み確率を返す。
5. Platt / Isotonic / Beta の各実装を生スコア入力に対応させる。
6. Calibration が未指定の場合は従来どおり `predict_proba`（確率値）を OOF/IF 予測に使用する。Calibration 有効時のみ生スコアベースの校正パスに入る。

### 影響範囲

- `BLUEPRINT.md §12.1`（入力仕様の変更）
- `BLUEPRINT.md §14.1`（`predict_raw` メソッド追加）
- `lizyml/estimators/base.py`（`predict_raw` 追加）
- `lizyml/estimators/lgbm.py`（`predict_raw` 実装）
- `lizyml/calibration/base.py`（IF 変更）
- `lizyml/calibration/platt.py`, `isotonic.py`（入力変更）
- `lizyml/calibration/cross_fit.py`（raw score を渡すよう変更）
- `lizyml/training/cv_trainer.py`（OOF 生スコア生成）

### 互換性

- `BaseCalibratorAdapter` の入力形式変更は破壊的。ただし Calibration は内部 IF であり公開 API ではないため、format_version 変更は不要。
- 既存 artifact の calibrator は確率値で学習されているため、load 互換に注意が必要。

### 代替案

- 確率値入力のまま維持する: 校正の理論的正しさが損なわれるため却下。

### 受け入れ基準

- `EstimatorAdapter.predict_raw()` が生スコアを返す。
- 校正器が生スコアで学習される。
- cross-fit 校正が raw score ベースで動作する。
- Calibration 未指定時は `predict_proba` で OOF/IF を生成する（動作変更なし）。
- BLUEPRINT §12.1 に入力形式が明記されている。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: Phase 23 で実装。Calibration IF の入力を確率値から生スコアに変更。Calibration 未使用時は従来の predict_proba パスを維持。

---

## 2026-03-06: Beta Calibration の実装

- ID: `H-0031`
- Status: `accepted`
- Scope: `Calibration`
- Related: `BLUEPRINT.md §12.2`

### 目的

BLUEPRINT §12.2 で列挙されている 3 つの校正手法（Platt / Beta / Isotonic）のうち、Beta Calibration のみ未実装。これを実装する。

### Proposal

1. `lizyml/calibration/beta.py` に `BetaCalibrator(BaseCalibratorAdapter)` を実装する。
2. Beta Calibration は `a * log(s) + b * log(1-s) + c` の 3 パラメーターモデルで、`scipy.optimize.minimize` で最適化する。
3. `calibration/registry.py` の `_NOT_IMPLEMENTED` から `"beta"` を削除し、正式に登録する。

### 影響範囲

- `lizyml/calibration/beta.py`（新規）
- `lizyml/calibration/registry.py`（登録変更）

### 互換性

- Config で `method="beta"` を指定可能になる（以前は `CALIBRATION_NOT_SUPPORTED` エラー）。
- 既存の Platt / Isotonic には影響なし。

### 代替案

- 外部ライブラリ（`betacal`）を依存に追加する: optional dependency を増やしたくないため、自前実装を選択。

### 受け入れ基準

- `method="beta"` で校正が動作する。
- cross-fit + OOF-only の契約を満たす。
- Platt / Isotonic と同一の BaseCalibratorAdapter IF を実装する。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: Phase 23 で実装。

---

## 2026-03-06: PurgedTimeSeries / GroupTimeSeries の Config・Model 接続

- ID: `H-0032`
- Status: `accepted`
- Scope: `Config | Split`
- Related: `BLUEPRINT.md §5, §10.2`

### 目的

Splitter クラス（`PurgedTimeSeriesSplitter`, `GroupTimeSeriesSplitter`）は実装済みだが、Config schema に対応する `Literal` がなく、`Model._build_splitter()` にルーティングもないため、ユーザーが利用できない。Config と Model を接続する。

### Proposal

1. Config schema に `PurgedTimeSeriesConfig`（`method: Literal["purged_time_series"]`）と `GroupTimeSeriesConfig`（`method: Literal["group_time_series"]`）を追加する。
2. `SplitConfig` の Union に上記を追加する。
3. `Model._build_splitter()` に `purged_time_series` / `group_time_series` のルーティングを追加する。
4. InnerValid 自動解決テーブルに `purged_time_series` → `time_holdout`、`group_time_series` → `group_holdout` を追加する。
5. 正規化エイリアスを追加する（`purged-time-series` → `purged_time_series` 等）。

### 影響範囲

- `lizyml/config/schema.py`（Config 追加）
- `lizyml/config/loader.py`（正規化追加）
- `lizyml/core/model.py`（ルーティング追加）
- `BLUEPRINT.md §5, §10.2, §10.3`

### 互換性

- 追加のみ。既存の 4 split method に影響なし。

### 代替案

- なし。

### 受け入れ基準

- `split.method: "purged_time_series"` / `"group_time_series"` で CV が動作する。
- InnerValid が自動解決される。
- 正規化エイリアスが機能する。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: Phase 23 で実装。

---

## 2026-03-06: 時系列 fold 期間情報の表示

- ID: `H-0033`
- Status: `accepted`
- Scope: `Public API | Plots`
- Related: `BLUEPRINT.md §13.3`

### 目的

時系列分割（`time_series` / `purged_time_series` / `group_time_series`）使用時に、fold ごとの期間情報（train の終端、valid の開始）を確認できる手段を提供する。

### Proposal

1. `FitResult.splits` に `time_col` の min/max 情報を fold ごとに記録する（`time_range` フィールド: `list[dict] | None`）。
2. `model.split_summary()` メソッドを追加し、fold ごとの期間情報を `pd.DataFrame` で返す。列: `fold`, `train_start`, `train_end`, `valid_start`, `valid_end`, `train_size`, `valid_size`。
3. 時系列でない場合は `time_range` なし、`split_summary()` は size 情報のみ返す。

### 影響範囲

- `BLUEPRINT.md §7.1`（FitResult.splits 拡張）
- `BLUEPRINT.md §4.1`（公開 API 追加）
- `lizyml/core/types/fit_result.py`（フィールド追加）
- `lizyml/core/model.py`（委譲メソッド追加）

### 互換性

- FitResult に optional フィールド追加。既存 artifact の load 互換は維持（`time_range` が None の場合はサイズ情報のみ）。

### 代替案

- 可視化（Gantt chart）のみ提供する: DataFrame 出力の方が汎用性が高いため、まず DataFrame を提供。

### 受け入れ基準

- 時系列分割時に `FitResult.splits` に期間情報が含まれる。
- `model.split_summary()` が DataFrame を返す。
- 非時系列でも size 情報は返す。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: Phase 23 で実装。

---

## 2026-03-06: Logging 出力先の統一

- ID: `H-0034`
- Status: `accepted`
- Scope: `Logging`
- Related: `BLUEPRINT.md §17`

### 目的

BLUEPRINT §17 で規定されている「`run_id` に基づく出力先（logs / artifacts / plots）の統一」が未実装。run_id ベースのディレクトリ管理を実装する。

### Proposal

1. `Model` に `output_dir` オプションを追加する（`Config` の `output` セクション or コンストラクタ引数）。
2. `output_dir` 指定時、`run_id` ベースのサブディレクトリ（`{output_dir}/{run_id}/`）を自動作成し、ログ・plot 保存先とする。（H-0111 注記: `fit()` / `tune()` がこのディレクトリに自動で書くのは `run.log` だけで、path 無しの `export()` は `{run_dir}/export` に書く（H-0039）。plot API は plotly の Figure を返すだけで、plot をファイルに書く経路は無い。BLUEPRINT §17 はこの実装を書く。#318）
3. `output_dir` 未指定時は現行動作（ログは標準出力、plot は返却のみ）を維持する。

### 影響範囲

- `BLUEPRINT.md §17`（仕様の具体化）
- `lizyml/core/logging.py`（出力先管理）
- `lizyml/core/model.py`（output_dir の受け渡し）

### 互換性

- `output_dir` はオプションのため既存動作に影響なし。

### 代替案

- MLflow 等の外部ツールに委ねる: 将来の拡張点として残すが、最小限の自前管理は必要。

### 受け入れ基準

- `output_dir` 指定時に `{output_dir}/{run_id}/` が作成される。
- ログファイルが出力先に保存される。
- 未指定時は既存動作を維持する。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: Phase 23 で実装。

---

## 2026-03-06: 解決済みパラメーターテーブル API

- ID: `H-0035`
- Status: `accepted`
- Scope: `Public API`
- Related: `BLUEPRINT.md §4.1`

### 目的

Notebook の「4.1 LightGBM Parameters」セルで手動実装しているパラメーター確認コードを、Model の公開メソッドとして提供する。ユーザーが booster 内部にアクセスする必要をなくし、1 行で解決済みパラメーターを確認できるようにする。

### Proposal

1. `model.params_table()` メソッドを追加する。
   - 戻り値: `pd.DataFrame`（index: `parameter`, 単一列: `value`）。
   - Config 由来の smart params（`auto_num_leaves`, `num_leaves_ratio`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`, `balanced`, `feature_weights`）と training 設定（`early_stopping.rounds`, `validation_ratio`）を含む。
   - fold 0 の学習済み booster から取得した解決済みネイティブパラメーター（`objective`, `num_leaves`, `min_data_in_leaf`, `min_data_in_bin`, `max_bin`, `learning_rate`, `max_depth`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l2`, `num_iterations` 等）を含む。
   - Config smart params（ratio 等）と resolved params（絶対値）は名前が異なるため衝突しない。同一テーブルに混在させることで、ユーザーは「指定した ratio」と「解決された絶対値」を対比確認できる。
   - 末尾に fold ごとの `best_iteration` 行を追加する。
   - `fit()` 未実行時は `MODEL_NOT_FIT` を送出する。
2. 出力イメージ:
   ```
                             value
   parameter
   objective                huber
   learning_rate            0.001
   max_depth                    5
   auto_num_leaves           True
   num_leaves_ratio           1.0
   num_leaves                  32
   min_data_in_leaf_ratio    0.01
   min_data_in_leaf           540
   min_data_in_bin_ratio    0.001
   min_data_in_bin             54
   max_bin                    511
   feature_fraction           0.7
   bagging_fraction           0.7
   bagging_freq                10
   lambda_l2             0.000001
   balanced                 False
   early_stopping_rounds      150
   validation_ratio           0.1
   num_iterations            1500
   best_iteration_0           487
   best_iteration_1           512
   ...
   ```

### 影響範囲

- `BLUEPRINT.md §4.1`（公開 API 追加）
- `lizyml/core/model.py`（委譲メソッド追加）
- Notebook の「4.1」セルを `model.params_table()` 1 行に置き換え可能

### 互換性

- 新規メソッド追加のみ。既存 API に変更なし。

### 代替案

- 2 列（`config` / `resolved`）で対比する: ratio → 絶対値の対応を明示的に示せるが、多くのパラメーターで片方が空欄になり冗長。単一列で十分識別可能（名前が異なるため）。

### 受け入れ基準

- `model.params_table()` が `pd.DataFrame` を返す。
- Config smart params と resolved booster params が同一テーブルに含まれる。
- fold ごとの `best_iteration` が含まれる。
- `fit()` 未実行時に `MODEL_NOT_FIT` を送出する。
- Notebook の「4.1」セルを `model.params_table()` に置き換えて動作確認。

---

## 2026-03-06: Smart Parameter の n_rows 基準を inner train サイズに変更

- ID: `H-0036`
- Status: `accepted`
- Scope: `Result の意味・shape（smart param 解決ロジック）`
- Related: `BLUEPRINT.md §5.3`

### 目的

Smart parameter（`min_data_in_leaf_ratio`, `min_data_in_bin_ratio`）の `n_rows` 基準が、現在は `fit()` に渡された全データセットサイズを使用している。実際にモデルが学習するデータは outer fold 分割 + inner valid 分割後のサブセットであり、5-fold + validation_ratio=0.1 の場合は全体の約 72% に減少する。ratio パラメーターの意図（実際の学習データサイズに対する割合）と乖離するため、n_rows を inner train サイズ（early stopping 用 validation 分割後）に変更する。

### Proposal

1. smart parameter の `n_rows` を「CVTrainer の各 fold における inner_valid 分割後の学習データ行数」とする。
2. `Model.fit()` での一括解決（現行）を廃止し、`CVTrainer.fit()` 内の fold ループ内で、inner_valid 分割後に smart params を解決する。
3. `auto_num_leaves` は `max_depth` のみに依存し `n_rows` を使わないため影響なし。`num_leaves_ratio` も `max_depth` ベースのため影響なし。影響を受けるのは `min_data_in_leaf_ratio` と `min_data_in_bin_ratio` のみ。
4. Tuner の trial 内でも同様に、CVTrainer 内部で fold ごとに解決する。
5. BLUEPRINT §5.3 の記述を更新し、`n_rows` の定義を明確化する。
6. `feature_weights` と `balanced`（`sample_weight`）は n_rows に依存しないため影響なし。

### 影響範囲

- `BLUEPRINT.md §5.3`（n_rows の定義明確化）
- `lizyml/core/model.py`（smart param 解決の移動）
- `lizyml/training/cv_trainer.py`（fold 内での smart param 解決追加）
- `lizyml/estimators/lgbm.py`（`resolve_smart_params` のインターフェース変更の可能性）
- `lizyml/tuning/tuner.py`（trial 内 smart param 解決の変更）
- `params_table()` の出力（fold ごとに異なる可能性のある値の表示方針）

### 互換性

- ratio の解決値が変わるため、同一 Config でも以前と異なる `min_data_in_leaf` / `min_data_in_bin` 値が生成される（破壊的変更）。
- ただし Artifacts の `format_version` や公開 API のシグネチャには影響しない。
- 既存の保存済みモデルには影響しない（解決済みパラメーターは booster に格納済み）。

### 代替案

1. **全データセットサイズ基準を維持し仕様明確化のみ**: 安定性・再現性の観点で合理的だが、ratio の意味が「全データに対する割合」に固定される。
2. **outer fold 基準**: inner valid 分割は考慮しない中間案。fold 間で均等分割なら安定するが、不均等分割（時系列等）では fold ごとに異なる。

### 受け入れ基準

- `min_data_in_leaf_ratio=0.01` で 5-fold + validation_ratio=0.1 の場合、解決値が全データの 0.72% 付近（inner train サイズ基準）になることをテストで確認。
- fold ごとの解決値が inner train サイズに基づいて正しく計算されること。
- Tuner の trial 内でも同一ロジックが適用されること。
- `params_table()` が fold 0 の解決値を正しく表示すること。
- 既存テストの回帰確認（seed 固定テストの期待値更新が必要な場合あり）。

---

## 2026-03-06: Phase 22 監査乖離クローズ — ドキュメント整合修正

- ID: `H-0037`
- Status: `accepted`
- Scope: `ドキュメント整合（BLUEPRINT 文言修正 + Notebook/テスト補完）`
- Related: `BLUEPRINT.md §5.2, §5.3, §5.4`, `PLAN.md Phase 22`

### 目的

Phase 22 監査で検出された BLUEPRINT の記述乖離（§5.3 balanced デフォルト、§5.2/§5.3 LGBMConfig 例）を実装/§5.4 と統一する。合わせて Notebook の feature_weights 解決後値確認セルと静的テストの不足を補完し、監査乖離を完全にクローズする。

### 対象

1. **BLUEPRINT §5.3 balanced 記述**: `balanced: bool = False` → `balanced: bool | None = None`（タスク依存自動解決: regression→False, binary/multiclass→True）。§5.4 Config Reference と一致させる。
2. **BLUEPRINT §5.2/§5.3 LGBMConfig 例**: `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` / `balanced` の説明文を現仕様（デフォルト値・自動解決ロジック）と一致するよう微修正。
3. **Notebook**: `tutorial_regression_tuning_lgbm.ipynb` に feature_weights (resolved) の確認セルを追加（設定時のみ表示）。
4. **テスト**: `tests/test_notebooks/test_notebook_cells.py` に feature_weights 解決後値確認セルの存在検証を追加。

### 影響範囲

- BLUEPRINT.md の文言修正のみ。公開 API / Config / Result の shape は変更なし。
- Notebook セル追加と静的テスト追加は既存動作に影響なし。

### 互換性

- 破壊的変更なし。

### 受け入れ基準

- BLUEPRINT §5.3 の balanced デフォルト記述が §5.4 Config Reference と一致していること。
- BLUEPRINT §5.2/§5.3 の LGBMConfig 例が現仕様と一致していること。
- `tutorial_regression_tuning_lgbm.ipynb` に feature_weights (resolved) セルがあること。
- Notebook 静的テストが feature_weights 解決後値確認セルの存在を検証すること。

### Decision

- Date: `2026-03-06`
- Result: `accepted`
- Notes: 変更ゲート非該当（文言修正 + テスト追加）。BLUEPRINT §5.2/§5.3 を修正し、開発タスクは Phase 22 の 22-O として追加。

---

## 2026-03-06: Phase 23 監査フォローアップ（23-C: BLUEPRINT準拠）

- ID: `H-0038`
- Status: `accepted`
- Scope: `Config | Split`
- Related: `BLUEPRINT.md §5.4, §10.2`, `PLAN.md Phase 23`

### Context

Requirements Audit の結果、Phase 23-C について BLUEPRINT と実装の乖離が確認された。  
BLUEPRINT §5.4 は `purged_time_series` の固有キーを `purge_gap` / `embargo_pct` と定義している一方、現実装は `purge_window` / `gap` を受け付けている。

本件は公開 Config 契約（split 設定）に該当するため、BLUEPRINT を正として整合させる方針を明示する。

### Proposal

1. `purged_time_series` の正式キーは BLUEPRINT 記載どおり `purge_gap` / `embargo_pct` とする。
2. `config/schema.py`・`config/loader.py`・`core/model.py`・splitter 実装を上記キー契約に合わせて更新する。
3. 既存ユーザー向けに `purge_window` / `gap` は移行期間中のみ後方互換として受け付け、明示警告を出す。
4. `embargo_pct` の split 動作をテストで固定し、リーク防止境界を明文化する。

### Impact

- `lizyml/config/schema.py`
- `lizyml/config/loader.py`
- `lizyml/core/model.py`
- `lizyml/splitters/purged_time_series.py`
- `tests/test_config/*`, `tests/test_e2e/test_time_series_splits.py`

### Compatibility

- 公開 Config 契約の是正であり、最終的には破壊的（legacy key 廃止時）。
- ただし移行期間を設け、legacy key を警告付きで受理することで段階移行可能とする。

### Alternatives Considered

1. 実装に合わせて BLUEPRINT を `purge_window` / `gap` に変更する  
   - 不採用。ユーザー指示（23-C は BLUEPRINT を正とする）と矛盾するため。
2. 互換レイヤーなしで即時切替する  
   - 不採用。既存 Config 利用者への影響が大きいため。

### Acceptance Criteria

- `split.method: "purged_time_series"` で `purge_gap` / `embargo_pct` が有効に解釈される。
- `purge_window` / `gap` 指定時は警告付きで同等動作し、移行案内が表示される。
- `embargo_pct` を含む split でリーク防止境界のテストが追加され、期待どおりに通過する。
- BLUEPRINT §5.4 / §10.2 と実装・テストのキー名が一致する。

### Migration

- 既存 Config の `purge_window` / `gap` は `purge_gap` / `embargo_pct` に置換する。
- 移行期間中は legacy key を警告付きで受理し、将来削除時期をリリースノートで告知する。

---

## 2026-03-06: Phase 23 監査フォローアップ（23-F: output_dir 契約完了）

- ID: `H-0039`
- Status: `accepted`
- Scope: `Config | Logging`
- Related: `BLUEPRINT.md §17`, `PLAN.md Phase 23`

### Context

Requirements Audit の結果、23-F は部分達成。  
現状は `Model(..., output_dir=...)` + `fit()` の経路のみ動作し、BLUEPRINT §17 の「Config or コンストラクタ」「fit/tune/export の統一出力先」要件を満たし切れていない。

### Proposal

1. `output_dir` を Config からも指定可能にする（優先順位は `constructor > config > 未指定`）。
2. `fit` だけでなく `tune` / `export` でも `{output_dir}/{run_id}/` を作成し、ログ出力を統一する。
3. 既存の未指定時挙動（標準出力中心、返却API中心）は維持する。

### Impact

- `lizyml/config/schema.py`
- `lizyml/core/model.py`
- `lizyml/core/logging.py`
- `tests/test_core/test_logging_output.py`（拡張）

### Compatibility

- 追加機能であり後方互換。
- `output_dir` 未指定ユーザーの挙動変更はない。

### Alternatives Considered

1. コンストラクタ引数のみ対応のまま維持する  
   - 不採用。BLUEPRINT §17 の契約に未達のため。
2. `fit` のみ対応のまま維持する  
   - 不採用。run 管理の統一要件を満たせないため。

### Acceptance Criteria

- Config 経由で `output_dir` を指定した場合に run ディレクトリが作成される。
- `fit` / `tune` / `export` の各経路で run ディレクトリとログファイルが作成される。
- コンストラクタ引数と Config 両方がある場合、優先順位がテストで保証される。
- 未指定時の既存挙動が回帰しない。

### Migration

- 移行必須なし（任意で Config に `output_dir` を追加可能）。

---

## 2026-03-07: TimeSeries CV 方針更新（time_col基準統一 + embargo改名）

- ID: `H-0040`
- Status: `accepted`
- Scope: `Config | Split | InnerValid`
- Related: `BLUEPRINT.md §5.4, §6.2, §10.2, §10.3`, `PLAN.md Phase 23`

### Context

TimeSeries 系 split（`time_series` / `purged_time_series` / `group_time_series`）の仕様が、`time_col` の扱い・パラメーター命名・ウィンドウ制御の観点で統一されていない。  
現状は「行順ベース」の実装が混在しており、ユーザーが `time_col` を指定しても split ロジックがその列で明示的にソートする契約になっていない。

### Proposal

1. 3 メソッド共通で `data.time_col` を必須化し、split 前に `time_col` 昇順で並べてから分割する。
2. 3 メソッド共通でウィンドウ制御キー `train_size_max` / `test_size_max` を持つ。
3. `time_series` / `group_time_series` は `gap`、`purged_time_series` は `purge_gap` を継続し、3 メソッドでギャップ指定を共通概念として扱う。
4. `purged_time_series` の `embargo_pct`（`float`）を `embargo`（`int`、Obs 数指定）に改名・型変更する。`gap` / `purge_gap` と同じ単位に統一。
5. 既存ユーザー向けに `embargo_pct` は移行期間中のみ警告付きで受理し、`int()` 変換の上 `embargo` へ正規化する。

### Impact

- `lizyml/config/schema.py`（split config 契約の更新）
- `lizyml/config/loader.py`（正規化・後方互換）
- `lizyml/core/model.py`（time_col 必須チェック、split 構築）
- `lizyml/splitters/time_series.py`
- `lizyml/splitters/purged_time_series.py`
- `lizyml/splitters/group_time_series.py`
- `lizyml/training/cv_trainer.py`（time_col 昇順前処理の適用位置に応じて）
- `tests/test_splitters/*`, `tests/test_e2e/test_time_series_splits.py`, `tests/test_e2e/test_split_summary.py`

### Compatibility

- `embargo_pct` -> `embargo` は公開 Config 契約の変更を含むため、最終的には破壊的。
- 移行期間中は `embargo_pct` を警告付き互換として受理し、段階移行可能にする。
- `time_col` 必須化は既存の「行順依存」設定に影響するため、エラーメッセージと移行ガイドを明示する。

### Alternatives Considered

1. 現行の「行順前提」運用を継続し、`time_col` 必須化しない  
   - 不採用。データ前処理依存で誤用しやすく、仕様の再現性を下げるため。
2. `embargo_pct` 名を維持して文言だけ調整する  
   - 不採用。指定単位の誤解が残るため、命名統一を優先。

### Acceptance Criteria

- 3 メソッドで `data.time_col` 未指定時は `CONFIG_INVALID` となる。
- `time_col` 非昇順データを与えても、`time_col` 昇順での分割結果が再現される。
- 3 メソッドすべてで `train_size_max` / `test_size_max` が有効に解釈される。
- `purged_time_series` で `embargo` が有効に動作する。
- `embargo_pct` 指定時は警告を出しつつ `embargo` と同等動作になる。
- 既存の leakage 防止テストと split_summary テストが回帰しない。

### Migration

- `split.method: "purged_time_series"` を使う既存 Config は `embargo_pct` を `embargo` に置換する。
- `time_series` / `purged_time_series` / `group_time_series` を使う既存 Config は `data.time_col` を必ず指定する。
- 既存の並び替え前提コードは、`time_col` の値が期待どおりの順序を持つことを確認する。

---

## 2026-03-07: LGBMAdapter: sklearn wrapper → Booster API 移行

- ID: `H-0041`
- Status: `accepted`
- Scope: `EstimatorAdapter | Training | Persistence`
- Related: `BLUEPRINT.md §14.2, §14.3`, `PLAN.md Phase 24`

### Context

LightGBM の sklearn wrapper（`LGBMRegressor` / `LGBMClassifier`）に、`early_stopping` callback 併用時に `model_to_string()` が空文字列を返す間欠バグが存在する（microsoft/LightGBM#7186）。
このバグは sklearn wrapper 内部の後処理（`engine.py:350` で `keep_training_booster=False` 時に実行される `model_from_string(model_to_string())` ラウンドトリップ）に起因し、約 5-10% の確率で `LightGBMError: Model file doesn't specify the number of classes` を発生させる。

LightGBM の Booster API（`lgb.train()`）では `keep_training_booster=True` がデフォルトであり、上記ラウンドトリップが発生しないため、このバグの影響を受けない。実際に 100 回の検証で 0 回の失敗を確認済み。

### Proposal

`LGBMAdapter.fit()` の内部実装を sklearn wrapper（`LGBMRegressor` / `LGBMClassifier`）から LightGBM Booster API（`lgb.train()`）に移行する。

1. **`fit()`**: `lgb.Dataset` を構築し、`lgb.train(params, train_set, valid_sets=[...], callbacks=[...], keep_training_booster=True)` で学習する。
2. **`predict()`**: `booster.predict(X)` を使用。regression はそのまま返却。classification は `objective` に応じて sigmoid/softmax 適用済みの値が返る。
3. **`predict_proba()`**: `booster.predict(X)` を使用。binary は `(n,)` → `(n, 2)` に変換。multiclass は `(n, k)` をそのまま返却。
4. **`predict_raw()`**: `booster.predict(X, raw_score=True)` を使用（現状と同じロジック、`booster_` 経由のアクセスが不要になる）。
5. **`importance()`**: `booster.feature_importance(importance_type=...)` を直接呼び出す。
6. **`get_native_model()`**: 戻り値を `lgb.Booster` に変更する。
7. **`best_iteration`**: `booster.best_iteration` から取得する。
8. **パラメーター変換**: sklearn 固有のパラメーター名（`n_estimators` → `num_boost_round`、`random_state` → `seed`）を Booster API に適切にマッピングする。

### Impact

- `lizyml/estimators/lgbm.py`（主要変更: fit / predict / predict_proba / get_native_model / _build_params）
- `lizyml/training/cv_trainer.py`（`evals_result_` → Booster API の `eval_results` への適応）
- `lizyml/training/refit_trainer.py`（同上）
- `lizyml/core/model.py`（`params_table()` の `.booster_` アクセスを `.get_native_model()` 直接に変更）
- `lizyml/explain/shap_explainer.py`（SHAP TreeExplainer は Booster を直接受け取れるため変更不要、ただし確認は必要）
- `lizyml/persistence/exporter.py`（joblib シリアライズ対象が Booster に変わるため確認）
- `tests/test_estimators/` `tests/test_e2e/`（`get_native_model()` 戻り値型、`.booster_` アクセスの更新）

### Compatibility

- **公開 API（`get_native_model()`）**: 戻り値が `LGBMRegressor | LGBMClassifier` → `lgb.Booster` に変更される。これは内部型（sklearn wrapper vs Booster）の変更であり、LightGBM 固有の下流コードに影響する。
- **`predict()` / `predict_proba()` / `predict_raw()` の shape 契約**: 変更なし。同一の入出力 shape を維持する。
- **`importance()` の shape 契約**: 変更なし。
- **Persistence**: joblib による `LGBMAdapter` のシリアライズ。`lgb.Booster` の `model_to_string()` / `model_from_string()` による保存・復元が必要。ただし `format_version=1` の互換性を維持するため、既存の保存済みモデル（sklearn wrapper ベース）のロードは引き続きサポートする必要がある。
- **SHAP**: `TreeExplainer` は `lgb.Booster` を直接受け取れるため互換性あり。

### Alternatives Considered

1. **テスト時に retry を追加して間欠エラーを許容する**
   - 不採用。根本原因が LightGBM の既知バグである以上、回避策を持つべき。ユーザー利用時にも影響する。
2. **`model_to_string()` 出力を post-fit で検証し、空の場合に再学習する**
   - 不採用。`LightGBMError` は `model_from_string()` 内部で raise されるため、post-fit 検証が間に合わない。
3. **`keep_training_booster=True` を sklearn wrapper に渡す**
   - 不採用。sklearn wrapper は `keep_training_booster` を外部パラメーターとして公開していない。
4. **LightGBM バージョンを制約する**
   - 不採用。4.3〜4.6 のすべてで再現するため、特定バージョンの除外では解決しない。

### Acceptance Criteria

- regression / binary / multiclass の全タスクで `lgb.train()` 経由の学習が動作する。
- `predict()` / `predict_proba()` / `predict_raw()` の出力 shape が移行前と同一である。
- `importance(kind="split")` / `importance(kind="gain")` が移行前と同一の結果を返す。
- `get_native_model()` が `lgb.Booster` を返す。
- `best_iteration` が正しく取得される。
- early stopping が正常に動作する（inner valid あり / なしの両方）。
- 学習履歴（`eval_history`）が cv_trainer / refit_trainer で正しく記録される。
- SHAP（`TreeExplainer`）が Booster 直接入力で動作する。
- 既存の persistence（export / load）が動作する。
- 既存テスト（782件）が回帰しない。
- notebook テスト（`tutorial_regression_tuning_lgbm.ipynb`）の間欠エラーが解消される。

### Migration

- `get_native_model()` の戻り値を `LGBMRegressor | LGBMClassifier` → `lgb.Booster` に変更。既存コードで `.booster_` 経由でアクセスしていた箇所は `.get_native_model()` 直接に変更する。
- `format_version=1` の既存保存モデルのロード互換は維持する（移行期間中は旧形式を検出して復元可能にする）。

---

## 2026-03-07: Model Facade の Mixin 分割

- ID: `H-0042`
- Status: `accepted`
- Scope: `Core | Architecture`
- Related: `BLUEPRINT.md §4.1, §19, 付録B`

### Context

`core/model.py` は Facade として assembly と delegation に徹しているが、1,451行・30+メソッドに肥大化している。メソッドは機能グループごとに明確に分かれており（plot系8メソッド、table/accessor系7メソッド、persistence系3メソッド等）、mixin による分割で可読性・保守性を改善できる。

公開 API（`Model` クラスのメソッドシグネチャ・戻り値）は一切変更しない。内部ファイル構成の変更のみ。

### Proposal

`core/model.py` を以下の mixin モジュールに分割し、`Model` クラスを多重継承で組み立てる。

1. **`core/_model_plots.py`** — `ModelPlotsMixin`
   - `residuals_plot()`, `roc_curve_plot()`, `calibration_plot()`, `probability_histogram_plot()`, `importance_plot()`, `plot_learning_curve()`, `plot_oof_distribution()`, `tuning_plot()`
   - 8メソッド、約300行

2. **`core/_model_tables.py`** — `ModelTablesMixin`
   - `evaluate_table()`, `residuals()`, `confusion_matrix()`, `importance()`, `params_table()`, `split_summary()`, `tuning_table()`
   - 7メソッド、約350行

3. **`core/_model_persistence.py`** — `ModelPersistenceMixin`
   - `export()`, `_resolve_export_path()`, `load()` (classmethod)
   - 3メソッド、約150行

4. **`core/model.py`** — `Model(ModelPlotsMixin, ModelTablesMixin, ModelPersistenceMixin)`
   - `__init__()`, `fit()`, `predict()`, `evaluate()`, `tune()`
   - プライベートヘルパー: `_build_splitter()`, `_build_inner_valid()`, `_make_inner_valid_factory()`, `_build_run_meta()`, `_require_fit()`, `_require_refit()`, `_load_data()`, `fit_result` プロパティ
   - 約600行

各 mixin は `self` の型を `Model` と仮定し、`_require_fit()` 等の共通ヘルパーを呼び出す。`TYPE_CHECKING` ガードで循環参照を回避する。

### Impact

- `lizyml/core/model.py`（分割元）
- `lizyml/core/_model_plots.py`（新規）
- `lizyml/core/_model_tables.py`（新規）
- `lizyml/core/_model_persistence.py`（新規）

### Compatibility

- **公開 API**: 変更なし。`from lizyml import Model` の利用者コードに影響しない。
- **import パス**: `lizyml.core.model.Model` は維持。mixin は `_` プレフィックスの非公開モジュール。
- **Persistence**: 変更なし（`format_version` 影響なし）。

### Alternatives Considered

1. **現状維持（分割しない）**
   - 不採用。1,451行は可読性の限界を超えており、今後のメソッド追加で悪化する。
2. **機能ごとに独立クラスに委譲（Composition パターン）**
   - 不採用。`model.plots.importance_plot()` のように API が変わり、破壊的変更になる。
3. **サブモジュールに分割し `__init__.py` で再 export**
   - 不採用。mixin の方がシンプルで、既存テストへの影響が最小。

### Acceptance Criteria

- `Model` の全既存テスト（861件）が回帰しない。
- `from lizyml import Model` および `from lizyml.core.model import Model` が引き続き動作する。
- 各 mixin ファイルが mypy strict でエラーゼロ。
- `model.py` が 700行以下に収まる。
- ruff lint / format がクリーン。

---

## 2026-03-07: テスト基盤の改善（conftest 集約・parametrize 強化・CI 拡張）

- ID: `H-0043`
- Status: `accepted`
- Scope: `Testing | CI`
- Related: `BLUEPRINT.md §18.1, §18.2`

### Context

テストスイート（861件、97%カバレッジ）は高品質だが、以下の保守性課題がある：

1. **ヘルパー関数の重複**: `_reg_df()`, `_bin_df()`, `_cfg()` 等のデータ生成ヘルパーが8+ファイルで重複定義されている。変更時に複数箇所の同期が必要。
2. **parametrize の活用不足**: タスク別（regression/binary/multiclass）テストが個別メソッドで書かれており、パラメタライズで統合できる余地がある。
3. **CI が develop PR 非対応**: 現状 main への PR のみで CI が実行される。develop への PR でも品質ゲートを回すべき。
4. **カバレッジ閾値なし**: `--cov-fail-under` が未設定で、カバレッジ回帰を検知できない。
5. **slow テストのローカルスキップ**: `@pytest.mark.slow` が定義されているが、ローカル開発時にデフォルトスキップする設定がない。
6. **optional dependency の "missing" テスト不足**: plotly / scipy の未インストール時パスが未テスト。

### Proposal

1. **conftest.py へのヘルパー集約**
   - `tests/conftest.py` に共通ヘルパー（`make_regression_df()`, `make_binary_df()`, `make_multiclass_df()`, `make_config()`）を定義する。
   - 各テストファイルのローカルヘルパーを conftest のヘルパーに置き換える。
   - データ生成は `seed` パラメーターを持ち、再現性を保証する。

2. **parametrize 強化**
   - E2E テスト（`test_e2e/`）でタスク横断のテストを `@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])` で統合する。
   - メトリクステストでタスク別の重複を削減する。

3. **CI の develop ブランチ対応**
   - `ci.yml` の `on.pull_request.branches` に `develop` を追加する。

4. **カバレッジ閾値の設定**
   - `pytest` 実行時に `--cov-fail-under=95` を追加する。

5. **slow テストのローカルデフォルトスキップ**
   - `pyproject.toml` の `[tool.pytest.ini_options]` に `addopts = "-m 'not slow'"` を追加する。
   - CI の main PR では明示的に `-m ""` で全テストを実行する。develop PR では slow を除外する。

6. **optional dependency の "missing" テスト追加**
   - plotly / scipy の未インストール時に `OPTIONAL_DEP_MISSING` エラーが発生することを検証するテストを追加する。

### Impact

- `tests/conftest.py`（大幅拡張）
- `tests/test_e2e/`（ヘルパー置換、parametrize 統合）
- `tests/test_config/`, `tests/test_training/`, `tests/test_tuning/` 等（ヘルパー置換）
- `.github/workflows/ci.yml`（develop 追加、`--cov-fail-under`）
- `pyproject.toml`（`addopts` 追加）
- `tests/test_plots/`, `tests/test_calibration/`（optional dep テスト追加）

### Compatibility

- **公開 API**: 変更なし。テスト基盤のみの変更。
- **テスト結果**: 既存テストの pass/fail は変わらない（リファクタリングのみ）。
- **CI**: develop PR でもゲートが走るようになる（追加のみ、既存動作に影響なし）。

### Alternatives Considered

1. **conftest を tests/ 直下ではなくサブディレクトリごとに配置**
   - 部分採用。共通ヘルパーは `tests/conftest.py`、サブディレクトリ固有のフィクスチャはサブディレクトリの `conftest.py` に配置する。
2. **pytest-lazy-fixture 等のプラグイン導入**
   - 不採用。依存を増やさず、標準の `conftest.py` + `pytest.fixture` で十分。

### Acceptance Criteria

- 共通ヘルパー（`_reg_df` 等）の重複定義が `tests/conftest.py` に集約され、各テストファイルからローカル定義が除去される。
- 全テスト（861件以上）が回帰しない。
- CI が `develop` ブランチへの PR でも実行される。
- カバレッジが 95% 未満の場合に CI が失敗する。
- `uv run pytest` でローカル実行時に slow テストがスキップされる。
- plotly / scipy の未インストール時テストが追加される。
- mypy / ruff がクリーン。

---

## 2026-03-09: Calibration CV の splitter 統一（BLUEPRINT 不一致の解消）

- ID: `H-0044`
- Status: `accepted`
- Scope: `Calibration | Split | Leakage`
- Related: `BLUEPRINT.md §10.1, §10.4, §10.5, §12.1`, `PLAN.md Phase 26`

### Context

BLUEPRINT §10.1 では「Splitter は外側 CV / early stopping / calibration で共通利用する」と定義されている。一方、現行実装の calibration cross-fit は `KFold` 固定で分割しており、`split.method` が `group_kfold` / `time_series` / `purged_time_series` / `group_time_series` の場合でも、group/time 制約を継承していない。

この不一致により、仕様上は守るべき分割境界（group overlap 禁止、時系列順、purge/embargo）が calibration 段階で崩れる余地がある。

### Proposal

1. calibration cross-fit の分割生成を `split.method` ベースに統一する。
   - `split.method` の family（kfold/stratified/group/time/purged/group_time）を calibration でも使用する。
   - fold 数のみ `calibration.n_splits` で上書きできるようにする。
2. `calibration.n_splits` と `split.n_splits` は独立値として維持する（一致必須にはしない）。
3. calibration 分割は splitters IF 経由で生成し、`KFold` 直接依存を廃止する。
4. `fit_result.splits.calibration` には実際に使用した calibration split を必ず保存する。
5. group/time 系で必要な補助情報（`groups`、時系列ソート後の行順）を calibration 分割にも適用する。
6. 分割不能（例: `n_splits` 過大、group 数不足、時系列条件不成立）は明示的なエラーで失敗させる。

### Impact

- `lizyml/calibration/cross_fit.py`（分割生成責務の見直し）
- `lizyml/core/model.py`（calibration 分割生成・引き渡し）
- `lizyml/core/_model_factories.py`（calibration splitter 構築ヘルパー追加）
- `tests/test_calibration/`（split.method 別の契約テスト追加）
- `tests/test_e2e/test_leakage_all.py`（group/time 境界の回帰防止テスト追加）

### Compatibility

- 公開 Config 形式は維持（`calibration.method`, `calibration.n_splits` は変更なし）。
- `split.method` が group/time 系の既存ユーザーは、calibration 分割の挙動が「ランダムKFold」から「split.method 準拠」に変わるため、`calibrated_oof` と関連メトリクス値が変化しうる。
- Artifacts shape / `format_version` 変更は不要（`splits.calibration` は既存フィールド内）。

### Alternatives Considered

1. 現行実装に合わせて BLUEPRINT を「calibration は KFold 固定」に修正する。
   - 不採用。BLUEPRINT の split/leakage 方針（outer/inner/calibration 一貫性）と矛盾するため。
2. `calibration.split_method` を新設して outer split と切り離す。
   - 不採用。公開 Config 拡張が必要で複雑化が大きい。まずは既存 `split.method` 継承で整合させる。
3. `calibration.n_splits` を廃止して `split.n_splits` に強制統一する。
   - 不採用。校正CVの分解能を独立に調整したい需要があるため。

### Acceptance Criteria

- `split.method="kfold"` で calibration 有効時、`len(splits.calibration) == calibration.n_splits` になる。
- `split.method="stratified_kfold"` で calibration 各 fold のラベル分布が極端に崩れない（層化分割として成立）。
- `split.method="group_kfold"` で calibration 各 fold に group overlap がない。
- `split.method="time_series"` で calibration 各 fold が時系列順（train < valid）を満たす。
- `split.method="purged_time_series"` で calibration 各 fold が `purge_gap + embargo` を満たす。
- `split.method="group_time_series"` で calibration 各 fold が group/time 境界を満たす。
- `split.n_splits != calibration.n_splits` のケースで outer と calibration が独立に動作する。
- 既存の leakage テスト（`cross-fit OOF != c_final`）が回帰しない。

### Decision

- Date: `2026-03-09`
- Result: `accepted`
- Notes: BLUEPRINT §10.5 / §12.1 に既に規定済みの契約を実装に反映する。`refactor/phase-26-calibration-split` ブランチで実施。

---

## 2026-03-09: evaluate_table の fold 列を OOF-per-fold に変更

- ID: `H-0045`
- Status: `accepted`
- Scope: `Evaluation | Public API | Contracts`
- Related: `BLUEPRINT.md §6.3, §7.1, §13.2`, `PLAN.md Phase 27`

### Context

現行の `evaluate_table()` は `fold_0..fold_N-1` に `if_per_fold`（train_idx 上の IF メトリクス）を表示している。実務上、fold ばらつきの確認は汎化性能（OOF）で行うことが多く、IF fold 値を `fold_n` として表示すると解釈ミスを誘発しやすい。

### Proposal

1. Evaluator の raw metrics に `oof_per_fold` を追加する。
   - 各 fold の `valid_idx` 上で metric を計算した dict の list（長さ = outer n_splits）。
2. `evaluate_table()` の `fold_0..fold_N-1` は `oof_per_fold` を表示する。
3. 既存の `if_mean` / `if_per_fold` は互換性のため維持する。
4. `evaluate_table()` の列意味を明記する。
   - `oof`: 全 OOF 集約値
   - `fold_n`: fold n の OOF（valid_idx）値
   - `if_mean`: IF 指標（参考値）

### Impact

- `lizyml/evaluation/evaluator.py`（`oof_per_fold` 追加）
- `lizyml/evaluation/table_formatter.py`（`fold_n` 参照元変更）
- `lizyml/core/types/fit_result.py`（metrics 契約 doc 更新）
- `tests/test_evaluation/`（契約テスト更新・追加）
- `tests/test_core/test_contracts.py`（metrics 階層ゴールデン更新）

### Compatibility

- `evaluate()` の raw 構造に `oof_per_fold` が追加される（後方互換な追加）。
- `evaluate_table()` の `fold_n` の意味は IF -> OOF に変わるため、値解釈は破壊的変更。
- `if_mean` / `if_per_fold` を維持することで、IF を参照する既存ユースケースは継続可能。
- Artifacts の top-level shape 変更はなく、`format_version` 変更は不要。

### Alternatives Considered

1. `fold_n` を維持し、`oof_fold_n` を別列追加する
   - 不採用。列が冗長になり、どちらを見るべきかが曖昧になるため。
2. IF 関連（`if_mean`, `if_per_fold`）を完全削除する
   - 不採用。既存利用との互換性影響が大きく、監査・デバッグ用途の需要が残るため。
3. `evaluate_table()` から fold 列を削除する
   - 不採用。fold ばらつき監視の要求を満たせないため。

### Acceptance Criteria

- `evaluate(metrics=[...])["raw"]` に `oof_per_fold` が含まれる。
- `oof_per_fold[i]` は `splits.outer[i][1]`（valid_idx）上で計算した metric と一致する。
- `evaluate_table()` の `fold_n` が `oof_per_fold[n]` を表示する。
- `if_mean` と `if_per_fold` は従来どおり計算・取得できる。
- 既存の OOF/calibration 契約テストが回帰しない。

### Decision

- Date: `2026-03-09`
- Result: `accepted`
- Notes: BLUEPRINT §7.1 / §13.2 に既に規定済みの契約を実装に反映する。`refactor/phase-27-oof-per-fold` ブランチで実施。

---

## 2026-03-09: 評価・可視化 API の IF/OOF 目的分類の明文化

- ID: `H-0046`
- Status: `accepted`
- Scope: `Evaluation | Plots | Public API | Contracts`
- Related: `BLUEPRINT.md §13.4 (新規)`, `PLAN.md Phase 28`

### Context

`evaluate_table()` の fold 列を OOF に変更（H-0045）した際、他の可視化・テーブル API にも IF（train_idx）と OOF（valid_idx）が混在していることが判明した。各 API が「診断目的（IF: 過学習検知）」と「汎化監視目的（OOF: モデル評価）」のどちらを主目的とするか、BLUEPRINT に正式な分類がない。

### Proposal

1. BLUEPRINT §13 に「評価・可視化 API の目的分類」サブセクション（§13.4）を追加する。
2. 既存 API を以下の 3 カテゴリに分類し、各 API のデータソースを明記する:
   - **汎化監視（OOF 優先）**: `evaluate_table()`（fold 列 = OOF）、`evaluate()`
   - **診断（IF + OOF 比較）**: `roc_curve_plot()`（IS/OOS）、`confusion_matrix()`（is/oos）、`residuals_plot()`（IS/OOS）
   - **学習過程監視**: `plot_learning_curve()`（train/valid loss）
3. 分類の原則: IS(In-Sample) = IF(train_idx) 集約値、OOS(Out-of-Sample) = OOF(valid_idx) 値。
4. 既存 API の挙動自体は変更しない（仕様の明文化のみ）。

### Impact

- `BLUEPRINT.md §13`（新規 §13.4 追加）
- 実装変更なし（明文化のみ）

### Compatibility

- 既存 API の挙動変更なし。
- ドキュメント・仕様の補足のみ。

### Alternatives Considered

1. 全 API を OOF のみに統一する
   - 不採用。IS/OOS 比較は過学習検知に有用であり、診断 API として残す価値がある。
2. 分類を BLUEPRINT に書かず、docstring のみで管理する
   - 不採用。API の目的が仕様として固定されないと、将来の変更で一貫性が崩れるリスクがある。

### Acceptance Criteria

- BLUEPRINT §13.4 に API 目的分類テーブルが追加される。
- 各 API のデータソース（IF/OOF/両方）が明記される。
- 既存テストが回帰しない（実装変更なし）。

### Decision

- Date: `2026-03-09`
- Result: `accepted`
- Notes: BLUEPRINT §13.4 に API 目的分類テーブルを追加済み。Phase 28 で確定。

---

## 2026-03-14: Isotonic Calibration LightGBM パラメーター強化

- ID: `H-0047`
- Status: `accepted`
- Scope: `Calibration | Internal`
- Related: `BLUEPRINT.md §12.2, §14.2.1`

### Context

現在の `IsotonicCalibrator` は最小限のデフォルトパラメーター（`n_estimators=200`, `max_depth=3`, `learning_rate=0.05`）のみで Early Stopping がなく、過学習リスクがある。また `LGBMRegressor`（sklearn wrapper）を使用しており、H-0041 で決定した Booster API（`lgb.train()`）統一方針と不整合がある。

### Proposal

1. `LGBMRegressor` → `lgb.train()`（Booster API）に移行する（H-0041 準拠）。
2. デフォルトパラメーターを以下に強化する（ユーザーは `calibration.params` で上書き可能）:
   ```python
   _ISOTONIC_DEFAULTS = {
       "objective": "binary",
       "metric": "binary_logloss",
       "monotone_constraints": [1],          # 常に強制（上書き不可）
       "monotone_constraints_method": "advanced",
       "num_leaves": 7,
       "max_depth": 3,
       "min_data_in_leaf_ratio": 0.01,       # fit 時に絶対値に解決
       "learning_rate": 0.03,
       "lambda_l2": 5.0,
       "min_gain_to_split": 0.0,
       "feature_fraction": 1.0,
       "bagging_fraction": 1.0,
       "bagging_freq": 0,
   }
   ```
3. `num_boost_round=1000` + Early Stopping（`patience=100`）を導入する。
4. Early Stopping 用 validation: calibration 学習データから 10% をランダムサンプリングする（`validation_ratio=0.1`, `seed=42` デフォルト、ユーザー上書き可能）。
5. `min_data_in_leaf_ratio` は fit 時に `min_data_in_leaf = max(1, ceil(n_train * ratio))` に解決する。
6. `objective="binary"` の Booster API predict は raw score を返すため、sigmoid 適用 + `np.clip(0, 1)` で確率に変換する。
7. calibration データが少数（< 20 行）の場合は Early Stopping を無効化して全データで学習する。

### Impact

- `lizyml/calibration/isotonic.py` — 実装変更（Booster API 移行 + パラメーター強化）
- `tests/test_calibration/test_isotonic_calibration.py` — テスト更新・追加

### Compatibility

- CalibrationResult の shape/contract は不変。
- 数値結果はデフォルトパラメーター変更により変わる。
- 公開 API（`CalibrationConfig`）の変更なし。既存の `calibration.params` dict で上書き可能。

### Alternatives Considered

1. sklearn wrapper のまま Early Stopping だけ追加する
   - 不採用。H-0041 の Booster API 統一方針と不整合が残る。
2. `IsotonicRegression`（sklearn）に置き換える
   - 不採用。LightGBM の単調制約のほうが柔軟であり、BLUEPRINT §12.2 の設計意図に合致する。

### Acceptance Criteria

- `IsotonicCalibrator` が `lgb.train()` を使用している。
- デフォルトパラメーターが Proposal 通りに設定されている。
- Early Stopping（patience=100）が機能し、1000 round 前に停止する。
- 内部 validation split（10%）が seed 固定で再現可能。
- ユーザーが `calibration.params` でデフォルトを上書きできる。
- `monotone_constraints=[1]` が常に強制される。
- 出力が [0, 1] 範囲、単調性を維持。
- 少サンプル（< 20 行）で Early Stopping が自動無効化される。
- 全テストが pass。

### Decision

- Date: `2026-03-14`
- Result: `accepted`
- Notes: Booster API 移行、デフォルトパラメーター強化、Early Stopping + 内部 validation split を実装済み。BLUEPRINT §12.2 に詳細を追加。Phase 29 で確定。

---

## 2026-03-14: Tuning Progress Callback

- ID: `H-0048`
- Status: `accepted`
- Scope: `Public API | Tuning`
- Related: `BLUEPRINT.md §4.1, §11`

### Context

`tune()` 実行時は trial 数が多いと待ち時間が長くなるが、進行状況を外部ツール（Widget 等）に通知する手段がない。外部ツール開発者向けに、trial ごとの進捗情報をリアルタイムで提供するコールバック API が必要。

### Proposal

1. `TuneProgressInfo` frozen dataclass を追加する:
   ```python
   @dataclass(frozen=True)
   class TuneProgressInfo:
       current_trial: int        # 現在の trial 番号（1-indexed）
       total_trials: int         # 全 trial 数
       elapsed_seconds: float    # 経過時間（秒）
       best_score: float | None  # これまでの最良スコア（None = まだ complete なし）
       latest_score: float | None  # 直近 trial のスコア（None = fail/pruned）
       latest_state: str         # "complete" | "pruned" | "fail"
   ```
2. `TuneProgressCallback` 型エイリアスを追加する:
   ```python
   TuneProgressCallback = Callable[[TuneProgressInfo], None]
   ```
3. `Tuner.__init__` に `progress_callback: TuneProgressCallback | None = None` パラメーターを追加する。
4. `Model.tune()` に `progress_callback: TuneProgressCallback | None = None` パラメーターを追加する。
5. Optuna の `study.optimize(callbacks=[...])` を活用して、各 trial 完了時に `TuneProgressInfo` を構築して `progress_callback` に渡す。
6. `progress_callback` 内で例外が発生した場合は catch して warning に変換し、tuning を中断させない。
7. `TuneProgressInfo` と `TuneProgressCallback` を `lizyml/__init__.py` の公開面に追加する。

### Impact

- `lizyml/core/types/tuning_result.py` — `TuneProgressInfo` dataclass + `TuneProgressCallback` 型追加
- `lizyml/tuning/tuner.py` — コールバック統合
- `lizyml/core/model.py` — `Model.tune()` シグネチャ変更
- `lizyml/__init__.py` — 公開面追加
- `tests/test_tuning/test_tuning_progress.py` — 新規テスト

### Compatibility

- 後方互換。`progress_callback` はデフォルト `None` で既存動作に影響なし。
- `TuningResult` の shape/contract は不変。

### Alternatives Considered

1. ログベースの進捗報告（`logging` 出力のみ）
   - 不採用。外部ツールがログをパースする必要があり、構造化されたコールバックのほうが使いやすい。
2. イベントバス / Pub-Sub パターン
   - 不採用。現時点ではコールバック 1 つで十分であり、過度に複雑化する。
3. `tqdm` / progress bar の表示
   - 不採用。CUI 向けの表示であり、Widget 等の外部ツールには不適切。コールバックのほうが汎用的。

### Acceptance Criteria

- `Model.tune(progress_callback=fn)` でコールバックが各 trial 完了時に呼ばれる。
- `TuneProgressInfo` の各フィールドが正しい値を持つ。
- `current_trial` が 1 から n_trials まで順に増加する。
- `elapsed_seconds >= 0` である。
- `best_score` が最初の complete trial 以降は `None` でない。
- `progress_callback=None`（デフォルト）で既存動作に影響なし。
- コールバック内例外が tuning を中断させない。
- `TuneProgressInfo` と `TuneProgressCallback` が `from lizyml import ...` で import 可能。
- 全テストが pass。

### Decision

- Date: `2026-03-14`
- Result: `accepted`
- Notes: `TuneProgressInfo` / `TuneProgressCallback` を定義し、`Tuner` / `Model.tune()` にコールバック統合を実装済み。BLUEPRINT §4.1 / §11.4 に記載。Phase 29 で確定。

---

## 2026-03-14: multiclassova 使用時の確率正規化

- ID: `H-0049`
- Status: `accepted`
- Scope: `Evaluation`
- Related: `BLUEPRINT.md §8, lizyml/evaluation/evaluator.py`

### Context

`objective="multiclassova"` で学習した場合、LightGBM の `booster.predict()` は各クラスに独立した sigmoid を適用するため、行ごとの合計が 1.0 にならない。sklearn の `roc_auc_score(multi_class="ovr")` は合計 1.0 をハードバリデーションしており、非正規化の出力を渡すと `ValueError` が発生する。`brier` / `logloss` も確率分布を前提とするため値が不正確になる。

起票元: LizyML-Widget (multiclass Fit で AUC 評価エラー)。

### Proposal

- `predict_proba()` の契約は変更しない（生の sigmoid 出力を返し続ける）。
- 評価パイプラインの責務として、`_pred_for_metric()` 内で `needs_proba=True` かつ `multiclass` かつ 2D の場合に行正規化を適用する。
- `_normalize_multiclass_proba()` を新設し、行ごとに `pred / row_sums` で正規化する（all-zero 行のゼロ除算ガード付き）。
- `multiclass` (softmax) の場合は既に合計 ≈ 1.0 のため冪等（no-op）。

### Impact

- 変更対象: `lizyml/evaluation/evaluator.py` の `_pred_for_metric()` 1 関数のみ。
- `predict_proba()` / `predict()` / `BaseMetric` Protocol / 個別メトリクスクラス / `_TASK_METRICS` は変更しない。

### Compatibility

- 後方互換。`multiclass` (softmax) では正規化が冪等のため出力値は実質不変。
- `multiclassova` 使用時のメトリクス値が修正される（バグ修正の性質）。

### Alternatives Considered

1. `predict_proba()` で正規化する（案 A）
   - 不採用。生の sigmoid 出力を保持する要件がある（LizyML-Widget 側で生値を使用）。
2. `BaseMetric` に `needs_normalized_proba` 属性を追加する（案 C）
   - 当初不採用としたが、レビューにより **採用に変更**（`needs_simplex` として実装）。
   - `auc_pr` / `brier` は per-class OvR 計算のため行正規化するとクラス内ランキングが変わる。
   - simplex が必要なメトリクス（`auc`, `logloss`）のみ正規化すべき。

### Acceptance Criteria

- `multiclassova` の非正規化出力で `roc_auc_score` がエラーなく動作する。
- softmax 出力は `assert_allclose` で実質不変。
- `needs_simplex=True` メトリクス（AUC, LogLoss）のみ行正規化される。
- `needs_simplex=False` メトリクス（AUCPR, Brier）は raw 値を受け取る。
- all-zero 行でゼロ除算が発生しない。
- binary / regression の `_pred_for_metric` は影響を受けない。
- `needs_proba=False` のメトリクスは影響を受けない。
- 全テストが pass。

### Decision

- Date: `2026-03-14`
- Result: `accepted`
- Notes: Evaluator 層での行正規化（案 B）を採用。ただし正規化対象を `needs_simplex=True` メトリクスに限定（案 C を統合）。`BaseMetric.needs_simplex` をデフォルト `False` の concrete property として追加し、`AUC` / `LogLoss` のみ `True` にオーバーライド。per-class OvR メトリクス（AUCPR, Brier）は raw 値を受け取る。

---

## 2026-03-15: Smart Parameter 統一 & TrainComponents 導入

- ID: `H-0050`
- Status: `accepted`
- Scope: `Training | Tuning | Result`
- Related: `BLUEPRINT.md §5.3, §6.1, §6.2, §7.2, §11.2`

### Context

現状 `resolve_smart_params`（fit 用、`LGBMConfig` を受け取る）と `resolve_smart_params_from_dict`（tune 用、`dict` を受け取る）の 2 関数が存在し、対応する smart params の範囲が非対称（tune 版は `feature_weights` / `balanced` を未対応）。また `TuningResult.best_params` が flat dict であるため、tune → fit 時に smart params のカテゴリ区別が失われ、Config 側の固定値で上書きされてしまう問題がある。fit / tune で CVTrainer への組み立てロジックも重複しており、一貫性・保守性を損なっている。

### Proposal

1. **`resolve_smart_params` を dict ベースに統一**: 第 1 引数を `LGBMConfig` → `dict[str, Any]` に変更。`extract_smart_params(config: LGBMConfig) -> dict` ヘルパーを追加。`resolve_smart_params_from_dict` を削除。fit / tune で同一関数を使用する。

2. **`TuningResult` をカテゴリ別に変更**: `best_params`（flat dict）を `best_model_params` / `best_smart_params` / `best_training_params` に分割。互換性のため `best_params` を computed property（flat view）として残す。

3. **`TrainComponents` 導入**: パラメータ解決結果を保持する dataclass（`estimator_factory` / `sample_weight` / `ratio_resolver` / `inner_valid`）。`Model._build_train_components()` で構築し、CVTrainer と RefitTrainer に同一インスタンスを渡すことで一貫性を構造的に保証する。

4. **`Model.fit()` / `Model.tune()` の共通化**: 両者とも `_build_train_components()` を経由して CVTrainer を構築する。tune の各 trial は `_build_train_components(model_params=..., smart_params=...)` を呼び、fit と同じコードパスを通る。

5. **`Tuner` のシンプル化**: Tuner の責務を Optuna study 管理のみに縮小。`objective` クロージャは Model 側で構築して注入する。Tuner から LGBM 固有の import をすべて除去する。

6. **`Model._best_params` 削除**: `_tuning_result` からカテゴリ別に取得する。パラメータ優先順位: `Config defaults < tune best < fit() 引数`。

### Impact

- `lizyml/estimators/lgbm.py`: `resolve_smart_params` 引数変更、`extract_smart_params` 追加、`resolve_smart_params_from_dict` 削除
- `lizyml/core/types/tuning_result.py`: field 構成変更
- `lizyml/core/model.py`: `TrainComponents` 追加、`_build_train_components` / `_merge_params` 追加、fit() / tune() 書き換え、`_best_params` 削除
- `lizyml/tuning/tuner.py`: コンストラクタ縮小、LGBM 固有ロジック除去

- 変更しないもの: `CVTrainer` / `RefitTrainer` / `config/schema.py` / `search_space.py` のインターフェース

### Compatibility

- `TuningResult.best_params` は computed property として残すため、読み取り側は互換。ただし `TuningResult` のコンストラクタは変更される（`best_params` → `best_model_params` + `best_smart_params` + `best_training_params`）。
- `Tuner` のコンストラクタは大幅に縮小されるが、内部 API のため外部互換性は影響なし。
- `resolve_smart_params_from_dict` は削除されるが、内部 API のため外部互換性は影響なし。

### Alternatives Considered

1. `TuningResult.best_params` に `overrides` 引数を追加する（fit 側で overrides 適用）
   - 不採用。カテゴリの区別が曖昧なまま残り、将来のアルゴリズム追加時に同じ問題が再発する。
2. EstimatorBuilder パターン（B案）を先に導入する
   - 不採用（段階的に実施）。tune → fit の smart params 問題を先に解決し、クリーンな状態で B案を検討する。

### Acceptance Criteria

- `resolve_smart_params_from_dict` が削除され、fit / tune が同一の `resolve_smart_params(dict, ...)` を使用している。
- `TuningResult` が `best_model_params` / `best_smart_params` / `best_training_params` を持ち、`best_params` property が flat view を返す。
- `_build_train_components()` が CVTrainer と RefitTrainer に同一の factory / resolver を提供している。
- tune() の各 trial が `_build_train_components()` を経由して CVTrainer を構築している。
- `Tuner` が LGBM 固有の import を持たない。
- tune → fit で smart params（`num_leaves_ratio` 等）が正しく引き継がれるテストが存在する。
- 既存テスト（910件）がすべて pass する。

### Decision

- Date: `2026-03-15`
- Result: `accepted`
- Notes: 議論の結果、「Config → Tune → Fit の一連のフローで同一コードパスを通る」設計を優先。B案（EstimatorBuilder）は本 Proposal 完了後に段階的に検討する。

---

## 2026-03-16: デッドコード削除と Foundation 整理

- ID: `H-0051`
- Status: `accepted`
- Scope: `Architecture | Internal`
- Related: `BLUEPRINT.md §2, §19, ARCHITECTURE.md`

### Context

アーキテクチャレビューの結果、以下のデッドコードと構造上の問題が発見された:
1. 本番コードで未使用のクラス/モジュールが複数存在する（`TargetTransformer`, `SplitPlan`, `HoldoutSplitter`, 未使用 Spec 群, `import_optional.py`）。
2. `EstimatorRegistry` / `SplitterRegistry` が `@register` デコレータで書き込みされるが `.get()` が呼ばれない（書き込み専用）。`MetricRegistry` / `CalibratorRegistry` は正当な lookup がある。
3. `types/` が `data/` に依存している（`DataFingerprint` の import）。ARCHITECTURE.md で定義した Layer 0 → Layer 1 の逆依存。
4. `splitters/` ↔ `specs/` の循環依存が存在する。

これらは ARCHITECTURE.md で定義した 5 層カテゴリアーキテクチャ（Layer 0: Foundation → Layer 1: Leaf → Layer 2: Composition → Layer 3: Optional → Layer 4: Facade）の前提条件である「DAG 構造」と「各 Leaf カテゴリの独立性」に違反する。

### Proposal

1. **デッドコード削除**:
   - `lizyml/features/transformers/target_transformer.py` を削除（完全未使用スタブ）
   - `lizyml/core/specs/split_plan.py` を削除（`_model_factories` に完全置換済み）
   - `lizyml/splitters/holdout.py` を削除（`SplitPlan` 経由のみ、他に呼び出し元なし）
   - `lizyml/core/specs/export_spec.py` を削除（未使用）
   - `lizyml/core/specs/training_spec.py` の `TrainingSpec` / `EarlyStoppingSpec` / `InnerValidSpec` を削除（未使用）
   - `lizyml/core/specs/calibration_spec.py` を削除（未使用）
   - `lizyml/core/specs/tuning_spec.py` を削除（未使用）
   - `lizyml/config/loader.py` の `config_to_split_spec` / `config_to_training_spec` / `config_to_tuning_spec` / `config_to_calibration_spec` / `config_to_export_spec` / `config_to_problem_spec` / `config_to_feature_spec` を削除（本番で未使用。`ProblemSpec` / `FeatureSpec` は `model.py` で直接構築しているため変換関数は不要）
   - `lizyml/utils/import_optional.py` を削除（全箇所がインライン try/except を使用）
   - `lizyml/splitters/__init__.py` の `_build_splitter(SplitSpec)` を削除（`_model_factories` の Config ベースと重複）

2. **書き込み専用 Registry の削除**:
   - `EstimatorRegistry` の `@register` デコレータを `LGBMAdapter` から除去し、`EstimatorRegistry` クラスを `registries.py` から削除
   - `SplitterRegistry` の `@register` デコレータを全 Splitter クラスから除去し、`SplitterRegistry` クラスを `registries.py` から削除
   - `MetricRegistry` / `CalibratorRegistry` は `.get()` 呼び出しがあるため維持

3. **DataFingerprint の移動**:
   - `lizyml/data/fingerprint.py` の `DataFingerprint` dataclass を `lizyml/core/types/artifacts.py` に移動
   - `compute()` 関数は `lizyml/data/fingerprint.py` に残す（`data/` が Foundation の型を返す形になり、逆依存が解消される）

4. **循環依存の解消**:
   - `specs/split_plan.py` 削除により `splitters/ → specs/` → `splitters/` の循環が自動解消

### Impact

- 削除対象はすべて本番コードで未使用（テストのみで使用）。公開 API・Result shape・format_version に影響なし。
- `ProblemSpec` / `FeatureSpec` / `SplitSpec` は維持（`ProblemSpec` / `FeatureSpec` は `model.py` で使用中、`SplitSpec` は将来の Spec-based パスの可能性を残す）。

### Compatibility

- 公開 API の変更なし。内部モジュールの削除のみ。
- `DataFingerprint` の import パスが `lizyml.data.fingerprint.DataFingerprint` → `lizyml.core.types.artifacts.DataFingerprint` に変更されるが、内部 API のため外部互換性は影響なし。

### Alternatives Considered

1. デッドコードを残し、将来の実装に備える
   - 不採用。Spec 層は `_model_factories` の直接 Config パスに完全に置き換えられており、復活の見込みがない。デッドコードの存在は保守性を下げ、アーキテクチャの理解を妨げる。

### Acceptance Criteria

- 削除対象ファイルがリポジトリに存在しないこと
- `splitters/ ↔ specs/` の循環依存が解消されていること
- `types/` から `data/` への依存が解消されていること（`DataFingerprint` が `types/artifacts.py` に存在すること）
- `EstimatorRegistry.get()` / `SplitterRegistry.get()` がコードベースに存在しないこと
- 既存テスト（962件）から削除対象のテストを除いた全テストが pass すること
- `ruff check` / `mypy` がクリーンであること

---

## 2026-03-16: Layer 間依存の浄化

- ID: `H-0052`
- Status: `accepted`
- Scope: `Architecture | Training | Evaluation`
- Related: `BLUEPRINT.md §2, §6.2, §6.3, §13.2, §14, §19, ARCHITECTURE.md`

### Context

ARCHITECTURE.md で定義した 5 層カテゴリアーキテクチャにおいて、以下の Layer ルール違反が存在する:

1. **training/ → evaluation/**: `cv_trainer.py` が `evaluation/oof.py` の `fill_oof`, `get_fold_pred`, `init_oof` を import している。これらは OOF アセンブリの ndarray ユーティリティであり、metric 計算（Evaluator の責務）とは無関係。Layer 2 の同層間依存。
2. **evaluation/ → calibration/**: `evaluator.py` が `CalibrationResult` を `isinstance` チェックし、calibrated metrics を直接組み立てている。Layer 2 → Layer 1 への不要な依存。Evaluator の責務は「raw predictions + y_true → metrics dict」であるべき。
3. **estimators/ → config/**: `lgbm.py` の `extract_smart_params(LGBMConfig)` が `config/schema.py` の `LGBMConfig` を直接参照している。Layer 1 の Leaf 間依存（Leaf カテゴリは互いに依存してはならない）。

### Proposal

1. **OOF ヘルパーを training/ に移動**:
   - `evaluation/oof.py` の `fill_oof`, `get_fold_pred`, `get_fold_raw`, `init_oof` を `training/oof_assembly.py`（新規）に移動
   - `cv_trainer.py` の import を `from lizyml.training.oof_assembly import ...` に変更
   - `evaluation/oof.py` は空にするか、後方互換の re-export のみ残す（内部 API のため即削除も可）

2. **Evaluator から calibration 依存を除去**:
   - `evaluator.py` の `evaluate()` は raw predictions のみを受け取り、`{"raw": {...}}` のみを返す
   - calibrated metrics の組み立ては **Facade**（`model.py` の `fit()` 内）が担当する: calibrated OOF を `evaluator.evaluate()` に別途渡して結果を `{"calibrated": {...}}` として統合する
   - `evaluator.py` から `CalibrationResult` の import と `isinstance` チェックを除去

3. **estimators/ から config/ 依存を除去**:
   - `extract_smart_params(LGBMConfig) -> dict` を `estimators/lgbm.py` → Facade（`model.py` または `_model_factories.py`）に移動
   - `lgbm.py` の `resolve_smart_params` / `resolve_ratio_params` は既に dict ベースのため変更不要
   - `lgbm.py` から `from lizyml.config.schema import LGBMConfig` を除去

### Impact

- **training/**: `cv_trainer.py` の import パスのみ変更。ロジックは同一。
- **evaluation/**: `Evaluator.evaluate()` の返り値から `"calibrated"` キーが消える。calibrated metrics は Facade 側で追加される。最終的な `FitResult.metrics` の shape は変更なし。
- **estimators/**: `lgbm.py` から `LGBMConfig` 依存が消える。`resolve_smart_params` は dict を受け取るため影響なし。
- **Facade**: `model.py` の `fit()` に calibrated metrics 組み立てロジックが追加される（Evaluator を2回呼ぶ形）。`extract_smart_params` の呼び出し元が移動する。

### Compatibility

- 公開 API の変更なし。`FitResult.metrics` の最終 shape は不変。
- `Evaluator.evaluate()` の返り値 shape が変更されるが、Evaluator は内部 API。

### Alternatives Considered

1. `evaluation/oof.py` を `core/` の共有ユーティリティに移動する
   - 不採用。OOF アセンブリは training loop 固有のロジックであり、Foundation に置く正当性がない。
2. Evaluator に calibrated_oof を引数で渡す（Evaluator 内で `{"calibrated": ...}` を生成）
   - 候補として残す。ただし Evaluator が CalibrationResult 型を知らなくても済む設計が優先。

### Acceptance Criteria

- `training/` から `evaluation/` への import が存在しないこと
- `evaluation/` から `calibration/` への import が存在しないこと
- `estimators/` から `config/` への import が存在しないこと
- `FitResult.metrics` の最終 shape が不変であること（`{"raw": {...}, "calibrated": {...}}` 構造を維持）
- 既存テスト（962件）がすべて pass すること
- カテゴリ間依存分析スクリプトで Layer ルール違反がゼロであること

---

## 2026-03-16: EstimatorProvider 導入（マルチアルゴリズム準備）

- ID: `H-0053`
- Status: `accepted`
- Scope: `Architecture | Estimators | Public API (internal)`
- Related: `BLUEPRINT.md §2, §14, §14.1, §19, §20, ARCHITECTURE.md`
- Depends: `H-0051, H-0052`

### Context

H-0050 で `_build_train_components` / `_merge_params` により fit/tune の共通化を達成した。しかし `model.py` は依然として LGBM に直接依存している:

```python
from lizyml.estimators.lgbm import LGBMAdapter, extract_smart_params, ...
isinstance(model_cfg, LGBMConfig)  # _merge_params 内 ×2
LGBMAdapter(task=..., params=...)  # make_estimator 内
default_space(cfg.task)            # tune() 内
```

EntityEmbedding 等の新アルゴリズムを追加するとき、`model.py` に `isinstance(model_cfg, EntityEmbeddingConfig)` を追加し続ける設計は持続可能でない。ARCHITECTURE.md の Layer ルール「Facade 以外の Layer は具象クラスを型ディスパッチしない」に違反する。

### Proposal

1. **EstimatorProvider protocol を定義** (`estimators/provider.py`):

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
   ```

2. **LGBMProvider を実装** (`estimators/lgbm/provider.py`):
   - 既存の `extract_smart_params`, `resolve_smart_params`, `resolve_ratio_params`, `default_space`, `default_fixed_params` を LGBMProvider のメソッドとして再配置
   - `build_estimator_factory` で `LGBMAdapter` を生成
   - `build_pipeline_factory` で `NativeFeaturePipeline` を返す

3. **Provider の解決** (`estimators/registry.py` または `_model_factories.py`):
   - `get_provider(model_cfg: ModelConfig) -> EstimatorProvider`
   - `ModelConfig` の `name` フィールドで dispatch（`"lgbm"` → `LGBMProvider`）
   - この dispatch は Facade 層（`_model_factories.py`）に置く

4. **model.py の書き換え**:
   - `from lizyml.estimators.lgbm import ...` を除去
   - `_merge_params` / `_build_train_components` / `tune` を provider 経由に変更
   - `isinstance(model_cfg, LGBMConfig)` チェックを除去

5. **新アルゴリズム追加時の手順** (目標):
   - `estimators/<name>/` に adapter + provider + config を作成
   - `config/schema.py` の `ModelConfig` union に追加
   - `_model_factories.py` の provider dispatch に追加
   - **model.py の変更: ゼロ**

### Impact

- `model.py`: LGBM 直接 import を除去。provider 経由に書き換え。
- `estimators/lgbm.py`: 既存関数を `LGBMProvider` に再配置。関数自体のロジックは変更なし。
- `tuning/search_space.py`: `default_space` / `default_fixed_params` を LGBMProvider に移動。`parse_space` / `suggest_params` / `split_by_category` は汎用のため tuning/ に残す。

### Compatibility

- 公開 API の変更なし。`Model.fit()` / `Model.tune()` / `Model.predict()` のシグネチャは不変。
- `EstimatorProvider` は内部 protocol。ユーザーが直接触れることはない。
- `LGBMAdapter` の import パスは `lizyml.estimators.lgbm.LGBMAdapter` を維持（`__init__.py` で re-export）。

### Alternatives Considered

1. Abstract base class (`ABCEstimatorProvider`) を使う
   - 不採用。Protocol の方が structural subtyping で柔軟。optional dependency のアルゴリズム（torch 系）でクラス継承を強制しない。
2. Factory 関数群を module-level で定義し、dict dispatch する
   - 不採用。Protocol の方が型安全で、mypy でチェック可能。
3. 現状の `isinstance` dispatch を維持し、新アルゴリズムごとに `elif` を追加する
   - 不採用。Open-Closed Principle に違反し、model.py がアルゴリズム追加のたびに変更される。

### Acceptance Criteria

- `model.py` に `from lizyml.estimators.lgbm import` が存在しないこと
- `model.py` に `isinstance(model_cfg, LGBMConfig)` が存在しないこと
- `LGBMProvider` が `EstimatorProvider` protocol を満たすこと（mypy で検証）
- 新アルゴリズム追加のテンプレートとして `estimators/lgbm/` のディレクトリ構造がドキュメント化されていること
- 既存テスト（962件）がすべて pass すること
- tune → fit で smart params が正しく引き継がれるテストが維持されていること

---

## H-0054: EstimatorProvider 完全化 — 残存 LGBM 固有依存の排除

- ID: `H-0054`
- Status: `accepted`
- Scope: `Architecture | EstimatorProvider | Internal`
- Related: `BLUEPRINT.md §2.1, §2.2, §14.4, HISTORY.md H-0053`
- Depends: `H-0053`

### Background

H-0053 で `EstimatorProvider` protocol を導入し、`model.py` のゼロ LGBM import を達成した。
しかしアーキテクチャ監査の結果、Facade 周辺と Layer 2 に LGBM 固有の知識が残存しており、
2つ目のアルゴリズム（EntityEmbedding 等）追加時にクラッシュまたはサイレント不具合が発生する。

### Problem Statement

| # | 問題 | 影響度 | ファイル |
|---|---|---|---|
| 1 | `cv_trainer.py` / `refit_trainer.py` が `categorical_feature=cat_cols or "auto"` を直書き | HIGH — 非 LGBM アダプタで `TypeError` | `training/cv_trainer.py:257`, `training/refit_trainer.py:124` |
| 2 | `_model_tables.py:params_table()` が `isinstance(model_cfg, LGBMConfig)` + `booster.params` 前提 | HIGH — 非 LGBM で `AttributeError` | `core/_model_tables.py:235` |
| 3 | `_build_run_meta` に `"lightgbm": _ver("lightgbm")` ハードコード | HIGH — BLUEPRINT §2.2「model.py 変更ゼロ」違反 | `core/model.py:713` |
| 4 | `estimators/provider.py` が `tuning/search_space.py` の `SearchDim` を import | HIGH — L1→L2 逆依存 | `estimators/provider.py:20` |
| 5 | `model.py:tune()` が `est.early_stopping_rounds = esr` を属性直書き | MEDIUM — 異名アダプタで silent no-op | `core/model.py:452` |
| 6 | `shap_explainer.py` が `NativeFeaturePipeline` を直 import | MEDIUM — L3→L1 具象依存 | `explain/shap_explainer.py:133` |
| 7 | `model.py` 836 行（800 行上限超過） | LOW — 保守性 | `core/model.py` |
| 8 | テスト `make_config()` が `"lgbm"` ハードコード、`fit()` docstring が "LightGBM" | LOW — 拡張時の障壁 | `tests/_helpers.py:125`, `core/model.py:144` |

### Proposed Changes

#### Phase A: 構造修正（2つ目のアルゴリズム追加の前提条件）

**A1. `SearchDim` 型を Foundation に移動**
- `SearchDim`, `FloatDim`, `IntDim`, `CategoricalDim`, `DimCategory` を `core/types/search_dim.py` に移動
- `tuning/search_space.py` は `parse_space`, `suggest_params`, `split_by_category`（Optuna 依存のロジック）のみ残す
- `estimators/provider.py` の import を `core/types/search_dim.py` に変更
- 影響: L1→L2 逆依存が解消

**A2. `categorical_feature` をアダプタ契約に移動**
- `BaseEstimatorAdapter` に `set_categorical_features(cols: list[str] | None) -> None` メソッド追加（デフォルト no-op）
- `LGBMAdapter` でオーバーライドし、`fit()` 内で `categorical_feature` kwarg に変換
- `cv_trainer.py` / `refit_trainer.py` から `categorical_feature=` kwarg 削除
- `TrainComponents` に `categorical_features: list[str] | None` フィールド追加
- CVTrainer は `estimator.set_categorical_features(cat_cols)` を呼び、その後 `estimator.fit()` を呼ぶ

**A3. `EstimatorProvider` に `runtime_deps()` / `params_summary()` 追加**
- `runtime_deps(self) -> dict[str, str]`: アルゴリズム固有の依存パッケージとバージョンを返す
- `params_summary(self, model: BaseEstimatorAdapter, model_cfg: Any) -> list[dict[str, Any]]`: params_table 用のパラメータ行を返す
- `LGBMProvider` で実装（現在 `_model_tables.py` にあるロジックを移植）

**A4. `_model_tables.py` から LGBMConfig 依存除去**
- `params_table()` を `provider.params_summary()` 経由に書き換え
- `LGBMConfig` import 削除
- `booster.params.get(k)` 直接参照を削除

**A5. `_build_run_meta` から `"lightgbm"` ハードコード除去**
- `provider.runtime_deps()` を呼び、返り値を `deps_versions` にマージ
- provider を `_build_run_meta` の引数に追加

#### Phase B: 品質改善

**B1. `early_stopping_rounds` を Provider 経由に**
- `EstimatorProvider.build_estimator_factory()` に `early_stopping_rounds` パラメータは既にある
- `model.py:tune()` の属性直書き（`est.early_stopping_rounds = esr`）を、`provider.build_estimator_factory()` 再呼び出しに変更

**B2. `shap_explainer` の Pipeline 復元を Provider 経由に**
- `compute_shap_importance` の引数に `pipeline_factory: Callable[[], BaseFeaturePipeline]` を追加
- `NativeFeaturePipeline` 直 import を削除
- Facade（`_model_plots.py`）が `provider.build_pipeline_factory()` を渡す

**B3. `model.py` ヘルパー抽出**
- `_has_metric_content`, `_filter_metrics` を `core/_model_metrics.py` に抽出
- `model.py` を 800 行以内に

**B4. テスト・docstring 整備**
- `make_config()` に `model_name: str = "lgbm"` パラメータ追加
- `fit()` docstring の "LightGBM parameters" を "Model parameters" に変更
- `Evaluator` docstring から "calibrated" 言及を削除
- `lgbm/__init__.py` に `LGBMProvider` re-export 追加

### Compatibility

- 公開 API の変更なし。`Model.fit()` / `Model.tune()` / `Model.predict()` のシグネチャは不変。
- `BaseEstimatorAdapter` に `set_categorical_features()` 追加（デフォルト no-op、後方互換）。
- `EstimatorProvider` に `runtime_deps()` / `params_summary()` 追加（protocol 拡張、内部のみ）。
- `SearchDim` の import パスが `tuning.search_space` → `core.types.search_dim` に変更（内部のみ、公開 API に含まれない）。

### Alternatives Considered

1. `categorical_feature` を `fit()` の `**kwargs` に任せ続ける
   - 不採用。新アダプタで `TypeError` が起きるリスクが高く、L2 の estimator 非依存性が破れる。
2. `SearchDim` を `estimators/` に移動する（L1 内で完結）
   - 不採用。`SearchDim` は tuning 以外（将来の config validation 等）でも使われる可能性がある。Foundation に置く方が汎用的。
3. `params_summary()` を `BaseEstimatorAdapter` のメソッドにする
   - 不採用。アダプタは「学習と予測」に徹すべき。テーブル表示はプレゼンテーション層の関心で、Provider が適切。

### Acceptance Criteria

- `_model_tables.py` に `LGBMConfig` import が存在しないこと
- `cv_trainer.py` / `refit_trainer.py` に `categorical_feature` が存在しないこと
- `model.py` に `"lightgbm"` 文字列リテラルが存在しないこと
- `estimators/provider.py` に `tuning/` からの import が存在しないこと
- `shap_explainer.py` に `NativeFeaturePipeline` import が存在しないこと
- `model.py` が 800 行以内であること
- 全テスト pass（932 件）
- mypy clean（86 ファイル）

---

## H-0055: StratifiedGroupKFold の Config 接続

- ID: `H-0055`
- Status: `implemented`
- Scope: `Config | Splitters`
- Related: `BLUEPRINT.md §5, §10`

### 目的

`StratifiedGroupKFoldSplitter`（既に `splitters/group_kfold.py` に実装済み）を Config → Model パイプラインに接続する。グループ制約と層化分割を同時に必要とするユースケース（例: 顧客IDでグループ分割しつつクラスバランスを維持）を Config 経由で利用可能にする。

### 影響範囲

- `config/schema.py`: `StratifiedGroupKFoldConfig` 追加、`SplitConfig` union 拡張
- `config/loader.py`: エイリアス追加（`stratified-group-kfold` 等）
- `core/_model_factories.py`: `_build_splitter_for_method` dispatch 追加、`_resolve_auto_inner_valid` にエントリ追加
- `BLUEPRINT.md §5, §10`: ドキュメント更新

### 互換性

- 既存 Config は影響なし（discriminated union への追加は後方互換）
- `StratifiedGroupKFoldSplitter` クラス自体は変更なし

### 代替案

なし。Splitter は既に実装・テスト済みであり、Config 接続のみが不足している。

### 受け入れ基準

- `method: "stratified_group_kfold"` で Config → Model → fit が完走すること
- エイリアス（`stratified-group-kfold` 等）が正規化されること
- InnerValid auto-resolution で `group_holdout` が選択されること（group 制約を維持）
- 全テスト pass、mypy clean

---

## H-0056: テスト基盤の体系的補強

- ID: `H-0056`
- Status: `accepted`（実装: 9bfeaf7、PR #33。H-0111 で Status を訂正、#319）
- Scope: `Testing`
- Related: `BLUEPRINT.md §18.1, §14.4, §15.2, §11`

### 目的

テスト評価（1007 テスト、97% カバレッジ）とピアライブラリ比較（scikit-learn / LightGBM / Optuna / FLAML / PyCaret）により特定された構造的ギャップを補填し、新アルゴリズム追加・format_version 変更・パラメータ組み合わせ爆発に対する回帰耐性を確保する。

### 背景

現テストスイートは契約テスト・リーク防止・再現性・エラーパスで高品質だが、以下の5カテゴリに構造的な不足がある。

### Proposal: 5 カテゴリのテスト補強

#### カテゴリ A: 実 Artifact 互換テスト（優先度: 高）

**現状の問題**: 同一バージョン round-trip と `analysis_context.pkl` 欠損の擬似 legacy のみ。過去版が吐いた実 artifact をロードする fixture がない。Legacy 校正経路（`model.py` 325 行目 `oof_raw_scores is None` → probability 入力で calibrate する else 分岐）は実質デッドコード扱いで未検証。

**追加テスト**:

1. **Frozen artifact fixture**: CI で生成した artifact を `tests/fixtures/v1_regression/` / `tests/fixtures/v1_binary_calibrated/` に格納。`Model.load()` → `predict()` → 既知の期待値と比較。LightGBM / XGBoost が保存形式互換で実施している手法に準拠。
2. **Legacy calibration path**: `oof_raw_scores=None` の FitResult を手動構築し、`predict()` 時に probability 経由で calibrate が走ることを確認。model.py 321–326 行の else 分岐のカバレッジを保証。
3. **format_version rejection 明示テスト**: `format_version=99` の metadata.json → `DESERIALIZATION_FAILED` で reject。`format_version=0`（過去）も同様。
4. **Booster model string roundtrip**: `model_to_string()` → `model_from_string()` の往復が LightGBM バージョン間で壊れないことの検証（LightGBM #7186 の回帰検知）。
5. **metadata.json 部分欠損**: 必須フィールド（`feature_names`, `task`, `run_id` 等）を1つずつ削除し、各欠損で正しいエラーメッセージが出ることを検証。

#### カテゴリ B: Provider/Adapter 共通 Invariant チェック（優先度: 高）

**現状の問題**: adapter / e2e テストは LGBM 前提の手書き happy-path が中心。共有データも 2 列の dense float DataFrame に偏る。scikit-learn は `check_estimator` / `parametrize_with_checks` で API 共通条件を一括検証し、LightGBM も `all_x_types` / `all_y_types` と sklearn check を回している。LizyML は provider/adapter ごとの共通チェック層を持たない。

**追加テスト（`check_provider` スイート）**:

1. **Protocol メソッド存在・戻り値型チェック**:
   - `check_extract_model_params_returns_dict`: `extract_model_params()` が `dict[str, Any]` を返す。
   - `check_extract_smart_params_returns_dict`: `extract_smart_params()` が `dict[str, Any]` を返す。
   - `check_runtime_deps_nonempty`: `runtime_deps()` が空でない `dict[str, str]` を返す。
   - `check_default_space_nonempty`: `default_space(task)` が空でない `list[SearchDim]` を返す。

2. **Factory → fit → predict 往復チェック**:
   - `check_estimator_fit_predict_roundtrip`: 全タスク型（regression / binary / multiclass）× provider で fit → predict が完走し、出力 shape が正しい。
   - `check_estimator_predict_proba_shape`: binary → `(n, 2)`、multiclass → `(n, k)`。regression → `UNSUPPORTED_TASK`。
   - `check_pipeline_factory_returns_pipeline`: `build_pipeline_factory()()` が `BaseFeaturePipeline` を返す。

3. **Pickle 往復チェック**:
   - `check_estimator_pickle_roundtrip`: fit 済み adapter を pickle → unpickle し、predict 結果が一致。

4. **Importance チェック**:
   - `check_importance_after_fit`: fit 後に `importance("split")` と `importance("gain")` が feature_names と同じキーの dict を返す。

5. **データ多様性 fixture**:
   - `dense_float_2col`（既存）、`dense_float_20col`（高次元）、`mixed_dtype`（float + int + category）、`with_missing`（NaN 列）、`single_feature`（1列）、`high_cardinality_cat`（100+ unique category）。
   - 各 fixture を `check_estimator_fit_predict_roundtrip` にパラメタライズ。

#### カテゴリ C: Tuning 再現性・失敗マトリクス（優先度: 中）

**現状の問題**: 再現性テストは fit/predict/evaluate まで。tuning 側は callback と基本成功/失敗のみ。同一 seed で `best_params` / `best_score` / trial 順が固定されるか未検証。全 trial 失敗時の分岐（`tuner.py` 167 行目）は未到達。Optuna は seed 固定と逐次実行を再現性の前提として明示している。

**追加テスト**:

1. **tune() 再現性**: 同一 seed・同一データ・同一 space で 2 回 `tune()` を実行し、`best_params`, `best_score`, `len(trial_history)`, trial 順序が完全一致することを検証。
2. **全 trial 失敗**: objective が常に例外を送出する mock を注入し、`TUNING_FAILED` + `context["n_trials"]` を検証。`tuner.py` 167–175 行の `if not completed` 分岐をカバー。
3. **部分 trial 失敗**: 一部 trial のみ失敗させ、成功 trial の中から best が正しく選択されることを検証。
4. **NaN/inf 返却時**: objective が `float("nan")` や `float("inf")` を返した場合の挙動を検証（Optuna 側の pruned 処理との整合）。
5. **Search space と Config params の衝突**: Config に `learning_rate=0.1` を設定しつつ、search space にも `learning_rate` を含め、tune 結果が Config 値を上書きすることを検証。
6. **空の search space**: `space={}` でデフォルト space が使用されることの明示テスト。

#### カテゴリ D: 入力ソース・dtype・境界値の E2E（優先度: 中）

**現状の問題**: DataSource 単体では CSV/Parquet を読めるが、Model entry まで通すテストは CSV 中心。共通 helper も単純な float DataFrame。Parquet 経由の fit/predict/export/load、nullable dtype、重複列、空/1行入力、カテゴリ順序ずれなどは見当たらない。LightGBM/XGBoost は `all_x_types` / `all_y_types` でコンテナ型・dtype 差分を広く回している。

**追加テスト**:

1. **Parquet フルパイプライン**: Parquet ファイル → `data.path` → fit → export → load → predict の完走。CSV 経由のみだった E2E を拡張。
2. **float32 入力**: `float32` DataFrame を `fit()` に渡し、OOF / predict 結果が `float64` で返ることを確認。scikit-learn の `global_dtype` fixture に相当。
3. **nullable dtype**: `pd.array([1, 2, None], dtype="Int64")` を含む DataFrame → fit が正常に動作するか、明確なエラーを返すかを検証。
4. **空 DataFrame (0行)**: `fit()` → 明確なエラーメッセージ（`DATA_SCHEMA_INVALID` 等）。
5. **1行 DataFrame**: CV 不可能な最小ケース → 明確なエラーメッセージ。
6. **重複列名**: `pd.DataFrame({"a": ..., "a": ...})` → 明確なエラーメッセージ。
7. **極端な値**: `inf` / `-inf` / 非常に大きい値を含む DataFrame での fit 挙動。
8. **カテゴリ順序ずれ**: 学習時 `["a", "b", "c"]` → 推論時 `["c", "a", "b"]` の順序違い。列ズレテスト（test_column_drift.py）の拡張。

#### カテゴリ E: パラメータ組み合わせの Pairwise テスト（優先度: 中）

**現状の問題**: 各パラメータを1つずつ検証しているが、相互作用のテストがない。全組み合わせの直積は爆発するが、Pairwise（2因子間カバレッジ）なら ~20-30 ケースで主要な相互作用を検出できる。

**因子と値**:

| 因子 | 値 |
|------|-----|
| task | `regression`, `binary`, `multiclass` |
| split_method | `kfold`, `stratified_kfold`, `group_kfold`, `time_series` |
| calibration | `None`, `"platt"` |
| early_stopping | `True`, `False` |
| n_estimators | `5`, `100` |

**追加テスト**:

1. **Pairwise fit 完走テスト**: 上記因子の pairwise 組み合わせ（約 20-30 ケース）を `@pytest.mark.parametrize` で生成し、「有効な組み合わせは例外なく fit 完走する」「無効な組み合わせ（例: calibration + regression）は明確なエラーを返す」を検証。
2. **個別の重要な相互作用テスト**:
   - `calibration` + `group_kfold`: calibration splitter が group 制約を尊重するか。
   - `balanced=True` + `multiclass`: sample_weight が正しく計算されるか。
   - `feature_weights` + `auto_num_leaves`: smart params 同士の相互作用。
   - `tuning` + `calibration`: tune → fit(calibration) で best_params と calibration が両立するか。
   - `n_estimators=1` + `early_stopping`: 最小ラウンドでのエッジケース。
   - `features.exclude` + `features.categorical`: 除外列がカテゴリ列の場合。

### 影響範囲

- `tests/` 以下への追加のみ。`lizyml/` の実装コードは変更しない。
- `tests/fixtures/` にfrozen artifact を追加（CI 生成スクリプト含む）。
- `tests/_helpers.py` にデータ多様性 fixture を追加。

### 互換性

- テスト追加のみのため破壊的変更なし。
- frozen artifact fixture は `format_version=1` のスナップショットであり、将来の version bump 時に migration テストの基盤となる。

### 代替案

- 全組み合わせ直積テスト → 実行時間爆発（数千ケース）。pairwise で十分な因子間カバレッジを達成。
- Property-based テスト（Hypothesis）→ scikit-learn / LightGBM / Optuna / FLAML / PyCaret の5ライブラリすべて未採用。将来の検討項目とする。
- 可視化回帰テスト（画像 diff）→ Optuna のみ別リポで実施。現時点では low priority。

### 受け入れ基準

- カテゴリ A: frozen artifact fixture からの `Model.load()` → `predict()` が期待値と一致。legacy calibration path（`oof_raw_scores=None`）のカバレッジ到達。
- カテゴリ B: `check_provider` スイートが LGBMProvider に対して全チェック pass。新 provider 追加時に自動で全チェックが走る構造。
- カテゴリ C: 同一 seed の `tune()` が `best_params` / `best_score` 完全一致。全 trial 失敗時に `TUNING_FAILED` を返す。
- カテゴリ D: Parquet / float32 / nullable dtype の E2E が pass。0行/1行/重複列で明確なエラー。
- カテゴリ E: pairwise 組み合わせ全ケースで fit 完走 or 明確なエラー。
- 全体: 既存 1007 テストに影響なし。カバレッジ 97%+ 維持。

---

## H-0057: Split-derived OOF Coverage（TimeSeriesCV の OOF メトリクス NaN 解消）

- ID: `H-0057`
- Status: `implemented`
- Scope: `Evaluation | Metrics`
- Related: `BLUEPRINT.md §13.2, §7.1`

### 目的

TimeSeriesCV（expanding window）使用時に、最初の期間のサンプルがどの validation fold にも含まれないため OOF 予測値が NaN のまま残り、`evaluate()` / `evaluate_table()` の全体 OOF メトリクスが NaN になる問題を解消する。

### 背景

- `TimeSeriesSplit(n_splits=K)` では先頭 `n_samples // (K+1)` 行程度が全 fold で train 側にのみ含まれ、validation に一度も現れない。
- 現行の `Evaluator.evaluate()` は `oof_pred` 全行で metric を計算するため、NaN が混入しメトリクスも NaN になる。
- NaN マスク（`np.isnan` で除外）は「バグで予測されなかった行」と「仕様上カバーされない行」を区別できず、潜在バグを見落とすリスクがある。

### Proposal: Split-derived OOF Coverage Mask

1. **`compute_oof_valid_mask(splits_outer, n_samples)`** を `oof_assembly.py` に追加。
   - `SplitIndices.outer` の全 fold の `valid_idx` の和集合から boolean mask を生成。
   - NaN 検出ではなく、split 構造から決定論的に導出。

2. **`Evaluator.evaluate()` の OOF メトリクス計算を変更**。
   - mask の True 行のみで `oof` メトリクスを計算。
   - **カバー行に NaN がある場合は `ValueError`**（予測パイプラインのバグとして検知）。（H-0111 注記: 実装は `LizyMLError(EVALUATION_FAILED)` を送出する。BLUEPRINT §13 は実装を書く。#319）
   - `oof_per_fold` / IF メトリクスは変更なし。

3. **`metrics["raw"]["oof_coverage"]`** を追加（float, 0.0–1.0）。
   - KFold: 常に `1.0`。TimeSeriesCV: `< 1.0`。

4. **`evaluate_table()`** で `df.attrs["oof_coverage"]` として公開。

### 影響範囲

| 対象 | 変更内容 |
|------|---------|
| `oof_assembly.py` | `compute_oof_valid_mask()` 追加 |
| `evaluator.py` | mask ベースの OOF 計算 + `oof_coverage` 追加 |
| `table_formatter.py` | `df.attrs` に `oof_coverage` |
| `_model_metrics.py` | calibrated パスは `splits` 保持済み → 変更不要 |
| FitResult | **変更なし**（mask は SplitIndices から導出） |

### 互換性

- **KFold（既存の主要ユースケース）**: 全行カバーのため挙動は完全に同一。
- **TimeSeriesCV**: `metrics["raw"]["oof"]` が NaN → 有効な数値に変わる（改善のみ）。
- **`metrics["raw"]` への `oof_coverage` キー追加**: 既存コードは未知キーを参照しない限り影響なし。`filter_metrics()` は非 dict 値をパススルーするため互換。

### 代替案（却下）

- **NaN マスク**: `np.isnan(oof_pred)` で除外。→ バグ由来の NaN も黙殺されるため却下。

### 受け入れ基準

- `compute_oof_valid_mask` が split indices から正しい bool mask を返す（unit test）。
- カバー行に NaN → `ValueError`（バグ検知テスト）。（H-0111 注記: 実装とテストは `EVALUATION_FAILED`。#319）
- 非カバー行の NaN は正常スキップ。
- KFold で `oof_coverage == 1.0`、TimeSeriesCV で `oof_coverage < 1.0`。
- TimeSeriesCV の OOF メトリクスが finite（NaN でない）。
- `evaluate_table().attrs["oof_coverage"]` が float。
- 既存テスト全通し（後方互換）。

### Decision

- Date: 2026-03-17
- Result: Accepted — split 構造からの決定論的マスク + バグ検知 assertion の方針で実装する。

### 備考

- `metrics["calibrated"]` には `oof_coverage` を含めない（現状維持）。calibrated の cross-fit 分割は `calibration.n_splits` で outer とは独立しており、coverage が異なりうるため、raw の値を流用すると不正確になる。
- この不整合は H-0058（Outer Split を Calibration で再利用する提案）で構造的に解消される予定。H-0058 が実装されれば calibrated の coverage は raw と一致するため、別途 coverage を公開する必要がなくなる。

---

## H-0058: Outer Split を Calibration Cross-fit で再利用

- ID: `H-0058`
- Status: `implemented`
- Scope: `Calibration | Split | Config`
- Related: `BLUEPRINT.md §10.5, §12, §13.2`, `H-0057`

### 目的

calibration cross-fit が outer CV とは独立した分割を使うことで生じる coverage 不整合・コード複雑性・概念的な非対称を構造的に解消する。

### 背景

現状（H-0057 後）では calibration cross-fit は `calibration.n_splits` で独立した分割を生成する。outer CV と同じ `split.method` を継承するが fold 数だけが独立しており、TimeSeriesCV 使用時に raw OOF と calibrated OOF のカバレッジが乖離する。

```
outer CV:       TimeSeriesSplit(n_splits=5) → coverage ≈ 83%
calibration CV: TimeSeriesSplit(n_splits=3) → coverage ≈ 75%
```

H-0057 では `_model_metrics.py` で splits を差し替える workaround を追加して対処したが、本質的には分割構造が二重になっていることが根本原因。

### リーク安全性

calibration cross-fit は `(oof_scores, y)` のみを入力とし、X は使わない（§12.1）。

3-fold の具体例（データ = A, B, C）:

| step | fold | 学習データ | 予測対象 |
|------|------|-----------|---------|
| Outer CV | 0 | X[B+C], y[B+C] → model_0 | oof[A] |
| Outer CV | 1 | X[A+C], y[A+C] → model_1 | oof[B] |
| Outer CV | 2 | X[A+B], y[A+B] → model_2 | oof[C] |
| Cal cross-fit | 0 | oof[B+C], y[B+C] → cal_0 | cal_oof[A] |
| Cal cross-fit | 1 | oof[A+C], y[A+C] → cal_1 | cal_oof[B] |
| Cal cross-fit | 2 | oof[A+B], y[A+B] → cal_2 | cal_oof[C] |

行 A に注目: `oof[A]` は A を見ていない model_0 が生成、`cal_oof[A]` は oof[A] を見ていない cal_0 が生成。同一行リーク経路なし。

C_final は `fit(oof[全行], y[全行])` で学習し推論専用。評価には使わない。

### Proposal

calibration cross-fit で `fit_result.splits.outer` をそのまま再利用する。

```python
# 現在（model.py）
cal_splitter = build_calibration_splitter(cfg)
cal_split_indices = list(cal_splitter.split(...))

# 変更後
cal_split_indices = fit_result.splits.outer
```

### 影響範囲

| 対象 | 変更内容 |
|------|---------|
| `CalibrationConfig.n_splits` | deprecated（残すが無視、UserWarning 出力） |
| `model.py _run_calibration()` | `build_calibration_splitter` → `fit_result.splits.outer` に置換 |
| `_model_factories.py` | `build_calibration_splitter` を deprecated 化 |
| `_model_metrics.py` | splits 差替えロジック削除（H-0057 workaround が不要に） |
| `SplitIndices.calibration` | outer と同一値（冗長だが互換性のため残す） |
| `cross_fit_calibrate()` | 変更なし（split_indices を受け取るだけ） |
| `BLUEPRINT.md §10.5` | calibration CV 規約を改訂 |
| `BLUEPRINT.md §13.2` | calibrated coverage が raw と一致する旨を追記 |

### 互換性

#### Config 互換性
- `calibration.n_splits` を指定した場合は `UserWarning` を出力し無視。
- `extra="forbid"` なのでフィールド自体は残す（削除すると既存 Config が壊れる）。
- 将来の `config_version` 更新時に削除を検討。

#### 保存互換性
- `SplitIndices.calibration` に outer と同一のリストを保存 → 既存の `Model.load()` は問題なし。
- `format_version` の変更は不要（データ構造は同一、値が変わるだけ）。

### 代替案（却下）

1. **calibrated に oof_coverage を追加（H-0057 案 B）**: H-0058 が来ると冗長フィールドになる。
2. **calibration.n_splits のデフォルトを outer に合わせる**: 結局独立分割が残り、method パラメータ（gap/embargo）の二重管理が消えない。
3. **現状維持**: `_model_metrics.py` の splits 差替え workaround を永続させることになる。

### 受け入れ基準

- `calibration.n_splits` 指定時に `UserWarning` が出力される。
- calibration cross-fit が `fit_result.splits.outer` を使用する。
- `SplitIndices.calibration` が outer と同一値。
- `_model_metrics.py` の splits 差替えロジックが削除される。
- TimeSeriesCV で calibrated OOF の実質 coverage が `metrics["raw"]["oof_coverage"]` と一致する。
- 既存の `Model.load()` で旧 artifact が問題なくロードできる。
- リーク検知テスト（`test_calibration_leakage`）が引き続き pass。

### Decision

- Date: 2026-03-17
- Result: Accepted — outer CV splits を calibration cross-fit でそのまま再利用する方針で実装する。

---

## H-0059: Codegen Export — LizyML 非依存の学習・推論コード生成

- ID: `H-0059`
- Status: `accepted`
- Decision: `v0.3.0 でリリース (2026-03-20)`
- Scope: `Export | Public API`
- Related: `BLUEPRINT.md §6.6, §15.4`, `skills/export/SKILL.md`

### 目的

LizyML で構築したモデルを本番環境に載せる際、ライブラリ全体を依存に含めるとデバッグが困難になる。`Model.export_code()` で **LizyML 非依存の学習・推論コード** を自動生成し、以下を実現する:

1. **再学習パイプライン**: 新データ到着時に同一設定で refit + calibrator 再構築ができる
2. **学習コードの透明性**: fit 時に何が起きているかを人間が読めるコードで確認・検証できる
3. **最小依存での本番推論**: LizyML なしで推論を実行できる

### 背景

現在の `Model.export()` は LizyML Artifact（joblib pickle）を出力し、`Model.load().predict()` で推論する。本番環境において:

1. **デバッグ困難**: エラー発生時に LizyML 内部を追う必要がある
2. **依存の重さ**: LizyML + pydantic + 全 optional deps が本番に必要
3. **再学習の不透明性**: `Model.fit()` の内部で何が起きているか追えない

### Proposal

#### 出力構造（3 ファイル + artifacts ディレクトリ）

```
{path}/
├── config.json             # 全設定の単一ソース（パラメータ変更はここだけ）
├── train.py                # 学習: pipeline fit → LightGBM refit → calibration
├── predict.py              # 推論: pipeline transform → predict → calibrate
├── requirements.txt        # 最小依存
├── test_equivalence.py     # LizyML との一致検証
└── artifacts/              # train.py が生成・更新
    ├── model.txt           # LightGBM Booster テキスト形式
    ├── pipeline_state.json # 学習済み feature pipeline 状態
    ├── calibrator.json     # Calibrator パラメータ（binary のみ）
    └── calibrator_model.txt # Isotonic Booster（該当時のみ）
```

#### config.json

ユーザーが確認・編集する唯一のファイル。`_` prefix はメタ情報（読み取り専用）。

```json
{
  "_generated_by": "lizyml 0.2.0",
  "_run_id": "7e77ba4b-...",
  "_task": "binary",
  "_target_col": "y",
  "_timestamp": "2026-03-19T12:00:00",

  "feature_names": ["age", "income", "category_a", "category_b"],
  "categorical_features": ["category_a", "category_b"],

  "lgbm_params": {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "verbosity": -1
  },
  "num_boost_round": 1000,
  "early_stopping_rounds": 50,
  "validation_ratio": 0.2,
  "seed": 42,

  "calibration_method": "platt",
  "calibration_n_splits": 5
}
```

#### train.py

```python
#!/usr/bin/env python3
"""Train a LightGBM model and fit a probability calibrator.

Usage:
    python train.py train_data.csv
    python train.py train_data.parquet --no-calibration

Steps:
    1. Fit feature pipeline (learn category mappings → pipeline_state.json)
    2. Train LightGBM on full data (→ model.txt)
    3. Generate OOF scores via CV for calibration
    4. Fit calibrator on OOF scores (→ calibrator.json)

Generated by lizyml — https://github.com/nbx-liz/LizyML
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

with open(ROOT / "config.json") as _f:
    CFG = json.load(_f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feature Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fit_pipeline(df: pd.DataFrame) -> dict:
    """Learn category mappings and save pipeline state."""
    expected = CFG["feature_names"]
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    mappings: dict[str, dict[str, int]] = {}
    for col in CFG["categorical_features"]:
        cats = sorted(str(v) for v in df[col].dropna().unique())
        mappings[col] = {v: i for i, v in enumerate(cats)}
        log.info("    %s: %d categories", col, len(cats))

    state = {
        "feature_names": expected,
        "categorical_features": CFG["categorical_features"],
        "category_mappings": mappings,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "pipeline_state.json", "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return state


def transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply fitted pipeline to a DataFrame."""
    X = df[state["feature_names"]].copy()
    for col, mapping in state.get("category_mappings", {}).items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapping)  # unseen → NaN
    return X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LightGBM Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_lgbm(X: pd.DataFrame, y: pd.Series, cat_cols: list[str]) -> lgb.Booster:
    """Train LightGBM with optional early stopping via holdout."""
    ds_full = lgb.Dataset(X, label=y, categorical_feature=cat_cols or "auto")
    callbacks: list = [lgb.log_evaluation(period=200)]
    train_set = ds_full
    valid_sets, valid_names = [ds_full], ["train"]

    ratio = CFG.get("validation_ratio", 0)
    es_rounds = CFG.get("early_stopping_rounds")
    if ratio > 0 and es_rounds:
        n = len(y)
        rng = np.random.default_rng(CFG["seed"])
        idx = rng.permutation(n)
        n_val = max(1, int(n * ratio))

        train_set = lgb.Dataset(
            X.iloc[idx[n_val:]], label=y.iloc[idx[n_val:]],
            categorical_feature=cat_cols or "auto",
        )
        valid_ds = lgb.Dataset(
            X.iloc[idx[:n_val]], label=y.iloc[idx[:n_val]],
            reference=train_set,
        )
        valid_sets = [train_set, valid_ds]
        valid_names = ["train", "valid"]
        callbacks.insert(0, lgb.early_stopping(es_rounds, verbose=True))
        log.info("    holdout: %d train / %d valid", n - n_val, n_val)

    booster = lgb.train(
        CFG["lgbm_params"], train_set,
        num_boost_round=CFG["num_boost_round"],
        valid_sets=valid_sets, valid_names=valid_names,
        callbacks=callbacks,
    )
    booster.save_model(str(ARTIFACTS / "model.txt"))
    log.info("    saved model.txt (best_iteration=%d)", booster.best_iteration)
    return booster


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _generate_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lightweight CV to produce OOF raw scores (logits)."""
    n_splits = CFG.get("calibration_n_splits", 5)
    task = CFG["_task"]
    seed = CFG["seed"]

    kf = (StratifiedKFold if task == "binary" else KFold)(
        n_splits=n_splits, shuffle=True, random_state=seed,
    )
    oof = np.full(len(y), np.nan)
    for i, (trn, val) in enumerate(kf.split(X, y)):
        log.info("    CV fold %d/%d", i + 1, n_splits)
        ds = lgb.Dataset(X[trn], label=y[trn])
        bst = lgb.train(CFG["lgbm_params"], ds,
                        num_boost_round=CFG["num_boost_round"],
                        callbacks=[lgb.log_evaluation(0)])
        oof[val] = bst.predict(X[val], raw_score=True)
    return oof


def _fit_platt(scores: np.ndarray, y: np.ndarray) -> dict:
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    lr.fit(scores.reshape(-1, 1), y)
    return {"method": "platt",
            "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def _fit_beta(scores: np.ndarray, y: np.ndarray) -> dict:
    from scipy.optimize import minimize
    s = np.clip(_sigmoid(scores), 1e-10, 1 - 1e-10)
    yf = y.astype(float)
    ls, l1s = np.log(s), np.log(1 - s)

    def nll(p):
        prob = np.clip(_sigmoid(p[0]*ls + p[1]*l1s + p[2]), 1e-10, 1-1e-10)
        return float(-np.sum(yf*np.log(prob) + (1-yf)*np.log(1-prob)))

    r = minimize(nll, x0=[1, 1, 0], method="L-BFGS-B")
    return {"method": "beta",
            "a": float(r.x[0]), "b": float(r.x[1]), "c": float(r.x[2])}


def _fit_isotonic(scores: np.ndarray, y: np.ndarray) -> dict:
    n = len(scores)
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "monotone_constraints": [1], "monotone_constraints_method": "advanced",
        "num_leaves": 7, "max_depth": 3, "learning_rate": 0.03,
        "lambda_l2": 5.0, "min_data_in_leaf": max(1, math.ceil(n * 0.01)),
        "verbose": -1, "seed": CFG["seed"],
    }
    rng = np.random.default_rng(CFG["seed"])
    n_val = max(1, int(n * 0.1))
    idx = rng.permutation(n)
    X_cal = scores.reshape(-1, 1)

    ds_t = lgb.Dataset(X_cal[idx[n_val:]], label=y[idx[n_val:]].astype(float))
    ds_v = lgb.Dataset(X_cal[idx[:n_val]], label=y[idx[:n_val]].astype(float),
                       reference=ds_t)
    bst = lgb.train(params, ds_t, num_boost_round=1000,
                    valid_sets=[ds_v], valid_names=["valid"],
                    callbacks=[lgb.early_stopping(100, verbose=False),
                               lgb.log_evaluation(0)])
    bst.save_model(str(ARTIFACTS / "calibrator_model.txt"))
    return {"method": "isotonic", "model_file": "calibrator_model.txt"}


_CAL_FITTERS = {"platt": _fit_platt, "beta": _fit_beta, "isotonic": _fit_isotonic}


def fit_calibrator(X: np.ndarray, y: np.ndarray) -> dict | None:
    """Generate OOF scores and fit calibrator. Returns params or None."""
    method = CFG.get("calibration_method")
    if not method or CFG["_task"] != "binary":
        return None

    log.info("[3/4] Generating OOF scores ...")
    oof = _generate_oof(X, y)

    log.info("[4/4] Fitting %s calibrator ...", method)
    params = _CAL_FITTERS[method](oof, y)
    with open(ARTIFACTS / "calibrator.json", "w") as f:
        json.dump(params, f, indent=2)
    log.info("    saved calibrator.json")
    return params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(df: pd.DataFrame, *, calibrate: bool = True) -> None:
    target = CFG["_target_col"]
    y = df[target]
    X_raw = df.drop(columns=[target])

    log.info("[1/4] Fitting feature pipeline ...")
    state = fit_pipeline(X_raw)
    X = transform(X_raw, state)

    log.info("[2/4] Training LightGBM ...")
    cat_cols = [c for c in CFG["categorical_features"] if c in X.columns]
    train_lgbm(X, y, cat_cols)

    if calibrate:
        fit_calibrator(X.values, y.values)
    else:
        log.info("[3/4] Calibration skipped")
        log.info("[4/4] —")

    log.info("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description="Train LightGBM (LizyML codegen)")
    p.add_argument("data", help="CSV or Parquet file")
    p.add_argument("--no-calibration", action="store_true")
    args = p.parse_args()

    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    target = CFG["_target_col"]
    if target not in df.columns:
        log.error('Target "%s" not found. Columns: %s', target, list(df.columns))
        sys.exit(1)

    train(df, calibrate=not args.no_calibration)


if __name__ == "__main__":
    main()
```

#### predict.py

```python
#!/usr/bin/env python3
"""Run inference with a trained model.

Usage:
    python predict.py test_data.csv
    python predict.py test_data.csv -o predictions.csv

Generated by lizyml — https://github.com/nbx-liz/LizyML
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

with open(ROOT / "config.json") as _f:
    CFG = json.load(_f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feature Transform (predict-time: no re-fitting)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_pipeline() -> dict:
    path = ARTIFACTS / "pipeline_state.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run train.py first.")
    with open(path) as f:
        return json.load(f)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Select expected columns and apply categorical encoding."""
    state = _load_pipeline()
    expected = state["feature_names"]

    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} column(s): {missing}. "
            f"Expected: {expected}"
        )

    extra = sorted(set(df.columns) - set(expected))
    if extra:
        log.warning("Ignoring %d extra column(s): %s", len(extra), extra)

    X = df[expected].copy()
    for col, mapping in state.get("category_mappings", {}).items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapping)  # unseen → NaN
    return X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Calibration (apply only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _load_calibrator() -> dict | None:
    path = ARTIFACTS / "calibrator.json"
    return json.load(open(path)) if path.exists() else None


def calibrate(raw_scores: np.ndarray, cal: dict) -> np.ndarray:
    """Map raw logits → calibrated probabilities."""
    m = cal["method"]
    if m == "platt":
        return 1 / (1 + np.exp(-(cal["a"] * raw_scores + cal["b"])))
    if m == "beta":
        s = np.clip(_sigmoid(raw_scores), 1e-10, 1 - 1e-10)
        logit = cal["a"] * np.log(s) + cal["b"] * np.log(1 - s) + cal["c"]
        return np.clip(_sigmoid(logit), 0, 1)
    if m == "isotonic":
        bst = lgb.Booster(model_file=str(ARTIFACTS / cal["model_file"]))
        return np.clip(bst.predict(raw_scores.reshape(-1, 1)), 0, 1)
    raise ValueError(f'Unknown calibration: "{m}"')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Predict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict(df: pd.DataFrame) -> dict[str, np.ndarray | None]:
    """Run inference. Returns {"pred": ..., "proba": ...}."""
    X = transform(df)
    booster = lgb.Booster(model_file=str(ARTIFACTS / "model.txt"))
    task = CFG["_task"]

    if task == "regression":
        return {"pred": np.asarray(booster.predict(X), dtype=np.float64),
                "proba": None}

    if task == "binary":
        proba = np.asarray(booster.predict(X), dtype=np.float64)
        cal = _load_calibrator()
        if cal:
            logits = np.asarray(booster.predict(X, raw_score=True), dtype=np.float64)
            proba = calibrate(logits, cal)
        return {"pred": (proba > 0.5).astype(np.int64), "proba": proba}

    if task == "multiclass":
        proba = np.asarray(booster.predict(X), dtype=np.float64)
        return {"pred": np.argmax(proba, axis=1).astype(np.int64), "proba": proba}

    raise ValueError(f'Unknown task: "{task}"')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    p = argparse.ArgumentParser(description="Predict (LizyML codegen)")
    p.add_argument("data", help="CSV or Parquet file")
    p.add_argument("-o", "--output", default="predictions.csv")
    args = p.parse_args()

    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    result = predict(df)

    out = pd.DataFrame({"pred": result["pred"]})
    if result["proba"] is not None:
        if result["proba"].ndim == 1:
            out["proba"] = result["proba"]
        else:
            for i in range(result["proba"].shape[1]):
                out[f"proba_{i}"] = result["proba"][:, i]
    out.to_csv(args.output, index=False)
    log.info("Saved %d rows → %s", len(out), args.output)


if __name__ == "__main__":
    main()
```

### 設計判断

| 判断 | 理由 |
|------|------|
| **2 ファイル構成** | train.py (~200行) と predict.py (~120行) で全体 ~320行。1ファイルでは長すぎ、4ファイルでは分散しすぎ |
| **config.json に設定集約** | コードを触らずにパラメータ変更可能。`_` prefix でメタ情報を分離 |
| **train.py に pipeline/calibration を内包** | 学習時しか使わないロジック（fit_pipeline, OOF生成, calibrator fit）を1箇所に集約 |
| **predict.py に transform/calibrate を内包** | 推論に必要なロジックだけ。本番デプロイ時は predict.py + config.json + artifacts/ のみ |
| **Booster テキスト形式** | pickle 不要、人間可読、バージョン間互換性高 |
| **Calibrator: method 別に適切な保存** | Platt/Beta → JSON 数値パラメータ、Isotonic → Booster テキスト |
| **OOF 生成に StratifiedKFold** | binary でクラス比保持（LizyML デフォルトと一致） |
| **refit + calibrator 再構築** | predict-only だと calibrator が陳腐化するリスクを回避 |
| **SHAP 非対応（初期版）** | 依存が重い。将来拡張 |

### 影響範囲

| 対象 | 変更内容 |
|------|---------|
| `Model` (公開 API) | `export_code(path)` メソッド追加 |
| `lizyml/codegen/` (新規) | generator.py: FitResult + config → ファイル生成 |
| `BaseCalibratorAdapter` | `export_params() -> dict` 追加 |
| `PlattCalibrator` | `export_params()`: coef_, intercept_ → a, b |
| `BetaCalibrator` | `export_params()`: _params → a, b, c |
| `IsotonicCalibrator` | `export_params()` + `save_model_text(path)` |
| `LGBMAdapter` | `save_model_text(path)` 追加 |
| `NativeFeaturePipeline` | `export_state_json(path)` 追加 |
| `BLUEPRINT.md §6.5, §15.3` | codegen export 仕様追記 |

### 互換性

- 既存 API (`export()` / `load()`) は変更なし。`export_code()` は新規追加。
- `format_version` 変更不要。codegen は既存 Artifact 形式とは独立。
- 将来の非 LGBM 対応時は Provider に `export_model_text()` を追加。

### 代替案（却下）

1. **Predict のみ codegen**: 再学習時に calibrator が陳腐化するリスク。LizyML が本番依存から外れない。
2. **4 ファイル分割**: pipeline.py / calibration.py を分離すると可読性向上だがファイル数が増えて見通しが悪化。
3. **1 ファイル構成**: 400行超で可読性低下。学習・推論の責務が混在。

### 制約・前提

- 初期実装は **LightGBM のみ**。
- CV は calibrator の OOF 生成のみ（評価メトリクスは含まない）。
- multiclass の calibration は対象外（binary のみ）。
- `train.py` 実行時に `scikit-learn` が必要（KFold / StratifiedKFold / LogisticRegression）。
- `predict.py` 実行時は `lightgbm` + `numpy` + `pandas` のみ（calibration 有無に関わらず）。

### 受け入れ基準

**学習 (train.py):**
- `python train.py data.csv` で全 artifacts が生成される。
- 生成コードに `import lizyml` が存在しない。
- 同一データ・同一 seed で refit モデルの予測値が `rtol=1e-7` で一致。
- `--no-calibration` で calibrator 学習がスキップされる。

**推論 (predict.py):**
- `python predict.py test.csv` で予測結果が出力される。
- `Model.predict()` と codegen 出力が `rtol=1e-7` で一致。
- regression / binary / multiclass の 3 タスクで pass。
- binary + Platt / Isotonic / Beta で calibrated probability が一致。
- 列ズレ検知が動作（missing → ValueError、extra → warning）。

**Calibrator 再学習:**
- 新データで `python train.py new_data.csv` → calibrator が再構築される。

**共通:**
- 依存が requirements.txt に収まる。
- E2E テスト: fit → export_code → train → predict → 結果検証。

---

## H-0060: blocked_group_kfold — 2軸交差検証（期間 × グループ）

- ID: `H-0060`
- Status: `accepted`
- Scope: `Split | InnerValid | Config | Public API`
- Related: `BLUEPRINT §5.4, §10`

### 目的

期間軸（blocks）とグループ軸（groups）の直積で交差検証を行う新しい split method `blocked_group_kfold` を追加する。時間的な前方検証とグループ間リーク防止を同時に実現する。

### 背景

既存の splitter は1軸の分割のみ対応する（時間 or グループ）。実務では以下のような2軸分割が必要になる。

- 金融: ユーザーID × 月次データ（2月以前で学習、3月以降で評価、学習/評価でユーザーが異なる）
- 医療: 患者ID × 受診期間
- 小売: 店舗ID × 週次/月次データ

既存の `group_time_series` はグループの出現順で時系列分割するが、以下を満たさない。

1. 任意の境界値（cutoff）で期間を区切れない
2. グループ軸で KFold できない（データ利用効率が低い）
3. expanding / sliding の窓制御ができない

### Proposal

#### Config 構造

```yaml
split:
  method: blocked_group_kfold
  blocks:                             # ── 期間軸（何で区切るか）
    col: date                         #   区切りに使うカラム
    cutoffs: ["2025-02", "2025-03"]   #   境界値リスト（valid 期間の開始点）
    mode: sliding                     #   expanding | sliding
    train_window: 2                   #   sliding 時: train に使う期間数
  groups:                             # ── グループ軸（何で分けるか）
    col: user_id                      #   グループ分割するカラム
    n_splits: 3                       #   グループの分割数
    stratify: auto                    #   auto | true | false
    shuffle: true                     #   グループ分割時のシャッフル
  min_train_rows: 10                  #   fold スキップ閾値（train）
  min_valid_rows: 5                   #   fold スキップ閾値（valid）
```

#### blocks セクション

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `col` | `str` | **必須** | 期間を定義するカラム名。ソート可能な型 |
| `cutoffs` | `list` | **必須** | 境界値リスト。各値が valid 期間の開始点 |
| `mode` | `"expanding" \| "sliding"` | `"expanding"` | train 期間の構成方式 |
| `train_window` | `int \| null` | `null` | `sliding` 時のみ有効。train に使う期間数 |

#### groups セクション

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `col` | `str` | **必須** | グループ分割するカラム名 |
| `n_splits` | `int` | **必須** | グループの分割数（K） |
| `stratify` | `"auto" \| true \| false` | `"auto"` | ターゲット分布による層化 |
| `shuffle` | `bool` | `true` | グループ分割時のシャッフル |

`stratify: auto` は binary/multiclass → 層化あり、regression → 層化なし。層化時はグループごとの代表ラベル（多数決クラス）で層化分割する。

#### 期間の定義

`cutoffs: [C₁, C₂, ..., Cₙ]` から `n+1` 個の期間を生成:

- P₀: `col < C₁`
- P₁: `C₁ ≤ col < C₂`
- Pₙ: `col ≥ Cₙ`

**expanding**: fold k の train = P₀ + ... + Pₖ、valid = Pₖ₊₁
**sliding** (`train_window=W`): fold k の train = 直前 W 期間、valid = Pₖ₊₁

時間 fold 数 = `len(cutoffs)`

#### Fold 生成アルゴリズム

```
for each 時間fold t:
    train_period_rows = 期間割り当てで train に属する全行
    valid_period_rows = 期間割り当てで valid に属する全行
    all_users = unique(groups[train_period_rows ∪ valid_period_rows])
    user_folds = StratifiedGroupKFold(all_users, n_splits=K)

    for each ユーザーfold u:
        train_users, valid_users = user_folds[u]
        train_idx = train_period_rows ∩ rows_of(train_users)
        valid_idx = valid_period_rows ∩ rows_of(valid_users)
        # 除外: train期間×valid_users, valid期間×train_users
        if len(train_idx) >= min_train_rows and len(valid_idx) >= min_valid_rows:
            yield (train_idx, valid_idx)
        else:
            warn and skip
```

合計 fold 数 = `len(cutoffs) × groups.n_splits − skip数`

#### Inner Valid（early stopping）

新規 strategy `BlockedGroupInnerValid` を追加する。

**自動解決ルール:**

| タスク | 戦略 | 動作 |
|---|---|---|
| binary / multiclass | `BlockedGroupInnerValid` | グループ分離 + 各クラス末尾グループ + 層化 |
| regression | `BlockedGroupInnerValid` | グループ分離 + 末尾グループ |
| フォールバック | `StratifiedTimeHoldoutInnerValid` | グループ数 < 4 で自動切替 |

**BlockedGroupInnerValid アルゴリズム:**

1. outer fold train 内のユニークグループを取得
2. 各グループの代表ラベルを算出（多数決クラス）※分類時のみ
3. 各グループの最終出現時刻でソート
4. 分類時: 各クラス内で末尾 `ratio` 分のグループを inner valid に割り当て（各クラス最低1グループ保証）
5. 回帰時: 末尾 `ratio` 分のグループを inner valid に割り当て
6. グループ単位で完全分離（同一グループが inner train/valid に跨がらない）

**フォールバック条件:** `n_unique_groups < 4` の場合、`StratifiedTimeHoldoutInnerValid`（各クラスの末尾行から `ratio` 分を取得）にフォールバックし、警告を出す。

**StratifiedTimeHoldoutInnerValid:** 各クラス内で時間順序を保持し、末尾 `ratio` 分を inner valid に取る。全クラスが inner valid に最低1行含まれることを保証しつつ、クラス内では時間順序を維持する。

#### バリデーション

- `blocks.col` と `groups.col` が同一カラムの場合 → `CONFIG_INVALID`
- `mode: "sliding"` で `train_window` 未指定 → `CONFIG_INVALID`
- `mode: "expanding"` で `train_window` 指定 → 警告（値は無視）
- `cutoffs` が空 → `CONFIG_INVALID`
- `blocks.col` の値が比較不能 → `DATA_SCHEMA_INVALID`

### 設計判断

| 判断項目 | 選択 | 理由 |
|---|---|---|
| Purge vs Group KFold | Group KFold | データ利用効率が高い。Purge は除外行が多すぎる |
| Config 構造 | セクション分離（blocks/groups） | 2つの軸の役割が視覚的に明確 |
| BaseSplitter IF 変更 | 変更なし | blocks.col 値は Facade がコンストラクタに注入 |
| Inner Valid | 専用 strategy（BlockedGroupInnerValid） | outer と同じグループ分離を inner でも適用 |
| 層化 | auto（タスク依存） | 既存慣例（StratifiedKFold デフォルト）と整合 |
| フォールバック | グループ数 < 4 で行レベル分割 | 少数グループ時の安定性確保 |

### 影響範囲

| 対象 | 変更内容 |
|---|---|
| **新規**: `lizyml/splitters/blocked_group_kfold.py` | `BlockedGroupKFoldSplitter` |
| **新規**: `lizyml/training/inner_valid.py` に追加 | `BlockedGroupInnerValid`, `StratifiedTimeHoldoutInnerValid` |
| `lizyml/config/schema.py` | `BlockedGroupKFoldConfig`, `SplitConfig` union 更新 |
| `lizyml/core/_model_factories.py` | factory 分岐, inner valid auto 解決 |
| `lizyml/core/model.py` | `blocks.col` 抽出 + splitter コンストラクタ注入 |
| `lizyml/splitters/__init__.py` | re-export |
| `BLUEPRINT.md` | §5.4, §10.2, §10.3 更新 |

### 互換性

- 既存 Config / splitter に変更なし（新規 method の追加のみ）
- BaseSplitter インターフェース変更なし
- 既存テストへの影響なし

### 代替案（却下）

1. **Purge 方式**: train/valid に跨がるグループを除去する。データ利用効率が低い（各 fold で 30-50% が除外される）。
2. **専用 splitter 量産**: Group+Time、Group+Group 等の組み合わせごとに専用クラスを作る。組み合わせ爆発。
3. **BaseSplitter IF 拡張**: `split()` に `time_order` パラメータを追加。既存全 splitter のシグネチャ変更が必要。

### 制約・前提

- `blocks.col` は順序付き型（比較演算可能）が必要
- Facade が `blocks.col` でデータをソートする（既存 TS method と同じ規約）
- 2軸分離の構造上、各 fold で「train期間 × valid_users」と「valid期間 × train_users」の行は除外される

### 受け入れ基準

**契約:**
- fold 数 = `len(cutoffs) × groups.n_splits − skip数`
- 各 fold で `train_users ∩ valid_users == ∅`
- 各 fold で train 行の `blocks.col` 値が train 期間内、valid 行が valid 期間内

**再現性:**
- 同一 seed → 同一 fold indices

**層化:**
- binary/multiclass で各 user fold のクラス分布が均等（±許容範囲）

**Inner Valid:**
- inner train/valid でグループ完全分離
- inner valid グループが時間的に遅いグループから選択される
- 分類タスクで各クラス最低1グループが inner valid に含まれる
- グループ数 < 4 でフォールバック発動 + 警告

**Edge case:**
- cutoffs 1つ → 1時間 fold × n_splits
- 全ユーザーが全期間に存在 → 除外多、正常動作
- valid 期間にデータがないユーザー → 正常動作
- min_train_rows / min_valid_rows 未満 → fold スキップ + 警告

## H-0061: LGBMAdapter でユーザー指定 metric を許可 + params_summary に metric 追加

- **ステータス**: Accepted
- **起票日**: 2026-03-28
- **関連 Issue**: #50, #51

### 目的

1. `_build_params()` が `params.metric` を常に破棄する問題を修正し、ユーザーが LightGBM の evaluation metric をカスタマイズできるようにする。
2. `params_summary()` の出力に `metric` を含め、Widget 等の下流が使用 metric を表示できるようにする。

### 影響範囲

- `lizyml/estimators/lgbm/adapter.py` — `_build_params()` の metric 処理変更
- `lizyml/estimators/lgbm/provider.py` — `params_summary()` に metric 行追加
- 学習履歴（`eval_history`）のキーがユーザー指定 metric に応じて変化する

### 互換性

- **後方互換**: `params` に `metric` 未指定時は従来通り `_TASK_METRIC[task]` がフォールバック
- `params_summary()` の返却は `list[dict]` のまま。行が 1 つ増えるだけで shape 変更なし

### バリデーション方針

- LightGBM に委任（案 A）。無効 metric は LightGBM がランタイムエラーを返す
- `LizyMLError` で wrap し、ユーザー指定 metric 値をコンテキストに含めてエラー箇所を特定可能にする

### 代替案

- 案 B: ホワイトリストで事前バリデーション → LightGBM バージョン依存で保守コスト大、却下

### 受け入れ基準（テスト観点）

- ユーザー指定 metric が Booster params に到達する
- 未指定時は `_TASK_METRIC` フォールバック
- 無効 metric で `LizyMLError`（context に metric 値を含む）
- `params_summary()` に metric 行が含まれる

## H-0062: plot_learning_curve() に metrics フィルタパラメータ追加

- **ステータス**: Accepted
- **起票日**: 2026-03-28
- **関連 Issue**: #52

### 目的

`plot_learning_curve()` に `metrics` パラメータを追加し、表示する metric をフィルタ可能にする。Widget 等の表示幅が限られた環境で、選択した metric のみプロットできるようにする。

### 影響範囲

- `lizyml/plots/learning_curve.py` — 関数シグネチャに `metrics: list[str] | None = None` 追加

### 互換性

- **完全後方互換**: `metrics=None`（デフォルト）で既存の全 metric プロット動作を維持
- 公開 API にオプショナル keyword-only パラメータを追加するのみ

### 仕様

- `metrics=None`: 全 metric をプロット（既存動作）
- `metrics=["auc"]`: eval_history キーの `/` 以降（metric 名部分）が一致するもののみ表示
- 一致する metric が 0 件の場合: `LizyMLError` で利用可能な metric 名を提示

### 代替案

- subplot の max_cols 制限 + ページング → 実装が複雑、Widget 側でフィルタする方が自然。却下

### 受け入れ基準（テスト観点）

- `metrics=None` で全 metric プロット（後方互換）
- `metrics=["auc"]` で該当 metric のみフィルタ
- 存在しない metric 指定で `LizyMLError`（利用可能な metric リスト付き）

## H-0063: Config 伝搬・実効性テスト網羅化

- **ステータス**: Accepted
- **起票日**: 2026-03-28

### 目的

Config の全フィールドが適切なクラス・関数に渡され、その値が実際の動作に反映されていることを検証するテストを追加する。現状のテストは「値が渡っている」伝搬テストが中心で、「渡った値が動作を変える」実効性テストが不足している。

### 影響範囲

- テストのみ。プロダクションコードの変更なし
- `tests/test_estimators/test_param_behavioral_effect.py`（新規）
- `tests/test_core/test_config_propagation.py`（既存の伝搬テスト補強）

### テスト設計

#### A. Booster パラメータ実効性テスト（2 値比較パターン）

異なる値で fit → 予測が変わることを検証:

| パラメータ | 値 A | 値 B |
|-----------|------|------|
| `learning_rate` | 0.01 | 0.5 |
| `max_depth` | 3 | 8 |
| `n_estimators` | 10 | 100 |
| `max_bin` | 63 | 511 |
| `lambda_l1` | 0 | 10.0 |
| `lambda_l2` | 0 | 10.0 |
| `bagging_fraction` + `bagging_freq` | 0.5/1 | 1.0/0 |
| `feature_fraction` | 0.3 | 1.0 |
| `boosting` | `gbdt` | `rf` |
| `metric` | `["auc"]` | `["binary_logloss"]` |
| `num_leaves` | 8 | 64（auto_num_leaves=False） |
| `min_data_in_leaf` | 5 | 50（直接指定） |

#### B. Smart Parameters 動作反映テスト

| パラメータ | 検証内容 |
|-----------|---------|
| `feature_weights` | 重み付きで fit → importance 順序が変わる |
| `balanced` | binary で scale_pos_weight → 不均衡データの予測分布が変わる |
| `min_data_in_leaf_ratio` vs 直接指定 | ratio と直接指定の排他動作 |

#### C. Training / Feature / Calibration 実効性テスト

| パラメータ | 検証内容 |
|-----------|---------|
| `early_stopping.random_state` | 同一 seed → 同一 inner split |
| `validation_ratio` | 0.1 vs 0.4 → inner valid サイズが比例 |
| `features.auto_categorical` | True で string 列が自動検出 |
| `calibration.params` | カスタムパラメータが calibrator に到達 |
| `verbosity` | `-1` 固定 → stdout に出力なし |
| `scale_pos_weight` | 1.0 vs 10.0 → 予測分布が変わる |
| `objective` | task 固定が Booster に正しく到達 |
| `num_class` | multiclass でクラス数が正しく設定 |

#### D. 伝搬テスト補強（不足分）

| パラメータ | 検証内容 |
|-----------|---------|
| `bagging_freq` | Booster params に到達 |
| `lambda_l1` / `lambda_l2` | Booster params に到達 |
| `first_metric_only` | Booster params に到達 |
| 任意パラメータ透過 | `path_smooth` 等の任意キーが Booster にそのまま到達 |

### 代替案

- 全パラメータの E2E テスト → 実行時間が長すぎる。adapter 単位の 2 値比較パターンで効率的にカバー

### 受け入れ基準（テスト観点）

- 上記 A〜D の全項目に対するテストが存在し pass する
- 既存テストに regression なし

## H-0064: LightGBM 学習用 metric の統合管理（マッピング・バリデーション・feval）

- **ステータス**: Done（PR #60）
- **起票日**: 2026-03-28
- **関連 Issue**: #57, #58, #59

### 目的

LizyML 評価用メトリクス名と LightGBM 学習用 `metric` パラメータの間にマッピング・バリデーション・カスタム feval 生成の統合管理層を導入する。現状、ユーザーは LizyML 名と LightGBM 名の両方を把握する必要があり、無効な metric 名の事前検証もない。

### 影響範囲

- `lizyml/estimators/lgbm/metric_bridge.py`（新規）— マッピング・バリデーション・feval 生成
- `lizyml/estimators/lgbm/adapter.py` — `_build_params()` 戻り値拡張、`fit()` の feval 注入
- `lizyml/estimators/lgbm/__init__.py` — re-export 追加（必要時）
- Config の `params={"metric": "..."}` の意味が拡張される（LizyML 名も受付可能に）

### 3 つの課題と解決方針

#### A. メトリクス名マッピング (#58)

LizyML 名と LightGBM 名が異なるメトリクスの自動変換:

| LizyML 名 | LightGBM 名 | タスク |
|-----------|-------------|-------|
| `logloss` | `binary_logloss` / `multi_logloss` | binary / multiclass |
| `auc_pr` | `average_precision` | binary / multiclass |

`accuracy` は LightGBM の `binary_error`/`multi_error` と意味が逆（higher is better vs lower is better）のため自動変換せず、feval で対応する。

#### B. ホワイトリストバリデーション (#57)

LightGBM ネイティブメトリクスのタスク別ホワイトリストを定義し、`_build_params()` 内でマッピング適用後に事前検証する。feval 対象メトリクスはバイパスする。

| タスク | 有効な LightGBM ネイティブメトリクス |
|-------|----------------------------------|
| regression | `l1`, `l2`, `rmse`, `quantile`, `mape`, `huber`, `fair`, `poisson`, `gamma`, `gamma_deviance`, `tweedie`, `r2` |
| binary | `binary_logloss`, `binary_error`, `auc`, `average_precision`, `cross_entropy`, `cross_entropy_lambda`, `kullback_leibler` |
| multiclass | `multi_logloss`, `multi_error`, `auc`, `auc_mu` |

#### C. feval カスタム関数 (#59)

LightGBM に存在しないメトリクスを `feval` 引数経由で注入:

| Metric | Regression | Binary | Multiclass | y_pred 変換 |
|--------|:---:|:---:|:---:|------------|
| `rmsle` | ✅ | | | そのまま |
| `f1` | | ✅ | ✅ | sigmoid / softmax → 閾値 / argmax |
| `brier` | | ✅ | ✅ | sigmoid / softmax |
| `ece` | | ✅ | | sigmoid |
| `precision_at_k` | | ✅ | | sigmoid |
| `accuracy` | | ✅ | ✅ | sigmoid / softmax → 閾値 / argmax |

### 互換性

- `_build_params()` は private API — 戻り値の拡張（feval リスト追加）は外部互換に影響しない
- `fit()` の外部シグネチャは変更なし
- `params={"metric": "binary_logloss"}` 等の既存指定はそのまま動作（マッピングは LizyML 名のみ変換）
- 既存の post-hoc エラー検出（defense in depth）は残す

### 代替案

1. **マッピングなし（LightGBM 名のみ受付）** — ユーザー体験が悪い。評価用と学習用で異なる名前を覚える必要がある
2. **feval なし（ネイティブメトリクスのみ）** — LizyML 独自メトリクスを学習時に使えない。機能制限が大きい
3. **バリデーションなし（現状維持）** — LightGBM が黙って無視するケースがあり、デバッグ困難

### 受け入れ基準（テスト観点）

- `params={"metric": "logloss"}` が binary で `binary_logloss` に、multiclass で `multi_logloss` に自動変換される
- 無効なメトリクス名が `_build_params()` 段階で `LizyMLError(CONFIG_INVALID)` を raise する
- タスク非互換メトリクス（regression + `auc` 等）がバリデーションで弾かれる
- `params={"metric": "f1"}` で binary 訓練時に eval_results に `f1` が記録される
- native + feval の混在（`["auc", "brier"]`）が動作する
- feval 付きの early_stopping が正常に機能する
- 全既存テストがパスする
- 新規テスト ~50 が追加される

## H-0065: パラメータ付き MetricEntry（precision_at_k の k 設定可能化）

- **ステータス**: Accepted
- **起票日**: 2026-03-28
- **関連**: H-0064 (metric_bridge)

### 目的

`precision_at_k` の `k` パラメータをユーザーが設定可能にする。Evaluation と Model Params（LightGBM 学習用 metric）の両方で独立した `k` を指定できるようにし、Plot 凡例と `params_summary()` で設定値を表示して事故を防止する。

### 設計方針（B-1: 使う場所で設定する）

`EvaluationConfig.metrics` と `model.lgbm.params.metric` の両方で `str | dict[str, dict[str, Any]]` 形式をサポートする。

```python
# Config 例（YAML 表記）
evaluation:
  metrics:
    - auc
    - precision_at_k:           # dict 形式: {metric_name: {param: value}}
        k: 20

model:
  lgbm:
    params:
      metric:
        - logloss
        - precision_at_k:
            k: 5               # Evaluation とは独立した k
```

### 型定義

```python
MetricEntry = str | dict[str, dict[str, Any]]
```

- `str`: 従来通りのデフォルトパラメータ（後方互換）
- `dict`: キーが metric 名、値がパラメータ辞書。キー数は必ず 1。

### 影響範囲

- `lizyml/metrics/registry.py` — `parse_metric_entry()` ユーティリティ新設、`get_metric()` に kwargs サポート
- `lizyml/config/schema.py` — `EvaluationConfig.metrics` の型を `list[MetricEntry]` に拡張
- `lizyml/evaluation/evaluator.py` — `evaluate()` が `list[MetricEntry]` を受け取る
- `lizyml/estimators/lgbm/metric_bridge.py` — `resolve_metrics()` が `list[MetricEntry]` を処理
- `lizyml/estimators/lgbm/adapter.py` — `_build_params()` が dict 形式 metric を処理
- `lizyml/estimators/lgbm/provider.py` — `params_summary()` で feval metric のパラメータ（k 等）を表示
- `lizyml/plots/learning_curve.py` — subplot_titles で metric パラメータを表示
- `lizyml/core/model.py` — `fit()` / `evaluate()` が MetricEntry を伝搬
- `lizyml/core/_model_metrics.py` — `filter_metrics()` が MetricEntry 対応

### name プロパティは変更しない

`PrecisionAtK.name` は `"precision_at_k"` のまま維持する。`k` の可視化は以下に限定:
- **Plot 凡例**: subplot_titles で `precision_at_k (k=20)` のように表示
- **params_summary()**: metric 行で `precision_at_k (k=5)` のように表示

### 互換性

- `list[str]` はそのまま動作（後方互換完全維持）
- `PrecisionAtK.name` 不変 → 結果 dict のキーは `"precision_at_k"` のまま
- `get_metric()` の既存呼び出し（引数なし）は従来通り動作
- `LGBMConfig.params` は `dict[str, Any]` のままで型変更なし（metric 値のパース時に dict を処理）

### 代替案

1. **案A: EvaluationConfig にトップレベル `precision_at_k` フィールド** — 将来パラメータ付き metric 追加時にフィールドが増える
2. **案B-2: k は EvaluationConfig のみ、Model Params は自動参照** — Model Params のみで使うケースに対応できない
3. **案B-3: metric_params セクションに集約** — Model Params との関連が初見で分からない

### 将来の拡張性

この `dict` 形式は `precision_at_k` 固有ではなく、将来の `ndcg@k` 等のパラメータ付きメトリクスにも汎用的に使える。

### 受け入れ基準（テスト観点）

- `metrics: ["auc", {"precision_at_k": {"k": 20}}]` で Evaluation 結果キーが `"precision_at_k"` で k=20 の値になる
- `params={"metric": [{"precision_at_k": {"k": 5}}]}` で feval が k=5 で動作し eval_history に記録される
- `metrics: ["precision_at_k"]` でデフォルト k=10 のまま動作（後方互換）
- 不正な dict 形式（キー数 ≠ 1、未知の metric 名、不正な k 値）がバリデーションエラー
- `params_summary()` で metric の k 値が表示される
- learning curve の subplot_titles で k 値が表示される
- 全既存テストがパスする

## H-0066: Codegen feval metric サポート（Metric Bridge 追従）

- **ステータス**: Accepted
- **起票日**: 2026-04-02
- **関連**: H-0059 (Codegen Export), H-0064 (Metric Bridge), H-0065 (MetricEntry)

### 目的

H-0064 で導入された feval metric（f1, brier, ece, precision_at_k, accuracy, rmsle, r2）が `export_code()` で生成される codegen 出力に反映されない問題を修正する。現状、`_build_params()` の feval 情報は `_, _` で破棄されており、feval-only metric 使用時は `lgbm_params.metric = "None"` が config.json に書き込まれる。これにより early stopping の監視指標が LizyML 本体と異なる挙動になる。

### 設計方針（B案: feval 再実装）

`train.py` テンプレートに pure numpy/scipy の feval callable を再実装し、`config.json` に feval metric のメタ情報を記録する。codegen 実行時に feval metric を検出し、生成コード内で同一の feval 関数を再構築する。

### 変更内容

1. **`config.json` 契約拡張**: `feval_metrics` フィールド追加
   ```json
   {
     "feval_metrics": [
       {"name": "f1", "params": {}, "greater_is_better": true, "needs_proba": false},
       {"name": "precision_at_k", "params": {"k": 20}, "greater_is_better": true, "needs_proba": true}
     ]
   }
   ```

2. **`train.py` テンプレート拡張**: feval セクション追加
   - `_codegen_sigmoid`, `_codegen_softmax` ヘルパー
   - 各 metric の pure numpy 実装（rmsle, r2, f1, brier, ece, precision_at_k, accuracy）
   - `build_feval_from_config()` ファクトリ: config.json → feval callable リスト
   - `train_lgbm()` の `lgb.train()` 呼び出しに `feval` パラメータ追加

3. **`_model_persistence.py` 修正**: feval メタ情報を config の metric 設定から再構築し `generate_code()` に渡す

### 影響範囲

- `lizyml/core/_model_persistence.py` — feval 情報の伝搬
- `lizyml/codegen/config_writer.py` — `feval_metrics` フィールド追加
- `lizyml/codegen/generator.py` — `feval_metrics` パラメータ追加
- `lizyml/codegen/templates.py` — `train.py` テンプレートに feval セクション追加
- `BLUEPRINT.md` §6.6 / §15.4 — feval 対応の追記

### 互換性

- `feval_metrics` が空リスト `[]` の場合、既存の codegen 出力と完全に同一（後方互換）
- `predict.py` / `test_equivalence.py` は予測のみなので変更不要
- `config.json` に新フィールド追加のみ（既存フィールドは不変）
- `generate_code()` / `build_config()` の新パラメータはデフォルト値 `[]` で後方互換

### 代替案

1. **案A: native metric にフォールバック** — feval metric を捨て、task default metric に差し替える。簡単だが学習挙動が変わる
2. **案C: 警告のみ** — feval 使用時に `UserWarning` を出すだけ。ユーザー体験が悪い

### 受け入れ基準（テスト観点）

- feval metric 使用時の `export_code()` で `config.json` に `feval_metrics` が正しく記録される
- `feval_metrics` が空リストの場合、既存テスト 73 件が全 PASS（後方互換）
- `train.py` テンプレートの各 feval 関数が LizyML 本体と同一の値を返す（`rtol=1e-10`）
- `metric="None"` + feval-only の組み合わせで `train.py` が正常に動作する
- multiclass feval（f1, brier, accuracy）の reshape + softmax が正しく動作する
- `precision_at_k` の `k` パラメータが config.json 経由で正しく伝搬される
- 品質ゲート（ruff / mypy / pytest）全 PASS

## H-0067: コードベース監査バグ修正バッチ（9件）

- **ステータス**: Accepted
- **起票日**: 2026-04-11
- **関連**: H-0057 (OOF Coverage), H-0058 (Outer Split Calibration), H-0064 (Metric Bridge)

### 目的

コードベース全体の監査で発見された 9 件のバグを一括修正する。メトリクス計算の正確性、leakage 境界の一貫性、防御的プログラミングの強化が対象。

### 変更内容

1. **ECE 計算式修正** (`metrics/classification.py`, `codegen/templates.py`):
   - 各 calibration bin 内の accuracy を `mean((y_pred >= 0.5) == y_true)`（二値化精度）から `mean(y_true)`（正例割合 = fraction-of-positives）に修正。標準的な ECE 定義に準拠。

2. **confusion_matrix_table NaN 除外** (`evaluation/confusion.py`):
   - OOS 混同行列に `compute_oof_valid_mask()` を適用し、構造的にカバーされない行（TimeSeriesCV 最初の期間等）を除外。修正前は NaN >= 0.5 → False → 偽の負例として計上されていた。

3. **リーク検知の短絡評価順序** (`data/validators.py`):
   - `np.allclose(dropna(), dropna())` の前に `isna().equals()` を評価するよう順序を変更。NaN 位置が異なる場合の `ValueError` が `except` で飲み込まれてリーク検知がスキップされる問題を修正。

4. **isotonic log_evaluation period** (`calibration/isotonic.py`):
   - `lgbm.log_evaluation(period=0)` を `period=-1` に変更。LightGBM 4.x で `period=0` は未定義挙動。`LGBMAdapter` の `period=-1` と統一。

5. **RefitTrainer pipeline leakage 境界** (`training/refit_trainer.py`):
   - pipeline を inner-train のみで fit するよう変更（CVTrainer と一致する leakage 境界）。最終的な `pipeline_state`（推論用）は別途全データで fit した pipeline から取得。`categorical_features` も final pipeline から取得。`NoInnerValid` 時は二重 fit を回避。

6. **cross_fit NaN guard** (`calibration/cross_fit.py`):
   - `val_idx` に NaN 行が含まれる場合の 3 分岐ガードを追加: all finite → `cal.predict()`、mixed → finite のみ predict + NaN は fallback、all NaN → fallback。

7. **calibrated metrics に oof_per_fold 追加** (`core/_model_metrics.py`):
   - `metrics["calibrated"]` に `oof_per_fold` を追加。IF metrics は leakage リスクのため引き続き除外。calibrated ブランチの構造: `{"oof": {...}, "oof_per_fold": [...]}`。

8. **HoldoutInnerValid 空 train ガード** (`training/inner_valid.py`):
   - `n_valid >= n_samples` の場合に `ValueError` を発出。修正前は空の train set が LightGBM に渡されて cryptic なエラーが発生。

9. **TimeHoldoutInnerValid 空 train ガード** (`training/inner_valid.py`):
   - 同上。`n_samples=1` でも発生する。

### 影響範囲

- `lizyml/metrics/classification.py` — ECE 計算式
- `lizyml/codegen/templates.py` — codegen ECE 計算式
- `lizyml/evaluation/confusion.py` — OOS 混同行列
- `lizyml/data/validators.py` — リーク検知
- `lizyml/calibration/isotonic.py` — log 抑制
- `lizyml/calibration/cross_fit.py` — NaN ガード
- `lizyml/training/refit_trainer.py` — pipeline fit 境界
- `lizyml/core/_model_metrics.py` — calibrated metrics 構造
- `lizyml/training/inner_valid.py` — 空 train ガード

### 互換性

- **ECE**: 計算結果が変わるが、修正前の値が誤りであるため後方互換の問題ではない
- **confusion_matrix_table**: NaN 行が除外されるため、TimeSeriesCV 使用時に行数が変わる
- **calibrated metrics**: `oof_per_fold` キーが追加される（追加方向、後方互換）
- **RefitTrainer**: 学習結果が微妙に変わる可能性（pipeline fit 境界変更）。現行の `NativeFeaturePipeline` は y 非使用のため実質的な影響なし
- **inner_valid**: 極端なエッジケースで新たに `ValueError` が発生するようになる

### 受け入れ基準（テスト観点）

- 各バグに対する回帰テスト（16 件追加）
- 既存テスト 1478 件が引き続き PASS（テスト総数 1495）
- 品質ゲート（ruff / mypy / pytest）全 PASS

## H-0068: Re-tune（Study Resume + 境界検知拡張）

- **ステータス**: Accepted
- **起票日**: 2026-04-11
- **スコープ**: Public API | Tuning | Types
- **関連**: BLUEPRINT.md §11, H-0048 (Progress Callback), H-0050 (TuningResult 3分割)

### 目的

初回 tuning 後に追加探索を行い、さらなる精度向上を目指す re-tune 機能を提供する。主に以下の 2 つの機能で構成する:

1. **Study Resume**: 前回の Optuna Study を保持し、追加試行を行う（TPE が過去試行から学習済み）
2. **境界検知 + 非対称拡張**: best params が探索空間の端に張り付いている次元を自動検知し、有望方向にのみ探索空間を拡張する

狭めるのではなく「まだ見ていない有望領域を探しに行く」発想。

### 背景・調査結果

- **Optuna**: `study.optimize()` 再呼び出しで試行追加可能。TPE sampler は過去試行を自動活用。`enqueue_trial()` で前回 best を初期候補に注入可能。
- **FLAML**: `points_to_evaluate` + progressive widening（低コスト→高コスト）。
- **PBT (DeepMind)**: best の ×0.8/×1.2 摂動で次世代生成。探索空間に明示境界なし。
- **共通リスク**: 同一 validation fold での繰り返し評価は過適合を招く。LizyML は OOF 評価のため単純リークはないが、試行数増加による選択バイアスは残る。

### Proposal

#### 1. `Model.tune()` API 拡張

```python
def tune(
    self,
    data: pd.DataFrame | None = None,
    *,
    resume: bool = False,
    n_trials: int | None = None,
    expand_boundary: bool | None = None,
    boundary_threshold: float = 0.05,
    progress_callback: TuneProgressCallback | None = None,
) -> TuningResult:
```

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `resume` | `False` | `True` の場合、前回の Study を再利用して追加試行を行う |
| `n_trials` | `None` | 追加試行数。`None` の場合 `config.tuning.optuna.params.n_trials` を使用 |
| `expand_boundary` | `None` | 境界拡張の有無。`None` の場合、デフォルト空間では `True`、ユーザー指定空間では `False` |
| `boundary_threshold` | `0.05` | 端判定の閾値（0.0〜1.0）。best の位置が端から threshold 以内なら拡張候補 |

制約:
- `resume=False` は現在と同一動作（完全な後方互換）
- `resume=True` で `tune()` 未呼び出しの場合は `LizyMLError(TUNING_FAILED)` を送出
- `expand_boundary=True` でユーザー指定空間の場合も動作する（明示許可）

#### 2. 境界検知ロジック

`tuning/search_space.py` に `detect_boundary()` と `expand_dims()` を追加。

**検知ルール**:
- `FloatDim` / `IntDim`（linear）: `(best - low) / (high - low) < threshold` → 下限近傍、`(high - best) / (high - low) < threshold` → 上限近傍
- `FloatDim` / `IntDim`（log）: 対数空間で同一の計算を行う
- `CategoricalDim`: 拡張不可（ログで通知のみ）

**拡張ルール**:
- linear: 端方向に `(high - low)` を追加（つまり range を 2 倍に拡張）
- log: 端方向に対数空間で 3 倍に拡張（例: low=0.0001 → low=0.0000333）
- `IntDim`: 拡張後の値を int に丸める。low は `max(1, new_low)` で下限ガード
- 反対側の端は据え置き（非対称拡張）

**戻り値型**:

```python
@dataclass(frozen=True)
class BoundaryDimStatus:
    name: str
    best_value: float | int | str | None
    low: float | int | None
    high: float | int | None
    position_pct: float | None   # 0.0〜1.0
    edge: str                    # "lower" | "upper" | "none"
    expanded: bool
    new_low: float | int | None
    new_high: float | int | None

@dataclass(frozen=True)
class BoundaryReport:
    dims: tuple[BoundaryDimStatus, ...]
    expanded_names: tuple[str, ...]
```

#### 3. Tuner の Study 保持

`Tuner.tune()` に `study` 引数を追加（省略時は従来通り新規作成）。`enqueue_trial` で前回 best を注入。

```python
def tune(
    self,
    objective: Any,
    metric_name: str = "rmse",
    *,
    study: Any | None = None,
    enqueue_params: dict[str, Any] | None = None,
) -> tuple[TuningResult, Any]:
    # Returns (result, study) — study を Model が保持して resume に使う
```

#### 4. TuningResult 拡張

```python
@dataclass(frozen=True)
class RoundSummary:
    round: int                        # 1-indexed
    n_trials: int
    best_score_before: float | None   # ラウンド開始前の best
    best_score_after: float           # ラウンド終了時の best
    expanded_dims: tuple[str, ...]
    space_snapshot: tuple[SearchDim, ...]

@dataclass(frozen=True)
class TuningResult:
    # --- 既存（変更なし） ---
    best_model_params: dict[str, Any]
    best_smart_params: dict[str, Any]
    best_training_params: dict[str, Any]
    best_score: float
    trials: list[TrialResult]
    metric_name: str
    direction: str
    # --- 追加 ---
    rounds: tuple[RoundSummary, ...]
    boundary_report: BoundaryReport | None
```

- `rounds` のデフォルトは `(RoundSummary(round=1, ...),)`（初回 tune でも 1 要素）
- `boundary_report` は `resume=True` 時のみ設定。初回 tune では `None`

#### 5. TuneProgressInfo 拡張

```python
@dataclass(frozen=True)
class TuneProgressInfo:
    # --- 既存 ---
    current_trial: int
    total_trials: int
    elapsed_seconds: float
    best_score: float | None
    latest_score: float | None
    latest_state: str
    # --- 追加 ---
    round: int                          # 1-indexed
    cumulative_trials: int              # 全ラウンド通算
    expanded_dims: tuple[str, ...]      # このラウンドで拡張された次元名
```

#### 6. TrialResult 拡張

```python
@dataclass(frozen=True)
class TrialResult:
    number: int
    params: dict[str, Any]
    score: float
    state: str
    round: int  # 追加: どのラウンドの試行か (1-indexed)
```

#### 7. tuning_table() 拡張

`round` 列と `state` 列を追加:

```
trial  round  rmse     learning_rate  num_leaves  state
0      1      0.312    0.005          128         complete
...
50     2      0.283    0.00008        300         complete
```

#### 8. boundary_table() 新設

`_model_tables.py` に `boundary_table()` メソッドを追加:

```
dim               best     low       high     position  edge   expanded  new_low  new_high
learning_rate     0.00015  0.0001    0.1      1.1%      lower  True      0.00001  0.1
num_leaves        251      16        256      97.9%     upper  True      16       512
feature_fraction  0.72     0.5       1.0      44.0%     none   False     —        —
```

#### 9. plot_tuning_history() 拡張

- ラウンド境界に縦の破線を追加
- ラウンドごとにアノテーション（拡張された次元名）
- best score の累積線はラウンドをまたいで連続描画

#### 10. ログ出力

```
INFO  tune.resume: expanding 2 of 5 dims
        learning_rate: lower bound 0.0001 → 0.00001 (best 0.00015 near lower edge)
        num_leaves: upper bound 256 → 512 (best 251 near upper edge)
INFO  tune.resume: enqueued previous best as initial trial
INFO  tune.resume: starting 30 additional trials (80 cumulative)
```

#### 11. Widget / Studio 連携仕様

LizyML Core は callback + 結果型でデータを提供し、Widget/Studio が消費する。

**Widget（リアルタイムモニタ）向け情報**:

| Widget 要素 | 情報源 |
|---|---|
| Round 表示 | `TuneProgressInfo.round` |
| 進捗バー | `TuneProgressInfo.cumulative_trials` |
| 改善幅 | `best_score` vs `RoundSummary.best_score_before` |
| 拡張パネル | `TuneProgressInfo.expanded_dims` |
| Score History | callback 呼び出しごとに蓄積 |

**Studio（ダッシュボード）向け情報**:

| Studio 要素 | 情報源 |
|---|---|
| Round History テーブル | `TuningResult.rounds` |
| Search Space Evolution | `RoundSummary.space_snapshot` |
| 収束判定 | `expanded_dims == ()` AND 改善 < threshold |
| boundary_table() | `TuningResult.boundary_report` |

収束判定ロジック自体は LizyML Core には含めず、Studio/Widget 側の責務とする。Core は判断材料のみを提供する。

### 影響範囲

- `lizyml/core/types/tuning_result.py` — TuningResult, TuneProgressInfo, TrialResult 拡張 + RoundSummary, BoundaryReport, BoundaryDimStatus 追加
- `lizyml/tuning/search_space.py` — detect_boundary(), expand_dims() 追加
- `lizyml/tuning/tuner.py` — study 引数追加、enqueue_trial 対応
- `lizyml/core/model.py` — tune() に resume/n_trials/expand_boundary/boundary_threshold 追加、`_study` 保持
- `lizyml/core/_model_tables.py` — tuning_table() に round/state 列追加、boundary_table() 新設
- `lizyml/plots/tuning.py` — plot_tuning_history() にラウンド区切り線追加
- `lizyml/__init__.py` — 新型の公開面追加

### 互換性

- `tune()` のデフォルト動作は変更なし（`resume=False`）→ 完全後方互換
- `TuningResult` に `rounds` と `boundary_report` フィールドが追加される。既存コードで positional args を使っている場合は影響があるが、frozen dataclass は keyword-only 使用が慣例
- `TuneProgressInfo` に 3 フィールド追加。callback が属性アクセスで使用している場合は影響なし（追加方向）
- `TrialResult` に `round` フィールド追加（デフォルト `1`）。追加方向
- `tuning_table()` に `round` と `state` 列が追加される（追加方向）
- `plot_tuning_history()` は初回 tune のみの場合、区切り線なしで従来と同一表示

### 代替案

1. **Space Narrowing（探索空間絞り込み）**: best 周辺に範囲を狭める。真の最適が範囲外にある場合に見逃すリスクが高い。過適合リスクも拡張型より高い。
2. **Successive Halving**: n_estimators を段階的に増やす多段評価。LizyML の CV ベース評価とは設計思想が異なる。
3. **拡張なしの純粋 Resume のみ**: 実装は簡単だが、探索空間の端に張り付いた場合に改善の余地がない。

### 受け入れ基準（テスト観点）

1. `resume=False` で既存テストが全 PASS（後方互換）
2. `resume=True` で累計試行数が正しく増加し、best_score が悪化しない
3. 境界検知: 端に張り付いた次元が正しく検知される（linear/log/categorical 各ケース）
4. 非対称拡張: 端方向のみ拡張され、反対側は据え置き
5. `TuningResult.rounds` が正しい RoundSummary を含む
6. `TuneProgressInfo` の追加フィールドが正しく報告される
7. `TrialResult.round` が正しいラウンド番号を持つ
8. `tuning_table()` に `round` / `state` 列が存在する
9. `boundary_table()` が BoundaryReport を正しく DataFrame に変換する
10. `plot_tuning_history()` でラウンド境界が描画される
11. ユーザー指定空間 + `expand_boundary=None` → 拡張されない
12. デフォルト空間 + `expand_boundary=None` → 拡張される
13. `resume=True` で未 tune → `TUNING_FAILED` エラー
14. 品質ゲート（ruff / mypy / pytest）全 PASS

## H-0069: `validation_ratio` を computed_field 化（Issue #95 構造的根治）

- **ステータス**: Accepted
- **起票日**: 2026-05-02
- **スコープ**: Public Config | Schema | Persistence
- **関連**: [Issue #95](https://github.com/nbx-liz/LizyML/issues/95), [LizyStudio #345](https://github.com/nbx-liz/LizyStudio/issues/345)

### 目的

`EarlyStoppingConfig.validation_ratio` と `inner_valid.ratio` が「両方 mutable / 両方 dump 出力 / 同期は validator の片方向のみ」という二重表現になっており、以下のバグを構造的に発生させている:

1. **Issue #95（顕在）**: `inner_valid` が `group_holdout` / `time_holdout` のとき `Model.save()` → `Model.load()` で round-trip が `ValidationError` で落ちる。LizyStudio #345 の production 500 エラーを引き起こしている
2. **codegen silent ratio mismatch（潜在）**: `inner_valid={method:..., ratio:0.25}` を渡しても `validation_ratio` は default 0.1 のまま。`_model_persistence.py:206` が `es.validation_ratio` を `export_code()` に渡すため、生成 `train.py` が誤った holdout 比率で動作する

これらは個別に対症修正（B 案: validator 内双方向同期）しても、二重表現自体が残るため再発リスクが高い。本 Proposal は `validation_ratio` を `inner_valid.ratio` から派生する read-only `@computed_field` に正規化し、Single Source of Truth 化することで二重表現を根絶する。

### 変更内容

1. **`EarlyStoppingConfig` schema 改訂** (`lizyml/config/schema.py`)
   - `validation_ratio: float | None = 0.1` を **削除**（stored field でなくなる）
   - `@computed_field` として `validation_ratio` プロパティを追加 — `inner_valid.ratio` を返す read-only
   - `_resolve_validation_ratio` validator を削除し、以下に置換:
     - `mode="wrap"` validator で legacy YAML 入力 (`{"validation_ratio": 0.1}` のみ) を `{"inner_valid": {"method": "holdout", "ratio": 0.1}}` に変換 + round-trip 入力 (`validation_ratio` と `inner_valid` 両方在) は `validation_ratio` を strip
     - mode="wrap" 内で `_inner_valid_explicit` PrivateAttr を「ユーザーが入力で `inner_valid` を明示し、かつ `validation_ratio` を併記しなかった場合のみ True」に設定（既存 auto-resolve セマンティクス維持）
   - `mode="after"` で `inner_valid is None` のとき default `HoldoutInnerValidConfig(method="holdout", ratio=0.1)` を補填

2. **下流の読み出し経路は無変更**
   - `_model_persistence.py:206`, `_model_tables.py:290` は `es.validation_ratio` を読むが、computed_field なので呼び出しは不変。値は自動的に正しくなる
   - `model.py:795` の `tp["validation_ratio"]` 読み出し（Tuner 探索次元）は引数辞書ベースなので無変更
   - `defaults.py:73` の `FloatDim("validation_ratio", ...)` も無変更（探索次元名として継続使用）

3. **既存 `model.lizyml` artifact 互換**
   - 旧 `metadata.json` には `validation_ratio: 0.1` が含まれる → mode="wrap" の round-trip strip ロジックで透過的に受理される
   - `format_version` bump 不要

### 影響範囲

- `lizyml/config/schema.py` — `EarlyStoppingConfig` schema（中核）
- `tests/test_config/test_early_stopping_defaults.py` — 既存テスト確認（API 不変なので PASS のはず）
- `tests/test_config/test_early_stopping_roundtrip.py` — 新規 round-trip 回帰テスト（Issue #95 受け入れ基準）
- `tests/regression/test_reg_issue_95_*.py` — 永続化互換テスト（旧形式 metadata.json の読み込み）
- BLUEPRINT.md — `EarlyStoppingConfig` セクションがあれば更新

### 互換性

- **YAML 入力（user-facing）**: 完全互換
  - `{"validation_ratio": 0.1}` → 内部で `inner_valid: holdout` に正規化（既存挙動と同じ）
  - `{"inner_valid": {...}}` → そのまま（既存挙動と同じ）
  - 両方併記 + 値一致 → OK（既存 round-trip allowance を継承）
  - 両方併記 + 値不一致 → `ValueError`（実コンフリクトの検知は維持）
- **`Model.load()` 互換**: 旧 artifact (`validation_ratio: 0.1` を含む metadata) はそのまま load 可能
- **`cfg.training.early_stopping.validation_ratio` の読み取り**: 引き続き動作（computed_field）
- **`auto-resolve` セマンティクス**: `_inner_valid_explicit` フラグの設定ルール変更なし（legacy YAML や round-trip では False、明示的 inner_valid では True）。`_model_factories.py:253` の挙動は不変
- **format_version**: 変更なし（schema 入出力契約は維持）

### 代替案

- **A. `isinstance` チェックを緩和するだけ**: Issue #95 の症状のみ修正。`validation_ratio ↔ inner_valid.ratio` の同期欠落は残存し codegen silent bug は手付かず → 場当たり修正
- **B. validator 内で双方向同期**: 同期欠落を修正するが二重表現は残存。新コンシューマが `validation_ratio` を mutable 前提で書くと再発リスク → 中庸
- **C. computed_field 化（本 Proposal）**: 二重表現を構造的に根絶。ユーザー API（read 経路）は完全互換 → **採用**
- **D. `validation_ratio` 完全撤廃**: 全 consumer を `inner_valid.ratio` 直読に変更。最もクリーンだが破壊的（YAML 入力 + Tuner 探索次元名）。次メジャー（v1.0）に持ち越し

### 受け入れ基準（テスト観点）

1. **Issue #95 直接修正**: 3 つの `InnerValidConfig` discriminant (`holdout` / `group_holdout` / `time_holdout`) について `model_validate(model_dump())` round-trip がすべて成功
2. **non-default ratio round-trip**: 上記 3 discriminant × `ratio ∈ {0.1, 0.25, 0.4}` の cross product でも round-trip 成功（隠れていた validation_ratio 同期欠落も同時解消）
3. **legacy YAML 互換**: `{"validation_ratio": 0.1}` 単独入力で `inner_valid` が `Holdout(ratio=0.1)` に正規化され、`_inner_valid_explicit=False`（auto-resolve 経路維持）
4. **明示的 inner_valid**: `{"inner_valid": {"method": "group_holdout", "ratio": 0.2}}` 入力で `_inner_valid_explicit=True`、`es.validation_ratio == 0.2`
5. **両方併記の整合性ガード**: `{"inner_valid": {ratio: 0.1}, "validation_ratio": 0.25}` は `ValidationError`（不整合の検知は維持）
6. **下流読み取り正常性**: `cfg.training.early_stopping.validation_ratio` が `inner_valid.ratio` と一致（computed）
7. **persistence 互換**: 旧 `validation_ratio: 0.1` を含む `metadata.json` から `Model.load()` 成功（パラメトライズ: 3 discriminant）
8. **codegen 整合**: `inner_valid={method: holdout, ratio: 0.25}` で fit したモデルを `export_code` した `train.py` が `validation_ratio=0.25` で動作（codegen silent bug の同時修正確認）
9. **auto-resolve 維持**: `validation_ratio: 0.1` + `split.method=group_kfold` の組み合わせで factory が `GroupHoldoutInnerValid` を返す（既存挙動）
10. **品質ゲート**: ruff / mypy / pytest 全 PASS、既存 1320+ テストが PASS

---

## H-0070: 非数値 Classification Target の自動エンコード（TargetEncoder 導入）

- **ステータス**: Accepted
- **起票日**: 2026-05-04
- **スコープ**: Public API | Foundation 型 | Data 層 | Persistence Format | Codegen
- **関連**: [Issue #98](https://github.com/nbx-liz/LizyML/issues/98), LizyStudio 観測（penguins.csv multiclass / target=species で `ValueError`）

### 目的

`task ∈ {binary, multiclass}` で y が非数値（object / str / `pd.StringDtype` / category-with-string-categories / bool）のとき、`Model.fit()` が LightGBM 層 (`_check_for_bad_pandas_dtypes`) で `ValueError: pandas dtypes must be int, float or bool` を出して落ちる。回避するにはユーザーが手動で `LabelEncoder` / `pd.factorize` する必要があり、`predict` 出力は整数コードのまま元ラベル（例 `"Adelie"`）への inverse 手段が無い。

これは ML ライブラリとしての基本機能ギャップであり、本 Proposal は task 駆動で y を自動エンコードし、`FitResult.target_encoder` 経由で predict / inference / codegen が元ラベルへ inverse_transform できる経路を整備する。同時に `task=regression` × 非数値 y の早期 reject も加える（現状は不明瞭エラーで死ぬ）。

### 変更内容

1. **Foundation: `TargetEncoder` 契約型新設** (`lizyml/core/types/target_encoder.py`)
   - `@dataclass(frozen=True) TargetEncoder { classes_: tuple[Any, ...], needs_encoding: bool, original_dtype: str }`
   - `TargetEncoder.fit(y, task) -> TargetEncoder`: regression / 数値 y は no-op、非数値 classification y は `pd.factorize`-equivalent
   - `transform(y) -> pd.Series` / `inverse_transform(codes) -> np.ndarray`
   - `TargetEncoder.no_op()` クラスメソッド: 旧 artifact migration / 数値 y 用の sentinel
   - 全カテゴリが Foundation 経由で参照可能（DAG 違反なし）

2. **`ErrorCode` 拡張** (`lizyml/core/exceptions.py`)
   - `TARGET_NOT_NUMERIC` — task=regression × 非数値 y を fit 開始前に reject
   - `TARGET_UNSEEN_LABEL` — 将来の explicit_classes 経路用ガード（v1 では fit 時の不変式チェックに使用）

3. **`FitResult` 拡張** (`lizyml/core/types/fit_result.py`)
   - `target_encoder: TargetEncoder = field(default_factory=TargetEncoder.no_op)` 追加
   - 数値 y / 旧 artifact では `needs_encoding=False` の sentinel が入るので consumer 側の分岐は最小

4. **Data 層改修** (`lizyml/data/dataframe_builder.py`)
   - `DataFrameComponents` に `target_encoder: TargetEncoder` 追加
   - `build()` シグネチャに `task: TaskType` 追加
   - 非数値 classification y → `TargetEncoder.fit` → `transform` 適用
   - regression × 非数値 y → `TARGET_NOT_NUMERIC` を fit 開始前に raise
   - **影響閉じ込め**: training/ / estimators/ / calibration/ は引き続き int y を見るのみで無変更

5. **Facade 配線** (`lizyml/core/model.py`)
   - `_prepare_training_data` → builder に `task=cfg.task` を渡し、components から encoder を受け取る
   - CVTrainer.fit 後に `dataclasses.replace(fit_result, target_encoder=encoder)` で注入
   - `predict()` の binary / multiclass 分岐で `pred = fit.target_encoder.inverse_transform(pred_codes)`
   - tune() 経路も同じ `_prepare_training_data` を使うため自動対応

6. **Persistence migration** (`lizyml/persistence/exporter.py`, `loader.py`)
   - `FORMAT_VERSION` を 1 → 2 に bump
   - loader: v1 metadata 検出時に no-op encoder を注入して FitResult を再構成（`target_encoder` フィールドが pickle に存在しなければ default 適用、joblib の dataclass デフォルト値で自動補填されるが、明示的に v1→v2 migration ルートを通す）

7. **Codegen 拡張** (`lizyml/codegen/templates.py`, `config_writer.py`)
   - 非数値 classification target の場合、predict.py に `_CLASSES = (...)` 定数 + `_decode(codes) -> np.ndarray` ヘルパーを emit
   - config.json に `target_encoder.classes_` を書き出し
   - 数値 target / regression は従来出力と完全互換

### 影響範囲

- **新規ファイル**: `lizyml/core/types/target_encoder.py`、関連テスト
- **変更ファイル**:
  - Foundation: `core/types/{__init__,fit_result}.py`, `core/exceptions.py`
  - Data: `data/dataframe_builder.py`
  - Facade: `core/model.py`
  - Persistence: `persistence/{exporter,loader}.py`
  - Codegen: `codegen/{templates,config_writer}.py`
- **無変更（疎結合維持）**: `splitters/` / `features/` / `estimators/` / `calibration/` / `metrics/` / `training/` / `evaluation/` / `tuning/`

### 不変条件 (Invariants-First)

| ID | Invariant |
|---|---|
| INV-1 | `FitResult.target_encoder.needs_encoding=True` ⇔ 元 y が非数値（fit 時点） |
| INV-2 | `predict().pred.dtype == 元 y dtype`（str → str, int → int, category → category） |
| INV-3 | `target_encoder.classes_` は sorted（`key=str`）かつ frozen。`classes_[i]` の `i` が int code |
| INV-4 | task=regression × 非数値 y → fit 開始前に `TARGET_NOT_NUMERIC` raise（LightGBM 層に到達しない） |
| INV-5 | format_version=1 artifact ロード時に no-op encoder が注入され、predict 挙動が v0.x と等価（数値 target の round-trip） |

### 互換性

- **既存ユーザー API**: 数値 y の fit/predict/save/load は完全互換（FitResult の追加フィールドは default 値）
- **predict 出力 dtype**: 非数値 classification の場合のみ `pred.dtype` が int → 元 dtype に変化（**新挙動**）。CHANGELOG で告知。下流（LizyStudio / Widget）は `pd.api.types.is_numeric_dtype(pred)` 分岐で吸収可能
- **format_version**: 1 → 2 に bump、loader が v1 を migration で受理
- **proba 列順契約**: 多クラス proba の列順は `target_encoder.classes_` の順（数値 y 互換のため、numeric 時は sorted 数値順 = 既存挙動と一致）

### 代替案

- **A. ユーザー手動エンコード（status quo）**: 全コンシューマがバラバラの前処理を実装。最悪の UX
- **B. Validate-and-reject のみ**: 非数値 y を弾いて clear エラーを出す。実装最小だが本質解決にならず、ユーザーは自前 LabelEncoder + inverse 経路を組む必要が残る
- **C. sklearn `LabelEncoder` 直接利用**: 標準だが `LizyMLError` 契約に合わせる薄い wrapper が必要、`classes_` numpy array が JSON 化を煩雑にする
- **D. 自前 `TargetEncoder` dataclass（本 Proposal）**: 例外契約整合・`frozen` で immutable・JSON 化容易・原 dtype 復元可能 → **採用**

### 受け入れ基準（テスト観点）

1. **Foundation 単体**: `TargetEncoder.fit(y, task)` の no-op / 非数値 / regression 透過パターン、`transform` / `inverse_transform` の round-trip（`tests/test_core/test_target_encoder.py`）
2. **regression reject**: `task=regression` × str y で `TARGET_NOT_NUMERIC` を fit 開始前に raise（INV-4）
3. **Data 層統合**: `dataframe_builder.build(df, ps, fs, task)` が encoder を返し、y が int 化される
4. **binary E2E**: 2 クラス str y（例 `["yes", "no"]`）で fit → predict 成功、`pred.dtype == y.dtype`、proba 列順が `classes_` 整合（INV-1, INV-2, INV-3）
5. **multiclass E2E**: 3 クラス str y（penguins-like）で fit → predict 成功、同上
6. **calibration**: binary + isotonic + str y で fit/predict/calibrated_oof 成功（int y 経路を通る確認）
7. **tune→fit→predict**: tune 経路も非数値 y で動く
8. **persistence 互換**: format_version=1 artifact (旧 fixture) を load して数値 y 用 predict 経路が等価動作（INV-5）
9. **codegen E2E**: 非数値 classification で `export_code` → 別プロセスで `predict.py` 実行 → 元ラベルが復元される
10. **品質ゲート**: ruff / mypy / pytest 全 PASS、既存 1320+ テスト全 PASS

## H-0071: sMAPE / WAPE — zero-tolerant percentage-style 回帰メトリクスの追加

- **ステータス**: Accepted
- **起票日**: 2026-05-05
- **決定日**: 2026-05-05
- **スコープ**: Public API (Metrics) | LGBM metric bridge | Codegen feval | Docs
- **関連**: [Issue #101](https://github.com/nbx-liz/LizyML/issues/101), LizyStudio v0.4.0 GUI 検証で発覚（target に 0 を含む sales/demand 系 regression で MAPE が `UNSUPPORTED_METRIC`）

### 目的

回帰メトリクス集合（`rmse`, `mae`, `r2`, `rmsle`, `mape`, `huber`）には **zero-tolerant な percentage-style 指標が無い**。`MAPE` は `y_true` に 0 を含むと `LizyMLError(UNSUPPORTED_METRIC)` を raise する仕様（[regression.py:152-157](lizyml/metrics/regression.py#L152-L157)）であり、これは数学的には正しいが、0 が valid 値となる sales / demand / count 回帰では percentage 系の代替が存在しない。

本 Proposal は **sMAPE**（symmetric MAPE）と **WAPE**（weighted absolute percentage error）の 2 指標を追加し、既存契約を破壊せずにギャップを埋める。LizyStudio 側でも「計算不能 metric の auto-disable warning」を companion Issue で別途対処予定だが、上流で tolerant な代替を提供するのが本質的な解。

### 変更内容

1. **新規 metric クラス 2 件** (`lizyml/metrics/regression.py`)
   - `@MetricRegistry.register("smape")` → `class SMAPE(BaseMetric)`
     - 式: `mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|)) * 100`、range `[0, 200]`（doubled 形）
     - `y_true == y_pred == 0` の行は `0/0` を 0 として扱う（perfect prediction 規約）
     - `greater_is_better=False`, `needs_proba=False`
   - `@MetricRegistry.register("wape")` → `class WAPE(BaseMetric)`
     - 式: `sum(|y_true - y_pred|) / sum(|y_true|) * 100`（= `MAE / mean(|y_true|) * 100`）
     - `sum(|y_true|) == 0` のときのみ `LizyMLError(UNSUPPORTED_METRIC, "WAPE is undefined when sum(|y_true|) is zero.")`
     - `greater_is_better=False`, `needs_proba=False`

2. **Registry 拡張** (`lizyml/metrics/registry.py:26`)
   - `_TASK_METRICS["regression"]` の frozenset に `"smape"`, `"wape"` を追加

3. **LGBM feval Bridge 連携** (`lizyml/estimators/lgbm/metric_bridge.py`)
   - LightGBM ネイティブに sMAPE / WAPE は無いため、`_FEVAL_METRICS["regression"]` 既存の `frozenset(["rmsle", "r2"])` に `"smape"`, `"wape"` を追加
   - `_build_feval()` は `BaseMetric` インスタンスを受けて feval callable を生成する汎用機構なので追加実装不要（H-0064 の設計を継承）
   - これにより `params={"metric": "smape"}` で early stopping / learning curve への接続が自動的に有効化される

4. **Re-export** (`lizyml/metrics/__init__.py`)
   - `from .regression import SMAPE, WAPE` 追加（既存パターン踏襲）

5. **Codegen 対応** (`lizyml/codegen/templates.py`)
   - Codegen の `_FEVAL_REGISTRY` は metric_bridge と独立した手書き dict のため、`_feval_smape` / `_feval_wape` 関数と registry エントリを追記
   - 関数定義は metric 本体と同一の数式を再現（`np.errstate` で 0/0 を 0 扱い、`sum(|y|)==0` で `ValueError`）
   - `tests/test_codegen/test_feval_codegen.py` にて lizyml 実装との数値等価性を rtol=1e-10 で検証

### 影響範囲

- **新規シンボル**: `lizyml.metrics.SMAPE`, `lizyml.metrics.WAPE`
- **変更ファイル**:
  - `lizyml/metrics/regression.py`（クラス 2 件追加）
  - `lizyml/metrics/registry.py`（task frozenset 拡張）
  - `lizyml/metrics/__init__.py`（re-export）
  - `lizyml/estimators/lgbm/metric_bridge.py`（feval 名 frozenset 拡張）
- **無変更**: BaseMetric IF / Evaluator / FeaturePipeline / Calibration / Persistence
- **Config 互換**: 既存の `eval_metrics: ["rmse", "mae"]` 等は完全互換、`smape`/`wape` を新規に列挙可能

### 不変条件 (Invariants-First)

| ID | Invariant |
|---|---|
| INV-1 | `SMAPE(y, y) == 0.0`（恒等予測でゼロ） |
| INV-2 | `SMAPE` 出力範囲 `[0, 200]`、`y_true == y_pred == 0` の行は寄与 0 |
| INV-3 | `WAPE(y, y) == 0.0`、`sum(|y_true|) == 0` でのみ `UNSUPPORTED_METRIC` raise（per-row 0 では raise しない） |
| INV-4 | `MetricRegistry.get("smape", "regression")` と `("wape", "regression")` が成功、binary / multiclass では `LizyMLError(METRIC_NOT_FOUND)`（H-0111 注記: `METRIC_NOT_FOUND` という `ErrorCode` は無く、レジストリは `UNSUPPORTED_METRIC` を送出する。#318） |
| INV-5 | `params={"metric": "smape"}` で `lgb.train` の `eval_results` に `smape` キーが現れる（feval 経由）|
| INV-6 | `greater_is_better=False`、`needs_proba=False`（両 metric 共通） |

### 互換性

- **後方互換**: 完全互換。既存 metric / Config / FitResult / Persistence に変更なし
- **format_version**: bump 不要（artifact 構造に変更なし）
- **LightGBM 依存**: feval 経由で実装するため LightGBM のバージョン要件に変更なし

### 代替案

- **A. 何もしない（status quo）** — 0 を含む regression データで percentage 系評価が不可能。問題本体の放置
- **B. MAPE の挙動緩和（`y_true == 0` を skip）** — 数学的に MAPE の定義を歪める。既存ユーザーへの silent な挙動変化、リグレッションリスク高 → 不採用
- **C. sMAPE のみ追加** — sMAPE は per-row 平均、WAPE は sum 比であり用途が異なる（imbalanced magnitude regression では WAPE の方が頑健）。Issue 起案者も両方要請 → 不採用
- **D. sMAPE + WAPE 同時追加（本 Proposal）** — 既存 metric IF に乗るだけ、追加コスト最小、用途分離明確 → **採用**

### 受け入れ基準（テスト観点）

1. **基本正解性** (`tests/test_metrics/test_regression.py`):
   - sMAPE: hand-computed example（例: y=[1,2,3], y_pred=[1,2,4] → 期待値を手計算で固定）
   - WAPE: hand-computed example、`MAE / mean(|y_true|) * 100` との一致
2. **エッジケース**:
   - sMAPE: `y_true = [0, 1]`, `y_pred = [0, 1]` で 0.0（INV-2）
   - sMAPE: `y_true = [0, 5]`, `y_pred = [0, 4]` で raise しない（MAPE 対比）
   - WAPE: `y_true = [0, 0, 0]`, `y_pred = [0, 1, 2]` で `UNSUPPORTED_METRIC` raise（INV-3）
   - WAPE: `y_true = [0, 1, 2]`（部分的 0）で raise しない
3. **属性契約**: `name`, `greater_is_better`, `needs_proba` の golden test（INV-6）
4. **Registry 統合**: `MetricRegistry.get("smape", "regression")` / `("wape", "regression")` 成功、binary 等では失敗（INV-4）
5. **LGBM feval E2E** (`tests/test_estimators/test_lgbm_feval.py` 拡張):
   - regression × `params={"metric": "smape"}` で fit 成功、`fit_result.history` に `smape` キーが現れる（INV-5）
   - 同上 `wape` パターン
   - `params={"metric": ["rmse", "smape", "wape"]}` の native + feval 混在
6. **Codegen E2E** (`tests/test_codegen/`):
   - regression + smape + wape を `eval_metrics` に含む config で `export_code` → 別プロセスで `predict.py` 実行 → equivalence 確認（feval bridge 継承の確認）
7. **CHANGELOG**: `Added` セクションに sMAPE / WAPE 追加を記載
8. **Docstring**: 各クラスに式・range・edge case 規約・MAPE との使い分け方針を明記
9. **品質ゲート**: ruff / mypy / pytest 全 PASS、既存 1320+ テスト全 PASS

### 参考実装方針（informational）

```python
# lizyml/metrics/regression.py（追加）

@MetricRegistry.register("smape")
class SMAPE(BaseMetric):
    """Symmetric Mean Absolute Percentage Error (range [0, 200])."""

    @property
    def name(self) -> str: return "smape"
    @property
    def needs_proba(self) -> bool: return False
    @property
    def greater_is_better(self) -> bool: return False

    def __call__(self, y_true, y_pred) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        denom = np.abs(y_true) + np.abs(y_pred)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(denom == 0, 0.0, 2 * np.abs(y_true - y_pred) / denom)
        return float(np.mean(terms) * 100)


@MetricRegistry.register("wape")
class WAPE(BaseMetric):
    """Weighted Absolute Percentage Error (= MAE / mean(|y_true|) * 100)."""

    @property
    def name(self) -> str: return "wape"
    @property
    def needs_proba(self) -> bool: return False
    @property
    def greater_is_better(self) -> bool: return False

    def __call__(self, y_true, y_pred) -> float:
        _validate_shapes(y_true, y_pred, self.name)
        denom = float(np.sum(np.abs(y_true)))
        if denom == 0.0:
            raise LizyMLError(
                code=ErrorCode.UNSUPPORTED_METRIC,
                user_message="WAPE is undefined when sum(|y_true|) is zero.",
                context={"metric": self.name},
            )
        return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)
```

## H-0072: Tuner / Model.tune に Optuna 永続化 storage を追加（resumable tuning）

- **ステータス**: Accepted
- **起票日**: 2026-05-06
- **決定日**: 2026-05-06
- **スコープ**: Public API (Tuner / Model.tune) | Tuning persistence | Docs
- **関連**: [Issue #105](https://github.com/nbx-liz/LizyML/issues/105), [LizyStudio#360](https://github.com/nbx-liz/LizyStudio/issues/360), BLUEPRINT.md §11.5（制約: 「study の永続化（RDB storage 等）は対象外」を改訂）

### 目的

LizyStudio v0.5 が要求する **24h+ Tune ジョブの再開可能性**（プロセス kill / サーバ再起動 / ネットワーク断 / ブラウザリロード後に最終 trial から resume）を実現するため、Optuna 標準の永続 storage（`JournalStorage` / `RDBStorage`）を `Tuner` / `Model.tune()` に薄く pass-through する。

H-0068 で既に `study=` 注入経路は導入済みだが、in-memory study はプロセス終了で消失するため、再アタッチ可能な **disk-backed study** を構築する手段が現状存在しない。本 Proposal は `storage` + `study_name` の 2 引数を追加し、Optuna 標準機能の薄い委譲として実装する（追加依存なし、Optuna 同梱機能）。

LizyStudio 側で trial loop を再実装することは、`enqueue_trial` / `progress_callback` / `round_number` / `prior_trials` / `expanded_dims` 等 H-0068 で導入済みの round 管理を二重化することになり保守上のアンチパターン。`Tuner.tune()` が study を構築する箇所で storage を受け取れるのが構造的に最適。

### 変更内容

1. **`Tuner.__init__()` に 2 引数を追加** (`lizyml/tuning/tuner.py:54-69`)

   ```python
   def __init__(
       self,
       dims: list[SearchDim],
       n_trials: int = 50,
       direction: Literal["minimize", "maximize"] = "minimize",
       timeout: float | None = None,
       seed: int = 42,
       *,
       progress_callback: TuneProgressCallback | None = None,
       storage: str | Any | None = None,    # NEW: optuna URL or BaseStorage
       study_name: str | None = None,        # NEW: study identifier (required when storage is given)
   ) -> None:
   ```

2. **`Tuner.tune()` の study 生成ロジック拡張** (`lizyml/tuning/tuner.py:115-117`)

   ```python
   if study is None:
       sampler = _optuna.samplers.TPESampler(seed=self.seed)
       if self.storage is None:
           study = _optuna.create_study(direction=self.direction, sampler=sampler)
       else:
           study = _optuna.create_study(
               direction=self.direction,
               sampler=sampler,
               storage=self.storage,
               study_name=self.study_name,
               load_if_exists=True,   # idempotent re-attach
           )
   ```

   - `storage is not None and study_name is None` → `LizyMLError(CONFIG_INVALID, "study_name is required when storage is provided.")`
   - 既存 `study=` 注入経路はそのまま（外部で構築済 study を渡す既存ユースケースは無変更）

3. **`Model.tune()` に同 2 引数を pass-through** (`lizyml/core/model.py:388-397`)

   ```python
   def tune(
       self,
       data: pd.DataFrame | None = None,
       *,
       resume: bool = False,
       n_trials: int | None = None,
       expand_boundary: bool | None = None,
       boundary_threshold: float = 0.05,
       progress_callback: TuneProgressCallback | None = None,
       storage: str | Any | None = None,    # NEW
       study_name: str | None = None,        # NEW
   ) -> TuningResult:
   ```

   - `Tuner(..., storage=storage, study_name=study_name)` に転送
   - `resume=True` と `storage=` の併用挙動: `self._study` が既に存在すれば従来通り（in-memory 続行）。`self._study is None` かつ `storage` 指定時は `load_if_exists=True` により journal から自動 resume（Studio の crash recovery シナリオ）

4. **trial 数カウントの整合**
   - `prior_trials = len(self._study.trials)` は study load 後に正しい値を返す（Optuna 仕様）
   - `round_number` は `Model._round_number` をベースに増分するため、journal resume 後の初回 `tune()` は round 1 から（journal 自体は round メタを持たない仕様）。round 履歴を永続化したい場合は別 Proposal で扱う

5. **BLUEPRINT.md §11.5 改訂**
   - 「Tuner は study オブジェクトの受け取り・返却に対応するが、study の永続化（RDB storage 等）は対象外」を削除
   - 新節 §11.5.x「Persistent Storage（H-0072）」を追加し、`storage` / `study_name` の引数仕様 / resume パターン / round 履歴非保証 を明記

6. **Docs（README または docs/tuning-resume.md）**
   - 最小サンプル: `Model.tune(storage="sqlite:///workspace/tune.db", study_name="job-42")` を 2 回呼び出し → 2 回目は途中再開
   - JournalStorage URL 形式の注意（`journal:///` は Optuna ≥ 3.x で利用可、SQLite は `sqlite:///`）

### 影響範囲

- **変更ファイル**:
  - `lizyml/tuning/tuner.py`（`__init__` 引数 2 件追加 + `create_study` 分岐）
  - `lizyml/core/model.py`（`tune()` 引数 2 件追加 + Tuner 構築時に転送）
  - `BLUEPRINT.md` §11.5 制約改訂 + 新節
  - `tests/test_tuning/test_tuner_persistence.py`（新規）
  - `tests/test_core/test_model_tune.py`（pass-through 引数のテスト追記）
  - README.md または docs/tuning-resume.md（resume 例）
  - `CHANGELOG.md`（Added セクション）
- **無変更**: `TuningResult` / `TrialResult` / `RoundSummary` / `BoundaryReport` / `progress_callback` / Persistence (Artifacts) / Calibration / Codegen
- **依存追加**: なし（Optuna 同梱の `JournalStorage` / `RDBStorage` をそのまま利用）。`sqlite:///` は標準 Python（追加 install 不要）。`mysql://` 等は利用者側で driver を入れる前提（Optuna 同様）

### 不変条件 (Invariants-First)

| ID | Invariant |
|---|---|
| INV-1 | `storage=None`（デフォルト）で挙動が H-0071 までと完全一致（in-memory study、disk IO ゼロ） |
| INV-2 | `storage=<url>` + `study_name=<name>` で trial 完了直後に journal/DB に追記され、process kill 後でもファイル/DB に N trials 残る |
| INV-3 | 同 storage + 同 study_name で `Model.tune()` 再呼び出し時、`load_if_exists=True` により完了済 trial を再実行せず resume（`len(study.trials)` が単調増加） |
| INV-4 | `storage` 指定 + `study_name=None` → `LizyMLError(CONFIG_INVALID)`（fail fast） |
| INV-5 | `storage` を後から変更した場合（同 process 内で異なる storage で 2 回目 tune）→ `study_name` が同一なら別 storage の trial と混ざらない（Optuna 仕様準拠） |
| INV-6 | `progress_callback` / `enqueue_params` / `expanded_dims` / `round_number` の挙動は `storage` 有無に依存しない |

### 互換性

- **後方互換**: 完全互換。`storage=None` がデフォルトで、現在の全テスト・全ユースケースに影響なし
- **format_version**: bump 不要（Artifact / FitResult / TuningResult 構造に変更なし）
- **Optuna バージョン要件**: `JournalStorage` の API は Optuna 3.0+。LizyML 既存の Optuna 依存範囲（`pyproject.toml` 確認）を満たす場合は追加制約なし。利用者が古い Optuna を使う場合は SQLite (`sqlite:///`) でフォールバック可能

### 代替案

- **A. 何もしない（status quo）** — LizyStudio v0.5 の crash recovery が実装不能。LizyStudio 側で `Tuner` をバイパスして optuna 直叩きする迂回が必要になり、`progress_callback` / round 管理を二重実装する保守地獄 → 不採用
- **B. LizyStudio 側で `study=` を毎回外部構築** — `study` を毎回外で作って渡す方式は、LizyStudio が `Tuner.tune()` の objective closure 構築に必要な `_build_train_components` 等の internal にアクセスする必要があり、private API への依存を強要する → 不採用
- **C. `Tuner` ではなく `Model` 側で storage を持つ** — `Model._study` の永続化責務は Model に持たせる案。しかし `Tuner` が study を生成する責務（`Tuner.tune()` 内 `create_study`）と分離されており、Model 側で持つと「Model が study を作って Tuner に渡す」逆転構造になる。現在の責務分離を保つほうが H-0068 設計と整合 → 不採用
- **D. `storage` + `study_name` を Tuner / Model.tune に追加（本 Proposal）** — Optuna 標準機能の薄い委譲。コード変更最小、責務分離維持、後方互換 100% → **採用**
- **E. round 履歴の永続化も同時に行う** — `RoundSummary` を journal に保存する拡張は別の管理レイヤー（Optuna study の system_attrs に詰める or 独立ファイル）が必要で、本 Issue のスコープ（trial 単位 resume）を超える → 別 Proposal（H-0073 候補）

### 受け入れ基準（テスト観点）

1. **後方互換** (`tests/test_tuning/test_tuner.py` 既存 + 新規):
   - `storage=None` で Tuner.tune を実行し、in-memory study が生成されることを確認（`study._storage` が `InMemoryStorage`）
   - 既存 H-0068 resume テスト（`study=` 注入）が PASS のまま
2. **永続化 happy path** (`tests/test_tuning/test_tuner_persistence.py` 新規):
   - tmp_path 配下に SQLite (`sqlite:///{tmp}/study.db`) で Tuner.tune → `len(study.trials) == n_trials` 確認
   - 同一 storage + study_name で 2 回目 Tuner.tune → `len(study.trials) == n_trials * 2` 確認（resume 動作）
3. **crash-and-resume** (新規、INV-2/INV-3):
   - objective が `trial.number == K` で `RuntimeError` を raise する細工 → catch して study が壊れないことを確認
   - 別 Tuner インスタンスを構築（`storage` + `study_name` 同一）→ 残り trial 実行 → `len(study.trials) == n_trials` で resume 完了
4. **fail fast** (INV-4):
   - `Tuner(..., storage="sqlite:///x.db", study_name=None).tune(...)` → `LizyMLError(CONFIG_INVALID)`
   - 同上 `Model.tune(storage=..., study_name=None)` → 同エラー
5. **`Model.tune()` pass-through** (`tests/test_core/test_model_tune.py` 拡張):
   - `Model.tune(storage="sqlite:///{tmp}/study.db", study_name="m1")` で TuningResult が H-0071 までと同型で返る
   - 2 回目呼び出し（`resume=False` だが `_study is None`、storage 指定）で journal から再アタッチし、trial 数が累積する
6. **progress_callback 互換** (INV-6):
   - `storage` 有無にかかわらず `TuneProgressInfo` が同型で発火、`current_trial` / `cumulative_trials` の値が一致
7. **JournalStorage URL** (Optuna 3.x 利用可能なバージョンで): SQLite と JournalStorage（`JournalFileBackend`）の両方で 1〜3 のテストを parametrize（環境依存があれば JournalStorage は skip 可、SQLite は必須）
8. **Docs**: README に最小 resume 例 + `docs/tuning-resume.md`（または既存 docs に節追加）に LizyStudio crash recovery 想定の使い方を記述
9. **CHANGELOG**: `Added` セクションに「`Tuner` / `Model.tune()` に `storage` / `study_name` 引数を追加（Issue #105）」を記載
10. **品質ゲート**: ruff / mypy / pytest 全 PASS、既存 1320+ テスト全 PASS

### Migration

破壊的変更なし。利用者側の対応は **任意**:

- 従来通り `Model.tune()` を引数なしで呼べば in-memory のまま（変更不要）
- 永続化を有効にするには `Model.tune(storage="sqlite:///workspace/tune.db", study_name="<unique>")` に変更
- 同 study_name で再呼び出しすれば自動的に途中再開

### 参考実装方針（informational）

```python
# lizyml/tuning/tuner.py（差分）

class Tuner:
    def __init__(
        self,
        dims: list[SearchDim],
        n_trials: int = 50,
        direction: Literal["minimize", "maximize"] = "minimize",
        timeout: float | None = None,
        seed: int = 42,
        *,
        progress_callback: TuneProgressCallback | None = None,
        storage: str | Any | None = None,
        study_name: str | None = None,
    ) -> None:
        if storage is not None and study_name is None:
            raise LizyMLError(
                code=ErrorCode.CONFIG_INVALID,
                user_message="study_name is required when storage is provided.",
                context={"storage": str(storage)},
            )
        self.dims = dims
        self.n_trials = n_trials
        self.direction = direction
        self.timeout = timeout
        self.seed = seed
        self.progress_callback = progress_callback
        self.storage = storage
        self.study_name = study_name

    def tune(self, objective, metric_name="rmse", *, study=None, ...):
        ...
        if study is None:
            sampler = _optuna.samplers.TPESampler(seed=self.seed)
            if self.storage is None:
                study = _optuna.create_study(direction=self.direction, sampler=sampler)
            else:
                study = _optuna.create_study(
                    direction=self.direction,
                    sampler=sampler,
                    storage=self.storage,
                    study_name=self.study_name,
                    load_if_exists=True,
                )
        ...
```

### Decision

- Date: 2026-05-06
- Result: accepted
- Notes: Issue #105 に対する upstream 対応として承認。Optuna 標準機能の薄い委譲、後方互換 100%、追加依存なし。round 履歴の永続化は別 Proposal（H-0073 候補）で扱う。

## H-0073: EstimatorProvider に build_export_params() を追加し codegen 経路から LGBM 固有コードを除去

- **ステータス**: Accepted
- **起票日**: 2026-05-10
- **決定日**: 2026-05-10
- **スコープ**: Public Protocol (`EstimatorProvider`) | Internal API (`_model_persistence.py`, `_model_factories.py`)
- **関連**: [Issue #109](https://github.com/nbx-liz/issues/109) (CRITICAL), [Issue #126](https://github.com/nbx-liz/issues/126) (MEDIUM), BLUEPRINT.md §2.2 / §14.4

### 目的

H-0053（EstimatorProvider 導入）で公開された Provider 抽象が、`Model.export_code()` 経路で**たった 1 箇所だけ破られている**。`lizyml/core/_model_persistence.py:154-179` が `LGBMAdapter` を直接 `isinstance` チェックし、private な `_build_params()` を直接呼び出す形でコード生成用パラメータを取得しているため、新しい Estimator（XGBoost / sklearn 等）を追加する際に**必ずこの persistence 層を編集する必要があり、Provider 抽象化の意義を半減させている**（#109 CRITICAL）。

加えて `BlockedGroupKFoldConfig` の `n_splits` 解決ロジックが `_model_persistence.py:174-179` と `_model_factories.py:173-181` に重複している（#126 MEDIUM）。本 Proposal は両者を 1 つの構造変更にまとめて解消する。

### 変更内容

1. **Provider Protocol に `build_export_params()` を追加** (`lizyml/estimators/provider.py`)

   ```python
   class EstimatorProvider(Protocol):
       ...
       def build_export_params(self, adapter: Any) -> dict[str, Any]:
           """Return booster params suitable for codegen export.

           Used by Model.export_code() to emit a self-contained train.py
           that does not depend on LizyML at runtime."""
   ```

2. **`LGBMProvider.build_export_params()` を実装** (`lizyml/estimators/lgbm/provider.py`)

   ```python
   def build_export_params(self, adapter: LGBMAdapter) -> dict[str, Any]:
       return adapter._build_params()  # private は LGBM パッケージ内で閉じる
   ```

   - `_build_params()` は `LGBMAdapter` の private のままで OK（同パッケージ内 access）
   - 将来 `XGBoostProvider` を追加するときも、同じ public method を実装するだけ

3. **`get_outer_n_splits(cfg) -> int` ユーティリティを `_model_factories.py` に追加**

   ```python
   def get_outer_n_splits(cfg: LizyMLConfig) -> int:
       if isinstance(cfg.split, BlockedGroupKFoldConfig):
           return cfg.split.groups.n_splits
       return cfg.split.n_splits
   ```

4. **`_model_persistence.py` 全面刷新**
   - `from lizyml.estimators.lgbm.adapter import LGBMAdapter` → 削除
   - `isinstance(adapter, LGBMAdapter)` → 削除
   - `adapter._build_params()` → `provider.build_export_params(adapter)` に置換
   - inline n_splits 解決 → `get_outer_n_splits(cfg)` 呼び出しに置換

### 影響範囲

- **変更ファイル**:
  - `lizyml/estimators/provider.py`（Protocol に 1 method 追加）
  - `lizyml/estimators/lgbm/provider.py`（実装追加）
  - `lizyml/core/_model_persistence.py`（LGBMAdapter import 削除 + dispatch 経由化）
  - `lizyml/core/_model_factories.py`（`get_outer_n_splits` 追加）
  - `tests/test_persistence/`（既存 73 codegen テストが pass することを確認、 isinstance ゼロを確認するテストを追加）
  - BLUEPRINT.md §14.4（Provider spec に `build_export_params` を追記）
- **無変更**: `Model` public API、`FitResult`、`PredictionResult`、Artifacts schema、Calibration、Tuning。

### 互換性

- **Provider Protocol への method 追加は破壊的**（既存の Provider 実装は新メソッドを実装する必要あり）。ただし現状 `LGBMProvider` のみ存在するため実害なし。
- `format_version` バンプ不要（Artifacts schema は無変更）。

### 代替案と不採用理由

| 代替案 | 不採用理由 |
|---|---|
| `LGBMAdapter._build_params()` を public 化 | LGBM 内部詳細をパッケージ外に漏らす。Provider 抽象の存在意義に反する |
| `_model_persistence.py` を Provider に丸ごと移管 | 過剰な責務移動。export 形式の決定は Facade 層の仕事 |
| `n_splits` 重複だけ解消し #109 は別 PR | #126 は #109 のサブセット。同じ refactor 経路で同時解消が効率的 |

### 受け入れ基準

- [ ] `grep -rn 'LGBMAdapter' lizyml/core/` → ゼロマッチ。
- [ ] `grep -rn 'isinstance.*LGBMAdapter' lizyml/core/` → ゼロマッチ。
- [ ] `grep -rn 'BlockedGroupKFoldConfig' lizyml/core/_model_persistence.py` → ゼロマッチ（factories へ集約）。
- [ ] 既存 codegen テスト（73 件）が無修正で pass。
- [ ] `EstimatorProvider.build_export_params` の契約テストを追加（呼び出すと `dict[str, Any]` が返る）。
- [ ] BLUEPRINT.md §14.4 に新メソッドが記載されている。

### Migration

- 外部の Provider 実装者向け（現状ゼロだが将来用）：`build_export_params(adapter) -> dict` を実装する必要がある。
- LizyML 利用者には影響なし（内部 refactor）。

### Decision

- Date: 2026-05-10
- Result: accepted
- Notes: コードレビューで CRITICAL とマークされた #109 + その subset の #126 を一括解消する内部 refactor。Provider 抽象化の整合性を取り戻すことで XGBoost / sklearn 拡張の素地を整える。

---

## H-0074: FitState frozen dataclass を導入し Mixin の private state 直接参照を解消

- **ステータス**: Accepted
- **起票日**: 2026-05-10
- **決定日**: 2026-05-10
- **スコープ**: Internal API (Mixin classes), Testability
- **関連**: [Issue #112](https://github.com/nbx-liz/issues/112) (HIGH)

### 目的

H-0042 で `model.py` を行数削減するため `ModelPlotsMixin` / `ModelTablesMixin` / `ModelPersistenceMixin` を切り出したが、各 Mixin が `self._y`、`self._X`、`self._cfg`、`self._provider`、`self._tuning_result`、`self._metrics`、`self._run_dir`、`self._output_dir` 等を**直接読み書きしている**。結果として:

1. **責務分離が見かけだけ**：Mixin 単独でユニットテスト不可（fit 済み Model が必要）
2. **silent breakage**：Model の private 属性をリネームすると Mixin が**型チェックなしに壊れる**（mypy は `if TYPE_CHECKING` ブロックで属性宣言される範囲しか追えない）
3. **入力契約が暗黙**：各 Mixin が必要とする state が分散しており、API として不明瞭

本 Proposal は **`FitState`（frozen dataclass）を中継層に導入**し、Mixin が必要な値を明示的に受け取る形に変える。

### 変更内容

1. **`FitState` を新規追加** (`lizyml/core/types/fit_state.py`)

   ```python
   from dataclasses import dataclass
   from pathlib import Path
   from lizyml.config.schema import LizyMLConfig
   from lizyml.core.types.fit_result import FitResult
   from lizyml.core.types.tuning_result import TuningResult
   from lizyml.estimators.provider import EstimatorProvider
   from lizyml.training.refit_trainer import RefitResult

   @dataclass(frozen=True)
   class FitState:
       """Snapshot of Model fit state consumed by Mixin methods (#112).

       Created by ``Model._get_fit_state()`` after fit/tune. Mixin methods
       receive this instead of reading ``self._*`` directly.
       """
       cfg: LizyMLConfig
       fit_result: FitResult
       refit_result: RefitResult | None
       tuning_result: TuningResult | None
       provider: EstimatorProvider
       output_dir: Path | None
       run_dir: Path | None
       y: pd.Series | None    # transient; absent after Model.load() w/o analysis_context
       X: pd.DataFrame | None
       metrics: dict[str, Any] | None
   ```

2. **`Model._get_fit_state() -> FitState` を追加**
   - 内部の `self._*` を 1 箇所に集約。Mixin 経由の参照は全てこれを通る。
   - 失敗時（`fit_result is None`）は既存の `_require_fit()` と同じ `LizyMLError` を raise。

3. **Mixin signatures を `state: FitState` 受け取りに変更**
   - 例: `ModelPlotsMixin.calibration_plot(self, *, state: FitState | None = None)` で渡されない場合は `self._get_fit_state()` 経由でフォールバック（破壊的変更回避）
   - 段階的移行：Phase 1 = `FitState` 経由でも `self._*` 経由でも動く / Phase 2 = `self._*` 直接 access を削除

4. **Mixin の単体テスト追加**
   - `tests/test_core/test_model_plots_mixin.py`：`FitState` を mock で組み立てて Mixin 単体で動作確認

### 影響範囲

- **変更ファイル**:
  - `lizyml/core/types/fit_state.py`（新規）
  - `lizyml/core/_model_plots.py`、`_model_tables.py`、`_model_persistence.py`（Mixin signatures + body 更新）
  - `lizyml/core/model.py`（`_get_fit_state()` 追加 + Mixin 呼び出し経路調整）
  - `tests/test_core/test_model_plots_mixin.py`（新規）
  - BLUEPRINT.md §3（責務分離の記述更新）
- **無変更**: 公開 API、Persistence 形式、Tuning、Plots 出力、Tables 形式。

### 互換性

- **公開 API 完全互換**。Model のメソッド名・シグネチャ・戻り値は変更なし。
- 内部 attribute (`self._cfg`, `self._fit_result`, ...) も維持（FitState はその snapshot）。

### 代替案と不採用理由

| 代替案 | 不採用理由 |
|---|---|
| Mixin を継承から composition に変更 | 既存の継承階層（Model(ModelPlotsMixin, ...)）が public 型表面なので破壊的 |
| `Protocol` で Model attrs を宣言 | mypy は通るが runtime 結合は変わらず、テスト可能性が改善しない |
| Phase 2（self._* 直接 access 削除）を同 PR で実施 | refactor 量が大きく review 負荷が跳ね上がる。段階的に進める |

### 受け入れ基準

- [ ] `FitState` frozen dataclass が定義されている。
- [ ] `Model._get_fit_state()` が動作し、既存テスト（特に `test_model_facade.py`）が pass。
- [ ] 各 Mixin に少なくとも 1 件の **mock FitState を使った単体テスト** を追加。
- [ ] 既存の Mixin tests がすべて pass。
- [ ] mypy --strict が pass。

### Migration

- ユーザーには影響なし。
- 拡張開発者向け：将来的に Mixin 内部から `self._*` 直接 access を削除する Phase 2 PR を予告（H-0074-Phase2 として別 Proposal）。

### Decision

- Date: 2026-05-10
- Result: accepted
- Notes: Mixin の責務分離を実態化する構造改善。Phase 1 では後方互換を保つ二重経路を許容し、Phase 2 で完全移行する段階的アプローチを承認。

---

## H-0075: TaskType Literal を全分岐サイトに伝搬し dispatch dict 化

- **ステータス**: Accepted
- **起票日**: 2026-05-10
- **決定日**: 2026-05-10
- **スコープ**: Internal type annotations, Code organisation
- **関連**: [Issue #122](https://github.com/nbx-liz/issues/122) (MEDIUM)

### 目的

`TaskType = Literal["regression", "binary", "multiclass"]` は既に `core/types/target_encoder.py` で定義されているが、コードベース内に `if task == "regression"` / `elif task == "binary"` 形式の分岐が **6 ファイル以上に散在**している。新しい task type（例: "ranking"、"multilabel"）を追加する際、全箇所を手で grep して書き換える必要があり、**漏れによる silent bug のリスクが高い**。

### 変更内容

1. **`TaskType` を全分岐サイトに type annotation として伝搬**

   対象ファイル（grep で確認）:
   - `lizyml/core/model.py:349` 等
   - `lizyml/evaluation/evaluator.py:57,59`
   - `lizyml/estimators/lgbm/metric_bridge.py:239,241`
   - `lizyml/core/_model_factories.py:55-62` (`_resolve_stratify`)
   - `lizyml/core/types/target_encoder.py`（既に使用済）
   - 他 grep で発見された箇所

2. **形が同じ `if/elif` チェーンを dispatch dict に置換**

   ```python
   # before
   if task == "regression":
       handler = handle_regression
   elif task == "binary":
       handler = handle_binary
   else:  # multiclass
       handler = handle_multiclass

   # after
   _TASK_DISPATCH: dict[TaskType, Callable[..., R]] = {
       "regression": handle_regression,
       "binary": handle_binary,
       "multiclass": handle_multiclass,
   }
   handler = _TASK_DISPATCH[task]
   ```

3. **特に metric_bridge.py / evaluator.py は確実に dispatch dict 化**
   - 同形分岐が 3 つ以上ある site のみが対象（`if-elif-else` を残す価値の低いケース）

4. **task が "regression" / "binary" / "multiclass" 以外を取れる箇所を grep で網羅し、エラーパスは `LizyMLError(UNSUPPORTED_TASK)` に統一**

### 影響範囲

- **変更ファイル**:
  - `lizyml/core/model.py`、`evaluation/evaluator.py`、`estimators/lgbm/metric_bridge.py`、`core/_model_factories.py` 等
  - `tests/test_*` 全般（既存 task 別パラメトリックテストは影響なし）
- **無変更**: 公開 API、戻り値の意味、metric の挙動、`TaskType` の値そのもの。

### 互換性

- **公開 API 完全互換**。`task` 文字列の値は変えない。
- 型注釈強化と内部リファクタのみ。

### 代替案と不採用理由

| 代替案 | 不採用理由 |
|---|---|
| `Task` Enum 化 | `TaskType` は外部 Config（YAML/JSON）から `str` で来るため Enum 化は変換コストを増やす。`Literal` のままが pydantic と相性が良い |
| `functools.singledispatch` | task は型ではなく value。singledispatch は不適切 |
| 全 if/elif を dispatch dict 化 | 1〜2 分岐の小さい if/elif は可読性が落ちる。3 分岐以上だけが対象 |

### 受け入れ基準

- [ ] `metric_bridge.py` と `evaluator.py` が dispatch dict を使用している。
- [ ] 全分岐サイトで `task` の type annotation が `TaskType` または `str` ではなく `TaskType`（後者は禁止）。
- [ ] 既存テスト全件 pass。
- [ ] mypy --strict が `task: TaskType` を正しく narrowing する（dispatch dict 経由で全 key カバレッジを確認）。

### Migration

- ユーザー影響なし。
- 拡張開発者向け：新しい task type を追加するときは `TaskType` Literal を更新するだけで dispatch table が mypy エラーで網羅性を強制できる。

### Decision

- Date: 2026-05-10
- Result: accepted
- Notes: 内部 refactor。新 task 追加時の漏れリスク削減と、mypy による網羅性チェックを獲得する。

---

## H-0076: Deprecation Warning に削除目標バージョンを明記し中央レジストリ化

- **ステータス**: Accepted
- **起票日**: 2026-05-10
- **決定日**: 2026-05-10
- **スコープ**: Documentation, User-facing warnings
- **関連**: [Issue #120](https://github.com/nbx-liz/issues/120) (MEDIUM), [Issue #121](https://github.com/nbx-liz/issues/121) (MEDIUM), [H-0058](#h-0058)

### 目的

複数の `DeprecationWarning` / `UserWarning` で「いつ削除するか」が記述されておらず、ユーザーは migration の緊急度を判断できない。具体的には:

- `lizyml/config/schema.py:115` (`purge_window`)
- `lizyml/config/schema.py:123` (`embargo_pct`)
- `lizyml/config/schema.py:131` (`gap` for purged_time_series)
- `lizyml/config/schema.py:475,495` (`CalibrationConfig.n_splits`)
- `lizyml/core/_model_factories.py:196` (`build_calibration_splitter`)

加えて H-0058 で `build_calibration_splitter` を deprecated にして以来、**実際の削除計画が記録されていない**ためテスト側も具体挙動を依存してしまっている（#120）。

### 変更内容

1. **すべての deprecation 文言に "Will be removed in v1.0." を追記**

   ```python
   warnings.warn(
       "`purge_window` is deprecated; use `purge_gap` instead. "
       "Will be removed in v1.0.",
       DeprecationWarning,
       stacklevel=2,
   )
   ```

2. **削除目標を `docs/DEPRECATIONS.md` に集約**（新規）

   | 対象 | 代替 | 削除予定 | Deprecated since |
   |---|---|---|---|
   | `EarlyStoppingConfig.validation_ratio` | `inner_valid.ratio` | v1.0 | H-0069 (2026-04) |
   | `CalibrationConfig.n_splits` | （outer split を再利用） | v1.0 | H-0058 (2026-04) |
   | `purge_window` | `purge_gap` | v1.0 | H-0021 |
   | `embargo_pct` | `embargo` | v1.0 | H-0021 |
   | `build_calibration_splitter()` | （内部実装で OOF split を再利用） | v1.0 | H-0058 |
   | `gap` (purged_time_series) | `purge_gap` | v1.0 | H-0021 |

3. **deprecation テストの整理**
   - `build_calibration_splitter` の動作に依存する既存テストを `pytest.warns(DeprecationWarning)` チェックに変更（実装非依存化）。
   - 削除時に最小コミットで除去できる状態にする。

4. **CI で `pytest.warns(DeprecationWarning, match="Will be removed in v")` を強制するテストを追加**
   - 全 deprecation メッセージに削除予定が含まれることをリグレッションテストで保証。

### 影響範囲

- **変更ファイル**:
  - `lizyml/config/schema.py`（deprecation 文言更新）
  - `lizyml/core/_model_factories.py`（同）
  - `docs/DEPRECATIONS.md`（新規）
  - `tests/test_config/`、`tests/test_calibration/`（テスト整理）
- **無変更**: 削除そのもの（v1.0 リリースまでは互換維持）、public API、Artifacts。

### 互換性

- **完全互換**。文言追加のみ、挙動は変えない。
- 実際の削除（v1.0）は別 Proposal（H-XXXX-removal）で扱う。

### 代替案と不採用理由

| 代替案 | 不採用理由 |
|---|---|
| 削除を即実施 | 互換性ポリシー違反。v1.0 は別タイムライン |
| バージョン記述を `pyproject.toml` のメタに集約 | warning メッセージから乖離する。docs/DEPRECATIONS.md が docs と warn 両方の SSOT として機能する |
| Python `@deprecated` decorator 移行 | Python 3.13+ の機能。3.10 サポート期間中は採用不可 |

### 受け入れ基準

- [ ] `docs/DEPRECATIONS.md` が存在し、上記 6 項目を含む。
- [ ] `grep -E "DeprecationWarning|deprecated" lizyml/` の全 warning に "v1.0" が含まれる（CI test で強制）。
- [ ] `build_calibration_splitter` の挙動依存テストが `pytest.warns` ベースに置き換わっている。
- [ ] 既存テスト全件 pass。

### Migration

- ユーザー向け：v1.0 までは現状の deprecation を継続。v1.0 リリース時に対応 PR で完全削除。
- 内部開発：新規の deprecation を追加する際は必ず DEPRECATIONS.md と "Will be removed in vX.Y." 形式の文言を伴うこと（CONTRIBUTING.md / CLAUDE.md にルール追記）。

### Decision

- Date: 2026-05-10
- Result: accepted
- Notes: ユーザーへの予告と内部の削除計画を一元管理する非機能改善。実際の削除は v1.0 リリースの直前に別 PR で実施する。

## H-0077: H-0074 Phase 2 — Mixin から self._* 直接 access を排除

- **ステータス**: Accepted
- **起票日**: 2026-05-10
- **決定日**: 2026-05-10
- **スコープ**: Internal API (`_model_plots.py`, `_model_tables.py`, `_model_persistence.py`, `core/types/fit_state.py`, `core/model.py`)
- **関連**: [Issue #112](https://github.com/nbx-liz/issues/112) (HIGH), H-0074 (Phase 1)

### 目的

H-0074 Phase 1 で `FitState` frozen dataclass と `Model._get_fit_state()` を導入したが、Mixin (`ModelPlotsMixin` / `ModelTablesMixin` / `ModelPersistenceMixin`) はまだ `self._cfg`, `self._y`, `self._X`, `self._fit_result`, `self._tuning_result`, `self._provider`, `self._metrics`, `self._run_dir`, `self._output_dir` を直接参照している（合計 59 箇所）。Phase 2 として、Mixin の Model 本体への結合を `state: FitState` 経由のみに揃え、Issue #112 の Acceptance criteria（"Mixin methods access only `state.*` and method-local variables"）を満たす。

### 変更内容

1. **`TuningState` frozen dataclass を `core/types/fit_state.py` に追加**

   `tuning_plot` / `tuning_table` / `boundary_table` は `tune()` のみ呼ばれた `fit()` 前の状態でも動作する必要があるため（既存テスト `tests/test_tuning/test_tuning_result.py` / `tests/test_plots/test_tuning_plot.py` で確認済み）、`FitState` の不変条件「fit 後 snapshot」を維持しつつ別経路を提供する。

   ```python
   @dataclass(frozen=True)
   class TuningState:
       cfg: LizyMLConfig
       tuning_result: TuningResult  # not None — required for tuning APIs
   ```

2. **`Model._get_tuning_state() -> TuningState` を追加**

   `_tuning_result is None` のとき `LizyMLError(MODEL_NOT_FIT)` を raise する単一の入口。tuning 系 Mixin メソッドはこれを通る。

3. **Mixin 全メソッドの書き換え**

   各メソッド冒頭で `state = self._get_fit_state()`（または `_get_tuning_state()`）を呼び、以降 `state.cfg / state.fit_result / state.y / state.X / state.metrics / state.tuning_result / state.provider / state.run_dir / state.output_dir` のみで完結させる。`self._<attr>` の直接 access を Mixin 内から完全に排除。`TYPE_CHECKING` ブロックの attribute stub も削除。

4. **`_resolve_export_path()` を Model facade に移動**

   既存実装は `self._run_dir = setup_output_dir(...)` で書き戻しを行うため、frozen な `FitState` 経由では実現できない。書き込み責務を Model facade に残し、Mixin の `export()` は facade method を呼ぶ形にする。

5. **Mixin 単体テストの追加**

   `tests/test_core/test_mixin_state_isolation.py` (新規) で mock の `FitState` / `TuningState` を Mixin に渡し、Mixin が `state.*` のみを参照していることを実証する単体テストを追加する。Issue #112 Acceptance criteria の 2 番目を満たす。

### 影響範囲

- **変更ファイル**:
  - `lizyml/core/types/fit_state.py`（`TuningState` 追加）
  - `lizyml/core/model.py`（`_get_tuning_state()` + `_resolve_export_path()` 追加）
  - `lizyml/core/_model_plots.py`（`self._*` → `state.*`）
  - `lizyml/core/_model_tables.py`（同上）
  - `lizyml/core/_model_persistence.py`（同上 + `_resolve_export_path` を facade 委譲化）
  - `tests/test_core/test_mixin_state_isolation.py`（新規）
- **無変更**: 公開 API、Persistence 形式、Tuning 挙動、Plots 出力、Tables 形式、Config schema。

### 互換性

- **公開 API 完全互換**。Mixin メソッド signature・戻り値は不変。
- 内部 attribute (`self._cfg`, `self._fit_result`, `_provider` 等) は Model 本体に維持。`FitState` / `TuningState` はその snapshot。
- format_version 影響なし。

### 代替案と不採用理由

| 代替案 | 不採用理由 |
|---|---|
| `FitState` を `fit_result: FitResult \| None` に緩めて単一入口化 | "fit 後 snapshot" の不変条件が崩れ、すべての Mixin メソッドで null check が必要になる。型安全性が低下 |
| Mixin メソッドに `state: FitState \| None = None` 引数を追加（外部 inject 可能） | 公開 API 表面が増え、ユーザーが内部構造を知る誘因になる。テスト用の入口は内部 helper で十分 |
| Mixin を継承から composition に変更 | 既存の継承階層が public 型表面（`isinstance(model, ModelPlotsMixin)` 互換）。本 Issue のスコープ外 |
| Phase 2 を tuning 系除外で実施 | Issue #112 Acceptance criteria を完全に満たさない |

### 受け入れ基準

- [ ] `grep -nE "self\._(cfg|y|X|fit_result|refit_result|tuning_result|metrics|provider|run_dir|output_dir)" lizyml/core/_model_plots.py lizyml/core/_model_tables.py lizyml/core/_model_persistence.py` の出力が空。
- [ ] 各 Mixin の `TYPE_CHECKING` ブロックから Model attribute stub が削除されている（`_get_fit_state` / `_get_tuning_state` / `_resolve_export_path` の宣言のみ残す）。
- [ ] `tests/test_core/test_mixin_state_isolation.py` で mock state を使った単体テストが各 Mixin に存在し、pass する。
- [ ] 既存 1709 テスト全件 pass。
- [ ] `uv run ruff check .` / `uv run ruff format --check .` / `uv run mypy lizyml/` がクリーン。

### Migration

- ユーザーには影響なし（公開 API 互換）。
- 拡張開発者向け：以後、Mixin に新メソッドを追加する際は `self._<private>` ではなく `state = self._get_fit_state()` パターンに従うこと。

### Decision

- Date: 2026-05-10
- Result: accepted
- Notes: Phase 1 (H-0074) で予告した Phase 2 の実装。tuning 系メソッドの fit-前-tune-後 ケースを `TuningState` 別経路で扱うことで `FitState` の "fit 後 snapshot" 不変条件を維持。

---

## H-0078: 探索空間の検証強化と `EstimatorProvider.parameter_bounds()` 導入

- **ステータス**: Accepted
- **起票日**: 2026-05-10
- **決定日**: 2026-05-10
- **スコープ**: Public API (`EstimatorProvider`), Internal Types (`SearchDim`, `BoundaryDimStatus`), `tuning/search_space.py`, `core/model.py`
- **関連**: [Issue #152](https://github.com/nbx-liz/issues/152) (severity: high), LizyStudio Issue #460（下流 UI）, PR [#153](https://github.com/nbx-liz/LizyML/pull/153) / [#154](https://github.com/nbx-liz/LizyML/pull/154) / [#156](https://github.com/nbx-liz/LizyML/pull/156)

### 目的

Re-tune (`expand_boundary=True`) を繰り返した際、`expand_dims` がパラメータ別の意味境界を超えて探索範囲を拡大してしまう。`learning_rate.high` が 1.0 を超え、`feature_fraction.high` が 1.0 を超え、`validation_ratio.low` が 0.0 にまで縮退する。さらに `parse_space` は `low >= high` や `log=True ∧ low <= 0` のような不正値を受け入れ、Optuna の trial 時にようやく汚いエラーで失敗する。

これらは LizyStudio や CLI 利用者を含む全ての Tuning 利用者に影響する。本 Proposal では以下の 3 階層で対応する。

1. **Parse-time validation**: `parse_space` が `low<high`・`log+positive` を即時拒否（早期失敗）。
2. **Provider-supplied bounds**: `EstimatorProvider.parameter_bounds(task)` でパラメータ別の意味境界を表現できる API を新設。LightGBM 用は `LGBMProvider` が知る。
3. **Bounded expansion**: `_expand_range` を `min_allowed` / `max_allowed` 認識にし、`Model.tune` が provider→dim に bounds を注入する。`BoundaryDimStatus.clamped_to_bound` で UI が badge できる。

### 変更内容

#### Phase 1 — `parse_space` 検証強化

`lizyml/tuning/search_space.py::parse_space()` に以下のチェックを追加。

- `"float"` / `"int"`: `low < high` が満たされない場合 `LizyMLError(CONFIG_INVALID, ...)` を raise。
- `log=True`: `low > 0` が満たされない場合 `LizyMLError(CONFIG_INVALID, ...)` を raise（log distribution は正の下限を要求）。

エラー文言には `param`, `low`, `high`, `log` を context に含める。

#### Phase 2 — `parameter_bounds` API + bounds-aware `_expand_range`

1. **`EstimatorProvider.parameter_bounds(task)` 追加**

   ```python
   def parameter_bounds(self, task: TaskType) -> dict[str, dict[str, float | int]]:
       """Return per-parameter meaningful bounds. Empty dict = unbounded."""
       ...
   ```

   ベースの Protocol で宣言。デフォルト実装はないため、各 Provider が実装する（未対応 estimator は `{}` を返してよい）。

2. **`LGBMProvider.parameter_bounds(task)` 実装**

   LightGBM 既知の意味境界を返す（Issue #152 の表を出発点に LightGBM docs と integration test で確定）:

   ```python
   {
       "learning_rate":          {"min": 1e-8, "max": 1.0},
       "feature_fraction":       {"min": 1e-3, "max": 1.0},
       "bagging_fraction":       {"min": 1e-3, "max": 1.0},
       "num_leaves_ratio":       {"min": 0.1,  "max": 2.0},
       "min_data_in_leaf_ratio": {"min": 1e-4, "max": 0.5},
       "min_data_in_bin_ratio":  {"min": 1e-4, "max": 0.5},
       "validation_ratio":       {"min": 0.05, "max": 0.5},
       "lambda_l1":              {"min": 0.0,  "max": 100.0},
       "lambda_l2":              {"min": 0.0,  "max": 100.0},
       "n_estimators":           {"min": 10,   "max": 10000},
       "max_depth":              {"min": -1,   "max": 30},
       "max_bin":                {"min": 2,    "max": 8192},
       "bagging_freq":           {"min": 0,    "max": 100},
       "early_stopping_rounds":  {"min": 1,    "max": 5000},
       "seed":                   {"min": 0,    "max": 2**31 - 1},
   }
   ```

3. **`SearchDim` (FloatDim/IntDim) に optional フィールド追加**

   `min_allowed: float | int | None = None`, `max_allowed: float | int | None = None`。`@dataclass(frozen=True)` の不変条件と後方互換性を維持。

4. **`_expand_range` を bounds-aware に拡張**

   引数に `min_allowed` / `max_allowed` を追加し、計算された `new_low` / `new_high` を境界でクランプ。両側がぶつかった場合は元値のまま返す（無限ループ防止のため `expanded=False` を上位で再判定）。

5. **`BoundaryDimStatus.clamped_to_bound: bool` フィールド追加**

   `expand_dims` 経由で expansion が境界に当たった場合 True。下流 UI が "max reached" badge を出すためのフラグ。デフォルト False で後方互換。

#### Phase 3 — `Model.tune` 配線

`Model._maybe_expand_boundary()` で `provider.parameter_bounds(cfg.task)` を取得し、`detect_boundary` 呼び出し前に各 dim へ bounds を注入する（または `detect_boundary` の signature を bounds-aware にする）。`expand_boundary=False` 時は既存挙動と同一（bounds は無視される）。

`expand_dims` も bounds 認識にし、`new_low/new_high` のクランプ + `clamped_to_bound` セットを行う。

### 影響範囲

- **変更ファイル**:
  - `lizyml/tuning/search_space.py`（`parse_space` 検証 / `_expand_range` bounds 対応 / `detect_boundary` `expand_dims` への bounds 配線）
  - `lizyml/core/types/search_dim.py`（`min_allowed` / `max_allowed` 追加）
  - `lizyml/core/types/tuning_result.py`（`BoundaryDimStatus.clamped_to_bound` 追加）
  - `lizyml/estimators/provider.py`（Protocol に `parameter_bounds` 追加）
  - `lizyml/estimators/lgbm/provider.py`（実装追加）
  - `lizyml/core/model.py`（`_maybe_expand_boundary` で bounds 注入）
  - テスト: `tests/test_search_space/test_parse_space_validation.py`（新規）、`tests/test_search_space/test_expand_dims_clamp.py`（新規）、`tests/test_estimators/test_lgbm_provider.py`（追記）、`tests/test_core/test_model_tune_uses_bounds.py`（新規）
- **無変更**: Persistence 形式、Calibration、Plots/Tables 出力 shape、Codegen export、既存 happy-path tuning 挙動。

### 互換性

- **`parse_space`**: 不正値を受け入れていたコードは新たに `CONFIG_INVALID` で fail-fast。これは "later & messier" → "earlier & clearer" への振る舞い変更で、**実害のあるユーザーコードは存在しない**（Optuna が trial で raise していたため）。format_version 影響なし。
- **`SearchDim`**: 新 field は optional でデフォルト None。既存呼び出しは無変更で動作。
- **`BoundaryDimStatus`**: 新 field は default False で後方互換。golden test の dim status 比較は更新が必要。
- **`EstimatorProvider`**: Protocol に method 追加。LizyML 内蔵 Provider (LGBM) は実装する。サードパーティ Provider が存在する場合は実装が必要だが現時点で該当なし。
- **`_expand_range`**: 新引数 `min_allowed` / `max_allowed` は keyword-only / optional。
- format_version 変更不要。

### 代替案と不採用理由

| 代替案 | 不採用理由 |
|---|---|
| `_expand_range` 内で param 名から bounds を直接 lookup | search_space.py が estimator-specific 知識を持つことになり、5-layer DAG (provider 経由のみ) の境界違反 |
| `default_space` に bounds をハードコード | `default_space` を使わずユーザーが `tuning.search_space` を指定した場合に bounds が効かない（Issue #152 のシナリオを完全には解決できない） |
| `parse_space` の検証緩和（warning のみ） | Optuna が後で raise するため、結局 fail。早期失敗で UX 改善が目的 |
| 単一 PR で全 Phase | レビュー負荷増・rollback 単位が粗い。3 Phase に分割し、各 Phase 単体で価値が出る形にする |
| Provider 不要（dim に bounds を直接書く） | `default_space` 以外（ユーザー定義空間）でも bounds が効くようにするには、param 名 → bounds の mapping が必要。estimator ごとの mapping を保持する責務は Provider が自然 |

### 受け入れ基準

- [ ] `parse_space` が `low >= high` を `LizyMLError(CONFIG_INVALID)` で拒否（テストあり）。
- [ ] `parse_space` が `log=True ∧ low <= 0` を `LizyMLError(CONFIG_INVALID)` で拒否（テストあり）。
- [ ] `EstimatorProvider.parameter_bounds(task)` が Protocol に追加されている。
- [ ] `LGBMProvider.parameter_bounds(task)` が LightGBM 固有 map を返す（テストあり）。
- [ ] `SearchDim`（FloatDim/IntDim）が optional `min_allowed` / `max_allowed` を持つ。
- [ ] `_expand_range` が bounds でクランプし、`BoundaryDimStatus.clamped_to_bound` を立てる（テストあり）。
- [ ] `Model.tune(re_tune=...)` が provider→dim に bounds を注入する（統合テストあり）。
- [ ] 既存 1709 テスト全件 pass。
- [ ] `uv run ruff check .` / `uv run ruff format --check .` / `uv run mypy lizyml/` がクリーン。

### Migration

- ユーザーには影響なし（公開 API 完全互換、新 field は optional）。
- サードパーティ Provider 実装者には `parameter_bounds(task) -> dict` の追加実装を求める。空 dict を返せば既存挙動と同一（unbounded expansion）。
- LizyStudio 側はこの Phase 完了後 `provider.parameter_bounds(...)` を介して UI の bound 制限を取得する流れに切り替える（別 Issue 管理）。

### Decision

- Date: 2026-05-10
- Result: accepted
- Notes:
  - **Phase 1 (PR #153)**: `parse_space` が `low >= high` と `log + low <= 0` を `LizyMLError(CONFIG_INVALID)` で拒否（+8 tests）。
  - **Phase 2 (PR #154)**: `EstimatorProvider.parameter_bounds(task)` Protocol method を追加し、`LGBMProvider` が 15 params の bounds を返す。`SearchDim`（FloatDim/IntDim）に optional `min_allowed`/`max_allowed` を追加。`_expand_range` が bounds でクランプし `BoundaryDimStatus.clamped_to_bound` を立てる（+25 tests）。
  - **Phase 3 (PR #156)**: `attach_bounds(dims, bounds)` ヘルパーを追加し、`Model._resolve_search_space` が `provider.parameter_bounds(cfg.task)` を search space に注入。default-space と user-supplied space の両方が bounds を自動取得（+10 tests）。Issue #152 の 5 ラウンド regression test 込み。
  - 後方互換性は完全維持。サードパーティ Provider は `parameter_bounds(task) -> {}` で従来挙動を再現可能。
  - リリース: v0.14.0 で配布。

## H-0079: silent objective override 修正と `EstimatorProvider.objective_choices()` / `metric_choices()` 導入

- **ステータス**: Accepted
- **起票日**: 2026-05-10
- **決定日**: 2026-05-10
- **スコープ**: Public API (`EstimatorProvider`), `lizyml/estimators/lgbm/adapter.py`, `lizyml/estimators/lgbm/defaults.py`, `lizyml/estimators/lgbm/metric_bridge.py`, `lizyml/estimators/lgbm/provider.py`, `lizyml/tuning/search_space.py` (`default_space` signature)
- **関連**: [Issue #159](https://github.com/nbx-liz/LizyML/issues/159), H-0078（同型の Provider 拡張パターン）, LizyStudio Issue #461（下流 UI consumer）, PR [#160](https://github.com/nbx-liz/LizyML/pull/160) / [#161](https://github.com/nbx-liz/LizyML/pull/161) / [#162](https://github.com/nbx-liz/LizyML/pull/162)

### 目的

LGBM Provider 層に存在する次の 2 つの問題を解消する。

1. **Bug — silent objective override**: `LGBMAdapter._build_params()` が user / Optuna trial 由来の `objective` を **無条件に strip** し、`_TASK_OBJECTIVE[task]` で再代入する。`default_space("regression")` の `CategoricalDim("objective", ("huber", "fair"))` で `"fair"` をサンプルした trial も実際は `"huber"` で学習され、`tuning_table` の `objective` 列が嘘になる。`metric` 側は H-0061 で同型の strip を解除済だが、`objective` は同型バグとして残置されていた（コミット `ba152b0`「fix(estimators): objective/metric stripped from user params (task-locked)」、2026-03-15）。
2. **API gap — private choice lists**: LizyStudio など下流 UI が "valid objective / metric per task" を提示する際、`_OBJECTIVE_CHOICES`（`defaults.py`）/ `_LGBM_NATIVE_METRICS` / `_FEVAL_METRICS`（`metric_bridge.py`）すべてが private。Provider レベルの公開 API が無いため、下流は (a) リストを再実装（drift risk）または (b) private symbol を import（layer 違反）の二択になる。さらに `_OBJECTIVE_CHOICES` は LightGBM が受理する `objective` enum の一部しか登録していない（regression: 9 → 2、binary: 3 → 1、multiclass: 2 → 2）。

H-0078 の `parameter_bounds(task)` と同型の Provider 拡張で両方を一度に解く。

### 対応方針

H-0078 と同じ 3-Phase 構成。Phase 単位で独立 PR 化し、各 Phase のみで価値が出る形にする。

#### Phase 1 — silent override 修正 + invariant guards

`LGBMAdapter._build_params()` の `user_params.pop("objective", None)` を **task-compatibility check** に置換する。

- 同 task 互換の値 → `params["objective"]` に上書き（lgb.train まで届く）。
- task 非互換の値（例: `task="binary"` で `objective="regression"`）→ `LizyMLError(CONFIG_INVALID)` を raise。既存の cross-task 注入防御テスト（`tests/test_code_review_fixes.py`）の意図は維持される。
- 末尾に **invariant assertion** を追加し、user 指定値が strip / 上書きされた場合に dev / test 環境で fail-fast（L5 in-code guard）。

`TASK_COMPATIBLE_OBJECTIVES: dict[TaskType, frozenset[str]]` を `defaults.py` に追加し、LightGBM 公式 enum の canonical 名のみ登録する（regression 9 / binary 3 / multiclass 2）。

#### Phase 2 — Provider choice APIs

`EstimatorProvider` Protocol に 2 つの method を追加する。

```python
class EstimatorProvider(Protocol):
    ...
    def objective_choices(self, task: TaskType) -> tuple[str, ...]:
        """Canonical objective names valid for *task*. No aliases."""
        ...

    def metric_choices(self, task: TaskType) -> dict[Literal["native", "feval"], tuple[str, ...]]:
        """Per-task valid metrics, split by source.

        - ``"native"``: LightGBM-evaluated metrics.
        - ``"feval"``:  LizyML custom metrics, wired as feval callables.

        Canonical names only, deterministic order, no duplicates across keys.
        """
        ...
```

`LGBMProvider` で実装。`default_space()` は optional `provider` 引数を受け取り、`provider.objective_choices(task)` から `CategoricalDim("objective", ...)` を構築する（既存 callers は無変更で動作）。

#### Phase 3 — 内部統合 + drift guards + docs

- `defaults.py:_OBJECTIVE_CHOICES` を削除し、`default_space` は `LGBMProvider().objective_choices()` を経由する。
- `metric_bridge._LGBM_NATIVE_METRICS` / `_FEVAL_METRICS` は private cache として残すが、authoritative source は Provider と明記。alias 受理（`l1`, `l2`, `mae`, `mse` 等）は metric_bridge 側に残し、`metric_choices()` の戻り値は canonical のみ。
- `MetricRegistry` の登録メトリクスが `metric_choices(task)["native"] ∪ metric_choices(task)["feval"]` で完全に被覆されることを drift test で担保（L4）。
- `docs/config-reference.md` に task 別 valid objectives 表を追加（L7）。

### Regression prevention（7 layers）

Issue #159 で要求された全 7 層を Phase に分散して実装する。

| Layer | 内容 | 対応 Phase |
|---|---|---|
| L1 | parametric end-to-end identity test（14 task×objective ペア） | Phase 1 |
| L2 | tune-sampled-objective が refit booster に届く identity guard | Phase 1 |
| L3 | provider drift smoke-fit（`objective_choices` / `metric_choices` の各値で実際に学習が通ること） | Phase 2 |
| L4 | `MetricRegistry` ↔ `metric_choices` 被覆 drift test | Phase 3 |
| L5 | `_build_params()` 末尾の invariant assertion | Phase 1 |
| L6 | CHANGELOG（Changed (potentially breaking)）+ DEPRECATIONS 行 | Phase 1 |
| L7 | `docs/config-reference.md` に task 別 valid objectives 表 | Phase 3 |

### 影響範囲

- **変更ファイル**:
  - Phase 1: `lizyml/estimators/lgbm/adapter.py`、`lizyml/estimators/lgbm/defaults.py`（`TASK_COMPATIBLE_OBJECTIVES` 追加）、`tests/test_estimators/test_lgbm_objective_identity.py`（新規）、`tests/test_tuning/test_tune_fit_identity.py`（追記）、`tests/test_estimators/test_lgbm_defaults.py`（修正：バグ前提アサート差し替え）、`CHANGELOG.md`、`docs/DEPRECATIONS.md`、`HISTORY.md`。
  - Phase 2: `lizyml/estimators/provider.py`（Protocol 追加）、`lizyml/estimators/lgbm/provider.py`（実装）、`lizyml/estimators/lgbm/defaults.py`（`default_space` signature）、`tests/test_estimators/test_provider_choice_apis.py`（新規）、`tests/test_estimators/test_provider_protocol_drift.py`（新規）、`HISTORY.md`、`CHANGELOG.md`。
  - Phase 3: `lizyml/estimators/lgbm/defaults.py`（`_OBJECTIVE_CHOICES` 削除）、`lizyml/estimators/lgbm/metric_bridge.py`（authoritative source コメント）、`tests/test_estimators/test_metric_choices_registry_coverage.py`（新規）、`docs/config-reference.md`、`HISTORY.md`、`CHANGELOG.md`。
- **無変更**: Persistence 形式、Calibration、Plots/Tables 出力 shape、Codegen export、CV / Splitter、Calibration の cross-fit 仕様。

### 互換性

- **`objective` の挙動変更**（Phase 1）: 同 task 互換値はこれまで silent に無視されていたが、今後は反映される。**過去の tune 結果を re-tune / refit すると metric が変動する可能性**がある（特に regression default_space の `objective="fair"` を引いていた trial）。CHANGELOG の "Changed (potentially breaking)" に明記し、DEPRECATIONS 行で言及する。
- **`EstimatorProvider` の Protocol 拡張**（Phase 2）: method 2 件を追加。LizyML 内蔵 Provider（LGBM）は実装する。サードパーティ Provider が存在する場合は実装が必要だが現時点で該当なし（H-0078 と同じ判断）。
- **`default_space()` signature**（Phase 2）: optional `provider` 引数を追加。既存 callers は無変更で動作。
- **format_version 変更不要**: 保存物の意味は変わらない（trained booster 自体は今も `_TASK_OBJECTIVE[task]` で学習されているため、export / persistence で書かれる値は修正前後で一致）。Re-tune / re-fit 後にのみ booster の objective が変わる。

### 代替案と不採用理由

| 代替案 | 不採用理由 |
|---|---|
| `_TASK_OBJECTIVE` を継続し、user 指定値を完全無視（現状維持） | tune の `tuning_table` が嘘を表示し続ける問題が解決しない。`default_space` が `objective` を tunable に出している以上、サンプル値が反映されないのは仕様矛盾 |
| 互換性のため warning のみで `_TASK_OBJECTIVE` を上書きしない | `tuning_table` と実際の booster が乖離し続ける（現状の bug と同じ）。サイレントが致命的なので fail-fast へ倒す |
| Issue #159 を 1 PR で bundle | レビュー範囲が大きく、Phase 1 の bug fix が API 追加と同時にしか出せなくなる。H-0078 の 3-PR 構成が機能した実績があるため踏襲 |
| `objective_choices` を `frozenset` で返す | 順序保証が無く、UI 側で安定した表示順を担保できない。`tuple[str, ...]` で順序固定 |
| `metric_choices` を flat な `tuple` にする | 下流 UI が "native vs feval" を区別できず、計算速度の違い（feval は callable 経由で遅い）を伝えられない。`dict[Literal, tuple]` で source を明示 |
| custom `fobj`（callable objective）対応を同梱 | 別問題（`ObjectiveRegistry` 設計が必要）。Out of scope として後続 Issue に回す |
| `_TASK_METRIC` の objective 連動（例: `objective="tweedie"` なら metric も tweedie 系へ） | 別問題。UX 改善で hard bug ではないため Out of scope |

### 受け入れ基準

- [ ] **Phase 1**:
  - [ ] `LGBMAdapter._build_params()` が同 task 互換の `objective` を pass-through し、cross-task 値を `LizyMLError(CONFIG_INVALID)` で reject する。
  - [ ] L1: 14 (task × objective) ペアの parametric end-to-end identity test が green（user 指定 `objective` が booster の `params["objective"]` に bit-for-bit 一致する）。
  - [ ] L2: tune-sampled-objective が refit booster に届く identity guard が green。
  - [ ] L5: `_build_params()` 末尾の invariant assertion が存在する。
  - [ ] L6: CHANGELOG「Changed (potentially breaking)」+ DEPRECATIONS 行。
  - [ ] 既存 1709 テスト全件 pass。
- [ ] **Phase 2**:
  - [ ] `EstimatorProvider.objective_choices(task) -> tuple[str, ...]` が Protocol に追加されている。
  - [ ] `EstimatorProvider.metric_choices(task) -> dict[Literal["native","feval"], tuple[str, ...]]` が Protocol に追加されている。
  - [ ] `LGBMProvider.objective_choices(task)` が canonical 9 / 3 / 2 を返す。
  - [ ] `LGBMProvider.metric_choices(task)` が canonical 名のみ・重複無し・順序固定で返す。
  - [ ] `default_space(task, provider=None)` が `provider.objective_choices(task)` を経由する。
  - [ ] L3: provider drift smoke-fit（全 objective / 全 native metric で smoke fit が通る）。
- [ ] **Phase 3**:
  - [ ] `_OBJECTIVE_CHOICES`（defaults.py）が削除されている。
  - [ ] L4: MetricRegistry ↔ `metric_choices` 被覆 drift test が green。
  - [ ] L7: `docs/config-reference.md` に task 別 valid objectives 表が追加されている。
- [ ] `uv run ruff check .` / `uv run ruff format --check .` / `uv run mypy lizyml/` / `uv run pytest` が全 Phase でクリーン。

### Migration

- ユーザー向け：
  - `LGBMConfig.params={"objective": ...}` を明示指定していたコードは、これまで silent に無視されていた値が今後は反映される。同 task 互換であれば動作上のデグレは無いが、metric が変わる可能性に注意。
  - `default_space("regression")` の tune 結果は、`"fair"` を引いた trial が今後は実際に `fair` で学習されるため、過去の tuning_table の score と乖離する可能性がある（過去の tuning_table は嘘だった）。
- サードパーティ Provider 実装者：Phase 2 で `objective_choices` / `metric_choices` の追加実装を求める。空の `tuple` / `{"native": (), "feval": ()}` を返せば「choice 提供なし」として扱われる（`default_space` 側はサンプル候補が無くなるので tune 不可、明示的なエラーにする）。

### Decision

- Date: 2026-05-10
- Result: accepted (Phase 1)
- Notes:
  - **Phase 1 (PR #160)**: `LGBMAdapter._build_params()` の `user_params.pop("objective", None)` を `_check_objective_compatible()` 経由の task-compat check に置換。`TASK_COMPATIBLE_OBJECTIVES`（regression 9 / binary 3 / multiclass 2）を `defaults.py` に追加。L1 parametric identity test（14 ペア + 7 cross-task reject）、L2 tune-sampled-objective identity test（regression）、L5 in-code invariant assertion を実装（+24 tests）。CHANGELOG「Changed (potentially breaking)」と DEPRECATIONS の行を追加。
  - **Phase 2 (PR #161)**: `EstimatorProvider.objective_choices(task) -> tuple[str, ...]` と `EstimatorProvider.metric_choices(task) -> dict[Literal["native","feval"], tuple[str, ...]]`（型 alias `MetricChoices`）を Protocol に追加。`LGBMProvider` に canonical 名のみの順序付きテーブル（regression 9 / binary 3 / multiclass 2 objectives、native/feval metric tuples）を実装。`default_space(task, provider=None)` を任意 provider 注入対応に拡張（既存 callers は無変更）。`_validate_objective_consistency()` をモジュールロード時に走らせ、`TASK_COMPATIBLE_OBJECTIVES` ↔ `_LGBM_OBJECTIVE_CHOICES` の drift を即座に検知（**意図的フェイルファスト**: drift があれば LizyML import 自体が失敗する。サイレントな整合性違反よりも process start 時クラッシュを優先する設計トレードオフ）。L3 provider drift smoke-fit（14 objectives + 21 native metrics = 35 fits）、API contract test 44 件（signature / no-aliases / no-duplicates / 各種 subset）を追加。
  - 既存 1709 → 1794（Phase 1）→ 1873 テスト全件 pass。
  - **Phase 3 (PR pending)**: `defaults._OBJECTIVE_CHOICES` を削除し、tune-safe な保守的サブセット `_DEFAULT_TUNE_OBJECTIVES` にリネームして意図を明示（`gamma`/`poisson`/`tweedie` 等は target 分布の制約が厳しく default tune には不向きのため非露出。ユーザーは `LGBMProvider().objective_choices(task)` で広い集合を取得して独自 search_space を組める）。L4 MetricRegistry 被覆 drift test を追加し、`metric_bridge._LGBM_NATIVE_METRICS["multiclass"]` に `auc` が誤登録されていた **pre-existing バグを発見**（LightGBM 4.x は multiclass で `params["metric"]=["auc"]` を `"Multiclass objective and metrics don't match"` で拒否）。whitelist から `auc` を削除し、ユーザーは `Model.evaluate(metrics=["auc"])` 経由 (sklearn OvR) または `auc_mu` (fit-time) を使うよう挙動を整理。`docs/config-reference.md` に L7 task-objective canonical 表 + target-distribution 制約 + Provider が source of truth であることを明記。
  - 既存 1873 → 1885 テスト全件 pass（+12 L4 drift coverage）。
  - **リリース予定**: v0.15.0 で 3 phase まとめて配布。

## H-0080: `training.seed` を outer splitter / isotonic calibrator に伝搬（`split.random_state` を sentinel None 化）

- **ステータス**: Accepted
- **起票日**: 2026-06-01
- **決定日**: 2026-06-01
- **スコープ**: Config schema (`KFoldConfig` / `StratifiedKFoldConfig` / `StratifiedGroupKFoldConfig` の `random_state`), `lizyml/config/loader.py`（default split 注入）, `lizyml/core/_model_factories.py`（`build_splitter` / `_build_splitter_for_method`）, `lizyml/core/model.py`（calibration params 構築）
- **関連**: [Issue #169](https://github.com/nbx-liz/LizyML/issues/169), v0.15.0 品質監査, H-0069（フィールド同期の前例）

### 目的（課題）

「single seed が deterministic に全乱数へ伝搬する」という再現性要件に反し、`training.seed` が **outer splitter と isotonic calibrator に届いていなかった**。

- `build_splitter()` は `BlockedGroupKFoldConfig` 分岐でのみ `cfg.training.seed` を転送し、一般分岐（KFold / StratifiedKFold / StratifiedGroupKFold）は `split.random_state`（既定 42）を直接使用していた。さらに `loader._normalize_split_default()` が split 省略時に `random_state: 42` を**ハードコード注入**していたため、最頻ケース（split 省略 + `training.seed` 指定）でも fold は 42 固定だった。
- isotonic calibrator は内部 validation split 用 seed を `calibration.params["seed"]`（既定 42）から取り、`training.seed` を見ていなかった。

結果として `training.seed=123` に変えても CV fold / calibrator split は 42 のままで、再シードに複数フィールドの lockstep 変更が必要な usability trap になっていた。

### 対応方針（Option A: sentinel None）

`split.random_state` を `int | None`（既定 `None`）に変更し、`None` を「`training.seed` を継承」の sentinel とする。解決は **splitter-build 時のみ**で行い、config には書き戻さない（H-0069 の dual-write round-trip 破壊を回避）。

- `KFoldConfig` / `StratifiedKFoldConfig` / `StratifiedGroupKFoldConfig`: `random_state: int | None = None`。
- `loader._normalize_split_default()`: default split から `random_state: 42` を除去（schema 既定 None を効かせる）。
- `build_splitter()`: 一般分岐でも `seed=cfg.training.seed` を `_build_splitter_for_method` に渡す。
- `_build_splitter_for_method()`: `random_state = explicit if explicit is not None else seed`（明示値が優先、未指定は training.seed、最終 fallback 42）。
- `model.py` calibration: `method == "isotonic"` かつ `calibration.params` に `seed` 不在のとき `cfg.training.seed` を `setdefault`。

代替案 Option D（`split.random_state` を computed mirror 化して seed を単一化）は、既存の明示指定ユーザー向け legacy 吸収が必要で破壊度が高く、独立 split seed の能力を失うため不採用。

### 経路の母集団（散文ではなく検査で閉じる）

> H-0111 注記（#319）: この節は `training.seed` の伝搬（H-0080）とは関係が無く、LightGBM のパラメーター名の経路についての検討である。経路の一覧 `ESTIMATOR_ROUTES` を決めたのは H-0093 で、H-0093 はこの節の 3 か所を、走査を直したうえで 6 か所に改めた。この節は経緯として残し、移動しない。

独立レビューが 2 ラウンド続けて「全経路を覆う」という主張を反証した。1 回目は構築後の config 変更と artifact 由来の `best_model_params`、2 回目は `export_code` の生成 params である。**どちらも回答は「呼び出し点を 1 つ足して文を 1 つ足す」だった** — 同じ形の主張が同じ形で 2 回破れており、3 回目が無いと考える理由が無い。

そこで母集団を**測って閉じる**。`lizyml/` 内で「パラメータ dict が学習器の前に置かれうる場所」を AST で走査すると 3 か所しかない:

| 場所 | 内容 |
|---|---|
| `estimators/lgbm/adapter.py` | パッケージ内で唯一の実行時 `lgb.train`。params は `self.params` |
| `estimators/lgbm/adapter.py` | `lgb.Dataset`。キーワードは API 引数であって調整対象パラメータではなく、コンストラクタのシグネチャと突き合わせる |
| `estimators/lgbm/provider.py` | `build_estimator_factory` の adapter 構築。**params が adapter に渡る唯一の入口**であり、その dict を作る側（`_merge_params` / 探索空間 / `export_code`）を Facade が塞いでいる |

この一覧は `tests/test_estimators/test_lightgbm_parameter_names.py` の `ESTIMATOR_ROUTES` として置き、走査が**一覧に無い場所を見つけた場合**と**一覧が存在しない場所を挙げている場合**の両方で落ちる。前者は新しい経路の追加をコミット時点で捕まえ、後者は「実際より広い被覆」を主張したままの記述を防ぐ。

codegen テンプレートは文字列定数の中にあり、このモジュールの AST では call にならないため上記の走査には現れない。別途テンプレート専用の走査が担当する。

**バックストップ（`build_estimator_factory` 内での再検査）は置かない。** 上記の走査が「Facade の 3 つの門を通らずに adapter へ届く経路は無い」ことを示している以上、その位置の検査は本番入力に対して決して発火しない — DC6 そのものになる。経路が増えたときに落ちるのは走査であって、沈黙するバックストップではない。

### 影響範囲 / 互換性

- **後方互換（無変更ケース）**: `training.seed` も既定 42 のため、全デフォルト構成・`training.seed` 未変更構成では実効 seed は 42 のまま。fold / OOF / calibrated は不変。
- **変化するケース（＝意図した修正）**: `training.seed` を非 42 に設定し、かつ `split.random_state` を明示していない構成。これらは fold が `training.seed` を反映するようになる（**potentially breaking**: OOF / metrics / split indices / 保存 artifact の fold 構成が変わりうる）。CHANGELOG に「Changed (potentially breaking)」として明記する。
- 明示 `split.random_state` 指定は従来通り優先され不変。
- `model_dump()` は `random_state: None` をそのまま round-trip（書き戻しなし）。保存 artifact 互換に `format_version` 変更は不要。
- inner_valid は既に `cfg.training.seed` を継承済（`build_inner_valid`, BLUEPRINT §10.3.1）のため対象外。明示 inner_valid config の `random_state`（既定 42）は本提案のスコープ外。

### 受け入れ基準（テスト観点）

- `build_splitter`: ①全デフォルト（`training.seed=42`, split `random_state` None）→ splitter `random_state == 42`（後方互換）、②`training.seed=123` + split `random_state` None → splitter `random_state == 123`、③`split.random_state=7` 明示 + `training.seed=123` → splitter `random_state == 7`（明示優先）。
- **seed sensitivity（invariant）**: `training.seed` のみ異なる 2 構成で OOF / outer split indices が変化する（同一なら fail）。同一 seed では bit 一致。
- isotonic calibrator: `calibration.params` に seed 不在時に `training.seed` を継承する。
- 既存テスト全件 green（loader default split の `random_state` ハードコード除去に伴う回帰なし）。


## H-0081: bit 一致の再現性保証を「固定 `(num_threads, CPU)` 環境」にスコープ明記（doc-scope）

- **ステータス**: Accepted
- **起票日**: 2026-06-01
- **決定日**: 2026-06-01
- **スコープ**: BLUEPRINT.md（§2 設計原則の再現性原則、§18.1.1 再現性テスト）, README.md（Design Priorities の bit 一致記述）。**コード変更なし（defaults 不変）**。
- **関連**: [Issue #170](https://github.com/nbx-liz/LizyML/issues/170), v0.15.0 品質監査

### 目的（課題）

BLUEPRINT は「同一 `config + seed` で bit 一致」を再現性の最優先保証として掲げるが、LightGBM のデフォルト（`feature_fraction` / `bagging_fraction` / `bagging_freq` による確率的サブサンプリング）は `deterministic` / `force_row_wise` / `num_threads` 未設定のままである。LightGBM の histogram 構築はスレッド数に依存するため、CPU / スレッド数が異なるマシン間（CI runner / ユーザー環境）では bit 一致が崩れうる。保証の文言が環境スコープを明示していないため、**文書上の over-promise** になっている。

### 対応方針（doc-scope: 保証をスコープ明記）

bit 一致保証を「**固定 `(num_threads, CPU)` 環境**」にスコープする旨を BLUEPRINT / README に明記する。defaults への `deterministic: true` / `force_row_wise: true` 追加は行わない。

- BLUEPRINT §2 設計原則: 再現性原則に「bit 一致は固定 `(num_threads, CPU)` 環境を前提とする」旨の注記を追加。
- BLUEPRINT §18.1.1 再現性テスト: 再現性テストが同一環境（同一スレッド数）を前提とすることを明記。
- README: Reproducibility 記述に同趣旨の注記を追加。

### 代替案（不採用）

defaults に `deterministic: true` + `force_row_wise: true` を追加し、クロス環境 bit 一致を実際に保証する案。**性能コスト**（`force_row_wise` による学習速度低下、`deterministic` のオーバーヘッド）が大きく、最頻ユースケースに恒久的なペナルティを課すため不採用。将来、クロス環境再現性を opt-in で提供する場合は別 Proposal（Change-Gate）とする。

### 影響範囲 / 互換性

- **ドキュメントのみ**。公開 API / Config / FitResult / PredictionResult / Artifacts / defaults いずれも不変。`format_version` 変更不要。
- 既存ユーザーの実行結果・保存 artifact に一切変化なし。

### 受け入れ基準（テスト観点）

- コード変更なしのため新規テスト不要。既存テスト全件 green（回帰なし）を確認する。
- BLUEPRINT / README の文言に「固定 `(num_threads, CPU)` 環境」のスコープが明記されていること（レビュー観点）。


## H-0082: `evaluate(None)` / `fit_result` の公開返却を selective deep-copy 化し internal state 汚染を防止

- **ステータス**: Accepted
- **起票日**: 2026-06-01
- **決定日**: 2026-06-01
- **スコープ**: `lizyml/core/model.py`（`evaluate(metrics=None)` 返却、`fit_result` property 返却）, `lizyml/core/types/fit_result.py`（`FitResult.__deepcopy__`）。`FitResult` は **非 frozen を維持**（dataclass のまま）。
- **関連**: [Issue #174](https://github.com/nbx-liz/LizyML/issues/174), v0.15.0 品質監査, `TuningResult`（frozen + defensive copy の前例）

### 目的（課題）

`FitResult` は非 frozen dataclass で mutable フィールド（`metrics` 等のネスト dict）を持ち、`evaluate(metrics=None)` と `fit_result` property は **live な内部参照をそのまま返す**。呼び出し側が返り値を変異させる（例: `m.evaluate()["raw"]["oof"]["rmse"] = 0`）と内部 `_metrics` が破壊され、`export()` が live な `_metrics` を読むため **export メタデータ汚染（再現性リスク）** に至る。一方 filtered path（`evaluate([...])`）は `filter_metrics` で fresh dict を返すため、**挙動が非対称**でもある。

### 対応方針（public return で selective deep-copy、FitResult 非 frozen 維持）

公開返却点でのみ copy を適用し、内部 live state を外部に貸し出さない。`fit_result` は **selective deep-copy**（mutable data は複製、学習済み estimator は reference 共有）とする。

- `evaluate(metrics=None)`: `return deepcopy(self._metrics)`（`metrics` は純粋なネスト dict のため全 deep-copy で問題なし）。
- `fit_result` property: `return deepcopy(self._require_fit())`。`FitResult.__deepcopy__` を実装し、
  - **deep-copy する mutable data**: `metrics` / `history` / `feature_names` / `dtypes` / `categorical_features` / `splits` / `data_fingerprint` / `run_meta` / `target_encoder` / `oof_pred` / `if_pred_per_fold` / `oof_raw_scores`。
  - **reference 共有する学習済み estimator**: `models` / `calibrator` / `pipeline_state`。
- **selective の理由**: `copy.deepcopy(LightGBM Booster)` は model 文字列 round-trip となり `booster.params`（`objective` 等）の fidelity を失う（`params["objective"]` が `None` 化）。export 汚染の実害ベクタは `metrics`（plain dict）であり、これを deep-copy で完全封鎖すれば再現性リスクは解消する。学習済み estimator まで複製すると **公開 `fit_result.models[i]` 経由の Booster metadata を劣化させる回帰**となるため共有する。
- 内部経路（Mixin / plot / persistence / export）は `FitState.fit_result`（`_require_fit()` 由来の live 参照）と `self._metrics` を直接使うため **copy のコストを負わない**。公開 property `Model.fit_result` は外部呼び出し専用であることをコード調査で確認済み（内部は `state.fit_result` を使用）。

### 代替案（不採用）

- **FitResult 全体を deepcopy**: 学習済み Booster の `params` fidelity を失い、公開 `fit_result.models[i]` の metadata を劣化させる回帰となるため不採用（本対応方針が selective とした根拠）。
- **FitResult を frozen 化**: ネスト dict / list は frozen dataclass でも mutable のままで根本解決にならず、内部で `FitResult` を構築・保持する多数の経路に破壊的影響が及ぶため不採用。
- **fit_result を borrowed reference として doc 明記のみ**: export 汚染という実害（再現性リスク）を残すため不採用。

### 影響範囲 / 互換性

- **Result の「形・意味」は不変**。返却される dict / FitResult の構造・値は完全に同一で、参照の同一性のみが変わる（`m.fit_result is m.fit_result` → `False`、`m.evaluate() is m.evaluate()` → `False`）。`format_version` 変更不要。
- `fit_result.models` / `calibrator` / `pipeline_state` は **reference 共有**（read-only 想定）。これらを呼び出し側が破壊的に変異させると内部 state に波及しうるが、学習済みモデルの故意変異は契約外の misuse とし、docstring に read-only である旨を明記する。export 汚染の現実的ベクタ（`metrics` 変異）は完全に封鎖される。
- 同一性に依存する利用（`is` 比較）は破壊されうるが、Result は値オブジェクトであり同一性依存は契約外。

### 受け入れ基準（テスト観点）

- **汚染防止（回帰トラップ）**: `evaluate(None)` の返り値ネスト dict を変異 → 後続 `export()` のメタデータ / `model.evaluate()` が影響を受けない。
- **fit_result 独立性**: `m.fit_result.metrics` を変異 → 内部 state（後続 `m.fit_result` / `export`）が不変。
- **estimator 共有**: `m.fit_result.models[i] is m._fit_result.models[i]`（identity 保持で Booster fidelity を維持）。
- **値の同一性**: copy 前後で構造・数値が bit 一致（`==`）する。
- 既存テスト全件 green。


## H-0083: export 時に各 .pkl の SHA-256 を metadata.json に記録し load 時に検証

- **ステータス**: Accepted
- **起票日**: 2026-06-01
- **決定日**: 2026-06-01
- **スコープ**: `lizyml/persistence/exporter.py`（metadata に `checksums` 追加）, `lizyml/persistence/loader.py`（load 前の digest 検証）。`FORMAT_VERSION` は **2 のまま据置（additive・後方互換 read）**。
- **関連**: [Issue #179](https://github.com/nbx-liz/LizyML/issues/179), v0.15.0 品質監査（SECURITY/LOW）

### 目的（課題）

`Model.load()` は `.pkl`（`fit_result` / `refit_model` / `analysis_context`）を `joblib.load` で復元するが、これは任意 Python を実行する。load 前検証は `metadata.json` のみで、**検証済み metadata と .pkl バイト列の間に整合バインドが無い**。良性 metadata に改竄 .pkl を組み合わせると ACE に至る。「trusted-source のみ」契約＋pickle-free codegen 代替の開示により LOW だが、安価な整合チェックで改竄/破損を検出できる。

### 対応方針（additive checksum、format_version 据置）

`export()` で各 .pkl の SHA-256 を計算し `metadata.json` に additive フィールドとして記録。`load()` で `joblib.load` 前に digest を照合し、不一致は `DESERIALIZATION_FAILED` を送出する。

- metadata 構造（additive）::

      "checksums": {
          "algorithm": "sha256",
          "files": {
              "fit_result.pkl": "<hex>",
              "refit_model.pkl": "<hex>",
              "analysis_context.pkl": "<hex>"   # 任意（存在時のみ）
          }
      }

- `export()`: .pkl を dump 後にバイト列の SHA-256 を計算し metadata に格納（書き込み順を「.pkl → metadata.json」に変更）。
- `load()`: `checksums` が存在し当該ファイルの digest が登録されていれば照合。`algorithm` が `sha256` 以外、または digest 不一致は `DESERIALIZATION_FAILED`（context に `file` / `expected` / `actual` を格納）。
- **後方互換 read**: `checksums` 不在（H-0083 以前の format_version 2、または format_version 1）の artifact は検証を skip して従来通り load する。これにより `FORMAT_VERSION` 据置で旧 artifact を読める。

### 代替案（不採用）

- **`FORMAT_VERSION` を 3 に bump**: checksum は additive で旧 read を壊さないため不要。migration 負荷を避ける（locked 決定）。
- **署名（HMAC/公開鍵）**: 鍵管理が必要で LOW リスクに対し過剰。SHA-256 は「改竄/破損検出」目的に十分（fully-trusted-but-malicious producer に対し pickle を安全化するとは主張しない）。
- **検証を warning に留める**: 整合バインドの意味を成さないため、不一致は fail-closed（例外）とする。

### 影響範囲 / 互換性

- **後方互換**: 旧 artifact（checksums 不在）は従来通り load 可能。新 artifact は旧 loader でも load 可能（`checksums` は未知フィールドとして無視される）。`FORMAT_VERSION` 不変。
- 公開 API（`export` / `load` の引数・戻り値）不変。`load()` の戻り `metadata` dict に `checksums` キーが増えるのみ。
- **TOCTOU 対策**: `load()` は .pkl のバイト列を 1 回だけ読み、in-memory で digest 照合後 `joblib.load(io.BytesIO(...))` で復元する（ファイルを再 open しない）。hash 後・load 前のすり替え窓を排除（security-review 指摘）。
- **脅威モデル（明示）**: `metadata.json` 自体は署名しないため、書き込み権限を持つ攻撃者は `checksums` を改竄/除去できる。本機能は「改竄/破損の検出」が目的であり、trusted-but-malicious producer に対し pickle を安全化するものではない（既存の trusted-source 契約を維持）。

### 受け入れ基準（テスト観点）

- **正常**: export→load round-trip が成功し、`metadata["checksums"]["files"]` に全 .pkl の SHA-256 が入る。
- **改竄検出（落ちるべき例）**: export 後に `fit_result.pkl` のバイトを書き換え→`load()` が `DESERIALIZATION_FAILED`（context に file/expected/actual）。`refit_model.pkl` / `analysis_context.pkl` も同様。
- **後方互換**: `checksums` を持たない metadata（旧 artifact 模擬）で `load()` が従来通り成功する。
- 既存 persistence テスト全件 green。


## H-0084: `FitState` / `TuningState` を Layer-0 `core/types/` から facade 隣接 `core/_model_state.py` へ移動

- **ステータス**: Accepted
- **起票日**: 2026-06-01
- **決定日**: 2026-06-01
- **スコープ**: `lizyml/core/types/fit_state.py` → `lizyml/core/_model_state.py`（rename/move）, importer 4ファイル（`model.py` / `_model_plots.py` / `_model_tables.py` / `_model_persistence.py`）, `ARCHITECTURE.md`（facade tree）。**振る舞い・公開 API 不変**（`FitState` は `core/types/__init__` 非エクスポートの内部型）。
- **関連**: [Issue #171](https://github.com/nbx-liz/LizyML/issues/171), H-0074 / H-0077（Mixin state isolation）, ARCHITECTURE.md §Layer 0（依存ゼロ不変条件）

### 目的（課題）

`FitState` / `TuningState` は `lizyml/core/types/`（5層 DAG の Layer-0、ARCHITECTURE.md で「依存ゼロ」と宣言）に置かれていたが、フィールドが構造的に Layer-1/2 型（`LizyMLConfig`, `EstimatorProvider`, `RefitResult`）を参照する。参照は `TYPE_CHECKING` 限定で runtime import cycle は無いが、**配置が Layer-0 不変条件に違反**する唯一の lower-imports-higher エッジだった。`FitState` は実態として「組み立て済み fit の facade snapshot」であり、Layer-0 ではなく facade 隣接が正しい住所。BLUEPRINT は内容（H-0074）を記録するが配置ルールを waive していない。

### 対応方針（facade-adjacent へ移動、内容不変）

- `core/types/fit_state.py` を `core/_model_state.py`（Layer-4 facade 隣接、`_model_metrics.py` / `_model_predict.py` と同列）へ移動。クラス定義・フィールド・docstring（内容）は不変。
- importer の import パスを `lizyml.core.types.fit_state` → `lizyml.core._model_state` に更新（4 ソース + 2 テスト）。
- `ARCHITECTURE.md` の Layer-4 facade ディレクトリツリーに `_model_state.py` を追記（併せて既存ツリーから欠落していた `_model_metrics.py` / `_model_predict.py` も補記）。
- これにより Layer-0（`core/types/`）は `FitResult` / `PredictionResult` / `TuningResult` / `artifacts` のみの「依存ゼロ」型に戻り、DAG の唯一の back-edge を解消する。

### 代替案（不採用）

- **現状維持 + facade-state 例外を明文化**: 不変条件を弱める方向で、DAG の「Layer-0 は基盤・依存ゼロ」保証を曇らせるため不採用。
- **`FitState` を Layer-0 に留め、参照型を Layer-0 へ降格**: `LizyMLConfig` / `EstimatorProvider` / `RefitResult` は本質的に上位層であり降格不可。

### 影響範囲 / 互換性

- **振る舞い・公開 API 不変**。`FitState` / `TuningState` は内部型（`core/types/__init__` 非エクスポート、`lizyml` トップレベル非公開）であり、利用者向けの import パス変更は無い。
- `format_version` 変更不要（Artifacts schema 無関係）。
- 純粋な配置移動 + import 更新 + doc 同期。

### 受け入れ基準（テスト観点）

- 既存テスト全件 green（`test_fit_state.py` / `test_mixin_state_isolation.py` の import パス更新後も挙動不変）。
- `core/types/` 配下に Layer-1/2 型を参照する型が残っていないこと（Layer-0 依存ゼロの回復）。
- structural refactor のため E2E gate（`tests/test_e2e/` + 診断 API 経路）green。

---

## H-0085: inner-valid 境界ポリシーの統一（pipeline fit 境界の矛盾解消 + purge/embargo の inner 伝播）

- **ステータス**: Accepted
- **起票日**: 2026-07-02
- **決定日**: 2026-07-02
- **スコープ**: `training/refit_trainer.py`（pipeline fit 境界）, `training/inner_valid.py`（`TimeHoldoutInnerValid` に gap 追加）, `core/_model_factories.py`（`_resolve_auto_inner_valid` の purge/embargo 受け渡し）, `evaluation/evaluator.py`（数値 target NaN の例外化）, `config`（time-order 下 shuffled inner_valid の警告）, `BLUEPRINT.md §6.2 / §10.3.1 L602 / §10.3.2 L629-631`。**公開 API・FitResult shape は不変**（`best_iteration` の数値は変わり得る＝再現性の観点で挙動変化）。
- **関連**: [Issue #208](https://github.com/nbx-liz/LizyML/issues/208), [#212](https://github.com/nbx-liz/LizyML/issues/212), [#207](https://github.com/nbx-liz/LizyML/issues/207) item 4, [#210](https://github.com/nbx-liz/LizyML/issues/210) item 3。BLUEPRINT §6.2 / §10.3。2026-07-02 full-package review。

### 目的（課題）

inner-valid（early-stopping）境界に関する 4 つの課題を、一貫した 1 つの決定として解消する。

1. **#208 — pipeline fit 境界の自己矛盾**: BLUEPRINT §6.2 L394-395 と §10.3.2 L626 は「pipeline は outer fold の train 全体で fit する（inner-train に狭めない）」と定めるが、§10.3.2 L629 は RefitTrainer について「pipeline は inner-train のみで fit する（CVTrainer と一致）」と**逆の境界**を記す。実装も分かれており、`cv_trainer.py:111` は outer train 全体、`refit_trainer.py:100-103` は inner-train のみで fit し、コメント「consistent with CVTrainer」は事実に反する。
2. **#212 — inner 境界の gap 欠落**: `purge_gap` / `embargo`（`time_series` の `gap`）は outer split のみに適用され、§10.3.1 L602 は「inner valid に伝搬しない」と明記。auto-resolve の `TimeHoldoutInnerValid`（`inner_valid.py:194-196`）は inner_train と inner_valid を gap ゼロで隣接させるため、look-ahead 構築 target が境界で重なり、全 outer fold の `best_iteration` を楽観的に汚染する。
3. **#207 item 4 — 数値 target の NaN 契約が未定義**: NaN-target 検証は label-encoded string target のみ（`core/types/target_encoder.py:126-134`）。regression/binary の数値 target では `Model.fit` の契約が未定義・未テスト。
4. **#210 item 3 — time-order 下 shuffled inner_valid が無警告**: 明示 `inner_valid: {method: holdout}`（random permutation）を `time_series` / `purged_time_series` outer split と組み合わせると、時間的にリークした early-stopping split が無警告で成立する（BLUEPRINT L599 が許容）。

### 対応方針（決定）

- **#208 → pipeline fit 境界を「outer-train 全体（Refit は全データ）」に統一する**。RefitTrainer を CVTrainer 側へ寄せ、pipeline を全データで 1 回 fit → 変換後に inner-train / inner-valid を slice（CVTrainer の `_build_iv_subsets` と同型）。これにより現行の二重 fit（L630 の推論用 pipeline 別 fit）を解消し、`best_iteration` 選択の境界を CV fold と一致させる。`refit_trainer.py:95-96` の虚偽コメントを訂正。BLUEPRINT §10.3.2 L629-631 を outer/full-train 境界へ改訂（§6.2 / L626 が正）。
  - 根拠: (a) 高優先の §6.2 が既に outer-train 境界を定義、(b) 現行 `NativeFeaturePipeline` は y-free（カテゴリ辞書は X のみ）で OOF・best_iteration とも実質リークしない、(c) refit の二重 fit と境界不一致という実バグを同時に解消。将来 y-dependent transform を導入する際は、その Proposal で「pipeline fit を inner-train に狭める」判断を改めて行う。
- **#212 → purge_gap / embargo（time_series の gap）を auto-resolve inner-valid へ伝播する**。`TimeHoldoutInnerValid` に gap パラメータを追加し、inner_train と inner_valid の間を `purge_gap + embargo`（time_series は `gap`）行だけ空ける。`_resolve_auto_inner_valid` が outer split 設定から gap を受け渡す。BLUEPRINT §10.3.1 L602 を「purge_gap / embargo（gap）は inner valid にも伝播する」に改訂（`n_splits` / `shuffle` / `random_state` / `train_size_max` / `test_size_max` は引き続き非伝播）。
- **#207 item 4 → 数値 target の NaN を明示的に拒否**。covered-OOF より前段、`Model.fit` の入口で数値 target に NaN があれば `LizyMLError(DATA_SCHEMA_INVALID)` を nan_count context 付きで送出する契約に固定する。
- **#210 item 3 → 警告に留める（エラー化しない）**。`time_series` / `purged_time_series` outer split と shuffle を伴う明示 inner_valid（holdout stratify、random permutation）の組み合わせで `UserWarning` を発する。L599 の「明示指定を尊重する」仕様は維持（spec-compatible）。

### 代替案（不採用）

- **#208 を inner-train のみに統一**: 最も厳格で将来の y-dependent transform に耐性があるが、高優先の §6.2 L394-395 を改訂する必要があり、CVTrainer が fold 毎に pipeline 再 fit するコスト増を伴う。現 pipeline が y-free で実害が early-stopping 語彙に限定される現状では過剰。y-dependent transform 導入時に再検討する。
- **#212 を明文化のみ（docstring caveat）**: 実装コストは最小だが、`purge_gap` を設定したユーザーの意図（境界リーク排除）を early-stopping 経路で裏切る穴が残るため不採用。
- **#210 item 3 をエラー化**: L599 の「明示 inner_valid を尊重」仕様と衝突するため、警告に留める。

### 影響範囲 / 互換性

- **公開 API・Config・FitResult / PredictionResult の shape は不変**。
- **挙動変化**: RefitTrainer の pipeline fit 境界変更と inner 境界の gap 追加により、既存モデルの `best_iteration`（ひいては学習済みモデル）が変わり得る。再現性契約上の変更であり、format_version は据え置き（Artifacts schema は不変）。CHANGELOG に「早期停止の分割境界が変わる」旨を明記する。
- 数値 target NaN の拒否は、従来 undefined だった経路を fail-fast にするもので、正常データには影響しない。

### 受け入れ基準（テスト観点）

- **#208**: RefitTrainer が全データで pipeline を 1 回 fit することを固定するテスト（二重 fit が無いこと）。CVTrainer / RefitTrainer が同一 pipeline fit 境界であることを検証（`test_pipeline_fit_boundary.py` に境界固定 assertion 追加）。
- **#212**: `purged_time_series`（purge_gap>0）で auto-resolve された inner-valid の inner_train 末尾と inner_valid 先頭の間に `purge_gap + embargo` 行の gap が存在することを固定する RED テスト（現行 zero-gap では落ちる）。
- **#207 item 4**: 数値 target に NaN を含む入力で `Model.fit` が `LizyMLError(DATA_SCHEMA_INVALID)` を nan_count context 付きで送出する RED テスト。
- **#210 item 3**: time-order outer split × shuffled 明示 inner_valid で `UserWarning` が発ることを検証するテスト。
- 上記いずれも「落ちるべき例」を含む（CLAUDE.md §6 の split/leakage/calibration 必須要件）。

## H-0086: Phase 3 契約/永続化/公開API の一括修正（FitResult 参照返し・tuned params 非永続化・config round-trip・top-level export）

- **ステータス**: Accepted
- **起票日**: 2026-07-03
- **決定日**: 2026-07-03
- **スコープ**: `core/model.py`（`fit()` の返却 / `load()` の metrics 共有）, `core/_model_persistence.py` + `persistence/exporter.py` + `persistence/loader.py`（tuned params の永続化・復元）, `config/schema.py`（inner_valid explicitness の round-trip 化）, `lizyml/__init__.py` + `core/types/__init__.py`（公開 re-export）, `BLUEPRINT.md`（公開 API surface / Artifacts metadata）。**format_version は据え置き（2、additive）**。
- **関連**: [Issue #204](https://github.com/nbx-liz/LizyML/issues/204), [#215](https://github.com/nbx-liz/LizyML/issues/215), [#203](https://github.com/nbx-liz/LizyML/issues/203), [#213](https://github.com/nbx-liz/LizyML/issues/213)。H-0082（deep-copy 防御）, H-0069（inner_valid canonical）, H-0083（metadata additive 前例）。2026-07-02 full-package review。

### 目的（課題）

full-package review が検出した契約・永続化・公開 API の 4 課題を、Phase 3 の契約クラスタとして一括で解消する。

1. **#204 — `fit()` / `load()` が内部 FitResult を参照で漏らす**: `fit_result` プロパティは H-0082 で selective deep-copy して internal state 汚染（→後続 `export()` の metadata 汚染）を防ぐが、主経路である `fit()` は `self._fit_result` と同一オブジェクトを返す（`model.py:275-277`）。`load()` も `instance._metrics = fit_result.metrics` で dict を共有する（`_model_persistence.py:192`）。最も使われる経路で防御が効いていない。
2. **#215 — tuned params が永続化されない**: best params は in-memory の `_tuning_result` overlay 経由で fit 時に適用される（`model.py:185-201, 878-897`）のみで、`export()` は fit/refit/config/metrics だけを書き（`exporter.py:96-108`）、`load()` は `_tuning_result` を復元しない（`_model_persistence.py:191-201`）。`Model.load()` 後の再 `fit()` は tuned params を失い config デフォルトで学習する — artifact は tuned で predict するのに再学習は defaults という silent な再現性ドリフト。
3. **#203 — config round-trip で明示 inner_valid が消失**: `model_dump()` は computed field `validation_ratio` を常に emit するため、再検証時に explicitness ヒューリスティック（`user_explicit_inner_valid = iv_in and not vr_present`）が `False` に倒れ、`_inner_valid_explicit`（`PrivateAttr`・非直列化）が失われる。factory は outer split から auto-resolve し直し、ユーザーの明示 `time_holdout` / `group_holdout` を silent に別戦略へ置換する（`export → load → fit` 経路を含む）。time/group データでは leakage-relevant。
4. **#213 — 契約型 / LizyMLError が top-level 未 export**: `Model.fit/predict/tune` は `FitResult` / `PredictionResult` / `TuningResult` を返し、公開メソッドは `LizyMLError` を送出するが、いずれも `lizyml` 直下から import できない（`__init__.py:13-22` は `Model` + 5 tuning 型のみ）。ユーザーは型注釈や `except LizyMLError` のために `lizyml.core.types` / `lizyml.core.exceptions`（private に見えるパス）へ手を伸ばす必要があり、公開/内部境界が曖昧化して将来のリファクタが事実上破壊的になる。

### 対応方針（決定）

- **#204 → `fit()` も selective deep-copy を返す**。`fit()` は `self._fit_result` に internal 参照を保持したまま、返却値は `FitResult.__deepcopy__`（H-0082 の selective copy）を通す。`load()` は `instance._metrics` を metrics dict の deep-copy にして internal と外部返却の共有を断つ。公開 API の shape・意味は不変（返却型は FitResult のまま）。回帰テスト: `fit()` の返却値を mutate → 後続 `export()` の metadata が汚染されないこと。
- **#215 → tuned params を metadata.json に additive 永続化し load で復元**。`export()` の metadata に `tuning` ブロック（`best_model_params` / `best_smart_params` / `best_training_params` / `best_score` / `metric_name` / `direction`）を追加し、`load()` が最小 `TuningResult`（`trials=()` / `rounds=()`）を復元して `_tuning_result` にセットする。これにより `load()` 後の再 `fit()` が tuned params を再現する。**format_version は 2 のまま（additive、H-0083 と同型）**: `tuning` キーの無い旧 artifact は従来どおり load でき `_tuning_result=None`（現行挙動）。**スコープ外**: optuna study 実体を要する完全な `tune(resume=True)`（trials/study の再構築）は本 Proposal では扱わず follow-up とする（params 復元により resume の seed には寄与するが study 継続は別途）。
- **#203 → explicitness を round-trip 安全な marker で直列化**。`validation_ratio` と同様に wrap-validator で pop される computed marker `inner_valid_explicit` を emit し、再検証時に入力 dict にあればそれを explicitness の source of truth として尊重する（無ければ従来ヒューリスティックにフォールバック）。これで `dump → reload` と `export → load → fit` がユーザーの明示 inner_valid 戦略を再現する。**settable な公開フィールドは追加しない**（computed かつ入力時 pop）。互換性: 旧 dump（marker 無し）はヒューリスティックにフォールバック＝現行挙動。paired-config-fields skill 準拠。
- **#213 → 契約型と例外を top-level に re-export**。`lizyml/__init__.py.__all__` に `FitResult` / `PredictionResult` / `TuningResult` / `LizyMLError` / `ErrorCode` / `load_config` / `TaskType` を追加、`core/types/__init__.py` に `DataFingerprint` を追加。公開 export set を固定するゴールデンテストを追加。純粋な additive（既存 import は不変）。

### 代替案（不採用）

- **#204 を「返却は参照のまま・doc で read-only を明記」**: コスト最小だが H-0082 の防御目的（export 汚染防止）を主経路で放棄するため不採用。
- **#215 を「loaded model を inference-only とし再 fit を warn/raise」**: 実装は軽いが、tune→export→load→再 fit という正当なワークフローを塞ぐ。params 復元の方がユーザー価値が高いため不採用（study 完全 resume のみ follow-up 送り）。
- **#215 で TuningResult 全体（trials 含む）を JSON 直列化**: metadata.json が肥大化し、TrialResult の非自明な直列化が必要。overlay に必要な best_* params + スコアに絞る。
- **#203 で `validation_ratio` を model_dump から除外**: round-trip は直るが、read-only mirror として dump 出力を読む外部/下流の想定を壊すため不採用。marker 追加の方が additive。
- **#213 で `lizyml.core.*` を公開パスとして追認**: 内部レイアウトを凍結してしまい将来のリファクタを縛るため不採用。

### 影響範囲 / 互換性

- **format_version は 2 のまま**。#215 の `tuning`・#203 の `inner_valid_explicit` はいずれも additive で、旧 artifact / 旧 dump は従来どおり load・再検証できる。
- **公開 API**: #213 は re-export の追加のみ（既存 import 不変）。#204 は返却型不変（同一オブジェクト → 独立コピーへ変わるのみ；参照同一性に依存する呼び出し側があれば挙動変化だが、契約は「読み取り専用の独立コピー」を明文化）。
- **挙動変化**: #203 の修正後、round-trip / load 経由の明示 inner_valid は auto-resolve されず明示戦略を保つ（＝リーク経路を塞ぐ正しい方向の変化）。#215 の修正後、load 後の再 fit は tuned params を再現する。いずれも CHANGELOG に明記。

### 受け入れ基準（テスト観点）

- **#204**: `fit()` の返却値の `metrics` を mutate → 内部 state と後続 `export()` 出力が汚染されないことを固定する回帰テスト。`load()` 後 `_metrics` が返却 metrics と別オブジェクトであること。
- **#215**: tune → export → load → 再 fit で tuned params が再現される契約テスト（load 前後で `_merge_params` の overlay が一致）。`tuning` キーの無い旧 metadata が load 可能（後方互換）な RED/GREEN テスト。
- **#203**: `{"inner_valid": {"method": "time_holdout", "ratio": 0.2, ...}}` を dump → reload して `_inner_valid_explicit` が保持される RED テスト（現行 False で落ちる）。`export → load → fit` で明示 inner_valid が auto-resolve されない leakage 観点テスト。純 legacy `{"validation_ratio": 0.1}` は従来どおり auto-resolve（非回帰）。
- **#213**: `lizyml` の top-level `__all__` を固定するゴールデンテスト（`FitResult` 等が import 可能・set が pin される）。

## H-0087: leakage validator を public API 化（dead-code 解消）＋空 `lizyml/utils/` 削除

- **ステータス**: Accepted
- **起票日**: 2026-07-03
- **決定日**: 2026-07-03
- **スコープ**: `lizyml/data/__init__.py`（3 validator の re-export + `__all__`）, `lizyml/utils/` 削除, docs（validator の言及追加）。**公開 API の additive 追加のみ**。`Model.fit` への自動配線は行わない（挙動不変）。
- **関連**: [Issue #216](https://github.com/nbx-liz/LizyML/issues/216)。2026-07-02 full-package review（dead-code 判定は cross-check 検証済）。leakage-first charter（CLAUDE.md §0）。

### 目的（課題）

`lizyml/data/validators.py` の 3 つの leakage validator（`validate_time_series_order` / `validate_no_target_leakage` / `validate_group_split`）は `LizyMLError` code とテストを備えた良質なコードだが、`lizyml/` 内に呼び出し箇所が皆無で、`lizyml.data` / top-level からも未 export・docs 未記載＝dead code。leakage-first を掲げる本ライブラリで leakage 検査ツールが利用不能な状態。加えて `lizyml/utils/` は 0 byte の空パッケージで誰も import していない。

### 対応方針（決定）

- **validator を `lizyml.data` の public API として re-export**し、docstring / docs に利用方法を記載する。ユーザーが `from lizyml.data import validate_time_series_order` 等で明示的に leakage 検査を呼べるようにする。**`Model.fit` への自動配線はしない**（既存の通過中 config に警告/例外を新たに出す挙動変更を避けるため。自動配線は将来別 Proposal で検討）。
- **空 `lizyml/utils/` を削除**する（import 参照ゼロを grep 確認済）。

### 代替案（不採用）

- **validator を削除**: charter 上価値ある leakage tooling とそのテストを失うため不採用。
- **`Model.fit` へ自動配線（warn/raise）**: leakage-first に最も合致するが、既存の通過中 config に新たな警告/例外を出す挙動変更（互換性リスク）を伴い、別 Proposal と RED テストが必要。本 Proposal のスコープ外とし follow-up とする。

### 影響範囲 / 互換性

- **純 additive**: `lizyml.data` に 3 シンボルを追加するのみ。既存 import（`from lizyml.data.validators import ...`）は不変。`Model.fit` の挙動は不変。`lizyml/utils/` 削除は参照ゼロにつき無影響。format_version 不変。

### 受け入れ基準（テスト観点）

- `lizyml.data.__all__` に 3 validator が含まれ、`from lizyml.data import ...` で import 可能なことを固定するゴールデンテスト。
- `lizyml/utils/` が存在せず、`import lizyml.utils` が失敗すること（削除の確認）。
- 既存 validator の振る舞いテストは不変で pass すること。

## H-0088: Layer-DAG ドリフトの解消（実在エッジの宣言 + BLUEPRINT §19 / 付録 B 同期）

- **ステータス**: Accepted
- **起票日**: 2026-07-03
- **決定日**: 2026-07-03
- **スコープ**: `ARCHITECTURE.md`（codegen の Layer 配置 + 宣言済みエッジ表）, `BLUEPRINT.md §19`（欠落モジュール追記 + codegen 追加）, `BLUEPRINT.md 付録 B`（H-0074 完了マーク）。**ドキュメント/仕様のみ。コード変更なし**。
- **関連**: [Issue #211](https://github.com/nbx-liz/LizyML/issues/211)。ARCHITECTURE.md §2.1 DAG, H-0052 / H-0054 / H-0073 / H-0074。2026-07-02 full-package review。

### 目的（課題）

宣言された 5 層 DAG（ARCHITECTURE.md / BLUEPRINT §2.1・§19）と実際の import グラフに乖離があり、「IF only / 下方向のみ」レビュールールが該当エッジで機能しない。

1. **eval→training エッジが未宣言**: `evaluation/{evaluator,confusion}.py` が `training/oof_assembly.py` の `compute_oof_valid_mask` を import（Layer 2 内の横依存、H-0052 の副作用）。循環なし。
2. **plots→calibration 具象ディスパッチ**: `plots/calibration.py` が Layer 1 具象 `CalibrationResult` を runtime import + isinstance dispatch（型ディスパッチは本来 Layer 4）。
3. **codegen/ に Layer 未割当**: 4 モジュール（833 行の `templates.py` 含む）が §19 / ARCHITECTURE.md に不在。実質 Layer 3。seam が estimator 固有（`generate_code(..., lgbm_params=...)`）で H-0073 の狙いと不整合。
4. **§19 / 付録 B ドリフト**: §19 が `core/_model_predict.py` / `core/_model_state.py` / `core/types/task.py` / `core/types/target_encoder.py` / `data/validators.py`（H-0087 で public 化）/ `codegen/` を欠く。付録 B が H-0074 FitState 移行を「整備中」と記すが 3 mixin は既に `_get_fit_state()` / `_get_tuning_state()` 使用済（H-0077 完了）。

### 対応方針（決定）

- **実在エッジを仕様に宣言する（型移動は follow-up）**。項目 1–3 のエッジはいずれも循環がなく、既存動作を保ったまま「仕様を実装に合わせる」ことでレビュールールを再び機能させる。ARCHITECTURE.md に codegen（Layer 3）を追加し、宣言済み横断エッジ（eval→training utility、persistence→training `RefitResult`（TYPE_CHECKING）、plots→calibration `CalibrationResult` dispatch）を rationale 付きで明記する。
- **§19 / 付録 B を実装へ同期する**（欠落モジュール追記、H-0074 を完了マーク）。
- **型の再配置は本 Proposal では行わない**（`compute_oof_valid_mask` / `RefitResult` / `CalibrationResult` の `core/types/` 昇格、`templates.py` 分割、codegen の estimator-agnostic 化）。いずれも shared-type contract / DAG に触れる別変更のため follow-up issue とする（codegen seam は既存 [#228](https://github.com/nbx-liz/LizyML/issues/228) を参照）。

### 代替案（不採用）

- **型を Layer 0 へ即時移動**: よりクリーンだが FitResult 契約に触れる shared-type 変更で、複数の import 経路と golden test に波及する。ドリフト解消（レビュールールの再機能化）が目的の本 Proposal では過剰。段階移行のため follow-up に分離。

### 影響範囲 / 互換性

- **コード・公開 API・format_version すべて不変**。ドキュメント/仕様のみ。実装は既に spec が記す実態に一致する方向へ更新するため、以後の DAG レビューが該当エッジで機能する。

### 受け入れ基準（テスト観点）

- ドキュメントのみのため runtime テストなし。BLUEPRINT §19 が実在モジュール（上記 5 + codegen）を網羅し、付録 B が H-0074 を完了として記すこと、ARCHITECTURE.md に codegen と宣言済みエッジが記載されることを目視レビューで確認する。

## H-0089: calibrated OOF metrics の fallback 透明化（CalibrationResult に per-fold fallback フラグ + metrics に fallback-row count）

- **ステータス**: Accepted
- **起票日**: 2026-07-04
- **決定日**: 2026-07-04
- **スコープ**: `lizyml/calibration/cross_fit.py`（`CalibrationResult` に additive フィールド 2 件 + `cross_fit_calibrate` の集計）, `lizyml/core/_model_metrics.py`（`metrics["calibrated"]` に `fallback_row_count` を追加）。
- **関連**: [Issue #218](https://github.com/nbx-liz/LizyML/issues/218)（metrics transparency 項）, H-0058（calibration が outer splits を再利用）, H-0054（calibrated metrics assembly）。2026-07-02 full-package review。

### 目的（課題）

cross-fit calibration には、fold の学習データに **有効な被覆スコアが無い**（例: TimeSeriesCV fold 0 の全 train 行が未被覆 = NaN）か、**単一クラス**の場合、その fold の validation 行に対して calibrator を fit できず、**未校正の生 OOF 確率（`oof_pred`）をそのまま埋める** fallback 経路が 3 つ存在する（`cross_fit.py` の no-covered-train / single-class / partial-NaN-val 分岐）。この fallback は無標識で `calibrated_oof` に混入し、`metrics["calibrated"]["oof"]` は「校正済み確率」と「未校正確率」のブレンド上で計算されるが、**その事実がどこにも surface されない**。行リークではないが（H-0058 で許容済みの挙動）、**metric の誠実性（honesty）**の問題であり、ユーザは校正メトリクスが部分的に未校正であることを知り得ない。

### 対応方針（決定）

1. `CalibrationResult` に **additive** フィールドを 2 件追加する（いずれも default 付きで後方互換）:
   - `fallback_fold_flags: list[bool]`（`default_factory=list`）— split_indices と同順。calibrator を fit できず fold 全体が未校正 fallback になった fold で `True`。
   - `n_fallback_rows: int = 0` — 全 fold 合計で、`cal.predict(...)` ではなく未校正 fallback を割り当てた validation 行数（部分 NaN-val 行を含む）。
2. `cross_fit_calibrate` のループでこれらを集計する（挙動そのものは不変 — 値は既存の fallback 経路をカウントするだけ）。
3. `assemble_calibrated_metrics`（`_model_metrics.py`）が `metrics["calibrated"]` に `fallback_row_count: int`（= `CalibrationResult.n_fallback_rows`）を追加し、校正メトリクスの横に fallback 規模を surface する。fallback が皆無の通常ケースは `0`。

### 影響範囲 / 互換性

- **契約変更（Result shape）**: `CalibrationResult` に 2 フィールド追加、`metrics["calibrated"]` に `fallback_row_count` キー追加。いずれも **additive**。既存の `metrics["calibrated"]["oof"]` / `oof_per_fold` の意味・値は不変（fallback 行は従来どおり blend に含まれる — 本変更は「標識を足す」だけで数値は変えない）。
- **format_version**: 据え置き（`2`）。`metrics` は fit 時に生成され dict として保存される純データで、旧アーティファクトの `metrics["calibrated"]` に本キーが無くても読み込みに支障はない（migration 不要）。`CalibrationResult` の新フィールドは default 付きのため、直接構築するコード（テスト等）も影響なし。旧 pickle の `CalibrationResult` を外部から読む経路では `getattr(cal, "n_fallback_rows", 0)` で防御する。
- **公開 API**: `FitResult.metrics` の shape が additive に拡張される。golden test（calibrated keys / 契約）を更新。

### 代替案（不採用）

- **fallback 行を `calibrated_oof` から除外して NaN 化**: 校正メトリクスから未校正行を完全に排除できるが、`calibrated_oof` の長さ・被覆契約（H-0058: raw OOF と同一被覆）を破壊し、下流の table / plot に波及する破壊的変更。誠実性は「除外」ではなく「標識化」で達成できるため過剰。
- **log のみで surface**: 実行時ログは事後監査に残らず、`FitResult` を受け取る評価コードから参照できない。metrics 契約に載せるのが最小で最も有用。

### 受け入れ基準（テスト観点）

- TimeSeriesCV（fold 0 全未被覆）相当の split で `cross_fit_calibrate` → `fallback_fold_flags[0] is True`、`n_fallback_rows == 当該 fold の fallback 行数`。
- fallback が発生しない通常の binary KFold + platt 構成で `fallback_fold_flags` が全 `False`、`n_fallback_rows == 0`、`metrics["calibrated"]["fallback_row_count"] == 0`。
- 単一クラス train fold を含む split で当該 fold flag が `True`、fallback 行数が一致。
- 既存の calibrated metrics テスト（`"oof" in cal`、`set(cal["oof"]) == set(raw["oof"])`）が引き続き green。

## H-0090: codegen の生成 train.py で time/group split を再現（#206 の shuffle-leak banner 撤去）

- **ステータス**: Accepted
- **起票日**: 2026-07-04
- **決定日**: 2026-07-04
- **スコープ**: `lizyml/codegen/templates.py`（生成 train.py に split 再現ロジックを追加、calibration OOF を再現 fold で生成、covered 行のみで calibrator fit）, `lizyml/codegen/config_writer.py`（config.json に `split` ブロック追加）, `lizyml/codegen/generator.py`（`generate_code(split=...)`、#206 banner/warning 撤去）, `lizyml/core/_model_persistence.py`（`_build_split_metadata(cfg)` で split spec を JSON 化）。
- **関連**: [Issue #228](https://github.com/nbx-liz/LizyML/issues/228)（#206 のフォローアップ）, [#206](https://github.com/nbx-liz/LizyML/issues/206)（最小対応: warn+banner）, H-0058（calibration は outer splits を再利用）, H-0060（BlockedGroupKFold）, [#211](https://github.com/nbx-liz/LizyML/issues/211) item 3（codegen seam）。

### 目的（課題）

#206 は最小対応として、`split.method` が `kfold` / `stratified_kfold` 以外のとき、生成 `train.py` に「shuffled random K-fold で retrain するため time/group 境界を跨いでリークする」旨の `UserWarning` + banner を出すに留めていた。retrain OOF（calibration 用）が実際にシャッフル分割のままで、時系列・グループデータの再学習が**境界を跨いでリーク**する状態は解消されていなかった。

### 対応方針（決定）

生成 `train.py` の calibration OOF CV を、`config.json["split"]` からモデルの `split.method` を**忠実に再現**する形に置き換える。LizyML 非依存を保つため、標準 3 種は sklearn を直接呼び（`time_series`=`TimeSeriesSplit`, `group_kfold`=`GroupKFold`, `stratified_group_kfold`=`StratifiedGroupKFold`）、独自 3 種（`purged_time_series` / `group_time_series` / `blocked_group_kfold`）は LizyML splitter の fold ロジックを pure numpy でテンプレに移植する。

- **ソート再現**: LizyML は分割前にデータを time_col（time系）または blocks.col（blocked）で `argsort()` する。生成側も同じ pandas `argsort()` で並べ替え、fold index を元の行順に戻す。
- **split metadata の JSON 化**: `_build_split_metadata(cfg)` が method 固有パラメータ（gap / purge_gap / embargo / cutoffs / mode / train_window / stratify / shuffle / random_state / min_train/valid_rows / n_splits / time_col / group_col）を**解決済みの値**（`stratify="auto"` は bool 化、`random_state` は `training.seed` へフォールバック）で serialize。テンプレは LizyML ロジック不要。
- **covered 行のみで calibrator fit**: time/group split では first-period / 未被覆行が全 validation fold から漏れて OOF が NaN になり得るため、calibrator は `~isnan(oof)` の covered 行のみで fit する（LizyML cross-fit C_final と同じ）。#206 で shuffled K-fold（全行被覆）だったため顕在化していなかった latent bug を解消。
- **#206 banner/warning 撤去**: 全 method を再現するため不要。

### 影響範囲 / 互換性

- **codegen 出力の意味変更（leakage 境界）**: 生成 `train.py` の retrain CV が shuffled K-fold から実 split に変わる。**リーク解消の是正**であり、`predict.py` / `model.txt` は不変。
- **config.json 追加キー `split`**: additive。旧 export（`split` なし）を読む生成コードは legacy の task-based shuffled K-fold にフォールバックする（後方互換）。
- **format_version**: 据え置き（生成コードは配布物で、`format_version` は Model artifact 側の契約）。
- **公開 API**: `generate_code` の `split_method: str` パラメータを `split: dict | None` に置換（内部 codegen API、外部利用なし）。

### 代替案（不採用）

- **LizyML splitter を生成コードから import**: 生成物の「LizyML 非依存」契約（export skill）を破壊するため不可。
- **元データの split indices を artifact に焼き込み**: retrain は新規データに対して行うため、固定 index では再現できない（split ロジックの移植が必須）。

### 受け入れ基準（テスト観点）

- 6 method（time_series / purged_time_series / group_time_series / group_kfold / stratified_group_kfold / blocked_group_kfold）+ stratified_kfold で、生成 `train.py` の `_resolve_folds(df, y)` が LizyML `build_splitter(...).split(...)` と **fold index 完全一致**（`tests/test_codegen/test_split_reproduction.py`）。
- time_series + calibration の export → 生成 `train()` が再現 fold で end-to-end に retrain し `calibrator.json` を生成（covered 行 fit）。
- `export_code` が対象 method で #206 warning/banner を出さず、`config.json["split"]` に method + 列名を serialize する。

## H-0091: tune() orchestration を writer-exempt な `_model_tuning.py` mixin へ抽出（H-0077 の read-only-mixin 不変条件を明示的に category 分割）

- **ステータス**: Accepted
- **起票日**: 2026-07-04
- **決定日**: 2026-07-04
- **スコープ**: `lizyml/core/model.py`（tune() + 5 helper を撤去、`ModelTuningMixin` を継承）, `lizyml/core/_model_tuning.py`（新規: writer-exempt orchestrator mixin）, `tests/test_core/test_mixin_state_isolation.py`（不変条件の category 分割を文書化 + writer-mixin 判定テスト追加）。
- **関連**: [Issue #237](https://github.com/nbx-liz/LizyML/issues/237)（#209 Phase 3 から deferred）, H-0074 / H-0077（Mixin state isolation）, #209（invariant-safe な部分は抽出済）。

### 目的（課題）

`core/model.py` は 800 行ガイダンスを超過（develop で 1244 行）。残る tuning orchestration（`tune()` + `_validate_tune_inputs` / `_resolve_search_space` / `_maybe_expand_boundary` / `_build_tune_objective` / `_run_tune_round`、約 471 行）が facade に残存している。#209 は invariant-safe な domain logic（`assemble_round_result` / `prepare_for_split` 等）のみ抽出し、orchestration は **H-0077 の read-only-mixin 不変条件と衝突**するため deferred した。

既存 mixin（`_model_plots` / `_model_tables` / `_model_persistence`）は **read-only な post-fit consumer** で、H-0077 により mixin body は `_get_fit_state()` / `_get_tuning_state()` 経由でのみ state を読む（`self._<private>` 直接 access 禁止、`tests/test_core/test_mixin_state_isolation.py` の静的ガードで強制）。一方 `tune()` は **writer**（`self._tuning_result` / `_study` / `_rounds` / `_run_dir` を書き換え、`self._merge_params` / `_build_train_components` を呼ぶ）。frozen state snapshot は writer に使えないため、素朴に mixin へ移すとガードに抵触する。

### 対応方針（決定）— writer-exempt orchestrator mixin category

mixin を 2 カテゴリに明示的に分ける:

1. **Diagnostic mixin（read-only）**: `_model_plots` / `_model_tables` / `_model_persistence`。post-fit consumer。H-0077 の read-only 規則が適用され、静的ガード（`_MIXIN_FILES`）が `self._<private>` を禁止する。
2. **Orchestrator mixin（writer-exempt）**: `_model_tuning`（新規）。`tune()` の mutating lifecycle 中に走る唯一の writer mixin。`self._<private>` への読み書きを許可し、静的ガードの `_MIXIN_FILES` に**含めない**（＝ read-only 規則の対象外であることを列挙で明示）。

これにより H-0077 の read-only 不変条件は diagnostic mixin に対して**無傷**のまま、orchestration を facade から分離して `model.py < 800 行` を達成する。純 refactor（振る舞い不変）。

### 不変条件（INV, testable）

- **INV-1**: diagnostic mixin（`_model_plots` / `_model_tables` / `_model_persistence`）は Model body の private state を書かない。読むのは `_get_fit_state()` / `_get_tuning_state()` 経由のみ。— 違反シナリオ: diagnostic mixin に `self._tuning_result = ...` が現れる → 静的ガードが検出。
- **INV-2**: writer mixin は **ちょうど 1 つ**（`_model_tuning`）で、writer-exempt として列挙される。diagnostic ガードの `_MIXIN_FILES` に `_model_tuning` は現れない。— 違反シナリオ: `_model_tuning` を `_MIXIN_FILES` に追加 → tune() の writer access を誤検出、または writer mixin が 2 つに増える。
- **INV-3**: `tune()` の抽出は振る舞いを変えない（OOF/best_params/rounds/study が抽出前後で一致）。— 違反シナリオ: 移動中に self._ 参照を取りこぼす → tuning E2E が fail。

### 失敗パス（Failure Paths Covered）

- **normal**: tune() → best_params/rounds/study が populate、後続 fit() が best_training_params 適用。
- **resume**: `tune(resume=True)` で既存 study を継続。
- **boundary expand**: `_maybe_expand_boundary` が探索空間境界を拡張する round。
- **exception**: tune 中の trial 例外は既存の callback（RuntimeWarning）で握られ、study は継続。
- **前提未達**: fit 前に diagnostic mixin メソッド呼び出し → `MODEL_NOT_FIT`（既存契約、不変）。

### 影響範囲 / 互換性

- **公開 API・FitResult・Artifacts・format_version すべて不変**。純 refactor。`Model` の MRO に `ModelTuningMixin` が加わるのみ（`tune()` は同じシグネチャで公開され続ける）。
- **BLUEPRINT §19 / ARCHITECTURE.md**: mixin 一覧に `_model_tuning`（orchestrator/writer category）を追記。

### 代替案（不採用）

- **案 A: H-0077 を緩めて「writer-exempt」フラグを diagnostic ガードに追加**: ガードのロジックが複雑化。ファイルリスト方式（diagnostic のみ列挙）の方が単純で、writer mixin は「リストに無い」ことで自然に exempt になる。
- **tune() を facade に残す**: model.py < 800 行が達成できず #237 未解決。

### 受け入れ基準（テスト観点）

- `core/model.py` < 800 行。
- 全 tuning + retune E2E green（振る舞い不変）: `test_tuning/` 一式、`test_tune_fit_identity.py`、`test_retune.py`、`test_tuning_persistence.py`。
- 静的ガード（`test_mixin_state_isolation.py::TestMixinPrivateAccessGuard`）が 3 diagnostic mixin で引き続き pass。
- 新規テスト: `_model_tuning.py` が `_MIXIN_FILES`（read-only ガード対象）に**含まれない**こと、`ModelTuningMixin` が `Model` の base に含まれ `tune` を提供することを assert。

## H-0092: §10.3.3 を H-0085 に追随させ、inner valid の gap 継承を「自動解決時のみ」に明文化（#265 / #266）

- **ステータス**: Accepted
- **起票日**: 2026-09-06
- **決定日**: 2026-09-06
- **スコープ**: `BLUEPRINT.md`（§0.1 に `ARCHITECTURE.md` の位置づけを追記、§10.3.3 全面改訂、§8.2 の shuffle 禁止を outer split に限定）, `ARCHITECTURE.md`（冒頭に派生文書である旨を明記、`format_version` 3 箇所 1→2）, `lizyml/training/inner_valid.py`（`BlockedGroupInnerValid` の回帰フォールバック先を修正、`StratifiedTimeHoldoutInnerValid` に空 train の fail-fast を追加）, `tests/test_docs/test_declared_versions.py`（新規）, `tests/test_training/test_inner_valid_purge_embargo.py` / `tests/test_training/test_blocked_group_inner_valid.py`（pin を追加）, `pyproject.toml`（`[tool.ruff] exclude` に `docs/audits/**` を追加し、併せて `force-exclude = true` を設定）, `docs/audits/2026-09-defect-discovery/`（調査 run の恒久アーカイブ）。
- **関連**: [Issue #265](https://github.com/nbx-liz/LizyML/issues/265), [Issue #266](https://github.com/nbx-liz/LizyML/issues/266), H-0085 / #212（gap 伝播の決定）, H-0060（§10.3.3 の初版）, H-0070（`format_version` 1→2）。

### 目的（課題）

**#265 は「2 つの規則が矛盾している」ではなく、H-0085 が片側だけ更新した取り残し（DC3）だった。**

`git blame` が経緯を確定させる。§10.3.1 L602 は `deacc9ee`（2026-07-02, H-0085）で「gap は inner valid に伝播する」に改訂された。一方 §10.3.3 の該当行は `f846ab2c`（2026-03-21, H-0060）のままで、「outer CV が purge / embargo を持っていても inner valid は追加の purge / embargo を持たない」という**改訂前の規則**を今も述べている。H-0085 の決定文自身が「§10.3.1 L602 を改訂」とだけ書いており、姉妹節の存在が見落とされた。

したがって修復は判断ではなく機械的な追随である。ただし取り残された同じブロックは、実装との乖離を他に 4 件抱えていた。

| BLUEPRINT の記述 | 実装 |
|---|---|
| `TimeHoldoutInnerValid(ratio)` | `__init__(self, ratio=0.1, gap=0)` |
| `n_valid >= n_samples` で `ValueError` | `n_valid + self.gap >= n_samples`（`training/inner_valid.py:195`） |
| `BlockedGroupInnerValid(ratio)` | `__init__(self, ratio=0.1, task="regression")` |
| `HoldoutInnerValid(ratio, stratify=False, random_state)` | `__init__(self, ratio=0.1, random_state=42, stratify=False)`（引数順が逆） |

姉妹エントリは構築子引数を列挙しているため、この省略は「引数が無い」という積極的な主張として読める。

**#266**: `ARCHITECTURE.md` は `format_version` を 3 箇所（:48 / :487 / :646）で 1 と述べていたが、`persistence/exporter.py:38` は H-0070 以来 2 である。3 箇所が揃って古びたまま CI は緑だった。文書が述べる定数をコードの定義と突き合わせる仕組みが無かったためで、3 箇所を手で直してもクラスは閉じない。

### 対応方針（決定）

1. **§10.3.3 を H-0085 に追随させる。** 5 つの strategy すべてを実装どおりの構築子シグネチャ（既定値つき）で記述し、`TimeHoldoutInnerValid` に gap 対応の分割規則と実際の `ValueError` 条件を書く。
2. **gap 継承を「自動解決時のみ」と明記する。** §10.3.1 が既に「明示指定した場合は外側 `split.method` を参照しない」と決めており、実装もそうなっている（`core/_model_factories.py:321→297` の自動経路は gap を渡し、`:356` の明示経路は渡さない）。§10.3.3 はこの限定を書いていなかったため、両節が同一入力に対して異なる予測をしていた。
3. **伝播量は「outer split の境界 gap」という概念で書き、`purge_gap + embargo` という合成では書かない。** 現行実装の合成（`_auto_inner_gap`）は保守的な選択であって必要量ではない。`embargo` が `purge_gap` と同じ valid 前 gap を動かしている件（前方連鎖 splitter では valid 後に train 行が置かれない）は別途の未起票事項であり、合成を BLUEPRINT に固定すると、その決着で仕様が再び古びる。概念で書けばどちらに転んでも真のままとなる。
4. **§8.2 の「shuffle 禁止」を outer split に限定する。** §10.3.1 L599 は「時間順の outer split に対して shuffle する inner split を明示指定した場合、警告した上で明示指定を尊重する」と決めており、実装も `UserWarning` を発する（`_model_factories.py:334-343`）。§8.2 の無限定な「禁止」はこれと衝突していた。
5. **`ARCHITECTURE.md` は派生文書であり規範性を持たないことを `BLUEPRINT.md` §0.1 に明記する。** 併せて `ARCHITECTURE.md` 冒頭にも同じ断りを置き、3 箇所の `format_version` を 2 に直す。

   当初は `CLAUDE.md` §1 の優先順位表に追記する想定だったが、**このリポジトリの `CLAUDE.md` は `.gitignore:156` で除外された追跡外ファイル**であり、PR に含められない。同様に §1 が 3 位に挙げる `AGENTS.md` もリポジトリに存在しない。順位の宣言は、追跡されておりかつ `ARCHITECTURE.md` を上位から規定できる `BLUEPRINT.md` に置くのが正しい。

6. **§10.3.3 の「回帰では `TimeHoldoutInnerValid` と同等」を実装側で真にする（本 Proposal で唯一の実装変更）。**

   文書を実装に追随させる作業中に、この一文だけは**実装が偽にしていた**ことが分かった。`BlockedGroupInnerValid.split` はグループ数 < 4 のとき `task` を見ずに `StratifiedTimeHoldoutInnerValid` へフォールバックする（`training/inner_valid.py:323`）。回帰の `y` は連続値なので 1 行 1 クラスとなり、クラスごとの末尾選択が全行を validation に入れ、**inner-train が空**になる。実行で再現した（`n=6, ratio=0.3`: `train=[] valid=[0..5]`。同条件の `TimeHoldoutInnerValid` は `train=[0..4] valid=[5]`）。

   ここで文書側を実装に合わせる（「回帰では層化フォールバックする」と書く）選択は取れない。空の inner-train は early stopping を成立させない不具合であって、記述すべき仕様ではないため。したがって修復は 2 点：

   - `BlockedGroupInnerValid` のフォールバック先を `task` で分岐させ、`regression` は `TimeHoldoutInnerValid` に送る（警告文もフォールバック先を正しく名乗る）。
   - `StratifiedTimeHoldoutInnerValid` 自体を、train が空になる `y` に対して `ValueError` で fail-fast させる。分岐だけでは、この strategy を直接指定した経路が同じ形で壊れたままになる（DC1: 黙って空の train を返す）。

   §10.3.3 は、この分岐が `task` で決まること、および層化不能な `y` が拒否されることを明記する。

### 影響範囲 / 互換性

- **公開 API・Config・FitResult・PredictionResult・Artifacts・`format_version` すべて不変。** 実装変更は決定 6 の 1 点のみで、公開シグネチャも Result の shape も変わらない。
- **振る舞いの変化は 2 つ、いずれも従来が壊れていた入力に限られる。** (a) `blocked_group_kfold` + 回帰 + グループ数 < 4 のとき、inner valid が空 train ではなく時間順 holdout になる。(b) `StratifiedTimeHoldoutInnerValid` を層化不能な `y` に直接適用すると、空 train を返す代わりに `ValueError` を送出する。どちらも従来は early stopping が成立しない状態であり、正常動作していた入力の結果は変わらない。
- §10.3.3 の gap 記述は H-0085 で既に決定済みの内容の明文化であり、新たな決定ではない。「明示指定時は継承しない」という限定のみが新しく書かれた文であり、これは実装の既存挙動と一致する（下記テストで固定）。
- `ARCHITECTURE.md` は本 Proposal 以降、`BLUEPRINT.md` から派生した説明資料として扱う。矛盾時は `BLUEPRINT.md` が正。

### 代替案（不採用）

- **案 A: §10.3.1 を §10.3.3 に合わせる（伝播しない方に戻す）。** H-0085 の決定を無効化することになる。H-0085 は look-ahead 防止という根拠を持ち、実装・テストとも伝播側で入っている。文書の古い側に実装を合わせるのは順序が逆。
- **案 B: 伝播量を `purge_gap + embargo` と明記する。** 現行実装と一致するが、`embargo` の意味に関する未決着の論点を仕様に固定してしまう。決着すれば BLUEPRINT が再び古びる。
- **案 C: `ARCHITECTURE.md` の 3 箇所を直すだけにする。** 3 箇所が揃って古びた原因（文書の定数を誰も検査していない）が残る。個別事例を閉じてクラスを開いたままにするのは、この監査が繰り返し見つけた形。
- **案 D: `ARCHITECTURE.md` を `BLUEPRINT.md` から自動生成する。** 本筋だが規模が別物。まず順位を確定させ、検査で drift を落とす。
- **案 E: 監査アーカイブの計測スクリプトを ruff に合わせて整形する（`exclude` を足さない）。** アーカイブは「その数値をどう測ったか」の証拠であり、整形は証拠の編集にあたる。`pyproject.toml` には既に同じ理由の除外が 3 件ある。同じ理由で `test_declared_versions.py` の `EXCLUDED_DIRS` からも外している。
- **案 F: `exclude` だけを足す（`force-exclude` を設定しない）。** これは実際に一度書いて落ちた。**ruff はコマンドラインに明示的に渡されたパスに `exclude` を適用しない**ため、`ruff check .` を走らせる CI では効き、staged ファイルを名前で渡す `.githooks/pre-commit` では効かない — 同じ宣言が呼び出し側によって別の意味になる（DC4 の形）。`force-exclude = true` を設定すると呼び出し方に依らず宣言が正となる。stdin-filename を使った両方向の対照で確認済み: `lizyml/` 配下の名前では未使用 import が 2 件検出され、`docs/audits/` 配下の同一内容では除外される。なお `.githooks/pre-commit` は既存 3 件の除外をシェル側にも重複して持っており（DC3）、`force-exclude` はその重複を将来解消できる前提条件でもある。

### 受け入れ基準（テスト観点）

- **`tests/test_docs/test_declared_versions.py`（新規）**: 現行仕様を述べる文書（`ARCHITECTURE.md` / `BLUEPRINT.md` / `README.md` / `docs/**`）が述べる `format_version` / `config_version` の値が、コードの定義（`persistence/exporter.py:FORMAT_VERSION`, `config/loader.py:SUPPORTED_CONFIG_VERSIONS`）と一致すること。修正前は `ARCHITECTURE.md` の 3 箇所で **RED**、修正後 green。
  - 追記型の記録（`HISTORY.md` / `CHANGELOG.md` / `PLAN.md`）は意図的に母集団外とし、除外理由を明記した上で**除外先ファイルの存在も検査する**（改名で母集団が黙って広がらないため）。監査アーカイブ（`docs/audits/`）も同様に除外する（報告対象の古い値をそのまま引用しているため）。
  - 検査が 1 件も一致しなくなった場合に vacuous pass しないよう、最小サイト数のガードに加え、**中心文書ごとに 1 サイト以上の寄与を要求する**（`ARCHITECTURE.md` だけが走査から外れても件数では気づけないため）。
  - **文法は accept / reject で閉じる。** 正規表現が突合するのは「定数名＋区切り」の**接頭部だけ**とし、行の残り全体を 3 分類する。トークンを取る書き方だとパターンが停止した先が不可視になり、境界が閉じない。accept（value）: 数字列の**直後**が行末・空白・囲み記号（`"` `` ` `` `）` `）` `」` `。` 等）のいずれかであること。type annotation: 残り全体が許可済みの型名 1 個であること（mermaid クラス図の `+config_version: int`。綴りを列挙し、未使用の綴りが残らないことも検査する）。reject: それ以外すべてを**失敗として報告する（skip しない）**。
  - 反例で検証する: `2bogus` / `2-bogus` / `2.5` / `02x` / 値なし / `two` はすべて reject、`config_version: int = 999` は初期化子を無視して annotation と読まずに reject、1 行に 2 つの宣言がある場合は両方を個別に分類する。値パターンに終端境界が無いと `2bogus` を `2` と読んで clean を報告し（DC2）、読めない宣言を黙って落とすと「見ていない」を「問題なし」と報告する（DC1）。
- **`tests/test_training/test_inner_valid_purge_embargo.py`（追加）**: 同一の outer 設定（`purged_time_series`, `purge_gap=3`, `embargo=2`）に対し、自動解決の inner valid は `gap == 5`、`inner_valid` を明示指定した inner valid は `gap == 0` になること。§10.3.3 が新たに書いた限定文に対応する pin であり、既存の自動解決テストはこの区別を検査していなかった。
- **`tests/test_training/test_blocked_group_inner_valid.py`（追加, `TestRegressionFallbackIsTimeOrdered`）**: 決定 6 の回帰テスト。(1) `BlockedGroupInnerValid(task="regression")` がグループ数 < 4 の連続値 `y` で**空でない** inner-train を返し、その分割が同一 `ratio` の `TimeHoldoutInnerValid` と一致すること（修正前は `train=[]` で **RED**）。(2) 警告文がフォールバック先として `TimeHoldout` を名乗ること。(3) `StratifiedTimeHoldoutInnerValid` を層化不能な連続値 `y` に直接適用すると `ValueError` になること。
- 既存の `TestTimeHoldoutGap` / `TestAutoResolvePropagatesGap`、および分類タスクの `BlockedGroupInnerValid` フォールバック既存テストは不変のまま green。

## H-0093: LightGBM パラメータ名の境界を閉じる（#261 / #262 / #268 の一部）

- **ステータス**: Accepted
- **起票日**: 2026-09-07
- **決定日**: 2026-09-07（PR [#275](https://github.com/nbx-liz/LizyML/pull/275) merge。外部レビュー 6 ラウンド、blocking 4/4/2/1/1/0）
- **スコープ**: `lizyml/estimators/provider.py`（`EstimatorProvider` に受理名を問う 2 メソッドを追加 = **公開 Protocol の変更**）, `lizyml/estimators/lgbm/provider.py`（実装）, `lizyml/estimators/lgbm/smart_params.py`（`feature_weights` → `feature_contri`）, `lizyml/core/_model_factories.py`（名前検証関数）, `lizyml/core/model.py`（`_merge_params` から呼ぶ）, `lizyml/core/_model_tuning.py`（study 開始前に探索空間を検査）, `lizyml/core/_model_persistence.py`（`export_code` の生成 params を検査）, `BLUEPRINT.md` §5.3 / §14.4, `tests/test_estimators/test_lightgbm_parameter_names.py`（新規）, `tests/test_tuning/test_search_space_name_validation.py`（新規）, `lizyml/calibration/isotonic.py`（自身が消費する名前の宣言）, `tests/test_calibration/test_calibration_param_names.py`（新規）, `tests/_ast_scan.py`（新規: LightGBM の束縛名を import から解決する走査ヘルパ）, `tests/test_estimators/test_param_behavioral_effect.py` / `tests/test_core/test_config_propagation.py`（既存テストの書き換え）。
- **関連**: [Issue #261](https://github.com/nbx-liz/LizyML/issues/261), [#262](https://github.com/nbx-liz/LizyML/issues/262), [#268](https://github.com/nbx-liz/LizyML/issues/268), H-0053（`EstimatorProvider` 導入）, H-0036（ratio params）。

### 目的（課題）

LightGBM に渡す名前を誰も検査していない。綴りを間違えた、あるいは存在しない名前は `lgb.train` に転送され、**LightGBM に黙って捨てられる**。`verbose=-1`（既定）が LightGBM 自身の警告を抑止するため、利用者には何も起きない。

**#261 — `feature_weights` は出荷以来ずっと無効だった。** これは「キー名の綴り違い」ではなく、**公開機能が一度も動作していなかった**という事実である。LightGBM 4.6.0 が定義する名前は `feature_contri` で、`feature_weights` は正式名にもエイリアスにも存在しない（LightGBM 自身の `LGBM_DumpParamAliases` で確認: 140 canonical / 307 aliases 込み）。

同一データ・同一 seed・`f0` の重みを 0.0 にして測定した結果:

```
baseline            : ['f1', 'f0', 'f2', 'f3', 'f4']   f0 gain 2704.69
feature_weights=... : ['f1', 'f0', 'f2', 'f3', 'f4']   f0 gain 2704.69   ← baseline と完全同一
feature_contri=...  : ['f1', 'f2', 'f3', 'f4', 'f0']   f0 gain    0.00   ← 効く
```

`BLUEPRINT.md:1425` は「`feature_weights` → importance 順序変化」を**検証すべき不変条件として宣言している**。この宣言は機能の全期間にわたり実装に対して偽だった（DC7）。既存の `TestFeatureWeightsE2E::test_feature_weights_applied` が「2 つの列名が importance dict に存在すること」しか見ておらず、重みの有無に関わらず成立する主張だったため検出されなかった。

**#262 — `tuning.optuna.space` の `category: model` 名を誰も検査していない。** 探索空間に存在しない名前を書くと、Optuna はその次元をサンプリングし、値は `lgb.train` に渡り、捨てられる。試行は完了し、スコアが出て、その次元は結果に何の影響も与えない。

**同じ穴が `model.params` にもある。** `model.params` は `tuning.optuna.space` の fit 側の双子であり、書いた内容はそのまま `lgb.train` に転送される。実行で確認済み: 存在しない名前を含む params で `lgb.train` はエラーを出さず 3 本の木を学習した。片方だけ塞いで双子を開けたままにするのは #270 が指摘する形そのものなので、同一のゲートで両方を塞ぐ。

### 対応方針（決定）

1. **`feature_contri` を発出する。** Config フィールド名 `feature_weights` は据え置き、`smart_params.py` が LightGBM に渡すキーだけを変える。`feature_pre_filter = False` の強制は現行どおり維持する。
2. **`EstimatorProvider` に「受理する名前」を問うメソッドを 2 つ追加する（公開 Protocol の変更）。**
   - `accepted_model_param_names() -> frozenset[str]`: その学習器がネイティブに受理する名前。
   - `smart_param_names() -> frozenset[str]`: smart parameter 名。誤って `category: model` や `model.params` に書かれた場合に**別のメッセージ**を返すために要る（利用者の意図はほぼ確実に smart 指定である）。

   LGBM 実装は前者を `LGBM_DumpParamAliases` から**モジュール読み込み時に 1 度だけ導出**する。手書きのリストにすると LightGBM の更新で名前が増減したときに黙って古びる（DC3）— それはこのゲートが検出するはずの欠陥そのものである。後者は `_SMART_PARAM_NAMES` の 1 タプルを正とし、`extract_smart_params` も同じタプルから構成する（両者が別々に名前を並べると同じ DC3 になる）。
3. **検証は Layer 4（Facade）に置く。`config/schema.py` には置かない。**

   `ARCHITECTURE.md` の DAG で `config/` と `estimators/` は**どちらも Layer 1 の兄弟**であり、Layer 1 は Layer 0 のみを参照できる。config 層から provider を import するのは層違反になる。したがって検証は、config と provider が**規則上はじめて出会える場所** = Facade で行う。エラーコードは `CONFIG_INVALID`、いかなる学習も始まる前に落ちる。
4. **発火点は「学習器に渡す直前」であり、構築時ではない。**

   当初は `Model.__init__` に置いたが、独立レビューが実行で 2 つの穴を示した。どちらも構築時検査という選択そのものの帰結である。

   - **呼び出し側は config の参照を持ち続ける。** `Model` は渡された `LizyMLConfig` をコピーせず保持するため、構築後に呼び出し側が書き換えた内容がそのまま学習器に届く。実測: `Model(cfg)` 構築後に `cfg.model.params` へ不正名を足して fit すると、3 回の `lgb.train` すべてにその名前が渡った。
   - **`best_model_params` は `__init__` の後に載る。** artifact から復元された tuned params は構築を経由しないため、構築時検査は原理的に届かない。実測: 復元後の再 fit で 3 回とも渡った。

   したがって検証は 4 か所で行う。いずれも同一関数の呼び出しであり、判断が分かれる複数の門ではない（4 つ目は決定 6 でレビュー中に加わった）。

   | 呼び出し点 | 覆う経路 |
   |---|---|
   | `_merge_params` のマージ後 | config の `model.params`、構築後に変更された config、artifact から復元された `best_model_params` |
   | tuning study 開始前 | `category: model` の探索空間。trial params はこの名前から生成されるため、空間を覆えば trial も覆う |
   | `export_code` の生成直前 | 生成される `train.py` が `CFG["lgbm_params"]` として `lgb.train` に渡す params。これは**学習済み adapter から**取られるため config 側のどちらの門も見ておらず、かつ `load()` を通した古い artifact がそのまま流れ込む |
   | 学習を始める各入口の先頭（`_fit_impl` と tuning の `_merge_params` 直後） | `calibration.params`。calibrator の Booster に渡る名前であり、config 側のどの門も見ていない。決定 6 を参照 |

   **`fit(params=...)` はこの一覧に含まれない。** 現在の `Model.fit` は `params` を `_merge_params` に転送していないため、この引数は**そもそも何にも効いていない**（実測: `{"not_a_lightgbm_parameter": 7, "num_leaves": 2}` を渡しても `lgb.train` 3 回のいずれにも届かず、`num_leaves` は既定値のままだった）。転送は #264 / PR 2 の題材であり、検証はすでに `_merge_params` の返り値に置かれているので、PR 2 は引数を配線するだけでこの経路も覆われる。
5. **`Model.load()` は検証しない。** 保存済み artifact は「実際に起きた fit の記録」であり、綴り違いを含む古い artifact の読み込みを拒否しても誰の役にも立たない。当初は決定 4 でそう書きながら**実装は逆だった** — `load()` は `cls(config)` を通るため、構築時検査は古い artifact の読み込みを拒否していた。発火点を移した結果、`load()` は成功し、その artifact から**再学習しようとした時点で**落ちる。これは正しい非対称性である: 記録は読めるべきだが、何も起こさない名前で学習を始めるべきではない。
6. **`calibration.params` も同じゲートで塞ぐ（レビュー round 3 で発見、当初スコープ外）。**

   `IsotonicCalibrator` は `calibration.params` を `_ISOTONIC_DEFAULTS` に上書きマージして `lgbm.train` に渡す。すなわち **`model.params` と全く同じ silent-discard がこの面にもある**。実測で確認済み: `IsotonicCalibrator({"num_leave": 7})` は例外を出さず、その名前がそのまま `lgbm.train` に届いた（`tests/test_calibration/test_calibration_param_names.py::test_the_calibrator_itself_forwards_an_unknown_name` が negative control としてこれを固定する）。

   これは当初 3 経路を塞いだ**後**に見つかった 4 つ目の経路であり、見落としの原因は経路の走査そのものにあった（決定 8）。PR 1 の受け入れ基準が「LightGBM が捨てる名前は拒否される」である以上、自身が列挙した経路を 1 つ未ゲートのまま出荷すると宣言が偽になる（DC5）ため、繰り延べずここで塞ぐ。

   3 点だけ `model.params` と扱いが異なる。

   - **対象は `method` が LightGBM を使う calibrator に限る。** `platt` / `beta` は numpy と scipy で fit し LightGBM に触れないため、その params を LightGBM の登録表で判定すると正当な config を誤って拒否する。どの method が該当するかは散文で宣言せず**走査する**: `test_lgbm_backed_calibrators_matches_the_scan` が登録済み calibrator の各モジュールの import を読み、`LGBM_BACKED_CALIBRATORS` と一致しなければ落ちる。`isotonic` 自身が未ゲートのまま出荷されたのと同じ経緯を、次の calibrator で繰り返さないための機構である。
   - **calibrator 自身が消費する名前は受理する。** `num_boost_round` / `validation_ratio` / `min_data_in_leaf_ratio` の 3 つは LightGBM に転送される前に取り出される。宣言はそれらを pop するコードの隣（`isotonic.py` の `CALIBRATOR_OWN_PARAM_NAMES`）に置き、Facade は `check_param_names(..., extra_accepted=...)` で受け取る。各面の例外がゲート関数側に溜まると、どの面の話なのか分からなくなるためである。
   - **発火点は `_merge_params` の直後であって `_run_calibration` ではない。** 当初は calibrator を組み立てる直前に置いた。「使用点で検査する」という決定 4 の方針には合っているが、`_run_calibration` は**外側 CV が終わった後**に走るため、絶対に使えない config のために全学習を先に払うことになる。レビュー round 4 が実行で示した: 不正な `num_leave` を置いた fit は、`CONFIG_INVALID` になる前に base model を 2 本学習していた。決定 4 と BLUEPRINT §12.2 の「学習が始まる前に落ちる」という記述が、この面に対してだけ偽だったことになる（DC5）。

     順序そのものをテストで固定する: `test_unknown_calibration_param_is_refused_before_any_training` は、不正名が `lgbm.train` に届かないことに加えて **`lgb.train` の呼び出し回数が 0 であること**を主張する。前者だけなら欠陥のある配置でも成立していた（実際していた）ため、順序は別の主張として要る。修正前は RED（Booster 3 本）。

     **さらにその修正も、入口を 1 つしか覆っていなかった。** レビュー round 5 が実行で示した: `tune()` は自前の入口を持ち、`model.params` と探索空間は検査するが `calibration.params` は問わない。したがって `tune()` → `fit()` の順で使うと、`fit()` が拒否する config が **study を丸ごと完走してから**落ちる（実測 `calls_after_tune=2`）。上の順序テストは直接 `fit()` しか通らないため green のままだった。

     この検査を入口で見落としたのは**これで 2 度目**である（1 度目は構築時に置いていたこと）。したがって修正は「もう 1 か所呼ぶ」ではなく、**学習する公開入口を列挙してそれぞれを実行で確かめる**形にする: `TRAINING_ENTRY_POINTS`（`fit` / `tune`）を parametrize し、各入口が「拒否が先、学習が後」であることを主張する。`predict` / `export_code` は学習しないので含めない。修正前は `tune` のセルだけが RED、`fit` のセルは green — 見落としの形そのものである。

     **ただし列挙しただけでは軸は閉じない。** 手書きの 2 件は何とも突き合わされておらず、将来学習する公開メソッドが増えても何も落ちない。これは本 PR が 3 ラウンドかけて学んだ「宣言した fixture は突き合わせた軸しか閉じない」形そのものなので、`NON_TRAINING_ENTRY_POINTS`（残り 21 件、各々に理由）を置き、**両者が `dir(Model)` の公開 callable を過不足なく分割すること**を主張する。母集団は `Model` の公開面から導出され、未分類のメソッドは失敗になる。分類が偽でないことも実行で確かめる: 非学習と宣言した各メソッドを未 fit の `Model` に対して呼び、Booster が 1 本も学習されないことを spy で主張する。両方向の RED 確認済み（分類漏れ / 実在しない名前）。

     **その実行確認自体が、3 件で空回りしていた。** 引数なしで `getattr(model, name)()` を呼ぶため、必須引数を持つメソッドは束縛の時点で `TypeError` になり本体が 1 行も走らない。spy が空なのは「学習しなかった」からではなく「何も実行されなかった」からで、緑のセルが何も検査していない — DC1 そのものである。実測 21 件中 3 件が該当し、うち 2 件は round 1/2 で実欠陥が見つかった `load` / `export_code` だった。修正は引数表 `NON_TRAINING_ENTRY_POINT_ARGS` を置き、**呼び出しが束縛できることを先に主張する**形にする（束縛失敗は空回りの明示メッセージで fail）。これで将来必須引数を持つ公開メソッドが分類された場合も、黙って緑にはならない。3 件とも本体に到達することを確認済み（`MODEL_NOT_FIT` / `MODEL_NOT_FIT` / `DESERIALIZATION_FAILED`）、および 3 件それぞれの引数を外すと当該セルが RED になることを確認済み。

   - **`seed` は例外に含めない。** `__init__` で pop されるが `merged["seed"]` で戻されるため実際に Booster に届き、かつ LightGBM が知っている名前なので基底の登録表がすでに受理する。pop されているという理由だけで例外表に足すと**偽の宣言**になる。この 3 名 / `seed` の区別は散文ではなく実行で固定されている: 宣言された各名前が本当に `lgbm.train` に到達しないこと、`seed` は到達すること、をスパイで主張する。

7. **例外表（exemption table）は出荷しない。** 中身が空の許可リストは何も許可しない経路であり DC6 になる。LightGBM が定義しない唯一のキー（`feature_weights`）への対処は「正しい名前を出す」ことであって「間違った名前を免除する」ことではない。免除が本当に必要になった PR が、その理由と自身の計測値を伴って表を導入すればよい。

8. **「どの経路を覆っているか」は散文で主張せず、走査した母集団として持つ。**

   レビュー 3 ラウンドが経路主張を 3 回反証した。1 回目「2 面」、2 回目「`CFG[...]` は覆われている」、3 回目は主張ではなく**走査そのもの**の欠陥だった。走査は受け手の名前を `{"lgb", "lightgbm"}` と直書きで照合していたため、`import lightgbm as lgbm` と書く `calibration/isotonic.py` が**丸ごと不可視**であり、走査は「経路は 3 つ、全て覆われている」と報告していた（DC1）。同じラウンドで、走査が adapter の構築（扉）だけを見て `build_estimator_factory(params=...)` の**呼び出し側（生産者）**を見ていないことも示された — 新しい呼び出し側を足しても検出は 0 件で、「経路が増えればテストが落ちる」という主張自体が偽だった。

   したがって `ESTIMATOR_ROUTES`（`tests/test_estimators/test_lightgbm_parameter_names.py`）は次の形を取る。

   - 受け手の名前は**各モジュール自身の import から解決する**（`tests/_ast_scan.py`）。慣例名 `{"lgb", "lightgbm"}` との和集合を取るのは、import が別の文字列定数にあるテンプレート断片で false-clean にならないためであり、過検出は loud failure にしかならない。
   - **生産者も走査対象**に含める。
   - 走査が本当に新しい経路を検出することを、**注入したソースに対する実行で**主張する（`HOSTILE_ROUTE_SHAPES`、10 ケース: エイリアス import / `from lightgbm import` / 改名付き `from` import / サブモジュール修飾 `lgb.basic.Dataset` / factory 呼び出し 2 形 / adapter 構築、および negative control 3 件 = `not_lgb.train` / `params=` の無い factory 呼び出し / 根が module でない `self.lgb.train`）。ここが散文だったことが、上記 2 つの見落としを許した箇所である。サブモジュール修飾の形は本ラウンドで走査を自分で攻撃して見つけた 3 つ目の false-clean であり、指摘されたものではない。

   走査後の経路母集団は **6 件**（当初の宣言は 3 件）。内訳は adapter の `train` / `Dataset`、adapter 構築（扉）、`build_estimator_factory` 呼び出し（唯一の生産者）、calibrator の `train` / `Dataset`。

   **この走査が主張する範囲を明示的に狭める。** Python の呼び出し文法は開いており（`getattr(lgb, "train")` / `functools.partial` / ディスパッチ表）、AST walker でそれを閉じることはできない。実際、本ラウンドで走査を 3 か所直したうち、**実在の欠陥を見つけたのは alias 解決の 1 件だけ**である（生産者走査は指摘された仮想形、サブモジュール修飾は自分で生成した仮想形を塞いだもの）。形を 1 つずつ追いかければ装置だけが際限なく育つ — 現に走査系は 557 行あり、本 PR の出荷差分 299 行を超えている。

   したがって走査の役割は「**明日の平凡な新しい呼び出し側を loud にする tripwire**」であって、「今日のコードが覆われていることの証明」ではない。後者の根拠は別にある: 門が使用点に置かれていること、H-0093 の実測 firing rate、そして実際の fit が実際の `lgb.train` に渡した内容を読む `test_train_param_names_are_lightgbm_names` である。以後のレビューで「この走査が見落とす呼び出し形がある」という指摘は、**その形が実際にツリーに存在する場合にのみ** blocking として扱う。

### Conditional-Activation Evidence

本 Proposal は `allow` 目的の条件（受理する名前だけを通す）を導入するため、`change-gate.md` により実装前の計測が要る。母集団は**出荷済みテストスイートが実際に構築する `LizyMLConfig` 全件**（静的な dict リテラル走査では `tests/_helpers.py` が override から組み立てる大半を取りこぼす）。`1d7c4e2` で再計測した:

```
Firing rate: 0/52 of configs carrying a tuning search space (#262 / PR 1; recorded by instruments/firing_rate_plugin.py over the shipped suite at 1d7c4e2)
Firing rate: 0/736 of configs carrying model.params (#262's twin surface / PR 1; same recording)
Firing rate: 0/3 of configs carrying calibration.params (#262's calibration surface / PR 1; recorded by a scratchpad pytest plugin over the shipped suite at 1d7c4e2)
```

875 configs 中、探索空間を持つものが 52、`model.params` を持つものが 736、`calibration` ブロックを持つものが 94、うち `calibration.params` を持つものが 3。**ゲートが拒否するものは 0 件**である（探索空間 / `model.params` の 2 行は 824 configs を母集団として先に計測し、calibration 面は決定 6 の判断のために後から同じ方法で計測した。母集団の差はその間に足したテストの分である）。`allow` 目的のゲートとしてこれは正常な結果であり（出荷済みコーパスに不正な名前が無いことを意味する）、DC6 ではないことは control で示す: `not_a_lightgbm_parameter` / `feature_weights_typo` / `nun_leaves` はいずれも受理集合の外、`num_leaves` / `learning_rate` / `feature_contri` / `num_leaves_ratio` はいずれも内側。実測された distinct な名前は探索空間 3 種・`model.params` 7 種で、いずれも受理集合内。`calibration.params` の distinct な名前は `num_boost_round` / `seed` の 2 種でいずれも受理集合内、`_ISOTONIC_DEFAULTS` の 13 キーも全て受理集合内（`min_data_in_leaf_ratio` は決定 6 の例外経由）、control の `not_a_lightgbm_parameter` は受理集合外。

なお §7 は `5712f41` で同じ 0/52・0/736 を記録している。本 Proposal は**その数値を引き写さず再導出した**（引き写しは本監査が繰り返し見つけてきた DC3 の形であるため）。母集団は 821→824 に増えているが、これは PR 0 が足したテストの分である。

### 影響範囲 / 互換性

- **公開 Protocol の破壊的変更**: `EstimatorProvider` にメソッドが 2 つ増える。`Protocol` を構造的に満たす外部実装は、この 2 つを実装しないと適合しなくなる。リポジトリ内の実装は `LGBMProvider` のみ。`format_version` は不変（Protocol は永続化されない）。
- **`feature_weights` を設定していた利用者の fit 結果が変わる。** これは本 PR で唯一の、既存の正常系に影響する変更である。従来は重みが**一切効いていなかった**ため、修正後は異なるモデルが学習される。その設定でチューニングした `best_params` や保存済み artifact は、重みが効いていないモデルを前提に得られたものである。値は保存された config のとおりに解釈されるようになる（従来は無視されていた）。**移行は不要だが、`feature_weights` 利用者は再チューニングを検討すべきである。**
- `model.params` / `tuning.optuna.space` / `calibration.params` に不正な名前を書いていた config は、**学習・チューニング・コード出力を開始した時点で**落ちるようになる（`Model(...)` の構築自体は成功する）。**計測上、出荷済みコーパスに該当は 0 件。**
- `calibration.params` の検査は `method: isotonic` のときだけ働く。`platt` / `beta` の `calibration.params` は従来どおり素通りする — ただし `PlattCalibrator` はそもそも `params` を一切読んでいない（構築時に捨てている）。これは本 PR が塞ぐ silent-discard とは別経路の別欠陥であり、本 PR では**変更しない**。
- `FitResult` / `PredictionResult` / `Artifacts` の形と意味は不変。split / leakage 境界に触れない。

### 代替案（不採用）

- **案 A: Config フィールド名も `feature_contri` に改名する。** 公開 Config の破壊的変更になる。`feature_weights` という名前は利用者にとって意味が通っており、変える理由は LightGBM の内部命名に合わせること以外に無い。BLUEPRINT §5.3 が「LightGBM に渡す」と書くだけで名前を書いていないのは、まさにこの分離が意図されていたためである。
- **案 B: `config/schema.py` で config-parse 時に弾く（当初計画）。** Layer 1 の兄弟間 import になり `ARCHITECTURE.md` の DAG に反する。層を曲げてまで早く落とす利益は無い — 学習・チューニング・出力のいずれも、開始する前に落ちる。
- **案 C: 受理名を手書きのリストで持つ。** LightGBM の更新で名前が増減したときに黙って古びる。このゲートが検出すべき欠陥を、ゲート自身が持つことになる。
- **案 D: `tuning/search_space.py` の `parse_space` にも provider を渡して二重に検査する。** Layer 2 は Layer 1 の抽象 IF を参照してよいので層違反ではないが、**同じ判断を下す門が 2 つできる**。両者が食い違ったときに正がどちらか決まらない。門は 1 つにする。
- **案 E: `Model.load()` でも検証する。** 綴り違いを含む古い artifact が読めなくなる。artifact は起きた事実の記録であり、それを拒否しても既存の利用者を困らせるだけである。

### 受け入れ基準（テスト観点）

- **`tests/test_estimators/test_lightgbm_parameter_names.py`（新規）**: LizyML が `lgb.train` に渡す全キーと `lgb.Dataset` に渡す全キーワードを、3 タスク × **provider が宣言する全 smart parameter** にわたって収集し、LightGBM 自身のエイリアス表と突き合わせる。権威は `LGBM_DumpParamAliases` であって手書きの一覧ではないため、LightGBM 側が名前を削除した場合にも落ちる。
  - smart parameter のケースは**`extract_smart_params` から導出する**。並べ挙げると母集団を名乗る標本になる。`test_every_smart_parameter_has_a_case` が、provider の宣言するどれかに値が無い場合に落ちること。
  - 呼び出しの照合は属性チェーンの厳密一致で行う（`str.endswith` は `not_lgb.train` にも一致してしまう — DC2）。
  - 修正前は `feature_weights` を含む 3 タスク分と、`model.params` の 3 タスク分が **RED**。
- **`tests/test_tuning/test_search_space_name_validation.py`（新規）**: 名前 3 種（LightGBM 名 / smart 名 `num_leaves_ratio` / 存在しない `not_a_lightgbm_parameter`）× `category` 3 値 × 3 タスク = **27 セル**。`category: model` かつ受理集合外のセルで `CONFIG_INVALID` になること、smart 名の場合は診断メッセージが `category: smart` を名指しすること、**拒否されたセルではその名前が `lgb.train` に到達しないこと**。
- **`feature_contri` の実効性**: `feature_weights` を設定した fit と設定しない fit で **importance の順序が変わる**こと。`BLUEPRINT.md:1425` が宣言し、これまで検証されていなかった不変条件そのもの。修正前は **RED**（順序が変わらない）。
- **構築後に現れる名前も検査されること**: (a) `Model(cfg)` 構築後に `cfg.model.params` を書き換えて fit → `CONFIG_INVALID`、かつその名前が `lgb.train` に**到達しない**。(b) artifact から復元した `best_model_params` に不正名がある状態で再 fit → 同様。(c) その config を持つ `Model` の**構築自体は成功する**（`load()` を塞がないこと）。(a) と (b) は構築時検査では通ってしまう経路、(c) は構築時検査が誤って塞いでいた経路であり、いずれも実測で確認する。
- **`export_code` の生成 params も検査されること**: 学習済み adapter の params に不正名がある状態で `export_code()` を呼ぶと `CONFIG_INVALID` になること。生成される `train.py` は `CFG["lgbm_params"]` をそのまま `lgb.train` に渡すため、ここを塞がないと「読み込みは許すが学習は拒む」という決定 5 の非対称性が出力経路から破れる。
- **codegen テンプレートの名前も検査すること**: `lizyml/codegen/templates.py` が LightGBM に渡す名前を AST で読み、同じ権威と突き合わせる。呼び出しの照合は属性チェーンの厳密一致（`endswith` は `not_lgb.train` にも一致する — DC2）。各 `lgb.train` サイトは **3 分類で網羅する**: ここで読める名前 / `CFG[...]` 由来（実行時ゲートが担保）/ **読めない（失敗、skip しない）**。テンプレート内の正当なキーを不正名に置換すると落ちること。
- **`tests/test_calibration/test_calibration_param_names.py`（新規）**: (a) negative control — ゲートを通さず `IsotonicCalibrator` を直接構築すると不正名が `lgbm.train` に**到達する**こと（到達しなくなったらゲートの存在理由が変わるので落ちる）。(b) **学習する公開入口それぞれ**（`fit` / `tune`）で `CONFIG_INVALID` になり、その名前が `lgbm.train` に到達せず、かつ **Booster が 1 本も学習されていない**こと（後半は決定 6 の順序主張。前半だけでは欠陥のある配置でも成立した。入口ごとに主張するのは、この検査が入口を 2 度取りこぼしたためである）。(c) `_ISOTONIC_DEFAULTS` の**全キー**が受理されること（誤拒否の検査）。(d) `CALIBRATOR_OWN_PARAM_NAMES` の**各名前**が本当に calibrator に消費され `lgbm.train` に届かないこと、および `seed` は届くこと。(e) `platt` / `beta` は検査されないこと。(f) `LGBM_BACKED_CALIBRATORS` が登録済み calibrator の import 走査と一致すること。
- **経路の走査自体が実行で検査されること**: `HOSTILE_ROUTE_SHAPES` の各ソースに対し、走査が期待どおりの検出集合を返すこと（negative control 2 件を含む）。これが無い間、「経路が増えればテストが落ちる」は 2 度偽だった。
- **既存テストの書き換え（追加ではなく置換）**: `test_param_reaches_booster` は、値が学習済み Booster に届くことに加え、**その名前が LightGBM の定義に存在すること**を検査する。`Booster.params` は渡した dict の反響であって解析結果ではない（実測: 存在しないキーもそのまま保持され、LightGBM が既定値で埋めたパラメータは現れない）ため、到達だけを主張しても捨てられる名前で成立してしまう。列挙された 8 つの名前を権威と突き合わせる後者が、この主張を意味あるものにする。
- `TestFeatureWeightsE2E::test_feature_weights_applied` は**差分**を主張する形に直す（現状の「2 つの列名が存在する」は重みの有無に関わらず成立する）。
- 全スイート green、`ruff check .` / `ruff format --check .` / `mypy lizyml/` クリーン。


## H-0094: `Model.fit(params=...)` を実際に転送し、不明名の出所を名指しする（#264）

- **ステータス**: Accepted
- **起票日**: 2026-09-07
- **決定日**: 2026-09-07
- **スコープ**: `lizyml/core/model.py`（`fit()` から override を渡す / `_merge_params` が名前の出所を持つ）, `lizyml/core/_model_factories.py`（決定 4 の拒否）, `lizyml/estimators/lgbm/smart_params.py`（`SMART_PARAM_TARGETS` 宣言 + `smart_managed_names`）, `lizyml/estimators/provider.py`（**公開 Protocol に 1 メソッド追加**）, `lizyml/estimators/lgbm/provider.py`（実装）, `tests/test_core/test_fit_params_override.py`（新規）, `tests/_train_spy.py`（新規: `lgb.train` / `lgb.Dataset` の記録を 1 か所へ）, `tests/test_estimators/test_lightgbm_parameter_names.py`（自前の記録器を共有ヘルパへ置換）, `BLUEPRINT.md` §12.2 / §18.1.3, `CHANGELOG.md`。
- **関連**: [Issue #264](https://github.com/nbx-liz/LizyML/issues/264), H-0093（名前検査の設置場所）, H-0050（`_merge_params` の優先順位）, [#277](https://github.com/nbx-liz/LizyML/issues/277)（本 PR で起票した calibration 側の同型欠陥）。

### 目的（課題）

`Model.fit(data=None, params=None)` は `params` を**公開シグネチャに持ち、docstring で「config の `model.params` を上書きする」と宣言している**。転送されていなかった。

受け皿である `_merge_params(self, provider, override=None)` のオーバーレイ自体は正しく、**呼び出し側が override を渡していなかった**（`model.py:203`）。したがって上書きは例外も警告もなく捨てられ、Booster は config の値で学習される。

出荷コードに対する実測（#264 より、本 PR で再現）:

```
Model._merge_params(override), declared default: None
invocations observed: 2  (one plain fit, one fit(params=...))
values it was bound to: [None, None]
invocations binding a non-default: 0
```

```python
dumps[0] == dumps[1]          # -> True   (booster texts identical)
# '[learning_rate: 0.001]' in both — the 0.5 override never arrived
```

DC4（inert wiring）。配管はあり、公開の書き手が誰も到達しない。

**この欠陥は「dict が正しいか」では捕まらない。** マージ後の dict は常に正しかった（誰も override を渡していないのだから）。だから受け入れ基準は**学習済み Booster** と `lgb.train` が実際に受け取った値に対して置く。

### 対応方針（決定）

1. **転送する（引数を削除するのではなく）。** `model.py` の呼び出しを
   `self._merge_params(provider, override=params)` にする。#264 が挙げるもう一方の方向（引数と docstring の削除）は §代替案 を参照。

2. **検査は転送先の dict の上に置く（H-0093 で既にそこにある）。** 転送だけを行うと H-0093 が閉じた境界が新しい入口から開く: `fit(params={"not_a_lightgbm_parameter": 1})` が無検査で `lgb.train` に届く。名前検査は `_merge_params` が返す dict に対して働くため、config・tune 結果・`fit()` 上書きの 3 入力すべてがそこを通る。

   ただし**これは `lgb.train` への全経路ではない**。trial params は `_merge_params` の後にマージされ（探索空間の parse 時検査でカバー）、`LGBMAdapter(params=...)` の直接構築と codegen が出す `lgb.train` は別方向からカバーされる（H-0093 決定 8）。主張は「**利用者の Config または `fit()` 呼び出し**が `lgb.train` の前に置ける名前は必ず provider の検査を通る」であり、「パッケージ内の全呼び出しが 1 関数を通る」ではない。

4. **スマートパラメーターが管理するネイティブ名は、上書きを受理せず拒否する（レビュー round 1 の指摘）。**

   転送しただけでは足りなかった。スマート解決は `_merge_params` より**後段**で走り、その結果が勝つ（`core/model.py`: `resolved_model = {**resolved_model, **smart_resolved}`）。したがって「最優先」は嘘になる。round 1 の実測:

   ```
   fit(params={"scale_pos_weight": 10})  -> lgb.train received 0.9354838709677419
   fit(params={"num_leaves": 12})        -> lgb.train received 32
   fit(params={"min_data_in_leaf": 3})   -> lgb.train received 1
   ```

   **受理して置き換えるのは、この PR が直している欠陥そのものの再演である。** そこで `LGBMConfig._validate_smart_params` が config に対して既に適用している方針（衝突は競合エラー）を `fit()` 入力にも適用する: 有効なスマートパラメーターが書くネイティブ名は `CONFIG_INVALID` で拒否し、**どのスマートパラメーターが管理しているか**と**それを無効化する方法**を message に書く。

   宣言（`SMART_PARAM_TARGETS`）は**コードから閉じる**。`resolve_smart_params` / `resolve_ratio_params` の `resolved[<文字列>] = ...` 代入を走査し、宣言と一致しなければ落ちるテストを置く。宣言だけの表は「4 つ目のネイティブ名を書き始めた日」に黙って古びる — それは本 PR が閉じている silence と同じ形である。

   さらに**表が実在の上書きを指していること自体を実行で確かめる**: 各名前について、管理しているスマートパラメーターを**無効化すると同じ上書きが `lgb.train` に素通しで届く**ことを主張する。これが無いと表は何を書いても拒否テストが通ってしまう。`balanced` は multiclass では sample weight を作る（パラメーター名ではない）ため、multiclass の `scale_pos_weight` は管理対象外であることも実行で固定する。

   **エイリアスまで閉じること（レビュー round 2 の指摘）。** ここまでの実装は文字列一致で、LightGBM がエイリアスを同一パラメーターとして解決することを見ていなかった。実測:

   ```
   auto_num_leaves=True   fit(params={"max_leaves": 12})
       -> lgb.train received (max_leaves=12, num_leaves=32), booster [num_leaves: 32]
   auto_num_leaves=False  fit(params={"max_leaves": 12})
       -> booster [num_leaves: 12]
   ```

   `max_leaves` は受理名なので H-0093 の検査も通り、決定 4 の管理表にも無いので拒否もされず、**LightGBM が canonical 側を優先するため上書きはまた黙って捨てられた**。したがって管理表は canonical 名で宣言し、判定時に**学習器が受理する全綴りへ展開する**。綴りの集合は列挙せず `LGBM_DumpParamAliases`（H-0093 と同じ権威）から導く。管理対象 6 名の綴りは実測 18 通り（`num_leaves` に 4、`min_data_in_leaf` に 4、`feature_contri` に 4 のエイリアス）。テストは全 18 綴り × 2 方向で回し、エイリアス展開を外すと**エイリアス 12 セルだけが RED**、canonical 6 セルは green になることを確認済み — 見落としの形そのものである。

   **適用範囲は `fit(params=)` のみ。** config 面の穴は round 12 で**面の全体を実行して**数えた（スマートパラメーター × 書き込む native 名 × LightGBM が受ける綴り = 18 通り）: **拒否 3 / 2 綴りが届く 12 / 黙って上書き 3**。「parse 時に 3 件が拒否済み」は**スマートパラメーターについては真、面については偽**であり、通過する 15 を数えていない（決定 8-3 で訂正）。実測済み、[#280](https://github.com/nbx-liz/LizyML/issues/280)。残り 2 件の衝突は出荷済み config に 0 件（上の firing rate）。`config/` から学習器の別名表へは層規約上届かないので、config 面の修正は「どこで拒否するか」の設計判断であり本 PR の入力ではない。探索空間面は 54/67 で該当するが、閉じると本リポジトリの 54 件が落ちるため #279 に分離した。**この非一貫性は認識したうえでの分離であり、H-0094 の主張は「`fit(params=)` について閉じた」までである。**

3. **不明名の拒否は出所を名指しする。** 3 入力が 1 つの dict にマージされてから検査されるため、従来はすべて `model.params` として報告していた。3 つのうち 2 つは**利用者を誤ったファイルに送る**。`_merge_params` が `origins` を持ち、`model.params` / `provider default fixed params` / `tuning best_model_params` / `fit(params=)` を名前ごとに区別する。優先順位が上の入力が出所を上書きするので、同名が複数入力にある場合は**実際に効いている方**が報告される。

5. **パラメーターの層は綴りではなく同一性でマージする（レビュー round 3 の指摘）。**

   `{**base, **override}` は綴りが違えば両方を残す。LightGBM はエイリアスを解決し、**両方あるときは canonical を採る**ため、上書きが黙って負ける。実測（booster から読んだ値、`_COMMON_DEFAULTS` は `learning_rate=0.001` を常に注入する）:

   | config | fit(params=) | 修正前 | 修正後 |
   |---|---|---|---|
   | `learning_rate: 0.07` | — | 0.07 | 0.07 |
   | **`eta: 0.07`** | — | **0.001** | **0.07** |
   | `learning_rate: 0.07` | `eta: 0.5` | **0.07** | **0.5** |
   | （無し） | `eta: 0.5` | **0.001** | **0.5** |

   2 行目が示すとおり、これは `fit(params=)` だけの問題ではない。**`_COMMON_DEFAULTS` は canonical 名で 11 個のパラメーターを毎回注入する**ので、そのどれかをエイリアスで書いた config は出荷以来ずっと無効だった。

   したがって修正は 2 か所:

   - `_merge_params` の 3 つの継ぎ目（config → provider 既定 fixed → tune best → `fit()` 上書き）を `overlay_params` に置き換える。上位層が名指すパラメーターの**別綴りを下位層から落とす**。
   - `LGBMAdapter._build_params` で、**利用者がどれかの綴りで名指しているパラメーターの既定値を落とす**。ここを直さないと、facade で綴りを揃えても既定の canonical が後段で再注入されて上書きがまた負ける（実測済み）。

   **学習器に 2 つの綴りを渡さない**形にしてあるので、結果は「LightGBM がどちらを優先するか」に依存しない。優先規則は実測したが、それに乗るのではなく、曖昧さを渡さないことで閉じている。

   `overlay_params` は**学習器が知らない名前を落とさない**。落とすと H-0093 の拒否がその名前を見られなくなり、綴り間違いがまた無言の no-op に戻る。これはテストで固定した。

6. **特別扱いされるパラメーターも同一性で取り出す（レビュー round 4 の指摘）。**

   決定 5 が「利用者がどれかの綴りで名指したパラメーターの既定値を落とす」ようにした結果、**adapter が文字列一致で特別扱いしていた 3 つのパラメーターに穴が開いた**。それまでは canonical の既定値が隣にあってエイリアスに勝っていたので露見しなかった。実測（binary タスク）:

   ```
   fit(params={"objective": "regression"})    -> CONFIG_INVALID（task 不一致で拒否）
   fit(params={"application": "regression"})  -> 学習された [objective: regression]
   fit(params={"objective": "binary", "application": "binary"}) -> KeyError 'objective'
   ```

   1 件目と 2 件目は同じパラメーターであり、**エイリアスで書くと `_check_objective_compatible` を丸ごと迂回して誤った objective で学習していた**（DC2 → DC1）。3 件目は、adapter が検証済みの `objective` を `params` に置いた後、`application` が user 側に残っているため決定 5 の既定値落としがそれを既定値と誤認して消し、末尾の不変条件が消えたキーを読んで落ちる（DC2）。

   したがって adapter の特別扱い（`objective` / `metric` / boosting round 数）は **`_pop_by_identity` で全綴りを取り出す**形にする。同じ層で 1 つのパラメーターが複数綴りで指定され、**値が異なる場合は `CONFIG_INVALID`** とする（同値なら無害なので通す）。どちらを採るかを dict の順序で決めるのは、本変更が消そうとしている欠陥そのものである。

   **facade 側にも同じ拒否を置く**（`check_duplicate_identities`）。adapter の拒否は特別扱いされる 3 つしか見ないが、通常のパラメーターは誰も pop しないため両綴りが dict に残り、学習器がどちらかを選んでしまう。これが冗長でないことは RED で確かめてある: facade の拒否を外すと `objective` のケースは通ったまま**通常パラメーターのケースだけが落ちる**。

   副次的に、`num_iterations` のエイリアス（`num_round` 等）で boosting 回数を指定できるようになった。従来は `n_estimators` という 1 綴りだけが `num_boost_round` に変換され、他の綴りは params に残って `lgb.train` の引数と食い違っていた。

### Conditional-Activation Evidence

**転送そのものには不要。** `if override:` は Change Gate が列挙する 6 つの目的（`skip` / `shorten` / `cache` / `select` / `allow` / `conditionally-activate`）のいずれでもなく、「上書きがあるかないか」という通常の必須動作の分岐である。`origins` も同様に、拒否メッセージの宛先を決めるだけで何かの発火条件ではない。

**決定 4 の拒否には必要（`allow` 目的のゲート）。** レビュー round 1 の指摘で追加した「スマートパラメーターが管理するネイティブ名を拒否する」検査は入力を条件付きで通す門なので、同型の衝突が実際にどれだけ起きているかを、出荷済みスイートが構築する全 config（912 件、`LizyMLConfig` を記録する pytest plugin で計測）に対して測った。

```
Firing rate: 0/824 of configs carrying model.params
Firing rate: 54/67 of configs carrying a category:model tuning space
Firing rate: 0/0 of shipped calls passing fit(params=...) -- no call site exists
```

3 行の意味は同じではない。

- **`model.params` は 0/824。** parse 時の既存チェックが 3 件を止めているうえ、残る 2 件（`balanced` / `feature_weights`）の衝突を書いている config が実際に無い。したがってこの面に検査を広げても、出荷済みの何も壊れない代わりに、何も捕まらない。
- **探索空間は 54/67。** これは**生きている欠陥**であり、本 PR では**閉じない**。閉じると本リポジトリ自身の 54 件が落ち、方向（拒否する / チューニング値を勝たせる / smart 次元へ写像する）は保守者の判断である。[#279](https://github.com/nbx-liz/LizyML/issues/279) に実行証拠つきで起票し、BLUEPRINT §5.3 に入口ごとの状態表を置いた。
- **`fit(params=)` は母集団 0。** 引数がこれまで何もしていなかったため、この引数を渡す呼び出しがコードベースに 1 つも存在しない（`grep` で新規テスト以外 0 件）。すなわち新しい拒否は**既存の何も拒否しない**。これは Change Gate が言う「測定不能」ではなく、母集団が空であることを測った結果である。

### 影響範囲 / 互換性

- **公開 API のシグネチャは不変**。`FitResult` / `PredictionResult` / `Artifacts` / `format_version` も不変。split / leakage 境界に触れない。
- **振る舞いは変わる。** これまで無視されていた `fit(params=...)` が効くようになる。すでにこの引数を使っていた利用者は、**今まで意図と違うモデルを得ていた**ことになる。CHANGELOG に Fixed として明記する。
- **「最優先」は 3 入力の中での最優先であって無条件ではない。** 有効なスマートパラメーターが管理するネイティブ名は、上書きされるのではなく**拒否される**（決定 4）。無条件の最優先を主張すると、決定 4 の前に実測された「受理して置換」を仕様として書くことになる。
- 不正な名前を `fit(params=)` に渡していた場合は `CONFIG_INVALID` で拒否されるようになる（H-0093 と同じ理由: 黙って捨てられるより拒否される方がよい）。
- `fit(params=)` は**その呼び出しに閉じる**。`_merge_params` は新しい dict を作るので利用者の config オブジェクトは書き換わらず、次の `fit()` は config の値に戻る。これはテストで固定する。

### 代替案（不採用）

- **引数と docstring を削除する。** #264 が挙げるもう一方の方向で、同じ公開 API 変更である。不採用の理由は `fit(params=tuning_result.best_model_params)` が文書化されたワークフローであり、**今日たまたま動いているのは tune 結果が別経路（`_tuning_result` オーバーレイ）で適用されるからにすぎない**こと。削除するとこのワークフローは書けなくなる。
- **config だけを検査し、転送後は検査しない。** 実装は小さいが、`fit(params=)` という無検査の入口を新設することになる。H-0093 が閉じたばかりの境界を同じ PR で開くことになるため不採用。
- **出所を持たず全て `model.params` と報告し続ける。** 追加コストは無いが、`fit(params=)` の綴り違いを config ファイルの問題として報告するため、利用者は存在しない行を探すことになる。

### 受け入れ基準（テスト観点）

`tests/test_core/test_fit_params_override.py`（新規、11 ケース）:

- **学習済み Booster が変わること**: `params` だけが異なる 2 回の fit で booster テキストが**異なり**、上書き側が上書き値を、対照側が config 値を実際に持つこと（修正前は両者バイト同一で RED）。
- **`lgb.train` が受け取った値**: 記録した全 `lgb.train` 呼び出しの `learning_rate` が上書き値のみであること。
- **優先順位の 2 段**: `fit(params=)` が tune 結果に勝つこと、tune 結果が config に勝つこと。後者は修正前から green で、前者だけが RED — 欠陥の形そのもの。
- **境界が開かないこと**: `fit(params={"not_a_lightgbm_parameter": 1})` が `CONFIG_INVALID` で拒否され、かつ **Booster が 1 本も学習されていないこと**。
- **出所の名指し**: 例外メッセージと `context["unknown"]` の `surface` が `fit(params=)` であること。smart param 名の場合も専用メッセージを保ったまま出所を名乗ること。
- **3 入力の同時判定**: `model.params` / `tuning best_model_params` / `fit(params=)` にそれぞれ不明名を置き、3 件が**それぞれの出所**で報告されること。
- **誤って何かを変えないこと**: `params=None` と `params={}` がともに no-op であること、上書きが呼び出しをまたいで残らず利用者の config を書き換えないこと。
- **決定 4（管理名の拒否、18 綴り × 2 方向）**: 学習器が受理する各綴りについて、(a) 管理するスマートパラメーターが有効なら `CONFIG_INVALID` で拒否され、書かれた綴り・canonical 名・スマートパラメーター名が message に現れ、**Booster が 1 本も学習されていない**こと。(b) そのスマートパラメーターを無効化すると、**同じ上書きが `lgb.train` に届く**こと。(b) が無ければ表は何を書いても (a) が通る。綴りの母集団は登録表から導出し、`accepted_spellings` が canonical しか返さなくなったら落ちるテストを別に置く（そうでないと全セルが通ったまま穴が戻る）。
- **管理表がコードと一致すること**: 解決関数の `resolved[...] =` 代入の走査と `SMART_PARAM_TARGETS` が一致すること。両方向の RED 確認済み（宣言のみの名前 / 走査にだけ現れる名前）。
- **スマート面の分割が閉じていること**: provider が申告する全スマートパラメーターが「ネイティブ名を書く」か「何も書かない」のどちらかに分類され、未分類が残らないこと。
- **対照**: 管理対象でない名前（`learning_rate`）は拒否されず届くこと、multiclass の `scale_pos_weight` は届くこと。
- **値の比較は順序を決めて行うこと**（レビュー round 6）: (1) 双方が長さを持つなら**長さ**（要素ごとの比較はブロードキャストし、空列は空の `all()` で何とでも一致する）、(2) 比較結果の**真偽値**（`np.float64(0.5) == 0.5` は `np.bool_` で `bool` ではないが `bool()` にはできる。ここを飛ばすと printed form に落ちて**有効な入力を拒否**する = round 5 の欠陥の再演）、(3) **要素ごと**、(4) 例外が出たものは **printed form**（比較そのものも handler の内側に入れること — `__eq__` が失敗する値がある）。テストは 25 の入力を「何についての事例か」でラベル付けし、**両方向**で主張する（2 つの拒否は逆順で比較するため、非対称な答えは同じ呼び出しを一方で拒否し他方で受理する）。3 つの指摘それぞれに対して RED 確認済み。
- **値の等価性は 1 か所に置き、両方の拒否が共有すること**（round 5 の修正に対する自己レビューと rounds 4-5 監査が独立に発見）。素の `!=` は numpy 配列に対して配列を返し、`bool()` が例外になる。**綴りが 1 つしか無くても**自分自身と比較していたため落ちた: `fit(params={"feature_contri": np.array([1.0, 2.0])})` が `ValueError` になっていた（学習前・入口で無条件に通る経路）。`lizyml/core/value_equality.py`（Layer 0、import 無し）に `values_differ` を置き、facade と adapter の両方が使う。要素ごとの比較は「全要素が等しいこと」に還元し、還元できない値は printed form へフォールバックする（弱い答えだが例外にはならない）。値の定義域（配列 / list / tuple / `None` / bool / 空 list）を両方の拒否と実際の fit で確認し、素の `!=` に戻すと 5 セルが RED。
- **決定 6 の重複拒否は printed form ではなく等価性で比較すること**（レビュー round 5）。`repr` 比較は `1` と `1.0` を別の値と見なし、**同じことを 2 度書いただけの呼び出しを拒否していた**（有効な入力を拒む = DC7 の向き）。`_pop_by_identity` は等価性で比較しているので、同じ入力がパラメーター名によって受理されたり拒否されたりしていた。テストは (a) 2 綴りの等価な値が実際の fit を通ること、(b) 2 つの拒否が同じ入力について一致すること（`True`/`1` を含む。LightGBM は bool を learning rate として解釈できないためこれはヘルパ層で確認）、(c) unhashable な値（`feature_contri` は list）で壊れないこと。RED 確認済み。
- **決定 6（特別扱いの同一性）**: (a) `application`（`objective` のエイリアス）に task 不一致の値を渡すと **`CONFIG_INVALID` で拒否され、Booster が 1 本も学習されない**こと。(b) 互換な値なら学習されること。(c) 同一パラメーターの 2 綴りが**同値なら通る**こと（KeyError にならない）。(d) 値が異なれば拒否され、両方の綴りが message に現れること。(e) boosting 回数が `n_estimators` / `num_iterations` / `num_round` のいずれでも効くこと（`lgb.train` に渡る `num_boost_round` で確認）。(f) `metrics` が metric として扱われること。(g) **adapter が同一性で pop する名前の集合**が、テストが持つ別名ケースの集合と一致すること（走査で導出）。(h) facade の重複拒否が冗長でないこと — 外すと通常パラメーターのケースだけが RED になる。
- **決定 5（同一性マージ）**: (a) config が canonical、`fit(params=)` がエイリアスのとき**上書きが勝つ**こと（booster から読む）。(b) config に無くても既定の canonical に勝つこと。(c) tune 結果がエイリアスでも config に勝つこと。(d) **`lgb.train` に渡る綴りが 1 つだけ**であること（結果だけを見るテストは、dict に両方残っていても通ってしまう）。(e) config だけにエイリアスがある場合も効くようになること（振る舞い変更、CHANGELOG に記載）。(f) 学習器が知らない名前は `overlay_params` に落とされないこと。両方の継ぎ目で RED 確認済み。

### 決定 7: 同一性は 4 つ目の継ぎ目にも、同一層規則は宣言した層すべてに（レビュー round 11）

11 ラウンド目は初めて**範囲を絞らず**、成果物の経路全体（`docs/` を除く diff 全体）に対して回した。直前 4 ラウンド（6 / 8 / 9 / 10）はいずれも「前ラウンドの修正」に絞られており、rounds 9-10 の monitor がその構造を指摘していた —
**「新しく書かれた装置に向けたラウンドは、出荷コードが正しいかどうかに関係なく装置の欠陥を見つける。round 10 の『本番欠陥ゼロ』はスコープが機械的に生んだ結果であって、成果物の状態を示していない」**。
範囲を広げた round 11 は**本番の欠陥を 3 件**返した。うち 2 件は、round 6 以降一度も変更されていないマージ経路そのものにあった。

1. **tuning の trial マージが 4 つ目の継ぎ目だった（DC1）。** 決定 5 は 3 つの継ぎ目を同一性マージにしたが、`_model_tuning.py` の objective 内 `{**base_model_params, **fixed, **model_p}` は綴りベースのまま残っていた。config に `learning_rate=0.001`、探索次元に `eta` があると、**trial は 0.001 で学習し、study には `eta=0.5` が best として記録され、その後の fit は 0.5 で学習する**。tuning が一度も評価していないモデルを選んでいた。実測（booster の `[learning_rate: ...]` 行）:

   ```
   best: {'eta': 0.5}
   tune: ['[learning_rate: 0.001]', '[learning_rate: 0.001]']
   fit:  ['[learning_rate: 0.5]',   '[learning_rate: 0.5]', ...]
   ```

   `overlay_params` を同じ順序（base → fixed → trial）で適用する。**この修正は本 PR の diff の外**（`_model_tuning.py`）にあるが、非一貫性を作ったのは本 PR である — 片側だけを同一性にしたため、trial の評価と選択が食い違うようになった。#279（スマートパラメーターが解決する探索次元）とは別物で、あちらはスマート解決による上書き、こちらは通常のエイリアス衝突である。

2. **同一層の重複拒否が 1 層にしか配線されていなかった（DC4）。** 決定 6 は「同じ層で 1 パラメーターが複数綴り・異なる値なら `CONFIG_INVALID`」と宣言したが、呼び出しは `fit(params=)` にしか無かった。`model.params` に `learning_rate` と `eta` を両方書いた config は両方が `lgb.train` に届き、LightGBM が黙って canonical 側を採る — 宣言はあり、実装もあり、その層には呼び出し側が無い。検査は facade（`_merge_params`）に置く。`config/` は層規約上 `estimators/` を import できないためである。

   ```
   Firing rate: 0/813 of pre-existing configs carrying model.params (本リポジトリの
   スイートが構築する config を `check_duplicate_identities` の呼び出し点で観測。
   814 件中 1 件が発火し、それは本変更と同時に足した回帰テストそのもの)
   ```

   出荷済み config は 1 件も壊れない。`allow` 目的の条件なので Change Gate の実測要件に従って測った。

   **さらに 4 つ目の層があった（rounds 10-11 monitor の指摘）。** 決定 7 を「宣言した層すべてに」と書いたので、monitor に「どの層に配線したのか」を問われた。`check_duplicate_identities` の呼び出しは 2 か所しか無く、**`calibration.params` は名前検査だけで同一性検査が無かった** — H-0093 が「config 側のどの門も見ていない 4 つ目の経路」と呼んだ層である。実測: `calibration.params: {"learning_rate": 0.001, "eta": 0.5}` は**両綴りが calibrator の `lgbm.train` に届き**、LightGBM が黙って canonical 側を採っていた。`check_calibration_param_names` の中に配線した（名前検査と同じ入口・同じ provider）。

   ```
   Firing rate: 0/22 of pre-existing configs carrying calibration.params
   (同じ観測。23 件中 1 件が発火し、それは本変更と同時に足した回帰テスト)
   ```

   **スマート層は綴りマージのままでよい（同 monitor の 2 つ目の候補、実測して否定）。** 他の全層を同一性でマージするのは学習器がエイリアスを解決するからであり、**スマートパラメーター名には学習器のエイリアスが 1 つも無い** — LizyML 自身の名前で、LightGBM はそれらを知らない。したがって 2 つ目の綴りで届く経路が存在しない。実測 0 件、テストで固定（名前が増えて古びる種類の主張なので、仮定ではなく主張として置く）。

3. **等価な配列を拒否していた（DC7）。** round 8 で要素ごとの還元ステップを削除したとき、真偽値にできない比較は印字形で判定することにし、その代償（dtype の違う等値な配列は「異なる」と報告される）を docstring に明記した。round 11 はその代償を**本番入口で実測**した: `np.array([1, 2])` と `np.array([1.0, 2.0])` はそれぞれ単独では学習でき、2 綴りで同時に書くと `CONFIG_INVALID` で拒否される。**代償を書いたことは、有効な入力を拒まないという要求を満たさない。**

   真偽値ステップと印字形の間に**変換ステップ**を入れる: 両辺に `tolist` があれば plain Python に変換し、**同じ**（ガード済みの）真偽値の問いをもう一度する。これは「任意のオブジェクトを反復すると何が出るか」という推測（round 8 が削除したもの）ではなく、文書化された変換のあとに通常の問いを繰り返すだけである。副次的に、印字形が要約で潰していた長い配列の差も正しく検出されるようになった。残る代償は plain Python への忠実な変換を持たない値（`DataFrame`、利用者独自のオブジェクト）だけで、ケース表がそこに到達する。

その他:

- `tests/_train_spy.py` は `lgb.train` / `lgb.Dataset` の記録器を 1 つにする。同じ計測器の 2 つ目の写しが既にあり、3 つ目を作る前に共有化した。`test_calibration_param_names.py` の `_TrainSpy` は**意図的に残す**: あれは `isotonic.lgbm` を名前で patch することで「calibrator の経路である」ことの証拠になっており、LightGBM 一般についての計測ではない。
- 全スイート green、`ruff check .` / `ruff format --check .` / `mypy lizyml/` クリーン。

### 決定 8: 綴りと容れ物は値ではない（レビュー round 12）

round 12 は round 11 と同じく**範囲を絞らない**ラウンドとして回した。結果は
`REQUEST_CHANGES` 2 件、どちらも `[P2]`、どちらも修正前にこちらで再現した。
以下 3 件目は、その後に**継ぎ目の全数列挙**（round 11-12 monitor へ持ち込む
問い）を実行して見つけたもので、レビュアーの指摘ではない。

1. **等価な列を容れ物の違いで拒否していた（DC7）。** 決定 7 の 3 番目は
   `np.array([1, 2])` と `np.array([1.0, 2.0])` の誤拒否を `tolist` 変換で閉じた。
   round 12 はその 1 つ隣を実測した: `np.array([1., 2.])` と `(1., 2.)` は
   `tolist` が配列だけを変え tuple を変えないので印字形まで落ち、`differ` になる。
   `feature_contri` と `feature_penalty` は LightGBM の同一パラメーターなので、
   同一層の同一性拒否が発火する — **`model.params` でも `fit(params=)` でも
   `CONFIG_INVALID`、`train_calls = 0`**。

   **これは round 11 が持ち込んだ退行ではない。** `tolist` ステップが無かった頃も
   この組は印字形で判定され、同じく differ になっていた。

   **受け入れ済みの判断を覆した。** `tests/test_core/test_value_equality.py` は
   `("a list and an equal tuple", [1.0, 2.0], (1.0, 2.0), True)` を持ち、
   「これは偶然ではなく判断である」と書いた test を添えていた。その判断は
   Python から論じていた（`[1.0, 2.0] == (1.0, 2.0)` は `False`）が、**問いを
   取り違えていた**。この関数が呼び出し元のために答える問いは「学習器は 2 つの値を
   見るか」であって「呼び出し元は同じ容れ物に手を伸ばしたか」ではない。実行して
   決めた:

   ```
   feature_contri        [1.0, 2.0] / (1.0, 2.0) / array([1., 2.]) / array([1, 2])
                         -> [feature_contri: 1,2]        identical trees: True
   monotone_constraints  [1, 0] / (1, 0) / array([1, 0])
                         -> [monotone_constraints: 1,0]  identical trees: True
   ```

   黙って一方が選ばれるわけではない — **選ぶべき差が無い**。round 11 の 3 番目を
   通したのと同じ論法である。

   修正は**独立したステップ**として比較の前に置く: テキストでない `Sequence` を
   `list` にする。`_as_plain_python` の拡張では足りない — `[1.0, 2.0] == (1.0, 2.0)`
   は真っ当な `bool` なので真偽値ステップが先に答えてしまい、変換ステップに届かない。
   `str` / `bytes` / `bytearray` は除外し、除外自体をケース表で固定した
   （`"ab"` と `("a", "b")` は differ）。

2. **calibration のエイリアスが、上書きしようとした既定値に負けていた（DC1）。**
   `IsotonicCalibrator.__init__` は `{**_ISOTONIC_DEFAULTS, **user}` と**綴りで**
   マージし、既定値は canonical で書かれている。したがって
   `calibration.params = {"eta": 0.5}` は名前検査も同一性検査も通り（呼び出し元は
   1 度しか書いていない）、`lgbm.train` には `learning_rate: 0.03` と並んで届き、
   LightGBM が canonical を採った。実測:

   ```
   learning_rate: 0.5 -> calibrator は {'learning_rate': 0.5}
   eta:           0.5 -> calibrator は {'learning_rate': 0.03, 'eta': 0.5}
   ```

   レビュアーは範囲を明示した — *「これは本 PR が触れた calibration 経路に残っていた
   既存の下流マージであり、本 PR が導入したとは主張しない」*。

   **修正の置き場所は層規約が決める。** `lizyml/calibration/` は
   `lizyml/estimators/` を import できないので calibrator にエイリアスを教えられない。
   `canonicalise_calibration_params` を facade 側（provider に既に届く場所）に置き、
   dict を渡す前に綴りを canonical に書き換える。calibrator 自身のキーは除外し、
   除外を assert で固定した: `num_boost_round` は `num_iterations` のエイリアスなので、
   canonical 化すると calibrator が pop するキーが消える。`random_state` は facade が
   供給する seed に対する同じ欠陥で、同じ書き換えで閉じた。

   **その書き換えが 1 件の振る舞いを変えたので、そこも閉じた。** calibrator は
   マージ後に `merged["verbose"] = -1` を強制していたが、**`verbose` は
   エイリアスで canonical は `verbosity`** である。LightGBM は canonical を優先する
   ので、`calibration.params = {"verbosity": 1}` は**本 PR 以前から**その強制を
   破っていた。canonical 化により `verbose` も同じ経路を通るようになり、非一貫が
   一貫した穴になる — なので強制を canonical 側に移した（`merged["verbosity"] = -1`、
   他綴りは pop）。`monotone_constraints` の強制が効いていたのは、そちらが最初から
   canonical だったからである。両方向をテストで固定した。

3. **同一層規則の 6 つ目の継ぎ目は、既に起票済みの設計判断だった（#280）。**
   継ぎ目の全数列挙で `check_smart_managed_overrides` に届いた。この検査は
   `fit(params=)` にしか配線されておらず、docstring は根拠として「config 面は
   parse 時に 5 件中 3 件が拒否済み」と書いていた。面の全体を実行した — スマート
   パラメーター × 書き込む native 名 × LightGBM が受ける綴り:

   ```
   population: 18
   DEFEATED   12/18   2 綴りが lgb.train に届き、LightGBM が canonical を採る
   REPLACED    3/18   resolver が利用者の値を黙って上書きする
   REFUSED     3/18
   ```

   **この欠陥は既知であり、BLUEPRINT.md §14.4 に正確に記載され、#280 として
   maintainer の判断待ちである。**`config/` から別名表に届かないため「どこで拒否
   するか」が設計判断になる、というのがその起票内容そのものである。したがって
   **本 PR では実装しない**。本 PR のコードにある欠陥は宣言の側で、
   「5 件中 3 件が拒否済み」はスマートパラメーターについては真だが**面については
   偽** — canonical 3 件を数え、通過する 15 の綴りと対象を数えていない。docstring を
   実測値に置き換え、#280 と BLUEPRINT §14.4 を指すようにした。**宣言を実態より広く
   書くことは、この PR が扱っている形そのものである（DC5）。**

   同じ類が calibration 層にもう 1 件ある（記録のみ、未修正）:
   `calibration.params = {"min_data_in_leaf": 7}` は、`IsotonicCalibrator.fit` が
   常在の既定 `min_data_in_leaf_ratio = 0.01` から `params["min_data_in_leaf"]` を
   無条件に書くため、**全綴りで** 7 ではなく `ceil(n × 0.01)` で学習する。本 PR 前後で
   結果は変わらない（canonical 綴りも以前から負けていた）。#280 と同じ設計判断に
   属するので、実装せず記録する。

#### 継ぎ目の全数列挙

`lizyml/` の中で「あるパラメーター dict が別のパラメーター dict に出会う」場所を
AST で列挙した（`{**a, **b}` / `.update` / `|` / 名前付きヘルパ 2 つ）。24 式。
そのうち**出所の異なる** 2 つの dict が出会うのは以下で、各行は実行して確かめた:

| 場所 | 解決 |
|---|---|
| `calibration/isotonic.py:97` | facade で canonical 化 — **決定 8-2** |
| `config/loader.py:108` | 同一層どうし。実行済み: 値が違えば拒否、同じなら学習 |
| `core/_model_factories.py:583` | `overlay_params` の中身、同一性を見る |
| `core/_model_tuning.py:457,458` | `overlay_params`（決定 7） |
| `core/_model_tuning.py:459` | スマート層。スマート名にエイリアスは無い（固定済み） |
| `core/model.py:469,472,503` | `overlay_params` |
| `core/model.py:481` | スマート層 |
| `core/model.py:749` | `canonicalise_calibration_params`（新規） |
| `estimators/lgbm/adapter.py:163` | ratio resolver が利用者の dict に出会う — **決定 8-3 / #280** |
| `estimators/lgbm/adapter.py:499` | 同一性を見る（rounds 1-2） |
| `estimators/lgbm/adapter.py:456,458` | `random_state` / `verbose`、上の重複排除に吸収される |
| `estimators/lgbm/provider.py:265` | `{**_COMMON_DEFAULTS, **effective_params}`。resolver が読み戻す唯一のキーは `max_depth` で、**エイリアスが無い**。実測し、古びないよう固定した |

残りの式はパラメーターのマージではない（`frozenset` の合併、行ビルダー 2 つ）。

#### Firing rate

スイートが構築する `LizyMLConfig` を全数記録し、**どのテストが作った config か**を
併記して、本変更自身の回帰テストと既存母集団を区別できるようにした。

```
Firing rate: 0/22 of pre-existing configs carrying calibration.params
             （calibrator が学習する値が変わるもの。24 件中 4 件が発火し、
               うち 2 件は本変更の回帰テスト、残り 2 件は round 11 の
               2 綴りテストで、潰れた側は同値か元々拒否される）
Firing rate: 0/1009 of pre-existing configs carrying model.params
             （本変更が解く拒否に掛かっていたもの。1010 件中 1 件が発火し、
               それは本変更の回帰テスト）
```

その他:

- `docs/audits/2026-09-defect-discovery/instruments/calibration_canonicalisation_firing_rate.py` を追加。
- 全スイート green、`ruff check .` / `ruff format --check .` / `mypy lizyml/` クリーン。

#### 決定 8 の追補: 探索空間もひとつの層だった（rounds 11-12 monitor の指摘 → 実行 → 修正）

rounds 11-12 monitor は `CONVERGING` / `continue` を返しつつ、上の継ぎ目表に 3 点の
反論を出した。verdict としてではなく finding として受け、3 点とも処理した。

1. **走査が宣言していた構文の集合に、その走査自身の指摘が住んでいる構文が無かった。**
   決定 8-2 の欠陥は `merged["verbose"] = -1`、8-3 の欠陥は
   `resolved["num_leaves"] = ...` で、どちらも `d[k] = v` である。表はそれらを
   *最寄りの宣言済み構文*の行に載せていた — つまり隣接コードを読んで見つけたので
   あって、走査が見つけたのではない。**自分が報告した欠陥の形を見られない走査で
   閉じた母集団は、近さで標本抽出しただけである。**

   構文集合を広げた（`d[k] = v` / `dict(a, **b)` / `f(**x)`）。候補は 24 → **48**。

2. **名指しされた継ぎ目は実在した。** `lizyml/tuning/search_space.py:215-223` は
   `params[dim.name] = trial.suggest_*` を次元ごとに書くので、**互いにエイリアスで
   ある 2 次元は同じ trial dict に両綴りを入れる**。`check_duplicate_identities` の
   呼び出しは 3 か所（`model.py:456,493` / `_model_factories.py:870`）で、空間の上には
   無い。実測:

   ```
   space = {learning_rate: [0.001, 0.01], eta: [0.4, 0.5]}
   -> 全 trial で両綴りが lgb.train に届き、learning_rate の値で学習
   -> best_model_params: {'learning_rate': 0.0064, 'eta': 0.4545}
   ```

   **`eta` 次元は sample され、Optuna が最適化し、どの trial にも影響しない。**
   study は何もしない軸で trial を順位づけ、`best_model_params` が死んだ綴りを
   記録するので、後続の `fit` もそれを運ぶ（DC1 + DC6）。

   #279（次元 × スマートパラメーターの衝突）とも #280（`model.params` ×
   スマートパラメーター）とも別物である。決定 6 が「宣言した層すべてに」と言う層で、
   呼び出し側が無かった 3 つ目 — round 11 の 2 番目と同じ形。

   `check_duplicate_space_dimensions` を study 開始前（名前検査の隣）に配線した。
   **ここには同値による免除が無い**: 2 次元は独立に sample するので、境界が何であれ
   1 パラメーターを 2 回名指しすることは曖昧である。両方向をテストで固定した。

   ```
   Firing rate: 0/69 of pre-existing configs carrying a category:model search space
   （70 件中 1 件が発火し、それは本変更と同時に足した回帰テスト）
   ```

3. **instrument が出荷されていなかった。** 決定 8 の表は走査から作ったのに、走査は
   scratchpad にしか無く、表を再生成できなかった — 本リポジトリ自身の規則で DC3。
   `instruments/parameter_merge_seams.py` として出荷した。走査が**できないこと**も
   明記してある: hint 語の絞り込みは識別子テキストのヒューリスティックであって型解析
   ではないので、hint 語のどれにも当たらない変数に入ったパラメーター dict は見えない。
   `HINTS` を定数として置いてあるのは、「走査が見落とした」を検証可能にするためである。

monitor の予測も記録しておく（採用ではなく記録）: *範囲を絞らなかったラウンド
（1-5, 7, 11, 12）はすべて `lizyml/` のファイルを名指ししている。round 13 は
`APPROVE` を予測しない。*

### 決定 9: 同じ値の別の書き方、そして LizyML 自身が握っているパラメーター（レビュー round 13）

round 13 も範囲を絞らず回した。`REQUEST_CHANGES` 3 件、すべて本番コード、すべて
修正前に再現した。**3 件とも round 12 が書いたコードではない** — 1 と 3 は、
エイリアスと転送が効くようになったことで本 PR が**露出させた**既存の読み手であり、
2 は本 PR が書いた関数の中だが、当該ケースは round 12 のステップより古い
（長さ 2 対 3 で、そのステップが無かった頃も拒否されていた）。

1. **エイリアスで書かれたカスタム metric が `export_code` で失われる（DC1）。**
   `_extract_feval_metadata` は `adapter.params.get("metric")` と**リテラル綴りで**
   読んでいた。`_build_params` は同じパラメーターを `_pop_by_identity` で読むので、
   **学習したコードと出力したコードが「呼び出し元は何を指定したか」で食い違っていた**。
   実測:

   ```
   metric        評価=['brier']  出力 metric='None'  feval=['brier']
   metrics       評価=['brier']  出力 metric='None'  feval=[]
   metric_types  評価=['brier']  出力 metric='None'  feval=[]
   ```

   生成コードは metric を失うだけでなく**動かない** — レビュアーが生成された
   `train_lgbm` を実行し、`ValueError: For early stopping, at least one dataset and
   eval metric is required` を得ている。同一性で読むよう直した。

   **この構文の母集団を列挙した。** `estimators/` / `persistence/` / `codegen/` /
   `core/` / `training/` でパラメーター dict をリテラル綴りで読む箇所は 4 件。
   `provider.py:473` が生きた 1 件で、`adapter.py:234,507` は `_pop_by_identity` の
   後（正規化済み）、`smart_params.py:146` は `max_depth`（エイリアス無し）。
   **これは継ぎ目走査が扱っていない構文である** — 「dict が dict に出会う」ではなく
   「利用者が綴った dict を 1 つの綴りで読む」。

2. **列とそのカンマ区切り文字列が「2 つの値」として拒否されていた（DC7）。** 実測:

   ```
   {feature_contri: [1, 2]}                          -> 学習
   {feature_penalty: "1,2"}                          -> 学習
   {feature_contri: [1, 2], feature_penalty: "1,2"}  -> CONFIG_INVALID
   2 つの単独ケースは同一の booster を学習する: True
   ```

   長さステップが `"1,2"` の**文字数**と `[1, 2]` の**要素数**を比べていた。

   **これは同じ関数で 3 ラウンド連続の「次の等価クラス」である** — round 11: dtype、
   round 12: 容れ物、round 13: テキスト文法。open grammar を 1 形式ずつ塞ぐ形なので、
   **追いかけるのではなく閉じる**書き方にした。

   **権威は推測せず読んだ。** `lightgbm/basic.py::_param_dict_to_str` は
   `list` / `tuple` / `set` / 1 次元 ndarray の**すべて**を、パラメーター名に関わらず
   `",".join(map(_to_string, val))` で書き、`str` はそのまま通す。つまり 2 つの形は
   **ワイヤ上で 1 つの値**であり、これが「パラメーターごとの知識」ではなく一様な
   ステップにできる理由である。テストは serialiser を**実行**する。

   比較は**テキストではなく要素ごと**にした。ワイヤ形式は正規形ではないからである:
   `[1.0, 2.0]` は `"1.0,2.0"`、`[1, 2]` は `"1,2"` になり、LightGBM はどちらも同じ
   double に解釈する。連結文字列を比べるとこの組を拒否してしまい、**同じ誤拒否が
   書式 1 段ずれて再発する**。

   **明示する限界**: 入れ子の文法は**扱わない**。`interaction_constraints` は
   `[[0, 1], [2]]` と `"[0,1],[2]"` を受けるが、それを読むには LightGBM が今後
   拡張しうる文法のパーサが要る — DC1 が警告する open-grammar そのものである。
   両者は「異なる」と報告し、ケース表で固定した。負のコントロール（`"1,2"` 対
   `[5, 6]` / `[1, 2, 3]`、`"auc"` 対 `["auc", "logloss"]`）も実行済み。
   **「同じ」の床には落とさない** — 落とすと、この門が存在する理由である DC1 を
   そのまま通してしまう。

3. **`training.*` が既に握っているネイティブパラメーター（DC1、両方向）。**
   `adapter.py:223` は `training.early_stopping.rounds` から作った callback を常に
   足す。実測: 上書きは毎回 `lgb.train` に届き、それでも config が停止を決めていた
   （`rounds: 2` + 上書き `10` → 3 イテレーション）。

   **修正方針を決めた実行**: callback を**切った**場合も inert ではない —
   LightGBM 自身がそのパラメーターを honour し、LizyML は検証セットを作っていないので
   `CONFIG_INVALID` が metric のせいにして落ちる。**安全に受理できる読みが存在しない。**

   **母集団を列挙し、全綴りで実行した。**

   ```
   training.early_stopping.rounds -> early_stopping_round
     early_stopping / early_stopping_round / early_stopping_rounds / n_iter_no_change
     4 綴りすべて受理され、それでも callback が決めていた
   training.seed -> seed
     random_seed / random_state / seed
     3 綴りすべて受理され、上書きが training.seed に黙って勝っていた
   ```

   `seed` は**逆方向**に失敗する（上書きが勝つ）ので、実行された run の再現性制御は
   config が宣言しているものではなかった。**2 方向が食い違うからこそ、どちらかを
   選ぶのではなく拒否する** — config のどこにも「どちらが効くか」は書いていない。
   検査は merge 後の dict に `origins` 付きで当て、利用者が直すべき入力を名指しする。

   ```
   Firing rate: 0/916 of configs with early stopping enabled and model.params
   Firing rate: 0/928 of configs with training.seed and model.params
   ```

#### 継ぎ目列挙の主張を格下げした

round 13 のプロンプトはレビュアーに「広げた走査がまだ見落とす継ぎ目を名指しせよ」と
明示的に求め、レビュアーは `config/loader.py:167`
（`node[last] = _coerce_env_value(value)`、環境変数上書きの書き込み）を挙げた。
カーソル変数名が `node` で hint 語に当たらなかったためである。実行して分類を確認した:
綴りが 2 つで値が違えば拒否、同値なら学習 — **列挙の穴であって欠陥ではない**、という
レビュアー自身の但し書きが正しい。hint 語に `node` / `cfg` / `config` を追加、候補
48 → **58**。

**そして主張自体を書き換えた。**「母集団を列挙した（閉じた）」は 2 回主張され、
2 回とも主張の 1 ラウンド以内に反証された — round 12 版は自分の 3 件中 2 件が住む
構文を宣言しておらず、round 13 版は `node` を見落とした。instrument の docstring は
**候補を生成する**こと、表が主張するのは**実行した分だけ**であること、そして
「開いた空間の走査を閉包と呼ぶこと」こそ本 run が他人の宣言に見つけ続けている DC5
であることを明記する。

その他:

- 全スイート **2474 passed**、`ruff check .` / `ruff format --check .` /
  `mypy lizyml/` クリーン。

#### 決定 9 の追補: 「閉じた」と言った直後に、閉じていないことを実行で示された（rounds 12-13 monitor）

決定 9-2 は「open grammar を追いかけるのではなく閉じた」と書いた。根拠は
`_param_dict_to_str` が `list` / `tuple` / `set` / 1 次元 ndarray の**すべて**を
一様に `","` で連結することであり、その一様な規則を適用したから、というものだった。

**rounds 12-13 monitor はその主張を実行で反証した。本ラウンドで自分でも再実行して
確認したうえで採用した。** `_comma_form_matches` は `isinstance(sequence, list)` で
入り口を絞っており、`_as_plain_sequence` は `collections.abc.Sequence` しか正規化
しない — ndarray はそれではなく、スカラーはそもそも列ではない。つまり
**docstring が名指しした 4 型のうち 1 型にしか届いていなかった。**

LightGBM 自身の serialiser を oracle にした再実行:

```
                        wire A          wire B          同一   結果
ndarray とそのテキスト  'p=1.0,2.0'     'p=1.0,2.0'     True   REFUSED
スカラーとそのテキスト  'p=0.5'         'p=0.5'         True   REFUSED
set とその list         'p=1.0,2.0'     'p=1.0,2.0'     True   REFUSED
list とそのテキスト     'p=1.0,2.0'     'p=1.0,2.0'     True   学習
```

**4 組中 3 組が LightGBM にバイト同一の文字列で届き、拒否されていた。**

`_wire_elements` を入れ、「この値に対して LightGBM は何を連結するか」を型に依らず
答えるようにした。スカラーは要素 1 個の wire 形式として扱う。テストは
**serialiser そのものを oracle にして型集合全体を回す** — 決定 9-2 の主張は本来
この形で検証されるべきだった。

**2 つの除外は「漏れ」ではなく「判断」として書き、ケースで固定した。**

- `set` / `frozenset`: LightGBM は連結するが、set に順序は無く、ここで列を取る
  パラメーターはすべて位置依存である（`feature_contri` は要素 *i* を特徴 *i* と
  読む）。同じに印字された 2 つの set は hash の偶然でそうなっただけであり、それを
  「同じ値」と認めると答えが hash 順に依存する。
- `None`: `_param_dict_to_str` は `None` を**送らない**。つまり「未指定」であって
  文字列 `"None"` ではない。

副産物として、仮定のままだった外部事実が 1 つ実行で確定した: **LightGBM は
`pd.Series` をパラメーターとして連結しない — `TypeError` で拒否する。** テストで固定。

**この追補自体が記録に値する形である。** 決定 9-2 は「閉じた」と書き、その主張は
1 ラウンド以内に反証された — 継ぎ目列挙の主張と**同じ経過**である。違いは、今回は
反証がループ外の monitor から来たこと、そして反証が具体的な値の組で来たことである。
主張を実測で書くこと自体は正しいが、**「型集合全体に適用した」と言うときは型集合全体で
実行すること**が、この 2 例から出る運用上の結論である。

### 決定 10: 探索空間も「入力」である — study が自分の次の一手に拒否される結果を返していた（レビュー round 14）

round 14 も範囲を絞らず回した。`REQUEST_CHANGES` **1 件（`[P1]`）** — round 5 以降で
最少であり、単一指摘のラウンドとしても round 5 以来である。

**指摘は、このラウンドのプロンプトが初めて投げた問いの答えだった。** rounds 12-13
monitor が「集合全体に適用した」という主張を 2 ラウンド続けて反証したので、round 14 の
プロンプトはレビュアーに *「この diff が集合を主張している箇所は、その集合の上で実行
されたのか」* を明示的に問わせた。返ってきた 1 件はまさにその形である。

**欠陥。** `check_training_managed_overrides` は `_merge_params` の中で走り、そこの
コメントは「every input at once をカバーする」と書いていた。`_merge_params` で出会う
**3 つの入力**については真だが、**4 つ目については偽**である — trial パラメーターは
その後、tune の objective で重なる。

したがって `category: model` の探索次元が `seed` や `early_stopping_round` を名乗ると、
**受理され、sample され、学習される** — そして直後の `fit()` が、その study が今作った
`best_model_params` を拒否する。両エントリの**全 7 綴り**で再現:

```
random_seed / random_state / seed / early_stopping / early_stopping_round /
early_stopping_rounds / n_iter_no_change
  -> いずれも booster を 2 個学習してから、次の fit が CONFIG_INVALID
```

**拒否が無かったのではなく、study が終わってから来ていた**のが欠陥である
（DC1 + DC4 + DC5）。

**修正。** `check_training_managed_space` を study 開始前、既にそこにある 2 つの
空間レベルの拒否（`check_param_names` / `check_duplicate_space_dimensions`）の隣に
配線した。空間も他と同じ 1 つの層であり、これは round 11 が
`check_duplicate_identities` について見つけ、決定 8 の追補が空間自身について見つけた
**同じ形の 3 例目**である。

**偽の主張は、修正だけでなく主張がなされた場所でも訂正した**: `model.py` のコメントは
merged-dict 検査がカバーする 3 入力を名指しし、カバーしない 1 つと、それがどこで
検査されるかを書く。

```
Firing rate: 0/70 of pre-existing configs carrying a category:model search space
（77 件中 7 件が発火し、それは本変更の回帰テストが 7 綴りを回した分）
```

#### レビュアーが「clean」と報告した内容（範囲付き）

- `fit(params={"eta": 0.5})` を binary / multiclass / regression で実行し、**全 CV
  booster と full-data refit booster**が `[learning_rate: 0.5]` を持つことを確認。
- training-managed の全 7 綴りを `model.params` と `fit(params=)` の両面で実行し
  **14/14 が学習前に拒否**。今回の指摘は tuning 層のみ。
- 学習済み adapter からの export パラメーター抽出を `metric` / `metrics` /
  `metric_types` で実行し、3 つとも Brier のメタデータを保持 — **round 13 の
  指摘 1 が独立に再実行されて確認された**。

レビュアー自身が「確立していない」と明記した範囲: フルスイート・lint・mypy は
再実行しておらず、ディスクへの export、生成プロジェクトの実行、実際の `Model.load()`
後の fit は read-only 制約下で実行していない。これらはこちらで実行した — フルスイート
**2500 passed**、`ruff check .` / `ruff format --check .` / `mypy lizyml/` クリーン。

#### 決定 10 の追補: D7 が発火していたのに記録が和らげていた（rounds 13-14 monitor）

rounds 13-14 monitor は `CONVERGING` / `redirect` を返し、2 つの指摘をした。両方とも
実行で確認したうえで採用した。

**1. D7 の authorship 条件が発火していた。** `git show 92e3d51` — round 13 の修正
コミットが `check_training_managed_overrides` **と**「every input at once をカバー」
という偽の主張の**両方**を書いている。つまり **round 14 の指摘は round 13 の修正が
書いたコードの欠陥**であり、これは D7 の authorship 条件の素直な読みで、この run で
初めて明確に発火した。

**round 14 の記録はそれを「同じ形の 3 例目」とだけ書き、発火したことを書いていなかった。**
条件は maintainer が**撤回済み**なのでループは止めない。しかし「発火したら和らげずに
記録する」はこの run 自身の基準であり、それを守れていなかった。monitor の指摘は正しい。

**2. リテラル読みの母集団だけが散文だった。** round 13 の記録は「grep で 4 件」と
書いたが、grep のパターンはどこにも残っておらず**再実行できない**。書き込み方向の
走査は positive control 付きで出荷されているのに、読み取り方向には対応物が無い。
この PR の他のすべての母集団は実行可能な宣言を持っていた。

monitor はさらに `adapter.py:455-458` を名指しし、liveness の判定は別モジュールとの
結合に依存するとして**判定を保留**した。**こちらで実行して判定した — 見た目より
小さい**: `random_seed=7` は自分の名前のまま `lgb.train` に届き、LightGBM はそれを
honour する（booster は `seed=7` と**バイト同一**、`seed=99` とは異なる）。**値は
失われていないので、これは欠陥修正ではなく一貫性の修正である**、とそう書いた。
facade 経路ではそもそも到達不能でもある（`training.seed` は常に設定される — 既定 42、
明示的 null は拒否 — ので training-managed 拒否が全綴りを先に claim する）。

**その整理が退行を 1 件生んだことも記録する。** 最初の版は `_pop_by_identity` を
使ったが、これは 2 綴り異値を**拒否**するので、`seed` が `random_state` に優先すると
いう受け入れ済みの決定（`test_lgbm_defaults.py`）を壊した。**命名の整理の副作用で
拒否の意味論を変えるのはこのコミットの仕事ではない**ので、優先順位をそのまま保つ形に
狭めた。

#### 2 つの宣言を出荷した

monitor の redirect は「層 × 検査の格子」と「読み取り方向の走査」を実行可能な宣言と
して出荷せよ、というものである。両方とも round 15 の前に出荷した。

- `tests/test_core/test_refusal_matrix.py` — 5 層 × 5 検査の格子。`wired` の各セルは
  **実行される**（拒否が発火し、層を名指しし、その前に何も学習していない）。`open` の
  セルは issue を名指しすることを要求し、表が矩形であることを要求し、
  `_model_factories` に列を持たない検査が存在しないことを確認する。**書いた直後に、
  wired と書きながら到達する入力が無いセルを 3 つ検出した** — 3 つとも入力を足した。
- `tests/test_estimators/test_literal_parameter_reads.py` — 読み取り方向の走査。
  エントリごとに理由を持つ許可リスト、両方向の陳腐化チェック、negative control 2 件を
  含む 7 件の hostile source。**実行したところ、未宣言の読み 2 件と、陳腐化した
  エントリ 2 件を検出した** — 許可リストは走査を回さずに読んで書いたものだったからで、
  これは round 13 の母集団がどう作られたかと同じ誤りである。

#### monitor の推奨のうち 1 点だけ採用しなかった

monitor は「2 つの宣言を出荷し、**round 15 をそれらに絞って**開け」と推奨した。
**絞ることは採用しない。** rounds 6 / 8 / 9 / 10 はいずれも直前ラウンドの修正に絞られ、
いずれも本番欠陥を出さなかった — それは**コードではなく範囲が作った結果**だと
この run 自身が確立している。monitor が名指しした作業の扱いとして確立している先例は
逆で、**ラウンドの前に処理し、ラウンドは絞らない**（rounds 10-11 / 11-12 monitor の
名指しした層はそれで「次ラウンドの指摘」ではなく「修正済みの欠陥」になった）。
round 15 は範囲を絞らず、新しい宣言もその中に入る。

全スイート **2529 passed**。

### 決定 11: 宣言が反証された — 格子の `n/a` が偽だった（レビュー round 15）

round 15 のプロンプトは、前ラウンドで出荷した 2 つの実行可能な宣言を**名指しで攻撃せよ**と
求めた。レビュアーはそうし、**格子自身の `n/a` の 1 つを反証した**。これは宣言が
意図どおり働いた結果である — 反証できる表は、反証できない散文より価値が高い。

`REQUEST_CHANGES` 2 件（ブロッキング）+ 1 件（非ブロッキング）。すべて修正前に再現。

1. **tuning が入れた早期停止設定が衝突ゲートから見えていなかった（DC1）。**
   `check_training_managed_overrides` は `cfg.training.early_stopping.enabled`
   だけを読んで `early_stopping_round` を claim するか決めていた。しかし
   `_build_train_components` は `best_training_params["early_stopping_rounds"]`
   があればそれを採る — **config が早期停止を無効にしていても**である。つまり
   study が config の設定を覆して早期停止を有効にでき、ゲートはそれを見ていなかった。

   実測（config 無効 / `category: training` で patience を 2 に tuning / `fit(params=)`
   で 10 を上書き）:

   ```
   tuned training params: {'early_stopping_rounds': 2, 'validation_ratio': 0.2}
   (上書き, callback rounds, iterations): [(10, [2], 4), (10, [2], 4), (10, [2], 4)]
   ```

   上書きは毎回 `lgb.train` に届き、全 booster が tuning 側の 2 で停止した。
   round 13 の指摘 3 と同じクラスで、**発動源が 1 つ違う**だけである。

   修正: `effective_early_stopping_rounds(cfg, training_overrides)` を
   「早期停止は有効か、patience はいくつか」の**唯一の定義**にし、
   `_build_train_components` と検査の**両方**がそれを使う。**この問いに 2 つの読みが
   あったこと自体が欠陥だった**ので、修理は 1 つにすることである。

2. **復元された `best_model_params` が同一層拒否をすり抜けていた（DC1 + DC4 + DC5）。**
   `overlay_params` は「重ねられる側」から競合綴りを落とすが、**overlay 自身が
   持っている**綴りはそのまま残す。`_merge_params` は overlay 内部の重複を検査して
   いなかった。実測:

   ```
   best_model_params = {"learning_rate": 0.1, "eta": 0.8}
     lgb.train には: [(0.1, 0.8), (0.1, 0.8), (0.1, 0.8)]
     booster:        [learning_rate: 0.1]
   ```

   レビュアーは実際の `export()` / `load()` 往復でも再現し、**主張しないこと**も明記
   した — 現在の `tune()` がそういう結果を作るとは言っていない。実際作れない
   （`check_duplicate_space_dimensions` が 2 次元 1 パラメーターを拒否する）。
   母集団は**本 PR より前に書かれた artifact** である。

   **そして格子はこのセルを問題なしと書いていた。**
   `tuning best_model_params × check_duplicate_identities` は
   `"n/a: overlaid by identity into a checked dict"` だった。この理由付けは**偽**である:
   overlay は「その下の層に対して」検査されるのであって、自分自身に対してではない。
   セルを `wired` にし、実行される入力を足した。**`wired` のセルには到達する入力が
   必要という harness が、この訂正を強制した。**

   修正: overlay の前に `best_model_params` へ `check_duplicate_identities`。
   `load()` 自体は依然としてその artifact を読む — artifact は「起きた fit の記録」で
   あり、読めなくして得をする人はいない。拒否は**再 fit** に属する。

3. **（非ブロッキング、ただし修正した）`params_table()` がエイリアス上書き後に
   過少報告していた。** `params_summary` は canonical 名の固定リストを booster の
   dict から**リテラル綴りで**読んでいた。実測:

   ```
   fit(params={"learning_rate": 0.5}) -> 表に learning_rate: 0.5
   fit(params={"eta": 0.5})           -> 表にどちらの名前も無い
                                         （booster はどちらでも 0.5 で学習）
   ```

   レビュアーは非ブロッキングとし「学習値ではなく報告の問題」と正しく限定した。
   **それでも修正した**: **本変更が動くようにするためだけに存在する経路**で run を
   誤報告するからであり、かつ round 13 の export 欠陥と**同じリテラル読み構文**だから
   である。前ラウンドで出荷した読み取り走査はこれを捕まえていない — `_DICT_NAMES` に
   `booster_params` が無く、キーがループ変数でリテラルでもない。レビュアーはそれを
   名指しし、それはその走査の docstring が自ら宣言している限界そのものである。

#### Firing rate

```
指摘 1: 出荷スイートで `category: training` の `early_stopping_rounds` 次元を持つ
        config は 1 件（`test_tuner_extended.py:38`）で、model 層の早期停止名は
        持たない。既存 config の拒否は 0 件。当該テストが通ることを確認済み。
指摘 2: 本リポジトリからは測定不能（母集団は旧版が書いた artifact であり、ここには
        無い）。代わりに境界を述べる: **`tune()` は今や重複綴りの
        `best_model_params` を作れない**（2 次元 1 パラメーターが study 前に拒否
        されるため）ので、これから書かれる artifact がこの拒否に掛かることはない。
非ブロッキング: 拒否は増えていない。空だった報告が埋まるだけである。
```

#### レビュアーが実行して clean と報告した内容

- **`_CELL_INPUTS` の 12 fixture すべてについて例外 traceback を検査**し、各々が
  名指しした checker に到達していること（別の理由で失敗しているのではないこと）を
  確認した。**これはこちらが自分の格子について実行できなかった検査である。**
- RAM 上の artifact I/O で実際の `export()` / `load()` 往復を実行 — 上書き無しの再 fit は
  config の `learning_rate=0.001` を復元し、新しい `eta=0.7` の上書きは 3 つの学習
  呼び出しすべてに届いた。**この経路はレビュアーによる実行としては round 7 以来である。**
- `export_code()` **と生成された `train_lgbm()`** を `metric` / `metrics` /
  `metric_types` で実行し、いずれも Brier 評価を保持し `learning_rate=0.5` の booster を
  学習した。

レビュアー自身が述べた限界: ディスク I/O をメモリ実装で置換したのでファイルシステムの
挙動は未検証、フルスイート・lint・mypy は再実行していない。こちらで実行 —
**2533 passed**、`ruff` / `ruff format --check` / `mypy` クリーン。

### 決定 12: 「唯一の定義」は 4 人の読み手を持っていた（rounds 14-15 monitor）

rounds 14-15 monitor は `CONVERGING` / `redirect` を返し、2 つの有限な列挙を名指しした。
**両方ともこちらで実行してから対応した。1 つは欠陥、1 つは clean。**

monitor は verdict では言えないことを 1 文で言った。記録として採用する:

> CONVERGING / DRIFTING の二分法が捉え損ねていて、15 ラウンド `APPROVE` が出ない実際の
> 理由はこれである — **maker が毎ラウンド、集合の上で実行していない普遍的な宣言を
> 出荷し、次のラウンドがそれを反証する。** 4 回連続である。

これはコードについてではなく**こちらの流儀について**の指摘であり、正しい。

#### 1 つ目の列挙 — 欠陥だった

決定 11 は `effective_early_stopping_rounds` を「**唯一の定義**」と宣言し、docstring は
trainer と検査が「食い違えない」と書いた。monitor はその問いの読み手を列挙した —
**4 人いて、宣言は 2 人の上でしか実行されていなかった。**

| 読み手 | 修正前 |
|---|---|
| `_model_factories.py` — 拒否 | 統一済み |
| `model.py` — trainer | 統一済み |
| `_model_persistence.py:239` — `export_code` へ | **config のみ** |
| `_model_tables.py:290` — `params_table` へ | **config のみ** |

実測（config patience 7、tuning patience 2）:

```
tuned patience  : 2
params_table    : 7
export_code     : 7
実際に使われた値: 2
```

**報告の問題より重い。** `export_code` は学習を再現するプロジェクトを生成するものであり、
**別のモデルを学習するプロジェクトを生成していた**。両方を共有定義に繋ぎ、4 人全員が
一致することを確認した。

monitor は帰結も明示した: この宣言はコミット `b737062`（round 15 の修正）にあるので、
**round 16 でここに指摘が出れば D7 の authorship 条件が round 15 に対して発火する**。
ラウンド前に処理することがそれを防ぐ、というのが rounds 10-11 / 11-12 で機能した先例
である。

#### 2 つ目の列挙 — 名指しされた死角のクラス、実行して clean

> **正しい名前で `lgb.train` に届いたパラメーターが、params dict ではない経路に
> 上書きされる。** この PR の計測器はすべて dict を読む。`adapter.py` は
> `num_boost_round=` を keyword で、早期停止を callback で、
> `categorical_feature=` / `weight=` を Dataset 構築時に渡す。
> `TRAINING_MANAGED_PARAMS` は 2 件しかないので、`num_iterations` と
> `categorical_feature` は**拒否も格子の列も持たない経路**である。

monitor は探し方も指定した: それらの経路をエイリアス表と掛け合わせ、
**booster が何をしたか**で判定せよ（`booster.params` ではなく — dict こそが見せない
ものだから）。その通りに実行し、**clean**:

```
num_boost_round 全 7 綴り     -> booster は 17 本（要求 17、config 6）、dict は空
categorical_feature 添字形式  -> booster に [categorical_feature: 0]
categorical_feature "name:"   -> LightGBMError が列名を挙げて明示的に失敗
```

`num_boost_round` は rounds 1-2 の修正が全綴りで効き続けている結果であり、`name:` の
失敗は**明示的**である（狩っているクラスは黙って負けることなので、明示的な失敗は
許容される側）。両方をテストで固定した — **clean の結果を durable にする半分がこれ**
である。

monitor は 3 つ目の候補（`lgb.Dataset` が `params=` 無しで構築される件）については
「`lgb.train` が未構築 Dataset に params を押し込む可能性があり、そうなら dead」として
**主張を控えた**。この自制があるからこそ、実際にした 2 つの主張は実行する価値があった。

#### 推奨は全面的に採用した

> `redirect` — 2 つの有限な列挙をラウンド前に実行し、**round 16 は範囲を絞らないこと**。
> probe-before-round は monitor が名指しした面を 2 度「指摘」ではなく「修正」に変えたが、
> ラウンドを絞ることは 4 度とも何も生まなかった。

**monitor の redirect が「絞るな」と推奨したのはこれが初めて**であり、しかもそれを
この run 自身の記録から導いている。

全スイート **2535 passed**。

### 決定 13: 報告面は「fit した model」を答える — D7 の authorship 条件が発火した（review round 16）

round 16（範囲を絞らない、head `bca3844`）は `REQUEST_CHANGES` を 2 件で返し、両方とも
こちらで再現した。

**1 件目は、決定 12 の修正自身が書いたコードの欠陥である。D7 の authorship 条件
（「round N の修正が書いたコードの欠陥が round N+1 で出る」）は、コミット `be2795a`
に対して発火した。** rounds 14-15 monitor はこの帰結を事前に明示していた。ここを弱めて
書かない — それは rounds 13-14 monitor が一度捕まえた癖そのものである。

#### 1 件目 — `params_table()` / `export_code()` が最新の tune を追いかけていた

決定 12 は 4 人の読み手を「実効値を再計算する」形で揃えた。だが `tune()` は
**tuning result を置き換えるだけで、fit 済み adapter は置き換えない**。実測:

```
fit した adapter が学習した patience : 7
params_table（tune 前）              : 7
tune 後、同じ adapter か             : True
params_table（tune 後）              : 2
export_code（tune 後）               : 2
```

**存在しない model について報告していた。** `export_code` は学習を再現するプロジェクトを
生成するので、決定 12 が閉じたはずの欠陥が向きを変えて戻ってきた形である。

reviewer の指示は明示的だった —「retained training state か、provider 経由で学習済み
adapter から解決せよ」。**両方を使った。どちらを使うかは値ごとに実行して決めた。**

#### 修正前に集合を列挙した — これがこの run の中心的な教訓

決定 12 の欠陥は「読み元を変える修正を、それを生んだ 1 つの lifecycle でしか実行して
いなかった」ことである。今回の集合は **lifecycle** であり、`export_code` が config から
渡す全引数と `params_table` が config から作る全行を、5 つの順序で実行した。

| 値 | 修正前 | 出所 |
|---|---|---|
| `early_stopping_rounds` | lifecycle 3 / 5b で誤り | **学習済み adapter**（`ExportParams` 経由） |
| `validation_ratio` | **lifecycle 2 で誤り（develop 由来、この PR の混入ではない）** | 保持した overlay（`FitState.applied_training_params`） |
| `seed` | 正しい | config のみ。trainer も `cfg.training.seed` しか読まない |
| `num_boost_round` | 正しい | 既に adapter 由来 |

5 つの lifecycle（`fit` / `tune→fit` / `fit→tune→report` / `fit→export→load` /
`fit→tune→export→load`）を全て実行した。

**`early_stopping_rounds` は adapter から読む。** `ExportParams` に
`early_stopping_rounds`（**default なし** — 「provider が設定しなかった」と「early
stopping が無効だった」を同じ値にするのは DC1 の形）を追加した。adapter は joblib で
保存されるので、これは lifecycle 5b でも正しい — **tuning result 経由の修正なら間違えて
いた唯一の lifecycle** である（artifact が持つ tuning result は、どの fit も消費していない）。

**`validation_ratio` は adapter に無い。** `FitState.applied_training_params` に、その fit が
実際に適用した overlay を保持する。実行して確認した内容:

```
inner-valid factory が呼ばれた ratio : [0.45]
params_table の validation_ratio     : 0.2   ← 修正前
```

`tuned_validation_ratio()` を唯一の定義とし、読み手 3 人（trainer / `params_table` /
`export_code`）を全て通した。

**述べる bound**: `load()` 後、この overlay は空である。artifact は tuning result を
記録するが「どの fit がそれを消費したか」を記録しないため、loaded model は config の
ratio に落ちる。これは `metadata.json` のキー追加＝変更ゲート案件なので、この PR では
やらない。**テストで固定し、issue に起票した。**

#### 2 件目 — comma 形式の比較が例外を出しうる（宣言違反）

`_comma_form_matches` の `float(element)` が `TypeError` / `ValueError` しか捕まえて
いなかった。`__float__` が `RuntimeError` を投げる `float` サブクラスで再現:

```
{'learning_rate': 0.5}              TRAINED True
{'eta': '0.5'}                      TRAINED True
{'learning_rate': 0.5, 'eta': '0.5'} RuntimeError conversion unavailable
```

module の宣言は「**この関数は `Exception` を送出しない**」「呼び出し側の値に触れる式は
すべて `try` の中にある」である。1 行下の `str(element)` は `try` の外にすらいなかった。
両方を広げ、`BaseException` は従来どおり伝播させる。

**durable な半分は、この cell が生まれた理由の方である。** 既存の cross product は
awkward な値どうしを比較するので `text` が `str` になることがなく、comma 形式の step は
要素に触れる前に `None` を返していた。**片側が文字列、片側が敵対的な要素の列** という
cell が存在しなかった。`_ELEMENT_BEHAVIOURS`（`__float__` 3 × `__str__` 2）を追加して
埋めた。

#### 範囲外として起票したもの

- **loaded model の `validation_ratio`**（上記 bound）— [#281](https://github.com/nbx-liz/LizyML/issues/281)。
- **`category: training` の `seed` 次元**: 実行すると受理・サンプル・`best_training_params`
  に格納されるが、trainer は `cfg.training.seed` しか読まないので**黙って無視される**
  （実測: 次元 123、学習は 0）。develop 由来で、この PR の経路上にない — [#282](https://github.com/nbx-liz/LizyML/issues/282)。

全スイート **2551 passed**。

### 決定 14: 失敗した呼び出しは、残るモデルの状態を書き換えてはならない（review round 17）

round 17（範囲を絞らない、head `04f3930`）は `REQUEST_CHANGES` を 2 件で返し、両方とも
こちらで再現した。

**2 件とも決定 13 の修正が書いたコードの欠陥である。D7 の authorship 条件はコミット
`a7ac071` に対して発火し、これで 2 ラウンド連続、しかも今回は 100%（古い欠陥は 1 件も
無い）。** この形は rounds 5-6 でも起きており、当時 D5 の停止条件判断の引き金になった。
再発を記録として残す。判断は standing instruction の下で管理者のものであり、
rounds 15-16 monitor の `escalate` は既に管理者へ上げてある。

#### 2 件に共通する 1 文

**どちらの修正も不変条件を宣言し、その不変条件の量化子ではなく「指摘が名指しした事例」
の上でしか実行していなかった。**

- 決定 13 の不変条件「`applied_training_params` は `fit_result` を記述する」は、
  **成功する 5 つの lifecycle** の上で実行され、失敗遷移はゼロだった。
- 決定 13 のもう 1 つ「呼び出し側の値に触れる式は guard の外に無い」は、
  **element 側の被演算子**の上で実行され、text 側は実行されなかった。

教訓は「もっと probe せよ」ではない。**実行すべき集合は不変条件自身の量化子**であり、
その量化子が「きっかけになった例」に黙って狭められた場所を見つけるのに、maker は
最も向いていない立場にいる、ということである。

#### 1 件目 — 拒否された fit が、残るモデルの状態を上書きしていた

`fit()` は各状態を「値が手に入った時点」で公開していた。それは fit が成功する前である。

```
その fit が学習した ratio : 0.2
tune 前                   : (0.2, 0.2)
tune 後                   : (0.2, 0.2)
拒否された fit            : LizyMLError
同じ adapter か           : True
拒否された fit の後       : (0.45, 0.45)
```

**修正前に集合を列挙して実行した。** `fit()` が成功前に書く `self._*` すべてを、
失敗点 2 つ（gate による拒否 / 学習中の例外）で実行した:

| 書かれるもの | 拒否時（修正前） | 修正後 |
|---|---|---|
| `_applied_training_params` | **上書き** | 保持 |
| `_X` / `_y` | **上書き** — 200 行が、残るモデルが見たことのない列を持つ 90 行に | 保持 |
| `_metrics` | refit が失敗しうる前に公開 | 結果と一緒に commit |
| `_provider` | 書き換わるが `cfg` 由来で、呼び出し間で変わりえない | 不変 |
| `_run_dir` | 保持 | 保持 |

`_applied_training_params` はこの PR のコードである。**`_X` / `_y` は既存**であり
（SHAP と診断が読むデータ）、修正が同じ 1 行の移動であり `fit()` 内で mid-flight に
読む箇所が無いので同梱した。

同じ問いを隣のメソッドで実行すると `tune()` も同じことをしていた: `self._X, self._y`
が study の前にあり、study が例外を投げると残る fit が見たことのない行を指していた。

**修正は「結果と一緒に commit する」こと。** 報告面が「その fit について」読む値は、
`fit()` の末尾で 1 かたまりとして、間に例外を投げうる文を置かずに公開する。`tune()` も
同様。

#### 2 件目 — text 側の被演算子が comma 形式の step を破れた

round 16 は `float(element)` と `str(element)` を guard した。`text.split(",")` は
一度も guard されておらず、**`str` サブクラスは `str` である**以上その override が走る。

**修正は「もう 1 つの `try`」ではない。** サブクラスのメソッド呼び出しを `try` で包めば
no-raise は満たすが、**正当な組（1 パラメーターを 2 通りに書いただけ）を拒否する**。
それはこの module が防ごうとしている欠陥のもう半分である。この module 自身の idiom は
**基底メソッドを unbound で呼ぶ**ことで、override を走らせない。

書く前に 2 つの事実を実行した:

```
str.split unbound      : ['0.5'] ['str']
element str() subclass : Text          <- str(element) はサブクラスを返しうる
```

2 つ目があるので `printed.strip()` も同じ扱いが要った。docstring は母集団を
**式 × 所有者 × 閉じ方**の表として持ち、module の宣言も「`try` の中、**または**基底型の
メソッドを unbound で」と書き換えた（後者は `_printed_forms_differ` で既に使われていた
のに宣言が触れていなかった）。

#### 計測器も拡張した — reviewer がその欠落を名指ししたため

> 「lifecycle grid は失敗した fit の遷移を含まないので、その 48 cell はこの性質を
> 確立できない。」

正しい。しかも同じ失敗の 1 段上である: 6 つの lifecycle はすべて成功系列だった。
`report_lifecycle_grid.py` は現在 8 lifecycle・**64 cell** を実行する。

```
cells: 64    agrees: 54   known-bound: 2   n/a: 8   DISAGREES: 0
```

#### 誤った理由で通っていた回帰テスト

このラウンドの remedy の中に、狩っているクラスが出たので記録する。`_X` / `_y` の主張の
最初の版は `fit → tune → 拒否された fit` を走らせていた。**`tune()` も `_X` / `_y` を
代入する**ので、`fit()` が一度も代入しないビルドに対しても通っていた。RED 検証が
「赤にならない」ことで捕まえた。現在は tune を挟まない独立したテストにしてある。

RED harness 側も同じ形の誤りだった: 移動した代入を**削除**するのは「修正前」ではない
（テストが直前の成功呼び出しの残りを読んでしまう）。両方とも元の位置へ**戻す**形にした。

全スイート **2592 passed**。

### 決定 15: 充足不能だったのは宣言のほうだった（review round 18、範囲限定）

round 18 は**修正箇所に限定して**開いた。rounds 15-16 / 16-17 の monitor が 2 回連続で
`DRIFTING` / `escalate` を返し、2 回目が「同じことを繰り返して今度はうまくいく、とは
言えない」と明言したため、round 18 を開かずに管理者の判断を仰いだ結果である。判断は
**「修正箇所に絞って開く」** — round 5 で管理者自身が出した第 4 の選択肢と同じ形。

#### 範囲限定ラウンドが取りに行った成果は、実際に得られた

**round 17 の状態公開の修正は、正面から攻撃されて持ちこたえた。** 3 ラウンドぶりに、
修正が clean で返ってきた最初の例である。

> CV 学習 / calibration / evaluation / full-data refit に失敗を注入。grouped fit
> フィールド 6 個すべてが従前のオブジェクト同一性を保持。`_run_tune_round` と
> `_assemble_tuning_result` での tune 失敗でも、検査した 12 フィールドすべて保持。
> 成功パスでの属性アクセスも計測し、6 フィールドは実行中に読まれないことを確認。

出荷した計測器も実行され、数字が完全に再現された（64 cells / 54 agree / 2 known-bound
/ 8 n/a / exit 0）。新しい回帰ケースは mutation test にかけられ、各々が「固定すると
主張している本番行」を実際に検出することが確認された。

#### 指摘は 1 件、そして**それは D7 authorship の 3 ラウンド連続発火**（`d83af2b`）

round 17 は `str.split` を unbound で呼ぶことでサブクラスの override を封じた。だが
**`isinstance` は `__class__` を読み、unbound descriptor は `type()` を読む**。
両者が食い違うのが **proxy**（`str` ではないが `__class__` に `str` を返す）である。

```
単独ではどちらも学習、booster 同一 : True
組にすると                        : TypeError: descriptor 'split' ...
```

round 17 の指摘は「比較すべきところで**送出した**」。今回はその鏡像で「比較すべきところで
**拒否した**」。bound の両半分とも効いており、round 17 の修正は前者を買って後者を売った。

#### 根本原因は guard ではなく**宣言**だった

reviewer は non-blocking として「`__class__` が raise する値は `isinstance` の時点で
bound を破り、それは `04f3930` でも同じだった」と報告した。これが 3 ラウンド続いた理由を
説明する: 宣言

> *この関数は `Exception` を送出しない*

は**あらゆる Python オブジェクト**を量化しており、**どんな実装でも充足できない**
（`__getattribute__` や `__class__` を raise させれば任意の式が落ちる）。
**これは宣言そのものに対する DC7 である。** だから毎ラウンド新しい dunder が見つかった。

#### module 自身が名指ししている権威に訊いた

docstring は `_param_dict_to_str` を値等価性の権威として既に名指ししている。推論ではなく
実行した:

```
Proxy (isinstance は str、type は違う)  -> 'learning_rate=0.5'
Rate (__class__ が raise)              -> RuntimeError: class unavailable
Text (素の str サブクラス)              -> 'learning_rate=0.5'
0.5                                    -> 'learning_rate=0.5'

type(str(Text('0.5')))          : str
type(str(SelfPrinting('0.5')))  : SelfPrinting
str.split はサブクラスを受ける   : ['0.5']
```

- **proxy は admit しなければならない。** serialiser が `learning_rate=0.5` を出す以上、
  LightGBM が学習する値である。`text` は unbound 呼び出しの前に `str()` で正規化する
  （serialiser 自身がそれに対して行うのと同じ操作）。
- **`__class__` が raise する値は bound の外であり、どんな guard でも変わらない。**
  serialiser 自体が呼び出し側の例外を投げるので、その値はどの綴りでも**学習を完了できず**、
  admit すべき組が存在しない。（当初この行は「`lgb.train` に届かない」と書いていた。
  rounds 17-18 monitor が経路を実行して訂正した — 直列化は LightGBM の**内側**で起きるので
  `lgb.train` には入り、そこで失敗する。支持できる主張はここに書いた形である。）

**bound を serialiser 相対に書き換えた:**

> *serialiser が受理する値については `Exception` を送出せず、serialiser が 1 つの値と
> みなす組は admit する。*

これは**閉じた、実行可能な母集団**である。旧来の宣言は開いており、3 ラウンドはそこに
費やされた。`_comma_form_matches` の式の表には `isinstance` を「**ここでは閉じない**」
という closure で 1 行加え、理由を書いた。

#### テストを「列挙」から「オラクルとの関係」に変えた

生成した cross product に対する `isinstance(result, bool)` は列挙であり、列挙こそが
古び続けたものである。text 側に `__class__` 軸（`normal` / `proxy` / `raises`）を加えて
108 通りにし、主張を**オラクルとの関係**にした:

> `values_differ` が送出するのは、`_param_dict_to_str` が送出するときだけである。

さらに「関係が空虚でないこと」を確かめる相棒テストを置いた。前件が一度も成立しない関係は
どんな実装でも満たされる — **列挙から逃げるために書いたテストの中の DC6** である。

宣言個数の guard は、新しい軸を初回実行で捕まえた（設計通り）。

全スイート **2666 passed**、grid exit 0。

### 決定 16: 欠けていたのは軸ではなく**関係**だった（review round 19、範囲限定）

round 19 も**修正箇所に限定して**開いた（round 18 と同じ枠組み）。blocking 1 件、
non-blocking 1 件。**2 件とも `fdf5cc2`（round 18 の修正）由来で、D7 authorship は
4 ラウンド連続の発火である。**

#### 1 件目 — 正規化が「別の formatter」を使っていた

round 18 は `str` 側の被演算子を `str()` で正規化した。だが serialiser は
**スカラー**を `f"{key}={val}"`（= `__format__`）で書き、**列の要素**は `_to_string`
（= `str`）で書く。この 2 つは片方だけを override した値で食い違い、両方向に出た:

```
F1a  wire form : a=0.5 | a=0.5   -> 1 つの値なのに values_differ True（拒否）
F1b  wire form : a=0.25 | a=0.5  -> 2 つの値なのに values_differ False（受理, DC1）
```

F1b が重い。`learning_rate=0.25` と `eta=0.5` が「同じ」と報告され、LightGBM が
残した方で学習していた。

**書く前に対応関係を実行した**（8 値 × 両 formatter）。`format` はスカラーの wire form に
8/8 一致、`str` は要素の wire form に 8/8 一致。**LightGBM が 2 つの関数を使い分けている
から、こちらも使い分ける**。下流の `str(element)` は要素用で正しく、揃えてはいけない。

#### 2 件目 — non-blocking と報告されたが修正した

`getattr(value, "tolist", None)` は `AttributeError` しか飲まないので、raise する
`tolist` **プロパティ**はそのまま外に出ていた。reviewer は round 18 より前からある
として non-blocking にしたが、**1 ラウンド前にこの PR が宣言した bound を反証する**ので
修正した（serialiser はその値を受理し、学習もできる ＝ bound の内側。決定 15 が除外した
`__class__` の raise とは違う）。

#### 本当の修復 — 関係の欠けていた半分

round 18 は列挙をオラクル関係に置き換えたが、**半分しか作っていなかった**:
「serialiser が raise するときだけ raise する」。もう半分「serialiser が 1 つの値と
みなす組は admit する」は一切主張していなかった。round 19 の両方向は、no-raise テストを
**構成上必ず通る**。

`__format__` を軸に足せば「この 1 例」は捕まる。**関係を足せばどの軸でも捕まる**:

- wire form が同じ ⟹ `values_differ` は `False`
- 両方が数値で異なる ⟹ `values_differ` は `True`

**存在した瞬間に 3 件見つけた:**

| 内容 | 処置 |
|---|---|
| `"0.5"` と proxy、両順（8 cell） | **修正** — 決定 15 自身の穴 |
| スカラー と 単一要素の列（4 cell） | **[#283](https://github.com/nbx-liz/LizyML/issues/283) に起票** |

proxy の穴は示唆的である: round 18 は `_comma_form_matches` の**中**で正規化しており、
そこは相手が列のときしか走らない。だから proxy を素のテキストと比べる経路は素通りだった。
**正規化は入口で 1 度、両方の被演算子に**行うようにした。

スカラー/単一要素の件を直さないのは、**現在拒否している組を admit するのは振る舞いの拡大**
（変更ゲートの `allow`）であり、実測 firing rate 付きの Proposal が要るからである。値は
母集団に残し `KNOWN_BOUNDS` に issue 番号を書いた（関係の非空虚性の witness を失わない
ため）。免除が不要になったら落ちる staleness 検査も置いた。

#### 母集団を「宣言」から「導出」へ

4 ラウンドかけて `__float__` → `__str__` → `split` → `__class__` → `__format__` /
`tolist` と、毎回「前のラウンドが思いつかなかった軸」を足してきた。**すべて人が選んだので、
すべて不完全だった。**

`DERIVED_HOSTILE_NAMES` は、この module が扱う型に対する **Python 自身の dunder 一覧**と、
module が文字列で引く属性名から導出する。各 1 個ずつ敵対化して関係に通す。導出が導出で
あることを確かめる相棒テストも置いた（明示 lookup 名が module のソースに実在するかまで
見るので、写しが古びれば落ちる）。**何も出なかった。それが走らせた意味である。**

#### 赤にならなかった RED 検証

記録しておく。最初の RED 実行で `format` を `str` に戻したのに**スイートは緑のままだった** —
reviewer の再現値を 1 つもテストに固定していなかったので、母集団に「`format` と `str` が
食い違う値」が存在しなかった。両方を、生成母集団にも出荷経路にも追加した。

全スイート **2895 passed / 6 skipped**、grid exit 0。

### 決定 17: formatter の返り値も呼び出し側のもの、そして関係が古い契約と矛盾していた（review round 20、範囲限定）

round 20 も**修正箇所に限定**して開いた。問いは rounds 18-19 monitor が名指しした識別子で
ある — **「強めた表明と宣言した bound は、直前の修正への再修正なしに独立の挑戦を生き延びるか」**。

**答えは「否」。** blocking 2 件、両方とも round 19 の修正由来。

#### 事前登録した停止条件が発火した

> **D7 の authorship 条件は 5 ラウンド連続で発火し、5 件すべてが
> `lizyml/core/value_equality.py` 内である**（rounds 16, 17, 18, 19, 20）。

この条件は round 20 を回す**前に**登録し、管理者に伝え、rounds 18-19 monitor にも示して
「適切である」と評価されていた。**判断は管理者に上げ、round 21 は開かない。**
選択肢は `DECISIONS-PENDING.md` の D8 にある。ただし**指摘自体は実在するので修正した** —
ループを止めるのは「次のラウンド」についての判断であって、既知の欠陥を出荷する理由ではない。

#### 1 件目 — formatter の返り値が override 可能な振る舞いを保持していた

round 19 は `format(value, "")` で正規化し、**返り値をそのまま返していた**。`format` は
`str` の**サブクラス**を返しうるので、そのサブクラスの override が比較を決めてしまう。

```
wire form : k=0.25 | k=0.50   -> 2 つの値なのに values_differ False（DC1）
wire form : k=0.5  | k=0.5    -> 1 つの値なのに values_differ True
```

前者は学習まで届く（両綴りを単独で学習させると booster は 0.25 と 0.5 で別物）。

**修復は module 自身の idiom を 1 段先に適用すること。** formatter の返り値に対して
`str.__str__` を unbound で呼ぶ。書く前に実行した:

```
str(x)          -> 'lied'   exact=False  len=99
str.__str__(x)  -> '0.25'   exact=True   len=4
```

`str()` では足りない — `__str__` に dispatch し、同じオブジェクトがそれも override
しうる。基底メソッドだけが LightGBM に実際に届く文字を返す。

#### 2 件目 — 新しい関係が、古い意図的な契約と矛盾していた

serialiser は `nan` を平然と書くので、「wire form が同じ ⟹ admit」は **2 つの NaN を
1 つの値として admit せよ**と要求する。だが module は**意図的に**それらを「異なる」と
報告し、理由も round 5 から書いてある（`nan != nan` なので、同じ値だと確立できるものが
何もない）。

round 19 は、serialiser を量化子に据えた関係を、**その量化子が指す契約と突き合わせずに**
書いた。**それは関係を導入して終わらせようとした失敗そのものの 1 段上である。**

NaN は理由付きの**宣言された例外**にした。そしてそのコストは論証ではなく実行で示した:
`_fit(learning_rate=nan)` は LightGBM が拒否する（`Check failed: (learning_rate) > (0.0)`）
ので、NaN の組にどんな verdict を出しても学習するモデルは変わらない。NaN は**例外だから
こそ**母集団に入れてある（宣言だけあって誰も到達しない例外は、この run が狩っている形）。

#### 3 度目の「誤った理由で通るテスト」

この PR で 3 度目であり、毎回 RED 検証だけが捕まえた。記録しておく。

1 件目の出荷経路の pin は、もう一方の綴りを最初 **float**（`eta: 0.50`）で書いていた。
数値は comma 形式の step を通り、そこは修正を戻しても正しく拒否する — つまりテストは
**何も固定しないまま通っていた**。reviewer 自身の再現に合わせてテキストにしたところ、
revert で 2 件とも赤になった。

#### 報告されたが直していないもの

- **`values_differ([], "")` が `True`**（両方 `k=` を書く）。`0b45250` でも同じで、
  範囲外。non-blocking。
- **導出チェックが「新しく追加された文字列 lookup」を検出しない。** reviewer が module の
  ソースをメモリ上で書き換えて lookup を 1 つ足したところ、相棒テストは通った。
  「宣言済みの名前がまだ在るか」は見ているが「実際の lookup が全部宣言されているか」は
  見ていない。現時点で欠けている lookup は無い。**こちらが書いた検査の実際の弱点である。**

全スイート **2898 passed / 62 skipped**、grid exit 0。

---

## H-0095: パラメーター値を入口で正規化し、比較の領域を閉じる（#264 後継 / D8 の帰結）

- **ステータス**: Accepted
- **起票日**: 2026-09-08
- **決定日**: 2026-09-08
- **スコープ**: `lizyml/core/_model_factories.py`（入口の正規化 + 受理集合の拒否）, `lizyml/core/value_equality.py`（閉じた型集合の上へ縮小）, `lizyml/core/model.py` / `lizyml/calibration/`（4 つの呼び出し元が正規化後の dict を使う）, `tests/test_core/test_value_equality.py`, `tests/test_core/test_fit_params_override.py`, `BLUEPRINT.md` §14.4, `CHANGELOG.md`。
- **関連**: H-0094（PR 2、決定 1-17）, [#264](https://github.com/nbx-liz/LizyML/issues/264), [#283](https://github.com/nbx-liz/LizyML/issues/283), `docs/audits/2026-09-defect-discovery/DECISIONS-PENDING.md` の **D8**。

### 目的（課題）

PR 2（#278）のレビューは **20 ラウンド回って `APPROVE` に到達しなかった**。うち
**rounds 16-20 は 5 連続で、直前のラウンドの修正が書いたコードに欠陥が出た**（D7 の
authorship 条件が 5 回発火）。**5 件すべてが `lizyml/core/value_equality.py` 内**である。

原因は個々の guard 漏れではなく、**この関数が判定しなければならない入力領域が開いている**
ことである。そしてそれが開いているのには 2 つの理由があった:

1. **型を名指せなかった。** このモジュールは自ら「標準ライブラリのみ」を課しており、
   numpy を `isinstance` で判定できないので `tolist` を探す等の duck typing に頼っていた。
   任意のオブジェクトが `__format__` / `__class__` / `tolist` / `__eq__` / `__len__` を
   どうにでも定義できるので、領域は構成上開く。
2. **誤拒否も欠陥だと、レビューが 2 度押し返した**（round 12 finding 1、round 13
   finding 2、どちらも DC7）。admit を増やすほど、任意オブジェクトについて理解すべき
   ことが増える。

**訂正すべき前提が 1 つある。** 「Layer 0 = 標準ライブラリのみ」は**アーキテクチャの規則
ではない**。`ARCHITECTURE.md` の「依存ゼロ」は*内部レイヤ*依存ゼロの意味で、Layer 0 の
他モジュール（`core/types/fit_result.py`, `core/_model_factories.py` 等）は numpy も
pandas も import している。**numpy を型として名指すことは最初から許されていた。**

### 提案

**パラメーター値を、surface の入口で 1 度だけ正規化する。** 受理集合の外は、学習が始まる
前に `CONFIG_INVALID` で拒否する。

受理集合は LightGBM 自身の受理集合から導出する（`lightgbm/basic.py`:
`_NUMERIC_TYPES = (int, float, bool)`、スカラーは
`isinstance(val, (str, Path, _NUMERIC_TYPES)) or _is_numeric(val)`、列は
`list` / `tuple` / `set` / 1-D ndarray）:

**⚠️ この表は superseded である。** 実装時の実測で 2 行が誤りと分かり（1-D ndarray /
`str` サブクラス）、以後 19 件の補正が積み上がった。**受理集合の正は本節末尾の
「契約の確定」であり、そこのブロックは生成物である。** 以下は提案が何を提案したかの
記録として残す。

| 入力 | 正規化後 |
|---|---|
| `None` / `bool` / `int` / `float` / `str` | そのまま（`str` サブクラスは厳密な `str` へ） |
| `Path` | そのまま |
| numpy スカラー | `.item()` |
| 1-D `ndarray` | `.tolist()` |
| `tuple` / `set` | `list` |
| 上記の列 | 要素ごとに同じ正規化 |
| **それ以外** | **`CONFIG_INVALID`（入口で、パラメーター名と受理集合を明示）** |

**文字列化はしない。** 素の型のまま正規化する — smart params の解決や boundary 展開は
数値演算をするので、文字列にすると壊れる。

配線先は既にある。`check_duplicate_identities` は **4 つの surface すべてが通る唯一の絞り**
として rounds 10-12 で配線・固定済みである（実測: `model.params` / `fit(params=)` /
`calibration.params` / `tuning best_model_params`）。ここを「検査するだけ」から
「検査して正規化した dict を返す」に変える。

### 影響範囲

`value_equality.py` は**閉じた素の型集合**の上でのみ動くようになり、劇的に縮む。
`__format__` / `__class__` / `tolist` / `__eq__` の敵対的実装は**入口を通らない**ので、
rounds 16-20 の指摘クラスは丸ごと消滅する。

`fit(params=...)` の型注釈は `dict[str, Any]` のままだが、**実効的な入力契約が狭まる**
ので公開 API の変更として扱う。

### 互換性

**LightGBM が受理するものの一部を、この提案は拒否する。** `_is_numeric` は `float(obj)`
が通れば何でも受けるので、独自 `__float__` を持つオブジェクトは LightGBM 的には有効である。

```
Firing rate: 7/1430 of every parameter value the suite constructs
             (measured by wrapping the shared identity check over the full
             suite at head 1403ba8; 2898 passed, 62 skipped)
```

内訳: `int` 1082 / `float` 187 / `str` 78 / `list` 45 / `ndarray` 14 / `tuple` 8 /
`bool` 6 / `None` 3 — **1423 件は受理集合の内側**。列の要素も全て素の型
（`float` 52 / `str` 22 / `int` 2）。**残る 7 件は `Equivalent` 2 / `Proxy` 2 /
`Conflicting` 1 / `FormatsToLiar` 1 / `Rate` 1 で、すべて rounds 16-20 が自分で構築した
敵対オブジェクトである。** 現実的な config 由来のものは 1 件も無い。

計測器は `docs/audits/2026-09-defect-discovery/instruments/parameter_value_type_census.py`
として**出荷する**（散文に写した数は古びる）。

**述べる bound**: これはこのリポジトリが構成する母集団であって、ライブラリのユーザー
コードは観測できない。だから拒否は「現実には何も拒否しない」ではなく
**「入口で、明示的に、学習前に拒否する」**として正当化する。**狩っているのは
「黙って違う値で学習する」ことなので、大声で拒否する側は許容できる半分である。**

### 代替案

**E: 同一性判定を provider protocol の背後へ移し、`_param_dict_to_str` を同一性の定義に
使う。** Codex（`gpt-6-astra`, effort medium）の評価では E > D > B > A > C で第 1 位
だったが、**実行して却下した**:

```
pair                      wire A     wire B    wire一致  現状
[1, 2]      vs '1,2'      1,2        1,2       True     admit
[1.0, 2.0]  vs '1,2'      1.0,2.0    1,2       False    admit  <- E なら誤拒否
(1.0, 2.0)  vs [1, 2]     1.0,2.0    1,2       False    admit  <- E なら誤拒否
0.5         vs '0.50'     0.5        0.50      False    admit  <- E なら誤拒否
```

**wire form は正準形ではない。** `_comma_form_matches` の docstring が既にそう書いている
—「比較が textual でなく elementwise なのは wire form が正準でないからで、joined string を
比較すると**この関数が除去するために存在する誤拒否そのもの**を起こす」。E は
**round 13 finding 2 の修正を、その理由が書いてある行ごと元に戻す**。数値を意識した比較を
wire の上に足せば救えるが、それは再実装に戻ることで E の存在理由が消える。

E の副次的コストも記録する: `_param_dict_to_str` は private であり `pyproject.toml` は
`lightgbm>=4.0` を許すのでバージョン幅の検証が要る、公開 protocol が 18 → 19 メソッドに
なる、実装を `estimators/` へ移す必要がある（F は `core/` のままでよい）。

**B: contract を狭める（round 13 の admission 撤回）** — E と同じ理由で誤拒否を再導入する。
**A: 範囲限定で続行** — rounds 18-19 monitor 自身が「blocking 数の減少はどちらの区別にも
ならない」と述べ、round 20 はその monitor の識別子に不合格になった。
**C: `APPROVE` を要求しない** — 受入要件を終わらせるだけで根拠を解決しない。
**D: レビュアーへの問いを変える** — 失敗した成果物の作者が受入基準を書き換える利益相反。

### 受け入れ基準（テスト観点）

1. **正規化は wire form を保存する。** 受理集合の全要素について
   `_param_dict_to_str({"k": normalise(x)}) == _param_dict_to_str({"k": x})`。
   **閉じた実行可能な性質**であり、既にテストにあるオラクルでそのまま書ける。これが
   本提案の中心的な受け入れ基準である。
2. **受理集合は LightGBM の受理集合から導出し、写さない。** 導出が導出であることを
   確かめるテストを置く（H-0094 決定 16 の `DERIVED_HOSTILE_NAMES` と同じ形）。
3. **拒否は入口で、学習前に、パラメーター名と受理集合を挙げて起きる。** `CONFIG_INVALID`。
4. **4 つの surface すべてで正規化が効く。** `model.params` / `fit(params=)` /
   `calibration.params` / `tuning best_model_params` — 宣言ではなく実行で確認する。
5. **rounds 16-20 の敵対オブジェクトが全て入口で拒否される。** 既存の回帰テストは
   「学習する」から「入口で拒否される」へ意味が変わるので、**削除せず書き換える**。
6. **`values_differ` は閉じた型集合の上で全域である。** 敵対母集団は入口を通らないので、
   H-0094 決定 16 の導出母集団は「入口の拒否」を確かめる側へ移る。
7. **#283（スカラー vs 単一要素の列）をこの提案で解決するかを明示的に決める。**
   正規化後は両者とも素の型なので、判断材料が揃う。


### 実装時の実測による提案の補正（2026-09-08）

受け入れ基準 1（wire 保存）を先に書いて実行したところ、**提案の受理表が 2 か所間違って
いた**。どちらも「型で正規化すれば bytes は変わらない」という暗黙の前提から来ており、
実際にはシリアライザが**位置によって別のフォーマッタを使う**ことが効く。

```
_param_dict_to_str:  スカラー位置  -> f"{key}={val}"  = __format__
                     要素位置      -> _to_string(v)   = str
```

1. **`1-D ndarray` → `.tolist()` は誤り。** 要素位置のフォーマッタは `str` であり、numpy の
   `str` と Python の `str` は同じ数値に対して別のテキストを書く。実測（27 通り中 8 件が
   不一致）:

   ```
   np.array([0.1], float32)   wire: 0.1     .tolist() 後: 0.10000000149011612
   np.array([0.1], float16)   wire: 0.1     .tolist() 後: 0.0999755859375
   ```

   **正しい要素変換は「その要素と同じテキストを印字する素の値」**。多くは `.item()` が
   それであり、そうでないところはテキストを parse し直す。スカラー位置は `.item()` で
   全 dtype 一致（実測 0 件の不一致）。

2. **`str` サブクラス → 厳密な `str` も誤り。** `__format__` を上書きしたサブクラスは
   スカラー位置で別の bytes を書くので、`str()` に落とすと wire が変わる。**厳密な型一致
   で受理し、サブクラスは拒否する**（rounds 18-20 の 3 オブジェクトがこれに当たる）。
   実測コストは 0 件。

3. **入れ子リストを受理表に追加。** `_to_string` は `list` 要素だけを
   `[` + カンマ結合 + `]` として書く。これは `interaction_constraints` の綴りそのもの
   なので受理する。要素位置の `tuple` / `set` / ndarray は Python や numpy の repr
   （`(1, 2)` / `[1 2]`）になり LightGBM が読めないため拒否する。

4. **素の代替が存在しない値は変換せずに拒否する。** `str(np.float16(1e3))` は `1e+03` で、
   これを印字する Python の数値は存在しない。丸めて通せば**呼び出し元が書いていない bytes
   で学習する**ことになるので、ここは拒否側に倒す。テストは「拒否が強制されたものである
   こと」（＝どの素の値も同じテキストを印字しないこと）を毎ケース検査する。

5. **受け入れ基準 8 を追加: 出口の表明。** 4 surface で正規化するのは**配線についての
   主張**であり、5 つ目の経路が後から足されたときに黙って崩れる（DC4）。
   `assert_plain_params` を `lgb.train` の 2 か所（`estimators/lgbm/adapter.py`、
   `calibration/isotonic.py`）に置き、**学習サイトの母集団をソースから導出して**
   全サイトが通っていることをテストで固定する。これで「閉じている」は主張ではなく性質に
   なる。

6. **`calibration.params` は 2 か所で正規化する。** 検査側（`check_calibration_param_names`）
   と、calibrator に渡る dict を作る側（`canonicalise_calibration_params`）の両方。
   calibrator は他の 3 surface を通らずに `lgbm.train` へ到達するため。正規化は冪等で
   あることをテストで固定してある。

7. **mapping を受理集合に加えた（実装中に発見）。** `metric` は
   `{"precision_at_k": {"k": 15}}` および `["auc", {"precision_at_k": {"k": 20}}]`
   という **LizyML の形**を持つ（H-0065）。提案の受理表にはこれが無く、そのままなら
   **出荷済みの設定形式を入口で拒否していた**（DC7 を自分で作るところだった）。
   `_build_params` がこれを feval に変換して除去するので、**入口と出口で受理集合が
   異なる**のが正しい: 入口は mapping を受理し、`lgb.train` の表明は拒否する。
   mapping のキーは厳密な `str`、値は再帰的に正規化する。

8. **`set` は入口で拒否する（振る舞いの縮小）。** 旧 `values_differ` は
   `{1.0, 2.0}` と `[1.0, 2.0]` を**別の値**として拒否していた。理由は「set に順序が
   無く、ここの列パラメーターはすべて位置依存だから、認めると答えがハッシュ順に依存する」
   というもので、**その理由は入口正規化でも消えない**。`list({1.0, 2.0})` は確かに
   シリアライザが書いたはずの bytes を書くが、それが**リテラルの `[1.0, 2.0]` と一致
   するかどうかはハッシュ順の偶然**である（実測: `list({3.0, 1.0, 2.0})` は
   `[3.0, 1.0, 2.0]` ではない）。よって `set` / `frozenset` は受理集合から外し、
   入口で `CONFIG_INVALID` にする。拒否は縮小側なので変更ゲートの `allow` には当たらない。
   実測コスト **0/1518**（この repository が構成するパラメーター値に set は 1 件も無い）。

8b. **リストの入れ子は深さ 2 まで（実装中に発見し、自己レビューで修正）。** mapping を
   受理する過程で `_plain_member` を再帰にしたところ、深さ 3 のリストが受理された。
   `_to_string` が意味を与えるのは深さ 2 まで（`interaction_constraints`）で、
   3 段目は Python の list repr で書かれるため **wire が変わる**。実測:
   `[[[np.float32(0.1)]]]` は `[[np.float32(0.1)]]` と書かれ、正規化後は `[[0.1]]`。
   深さ 3 以上は拒否する。**母集団が深さ 3 を生成していなかったので wire 保存の性質
   テストはこれを見られなかった** — 母集団に深さ 3 と set を加え、
   「拒否は宣言した 3 つの理由のいずれかに当たる」ことを毎ケース検査するようにした。

9. **`value_equality.py` を閉じた集合の上へ縮めた（受け入れ基準 6）。** 494 行 →
   約 140 行。消えたのは敵対オブジェクト向けの防御だけである:
   `_as_wire_text`（`__format__` / `__class__` プロキシ対策）、`_as_plain_python`
   （`tolist` 探索）、`_as_plain_sequence`、`_length_or_none`、
   `_printed_forms_differ`（`repr` の床）、および unbound な `str.split` /
   `str.strip` / `str.__str__` 呼び出しと広い `except Exception`。**残したのは
   round 12/13 が買った admission**（列とそのカンマ形は 1 つの値、比較は textual では
   なく elementwise）である。宣言する bound は
   **「`param_domain` が受理する値の上で全域であり、そのどれでも raise しない」**に
   変わった。有限で列挙可能なので、**受理母集団を import して実行する**ことで確かめる
   （散文で言い直さない ＝ DC3 回避）。

#### 実装後の実測

```
Firing rate: 14/1518 of every parameter value the suite constructs
             (measured by wrapping `normalise_params` over the full suite
             after implementation)
```

14 件の内訳は rounds 16-20 の敵対オブジェクト 10 件と、拒否経路を実行するために
`test_refusal_matrix.py` が構築した 4 件で、**すべてこの PR 自身のテストが作ったもの**。
実運用の config 由来の値は 1 件も拒否されていない。計測器
`instruments/parameter_value_type_census.py` は `normalise_params` を包むように更新済み。

**スイート**: 7443 passed / 256 skipped、`ruff` / `mypy` clean。

#### review round 21 が見つけた 3 件（2026-09-08、unscoped）

**関係監視が事前に宣言した反証条件が、finding 1 でちょうど発火した** —
「導出が生成しなかった形であって、受理すると `lgb.train` に届く bytes が変わるもの」。

10. **numpy を継承で受理していた（DC1、`deliverable-path`）。** `isinstance(value,
    np.generic)` は `np.float64` の**サブクラス**（`__format__` が嘘をつく）と
    **`np.timedelta64`** を通した。後者が非自明で、実行して分かった:
    **`np.timedelta64` は `np.integer` のサブクラスである。** 実測:

    ```
    値                                      caller の wire    学習に届く wire
    np.float64 サブクラス（__format__ が嘘）    0.9              0.1
    np.timedelta64(1, "ns")                  1 nanoseconds    1
    ```

    どちらも fit は完了する。出口の表明は変換**後**の素の値を見るので検出できない。
    修正は **2 段の防御**で、それぞれ別のものを買う:
    - **`NUMPY_SCALAR_TYPES` を numpy 自身の階層から導出し厳密型一致で受理** →
      買うのは「**正規化中に呼び出し元のコードが 1 行も走らない**」こと。
      `.item()` も `__format__` も numpy 自身の実装になり、rounds 16-20 の軸が
      構成上消える。
    - **`format(plain, "") != format(value, "")` なら拒否** → `timedelta64` を
      捕まえるのはこちら（型集合の中にいるので 1 段目では捕まらない）。

    RED 検証で 2 段が別々に効いていることを確認した。

11. **`values_differ` が全域でなかった（DC7、`deliverable-path`）。** Python の `int` に
    幅は無いので `10**400` は受理集合の内側のごく普通の値（シリアライザは桁を書く）だが、
    `float()` は `OverflowError` を投げ、`(TypeError, ValueError)` しか囲っていなかった。
    **「受理集合の上で全域」という宣言そのものを反証する** — 宣言は正しく、コードが
    例外 1 つ足りなかった。

12. **導出テストが「広がり」を検出できなかった（DC3、`periphery`）。** 列型のテストは
    「こちらが受理する名前が join 分岐に現れるか」しか見ておらず、**シリアライザが新しい
    列型を得ても永遠に通る**。レビュアーが in-memory で広げたソースを食わせて実証した。
    join 分岐の `isinstance` タプルから**名前を抽出して集合として比較**するよう修正。
    実測: `deque` / `array` の追加で落ち、`frozenset` の追加では通る（正しい —
    `frozenset` は `REFUSED_SEQUENCE_TYPES` に記録済み）。

**この 3 件は「同じサイクルの引っ越し」ではない。** 10 は DC1 だが**受理の入口の型判定**の
欠陥であって「比較が値を理解し損ねた」欠陥ではなく、修正は個別の guard ではなく
**呼び出し元コードが走らないようにする構成上の変更**である。11 と 12 は宣言と実装の
ずれで、どちらも宣言のほうが正しかった。記録は `results/pr2_codex_round21.md`。

13. **`np.ndarray` も厳密な型一致で受理する（rounds 20-22 監視が名指しし、こちらで
    実測して見つけた）。** 監視は「`isinstance(value, np.ndarray)` が 2 か所残っており、
    安全だと検証していない」と明示した。実行したところ **1 件出た**:

    ```
    値                                          caller の wire   正規化後の wire
    1-D ndarray サブクラス（__iter__ が毎回変わる）   1.0,1.0          2.0,2.0
    ```

    **サブクラスを反復すると呼び出し元の `__iter__` が走り、呼び出し元のメソッドは
    2 度同じものを返す義務を負わない。** `np.matrix` / masked array / `ndim` が嘘を
    つくサブクラスは元から拒否されていたが、これは通っていた。修正はスカラーの門と
    同じ形（**`type(value) is np.ndarray`**）。1-D 判定も**シリアライザ自身が読む
    `len(shape)`** に合わせた（`ndim` と `shape` は numpy の配列では一致するが、
    他では一致する保証がない）。

14. **呼び出し元が numpy を自称できた（DC1、review round 22）。** 13 までの導出は
    `np.integer` などの `__subclasses__()` を **import 時に**歩き、
    `kind.__module__.split(".")[0] == "numpy"` で絞っていた。**`__module__` は
    クラス本体に書けるただの属性である。** レビュアーは
    `class Disguised(np.float64): __module__ = "numpy"` と書き、さらに
    `__format__` が `item()` の立てるフラグで答えを変えるようにして、
    **2 段の防御を同時に破った**:

    ```
                              caller が書く wire   学習に届く wire
    model.params                learning_rate=0.9   0.1
    fit(params=)                learning_rate=0.9   0.1
    tuning best_model_params    learning_rate=0.9   0.1
    calibration.params          learning_rate=0.9   0.1
    ```

    fit は完了する。加えて走査が import 時なので、**そのクラスが lizyml の import より
    前に定義されたかどうかで答えが変わっていた**。

    修正: **型集合を `vars(numpy)` から読む** — 「numpy がその名前で export している型か」
    は**同一性**の問いであり、呼び出し元が主張できず、import 順にも依存しない。
    加えて **`format(value, "")` を `.item()` の前に読む**（「もう一方の検査が効いている
    ことに正しさが依存する検査」は 2 段目ではない）。

    **RED 検証をやり直した。** 最初に書いたテストは**どちらの revert でも緑**だった —
    witness をテスト本体で定義すると import より後になるので走査実装でも集合に入らず、
    状態を持つ値は型集合を通れないので順序の検査に届かない。**導出関数を witness
    定義後に呼び直す**形と、**型集合を monkeypatch で緩めて 2 段目だけを単独で試す**形に
    書き直して両方 RED を確認した。**この run で「テストが別の理由で緑だった」のは 5 回目。**

    なお **round 22 の verdict は取得できていない** — provider 側のコンテンツフィルタで
    実行が中断された（敵対的オブジェクトを構築する手法自体が誤検知されたと見られる）。
    上記はログに残っていた再現である。記録は `results/pr2_codex_round22.md`。

15. **`in` は同一性ではなかった（DC1、round 23）。**
    `type(value) in NUMPY_SCALAR_TYPES` は `frozenset` の探索であり、判定は
    **呼び出し元の `__hash__` / `__eq__`** で行われる。クラスのそれらは**メタクラス**から
    来るので呼び出し元が書ける。実測: `hash(np.float64)` を返し `__eq__` が真になる
    メタクラスを持つクラスは、**numpy を継承せず、`__module__` も名乗らず、import 順にも
    依存せずに**通過し、自前の `__format__` と `item()` が正規化の中で走った。
    `PLAIN_SCALAR_TYPES`（tuple、`x is e or x == e`）にも同じ穴。
    **修正: 全ての門を `is` 比較にする** — `is` は Python で唯一呼び出し元が参加できない
    比較である。

16. **`vars(numpy)` は書き込み可能（DC1、round 23）。** `np.Injected = Injected` を
    import より前に 1 行書くだけで型集合に入る。checker の指摘の核心:
    **「Python のどんな名前空間の読み取りも呼び出し元から独立ではない。『呼び出し元が
    自称できない』は、どんな導出も提供できない性質である。」**

    **修正: 決め手を名前空間から numpy 自身の dtype レジストリへ移す。** 候補は
    `vars(np)` / `np.sctypeDict` から**列挙するだけ**（上位集合でよい）とし、採用は
    **`np.dtype(kind).type is kind` の往復**で決める。サブクラスは基底に解決されるので
    通らない（実測: `Injected -> float64`）。

    **そして bound を書き直した。** これはパラメーターの**値**に対して領域を閉じる。
    **プロセス内で既に numpy の一部を差し替えた呼び出し元に対する sandbox ではない** —
    `numpy.dtype` を差し替えられる者は `numpy.float64` もこのモジュールも差し替えられる。
    **この run で「宣言が達成不能だった」のは 3 度目**（round 18 の「何に対しても
    raise しない」、round 20 の NaN、そして今回）。**達成可能な宣言に書き直すのが正しい
    修復である。**

17. **要素位置の門が未検証だった（round 23）。** `_plain_element` の厳密型一致を
    `isinstance` に緩めてもファイル全体が緑のままだった（DC6 の形 — 防御は正しく、
    それを行使するテストが無い）。振る舞いは実在する。テストを追加。

    **なお round 23 は Codex ではなく `policy:fresh-checker` の read-only checker が
    実行した。Codex は 3 回連続で provider 側のコンテンツフィルタに落ちており、
    3 回目でプロンプトではなく `param_domain.py` の中身自体が反応していると判断して
    経路を変えた（同一エラー 3 連続で approach を変える運用ルール）。
    ⚠️ したがってマージゲートの「Codex APPROVE」は依然として未取得である。**
    記録は `results/pr2_codex_round23.md`。

18. **受理した型が「書ける値」とは限らない（DC1、round 24）。** 2 つの値が surface を
    通り、学習前の表明も通り、**LightGBM の内側で raise した**:

    - **`PurePosixPath` / `PureWindowsPath`** — シリアライザは
      `isinstance(val, (str, Path, ...))` で判定するが、**pure path は `Path` では
      ない**。受理集合に入れていたのがそのまま誤り。
    - **`10**5000`** — Python の `int` に幅は無いが、**十進変換の上限（既定 4300 桁）を
      超えると `str()` は桁を返さず raise する**。round 21 で `float()` の
      `OverflowError` を直したが、`str()` の `ValueError` は別物だった。

    **修正: 型からの推定をやめ、文字列を実際に要求する。** `_written_or_refused` を
    scalar 位置（`format`）と element 位置（`str`）の両方に置き、書けない値は入口で
    `CONFIG_INVALID`。path は `Path` のフレーバーだけに絞り、
    **「受理する path 型はすべて `issubclass(kind, pathlib.Path)`」をテストで固定**する
    （シリアライザ自身の判定から導出）。`values_differ` の `str(element)` も
    例外ハンドラの中へ入れた。

    **round 24 は Codex が完走した。** rounds 22-23 の中断はコードではなく
    **レビュー依頼の書き方**が原因で、過去のすり抜けを並べた表・「呼び出し元のコードが
    走る経路」という問い・煽りを外し、**契約の検証**として書き直したところ通った。
    受け入れ基準 2/3/4/5/7/8/9 は合格しており、特に 8 は**レビュアーが AST で学習
    サイトを列挙**して確認している。記録は `results/pr2_codex_round24.md`。

19. **`export_code` は受理集合の 3 番目の消費者である（提案が名指していなかった）。**
    実測: `fit(params={"forcedsplits_filename": Path("f.json")})` は**学習が通り**、
    その後 `export_code` が `TypeError: Object of type PosixPath is not JSON
    serializable` で落ちる。`json.dump` に path のエンコーダは無い。

    **修正: path は入口でテキストにする。** シリアライザは path をスカラー
    フォーマッタで書き、path のそれは自身のテキストなので **bytes は同じ**であり、
    テキストは下流の全員が運べる唯一の形である。受理集合からは型が 1 つ減る。

    **これは提案の補正であって、実装の詳細ではない。** H-0095 が宣言した消費者は
    **4 surface と 2 つの `lgb.train` 表明サイト**だけで、`export_code` は
    1 度も出てこない。rounds 23-25 の監視がこれを `DRIFTING` の根拠に挙げた ——
    「受理集合の定義が、提案が名指していない第 3 の消費者によって決められている」。
    **指摘は正しい。よってここに消費者として明記する**: 正規化後の値は
    **`json.dump` できること**も要件である。

    `_path_text` に置いた変換の検査は**到達不能だったので外した**（DC6）。
    厳密型一致がサブクラスを弾くので、`__format__` を持つ path は入口に届かない。
    `pathlib` は `__format__` を定義しないので、この 2 型については
    `format(p, "") == str(p)` が構成上成り立つ —— **それをテストで固定した**。
    最初に書いたテストは検査を外しても緑のままで、**この PR で「テストが別の理由で
    緑」は 6 回目**である。

#### 受け入れ基準 7 の決定: **#283 は H-0095 では解決しない**

スカラー `0.5` と単一要素の列 `[0.5]` は同じ bytes を書くが、正規化後も `float` と `list`
であり、依然として拒否される。**admit するには「学習器にどちらを渡すか」を決める必要が
あり、それは別の決定である**（`allow` ＝振る舞いの拡大なので firing rate 付きの Proposal が
要る）。#283 は open のままとし、`KNOWN_BOUNDS` の免除もそのまま残す。

### 契約の確定（2026-09-08、D10 の帰結）

**上の「提案」節の受理表と、それに続く補正 1-19 は、この節に置き換わる。**
削除はしない（提案がどう動いたかの記録である）が、**受理集合と消費者の正は以下**であり、
補正の列を読んで再構成する必要はない。

書き直した理由は D10 に記録した。24 ラウンドで `APPROVE` が出なかったのは、
受入基準が実質「**この検証を破る値は存在するか**」を問うていたからで、これは Python の
あらゆるオブジェクトを渡る全称命題であり、**開いた領域に対する「反例なし」は有限の
レビューでは示せない**。以下は「破れるか」ではなく「**契約通りか**」を判定できる形に
書き直したものである。

#### 1. 受理集合（位置 × 厳密型）

**シリアライザは位置ごとに別のフォーマッタを使う**ので、集合も位置ごとに定義する
（スカラー位置は `__format__`、列の要素位置は `str`）。以下のブロックは
`lizyml/core/param_domain.py` から**生成**したものであり、散文に写した表ではない。

再生成 / 差分検出:

```
uv run python docs/audits/2026-09-defect-discovery/instruments/param_domain_contract.py
uv run python docs/audits/2026-09-defect-discovery/instruments/param_domain_contract.py --check
```

`--check` はこのブロックとモジュールを突き合わせ、乖離したら非零で終了する（DC3）。
**乖離検出が実際に落ちることは確認済み**（numpy の版を 1 文字変えて exit 1）。

ブロックが **numpy の版と platform を書いている**のは飾りではない。
`longdouble` / `longlong` / `ulonglong` は C の型に対する別名であり、**platform に
よっては別の型に解決されて集合から消える**。このブロックは「受理集合」ではなく
「**この numpy・この platform での受理集合**」であり、別環境で `--check` が落ちるのは
ドリフトではなく環境差である。

**型集合は値集合より広い**、というのもブロックが**測って書いている**（散文の注ではない）。
`timedelta64` は `numpy.integer` なので導出が型を通し、値は `format` 検査が拒否する。
`longdouble` は 2 例目で、しかも**位置で非対称**である（`.item()` が Python の float を
返さないのでスカラー位置は拒否、要素位置はテキストを parse し直して受理）。
手で書いた注は 1 例目しか挙げておらず、2 例目は計測して初めて出た。

なお `param_domain.ACCEPTED_DESCRIPTION` は**拒否メッセージに載せる利用者向けの要約**で
あって契約ではない。契約はこのブロックと下の要件表である。

<!-- param-domain-contract:begin -->
```text
numpy               2.4.2
platform            linux x86_64

scalar position     NoneType, bool, float, int, str
  converted         PosixPath, WindowsPath -> str
  converted         numpy scalar -> .item(), checked by format()
element position    NoneType, bool, float, int, str
  converted         PosixPath, WindowsPath -> str
  converted         numpy scalar -> the plain value printing as str(x)
sequence            list, tuple, 1-D ndarray
  member            scalar, list (depth 2 only), dict
mapping             dict with exact-str keys, values normalised
refused sequence    frozenset, set

numpy scalar types (exact type; derived by np.dtype(k).type is k)
  numpy.bool, numpy.float16, numpy.float32, numpy.float64, numpy.int16, numpy.int32, numpy.int64, numpy.int8, numpy.longdouble, numpy.longlong, numpy.str_, numpy.timedelta64, numpy.uint16, numpy.uint32, numpy.uint64, numpy.uint8, numpy.ulonglong

  the type set is wider than the value set, and by position:
  refused in scalar position   numpy.longdouble, numpy.timedelta64
  refused in element position  numpy.timedelta64
  (one constructed value per type -- a measurement of these values,
  not a proof about every value of the type)
```
<!-- param-domain-contract:end -->

**各行を固定しているテスト**（テストの無い行は、次のラウンドが見つける行である）:

| 受理集合の行 | 固定しているテスト（`tests/test_core/test_param_domain.py`） |
|---|---|
| スカラー位置の素の型 | `test_the_scalar_types_are_the_ones_the_serialiser_names` |
| path を受理してテキストにする | `test_every_path_type_accepted_is_one_the_serialiser_accepts` / `test_a_path_is_carried_on_as_its_text` / `test_the_path_conversion_is_safe_because_of_the_types_admitted` |
| numpy 型集合の導出 | `test_the_numpy_scalar_types_are_derived_from_numpy` / `test_the_numpy_type_set_is_what_numpy_resolves_to_itself` |
| numpy 型集合が呼び出し元から独立であること | `test_a_caller_class_cannot_claim_to_be_a_numpy_type` / `test_a_class_put_into_the_numpy_namespace_is_still_refused` |
| numpy スカラーの変換を `format` で検査すること | `test_a_numpy_value_whose_conversion_would_lose_bytes_is_refused` / `test_the_written_form_is_read_before_the_conversion` / `test_each_defence_is_load_bearing_for_something_different` |
| 要素位置の門と、素の代替の探索 | `test_the_element_position_admits_by_identity_too` / `test_a_reduced_precision_element_that_does_have_a_stand_in_is_accepted` / `test_a_value_with_no_plain_stand_in_is_refused_rather_than_rounded` |
| 列の型 | `test_the_sequence_types_are_the_ones_the_serialiser_joins` / `test_a_numpy_array_subclass_is_refused_for_the_same_reason_a_scalar_is` |
| `set` の拒否 | `test_a_set_is_refused_rather_than_ordered_by_hash` |
| リストの入れ子は深さ 2 まで | `test_the_one_nested_shape_lightgbm_reads_is_accepted` / `test_a_list_nested_deeper_than_the_serialiser_reads_is_refused` |
| mapping は入口で受理し出口で拒否 | `test_a_metric_entry_written_as_a_mapping_is_accepted_at_the_surface` / `test_a_mapping_that_survived_to_the_trainer_is_a_defect` |
| 門は `is` であること | `test_membership_is_identity_and_not_the_callers_own_equality` |
| 受理した型が書ける値とは限らないこと | `test_an_accepted_type_is_not_by_itself_a_writable_value` / `test_an_integer_too_large_for_a_float_is_accepted_and_compared` |
| 境界そのもの（受理と拒否の分割） | `test_the_refused_subset_is_exactly_the_declared_boundary` / `test_a_refusal_inside_the_candidate_set_is_forced_not_chosen` |
| 書く文字が UTF-8 に encode できること（round 25 追加） | `test_a_string_neither_consumer_can_encode_is_refused` |
| 「正規化が値を変えなかった」を**型と同一性**で判定すること（round 26 追加） | `test_unchanged_is_decided_by_type_and_not_by_printed_text` |
| 母集団の型軸が受理型集合と一致すること（round 25 追加） | `test_the_population_covers_every_admitted_numpy_type` |

#### 2. 消費者と、それぞれが課す要件

**正規化した値の消費者は、学習器だけではない。** 提案は 4 surface と `lgb.train` 2 サイト
しか名指しておらず、`export_code` は 19 番目の補正で**実装のほうから**現れた
（rounds 23-25 監視が `DRIFTING` の根拠に挙げた）。ここで全部名指す。

| 消費者 | sink | 課す要件 | 受理母集団の上で実行しているオラクル |
|---|---|---|---|
| 学習（adapter） | `estimators/lgbm/adapter.py` の `lgb.train` | 値が `is_plain`（mapping は adapter が消費済み） | `test_the_exit_assertion_passes_everything_the_normaliser_produces` / `test_the_exit_assertion_is_called_at_every_place_that_trains` |
| 学習（calibrator） | `calibration/isotonic.py` の `lgbm.train` | 同上 | 同上（学習サイトの母集団はソースから導出） |
| 学習に届く bytes | 上記 2 サイトのシリアライザ | **正規化の前後で wire が同じ** | `test_normalising_does_not_change_the_bytes_the_estimator_is_sent` |
| `export_code` | `codegen/artifact_writer.py` の `json.dump`（`config.json`） | **`json.dump` で書けること** | `test_every_accepted_value_can_be_written_as_json` |
| **学習器と `export_code` の両方**（round 25 追加） | LightGBM の `_c_str` / `artifact_writer` の `encoding="utf-8"` | **書く文字が UTF-8 に encode できること** | `test_a_string_neither_consumer_can_encode_is_refused` |
| 2 度正規化する経路 | `calibration.params`（検査側と calibrator dict 生成側） | **冪等** | `test_normalising_twice_is_normalising_once` |
| ~~同一性比較~~ **（H-0096 で消滅）** | ~~`core/value_equality.py`~~ | ~~受理集合の上で全域、どれでも raise しない~~ | **この行は superseded。** H-0096 が同一層の重複綴りを値によらず拒否するようにしたので、比較そのものが消え、モジュールごと削除した。要件が緩んだのではなく**消費者が居なくなった** |
| 述語 | `is_accepted` / `is_plain` | 正規化関数と一致すること（**両向き**） | `test_the_predicate_and_the_normaliser_agree` / `test_every_accepted_value_normalises_into_the_closed_set` / `test_the_predicates_refuse_everything_the_normaliser_refuses` |

**UTF-8 の行は round 25 が見つけた**（記録: `results/pr2_codex_round25.md` 指摘 1）。
「学習器は `_param_dict_to_str` が書くものを読む」「`export_code` は json 化できれば
よい」はどちらも真で、**どちらも足りていなかった** —— 両消費者ともそのあと UTF-8 に
encode する。孤立サロゲート `"\ud800"` は受理型の `str` であり、正規化・出口の表明・
`json.dumps` オラクルをすべて通ってから両消費者で `UnicodeEncodeError` になっていた。
**要件を 1 つ持つ消費者を「1 つの要件で足りる」と読んだのが誤りである。**

**round 26 が、この 2 つの修正の続きを 2 件出した**（記録: `results/pr2_codex_round26.md`）。

- **UTF-8 検査は 1 か所を漏らしていた**: `_plain_element` の numpy 分岐が
  テキストを直接読んでいたため、`np.str_` の孤立サロゲートが**素の文字列に正規化され**、
  その結果を再び正規化すると拒否される（**冪等性違反**）。修正は他の位置と同じ門
  （`_written_or_refused`）を通すこと。レビュアーは**サロゲート 2048 個 × 12 構成**を
  列挙して、この 1 形だけが通っていたことを示した。
- **`repr` で「変わっていない」を判定したのが誤りだった**（**こちらの修正が書いた欠陥**）。
  `repr` は表示テキストであり、numpy が公式に持つ `printoptions(legacy="1.25")` の下で
  `np.int64(1)` と `1` は同じに印字される。**この PR が 24 ラウンドかけて排除してきた
  「呼び出し元が参加できる比較」を、値の比較側で開けた**ことになる。
  判定は**型の再帰的な一致と、スカラーの同一性（`is`）**で行う形に直した。

**述語の行も round 25 が見つけた**（指摘 2）。述語は受理集合を**自分の言葉で言い直して
いた**ため、round 24 が正規化だけを「実際に文字を書ける値」へ狭めたときに置き去りに
なり、`10**5000` が両述語と出口の表明を通って正規化にだけ拒否された。
**1 つの境界に対する宣言が 2 つあったのが原因**なので、述語は**正規化関数を呼ぶ**形に
した（`is_accepted` = 正規化が通り、かつ結果が入力と同じ）。
一致テストも**両向きにした** —— 従来は受理母集団しか走査しておらず、
**緩すぎる述語を構成上見られなかった**。

**閉じられるのは「要件のリスト」であって「消費者のリスト」ではない。** 各要件は
受理母集団**全体**の上で実行するオラクルを持ち、そこは閉じている。消費者のリストは
各要件の**根拠**であって、走査で閉じたものではない。

sink の走査は `instruments/param_domain_contract.py` として出荷するが、
**それを閉包と呼ばない**。`parameter_merge_seams.py` の docstring が同じ主張を
2 回して 2 回反証されたのと同じ理由であり、実際に**この走査も初版で 1 件落とした** ——
calibrator は同じライブラリを `lgbm` という別名で import しており、
`lgb` しか知らない版はその `train` を見なかった。既知のサイトと突き合わせて見つけた。

**要件を課さない sink**（走査に出るが、パラメーター値を運ばない、あるいは運んでも
raise しない）:

- `persistence/exporter.py` の `joblib.dump` — pickle であり、素の値はすべて運べる
- `persistence/exporter.py` の `json.dumps(metadata, default=str)` — 受け取るのは
  config であって正規化後の dict ではなく、`default=str` があるので raise しない
- `features/pipelines_native.py` の `json.dump` — pipeline state であってパラメーターではない

#### 3. 宣言する bound（スコープ外を明示する）

**達成不能な宣言を書くのは DC7 であり、この run で 3 度書き直している。** よって
スコープ外を事実として書く。

1. **プロセス内で numpy の一部を差し替え済みの呼び出し元に対する sandbox ではない。**
   `numpy.dtype` を差し替えられる者は `numpy.float64` もこのモジュールも差し替えられる。
   閉じるのはパラメーターの**値**に対してである。
2. **`Model.load()` は検査しない。** artifact は起きた fit の記録であり、読めなく
   する理由がない（§14.4 既定）。したがって**このバージョンより前に書かれた artifact**
   は、受理集合の外の値を持つ adapter を復元しうる。**実測**: 復元した adapter の
   params に path を入れて `export_code` を呼ぶと
   `TypeError: Object of type PosixPath is not JSON serializable` になる
   （H-0095 の前と同じ振る舞いであり、この提案が悪化させたものではない）。
   閉じているのは「**このプロセスで 4 surface を通って入った値**」である。
3. **入口と出口で受理集合が違う**のは意図である（mapping）。1 つの述語で両端を
   賄うと、緩いほうに合わせることになり、学習器に必要な bound が言えなくなる。

#### 4. この契約に対してレビューが答える問い

「この検証を破る値はあるか」ではなく、**「1 の集合と 3 の bound の下で、2 の各要件が
受理母集団の上で成り立っているか」**である。前者は Python のあらゆるオブジェクトを
渡る全称命題で有限のレビューでは閉じない。後者は**実行可能で、実際に実行している**。

**ただし「受理母集団」は領域そのものではない**（round 25 指摘 3 の後半による訂正）。
§1 の値領域は**無限**である —— 文字列・整数・コンテナの中身に上限が無い。
`ACCEPTED_POPULATION` は**その有限標本**であり、テストが主張しているのは
**「この標本の上での網羅」**であって「すべての受理値について成り立つ」ではない。
両者を書き分けること。**標本の閉じ方**は次の 2 つで担保する:

- **型軸は受理型集合から導出する**（`_NUMPY_SCALAR_TYPES` はモジュールの
  `NUMPY_SCALAR_TYPES` を読む。ベタ書きだった版は 5 型を落としていた ——
  `test_the_population_covers_every_admitted_numpy_type` が固定する）。
- **拒否の理由は閉じた列挙**であり、標本の中のどの拒否も 6 つの宣言理由のいずれかに
  当たること、かつ理由が予測する値はすべて拒否されることを両向きで確かめる
  （`test_the_refused_subset_is_exactly_the_declared_boundary`）。
  理由は位置ごとに評価する —— **同じ型が scalar 位置で拒否され element 位置で
  受理される**ことがあるためで、`longdouble` が実例である。

### 決定: 探索空間の `choices` は型の同一性で判定する（受け入れレビュー round 29 / [#287](https://github.com/nbx-liz/LizyML/issues/287)）

**この決定は探索空間を 5 つ目の正規化 surface にするものではない。** 探索空間は
これまでどおり numpy スカラーを**すべて拒否**する。変わるのは、**Python のスカラーを
継承している 2 型（`np.float64` / `np.str_`）が今まですり抜けていた**のを止める点だけである。

#### 何が起きていたか

`lizyml/tuning/search_space.py` の `_validate_categorical_choices` は
`(NoneType, bool, int, float, str)` を **`isinstance`** で判定していた。
`np.float64` は `float` のサブクラス、`np.str_` は `str` のサブクラスなので**通る**。
サンプルされた値を正規化する場所は無いので、値は numpy のまま
`_model_tuning.py` の trial overlay に載り、adapter の出口表明
（`assert_plain_params`）が全トライアルで拒否する。Optuna は各トライアルを FAIL と
記録し、全滅した結果**利用者が見るのは `TUNING_FAILED: All tuning trials failed.
Check parameter ranges.`** —— range は正しいのに range を疑えと言われる。

実測（numpy 2.4.2、`instruments/space_choice_normalisation.py`）:

```
  np.float64 (eta       ): TUNING_FAILED  All tuning trials failed. Check parameter ranges.
  np.float32 (eta       ): CONFIG_INVALID Categorical dim 'eta' has invalid choice at index 0 ...
    np.int64 (num_leaves): CONFIG_INVALID Categorical dim 'num_leaves' has invalid choice at index 0 ...
     np.str_ (boosting  ): TUNING_FAILED  All tuning trials failed. Check parameter ranges.
 plain float (eta       ): tuned
```

**範囲は `categorical` だけ。** `FloatDim` は `float(spec["low"])`、`IntDim` は
`int(spec["low"])` と parse 時に強制変換するので、numpy の境界はサンプル前に素の値になる。
リテラルをそのまま運ぶのは `choices` だけである。

#### 対応方針（決定）

**`isinstance` を型の同一性へ変える** ——
`any(type(value) is plain for plain in _ALLOWED_CHOICE_TYPES)`。

`type(value) in _ALLOWED_CHOICE_TYPES` **とは書かない**。`in` はタプルの各要素に
「その型と**等しいか**」を尋ねるので、クラスの `__eq__` ——**メタクラス**から来る＝
呼び出し元が書ける —— に依存する。テストで固定する。

**訂正（round 30 の指摘、実測で確認）**: 初版は「ハッシュと等価による探索」と書いたが、
**タプルの `in` はハッシュを引かない**。`__hash__` が呼ばれるのは `set` / `frozenset`
の場合で、`BLUEPRINT.md` §14.4 の記述（round 23）はそちらについてのものである。
実測: `float` と等価を返すメタクラスを持つクラスはタプルの `in` を通り、
そのとき `__hash__` は一度も呼ばれない。**結論（`in` を避けて同一性で見る）は変わらないが、
理由は「呼び出し元が書ける等価」だけである。**

**この門は受理集合の言い直しではない。** 守っている境界が違う ——
`param_domain` の受理集合は「LightGBM に書ける値」であり、**numpy スカラーを受理して
変換する**。こちらは「Optuna の `CategoricalDistribution` が受け取れる値」かつ
「正規化を経ずに出口表明へ到達しても通る値」であり、**素の Python スカラーだけ**である。
1 つの境界に宣言を 2 つ持たないという規則（§14.4、round 25）に反しない。

メッセージも直す。現行の「Each choice must be a scalar (str, int, float, bool, or
None)」は、**まさに `float` である値を拒否したときに自己矛盾する**。
「a plain Python scalar (not a numpy scalar)」と書く。

#### Firing rate

```
Firing rate: 0/54 of the choices in the shipped suite would be newly refused
             (42 categorical dimensions, 54 choices: 51 plain Python scalars,
             3 already refused today and all three from tests asserting that
             refusal, 0 subclassing a plain scalar; measured by
             instruments/space_choice_type_firing_rate.py over the full suite at
             22b11b3, report in results/pr2_space_choice_measurement.txt)
```

`allow` 条件を**狭める**変更なので、上の実測は「狭めて壊れる母集団が空である」ことの
証拠として記録する。

**この数値の bound（round 30 と監視が独立に指摘、そのまま記録する）**:
`0/54` は**計測器がスイート上で観測した choice の出現回数**についての記述的な比率であり、
そのうち 3 件は今日すでに無効な出現である。**一意な config についての比率でもなければ、
ありうる全呼び出し元についての証拠でもない。** 狭める対象の 2 型が今日すでに
`TUNING_FAILED` になることは実測しているが、それは**この 2 型を使う config が
「動いている」とは言えない**という意味であって、**「そうした config が存在しない」ことを
立証するものではない**（初版はそう書いていた。訂正する）。

#### 残る不整合（この決定では閉じない）

4 つの正規化 surface は numpy スカラーを**受理して変換する**のに、探索空間は**拒否する**。
この決定はその差を**縮めない** —— すり抜けを止めるだけである。
「`choices` も他の 4 surface と同じように numpy を受理すべきか」は別の判断で、
そちらを採るなら trial overlay か parse 時のどちらかで正規化を走らせる提案が必要になる。
**[#287](https://github.com/nbx-liz/LizyML/issues/287) をその問いのために開いたままにする。**

#### 受け入れ基準（テスト観点）

1. `np.float64` / `np.str_` を `choices` に書くと **`CONFIG_INVALID`** で、
   **`TUNING_FAILED` ではなく**、次元名を名指し、**Booster が 1 本も学習されない**。
2. `np.float32` / `np.int64` / `np.bool_` は**従来どおり**同じ形で拒否される（回帰させない）。
3. 素の値は従来どおり tuning が通る（対照 —— 無いと「全部拒否」でも 1 と 2 が通る）。
4. 門が**型の同一性**を見ていること —— `float` と等価を装うメタクラスを持つクラスが
   通らない。

---

## H-0098: Derive parameter-domain predicates from one structural walk

- Status: accepted -- implemented in 5fb8a80 (#291); corrected in H-0111 (#319)
- Date: 2026-09-10
- Related: BLUEPRINT §14.4; H-0095; issues #284 and #287.

### Purpose and scope

Replace the three structural walks in `core/param_domain.py` with one walk
returning a normalized value, whether it was unchanged, and whether it contains
a mapping. The surface and training predicates consume these results directly.
Scalar and element formatting remain distinct, following the estimator wire.

Review repair: when the training boundary disallows mappings, the same dictionary
dispatch refuses before traversing members. This preserves controlled rejection
of cyclic dictionaries; there is no separate mapping-search walk. Surface
normalization behavior is unchanged.

### Compatibility and migration

Preserve the current accepted set and scalar/sequence-position asymmetry.
Identity determines unchanged scalar values; container branches aggregate child
results during normalization. No caller-controlled equality or display is used.
Public signatures and persisted formats are unchanged; no migration is needed.

For #287 retain the H-0095 categorical choice decision: exact plain Python
scalars only. Numeric range bounds are converted at parsing; choices are carried
literally and reject numpy scalars before a study. This deliberate difference
from the four normalization surfaces is retained, not silently widened.
The former numpy-choice TUNING_FAILED symptom is already fixed at base 2436a66:
the shipped instrument returns CONFIG_INVALID with a successful plain control.

### Alternatives

Keeping separate recursive predicates retains the drift seam of #284.
Normalizing categorical choices would widen the current contract and needs a
separate compatibility decision; per-trial normalization repeats avoidable work.

### Acceptance criteria

- Existing parameter population preserves wire bytes, JSON/UTF-8 behavior,
  idempotence, mapping exclusion and caller-independent unchanged decisions.
- One structural dispatch derives normalized value and both predicate facts.
- `param_domain_contract.py --check` still verifies the generated contract.
- Every literal-bearing search dimension is covered: numpy categorical values
  fail CONFIG_INVALID naming the dimension; float/int bounds become plain;
  plain categorical controls remain accepted.
- Lint, format, type checks and non-slow suite pass before external review.

## H-0096: 同一層の重複綴りを値によらず拒否する（H-0094 決定の改訂 / D13 の帰結）

- **ステータス**: Accepted
- **起票日**: 2026-09-09
- **決定日**: 2026-09-09（決定の記録は `DECISIONS-PENDING.md` の **D13** = 経路 1）
- **スコープ**: `lizyml/core/_model_factories.py`（`check_duplicate_identities`）, `lizyml/estimators/lgbm/adapter.py`（`_pop_by_identity`）, **`lizyml/core/value_equality.py`（削除）**, `tests/test_core/test_value_equality.py`（削除）, `tests/test_core/test_fit_params_override.py`（許容ケース → 拒否ケース）, **`BLUEPRINT.md` §14.4**, `CHANGELOG.md`。
- **関連**: H-0094（決定 6 / round 5 で入れた同値許容）, H-0095（比較の領域を閉じる提案）, [#264](https://github.com/nbx-liz/LizyML/issues/264), `docs/audits/2026-09-defect-discovery/DECISIONS-PENDING.md` の **D13**, `results/pr2_why_no_approve.md`, `results/pr2_prior_art.md`。

### 目的（課題）

PR 2（#278）は **26 ラウンド回って `APPROVE` に到達しなかった**。原因解析
（`results/pr2_why_no_approve.md`、実測）が特定した原因は 2 つで、独立している。

**原因 A（設計）。** H-0094 round 5 が入れた「**同一層の重複綴りは、値が等しければ許し、
違えば拒否する**」という 1 行が、**任意の Python 値についての全域な等価判定**を要求する。
`values_differ` の呼び出し元は今も 2 か所しかなく、どちらもまさにこの問いである。
その 1 つの述語が `value_equality.py` + `param_domain.py` = **728 行**、
production commit **51 件中 30 件**、**rounds 18-26 の 9 連続**を生んでいる。
そして「誤拒否も欠陥」（round 12/13 が 2 度押し返した DC7）なので、
**正しくあるには領域を広げねばならず、証明可能であるには狭めねばならない** ——
2 つは逆向きに引き、広げるたびに新しい位置が開く。

**原因 B（手続き）。** round 21 以降、レビューの問いが「**どんな値でも門を破れないか**」
という全称命題になった。反例でしか答えられないので `APPROVE` の出口が無い。
**本提案は原因 A を除去する。原因 B はレビュー依頼の書き方で別に扱う。**

### 実測（この提案が根拠にしているもの）

**1. 許容分岐は出荷済み母集団で 0 回発火する。** 4 surface すべての呼び出し点を包んで
スイート全体（`9731 passed`）を計測した
（`docs/audits/2026-09-defect-discovery/instruments/duplicate_tolerance_firing_rate.py`）:

```
one parameter under two spellings: 51
  REFUSED  (different values): 14
  TOLERATED (equal values):    37
帰属: 37/37 が tests/test_core/test_fit_params_override.py（本 PR が追加した file）
      pre-existing のヒット: 0
```

round 11 の `0/811`、round 12 の `0/1009` / `0/22`、round 13 の `0/916` / `0/928`、
round 14 の `0/70`、H-0093 の `0/736` / `0/52` / `0/3` と整合する。

**2. LightGBM 自身は値を比較しない。** verbosity を既定に戻して fd レベルで捕捉すると、
**等しくても違っても重複そのものを警告する**
（`instruments/lgbm_duplicate_alias_behaviour.py`）:

```
[Warning] learning_rate is set=0.5, eta=0.5 will be ignored. Current value: learning_rate=0.5
```

優先順位は**決定的で dict の順序に依存しない**（`eta` はどちらの順でも `shrinkage_rate`
に勝つ）。したがって round 4 の拒否理由「どの値が効くかは*書いたもの*ではなく*ライブラリ*
で決まる」は**半分しか正しくない** —— 決定的であり、LightGBM 自身がそう言う。
**拒否の根拠は「不可視だから」ではなく「利用者が 1 つの設定を 2 度書いており、
どちらを意図したか LizyML には決められないから」に置き換わる。**

**3. 先行事例に `values_differ` の形は無い**（`instruments/duplicate_key_prior_art.py`、
`results/pr2_prior_art.md`）。調べた 9 処理系のうち意味的な値の等価で分岐するのは
C プリプロセッサだけで、その C ですら**トークン列の同一性**（構文的）で判定する。
Python の呼び出しは**等しくても `TypeError`**、pydantic は alias が無言で勝ち、
Go yaml.v3 / Ruby Psych はエラー、PyYAML / PostgreSQL / dict / json は後勝ち。

### 提案

**同一層で 1 つのパラメーターが 2 つ以上の綴りで書かれていたら、値によらず
`CONFIG_INVALID` で拒否する。**

- `check_duplicate_identities`（4 surface）: canonical 名でグループ化し、**要素が 2 つ
  以上のグループがあれば拒否**。値は読まない。
- `_pop_by_identity`（adapter）: 受理綴りが 2 つ以上供給されていれば拒否。値は読まない。
- **`lizyml/core/value_equality.py` を削除する。** 呼び出し元が消えるため。
  カンマ形式の同一視（`feature_contri: [1,2]` と `feature_penalty: "1,2"` を 1 つの値と
  みなす、round 13）も同時に消える —— **これは「同じ値か」を問うことをやめた帰結であり、
  「2 つの綴り」であることに変わりはないので拒否側に入る。**
- 拒否メッセージは surface と**綴り**を名指す。**値は名指さない**（提案時は「綴りと値を
  名指す」と書いていたが、**レビュー round 27 がそれを欠陥として差し戻した**ので訂正する）。
  規則は綴りの数だけで判定するのに、**報告**が値を読んでいた ——
  `sys.get_int_max_str_digits()` 桁を超える `int` は十進テキストを持たないので `str()` が
  raise し、約束した `CONFIG_INVALID` が素の `ValueError` になる。**値は context からも
  外す**: `LizyMLError.__repr__` は context を `!r` で描画するので、そこに残せば
  印字できない値を下流へ渡すことになる。綴りは `str` のキーであり必ず印字できる。

**`lizyml/core/param_domain.py` の振る舞いはこの提案では変更しない**（`:11` の docstring が
`values_differ` を存在理由として名指しているので、そこだけ再定義する）。理由を実測で述べる。

**縮小の規模を正確に書いておく。** D13 の選択肢提示では「`param_domain` は消費者を失い、
残るのは export だけで出口側で解ける」と書いたが、**本提案が実際に削るのは 157 行
（`value_equality.py`）と「同じ値か」という問いであって、728 行ではない。**
`param_domain` の 571 行は残る。

比較（消費者行 7）は消えるが、**他の消費者は残り、そのうち `export_code` は本 PR とは
独立の既存欠陥を直している**。`origin/develop`（`ccae32b`、本 PR 以前）で実行した:

```
model.params = {"feature_contri": np.array([1.0, 1.0])}
  -> fit ok, export_code -> TypeError: Object of type ndarray is not JSON serializable
```

**この欠陥は config 面に元からあり、#264 とも重複検出とも関係が無い。**
`param_domain` を削ると再発する。また `is_accepted` / `_is_unchanged` は
`assert_plain_params`（学習サイトの出口表明）にのみ仕えており、比較の消費者ではない。
**縮小の範囲を「比較のために存在したもの」に限る**のが本提案の立場であり、
`param_domain` の再設計は別提案とし、**[#284](https://github.com/nbx-liz/LizyML/issues/284)
に起票済み**（構造走査が 3 つあり一致を保つ機構が無い＝DC3。round 24/25/26 は、
この境界を言い直すたびに指摘が出たことを示している）。**繰り延べではなく追跡対象である。**

### 影響範囲

- **公開 API のシグネチャは変わらない。** 変わるのは**受理される入力**である。
- これまで通っていた「1 パラメーター × 2 綴り × 等しい値」が `CONFIG_INVALID` になる。
  4 surface（`model.params` / `fit(params=)` / `calibration.params` /
  `tuning.optuna.space`）と adapter の全部。
- `FitResult` / `PredictionResult` / `Artifacts` の形と意味は変わらない。
  `format_version` の変更は不要。
- **`BLUEPRINT.md` §14.4（1291 行目）の改訂が必要。** 現行は
  「同じ層で 1 パラメーターが複数綴り・異なる値で指定されたら `CONFIG_INVALID` とすること
  （**同値は通す**）」と書いている。**この括弧を削る。**
  （なお §5.3 が固定しているのはスマートパラメーターと `params` の競合であって
  別名重複ではない。改訂対象は §14.4 だけである。）

### 互換性

**破壊的変更である。** ただし影響は以下に限られ、実測で裏づけがある。

- **出荷済みの config / テストで壊れるものは 0 件**（上の実測 1、pre-existing 0/37）。
- **保存済み Artifacts の読み込みには影響しない**: 検査は入口（fit 前）でのみ走る。
  ただし `best_model_params` を含む復元経路は `tuning best_model_params` surface を
  通るため、**過去のバージョンが書いた重複綴りの `best_model_params` は
  読み込み時に拒否される** —— これは H-0094 決定 15 が既に「異なる値なら拒否」として
  導入した経路であり、本提案はその条件を広げる。
  **この母集団は測れない**（round 15 が既にそう記録している: 母集団は過去バージョンが
  書いた artifact であり、本リポジトリはそれを保持していない）。**測る代わりに bound を
  述べる**: `tune()` は round 11 以降、重複次元を study 開始前に拒否するので、
  **新しい artifact はこの形を作れない**。影響を受けうるのは round 11 より前に
  書かれた artifact に限られる。
- 利用者にとっての回避は自明である（**綴りを 1 つに減らす**）。拒否メッセージが
  両方の綴りを名指す。

### 代替案（すべて検討し、実測で棄却した）

| 案 | 棄却理由 |
|---|---|
| **現状維持 + 位置を計測**（D13 選択肢 1） | 原因 A が残る。領域の拡大↔閉包の綱引きが続く |
| **wire 比較**（C のトークン同一性に相当） | numpy / ndarray / tuple / カンマ文字列は無料で解けるが、**`1` vs `1.0` は拒否に戻る**（実測）。round 5 の緊張は解消せず境界が構文的に移るだけ。さらに `_param_dict_to_str` の `_is_numeric` は `try: float(obj)` なので `__float__` を持つ任意クラスが通り、型の門は別途必要 |
| **sink に型判定を委譲** | 同上の `_is_numeric` の穴に加え、実測で `set`(hash 順) / 入れ子の深さ / サロゲート str / `None` 無言脱落 の 4 つの誤受理。**「構成上閉じる」は成り立たない** |
| **警告して決定的に選ぶ**（LightGBM / pydantic の答え） | **#264 はまさに「黙って上書きされた」ことの報告**であり、警告は利用者が読む前提の仕組みである。本リポジトリの最優先は再現性であり、`CONFIG_INVALID` の既存方針（BLUEPRINT §5.3）と整合しない |
| **PR を分割**（D13 選択肢 2） | 併用可能だが、分割しても A の設計判断は残る。D13 は経路 1 を選んだ |

### 受け入れ基準（テスト観点）

1. **等しい値の重複綴りが、5 か所すべてで `CONFIG_INVALID` になる** ——
   `model.params` / `fit(params=)` / `calibration.params` / `tuning.optuna.space` /
   adapter の `_pop_by_identity`。**これが振る舞いの変更点であり、
   既存の 37 件の許容ケースを拒否ケースへ書き換える。**
2. **異なる値の重複綴りは、これまで通り拒否される**（回帰させない）。
3. **1 つの綴りしか書かれていない場合は、これまで通り学習に届く**（対照）。
   `feature_contri: [1,2]` 単独 / `feature_penalty: "1,2"` 単独のどちらも通ること ——
   カンマ形式の同一視を消したことで**単独の値**が壊れていないことを固定する。
4. **`lizyml/core/value_equality.py` が存在せず、production に import が 1 つも残らない**
   （走査テストで固定する。DC4 の裏返し）。
5. 拒否メッセージが **surface と、書かれた全綴り**を名指す。**値は名指さない** ——
   message からも context からも外す。提案時は「その値」も名指すと書いていたが、
   **レビュー round 27 がそれを欠陥として差し戻した**（値の印字が例外を出せば、
   約束した `CONFIG_INVALID` の代わりに別の例外が飛ぶ）ので訂正する。
   **adapter 側（`_pop_by_identity`）は surface を名指さない** —— 引数に取らないため。
   これは既知の不一致で、処分は D14 の受け入れ基準文書に記録し、[#286](https://github.com/nbx-liz/LizyML/issues/286) として起票した。
6. **`param_domain.py` の振る舞いが変わっていない**こと ——
   `test_param_domain.py` が無変更で通る。
7. フルスイートが緑で、**`export_code` + ndarray の既存修復が保たれている**
   （`origin/develop` で再現した `TypeError` が本ブランチでは起きないことを固定する）。

#### Firing rate

```
Firing rate: 0/37 of the tolerance branch's occurrences come from pre-existing code
             (37/37 from tests/test_core/test_fit_params_override.py, a file this PR
             adds; recorded by instruments/duplicate_tolerance_firing_rate.py over the
             shipped suite at 251353d, wrapping all four surfaces plus the adapter)
```

本提案は許容条件（`allow`）を**除去**するものであり、条件を追加しない。上の実測は
「除去して壊れる母集団が空である」ことの証拠として記録する。

---

## H-0097: adapter の拒否が出所を名指す（#286 とその同クラス 2 件 / PR 2b）

### Revision 2 — merged-input validation (2026-09-09)

- Status: accepted -- implemented by PR #290 (merge 2436a66); corrected in H-0111 (#319). This revision supersedes the implementation direction and
  acceptance criteria below; the original proposal remains as historical evidence.
- Purpose: identify the winning input when an objective or metric is rejected.
- Scope: the merged model parameters in `Model._merge_params`, shared LightGBM
  validation helpers, adapter seed/verbosity duplicate handling, and regression tests.
- Decision: validate objective and metric values after overlays, while per-key
  `origins` still exists. Resolve aliases using the existing canonical-name table.
  Attach the written parameter and its origin only to `CONFIG_INVALID` errors;
  preserve existing context, debug information, and exception chaining.
- Keep the same validators in the adapter for direct construction and trial
  overlays. Share rule definitions rather than removing these protections.
- Compatibility: no public constructor or provider Protocol change. Valid inputs
  and precedence remain unchanged. Direct adapter calls with duplicate seed or
  verbosity spellings now raise `CONFIG_INVALID`, regardless of equal values.
- Alternatives: a single adapter-wide surface misattributes mixed inputs; per-key
  adapter provenance would widen the public Protocol. Neither is required here.
- Migration: write seed and verbosity once under any accepted spelling. No
  persistence format change. #286 remains open pending disposition; six reproduced
  facade cases refuse before the adapter. #284/#287 remain PR 2c.
- Correction to the handoff: `test_seed_takes_priority_over_random_state` does
  exist and dates to commit `6619d7eb` (2026-03-07). This revision intentionally
  replaces that behavior; the original claim that no such test exists is false.
- Acceptance: invalid objective/metric aliases name the winning input before
  training; valid higher-priority replacements train; mixed origins stay distinct;
  direct adapter protection and single-spelling conversion remain; unrelated
  exceptions propagate unchanged. The updated PR 2b acceptance table is authoritative.
- Limit: this revision covers parameters entering the merged model-input boundary,
  not provenance for later sampled trial values or arbitrary future adapter errors.

### Original proposal (superseded by Revision 2 above)

Fit-only boundary clarification: `Model.fit()` requests value validation after
its final overlay. `Model.tune()` retains adapter validation after trial overlays;
rejecting the base value before a valid sampled replacement would be a regression.
`test_tuning_validates_after_sampled_overlay` pins this compatibility case, which
was reproduced as passing at the base and failing in the first local candidate.


- **ステータス**: Superseded（上の Revision 2 が実装の方向と受け入れ基準を置き換え、Revision 2 が PR #290 で実装された。H-0111 で訂正、#319）
- **起票日**: 2026-09-09
- **スコープ**: `lizyml/estimators/lgbm/adapter.py`（拒否の出所付与、`_build_params` の 6 か所目）, `tests/test_core/test_fit_params_override.py`, `tests/test_estimators/test_lgbm_defaults.py`, `CHANGELOG.md`。
- **関連**: H-0094 決定 3（出所の名指し）, H-0096, [#286](https://github.com/nbx-liz/LizyML/issues/286), [#285](https://github.com/nbx-liz/LizyML/issues/285), 計画 `phase3-plan.md` §12.4, 導出結果 `results/pr2b_rule_positions.md`。

### 目的（課題）

H-0094 決定 3 は「拒否は利用者が直すべき入力を名指す」と宣言した。同じ誤りを
`model.params` に書いた場合と `fit(params=)` に書いた場合で、**別の住所**が返ることが要件である。

**規則は宣言より多くの位置を縛っていた。** 計画 Revision 6 §12.4 の手続きに従い、
実装前にソースから位置を導出した（`instruments/refusal_surface_positions.py`）。

```
CONFIG_INVALID raising functions: 33
  EVERY RAISE NAMES A SURFACE      6
  SOME NAME AND SOME DO NOT        0
  NO RAISE NAMES A SURFACE        27
```

27 のうち大半は単一入口からのみ到達するので出所は自明である。
**パラメーター経路にあり、2 つの surface から同じメッセージを返すものが 3 つあった**
（実測。同じ入力を両 surface から入れて全文比較した）:

| 位置 | 実測 |
|---|---|
| `adapter.py:28 _pop_by_identity` | 両者バイト同一、surface 無し（**#286**） |
| `adapter.py:86 _check_objective_compatible` | 両者バイト同一、surface 無し（**未起票**） |
| `metric_bridge.py` の metric 検証 | 両者バイト同一、surface 無し（**未起票**） |

対照 —— `check_param_names` は両者で**異なる**メッセージを返す（規則が働いている例）。

**後の 2 つは起票していない。** §12.4 の手続きどおり、**本提案の中の位置として処分する**
—— 後続のレビューラウンドが 1 件ずつ発見して issue にするのを避けるために導出した。

### 対応方針（決定）

1. **adapter は自分が構築された出所を保持し、拒否に付与する。**
   `LGBMAdapter.__init__` に **`surface: str | None = None`** を追加する（追加のみ、既定値あり）。
   facade は構築時に出所を渡す。**直接構築では `None` のままで、メッセージは住所を省く**
   —— 直接構築に出所は存在しないので、無いものを名乗らせない。

2. **付与は 1 か所で行う。** 3 つの署名を書き換えるのではなく、
   **`_build_params()` と `fit()` の外周で `CONFIG_INVALID` を捕まえ、
   まだ surface を名乗っていなければ住所を足して再送出する。**
   `raise ... from error` で連鎖を保つ。**握り潰さない** —— 常に再送出する。

   この形を選ぶ理由は**クラスを閉じるため**である。署名を 3 つ書き換えると 3 つの
   インスタンスは直るが、**次に adapter へ足される拒否は同じ欠陥を持って生まれる**。
   外周での付与は、**今後の拒否も含めて**規則を満たす。

3. **`_build_params` の 6 か所目（`seed` / `verbosity`）を `_pop_by_identity` 経由にする（#285）。**

   **この修正は既存の決定を撤回しない。** 実装時のコメントは「`seed` が `random_state` に
   優先するという受理済みの決定があり、ヘルパー経由にすると撤回になる」と書いていたが、
   **記録されているのは単一綴りの変換だけである** ——
   `BLUEPRINT.md:1187` と `HISTORY.md:2527`（sklearn 名 → Booster API 名）、および
   `test_lgbm_defaults.py:59-88`（`random_state=77` **だけ**を書くと `seed=77` になる）。
   **2 綴りが同時に書かれたときどちらが勝つかを固定した文書もテストも存在しない。**
   `_pop_by_identity` は 2 綴りあるときだけ拒否し、1 綴りなら pop して canonical で
   書き戻すので、**単一綴りの変換は保たれる**。当該コメント自体も訂正する。

### 影響範囲

- **公開 API**: `LGBMAdapter.__init__` に既定値つきの引数が 1 つ増える（後方互換）。
- **観測可能な振る舞い**: facade 経由の拒否メッセージが住所を含むようになる。
  直接構築の拒否メッセージは**変わらない**。
- **`_build_params` 直接構築で 2 綴りを書いた場合**、これまで黙って片方が選ばれていたのが
  `CONFIG_INVALID` になる。

### 互換性

```
Firing rate: 0 of every public surface -- {"seed": 7, "random_state": 7} and
             {"verbose": -1, "verbosity": -1} are already CONFIG_INVALID through
             model.params, fit(params=), calibration.params and the search space,
             because every surface runs check_duplicate_identities first (measured
             at 5715ee2, pinned by test_the_sixth_site_is_unreachable_from_every_surface).
             The only caller that reaches the picking branch is direct construction
             of LGBMAdapter, which is not a public entrance.
```

`allow` を**狭める**変更なので実測を記録する。**公開経路から到達可能な母集団は空である。**

### 代替案（検討して棄却）

1. **3 つの署名に `surface` を通す。** インスタンスは直るが**クラスが閉じない** ——
   次に足される拒否が同じ欠陥を持って生まれる。棄却。
2. **facade 側で objective / metric を先に検査する。** 1 つの境界に宣言を 2 つ持つことになり
   `BLUEPRINT.md` §14.4（round 25 の指摘）に反する。棄却。
3. **6 か所目を据え置く**（到達不能を根拠に）。到達不能は**現在の**facade についての測定であり、
   新しい入口が増えれば黙って開く。**5 か所が拒否し 1 か所が選ぶ**という非一貫性は
   DC4 の形そのものなので、閉じる。

### 受け入れ基準（テスト観点）

1. **3 つの位置すべてで、`model.params` と `fit(params=)` が別の住所を返す**
   —— `_pop_by_identity` の重複拒否、`_check_objective_compatible` の task 不一致、
   metric 検証の 3 つ。**メッセージが両 surface で異なることを主張する**
   （同一でないことだけでなく、それぞれが正しい住所を名乗ること）。
2. **直接構築では住所を名乗らない** —— 出所が無いのに名乗るのは偽である。
3. **付与は握り潰さない** —— 外周が捕まえた `CONFIG_INVALID` は必ず再送出され、
   `code` と `context` の既存キーが保たれる。**例外を出さない入力では何も変わらない。**
4. **今後の拒否も規則を満たす** —— adapter に新しい `CONFIG_INVALID` を足したテスト用の
   経路が、住所を明示的に書かなくても住所つきで届くこと（クラスが閉じている証拠）。
5. **6 か所目が 2 綴りを拒否する** —— `LGBMAdapter(params={"verbose": -1, "verbosity": 0})`
   が `CONFIG_INVALID`。
6. **単一綴りの変換は保たれる**（回帰させない）—— `test_lgbm_defaults.py` の
   `random_state` → `seed`、`verbose` → `verbosity` が無変更で通る。
7. **導出が最新であること** —— `instruments/refusal_surface_positions.py` の出力が、
   準拠 6 + 本提案で直す 3 を反映すること。

### Rule positions

```
Rule positions (R4, surface naming): 33 CONFIG_INVALID raising functions,
  derived by AST over lizyml/ (instruments/refusal_surface_positions.py)
  complying     : 6 name a surface in every raise
  fixed here    : 3 -- _pop_by_identity (#286), _check_objective_compatible,
                  metric_bridge metric validation
  dispositioned : 23 reachable from one entrance only (config reader, plots,
                  dataframe builder, splitters, search space, calibrator
                  construction), plus assert_plain_params which names its sink
                  by design rather than its entrance
```

### 分割点（事前宣言）

**本提案は PR 2 の残余のうち adapter 側だけを扱う。**
`param_domain` 側（**#284** の三重走査、**#287** の探索空間と 4 surface の不整合）は
**PR 2c で別提案**とする。理由は測定である —— PR 2 が 30 ラウンドを要したうち
rounds 16-26 は受理集合の面で費やされており、その面の再構成を adapter の作業と
束ねると同じ形が再現する。

**ラウンド予算 8**（計画 Revision 6 §12.6）。8 で一度止めて、完了基準に照らして
受け入れ／範囲限定でもう 1 回／さらなる分割 を判断する。

**#283 は本提案に含まない** —— H-0096 が `values_differ` ごと削除したため再現せず、
2026-09-09 に superseded として close した。

## H-0099: Reconcile tuning direction and refuse unconsumed search dimensions

- Status: accepted
- Scope: Config and tuning admission (#258, #279, #282; Phase 3 PR 3)
- Related: BLUEPRINT.md tuning and parameter ownership contracts

### Purpose

Every study must optimize the objective in its declared metric orientation.
Every accepted model/training search dimension must reach its consumer rather
than being silently discarded or overwritten by smart resolution.

### Proposal and decision

1. An omitted or null `tuning.optuna.params.direction` means automatic selection
   from the first effective evaluation metric's `greater_is_better`. Explicit
   `minimize`/`maximize` must agree or raise `CONFIG_INVALID` before study creation.
   Use null as the serialized automatic value: `model_fields_set` alone loses
   provenance on ordinary Config dump/load. This refines the Phase 3 plan's
   implementation suggestion while preserving its intended behavior.
2. Validate an existing or persisted Optuna study's direction before enqueue or
   optimize. Refuse a mismatch instead of relabeling an already-created study.
3. Refuse model dimensions claimed by active smart parameters, including native
   aliases. Reuse the provider's smart ownership authority. Validate the resolved
   space, including defaults/resume, before constructing the study. A search
   dimension that can activate a conflicting smart owner is also refused.
4. Refuse training dimensions other than `early_stopping_rounds` and
   `validation_ratio`, the two overrides actually consumed by training. Errors
   identify `tuning.optuna.space` and the offending name. Preserve valid default
   spaces and actual consumption of both supported training overrides.

### Scope and compatibility

The Config schema, tuning orchestrator, search admission, study direction check,
documentation and affected fixture declarations change. No estimator precedence
rule, public Result shape, persistence format version or dependency changes.
Matching explicit directions remain supported. Historical dumps containing an
explicit contradictory direction are refused; their provenance cannot be guessed.
Existing studies optimized in the wrong direction require a new study name.

Firing rate: 55/88 model-space configs, 9/23 training-space configs, and 0/159 explicit-direction configs observed through LizyMLConfig.model_validate during the full shipped baseline suite at 5fb8a809 (162 accepted tuning configs; provider alias/active-owner checks and effective first-metric orientation; direct constructor calls outside this recorder are not counted).

Baseline suite: 7756 passed, 230 skipped, 8 deselected. Added direction regressions
before implementation: 32 failed, 12 passed over the 22 registry-derived pairs
(10 wrong inferred orientations and 22 missing explicit-conflict refusals).

### Alternatives

- Keeping minimize as the automatic default silently picks the wrong extremum.
- Remembering explicitness only in memory changes meaning after dump/load.
- Letting sampled model values override smart settings changes existing
  precedence; rewriting model dimensions as smart ones guesses user intent.
- Sampling unused training settings creates misleading successful results.

### Migration

Omit direction or use null for automatic orientation. Remove contradictory
explicit values. To tune native leaves directly, disable `model.auto_num_leaves`;
otherwise declare the meaningful smart dimension. Remove unsupported training
dimensions rather than treating their recorded values as applied settings.
Update shipped fixtures intentionally requesting native leaves to disable their
smart owner, and convert regression cases for discarded settings to refusals.

### Acceptance criteria

- Registry-derived tests execute all 22 task/metric pairs and select the correct
  extremum; explicit contradictions fail before study creation.
- Defaults, parameterized metrics, Config round trips and existing study
  direction mismatches have regression coverage.
- Every active smart owner/native spelling is refused before training; disabled
  owners allow sampled model values to reach LightGBM. Supported training values
  still reach their consumers; unsupported names never start a study.
- Existing valid default spaces and tune/fit identity remain covered. All
  applicable lint, format, type and test gates pass.

## 2026-09-10: Merge partial tuning spaces with provider defaults

- ID: `H-0102`（2026-10-01 に `H-0100` から付け替え。PR 3c の `H-0100` と番号が重複していたため。参照の少ないこちらを動かした。H-0101 を参照）
- Status: `accepted`
- Scope: `Config | Tuning`
- Related: `H-0024`, `H-0068`, `H-0099`, `BLUEPRINT.md §11.3`

### Purpose

Resolve H-0024's contradictory replacement and per-dimension merge clauses.
A partial space must retain unspecified provider dimensions. At develop
`21c87c0cc02ce291c96f18989e8010cf19f98365`, all nine executed cases
(three tasks times learning_rate/eta/shrinkage_rate) lose nine defaults.
These are controlled regression cases, not natural usage prevalence.

### Proposal

- Add `tuning.optuna.space_mode: merge | replace`, default `merge`.
- Merge starts with provider defaults. User dimensions replace whole default
  dimensions by parameter identity; additional dimensions are appended. Model
  identity uses the provider canonical-name map. Other identities use category
  and name. Preserve the user's spelling and complete dimension specification.
- Same-layer duplicate user identities remain invalid. Merge does not waive
  category, training ownership or smart ownership checks.
- Replace uses only the user space, including an empty space, with no default
  fixed parameters. This preserves intentionally small/native-only spaces.
- Merge keeps existing fixed-value precedence: base model < fixed < sampled.
- Automatic boundary expansion still requires an empty/omitted user space in
  merge mode. Partial spaces require explicit `expand_boundary=True`.
- Resume reuses stored dimensions, bounds, expansion and fixed-default policy;
  later Config mutations must not reintroduce defaults into the stored space.

### Rule positions and scope

Searching `default_space(` and `_resolve_search_space(` under `lizyml/` finds
one operational call in tune, one resolver, the provider protocol/implementation
and default factory. The fresh/resume resolver and objective fixed/sample
overlays are the affected consumers. Factories still return provider defaults.
Direct Tuner and parse_space consume explicit dimensions, not Config merge
policy. This bounded inventory does not claim to cover downstream providers.

### Compatibility and Migration

Partial spaces inherit dimensions, potentially increasing training cost or
exposing smart/native conflicts. Set `space_mode: replace` to preserve previous
nonempty-space behavior. Empty/omitted spaces still use defaults. Config
serialization retains the mode. Artifact format remains unchanged and older
artifacts remain readable. Start a fresh study when changing space semantics;
old trials are not observations of the new space.

### Alternatives

Implicit replacement violates partial-space retention. Removing replacement
would unnecessarily remove legitimate explicit-only tuning workflows.

### Acceptance criteria

- All task defaults survive overrides, including model aliases; smart/training
  overrides preserve category and additional dimensions survive.
- Replace preserves explicit-only spaces and fixed-default policy.
- Config mode round-trips and rejects unknown values.
- Real Model.tune trials retain default dimensions and apply user overrides.
- Resume preserves dimensions, bounds and original fixed-default policy.
- Existing admission failures remain pre-study failures.
- Ruff, mypy and regression/full suites pass.

### Decision

- Date: `2026-09-10`
- Result: `accepted`
- The approved PR 3b scope requires partial-space retention. Explicit replacement
  preserves the old workflow without continuing silent default removal.
- Source inspection additionally identified `Model._merge_params` as the fit
  consumer of fixed defaults and `ModelPersistenceMixin.load` as the legacy
  Config rehydration boundary. Both are updated in this change. Legacy nonempty
  artifact spaces receive replace mode when the stored mode is absent.
- Review found and reproduced fresh merge-to-replace trial/fit divergence and
  loss of fixed policy after export/load with changed Config. Fresh trials now
  receive the new round's policy explicitly. FitState and the existing exporter
  carry optional effective fixed-policy metadata; load restores it while
  retaining the legacy fallback for artifacts without the field.
- A later review found that removed sampled model/smart dimensions still leaked
  into fresh trials through the prior best overlay. Actual-flow regressions
  reproduced both categories. Fresh studies now omit that entire prior tuning
  overlay; fit and resume continue to reuse the successful tuning result.
- Fresh-study admission also excludes prior best training parameters when
  checking native parameter conflicts. A real tune-to-fit regression reproduces
  obsolete early-stopping settings incorrectly refusing a fresh study; resume
  and fit still reject conflicts with the retained successful training policy.
- Resolved training dimensions can introduce ownership even when Config early
  stopping is disabled. Admission now checks those dimensions against effective
  native parameters and model dimensions before study creation. Regression cases
  cover inherited merge defaults, explicit replacement dimensions and aliases.

---

## H-0100: `calibration.params` を 3 手法すべてで反映し、Platt を原典の方法で推定する（#277 / PR 3c）

- **ステータス**: Accepted（管理者決定 2026-09-15）
- **起票日**: 2026-09-15
- **決定日**: 2026-09-15
- **スコープ**: `lizyml/calibration/{platt,beta,isotonic,base,_optimizer}.py`、`lizyml/core/_model_factories.py`（検査と前処理）、`lizyml/core/model.py`（calibration の配線）、`lizyml/core/_model_persistence.py`・`lizyml/codegen/{config_writer,generator,templates}.py`（生成コードでの再現）、`README.md`、`BLUEPRINT.md` §12.2 / §15.4、`CHANGELOG.md`、テスト。
- **関連**: [#277](https://github.com/nbx-liz/LizyML/issues/277)、`b465240`（`calibration.params` 導入）、H-0030、H-0031、H-0047、H-0058、H-0059、H-0090、H-0093、H-0094 決定 8、H-0095。設計と記録: `docs/audits/2026-09-defect-discovery/results/pr3c_design.md`（改訂 3）、`pr3c_design_review_round1.md`、`pr3c_design_review_round2.md`、`pr3c_monitor_round1.md`、`pr3c_platt_intercept_research.md`、`pr3c_calibration_params_measurement.txt`。

### 目的（課題）

`CalibrationConfig.params` は `b465240` で「method-specific overrides」として導入され、`get_calibrator(name, params=None)` が 3 手法共通の入口になった。**`isotonic` は反映しているが、`platt` と `beta` はコンストラクタで受け取って捨てている**（#277 の実測: `{"not_a_real_option": 123, "C": 0.001}` を渡しても calibrated メトリクスは完全一致、警告なし。DC4）。

さらに実装を読むと 2 つの欠落がある。

1. **`export_code` の生成コードは 3 手法すべてで `calibration.params` を捨てる。** `config.json` に calibration params が無く、`_fit_platt` / `_fit_beta` / `_fit_isotonic` は既定値を直書きしている。H-0059 は「新データで同一設定のまま calibrator を再構築できる」ことを約束している。
2. **facade は LightGBM の名前正規名化を method を問わず適用する**（`model.py` の `canonicalise_calibration_params`）。platt / beta の名前を反映し始めると、LightGBM の別名表で書き換えられて消費者に届かなくなる。

### 管理者の決定

1. **反映する**（#277 の選択肢 2。拒否ではない）。元の機能設計を重視する。
2. beta の上書き範囲は `x0 / method / bounds / tol / options`。
3. **platt の既定を原典の Platt（正則化なし・目標値平滑化）に寄せる。本 PR で行う。**
4. **platt は自前の Platt 最尤推定で実装する**（`scipy.optimize.minimize`）。上書き範囲は `x0 / method / bounds / tol / options / target_smoothing`。
5. intercept を正則化する経路（`liblinear`）は設けない —— 決定 4 で LogisticRegression を使わなくなるため、経路そのものが無くなる。

### 対応方針（決定）

#### 決定 1: Platt を原典どおりに推定する

BLUEPRINT §12.2 が名指す「Platt Scaling」は Platt (1999) の方法である: `P(y=1|f) = 1/(1 + exp(A·f + B))`、**A（slope）と B（intercept）を同時に最尤推定**、目標値は平滑化（`t+ = (N+ + 1)/(N+ + 2)`, `t− = 1/(N− + 2)`）、正則化項なし。**intercept はモデルの定義に含まれ、外せない**（研究ノートの実測: intercept なしは offset のずれで ECE 0.02 → 0.13〜0.20）。

LizyML の Phase 13 実装（`2ac4331`）は `LogisticRegression(C=1.0)`（L2・目標値 0/1）で、原典から逸脱していた。**本提案はこれを意図的に変える。** 利益を過大に書かない —— 実測の差は n=2000 で無し、n=100 で ECE 0.114 → 0.107 程度であり、変更の主な根拠は「BLUEPRINT が名指す手法の定義に合わせる」ことである。

- 係数は export 形式 `sigmoid(a·s + b)` で持つ（`a = −A`, `b = −B`）。`export_params` と `predict` の式は変えない。
- 既定: `target_smoothing=True`、`method="L-BFGS-B"`、初期値は Platt / scikit-learn と同じ `a=0`, `b=−log((N−+1)/(N++1))`、L-BFGS-B の options は scikit-learn `_sigmoid_calibration` と同じ `gtol=1e-6`, `ftol=64·eps`。
- `max(|s|) ≥ 30` のときは scikit-learn と同じくスコアを `k = max(|s|)` で割って最適化する。**利用者の `x0` と `bounds` の slope 成分を `k` 倍してから解き、結果の slope を `k` で割って戻す**（制約付き問題を同じにするため）。beta はスコアを確率にしてから対数を取るので縮尺しない。

#### 決定 2: 手法ごとの受理契約を calibrator が宣言し、facade が学習前に検査する

- 各 calibrator が classmethod `validate_params(params)` で**受理名・値の形・最適化手法の契約**を宣言する（`lizyml/estimators/` を import しない）。
- 利用者が書く `x0` と `bounds` は **`export_params` と同じ係数**で表す（platt `(a, b)`、beta `(a, b, c)`）。`bounds` は係数ごとの 2 要素**リスト**のリスト（H-0095 の受理集合で列の member は list であり tuple は拒否。受理集合は広げない）。
- **受理する最適化手法は閉じた表**: `L-BFGS-B`（既定）/ `TNC` / `SLSQP` / `trust-constr` / `Powell` / `Nelder-Mead` は `bounds` 可、`BFGS` / `CG` は `bounds` 不可（併記は拒否）。ヘッセ行列を要する手法は受理しない。
- **`tol` と `options` の優先順位**: 利用者の `options` のキー ＞ 利用者の `tol` ＞ LizyML の既定。scipy は `tol` を `options.setdefault` で渡すので、利用者が `tol` を書いたときは LizyML の既定 options のうち `tol` が設定するキーを入れない。既定の options は手法ごとに持つ（L-BFGS-B の `ftol` を BFGS に渡さない）。
- **`options` のキーは実物の scipy で確かめる**: 検査時に小さな既知の問題に `minimize` を 1 回かけ、`OptimizeWarning: Unknown solver options` を拒否に変える（名前の表を写さない。scipy のバージョン差に追随する）。
- facade の `check_calibration_param_names` が method ごとの宣言を呼ぶ。**位置は変えない**: `fit()` と `tune()` のマージ直後、Booster も study も学習する前（H-0093 決定 6）。**LightGBM 固有の名前検査は isotonic 限定のまま。**

#### 決定 3: 前処理を method で分ける

値の正規化（H-0095 `normalise_params`, `surface="calibration.params"`）は 3 手法すべて。**LightGBM の名前正規名化は isotonic のみ**（自前キー除外は従来どおり）。platt / beta の名前は書き換えない。実行時と `export_code` は同じ前処理を使う。

#### 決定 4: 生成コードで再現する

経路: `_model_persistence.export_code` → `generator.generate_code` → `config_writer.build_config` → `config.json` の `calibration_params`（前処理後の実効値）→ 生成 `train.py::fit_calibrator` → `_CAL_FITTERS[method](scores, y, params)`。生成 fitter は実行時と同じモデル・既定値・上書き・手法の契約・縮尺を使う。`_fit_isotonic` は `"verbosity": -1` に揃え、自前キーの上書きと 20 行未満で early stopping を切る振る舞い（H-0047）まで再現する。生成 `requirements.txt` は platt または beta のとき scipy を明示し、テンプレートの注記と README の依存の記述を同時に直す。

#### 決定 5: 旧 artifact の platt calibrator を読み込み時に移行する

`fit_result.pkl` は calibrator オブジェクトごと pickle されており、旧 `PlattCalibrator` は `_model: LogisticRegression` を持つ。`PlattCalibrator.__setstate__` が旧状態を `(a, b)` に変換する（`coef_[0, 0]`, `intercept_[0]`）。未学習の旧状態（`_model=None`）は新しい未学習状態に、新しい状態はそのまま。predict は数値的に安定な sigmoid で計算し、通常のスコアと極端なスコアで旧 predict と許容誤差内で一致させる。

**`FORMAT_VERSION` は 2 のまま**: 公開の Artifact 契約（ディレクトリ構成・metadata・`Model.load()` の振る舞い）が変わらず、旧モデルの推論を保つ内部状態の変換であって、`CLAUDE.md` §3 の破壊的変更に当たらない。**保証範囲の外**: H-0030 より前（確率を入力にしていた時期）の artifact。

#### 決定 6: 文書の食い違いを訂正する

BLUEPRINT §12.2 と H-0047 は「isotonic の `Booster.predict()` は raw score を返すので sigmoid を適用する」と書くが、`objective="binary"` の `Booster.predict()` は確率を返し、実装（`isotonic.py` の predict）は sigmoid を適用しない。**実装が正しく、文書が古い。** BLUEPRINT を訂正する（H-0047 の本文は記録として残す）。

### Rule positions

```
Rule positions (calibration.params reaches its consumer or is refused before training):
  derived from CalibratorRegistry (3 calibrators) x consumers (runtime cross-fit +
  C_final, generated _CAL_FITTERS) = 6, plus entrances fit/tune = 2, plus the
  legacy-artifact reader = 1
  complying     : 1 -- runtime isotonic (and the entrance check, for isotonic only)
  fixed here    : runtime platt, runtime beta, generated platt/beta/isotonic,
                  both entrances for platt/beta, the legacy platt reader
  dispositioned : direct construction of a calibrator outside the registry, and any
                  route that hands a calibrator values other than calibration.params
  bound         : "3 x 2" names the categories to cover, not proof that values are
                  carried -- the proof is the reach/effect tests; the derivation is
                  pinned by tests requiring every registered calibrator to declare
                  validate_params and to have a generated fitter
```

### Firing rate

```
Firing rate: 2/75 of the calibrated platt and beta configs the shipped suite builds
             carry a non-empty calibration.params (platt 1/71, beta 1/4); both come from
             test_calibrators_that_do_not_use_lightgbm_are_not_checked, which pins the
             old accept-and-ignore behaviour and changes with this proposal. 0 from any
             other test. (instruments/calibration_params_firing_rate.py, replayed over
             the full suite at 92e32ee on branch fix/phase3-pr3c-calibration-params;
             report in results/pr3c_calibration_params_measurement.txt)
```

### 影響範囲

- 公開 API（`CalibrationConfig`、`get_calibrator` のシグネチャ）は変えない。`BaseCalibratorAdapter` に `validate_params` を追加する（既定は空でない params を拒否）。
- **platt の既定が変わる** → 新しい fit の calibrated 結果が変わる。
- **これまで無視されていた platt / beta の params が効く**。LogisticRegression の引数名（`C` 等）は未知名として拒否される（これまでも効いていなかった）。
- 生成 `config.json` にキー `calibration_params` が増える。生成 `requirements.txt` と README に platt でも scipy が載る。

### 互換性

- **旧 artifact は読み込め、predict は変わらない**（決定 5）。`format_version` は 2。
- 既に生成済みのコードは影響を受けない。
- 依存: 新しいインストール要件は増えない（インストール済み scikit-learn 1.8.0 のメタデータは `scipy>=1.10.0` を必須にしている。scikit-learn の全バージョンの下限までは確認していない）。CI の lowest-direct レーンは `--frozen` なので、**scikit-learn 1.3 / scipy 1.10 を実際に入れた隔離環境で確認し、解決されたバージョンを記録する**。

### 代替案（検討して棄却）

1. **拒否する**（#277 の選択肢 1）。管理者が反映を選んだ。元の設計（`b465240` の method-specific overrides）とも反する。
2. **LogisticRegression を使い続け、目標値平滑化を重み複製で再現する。** 数値は scikit-learn の Platt と許容誤差内で一致した（研究ノート §6.1）が、scikit-learn 1.8 で `penalty` が非推奨になり、正則化なしの指定がどう書いても警告を出す。受理名を LogisticRegression のシグネチャから導出すると、受理される config が scikit-learn のバージョンで変わる。
3. **既定値の変更を別 PR にする**（ループ監視の `redirect` 勧告）。管理者が本 PR を選んだ。分割点は事前宣言した（下記）。

### 分割点（事前宣言）

「platt の既定値変更・自前 MLE・旧 artifact 移行」は「3 手法への params の反映・前処理の分離・入口検査・生成コードの再現」から分けてコミットできる塊として扱う。ラウンド予算 8 に達したとき、またはこの塊だけに検証の問題が残ったときは PR 3c-2 に切り出す。

### 受け入れ基準（テスト観点）

`docs/audits/2026-09-defect-discovery/results/pr3c_acceptance_criteria.md` の対応表を正とする。要旨:

1. 既定の platt が scikit-learn の `_sigmoid_calibration` と許容誤差内で一致する（参照はテストでのみ使う）。
2. offset のずれがあるスコアで intercept が推定され、intercept を 0 に固定した fit より損失が小さい。
3. 3 手法で params が観測可能な効果を持ち、cross-fit の全 fold と C_final に届く。
4. `tol` の効き目と `options` の優先、手法表の各手法での fit、組み合わせと未知 option の拒否。
5. 大きなスコアで、`x0` / `bounds` が書いた座標で効き、縮尺しない解き方と同じ問題になる。
6. 違反は fit でも tune でも Booster と study が学習される前に `CONFIG_INVALID`、出所 `calibration.params`。
7. platt / beta の名前は正規名化されず、isotonic の別名は従来どおり正規名になる。
8. 生成 fitter が実行時と一致し、params で変わり、生成コードで再学習が走る。`config.json` に実効値。
9. 旧 platt calibrator の artifact が通常・極端なスコアで同じ predict を返す。未学習の旧状態、新状態の再読込も通る。
10. 既定の platt / beta の fit で警告が出ない。登録された calibrator すべてに `validate_params` の宣言と生成 fitter がある。README と生成 requirements の scipy の記述が一致する。

## H-0101: `TimeHoldoutInnerValid.gap` を「自動解決だけが設定する引数」と明文化し、HISTORY の ID 重複を解消する（#265 / PR 3d）

- **ステータス**: Accepted
- **起票日**: 2026-10-01
- **決定日**: 2026-10-01
- **スコープ**: `BLUEPRINT.md` §10.3.3（明示指定時の gap の記述を訂正、`gap` が Config フィールドでないことを明記）, §11.3（H-0100 → H-0102）, `HISTORY.md`（2026-09-10 の探索空間マージ項目の ID を H-0102 に付け替え）, `lizyml/core/_model_persistence.py` / `tests/test_tuning/test_default_space.py`（コメント内の ID 参照のみ）, `tests/test_training/test_inner_valid_purge_embargo.py`（追加）, `tests/test_docs/test_history_ids.py`（新規）
- **関連**: [Issue #265](https://github.com/nbx-liz/LizyML/issues/265), H-0092（#265 の本体を解決）, H-0085（gap 伝播）, [#268](https://github.com/nbx-liz/LizyML/issues/268)（Config から届かない構築子引数の一覧）, H-0100（calibration）, H-0102（旧 ID H-0100 の探索空間マージ）

### 目的（課題）

**1. #265 の close レビューで残った 2 点。** 2026-10-01 の独立した close レビューが、H-0092 の修復の後に残る次の 2 点を示した。

- `BLUEPRINT.md` §10.3.3 は「`training.early_stopping.inner_valid` を明示指定した場合は継承せず、明示された値（既定 `gap=0`）を使う」と書いていた。**明示できる値は存在しない。** `TimeHoldoutInnerValidConfig` は `method` と `ratio` だけを持ち `extra="forbid"`（`lizyml/config/schema.py`）、明示指定の factory は `ratio` だけを渡す（`core/_model_factories.py`）。満たせる入力が無い宣言である（DC7 の形）。
- #265 の DoD は「Config から届かない `TimeHoldoutInnerValid.gap` の処分を記録する」ことを求めていた。H-0092 は現状維持を暗に含むが、処分を明文では書いていない。

**2. H-0100 の重複。** 2026-09-10 の探索空間マージ項目（PR 3b、#293）は本文の ID 行で `H-0100` を名乗り、2026-09-15 の calibration 項目（PR 3c、#296）も見出しで `H-0100` を名乗っていた。並行したブランチがそれぞれ次の空き番号を取り、両方がマージされた。ID は各ブランチ上で採番され、重複はマージ後にしか存在しないので、どちらのブランチの検査にも映らない。

### 対応方針（決定）

1. **`gap` は自動解決だけが設定する構築子引数とし、Config には出さない。** 明示指定の経路は常に `gap=0` である。`gap` キーは `extra="forbid"` により検証で拒否される（黙って捨てられはしない）。境界 gap が必要な利用者は outer split に `purge_gap` / `embargo` / `gap` を設定し、inner valid を自動解決に任せる。これで #268 の `TimeHoldoutInnerValid.gap` 行は「意図的に Config 外」として処分される。
2. **§10.3.3 の記述を「明示指定時は常に `gap=0`」に訂正し、`gap` が Config フィールドでないことと、その代替手段を明記する。**
3. **ID の重複は、参照の少ない探索空間マージ項目を `H-0102` に付け替えて解消する。** 参照は探索空間側が 5 か所（HISTORY 本文 1、BLUEPRINT 2、コードとテストのコメント 2）、calibration 側が約 60 か所だった。付け替えた項目の ID 行に旧番号と理由を残す。
4. **HISTORY の各項目が ID をちょうど 1 つ宣言し、同じ ID を 2 項目が宣言しないことをテストで固定する**（`tests/test_docs/test_history_ids.py`）。文法は閉じる: 項目は fenced code の外の `## ` 見出し、ID の綴りは実測した 2 種（見出しの `## H-NNNN` と本文の ``- ID: `H-NNNN` ``）。ID が 0 個・2 個の項目は名前つきで失敗にし、読み飛ばさない。実測: 2026-10-01 の HISTORY.md は 102 項目、全項目がどちらかの綴りでちょうど 1 つを宣言し、重複は H-0100 の 1 件のみ。

### 互換性

- 振る舞いの変更は無い。`gap` キーは以前から検証で拒否されており、明示指定の経路は以前から `gap=0` だった。今回はそれを仕様に書き、テストで固定した。
- `format_version` / 公開 API / Result の形と意味は変わらない。
- `H-0100` を探索空間マージの意味で引用していた外部の記録は `H-0102` と読み替える。リポジトリ内の参照はすべて付け替えた。

### 代替案（検討して棄却）

1. **`gap` を Config に公開する。** 新機能であり、#265 の残作業の範囲を超える。outer split の設定で同じ目的を達成できる。必要になれば別の Proposal で扱う。
2. **ID を付け替えず、両項目に注記を付ける。** 「H-0100」が 2 つの決定を指し続け、引用のたびに曖昧になる。
3. **calibration 側を付け替える。** 参照が約 60 か所あり、コード・テスト・監査記録の変更が大きい。

### 受け入れ基準（テスト観点）

- `tests/test_training/test_inner_valid_purge_embargo.py::TestExplicitInnerValidDoesNotInheritGap::test_gap_is_not_a_config_field`: 明示指定の inner valid に `gap` を書くと `CONFIG_INVALID`（pydantic の `extra_forbidden`、対象は `gap` キー）。既存の `test_explicit_time_holdout_gets_no_gap` / `test_auto_and_explicit_differ_for_the_same_outer_split` は不変のまま green。
- `tests/test_docs/test_history_ids.py`: 現在の HISTORY.md で違反 0 件、100 項目以上を解析（空回りの防止）。重複・ID 無し・ID 2 個・fenced code 内の見出しの 4 形を合成入力で検査する。**付け替え前の HISTORY.md では H-0100 の重複を報告して失敗することを確認した。**

## H-0103: 最終 refit を CV fold と同じ重み付けで学習させ、`RefitTrainer.fit` の入力差を方針として固定する（#269 / PR 4）

- **ステータス**: Accepted
- **起票日**: 2026-10-01
- **決定日**: 2026-10-01（設計レビュー round 1 の指摘で受け入れ基準を改訂、コードレビュー round 1 で APPROVE）
- **スコープ**: `lizyml/training/refit_trainer.py`（`fit` に `sample_weight` を追加し、inner valid があれば inner-train 行に絞る）, `lizyml/core/model.py`（`RefitTrainer.fit` の呼び出しに `sample_weight=tc.sample_weight` を渡す）, `BLUEPRINT.md` §5.3（`balanced`）/ §8 手順 8 / §10.3（refit の inner valid）, `tests/test_training/test_cv_refit_parity.py`（新規）
- **関連**: [Issue #269](https://github.com/nbx-liz/LizyML/issues/269), H-0050（`TrainComponents` を CV と refit で共有）, H-0085（refit の pipeline fit 境界）, H-0036（ratio params を inner-train の大きさで解決）, [#301](https://github.com/nbx-liz/LizyML/issues/301)（生成コード側の同じ規則、繰り延べ）, [#263](https://github.com/nbx-liz/LizyML/issues/263) / [#272](https://github.com/nbx-liz/LizyML/issues/272)（fingerprint を誰も照合しない件、PR 6）

### 目的（課題）

`RefitTrainer.fit` は `CVTrainer.fit` の 7 つの入力のうち 3 つ（`X` / `y` / `groups`）しか受け取らない。欠けている 4 つ（`sample_weight` / `time_values` / `data_fingerprint` / `run_meta`）について、受け取るか、受け取らない理由を方針として書くかを決める。

**実害があるのは `sample_weight` だけで、multiclass に限られる。** `balanced` はタスクごとに解決の仕方が違う（`BLUEPRINT.md` §5.3、`estimators/lgbm/smart_params.py`）: regression は `UNSUPPORTED_TASK`、binary は native パラメーター `scale_pos_weight` になって estimator factory 経由で両 trainer に届く、multiclass だけが行ごとの `sample_weight` 配列になる。実測（`develop` `96171da`、クラス不均衡な 639 行、3 fold、early stopping は既定どおり有効）: `balanced: true` では **CV の 3 fold の学習 Dataset（inner-train 383 行ずつ）はすべて重み付き、refit の学習 Dataset（inner-train 575 行）は重みなし**。eval set はどれも重みなし。`balanced: false` ではどれも重みなし。設計レビュー round 1 が `balanced: null`（multiclass の既定）でも同じ欠陥を実行で再現した。

つまり **multiclass の `balanced` では、利用者に OOF 指標として見せたモデル群と、predict / export に使う最終モデルが違う重み付けで学習されている。** `BLUEPRINT.md` §8 手順 8 は「同一の `TrainComponents` を使用し、CV との一貫性を構造的に保証する」と書くが、`TrainComponents.sample_weight` を refit は読んでいないので、この記述は今の実装では偽である。

### 対応方針（決定）

1. **`sample_weight` は refit に渡す。** `CVTrainer._fit_estimator` と同じ規則に揃える: inner valid があれば inner-train 行の重み（`sample_weight[inner_train_rel]`）だけを estimator に渡し、inner-valid 行には重みを付けない。inner valid が無ければ全行の重みを渡す。重みは `Model.fit` が `TrainComponents.sample_weight`（全データの `y` から計算済み）として既に持っているので、新しい計算はしない。呼び出し側は `core/model.py` の `refit_trainer.fit(X, y, groups)` に `sample_weight=tc.sample_weight` を足す。
2. **`time_values` は渡さない（方針）。** 時間順の outer split（`time_series` / `purged_time_series` / `group_time_series`）では、`data/dataframe_builder.py` が CV の前に全行（`X` / `y` / `groups` / 時間列）を時間列で並べ替える。refit はその並べ替え済みの `X` / `y` を受け取り、**自動解決される** inner valid の strategy（時間順 split では `TimeHoldoutInnerValid` など）は行順で分割する（時間値を読まない）。`method: holdout` を明示指定すれば並べ替え後でもシャッフルされるが、それは BLUEPRINT §10.3.1 が警告つきで認める利用者の選択であり、CV と refit で同じに働く。`CVTrainer` も inner split に `time_values` を渡しておらず、使い道は fold ごとの時間範囲を `FitResult.splits.time_range` に記録することだけで、全データを 1 回学習する refit にはそれに当たる記録が無い。**#269 の「refit は時間列を見られないので時間順の inner split ができない」は成り立たない**: 並べ替えが両 trainer の前に済んでいる。この根拠はテストで固定する（行をシャッフルして渡しても、refit に届く `y` が時間順であること）。**根拠の範囲**: 並べ替えは `core/model.py` の `_TS_METHODS`（上記 3 手法）に限られる。時間順でない outer split に `time_holdout` の inner valid を明示指定した場合は並べ替えが起きず、**CV も refit も入力の行順の末尾を** inner-valid にする（`validate_time_series_order` は公開されているが `Model.fit` からは呼ばれない）。これは CV と refit の差ではないので本 Proposal の範囲外である。
3. **`data_fingerprint` は渡さない（方針）。** `Model.fit` が 1 回の呼び出しの中で、両 trainer に渡す同じ `X` から 1 度だけ計算し（`fp_compute(X)`）、`FitResult.data_fingerprint` に記録する。refit は同じ呼び出しの中で同じ `X` から作られ、artifact には `FitResult` と一緒に保存される。refit に渡しても同じ値をもう 1 つ持つだけである。**fingerprint を誰も照合していないこと**（`lizyml/` の中で `data_fingerprint` を読むのは記録する 2 か所だけ）は #263 / #272（PR 6）の範囲であり、本 Proposal では扱わない。
4. **`run_meta` は渡さない（方針）。** fit 呼び出し 1 回につき 1 つの記録（バージョンと config）であり、`FitResult` が持つ。refit はその fit の一部である。
5. **2 つの trainer の入力差を恒久検査にする。** `inspect.signature` で両方の `fit` を読み、一方だけが受け取る入力はすべて「もう一方も受け取る」か「本 Proposal の決定番号を持つ方針」のどちらかであることを主張する。方針に登録した名前が実際に `RefitTrainer.fit` に無く `CVTrainer.fit` に有ることも主張する（誰かが後で渡し始めたら、登録が古びたことが落ちて分かる）。
6. **生成コード（`export_code` の `train.py`）は本 PR で直さず、#301 に繰り延べる。** 下記「規則が縛る位置」の 2。

### 規則が縛る位置（ソースから導出、実装前）

規則: **最終モデルは、CV の各 fold と同じ重み付けで学習する。** 位置の導出: `lizyml/` 全体を `lgb.train(` / `lgbm.train(` / `lgb.Dataset(` / `estimator.fit(` で grep し、学習を行う呼び出しを列挙した（2026-10-01、`96171da`）。bound: この 4 綴りで呼ばない学習経路は列挙の外にある。ただし H-0093 の `tests/_ast_scan.py` が LightGBM への経路の母集団を別途 AST で固定しており、新しい経路はそちらで検出される。

| # | 位置 | 規則との関係 | 本 PR |
|---|---|---|---|
| 1 | `training/refit_trainer.py` `fit` → `estimator.fit` | 最終モデルを学習する。重みを受け取っていない | **修正** |
| 2 | `codegen/templates.py` `train_lgbm`（生成 `train.py`） | 生成プロジェクトでの再学習。multiclass の重みを一切計算しない（実測: `config.json` / `train.py` に重みに当たるものが無い。binary は `lgbm_params.scale_pos_weight` で届く） | **繰り延べ #301**。生成 `train_lgbm` は早期停止の分割からして LizyML の refit と異なる（`InnerValidStrategy` ではなくシード付きランダム holdout）ので、「LizyML と同じく学習する」は今の生成コードが約束していない保証であり、重みだけ直しても約束にならない。**繰り延べで外れる保証**: 生成コードから再学習した multiclass `balanced` モデルは重みなしで学習される。`predict.py` は export された booster を読むので影響しない |
| 3 | `training/cv_trainer.py` `_fit_estimator` | 規則の基準側（既に重み付き） | 変更なし |
| 4 | `core/_model_tuning.py` の `CVTrainer` | tune は CV だけを回し refit しない。既に重みを渡している | 変更なし |
| 5 | `calibration/isotonic.py` `lgbm.train`、`codegen/templates.py` `_generate_oof` / `_fit_isotonic` | calibration は binary 専用（multiclass は `CALIBRATION_NOT_SUPPORTED`）。binary の `balanced` は重み配列を作らない | 対象外 |

### 互換性

- **multiclass で `balanced` が有効な fit（`balanced: null` の既定も含む）は、最終モデルが変わる。** refit が重み付きで学習するので、`predict` / `export` の結果が変わりうる（クラスがほぼ均等なら重みはほぼ 1 で、結果がほとんど変わらないこともある）。CV の各 fold と OOF 指標は変わらない。これは欠陥の修正であり、利用者が見てきた OOF 指標を出したモデルに、最終モデルが揃う方向の変化である。
- regression / binary、および `balanced: false` の multiclass は変わらない（実測で確認する。受け入れ基準 2）。
- `format_version` / `FitResult` / `RefitResult` / `PredictionResult` の形と意味は変わらない。保存済み artifact はそのまま読め、predict も変わらない（保存された booster を使う）。
- `RefitTrainer.fit` に省略可能なキーワード引数が 1 つ増える。既存の呼び出しはそのまま動く（`RefitTrainer` は `lizyml.training` から公開されている）。
- Firing rate: 本 Proposal は skip / shorten / cache / select / allow / conditionally-activate のいずれの条件も新設しない。既に計算されて CV に渡っている重みを refit にも渡すだけであり、重みが有るかどうかの分岐は既存の `balanced` の解決（`smart_params.py`）が決める。

### 代替案（検討して棄却）

1. **`time_values` / `data_fingerprint` / `run_meta` も渡す（対称性のため）。** 受け取っても使い道が無い引数になる。`time_values` は refit に記録先が無く、fingerprint と run_meta は同じ値の複製である。使われない入力を足すことは、#268 が数えた「届かない knob」を増やすことになる。
2. **refit で重みを inner-train ではなく全行に付ける。** CV の規則（inner-valid 行には重みを付けず eval set として使う）とずれ、「CV と同じ重み付け」にならない。
3. **生成コードも本 PR で直す。** 上記の位置 2 の理由で、何を再現すると約束するかの決定が先に要る。#301 で Proposal を立てる。

### 受け入れ基準（テスト観点）

`docs/audits/2026-09-defect-discovery/results/pr4_acceptance_criteria.md` の対応表を正とする。要旨:

1. multiclass で `balanced` が有効な 3 つの書き方（`true` / `null` / 省略 = 既定）それぞれで、refit の学習 Dataset が受け取る重みが `compute_sample_weight("balanced", y)` の該当行と**値で**一致する。early stopping あり（inner-train 行に絞る）と、なし（全行）の両方。CV fold の重みも同じ規則で一致し、eval set は重みを持たない。修正前は RED。
2. 重み配列を作らない組み合わせを個別に固定する: regression `true` は `UNSUPPORTED_TASK` で学習 0 回 / regression `false` は重みなしで学習 / binary `true` は重み配列なしで、CV と refit の params に同じ `scale_pos_weight` が載る / binary `false` と multiclass `false` は重み配列も `scale_pos_weight` もなし。（multiclass `true` は基準 1。）
3. 時間順の split に行をシャッフルして渡したとき、`RefitTrainer.fit` に届く `y` が時間順に並んでいる（決定 2 の根拠を実行で固定する）。
4. 入力差の恒久検査（決定 5）: `inspect.signature` から読んだ差がすべて受理済みか方針登録済みで、登録名は実際に `RefitTrainer.fit` に無く `CVTrainer.fit` に有る。
5. `BLUEPRINT.md` §5.3 / §8 手順 8 / §10.3 が、refit の重み付けと 3 つの方針を述べる。

## H-0104: feature pipeline の拡張点を宣言どおり使えるようにし、未知カテゴリの置換を報告し方針を Config に出す（#259 / #260 / PR 5）

- **ステータス**: Accepted
- **起票日**: 2026-10-01
- **決定日**: 2026-10-01（設計レビュー round 1 の指摘で改訂。コードレビューは round 2 で authorship 停止条件が発火、管理者判断で広範な round 3、round 4 で APPROVE）
- **スコープ**: `lizyml/features/pipeline_base.py`（`transform_with_warnings` の既定実装、`get_state` の `categorical_cols` を文書化）, `lizyml/features/column_check.py`（新規: 推論時の列検査の唯一の実装）, `lizyml/features/pipelines_native.py`, `lizyml/features/encoders/categorical_encoder.py`（置換の報告）, `lizyml/core/_model_predict.py`（facade で列検査）, `lizyml/config/schema.py`（`FeaturesConfig.unseen_policy`）, `lizyml/estimators/provider.py` / `lizyml/estimators/lgbm/provider.py`（`build_pipeline_factory(unseen_policy=...)` = **公開 Protocol の変更**）, `lizyml/core/model.py` / `lizyml/core/_model_tuning.py`（呼び出し）, `lizyml/codegen/templates.py` / `config_writer.py` / `generator.py`（生成 `predict.py` の置換ログと欠損の扱い、生成 `train.py` が方針を保つ）, `BLUEPRINT.md` §5.4 / §9.2, `docs/config-reference.md`, `ARCHITECTURE.md`, テスト（新規 `tests/test_features/test_pipeline_conformance.py` / `tests/test_features/test_unseen_policy.py` / `tests/test_codegen/test_unseen_policy_codegen.py`）
- **関連**: [Issue #259](https://github.com/nbx-liz/LizyML/issues/259), [Issue #260](https://github.com/nbx-liz/LizyML/issues/260), H-0054（pipeline factory を provider 経由に）, H-0085（pipeline の fit 境界）, #205（生成 `predict.py` が `unseen_policy` を再現）, PR 6（#263: `INCOMPATIBLE_COLUMNS` をこの検査から出す）

### 目的（課題）

**#259.** 推論経路（`core/_model_predict.py`）と SHAP（`explain/shap_explainer.py`）は `pipeline.transform_with_warnings(X)` を無条件に呼ぶが、このメソッドは `BaseFeaturePipeline` の抽象インターフェース（`fit` / `transform` / `get_state` / `load_state`）に無い。宣言どおりに 4 メソッドだけ実装した pipeline は、学習は通り、`predict` で `AttributeError` になる。

**到達可能性（正直に書く）**: 自作の pipeline が `Model` に入る経路は provider の `build_pipeline_factory` だけで、provider は `get_provider` が `lgbm` を直書きで選ぶ。**公開の登録手段は無い。** したがって今日この欠陥に当たるのは、本体に新しい学習器（BLUEPRINT の `EstimatorProvider` 節が `build_pipeline_factory` の用途として挙げる例: EntityEmbedding）を足す開発者か、provider を差し替える利用者だけであり、出荷コードでの発火は 0 件（実装は `NativeFeaturePipeline` だけ）。それでも直すのは、**宣言したインターフェースと実行時に要求されるものが食い違う契約の欠陥**だからである。

**#260.** `CategoricalEncoder` は推論時の未知カテゴリを、既定（`unseen_policy="mode"`）で学習時の最頻値に**黙って**置き換える。警告も `PredictionResult.warnings` も無い。`BLUEPRINT.md` §7.3 は `warnings` を「補正が走った場合の通知」と定め、同じメソッドの列ズレ（余剰列）は報告しているので、この分岐だけが矛盾している。方針は Config から選べない（`FeaturesConfig` に無く、provider は引数なしで pipeline を作る）。

### 対応方針（決定）

1. **`BaseFeaturePipeline.transform_with_warnings` を具体メソッドとして追加する。** 既定実装は `(self.transform(X), [])`。抽象にすると既存の外部サブクラスを壊すので取らない。`NativeFeaturePipeline` は従来どおり上書きする。
2. **`get_state()` の `"categorical_cols"` キーは任意（既定は空）と文書化する。** trainer は既に `get_state().get("categorical_cols", [])` で読み、無ければ「カテゴリ列なし」として学習する。基底クラスの docstring と BLUEPRINT に、このキーが estimator に渡すカテゴリ列の宣言であることを書く。
3. **推論時の列検査を facade に置く（計画 §PR 5 の「インターフェースの一部にする」の具体化）。** 計画がそう決めた理由は「自作の pipeline が検査をすり抜けられないように」（PR 6 が `INCOMPATIBLE_COLUMNS` を全経路で出すため）である。**基底クラスの上書き可能なメソッドに置いてもその理由は満たせない**（サブクラスが上書きすれば検査は消える）。pipeline に渡す**前**に facade で検査すれば、構成上すり抜けられない。実装は 1 つにする: `features/column_check.py` の純関数が `(X, feature_names)` を受け取り、不足列は `DATA_SCHEMA_INVALID`、余剰列は警告を返し、学習時の列順で選んだ `X` を返す。`run_predict` はこれを呼んでから pipeline に `X[feature_names]` を渡し、`NativeFeaturePipeline.transform_with_warnings` も単体利用のために同じ関数を呼ぶ（facade が先に列を選ぶので、推論経路で余剰列の警告が 2 度出ることはない — テストで確かめる）。fit 時の列は `RefitResult.feature_names` から取る。**範囲は今日の振る舞い（不足 / 余剰）だけ**で、dtype の不一致 → `INCOMPATIBLE_COLUMNS` は PR 6 の RED としてこの関数に足す。**bound**: 新しいデータを pipeline に通す **`Model` facade の**公開の入口は `Model.predict` → `run_predict` の 1 か所だけ（pipeline を直接使う場合は、その pipeline 自身の検査に任される）（`lizyml/` 全体を `transform_with_warnings` / `pipeline.transform(` / `.load_state(` で grep した。SHAP は学習時のデータを変換する）。
4. **`CategoricalEncoder` は行った置換を返す。** `transform_with_warnings(X) -> (X, warnings)` を足し、`transform` はそれに委ねる。`NativeFeaturePipeline.transform_with_warnings` が列ズレの警告に連結するので、`PredictionResult.warnings` に届く。
5. **`"nan"` も警告する。** 未知カテゴリを欠損にするのも適用された補正である（§7.3）。方針ごとの観測結果は: `"mode"` = 警告 + 最頻値、`"nan"` = 警告 + 欠損、`"error"` = `DATA_SCHEMA_INVALID`。
6. **`FeaturesConfig.unseen_policy: Literal["mode", "nan", "error"] = "mode"` を追加する。** 既定は現行の `"mode"` なので既存の config の挙動は変わらない。値は `EstimatorProvider.build_pipeline_factory(unseen_policy=...)`（キーワード引数、既定 `"mode"`）で pipeline に届ける。`fit` と `tune`（CV）は config の値を渡す。**推論時の方針は保存済みの pipeline 状態から復元される**（`CategoricalEncoder.get_state()` が `unseen_policy` を持つ）ので、`run_predict` は引数を渡さない。artifact は fit が適用した方針を記録しており、読み込み後の config ではなくそれに従う。
7. **方針は CV の検証 fold にも同じく効く。** 同じ encoder の `transform` が各 outer-valid fold に掛かる。検証 fold は学習に使っていないデータであり、推論時と同じ状況だからである。**ただし fold ごとに未知カテゴリが生じる列は限られる**（実装中に実測）: 既定の `auto_categorical: true`、または `features.categorical` に挙げた列は、データ構築時（`data/dataframe_builder.py` `_apply_categorical`）に**全行の値で** `category` 型になり、encoder はその宣言済みカテゴリを学ぶので、どの fold でも未知にならない。fold ごとに未知が生じるのは、`auto_categorical: false` で、`NativeFeaturePipeline` が文字列型としてカテゴリ扱いする列だけである。その場合 `"error"` では、学習 fold に無い値が検証 fold にだけ現れると **`fit` が `DATA_SCHEMA_INVALID` で止まる**。これを §9.2 に書く。実測（`develop` `2d3bc54`、既定を `"error"` に差し替えてフルスイートを実行）: CV 中にこの拒否が起きたのは、それを起こすために作られたテスト（`test_valid_only_category_raises_proving_train_only_fit`）1 件だけで、自然な設定での発火は 0 件。
8. **fit 中（CV の検証 fold）と SHAP 重要度での置換は報告しない（範囲の外）。** `FitResult` にも SHAP 重要度の戻り値にも警告の通り道が無い。#260 の DoD は推論時の報告を求めている。**SHAP 重要度は方針をそのまま適用する**（設計レビュー round 1 の blocking 1 で訂正）: SHAP 重要度は**最後の CV fold の** pipeline 状態（`FitResult.pipeline_state`）で学習データ全体を変換する。スライディング窓（`train_size_max`）では、その fold のどこにも属さない行があり、決定 7 の文字列列ではその行だけが持つ値が未知になる。実測: `"error"` では `fit` が通ったうえで `importance(kind="shap")` が `DATA_SCHEMA_INVALID`、`"mode"` では報告なしに置換する。利用者が `"error"` を選んだ以上、変換する場所で拒否するのは方針どおりであり、これを仕様として固定する（テスト: `test_shap_importance_applies_the_stored_policy_outside_the_last_fold`）。最後の fold の pipeline を全行に掛けること自体は本 PR 以前からの性質で、[#303](https://github.com/nbx-liz/LizyML/issues/303) に切り出した。（H-0114 注記: この SHAP の節は H-0114 で置き換えた。SHAP 重要度は fold k の検証行を fold k の pipeline 状態で変換し、`error` で `fit` が通ったモデルの SHAP 重要度は送出しない。固定テストは `test_shap_importance_uses_each_fold_pipeline_outside_the_last_fold` に書き換えた。）**外れる保証**: `"mode"` / `"nan"` で CV の検証 fold や SHAP 重要度の対象行に未知カテゴリがあると、OOF 指標と SHAP 重要度はその置換を経た値で計算され、利用者には知らされない（決定 7 の条件の列に限る）。
9. **生成 `predict.py` も置換をログに出す。** 生成コードは既に 3 方針を状態から再現している（#205）。報告の規則の位置として、余剰列と同じく `log.warning` を足す。**あわせて欠損値を未知カテゴリとして扱わないようにする**（実装中に発見）: 生成コードは `astype(str)` で欠損を文字列 `"nan"` にしてから対応表を引くため、欠損が未知と区別されず、`"mode"` では最頻値に置き換わり、`"error"` では拒否されていた。実行時の encoder は欠損を欠損のまま残すので、方針は「値があり、対応表に無い」行だけに掛ける。
10. **生成 `train.py` は方針を保ったまま pipeline 状態を書き直す**（設計レビュー round 1 の blocking 2）。生成 `train.py` の `fit_pipeline` は `pipeline_state.json` を作り直すが、`unseen_policy` も `unseen_codes` も書いていなかったので、再学習後の `predict.py` は既定の `"nan"` に落ちていた（エクスポート時の `"mode"` / `"error"` が失われる）。`config.json` に fit が適用した方針を載せ、`fit_pipeline` はそれと再学習データの最頻値のコードを状態に書く。**生成コードの範囲（コードレビュー round 3 の広範な照合で確定）**: 列の型 11 種 × {既知・未知・欠損・同値} × 3 方針の全 132 通りを実行時と照合し、本 PR の主張の内側の食い違い（float32 の `category` 列でエクスポート時に最頻値コードが落ちる）は修正した。**外れる保証**（本 PR 以前からの性質、[#304](https://github.com/nbx-liz/LizyML/issues/304)）: 生成コードはカテゴリを `str()` で引くので `1` と `"1"` が 1 つにまとまり、再学習は観測値だけから対応表を作るので宣言済みで未使用のカテゴリが未知になる。

### 規則が縛る位置（ソースから導出、実装前）

規則 A: **設定された `unseen_policy` が、データを変換するすべての場所で効く。** 規則 B: **推論時に適用した補正は呼び出し側に報告する。** 導出: `lizyml/` を `transform_with_warnings` / `pipeline.transform(` / `.load_state(` / `build_pipeline_factory` で grep し、`lizyml/codegen/` を `unseen` で grep した（`2d3bc54`）。

| # | 位置 | 規則 A | 規則 B | 本 PR |
|---|---|---|---|---|
| 1 | `core/model.py` fit の `pipeline_factory`（CV と refit が共有） | 今は既定の `"mode"` 固定 | — （fit に警告の通り道なし、決定 8） | config の値を渡す |
| 2 | `core/_model_tuning.py` の CV | 同上 | — | config の値を渡す |
| 3 | `core/_model_predict.py` `run_predict` | 保存状態から復元（適合） | 置換を報告しない | **報告する**（決定 4）、列検査を facade へ（決定 3） |
| 4 | `core/_model_tables.py` → `explain/shap_explainer.py`（SHAP 重要度） | 保存状態（最後の CV fold の pipeline）から復元。方針は適用される | 報告の通り道なし。スライディング窓では最後の fold に属さない行の値が未知になりうる（初版の「新しい未知カテゴリは生じない」は誤り、設計レビューが反証） | 変更なし、決定 8 として仕様化しテストで固定 |
| 5 | 生成 `predict.py` の `transform` | 状態から 3 方針を再現（#205）。ただし欠損を未知として扱っていた | 余剰列はログ、置換はログなし | **置換をログに出し、欠損は欠損のまま**（決定 9） |
| 6 | 生成 `train.py` の `fit_pipeline`（`pipeline_state.json` を書き直す） | 方針と最頻値コードを書かず、再学習後の予測が `"nan"` に落ちる | — | **方針と最頻値コードを書く**（決定 10）。初版は「対象外」としていた（設計レビューが反証） |

### 互換性

- 既定は `"mode"` のまま。既存の config・artifact の予測値は変わらない。
- **`PredictionResult.warnings` の内容が変わる**: 未知カテゴリがあると警告が入る（形は変わらない）。警告が空であることに依存する呼び出し側は影響を受ける。
- `FeaturesConfig` に任意のキーが 1 つ増える（`extra="forbid"` なので、今は `unseen_policy` を書くと `CONFIG_INVALID`）。`config_version` は据え置き（任意キーの追加）。
- **公開 Protocol の変更**: `EstimatorProvider.build_pipeline_factory` に既定値つきのキーワード引数が増える。facade は fit / tune でこの引数を渡すので、引数を受け取らない provider 実装は fit で失敗する。本体の provider は `LGBMProvider` だけで、公開の登録手段は無い。
- `BaseFeaturePipeline` に具体メソッドが 1 つ増える（既存サブクラスはそのまま動く）。
- `format_version` は据え置き。pipeline 状態は既に `unseen_policy` を保存している。
- **生成コード**: 新しくエクスポートした `config.json` に `unseen_policy` が増える。生成 `predict.py` は欠損値を未知として扱わなくなる（`"mode"` で欠損が最頻値に置き換わっていた・`"error"` で欠損が拒否されていた挙動が、実行時と同じく欠損のままになる）。既にエクスポート済みのファイルは変わらない。
- **`"error"` を選んだ場合**、決定 7 の条件の列では `fit` が、スライディング窓ではさらに SHAP 重要度が `DATA_SCHEMA_INVALID` で止まりうる（既定の `"mode"` では起きない）。
- Firing rate: 本 Proposal は skip / shorten / cache / select / allow / conditionally-activate の条件を新設しない。警告は置換が起きたときに必ず出る報告であり、`unseen_policy` は既存の encoder の分岐を Config から選べるようにするだけである。

### 代替案（検討して棄却）

1. **`transform_with_warnings` を抽象にする。** 定義時に要求が分かるが、既存の外部サブクラスを壊す。
2. **列検査を基底クラスの具体メソッドに置く（計画の文言どおり）。** サブクラスが上書きすれば検査が消え、計画が挙げた理由を満たさない。fit 時の列は pipeline ではなく `FitResult` / `RefitResult` にあるので、facade の方が情報も揃っている。
3. **`"error"` を推論時だけに効かせる（CV では `"mode"`）。** 同じ方針が fit と predict で違う意味を持つことになり、検証 fold を推論の代理とする OOF の前提とずれる。
4. **fit 中の置換も報告する。** `FitResult` に警告の通り道を足す設計判断が要り、#260 の DoD を超える。

### 受け入れ基準（テスト観点）

`docs/audits/2026-09-defect-discovery/results/pr5_acceptance_criteria.md` の対応表を正とする。要旨:

1. 4 つの抽象メソッドだけを実装した pipeline（provider の factory に差し込む。公開の登録手段は無いので差し込みで代える）が、`fit` → `predict` → SHAP 重要度 / `predict(return_shap=True)` を通る。修正前は `predict` で `AttributeError`。`categorical_cols` を持たない状態でも学習できる。
2. facade の列検査: 自作 pipeline（自分では列を検査しない）でも、不足列は `DATA_SCHEMA_INVALID`、余剰列は警告 1 件（`NativeFeaturePipeline` でも 1 件で、2 件にならない）。
3. `UnseenPolicy` の全値（型から読む）を Config から指定して、推論時の観測結果がそれぞれ: `"mode"` = 警告 + 最頻値と同じ予測、`"nan"` = 警告 + 欠損と同じ予測（最頻値に置換した予測と欠損にした予測が異なる、識別できるデータで）、`"error"` = `DATA_SCHEMA_INVALID`。fit と tune が作る pipeline に指定した方針が載る。自作 pipeline が自分で出した警告は変えずに届く。Config の値の集合と `UnseenPolicy` が一致する。修正前は Config が `CONFIG_INVALID`、既定で警告が空。
4. 指定した方針が refit の pipeline 状態に載り、`Model.load()` 後の `predict` でもその方針と警告が保たれる。
5. `"error"` で検証 fold にだけ現れる値があると、`auto_categorical: false` では `fit` が `DATA_SCHEMA_INVALID`、`auto_categorical: true` では通る（決定 7 の固定）。スライディング窓の SHAP 重要度は `"error"` で `DATA_SCHEMA_INVALID`、`"mode"` で通る（決定 8 の固定）。
6. 生成コード: `config.json` と `pipeline_state.json` に方針が載る。`predict.py` は `"mode"` / `"nan"` の置換でログを出し、欠損は欠損のまま（`"error"` でも拒否しない）。`train.py` で再学習しても方針と最頻値コードが保たれる。
7. `BLUEPRINT.md` §5.4 / §9.2、`docs/config-reference.md`、`ARCHITECTURE.md` が基底クラスと一致する。

## H-0105: LightGBM の feval に渡る確率を変換し直さない（#306 / PR 5b）

- **ステータス**: Accepted
- **起票日**: 2026-10-01
- **決定日**: 2026-10-01（管理者の指示で事実確認を含む外部レビューを 3 ラウンド実施。round 1・2 で文書の事実誤認を計 6 件訂正し、テストの母集団を受理される全目的関数に拡大。round 3 で APPROVE）
- **スコープ**: `lizyml/estimators/lgbm/metric_bridge.py`（`_build_feval` の `feval_fn`、未使用になった `_sigmoid` / `_softmax` を削除）, `lizyml/codegen/templates.py`（生成 `train.py` の feval、未使用になった `_softmax` を削除）, `tests/test_estimators/test_feval_probabilities.py`（新規）, `tests/test_estimators/test_lgbm_metric_bridge.py` / `tests/test_codegen/test_feval_codegen.py`（誤った前提で書かれたテストの書き直し）
- **関連**: [Issue #306](https://github.com/nbx-liz/LizyML/issues/306), H-0064（feval の導入）, H-0066（生成コードの feval）

### 目的（課題）

LightGBM 4 は、組み込みの目的関数では **feval に変換済みの予測を渡す**: binary は確率（1 次元）、multiclass と multiclassova は確率（2 次元 `(n, num_class)`）、回帰は値。`metric_bridge._build_feval` はこれを生のスコアと仮定し、binary に `sigmoid`、multiclass に `reshape` + `softmax` を**もう一度**掛けていた。docstring もその仮定を書いていた。

実測（`develop` `fc9d820`、lightgbm 4.6.0）:

- feval が受け取る値（300 行、20 ラウンドの `lgb.train`）: binary は 0.058〜0.927 で形 `(300,)`、multiclass と multiclassova は 2 次元 `(300, 3)`。`cross_entropy` も確率（1 次元）。**例外は `cross_entropy_lambda`** で、出力は 1 を超える（400 行・15 ラウンドで 6 回目から 1 超、最大 2.03）。これは確率ではなく、LizyML 全体の扱いの問題として [#307](https://github.com/nbx-liz/LizyML/issues/307) に切り出した（feval とは無関係に `Model.fit` が評価器で失敗する）。LightGBM 4.0 の文書も、multiclass の feval 入力を 2 次元と定めている。
- binary のラベル指標（`accuracy` / `f1`）: `sigmoid(p) >= 0.5` がすべての `p >= 0` で成り立つので、全行が陽性になり値が一定（400 行・40 ラウンドの一例で `accuracy` は陽性率のまま。実際の accuracy は 0.63 前後から 0.93 前後へ上がる。数値はデータに依存する）。LizyML の既定値は `first_metric_only` を `False` に設定し（`estimators/lgbm/defaults.py`）、adapter の early stopping コールバックも引数を省いて既定の `False` を使うので、すべての指標が監視され、一定の指標は改善しない。**その結果、待機回数（patience）を使い切った時点で 1 回目が最良として選ばれ、最終的なモデルは木 1 本になる**（`Model.fit`、binary、2000 行、3 fold、patience 20: best_iteration 1/1/1、OOF AUC 0.7645。`binary_logloss` なら 93/80/104、0.8672。設計レビューが同じ設定で全行を再現した。素の `lgb.train` で patience 20 なら 21 ラウンド評価して 1 回目を選ぶことも、レビューが確かめた）。
- 確率の指標（binary の `brier` / `ece`、multiclass の `brier`）: 二重変換した確率で計算され、early stopping の位置と学習曲線がずれる（binary brier は報告 0.2123 対 実際 0.1087、multiclass brier は 0.1649 対 0.1043。`brier` を指定した fit の best_iteration は 180/100/201）。`logloss` は LightGBM ネイティブの指標に変換されるので feval を通らず、影響しない（#306 の本文にある logloss の数値は、橋渡しを直接呼んだ合成の例である）。
- 順位だけで決まる指標（binary の `precision_at_k`、multiclass の argmax による `accuracy` / `f1`）は、変換が単調なので値が変わらなかった。回帰は変換していないので影響なし。

既存のテスト（`test_lgbm_metric_bridge.py`）は、feval に人工の logit を渡し、期待値も同じ変換で計算していた。**宣言（生のスコアが来る）と同じ前提で書いたフィクスチャ**なので、誤りを検出できなかった（DC7）。

### 対応方針（決定）

1. **feval は LightGBM が渡した値を変換し直さない。** 指標に渡す値は評価器と同じ規則（`evaluation.evaluator._pred_for_metric`）で作る: ラベルの指標には閾値 0.5（binary）または argmax（multiclass）、確率の分布を必要とする multiclass の指標（`needs_simplex`）には行で正規化した確率（multiclassova の行は合計が 1 にならない）、それ以外は渡された確率そのまま。これで同じ指標名が学習曲線と `FitResult.metrics` で同じ意味になる。
2. **multiclass で 2 次元でない値が来たら、推測で並べ替えずに `EVALUATION_FAILED` で止める。** 依存の下限は `lightgbm>=4.0` であり、4.6.0 で 2 次元であることは実測し、LightGBM 4.0 の文書も 2 次元と定めている。**4.0.0 を実行しての確認はしていない**。最低依存の CI レーンは `--frozen` で動き下限を入れないので（#295）、CI でも測られていない。下限の版が 1 次元を渡すなら、この検査が `EVALUATION_FAILED` として止める（黙って誤った並べ替えをするより良い）。#295 が下限を実際に入れるようになった時点で測られる。
3. **生成 `train.py` の feval も同じく変換しない**（同じ欠陥があった）。multiclass で 2 次元でなければ `ValueError`。生成 `config.json` の feval 情報は `needs_simplex` を持たないが、multiclass で feval に回る指標（`f1` / `brier` / `accuracy`）はどれも `needs_simplex` が偽なので、確率をそのまま渡せば実行時と一致する。
4. **誤った前提のテストは削除せず書き直す。** 元のデータ（logit）を、LightGBM が渡す確率に変換してから feval に渡し、期待値はその確率から指標を直接計算する。生成コードの「`_softmax` を含む」というテストは「feval が変換し直さない」というテストに置き換える。
5. **恒久検査**: feval に回る全指標（`_FEVAL_METRICS` から読む）× LizyML が受理するすべての目的関数（`TASK_COMPATIBLE_OBJECTIVES` から読む。設計レビューが、手書きの 4 種類では `cross_entropy` / `cross_entropy_lambda` が抜けていると指摘した）で、本物の `lgb.train` を回し、feval の値が「LightGBM が渡した値に評価器の規則を当てた指標値」と一致すること、**評価器の規則自体が失敗する場合（`cross_entropy_lambda` の 1 超の出力を確率の指標に渡す等）は feval も同じ型で失敗する**ことを主張する。学習曲線が、`FitResult.metrics` では計算できない数値を報告しないためである。期待値の側は手書きのフィクスチャではなく、LightGBM の実際の出力から作る。

### 規則が縛る位置（ソースから導出）

規則: **LightGBM が feval に渡す予測は、その目的関数の出力として扱い、変換し直さない。** 導出: `lizyml/` を `feval` / `_sigmoid` / `_softmax` で grep した（`fc9d820`）。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `estimators/lgbm/metric_bridge.py` `_build_feval` | 修正 |
| 2 | `codegen/templates.py` 生成 `train.py` の `build_feval_from_config` | 修正 |
| 3 | `codegen/templates.py` の `_sigmoid`（生成コードの較正: beta / platt の生スコア） | 対象外（生スコアに掛ける正しい用途） |

### 互換性

- **`model.params.metric` に feval に回る分類の指標（binary: `accuracy` / `f1` / `brier` / `ece`、multiclass: `brier`）を含み early stopping を使う config は、学習結果が変わりうる。** early stopping が選ぶ反復が正しい値に基づくようになるため（値が変わっても、選ばれる反復が変わらない場合もある）。特に `accuracy` / `f1` は 1 回目が選ばれていたものが学習するようになる。学習曲線（`history`）のこれらの値は、early stopping の有無によらず正しくなる。
- `precision_at_k`、multiclass の `accuracy` / `f1`、回帰の feval 指標は値が変わらない。`logloss` / `auc` は feval を通らない（LightGBM ネイティブの指標に変換される。タスクで使えない組み合わせは検証で拒否される）。
- `cross_entropy_lambda` で確率の指標を feval に指定した場合、以前は sigmoid で [0, 1] に押し込まれた値で黙って計算していたが、今は評価器と同じ値を渡すので、評価器と同じ振る舞いになる: 1 を超える値を拒否する `brier` は失敗し、`ece` と `precision_at_k` は 1 を超える値のまま数値を返す（設計レビューが実測: 最大 2.03 の出力で ece 0.0925、precision_at_k 1.0）。この目的関数はそもそも既定の指標でも `Model.fit` が失敗する（#307）。
- 保存済み artifact はそのまま読め、predict も変わらない（保存された booster を使う）。`format_version` は据え置き。
- 生成コード: 新しくエクスポートした `train.py` の feval が正しくなる。既にエクスポートしたファイルは変わらない。
- Firing rate: 本 Proposal は skip / shorten / cache / select / allow / conditionally-activate の条件を新設しない。multiclass の形の検査は、LightGBM 4 の実際の出力では発火しない防御である。

### 代替案（検討して棄却）

1. **生のスコアを受け取るよう LightGBM の設定を変える（`raw_score` 等）。** feval の入力形式は目的関数で決まり、利用者の指定する目的関数ごとに分岐が要る。確率をそのまま使う方が単純で、評価器と同じ値になる。
2. **multiclass で 1 次元が来たら古い形式とみなして並べ替える。** 依存の下限は 4.0 で、旧形式（クラスごとに連結）を推測で並べ替えると誤りが黙って通る。止める方を選んだ。

### 受け入れ基準（テスト観点）

1. feval に回る全指標 × 受理されるすべての目的関数（`TASK_COMPATIBLE_OBJECTIVES`）で、本物の `lgb.train` の各ラウンドの feval の値が、LightGBM が渡した値に評価器の規則を当てた指標値と一致する。評価器の規則が失敗するセルでは feval も同じ型で失敗する。修正前は binary の `accuracy` / `f1` / `brier` / `ece` と multiclass / multiclassova の `brier` で RED。
2. `Model.fit`（binary、`metric: accuracy` / `f1`、early stopping）で、どの fold の best_iteration も 1 より大きく、学習曲線の値が変化する。修正前は RED。
3. 生成 `train.py` の feval が、確率を入力として指標を直接計算した値と一致する（binary `accuracy` / `brier`、multiclass `brier`）。修正前は RED。
4. 書き直した既存テストのうち、**値が変わる指標**（binary の `accuracy` / `f1` / `brier` / `ece`、multiclass の `brier`）を扱うものは修正前の実装で失敗し、修正後に通る。**値が変わらない指標**（`precision_at_k`、multiclass の `accuracy` / `f1`、回帰）を扱うものは修正の前後どちらでも通る不変条件として残す（初版はすべてが失敗すると書いていたが誤りで、設計レビューが反証した）。

## H-0106: 宣言されたすべての `ErrorCode` を発生させ、`config_version` の検査をすべての入口に置く（#263 / #272 / PR 6）

- **ステータス**: Accepted
- **起票日**: 2026-10-01
- **決定日**: 2026-10-01（事実確認を含む外部レビュー: 設計レビュー 3 ラウンド（blocking 7 → 1 → 0。文書の事実誤認を計 8 件訂正し、計測器を直した）、コードレビュー 2 ラウンド（blocking 6 → 0。旧 loader と異なっていた版の変換を戻し、`"2.0"` の迂回を発見、空振りしうる 2 つのテストを強化）。各ラウンドの前に absolute / relational monitor。RED テストの段で #309 を発見）
- **スコープ**: `lizyml/core/exceptions.py`（`DATA_FINGERPRINT_MISMATCH` を削除）, `lizyml/features/column_check.py`（dtype の検査 → `INCOMPATIBLE_COLUMNS`）, `lizyml/core/_model_predict.py`（呼び出し）, `lizyml/metrics/classification.py`（確率の検査 → `METRIC_REQUIRES_PROBA`）, `lizyml/config/version.py`（新規: 版の定義と検査を 1 か所に）, `lizyml/config/schema.py` / `lizyml/config/loader.py` / `lizyml/core/model.py`（版の検査を呼ぶ）, `BLUEPRINT.md` §4 の Config 表 / §16.2, `docs/api.md`, `docs/DEPRECATIONS.md`, `PLAN.md`, `CHANGELOG.md`, テスト（下記）
- **関連**: [Issue #263](https://github.com/nbx-liz/LizyML/issues/263), [Issue #272](https://github.com/nbx-liz/LizyML/issues/272), [Issue #307](https://github.com/nbx-liz/LizyML/issues/307)（`cross_entropy_lambda`）, [Issue #271](https://github.com/nbx-liz/LizyML/issues/271)（`SUPPORTED_CONFIG_VERSIONS` は未文書の 8 名の 1 つ）, [Issue #295](https://github.com/nbx-liz/LizyML/issues/295)（依存の下限が CI で実行されない）, H-0104（推論時の列検査を facade に置いた）, H-0105, #210（文字列の版の迂回を閉じた）
- **実測の記録**: `docs/audits/2026-09-defect-discovery/results/pr6_measurements.txt`（`develop` `1abf7fb`。再生成するスクリプトは同じディレクトリの `../instruments/pr6_*.py`）

### 目的（課題）

**#263**: `ErrorCode` の 20 メンバーのうち 3 つ（`DATA_FINGERPRINT_MISMATCH` / `INCOMPATIBLE_COLUMNS` / `METRIC_REQUIRES_PROBA`）は、`lizyml/` のどの `raise` も出さない（`ast.Raise` の部分木を走査して確認）。エラーコードは「その条件を検出する」という主張であり、文書化されて出ないコードは、検査が無いのに有ると書いていることになる（DC4 / DC5）。

**#272**: `BLUEPRINT.md` の Config 表は `config_version` を「`1` のみサポート」とするが、検査は `config/loader.py` の `_check_config_version` にしかなく、`load_config` を通らない入口では効かない。実測（`pr6_config_version_probe.py` / `pr6_config_version_env_probe.py`、pydantic 2.12.5）:

| 入口 | `config_version=2` | `config_version=False` |
|---|---|---|
| `load_config(dict)` / `Model(dict)` | `CONFIG_VERSION_UNSUPPORTED` | **受理され `0` として保存** |
| `LizyMLConfig.model_validate(dict)` → `Model(instance)` | **受理** | **受理（`0`）** |
| `LizyMLConfig.model_construct(...)` / 検証済みインスタンスへの代入 / `model_copy(update=...)` → `Model(instance)` | **受理** | **受理（検証を通らないので `bool` の `False` のまま保持）** |
| `load_config(dict v=1)` + 環境変数 `LIZYML__config_version=2` | **受理（`2`）** | `"false"` で **受理（`0`）** |

#272 が書いた入口（インスタンス）に加え、**dict の経路にも 2 つの迂回がある**: loader の検査は `bool` を pydantic に任せて素通しし（`False` は `int` の `0` に変換される）、環境変数の上書き（`loader.py:244`）は版の検査（`loader.py:240`）の**後**に適用される。

### 対応方針（決定）

1. **`DATA_FINGERPRINT_MISMATCH` を削除する。** 予測時に照合できる、冗長でない条件が存在しない（計画 `phase3-plan.md` §PR 6 で実測済み）: `row_count` は正当な予測バッチで異なりうる（学習時と同じ行数である理由が無い）。`file_hash` は `Model.fit(df)` では常に `None`（`fp_compute(X, file_path=None)`）で、`predict` はファイルを受け取らない。`column_hash` は列順に依存し、列順を入れ替えた frame は今日正しく予測できる。意味のある列のずれ（不足 → `DATA_SCHEMA_INVALID`、余剰 → 警告、dtype → 決定 2）は別の検査が報告する。`DataFingerprint` 型とその記録は変えない（来歴の記録として残す。予測時に照合しているとは主張しない）。`RESERVED` のような保留表は作らない（計画 round 6: 保留は処置ではなく受け入れ規則の変更であり DC5）。
2. **`INCOMPATIBLE_COLUMNS`: 学習時に数値だった列が、予測時に数値でない dtype で届いたら拒否する。**
   - **「数値」の定義（受理集合を閉じる）**: dtype の scalar 型が numpy の整数・浮動小数・bool 型のサブクラスで、`timedelta64` と `longdouble` を除くもの。LightGBM 4 が pandas の列に課す規則と同じ集合である。pandas の `is_numeric_dtype`（と `is_bool_dtype`、complex を除く）は使わない: pyarrow の数値 dtype 3 種（`int64[pyarrow]` / `double[pyarrow]` / `bool[pyarrow]`）と `longdouble`（`float128`）を数値と判定するが LightGBM はそれを拒否するので、33 種の dtype で 4 件食い違った（設計レビュー round 1 が計数を訂正。初版は `longdouble` を数えず 3 件と書いた）。numpy 型の規則では **33 種すべてで「規則が受理 ⇔ 本物の `predict` が成功」**（0 件の不一致、`pr6_dtype_rule_probe.py`）。
   - **「学習時に数値」**: `FitResult.dtypes`（`str(X[col].dtype)`、`auto_categorical` の変換**後**に記録）を `pandas.api.types.pandas_dtype` で読み戻し、同じ規則で判定する。学習時に `category` だった列（`str` / `object` / `string` も既定の `auto_categorical` で `category` になる）は検査しない: encoder が予測時の値を扱い、未知値は `unseen_policy` に従う（H-0104）。**ただし encoder がすべての dtype を扱えるわけではない**: 整数の category に `float16` / `longdouble` が届くと、encoder の中で pandas の生の例外になる（RED テストを書く段で発見。#309 に切り出し、該当 4 セルは strict な xfail で固定）。この規則の範囲外（数値で学習した列ではない）なので本 PR では直さない。
   - **読み戻せない dtype 文字列は、明示的な免除とする**: その列は検査せず、予測は今日と同じに進む。下流が失敗するとは限らない（LightGBM が検査するのは到着したデータで、記録された文字列ではない。記録を読み戻せない文字列に差し替えても数値の予測は成功した、設計レビュー round 1）。初版は「下流の LightGBM が生の例外を出す」と書いたが、それは偽だった。免除する理由: 予測時の dtype から学習時の型を推測すると誤った拒否を生み、記録の無い列を拒否すると古い artifact の正しい予測を壊す。**bound**: fit できる 23 種の dtype はすべて読み戻せる文字列を記録した（fit できない 10 種は LightGBM が fit で拒否する、`pr6_dtype_rule_probe.py`）。恒久テストがこの 23 種の読み戻しを主張し、免除の振る舞い（読み戻せない記録の列は拒否されず予測が進む）も別のテストで固定する。pandas が綴りを変えて読み戻せなくなれば、黙って検査が外れるのではなく前者が赤になる。
   - **置き場所**: H-0104 の `select_training_columns` に省略可能な引数 `dtypes` を足し、`run_predict` が `fit_result.dtypes` を渡す。不足列の `DATA_SCHEMA_INVALID` を先に判定し、そのあと dtype を判定する。違反した列は 1 回の例外ですべて報告する（学習時の列順）。`context = {"columns": [{"column", "fit_dtype", "predict_dtype"}, ...]}`。
   - **今日の振る舞いとの差**: 拒否する入力はすべて今日も失敗している（LightGBM の `pandas dtypes must be int, float or bool`、`category` で届くと `train and valid dataset categorical_feature do not match`、`datetime` / `timedelta` は numpy の `DTypePromotionError`）。`NativeFeaturePipeline`（`get_provider` が返す唯一の pipeline）では、成功している予測は 1 件も変わらない。変わるのは、列名と両方の dtype を持つ LizyML の例外になること。**ただし自作 pipeline では、成功している予測を拒否しうる**: 予測時に文字列を数値へ変換する自作 pipeline は、数値で学習した列に数値の文字列が届いても今日は同じ予測を返すが、facade の検査は変換の前に拒否する（設計レビュー round 1 が実行で確認）。**これは意図した制約である**: 学習時に数値だった列は、どの pipeline でも予測時に数値で届くことを入力の契約とする。公開された手段で作れる構成は影響を受けない（リポジトリの範囲で確かめた。リポジトリの外の、私的な内部への差し込みは確かめられない）: 自作 pipeline を登録する公開の手段が無く（`core/_model_factories.get_provider` は `lgbm` だけを返し、pipeline は provider が作る）、4 メソッドの自作 pipeline が推論経路を通るようにした H-0104 自体が未リリースである。
3. **`METRIC_REQUIRES_PROBA`: 確率を必要とする組み込み指標（`needs_proba` が真）が、確率でない値を受け取ったら拒否する。**
   - **計画との違い**: 計画 `phase3-plan.md` §PR 6 は「確率を持たないタスクに `needs_proba` の指標が求められたとき、指標の dispatch で出す」としていた。**その条件は `UNSUPPORTED_METRIC` が先に拒否するので到達できない**（`metrics/registry.py` の `_TASK_METRICS`: 回帰に確率の指標は無い）。到達できる条件は、指標クラスに確率でない値が渡ることである。計画の該当段落とファイル一覧は本 PR の最初のコミットで直す。
   - **規則**: 確率とは、数値に変換でき、すべて有限で、[0, 1] に収まる値。加えて、`y_true` のクラスが 3 つ以上なのに `y_pred` が 1 次元なら拒否する（多クラスの確率は 2 次元）。`context = {"metric", "reason", ...}`。**検出できないもの**: 0/1 のハードラベルは正当な確率でもあるので区別できない。
   - **順位だけで決まる指標（`auc` / `auc_pr` / `precision_at_k`）も同じ規則で拒否する**（代替案 2 を参照）。
   - **到達する経路**: (a) 公開の `lizyml.metrics` のクラスを直接呼ぶ利用者。今日、binary の logit を渡すと `auc` / `auc_pr` / `ece` / `precision_at_k` は黙って計算し、`brier` / `logloss` は scikit-learn の生の `ValueError`。多クラスの `y_true` に 1 次元のラベルを渡すと、`logloss` / `auc` / `auc_pr` / `brier` は生の `ValueError`、binary 専用の `ece` は `0.0`、`precision_at_k` は `2.0` を黙って返す（`y_true = y_pred = [0,1,2,0,1,2]`、設計レビュー round 1。初版は全指標が `ValueError` と書いたが偽だった。`ece` / `precision_at_k` は multiclass の指標の登録には無く、`Model` からは届かない）。(b) `objective: cross_entropy_lambda`（#307）: 出力が確率ではない。
   - **対象外**: 利用者が自作した `BaseMetric` のサブクラスは検査されない（検査は組み込みの 6 クラスの `__call__` に置く。抽象基底の契約は変えない）。
4. **`config_version`: 版の定義と検査を `lizyml/config/version.py` の 1 か所にし、`LizyMLConfig` ができるすべての入口から呼ぶ。**
   - `SUPPORTED_CONFIG_VERSIONS` と `check_config_version(value)` をこのモジュールに置く（`schema.py` は `loader.py` を import できないため）。`lizyml.config.loader.SUPPORTED_CONFIG_VERSIONS` は同じオブジェクトの再 export として残す（公開名を壊さない。同一性をテストする）。型は今と同じ `list[int]`。
   - **schema**: `LizyMLConfig.config_version` に field validator（after）を置き、検査を呼ぶ。`model_validate` / コンストラクタ / `load_config` の検証段（環境変数の上書きの後）を通るすべてが共有する。pydantic 2.12.5 では validator が投げた `LizyMLError` は `ValidationError` に包まれずにそのまま伝わる（実測）。依存の下限 `pydantic>=2.0` では実行していない（#295: 最低依存の CI レーンは下限を入れない）。下限の版が包むなら `load_config` は `CONFIG_INVALID` として報告することになり、#295 が下限を入れるようになった時点で測られる。
   - **loader**: 生の値での検査（`loader.py:240`）は先頭に残す。loader が拒否する値について、利用者が書いた綴り（`"2"` / `"2.0"` / `2.5` 等）を `context["config_version"]` に保つため（loader が版と読めず pydantic が変換した値を schema の検査が拒否する場合は、変換後の値になる）。検査関数は schema と同じものを呼ぶ。
   - **`Model.__init__` のインスタンスの分岐**: 検証を通らないインスタンス（`model_construct`、代入、`model_copy(update=...)`）のため、`check_config_version(config.config_version)` を 1 行で呼ぶ（facade に判断を書かない。CLAUDE.md §3）。
   - **`bool`**: 検査関数は値を `int()` で変換してから判定する（今日の loader の `bool` を素通しする早期 return は削除する）。だから検証を通る経路では `False` → `0`、検証を通らない経路では `bool` の `False` のまま届いても、どちらも `int(False) == 0` で拒否される。`True` は全経路で `1` として受理される（検証を通る経路では pydantic の lax な `int` 変換と同じ、他のすべての `int` フィールドと同じ）。`int()` に変換できない値は、今日と同じく pydantic の型検査に任せる（`CONFIG_INVALID`）。インスタンスの分岐で変換できない値が来た場合は `CONFIG_VERSION_UNSUPPORTED` とする（pydantic に任せる段が無いため）。**変換の細則（コードレビュー round 1 で旧 loader と突き合わせて決めた）**: loader では小数の float を今日と同じく切り捨てる（`2.5` / `-1.5` / `0.5` は `CONFIG_VERSION_UNSUPPORTED`、`1.5` は 1 として通り pydantic が `CONFIG_INVALID`。初版の実装はこれを `CONFIG_INVALID` に変えていた）。インスタンスの分岐では切り捨てず、整数でない float は拒否する（`1.5` を版 1 として受理しないため）。文字列は `int()` で読めなければ整数値の float として読む: **今日 `"2.0"` は loader の検査を通り、pydantic が 2 に変換して版 2 として受理されていた**（dict の経路の 3 つ目の迂回。コードレビュー round 1 が発見）。本 PR の後は loader が `"2.0"` のまま拒否する。`inf` は今日 loader の `int()` で生の `OverflowError` になっていたが、本 PR の後は版ではない値として pydantic の `CONFIG_INVALID` になる。
   - **対象外**: `Model` に渡した**後**に呼び出し側がそのインスタンスを書き換えることは防がない（`Model` は渡されたオブジェクトを保持する。コピーは本 PR の範囲外）。
   - `BLUEPRINT.md` の Config 表の `config_version` 行に、定義の場所（`lizyml.config.loader.SUPPORTED_CONFIG_VERSIONS`）を書く。#271 の 8 名のうち 1 名を処理する（#271 は閉じない）。
5. **恒久検査**:
   - `test_error_code_population.py`（静的）: `lizyml/` の `ast.Raise` の部分木に現れる `ErrorCode.X` を集め、`set(ErrorCode)` と一致すること。コメントや docstring の言及は数えない。自己検査として、走査が 10 メンバー以上を見つけることを主張する（誤った作業ディレクトリで走査対象が空になり、全 20 メンバーを「未発生」と報告した事例が計画の検証で起きた）。
   - `test_error_code_raising.py`（振る舞い）: キーが `set(ErrorCode)` と等しい dict で、各メンバーについて条件を作り、出る `code` と、その raise 箇所が渡す `context` のキーを主張する。`if False: raise ...` のような到達しない raise は静的検査を通るが、こちらで落ちる。
   - 文書の完全性: `docs/api.md` の例外コード表と `BLUEPRINT.md` §16.2 の一覧が `set(ErrorCode)` と一致すること（DC3）。今日 `docs/api.md` には `METRIC_REQUIRES_PROBA` / `TARGET_NOT_NUMERIC` / `TARGET_UNSEEN_LABEL` が無く、§16.2 は「例」と題して `EVALUATION_FAILED` / `CALIBRATION_NOT_FITTED` / `TARGET_NOT_NUMERIC` / `TARGET_UNSEEN_LABEL` を欠く。本 PR で両方を完全にし、§16.2 の題から「例」を外す。

### 規則が縛る位置（ソースから導出）

**規則 A（dtype）: Model facade が新しいデータを pipeline に渡す前に、学習時に数値だった列の dtype を検査する。** 導出: H-0104 と同じ grep（`transform_with_warnings` / `.load_state(` / `pipeline.transform(`、`1abf7fb`）。

| # | 位置 | 本 PR |
|---|---|---|
| A1 | `core/_model_predict.py` `run_predict`（`Model.predict` の唯一の入口） | 検査を呼ぶ |
| A2 | `explain/shap_explainer.py:161`（学習時のデータを変換する） | 対象外（新しいデータではない） |
| A3 | `features/pipelines_native.py:106`（`NativeFeaturePipeline` を単体で使う場合） | 対象外（pipeline の状態は学習時の dtype を持たない。facade の経路は A1 が先に検査する） |

**規則 B（確率）: `needs_proba` が真の組み込み指標は、確率でない値を拒否する。** 導出: 登録されたすべての指標を構築して `needs_proba` を読んだ（16 指標中 6）。

| # | 位置 | 本 PR |
|---|---|---|
| B1-B6 | `metrics/classification.py` の `LogLoss` / `AUC` / `AUCPR` / `Brier` / `ECE` / `PrecisionAtK` の `__call__` | 先頭で検査 |
| B7 | 呼び出し側: `evaluation/evaluator.py` `_compute_metrics`、`estimators/lgbm/metric_bridge.py:278`（feval） | 変更なし（例外を包まないので、同じ例外がそのまま伝わる） |
| B8 | 呼び出し側: tuning（`tuning/tuner.py` の `study.optimize(..., catch=(LizyMLError, ValueError, RuntimeError))`） | 変更なし（その trial が失敗として記録され、全 trial が失敗すれば `TUNING_FAILED`。今日の scikit-learn の `ValueError` と同じ扱い） |

**規則 C（版）: `LizyMLConfig` が `Model` に入るすべての経路で `config_version` を検査する。** 導出: `lizyml/` を `LizyMLConfig(` / `model_validate` / `model_construct` / `load_config(` で grep した。`Model.load()` は保存された dict を `Model(config)` に渡すので C2 を通る。

| # | 位置 | 本 PR |
|---|---|---|
| C1 | `config/schema.py` `LizyMLConfig.config_version` の field validator | 新設 |
| C2 | `config/loader.py:240` `load_config`（生の値） | 共通の検査関数を呼ぶ |
| C3 | `core/model.py` `Model.__init__` のインスタンスの分岐 | 新設（1 行） |

### 互換性

- **`ErrorCode.DATA_FINGERPRINT_MISMATCH` が無くなる。これは公開名を壊す変更であり、そう認めて行う。** 何もこれを出したことがないので、この code の例外を実際に受け取った呼び出し側は無い。しかし、別の例外の `code` をこのメンバーと比べる防御的な処理（`if e.code == ErrorCode.DATA_FINGERPRINT_MISMATCH:`）は書けて、それは評価された時点で `AttributeError` になる（初版は「捕まえていた呼び出し側は存在しえない」と書いたが、それは比べる処理の存在を否定しないので偽だった）。v1.0 を待たずに削除し、`docs/DEPRECATIONS.md` と `CHANGELOG.md` に破壊的変更として記録する（H-0079 の「already enforced」行と同じ扱い）。廃止期間を置く案（metaclass でメンバーへのアクセスに `DeprecationWarning` を出す。設計レビュー round 1 が実行で可能と確認した）は代替案 7 で棄却する。
- **予測時の dtype**: `NativeFeaturePipeline` での成功していた予測は変わらない（決定 2）。予測時に文字列を変換する自作 pipeline は拒否されうるが、自作 pipeline を登録する公開の手段は無い（決定 2）。失敗していた予測は、生の例外ではなく `INCOMPATIBLE_COLUMNS` になる。`FitResult.dtypes` は既存の必須フィールドなので、保存済みの artifact にも同じ検査が効く。`format_version` は据え置き。
- **指標**: 公開の指標クラスに logit 等を渡して `auc` / `auc_pr` / `ece` / `precision_at_k` の値を得ていた呼び出しは、`METRIC_REQUIRES_PROBA` になる。**`objective: cross_entropy_lambda` で指標を `auc` / `auc_pr` / `ece` / `precision_at_k` に限った `Model.fit` は、今日は成功し最大 3.148 の「確率」を返すが（#307 のデータで実測、`pr6_xentlambda_metric_probe.py`）、本 PR の後は、出力が 1 を超える限り評価器で `METRIC_REQUIRES_PROBA` になる。** 出力が 1 を超えない短い fit（1 本の木で 0.42〜0.59、クローズレビューが実行）は今も成功する。これは意図した変更である（確率でない出力を確率として報告しなくなる）。目的関数そのものの扱いは #307 で決める。`logloss` / `brier` を含む場合は今日も scikit-learn の生の `ValueError` で失敗しており、名前の付いた例外に変わるだけ。`tune()` の trial の中でこの例外が出た場合は、その trial が失敗として記録され、全 trial が失敗すれば `TUNING_FAILED` になる（B8、tuner の `catch` による。`tune()` での発生は実行していない）。
- **feval のテスト**: `tests/test_estimators/test_feval_probabilities.py` の「評価器の規則が失敗するなら feval も同じ型・同じ文言で失敗する」分岐は、そのまま成り立つ。feval は指標の例外を包まない（`metric_bridge.py:278` で指標を直接呼ぶ）ので、評価器側と同じ `LizyMLError` が出る。`cross_entropy_lambda` の `ece` / `precision_at_k` のセル（今日は計算できている 40 回）は、この分岐に移る。
- **`test_metric_entry_integration.py::test_feval_returns_display_name` を書き直す（削除しない）。** feval の表示名を確かめるテストだが、入力に「LightGBM が feval に logit を渡し、sigmoid を通す」という H-0105 が誤りとした前提の合成 logit（-2〜2）を使っており、本 PR の後は `METRIC_REQUIRES_PROBA` で失敗する。PR 5b の書き直しから漏れたのは、`precision_at_k` は順位だけで決まり値が変わらなかったため。入力を同じ logit の sigmoid（LightGBM が実際に渡す確率）に替え、表示名・向き・値の型の主張はそのまま残す。
- **`config_version`**: `LizyMLConfig` インスタンスの入口（`False` を保持したインスタンスを含む）、環境変数 `LIZYML__config_version`、`config_version: false` で、サポート外の版が拒否されるようになる。今日これらで受理されていたサポート外の版は、何の効果も持たなかった（#272: 版 2 で学習したモデルはバイト単位で同一）。`context["config_version"]` は loader が拒否する場合は利用者が書いた値（`"2"` / `"2.0"` / `2.5` 等）、schema の検査が拒否する場合は変換後の `int`。`"2.0"`（今日は版 2 として受理）も拒否されるようになる。小数の float の扱いは今日と同じ（`2.5` は `CONFIG_VERSION_UNSUPPORTED`、`1.5` は `CONFIG_INVALID`）。`inf` は生の `OverflowError` から `CONFIG_INVALID` に変わる。
- 生成コード（`export_code`）は変わらない。

**Firing rate**（本 Proposal は `allow` の条件を 2 つ新設し（dtype、確率）、既存の 1 つ（版）を新しい位置に広げ、dtype の検査に免除を 1 つ置く。4 つとも測った。数値は設計レビュー round 1 の指摘で計測器を直した後の実行のもの: 判定を本 Proposal の規則と完全に一致させ、版の分母を「完了した呼び出し」に限り、`LizyMLConfig.__init__` と `model_validate*` のすべてを包み、計測器の起動時に陽性・陰性の対照を実行する。レビューは直す前の計測器で 112 / 7053 / 1254 / 1280 と 61 件を再現した。直した後の版の分母 1253 / 1375 はレビューを経ていない）:

Firing rate: 0/112 of `run_predict` calls in the full test suite (`develop` `1abf7fb`; `pr6_firing_plugin.py` wraps `run_predict` — every `Model.predict` that gets past the fitted-state check — and applies the numpy-scalar-type rule to `FitResult.dtypes` against the predict-time frame; 8108 passed)

Firing rate: 0/23 of fittable dtypes record a `FitResult.dtypes` string that `pandas_dtype` cannot parse (the unreadable-metadata exemption; `pr6_dtype_rule_probe.py` fits each of the 33 dtypes once; the 10 that cannot be fitted are refused by LightGBM at fit)

Firing rate: 61/7053 of calls to the six `needs_proba` metrics in the full test suite (same run; 60 in `test_feval_probabilities.py`'s `cross_entropy_lambda` cells -- 20 already raised, 40 `ece` / `precision_at_k` computed and would now be refused -- and 1 in `tests/test_metrics/test_metric_entry_integration.py::test_feval_returns_display_name`, which hands the feval synthetic logits in [-2, 2]; no `Model.fit` cell passes a non-probability. The design review round 2 corrected the attribution: the first version said all 61 were in the `cross_entropy_lambda` cells)

Firing rate: 0/1253 of completed `Model.__init__` calls and 0/1375 of completed pydantic validations of `LizyMLConfig` (`__init__`, `model_validate`, `model_validate_json`, `model_validate_strings`) in the full test suite produced a config whose `config_version` the new check rejects (same run; calls the loader already refused are not completions and are not counted)

`config_version` の検査は「サポート外の版を拒否する」という既存の規則の位置を増やすだけで、受理する集合（`[1]`）は変えない。

### 代替案（検討して棄却）

1. **`DATA_FINGERPRINT_MISMATCH` を残して予測時の照合を実装する。** 照合できる成分が無い（決定 1）。列の集合だけを比べる狭い照合は、既存の不足列・余剰列の検査と重複する。
2. **確率の検査を、値の較正に依存する指標（`logloss` / `brier` / `ece`）だけにし、順位だけで決まる指標（`auc` / `auc_pr` / `precision_at_k`）にはスコアを許す。** scikit-learn の `roc_auc_score` は任意のスコアを受け取るので、利用者には便利である。しかし `needs_proba` は「確率が必要」という宣言で、LizyML の内部（評価器、feval）はこの宣言に従って確率を渡す。順位の指標に 1 を超える値が届くのは、常に上流の欠陥（#307 のように確率でない出力を確率として扱っている）であり、それを黙って計算したことが #307 の一部を隠していた。宣言を指標ごとに別の意味にしないため、6 指標に同じ規則を当てる。スコアで AUC を計算したい利用者は scikit-learn を直接使える。
3. **`INCOMPATIBLE_COLUMNS` の判定に pandas の `is_numeric_dtype` を使う。** pyarrow の数値 dtype 3 種と `longdouble` で LightGBM と食い違う（4/33）。
4. **`INCOMPATIBLE_COLUMNS` を「学習時と完全に同じ dtype」とする。** `int64` で学習して `float64` で予測すると今日は正しく予測できる（実測で予測値も一致）。成功している予測を拒否することになる。
5. **版の検査を schema の validator だけにし、loader の生の値の検査を削除する。** 利用者が書いた綴りが `context` から失われる。どちらも同じ検査関数を呼ぶので、規則は 1 か所のまま。
6. **`LizyMLConfig` に `validate_assignment=True` を設定して代入を検査する。** 全フィールドの代入の振る舞いが変わる（範囲外）。`model_construct` は防げない。
7. **`DATA_FINGERPRINT_MISMATCH` に廃止期間を置く（metaclass でアクセス時に `DeprecationWarning`、v1.0 で削除）。** 期間中は「宣言され、何も出さないメンバー」が残り、それを許すには母集団のテストに免除を書くことになる。計画 round 6 が棄却した保留表（受け入れ規則の変更、DC5）と同じ形である。`ErrorCode` に metaclass を足すことも、enum 全体の振る舞いに触れる。比べる処理を書いた利用者への影響（`AttributeError`）は、破壊的変更として `CHANGELOG.md` に書く。
8. **dtype の検査を pipeline の変換の後（推定器に渡す直前）に置く。** 変換する自作 pipeline を通せるが、学習時に変換後のどの列が `category` だったかの記録が無い（`categorical_cols` を持たない自作 pipeline もある）。数値の列に `category` が届く場合（LightGBM が `categorical_feature do not match` で失敗する）を、記録なしでは判定できない。入力の契約として変換の前に置く。

### 受け入れ基準（テスト観点）

詳細と証拠のテスト名は `docs/audits/2026-09-defect-discovery/results/pr6_acceptance_criteria.md`。

1. `ErrorCode` の全メンバーが `ast.Raise` に現れ（静的、10 メンバー以上を見つける自己検査付き）、かつ条件を作ると出る（振る舞い、キーが `set(ErrorCode)` の dict）。修正前は 3 メンバーで RED。
2. 学習時に数値の列について、33 種の到着 dtype のすべてで「規則が受理 ⇒ `predict` 成功、規則が拒否 ⇒ `INCOMPATIBLE_COLUMNS` と `context`」。学習時に `category` の列は dtype の規則で検査されない（整数の category に `float16` / `longdouble` が届く 4 セルは #309 の strict な xfail）。不足列は dtype より先に `DATA_SCHEMA_INVALID`。列を検査しない自作 pipeline でも同じ例外。予測時に文字列を変換する自作 pipeline でも、数値で学習した列に文字列が届けば拒否される（意図した制約の固定）。`Model.load()` 後も同じ。fit できる 23 種の dtype の記録文字列がすべて読み戻せる。読み戻せない記録の列は検査されず予測が進む（免除の固定）。
3. `needs_proba` の全指標（登録から読む）について、有限でない値・[0, 1] の外・3 クラス以上で 1 次元・数値でない値が `METRIC_REQUIRES_PROBA` になる。binary の正当な確率と 0/1 のハードラベル、数値の object 配列は通り、値は修正前と同じ。multiclass に対応する 4 指標は 2 次元の正当な確率で通り、2 次元で [0, 1] の外なら拒否される。0/1 以外の 2 値ラベル（`[3, 7]` 等）の binary は 1 次元の規則で拒否されない。`cross_entropy_lambda` で `auc` に限った `Model.fit` が `METRIC_REQUIRES_PROBA` になる。
4. `config_version`: 入口（`load_config(dict)`、`Model(dict)`、`model_validate`、`Model(model_validate)`、`Model(model_construct)`、`Model(代入)`、`Model(model_copy(update=))`、環境変数の上書き）× 版（`1` / `2`）のすべてのセルで、`1` は受理、`2` は `CONFIG_VERSION_UNSUPPORTED`。`False` はすべての入口で拒否され、検証を通らない 3 経路では `Model` が受け取る時点で拒否される（構築・代入・コピーの時点ではない）。`True` と `"1"` は受理される。`loader.SUPPORTED_CONFIG_VERSIONS is version.SUPPORTED_CONFIG_VERSIONS`。
5. `docs/api.md` の例外コード表と `BLUEPRINT.md` §16.2 が `set(ErrorCode)` と一致する。
6. 既存のテストは削除しない。`tests/test_core/test_exceptions.py` の一覧は `DATA_FINGERPRINT_MISMATCH` を除いて更新する（列挙の SSOT は enum。この一覧は削除の意図を固定する）。

## H-0107: 漏洩検査が比較できない列を黙って飛ばさない（#267 / PR 7）

- **ステータス**: Accepted
- **起票日**: 2026-10-01
- **決定日**: 2026-10-01（事実確認を含む外部レビュー: 設計レビュー 3 ラウンド（blocking 4 → 1 → 0。報告の順序を決め、`TypeError` / `ValueError` 以外も包むことと `cause` の同一性を基準に加え、分母と母集団の記述を訂正）、コードレビュー 2 ラウンド（blocking 2 → 0。文書とコメントの文言）。各ラウンドの前に absolute / relational monitor。設計レビューの指摘で #311 を起票）
- **スコープ**: `lizyml/data/validators.py`（`validate_no_target_leakage` の `except (TypeError, ValueError): pass` を置き換え）, `docs/api.md`（漏洩検査の節と `DATA_SCHEMA_INVALID` の行）, `CHANGELOG.md`, テスト（`tests/test_data/test_leakage_validator_unchecked_column.py`、新規）
- **関連**: [Issue #267](https://github.com/nbx-liz/LizyML/issues/267), H-0087（漏洩検査を公開 API にした）, 計画 `phase3-plan.md` §PR 7
- **実測の記録**: `docs/audits/2026-09-defect-discovery/results/pr7_measurements.txt`（`develop` `97381db`。スクリプトは `../instruments/pr7_*.py`）

### 目的（課題）

`lizyml.data.validate_no_target_leakage` は、各列を目的変数と比べる呼び出し（`_series_perfectly_correlated`）を `try` で囲み、`TypeError` / `ValueError` を捕まえて**何もせずに次の列へ進む**（コメントは「Non-comparable types; skip」）。比べられなかった列は検査されていないのに、呼び出し側には検査して漏洩が無かった場合と同じ空のリストが返る（DC1）。

#267 の時点では、17 種の列でこの handler に入る入力が見つからず、「handler は死んでいる」のか「到達する入力を見つけていない」のかが決まっていなかった。計画の step 1（378 セル）で、**数値を名乗る `ExtensionDtype`（`_is_numeric = True`）で、配列が `__array__` か `isna` で例外を出すもの**が到達することがわかった（15/378）。その計測器はリポジトリに残っていないので、`develop` `97381db` で再現した（`pr7_hostile_numeric_probe.py`）: 2 つの形 × 目的変数 5 種（int64 / float64 / bool / Int64 / complex128）の 10 セルすべてで、比べる呼び出しが `TypeError` / `ValueError` を出し、`validate_no_target_leakage(..., raise_on_violation=True)` は `[]` を返した。

一方、普通の列は到達しない: PR 6 の 33 種の dtype の列 × 元の順 / 逆順 × 目的変数 7 種（int64 / float64 / bool / Int64 / object の文字列 / category / string）の 462 セルで、比べる呼び出しの例外は 0（`pr7_dtype_sweep.py`。列は目的変数とは独立に作るので、どのセルでも目的変数と一致する列と一致しない列の両方を作ったわけではない。設計レビュー round 1 が表現を訂正）。フルスイートでも、比べる関数の呼び出し 13 回（検査を通る 9 回と、関数を直接呼ぶテストの 4 回）で例外は 0（`pr7_swallow_plugin.py`、8424 passed）。

### 対応方針（決定）

0. **報告の順序は今日のまま。** 列は先頭から順に検査し、`raise_on_violation=True` で漏洩している列に先に当たれば、今日と同じくその場で `LEAKAGE_SUSPECTED` を出す（比較できない列がその後ろにあっても、漏洩の報告を後回しにしない）。保証するのは「目的変数が frame にあるとき、戻り値（`[]` または警告のリスト）を返すのは、すべての列を比べられたときだけ」であり、比較できない列があれば戻り値は返らない（目的変数が無いときの `[]` は #311、下記の対象外）。
1. **比べられなかった列は黙って飛ばさず、列名を付けて報告する。** 比べる呼び出しが例外を出したら、`LizyMLError(DATA_SCHEMA_INVALID)` を出す。`context = {"column", "target"}`、`cause` に元の例外を付ける。`raise_on_violation` の値によらず出す: 警告のリストは「漏洩の疑い」を表すので、「検査できなかった」を同じリストに入れると、呼び出し側が文言を読み分けない限り 2 つが混ざる（代替案 2）。
2. **捕まえる範囲は比べる呼び出しだけにし、例外の型は限らない（`Exception`）。** 今日捕まえていた `TypeError` / `ValueError` に加え、拡張配列が出しうる他の例外（`OverflowError` / `AttributeError` 等）も、列名の無い生の例外として外に出るより、どの列で失敗したかを示す方がよい。漏洩を見つけたときの `LEAKAGE_SUSPECTED` は `try` の外で出す（今日は `try` の中にある。型が違うので捕まらないが、範囲を正しくする）。
3. 古いコメント「Non-comparable types; skip」は、それが説明していたコードと一緒に消す。`_series_perfectly_correlated` の docstring（NaN の位置を先に比べる理由）はそのまま残す。

**計画との違い**: 計画 §PR 7 は「handler を削除し、例外をそのまま伝える」としていた。削除だけでは、伝わる例外は拡張配列が出した生の例外（例: `TypeError: _HostileArray.__array__`）で、どの列で起きたかが分からない。#267 の DoD の「`n` 個の列が検査されなかったことを呼び出し側が区別できる」を満たすには列名が要るので、`ErrorCode` を付けて出し直す。黙って飛ばす経路は無くなる点は計画と同じ。

### 規則が縛る位置（ソースから導出）

規則: **漏洩検査は、検査できなかった列を検査済みとして扱わない。** 導出: `lizyml/data/validators.py` の `except` を grep した（`97381db`、1 か所）。他の 2 つの検査（`validate_time_series_order` / `validate_group_split`）に例外を捕まえる箇所は無い。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `data/validators.py` `validate_no_target_leakage` の `except (TypeError, ValueError): pass` | 置き換え |

### 互換性

- 普通の列（462 セル）とテストスイートの入力では、振る舞いは変わらない（どれもこの経路に入らない）。
- 比べられない列（数値を名乗り、比較で例外を出す拡張配列）を含む frame の今日の振る舞いは、比較が出す例外の型で 2 つに分かれる。`TypeError` / `ValueError` なら handler がその列を飛ばし、残りの列だけの結果（漏洩が無ければ `[]`、あれば `LEAKAGE_SUSPECTED` か警告）が返る。それ以外（`OverflowError` 等）は handler が捕まえないので、列名の無い生の例外として外に出る。本 PR の後は、比較の例外の型によらず、その列に達した時点で `DATA_SCHEMA_INVALID` になる（漏洩している列が先にあり `raise_on_violation=True` なら、今日と同じく先に `LEAKAGE_SUSPECTED`）。`TypeError` / `ValueError` の場合はその列を落としてから検査し直せば今日と同じ結果になる（設計レビュー round 2 の指摘で、例外の型による違いを書き分けた）。
- **対象外: 目的変数が frame に無いとき `[]` を返す振る舞い**（`validators.py` の先頭の `if target not in df.columns: return []`）。これも「検査していない」を「漏洩なし」と同じ形で返すが、例外を握りつぶす #267 の handler とは別の経路で、既存のテスト `tests/test_data/test_validators_edge.py::test_leakage_missing_target` が意図した振る舞いとして固定している。変えるかどうかは公開 API の判断なので [#311](https://github.com/nbx-liz/LizyML/issues/311) に切り出した。
- 公開 API の形（引数・戻り値）は変わらない。`docs/api.md` の `DATA_SCHEMA_INVALID` の行に、この条件を足す。

**Firing rate**: 本 Proposal は skip / shorten / cache / select / allow / conditionally-activate の条件を新設しない。今ある `skip`（例外の列を飛ばす）を取り除く。取り除く条件の発火の測定:

Firing rate: 10/10 of numeric-declared extension columns that raise in `__array__` or `isna` (2 shapes x 5 target dtypes, `pr7_hostile_numeric_probe.py`), 0/462 of ordinary cells (33 column dtypes x original / reversed order x 7 target dtypes, `pr7_dtype_sweep.py`), and 0/13 of the comparison-helper calls in the full test suite (9 through the validator, 4 direct helper tests; `pr7_swallow_plugin.py`) enter the handler being removed (`develop` `97381db`)

### 代替案（検討して棄却）

1. **handler を削除して例外をそのまま伝える（計画の案）。** 黙って飛ばす経路は無くなるが、生の例外は列名を持たないので、どの列が検査されなかったかが呼び出し側に分からない。
2. **`raise_on_violation=False` のときは警告のリストに「列 X は検査できなかった」を入れて続ける。** 文言で書き分けることはできるが、リストは漏洩の疑いの一覧として使われるので、呼び出し側が文字列を読み分けない限り 2 つが混ざる（設計レビュー round 1 の指摘で「区別できない」から表現を改めた）。戻り値の型を変えて 2 つを分けるのは公開 API の変更で、到達する入力（意図的に壊れた拡張配列）に比べて大きすぎる。
3. **新しい `ErrorCode`（例: `LEAKAGE_CHECK_FAILED`）を足す。** 列を目的変数と比べられないのは、列の型がこの検査に使えないという data schema の問題で、既存の `DATA_SCHEMA_INVALID` で表せる。メンバーを増やすと、PR 6 の母集団の検査（全メンバーが発生する）にも条件を 1 つ足すことになる。

### 受け入れ基準（テスト観点）

詳細と証拠のテスト名は `docs/audits/2026-09-defect-discovery/results/pr7_acceptance_criteria.md`。

1. 比較で例外を出す 3 つの形（`__array__` が `TypeError` / `isna` が `ValueError` / `__array__` が `OverflowError`。最後は `TypeError` / `ValueError` 以外も包むことの確認）× 目的変数 5 種 × `raise_on_violation` の 2 値で、`validate_no_target_leakage` が `DATA_SCHEMA_INVALID` を出し、`context` の `column` / `target` が正しく、`cause` が元の例外**そのもの**（同じオブジェクト）。修正前は `[]` が返る（`OverflowError` の形は生の例外が漏れる）ので RED。
2. 比較できない列と漏洩している列の 2 つの順 × `raise_on_violation` の 2 値: 比較できない列が先なら `DATA_SCHEMA_INVALID`。漏洩列が先で `True` なら `LEAKAGE_SUSPECTED`（今日と同じ即時の報告）、`False` なら警告を溜めた後に比較できない列で `DATA_SCHEMA_INVALID`。どの場合も戻り値は返らない。
3. 普通の列では今日と同じ: 漏洩している列は `LEAKAGE_SUSPECTED`（`code` と `context["leaking_column"]` まで一致。`raise_on_violation=False` なら警告 1 件）、していない列は `[]`。漏洩の `LizyMLError` が包み直されないことはこの行で確かめる。
4. `validators.py` に `except ...: pass` も「Non-comparable」のコメントも残っていない。
5. 既存のテストは削除しない。

## H-0108: 既定値付きの構成値の出どころを全数で分類し、Config のキーが設定しないものを BLUEPRINT に書く（#268 / PR 8）

- **ステータス**: Accepted
- **起票日**: 2026-10-01
- **決定日**: 2026-10-01（事実確認を含む外部レビュー: 設計レビュー 4 ラウンド（blocking 6 → 1 → 1 → 0。分類の規則と優先順位を明文化し、経路ごとのセルを足し、#313 を発見）、コードレビュー 3 ラウンド（blocking 6 → 1 → 0。回帰の代替経路のセル、§5.5 の位置の検査、#268 への訂正コメント）。各ラウンドの前に absolute / relational monitor。**#268 と計画の前提（名前の照合）が誤りだったので、承認済みの「9 個を公開」を置き換えた**）
- **スコープ**: `BLUEPRINT.md`（§5.5 を新設）, `tests/test_config/_knob_registry.py` / `tests/test_config/test_knob_reachability.py`（新規）, `tests/test_plots/test_model_plot_options.py` / `tests/test_tuning/test_detect_boundary_threshold.py`（新規）, `docs/audits/2026-09-defect-discovery/phase3-plan.md`（§3 / §PR 8 / §6 / §7 の訂正）, `CHANGELOG.md`。**production コードは変えない。**
- **関連**: [Issue #268](https://github.com/nbx-liz/LizyML/issues/268), [Issue #313](https://github.com/nbx-liz/LizyML/issues/313)（設計レビューで発見）, H-0065（指標を dict で書く形）, H-0104（`features.unseen_policy`）, H-0101（`TimeHoldoutInnerValid.gap` は自動解決だけが設定する）, H-0030（較正は生のスコア）
- **実測の記録**: `docs/audits/2026-09-defect-discovery/results/pr8_measurements.txt`（`develop` `91a698b`。`../instruments/pr8_write_measurements.py` で再生成）

### 目的（課題）

#268 は、公開クラスの `__init__` の既定値付き引数 74 個のうち 25 個を「Config から届かない」とし、計画（`phase3-plan.md` §PR 8、§6 で承認済み）は「9 個を公開、13 個を方針として書く」とした。**この分類は、引数の名前と Config のフィールド名を照合して作られていた。** 実行して確かめると、名前が違うだけで届いているものが多い（`pr8_reachability_probe.py`: 既定でない値を Config に書いて fit / tune し、コンストラクタが受け取った値を記録した）:

- 分割器の `max_train_size` / `max_test_size`（6 個）: `split.train_size_max` / `split.test_size_max` から届く。この 2 つのキーは 2026-03-07（`5daaffd`）から存在する。
- `PrecisionAtK.k` / `ECE.n_bins` / `HuberLoss.delta`: 指標を dict で書く形（`{"precision_at_k": {"k": 20}}`、H-0065、2026-03-28 `0f37464`）で `evaluation.metrics` から届く。`k` と `n_bins` は `model.params` の `metric`（feval）からも届く。**`huber` は `model.params` の `metric` では LightGBM 組み込みの指標として扱われ、`delta` は黙って捨てられる**（無効な引数も検査されない。設計レビュー round 1 が発見し、#313 に切り出した）。
- `LGBMAdapter.early_stopping_rounds`: `training.early_stopping.rounds` から届く。
- `Tuner.progress_callback` / `.storage` / `.study_name`: `Model.tune(...)` の引数から届く。

つまり計画の「公開する 9 個」は 9 個とも、計画を書いた時点ですでに Config から届いていた。**承認された決定（9 個の公開）を、その前提の計測が誤っていたので置き換える。** 逆に、#268 の一覧に無い `StratifiedKFoldSplitter.shuffle` は、構築箇所が常に定数 `True` を渡し、`StratifiedKFoldConfig` に `shuffle` が無いので変えられない。

### 対応方針（決定）

1. **74 個を、値の出どころで 5 種類に分類する**（`tests/test_config/_knob_registry.py`）。分類はそのクラスを構築するすべての本番の経路で読み、最初に当てはまるものを採る（設計レビュー round 1 の指摘で、規則と優先順位を明文化した）:
   - **config（60）**: Config のキーの値がそのまま渡る（一部の経路だけでも）。経路によって出どころが違うもの（明示の inner valid か自動解決か、`split.random_state` が無いときの `training.seed` など）は、経路ごとに書く。`task` を受け取る 5 個、`Model.output_dir`（Config の `output_dir`。引数を渡せばそちらが優先）、自動解決で outer の `split.gap` を受け取る `TimeHoldoutInnerValid.gap`、代替経路で `validation_ratio` を受け取る `StratifiedTimeHoldoutInnerValid.ratio` もここに入る（後の 2 つは設計レビュー round 2 の指摘で derived から移した）。
   - **api（4）**: Config のキーは設定せず、公開の呼び出しの引数が設定する（`Model(data=)`、`Model.tune(progress_callback=, storage=, study_name=)`）。利用者が選べるので、#268 の意図（利用者に知らせずに決めている値をなくす）を満たす。
   - **derived（3）**: ライブラリがデータから、または設定から 1 つの値を渡すのではない規則で決める（クラス数の 2 つ、較正の有無）。規則を書く。
   - **policy（2）**: ライブラリが値を固定している（定数を渡すか既定値のままにする）。Config のキーも公開の引数も変えられない: `LGBMAdapter.verbose_eval = -1`、`StratifiedKFoldSplitter.shuffle = True`（初版は「どの呼び出し元も渡さない」と書いたが、`shuffle` は構築箇所が定数で渡すので誤りだった）。
   - **internal（5）**: 振る舞いの設定ではない（`LizyMLError` の 3 つ、2 つのトレーナーの `ratio_param_resolver`）。
2. **Config に新しいキーは足さない。** policy の 2 個は方針として書く:
   - `verbose_eval = -1`: LightGBM の反復ごとの評価ログを出さない。学習曲線は `FitResult.history` に記録されるので、ログは情報を増やさない。
   - `StratifiedKFoldSplitter.shuffle = True`: `stratified_kfold` は常にシャッフルしてから層化し、順序は `split.random_state`（無ければ `training.seed`）が決める。行の順序に意味があるデータには時系列の分割を使う。公開すると、承認されていない Config の面を増やすことになる（代替案 1）。
3. **Config のキーが設定しない 14 行（api / derived / policy / internal）を `BLUEPRINT.md` §5.5 の表に書く**（経路による条件を含めて）。表の行の集合と種類が台帳と一致することをテストで確かめる（行を読み取って集合で比べる）。
4. **恒久検査**（`tests/test_config/test_knob_reachability.py`）:
   - 母集団はテスト時に AST で数える（新しい構成値は分類されるまで失敗する）。台帳のキーが母集団と一致すること。数え方が壊れて空になった場合を「全部分類済み」と読まないよう、60 個以上を見つけることも確かめる。
   - config の 60 行すべてに実行セルがあり（`Model.output_dir` だけは Config のキーを `Model.__init__` の中で解決するので、別のテストで確かめる）、**出どころが違う経路にはそれぞれセルがある**: 明示した inner valid と自動解決（`random_state` ← `training.seed`、`stratify` ← outer の分割手法、`ratio` ← `validation_ratio`）、`split.random_state` が無いときの `training.seed`、指標の `evaluation.metrics` と feval、`BlockedGroupInnerValid` の分類 / 回帰の代替経路（設計レビュー round 1 が、自動解決の経路の seed を定数に変える変異が初版の全テストを通ることを示し、コードレビュー round 1 が回帰の代替経路のセルが無いことを示した）。各セルは本物の `Model.fit` / `tune` でコンストラクタが受け取った値を確かめる。多くのセルは既定でない値を Config に書く。いくつかは既定値と同じ値になる分岐（回帰の task、自動解決の `stratify=False`）を確かめるもので、届いたことの証拠にはならないので、どの行にも既定でない値のセルが少なくとも 1 つあることを別のテストで確かめる（コードレビュー round 1 の指摘）。名前の照合はしない。
   - api の行、derived の行（クラス数、`collect_raw_scores` の fit と tune。tune は較正を設定した場合も）、`TimeHoldoutInnerValid.gap` の明示の経路（0）、policy の 2 行（固定値のまま）を実行で確かめる。
5. **#268 の「どこからも使われていない公開オプション 4 つ」**: `Model(data=)` は今のテストスイートで使われている。`detect_boundary(threshold=)` は既定値と同じ `0.05` でしか呼ばれていない（設計レビュー round 1 の指摘で、初版の「使われている」を訂正）。`Model.importance_plot(top_n=)` と `Model.plot_learning_curve(metrics=)`（`Model` のメソッドを通す形）は使われていない。この 3 つにテストを足す（`detect_boundary` は既定値では端にならず `threshold=0.2` では端になる値で、判定が変わることを確かめる）。

### 規則が縛る位置（ソースから導出）

規則: **公開クラスの既定値付き引数はすべて、Config のキーから実行で届くことが確かめてあるか、§5.5 に出どころが書いてある。** 導出: `lizyml/` 全体の AST（`ClassDef` で名前が `_` で始まらないもの、その `__init__` の既定値付き位置引数とキーワード専用引数）。構築箇所は `pr8_construction_sites.py` が列挙した。既定のままにする箇所とリテラルを渡す箇所は、1 つずつ読んだ:

- `shap_explainer.py:160`: 保存済みの状態を `load_state` で読むので、`unseen_policy` は保存値に従う。
- `_model_tuning.py:551`: tune の trial の CV では trial の評価を較正しないので、`collect_raw_scores` は既定の `False`。
- `_model_factories.py:294` / `:301`: outer split の手法から `stratify` を決める、自動解決の経路。
- `_model_factories.py:357` / `inner_valid.py:346`: gap を渡さない明示の経路と、代替の経路。
- `_model_factories.py:118`: `StratifiedKFoldSplitter(shuffle=True)`。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | 74 個の構成値（台帳） | 分類し、テストで母集団と一致させる |
| 2 | `BLUEPRINT.md` §5.5 | 新設（14 行） |

### 互換性

- production コードを変えないので、振る舞いは変わらない。追加するのはテストと文書だけ。
- 新しい公開クラスや引数を足すと、台帳に分類するまで `test_registry_classifies_exactly_the_census` が失敗する。これは意図した変更（分類しないまま既定値で決める構成値を増やさない）。
- #313（`model.params` の `metric` で、組み込みの指標の dict の引数が捨てられる）は別の PR で決める。

**Firing rate**（台帳は「Config のキーが設定しなくてよい」構成値を許す `allow` の表である）:

Firing rate: 14/74 of defaulted public constructor knobs are allowed without a Config key setting their value (api 4, derived 3, policy 2, internal 5), each stated with its source in BLUEPRINT §5.5; the other 60 receive a Config key's value on at least one path, each such path executed (`develop` `91a698b`; census by AST sweep, `pr8_knob_census.py`; executed by `pr8_reachability_probe.py` and by `test_knob_reachability.py`)

### 代替案（検討して棄却）

1. **`StratifiedKFoldConfig` に `shuffle` を足す。** sklearn の `StratifiedKFold` は `shuffle=False` を許すので、選べるようにすることはできる。しかし、これは計画も利用者も承認していない Config の面で、#268 にも挙がっていなかった。層化しつつ行の順序を保ちたい要求は今のところ無く、時系列の分割で表せる。方針として書き、要求が出たときに Proposal で公開する。
2. **計画どおり 9 個を「公開する」。** 9 個ともすでに Config から届いているので、作業は存在しない。
3. **#268 と同じく名前の照合で検査を作る。** 今回の誤りを生んだ方法そのものである（`max_train_size` と `train_size_max`、dict で書く指標の引数）。実行して値が届くことを確かめる。
4. **derived / internal を Config に出す。** クラス数は他の設定やデータから決まり、別の値を選ぶと矛盾する。`LizyMLError` の中身は振る舞いではない。

### 受け入れ基準（テスト観点）

詳細と証拠のテスト名は `docs/audits/2026-09-defect-discovery/results/pr8_acceptance_criteria.md`。

1. 台帳のキーの集合が AST の母集団と一致し、母集団は 60 個以上。
2. config の 60 行すべてに実行セルがあり、出どころが違う経路にはそれぞれセルがある。各セルで、Config に書いた既定でない値がコンストラクタに届く（較正の `params` と `LGBMAdapter.params` は、書いた項目が含まれる）。`Model.output_dir` は Config のキーと引数の優先順位を確かめる。
3. api の 4 行が本物の呼び出しで届く。
4. `BLUEPRINT.md` §5.5 の表の行と種類が、台帳の config 以外の 14 行と一致する。修正前は §5.5 が無いので RED。
5. derived の行（クラス数、`collect_raw_scores`）、`TimeHoldoutInnerValid.gap` の明示の経路、policy の 2 行が、実行で台帳の記述どおり。
6. `Model.importance_plot(top_n=1)` が 1 特徴だけを描き、`Model.plot_learning_curve(metrics=[...])` が指定した指標だけを描き、`detect_boundary` の `threshold` が判定を変える。

## H-0109: fit が適用した training overlay を artifact に記録し、`load()` 後の報告面が同じ値を答える（#281 / PR 8b）

- **ステータス**: Accepted
- **起票日**: 2026-10-01
- **決定日**: 2026-10-01（事実確認を含む外部レビュー: 設計レビュー 6 ラウンド（blocking 4 → 1 → 1 → 2 → 1 → 0。round 1 は全数計測への `evaluate()` などの追加、gain の誤差上限、事実誤認、状態遷移の受け入れ基準。round 2〜5 は gain の誤差上限の導出（binary32 への読み戻し、非正規数、丸めの補題の向き、限定句）。#315 を起票）、コードレビュー 2 ラウンド（blocking 4 → 0。round 1 は、学習が受け入れる categorical の `"0.45"` / `True` を生のまま記録すると `load()` が拒否する欠陥（DC7）、`10**400` の `OverflowError`、bool の節の単独の検査、context の記述）。各ラウンドの前に absolute / relational monitor）
- **スコープ**: `lizyml/persistence/exporter.py`（`metadata.json` に `applied_training_params` を追加）, `lizyml/core/_model_persistence.py`（export で渡し、load で検査して復元）, `lizyml/core/model.py` / `lizyml/core/_model_state.py`（「不明」を `None` で表す）, `lizyml/core/_tuning_validation.py`（training の次元名を 1 つの定数に）, `BLUEPRINT.md`（§7.4 / §15.1 / l.1455 の bound）, テスト（新規 2 ファイルと共有の全数計測モジュール `tests/test_persistence/_load_census.py`、既存 1 件の docstring）, `docs/audits/2026-09-defect-discovery/instruments/report_lifecycle_grid.py`, `CHANGELOG.md`
- **関連**: [Issue #281](https://github.com/nbx-liz/LizyML/issues/281), [Issue #315](https://github.com/nbx-liz/LizyML/issues/315)（本 PR の全数計測で発見、対象外）, H-0094 決定 13（本提案が置き換える bound）と決定 14（fit の commit をまとめる）, H-0086（`tuning` ブロック、trial を保存しない決定）, H-0083（追加のキーで `format_version` を上げない前例）
- **実測の記録**: `docs/audits/2026-09-defect-discovery/results/pr8b_measurements.txt`（`develop` `036cd18`。各計測器は `../instruments/pr8b_*.py`）

### 目的（課題）

H-0094 決定 13 は、`params_table()` と `export_code()` が「この fit が何を使ったか」を答えるようにした。patience は学習済み adapter から読む（joblib で保存されるので `load()` を越える）。inner valid の比率は adapter に記録が無いので、fit が適用した overlay（`FitState.applied_training_params`）から読む。**この overlay は artifact に無い**ので、`load()` 後は config の比率に落ちる。決定 13 はこれを bound として書き、テストで固定した。

この bound が実際に誤った値を答える lifecycle は `tune → fit → export → load` である（#281）。`export()` は model の**現在の** tuning result を `tuning` ブロックに書くので、`fit → tune → export` の artifact も、どの fit も消費していない overlay を持つ。実測: `tuning.best_training_params` は `tune_fit` と `fit_tune` で同じ `{'early_stopping_rounds': 219, 'validation_ratio': 0.45}` だった。tuning ブロックから「fit が適用した値」を復元すると、`fit_tune` で誤る。だから記録が別に要る。

計画 §3 の恒久検査は「報告面が答える値はすべて `load()` を越えて残る」である。範囲を #281 の 1 値に限らず、全数で測った（`pr8b_load_census.py`。定義は `tests/test_persistence/_load_census.py` にあり、恒久検査と共有する）。母集団は `Model` の公開名 24 個で、そのうち 4 個（`fit` / `tune` / `export` / `load`）は報告面でない理由を書き、残り 20 個を 22 面として読む（`importance` は split / gain / shap の 3 面、`export_code` は `generate_code` に渡す引数、`fit_result` は `FitResult` のフィールド）。設計レビュー round 1 が `evaluate()` の欠落を指摘し、`fit_result` と `predict()` も加えた。**bound**: 各面は既定の引数で呼ぶ（`importance` の kind を除く）。`confusion_matrix(threshold=)` などの既定でない引数は測っていない。

22 面 × 4 構成（regression / binary / platt 較正付き binary / multiclass）× 5 lifecycle（`fit` / `tune_fit` / `fit_tune` / `tune_fit_reexport` / `tune_resume_fit`）= 440 セル。80 セルが `load()` 後に違う:

| 面 | セル | 原因 | 本提案 |
|---|---|---|---|
| `params_table` / `export_code` | 12 / 12 | `validation_ratio`: tuned 0.45 → config 0.2（#281） | 直す |
| `importance("gain")` | 20 | LightGBM のモデルテキストは `split_gain=` を有効数字 6 桁で書き、読み込みで binary32 に戻す。pickle した adapter の gain はテキスト往復の gain と完全に一致する。予測と OOF は、測った fixture では bit 単位で一致した（`pr8b_gain_precision.py`） | 直さず書く（split ごとに相対 5.06e-6 + 絶対 2^-150） |
| `tuning_table` / `tuning_plot` / `boundary_table` | 16 / 16 / 4 | H-0086 が trial の履歴、round、境界の報告を保存しないと決めた。load 後は空の表、trace の無い図、「`tune(resume=True)` を実行せよ」という `MODEL_NOT_FIT` になる | #315 に切り出す |

`evaluate` / `evaluate_table` / `fit_result` / `predict` を含む他の面は、測った 440 セルではすべて一致した。

#315 は判断が要る（trial を保存するか、保存しないことを面が言うか）ので本提案に含めない。繰り延べで覆われないまま残る保証は「load 後の報告面は、export 前と同じ値を答えるか、答えられないと言う」で、#315 にそう書いた。したがって本提案の恒久検査は「すべての値が残る」の証明ではなく、**上の母集団と bound の中で、違うセルが宣言した例外と等しいことを固定する回帰の契約**である。

### 対応方針（決定）

1. **`metadata.json` に最上位のキー `applied_training_params` を足す。** 値は、artifact のモデルを作った fit が適用した overlay の dict で、training が読むときに変換した値を持つ（patience は `int()`、比率は `float()`。`_model_factories.applied_training_overlay`）。overlay を使わなかった fit は `{}`。生の `best_training_params` をそのまま記録すると、training が変換して受け入れる categorical の `"0.45"` や `True` を記録し、その artifact を `load()` が拒否した（コードレビュー round 1 が実行で示した）。`format_version` は 2 のまま（追加のキーだけで、H-0083 `checksums` / H-0086 `tuning` と同じ形）。
2. **「不明」と「overlay なし」を分ける。** `Model._applied_training_params` と `FitState.applied_training_params` を `dict[str, Any] | None` にする。`None` = 不明（記録の無い artifact から load した）、`{}` = fit は overlay を使わなかった、dict = 適用した overlay。export はキーを `None` でないときだけ書く。理由: 潰すと、記録の無い（修正前の）artifact を load して再 export したとき、`{}` という確定した記録を偽って書くことになる。状態の遷移: `__init__` は `{}`（fit 前は誰も読まない）。成功した `fit()` だけが記録を置き換える（決定 14 の commit の中）。`load()` はキーがあれば検査して復元し、無ければ `None`。`tune()`、拒否された `fit()`、学習中に失敗した `fit()` は記録を変えない（記録は保持しているモデルの fit を説明し続ける）。再 export は記録をそのまま書く（`None` なら書かない）。
3. **`load()` で記録を検査してから復元する。** 拒否するもの（`DESERIALIZATION_FAILED`。context にはパスと値の型、キーがあればキー）: dict でない、知らないキー（training の次元として消費される `early_stopping_rounds` / `validation_ratio` 以外）、bool、`int` でない patience、`float` でない比率、非有限の値、`(0, 1)` の外の `validation_ratio`。JSON の整数は上限が無いので、有限性は `float` にだけ問う（`math.isfinite(10**400)` は `OverflowError` を出す。コードレビュー round 1）。理由: 受け入れて報告面の `float()` で落ちる、または生成コードに不可能な比率を渡すのは、読み込みで通して後で失敗する形（DC1 / DC7）である。比率の範囲は、inner valid の戦略がすべて `0 < ratio < 1` を要求する（`training/inner_valid.py` の 5 か所）ので、どの fit も範囲外の比率を適用できない。patience には範囲を設けない: 報告面は patience を記録ではなく adapter から読み、学習経路は `int()` するだけで範囲を検査しない（設計レビュー round 1 の非 blocking の指摘に従い、学習経路より厳しい規則を記録に置かない）。2 つの名前は `_tuning_validation.py` がすでに使っている集合で、1 つの定数にして両方から読む。キーが無い artifact は今までどおり読め、`None` になる（決定 13 の bound は、記録の無い artifact に限って残る）。
4. **読み手は変えない。** `tuned_validation_ratio(None)` はすでに `None`（= config の比率）を返す。`params_table` と `export_code` は、記録があれば記録を、無ければ config を答える。
5. **恒久検査**（`tests/test_persistence/test_reporting_surfaces_survive_load.py`、定義は `_load_census.py`）: 440 セルを実行し、`load()` 前後の読みを比べる。違うセルの集合が宣言した例外の集合と**等しい**ことを確かめる: gain 重要度は相対 5.1e-6 + 絶対 2^-126 以内なら一致と数え、tuning の 3 面は #315 のセルだけが違ってよい。#315 を直すと、このテストがそれを知らせる。`INVENTORY` が `Model` の公開名と一致することも確かめる（新しい公開メソッドは、読むか理由を書くまで失敗する）。空振りの防止として、22 面それぞれに修正前のモデルが読みを返したセルがあること、tuned と config の比率が違うセルが 12 あることも確かめる。

gain の許容差は観測からではなく形式から決める。split の gain g ごとに: 有効数字 6 桁への丸めは最大で 6 桁目の半単位、つまり相対 5e-6 動かす。binary32 への読み戻し（最近接への丸め）は、binary32 の有限の範囲のすべての値 d について |fl(d) − d| ≤ 2^-24·|d| + 2^-150 を満たす。相対の項は正規数の範囲のため、絶対の項（非正規数の間隔の半分）は通常の相対の上限 2^-24 が成り立たない入力（非正規数の範囲全体と、正規数の最小値へ丸め上がる値）のためにある。この補題は丸める前の d について述べる: d = 2^-126 − 2^-150 は正規数 2^-126 に丸め上がり、相対誤差はわずかに 2^-24 を超える（設計レビュー round 3）が、絶対の項がそれを覆う。合わせると |after − g| ≤ C·|g| + 2^-150、C = (1 + 5e-6)(1 + 2^-24) − 1 ≈ 5.0596e-6。特徴の gain は n 個の非負の split gain の和 G なので、C·G + n·2^-150 を超えない。split が 2^24 個未満のモデルなら n·2^-150 < 2^-126（binary32 の最小の正規数）である（倍精度の和の誤差は桁違いに小さい）。テストの許容差は相対 5.1e-6 + 絶対 2^-126。経緯: 初版は 5e-6 と書いて binary32 への読み戻しを落とし（設計レビュー round 1 が 1 split の合成モデルで 5.054e-6 を測った）、第 2 版は相対の項だけを書いて正規数だけを掃いた（round 2 が約 1e-39 の非正規数の gain で 5.602e-6 を測った。新しい上限では 5.605e-45 ≤ 5.763e-45 で内側）。第 3 版は丸めの補題を「結果が正規数かどうか」で分けたが、境界で偽だった（round 3。上限の値は変わらない）。`pr8b_gain_precision.py` は正の有限な binary32 を 10 進の指数 1e-45〜1e38 で掃き（非正規数を含む 334,506 個）、誤差と上限の比の最大は 0.999 で、すべて上限内に収まる。正規数での相対誤差の最大は 5.0544e-6（round 1 の反例と同じ値）。計測器が学習した 3 タスクのモデルでは、split gain はすべて正だった。

### 規則が縛る位置（ソースから導出）

規則: **fit が適用し、学習済み adapter が記録していない training の値は、artifact に記録され、`load()` 後の報告面はその記録を読む。** 導出: `lizyml/` 全体で `applied_training_params` を grep した全件（10 行、`pr8b_measurements.txt` §5）と、artifact を書く・読む 2 つの関数。adapter が記録していない training の値は、training の次元として消費される 2 つの名前のうち `validation_ratio` だけである（`early_stopping_rounds` は adapter の属性）。導出の bound: training の次元名は `_tuning_validation.py` の検査が 2 つに閉じている。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `model.py:160`（初期値） | `{}` のまま（fit 前は読まれない） |
| 2 | `model.py:333`（fit の commit） | `applied_training_overlay` で変換した値を入れる（`_model_factories.py` に新設） |
| 3 | `model.py:921`（`FitState` へ写す） | `None` を写せるようにする |
| 4 | `_model_state.py:52, 53, 76`（`FitState` のフィールドと、`tuning_result` / `applied_training_params` の docstring） | 型を `dict | None` に、docstring の bound を書き換える |
| 5 | `_model_tables.py:301, 314`（`params_table`） | コメントの bound を書き換える（読み方は変えない） |
| 6 | `_model_persistence.py:211, 269`（`export_code`） | 同上 |
| 7 | `_model_persistence.py` `export()` | 記録を exporter に渡す |
| 8 | `exporter.py` `export()` | キーを書く（`None` なら書かない） |
| 9 | `_model_persistence.py` `load()` | 検査して復元する |
| 10 | `_tuning_validation.py:62` | 次元名を定数にする |

### 互換性

- **読み込み**: キーの無い artifact（修正前の export）は今までどおり読める。比率は config に落ちる（決定 13 の bound と同じ）。修正前の export が書く最上位のキー集合を測った（`pr8b_metadata_keys_before.txt`）。修正後の export がその集合にこのキーだけを足すことをテストで確かめ、テストではキーを消した artifact を修正前の artifact として使う（キー集合が同じことは確かめるが、他のキーの値は現行の exporter が書いたものである）。pickle は変わらない（`FitResult` / `RefitResult` に触れない）。
- **書き込み**: 修正後の export は追加のキーを 1 つ書く。現行（`036cd18`）の loader は知らない最上位のキーを無視する（設計レビューが実行で確かめた。それより古いリリースは確かめていない）ので、新しい artifact を現行のライブラリで読むこともできる。
- **振る舞いの変化**: `tune → fit → export → load` の後の `params_table()` の `validation_ratio` 行と `export_code()` が生成する比率が、config の値から fit が使った値に変わる。後者は生成される `train.py` の inner valid の比率を変える（#281 の修正そのもの）。
- **新しく拒否するもの**: 不正な記録を持つ artifact。修正前のライブラリはこのキーを書かず、修正後のライブラリは training が変換した値（`int` の patience と `(0, 1)` の `float` の比率）だけを書くので、拒否されうるのは手で編集された metadata だけである。
- **`FitState.applied_training_params` の型**: `dict` から `dict | None` に広がる。`FitState` は内部の型で、読み手は `tuned_validation_ratio` の 2 か所だけ（すでに `None` を受け取る）。

**Firing rate**: 本提案の条件分岐（キーがあれば復元、無ければ `None`）は、互換性のための読み分けで、Change Gate の 6 つの目的（skip / shorten / cache / select / allow / conditionally-activate）のどれでもない。参考に、直す値の母集団を書く: 全数計測の 440 セル中、#281 のセルは 24（fit が tuning result を消費した 3 lifecycle × 4 構成 × 2 面）。

### 代替案（検討して棄却）

1. **`FitResult` にフィールドを足す。** `FitResult` は公開の Result 契約で、pickle 済みの古い `FitResult` には属性が無いので migration が要る。`metadata.json` のキーなら JSON で読め、H-0083 / H-0086 と同じ追加の形で済む。
2. **load 時に tuning ブロックの `best_training_params` から推測する。** `fit_tune` の artifact で誤る（その fit は overlay を使っていない）。#281 の本文と決定 13 が、まさにこの理由で避けた方法である。
3. **adapter に比率を持たせる。** adapter は推定器の境界で、inner valid の分割は trainer の責務である。比率は adapter の学習に入ってこない値なので、adapter に記録すると責務が混ざる。
4. **「不明」と「overlay なし」を `{}` に潰す。** 修正前の artifact を load して再 export すると、確定した `{}` を偽って書く（決定 2）。
5. **tuning の 3 面もこの PR で扱う。** trial を保存するかどうかは H-0086 の棄却した代替案を覆す判断で、#281 とは別に決める（#315）。

### 受け入れ基準（テスト観点）

詳細と証拠のテスト名は `docs/audits/2026-09-defect-discovery/results/pr8b_acceptance_criteria.md`。

1. `export()` が `applied_training_params` を書く: `fit` → `{}`、`tune → fit` → `best_training_params` と等しい、`fit → tune` → `{}`。値の型は JSON の往復で変わらない。修正前の最上位のキー集合にこのキーだけが加わる。修正前はキーが無いので RED。
2. `tune → fit → export → load` の後、`params_table()` と `export_code()` が tuned の比率を答える（3 タスク）。`load → export → load` で記録が残る。修正前は RED。
3. 記録の無い artifact は読めて config に落ち、状態は `None` で、再 export してもキーを書かない。
4. 不正な記録（dict でない、知らないキー、bool、文字列、null、非有限、範囲外の比率）は `load()` で `DESERIALIZATION_FAILED`。修正前は黙って無視されるので RED。
5. `load()` の後の成功した `fit()` は、その fit が適用した overlay を記録する。`load()` の後の `tune()`、拒否された `fit()`、学習中に失敗した `fit()` は、記録（既知でも不明でも）を変えず、再 export もそれを保つ。
6. 440 セルの恒久検査: 違うセルの集合 = 宣言した例外（gain は相対 5.1e-6 + 絶対 2^-126 以内、tuning の 3 面は #315 のセル）。`INVENTORY` = `Model` の公開名。22 面すべてが読みを返したセルを持つ。修正前は #281 の 24 セルで RED。
7. `report_lifecycle_grid.py` の `known-bound` の 2 セルが `agrees` になる。

## H-0110: 決定済みの提案を BLUEPRINT に畳み込み、全提案の処分を内容で検査する（#271 / PR 9）

- **ステータス**: Accepted
- **起票日**: 2026-10-06
- **決定日**: 2026-10-06（事実確認を含む外部レビュー: 設計レビュー 3 ラウンド（blocking 5 → 3 → 1。round 1 は HISTORY の fence の文法、`[names]` の母集団、決定 4 の受け入れ基準、dataclass の 2 clause、RED の件数。round 2 は同じ id の `- ID:` 行の重複と文書の 2 点。round 3 は形の崩れた `- ID:` 行）で予算を使い切り、最後の修正だけの狭い事実確認が APPROVE。round 2 の前に absolute monitor、round 3 の前に relational monitor（いずれも continue）。記録は `results/pr9_design_review_round{1,2,3}.md`、`pr9_design_monitor_round{2,3}.md`、`pr9_design_fixcheck.md`）
- **スコープ**: `BLUEPRINT.md`（§4〜§19 の 32 節）, `docs/proposal_dispositions.toml`（新規）, `tests/test_docs/test_proposal_blueprint_coverage.py`（新規）, `tests/test_docs/_history_grammar.py`（新規。`test_history_ids.py` の文法を移す）, `tests/test_docs/test_history_ids.py`（import のみ）, `docs/audits/2026-09-defect-discovery/`（計測器・結果・計画・manifest）
- **関連**: [Issue #271](https://github.com/nbx-liz/LizyML/issues/271)（本文と 2026-09-06 のコメント）, 計画 §4 PR 9（Revision 5）, H-0101（HISTORY の id の文法）, H-0083（代表例）
- **実測の記録**: `docs/audits/2026-09-defect-discovery/results/pr9_census_13fb9d7.txt`, `results/pr9_clause_inventory.md`（`results/pr9_inventory_{A,B}.json` から生成）, `results/pr9_dispositions_C.json`
- **管理者決定（2026-10-06、キックオフ）**: 処分は別の TOML ファイルに置く / 検査の対象は Status を問わず全提案 / PR は 1 本で BLUEPRINT の節ごとにコミット / レビュー予算は設計 3 回 + 受け入れ 4 回

### 目的（課題）

HISTORY.md で決まり、実装され、BLUEPRINT.md に畳み込まれなかった決定がある（#271）。代表例は H-0083 で、artifact の `.pkl` ごとの SHA-256 を `metadata.json` に書き、load 時に検証する。`13fb9d7` でも BLUEPRINT は `checksum` / `sha256` / `CHECKSUM_ALGORITHM` に 1 度も触れない。

#271 の母集団（40 entries / 129 clauses / 77 edits、2026-09-06）は、そのまま実行できなかった。129 clause のうち 65 は記録が残っておらず復元できない（`instruments/extract_missed_clauses.py`）。残る 64 も 2026-09-06 の BLUEPRINT に対する判定で、その後 Phase 3 の PR が BLUEPRINT を編集した。HISTORY は 92 件から 110 件に増え、BLUEPRINT が id を引用しない提案は 57 件のままだが中身が変わった（H-0030 / H-0031 が引用され、H-0105 / H-0107 が引用されない）。

そこで `13fb9d7` に対して洗い直した。対象は #271 の義務ありの 40 件と、新たに引用されない H-0105 / H-0107 の 42 件。2 つの read-only の調査が、提案ごとに clause を先に列挙し、BLUEPRINT の行を引いて判定した。正解の分かっている clause 3 件をラベルなしで混ぜ、3 件とも正しい行で `stated` と返った。

| | |
|---|---|
| entries / clauses | 42 / 307 |
| stated / missed / contradicted / superseded / not_in_force / off_surface | 111 / 113 / 14 / 17 / 6 / 46 |
| BLUEPRINT の編集が要る entry | 41（H-0024 だけは義務なし: 7 clause は書かれており、3 clause は H-0099（最適化の向き）と H-0102（space の merge）が置き換えた） |
| 編集が要る clause | 127（32 節） |

表の数は設計レビュー round 1 の後の値である。調査は H-0002 の「`FitResult` / `PredictionResult` / `SplitIndices` / `RunMeta` は dataclass」と H-0082 の「`FitResult` は frozen でない dataclass」を `off_surface` としたが、dataclass であることは golden test が `dataclasses.fields` で固定している公開の表現なので（`tests/test_e2e/test_golden_contracts.py`）、`missed` に直した（round 1 の指摘）。

#271 の 129 より多いのは、列挙をやり直して clause の粒度が細かくなったことと、2 件が加わったためである。どの clause がどの節に入るかは `results/pr9_clause_inventory.md` が一覧にしている。

### 対応方針（決定）

1. **処分ファイル `docs/proposal_dispositions.toml`**。最上位は `[proposals]` と `[names]` の 2 表だけ。`[proposals."H-xxxx"]` は HISTORY の id ごとに 1 行で、`disposition` は次の 4 つのどれか:
   - `specified` + `anchors`（1 個以上）+ 任意の `note`: 各 anchor は BLUEPRINT.md と、その提案自身の HISTORY entry の**両方**に全単語一致で現れる。全単語一致は、前後に `[A-Za-z0-9_]` が無いこと（`checksum` は `checksum_algorithm` に一致しない。DC2）。提案の id は anchor にならない（id の存在は内容の存在ではない。#271）。
   - `no_obligation` + `reason`: CLAUDE.md §3 の面で、いま有効な決定を何もしていない。
   - `superseded` + `superseded_by`（別の提案の id）+ `reason`。
   - `pending` + `reason`: まだ決まっていない。
   知らない key、知らない disposition、空の文字列は失敗にする（読み飛ばさない）。
2. **母集団は HISTORY の登録簿そのもの**。処分の行の集合は HISTORY が宣言する id の集合と両方向で等しい。提案ごとのテストは HISTORY の id で parametrize するので、件数は保存せず HISTORY から毎回再生成する。id の文法は H-0101 の `test_history_ids.py` と同じものを `tests/test_docs/_history_grammar.py` に移して共有する（解析器を 2 つにしない）。Status 行は使わない: 書き方が 9 通りあり、H-0009〜H-0012 と H-0056 は実装済みなのに `proposed` のままである。
3. **洗い直しで `missed` / `contradicted` の 127 clause を BLUEPRINT に書く**。BLUEPRINT とコードが食い違うときは、コードが提案どおりなら BLUEPRINT を直す。BLUEPRINT か提案が書いていてコードがしないものは、どちらが正しいかが判断なので直さず Issue にする（下記）。
4. **提案の無い実装を BLUEPRINT に合わせる 2 点を、ここで決定として記録する**（doc-hierarchy の「仕様が古く実装が正しい場合は HISTORY に記録して BLUEPRINT を更新する」）:
   - multiclass の `PredictionResult.proba` は `(n, k)` を返す。BLUEPRINT §7.3 と H-0002 は binary だけを書いていた。広げた提案は見つからなかった。公開の振る舞いであり、変えれば破壊的になるので、BLUEPRINT を実装に合わせる。
   - BLUEPRINT に残っていたテンプレートの仮名を実際の名前にする: `yourlib_version` → `lizyml_version`、`deps_version` → `deps_versions`（`core/types/artifacts.py`）、`YourLibError` → `LizyMLError`（`core/exceptions.py`、`lizyml/__init__.py` が公開）。
5. **#271 が挙げた未文書化の公開名**は `[names]` 表に置く。`documented`（`where` = `BLUEPRINT.md` か `docs/api.md` に全単語一致）か `internal` + `reason`。`CHECKSUM_ALGORITHM` は H-0083 の畳み込みで BLUEPRINT に書く。`SUPPORTED_CONFIG_VERSIONS` はすでに BLUEPRINT にある。`TASK_TYPES` は `lizyml` から再公開されず、公開名は `TaskType` なので internal。`plots/_theme.py` の 3 定数はモジュール自体が private なので internal。
6. **洗い直しの対象外の 68 件**は clause ごとには監査しない。処分と anchor を決める（`results/pr9_dispositions_C.json`。specified 64 / superseded 2 / no_obligation 2 / pending 0）。ただし、その調査が「決まっていて BLUEPRINT に無い」と報告した 5 件（H-0057 の covered OOF の NaN 拒否、H-0085 の数値 target の NaN 拒否、H-0087 の 3 つの validator と fit に配線しない規則、H-0104 の `transform_with_warnings` と `get_state` の `categorical_cols`、H-0106 の `INCOMPATIBLE_COLUMNS` / `METRIC_REQUIRES_PROBA` を出す条件）は #271 と同じ欠陥なので、あわせて BLUEPRINT に書く。
7. **anchor は畳み込みを示せるものを選ぶ。** BLUEPRINT にすでにある語だけを anchor にすると、畳み込みの前後どちらでも検査が通る。`13fb9d7` で実測すると、書き足しが要る 41 件のうち 23 件がそうだった。そこで、書き足しが要る 46 件（41 件 + 上の 5 件）は、それぞれ少なくとも 1 つ、`13fb9d7` の BLUEPRINT に無く、畳み込みが書く anchor を持つ（`instruments/pr9_discriminating_anchors.py --base 13fb9d7` が exit 0）。
8. **H-0110 自身の行**は `specified` で、anchor は BLUEPRINT に書く `proposal_dispositions.toml`。以後、新しい提案を足す PR は、その提案の行を同じ PR で足す。

Firing rate: 4/110 of HISTORY proposals at 13fb9d7 carry an exempt disposition (H-0010, H-0044 superseded; H-0037, H-0075 no_obligation; counted from `docs/proposal_dispositions.toml`, `results/pr9_measurements.txt` item 5)

### 影響範囲 / 互換性

- コードは変えない。公開 API / Config / FitResult / PredictionResult / Artifacts の形と意味は変わらない。BLUEPRINT が実装を述べるようになるだけである。`format_version` は変えない。
- 新しい依存は無い。TOML は Python 3.11 以上で標準の `tomllib`、3.10 では `tomli` で読む。`tomli` は 3.10 で pytest 自身が依存している（`uv.lock`）ので、テストが走る環境には必ずある。
- 新しい提案を足す PR は、`docs/proposal_dispositions.toml` に 1 行足さないと CI が落ちる。これが恒久検査の意図である。

### 代替案（不採用）

- **#271 の 129 clause をそのまま実行する**: 65 clause は記録が無く、残りも古い BLUEPRINT に対する判定である。
- **id の引用を検査する**: #271 が示したとおり、33 件は id 無しで完全に書かれ、`(H-0083)` を 1 つ足せば checksum が書かれないまま通る。
- **処分を HISTORY の各 entry に 1 行ずつ書く**: 110 entry を編集し、HISTORY の文法が開く（管理者決定で不採用）。
- **Status が accepted / implemented の提案だけを検査する**: Status の文法と、実態とずれた Status に依存する（管理者決定で不採用）。
- **anchor を BLUEPRINT にだけ求める**: BLUEPRINT にある無関係な語で通る。提案自身の entry にも求めれば、anchor が提案の決定と結びつく。

### 受け入れ基準（テスト観点）

1. `test_every_proposal_has_exactly_one_row`: 処分の行 = HISTORY の id（両方向）。行の無い提案、提案の無い行はそれぞれ名前付きで失敗する。
2. `test_proposal_disposition_holds[H-xxxx]`: HISTORY の id ごとに 1 件で、件数は HISTORY から導出する（`13fb9d7` で 110）。全件が通る。
3. 文法の拒否（単体）: 上位文字列の不一致（`checksum` / `checksum_algorithm` など 5 例）、境界のある一致（4 例）、不正な行 15 種、両方向の差、`[names]` の不正 4 種。
4. **RED**: 未編集の BLUEPRINT（`13fb9d7`）に対して、提案の行 19 件と `[names]` の `CHECKSUM_ALGORITHM` の 20 件が失敗する（`0655e05` で実測）。19 件の内訳は、書き足しが要る entry 18 件と H-0110 自身の行（anchor `proposal_dispositions.toml` がまだ BLUEPRINT に無い）。書き足しの要らない既存の提案の行は 1 件も失敗しない（`results/pr9_measurements.txt` item 6）。畳み込みの後は、書き足しが要る 46 件すべてが、`13fb9d7` に無かった anchor を持つ（方針 7）。マージ後の木では、H-0083 の checksum の記述を BLUEPRINT から消す変更で失敗する（phase3 manifest の `red_mutation`。マージ前の木で走らせる RED は、`.py` でない `docs/proposal_dispositions.toml` が before tree に入らないため、欠陥ではなくファイルが無いことで失敗するだけになる）。
5. `test_public_name_dispositions_hold`: `[names]` の 6 名すべてが成り立つ。
6. 畳み込む 127 clause、方針 6 の 5 件、決定 4 の 2 点（multiclass の `proba`、仮名 3 つの改名。`lizyml_version` / `deps_versions` は H-0002 の RunMeta の clause と重なるので、その行で両方を示す）、方針 8 の 1 行、それぞれについて、BLUEPRINT のどの行が述べるかを `results/pr9_fold_map.md` が示す（レビューで確かめる宣言。計測器は anchor の存在までしか読めない）。
7. `instruments/pr9_discriminating_anchors.py --base 13fb9d7` が exit 0（46/46）。

### 本 PR で直さず Issue にするもの

- **[#318](https://github.com/nbx-liz/LizyML/issues/318)**: BLUEPRINT か提案が書いていてコードがしないもの（`not_in_force` の 6 件のうち、決定 4 で扱う multiclass の `proba` を除く 5 件と、畳み込みの作業で見つかった H-0078 の 1 件: 両側が境界に当たったときに `expanded=False` を再判定するという約束を `detect_boundary` が守らない）: `embargo_pct` の `int()` 変換（コードは端数を拒否、#210）、plot を `{output_dir}/{run_id}/` に保存（コードは `run.log` だけ書く）、`migrations/v1_to_v2.py` の追加（load 時の自動 migration の半分は H-0070 で有効）、Metric の `supports_task` 属性（BLUEPRINT は Metric IF の属性として挙げるがコードに無い）、`METRIC_NOT_FOUND`（その ErrorCode は無く、実際は `UNSUPPORTED_METRIC`）。この 5 件は、どちらが正しいかが判断なので、BLUEPRINT は直さない。これとは別に、`evaluate_table()` の `cal_fold_*` 列は BLUEPRINT にだけあり、H-0005 は `cal_oof` 列だけを決め、コードもそうしているので、この 1 件は Issue にせず、畳み込みとして BLUEPRINT を直す。
- **[#319](https://github.com/nbx-liz/LizyML/issues/319)**: HISTORY の Status 行が実態とずれている 7 件（H-0009〜H-0012、H-0056、H-0097、H-0098）、コードが送出しないエラー名を書く 2 件（H-0057 の `ValueError`、H-0009 の `INVALID_CONFIG`。BLUEPRINT にはコードどおり `EVALUATION_FAILED` / `CONFIG_INVALID` を書いた）、H-0080 の entry の中にある、seed と関係の無いパラメーター名の経路の節（HISTORY 6664〜6681 行）。

## H-0111: 仕様とコードのずれ 6 件を決着させ、HISTORY の記録のずれを直す（#318 / #319 / #321）

- **ステータス**: Accepted
- **起票日**: 2026-10-06
- **決定日**: 2026-10-06（管理者の判断: #318 は「行 1〜5 は文書をコードに合わせ、行 6 はコードを H-0078 に合わせる」、#321 は選択肢 (a)）
- **スコープ**: `lizyml/tuning/search_space.py`（`detect_boundary` の再判定）, `lizyml/core/_model_tuning.py`（ログの文言）, `BLUEPRINT.md`（§10 の `embargo_pct`、§11 の境界拡張、§13.1 の Metric IF、§17 の `output_dir`）, `HISTORY.md`（H-0003 / H-0009〜H-0012 / H-0034 / H-0056 / H-0057 / H-0071 / H-0080 / H-0097 / H-0098 への注記と Status の訂正）, `docs/proposal_dispositions.toml`, テスト（`tests/test_tuning/test_retune.py`、`tests/test_tuning/test_dimension_consumption.py`）, `docs/audits/2026-09-defect-discovery/`（manifest の #279 行、`MANIFEST.md`、`phase3-plan.md` §8、計測器 `instruments/h0111_pinned_dims.py`）, `CHANGELOG.md`
- **関連**: [Issue #318](https://github.com/nbx-liz/LizyML/issues/318), [Issue #319](https://github.com/nbx-liz/LizyML/issues/319), [Issue #321](https://github.com/nbx-liz/LizyML/issues/321), H-0078（項目 4: 両側が境界に当たったときの `expanded=False` の再判定）, H-0110（本提案の 3 件を Issue に切り出した畳み込み）

### 目的（課題）

H-0110 の畳み込みは、決定済みで実装済みの節を BLUEPRINT に写した。そのとき、BLUEPRINT か提案が書いていてコードがしない 6 件（#318）と、HISTORY 自身の記録のずれ（#319）を、どちらが正しいかの判断が要るので Issue に切り出した。PR 9b の後に、Phase 3 の manifest の母集団を定数で宣言した行が、文書やコードの成長で古くなりうること（#321、DC3）も分かった。本提案はこの 3 件を決着させる。

### 対応方針（決定）

1. **#318 行 1〜5: 文書をコードに合わせる。** どれもコードの挙動が意図どおりで、文書だけが古い。
   - 行 1（H-0040）: 旧キー `embargo_pct` は `int()` で変換しない。整数値（`3` / `3.0`）は受理し、端数（`0.05`）と bool は `CONFIG_INVALID` で拒否する（#210。切り捨てると漏洩防止の gap が `0` に潰れる）。BLUEPRINT §10 を直す。
   - 行 2（H-0034）: `fit()` / `tune()` が `{output_dir}/{run_id}/` に自動で書くのは `run.log` だけ（path 無しの `export()` は `{run_dir}/export` に書く、H-0039）で、plot は保存しない（plot API は plotly の Figure を返す。`lizyml/` に `write_html` / `write_image` は無い）。BLUEPRINT §17 を直し、H-0034 に注記する。
   - 行 3（H-0003）: `migrations/` パッケージは無い。v1 の artifact は loader の中でメモリ上で引き上げる（H-0070、BLUEPRINT §15.2 は既にそう書く）。H-0003 に注記する。
   - 行 4（H-0014）: `BaseMetric` に `supports_task` は無い。タスクとの対応はレジストリが持ち、扱わないタスクで引くと `UNSUPPORTED_METRIC`。BLUEPRINT §13.1 の Metric IF の属性一覧から外す。
   - 行 5（H-0071 INV-4）: `METRIC_NOT_FOUND` という `ErrorCode` は無く、送出されるのは `UNSUPPORTED_METRIC`。BLUEPRINT は H-0110 で既に正しい。H-0071 に注記する。
2. **#318 行 6: コードを H-0078 項目 4 に合わせる。** H-0078 は「両側がぶつかった場合は元値のまま返し、無限ループ防止のため `expanded=False` を上位で再判定する」と決めたが、`detect_boundary` は端に近い次元をクランプのあとも `expanded=True` とし、`expanded_names` に入れていた。既に `min_allowed` / `max_allowed` に貼り付いた次元は、`tune(resume=True)` のたびに範囲の変わらない「拡張」として報告され続ける。これは新しい決定ではなく、H-0078 の決定の実行である。規則: **クランプと `IntDim` の丸めのあとで `(new_low, new_high)` が元の `(low, high)` と等しいなら、`expanded=False`、`new_low` / `new_high` は `None`、`expanded_names`（したがって `RoundSummary.expanded_dims`）に入れない。** `clamped_to_bound` は変えない: `min_allowed` / `max_allowed` のクランプが効いたなら `True` のまま残す（下流 UI の「上限に達した」表示の根拠）。linear の `0.0` 下限と `IntDim` の `max(1, ...)` はもともと `clamped_to_bound` を立てないので `False` のまま。`IntDim` の拡張は整数で計算する（コードレビュー round 1: float を経由すると `2**53` を超える `high` と `high + 1` が同じ値になり、まだ 1 つ広げられる次元を「範囲が変わらない」と誤判定した）。
   - **適用範囲を H-0078 の文面より広げる。** H-0078 は `min_allowed` / `max_allowed` だけを書くが、同じ理由（範囲の変わらない拡張を毎ラウンド繰り返さない）は、それより古い 2 つのクランプにも当てはまる: linear の下限 `0.0`（#110）と `IntDim` の `max(1, ...)`。原因を問わず「範囲が変わらない」で判定する。`tests/test_tuning/test_retune.py::test_int_dim_lower_guard` は `IntDim(low=1)` の空の拡張を `expanded=True` として固定していたので、ガードが効きつつ範囲が動く `low=2` に変え、`low=1` は新しいテストで `expanded=False` を固定する。
3. **#319: HISTORY の記録を注記で直し、本文は書き換えない。** Status 行の 7 件（H-0009 / H-0011 / H-0012 / H-0056 / H-0098 は `accepted`、H-0010 は `superseded`（H-0069）、H-0097 は Revision 2 が `accepted`（PR #290、merge `2436a66`）で元の提案は Revision 2 に置き換えられた）。コードが送出しないエラー名の 2 件（H-0057 の `ValueError` → `EVALUATION_FAILED`、H-0009 の `INVALID_CONFIG` → `CONFIG_INVALID`）は、該当行に「H-0111 注記」を付ける。H-0080 の中にある LightGBM のパラメーター名の経路の節は、H-0093 の検討の一部で、H-0093 が `ESTIMATOR_ROUTES` を決めてこの節の 3 か所を 6 か所に改めた。節は移動せず、冒頭に注記を置く（移すと H-0093 の改訂済みの記述と古い 3 か所の表が並ぶため）。
4. **#321: 選択肢 (a)。** 定数の `population` を持つ 6 行のうち、5 行はテスト自身が宣言する設計上の件数で、文書やコードの成長では変わらない。残る #279 の行（smart parameter が主張する native の綴り 18 個）は、provider の alias 表から数えるので、#266 と同じ形で古くなりうる。これを `derived_from`（`len(CLAIMED)`）に変える。導出は自分自身を数えるので、綴りが減っても気づけない。PR 9b が #266 で `MIN_SITES` を置いたのと同じく、`MIN_CLAIMED = 18` の下限をテストに置く。

### 規則が縛る位置（ソースから導出）

規則 2 が縛るのは、境界の報告を作る位置と、それを読む位置である。導出: `lizyml/` 全体で `expanded_names` / `.expanded` / `detect_boundary` / `expand_dims` を grep した全件。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `tuning/search_space.py` `detect_boundary` | 再判定を足す（唯一の書き手） |
| 2 | `tuning/search_space.py` `expand_dims` | 変えない（`expanded=True` の次元だけを読むので、再判定した次元は素通しになる） |
| 3 | `core/_model_tuning.py` `_maybe_expand_boundary` | ログの文言を「端に近い次元が無い」から「拡張した次元が無い」に変える（端に近くても貼り付いた次元だけのときに偽になるため） |
| 4 | `tuning/rounds.py` / `core/_model_tuning.py`（`RoundSummary.expanded_dims`、進捗の `expanded_dims`） | 変えない（`expanded_names` を写すだけ） |
| 5 | `core/_model_tables.py` `boundary_table` | 変えない（`expanded` 列は再判定後の値を表示する） |

### 互換性

- **振る舞いの変化**: 端に近く、かつ既に境界に貼り付いた次元について、`BoundaryDimStatus.expanded` が `True` から `False` に、`new_low` / `new_high` が元の値から `None` に変わり、`expanded_names`・`RoundSummary.expanded_dims`・`boundary_table()` の `expanded` 列・進捗コールバックの `expanded_dims` から消える。探索空間そのものは変わらない（範囲が変わらない拡張を素通しにするのは修正前の `expand_dims` の結果と同じ）。
- **該当する次元の数（参考）**: LightGBM の既定の探索空間に provider の境界を付けると、3 タスクとも数値の 9 次元のうち 2 次元（`feature_fraction` と `bagging_fraction` の上限 1.0）が最初から上限に貼り付いている（`instruments/h0111_pinned_dims.py`）。最良値がその上端に来た `tune(resume=True)` で、修正前は毎ラウンド報告されていた。
- **Firing rate**: 本提案の分岐は、拡張と非拡張という通常の 2 つの振る舞いのどちらを報告するかの訂正で、Change Gate の 6 つの目的（skip / shorten / cache / select / allow / conditionally-activate）のどれでもない。
- `format_version` / Config / 公開 API の形は変わらない。文書の訂正（方針 1・3）は振る舞いを変えない。
- **bound（宣言）**: 規則 2 は「拡張の結果が元の範囲と等しいか」を厳密に判定する（`IntDim` は整数で計算する）。その前段の端の検出（`_position_pct` / `_detect_edge`）は H-0068 のまま float で行い、本提案は変えない。したがって、`2**53` を超える `IntDim` の範囲（例: `[2**53, 2**53 + 1]`）では、最良値が端にあっても位置が潰れて端を検出しないことがある（コードレビュー round 2 が実行で示した。修正前から同じで、本提案が持ち込んだものではない）。LightGBM のパラメーターの境界は `seed` の `2**31 - 1` が最大だが、provider の境界は拡張だけを縛り、利用者が書いた初期の範囲は検証もクランプもしない（`parse_space` と `attach_bounds` は `seed` の `[2**53, 2**53 + 1]` を受け取ったまま `max_allowed=2147483647` を付ける。コードレビュー round 3 が実行で示した）。したがって、利用者が `2**53` を超える整数の範囲を書けば、境界の有無に関わらずこの限界に届く。provider の境界が付いた名前では、その範囲はすでに意味のある範囲の外である。端の検出を整数で厳密にすることは本提案の範囲外とする。

### 代替案（検討して棄却）

1. **行 6 も文書をコードに合わせる。** H-0078 の再判定は、無限ループと誤った報告を防ぐために決めたもので、それを外す理由が無い。
2. **H-0078 の文面どおり `min_allowed` / `max_allowed` だけを再判定する。** `IntDim(low=1)` と linear の `low=0.0` の空の拡張は報告され続ける。理由が同じなので規則を分けない。
3. **再判定した次元の `new_low` / `new_high` に元の値を入れる。** 端に近くない次元と categorical は `None` を入れており、「`expanded=False` なら `None`」の形に揃える。
4. **#319 の H-0080 の節を H-0093 に移す。** H-0093 は同じ経路を走査の修正後の 6 か所で書いており、古い 3 か所の表を並べると矛盾した記述が 1 つの entry に入る。
5. **#321 で (b)（PR ごとに `phase3_gap.py` を回す）も行う。** 計測器は worktree と GitHub API を使うので CI で回せず、手作業の規則になる。(a) で古くなりうる行が無くなるので、管理者は (a) だけを選んだ。

### 受け入れ基準（テスト観点）

1. `max_allowed` / `min_allowed` に貼り付いた次元（float の上限・下限、log の上限、int の上限）で、端に近い最良値を与えると `expanded=False`、`clamped_to_bound=True`、`new_low` / `new_high` は `None`、`expanded_names == ()`、`expand_dims` は次元をそのまま返す（`TestNoOpExpansionIsNotExpanded`）。修正前は RED。
2. linear の `low=0.0` と `IntDim(low=1)` の下端で `expanded=False`、`clamped_to_bound=False`。修正前は RED。
3. クランプが効いても端が動く次元は `expanded=True`、`clamped_to_bound=True`。貼り付いた次元と動く次元が混ざると、`expanded_names` は動く次元だけ。
4. `IntDim(low=2)` の下端の拡張は `max(1, ...)` で `new_low == 1` になる（ガードのテストを保つ）。
5. manifest の #279 行は `derived_from` を持ち `population` を持たない。そのスニペットは HEAD で `18` を出す。`test_claimed_spellings_are_found` は綴りが 18 未満になると落ちる。
6. `tests/test_docs/test_history_ids.py` と `test_proposal_blueprint_coverage.py` が緑のまま（H-0111 の処分の行を含む）。

## H-0112: 漏洩検査は、名指しされた列が無いときに「問題なし」と答えない（#311）

- **ステータス**: Accepted
- **起票日**: 2026-10-06
- **決定日**: 2026-10-06（管理者の判断: target 列が無いときは `DATA_SCHEMA_INVALID` を送出する。同じ形の `validate_time_series_order` の `time_col` も同じ PR で扱う）
- **スコープ**: `lizyml/data/validators.py`（`validate_no_target_leakage` と `validate_time_series_order` の早期 `return []`）, `BLUEPRINT.md` §8.2, `docs/api.md`（漏洩検査の節と `DATA_SCHEMA_INVALID` の行）, `CHANGELOG.md`, テスト（`tests/test_data/test_validators_edge.py` の 2 件を決定に合わせて更新、`tests/test_data/test_leakage_validator_missing_column.py` を新規）, 計測器 `docs/audits/2026-09-defect-discovery/instruments/h0112_missing_column_plugin.py`
- **関連**: [Issue #311](https://github.com/nbx-liz/LizyML/issues/311), H-0087（3 つの検査を公開 API にした）, H-0107（比較できない列を `DATA_SCHEMA_INVALID` にした。`raise_on_violation` によらず送出する前例）

### 目的（課題）

`lizyml.data.validate_no_target_leakage(df, target)` は、`target` が `df` の列に無いと最初の行で `return []` する。全列を比べて漏洩が無かったときと同じ `[]` なので、目的変数の名前を打ち間違えた呼び出しや、目的変数を含まない frame を渡した呼び出しは、何も比べていない検査から「問題なし」を受け取る（DC1）。H-0107 は比較できない列についてこれを塞ぎ、「target が frame にあるとき、戻り値が返るのは全列を比べたときだけ」を保証したが、target が無いときは #311 に残した。

`validate_time_series_order(df, time_col)` も、`time_col` が列に無いと `return []` する。並びを何も調べていないのに、並んでいたときと同じ `[]` を返す。同じ形なので、管理者の判断で同じ提案で扱う。3 つ目の `validate_group_split(groups, train_idx, valid_idx)` は列名を取らないので対象外。

### 対応方針（決定）

1. **名指しされた列が `df` に無ければ `LizyMLError(DATA_SCHEMA_INVALID)` を送出する。** `validate_no_target_leakage` は `target`、`validate_time_series_order` は `time_col` について。`user_message` は列名と、検査が何も調べていないことを書く。`context` は `dataframe_builder` の列の欠落と同じ形 `{"missing_columns": [<name>], "available_columns": list(df.columns)}` に、役割のキー（`"target"` / `"time_col"`）を加える。
2. **`raise_on_violation` によらず送出する。** `raise_on_violation=False` の戻り値は「漏洩の疑い」の警告のリストで、「検査できなかった」を同じリストに入れると、呼び出し側が文言を読み分けない限り 2 つが混ざる（H-0107 決定 1 と同じ理由）。
3. **検査は列の有無を最初に、完全なラベルで確かめる。** 列が無ければ、どの列も比べず、並びも調べずに送出する。有無は `df.columns.to_flat_index()` に対して調べる: `MultiIndex` の列では `"a" in df.columns` が部分キーで真になり、存在しない列について部分 frame を検査して `[]` を返していた（コードレビュー round 1 が実行で示した）。したがって、2 つの検査の戻り値（`[]` か警告のリスト）は「名指しされた列があり、検査を最後まで行った」ことを意味する。

### 規則が縛る位置（ソースから導出）

規則: **漏洩検査は、名指しされた列が無いことを「問題なし」として返さない。** 導出: `lizyml/data/validators.py` で `not in df.columns` と `return []` を grep した全件（`fe20bf6`）。`not in df.columns` は 2 か所で、どちらも直後に `return []` する。残りの `return []` は 2 つで、`validate_time_series_order` が並びを確かめたあと（`:48`）と、`validate_group_split` が重なりを確かめたあと（`:173`）の、検査を終えた正常な戻り値である。`lizyml/` の中にこの 2 つの検査を呼ぶ箇所は無い（`Model.fit` には配線しない、H-0087）。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `data/validators.py` `validate_time_series_order` の `if time_col not in df.columns: return []` | 送出に置き換える |
| 2 | `data/validators.py` `validate_no_target_leakage` の `if target not in df.columns: return []` | 送出に置き換える |

### 互換性

- **振る舞いの変化**: 名指しされた列の無い frame を渡すと、`[]` が返っていたところで `DATA_SCHEMA_INVALID` が送出される。`raise_on_violation=False` でも同じ。列がある呼び出しの振る舞いは変わらない。
- **影響を受ける呼び出し（計測）**: フルスイートで 2 つの検査の呼び出しを数えた（`instruments/h0112_missing_column_plugin.py`、`fe20bf6`）。`validate_no_target_leakage` 46 回中 1 回、`validate_time_series_order` 5 回中 1 回で列が無く、その 2 回はどちらも今日の `[]` を固定しているテスト（`tests/test_data/test_validators_edge.py::test_leakage_missing_target` / `test_time_series_missing_col`）である。この 2 件は削除せず、決定に合わせて書き換える。
- **Firing rate**: 本提案の条件（列が無ければ送出）は入力の検証であり、Change Gate の 6 つの目的（skip / shorten / cache / select / allow / conditionally-activate）のどれでもない。参考の計測は上のとおり 2/51。
- `format_version` / Config / `Model` の公開 API は変わらない。2 つの関数のシグネチャも変わらない。

### 代替案（検討して棄却）

1. **`[]` を返したまま docstring と `docs/api.md` に書く。** 呼び出し側は戻り値だけでは「検査した」と「検査しなかった」を区別できず、文書を読まない呼び出しには DC1 が残る。
2. **`raise_on_violation=False` のときは警告をリストに入れる。** H-0107 決定 1 と同じ理由で、「漏洩の疑い」と「検査できなかった」が 1 つのリストに混ざる。
3. **`LEAKAGE_SUSPECTED` で送出する。** 列が無いのは漏洩の疑いではなく、入力の形の誤りである。他の列の欠落（`dataframe_builder`、`column_check`）と同じ `DATA_SCHEMA_INVALID` にそろえる。
4. **`validate_no_target_leakage` だけを直す。** `validate_time_series_order` に同じ形が残る。管理者が両方を選んだ。

### 受け入れ基準（テスト観点）

1. `validate_no_target_leakage(df, "<無い列>")` は `raise_on_violation` が `True` でも `False` でも `DATA_SCHEMA_INVALID` を送出し、`context["target"]` と `context["missing_columns"] == [<名前>]`、`context["available_columns"] == list(df.columns)` を持つ。修正前は `[]` を返すので RED。
2. `validate_time_series_order(df, "<無い列>")` も同じ（`context["time_col"]`）。修正前は RED。
3. 大文字小文字だけが違う名前（`"Y"` と `"y"`）は無い列として扱う（打ち間違いの例。#311 の再現）。`MultiIndex` の列で部分キー（`"a"`）は無い列として扱い、完全なラベル（`("a", "x")`）は今日と同じく検査する。
4. 列があるときの振る舞いは変わらない: 漏洩・並びの乱れがあれば今日と同じ `LEAKAGE_SUSPECTED` か警告、無ければ `[]`。
5. `tests/test_data/test_validators_edge.py` の 2 件は削除せず、決定に合わせて送出を確かめる形に書き換える。
6. `docs/api.md` の `ErrorCode` の表と漏洩検査の節、BLUEPRINT §8.2 が新しい振る舞いを書き、`tests/test_docs/` が緑のまま（H-0112 の処分の行を含む）。

## H-0113: 確率校正の最適化が収束しなかったら、その係数を使わずに送出する（#297）

- **ステータス**: Accepted
- **起票日**: 2026-10-06
- **決定日**: 2026-10-06（管理者の判断: 既定の設定での発火率を測り、0 なら送出する。送出する code は新設の `ErrorCode.CALIBRATION_FAILED`）
- **スコープ**: `lizyml/core/exceptions.py`（`CALIBRATION_FAILED` を追加）, `lizyml/calibration/_optimizer.py`（収束の検査）, `lizyml/calibration/platt.py` / `beta.py`（検査を呼ぶ）, `lizyml/calibration/cross_fit.py`（失敗した fold を context に付ける）, `lizyml/codegen/templates.py`（生成される `train.py` の校正器）, `BLUEPRINT.md`（§12 の校正、§16.2 の `ErrorCode` 一覧）, `docs/api.md`, `CHANGELOG.md`, テスト（`tests/test_calibration/test_calibration_failed_optimisation.py` 新規、`tests/test_codegen/test_calibration_params_codegen.py` 追記、`ErrorCode` の一覧を固定する `tests/test_core/test_exceptions.py` と、各 code を実際に送出させる `tests/test_core/test_error_code_raising.py` に 1 行ずつ追加）, 計測器 `docs/audits/2026-09-defect-discovery/instruments/i297_minimize_census_plugin.py`
- **関連**: [Issue #297](https://github.com/nbx-liz/LizyML/issues/297), H-0100（`calibration.params` で最適化の設定を受け付けた）, H-0058（校正は outer split を使う cross-fit）

### 目的（課題）

`PlattCalibrator.fit` と `BetaCalibrator.fit` は `scipy.optimize.minimize` の結果の `x` を、`success` を見ずに係数として使う。最適化が収束しなかったとき（反復の上限、関数評価の上限、線探索の失敗）、校正器は途中の係数、最悪の場合は初期値（Platt なら `a = 0` と事前確率の対数オッズ）を持ったまま、成功した校正器として `fit()` から返り、C_final として export される。警告は出ない（DC1）。生成される `train.py` の校正器も同じ形である。

#297 は `calibration_params={"options": {"maxls": 1}}` で Platt の 3 fold と C_final がすべて `success=False`（`ABNORMAL:`）のまま初期値を出荷したことを示した。scikit-learn の `_sigmoid_calibration` も `success` を見ないので、既定の振る舞いは参照実装と同じだが、失敗した fit を成功として扱うことに変わりはない。

### 発火率（実装前に計測）

`instruments/i297_minimize_census_plugin.py` で、フルスイート（`113d20a`）の `minimize` の呼び出しをすべて記録し、呼び出し元と `success` を数えた。校正器が利用者の `calibration.params` を持つ呼び出しを「custom」、持たない呼び出しを「default」とした。

```
platt  default  360 呼び出し  失敗 0
beta   default   37 呼び出し  失敗 0
platt  custom    50 呼び出し  失敗 0
beta   custom    27 呼び出し  失敗 0
生成コード        12 呼び出し  失敗 0
```

Firing rate: 0/397 of default-setting calibrator minimize calls in the full test suite (`instruments/i297_minimize_census_plugin.py` at `113d20a`; 0/77 custom, 0/12 generated)

既定の設定で今日成功している fit を、送出に変えて壊すことはこの母集団では起きない。一方、利用者が `options` で反復の上限などを付ければ失敗は起きる: 同じデータで `{"options": {"maxiter": 1}}` は Platt と Beta の両方で `success=False`（`STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`）、`{"options": {"maxls": 1}}` は Beta で `ABNORMAL:` になる（計測器の確認用の実行）。テストのデータは合成データで、実データのスコア分布で既定の設定が失敗するかは測っていない。

### 対応方針（決定）

1. **`minimize` の `success` が偽なら `LizyMLError(CALIBRATION_FAILED)` を送出し、その係数を校正器に残さない。**`fit()` は始めに前の係数を消すので、成功した fit のあとの refit が失敗した場合も、前の fit の係数で `predict()` / `export_params()` が動くことはない（コードレビュー round 1 が、成功のあとの失敗で前の係数が残ることを実行で示した）。 Platt と Beta の両方。理由（反復の上限、関数評価の上限、線探索の失敗）は区別しない: どれも、返った係数が尤度の最大点であることを示していないので、校正として使えない。`context` は `calibrator`（`"platt"` / `"beta"`）、`method`、scipy の `message` と `status`、`nit`（反復回数）を持つ。`user_message` は、`calibration.params` の `options` が最適化を制限しているなら緩めるよう書く。
2. **`ErrorCode.CALIBRATION_FAILED` を新設する。** 既存の code は意味が合わない: `CONFIG_INVALID` は設定の誤りで、既定の設定での失敗には当たらない。`CALIBRATION_NOT_FITTED` は fit 前の呼び出しを表す。enum への追加だけで、既存の code は変わらない。
3. **cross-fit では、どこで失敗したかを付ける。** `cross_fit` は fold の校正器の失敗に `context["stage"] = "cross_fit"` と `context["fold"]`（0 始まりの fold 番号）を、C_final の失敗に `context["stage"] = "c_final"` を加えて送出し直す（`cause` は元の例外）。他の `LizyMLError` と例外はそのまま通す。
4. **生成される `train.py` も同じ規則に従う。** 生成コードは LizyML に依存しないので、`_run_minimize` が `success` の偽を `RuntimeError`（`calibration (<method>) did not converge: <message>`）にする。生成コードの他の拒否（`ValueError`）は設定の誤りを表し、これは実行時の失敗なので型を分ける。
5. **isotonic は対象外。** `minimize` を使わない。

### 規則が縛る位置（ソースから導出）

規則: **確率校正の係数は、`minimize` が成功を報告したときだけ使う。** 導出: `lizyml/` で `minimize(` を grep した全件（`113d20a`）。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `calibration/platt.py` `PlattCalibrator.fit` | 検査を足す |
| 2 | `calibration/beta.py` `BetaCalibrator.fit` | 検査を足す |
| 3 | `codegen/templates.py` `_run_minimize`（生成コードの Platt と Beta の両方が通る） | 検査を足す |
| 4 | `calibration/_optimizer.py` の `options` の名前の検査（小さな 2 次の問題で 1 回 `minimize` を実行し、未知の option の警告を拒否に変える。結果の係数は使わない） | 変えない: この呼び出しは校正の係数を作らない |

### 互換性

- **振る舞いの変化**: `minimize` が収束しなかった Platt / Beta の校正は、今日は途中の係数で成功していたところで `CALIBRATION_FAILED` を送出する（`Model.fit` の校正の段階で、cross-fit の fold か C_final か）。上の計測では、既定の設定でもテストの `calibration.params` でもこの分岐に入る呼び出しは 0 件である。入るのは、利用者が `options` で最適化を強く制限した場合と、既定の設定で収束しない（未測定の）実データの場合である。後者で今日得ていたのは、尤度の最大点ではない係数である。
- **生成コード**: 同じ場合に、生成される `train.py` は `RuntimeError` で止まる。
- **公開 API**: `ErrorCode` に `CALIBRATION_FAILED` が加わる。`docs/api.md` と BLUEPRINT §16.2 の一覧を更新する（`tests/test_docs/test_error_code_docs.py` が enum と一致させる）。`format_version` / Config は変わらない。

### 代替案（検討して棄却）

1. **警告して結果を使う。** 校正済みの指標と C_final が、尤度の最大点でない係数で計算・出荷される。警告は読まれないことがあり、#297 の「失敗した fit を成功として扱う」が残る。
2. **警告して、文書化した形（未校正の確率）に落とす。** cross-fit にはすでに fold の fallback（単一クラス、学習行なし）があるが、それはデータの構造で校正できない場合のためで、最適化の失敗を同じ扱いにすると、利用者の設定ミスが「一部の fold は未校正」という目立たない結果に変わる。
3. **反復の上限で止まった場合だけは許す。** 上限で止まった係数も最大点ではない。利用者が上限を付けたのが意図でも、校正としての品質を保証できない。上限を緩めるよう `user_message` で示す。
4. **`CONFIG_INVALID` を使う。** 既定の設定での失敗まで「設定の誤り」になる。

### 受け入れ基準（テスト観点）

1. 本物の失敗した `minimize`（`{"options": {"maxiter": 1}}`、Platt と Beta）で `fit()` が `CALIBRATION_FAILED` を送出し、`context` に `calibrator` / `method` / `message` / `status` / `nit` を持つ。係数は設定されず、`predict()` は `CALIBRATION_NOT_FITTED`。成功した fit のあとに同じ校正器で失敗した refit も、`predict()` / `export_params()` を `CALIBRATION_NOT_FITTED` にする。修正前は fit が成功するので RED。
2. `cross_fit_calibrate` で fold の失敗は `stage="cross_fit"` と `fold` を、C_final の失敗は `stage="c_final"` を持つ（fold の校正器だけを失敗させる factory と、C_final だけを失敗させる factory で確かめる）。修正前は RED。
3. `Model.fit` で、`calibration.params` に `{"options": {"maxiter": 1}}` を付けた binary の Platt と Beta が `CALIBRATION_FAILED` を送出する。修正前は RED。
4. 生成される `train.py` の Platt と Beta の校正器は、同じ設定で `RuntimeError` を送出する。修正前は RED。
5. 既定の設定の校正（既存のテスト）は変わらず通る。`docs/api.md` と BLUEPRINT §16.2 の `ErrorCode` 一覧が enum と一致する（`test_error_code_docs.py`）。

## H-0114: SHAP 重要度は、各 fold のモデルをその fold の pipeline で変換した行で説明する（#303）

- **ステータス**: Accepted
- **起票日**: 2026-10-06
- **決定日**: 2026-10-07（管理者の判断 2026-10-06: 方針 A = fold ごとの pipeline 状態を `FitResult` に記録して SHAP 重要度で使う。Codex の設計レビュー 2 run を経て、run 2 の round 2 で APPROVE）
- **スコープ**: `lizyml/core/types/fit_result.py`（フィールド追加、`_SHARED_ON_COPY`、docstring の「`None` になりうるフィールド」の文と共有の説明）, `lizyml/core/model.py`（`fit_result` プロパティの docstring の共有フィールドの列挙）, `lizyml/training/cv_trainer.py`（fold ごとの状態を記録）, `lizyml/explain/shap_explainer.py`（`compute_shap_importance` が fold ごとの状態を使う）, `lizyml/core/_model_tables.py`（呼び出しと、状態を持たない artifact の拒否）, `BLUEPRINT.md`（§7 の FitResult の一覧と共有フィールドの文、§9.2、§13 の SHAP 重要度、§15.1 の保存対象、§15.2）, `docs/api.md`（FitResult の表と「`None` になりうるフィールド」の文、`importance()` の Raises、`load()` の説明、`ErrorCode` 表の `MODEL_NOT_FIT`）, `docs/proposal_dispositions.toml`, `CHANGELOG.md`, テスト（`tests/test_features/test_unseen_policy.py` の固定テストを新しい契約に書き換え、`tests/test_explain/test_shap_importance_per_fold_pipeline.py` 新規、`tests/test_persistence/` に旧 artifact の読み込み、`tests/test_core/test_contracts.py` / `tests/test_e2e/test_golden_contracts.py` / `tests/test_core/test_result_isolation.py` のフィールド一覧と共有の検査）。既存の一覧のずれ（`docs/api.md` に `target_encoder` が無い、BLUEPRINT §7 に `oof_raw_scores` が無い）とその整合検査は [#326](https://github.com/nbx-liz/LizyML/issues/326) に切り出し、本 Proposal は新しいフィールドを両方の一覧に足すだけにする
- **関連**: [Issue #303](https://github.com/nbx-liz/LizyML/issues/303)（測定と独立 critique はコメント 6013323756）, H-0104（決定 8 の SHAP の節を本 Proposal が置き換える）, H-0007（fold 平均の SHAP 重要度）, H-0054（provider の pipeline factory）, H-0082（selective deep copy）, H-0026（`analysis_context` の無い artifact の診断 API を `MODEL_NOT_FIT` にした前例）, H-0003（format_version の規則）

### 目的（課題）

`Model.importance(kind="shap")` は `FitResult.pipeline_state`（**最後の** CV fold の pipeline の `get_state()`、`cv_trainer.py` `_build_result`）で学習データの全行を変換し、各 fold のモデルをその fold の検証行で説明する。各 fold のモデルはその fold の pipeline で変換したデータで学習し、OOF もその pipeline で変換した検証行で予測している。最後の fold の pipeline が知っているカテゴリの集合が fold の pipeline と違うと、SHAP 重要度は、モデルが学習でも OOF でも見なかった符号化の行を説明する。

対象は `features.auto_categorical: false` で pipeline がカテゴリとして扱う文字列列だけである（既定の `auto_categorical: true` と `categorical` 指定の列はデータ構築時に全行の値で `category` 型になり、どの fold の pipeline も同じ値を知る）。

### 測定（#303 コメント 6013323756、`113d20a`、Codex の read-only critique 1 round を経た値）

`Model.fit` → fold のモデル → `compute_shap_values` を通しで実行した。各 fold の検証行を「最後の fold の pipeline」（現状）と「fold 自身の pipeline」（`CVTrainer` と同じく `X[train]` で fit し直したもの）で変換して比べた。fold 自身の pipeline で再現した予測は保存された OOF と完全に一致した（最大誤差 0、regression / binary / multiclass）。

| 場合 | 結果 |
|---|---|
| 両方の pipeline が知る値（category の code の並びだけが違う） | 差 0（442 行）。LightGBM は学習時の category の並びで値ごとに remap する（`lightgbm/basic.py:852-856`、4.6.0）。陰性対照（remap を効かなくした入力）は最大差 0.0323 を出すので、probe は差を検出できる |
| fold が知らず最後の fold が知る値（後の時期に現れるカテゴリ、expanding window、`unseen_policy: mode`） | 480 行中 38 行で SHAP が違う。OOF では fold のモデルは最頻値への置換を見たが、SHAP 重要度は LightGBM の remap 後の欠損を見る。binary / multiclass でも同じ |
| fold が知り最後の fold が知らない値（sliding window、`train_size_max`） | `mode` / `nan` で 33 行が違う（最大差 0.072）。`error` では `fit` が通ったうえで `importance(kind="shap")` が `DATA_SCHEMA_INVALID` |
| 戻り値への影響 | `cat` の重要度（現状 / fold 自身）: regression 0.01181 / 0.01114、binary 0.02576 / 0.02746、multiclass 0.02444 / 0.02526（約 3〜6 %）。sliding window の例で 0.03107 / 0.03043 |
| 影響なし | `auto_categorical: true`、`categorical` 指定、数値列だけ: 差 0 |

### 対応方針

1. **`FitResult` に `pipeline_state_per_fold: list[Any] | None = None` を追加する。** 名前は既存の `if_pred_per_fold` に揃える。`CVTrainer` は fold ごとに fit した pipeline の `get_state()` を順に記録し、常に埋める（`len == len(splits.outer)`、`[-1]` は `pipeline_state` と同じ状態）。`None` は「fold ごとの状態が無い」ことを表す。`CVTrainer` が作る `FitResult` は常にリストを持つので、`None` になるのは本 Proposal 以前の LizyML が作った `FitResult`（旧 artifact）と、利用者がこのフィールドを省いて直接構築した `FitResult`（`FitResult` は `lizyml/__init__.py` から公開されている）である。facade はどちらも同じに扱う（決定 7）。最後のフィールドの後ろに置き、既定値を持つので、既存の位置引数・キーワード引数の構築は変わらない。
2. **`pipeline_state` の意味は変えない**（最後の fold の状態のまま）。`pipeline_state_per_fold[-1]` と重複するが、削除や型の変更は破壊的変更で `format_version` を上げる（H-0003）。本 Proposal では残し、削除は扱わない。lizyml 内の読み手は SHAP 重要度だけである（下の表）。
3. **`compute_shap_importance` は fold k のモデルを、fold k の状態を load した pipeline で `X.iloc[valid_idx_k]` だけを変換した行で説明する。** 引数 `pipeline_state_per_fold: list[Any] | None = None` を**末尾（`pipeline_factory` の後）**に足す。既存の位置引数（7 番目は `pipeline_factory`）の意味は変わらない。与えられたときは長さが `models` と一致しなければ `ValueError`（内部の不整合）。与えられないときは従来どおり `pipeline_state` 1 つで全行を変換する（この関数を直接呼ぶ利用者の互換。どの状態で変換するかは呼び出し側が渡す状態が決める）。facade は常に fold ごとの状態を渡す。
   - 結果: SHAP 重要度が説明する行の符号化は、OOF 予測が使った符号化と同じになる。`unseen_policy` の置換も OOF と同じ置換になる（fit 中と同じく報告しない。H-0104 決定 8 の「報告しない」は変わらない）。
   - `error`: fold の検証行に fold 自身が知らない値があれば `fit` が先に `DATA_SCHEMA_INVALID` で止まるので、`fit` が通ったモデルの `importance(kind="shap")` はこの理由で送出しない。
4. **`pipeline_state_per_fold` を `_SHARED_ON_COPY` に加える。** `pipeline_state` と同じ理由（学習済みの状態、read-only の慣例。H-0082）。リストの容器だけを新しくする（`models` と同じ扱い）。
5. **保存形式: `FORMAT_VERSION` は 2 のまま。** `fit_result.pkl` は `FitResult` 全体の pickle なので、新しいフィールドは exporter / `metadata.json` / `checksums` を変えずに保存される。BLUEPRINT §15.2 の「フィールドの追加は後方互換の変更で `format_version` を上げない」に当たる。
6. **旧 artifact（フィールドを持たない `fit_result.pkl`）の読み込み。** 旧 pickle は `__dict__` をそのまま復元するので属性が無い。`= None` の既定値はクラス属性として残るため、属性アクセスは `None` を返す（`default_factory` にするとクラス属性が無く `AttributeError` になる。H-0070 が `hasattr` を要した理由）。loader は変えない。`predict()` / `evaluate()` / 他の診断 API は従来どおり動く。
7. **fold ごとの状態が無いときの `importance(kind="shap")` は `MODEL_NOT_FIT` で拒否する**（設計レビュー round 1 が代替案 3 より推奨した）。`pipeline_state_per_fold is None` のとき、facade は `LizyMLError(MODEL_NOT_FIT)` を送出する。`user_message` は「この `FitResult` は fold ごとの pipeline 状態（`pipeline_state_per_fold`）を持たない。H-0114 より前の版で作られた artifact か、このフィールドを省いて構築した `FitResult` である。現在の版で `fit()` し直す」（再 export だけでは状態は作られない）。どちらの原因かは区別しない（`None` からは区別できない）。`context` は `task` / `kind` / `method` / `missing="pipeline_state_per_fold"`。前例は H-0026（`analysis_context` の無い artifact の診断 API を `MODEL_NOT_FIT` で明示的に失敗させる）。`importance_plot(kind="shap")` は `importance()` を通るので同じ。
   - **検査の順序**: この検査は既存の `state.X is None`（`analysis_context` の無い artifact）の検査より**先**に行う。両方を欠く古い artifact でも `missing="pipeline_state_per_fold"` になり、示す対処（fit し直す）は両方を解消する。`X` だけを欠く場合の既存のエラーは変わらない。
8. **文書**: `docs/api.md` の `importance()` の Raises、`load()` の説明（`analysis_context` があっても H-0114 以前の artifact では SHAP 重要度が使えない）、`ErrorCode` 表の `MODEL_NOT_FIT` の説明（fit 前に加えて、診断に必要な状態を持たない読み込み済みモデル）を更新する。BLUEPRINT §15.2 の旧 artifact の診断 API の行にも同じことを足す。FitResult の一覧（`docs/api.md` の表と BLUEPRINT §7）には新しいフィールドを足す。

### 規則が縛る位置（ソースから導出）

規則: **fold のモデルに渡す行は、その fold の pipeline 状態で変換する。** 導出: `lizyml/` で `pipeline_state` / `load_state(` / `fit_result.models` / `compute_shap_importance` / `importance(kind` を grep した全件（`5d04d08`）。

| # | 位置 | fold のモデルに変換した行を渡すか | 本 PR |
|---|---|---|---|
| 1 | `explain/shap_explainer.py` `compute_shap_importance`（呼び出し元 `core/_model_tables.py:170`） | 渡す（最後の fold の状態で変換） | fold ごとの状態に変える |
| 2 | `core/_model_plots.py:185` `importance_plot(kind="shap")` | #1 を `importance()` 経由で呼ぶ | 変えない（#1 で直る） |
| 3 | `core/_model_tables.py:180-187` / `plots/importance.py:85` の split / gain 重要度 | 渡さない（モデル内部の値） | 変えない |
| 4 | `core/_model_predict.py:56` の推論と `return_shap=True` | 渡さない（refit モデルと refit pipeline の組） | 変えない |
| 5 | `core/_model_persistence.py:336` の `export_code` | 渡さない（refit の状態を書き出す） | 変えない |
| 6 | `training/cv_trainer.py` の fold 学習と OOF | その fold の pipeline で変換済み（規則の基準） | 状態を記録するだけ |

### 互換性

- **振る舞いの変化（新しく fit したモデル）**: `auto_categorical: false` でカテゴリとして扱う文字列列があり、fold ごとの pipeline が知る値の集合が違う場合だけ、`importance(kind="shap")` の値が変わる（測定で 3〜6 %）。sliding window + `unseen_policy: "error"` では、今日 `DATA_SCHEMA_INVALID` を送出する `importance(kind="shap")` が値を返す。上の測定で、既定の `auto_categorical: true`、`categorical` 指定、数値列だけの場合は差 0。
- **旧 artifact**: `importance(kind="shap")` / `importance_plot(kind="shap")` が、今日の値（最後の fold の状態による）を返す代わりに `MODEL_NOT_FIT` を送出する（決定 7）。他の API は変わらない。フィールドを省いて直接構築した `FitResult` を使うモデルも同じ。
- **公開 API**: `FitResult` にフィールドが 1 つ加わる（追加のみ、既定値あり）。`compute_shap_importance` の末尾に省略可能な引数が 1 つ加わる（既存の位置引数・キーワード引数の呼び出しは変わらない）。`format_version` / Config / `PredictionResult` / `metadata.json` は変わらない。
- **H-0104 決定 8 の SHAP の節を置き換える**: 「SHAP 重要度は最後の CV fold の pipeline 状態で学習データ全体を変換する…これを仕様として固定する」と、その固定テスト `test_shap_importance_applies_the_stored_policy_outside_the_last_fold` は本 Proposal の契約に書き換える（テストは削除せず、`test_shap_importance_uses_each_fold_pipeline_outside_the_last_fold` として、受け入れ基準 2 のデータで新しい振る舞いを固定する。旧 fixture は差を検出できなかった）。BLUEPRINT §9.2 の該当行も書き換える。

### 代替案（検討して棄却）

1. **`pipeline_state` を fold ごとの状態のリストに変える。** 型と意味の変更で `format_version` を上げ、v2 → v3 の migration が要る。重複は消えるが、`pipeline_state` を読む外部コードを壊す。追加で足りるので採らない。
2. **旧 artifact では最後の fold の状態に戻して警告する。** 旧 artifact だけ #303 の欠陥を出し続け、値が新しい fit と黙って食い違う（警告は読まれないことがある）。採らない。
3. **旧 artifact では fold ごとの状態を作り直す。** `analysis_context` の `X` と `splits.outer` から、`provider.build_pipeline_factory()`（`unseen_policy` は保存された状態から）で fold ごとに fit し直す。#303 の critique は、この再現が OOF と完全に一致することを確かめた（最大誤差 0）。旧 artifact でも正しい値が出る利点があるが、説明の経路に pipeline の fit を持ち込み、artifact の内容ではなく**読み込んだ時点の版の** pipeline の実装で状態を作る（後の版で pipeline の fit が変わると、学習時の状態と黙って食い違う）。決定 7 の拒否より利用者に優しいので、設計レビューで判断を仰ぐ代替として残す。
4. **SHAP 重要度の対象を、どの fold にも共通して知られた値の行に限る。** 重要度の母集団が変わり、fold の検証行の平均という H-0007 の定義から外れる。採らない。
5. **記録せずに、毎回 fold ごとに pipeline を fit し直す（新しい fit でも）。** 代替案 3 と同じ問題（読み込み時の版の実装に依存）を新しい artifact にも持ち込む。記録は fold あたり状態 1 つ（カテゴリの一覧と最頻値）で小さい。採らない。

### 設計レビューでの判断（round 1）

- **旧 artifact**: 決定 7（拒否）を採る。旧 artifact には学習時の fold ごとの状態が無く、作り直し（代替案 3）は読み込んだ時点の版の pipeline 実装に依存するので、`5d04d08` で誤差 0 でも将来の一致を保証しない。
- **`categorical_features`**: 最後の fold の `categorical_cols` から取ることを維持する。`NativeFeaturePipeline.fit` は列の dtype（`category` か文字列か）だけで決める（`pipelines_native.py:56-61`）。レビューの probe で、object / StringDtype / category / 数値の列について、学習した値が違う 2 つの行の部分集合で同じ一覧が返ることを確かめた。独自の pipeline（H-0054）で fold ごとに違いうる場合は本 Proposal の範囲外とする。受け入れ基準 4 で、記録したすべての状態の `categorical_cols` が `categorical_features` と一致することを固定する。

### 受け入れ基準（テスト観点）

1〜2 と 5〜7 は修正前に RED になる。3 は修正前から通る陰性対照（変えてはいけない場合の回帰）。4・8〜10 は新しい契約の固定。

1. **fold が知らず最後の fold が知る値**（expanding window、`auto_categorical: false`、`unseen_policy: "mode"`、LightGBM のカテゴリ既定を緩めて `cat` で分岐させる。#303 の probe のデータ。regression / binary / multiclass のそれぞれで。測定は 3 つの task で差を出した）: `importance(kind="shap")` が、fold ごとに `X[train]` で fit し直した pipeline で検証行を変換して計算した値と一致し、最後の fold の状態で計算した値とは違う。陽性の確認として、**値の集合が違う fold のモデル**（`fit_result.models[k]`）が `cat` で分岐している（`importance(kind="split")["cat"] > 0`）。同じデータの `unseen_policy: "nan"` でも独立の再計算と一致する（この向きで `nan` が最後の fold の値と違うかは測っていないので、差は求めない）。
2. **fold が知り最後の fold が知らない値**（sliding window、`train_size_max`、3 fold 以上）: ある fold の学習行と検証行にあり、最後の fold の学習行に無い値を置く（既存の fixture は 2 fold で、その値が検証行に入らないため、修正前でも最後の fold と fold 自身で差が 0 になる。round 1 で実測）。`unseen_policy: "error"` で `fit` が通ったモデルの `importance(kind="shap")` が値を返す（修正前は `DATA_SCHEMA_INVALID`）。`"mode"` と `"nan"` のそれぞれで、1 と同じ独立の再計算と一致し、最後の fold の状態で計算した値とは違い（測定では両方で 33 行が違った）、値を持つ fold のモデルが `cat` で分岐している。既存の `test_shap_importance_applies_the_stored_policy_outside_the_last_fold` をこの契約とデータに書き換える（削除しない）。
3. **影響しない場合（陰性対照、修正前から GREEN）**: 1 と同じ、時期によって値の集合が変わる文字列列を持つデータで、(a) `auto_categorical: true`（既定）と (b) `auto_categorical: false` に `features.categorical: ["cat"]` を指定した設定のそれぞれについて、`importance(kind="shap")` が最後の fold の状態で計算した値と一致する。対照が空振りでないことの確認として、各 fold の記録した状態が知るカテゴリの集合がすべての fold で同じであり、fold のモデルが `cat` で分岐している。数値列だけのデータでも一致する。
4. **記録の契約**: regression / binary / multiclass で `len(pipeline_state_per_fold) == len(splits.outer)`、`pipeline_state_per_fold[-1] == pipeline_state`、各要素が fold の `X[train]` で fit し直した pipeline の `get_state()` と等しく、各要素の `categorical_cols` が `FitResult.categorical_features` と等しい。
5. **保存**: export → load で `pipeline_state_per_fold` が保たれ、`importance(kind="shap")` が export 前と一致する。`FORMAT_VERSION == 2` のまま。
6. **旧 artifact**: 属性を `__dict__` から消した `FitResult` を `fit_result.pkl` として書き（`checksums` を書き直す）、次を確かめる。
   - `Model.load()` が通り、`predict()` が export 前と一致する。読み込んだ `FitResult` で `pipeline_state_per_fold is None`、`Model.fit_result`（`__deepcopy__` を通る）と `dataclasses.replace` が成功する。本物の v1 の形（`target_encoder` と `pipeline_state_per_fold` の両方を `__dict__` から消し、`metadata.json` の `format_version` を 1 にした artifact）でも load が通り、`_migrate_fit_result` の `replace` の分岐を通って（`loader.py:51-54` は `target_encoder` が無いときだけ `replace` を呼ぶ）、`target_encoder` が no-op、`pipeline_state_per_fold is None` になる。
   - `importance(kind="shap")` が `MODEL_NOT_FIT` を送出する。`context` は `task` / `kind`（`"shap"`）/ `method`（`"importance"`）/ `missing`（`"pipeline_state_per_fold"`）をすべて持ち、`user_message` は 2 つの原因（H-0114 より前の版の artifact と、このフィールドを省いて構築した `FitResult`）と対処（`fit()` し直す）をどちらも述べる。
   - SHAP 以外は変わらない: `importance(kind="split")` / `importance(kind="gain")` が export 前と同じ値を返し、`importance_plot(kind="split")` が送出しない（plotly があるとき）。拒否の検査が `kind == "shap"` の分岐の外に置かれると RED になる。
   - `analysis_context.pkl` も消した artifact でも同じ `missing` になる（決定 7 の検査の順序）。`analysis_context.pkl` だけを消した新しい artifact は従来どおりの `MODEL_NOT_FIT` で、`missing` を持たない。
7. **直接構築した `FitResult`**: このフィールドを省いて構築した `FitResult` は `pipeline_state_per_fold is None` で、それを使うモデルの SHAP 重要度は 6 と同じ `context` と `user_message` の検査で拒否され、split / gain の重要度は値を返す。
8. **`compute_shap_importance` の互換**: 7 つの位置引数（`pipeline_factory` まで）の従来の呼び出しは、渡した `pipeline_state` 1 つで全行を変換した従来の値を返す。`pipeline_state_per_fold` の長さが `models` と違えば `ValueError`。
9. **copy**: `Model.fit_result` の戻り値で `pipeline_state_per_fold` の要素は内部と同一オブジェクト、リストは別オブジェクト（`test_result_isolation.py`）。
10. **フィールド一覧と文書**: `test_contracts.py` / `test_golden_contracts.py` の `FitResult` のフィールド一覧、`docs/api.md` の表、BLUEPRINT §7 に `pipeline_state_per_fold` が加わる。`docs/api.md` の `importance()` / `load()` / `MODEL_NOT_FIT` の記述が決定 7 を述べる。BLUEPRINT §15.1 の保存対象に `pipeline_state_per_fold` が加わる。`None` になりうるフィールドと参照を共有するフィールドを列挙する文（`docs/api.md` の FitResult の冒頭、BLUEPRINT §7 の selective deep copy の箇条、`FitResult` の docstring、`Model.fit_result` の docstring）が新しいフィールドを含む。BLUEPRINT は §9.2 の SHAP 重要度の行（最後の fold の pipeline で全行を変換する、を fold 自身の pipeline に書き換える）、§13 の SHAP 重要度（fold k の検証行を fold k の pipeline 状態で変換する）、§15.2（fold ごとの状態を持たない artifact の SHAP 重要度は `MODEL_NOT_FIT`、検査の順序）を本 Proposal の決定どおりに述べる。`docs/proposal_dispositions.toml` の H-0114 は `specified` になり、その anchor（`pipeline_state_per_fold` を含む）が BLUEPRINT と本 Proposal の両方に現れる（`test_proposal_blueprint_coverage.py`）。

### 実装時の記録

- **受け入れ基準 1 の `nan`**: 修正前から通る。このデータでは、最後の fold の pipeline が値 `b` を fold のモデルの知らないカテゴリとして符号化し、LightGBM はそれを欠損として扱うので、fold 自身の `nan` 置換と同じ入力になる。基準 1 が `nan` に差を求めなかったとおりである。
- **受け入れ基準 2 のデータ**: 最初に作ったデータ（値 `e` を 40 %、効果 -4）では、fold のモデルが `cat` を `c` と `d` でしか分岐せず、`e` が欠損と同じ枝に落ちたため、`nan` で fold 自身と最後の fold の値が一致した（booster の dump で確認）。`e` を 60 %、効果 +6 にして、`mode` と `nan` の両方で差が出ることを確かめた。テストの docstring に理由を残した。

## H-0115: `purged_time_series` の `embargo` を `purge_gap` に統合する（#273）

- **ステータス**: Accepted
- **起票日**: 2026-10-07
- **決定日**: 2026-10-07（管理者の判断 2026-10-07: `embargo` を `purge_gap` に統合する。中立な名前の 2 つ目の gap への改名と、inner split を `purge_gap` だけにする案は採らない。Codex の設計レビュー 3 round の後、修正の確認 2 回の 2 回目で APPROVE）
- **スコープ**: `lizyml/config/schema.py`（`PurgedTimeSeriesConfig` の `embargo` フィールドを外し、旧キーとして `purge_gap` に加算）, `lizyml/splitters/purged_time_series.py`（`embargo` 引数の非推奨化）, `lizyml/core/_model_factories.py`（splitter の構築と `_auto_inner_gap`）, `lizyml/training/inner_valid.py`（docstring とエラー文）, `lizyml/core/_model_persistence.py` と `lizyml/codegen/templates.py`（生成コードの split 設定）, `BLUEPRINT.md`（§5 の既定値表と旧キーの規則、§5.5 の構成値の表と件数、§10.2、§10.3.1、§10.3 の inner gap の説明）, `docs/config-reference.md`（既定値表、gap の表、図）, `docs/DEPRECATIONS.md`（行の追加と、旧キーの出典 H-0021 の誤りの訂正）, `notebooks/tutorial_time_series_lgbm.ipynb`（`embargo` を使うセルと説明）, `docs/proposal_dispositions.toml`, `CHANGELOG.md`, テスト（`tests/test_config/_knob_registry.py` の `PurgedTimeSeriesSplitter.embargo` を `api` に分類し直し、`test_knob_reachability.py` の `embargo` の `config` のセルを外して `api` の到達の証拠を足す、`tests/test_splitters/`、`tests/test_config/`、`tests/test_training/test_inner_valid_purge_embargo.py`、`tests/test_e2e/test_time_series_*`、`tests/test_codegen/test_split_reproduction.py`、`tests/test_calibration/test_calibration_split.py` の `embargo` を使う箇所を新しい契約に書き換え、統合の新規テストを追加）
- **関連**: [Issue #273](https://github.com/nbx-liz/LizyML/issues/273), H-0038（`purge_window` → `purge_gap`、`gap` の旧キー）, H-0040（`embargo` を観測数にし、`embargo_pct` を旧キーにした）, H-0076（非推奨の登録簿と v1.0 削除）, H-0085（outer の境界 gap を inner valid に伝える）, H-0111（旧キーの端数を拒否）, [#265](https://github.com/nbx-liz/LizyML/issues/265)（inner gap の伝搬規則の矛盾。close 済み）

### 目的（課題）

`purged_time_series` は `purge_gap` と `embargo` を別のつまみとして公開しているが、splitter は両方を同じ位置で引く（`purged_time_series.py` `train_end = (k + 1) * fold_size - self.purge_gap - self.embargo`）。2 つは同じ「学習の末尾と検証の先頭の間の除外」を広げる 1 つのつまみである。

文献（purging and embargo）で embargo は方向を持つ仕組みである: **検証ブロックの後ろ**にある学習行の先頭を除く（後ろ向きの特徴量窓が検証期間と重なる行を落とすため）。この splitter は前向き連鎖（expanding window）で、学習は常に検証より前にある（#273 で全 fold を実行: 検証の後ろにある学習行は 0）。したがって、名前が約束する仕組みには働く場所が無く、名前を知る利用者は意図と逆のもの（検証の**前**の学習行の追加除外）を得る。`docs/config-reference.md` の図は `embargo` を検証の後ろに描いており、実装と食い違う。

これは漏洩ではない（除外が増える方向で、安全側）。欠陥は、別の意味を持つ用語が 2 つ目のつまみとして並んでいることである。

### 対応方針

1. **つまみを `purge_gap` 1 つにする。** `PurgedTimeSeriesConfig` から `embargo` フィールドを外す。`purge_gap` は「学習の末尾と検証の先頭の間で除く観測数」で、検証の前にだけ働く。
2. **`embargo` キーは旧キーとして受理し、値を `purge_gap` に加算する。** 値が `0` でも、キーがあれば `DeprecationWarning`（v1.0 で削除、H-0076）。文面は、`embargo` が `purge_gap` と同じ位置を広げていたこと、値を `purge_gap` に足したこと、`purge_gap` に合計を書くことを述べ、`purge_gap` の名前を含む。
3. **旧キー `embargo_pct` と `gap`（`purged_time_series`）も `purge_gap` に加算する。** 今日は `embargo` に写している（H-0040）。`embargo` / `embargo_pct` / `gap` は同じ 2 つ目の gap の 3 つの綴りなので、**同時に指定できるのは 1 つまで**（2 つ以上は `CONFIG_INVALID`）。今日も 2 つ目以降の綴りは残って `extra="forbid"` で拒否される（`embargo_pct` → `embargo` の後の `gap` も同じ）ので、この点で受理する集合は変わらない。`purge_window` → `purge_gap` の規則（H-0038）は変えない。
4. **値の検査と順序**（受理する値の集合はここで変わる。round 1 の実測に基づく）:
   1. `purge_window` を `purge_gap` に写す（今日どおり）。
   2. 2 つ目の gap の綴り（3 つのうち 1 つ）を取り出し、**綴りごとに今日と同じ規則で**整数に読む（round 2 の指摘: `_legacy_obs_count` は 2 進の `float` を経由するので、`embargo` に使うと受理する値が変わる）。
      - `embargo`: 今日と同じ pydantic の int として検証する（lax。`"1e3"` は拒否、`"9007199254740993"` は丸めずに受理、`"1e400"` は拒否）。**変化は 1 つだけ**: その前に bool を拒否する（今日 `True` は `1` として受理されていた。round 1 の実測。観測数に bool を書く正当な入力は無い）。
      - `embargo_pct` / `gap`: 今日と同じ `_legacy_obs_count`（整数、整数値の float、整数値を表す文字列は受理し、端数と bool は拒否）。**ただし** `_legacy_obs_count` は今日、`"1e400"` のような値で `int(float(...))` の `OverflowError` を捕まえずに送出する（`load_config` は `ValidationError` しか捕まえないので `CONFIG_INVALID` にならない）。この経路を触るので、`OverflowError` を `ValueError` に変えて `CONFIG_INVALID` にする。
   3. `purge_gap` は今日どおり pydantic の int として検証する（`"5"` や `5.0` の受理は変えない）。そのうえで、**`purge_gap` と 2 つ目の gap はそれぞれ独立に 0 以上**でなければならない。**変化**: 今日、負の `purge_gap` / `embargo` は Config の検証を通り、fit で splitter が `ValueError` を送出していた。これを Config 検証の失敗（`CONFIG_INVALID`）に早める。加算の前に検査するので、`purge_gap: -1, embargo: 2` のように負の値を加算で隠すことはできない（今日も fit で拒否されている）。
   4. 加算する: `purge_gap = purge_gap + (2 つ目の gap)`。
   実装は `mode="wrap"` などの validator で、`purge_gap` の型の検証の後に 3〜4 を行う。拒否はどれも Config 検証の失敗（`CONFIG_INVALID`）。
5. **除外の量は変わらない。** 今日受理されて fit できた入力（`True` の `embargo` を除く）では、新しい `purge_gap` は今日の `purge_gap + embargo` に等しい。outer の fold も、H-0085 で inner valid に伝える境界 gap も、今日と同じ行になる。
6. **inner valid の境界 gap の規則は `purge_gap` だけで述べる。** `_auto_inner_gap` は `purged_time_series` で `purge_gap` を返す（`time_series` の `gap` は変えない）。outer の splitter と inner の gap はどちらも Config の `purge_gap` 1 つから作るので、2 つが食い違う余地が無い。BLUEPRINT §10.3.1 の「`purge_gap + embargo`」は「outer split の境界 gap（`purged_time_series` では `purge_gap`）」と書き換える。
7. **`PurgedTimeSeriesSplitter(embargo=...)` 引数も非推奨にする。** `lizyml.splitters` から公開されているため。引数の既定値を `None`（指定なし）にし、**指定されれば `0` でも** `DeprecationWarning` を出して `purge_gap` に加算する（省略と明示の `0` を区別するため）。`purge_gap` と `embargo` はそれぞれ負なら今日どおり `ValueError`（加算の前に検査）。属性 `embargo` は持たない（`purge_gap` が合計を持つ）。
8. **生成コード（`export_code`）**: split 設定のブロックは `embargo` を書かず、生成される `_purged_ts_folds` は `purge_gap` だけを引く。
9. **チュートリアルとテストの登録簿**: `notebooks/tutorial_time_series_lgbm.ipynb` の `"embargo": 50` を `purge_gap` に足した値に、校正が `embargo` を継承するという説明を `purge_gap` に書き換える。`tests/test_config/_knob_registry.py` の `PurgedTimeSeriesSplitter.embargo` は**外さず、種類を `config` から `api` に変える**（round 2 の指摘: 引数は v1.0 まで残り、台帳は既定値付きの全引数の網羅を `test_knob_reachability.py` で要求する。round 3 の指摘: §5.5 の定義で、公開の呼び出しの引数が値を変えるものは `policy` ではなく `api` である。`PurgedTimeSeriesSplitter` は `lizyml.splitters` から公開され、直接構築すれば値が変わる）。値の出どころは「`PurgedTimeSeriesSplitter(embargo=...)`（非推奨。`purge_gap` に加算され、v1.0 で削除。Config のどのキーも渡さない）」。`test_knob_reachability.py` の `embargo` の `config` の実行セルは外し、`purge_gap` のセルは残す。`api` の行と同じく、直接構築で値が splitter に届く（`purge_gap` に加算される）ことを実行で確かめるセルを足す。BLUEPRINT §5.5 の表に同じ行を加え、件数（74 個のうち Config のキーが渡す 60 個、残り 14 個）を 59 個と 15 個に直す。
10. **`docs/DEPRECATIONS.md` の出典を訂正する**: 旧キー `purge_window` / `embargo_pct` / `gap` の出典は H-0021（LightGBM の smart parameter の Proposal）ではなく、H-0038（`purge_window`、`gap`）と H-0040（`embargo_pct`）である。H-0076 の本文の表も同じ誤りを持つが、HISTORY の記録は書き換えず、`docs/DEPRECATIONS.md` を正す。
11. **BLUEPRINT §10.2 に幾何の事実を書く**: `purged_time_series` は前向き連鎖で、どの fold でも学習の最大 index は検証の最小 index より小さい。検証の後ろに学習行が無いので、検証の後ろを除く embargo の置き場所は無い。後ろ向きの embargo が必要になれば、検証ブロックの後ろに学習行を持つ別の splitter が要る（本 Proposal の範囲外）。

### 規則が縛る位置（ソースから導出）

規則: **`purged_time_series` の除外は `purge_gap` 1 つで表し、`embargo` は旧キーとしてだけ受理して `purge_gap` に加算する。** 導出: リポジトリ全体で `rg -l -i '\bembargo(?:_pct)?\b' .` の全件（`6d4c799`。round 1 で notebook と knob registry の漏れを指摘されて取り直した）。HISTORY の過去の記録と `docs/audits/` の監査記録は書き換えない。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `config/schema.py` `PurgedTimeSeriesConfig`（フィールドと `_normalize_legacy_keys`） | フィールドを外し、`embargo` / `embargo_pct` / `gap` を `purge_gap` に加算 |
| 2 | `splitters/purged_time_series.py`（構築子、`split` の `train_end`） | 引数を非推奨にし、`purge_gap` だけを引く |
| 3 | `core/_model_factories.py:150`（splitter の構築） | `embargo` を渡さない |
| 4 | `core/_model_factories.py:264-270` `_auto_inner_gap` | `purge_gap` を返す |
| 5 | `training/inner_valid.py:175, 201`（docstring とエラー文） | `purge_gap` だけを述べる |
| 6 | `core/_model_persistence.py:66`（生成コードの split ブロック） | `embargo` を書かない |
| 7 | `codegen/templates.py:323, 334, 514`（生成される fold の再現） | `embargo` を引かない |
| 8 | `BLUEPRINT.md` §5（既定値表 l.348、旧キー l.355-356）、§10.2（l.685）、§10.3.1（l.701）、§10.3（l.755） | 書き換える |
| 9 | `docs/config-reference.md`（l.80, 98, 108, 116, 142-148, 572） | 書き換える（図の誤りを含む） |
| 10 | `docs/DEPRECATIONS.md`（l.17-19, 76-82） | `embargo` / `embargo_pct` / `gap` → `purge_gap`、splitter の引数を登録、出典を H-0038 / H-0040 に訂正 |
| 11 | `notebooks/tutorial_time_series_lgbm.ipynb`（l.335, 371） | `purge_gap` に書き換える |
| 12 | `tests/test_config/_knob_registry.py`（l.130, 181）、`tests/test_config/test_knob_reachability.py`（l.212, 248, 409）、`BLUEPRINT.md` §5.5（l.405-430 の件数と表） | `PurgedTimeSeriesSplitter.embargo` を `api` に分類し直し、`config` の実行セルを外して `api` の到達のセルを足し、§5.5 の表と件数を直す |
| 13 | 既存のテストで `embargo` を書くもの（`tests/test_splitters/`、`tests/test_config/test_config_hardening.py`、`tests/test_training/test_inner_valid_purge_embargo.py`、`tests/test_e2e/test_time_series_*`、`tests/test_codegen/test_split_reproduction.py`、`tests/test_calibration/test_calibration_split.py`） | 旧キーの契約を確かめるもの以外は `purge_gap` に書き換える（削除しない） |

### 互換性

- **受理する値の変化**（決定 4。これ以外の値の受理と変換は綴りごとに今日と同じ）: `embargo: true` は今日 `1` として受理されたが、`CONFIG_INVALID` になる。負の `purge_gap` / `embargo` は今日 fit で splitter の `ValueError` だったが、Config 検証の `CONFIG_INVALID` になる（早くなるだけで、受理されていた入力ではない）。`embargo_pct` / `gap` の桁あふれする値（`"1e400"`）は今日 `OverflowError` が漏れていたが、`CONFIG_INVALID` になる。
- **除外の量と分割は変わらない**（決定 5）。`fit` / `tune` / OOF / inner valid / 校正の cross-fit（outer split を再利用、H-0058）の行は、どの入力でも今日と同じ。
- **警告**: `embargo`（0 を含む）を書いた Config は `DeprecationWarning` を出す。`embargo_pct` / `gap` の警告文は移行先が `purge_gap` に変わる。
- **`model_dump()` / `config_normalized`**: `embargo` キーが無くなり、`purge_gap` が合計を持つ。Config を dump して読み直す往復は、警告なしに同じ分割を再現する。
- **保存済みの artifact**: 以前の `metadata.json` の `config` は `embargo`（多くは `0`）を持つ。`Model.load()` は Config を検証し直すので、`purged_time_series` の artifact は `DeprecationWarning` を出し、値を `purge_gap` に加算して同じ分割を得る。`predict()` の結果は変わらない。`format_version` は変えない（保存形式は変わらない）。**v1.0 で `embargo` キーを削除するときは、読み込みの経路でこの旧キーを正規化し続ける必要がある**（さもないと以前の artifact が `extra="forbid"` で読めなくなる）。これを `docs/DEPRECATIONS.md` に注記する。
- **公開 API**: `PurgedTimeSeriesConfig.embargo` 属性と `PurgedTimeSeriesSplitter.embargo` 属性が無くなる（構築時のキー・引数は v1.0 まで受理）。

### 代替案（検討して棄却）

1. **中立な名前の 2 つ目の gap に改名する**（例 `extra_gap`）。2 つ目のつまみの存在理由（`purge_gap` と区別される仕組み）を説明できない。管理者の判断で採らない。
2. **改名し、inner valid の境界 gap を `purge_gap` だけにする**（#273 の主張: 追加の除外に漏洩上の理由は無い）。inner-train の行が増える振る舞いの変更になる。統合すれば 2 つ目の量そのものが無くなるので、この区別が要らない。
3. **後ろ向きの embargo を実装する**: 検証ブロックの後ろに学習行を持つ別の splitter が要り、この splitter の変更ではない。
4. **乖離を文書に書くだけにする**: 名前を知る読み手を誤らせ続ける。
5. **`embargo: 0` だけ警告しない**: 以前の artifact の読み込みで出る警告は減るが、非推奨のキーを黙って受理する条件が増え、v1.0 の削除で同じ問題が残る（上の注記）。採らない。

### 受け入れ基準（テスト観点）

2〜4、7〜10 は修正前に RED になる（`embargo` が `purge_gap` に足されない、属性や `model_dump()` の `embargo` が残る、値の検査が無い、文書と登録簿が古い）。1・5・6 は修正前から通る互換の固定で、除外の量が変わらないこと（決定 5）を守る。

1. **幾何の事実**（修正前から GREEN、固定）: `PurgedTimeSeriesSplitter` の全 fold で、学習の最大 index が検証の最小 index より小さい。`n_samples`、`n_splits`、`purge_gap`、`max_train_size` / `max_test_size` の組み合わせで確かめる。将来の splitter の変更でこれが崩れたら落ちる。
2. **統合**: `{"purge_gap": 5, "embargo": 2}` は `DeprecationWarning` を出し、`purge_gap == 7`、`model_dump()` に `embargo` が無い。`n=240, n_splits=4` の fold が #273 の実測（fold 0: 学習 `0..40`、検証 `48..95`、…、除外 7）と一致し、`{"purge_gap": 7}` の fold とも一致する。修正前は `purge_gap == 5` のままなので RED。
3. **旧キーと値の検査**: `embargo: 0` も `DeprecationWarning` を出す。`embargo_pct: 2` と `gap: 2` も `purge_gap` に加算する。3 つの綴りの警告文はどれも `purge_gap` を名指す（文面を正規表現で確かめる）。3 つのうち 2 つ以上は `CONFIG_INVALID`。2 つ目の gap の `True` / 端数（`0.5`）/ 負の値、負の `purge_gap`（`purge_gap: -1, embargo: 2` を含む）は `CONFIG_INVALID`（`embargo: true` と負の値は修正前は Config 検証を通るので RED）。`purge_gap: "5"` と `embargo: "2"` は今日どおり受理され、`purge_gap == 7`。綴りごとの境界の値は今日と同じ: `embargo: "1e3"` と `embargo: "1e400"` は `CONFIG_INVALID`、`embargo: "9007199254740993"` は丸めずに加算される。`gap: "1e400"` と `embargo_pct: "1e400"` は `CONFIG_INVALID`（修正前は `OverflowError` が漏れるので RED）。
4. **splitter**: `PurgedTimeSeriesSplitter(purge_gap=5, embargo=2)` は `DeprecationWarning` を出し、`PurgedTimeSeriesSplitter(purge_gap=7)` と同じ fold を返す。`PurgedTimeSeriesSplitter(embargo=0)` も警告し、`embargo` を省いた構築は警告しない。`purge_gap=-1, embargo=2` と負の `embargo` は `ValueError`。構築した splitter に `embargo` 属性が無い。
5. **inner valid**: `embargo` を書いた Config の自動解決された inner valid の gap が、outer の除外（`purge_gap` の合計）と等しい。#273 の例（outer 学習 192 行、ratio 0.1）で inner-train は今日と同じ 166 行。
6. **end-to-end**: `{"purge_gap": 5, "embargo": 2}` と `{"purge_gap": 7}` で、`fit` は同じ `splits` と OOF を返し、binary の Platt 校正つきの `fit` は同じ校正の fold の index を返し（`test_calibration_split.py` の「`purge_gap` より大きい」という緩い比較を、一致の比較に書き換える）、`tune`（少数の trial、seed 固定）は同じ trial の score を返す。
7. **Config の往復**: 旧キーで作った Config の `model_dump()` に `embargo` が無く、それを読み直すと警告を出さず（警告をエラーにして確かめる）同じ `purge_gap` になる。`PurgedTimeSeriesConfig` に `embargo` 属性が無い。
8. **以前の artifact**: export した artifact の `metadata.json` の `config.split` を `{"purge_gap": 5, "embargo": 2}` に書き換える（`purge_gap: 7` で fit したモデル）。`load()` は `DeprecationWarning` を出し、読み込んだモデルの Config が `purge_gap == 7` で `embargo` を持たない。分割を使う経路で確かめる: 読み込んだモデルの `export_code` の split ブロックが `purge_gap: 7` で、読み込んだモデルで `fit` し直した `splits` が元のモデルと一致する。`predict()` は export 前と一致する。`embargo: 0` の artifact も読める。
9. **生成コード**: `export_code` の split ブロックに `embargo` が無く、生成された `train.py` の fold が `purge_gap` の合計で再現される（`test_split_reproduction.py`）。
10. **文書と登録簿**: BLUEPRINT §10.2 が決定 11 の幾何を述べ、§5 の Config の表と旧キーの規則、§10.3.1、§10.3 に `embargo` が Config の旧キーとしてしか現れない（§5.5 の構成値の表の `PurgedTimeSeriesSplitter.embargo` の `api` の行は、非推奨の構築子引数を述べるもので、これに当たらない）。`docs/config-reference.md` の図と表に `embargo` の列が無い。`docs/DEPRECATIONS.md` に `embargo` → `purge_gap`、`embargo_pct` / `gap` → `purge_gap`、splitter の引数、v1.0 の読み込み経路の注記があり、旧キーの出典が H-0038 / H-0040。チュートリアル notebook が `embargo` を使わない。knob registry の `PurgedTimeSeriesSplitter.embargo` が `api` で、直接構築で値が届くことを実行で確かめるセルがあり、`test_knob_reachability.py` の網羅の照合が通り、BLUEPRINT §5.5 の表と件数（59 個 / 15 個）が台帳と一致する（同じテストが照合する）。`docs/proposal_dispositions.toml` の H-0115 は `specified`。

### 実装時の記録

- **BLUEPRINT の引用の固定**: Phase 3 の fold map（`test_pr9_fold_map_check.py`）は §5 の「`gap` → `embargo`（H-0038 は `embargo_pct` に写すと決めたが」を、H-0085 の disposition は「`purge_gap + embargo`」を、それぞれ BLUEPRINT の文として固定している。過去の監査記録は書き換えず、BLUEPRINT ではそれらを H-0115 より前の規則として残した（§5 の旧キーの規則、§10.3.1）。
- **`PurgedTimeSeriesConfig` の検証**: `mode="wrap"` の validator で、`purge_gap` を含む残りのフィールドを先に検証してから、合計を `purge_gap` に入れて検証し直す（結果は `purge_gap` を直接書いた入力と同じ作り方になり、`model_fields_set` にも入る）。

## H-0116: `None` 以外の文字列でない `objective` を `CONFIG_INVALID` で拒否する（#270）

- **ステータス**: Accepted
- **起票日**: 2026-10-07
- **決定日**: 2026-10-07（管理者の判断: #270 の PR で直す）
- **スコープ**: `lizyml/estimators/lgbm/param_validation.py`（`check_objective_compatible`）, `BLUEPRINT.md` §14.2, `CHANGELOG.md`, テスト（`tests/test_estimators/test_h0079_followup.py`）
- **関連**: [Issue #270](https://github.com/nbx-liz/LizyML/issues/270), H-0079（task 非互換の `objective` を `CONFIG_INVALID` にした。G3 は端の入力でも未加工の `TypeError` / `KeyError` を出さないと定めた）

### 目的（課題）

`check_objective_compatible(task, objective)` は `objective not in TASK_COMPATIBLE_OBJECTIVES[task]` で互換性を調べる。集合の包含はハッシュを使うので、dict や list の値はそこで `TypeError: unhashable type` を送出し、`CONFIG_INVALID` にならない。`Model.fit` の 2 つの経路（`model.params` と `fit(params=)`）で再現した（`1d41b66`）。facade の `check_param_values` は `LizyMLError` だけを捕まえるので、`TypeError` はそのまま利用者に届く。

これを固定するはずのテスト `test_dict_form_objective_raises` は `pytest.raises((LizyMLError, TypeError))` で、`TypeError` も「許容」していた。#270 の再点検で WEAK と判定し、`CONFIG_INVALID` だけを受け入れる形に締めたところ RED になった。

### 対応方針（決定）

1. **`check_objective_compatible` は、文字列でない値を包含の検査より前に `CONFIG_INVALID` で拒否する。** 受理される objective はすべて文字列なので、文字列でない値はどれも互換ではない。ただし明示の `None` は今日どおり「上書きなし」で、拒否しない: 2 つの呼び出し元（`_model_factories.py` の `check_param_values` と `adapter.py` の `_build_params`）は `None` をこの検査に渡さず、task の既定の objective で学習する。`context`（`task` / `objective` / `valid_objectives`）は task 非互換のときと同じ形にする。`user_message` は、文字列なら今日どおりその値を、文字列でない値なら型名（`objective of type 'dict'`）を示す。
2. **包含の検査とメッセージの組み立てでは、拒否する値を書式化もハッシュもしない。** 文字列でない値は型名だけで示し、文字列は `str` の素の複製（`str.__str__`）で包含を調べて表示する。値を書式化すると、その `__repr__` / `__format__` が例外を出したとき、`CONFIG_INVALID` の前に未加工の例外が漏れる（キーが表示できない dict で再現）。`str` の部分クラスが上書きした `__hash__` / `__eq__` / `__format__` / `__str__` / `__repr__` も同じ理由で使わない。保証はこの 2 つの操作に限る。**範囲外**: 値の型の参照やその属性の参照が例外を出すもの（`isinstance` が読む `__class__`、`__name__` で例外を出すメタクラスなど。`Model.fit` の経路では手前の値の型の検査が先に型名を読むので、そこで漏れる）と、送出した後にエラーを表示すること（`context["objective"]` は書いた値を保つので、`repr(error)` はその値の `__repr__` を呼ぶ）。拒否した値の報告で値のコードが一切動かないことは、Python では保証できない（traceback が局所変数を表示するときも `repr` を呼ぶ）ので約束しない。
3. 呼び出し元は変えない。facade（`check_param_values`）は今日どおり `CONFIG_INVALID` に入力の層の名前を付けて送出し直す。

### 規則が縛る位置（ソースから導出）

規則: **組み込みの型の `objective`（dict / list / set / int など）が、互換性の検査の包含で未加工の `TypeError` / `KeyError` を出さない。** 導出: `lizyml/` で `check_objective_compatible` と `TASK_COMPATIBLE_OBJECTIVES` を grep した全件（`1d41b66`）。包含を調べるのは `check_objective_compatible` 1 か所で、それを呼ぶのは 2 か所。`provider.py` の `TASK_COMPATIBLE_OBJECTIVES` は `objective_choices` の一致の確認と choices の生成で、利用者の値を包含で調べない。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `estimators/lgbm/param_validation.py` `check_objective_compatible` | 文字列でない値を先に拒否する |
| 2 | `core/_model_factories.py` `check_param_values`（`model.params` / `fit(params=)` の facade の経路） | 変更なし（1 の結果を層の名前付きで送出し直す） |
| 3 | `estimators/lgbm/adapter.py` `_build_params`（adapter を直接作る経路） | 変更なし（1 の結果がそのまま届く） |

### 互換性

- **振る舞いの変化**: dict / list などハッシュできない `objective` は、`TypeError` だったところで `CONFIG_INVALID` になる。ハッシュできる文字列でない値（`42` など）は今日も `CONFIG_INVALID` で、変わらない（`user_message` は値の代わりに型名を示す）。通常の `str` の値の振る舞いは変わらない。`str` の部分クラスは、上書きしたメソッドを使わず文字列の内容で判定する。明示の `None` は今日どおり「上書きなし」（task の既定）で、変わらない。
- どちらの場合も学習の前に止まる（今日の `TypeError` も学習の前）。学習できていた通常の入力（文字列と組み込みの型の値）で拒否されるものは無い。決定 2 の範囲外の値は、この評価の対象でもない。
- **Firing rate**: 本提案の条件は入力の検証であり、Change Gate の 6 つの目的のどれでもない。
- `format_version` / Config のスキーマ / 公開 API のシグネチャは変わらない。

### 代替案（検討して棄却）

1. **`TypeError` を捕まえて `CONFIG_INVALID` に変える。** 包含の検査が別の理由で出す例外まで同じ扱いになる。値の型で先に分けるほうが、何を拒否したかが明確である。
2. **Config のスキーマ（pydantic）で `objective` を文字列に限る。** `fit(params=)` と adapter の直接構築は Config を通らないので、穴が残る。
3. **テストを元の「`TypeError` も可」に戻す。** H-0079 G3 の契約（未加工の `TypeError` を出さない）と矛盾する。

### 受け入れ基準（テスト観点）

1. adapter を直接作る経路で `{"objective": {"huber": {}}}` は `CONFIG_INVALID` で、`context["objective"]` に書いた値が入る。修正前は `TypeError` で RED。
2. `Model.fit` の `model.params` と `fit(params=)` の両方で、dict と list の `objective` は `CONFIG_INVALID` で、`lgb.train` は一度も呼ばれない。修正前は dict / list で RED。set は Python が包含の検査で frozenset に変えるので修正前から `CONFIG_INVALID`（同じテストで固定）。
3. 既存の `objective` のテスト（`test_h0079_followup.py`、`tests/test_estimators/`）が緑のまま。
4. BLUEPRINT §14.2 が文字列でない値の扱いを書き、`tests/test_docs/` が緑のまま（H-0116 の処分の行を含む）。
5. 明示の `None` は拒否されない: `model.params` と `fit(params=)` の両方で `lgb.train` は task の既定の objective を受け取る（`test_none_objective_through_model_fit_trains_on_the_default`）。adapter を直接作る経路は `test_none_objective_falls_back_to_default` が固定する。
6. 包含の検査とメッセージの組み立てで、決定 2 が挙げた操作が値に対して走らない（範囲外の 2 つは固定しない）: 表示できないキーを持つ dict は adapter を直接作る経路で `CONFIG_INVALID`（`objective of type 'dict'`）で、修正前はキーの例外が漏れて RED（`test_an_unprintable_objective_is_still_config_invalid[adapter]`。`Model.fit` の経路は値の型の検査が先に `CONFIG_INVALID` で止めるので、同じテストの `[fit_params]` はそれを固定する）。`__hash__` / `__eq__` / `__format__` / `__str__` / `__repr__` が例外を出す `str` の部分クラスは、互換でない内容なら `CONFIG_INVALID`、互換な内容なら受理（`test_a_hostile_string_objective_is_judged_by_its_text`。修正前は RED）。

## H-0117: リリースを develop → main の Merge commit に限り、git 管理下の運用ルールを CONTRIBUTING.md に置く

- **ステータス**: Accepted
- **起票日**: 2026-10-08
- **決定日**: 2026-10-08（管理者の判断: v0.18.0 の前に、リリース経路と文書の役割を直す計画を承認）
- **スコープ**: `scripts/release.py`, `.github/workflows/auto-release.yml`, `.github/scripts/`（`check_release_merge.sh` / `release_version.sh` / `create_release_tag.sh`、いずれも新規）, `CONTRIBUTING.md`, テスト（`tests/test_release/`）, `docs/proposal_dispositions.toml`
- **関連**: Codex の設計レビュー `lizyml-agents-critique`（2026-10-08、指示ファイルの整理の検討で見つかった）、`CONTRIBUTING.md` の Release 節

### 目的（課題）

リリースの手順は `CONTRIBUTING.md` が「develop から main への PR を Merge commit で統合し、`auto-release.yml` がタグを作る」と定めている。v0.8.1 以降の 13 回のリリースはすべてこの形だった。しかし、手順を実行する道具と、手順を説明する文書の両方に、この形から外れる道が残っている。

1. **`scripts/release.py` が develop に直接 commit して push する。** `CHANGELOG.md` に未 commit の変更があると、develop の上で commit し、続けて `git push origin develop` を実行する。`CONTRIBUTING.md` は develop への直接 commit と直接 push を禁じ、CHANGELOG は feature PR で入れると定めている。
2. **`auto-release.yml` がリリースの形を確かめない。** 起動条件は「main に merge された PR のタイトルが `release:` で始まる」だけである。head が develop でない PR（2026-04-02 までの `release/vX.Y.Z` ブランチ）や、Squash merge された PR でもタグを作る。Squash merge は main だけにある commit を作り、main と develop の履歴を切り離す。加えて、PR のタイトルを `run:` の中で `${{ github.event.pull_request.title }}` として直接展開しているので、タイトルの文字列がシェルとして解釈される。
3. **`CONTRIBUTING.md` の文書の優先順位が、この public リポジトリでは git 管理下にないファイルを挙げている。** 3 位の `AGENTS.md` と 4 位の `skills/*` は、エージェントへの指示のためのローカルファイルで、`.gitignore` の対象である（管理者の方針: public リポジトリでは指示ファイルを git で管理しない）。新しい clone、CI、commit を読むレビューのどれにも届かない文書が、運用ルールの正として挙がっている。`PLAN.md` の位置づけは書かれていない。言語の規約も `CLAUDE.md` を挙げている。

### 対応方針（決定）

1. **`scripts/release.py` は commit も push もしない。** `CHANGELOG.md` に未 commit の変更があれば、feature PR で入れるよう案内して終了コード 1 で止める。ローカルの develop が `origin/develop` と一致しなければ（未 push の commit がある、または遅れている）、同じく止める。どちらも満たせば、`release: vX.Y.Z` というタイトルで develop → main の PR を作るだけにする。
2. **`auto-release.yml` は、タグを作る前に PR の形を確かめる。** 確かめる内容は 3 つで、`.github/scripts/check_release_merge.sh` に置く。
   1. PR の head ブランチが `develop` であること。
   2. head が同じリポジトリのものであること。
   3. merge commit（`merge_commit_sha`）の親がちょうど 2 つであること。

   1 つでも満たさなければ、タグも Release も作らずに失敗する。

   加えて、次の 3 つを定める。

   - **バージョンは、タイトル全体が `release: vX.Y.Z` と一致するときだけ読む**（`.github/scripts/release_version.sh`）。`-rc1` などの接尾辞、2 つ目のバージョン、後ろに続く文字列があれば拒否する。部分一致で読むと、`release: v0.18.0-rc1` から正式版のタグ `v0.18.0` を作ってしまう。
   - **タグは checkout の HEAD ではなく `merge_commit_sha` に付け、再実行に耐える形で作る**（`.github/scripts/create_release_tag.sh`）。同じタグが既に `merge_commit_sha` を指していれば再利用して push し直し、別の commit を指していれば拒否する。タグを push した後に GitHub Release の作成や PyPI の起動が失敗しても、workflow を再実行すれば同じリリースを完了できる。GitHub Release も、既にあれば作り直さない。
   - PR のタイトルなど利用者が書ける値と、前の step の出力は、`env:` を経由してシェルに渡し、`run:` の中で `${{ }}` を展開しない。
3. **git 管理下の運用ルールの正を `CONTRIBUTING.md` にする。** 文書の優先順位は `BLUEPRINT.md` > `HISTORY.md` > `CONTRIBUTING.md` > 実装コードとする。エージェント向けの指示ファイル（`CLAUDE.md`, `.claude/AGENTS.md`, `.claude/skills/`）はローカルで git の管理外にあり、このリポジトリの規則を要約するだけで上書きしないと書く。`PLAN.md` は予定と進捗の文書であって仕様ではないと書く。言語の規約の `CLAUDE.md` を、指示ファイル全般を指す書き方に直す。

### 規則が縛る位置（ソースから導出）

規則: **リリースのタグは、develop を Merge commit で main に統合した commit にだけ付き、その過程で develop に直接 commit も push もしない。** 導出: リポジトリで `git push`、`git commit`、`git tag`、`gh pr create --base main`、`release:` を grep した全件（`ac6c67d`）。

| # | 位置 | 本 PR |
|---|---|---|
| 1 | `scripts/release.py`（CHANGELOG の commit と develop の push） | 拒否に置き換える |
| 2 | `scripts/release.py`（`gh pr create --base main --head develop`） | 残す。タイトルを `release: vX.Y.Z` に揃える |
| 3 | `.github/workflows/auto-release.yml`（`git tag` と `git push origin <tag>`、タイトルからのバージョンの読み取り、`gh release create`） | 形の検査の後に、完全一致のタイトルから読んだバージョンで、`merge_commit_sha` に再実行に耐える形でタグを付ける。Release も既にあれば作り直さない |
| 4 | `.github/workflows/release.yml`（PyPI 公開。3 からの `workflow_dispatch` のほか、GitHub Release の `published` と手動の `workflow_dispatch` でも起動する） | 変更なし。3 が拒否すればこの経路からは起動しない。手動で Release を公開する経路と手動の起動は運用者の操作で、本提案の範囲外 |
| 5 | `CONTRIBUTING.md` の Release 節 | 手順そのものは変えない（すでに正しい）。手順 2 に `release.py` で同じ PR を作れることと、その拒否条件を、手順 5 に `auto-release.yml` の検査（PR の形、タイトルの完全一致）と、再実行でタグを再利用することを書き足す |
| 6 | `.githooks/protected-refs.sh`（`develop` を保護ブランチに含める）と `.githooks/pre-push`（保護ブランチへの push を拒否する） | 変更なし。develop への直接 push をローカルで拒否する仕組みで、決定 1 と同じ規則を git の側で支える。届くのは hooks を入れた clone だけ |
| 7 | `PLAN.md` の 25-G（v0.1.1 のリリース手順。「main に `git tag v0.1.1` を打つ」） | 変更なし。v0.1.1 当時の作業記録で、決定 3 のとおり `PLAN.md` は仕様ではない。現在の手順は `CONTRIBUTING.md` が定める |

### 互換性

- ライブラリの公開 API、Config、Result、`format_version` は変わらない。
- **`scripts/release.py` の振る舞いが変わる。** これまで代わりに行っていた CHANGELOG の commit と develop の push をしなくなる。運用者は、`CONTRIBUTING.md` のとおり CHANGELOG の feature PR を先に merge しておく必要がある。PR のタイトルから `— LizyML X.Y.Z` が消え、`CONTRIBUTING.md` の `release: vX.Y.Z` と一致する。
- **`auto-release.yml` は、形の違うリリース PR と、タイトルが `release: vX.Y.Z` と完全に一致しないリリース PR を拒否するようになる。** その場合はタグも Release も PyPI 公開も起きない。過去に `scripts/release.py` が作っていた `release: vX.Y.Z — LizyML X.Y.Z` という形のタイトルも拒否される。この PR で `release.py` はその形を作らなくなる。
- 既にタグがある状態での再実行は、これまで「Tag already exists」で止まっていた。タグが `merge_commit_sha` を指していれば続行するようになる。
- **Firing rate**: 13/15 for the head and parent checks, and 8/15 with the exact-title check added, of release-titled PRs merged into main whose merge commit is still in history (replayed with `gh pr list --base main --state merged` and `git rev-list --parents`). The head and parent checks refuse the 2 `release/v0.7.3-fix` and `release/v0.8.0` squash merges of 2026-04-02 under the superseded rule. The title check also refuses 5 legitimate releases (#97, #100, #104, #151, #158) whose titles carried the `— LizyML X.Y.Z` suffix that `scripts/release.py` generated until this PR; with the suffix removed here, every release made by the documented procedure passes. The 14 release PRs before the 2026-04-02 history rewrite have no merge commit left to inspect.

### 代替案（検討して棄却）

1. **GitHub の ruleset で main への merge 方式を Merge commit だけに制限する。** サーバー側で効く点は強いが、設定はリポジトリの外にあり、PR の差分としてレビューできない。リポジトリ全体の merge 方式の設定では、develop への Squash merge と main への Merge commit を両立できない。補完として後で検討できるが、本提案の代わりにはしない。
2. **`release.py` が CHANGELOG 用のブランチを切って PR まで作る。** 自動化は進むが、手順が 2 本の PR にまたがり、途中で止まったときの扱いが増える。手順書どおりの「先に feature PR」を守らせる方が単純である。
3. **`CONTRIBUTING.md` の優先順位に指示ファイルを残す。** 新しい clone や CI に届かない文書が正として挙がり続ける。今回それらが黙ってリンク切れになっていたことが、この形の弱さを示している。

### 受け入れ基準（テスト観点）

1. `release.py` は、`CHANGELOG.md` に未 commit の変更があると終了コード 1 で止まり、`git commit` も `git push` も実行しない（実行したコマンドを記録して確かめる）。
2. `release.py` は、ローカルの develop と `origin/develop` が一致しないと終了コード 1 で止まり、`gh pr create` を実行しない。
3. `release.py` の正常系は、`git push` を一度も実行せず、`gh pr create --base main --head develop` をタイトル `release: vX.Y.Z` で 1 回だけ実行する。
4. `check_release_merge.sh` を実際のリポジトリで実行し、2 つの親を持つ merge commit と head `develop` の組では成功し、親が 1 つの commit、親が 3 つの commit（octopus merge）、head が `develop` 以外、head が別リポジトリ、merge commit が checkout に無い、merge commit が空の 6 通りでは失敗する。
5. `release_version.sh` は `release: v0.18.0` から `tag=v0.18.0` と `version=0.18.0` を出力し、接尾辞付き、バージョン 2 つ、改行入り、`— LizyML` 付き、`v` なし、大文字始まり、先頭の空白、バージョンなし、空文字列のタイトルでは失敗して何も出力しない。
6. `create_release_tag.sh` を `origin` を持つ実際のリポジトリで実行し、新規のタグは merge commit に付いて push される。再実行は成功してタグを変えない。push されていないタグは再実行で push される。別の commit を指す既存のタグは拒否して何も push しない。checkout に無い merge commit は拒否する。
7. `auto-release.yml` は、検査・バージョン・タグの step をこの順に置いて上の 3 つのスクリプトを実行し、各 step の `env:` は期待どおりの対応表と完全に一致し、どの `run:` も `${{ }}` を展開しない（YAML を読んで確かめる）。
8. `CONTRIBUTING.md` の文書の優先順位と言語の規約が決定 3 のとおりで、`tests/test_docs/` が緑のまま。

## H-0118: 結果フィールドの一覧を dataclass に揃え、テストで固定する（#326）

- **ステータス**: Accepted
- **起票日**: 2026-10-08
- **決定日**: 2026-10-08（管理者の判断: v0.18.0 の前の文書更新に #326 を含める）
- **スコープ**: `BLUEPRINT.md` §7.1, `docs/api.md`（FitResult の表）, テスト（`tests/test_docs/test_result_field_inventories.py`、新規）, `docs/proposal_dispositions.toml`
- **関連**: [Issue #326](https://github.com/nbx-liz/LizyML/issues/326), H-0002（結果の型をゴールデンテストで固定）, H-0030（校正の入力を生スコアにした）, H-0070（`target_encoder`）, H-0114（`pipeline_state_per_fold` を両方の一覧に足した）

### 目的（課題）

`FitResult` のフィールドを文章で並べた一覧が 2 つあり、どちらも dataclass（`lizyml/core/types/fit_result.py`）とずれていた。`BLUEPRINT.md` §7.1 には `oof_raw_scores` が無く、`docs/api.md` の FitResult の表には `target_encoder` が無い。ゴールデンテストは dataclass を固定するだけなので、文書がずれても CI は緑のままだった（DC3）。

### 対応方針（決定）

1. **実装が正で、文書を実装に揃える。** §7.1 に `oof_raw_scores` を足し、`docs/api.md` の表に `target_encoder` を足す。フィールドの形も意味も変えない。
2. **一覧と dataclass の照合をテストにする。** 各文書の一覧からフィールド名の集合を読み取り、`dataclasses.fields` と比べる。対象は `FitResult`（§7.1 と `docs/api.md`）と `PredictionResult`（§7.3 と `docs/api.md`）の 4 組。一覧を読み取れなかった場合も失敗させる（空の集合で通さない）。

### 互換性

- コードも公開 API も変わらない。文書の記述と、それを検査するテストだけが増える。
- 今後 `FitResult` か `PredictionResult` にフィールドを足すときは、2 つの文書の一覧にも足さないとテストが落ちる。
- **Firing rate**: 本提案の条件は文書の検査であり、Change Gate の 6 つの目的のどれでもない。

### 代替案（検討して棄却）

1. **文書の一覧をやめ、dataclass の docstring を参照させる。** 一覧は仕様の一部（BLUEPRINT）と利用者向けの説明（api.md）を兼ねており、消すと読み手が型の定義を読みに行く必要がある。
2. **`SplitIndices` と `RunMeta` も同じテストに含める。** #326 は検討を勧めているが、§7.1 では両者を入れ子の箇条書きで説明しており、決まった形の一覧になっていない。照合のために書き方を変えるのは本提案の範囲を超えるので、今回は入れない。

### 受け入れ基準（テスト観点）

1. `test_result_field_inventories.py` の 4 組がすべて緑。修正前は `FitResult` の 2 組が赤（§7.1 に `oof_raw_scores` が無い、`docs/api.md` に `target_encoder` が無い）。
2. どちらかの文書から 1 つのフィールドを消すと、その組が赤になる。
3. `tests/test_docs/` が緑のまま。

## H-0119: ノートブック索引を閉じた契約にする（#334）

- **ステータス**: Accepted
- **起票日**: 2026-10-09
- **決定日**: 2026-10-09（管理者の判断: 案 E を 2 PR で、CI の起動条件は 6. のとおり。Codex の Proposal review 3 round の最終 round で APPROVE）。改訂: 2026-10-10（決定 1、管理者の判断。Codex の改訂レビューは 3 回の run で行い、3 回目の run の round 1 で APPROVE）
- **スコープ**: `notebooks/*.ipynb`（メタデータとセルのタグ）, `docs/examples.md`（生成する部分）, `lizyml/_extras.py`（新規・非公開）, `scripts/examples_index.py`（新規）, `tests/test_docs/test_examples_index.py`（置き換え）, `tests/test_notebooks/`（実行の記録、`test_index_execution.py` を新設、ネットワーク失敗のマーカー一覧を共有モジュールへ移す）, `.github/workflows/ci.yml`（matrix を作るジョブ、(a)(b) の 2 つのジョブ、gate ジョブを追加）, `scripts/notebook_index_gate.py`（gate の判定、新規）, `.github/scripts/notebook_index_scope.sh`（決定 1 で削除）, `pyproject.toml` / `uv.lock`（依存グループ `notebooks` を追加）, `docs/proposal_dispositions.toml`
- **関連**: [Issue #334](https://github.com/nbx-liz/LizyML/issues/334), PR #335（索引の修正と静的な検査）, H-0118（文書の一覧を実装と照合する、同じ DC3 の型）, H-0117（運用ルールの正は `CONTRIBUTING.md`）

### 目的（課題）

`docs/examples.md`（ノートブックの索引）は v0.18.0 に古いまま入り、ノートブックが呼ばないメソッドを挙げ、8 本中 7 本で extras を誤っていた。PR #335 で索引を直し、`ast` でノートブックの呼び出しを読み取って索引と照合するテストを足した。

ところが Codex の 2 つの run では、文書はどの round でも正しいと確認された一方、最後の 10 件の指摘のうち 9 件がこのテストの静的解析の抜け道だった。コメント、受け手、位置引数、`**`、重複見出し、代入順と再代入、到達しない分岐、`def` / `class` による再代入、`tutorial_*` だけの照合、式の中の条件付き実行と、round ごとに形が 1 つずつ増えた。

選択肢の批評（run `examples-options-critique`、head `aa9228f`）は次を示した。

- 「呼び出しが実行されるか」は、`ast` のノードの種類を全部分類しても決まらない。同じノードでも、どの子がどの位置にあるかで実行されるかが変わる。
- 前の文が例外や `sys.exit` で止まれば後ろは実行されないが、任意の Python でこれを判定することはできない。

検査の対象を Python の構文全体から、索引の約束に必要な小さな文法へ移す。

### 対応方針（提案）

1. **索引の約束を狭める。** 索引が保証するのは「各ノートブックが例として示すメソッド」と「そのノートブックの実行に必要な extras」の 2 つだけである。ノートブックの全呼び出しの一覧ではない。各節の説明文は手書きとし、約束の外に置く。

2. **ノートブックごとの宣言を正にする。** 各ノートブックの `metadata.lizyml.index` は、JSON object で、キーは `models`、`methods`、`extras` のちょうど 3 つとする。ほかのキーがあれば失敗し、キーが欠けても失敗する。3 つの値はどれも、文字列の JSON 配列で、重複が無く、昇順に並んでいなければならない。
   - `models`: 空でない。各要素は Python の識別子で、キーワードではない。例の受け手になる変数名を並べる。
   - `methods`: 空でない。各要素は `Model` の公開メソッド名で、`_` で始まらず、`inspect.getmembers(Model)` に callable として存在する。継承した mixin のメソッドも含む。公開メソッドは instance method（`Model` の MRO 上の関数）に限り、classmethod / staticmethod / property は除く。受け手の検査がインスタンスに束縛するためである（実装時に追加、2026-10-09、実装 review round 1）。
   - `extras`: 空でもよい。各要素は対応表（4.）に現れる extra 名で、現行では `explain`、`plots`、`tuning` のどれかである。

   宣言が無いノートブックや、型・キー・並びのどれかが規則に反するノートブックは失敗させる。

3. **例は `index-example` タグのセルに置き、閉じた文法で書く。** タグ付きセルを `ast` で読み、次の文法に当てはまらないものはすべて失敗させる。

   - **セル**: `%` / `!` で始まる行を含まない。文が 1 つ以上ある。
   - **文**: 次の 2 つだけを許す。
     - 式文 `R.m(引数…)`
     - 代入 `N = R.m(引数…)`

     条件は次のとおり。
     - `R` は `models` の名前、`m` は `methods` の名前である。
     - `N` は単純な名前で、`models` に含まれない。
     - 代入の左辺はちょうど 1 つ。
   - **引数**: 位置引数とキーワード引数（`*` / `**` の展開は不可）。値は次の「値」に限る。
   - **値**: 次のどれか。それ以外の式は拒否する。
     - 定数
     - 名前
     - 名前と属性の連鎖（`a.b.c`）
     - 値を添字にした値の添字（`a["x"]`、`a[0]`）
     - 単項マイナスの定数
     - 値の list / tuple / set / dict

     したがって、関数呼び出し、`lambda`、内包表記、`and` / `or`、条件式、代入式、`await` などは拒否される。
   - **extras に関わる引数**（対応表が条件に使う `kind` と `return_shap`）は、定数で書く。`importance` と `importance_plot` の `kind` は、位置引数 0 で渡してもよい。

   セルとメソッドの対応は、次の 2 つで照合する。
   - タグ付きセルに現れる `m` の集合は `methods` と一致する。
   - タグ付きセルが 1 つも無いノートブックは失敗する。

   タグの無いセルには制約を置かない。

   現行のノートブックには、2 つの文の形に収まらない呼び出しがあり、例として示す場合は書き直す。たとえば次の 2 つである（2026-10-09 時点、round 2 の指摘）。
   - `model.plot_learning_curve().show()`（回帰）→ タグ付きセルで `fig = model.plot_learning_curve()` とし、`fig.show()` はタグの無いセルに置く。
   - `pd.DataFrame({"pred": model.predict(X_new).pred})`（codegen）→ タグ付きセルで `result = model.predict(X_new)` とし、残りはタグの無いセルに置く。

   書き直しは次の 3 つの形に限る。
   - 呼び出しの連鎖を代入とタグの無いセルに分ける。
   - 式の中の呼び出しを先に代入する。
   - 2 つの形の文と、それ以外の文（`print` など）が同じセルにあれば、セルを分ける。

   どのノートブックも、実行に要る extra のそれぞれについて、それを導くタグ付きの文を少なくとも 1 つ持つ必要がある（4. と 6.(b) から従う）。実装時の調べ（2026-10-09）では、8 本で約 10 か所の書き直しが要る。

4. **メソッドから extra への対応表をパッケージ内に置く**（`lizyml/_extras.py`、非公開）。

   項目は、現行コード（2026-10-09、`d72b20a`）から調べたものである。

   | メソッド | 条件 | extra |
   |---|---|---|
   | `tune` | 常に | `tuning` |
   | `predict` | `return_shap=True` | `explain` |
   | `importance` | `kind="shap"` | `explain` |
   | `importance_plot` | 常に | `plots` |
   | `importance_plot` | `kind="shap"` | `explain`（`plots` に加えて） |
   | `residuals_plot`、`roc_curve_plot`、`calibration_plot`、`probability_histogram_plot`、`plot_learning_curve`、`plot_oof_distribution`、`tuning_plot` | 常に | `plots` |

   条件に使う引数の既定値は、`predict(return_shap=False)`、`importance(kind="split")`、`importance_plot(kind="split")`、`residuals_plot(kind="all")` である。引数を省いた場合は、この既定値で判定する。

   scipy は scikit-learn と lightgbm が無条件で要求するため、base install に必ず入っている。そのため、`calibration` extra（Beta）も、`residuals_plot` の qq / all が使う scipy も、対応表には入れない。

   宣言の `extras` は、タグ付きセルの呼び出しから対応表で導いた集合と一致しなければならない。

5. **`docs/examples.md` の先頭を生成領域にする**（決定 1 で改訂、2026-10-10）。

   - **生成領域**: `docs/examples.md` の先頭の K 行を生成領域とする。K は、ノートブックの数に 8 を足した数である。生成領域は次の 2 つだけから作り、次の K 行と完全に一致しなければならない。
     - `notebooks/*.ipynb` のファイル名を、Python の `sorted()`（文字列のコードポイント順）で並べた順
     - 各ノートブックの宣言（2.）
     1. `# Notebook Index`
     2. 空行
     3. `<!-- Generated from each notebook's metadata.lizyml.index by scripts/examples_index.py. Do not edit this region by hand. rows=N sha256=D -->`。N はノートブックの数を、先頭に 0 を付けない十進数で書く。D は、7. の N 行の中身を `"\n".join(rows).encode("utf-8")` としたバイト列の SHA-256 を、64 文字の小文字の 16 進数で書く。行の区切りそのものは D に含めない。
     4. 空行
     5. `| Notebook | Demonstrates | Extras required |`
     6. `|---|---|---|`
     7. ノートブックごとに 1 行。`` | `<名前>.ipynb` | `m1()`, `m2()` | `pip install 'lizyml[e1,e2]'` | ``。メソッドは `methods` の順、extras は `extras` の順に並べる。`extras` が空なら、3 列目は `none (base install)` とする。
     8. 空行
     9. `<!-- index:end -->`
   - **名前**: `<名前>` に使える文字は、ASCII の英字と数字、`_`、`.`、`-` だけとする。これ以外の文字を含む名前のノートブックがあれば、`--check` も `--write` も失敗する。宣言（2.）に誤りのあるノートブックがある場合と、`notebooks/*.ipynb` が 1 本も無い場合も同じである。いずれの場合も、`--write` はファイルを変えない。そのため N は常に 1 以上である（決定 1 の改訂の review round 3 で明記、2026-10-10）。
   - **行の区切り**: 行は LF、CRLF、CR のどれで区切られていてもよい。照合するのは各行の中身で、区切りの種類は照合しない。
   - **検査の範囲**: `--check` が見るのは生成領域の行だけである。生成領域より後ろの手書きの部分（ノートブックごとの説明、extras の入れ方など）は読まず、書き方も制限しない。また、その表示も保証しない。Markdown として解釈せず、物理的な行が一致するかだけを見る。
   - **失敗する場合**: 先頭の K 行が上の行と 1 行でも一致しなければ、`--check` は失敗する。ファイルが K 行より短い場合も含む。これには、行の欠け、余分な行、行の順序の違い、古い内容、ノートブックの増減、K 行目に `<!-- index:end -->` が無い場合が含まれる。
   - **`--write`**:
     - **古い生成領域**: 古い 3 行目から、古い行数 N と古いダイジェスト D を読む。古い生成領域は、ファイルの先頭の N+8 行とする。形を探すことはしない（決定 1 の改訂の review round 2 と、批評 run `h0119-boundary-critique` を受けて、管理者の判断、2026-10-10）。

       次のすべてを満たす場合だけ、古い生成領域を受け付ける。
       - 古い 3 行目が上の形である。N は先頭に 0 の無い正の十進数、D は 64 文字の小文字の 16 進数
       - 1、2、4、5、6 行目が、上の固定の行と完全に一致する
       - 続くちょうど N 行が、どれも `| ` で始まる
       - その次の行が空行で、さらにその次の行が `<!-- index:end -->` である
       - その N 行から上の方法で計算した SHA-256 が、D と一致する

       1 つでも満たさなければ、`--write` は失敗する。その場合は「生成領域を版管理から戻して、もう一度実行する」よう報告し、ファイルを変えない。

       報告には、最初に満たさなかった条件を、次の理由のどれか 1 つとして含める。条件は上の順に調べる。
       - `header-format`: 3 行目が上の形でない。N または D の書式の誤りを含む
       - `fixed-lines`: 固定の行が一致しない
       - `row-prefix`: 数えた行が `| ` で始まらない
       - `boundary`: 空行、または `<!-- index:end -->` が無い。ファイルが短い場合を含む
       - `digest-mismatch`: D が一致しない

       テストは、この理由を確かめる（決定 1 の改訂の review run 2 の round 2 で追加、2026-10-10）。
     - **置き換え**: 受け付けた古い N+8 行を、今の宣言から作った K 行に置き換える。古い N+8 行より後ろの部分は、行の区切りも含めて 1 バイトも変えない。
     - **保証の境界**: D は、うっかりした編集を検出するための整合性の検査である。故意の改ざんを防ぐ認証ではない。N と D の両方を計算し直して書き換えたファイルは、その N+8 行を古い生成領域として指定したものとみなす。
     - **行の区切り**: 置き換えた K 行の区切りには、すべて、古い生成領域の 1 行目の区切りを使う。
     - **書き込み**:
       - 名前と宣言の検査、古い生成領域と D の検査は、書き込みより前に行う。
       - 書き込みは、同じディレクトリに作った一時ファイルに書いて閉じた後、`os.replace` で置き換える。
       - 置き換えより前に失敗した場合は、一時ファイルを消す。元のファイルは変わらない。
       - 電源断などへの耐久性は約束しない。

   `scripts/examples_index.py` は、`--write` で生成領域を書き換え、`--check` で不一致を報告する。テストは `--check` と同じ関数を呼ぶ。

6. **CI にジョブを足す。** 対応表の検査 (a) とノートブックの実行 (b) の 2 つの検査ジョブ、その matrix を作るジョブ、常に走る gate ジョブの 4 つである（ジョブの数は決定 1 の改訂の review round 3 で訂正、2026-10-10）。

   **(a) 対応表の検査。** extras は `explain`、`plots`、`tuning` の 3 つで、それぞれについて「その extra だけを除いた環境」を作る（matrix 3）。環境ごとに、その extra を要る対応表の項目を全部呼ぶ。そして `OPTIONAL_DEP_MISSING` が出ること、`context["package"]` がその extra のパッケージであることを確かめる。

   環境は `uv sync --frozen --no-dev --group notebooks --extra <残りの 2 つ>` で作る。実行は `uv run --no-sync --no-dev …` で行い、作った環境を作り直させない。実行の前に、除いたパッケージが import できないこと、残した 2 つが import できることを確かめる。

   各項目の前提は次のとおり。どれも、前提の検査で別のエラーが出た時点で失敗とする（依存の guard まで届かなかったものを通さない）。

   | 項目 | 前提 |
   |---|---|
   | `tune` | `tuning:` を持つ回帰の Config（`n_trials: 2`）で、`tune()` を呼ぶ |
   | `predict(return_shap=True)`、`importance(kind="shap")`、`importance_plot`、`residuals_plot(kind="scatter")`、`plot_oof_distribution` | 小さな合成データで fit した回帰モデル |
   | `plot_learning_curve` | inner valid を使う early stopping で fit した回帰モデル |
   | `roc_curve_plot` | fit した二値分類モデル |
   | `calibration_plot`、`probability_histogram_plot` | isotonic の校正付きで fit した二値分類モデル |
   | `tuning_plot` | `tuning` を残し `plots` を除いた環境で `tune()` を終えたモデル |
   | `importance_plot(kind="shap")` | `explain` を除いた環境（shap が先に検査される）と、`plots` を除いた環境の両方で確かめる |

   この検査は、対応表の過大な記載（要らない extra）を防ぐ。

   **(b) ノートブックの実行。** ノートブックごとの matrix（8）で行う。
   - **環境**: そのノートブックが宣言した extras だけを入れた環境を作る（`uv sync --frozen --no-dev --group notebooks` に、宣言した extra ごとに `--extra <e>`）。実行は `uv run --no-sync --no-dev python -m pytest tests/test_notebooks/test_index_execution.py -m slow -k <名前>` で行う。このテストには `slow` を付ける。`pyproject.toml` の `addopts` が `-m 'not slow'` なので、`-m slow` を明示しないと選ばれない（実装時の調べで判明、2026-10-09）。
     - develop 向けの通常の lane では、このテストは走らない。
     - main 向けの quality lane（`-m ""`）では、既存の slow な実行テストと同じく dev 環境でも走る。その場合は呼び出しの記録だけを確かめ、extras の分離は確かめない。
   - **環境の確認**: pytest を起動する前の別の CI step で、`explain`、`plots`、`tuning` の各パッケージについて、宣言した extra のものは import でき、宣言していないものは import できないことを確かめる。テストの中に置かないのは、dev 環境（main 向けの quality lane）では宣言していない extra も入っているためである。
   - **カーネル**: ipykernel の native kernel を使い、テストプロセスの `sys.executable` で起動する。codegen のノートブックは `python` を PATH から探して起動しているので、`sys.executable` に変える。
   - **呼び出しの記録**: 実行する notebook の内容は、メモリ上でだけ書き換える（ファイルは変えない）。
     - 先頭に記録用のセルを足す。このセルは `Model` の公開メソッドを、継承した mixin のものも含めて包み、呼び出しを記録する。記録の対象は、ほかの `Model` メソッドの中から呼ばれたものを除いた、一番外側の呼び出しだけである。
     - タグ付きセルの各文について、その直前に「受け手 `R` が `Model` のインスタンスであることを確かめ、`(id(R), m)` を待つ」処理を挟む。直後には、「その呼び出しが一番外側で記録されたことを確かめる」処理を挟む。
     - 末尾のセルで、すべてのタグ付きの文がこの確認を通ったことを確かめる。

     これで、宣言した受け手の `Model` で、宣言したメソッドが、その場所で実際に実行されたことを保証する。
   - **extras の不足**: extras が足りなければ実行が失敗するので、対応表の漏れと宣言の不足もここで捕まる。
   - **データ取得の失敗**:
     - 実行は最大 3 回（初回と再試行 2 回）。毎回、新しいカーネルで、notebooks ディレクトリを新しい一時ディレクトリに写したものを作業ディレクトリにして実行する。
     - 再試行するのは、失敗の `CellExecutionError` の文字列に、既存のマーカー一覧（`tests/test_notebooks/test_notebook_execution.py` の `_NETWORK_ERROR_MARKERS`）のどれかが含まれる場合だけとする。その一覧は新しいモジュールに移し、既存のテストはそこから読む。
     - それ以外の失敗は、再試行せずに落とす。3 回とも上記の失敗なら落とし、skip しない（DC1）。

   **起動条件**（決定 1 で改訂、2026-10-10）: (a)(b) は、base や変わったパスに関係なく、ci.yml が走るすべての PR と main への push で走らせる。最後に常に走る gate ジョブを置き、matrix を作るジョブと (a)(b) がすべて成功した場合だけを通す。失敗、skip、cancel はどれも gate を失敗させる。必須にする check はこの gate 1 つとする。変わったパスで実行を絞る仕組みは置かない（それまでの起動判定 `.github/scripts/notebook_index_scope.sh` は削除する）。

7. **PR #335 の静的な検査を置き換える。** これまでの review で見つかった反例をすべて新しい仕組みで再生し、どれも失敗することを確かめる。

   既存の slow な実行テスト（`tests/test_notebooks/test_notebook_execution.py`）は、マーカー一覧の移動以外は範囲外とする。

### 互換性

- 公開 API、Config、結果の型は変わらない。`lizyml/_extras.py` は非公開で、既存のエラーメッセージや guard は変えない。
- ノートブックのメタデータとセルのタグは、表示にも実行にも影響しない。codegen のノートブックが子プロセスを起動するコマンドだけを `sys.executable` に変える。
- 依存グループ `notebooks`（`nbconvert`、`ipykernel`、`pytest`）を新設する。新しいパッケージは追加しない（どれも `dev` に入っている）。`uv.lock` が更新される。
- **Firing rate**: 決定 1 の改訂で、6. の起動条件は無条件になった。`skip` / `select` にあたる条件は無いので、発火率の記録は要らない。改訂前の条件（develop 向け PR では index のパスが変わったときだけ実行する）の発火率は、**11/134 of develop の first-parent commit、2026-04-02 以降**だった（`d72b20a`、2026-10-09 に測定）。

### 代替案（検討して棄却）

1. **`ast` の全ノードを「必ず実行される／されないかもしれない」に分類し、未分類のノードで失敗させる**（案 D）。上記の批評のとおり、実行されるかは位置と、それより前の文の結果で決まるので閉じない。
2. **索引の検査を「ノートブックの一覧の一致」と「import から導く extras」に絞る**（案 B）。8 本ともオプションのパッケージを直接 import していないので、extras はすべて空と導かれる。その結果、今回の不具合（extras の誤り）を捕まえられなくなる。
3. **静的な検査をやめ、実行の記録だけで照合する**（案 C 単独）。正しさは保てるが、普段の PR で走る速い検査が無くなり、索引のずれに気づくのが実行ジョブの起動時まで遅れる（この棄却は改訂前に判断した。当時の起動条件では、develop 向け PR で実行ジョブが起動するのは 11/134 だった。決定 1 で起動は無条件になったが、静的な検査は、実行ジョブより早く、ノートブックを動かさずに、ずれを報告するので残す）。
4. **対応表をテストのデータとして `tests/` に置く。** 対応表はパッケージの振る舞い（どのメソッドがどの extra を要るか）を述べるので、コードの隣で変更されるべきである。生成スクリプトもテストもそこから読む。
5. **`sys.modules` から extras を導く。** `import lizyml` の時点で optuna の import が試みられ、scipy は scikit-learn と lightgbm が必ず読み込むので、過大にも過小にもなる（DC6）。
6. **起動判定を残して直し続ける**（決定 1 の案 S2）。merge base が複数ある場合とサブモジュールの設定の 2 点を塞いでも、git が変更を報告する形と設定は開いたままで、review が次の形を見つける。
7. **起動判定を「ほぼ正しい最適化」と契約に書く**（案 S3）。main 向け PR が最後の守りになるが、develop の貢献者に、変更を検査していない緑の gate を見せることになる。省けるのは runner の時間だけで、待ち時間は省けない。
8. **`docs/examples.md` を markdown-it-py で読む**（案 M1）。手書きの文法よりは閉じるが、GitHub の表示（GFM）との一致は保証できない。また、dev 依存を直接足すことになる。
9. **行の文法を直し続ける**（案 M2）。CommonMark を手書きで真似る限り、review は次の形を見つける（実装 review run 2〜5 で 15 件）。
10. **`docs/examples.md` 全体をメタデータから生成する**（案 M4）。説明文を構造化データに移す手間がかかる。また、説明文を差し込む形にすると、また文法が要る。

### 受け入れ基準（テスト観点）

1. **宣言（2.）の検査**: 各規則について、守った宣言が通るテストと、破った宣言が失敗するテストがある。対象は、キーの過不足、型、重複、並び、空、識別子、存在しないメソッド、未知の extra である。
2. **タグ付きセルの文法（3.）の検査**: 許す各形（2 つの文、「値」の各形、位置引数の `kind`）が通るテストがある。拒否する各形（ほかの文、ほかの式、`*` / `**`、`models` 以外の受け手、`N` が `models` の名前、マジック行、空のセル）が失敗するテストがある。
3. **これまでの反例の再生**: すべてを新しい仕組みで再生し、どれも失敗する。静的な検査か実行の記録のどちらかで落ちればよい。対象は次のとおり。
   - `and` / `or` の後ろの呼び出し
   - 空の内包表記
   - 条件式
   - 到達しない分岐
   - 再代入
   - `def` / `class` の再代入
   - 重複見出し（生成領域の行が重複すると失敗する）
   - マジック行
   - 一覧に無いノートブック（生成領域に行が無いノートブックがあると失敗する）
   - コメントだけの呼び出し
   - `Model` に見せかけた別の受け手
4. **`docs/examples.md` の照合（5.）**: リポジトリの `docs/examples.md` で、生成領域が生成し直した行と一致する。次の操作をしたときは、それぞれ失敗する。
   - どれか 1 本のメタデータを変える
   - ノートブックを 1 本足す、または消す
   - 行を消す、足す、並べ替える、書き換える（題、生成元のコメント、表の見出し、表の区切り、ノートブックの行、空行、`<!-- index:end -->`）
   - `<!-- index:end -->` の行を消す。手書きの部分に別の `<!-- index:end -->` の行がある場合を含む
   - 名前に使えない文字を含むノートブックを置く

   次のことを確かめるテストもある。
   - **並び順**: 生成領域の行の順は、ファイル名のコードポイント順である（大文字で始まる名前が、小文字で始まる名前より前に来る）。
   - **検査の範囲**: 生成領域より後ろに何を書いても、`--check` の結果は変わらない。
   - **行の区切り**: LF、CRLF、CR のどれで区切っても、`--check` は同じ結果になる。
   - **`--write` で直せる場合**: 次の各場合に `--write` が成功し、結果が今の宣言から作った K 行になる。
     - 古いが壊れていない生成領域で、ノートブックの数が変わらない
     - 古いが壊れていない生成領域で、ノートブックを足した、または消した（古い N と今の数が異なる）
   - **`--write` の置き換え**: 古い生成領域が正しければ、後ろの部分は 1 バイトも変わらない。後ろの部分に、`| ` で始まる行、空行、1 つ以上の `<!-- index:end -->` の行がある場合も同じである。区切りが混在するファイルでも、置き換えた K 行は、すべて古い 1 行目の区切りになる。LF、CRLF、CR、混在のどれでも、D は同じ値になる。
   - **ダイジェストの既知の値**（決定 1 の改訂の review run 2 の round 1 で追加、2026-10-10）: 次の 2 行に対する D は、`9fa91b360618931dd5717f9cf60c166f0058cc92734b624f9c49800d3676ab92` である。
     - 1 行目: `` | `a.ipynb` | `fit()` | none (base install) | ``
     - 2 行目: `` | `b.ipynb` | `predict()` | café | ``（`é` は U+00E9）

     区切りや符号化を誤った場合は、これと違う値になる。区切りを付けずにつなぐと `40f73248…`、末尾にも LF を付けると `c5c08be6…`、UTF-8 ではなく Latin-1 で符号化すると `c3acf5de…` である。テストでは、この値と一致することを確かめる。
   - **古い生成領域を拒否する場合**（決定 1 の改訂の review run 2 の round 3 で、各場合の変更と理由を 1 つずつに定めた、2026-10-10）
     - **基準のファイル**: 基準は次の 12 行のファイルである。
       - 1〜10 行目が古い生成領域で、N=2 である。
       - 7、8 行目は、上の既知の値の 2 行である。
       - 3 行目の D は、その既知の値である。
       - 11 行目は空行、12 行目は `## Notes` である。

       この基準を `--write` すると成功する。
     - **拒否の確かめ方**: 次の各場合は、基準に表の変更だけを加える。どの場合も `--write` は失敗し、ファイルは 1 バイトも変わらず、一時ファイルも残らない。テストは、報告された理由が表のとおりであることを確かめる。

     | 基準への変更 | 理由 |
     |---|---|
     | 3 行目を `rows=0` にし、D を空のバイト列の SHA-256 にする | `header-format` |
     | 3 行目を `rows=02` にする | `header-format` |
     | D を同じ値の大文字にする | `header-format` |
     | D の最後の 1 文字を消す（63 文字） | `header-format` |
     | D の 1 文字を `g` にする | `header-format` |
     | 3 行目の `Do not edit` を `Do edit` にする（N と D は変えない） | `header-format` |
     | 5 行目の `Demonstrates` を `Methods` にする | `fixed-lines` |
     | 8 行目を `x` で始まる行にし、D をその 2 行から計算し直す | `row-prefix` |
     | N を 3 にする（9 行目の空行が 3 行目の表の行として数えられる） | `row-prefix` |
     | N を 1 にする（8 行目が空行の位置に来る） | `boundary` |
     | 9 行目の空行を消す | `boundary` |
     | 10 行目の `<!-- index:end -->` を `<!-- end -->` にする | `boundary` |
     | 9、10 行目を消し、その位置に、`c.ipynb` の表の行（7 行目の名前と `fit()` を変えた行）、空行、`<!-- index:end -->` の 3 行を置く（決定 1 の改訂の review round 2 の反例） | `boundary` |
     | 上の反例のファイルで、さらに N を 3 にする（D は変えない） | `digest-mismatch` |
     | D を別の 64 文字の小文字の 16 進数にする | `digest-mismatch` |
     | 7 行目の `fit()` を `fit2()` にする（D は変えない） | `digest-mismatch` |

     - **書式の誤りの理由**: D の書式の誤りは、ダイジェストの値とは独立に、理由 `header-format` で確かめる。正しい D は小文字の 64 文字なので、書式を誤った D は、計算し直した値と一致させられないためである。
     - **生成領域の検査の前に失敗する場合**: 次の場合も、`--write` は失敗し、ファイルを変えず、一時ファイルも残さない。これらの検査は古い生成領域の検査より前に行うので、上の理由は報告しない。
       - 名前に使えない文字を含むノートブックがある
       - 宣言に誤りがある
       - `notebooks/*.ipynb` が 1 本も無い（`--check` も失敗する）
     - **書き込みの途中で失敗する場合**: 一時ファイルへの書き込みの途中、または置き換えの前に失敗させても、ファイルは変わらず、一時ファイルも残らない。
   - **保証の境界**: N と D の両方を、整合するように計算し直したファイルでは、その N+8 行が古い生成領域として置き換えられる。テストでは、これを保証の境界として確かめる。
   - **一時ファイル**: 一時ファイルは対象と同じディレクトリに作られる。`os.replace` が呼ばれる時点で、一時ファイルは閉じていて、中身は書き終わっている。
5. **対応表の検査（6.(a)）**: 対応表の全項目が、その extra を除いた環境で、期待どおりのパッケージ名つきの `OPTIONAL_DEP_MISSING` を出す。前提の検査で別のエラーが出た場合は失敗する。
6. **ノートブックの実行（6.(b)）**: 8 本すべてが、宣言した extras だけの環境で skip なしに実行できる。タグ付きの文はすべて、宣言した受け手の `Model` で実行されたと記録される。次の場合は、それぞれ失敗する。
   - extras を宣言しているノートブックで、宣言から 1 つ減らす
   - 宣言にある呼び出しを、実行されない位置に移す
   - 受け手を `Model` でないものにする

   各ジョブの所要時間を記録する。

   CI の構成（6. の起動条件）は、`.github/workflows/ci.yml` を読むテストで確かめる。
   - matrix を作るジョブ、(a)、(b)、gate の各ジョブに、変わったパスや base で実行を絞る条件（`if:`）が無い。gate だけは `if: always()` を持つ。
   - (a)(b) は matrix を作るジョブに、gate は 3 つのジョブすべてに依存する。
   - ワークフローの起動条件（`on:`）が、PR と main への push を含む。
   - `.github/scripts/notebook_index_scope.sh` が存在しない。

   gate の判定は、3 つのジョブの結果（success、failure、cancelled、skipped）の全組み合わせで確かめる。すべて success の場合だけが通る。
7. **記録用の仕組みの単体テスト**: notebook を実行せずに `Model` を直接使い、次のことを確かめる。
   - 包んだ後も、`Model.load` が classmethod として、クラスからもインスタンスからも同じように束縛されて呼べる。ほかの classmethod / staticmethod / property があれば、それも同じように扱われる。
   - 包んだ後も、`Model` の公開メソッドの名前、signature、docstring が変わらない。
   - 公開メソッドの中から別の公開メソッドが呼ばれた場合（`importance_plot(kind="shap")` が中で `importance` を呼ぶ）、記録はちょうど `[("importance_plot", id)]` の 1 件になる。中の呼び出しは記録されない。
   - 例外で抜けた呼び出しの後も、深さの数え方が崩れない。
   - 記録を外した後は、`Model` が元のメソッドに戻る。
8. **再試行の単体テスト**: 実行部分を差し替えられる形にし、本物のカーネルを使わずに次のことを確かめる。
   - 1 回目で成功すると、試行は 1 回で終わる。
   - マーカーに当たる失敗の次に成功すると、試行は 2 回で、成功として終わる。
   - マーカーに当たる失敗が 3 回続くと、試行は 3 回で失敗として終わり、skip にはならない。
   - マーカーに当たらない失敗は、1 回で失敗として終わる。
   - 試行ごとに、カーネルと作業ディレクトリが別のものになる（識別子が試行の間で異なる）。
9. **review**: 新しい review run を 1 回、Codex APPROVE まで通す。review が確かめるのは、5. と 6. の各条文、受け入れ基準 1〜8 の各項目にテストがあるかと、そのテストが通るかである。条文に無い形（CommonMark や GFM の書き方、git のパスの表し方など）を探すことは求めない。条文に無い形が見つかっても、それは次の round に進む理由にならない。保証を広げたい場合は、別の Proposal とする。各条文にテストがあり、どの major の指摘も、改訂後の条文に反することを示していなければ、run を止める。

### 決定 1: 外部の仕組みとの完全一致をやめ、生成した行と無条件の実行に絞る（実装 review run 1〜5 の後、2026-10-10、管理者の判断）

- **経緯**: 実装 review は 5 つの run で、major の指摘を 29 件出した。指摘のあった場所は次のとおり。
  - 本体（宣言、タグ付きセル、対応表、記録、実行ジョブ、gate）: 5 件で、すべて run 1 の round 1
  - `docs/examples.md` の検査: 15 件
  - develop 向け PR の起動判定: 5 件
  - `--write`: 3 件
  - CHANGELOG: 1 件

  各 run の最後では、受け入れ基準 1〜8 が成り立っていると報告されていた。
- **状況の批評**（run `h0119-situation-critique`、head `4cf5af2`）による原因は次のとおり。
  - 20 件は、外部の仕組みとの完全一致を約束したことから生じた。5. は「CommonMark の表示どおりに読む」、6. は「git がどう報告しても、変わった index のパスを見落とさない」と約束していた。どちらも手書きのコードで真似ていた。
  - review の依頼は、あらゆる形を探すことを求めていた。また、指摘のたびに契約を書き足したので、目標が動いた。
  - 修正そのものが新しい欠陥を生んだ例が、少なくとも 6 件ある。
- **`docs/examples.md` の目的**（同じ批評で確認）: README から「Jupyter notebook index」としてリンクされた、利用者向けの案内である。ノートブックを選び、必要な extras を入れるために使う。ドキュメントサイトは無く、GitHub 上で GFM として表示される。守るべきなのは、メソッドの一覧、extras、ノートブックの漏れの 3 つが正しいことである。手書きの Markdown が、表示どおりに読まれることまでの証明は要らない。
- **決定**:
  - 5. は、ファイル先頭の生成領域の物理的な行が完全に一致することだけを検査する。CommonMark の解釈と、表示の保証をやめる（案 M3 を強化した形）。
  - 6. の起動条件は無条件にし、起動判定を削除する（案 S1）。待ち時間は増えない。index のジョブは 14〜66 秒で、Quality ジョブ（最長 258 秒）と並列に走る。増えるのは runner の時間で、1 PR あたり約 5 分である（`4cf5af2` の CI で観測。保証ではない）。
  - `--write` は、一時ファイルを経由して置き換える。
  - それまでの行の文法、マーカー候補、空白と行の区切りの規則、起動判定に関する、実装時の注記は削除する。それらを確かめていたテストも、条文と一緒に取り除く。
- **互換性**: 公開 API には影響しない。`docs/examples.md` の構成は変わる。各ノートブックの説明の節の前に、生成された一覧の表が来る。

## H-0120: 生成 train.py が LizyML の refit モデルを再現する約束を戻す（#301、#304）

- **ステータス**: Proposed
- **起票日**: 2026-10-10
- **スコープ**: `lizyml/codegen/`（`config_writer.py`、`templates.py`、`artifact_writer.py`、`generator.py`）、`lizyml/core/_model_persistence.py`（export に渡す値）、`BLUEPRINT.md` §6.6 / §15.4、`tests/test_codegen/`（再現の行列テストを新設）、`CHANGELOG.md`、`docs/proposal_dispositions.toml`
- **関連**: [Issue #301](https://github.com/nbx-liz/LizyML/issues/301)、[Issue #304](https://github.com/nbx-liz/LizyML/issues/304)、H-0059（codegen）、H-0073、H-0090（OOF の fold の再現）、H-0103（inner valid）、H-0105（feval）、#269 の決定（refit の重み。HISTORY の #269 の項が、生成 `train.py` が何を再現するかの決定を #301 に先送りしている）

### 目的（課題）

H-0059 は `export_code` の目的の 1 つ目を「新データ到着時に同一設定で refit と calibrator の再構築ができること」とし、受け入れ基準に「同一データ・同一 seed で refit モデルの予測値が `rtol=1e-7` で一致する」を置いた。この約束はその後の変更で崩れ、それを確かめるテストも無かった（`tests/test_codegen/` で、生成 `train.py` で再学習したモデルを LizyML と照合するものは無い。照合しているのは、export した booster を読む `predict.py` だけである）。

実測（2026-10-10、`fabac47`、n=500）では、early stopping を使う 9 ケースはすべて一致しなかった（最大の差 0.011〜0.276。multiclass の `balanced` では 500 行中 211〜256 行でクラスが変わる）。early stopping を使わない場合は一致したが、multiclass の `balanced` だけは一致しなかった。LizyML の refit が実際に使った inner valid の分割と重みを与えて学習し直すと、15 ケースすべてで差は 0.0 になった。この実測は調査用のスクリプトで行い、リポジトリには残していない。受け入れ基準 1 の行列テストが、これを恒久的な検査として置き換える。

管理者の判断（2026-10-10）: 約束を H-0059 のものに戻す。#304（カテゴリの符号）も、同じ約束に必要なので本 Proposal に含める。

### 約束（本 Proposal が定める範囲）

**約束**: `Model.fit(df)` の後に `export_code(path)` で生成したプロジェクトで、`python train.py <data>` を同じデータで実行したとする。このとき、`train.py` が書く `artifacts/model.txt` と `artifacts/pipeline_state.json` による校正前の予測は、どの入力行に対しても、LizyML の refit モデルの校正前の予測と `rtol=1e-7` で一致する。

校正前の予測とは、次のものをいう。
- regression: 予測値
- binary: 陽性クラスの確率（校正前）
- multiclass: 各クラスの確率

**「同じデータ」の定義**: `Model.fit` に渡した DataFrame と同じ行・同じ列・同じ行順のデータを、parquet で保存したもの。生成される `requirements.txt` に、parquet を読むための `pyarrow` を加える（方針 7）。

**型の混じった列**: 1 つの列に型の混じった値（例: 文字列の `"1"` と整数の `1`）を持つ DataFrame は、parquet（pyarrow）でも CSV でも、元の型のまま保存できない。そのような列を持つ fit は、再学習の約束の外とする。ただし export 直後の `predict.py` は型付きの状態（方針 5）を読むので、LizyML と同じ符号を使う（受け入れ基準 1）。

**CSV の場合**（管理者の判断、2026-10-10）: CSV は値の型と `category` dtype を保てない。CSV で保存したデータについては、次の条件を満たす場合に限って約束する。この限定は BLUEPRINT §15.4 に書く。
- 生成 `train.py` が `pd.read_csv` で読み、宣言されたカテゴリ（`config.json` の `declared_categories`、方針 5）を当てた後の、カテゴリ列、時間の列、グループの列の値と dtype が、fit 時のものと等しい。

この条件を満たさない例は、型の混じった列、文字列として読まれる時刻の列である。

**前提**:
- 同じ計算機で実行する。
- 同じ版の LightGBM、numpy、pandas、scikit-learn で実行する。分割、並べ替え、カテゴリの符号は、これらの版によって変わりうる。export 時の 4 つの版を `config.json` に記録し、生成 `train.py` は実行時の版が記録と違えば警告を出す。学習は止めない（管理者の判断、2026-10-10。新しい環境での再学習を妨げないため）。
- LightGBM 自身の決定性に依存する設定（`num_threads`、`force_col_wise` / `force_row_wise`、`deterministic`）は、どちらの側でも LightGBM の既定のままである。この場合、約束は LightGBM 自身の決定性の範囲で成り立つ（LightGBM は既定では col-wise と row-wise を所要時間で選ぶ）。`model.params` で `deterministic: true`、`force_col_wise: true`、`num_threads: 1` を指定すると、これらは `config.json` の `lgbm_params` に書かれるので、両方の側が同じ設定で学習する。受け入れ基準のテストは、この指定で実行する。

**約束しないもの**（明記する）:
- `Model.load()` で読み込んだモデルからの export のうち、`applied_training_params` の無い（H-0109 より前の）artifact のもの。このとき、tune が決めた `ratio` は分からない（H-0109）。patience は、保存された adapter が持つので分かる（`build_export_params` が adapter から読む、BLUEPRINT §14.4）。
- 校正器。H-0059 の約束は「校正器が作り直される」までで、値の一致は約束していない。生成 `train.py` の校正用 OOF は fold の分割だけを再現し（H-0090）、fold のモデルは再現しない。そのため、binary の校正後の確率は一致しない。
- 異なるデータで学習し直した場合。このときは同じ規則で学習するが、比較の対象となる LizyML のモデルは無い。
- 本 Proposal より前の版が生成したプロジェクト。生成されたコードはそのプロジェクトの中で完結しているので、古いプロジェクトは古い動作のままである。

### 対応方針（提案）

約束を崩していた原因は 7 つで、ソースから導出した（「規則が縛る位置」）。それぞれを次のように直す。

1. **早期停止の分割を再現する。**
   - **2 つの値を独立に書く。** LizyML の refit では、inner valid の分割（検証集合があるか）と early stopping の patience（callback があるか）が別々に決まる。callback が付くのは、検証集合があり、かつ patience が `None` でない場合だけである（`estimators/lgbm/adapter.py` の callback の構築）。tune が `validation_ratio` だけを変えた fit では、検証集合はあるが callback は無い。tune が patience だけを決め、config で early stopping を無効にした fit では、adapter は patience を持つが検証集合は無いので、callback は無い（`effective_early_stopping_rounds` は config の `enabled` に関わらず tune の patience を返し、`model.py` がそれを adapter に入れる。inner valid は作られない）。そこで `config.json` には次の 2 つを、どちらも必ず書く。
     - `inner_valid`: refit が実際に使った strategy の設定。refit に検証集合が無かった fit では `null`。
     - `early_stopping_rounds`: refit の adapter が実際に持っていた patience（`ExportParams`）。無かった場合は `null`。

     生成 `train.py` は 2 つとも既定値なしで読む（キーが無ければ失敗する）。そして LizyML と同じ規則で使う。`inner_valid` が `null` でなければ、分割して学習行だけで学習し、検証集合を渡す。callback を付けるのは、それに加えて `early_stopping_rounds` が `null` でない場合だけである。「書かれていない」と「無効」を同じ値にしない（BLUEPRINT §14.4 の `ExportParams.early_stopping_rounds` と同じ理由）。
   - **`inner_valid` の値の出どころ。** refit が使った strategy のオブジェクトは保存されていない（`Model.fit` の中の局所変数で、`RefitResult` にも `FitState` にも無い）。そのため export 時に、fit が strategy を作ったのと**同じ関数**（`_model_factories.py` の自動解決と明示指定の経路）を、fit と同じ入力で呼び直して作る。入力は次のとおりで、どれも保存されている。
     - config から: outer split の method と inner gap（`purge_gap` / `gap`）、明示した `inner_valid`（method、`random_state`、`stratify`）、`training.seed`、タスク。
     - fit が適用した値から: `ratio`。tune が変えうるので、H-0109 の `applied_training_params` から読む（H-0094 決定 13 と同じ理由）。
     - 列名（グループの列、時間の列）: strategy のオブジェクトは列名を持たないので、config のデータの指定から読む。

     書く中身は、こうして作った strategy の種類と、その `ratio`、`random_state`、`stratify`、`gap`、および列名である。例: 明示した `time_holdout` は、outer split が `purge_gap` を持っていても、同じ関数が `gap=0` で作るので、書く値は `0` になる。
   - 生成 `train.py` は、`lizyml/training/inner_valid.py` の 4 つの strategy を移した関数で、同じ分割を作る（H-0090 が outer split で行ったのと同じ方法）。対象は `HoldoutInnerValid`（層化あり／なし）、`GroupHoldoutInnerValid`、`TimeHoldoutInnerValid`（`gap` を含む）、`BlockedGroupInnerValid` の 4 つである。LizyML と同じ numpy と scikit-learn の呼び出しを使い、検証行の数の丸め方（切り上げ／切り捨て）と学習行の並び順も合わせる。
2. **学習前の行の並び順を再現する。** LizyML は、時間順の outer split（`time_series`、`purged_time_series`、`group_time_series`）と `blocked_group_kfold` で、学習の前に行を並べ替える（`data/dataframe_builder.py`）。生成 `train.py` も、学習の前に LizyML と同じ呼び出し（同じ列の `Series.argsort()`、既定の `kind="quicksort"`）で並べ替える。これは安定ソートではないので、同じ値の時刻やブロックの間の順序は、ソートの実装（numpy の版）で決まる。同じ版で同じ入力なら同じ順序になる（前提を参照）。
3. **multiclass の `balanced` の重みを再現する。** export 時に、refit が重みを使ったかどうかと、その規則（`balanced`）を `config.json` に書く。生成 `train.py` は LizyML と同じ式（`compute_sample_weight("balanced", y)` と同じ値）で行ごとの重みを計算し、inner valid の学習行にだけ付ける。検証行には付けない。binary は、これまでどおり `scale_pos_weight` で届く。
4. **評価関数（feval）の一致を確かめる。** 生成 `train.py` の評価関数が、LizyML の評価関数と同じ値を返し、同じ round で早期停止させることを、受け入れ基準 1 の行列で確かめる。現時点で既知のずれは無い。multiclass で使える評価関数（`f1`、`brier`、`accuracy`）はどれも `needs_simplex` を使わない（H-0105 に記録済み。`metrics/classification.py`）。行列でずれが見つかった場合に限って直す。
5. **カテゴリの符号を再現する（#304）。**
   - **生成 `train.py` の `fit_pipeline` は、`CategoricalEncoder.fit` と同じ呼び出しでカテゴリと最頻値を決める**（`features/encoders/categorical_encoder.py`）。
     - `category` dtype の列では、`series.cat.categories`（宣言された順）を使う。
     - それ以外の列では、`sorted(series.dropna().unique().tolist(), key=str)` を使う。
     - 最頻値も同じ規則で決める。カテゴリが 1 つ以上あれば、`series.mode()` が空でなければその先頭、空なら（宣言されたカテゴリはあるが、値がすべて欠損の列など）カテゴリの先頭とする。カテゴリが 1 つも無ければ `None` とする。
     - 値は `str` にせず、値のまま区別する。符号は、カテゴリの並びの中の位置である。
   - **宣言されたカテゴリを `config.json` に書く。** fit 時に `category` dtype だった列について、そのカテゴリの並びを `config.json` の `declared_categories`（列名 → 値の配列）に書く。生成 `train.py` は、読んだデータのその列を、このカテゴリで `category` dtype に直してから `fit_pipeline` に渡す。CSV で dtype が失われても、宣言されたカテゴリは失われない。
   - **`pipeline_state.json` の形**: カテゴリ列ごとに `{"categories": [値, ...], "mode": 値}` と書く。配列の位置が符号である。値は型付きの JSON の値（文字列、数、真偽値）とし、JSON のオブジェクトのキー（常に文字列）には使わない。`config.json` の `declared_categories` も同じ表し方をする。
   - **受け付ける値の型**（閉じた集合）:
     - Python の `str`、`int`、`float`、`bool`。
     - numpy のスカラー（`np.generic`）のうち、`.item()` がこれらのどれかになり、元の値と等しいもの（`np.int64`、`np.float64`、`np.bool_`、`np.str_` など）。書く前に `.item()` で直す。
     - 欠損値（`None`、`NaN`）はカテゴリにならない（`CategoricalEncoder` と同じ）ので、この判定の対象外である。
   - **拒否**: 受け付ける型の外のカテゴリを持つ fit では、`export_code` が `LizyMLError` で拒否し、何も書かない。たとえば `tuple`（JSON の配列にはなるが、受け付けない）、`bytes`、`pd.Timestamp`、`decimal.Decimal` である。黙って `str` にすることはしない。
   - #309（`CategoricalEncoder` 自身が float16 / longdouble で出す生の例外）は、LizyML 側のエラーの契約の問題なので、本 Proposal の範囲外とする。
   - 生成 `predict.py` の `transform` も、同じ符号を使う。
6. **実行の決定性と版**: 約束の前提（同じ計算機、4 つのライブラリの同じ版、決定性の設定）を BLUEPRINT §15.4 に書く。export 時の LightGBM、numpy、pandas、scikit-learn の版を `config.json` の `_versions` に記録する。生成 `train.py` は、実行時の版が記録と違えば、どの版が違うかを警告し、学習は続ける。テストは `model.params` に `deterministic: true`、`force_col_wise: true`、`num_threads: 1` を指定して実行する。
7. **parquet を読めるようにする。** 生成 `train.py` は parquet を `pd.read_parquet` で読むが、生成される `requirements.txt` には parquet の読み込みに要る `pyarrow` が無い。`requirements.txt` に `pyarrow` を加える。

### 規則が縛る位置（ソースから導出、実装前）

規則: **生成 `train.py` は、LizyML の refit が学習に使う入力をすべて同じにする。**

位置の導出: refit の経路（`Model.fit` → `RefitTrainer.fit` → LGBM adapter）が学習に使う入力をソースから列挙し、それぞれについて生成コードの位置を調べた（2026-10-10、`fabac47`）。bound: この経路で `lgb.train` に届く入力（データ、行の並び、特徴量の符号、params、num_boost_round、Dataset の引数、重み、検証集合、callback、評価関数）。

| # | 入力 | LizyML の位置 | 生成コードの位置 | 本 PR |
|---|---|---|---|---|
| 1 | inner valid の分割 | `training/inner_valid.py`、`core/_model_factories.py` の自動解決と明示指定、inner gap | `templates.py` `train_lgbm`（全 method で乱数の holdout） | **修正**（方針 1） |
| 2 | 学習前の行の並び順 | `data/dataframe_builder.py`（時間順と blocked） | `templates.py` `train()`（入力の順のまま） | **修正**（方針 2） |
| 3 | 行ごとの重み | `training/refit_trainer.py`、`estimators/lgbm/smart_params.py` | 無い | **修正**（方針 3） |
| 4 | 検証集合があるか、callback があるか | `_model_factories.py` の inner valid の構築と `effective_early_stopping_rounds`、`estimators/lgbm/adapter.py` の callback の構築 | `config.json` の `validation_ratio` と `early_stopping_rounds`（`ratio > 0 and rounds` で両方を一緒に決める） | **修正**（方針 1 の最初の項） |
| 5 | 評価関数 | `estimators/lgbm/metric_bridge.py` | `templates.py` の評価関数 | **確認**（方針 4。既知のずれは無い） |
| 6 | カテゴリの符号 | `features/encoders/categorical_encoder.py` | `templates.py` `fit_pipeline` / `transform`、`artifact_writer.py` | **修正**（方針 5） |
| 7 | params、smart params、tune の結果、`num_boost_round`、最終 iteration、Dataset の引数、検証集合、`first_metric_only`、ラベルの変換、落とす列 | adapter と provider（`_build_params`）、`target_encoder.py`、`dataframe_builder.py` | refit の adapter から `config.json` へ書かれる | 変更なし（一致していることを行列テストで確かめる） |

### 互換性

- **公開 API、Config、`FitResult`、`PredictionResult`、LizyML の artifact（`format_version`）は変わらない。**
- **`export_code` の出力は変わる。**
  - `config.json` に、`inner_valid`（または `null`）、`early_stopping_rounds`（または `null`）、重みの設定、`declared_categories`、`_versions` が増える。
  - `pipeline_state.json` のカテゴリは、列ごとの `{"categories": [...], "mode": ...}` として型付きで書かれる。
  - 生成されるプロジェクトはその中で完結しているので、既存のプロジェクトはそのまま動く。
- **`export_code` が新しく拒否する場合がある。** 方針 5 の受け付ける型の外のカテゴリを持つ fit である。これまでは `str` にして黙ってずれていた。
- **生成 `train.py` で再学習した結果が変わる。** LizyML と一致するようになる。
- **Firing rate**: export の拒否（方針 5）は `allow` の条件にあたる。その他の分岐（分割の種類、並べ替えの有無、重みの有無、検証集合と callback の有無）は、LizyML の fit が既に下した判断を生成コードに写すだけで、新しい条件ではない。拒否の発火率は次のとおりである。
  - **Firing rate: 0/4559 of `CategoricalEncoder.fit` の呼び出し（うちカテゴリを 1 つ以上持つもの 390）、テストスイート全体（`fabac47`、9526 passed）**。測り方: `CategoricalEncoder.fit` を包む pytest プラグインで、fit ごとに、受け付ける型の外のカテゴリがある列を数えた（2026-10-10）。
  - 発火は 0 件である。この拒否は最適化ではなく、黙ってずれる export を止めるための安全側の拒否なので、発火しないことは欠陥ではない。テストスイートの母集団は、受け付けない型をほとんど含まないと考えられる。そのため、拒否の各分岐は受け入れ基準 2 で直接テストする。

### 代替案（検討して棄却）

1. **重みの規則だけを揃える（約束を絞る）。** 管理者は、H-0059 の約束に戻すことを選んだ（2026-10-10）。
2. **LizyML が使った分割の行番号を `config.json` に保存する。** 同じデータでは一致するが、新しいデータでの再学習という目的に使えない。
3. **生成コードから `lizyml` を import する。** 「LizyML 非依存」という H-0059 の目的に反する。
4. **校正器の一致も約束する。** 校正用 OOF の各 fold のモデル（fold ごとの pipeline、inner valid、重み）まで再現する必要があり、H-0059 の約束を超える。

### 受け入れ基準（テスト観点）

1. **再現の行列**: 次の各ケースで、`Model.fit` → `export_code` → 同じデータで `train.py` → `artifacts/` による校正前の予測が、LizyML の refit モデルの予測と `rtol=1e-7` で一致する。どのテストも `model.params` に `deterministic: true`、`force_col_wise: true`、`num_threads: 1` を指定する。ケースは次の列挙で固定する。
   - **タスクと outer split**: 3 つのタスク（regression、binary、multiclass）と 8 つの method（`kfold`、`stratified_kfold`、`group_kfold`、`stratified_group_kfold`、`time_series`、`purged_time_series`、`group_time_series`、`blocked_group_kfold`）の 24 の組み合わせのすべて。early stopping を有効にした既定の設定で行う。`Model.fit` がその組み合わせを拒否する場合は、テストは拒否されること（`LizyMLError`）を確かめる。これで、24 の組み合わせのそれぞれが「再現する」か「fit が拒否する」のどちらかに入る。
   - **early stopping を無効にした設定**: 3 つのタスクのそれぞれ。
   - **multiclass の `balanced`**: `true`、`null`（既定）、`false` の 3 つ。
   - **明示した `inner_valid`**: method、`ratio`、`random_state`、`stratify` のそれぞれを既定から変えたもの。
   - **inner gap**: `purged_time_series` で `purge_gap` を 0 以外にした、自動解決の fit（gap が inner valid に渡る）。同じ outer 設定に `time_holdout` を明示した fit（gap は 0）。
   - **検証集合と callback の組み合わせ**: 両方ある（既定）、両方ない（early stopping 無効）、検証集合だけある（tune が `validation_ratio` だけを変えた fit）、patience だけある（tune が patience を決め、config で early stopping を無効にした fit）の 4 つ。
   - **同じ値の時刻**: `time_series` で、時間の列に同じ値が複数ある fit。
   - **評価関数**: 生成コードが再実装する 9 つ（`rmsle`、`r2`、`f1`、`brier`、`ece`、`precision_at_k`、`accuracy`、`smape`、`wape`）と 3 つのタスクの組み合わせのすべて。LizyML がそのタスクでその評価関数を受け付けない場合は、拒否されることを確かめる。
   - **カテゴリ**（parquet で行う）: 文字列、整数、宣言だけされたカテゴリ（`category` dtype）、宣言されたカテゴリを持ち値がすべて欠損の列（最頻値がカテゴリの先頭になる）、欠損値を含む列。
   - **型の混じった列**: `"1"` と `1` が混じった object 列を持つ fit で、export 直後の `predict.py`（再学習の前）の予測が `Model.predict` と一致する。この列は保存できないので、再学習の行列には入れない（約束の外）。
   - **CSV**: 文字列のカテゴリ列、宣言されたカテゴリを持つ列、数値の時間の列を持つ fit を、CSV で保存して `train.py` に渡す（`declared_categories` を当てた後に約束の CSV の条件を満たすケース）。
   - **生成プロジェクトの依存**: 生成される `requirements.txt` が `pyarrow` を含む。
2. **拒否**: 受け付ける型の判定の各分岐をテストする。
   - 受け付ける: `str`、`int`、`float`、`bool`、`np.int64`、`np.float64`、`np.bool_`、`np.str_`。
   - 拒否する（numpy 以外）: `tuple`、`bytes`、`pd.Timestamp`、`decimal.Decimal`。
   - 拒否する（numpy のスカラーで、`.item()` が受け付ける型にならないもの）: `np.bytes_`、`np.datetime64`、`np.complex128`。
   - 拒否されたどの場合も、`export_code` が `LizyMLError` を送出し、出力先に何も書かない。
3. **版の警告**: `config.json` の `_versions` を実行環境と違う値にしたとき、生成 `train.py` が違う版を挙げて警告し、学習は最後まで行う。
4. **負の対照**: 方針 1、2、3、5 の修正と、`declared_categories` の復元を 1 つずつ元に戻すと、行列のどれかのケースが失敗する。方針 4 は確認だけで修正を伴わないので、負の対照の対象にしない。
5. **既存の照合**: `test_equivalence.py`（export した booster を `predict.py` が読んだ予測の一致）と H-0090 の fold の再現は、引き続き通る。
6. **文書**: BLUEPRINT §6.6 / §15.4 に、約束、前提（計算機、4 つの版、決定性の設定）、CSV の条件、約束しないものを書く。
7. **review**: Codex の review run を APPROVE まで通す。review が確かめるのは、上の約束、規則が縛る位置、受け入れ基準 1〜6 の各項目にテストがあり、そのテストが通り、違反すれば失敗するかである。約束の範囲の外にある形を探すことは求めない。
