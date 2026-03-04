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
- Status: `proposed`
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
- Status: `proposed`
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
- Status: `proposed`
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

- `format_version=1` から `format_version=2` への移行が必要になった場合、`lizyml/persistence/migrations/v1_to_v2.py` を追加し、ロード時に自動マイグレーションを試みる（または明示的エラーで移行を促す）。

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
- Status: `proposed`
- Scope: `Public API | Plots`
- Related: `BLUEPRINT.md §4.1, §13.3`

### Context

H-0006 で `residuals_plot()` を実装したが、以下の不足がある。

1. 残差散布図（predicted vs residual）が未実装。
2. 常に 2 パネル（histogram + QQ）が表示され、個別選択できない。
3. In-Sample（IF）と Out-of-Sample（OOF）の傾向比較ができない。

### Proposal

- `residuals_plot(kind=...)` に `kind` 引数を追加する。
  - `"scatter"`: 残差散布図（x=predicted, y=residual）。IS と OOS を色分けオーバーレイ。y=0 の参照線。
  - `"histogram"`: 残差ヒストグラム。IS と OOS を色分けオーバーレイ。mean/std アノテーション（OOS のみ）。
  - `"qq"`: QQ plot（OOS 残差のみ）。45 度参照線。
  - `"all"`: 上記 3 つを横並びサブプロットで表示（デフォルト）。
- 内部関数 `plot_residuals()` のシグネチャを変更し、`FitResult` + `y_true` を受け取る形式に統一する（他の plot 関数と同じパターン）。
- IS データは `fit_result.if_pred_per_fold[i]` + `fit_result.splits.outer[i][0]`（train_idx）から組み立てる。
- `kind` の値が不正な場合は `LizyMLError(INVALID_CONFIG)` を返す。

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

- `model.residuals_plot(kind="scatter")` が Plotly Figure を返し、IS/OOS 両方のトレースを含む。
- `model.residuals_plot(kind="histogram")` が IS/OOS オーバーレイのヒストグラムを返す。
- `model.residuals_plot(kind="qq")` が QQ plot を返す。
- `model.residuals_plot(kind="all")` が 3 サブプロットの Figure を返す。
- `model.residuals_plot()` がデフォルトで `kind="all"` として動作する。
- 不正な kind 値で `INVALID_CONFIG` エラーが返る。

---

## 2026-03-04: EarlyStoppingConfig に validation_ratio エイリアス追加

- ID: `H-0010`
- Status: `proposed`
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
- Status: `proposed`
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
