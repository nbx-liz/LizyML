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

- `model.residuals_plot(kind="scatter")` が Actual vs Predicted の Plotly Figure を返し、IS/OOS 両方のトレースと y=x 参照線を含む。
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

---

## 2026-03-04: residuals_plot() の IS/OOS サンプル数バランシング

- ID: `H-0012`
- Status: `proposed`
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
2. `output_dir` 指定時、`run_id` ベースのサブディレクトリ（`{output_dir}/{run_id}/`）を自動作成し、ログ・plot 保存先とする。
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
- Status: `proposed`
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
   - **カバー行に NaN がある場合は `ValueError`**（予測パイプラインのバグとして検知）。
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
- カバー行に NaN → `ValueError`（バグ検知テスト）。
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
| INV-4 | `MetricRegistry.get("smape", "regression")` と `("wape", "regression")` が成功、binary / multiclass では `LizyMLError(METRIC_NOT_FOUND)` |
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
