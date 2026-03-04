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
- Status: `proposed`
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
- Status: `proposed`
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
- Status: `proposed`
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
- Status: `proposed`
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
- Status: `proposed`
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
- Status: `proposed`
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
- Status: `proposed`
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
- Status: `proposed`
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
- Status: `proposed`
- Scope: `Config | EstimatorAdapter`
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
- Status: `proposed`
- Scope: `Config | EstimatorAdapter`
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
- Status: `proposed`
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

---

## 2026-03-05: デフォルト Tuning Space の導入（タスク別デフォルト探索空間 + Tuner 拡張）

- ID: `H-0024`
- Status: `proposed`
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
