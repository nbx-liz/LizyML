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
- Status: `proposed`
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
