# LizyML 開発計画

## 概要

BLUEPRINT.md に基づき、Config駆動のML分析ライブラリ LizyML をゼロから構築する。
本計画は依存関係の順序を尊重し、各フェーズで動作確認・テストを行いながら段階的に進める。

---

## フェーズ一覧

| Phase | 名称 | 主な成果物 | 依存先 |
|-------|------|-----------|--------|
| 0 | 開発環境・プロジェクト骨格 | pyproject.toml (build-system/metadata/optional-deps), py.typed, ディレクトリ構成, uv.lock | なし |
| 1 | 基盤レイヤー | exceptions, logging, seed, import_optional | Phase 0 |
| 2 | Config & Specs | pydantic schema, loader, Spec群 | Phase 1 |
| 3 | データレイヤー | DataSource, DataFrameBuilder, validators, fingerprint | Phase 1 |
| 4 | 型契約（Result/Artifacts） | FitResult, PredictionResult, Artifacts | Phase 1 |
| 5 | Splitters | Splitter IF, KFold, GroupKFold, TimeSeries系 | Phase 1 |
| 6 | Feature Pipeline | pipeline_base, native pipeline, encoders, transformers | Phase 1, 3 |
| 7 | Metrics | Metric IF, regression/classification metrics, registry | Phase 1, 4 |
| 8 | EstimatorAdapter | EstimatorAdapter IF, LightGBM adapter | Phase 1, 7 |
| 9 | Training Core（CV/InnerValid） | CVTrainer, RefitTrainer, InnerValidStrategy, OOF生成 | Phase 4-8 |
| 10 | Evaluation | Evaluator, thresholding | Phase 4, 7, 9 |
| 11 | Model Facade（fit/evaluate/predict） | core/model.py | Phase 2-10 |
| 12 | Tuning | SearchSpace, OptunaTuner, TuningTrainer | Phase 9, 11 |
| 13 | Calibration | Platt, Isotonic, Beta calibrator | Phase 9, 10 |
| 14 | Persistence & Export | serializer, model_store, Model.load() | Phase 4, 11 |
| 15 | Explain（SHAP等） | Explainer IF, SHAP, lgbm_contrib | Phase 8, 11 |
| 16 | Plots | learning_curve, importance, residuals, calibration, classification | Phase 4, 10 |
| 17 | E2Eテスト・統合テスト | 全パイプラインの結合テスト | Phase 11-16 |
| 18 | CI・PyPI 配布検証 | GitHub Actions ワークフロー, sdist/wheel build, twine check, install smoke test | Phase 17 |

---

## Phase 0: 開発環境・プロジェクト骨格

**SKILL:** skills/dev-environment/SKILL.md

### 0-1. pyproject.toml 作成

BLUEPRINT §15.4 に準拠し、PyPI 配布に必要な要素を初期から含める。

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "lizyml"
version = "0.1.0"
description = "Config-driven ML analysis library for regression and classification"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [{ name = "LizyML Authors" }]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "pydantic>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "lightgbm>=4.0",
    "pyyaml>=6.0",
    "joblib>=1.3",
]

[project.urls]
Homepage = "https://github.com/nbx-liz/LizyML"
Repository = "https://github.com/nbx-liz/LizyML"

[project.optional-dependencies]
tuning = ["optuna>=3.0"]
explain = ["shap>=0.44"]

[dependency-groups]
dev = [
    "ruff>=0.8",
    "mypy>=1.10",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pandas-stubs>=2.0",
    "build>=1.0",
    "twine>=6.0",
]
```

**PyPI 配布要件（BLUEPRINT §15.4 対応）:**
- `[build-system]` を PEP 517/518 準拠で定義（hatchling）
- `[project]` に name/version/description/readme/license/authors/classifiers/urls を必須で記載
- optional dependency を `[project.optional-dependencies]`（配布利用者向け）と `[dependency-groups]`（開発者向け）に分離
- バージョン定義の正は `pyproject.toml` の `version` フィールド 1 箇所。`lizyml/__init__.py` の `__version__` は参照用

### 0-1b. py.typed マーカー

`lizyml/py.typed` を空ファイルとして作成し、型ヒントを配布対象に含める。

### 0-2. ツール設定（pyproject.toml 内）

- **Ruff:** line-length=88, select=["E","F","I","W","UP","B","SIM"], target-version="py310"
- **mypy:** strict=true, warn_return_any/warn_unused_configs/disallow_untyped_defs
- **pytest:** testpaths=["tests"]

### 0-3. ディレクトリ構成の作成

BLUEPRINT 19章に準拠したディレクトリと `__init__.py` を作成する。

```
LizyML/
  pyproject.toml
  BLUEPRINT.md / HISTORY.md / CLAUDE.md / README.md
  lizyml/
    __init__.py
    config/
    core/
      types/
      specs/
    data/
    features/
      encoders/
      transformers/
    splitters/
    estimators/
    training/
    tuning/
    metrics/
    evaluation/
    calibration/
    explain/
    plots/
    persistence/
    utils/
  tests/
    conftest.py
    test_config/
    test_core/
    test_data/
    test_features/
    test_splitters/
    test_estimators/
    test_training/
    test_tuning/
    test_metrics/
    test_evaluation/
    test_calibration/
    test_explain/
    test_plots/
    test_persistence/
    test_e2e/
```

### 0-4. uv 環境構築

```bash
uv sync
uv run ruff check .
uv run mypy lizyml/
uv run pytest
```

### 0-5. develop ブランチ作成

- `main` から `develop` ブランチを切る。
- 以降の開発は全て `develop` ベースのフィーチャーブランチで行う。

### DoD
- `uv sync` が通る
- `uv run ruff check .` がクリーン
- `uv run mypy lizyml/` がクリーン
- `uv run pytest` が（空だが）成功する
- `uv build` で sdist / wheel が生成できる
- `lizyml/py.typed` が存在する

---

## Phase 1: 基盤レイヤー

**SKILL:** skills/exceptions-and-logging/SKILL.md, skills/optional-dependencies/SKILL.md

### 1-1. core/exceptions.py

```python
class LizyMLError(Exception):
    """統一例外。code / user_message / debug_message / cause / context を持つ。"""

class ErrorCode(str, Enum):
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_VERSION_UNSUPPORTED = "CONFIG_VERSION_UNSUPPORTED"
    DATA_SCHEMA_INVALID = "DATA_SCHEMA_INVALID"
    DATA_FINGERPRINT_MISMATCH = "DATA_FINGERPRINT_MISMATCH"
    LEAKAGE_SUSPECTED = "LEAKAGE_SUSPECTED"
    LEAKAGE_CONFIRMED = "LEAKAGE_CONFIRMED"
    OPTIONAL_DEP_MISSING = "OPTIONAL_DEP_MISSING"
    MODEL_NOT_FIT = "MODEL_NOT_FIT"
    INCOMPATIBLE_COLUMNS = "INCOMPATIBLE_COLUMNS"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    UNSUPPORTED_METRIC = "UNSUPPORTED_METRIC"
    METRIC_REQUIRES_PROBA = "METRIC_REQUIRES_PROBA"
    TUNING_FAILED = "TUNING_FAILED"
    CALIBRATION_NOT_SUPPORTED = "CALIBRATION_NOT_SUPPORTED"
    SERIALIZATION_FAILED = "SERIALIZATION_FAILED"
    DESERIALIZATION_FAILED = "DESERIALIZATION_FAILED"
```

- context dict（fold, config_path, column, estimator等）を必須で受け取る設計。
- `__str__` でユーザー向けメッセージを返し、`__repr__` でデバッグ情報を含める。

### 1-2. core/logging.py

- `run_id` 生成（UUID or timestamp-based）。
- 構造化ログ（`logging` + JSON formatter）。
- `print` 禁止。PII/生値を出さない。
- fold / config hash / data fingerprint / split hash をログに含められるようにする。

### 1-3. core/seed.py

- seed 管理ユーティリティ。
- `set_global_seed(seed)`: numpy / random / (optional) torch の seed を統一設定。
- `derive_seed(base_seed, fold_index)`: fold ごとに決定論的な seed を生成。

### 1-4. utils/import_optional.py

- `import_optional(module_name, package_name, install_hint)` 関数。
- 未インストール時に `OPTIONAL_DEP_MISSING` の `LizyMLError` を raise。
- user_message にインストール手順を含める。

### 1-5. utils/array.py, utils/pandas.py

- 配列操作・DataFrame操作の薄いヘルパー。
- 必要になった時点で中身を追加する（最初は空でもよい）。

### テスト
- `LizyMLError` の code / context / メッセージ検証
- `import_optional` の未インストール時の例外コード検証
- seed 再現性テスト

### DoD
- 例外を投げて code / context が正しいことをテストで検証
- `import_optional` テスト通過
- Lint / Type check クリーン

---

## Phase 2: Config & Specs

**SKILL:** skills/config-and-specs/SKILL.md

### 2-1. config/schema.py

pydantic v2 を使用し、`model_config = ConfigDict(extra="forbid")` で未知キーを拒否。

主要モデル:
- `LizyMLConfig`（トップレベル）
  - `config_version: int`（必須）
  - `task: Literal["regression", "binary", "multiclass"]`
  - `data: DataConfig`
  - `features: FeaturesConfig`
  - `split: SplitConfig`
  - `model: ModelConfig`（discriminated union で lgbm / sklearn 等を切り替え）
  - `training: TrainingConfig`
  - `tuning: Optional[TuningConfig]`
  - `evaluation: EvaluationConfig`
  - `calibration: Optional[CalibrationConfig]`

各 sub-config も `extra="forbid"` を適用。

### 2-2. config/loader.py

- `load_config(source: dict | str | Path) -> LizyMLConfig`
  - dict: そのまま validate
  - str/Path: `.json` or `.yaml`/`.yml` を判定して読込 → validate
- 環境変数 override: `LIZYML__` prefix、`__` でネスト区切り
- 正規化: alias 吸収（例: `k-fold` → `kfold`）、deprecated key の警告

### 2-3. core/specs/

Config → Spec の変換レイヤー。正規化済み Config から生成し、下流に渡す唯一の入力。

- `problem_spec.py`: `ProblemSpec(task, target, features, ...)`
- `feature_spec.py`: `FeatureSpec(feature_names, categorical, exclude, auto_categorical, ...)`
- `split_spec.py`: `SplitSpec(method, n_splits, random_state, group_col, time_col, ...)`
- `training_spec.py`: `TrainingSpec(early_stopping, inner_valid, ...)`
- `tuning_spec.py`: `TuningSpec(space, n_trials, direction, ...)`
- `calibration_spec.py`: `CalibrationSpec(method, ...)`
- `export_spec.py`: `ExportSpec(format, ...)`

### 2-4. core/registries.py

- `EstimatorRegistry`: estimator 名 → adapter class のマッピング
- `SplitterRegistry`: split method 名 → splitter class
- `MetricRegistry`: metric 名 → metric class
- `CalibratorRegistry`: calibration method 名 → calibrator class
- Registry は dict ベースの薄い実装。decorator パターンで登録可能にする。

### テスト
- 正常 Config の validate 成功
- 未知キー混入時のエラー（typo 検知）
- 必須キー欠落時のエラー
- YAML/JSON/dict 各形式のロード
- 環境変数 override
- 各 Spec への変換

### DoD
- pydantic schema 定義完了
- loader が dict / JSON / YAML を正しく読める
- 不正 Config に対して CONFIG_INVALID を返す
- Spec 変換が網羅的にテストされている

---

## Phase 3: データレイヤー

### 3-1. data/datasource.py

- `DataSource` クラス: CSV / Parquet / DataFrame を読み込む。
- `read(source: str | Path | pd.DataFrame) -> pd.DataFrame`
- 読むだけに限定（加工しない）。

### 3-2. data/dataframe_builder.py

- `DataFrameBuilder`: target / time / group を分離し、特徴量 DataFrame を構成。
- `FeatureSpec` に基づいて exclude / categorical 指定を反映。
- auto_categorical: 非数値列を自動検出。

### 3-3. data/validators.py

- `TimeSeriesValidator`: ソート済みか、未来情報混入疑い、shuffle禁止の検証
- `GroupValidator`: group 跨ぎ、分割条件の不整合
- `LeakageValidator`: target と完全一致の列、時間逆転検知
- 各バリデータは `LEAKAGE_SUSPECTED` / `DATA_SCHEMA_INVALID` を raise。

### 3-4. data/fingerprint.py

- `DataFingerprint`: `row_count`, `column_hash`（列名+dtype+順序）, optional `file_hash`
- `compute(df: pd.DataFrame, file_path: Optional[Path]) -> DataFingerprint`
- Artifacts に保存して再現性を担保。

### テスト
- CSV / Parquet / DataFrame の読み込み
- DataFrameBuilder の target 分離・exclude・auto_categorical
- Validator が違反を検知する例（「落ちるべき例」）
- fingerprint の再現性（同一データで同一ハッシュ）

### DoD
- DataSource → DataFrameBuilder → Validator → fingerprint の一連のパイプラインがテスト通過

---

## Phase 4: 型契約（Result/Artifacts）

**SKILL:** skills/evaluation-contracts/SKILL.md

### 4-1. core/types/fit_result.py

```python
@dataclass
class FitResult:
    oof_pred: np.ndarray
    if_pred_per_fold: list[np.ndarray]
    metrics: dict  # {"raw": {"oof": {...}, "if_mean": {...}, "if_per_fold": [...]}, "calibrated": {...}}
    models: list[Any]  # fold ごとのモデル
    history: list[dict]  # fold ごとの eval history / best_iteration
    feature_names: list[str]
    dtypes: dict[str, str]
    categorical_features: list[str]
    splits: SplitIndices
    data_fingerprint: DataFingerprint
    pipeline_state: Any
    calibrator: Optional[Any]
    run_meta: RunMeta
```

### 4-2. core/types/predict_result.py

```python
@dataclass
class PredictionResult:
    pred: np.ndarray
    proba: Optional[np.ndarray]  # binary のみ
    shap_values: Optional[np.ndarray]  # 要求時のみ
    used_features: list[str]
    warnings: list[str]
```

### 4-3. core/types/artifacts.py

- `SplitIndices`: outer CV / inner valid / calibration CV の全 indices
- `RunMeta`: lizyml_version / python_version / deps_version / config_normalized / config_version
- `FitArtifacts`: FitResult + 追加メタ情報

### テスト（ゴールデンテスト）
- FitResult / PredictionResult のキー・階層・shape・dtype を固定するスナップショットテスト
- 契約に違反する構造を検知するテスト

### DoD
- 型定義完了、ゴールデンテスト通過
- shape / 階層の破壊を検知できる

---

## Phase 5: Splitters

**SKILL:** skills/splitters-and-splitplan/SKILL.md

### 5-1. splitters/base.py

```python
class BaseSplitter(ABC):
    @abstractmethod
    def split(self, n_samples: int, y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(train_indices, valid_indices) を yield する。"""
```

- DataFrame を受け取らない（index のみ）。
- shuffle / 副作用禁止。

### 5-2. 具体 Splitter

- `splitters/kfold.py`: KFold / StratifiedKFold
- `splitters/group_kfold.py`: GroupKFold / StratifiedGroupKFold
- `splitters/time_series.py`: TimeSeriesSplit
- `splitters/purged_time_series.py`: PurgedTimeSeriesSplit（purge window 付き）
- `splitters/group_time_series.py`: GroupTimeSeriesSplit

全て `BaseSplitter` を継承し、`SplitterRegistry` に登録。

### 5-3. SplitPlan

- outer / inner_valid / calibration の分割を統合管理。
- `SplitPlan.create(split_spec, training_spec, calibration_spec)` で生成。
- indices を Artifacts に保存（再現性）。

### テスト
- 各 Splitter が正しい index を返す
- seed 固定で再現性テスト
- time / group 制約の違反検知（「落ちるべき例」）
- SplitPlan が outer / inner / calibration を統合管理

### DoD
- 全 Splitter がテスト通過
- SplitPlan の indices 保存・再現性が検証されている

---

## Phase 6: Feature Pipeline

**SKILL:** skills/feature-pipeline-and-leakage/SKILL.md

### 6-1. features/pipeline_base.py

```python
class BaseFeaturePipeline(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseFeaturePipeline": ...
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame: ...
    @abstractmethod
    def get_state(self) -> dict: ...
    @abstractmethod
    def load_state(self, state: dict) -> "BaseFeaturePipeline": ...
```

### 6-2. features/pipelines_native.py

- LightGBM 向けの native pipeline。
- auto_categorical 検出、categorical dtype 強制。
- 列ズレ方針: 余剰列→警告/無視、不足列→エラー、unseen category→ポリシー選択。

### 6-3. features/encoders/categorical_encoder.py

- カテゴリ辞書の保持・unseen category の処理。

### 6-4. features/transformers/

- `target_transformer.py`: 目的変数変換（log等）の状態管理。
- `feature_transformer.py`: 特徴量変換（正規化等）。

### テスト
- fit → transform の状態保持（カテゴリ辞書）
- 列ズレ検知テスト（余剰/不足/unseen）
- pipeline state の save / load
- **リークテスト**: validation データで fit していないことの検証

### DoD
- native pipeline が fit/transform/state管理を正しく行う
- 列ズレポリシーがテストで検証されている
- リーク防止のテスト通過

---

## Phase 7: Metrics

**SKILL:** skills/metrics/SKILL.md

### 7-1. metrics/base.py

```python
class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def needs_proba(self) -> bool: ...
    @property
    @abstractmethod
    def greater_is_better(self) -> bool: ...
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
```

### 7-2. metrics/regression.py

- RMSE, MAE, R2, RMSLE 等

### 7-3. metrics/classification.py

- LogLoss, AUC (ROC/PR), F1, Accuracy, Brier Score, ECE 等

### 7-4. metrics/registry.py

- `MetricRegistry` に全メトリクスを登録。
- 名前から検索、task との互換性チェック。

### テスト
- 各メトリクスの数値検証（既知の入出力ペア）
- `needs_proba` / `greater_is_better` の正しさ
- shape 不一致時の例外
- Registry からの名前引き

### DoD
- 回帰・分類の主要メトリクスが実装・テスト済み
- Registry 経由でメトリクスを取得できる

---

## Phase 8: EstimatorAdapter

**SKILL:** skills/add-estimator-adapter/SKILL.md

### 8-1. estimators/base.py

```python
class BaseEstimatorAdapter(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs) -> "BaseEstimatorAdapter": ...
    @abstractmethod
    def predict(self, X) -> np.ndarray: ...
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...
    @abstractmethod
    def importance(self, kind: str = "split") -> dict[str, float]: ...
    @abstractmethod
    def get_native_model(self) -> Any: ...
```

### 8-2. estimators/lgbm.py

- LightGBM の `LGBMRegressor` / `LGBMClassifier` をラップ。
- objective / metric の自動設定（task に基づく）。
- categorical 処理の統一。
- early stopping callback の吸収。
- importance: split / gain / SHAP 対応。

### テスト
- 小規模データで fit → predict の E2E
- regression / binary / multiclass 各タスクの動作
- early stopping 時の best_iteration 取得
- importance の取得
- 再現性テスト（seed 固定）

### DoD
- LightGBM adapter が 3 タスクで動作
- テスト通過、Lint/Type check クリーン

---

## Phase 9: Training Core（CV / InnerValid / OOF）

**SKILL:** skills/training-cv-and-inner-valid/SKILL.md

### 9-1. training/inner_valid.py

```python
class BaseInnerValidStrategy(ABC):
    @abstractmethod
    def split(self, X_train, y_train, groups=None) -> tuple[
        tuple[np.ndarray, np.ndarray],  # (inner_train_idx, inner_valid_idx)
    ]: ...
```

- `HoldoutInnerValid(ratio, stratify, group, time, random_state)`
- `InnerKFoldValid(n_splits, pick_fold, ...)`
- time / group 制約を内側でも厳守。

### 9-2. training/cv_trainer.py

- `CVTrainer`: 外側 CV loop を実行。
  1. fold ごとに train/valid を分割。
  2. InnerValidStrategy で early stopping 用分割を生成。
  3. FeaturePipeline.fit は train データのみで行う（リーク防止）。
  4. EstimatorAdapter.fit を実行。
  5. OOF / IF を生成（evaluation/oof.py に委譲）。
  6. history（eval_history, best_iteration）を保存。
  7. FitResult を構成して返す。

### 9-3. training/refit_trainer.py

- `RefitTrainer`: 全データで再学習。
- best_params を使用。

### 9-4. evaluation/oof.py

- OOF 予測の組み立てロジック。
- **同一行リーク禁止**: そのfold の valid 部分のみ予測を埋める。

### テスト
- CV loop の正常動作（小規模データ）
- OOF の shape・行対応が正しいこと
- **OOF リークテスト**: train に含まれた行が OOF 予測に混入しないことを検証
- InnerValid の time/group 制約テスト
- history の保存
- 再現性テスト（seed 固定）

### DoD
- CVTrainer が FitResult を正しく返す
- OOF リーク防止テスト通過
- InnerValid のテスト通過

---

## Phase 10: Evaluation

**SKILL:** skills/evaluation-contracts/SKILL.md

### 10-1. evaluation/evaluator.py

- `Evaluator`: FitResult + メトリクスリストを受け取り、評価結果を返す。
- 出力構造（固定）:
  ```python
  {
      "raw": {
          "oof": {"rmse": 0.123, "mae": 0.098},
          "if_mean": {"rmse": 0.120, "mae": 0.095},
          "if_per_fold": [{"rmse": ..., "mae": ...}, ...]
      },
      "calibrated": {  # binary のみ、calibrator 有効時
          "oof": {...},
          "if_mean": {...},
          "if_per_fold": [...]
      }
  }
  ```
- Evaluator 外でのメトリクス計算・フォーマッティングは禁止。

### 10-2. evaluation/thresholding.py（任意）

- binary 分類の閾値最適化。
- F1 最大化等のポリシーを提供。

### テスト（ゴールデンテスト）
- 評価出力の構造（キー・階層）が固定されていることを検証
- raw / calibrated の並列構造
- per_fold の数が n_splits と一致

### DoD
- Evaluator が FitResult から正しい構造の評価結果を返す
- ゴールデンテスト通過

---

## Phase 11: Model Facade（fit / evaluate / predict）

### 11-1. core/model.py

```python
class Model:
    def __init__(self, config: dict | str | Path): ...
    def fit(self, params: Optional[dict] = None) -> FitResult: ...
    def evaluate(self, metrics: Optional[list[str]] = None) -> dict: ...
    def predict(self, X, return_shap: bool = False) -> PredictionResult: ...
    def importance(self, kind: str = "split") -> dict: ...
    def tune(self) -> TuningResult: ...
    def export(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> "Model": ...
```

**Model の責務（組み立てのみ）:**
1. Config validate → ProblemSpec 生成
2. DataSource → DataFrame 読み込み
3. Registry 経由で各コンポーネントを選択
4. Trainer / Evaluator / Persistence に委譲
5. FitResult / Artifacts を保持

**Model に置かないこと:**
- OOF/IF 生成ロジック → evaluation/oof.py
- metric 計算 → evaluation/evaluator.py
- LGBM 固有処理 → estimators/lgbm.py
- plot 実装本体 → plots/*
- 保存形式の詳細 → persistence/*

### テスト（E2E 小規模）
- Config → fit → evaluate → predict の一貫パイプライン
- regression / binary の基本フロー
- predict 時の列ズレ検知
- 再現性テスト

### DoD
- Model が Config から fit / evaluate / predict を実行できる
- 責務分離が守られている（Model にロジックがない）

---

## Phase 12: Tuning

**SKILL:** skills/tuning/SKILL.md

### 12-1. tuning/search_space.py

- Optuna 非依存の SearchSpace 表現。
- 離散・連続・対数・カテゴリを統一。

### 12-2. tuning/base.py

```python
class BaseTuner(ABC):
    @abstractmethod
    def optimize(self, objective_fn, space, n_trials, ...) -> TuningResult: ...
```

### 12-3. tuning/optuna_tuner.py

- Optuna ベースの Tuner 実装。
- SearchSpace → Optuna trial への変換。
- objective = CV aggregated score。
- TuningResult: best_params, best_score, study summary。

### 12-4. training/tuning_trainer.py

- Tuning と CV の統合。
- **リーク回避**: tuning に使った CV で最終性能を主張しない。
- デフォルトは holdout or CV + テストセット方式。

### テスト
- SearchSpace の正しい変換
- 小規模データで tune → fit の一貫フロー
- best_params が fit に正しく渡される
- リーク回避方針の検証

### DoD
- tune → fit パイプラインが動作
- TuningResult の契約テスト通過

---

## Phase 13: Calibration

**SKILL:** skills/calibration/SKILL.md

### 13-1. calibration/base.py

```python
class BaseCalibrator(ABC):
    @abstractmethod
    def fit(self, s_oof: np.ndarray, y: np.ndarray) -> "BaseCalibrator": ...
    @abstractmethod
    def predict(self, s: np.ndarray) -> np.ndarray: ...
```

- **入力は s_oof（1D）と y のみ。X は受け取らない。**

### 13-2. 具体 Calibrator

- `calibration/platt.py`: Platt Scaling（LogisticRegression）
- `calibration/isotonic.py`: Isotonic Regression
- `calibration/beta_calibration.py`: Beta Calibration

### 13-3. Calibration の統合

- CVTrainer 内で cross-fit 学習。
  1. base model の OOF スコアを取得。
  2. calibration CV で cross-fit し、calibrated OOF を生成。
  3. C_final を全 OOF で学習。
  4. 推論時は C_final を使用。
- 評価は cross-fit で生成した calibrated 値で行う（C_final ではない）。

### テスト
- 各 Calibrator の fit / predict
- **リークテスト（MUST）**: 同一行で学習した calibrator で評価していないことの検証
- C_final と cross-fit の区別
- Calibrator が X を受け取れないことの検証

### DoD
- 3 種の Calibrator が動作
- リーク防止テスト通過
- CVTrainer との統合テスト通過

---

## Phase 14: Persistence & Export

**SKILL:** skills/persistence-and-migration/SKILL.md, skills/export/SKILL.md

### 14-1. persistence/serializer.py

- `format_version` を必須で保存。
- 保存対象:
  - lizyml_version / python_version / deps_versions
  - config_normalized
  - schema (feature_names / dtypes / categorical policy)
  - split indices
  - data_fingerprint
  - pipeline_state
  - models / calibrator
  - metrics / history
- 互換性: 読めない format_version は明示的に拒否。

### 14-2. persistence/model_store.py

- `save(path, artifacts)`: ディレクトリにまとめて保存。
- `load(path) -> artifacts`: メタデータ検証付きで復元。

### 14-3. Model.load() 統合

- `Model.load(path)` で復元後:
  - `predict(X_new)` が実行可能
  - `evaluate()` で学習時の評価情報を参照可能

### テスト（E2E）
- save → load → predict の一貫テスト
- format_version 不一致時のエラー
- メタデータ欠損時のエラー
- load 後の evaluate で学習時情報が取得できること

### DoD
- export → load → predict / evaluate のE2Eテスト通過
- format_version 管理が機能している

---

## Phase 15: Explain（SHAP等）

**SKILL:** skills/explain/SKILL.md

### 15-1. explain/base.py

```python
class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, model, X) -> np.ndarray: ...
    @property
    @abstractmethod
    def method_name(self) -> str: ...
```

### 15-2. explain/shap.py

- SHAP ライブラリを使用（optional dependency）。
- TreeExplainer（LGBM向け）。

### 15-3. explain/lgbm_contrib.py

- LightGBM 内蔵の feature contribution。
- SHAP ライブラリ不要。

### テスト
- shap_values の shape 検証
- optional dependency 未インストール時のエラーコード
- predict(return_shap=True) のE2E

### DoD
- SHAP / lgbm_contrib が動作
- PredictionResult.shap_values に正しく格納

---

## Phase 16: Plots

**SKILL:** skills/plots/SKILL.md

### 16-1. plots/learning_curve.py

- fold ごとの学習曲線。入力: FitResult.history。

### 16-2. plots/importance.py

- 特徴量重要度の棒グラフ。入力: importance dict。

### 16-3. plots/residuals.py

- 残差分布。入力: FitResult.oof_pred + y_true。

### 16-4. plots/calibration.py

- Reliability diagram / ECE。入力: FitResult (calibrated/raw)。

### 16-5. plots/classification.py

- ROC / PR / Confusion Matrix。入力: FitResult。

**全プロットの鉄則:**
- FitResult（+ 最小限の補助データ）のみを入力とする。
- 推論・再計算をしない。データ不足時はエラー。

### テスト
- 各プロットが例外なく実行できる（matplotlib backend を "Agg" に設定）
- 必要データ不足時のエラー

### DoD
- 主要プロットが FitResult から生成可能
- Model のメソッド（importance_plot, residuals_plot 等）が薄いラッパーとして動作

---

## Phase 17: E2E テスト・統合テスト


### 17-1. 全パイプライン E2E テスト

小規模データ（合成データ）を用いた完全なフロー:

```
Config → tune → fit → evaluate → predict → export → load → predict
```

対象タスク: regression, binary, multiclass

### 17-2. ゴールデンテスト

- FitResult / PredictionResult / Artifacts のスキーマ固定テスト
- 変更時に意図的に破壊されることを検証

### 17-3. リークテスト群

- OOF リーク検知
- Calibration リーク検知
- FeaturePipeline リーク検知（validation データで fit していない）
- tuning リーク回避の検証

### 17-4. 再現性テスト

- 同一 config / seed で split indices と主要指標が一致

### 17-5. 列ズレテスト

- 余剰列 / 不足列 / unseen category のポリシー動作

### 17-6. Public API surface テスト

- `from lizyml import Model` 等のトップレベル公開面が ImportError なく動作する
- `__init__.py` の re-export 一覧が期待通りであることを検証（名前列挙）

### 17-7. バージョン一致テスト

- `lizyml.__version__` と `importlib.metadata.version("lizyml")` が一致する

### 17-8. README サンプルコードテスト

- `README.md` から最短利用例を抽出し、`SyntaxError` / `ImportError` なく実行可能なことを検証（データ依存部分はモック可）

### DoD
- 全 E2E テスト通過
- 全リークテスト通過（「落ちるべき例」含む）
- 再現性テスト通過
- Public API surface テスト通過
- バージョン一致テスト通過
- README サンプルコードテスト通過

---

## Phase 18: CI・PyPI 配布検証

**SKILL:** skills/dev-environment/SKILL.md, skills/release/SKILL.md

### 18-1. GitHub Actions ワークフロー

`.github/workflows/ci.yml`:

```yaml
on:
  pull_request:
    branches: [main]

jobs:
  quality:
    strategy:
      matrix:
        python-version: ["3.10", "3.12"]    # 下限 + 最新安定版
    steps:
      - uv sync --frozen
      - uv run ruff check .
      - uv run ruff format --check .
      - uv run mypy lizyml/
      - uv run pytest --cov=lizyml

  quality-lowest-deps:                       # 依存下限バージョンテスト
    steps:
      - uv sync --frozen --resolution lowest-direct
      - uv run pytest

  distribution:
    steps:
      - uv build                             # sdist + wheel
      - uv run twine check dist/*            # 配布メタデータ検証
      - pip install dist/*.whl               # install smoke test
      - python -c "from lizyml import Model; print('import ok')"
```

### 18-2. PyPI 配布検証（BLUEPRINT §18.2 対応）

- `sdist / wheel` の build を CI で必ず実行し、ビルド可能性を担保する。
- `twine check` 相当で配布メタデータ（README 表示・classifiers 等）を検証する。
- install smoke test: 配布物からの import と README の最短利用例が破綻していないことを確認する。
- README の import 例がトップレベル公開面（`lizyml/__init__.py`）と一致していることを検証する。

### 18-3. 複数 Python バージョン・依存下限テスト（BLUEPRINT §18.2 対応）

- `requires-python` の下限（3.10）と最新安定版（3.12）の両方でテストを実行する。
- `uv` の `--resolution lowest-direct` を使い、依存の下限バージョンでテストを実行して下限が嘘でないことを保証する。

### DoD
- main への PR で CI が自動実行
- uv.lock 整合性・Lint・Format・TypeCheck・テストが全て通る
- `uv build` で sdist / wheel が正常に生成される
- `twine check` がエラーなしで通る
- install smoke test で import が成功する
- Python 3.10 と 3.12 の両方でテストが通る
- 依存下限バージョンでテストが通る

---

## 追加計画: テストカバレッジ改善（BLUEPRINT 準拠）

現時点では unit test は十分に整っている一方、BLUEPRINT の MUST を
「Facade 経由で本当に守れているか」を固定する統合テストが相対的に薄い。
次の改善は、既存機能の実装追加より先に、仕様逸脱を早期に検知するための
テスト拡充として実施する。

### T-1. Split 設定の統合テスト強化（最優先）

**対象:** BLUEPRINT §10, §6.2, §3.3

- `Model.fit()` が `split.method` を正しく反映することを E2E で検証する。
- 少なくとも以下を追加する。
  - `stratified_kfold`: 各 fold の target 比率が極端に崩れない
  - `group_kfold`: `FitResult.splits.outer` 上で group overlap が発生しない
  - `time_series`: 各 fold で `train_idx.max() < valid_idx.min()` を満たす
- splitter 単体ではなく、`Model` / `CVTrainer` / `FitResult.splits` まで通す。

**DoD**
- `tests/test_e2e/` または `tests/test_model_facade.py` に non-`kfold` 系の統合テストが追加されている
- split 設定を無視した場合に落ちるテストになっている

### T-2. Config 互換性ゲートのテスト追加

**対象:** BLUEPRINT §5, §15.2, §16.2

- `config_version` の必須性だけでなく、未対応 version の拒否をテストで固定する。
- 少なくとも以下を追加する。
  - `config_version` 欠落 → `CONFIG_INVALID`
  - `config_version` が未対応値（例: `999`） → `CONFIG_VERSION_UNSUPPORTED`
  - `Model(config=...)` 経由でも同じ例外契約を満たす

**DoD**
- `tests/test_config/` に unsupported version の失敗ケースが追加されている
- 例外 code が `CONFIG_VERSION_UNSUPPORTED` で固定されている

### T-3. Calibration 契約の固定テスト追加

**対象:** BLUEPRINT §12, §7.1, §10.4

- calibration の「リークしない」だけでなく、返却 shape と保存対象を固定する。
- 少なくとも以下を追加する。
  - `FitResult.splits.calibration` が calibration 有効時に保存される
  - `metrics["calibrated"]` が `raw` と同型の階層を持つ
    （`oof`, `if_mean`, `if_per_fold`）
  - calibration 有効モデルの export / load 後も予測時に校正器が適用される
  - `beta` を公開設定に含める場合は、実装済みであること、未実装なら明示的に失敗すること

**DoD**
- `tests/test_calibration/` と `tests/test_persistence/` に calibration 契約テストが追加されている
- calibration の split / metrics の shape 変更が破壊的変更として検知される

### T-4. Evaluation 契約の追加テスト

**対象:** BLUEPRINT §6.3, §13.2

- `evaluate()` のデフォルト経路だけでなく、指定メトリクス経路を固定する。
- 少なくとも以下を追加する。
  - `Model.evaluate(metrics=[...])` が指定メトリクスで再計算できる
  - task 非対応 metric 指定時に正しい例外 code を返す
  - load 後の `Model.evaluate(metrics=[...])` の契約を定義し、成功または明示的失敗を固定する

**DoD**
- `tests/test_e2e/` または `tests/test_evaluation/` に custom metrics 経路のテストが追加されている
- `evaluate(metrics=[...])` の挙動が曖昧でなくなる

### T-5. Persistence 保存対象の固定テスト

**対象:** BLUEPRINT §15.1, §7.3

- `metadata.json` の最低限キーだけではなく、保存契約をより厳密に固定する。
- 少なくとも以下を追加する。
  - `format_version`
  - `config_normalized`
  - `metrics`
  - `feature_names`
  - `run_id`
  - `FitResult` 内の `splits / data_fingerprint / pipeline_state / run_meta`
- 「roundtrip できた」だけでなく、「必要情報が保持される」を検証する。

**DoD**
- `tests/test_persistence/` に保存対象の明示的な契約テストが追加されている
- 保存項目の欠落や key 名変更がテストで検知される

### T-6. Column Drift / PredictionResult 意味の固定

**対象:** BLUEPRINT §7.2, §9.2

- 列ズレを単に「落ちない」で終わらせず、戻り値の意味まで固定する。
- 少なくとも以下を追加する。
  - extra column 時に `PredictionResult.warnings` に警告が入る
  - `used_features` が学習時の列順を保持する
  - missing column は warning ではなく hard error になる

**DoD**
- `tests/test_e2e/test_column_drift.py` で warnings / used_features の意味が固定されている
- 列ズレポリシーの挙動変更を破壊的変更として検知できる

### T-7. Config / Data 入口の E2E テスト

**対象:** BLUEPRINT §5.1, §8.1

- loader 単体ではなく、Facade の入口として複数の入力形式を固定する。
- 少なくとも以下を追加する。
  - `Model(config=<dict>)`
  - `Model(config=<json path>)`
  - `Model(config=<yaml path>)`
  - `Model(config=<LizyMLConfig instance>)`
  - `fit(data=None)` かつ `config.data.path` からの学習

**DoD**
- `tests/test_e2e/` に config path / data path 経路の統合テストが追加されている
- README / 公開 API と実際の入口が乖離していないことを確認できる

### 実施順序

1. `T-1 Split 設定の統合テスト`
2. `T-2 Config 互換性ゲート`
3. `T-3 Calibration 契約`
4. `T-4 Evaluation 契約`
5. `T-5 Persistence 保存対象`
6. `T-6 Column Drift / PredictionResult`
7. `T-7 Config / Data 入口`

理由:
- まず split / config / calibration を押さえると、BLUEPRINT の MUST 逸脱を最も早く検知できる。
- その後に evaluate / persistence / prediction 契約を固定すると、戻り値と保存互換性のブレを抑えられる。
- 最後に入口 E2E を固めると、README と公開 API の実使用導線を保証できる。

### 完了条件（この改善計画全体の DoD）

- BLUEPRINT の MUST に対応する「落ちるべき例」が各主要領域に追加されている
- `tests/test_e2e/` が unit test の穴埋めではなく、公開 API 契約の固定テストとして機能している
- split / leakage / calibration / persistence の仕様逸脱を、実装変更時にテストが即時検知できる
- 新規テスト追加後も `uv run pytest`, `uv run ruff check .`, `uv run mypy lizyml/` が通る

---

## 作業ブランチ戦略

```
main
  └── develop
        ├── feat/phase-0-dev-env
        ├── feat/phase-1-foundation
        ├── feat/phase-2-config-specs
        ├── feat/phase-3-data-layer
        ├── ...
        └── feat/phase-18-ci
```

- 各 Phase は `develop` ベースのフィーチャーブランチで作業。
- PR は `develop` に対して作成。
- 全 Phase 完了後、`develop` → `main` の PR で CI を通して統合。

---

## 補足: HISTORY.md Proposal が必要なタイミング

以下の変更は実装前に HISTORY.md に Proposal を追加する:

- 公開 API（Model / Config / FitResult / PredictionResult / Artifacts）
- Result の意味・shape
- split / leakage 境界
- export / simulate のフォーマット
- optional dependency 追加
- 保存互換性（format_version）

Phase 2（Config）・Phase 4（型契約）・Phase 14（Persistence）が特に該当する。
これらのフェーズ開始時に、まず Proposal を HISTORY.md に記録してから実装に入る。

---

## 見積もり

本計画は見積もりを含まない（CLAUDE.md の方針に従い、時間予測は行わない）。
各 Phase の DoD を満たした時点で次 Phase に進む。
