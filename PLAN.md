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
| 19 | evaluate_table / residuals / SHAP importance + Plotly 移行 | table_formatter, residuals plot, SHAP importance, Plotly 移行 | Phase 15, 16 |
| 20 | 分類強化 / Multiclass メトリクス・OvR ROC / InnerValid 拡張（Stratified デフォルト / Precision@K / ROC / Confusion Matrix / Calibration 可視化 / Multiclass メトリクス・OvR ROC / InnerValid stratified・group・time holdout） | StratifiedKFold デフォルト化, precision_at_k, ROC Curve (binary+multiclass OvR), Confusion Matrix, Calibration Curve, Probability Histogram, multiclass metrics (auc/auc_pr/brier OvR), InnerValid stratified/group/time holdout + デフォルト自動解決 | Phase 7, 9, 13, 16, 19 |
| 21 | LightGBM パラメーター強化 | LGBMConfig スマートパラメーター, デフォルトプロファイル, TuningResult + tuning_table(), デフォルト Tuning Space, Notebook 更新 | Phase 8, 9, 11, 12 |
| 22 | 監査乖離の是正 + 追加開発 | load 後診断 API, Config デフォルト修正, tuning_plot, fit_result プロパティ, Notebook 全項目 Config, params_table, n_rows inner train 基準, ドキュメント整合 | Phase 20, 21 |
| 23 | Calibration 強化・時系列拡張 | raw score Calibration, Beta Calibration, PurgedTimeSeries/GroupTimeSeries Config 接続, split_summary, 時系列 Notebook, Logging 統一 | Phase 22 |
| 24 | LGBMAdapter Booster API 移行 | sklearn wrapper → `lgb.train()` 移行, predict/proba/raw shape 維持, get_native_model 戻り値変更, 学習履歴適応, persistence 互換 | Phase 23 |
| 25 | Model Facade 分割・テスト基盤改善 ✅ | model.py mixin 分割, conftest 集約, parametrize 強化, CI develop 対応, カバレッジ閾値, slow テストスキップ, optional dep テスト補完 | Phase 24 |

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

## Phase 19: evaluate_table / residuals_plot / importance_plot(kind="shap") + Plotly 移行

**HISTORY:** H-0005, H-0006, H-0007, H-0008
**依存:** Phase 15 (Explain/SHAP), Phase 16 (Plots)
**ブランチ:** `feat/phase-19-eval-table-residuals-shap`

### 背景

Notebook 利用を通じて、「ユーザーにコードを書かせない」思想に対する不足が判明。
併せて全プロットを matplotlib から Plotly に移行し、インタラクティブな可視化を提供する。

### 19-A: Plotly 移行（H-0008）

1. `pyproject.toml` の optional dep `plots` を `matplotlib>=3.7` → `plotly>=5.0` に変更
2. `dependency-groups` (dev) も同様に変更
3. mypy overrides: `matplotlib.*` → `plotly.*`
4. 既存 plots を Plotly に書き換え:
   - `lizyml/plots/importance.py`: 横棒グラフ → Plotly
   - `lizyml/plots/learning_curve.py`: 折れ線グラフ → Plotly
   - `lizyml/plots/oof_distribution.py`: ヒストグラム → Plotly
5. optional dep sentinel を `_mpl` → `_plotly` に変更
6. `tests/test_plots/test_plots.py` を Plotly Figure アサーションに更新

### 19-B: evaluate_table（H-0005）

1. `lizyml/evaluation/table_formatter.py` 新規作成
   - `format_metrics_table(metrics: dict) -> pd.DataFrame`
   - 行 = メトリクス名、列 = `oof`, `if_mean`, `fold_0`...`fold_N-1`, `cal_oof`（存在時）
2. `lizyml/core/model.py` に `evaluate_table()` 追加（table_formatter に委譲）
3. `tests/test_evaluation/test_table_formatter.py` 新規テスト

### 19-C: residuals / residuals_plot（H-0006）

1. `lizyml/core/model.py`:
   - `self._y` を `fit()` 中に一時保持（export には含めない）
   - `residuals()`: 回帰専用、`y - oof_pred` を返す
   - `residuals_plot()`: `plots/residuals.py` に委譲
2. `lizyml/plots/residuals.py` 新規作成
   - `plot_residuals(residuals)`: ヒストグラム + QQ plot の 2 パネル（Plotly）
3. `tests/test_plots/test_residuals.py` 新規テスト

### 19-D: importance(kind="shap") / importance_plot(kind="shap")（H-0007）

1. `lizyml/core/model.py`:
   - `self._X` を `fit()` 中に一時保持（export には含めない）
   - `importance(kind="shap")`: `shap_explainer.compute_shap_importance()` に委譲
   - `importance_plot(kind="shap")`: `importance()` で dict を取得し `plot_importance_from_dict()` に委譲
2. `lizyml/explain/shap_explainer.py` に `compute_shap_importance()` 追加
   - pipeline_state から NativeFeaturePipeline を復元 → X を transform
   - fold ごとに validation データで SHAP 計算 → mean(|SHAP|) を fold 平均
3. `lizyml/plots/importance.py` に `plot_importance_from_dict()` 追加
4. テスト追加

### テスト

- `test_evaluation/test_table_formatter.py`: 構造、calibrated、fold 列、E2E、MODEL_NOT_FIT
- `test_plots/test_residuals.py`: shape、回帰専用、load 後エラー、Plotly Figure、OPTIONAL_DEP_MISSING
- `test_plots/test_plots.py`: 既存テスト Plotly 更新、importance_plot(kind="shap")
- `test_explain/`: SHAP importance dict、load 後エラー、OPTIONAL_DEP_MISSING

### DoD

- 全プロットメソッドが Plotly Figure を返す
- `evaluate_table()` が DataFrame を返す
- `residuals()` / `residuals_plot()` が回帰タスクで動作
- `importance(kind="shap")` / `importance_plot(kind="shap")` が動作
- 全テスト・lint・mypy 通過

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

**対象:** BLUEPRINT §15.1, §7.4

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

**対象:** BLUEPRINT §7.3, §9.2

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

## Phase 20: 分類強化 / Multiclass メトリクス・OvR ROC / InnerValid 拡張

**HISTORY:** H-0013, H-0014, H-0015, H-0016, H-0017, H-0018, H-0019, H-0020
**SKILL:** skills/config-and-specs/SKILL.md, skills/metrics/SKILL.md, skills/plots/SKILL.md, skills/evaluation-contracts/SKILL.md, skills/training-cv-and-inner-valid/SKILL.md, skills/splitters-and-splitplan/SKILL.md
**依存:** Phase 7 (Metrics), Phase 9 (Training Core), Phase 13 (Calibration), Phase 16 (Plots), Phase 19 (Plotly)

### 背景

Binary / Multiclass 分類 Notebook の作成にあたり、StratifiedKFold デフォルト化、Precision@K メトリクス、ROC Curve、Confusion Matrix テーブル、Calibration 可視化、Multiclass メトリクス拡張が不足していることが判明。また、EarlyStopping 用の InnerValid が holdout（ランダム）のみで、外側 CV の split 方式に応じた stratified / group / time-aware な内側分割ができないことも判明。

### 20-A: StratifiedKFold デフォルト化 + KFold 警告（H-0013）

1. `lizyml/config/loader.py`: 正規化で `task` が `binary`/`multiclass` かつ `split.method` 未指定時に `stratified_kfold` をデフォルトにする
2. `lizyml/core/model.py`: 分類タスクで `method="kfold"` 明示指定時に `warnings.warn()` を出す
3. テスト: デフォルト切替、警告出力、回帰タスク非影響

### 20-B: Precision at K メトリクス（H-0014）

1. `lizyml/metrics/classification.py`: `PrecisionAtKMetric` クラス追加（`needs_proba=True`, `k` パラメータ）
2. `lizyml/metrics/registry.py`: `TASK_METRICS["binary"]` に登録
3. テスト: 正常系、k パラメータ変更、regression/multiclass での UNSUPPORTED_METRIC

### 20-C: ROC Curve プロット — binary + multiclass OvR（H-0015, H-0019）

1. `lizyml/plots/classification.py` 新規作成: `plot_roc_curve(fit_result, y_true)`
   - **binary**: IS/OOS の 2 本の ROC Curve を重ね描き、AUC 値を凡例に表示
   - **multiclass**: IS/OOS を subplot 横並びにし、クラスごとの OvR ROC Curve を描画。各クラスの AUC 値を凡例に、macro 平均 AUC も表示
   - IS: `if_pred_per_fold` + `splits.outer` train_idx から集約
   - OOS: `oof_pred` から算出
2. `lizyml/plots/__init__.py`: export 追加
3. `lizyml/core/model.py`: `roc_curve_plot()` メソッド追加
4. テスト: Figure 構造、IS/OOS トレース存在、binary/multiclass 両方、regression エラー

### 20-D: Confusion Matrix テーブル（H-0016）

1. `lizyml/evaluation/confusion.py` 新規作成: `confusion_matrix_table(fit_result, y_true, threshold=0.5)`
   - 戻り値: `{"is": pd.DataFrame, "oos": pd.DataFrame}`
   - IS: `if_pred_per_fold` + `splits.outer` train_idx から集約
   - OOS: `oof_pred` から算出
   - binary: threshold、multiclass: argmax
2. `lizyml/core/model.py`: `confusion_matrix()` メソッド追加
3. テスト: DataFrame shape、threshold パラメータ、multiclass 対応、regression エラー

### 20-E: Calibration Curve + Probability Histogram（H-0017）

1. `lizyml/plots/calibration.py` 新規作成:
   - `plot_calibration_curve(fit_result, y_true)`: Raw/Calibrated の Reliability Diagram（bin 数デフォルト 10、理想線 y=x）
   - `plot_probability_histogram(fit_result)`: Raw/Calibrated の確率分布ヒストグラム重ね描き
   - データソース: `oof_pred`（Raw）と `calibrator.calibrated_oof`（Calibrated）。`c_final` は使用しない。
2. `lizyml/plots/__init__.py`: export 追加
3. `lizyml/core/model.py`: `calibration_plot()`, `probability_histogram_plot()` メソッド追加
4. テスト: Figure 構造、Raw/Calibrated トレース存在、calibration 未有効時エラー、binary 以外エラー

### 20-F: Multiclass メトリクス拡張 — AUC/Average Precision/Brier OvR（H-0018）

1. `lizyml/metrics/classification.py`:
   - `AUCMetric.__call__`: `y_pred.ndim == 2` の場合 `roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')` に分岐
   - `AUCPRMetric.__call__`: `y_pred.ndim == 2` の場合 One-Hot 展開 + クラスごと `average_precision_score` の macro 平均
   - `BrierMetric.__call__`: `y_pred.ndim == 2` の場合 One-Hot 展開 + クラスごと `brier_score_loss` の macro 平均
   - 各クラスの `supports_task` に `"multiclass"` を追加
2. `lizyml/metrics/registry.py`: `TASK_METRICS["multiclass"]` に `auc`, `auc_pr`, `brier` を追加
3. テスト: multiclass での正常系、binary 既存動作の非変更、regression での UNSUPPORTED_METRIC

### 20-G: InnerValid の split method 設定対応 — stratified / group / time-aware holdout（H-0020）

1. `lizyml/config/schema.py`:
   - `HoldoutInnerValidConfig` に `stratify: bool = False` フィールド追加
   - `GroupHoldoutInnerValidConfig(method="group_holdout", ratio, random_state)` 追加
   - `TimeHoldoutInnerValidConfig(method="time_holdout", ratio)` 追加
   - `InnerValidConfig` を discriminated union に拡張
   - `EarlyStoppingConfig.inner_valid` の型を `InnerValidConfig | None` に変更
2. `lizyml/training/inner_valid.py`:
   - `HoldoutInnerValid`: `stratify=True` 時に `StratifiedShuffleSplit(n_splits=1, test_size=ratio)` を使用
   - `GroupHoldoutInnerValid`: group 単位で末尾グループを validation に割り当て（group overlap 禁止）
   - `TimeHoldoutInnerValid`: 末尾 ratio 割合を validation に割り当て（shuffle なし、時系列順維持）
3. `lizyml/core/model.py`: `_build_inner_valid()` にデフォルト自動解決ロジック追加
   - 外側 `split.method` に応じて `stratified_kfold` → `holdout(stratify=True)`, `group_kfold` → `group_holdout`, `time_series` → `time_holdout`, `kfold` → `holdout(stratify=False)`
4. `lizyml/training/cv_trainer.py`: `inner_valid.split()` に `groups` 引数を渡す
5. テスト: デフォルト自動解決、明示指定の優先、stratified 層化検証、group overlap 禁止、time-aware shuffle 禁止

### テスト

- `tests/test_config/test_stratified_default.py`: デフォルト切替、警告、回帰非影響
- `tests/test_metrics/test_precision_at_k.py`: 正常系、k パラメータ、タスク非対応エラー
- `tests/test_metrics/test_multiclass_metrics.py`: AUC OvR、Average Precision OvR、Brier OvR の正常系 + binary 非変更
- `tests/test_plots/test_classification.py`: ROC Curve の IS/OOS トレース（binary + multiclass OvR）
- `tests/test_evaluation/test_confusion.py`: Confusion Matrix の shape、threshold、multiclass 対応
- `tests/test_plots/test_calibration.py`: Calibration Curve/Histogram のトレース、calibration 未有効エラー
- `tests/test_training/test_inner_valid.py`: stratified holdout 層化検証、group holdout overlap 禁止、time holdout shuffle 禁止、デフォルト自動解決、明示指定の優先

### DoD

- `task="binary"` で split 未指定時に StratifiedKFold が使われる
- `precision_at_k` が binary タスクの evaluate で利用できる
- `roc_curve_plot()` が binary は IS/OOS 重ね描き、multiclass は OvR ROC を描画する
- `confusion_matrix()` が IS/OOS の Confusion Matrix テーブルを返す（binary + multiclass）
- `calibration_plot()` が Raw/Calibrated の Reliability Diagram を描画する
- `probability_histogram_plot()` が Raw/Calibrated の確率分布を描画する
- `task="multiclass"` で `evaluate(metrics=["auc", "auc_pr", "brier"])` が OvR macro 平均を返す
- `inner_valid` 未指定時に外側 CV method に応じた stratified / group / time holdout が自動解決される
- `inner_valid` 明示指定が外側 CV method に関わらず優先される
- `time_holdout` で shuffle が行われない、`group_holdout` で group overlap が発生しない
- 全テスト・lint・mypy 通過

---

## Phase 21: LightGBM パラメーター強化

**HISTORY:** H-0021, H-0022, H-0023, H-0024
**SKILL:** skills/config-and-specs/SKILL.md, skills/add-estimator-adapter/SKILL.md
**依存:** Phase 8 (EstimatorAdapter), Phase 9 (Training Core), Phase 11 (Model Facade)

### 背景

LightGBM のパラメーター設定において、データサイズ依存のパラメーター手動計算が必要、デフォルト値が LightGBM ライブラリ任せで実務に最適化されていない、という課題がある。スマートパラメーター機能とタスク別デフォルトプロファイルを追加し、最小限の Config で妥当な精度のモデルを構築できるようにする。

### 21-A: LGBMConfig スマートパラメーター（H-0021）

1. `lizyml/config/schema.py`:
   - `LGBMConfig` に `auto_num_leaves`, `num_leaves_ratio`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`, `feature_weights`, `balanced` フィールド追加
   - `model_validator` で `auto_num_leaves=True` + `params.num_leaves` 等の競合検知
   - `num_leaves_ratio`（0 < ratio ≤ 1）、`min_data_in_leaf_ratio` / `min_data_in_bin_ratio`（0 < ratio < 1）の範囲バリデーション
2. `lizyml/estimators/lgbm.py`:
   - `resolve_smart_params(params, n_rows, feature_names, y, task)` 関数追加
   - auto_num_leaves → `num_leaves` 算出（`clamp(ceil(base × ratio), 8, 131072)`）
   - ratio params → 絶対値変換（`max(1, ceil(n_rows × ratio))`）
   - feature_weights dict → リスト変換 + `feature_pre_filter=False` 強制
   - balanced → binary: `scale_pos_weight`, multiclass: `sample_weight` 算出
3. `lizyml/core/model.py`:
   - `fit()` で `n_rows` / `feature_names` / `y` を `resolve_smart_params()` に渡す
   - 解決済みパラメーターを `LGBMAdapter` に渡す
4. テスト:
   - `tests/test_config/test_lgbm_smart_params.py`: バリデーション（競合、範囲外、未知特徴量）
   - `tests/test_estimators/test_lgbm_resolve.py`: 解決ロジック（num_leaves 算出、ratio 変換、feature_weights リスト化、balanced 重み算出）

### 21-B: デフォルトパラメータープロファイル（H-0022）

1. `lizyml/estimators/lgbm.py`:
   - `_TASK_OBJECTIVE` 更新: regression を `regression` → `huber` に変更
   - `_TASK_METRIC` 更新: タスク別メトリクスリスト化（`[huber, mae, mape]` 等）
   - `_build_params()` にデフォルト値追加: `boosting`, `learning_rate`, `max_depth`, `max_bin`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l1`, `lambda_l2`, `n_estimators`, `first_metric_only`
2. `lizyml/config/schema.py`:
   - `EarlyStoppingConfig` のデフォルト値変更: `enabled=True`, `rounds=150`, `validation_ratio=0.1`
3. 既存テスト更新:
   - デフォルト値変更に伴う再現性テスト・seed 固定テストの期待値更新
   - `EarlyStoppingConfig` デフォルト変更に伴うテスト修正
4. テスト:
   - `tests/test_estimators/test_lgbm_defaults.py`: タスク別 objective/metric デフォルト、共通デフォルト適用、params 上書き
   - `tests/test_config/test_early_stopping_defaults.py`: 新デフォルト値の検証

### 21-C: Notebook 更新

Phase 21 実装完了後、既存 Notebook を更新する。

1. `notebooks/tutorial_regression_lgbm.ipynb`:
   - Config を Phase 21 のデフォルトプロファイルに合わせて更新（不要な明示指定を削除し、デフォルトに任せる）
   - スマートパラメーター（`auto_num_leaves`, `min_data_in_leaf_ratio` 等）の使用例を Config に追加
   - **LightGBM パラメーター確認セル**を追加: fit 後に全パラメーターを一覧表示
     - LGBMConfig のスマートパラメーター設定値（`auto_num_leaves`, `num_leaves_ratio`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`, `feature_weights`, `balanced`）
     - スマートパラメーター解決結果（`num_leaves` 算出値、`min_data_in_leaf` / `min_data_in_bin` 変換後の絶対値、`feature_weights` リスト、`scale_pos_weight` 等）
     - LightGBM ネイティブパラメーター（`objective`, `metric`, `boosting`, `learning_rate`, `max_depth`, `max_bin`, `feature_fraction`, `bagging_fraction` 等 — `booster_.params` から取得）
     - Training パラメーター（`early_stopping.rounds`, `validation_ratio`, `n_estimators`）
     - fold ごとの `best_iteration`
2. 将来の binary / multiclass Notebook にも同様のパラメーター確認セルを含める

### 21-D: TuningResult 型と tuning_table() API（H-0023）

1. `lizyml/core/types/tuning_result.py`:
   - `TrialResult` dataclass: `number`, `params`, `score`, `state`
   - `TuningResult` dataclass: `best_params`, `best_score`, `trials`, `metric_name`, `direction`
2. `lizyml/tuning/tuner.py`:
   - `tune()` の戻り値を `dict[str, Any]` → `TuningResult` に変更
   - `study.trials` から `TrialResult` リストを構築
3. `lizyml/core/model.py`:
   - `tune()` の戻り値を `TuningResult` に変更
   - `self._tuning_result` として保持
   - `tuning_table() -> pd.DataFrame` メソッド追加（列: `trial`, メトリクス名, 探索パラメーター名）
4. テスト:
   - `tests/test_tuning/test_tuning_result.py`: TuningResult 構造、trials 保持、tuning_table DataFrame 形状
   - 既存の tune テストの戻り値アサーション更新

### 21-E: デフォルト Tuning Space（H-0024）

1. `lizyml/tuning/search_space.py`:
   - `SearchDim` にカテゴリ属性 (`category: Literal["model", "smart", "training"]`) を追加
   - `default_space(task: str) -> list[SearchDim]` 関数追加: タスク別デフォルト探索空間を返す
   - `default_fixed_params(task: str) -> dict[str, Any]` 関数追加: 固定パラメーター（`auto_num_leaves=True`, `first_metric_only=True`, `metric`）を返す
2. `lizyml/tuning/tuner.py`:
   - `tune()` の objective 内で `trial_params` をカテゴリ別に分類し適用:
     - `model` → `estimator_factory` に渡す（現行通り）
     - `smart` → `resolve_smart_params()` に渡す
     - `training` → `early_stopping_rounds` を estimator に渡し、`validation_ratio` で `InnerValidStrategy` を再構築
   - `inner_valid` を `inner_valid_factory: Callable[[float], BaseInnerValidStrategy]` に変更（`validation_ratio` が探索対象の場合）
3. `lizyml/core/model.py`:
   - `tune()` で `tuning.optuna.space` が空の場合に `default_space(task)` を使用
   - 拡張 `estimator_factory` / `inner_valid_factory` の構築
   - 固定パラメーター（`first_metric_only`, `metric`）を base_params に適用
4. テスト:
   - `tests/test_tuning/test_default_space.py`: タスク別デフォルト空間の次元数・型・範囲、固定パラメーター
   - `tests/test_tuning/test_tuner_extended.py`: smart params / training params の per-trial 適用

### テスト

- `tests/test_config/test_lgbm_smart_params.py`: auto_num_leaves 競合、ratio 範囲、feature_weights 未知特徴量、balanced + regression
- `tests/test_estimators/test_lgbm_resolve.py`: num_leaves 算出ロジック、ratio→絶対値変換、feature_weights リスト化 + feature_pre_filter、balanced scale_pos_weight/sample_weight
- `tests/test_estimators/test_lgbm_defaults.py`: タスク別 objective/metric、共通デフォルト、params 上書き優先
- `tests/test_config/test_early_stopping_defaults.py`: enabled=True / rounds=150 / validation_ratio=0.1
- `tests/test_tuning/test_tuning_result.py`: TuningResult 構造、trials 保持、tuning_table DataFrame
- `tests/test_tuning/test_default_space.py`: デフォルト空間定義、固定パラメーター
- `tests/test_tuning/test_tuner_extended.py`: smart/training params の per-trial 適用

### DoD

- `auto_num_leaves=True` + `max_depth=5` → `num_leaves=ceil(32 × ratio)` が正しく算出される
- ratio パラメーターが `n_rows` から正しく解決される
- `feature_weights` dict がリスト変換され `feature_pre_filter=False` が設定される
- `balanced=True` でクラス重みが算出される（binary: scale_pos_weight、multiclass: sample_weight）
- デフォルトパラメーターが全タスクで正しく適用される（objective/metric/共通パラメーター）
- `params` 指定がデフォルトを上書きする
- `auto_num_leaves=True` + `params.num_leaves` 指定 → `CONFIG_INVALID`
- `balanced=True` + regression → `UNSUPPORTED_TASK`
- `model.tune()` が `TuningResult` を返し、`best_params` / `best_score` / `trials` を含む
- `model.tuning_table()` が全 trial の DataFrame を返す（列: trial, メトリクス名, 探索パラメーター名）
- `tune()` 未実行時の `tuning_table()` が `MODEL_NOT_FIT` エラー
- `tuning.optuna.space` が空の場合にタスク別デフォルト空間が自動適用される
- デフォルト空間で `num_leaves_ratio` / `min_data_in_leaf_ratio`（smart）と `early_stopping_rounds` / `validation_ratio`（training）が trial ごとに変更される
- `first_metric_only=True` と `metric` がデフォルトで固定適用される
- ユーザー指定の `space` がデフォルトを上書きする
- 全テスト・lint・mypy 通過

---

## Phase 22: 監査乖離の是正（Phase 20/21 Follow-up）

**HISTORY:** H-0025, H-0026
**SKILL:** skills/spec-update/SKILL.md, skills/history-proposals/SKILL.md, skills/testing/SKILL.md
**依存:** Phase 20, Phase 21

### 背景

Requirements Audit（Phase 20/21 実装後）で、仕様との部分的乖離が 4 点検出された。  
このうち 22-A（load 後 API 境界）は、`H-0025` の「load 後不可へ厳格化」から `H-0026` の「load 後も診断 API 利用可能」へ方針変更済み。

### 22-A: `Model.load()` 後診断 API の利用可能化（H-0026 / H-0025-1 superseded）

1. `lizyml/core/model.py`:
   - `Model.load()` 後でも以下 API が実行可能になるよう境界を更新する:
     `residuals()`, `residuals_plot()`, `importance(kind="shap")`, `roc_curve_plot()`,
     `confusion_matrix()`, `calibration_plot()`, `probability_histogram_plot()`
   - `fit()` 実行直後と `load()` 復元直後で、上記 API の意味・出力契約が一致することを保証する
2. `lizyml/persistence/*`:
   - load 後診断 API に必要な `analysis_context`（`y_true`, `X_for_explain`）を export 対象に追加
   - loader で `analysis_context` を復元し、Model インスタンスへ受け渡す
3. 互換性対応:
   - 既存 artifact（`analysis_context` なし）を load 可能に維持
   - 既存 artifact で診断 API を呼んだ場合は、再 export を促す明示的エラー（`MODEL_NOT_FIT`）を返す

### 22-B: `GroupHoldoutInnerValid` の validation 割当ポリシーを仕様準拠化（H-0025）

1. `lizyml/training/inner_valid.py`:
   - `GroupHoldoutInnerValid` の validation group 選定を「shuffle 後の末尾」ではなく「入力順の末尾 group（time/order aware）」に変更
   - group overlap 禁止は維持
2. テスト:
   - 入力順に対して末尾 group が validation に入ること
   - 既存の overlap 禁止・再現性テストを維持

### 22-C: `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` の範囲検証を追加（H-0025）

1. `lizyml/config/schema.py`:
   - `min_data_in_leaf_ratio` は `0 < ratio < 1` を検証
   - `min_data_in_bin_ratio` は `0 < ratio < 1` を検証
2. テスト:
   - `0`, `1`, 負値, `1超` が `CONFIG_INVALID` になること
   - 正常値（例: `0.01`, `0.2`）が受理されること

### 22-D: Notebook のスマートパラメーター確認セルを網羅化（H-0025）

1. `notebooks/tutorial_regression_lgbm.ipynb`:
   - 表示項目に `min_data_in_bin_ratio`, `feature_weights`, `balanced` を追加
   - 解決結果として `min_data_in_bin`, `feature_weights`, `scale_pos_weight`/`sample_weight` の確認例を追加
2. `notebooks/tutorial_binary_lgbm.ipynb`, `notebooks/tutorial_multiclass_lgbm.ipynb`:
   - 21-C 方針に合わせて同等のパラメーター確認セルを追記（必要最小限）

### テスト

- `tests/test_plots/test_residuals.py`: load 後に `residuals()` / `residuals_plot()` が実行可能
- `tests/test_explain/test_explain.py`: load 後に `importance(kind="shap")` が実行可能
- `tests/test_plots/test_classification_plots.py`: load 後に `roc_curve_plot()` が実行可能
- `tests/test_evaluation/test_confusion.py`: load 後に `confusion_matrix()` が実行可能
- `tests/test_plots/test_calibration_plots.py`: load 後に `calibration_plot()` / `probability_histogram_plot()` が実行可能
- `tests/test_persistence/test_persistence.py`: legacy artifact（`analysis_context` なし）で load は成功し、診断 API は明示的エラー
- `tests/test_training/test_inner_valid_extensions.py`: group 末尾割当テストを追加
- `tests/test_config/test_lgbm_smart_params.py`: ratio 範囲バリデーションケースを追加
- Notebook は実行確認（最低限、追加セルの静的整合）

### DoD

- `Model.load()` 後に 7 つの診断 API（residuals / residuals_plot / shap importance / roc / confusion / calibration / probability histogram）が利用可能
- load 後診断 API の出力契約が fit 後と整合する
- `analysis_context` を含む artifact を export/load できる
- 既存 artifact（`analysis_context` なし）は load 互換を維持し、診断 API 呼び出し時は再 export を促す明示的エラー
- `GroupHoldoutInnerValid` が入力順末尾 group を validation に割り当てる
- `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` の範囲外値が `CONFIG_INVALID`
- Regression/Binary/Multiclass Notebook にスマートパラメーター確認セルが揃う
- 追加/更新テストが通過する

### 22-E: legacy artifact での診断 API ガード統一（H-0026 Follow-up）

1. `lizyml/core/model.py`:
   - `probability_histogram_plot()` で `analysis_context` 未復元（`self._y is None`）時に `MODEL_NOT_FIT` を返す。
   - `roc_curve_plot()` / `confusion_matrix()` / `calibration_plot()` / `residuals()` / `importance(kind="shap")` と同一の「再 export を促す」エラーメッセージ方針に統一する。
2. `tests/test_persistence/test_persistence.py`:
   - legacy artifact（`analysis_context` なし）で、少なくとも `probability_histogram_plot()` が `MODEL_NOT_FIT` になることを追加検証する。
   - 可能なら 7 診断 API 全体を table-driven で検証し、`predict()` / `evaluate()` は引き続き成功することを同時に担保する。

### 22-F: Notebook の解決結果確認セルの不足補完（H-0025 Follow-up）

1. `notebooks/tutorial_regression_lgbm.ipynb`:
   - `min_data_in_bin` に加えて、解決後 `feature_weights` の確認例を明示する（利用時のみ表示）。
2. `notebooks/tutorial_binary_lgbm.ipynb`:
   - `scale_pos_weight` の確認例を維持しつつ、解決後 `feature_weights` の確認例を追加する。
3. `notebooks/tutorial_multiclass_lgbm.ipynb`:
   - `sample_weight` の確認例（クラス頻度由来の重み確認）を追加する。
   - 解決後 `feature_weights` の確認例を追加する。
4. テスト:
   - Notebook の静的整合テスト（`nbformat` ベース）を追加し、3本で `min_data_in_bin_ratio` / `feature_weights` / `balanced` と task 別確認項目（binary: `scale_pos_weight`, multiclass: `sample_weight`）が存在することを検証する。

### 22-E/22-F 追加 DoD

- legacy artifact（`analysis_context` なし）で診断 API を呼ぶと、仕様どおり `MODEL_NOT_FIT` + 再 export ガイダンスを返す。
- legacy artifact でも `predict()` / `evaluate()` は互換性を維持する。
- 3 本の Notebook で smart parameter の「設定値」だけでなく「解決後値」の確認例が揃う（`min_data_in_bin`, `feature_weights`, binary は `scale_pos_weight`, multiclass は `sample_weight`）。
- 追加した Notebook 整合テストが通過する。

### 22-H: Config デフォルト値修正（H-0027）

1. `lizyml/config/schema.py`:
   - `LGBMConfig.min_data_in_leaf_ratio` のデフォルトを `None` → `0.01` に変更。
   - `LGBMConfig.min_data_in_bin_ratio` のデフォルトを `None` → `0.01` に変更。
   - `LGBMConfig.balanced` のデフォルトを `False` → `None` に変更（`None` はタスク依存自動解決: regression→False, binary/multiclass→True）。
2. BLUEPRINT §5.4 に Config Reference（全キー一覧）を追加（完了済み）。
3. テスト:
   - 既存テストのうちデフォルト値に依存するケースを更新。
   - デフォルト値が正しく適用されることを検証するテストを追加。

### 22-I: Tuning 探索状況の可視化（H-0028）

1. `lizyml/plots/tuning.py`（新規）:
   - `plot_tuning_history(tuning_result)` 関数を実装。
   - X 軸 = trial 番号、Y 軸 = スコア値。
   - 完了/枝刈り/失敗を色分け（marker color）。
   - 最良スコアの累積推移ラインを重ね描き。
   - Plotly optional dependency。
2. `lizyml/core/model.py`:
   - `tuning_plot()` メソッドを追加（委譲のみ）。
   - `tune()` 未実行時は `MODEL_NOT_FIT`。
3. テスト:
   - `tune()` 後に `tuning_plot()` が Figure を返す。
   - `tune()` 未実行時に `MODEL_NOT_FIT`。
   - Plotly 未インストール時に `OPTIONAL_DEP_MISSING`。

### 22-J: `Model.fit_result` プロパティ（H-0029）

1. `lizyml/core/model.py`:
   - `@property fit_result` を追加（`self._require_fit()` を返すだけ）。
2. テスト:
   - `fit()` 後に `model.fit_result` が `FitResult` を返す。
   - `fit()` 未実行時に `MODEL_NOT_FIT`。

### 22-K: Tutorial Notebook で全 Config 項目指定

1. `notebooks/tutorial_regression_tuning_lgbm.ipynb` を更新:
   - README の Config Reference に記載されている全キーを Config に明示指定する。
   - 各セクション（data, features, split, model, training, tuning, evaluation）を網羅する。
   - スマートパラメーター（auto_num_leaves, ratio系, feature_weights, balanced）を含む。
   - 学習後に以下を確認するセルを追加:
     - LGBMConfig Smart Parameters の設定値
     - Training defaults
     - 解決後 LightGBM ネイティブパラメーター（Fold 0 の booster params）
     - feature_weights の解決結果
     - Best iteration per fold
2. テスト:
   - Notebook が `nbconvert --execute` で正常に実行可能。

### 22-H〜K 追加 DoD

- Config デフォルト値が README/BLUEPRINT と一致する（`min_data_in_leaf_ratio=0.01`, `min_data_in_bin_ratio=0.01`, `balanced=None`（task auto: regression→False, binary/multiclass→True））。
- `model.tuning_plot()` が Plotly Figure を返す。
- `model.fit_result` が FitResult を返す。
- Tutorial Notebook が全 Config 項目を網羅し、解決後パラメーターの確認セルが含まれる。
- 追加/更新テストが通過する。

---

### 22-G: Notebook Config 手本化・実行エラー修正（22-F Follow-up）

1. **Binary / Multiclass notebook 実行エラー修正**:
   - Config で `params.num_leaves=31` を指定していたが `auto_num_leaves=True`（デフォルト）と競合し `CONFIG_INVALID` で実行不可。
   - `params` から `num_leaves`, `n_estimators`, `learning_rate`, `min_child_samples` を削除し、Phase 21 デフォルトプロファイルに委ねる形に修正。
2. **3 本の Notebook Config を Tutorial 手本として整備**:
   - 主要スマートパラメーター（`min_data_in_leaf_ratio`, `min_data_in_bin_ratio`）を明示指定。
   - Binary / Multiclass では `balanced: True` を明示し、解決後の `scale_pos_weight` / `sample_weight` を確認セルで表示。
   - デフォルトで十分なパラメーター（`n_estimators`, `learning_rate`, `max_depth`, early stopping 等）はコメントで明記しつつ省略。
3. **Multiclass notebook の `sample_weight` 表示修正**:
   - adapter 属性ではなく `compute_sample_weight("balanced", y)` で再計算して表示する方式に変更（sample_weight はモデルに保存されないため）。
4. **`min_data_in_bin_ratio` のデフォルト値について**:
   - `LGBMConfig.min_data_in_bin_ratio` のデフォルトは `None`（未指定）。
   - Notebook では Tutorial 手本として `0.001` を明示指定。
5. **3 本の Notebook が `nbconvert --execute` で正常に実行可能なことを確認済み**。

### 22-G DoD

- Binary / Multiclass notebook が `CONFIG_INVALID` なく正常に実行可能。
- 3 本の Config が Tutorial 手本として主要スマートパラメーターを明示指定している。
- Multiclass notebook で `sample_weight` の解決後値が正しく表示される。
- 静的整合テスト（24 cases）が全パスする。
- 全テスト 735 パス、lint / mypy 全クリア。

### Phase 22 残タスク（22-G / 22-K Follow-up）— 完了済み ✅

**SKILL:** skills/spec-update/SKILL.md, skills/testing/SKILL.md

1. ~~`notebooks/tutorial_regression_tuning_lgbm.ipynb`: feature_weights (resolved) 確認セル追加~~ → 完了済み。
2. ~~Notebook 実行テスト（`tests/test_notebooks/test_notebook_execution.py`）追加~~ → 完了済み（4 本対象、`@pytest.mark.slow`）。
3. ~~CI 連携: Notebook 実行テストを CI に組み込み~~ → 完了済み。

### Phase 22 残タスク DoD — 達成済み ✅

- `tutorial_regression_tuning_lgbm.ipynb` に `feature_weights (resolved)` の確認セルが追加され、22-K 要件を満たす。
- 22-G/22-K 対象 Notebook の `nbconvert --execute` テストが追加され、ローカルでパスする。
- CI 上でも Notebook 実行テストが実行され、失敗時に PR をブロックできる。

### 22-L: Notebook パラメーター明示化 + fit_result 統一

全 4 本の Tutorial Notebook に対し、LightGBM パラメーター 14 項目を明示指定して手本化する。

**対象パラメーター（14 項目）:**
`objective`, `n_estimators`, `learning_rate`, `max_depth`, `num_leaves_ratio`, `min_data_in_leaf_ratio`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l2`, `max_bin`, `min_data_in_bin_ratio`, `early_stopping_rounds`, `validation_ratio`

1. **Tuning Notebook** (`tutorial_regression_tuning_lgbm.ipynb`):
   - §3 Config: `tuning.optuna.space` に 14 パラメーターの探索範囲を明示定義する（空 `{}` → 明示 space）。
     - model category: `objective`, `n_estimators`, `learning_rate`, `max_depth`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l2`, `max_bin`
     - smart category: `num_leaves_ratio`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`
     - training category: `early_stopping_rounds`, `validation_ratio`
   - §3 Config: `model.params` からは tuning で探索するパラメーターを除外（探索と固定値の重複を防ぐ）。
   - §5.1 LightGBM Parameters: `fit_result.models[0]` を `model.fit_result.models[0]` に統一する。

2. **非 Tuning Notebook** 3 本（`tutorial_regression_lgbm.ipynb`, `tutorial_binary_lgbm.ipynb`, `tutorial_multiclass_lgbm.ipynb`):
   - §3 Config: `model.params` に 14 パラメーターを明示設定する。
     - `model.params`: `objective`, `n_estimators`, `learning_rate`, `max_depth`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l2`, `max_bin`
     - Config smart params: `num_leaves_ratio`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`（既存維持）
     - Config training: `early_stopping.rounds`, `early_stopping.validation_ratio`（既存維持）
   - §5.1 相当の確認セル: `fit_result` ローカル変数 → `model.fit_result` プロパティに統一する（該当箇所がある場合）。

3. **テスト**:
   - 4 本の `nbconvert --execute` 実行テストがパスすること。
   - 既存の静的整合テストがパスすること。

### 22-L DoD

- Tuning Notebook: 14 パラメーターの search space が明示定義されている。
- 非 Tuning Notebook: 14 パラメーターが `model.params` + smart params + training で明示されている。
- 全 Notebook で `model.fit_result` プロパティ経由のアクセスに統一されている。
- 4 本が `nbconvert --execute` で正常に実行可能。

### 22-M: `model.params_table()` — 解決済みパラメーターテーブル API（H-0035）

1. **`lizyml/core/model.py`**:
   - `params_table()` メソッドを追加する。`_require_fit()` で fit 済みチェック。
   - Config smart params（`auto_num_leaves`, `num_leaves_ratio`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`, `balanced`, `feature_weights`, `early_stopping.rounds`, `validation_ratio`）を収集。
   - fold 0 の booster から解決済みネイティブパラメーター（`objective`, `num_leaves`, `min_data_in_leaf` 等）を収集。
   - fold ごとの `best_iteration` を末尾に追加。
   - 単一列 `value` の `pd.DataFrame`（index: `parameter`）を返す。

2. **Notebook 更新**:
   - 全 4 本の「§4.1 LightGBM Parameters」セルを `model.params_table()` 1 行に置き換える。

3. **テスト**:
   - `model.params_table()` が `pd.DataFrame` を返すこと。
   - Config smart params と resolved booster params が含まれること。
   - `fit()` 未実行時に `MODEL_NOT_FIT` が送出されること。
   - 4 本の `nbconvert --execute` 実行テストがパスすること。

### 22-M DoD

- `model.params_table()` が `pd.DataFrame`（index: `parameter`, 列: `value`）を返す。
- Config smart params + resolved booster params + fold ごとの `best_iteration` が含まれる。
- `fit()` 未実行時に `MODEL_NOT_FIT` を送出する。
- Notebook の「4.1」セルが `model.params_table()` に置き換えられている。
- 4 本が `nbconvert --execute` で正常に実行可能。

### 22-N: Smart Parameter の n_rows 基準を inner train サイズに変更（H-0036）

1. **`lizyml/estimators/lgbm.py`**:
   - `resolve_smart_params()` / `resolve_smart_params_from_dict()` に渡す `n_rows` の定義を「inner valid 分割後の学習データ行数」に変更。
   - `auto_num_leaves` / `num_leaves_ratio` は `max_depth` のみに依存するため変更不要。

2. **`lizyml/core/model.py`**:
   - `Model.fit()` から smart param の一括解決を除去。
   - `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` の解決ロジックを CVTrainer に委譲。
   - `feature_weights` / `balanced` / `auto_num_leaves` は n_rows 非依存のため、Model.fit() での解決を維持。

3. **`lizyml/training/cv_trainer.py`**:
   - fold ループ内で inner_valid 分割後の行数を取得し、ratio パラメーターを解決。
   - estimator_factory の呼び出し前に解決済みパラメーターを estimator に適用する仕組みを追加。

4. **`lizyml/tuning/tuner.py`**:
   - trial 内の smart param 解決で `self.n_rows` の代わりに CVTrainer 内部で fold ごとに解決する方式に移行。

5. **テスト**:
   - 5-fold + validation_ratio=0.1 で `min_data_in_leaf_ratio=0.01` → 解決値が inner train サイズ × 0.01 であることを検証。
   - early stopping 無効時は outer fold サイズ基準であることを検証。
   - Tuner trial 内でも同一ロジックが適用されることを検証。
   - 既存テストの回帰確認（seed 固定テストの期待値更新が必要な場合あり）。

### 22-N DoD

- `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` の `n_rows` が inner train サイズ基準で解決される。
- early stopping 無効時は outer fold サイズ基準。
- fold ごとに異なる inner train サイズの場合、各 fold で個別に解決される。
- Tuner trial 内でも同一ロジックが適用される。
- 既存テスト全件パス（期待値の更新を含む）。

### 22-O: 監査乖離クローズ — ドキュメント整合 + Notebook/テスト補完（H-0037）

1. **BLUEPRINT §5.3 balanced 記述統一**（ドキュメント修正のみ）:
   - `balanced: bool = False` → `balanced: bool | None = None`（タスク依存自動解決）に修正。§5.4 Config Reference と一致させる。
2. **BLUEPRINT §5.2/§5.3 LGBMConfig 例の微修正**（ドキュメント修正のみ）:
   - `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` のデフォルト値（`0.01`）を §5.3 に明記。
   - §5.2 Config 例の `balanced` コメントを `None`（タスク依存自動）に修正。
3. **`notebooks/tutorial_regression_tuning_lgbm.ipynb`**:
   - feature_weights (resolved) の確認セルを追加する（設定時のみ表示する条件付き）。
   - Fold 0 の学習済みモデルから取得し、Config 設定値との対比を明示する。
4. **`tests/test_notebooks/test_notebook_cells.py`**:
   - Tuning Notebook に feature_weights の「解決後値確認」セル（`feature_weights` キーワードを含むコードセル）が存在することを検証するテストを追加。

### 22-O DoD

- BLUEPRINT §5.3 の balanced デフォルト記述が `bool | None = None` + タスク依存自動解決の説明になっている。
- BLUEPRINT §5.2 Config 例の balanced コメントが `None` になっている。
- BLUEPRINT §5.3 に `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` のデフォルト値 `0.01` が明記されている。
- `tutorial_regression_tuning_lgbm.ipynb` に feature_weights (resolved) の確認セルが追加されている。
- Notebook 静的テストが feature_weights 解決後値確認セルの存在を検証する。
- 全テストがパスする。

---

## Phase 23: Calibration 強化・時系列拡張

**HISTORY:** H-0030, H-0031, H-0032, H-0033, H-0034, H-0040
**SKILL:** skills/calibration/SKILL.md, skills/splitters-and-splitplan/SKILL.md, skills/plots/SKILL.md, skills/exceptions-and-logging/SKILL.md
**依存:** Phase 22

### 背景

Requirements Audit で検出された未実装項目（Beta Calibration、PurgedTimeSeries/GroupTimeSeries Config 接続）と、新規要求（Calibration raw score 入力、時系列 fold 期間表示、Logging 統一、時系列 Tutorial Notebook）を Phase 23 でまとめて対応する。

### 23-A: Calibration に生スコア（raw score）を渡す（H-0030）

1. `lizyml/estimators/base.py`:
   - `predict_raw(X)` 抽象メソッドを追加（sigmoid/softmax 適用前の生スコアを返す）。
2. `lizyml/estimators/lgbm.py`:
   - `predict_raw(X)` を実装。LightGBM の `predict(raw_score=True)` を使用。
3. `lizyml/calibration/base.py`:
   - `fit(oof_scores, y)` / `predict(scores)` のドキュメントを更新（入力は生スコア）。
4. `lizyml/calibration/platt.py`, `isotonic.py`:
   - 生スコア入力に対応（Platt は LogisticRegression で自然に対応、Isotonic は入力スケールのみ注意）。
5. `lizyml/calibration/cross_fit.py`:
   - OOF 生スコアを校正器に渡すよう変更。
6. `lizyml/training/cv_trainer.py`:
   - Calibration 有効時に `predict_raw` で OOF 生スコアを生成。Calibration 無効時は従来の `predict_proba` を維持。
7. テスト:
   - raw score 入力で Platt / Isotonic が動作する。
   - Calibration 無効時は `predict_proba` パスが維持される。

### 23-B: Beta Calibration（H-0031）

1. `lizyml/calibration/beta.py`（新規）:
   - `BetaCalibrator(BaseCalibratorAdapter)` を実装。
   - 3 パラメーターモデル `a * log(s) + b * log(1-s) + c` を `scipy.optimize.minimize` で最適化。
   - 生スコア入力に対応（23-A に依存）。
2. `lizyml/calibration/registry.py`:
   - `_NOT_IMPLEMENTED` から `"beta"` を削除し、`BetaCalibrator` を登録。
3. テスト:
   - `method="beta"` で cross-fit 校正が動作する。
   - OOF-only / リーク禁止契約を満たす。

### 23-C: PurgedTimeSeries / GroupTimeSeries の Config・Model 接続（H-0032）

1. `lizyml/config/schema.py`:
   - `PurgedTimeSeriesConfig`（`method: Literal["purged_time_series"]`, `n_splits`, `purge_gap`, `embargo_pct`）を追加。
   - `GroupTimeSeriesConfig`（`method: Literal["group_time_series"]`, `n_splits`）を追加。
   - `SplitConfig` の Union に上記を追加。
2. `lizyml/config/loader.py`:
   - `_SPLIT_METHOD_ALIASES` に `purged-time-series`, `purgedtimeseries`, `group-time-series`, `grouptimeseries` を追加。
3. `lizyml/core/model.py`:
   - `_build_splitter()` に `purged_time_series` / `group_time_series` のルーティングを追加。
   - `_build_inner_valid()` に対応する自動解決を追加（`purged_time_series` → `time_holdout`, `group_time_series` → `group_holdout`）。
4. テスト:
   - `split.method: "purged_time_series"` / `"group_time_series"` で fit が動作する。
   - 正規化エイリアスが機能する。
   - InnerValid 自動解決テストを追加。

### 23-D: 時系列 fold 期間情報の表示（H-0033）

1. `lizyml/core/types/fit_result.py`:
   - `SplitIndices` に `time_range: list[dict] | None = None` フィールドを追加（fold ごとの `train_start/train_end/valid_start/valid_end`）。
2. `lizyml/training/cv_trainer.py`:
   - 時系列分割時に `time_col` の min/max を fold ごとに記録して `time_range` に格納。
3. `lizyml/core/model.py`:
   - `split_summary()` メソッドを追加（委譲のみ）。
   - fold ごとの `train_size`, `valid_size` を DataFrame で返す。時系列分割時は `train_start/train_end/valid_start/valid_end` も含む。
4. テスト:
   - 時系列分割で `split_summary()` が期間情報を含む DataFrame を返す。
   - 非時系列でもサイズ情報は返す。

### 23-E: 時系列 Tutorial Notebook

1. `notebooks/tutorial_time_series_lgbm.ipynb`（新規）:
   - 公開データセット（5個以上の特徴量を含む時系列データ）をダウンロードして使用。
   - `split.method: "time_series"` で CV を実行。
   - `split_summary()` で fold 期間情報を表示。
   - 学習・評価・予測の一連のワークフローを実演。
   - 解決後パラメーターの確認セルを含む。
2. テスト:
   - Notebook が `nbconvert --execute` で正常に実行可能。

### 23-F: Logging 出力先の統一（H-0034）

1. `lizyml/core/logging.py`:
   - `setup_output_dir(output_dir, run_id)` 関数を追加。`{output_dir}/{run_id}/` ディレクトリを作成。
   - ファイルハンドラをセットアップし、構造化ログをファイルに出力。
2. `lizyml/core/model.py`:
   - `Model` コンストラクタに `output_dir: str | Path | None = None` パラメーターを追加。
   - `output_dir` 指定時、fit/tune/export のログを `{output_dir}/{run_id}/` に保存。
3. テスト:
   - `output_dir` 指定時にディレクトリとログファイルが作成される。
   - 未指定時は既存動作を維持。

### Phase 23 DoD

- `EstimatorAdapter.predict_raw()` が生スコアを返し、Calibration 時に使用される。
- `method="beta"` で校正が動作する。
- `split.method: "purged_time_series"` / `"group_time_series"` で CV が動作する。
- `model.split_summary()` が fold 情報を DataFrame で返す（時系列時は期間情報含む）。
- 時系列 Tutorial Notebook が公開データで実行可能。
- `output_dir` 指定時にログが統一ディレクトリに出力される。
- 追加/更新テストが通過する。

### Phase 23 残タスク（Requirements Audit 2026-03-06）

**HISTORY:** H-0038, H-0039  
**SKILL:** skills/spec-update/SKILL.md, skills/history-proposals/SKILL.md, skills/splitters-and-splitplan/SKILL.md, skills/calibration/SKILL.md, skills/dev-environment/SKILL.md, skills/testing/SKILL.md

監査結果により 23-C / 23-E / 23-F / 23-G に未達があるため、以下を追加対応する。

1. **23-C Follow-up（BLUEPRINT優先で契約是正）**
   - `purged_time_series` の Config 契約を BLUEPRINT §5.4 準拠（`purge_gap`, `embargo_pct`）に統一する。
   - `H-0040` により最終契約は `embargo`（`embargo_pct` は移行期間の互換キー）へ更新する。
   - legacy key（`purge_window`, `gap`）は移行期間のみ警告付き受理とし、正規キーへ正規化する。
   - split 境界（purge + embargo）のテストを追加し、リーク防止を検証する。
2. **23-E Follow-up（Notebook ワークフロー完了）**
   - `tutorial_time_series_lgbm.ipynb` に `predict` 実演セルを追加し、「学習・評価・予測」の一連フローを満たす。
   - `tests/test_notebooks/test_notebook_execution.py` の `nbconvert --execute` 対象に `tutorial_time_series_lgbm.ipynb` を追加する。
3. **23-F Follow-up（output_dir 統一の完了）**
   - `output_dir` を Config からも指定可能にし、優先順位を `constructor > config > 未指定` で固定する。
   - `fit` だけでなく `tune` / `export` でも `{output_dir}/{run_id}/` へログを統一出力する。
   - 回帰テストで「未指定時の既存挙動維持」を担保する。
4. **23-G Follow-up（Isotonic 実装の BLUEPRINT 準拠）**
   - `lizyml/calibration/isotonic.py` を、BLUEPRINT §12.2 の方針（LGBM 単調制約利用）に沿う実装へ置換する。
   - calibrator IF（`fit(oof_scores, y)` / `predict(scores)`）とリーク防止境界（OOF-only + cross-fit 評価）は維持する。
   - `tests/test_calibration/` に Isotonic 実装方式の回帰テスト（単調制約設定・出力レンジ・E2E）を追加する。

### Phase 23 残タスク（監査アップデート 2026-03-06）

再監査の結果、未達として残っているのは **23-C / 23-F** のみ。23-E / 23-G は完了扱いとする。

1. **23-C Follow-up 2（embargo 実動作の実装）**
   - `PurgedTimeSeriesSplitter` で `embargo`（旧: `embargo_pct`）を「設定値の保持」だけでなく、split 境界計算に実際に反映する。
   - purge + embargo の境界で train/valid のリーク防止が満たされることをテストで固定する。
   - `tests/test_splitters/test_splitters.py` または `tests/test_e2e/test_time_series_splits.py` に、`embargo > 0` のときの境界検証を追加する。
2. **23-F Follow-up 2（export 経路の output_dir 統一）**
   - `Model.export()` 経路でも `{output_dir}/{run_id}/` を作成し、`fit/tune/export` で出力先契約を統一する。
   - constructor/config の優先順位（`constructor > config > 未指定`）を export 経路でも崩さない。
   - `tests/test_core/test_logging_output.py` に export 経路の回帰テストを追加する。

### Phase 23 残タスク DoD

- 23-C: BLUEPRINT §5.4 のキー契約（`purge_gap`, `embargo`）と実装が一致し、`embargo_pct` を含む legacy key は警告付き互換として扱われる。
- 23-E: 時系列 Notebook が `fit -> evaluate -> predict` を実演し、`nbconvert --execute` テストに含まれてパスする。
- 23-F: Config/コンストラクタ双方の `output_dir` が仕様どおりに解決され、`fit/tune/export` で run ディレクトリにログが出力される。
- 23-G: `isotonic` が BLUEPRINT §12.2 の実装方針（LGBM 単調制約利用）を満たし、既存 calibration 契約テストが回帰しない。
- 追加テストが pass し、既存 E2E/Notebook テストに回帰がない。
- 23-C（Follow-up 2）: `embargo`（旧: `embargo_pct`）が split 計算に反映され、`embargo > 0` で embargo 境界のリーク防止テストが pass する。
- 23-F（Follow-up 2）: export 経路でも `output_dir` 契約が成立し、`fit/tune/export` の全経路で run ディレクトリ作成テストが pass する。

### Phase 23 残タスク（TimeSeries CV 方針更新 2026-03-07）

**HISTORY:** H-0040  
**SKILL:** skills/spec-update/SKILL.md, skills/history-proposals/SKILL.md, skills/config-and-specs/SKILL.md, skills/splitters-and-splitplan/SKILL.md, skills/training-cv-and-inner-valid/SKILL.md, skills/testing/SKILL.md

1. ~~**23-H: time_col 基準の分割統一**~~ ✅
   - `time_series` / `purged_time_series` / `group_time_series` は `data.time_col` 必須にする。
   - split 直前に `time_col` 昇順へ並べた index で分割する契約に統一する。
   - `time_col` 未指定時は `CONFIG_INVALID` を返す。
2. ~~**23-I: TimeSeries 共通パラメーター追加**~~ ✅
   - 3 メソッド共通で `train_size_max` / `test_size_max` を Config と splitter 契約に追加する。
   - `time_series` / `group_time_series` の `gap` と `purged_time_series` の `purge_gap` の適用境界をテストで固定する。
3. ~~**23-J: embargo 命名への移行**~~ ✅
   - `purged_time_series` の正式キーを `embargo` に変更する。
   - `embargo_pct` は移行期間のみ警告付き互換として受理し、`embargo` に正規化する。
   - notebook / ドキュメント / テストのキー名を `embargo` に統一する。

### Phase 23 残タスク（TimeSeries CV 方針更新）DoD

- 23-H: 3 メソッドで `data.time_col` 未指定時に明示エラーになり、`time_col` 昇順分割の再現性テストが pass する。
- 23-I: `train_size_max` / `test_size_max` が 3 メソッドで有効に動作し、各 split 境界テストが pass する。
- 23-J: `embargo` が正式キーとして動作し、`embargo_pct` は警告付き互換で同等動作、移行テストが pass する。
- 既存の leakage 防止・split_summary・Notebook 実行テストに回帰がない。

---

## Phase 24: LGBMAdapter Booster API 移行

**HISTORY:** H-0041
**SKILL:** skills/add-estimator-adapter/SKILL.md
**依存:** Phase 23

### 背景

LightGBM の sklearn wrapper（`LGBMRegressor` / `LGBMClassifier`）に、`early_stopping` callback 併用時に `model_to_string()` が空文字列を返す間欠バグが存在する（microsoft/LightGBM#7186）。Booster API（`lgb.train()`）では `keep_training_booster=True` によりこのバグを回避できる。

### 24-A: LGBMAdapter.fit() を Booster API に移行

1. `lizyml/estimators/lgbm.py`:
   - `fit()`: `LGBMRegressor(**params).fit(...)` → `lgb.Dataset` 構築 + `lgb.train(params, train_set, valid_sets=..., callbacks=..., keep_training_booster=True)` に置換。
   - `_model` の型を `LGBMRegressor | LGBMClassifier | None` → `lgb.Booster | None` に変更。
   - `_build_params()`: sklearn 固有パラメーター名（`n_estimators`, `random_state`, `verbose`）を Booster API 名（`num_boost_round` は引数、`seed`、`verbosity`）に変換。`num_boost_round` はパラメーター dict から分離して `lgb.train()` の引数として渡す。
   - `best_iteration`: `booster.best_iteration` から取得。
   - `evals_result` dict をコールバック経由で収集（`lgb.record_evaluation(evals_result)` を使用）。
2. テスト:
   - regression / binary / multiclass の全タスクで `lgb.train()` 経由の学習が動作する。
   - `best_iteration` が正しく取得される。
   - early stopping が inner valid あり / なしの両方で動作する。

### 24-B: predict / predict_proba / predict_raw の適応

1. `lizyml/estimators/lgbm.py`:
   - `predict()`: `booster.predict(X)` を使用。regression はそのまま返却。
   - `predict_proba()`: `booster.predict(X)` を使用。binary は `(n,)` → `np.column_stack([1-p, p])` で `(n, 2)` に変換。multiclass は `(n, k)` をそのまま返却。
   - `predict_raw()`: `booster.predict(X, raw_score=True)` を直接使用（`booster_` 経由のアクセスが不要になる）。
   - `_require_fitted()` の戻り値型を `lgb.Booster` に変更。
2. テスト:
   - 全タスクで `predict()` / `predict_proba()` / `predict_raw()` の出力 shape が移行前と同一。
   - binary `predict_proba()` が `(n, 2)` を返す。

### 24-C: importance / get_native_model の適応

1. `lizyml/estimators/lgbm.py`:
   - `importance()`: `booster.feature_importance(importance_type=...)` を直接使用。
   - `get_native_model()`: 戻り値型を `lgb.Booster` に変更。
2. `lizyml/core/model.py`:
   - `params_table()`: `.get_native_model().booster_` → `.get_native_model()` に変更（Booster を直接使用）。
3. テスト:
   - `importance(kind="split")` / `importance(kind="gain")` が動作する。
   - `get_native_model()` が `lgb.Booster` インスタンスを返す。

### 24-D: cv_trainer / refit_trainer の学習履歴適応

1. `lizyml/training/cv_trainer.py`:
   - `native.evals_result_` → `evals_result` dict（`LGBMAdapter` から取得する方法に変更）。
   - `LGBMAdapter` に `eval_result` プロパティを追加し、`lgb.record_evaluation()` で収集した結果を返す。
2. `lizyml/training/refit_trainer.py`:
   - 同上。
3. テスト:
   - 学習曲線プロットが移行前と同様に動作する。

### 24-E: SHAP / Persistence の確認と適応

1. `lizyml/explain/shap_explainer.py`:
   - `TreeExplainer` が `lgb.Booster` を直接受け取ることを確認。必要に応じて型ヒントを調整。
2. `lizyml/persistence/exporter.py` / `loader.py`:
   - `lgb.Booster` の joblib シリアライズが動作することを確認。
   - `format_version=1` の既存保存モデルのロード互換を確認（旧形式の sklearn wrapper ベースのモデルも復元可能にする）。
3. テスト:
   - SHAP が Booster 直接入力で動作する。
   - export / load のラウンドトリップが動作する。
   - 既存の persistence テストに回帰がない。

### 24-F: テスト更新と回帰確認

1. `tests/test_estimators/test_estimators.py`:
   - `get_native_model()` の戻り値型チェックを `lgb.Booster` に更新。
   - `.booster_` 経由のアクセスを直接 Booster アクセスに更新。
2. `tests/test_e2e/test_model_facade.py`:
   - `params_table()` テストの `.booster_` アクセスを更新。
3. 全テスト（782件）の pass を確認。
4. notebook テスト（`tutorial_regression_tuning_lgbm.ipynb`）の間欠エラー解消を確認（複数回実行）。

### DoD

- 24-A: `lgb.train()` 経由で regression / binary / multiclass が学習でき、`best_iteration` / early stopping が動作する。
- 24-B: `predict()` / `predict_proba()` / `predict_raw()` の出力 shape が移行前と同一であり、全タスクのテストが pass する。
- 24-C: `importance()` / `get_native_model()` が Booster 直接で動作し、`params_table()` が回帰しない。
- 24-D: `eval_result` による学習履歴収集が cv_trainer / refit_trainer で動作し、学習曲線プロットが回帰しない。
- 24-E: SHAP / export / load が回帰しない。
- 24-F: 全テスト（861件）が pass し、notebook テストの間欠エラーが解消される（notebook 5/5 pass 確認済み）。
- 24-G: `_build_params()` が Booster 名称へ完全正規化し、対応ユニットテストが pass する。
- 24-H: テスト件数が実測値で記録される。
- 24-I: `H-0041` の Status が `accepted` に更新される。

### Phase 24 残タスク（Requirements Audit 2026-03-07）

**HISTORY:** H-0041
**SKILL:** skills/add-estimator-adapter/SKILL.md, skills/testing/SKILL.md, skills/spec-update/SKILL.md

1. **24-G: Booster パラメーター名正規化の補完**
   - `lizyml/estimators/lgbm.py::_build_params()` で `params.random_state` / `params.verbose` を `seed` / `verbosity` に正規化する。
   - `seed` / `verbosity` が同時指定された場合の優先順位を実装・テストで固定する。
   - `_build_params()` 後に Booster 非推奨キー（`random_state`, `verbose`, `n_estimators`）が残らないことをテストで担保する。
2. **24-H: 完了判定用テスト基準の更新**
   - Phase 24 DoD の「全テスト（782件）」を現行件数に更新し、実測 pass 結果で固定する。
   - `tests/test_notebooks/test_notebook_execution.py -k tuning` を複数回実行し、間欠エラー非再発を記録する。
3. **24-I: 仕様記録の状態整合**
   - Phase 24 DoD 達成後に `HISTORY.md` の `H-0041` を `proposed` から `accepted` に更新する。

### Phase 24 残タスク DoD

- 24-G: `_build_params()` が Booster 名称へ完全正規化し、対応ユニットテストが pass する。
- 24-H: 現行テスト件数ベースのフルテストが pass し、tuning notebook 実行テストの複数回 pass を確認できる。
- 24-I: `H-0041` の Status が `accepted` になり、PLAN/BLUEPRINT/HISTORY の記載が矛盾しない。

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

## Phase 25: Model Facade 分割・テスト基盤改善

**HISTORY:** H-0042, H-0043
**SKILL:** skills/testing/SKILL.md, skills/dev-environment/SKILL.md
**依存:** Phase 24

### 背景

品質評価の結果、以下の改善点が特定された：
- `core/model.py` が 1,451行に肥大化しており、mixin 分割で可読性・保守性を改善できる（H-0042）。
- テストスイートのヘルパー関数が8+ファイルで重複定義されており、conftest への集約と parametrize 活用で DRY 化できる（H-0043）。
- CI が main PR のみで実行されており、develop PR でも品質ゲートを回すべき（H-0043）。

### 25-A: Model Facade の Mixin 分割

1. `lizyml/core/_model_plots.py` を新規作成し、`ModelPlotsMixin` に plot 系メソッド（8メソッド）を移動する。
   - `residuals_plot()`, `roc_curve_plot()`, `calibration_plot()`, `probability_histogram_plot()`, `importance_plot()`, `plot_learning_curve()`, `plot_oof_distribution()`, `tuning_plot()`
2. `lizyml/core/_model_tables.py` を新規作成し、`ModelTablesMixin` に table/accessor 系メソッド（7メソッド）を移動する。
   - `evaluate_table()`, `residuals()`, `confusion_matrix()`, `importance()`, `params_table()`, `split_summary()`, `tuning_table()`
3. `lizyml/core/_model_persistence.py` を新規作成し、`ModelPersistenceMixin` に persistence 系メソッド（3メソッド）を移動する。
   - `export()`, `_resolve_export_path()`, `load()` (classmethod)
4. `lizyml/core/model.py` を `Model(ModelPlotsMixin, ModelTablesMixin, ModelPersistenceMixin)` の多重継承に変更する。
   - `model.py` には `__init__()`, `fit()`, `predict()`, `evaluate()`, `tune()` とプライベートヘルパーのみを残す。
5. 各 mixin で `TYPE_CHECKING` ガードにより循環参照を回避する。
6. mypy strict / ruff がクリーンであることを確認する。

### 25-B: conftest.py へのヘルパー集約

1. `tests/conftest.py` に共通ヘルパーを定義する：
   - `make_regression_df(n=200, seed=0)` → `pd.DataFrame`
   - `make_binary_df(n=200, seed=0)` → `pd.DataFrame`
   - `make_multiclass_df(n=200, n_classes=3, seed=0)` → `pd.DataFrame`
   - `make_config(task, **overrides)` → `dict`
2. 各テストファイルのローカルヘルパー（`_reg_df()`, `_bin_df()`, `_cfg()` 等）を conftest のヘルパーに置き換える。
3. サブディレクトリ固有のフィクスチャはサブディレクトリの `conftest.py` に残す。

### 25-C: parametrize 強化

1. `tests/test_e2e/` でタスク横断テストを `@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])` で統合する。
2. `tests/test_metrics/` でタスク別の重複テストを統合する。
3. 統合後もテスト件数（861件以上）と機能カバレッジを維持する。

### 25-D: CI の develop ブランチ対応

1. `.github/workflows/ci.yml` の `on.pull_request.branches` に `develop` を追加する。
2. develop PR では slow テストを除外する（`-m "not slow"`）。main PR では全テストを実行する（`-m ""`）。
3. `--cov-fail-under=95` を CI の pytest 実行に追加する。

### 25-E: slow テストのローカルスキップ

1. `pyproject.toml` の `[tool.pytest.ini_options]` に `addopts = "-m 'not slow'"` を追加する。
2. CI（main PR）では `-m ""` で上書きして全テストを実行する。

### 25-F: optional dependency の "missing" テスト補完

1. plotly 未インストール時に `OPTIONAL_DEP_MISSING` エラーが発生するテストを追加する。
2. scipy 未インストール時に `OPTIONAL_DEP_MISSING` エラーが発生するテストを追加する。
3. テストは `unittest.mock.patch` で該当モジュールの `_plotly` / `_scipy` 変数を `None` にする方式で実装する（既存の optuna / shap パターンに準拠）。

### 25-G: リリース（v0.1.1）

1. CHANGELOG.md に `[0.1.1]` エントリを追加する。Phase 25 の変更内容（内部リファクタリング・テスト基盤改善・CI 拡張）を記載する。
2. develop → main の PR を作成し、Merge commit で統合する。
3. main に `git tag v0.1.1` を打つ。hatch-vcs により自動的にバージョンが反映される。

### DoD

- 25-A: `model.py` が 700行以下に収まり、全テスト（861件以上）が回帰しない。mypy strict / ruff がクリーン。
- 25-B: 共通ヘルパーの重複定義が `tests/conftest.py` に集約され、各テストファイルからローカル定義が除去される。
- 25-C: parametrize によりテストロジックの重複が削減され、テスト件数とカバレッジが維持される。
- 25-D: CI が develop PR でも実行される（slow 除外）。main PR では全テスト実行。カバレッジ 95% 未満で CI が失敗する。
- 25-E: `uv run pytest` でローカル実行時に slow テストがスキップされる。
- 25-F: plotly / scipy の未インストール時テストが追加され、全 optional dependency の "missing" パスがカバーされる。
- 25-G: CHANGELOG.md に `[0.1.1]` が記載され、main に `v0.1.1` タグが打たれる。

---

## 見積もり

本計画は見積もりを含まない（CLAUDE.md の方針に従い、時間予測は行わない）。
各 Phase の DoD を満たした時点で次 Phase に進む。
