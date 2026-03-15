# Architecture

LizyML のアーキテクチャは **5 層のカテゴリ（圏）** で構成される。
各カテゴリは明確な境界を持ち、依存は常に上位層→下位層の DAG（非巡回有向グラフ）のみ。

## Layer Map — カテゴリ依存 DAG

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'fontSize': '13px',
    'primaryColor': '#2d3748',
    'primaryTextColor': '#e2e8f0',
    'lineColor': '#a0aec0',
    'background': '#1a202c'
  }
}}%%
flowchart TB
    subgraph L0["Layer 0 — Foundation"]
        direction LR
        CORE["core/<br/>exceptions, logging"]
        TYPES["types/<br/>FitResult, PredictionResult,<br/>TuningResult, RunMeta,<br/>SplitIndices, DataFingerprint"]
    end

    subgraph L1["Layer 1 — Leaf Categories"]
        direction LR
        CONFIG["config/<br/>LizyMLConfig<br/>pydantic schemas"]
        DATA["data/<br/>datasource<br/>dataframe_builder<br/>fingerprint"]
        SPLIT["splitters/<br/>BaseSplitter<br/>+ 7 concrete"]
        FEAT["features/<br/>BaseFeaturePipeline<br/>NativeFeaturePipeline"]
        EST["estimators/<br/>BaseEstimatorAdapter<br/>LGBMAdapter<br/>smart params"]
        METRIC["metrics/<br/>BaseMetric<br/>+ registry<br/>+ 14 concrete"]
        CAL["calibration/<br/>BaseCalibratorAdapter<br/>cross_fit_calibrate<br/>+ 3 concrete"]
    end

    subgraph L2["Layer 2 — Composition"]
        direction LR
        TRAIN["training/<br/>CVTrainer<br/>RefitTrainer<br/>TrainComponents<br/>inner_valid"]
        EVAL["evaluation/<br/>Evaluator<br/>table_formatter"]
        TUNE["tuning/<br/>Tuner<br/>SearchDim"]
    end

    subgraph L3["Layer 3 — Optional"]
        direction LR
        EXPLAIN["explain/<br/>shap_explainer<br/>(optional dep)"]
        PLOTS["plots/<br/>8 plot modules<br/>(optional dep)"]
        PERSIST["persistence/<br/>exporter + loader<br/>format_version=1"]
    end

    subgraph L4["Layer 4 — Facade"]
        MODEL["Model<br/>+ PlotsMixin<br/>+ TablesMixin<br/>+ PersistenceMixin<br/>+ _model_factories"]
    end

    %% Layer 0 has no deps (foundation)

    %% Layer 1 → Layer 0 only
    CONFIG --> CORE
    DATA --> CORE
    SPLIT --> CORE
    FEAT --> CORE
    EST --> CORE
    METRIC --> CORE
    CAL --> CORE

    %% Layer 2 → Layer 0 + Layer 1 interfaces
    TRAIN --> CORE
    TRAIN -.->|BaseSplitter IF| SPLIT
    TRAIN -.->|BaseFeaturePipeline IF| FEAT
    TRAIN -.->|BaseEstimatorAdapter IF| EST
    TRAIN --> TYPES

    EVAL --> CORE
    EVAL -.->|BaseMetric IF| METRIC
    EVAL --> TYPES

    TUNE --> CORE
    TUNE --> TYPES

    %% Layer 3 → Layer 0/1/2
    EXPLAIN -.->|BaseEstimatorAdapter IF| EST
    EXPLAIN --> CORE
    PLOTS --> CORE
    PLOTS --> TYPES
    PERSIST --> CORE
    PERSIST --> TYPES

    %% Layer 4 → all layers
    MODEL --> CONFIG
    MODEL --> DATA
    MODEL --> TRAIN
    MODEL --> EVAL
    MODEL --> TUNE
    MODEL --> CAL
    MODEL --> EXPLAIN
    MODEL --> PLOTS
    MODEL --> PERSIST
    MODEL --> TYPES

    %% Styling
    style L0 fill:#1a365d,stroke:#4a90d9,color:#e2e8f0
    style L1 fill:#1a332a,stroke:#68d391,color:#e2e8f0
    style L2 fill:#2d2040,stroke:#b794f4,color:#e2e8f0
    style L3 fill:#3b2f20,stroke:#ed8936,color:#e2e8f0
    style L4 fill:#3b1c1c,stroke:#fc8181,color:#e2e8f0
```

### 依存ルール

| ルール | 説明 |
|---|---|
| **下方向のみ** | 各カテゴリは自分より上の Layer にのみ依存する |
| **IF 経由** | Layer 2 は Layer 1 の**抽象 IF のみ**を参照（具象クラスを import しない） |
| **Facade 独占** | 具象クラスの組み立て・型ディスパッチは Layer 4 のみが行う |
| **循環禁止** | カテゴリ間の双方向依存は許可しない |

### 射（カテゴリ間契約）

| From | To | 契約 |
|---|---|---|
| Config → Facade | `LizyMLConfig` (validated dict) |
| Data → Facade | `DataFrameComponents(X, y, time_col?, group_col?)` |
| Splitters → Training | `BaseSplitter.split(n, y, groups) → Iterator[(train_idx, valid_idx)]` |
| Features → Training | `BaseFeaturePipeline.fit(X, y)`, `.transform(X) → DataFrame` |
| Estimators → Training | `BaseEstimatorAdapter.fit()`, `.predict()`, `.predict_proba() → ndarray` |
| Metrics → Evaluation | `BaseMetric.__call__(y_true, y_pred) → float` |
| Calibration → Facade | `BaseCalibratorAdapter.fit(scores, y)`, `.predict(scores) → ndarray` |
| Training → Facade | `FitResult`, `RefitResult` |
| Evaluation → Facade | `dict` (structured metrics) |
| Tuning → Facade | `TuningResult` |

---

## Layer 0 — Foundation

依存ゼロ。全カテゴリが参照する共通基盤。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'primaryColor': '#2d3748', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'background': '#1a202c', 'mainBkg': '#2d3748', 'nodeBorder': '#718096'}}}%%
classDiagram
    direction LR

    class LizyMLError:::foundationClass {
        +code: ErrorCode
        +user_message: str
        +debug_message: str?
        +cause: Exception?
        +context: dict
    }
    class ErrorCode:::foundationClass {
        <<enum>>
        CONFIG_INVALID
        DATA_SCHEMA_INVALID
        MODEL_NOT_FIT
        TUNING_FAILED
        ...
    }

    class FitResult:::foundationClass {
        +oof_pred +oof_raw_scores?
        +if_pred_per_fold +metrics
        +models +history
        +feature_names +dtypes
        +categorical_features
        +splits: SplitIndices
        +data_fingerprint: DataFingerprint
        +pipeline_state
        +calibrator: CalibrationResult?
        +run_meta: RunMeta
    }
    class PredictionResult:::foundationClass { +pred +proba +shap_values +used_features +warnings }
    class RefitResult:::foundationClass { +model +pipeline_state +feature_names +train_pred +history }
    class TuningResult:::foundationClass {
        +best_model_params +best_smart_params +best_training_params
        +best_score +trials +metric_name +direction
        +best_params <<property>>
    }
    class TrialResult:::foundationClass { +number +params +score +state }
    class CalibrationResult:::foundationClass { +c_final +calibrated_oof +method +split_indices }
    class SplitIndices:::foundationClass { +outer +inner? +calibration? +time_range? }
    class RunMeta:::foundationClass { +lizyml_version +python_version +deps_versions +run_id +timestamp }
    class DataFingerprint:::foundationClass { +row_count +column_hash +file_hash? }

    FitResult *-- SplitIndices
    FitResult *-- RunMeta
    FitResult *-- DataFingerprint
    FitResult o-- CalibrationResult
    TuningResult *-- TrialResult

    classDef foundationClass fill:#1a365d,stroke:#4a90d9,color:#e2e8f0
```

---

## Layer 1 — Leaf Categories

Foundation のみに依存。互いに依存しない。

### config/

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'primaryColor': '#2d3748', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'background': '#1a202c', 'mainBkg': '#2d3748', 'nodeBorder': '#718096'}}}%%
classDiagram
    direction TB

    class LizyMLConfig:::configClass {
        +config_version: int
        +task: TaskType
        +data: DataConfig
        +features: FeaturesConfig
        +split: SplitConfig
        +model: ModelConfig
        +training: TrainingConfig
        +tuning: TuningConfig?
        +evaluation: EvaluationConfig
        +calibration: CalibrationConfig?
        +output_dir: str?
    }
    class DataConfig:::configClass { +path +target +time_col +group_col }
    class FeaturesConfig:::configClass { +exclude +auto_categorical +categorical }
    class TrainingConfig:::configClass { +seed +early_stopping: EarlyStoppingConfig }
    class EarlyStoppingConfig:::configClass { +enabled +rounds +inner_valid +validation_ratio }
    class EvaluationConfig:::configClass { +metrics: list }
    class CalibrationConfig:::configClass { +method +n_splits +params }
    class TuningConfig:::configClass { +optuna: OptunaConfig }

    LizyMLConfig *-- DataConfig
    LizyMLConfig *-- FeaturesConfig
    LizyMLConfig *-- TrainingConfig
    LizyMLConfig *-- EvaluationConfig
    LizyMLConfig *-- CalibrationConfig
    LizyMLConfig *-- TuningConfig
    TrainingConfig *-- EarlyStoppingConfig

    %% SplitConfig — discriminated union on "method"
    class KFoldConfig:::configClass { +method="kfold" }
    class StratifiedKFoldConfig:::configClass { +method="stratified_kfold" }
    class GroupKFoldConfig:::configClass { +method="group_kfold" }
    class TimeSeriesConfig:::configClass { +method="time_series" }
    class PurgedTimeSeriesConfig:::configClass { +method="purged_time_series" }
    class GroupTimeSeriesConfig:::configClass { +method="group_time_series" }

    LizyMLConfig *-- KFoldConfig : split
    LizyMLConfig *-- StratifiedKFoldConfig : split
    LizyMLConfig *-- GroupKFoldConfig : split
    LizyMLConfig *-- TimeSeriesConfig : split
    LizyMLConfig *-- PurgedTimeSeriesConfig : split
    LizyMLConfig *-- GroupTimeSeriesConfig : split

    %% ModelConfig — discriminated union on "name"
    class LGBMConfig:::configClass {
        +name="lgbm"
        +params: dict
        +auto_num_leaves +num_leaves_ratio
        +min_data_in_leaf_ratio +min_data_in_bin_ratio
        +feature_weights +balanced
    }
    LizyMLConfig *-- LGBMConfig : model

    %% InnerValidConfig — discriminated union on "method"
    class HoldoutInnerValidConfig:::configClass { +method="holdout" +ratio +stratify }
    class GroupHoldoutInnerValidConfig:::configClass { +method="group_holdout" +ratio }
    class TimeHoldoutInnerValidConfig:::configClass { +method="time_holdout" +ratio }
    EarlyStoppingConfig *-- HoldoutInnerValidConfig : inner_valid
    EarlyStoppingConfig *-- GroupHoldoutInnerValidConfig : inner_valid
    EarlyStoppingConfig *-- TimeHoldoutInnerValidConfig : inner_valid

    classDef configClass fill:#22543d,stroke:#68d391,color:#e2e8f0
```

### splitters/ + features/ + estimators/ + metrics/ + calibration/ + data/

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'primaryColor': '#2d3748', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'background': '#1a202c', 'mainBkg': '#2d3748', 'nodeBorder': '#718096'}}}%%
classDiagram
    direction TB

    %% === splitters/ ===
    class BaseSplitter:::ifClass {
        <<abstract>>
        +split(n, y, groups) Iterator
    }
    class KFoldSplitter:::leafClass { }
    class StratifiedKFoldSplitter:::leafClass { }
    class GroupKFoldSplitter:::leafClass { }
    class TimeSeriesSplitter:::leafClass { }
    class PurgedTimeSeriesSplitter:::leafClass { }
    class GroupTimeSeriesSplitter:::leafClass { }
    KFoldSplitter --|> BaseSplitter
    StratifiedKFoldSplitter --|> BaseSplitter
    GroupKFoldSplitter --|> BaseSplitter
    TimeSeriesSplitter --|> BaseSplitter
    PurgedTimeSeriesSplitter --|> BaseSplitter
    GroupTimeSeriesSplitter --|> BaseSplitter

    %% === features/ ===
    class BaseFeaturePipeline:::ifClass {
        <<abstract>>
        +fit(X, y) +transform(X)
        +transform_with_warnings(X)
        +get_state() +load_state()
    }
    class NativeFeaturePipeline:::leafClass { }
    NativeFeaturePipeline --|> BaseFeaturePipeline

    %% === estimators/ ===
    class BaseEstimatorAdapter:::ifClass {
        <<abstract>>
        +fit() +predict() +predict_proba()
        +predict_raw() +importance()
        +update_params() +get_native_model()
        +best_iteration +eval_results
    }
    class LGBMAdapter:::leafClass {
        -_model: Booster
        +task +params +early_stopping_rounds
    }
    LGBMAdapter --|> BaseEstimatorAdapter

    %% === metrics/ ===
    class BaseMetric:::ifClass {
        <<abstract>>
        +name +needs_proba +needs_simplex
        +__call__(y_true, y_pred) float
    }
    class MetricRegistry:::leafClass {
        +get(name) BaseMetric
        +get_metrics_for_task(names, task) list
    }
    MetricRegistry ..> BaseMetric : lookups

    %% === calibration/ ===
    class BaseCalibratorAdapter:::ifClass {
        <<abstract>>
        +name: str
        +fit(scores, y) +predict(scores)
    }
    class PlattCalibrator:::leafClass { }
    class IsotonicCalibrator:::leafClass { }
    class BetaCalibrator:::leafClass { }
    PlattCalibrator --|> BaseCalibratorAdapter
    IsotonicCalibrator --|> BaseCalibratorAdapter
    BetaCalibrator --|> BaseCalibratorAdapter

    class cross_fit_calibrate:::leafClass {
        +cross_fit_calibrate(oof_scores, y, factory, splits) CalibrationResult
    }
    cross_fit_calibrate ..> BaseCalibratorAdapter : uses via factory

    %% === data/ ===
    class datasource:::leafClass {
        +read(path) DataFrame
    }
    class dataframe_builder:::leafClass {
        +build(df, problem_spec, feature_spec) DataFrameComponents
    }
    class DataFrameComponents:::leafClass { +X +y +time_col? +group_col? }
    dataframe_builder ..> DataFrameComponents : produces

    classDef ifClass fill:#1e3a5f,stroke:#4a90d9,color:#e2e8f0
    classDef leafClass fill:#22543d,stroke:#68d391,color:#e2e8f0
```

---

## Layer 2 — Composition

Layer 1 の **抽象 IF のみ** を参照。具象クラスを知らない。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'primaryColor': '#2d3748', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'background': '#1a202c', 'mainBkg': '#2d3748', 'nodeBorder': '#718096'}}}%%
classDiagram
    direction TB

    %% === training/ ===
    class TrainComponents:::compClass {
        <<frozen dataclass>>
        +estimator_factory: Callable → BaseEstimatorAdapter
        +sample_weight: ndarray?
        +ratio_resolver: Callable?
        +inner_valid: BaseInnerValidStrategy
    }
    class CVTrainer:::compClass {
        +outer_splitter: BaseSplitter
        +inner_valid: BaseInnerValidStrategy
        +pipeline_factory: Callable
        +estimator_factory: Callable
        +task +n_classes +ratio_param_resolver
        +collect_raw_scores: bool
        +fit(X, y, groups, ...) FitResult
    }
    class RefitTrainer:::compClass {
        +inner_valid: BaseInnerValidStrategy
        +pipeline_factory: Callable
        +estimator_factory: Callable
        +task +ratio_param_resolver
        +fit(X, y, groups) RefitResult
    }
    class BaseInnerValidStrategy:::ifClass {
        <<abstract>>
        +split(n, y, groups) tuple?
    }
    class NoInnerValid:::compClass { }
    class HoldoutInnerValid:::compClass { +ratio +stratify +random_state }
    class GroupHoldoutInnerValid:::compClass { +ratio +random_state }
    class TimeHoldoutInnerValid:::compClass { +ratio }
    NoInnerValid --|> BaseInnerValidStrategy
    HoldoutInnerValid --|> BaseInnerValidStrategy
    GroupHoldoutInnerValid --|> BaseInnerValidStrategy
    TimeHoldoutInnerValid --|> BaseInnerValidStrategy

    TrainComponents ..> CVTrainer : provides factory/resolver/iv
    TrainComponents ..> RefitTrainer : provides same

    CVTrainer o-- BaseInnerValidStrategy : inner_valid

    %% === evaluation/ ===
    class Evaluator:::compClass {
        +task: TaskType
        +evaluate(fit_result, y, metric_names) dict
    }

    %% === tuning/ ===
    class Tuner:::compClass {
        +dims: list~SearchDim~
        +n_trials +direction +timeout +seed
        +progress_callback: TuneProgressCallback?
        +tune(objective, metric_name) TuningResult
    }
    class SearchDim:::compClass {
        <<abstract>>
        +name: str
        +category: model|smart|training
    }
    class FloatDim:::compClass { +low +high +log }
    class IntDim:::compClass { +low +high }
    class CategoricalDim:::compClass { +choices }
    FloatDim --|> SearchDim
    IntDim --|> SearchDim
    CategoricalDim --|> SearchDim
    Tuner o-- SearchDim : dims

    classDef ifClass fill:#1e3a5f,stroke:#4a90d9,color:#e2e8f0
    classDef compClass fill:#2d2040,stroke:#b794f4,color:#e2e8f0
```

**training/ が参照するのは IF のみ:**
- `BaseSplitter` (splitters/)、`BaseFeaturePipeline` (features/)、`BaseEstimatorAdapter` (estimators/)
- 具象クラス (`KFoldSplitter`, `LGBMAdapter` 等) を import しない

---

## Layer 3 — Optional

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'primaryColor': '#2d3748', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'background': '#1a202c', 'mainBkg': '#2d3748', 'nodeBorder': '#718096'}}}%%
classDiagram
    direction LR

    class shap_explainer:::optClass {
        +compute_shap_values(model, X, task) ndarray
        +compute_shap_importance(models, X, splits, ...) dict
    }

    class plots:::optClass {
        +plot_importance()
        +plot_learning_curve()
        +plot_oof_distribution()
        +plot_residuals()
        +plot_roc_curve()
        +plot_calibration_curve()
        +plot_probability_histogram()
        +plot_tuning_history()
    }

    class persistence:::optClass {
        +export(path, fit_result, refit_result, ...) void
        +load(path) tuple
        +FORMAT_VERSION = 1
    }
    class AnalysisContext:::optClass { +y_true +X_for_explain }
    persistence ..> AnalysisContext : saves/restores

    classDef optClass fill:#3b2f20,stroke:#ed8936,color:#e2e8f0
```

**依存先**: Foundation の型 (`FitResult`, `TuningResult` 等) のみ。Layer 1/2 の具象を直接参照しない。

---

## Layer 4 — Facade

**唯一**全カテゴリを知る層。具象クラスの組み立てとディスパッチを担当。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '11px', 'primaryColor': '#2d3748', 'primaryTextColor': '#e2e8f0', 'lineColor': '#a0aec0', 'background': '#1a202c', 'mainBkg': '#2d3748', 'nodeBorder': '#718096'}}}%%
classDiagram
    direction TB

    class Model:::facadeClass {
        -_cfg: LizyMLConfig
        -_fit_result: FitResult?
        -_refit_result: RefitResult?
        -_tuning_result: TuningResult?
        +fit(data, params) FitResult
        +predict(X, return_shap) PredictionResult
        +evaluate(metrics) dict
        +tune(data, progress_callback) TuningResult
        +export(path) Path
        +load(path)$ Model
        -_merge_params(override) tuple
        -_build_train_components(X, y, ...) TrainComponents
        -_prepare_training_data(data) tuple
        -_run_calibration(...) FitResult
    }
    class ModelPlotsMixin:::mixinClass {
        +residuals_plot() +roc_curve_plot()
        +calibration_plot() +probability_histogram_plot()
        +importance_plot() +plot_learning_curve()
        +plot_oof_distribution() +tuning_plot()
    }
    class ModelTablesMixin:::mixinClass {
        +evaluate_table() +residuals()
        +confusion_matrix() +importance()
        +tuning_table() +params_table()
        +split_summary()
    }
    class ModelPersistenceMixin:::mixinClass {
        +export(path) Path
        +load(path)$ Model
    }
    class _model_factories:::facadeClass {
        +build_splitter(cfg) BaseSplitter
        +build_inner_valid(cfg) InnerValidType
        +build_calibration_splitter(cfg) BaseSplitter
        +make_inner_valid_factory(cfg) Callable
    }

    Model --|> ModelPlotsMixin : mixin
    Model --|> ModelTablesMixin : mixin
    Model --|> ModelPersistenceMixin : mixin
    Model ..> _model_factories : uses

    classDef facadeClass fill:#3b1c1c,stroke:#fc8181,color:#e2e8f0,stroke-width:2px
    classDef mixinClass fill:#4a3728,stroke:#ed8936,color:#e2e8f0
```

**Facade の責務**: Config を読み、各カテゴリの具象クラスを選択・組み立て・接続する。
ロジック（OOF 生成、metric 計算、学習ループ等）は一切持たない。

---

## 処理フロー

### fit()

```
Facade (Model)
  │
  ├─ 1. Data ── datasource.read → dataframe_builder.build
  │              → (X, y, groups)
  │
  ├─ 2. Config ── _merge_params(override)
  │              Config defaults < tune best < fit() args
  │              → (model_params, smart_params)
  │
  ├─ 3. Estimators ── resolve_smart_params(dict) → resolved params
  │    Config ── _model_factories.build_inner_valid(cfg)
  │              → TrainComponents (frozen dataclass)
  │
  ├─ 4. Training ── CVTrainer.fit(X, y, groups, ...)
  │    ┌─ Splitters ── outer_splitter.split() → indices
  │    ├─ per fold:
  │    │   ├─ Features ── pipeline.fit(X_train) → transform
  │    │   ├─ Estimators ── estimator.fit(X, y, X_valid, y_valid)
  │    │   └─ predictions → OOF / IF
  │    └─ → FitResult (metrics={})
  │
  ├─ 5. Calibration ── cross_fit_calibrate(oof_scores, y)
  │              → FitResult + CalibrationResult
  │
  ├─ 6. Evaluation ── Evaluator.evaluate(fit_result, y, metric_names)
  │    └─ Metrics ── registry → BaseMetric.__call__()
  │              → structured metrics dict
  │
  └─ 7. Training ── RefitTrainer.fit(X, y, groups)
                 → RefitResult (final model for inference)
```

### tune()

```
Facade (Model)
  │
  ├─ 1. Data ── _prepare_training_data
  │
  ├─ 2. Config ── _merge_params → base params
  │
  ├─ 3. Tuning ── parse_space or default_space
  │              → list[SearchDim], fixed params
  │
  ├─ 4. Facade builds objective closure:
  │    objective(trial):
  │      ├─ Tuning ── suggest_params → split_by_category
  │      ├─ Estimators ── resolve_smart_params
  │      ├─ Training ── CVTrainer.fit() → FitResult
  │      └─ Evaluation ── Evaluator → score
  │
  ├─ 5. Tuning ── Tuner.tune(objective)
  │              → TuningResult
  │
  └─ stored for next fit()
```

### predict()

```
Facade (Model)
  │
  ├─ 1. Features ── pipeline.load_state → transform_with_warnings
  │
  ├─ 2. Estimators ── model.predict / predict_proba
  │    └─ Calibration ── c_final.predict (binary)
  │
  ├─ 3. Explain ── compute_shap_values (optional)
  │
  └─ → PredictionResult
```

### export() / load()

```
export:  FitResult + RefitResult + Config + AnalysisContext
         → metadata.json + fit_result.pkl + refit_model.pkl
         format_version=1

load:    metadata.json → validate format_version
         → Model instance (predict / evaluate / plots ready)
```

---

## モジュール構成

```
lizyml/
│
├── core/                           ── Layer 0: Foundation ──
│   ├── exceptions.py               LizyMLError + ErrorCode
│   ├── logging.py                  logger + run_id + output_dir
│   ├── registries.py               MetricRegistry (generic)
│   └── types/
│       ├── fit_result.py           FitResult (13 fields)
│       ├── predict_result.py       PredictionResult
│       ├── tuning_result.py        TuningResult, TrialResult
│       └── artifacts.py            RunMeta, SplitIndices, DataFingerprint
│
├── config/                         ── Layer 1: Config ──
│   ├── schema.py                   pydantic schemas (extra="forbid")
│   └── loader.py                   YAML/JSON/dict → LizyMLConfig
│
├── data/                           ── Layer 1: Data ──
│   ├── datasource.py               CSV / Parquet / DataFrame
│   ├── dataframe_builder.py        X/y/groups 分離 + categorical
│   └── fingerprint.py              DataFingerprint 計算
│
├── splitters/                      ── Layer 1: Splitting ──
│   ├── base.py                     BaseSplitter
│   ├── kfold.py                    KFoldSplitter, StratifiedKFoldSplitter
│   ├── group_kfold.py              GroupKFoldSplitter
│   ├── time_series.py              TimeSeriesSplitter
│   ├── purged_time_series.py       PurgedTimeSeriesSplitter
│   └── group_time_series.py        GroupTimeSeriesSplitter
│
├── features/                       ── Layer 1: Features ──
│   ├── pipeline_base.py            BaseFeaturePipeline
│   ├── pipelines_native.py         NativeFeaturePipeline
│   └── encoders/                   CategoricalEncoder
│
├── estimators/                     ── Layer 1: Estimators ──
│   ├── base.py                     BaseEstimatorAdapter
│   └── lgbm.py                     LGBMAdapter + smart param resolution
│
├── metrics/                        ── Layer 1: Metrics ──
│   ├── base.py                     BaseMetric
│   ├── registry.py                 MetricRegistry helpers
│   ├── regression.py               RMSE, MAE, R2, RMSLE, MAPE, Huber
│   └── classification.py           LogLoss, AUC, AUCPR, F1, Accuracy, Brier, ECE
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
│   ├── train_components.py         TrainComponents (frozen dataclass)
│   └── inner_valid.py              BaseInnerValidStrategy + 4 concrete
│
├── evaluation/                     ── Layer 2: Evaluation ──
│   ├── evaluator.py                Evaluator (structured metrics)
│   ├── table_formatter.py          evaluate_table 整形
│   └── confusion.py                confusion_matrix_table
│
├── tuning/                         ── Layer 2: Tuning ──
│   ├── tuner.py                    Tuner (Optuna study management)
│   └── search_space.py             SearchDim, parse/suggest/split_by_category
│
├── explain/                        ── Layer 3: Explain (optional) ──
│   └── shap_explainer.py           compute_shap_values / importance
│
├── plots/                          ── Layer 3: Plots (optional) ──
│   ├── importance.py               feature importance bar chart
│   ├── learning_curve.py           training/validation loss curve
│   ├── oof_distribution.py         OOF prediction distribution
│   ├── residuals.py                scatter/histogram/QQ
│   ├── classification.py           ROC curve
│   ├── calibration.py              reliability diagram + probability histogram
│   └── tuning.py                   tuning history plot
│
├── persistence/                    ── Layer 3: Persistence ──
│   ├── exporter.py                 export() + FORMAT_VERSION
│   └── loader.py                   load() + format_version validation
│
└── core/                           ── Layer 4: Facade ──
    ├── model.py                    Model (組み立てと委譲のみ)
    ├── _model_factories.py         splitter/inner_valid/calibration 構築
    ├── _model_plots.py             ModelPlotsMixin
    ├── _model_tables.py            ModelTablesMixin
    └── _model_persistence.py       ModelPersistenceMixin
```

---

## 凡例

| 色 | Layer | 意味 |
|---|---|---|
| 青 | 0 | **Foundation** — 全カテゴリが参照する型・例外・ログ |
| 緑 | 1 | **Leaf** — 互いに依存しない独立カテゴリ |
| 紫 | 2 | **Composition** — Layer 1 の IF を組み合わせるカテゴリ |
| 橙 | 3 | **Optional** — Foundation の型のみ参照する追加機能 |
| 赤 | 4 | **Facade** — 唯一全カテゴリを知り、組み立てる |

---

## 現状との差分（実装ロードマップ）

現在の実装はこの目標構造に概ね沿っているが、以下の修正が必要:

| # | 問題 | 目標 Layer ルール違反 | 修正方針 |
|---|---|---|---|
| 1 | `training/` が `evaluation/oof.py` に依存 | L2 → L2 (同層間の依存) | `oof.py` の `fill_oof`, `get_fold_pred` を `training/` に移動 |
| 2 | `evaluation/` が `calibration/` に依存 | L2 → L1 (逆方向ではないが不要) | calibrated metrics の組み立てを Facade に移動 |
| 3 | `estimators/` が `config/` に依存 | L1 → L1 (Leaf 間の依存) | `extract_smart_params(LGBMConfig)` を Facade に移動 |
| 4 | `types/` が `data/` に依存 | L0 → L1 (逆方向) | `DataFingerprint` を `types/` に移動 |
| 5 | `splitters/` ↔ `specs/` 循環 | 循環 | 死んだ Spec 層を削除 |
| 6 | デッドコード多数 | — | `TargetTransformer`, `SplitPlan`, 未使用 Spec 等を削除 |
