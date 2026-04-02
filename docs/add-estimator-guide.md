# 新規 Estimator 実装ガイド

LightGBM 以外の学習アルゴリズム（XGBoost, CatBoost, sklearn, PyTorch 等）を追加する際に実装が必要な全項目を網羅する。

---

## 1. ディレクトリ構成

```
lizyml/estimators/<name>/
├── __init__.py           # Public exports
├── adapter.py            # BaseEstimatorAdapter 実装
├── provider.py           # EstimatorProvider 実装
├── defaults.py           # タスク別デフォルト、探索空間
├── smart_params.py       # Smart parameter 解決ロジック（任意）
└── metric_bridge.py      # Metric 名変換 + feval 生成（任意）
```

LightGBM の実装 (`lizyml/estimators/lgbm/`) が全項目のリファレンス。

---

## 2. 必須: BaseEstimatorAdapter の実装

`lizyml/estimators/base.py` の抽象メソッドを全て実装する。

### 2.1 抽象メソッド（6 個）

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `fit(X_train, y_train, X_valid?, y_valid?, **kwargs)` | `self` | 学習。`sample_weight` は `kwargs` 経由。early stopping 使用時は `best_iteration` を設定 |
| `predict(X)` | `ndarray[float64]` (1D) | 回帰: 予測値。二値分類: 正例確率（1D）。多値分類: クラスラベル（argmax 後の整数配列） |
| `predict_proba(X)` | `ndarray[float64]` (2D) | 二値: `(n, 2)`。多値: `(n, k)`。回帰は `LizyMLError` |
| `predict_raw(X)` | `ndarray[float64]` | Calibration / feval 用の生スコア（logit 等）。回帰は `predict()` と同一 |
| `importance(kind)` | `dict[str, float]` | `kind="split"` or `"gain"`。キーは `fit()` 時の列名と一致必須 |
| `get_native_model()` | `Any` | 内部モデルオブジェクト（`Booster`, `Estimator` 等） |

> **注意**: `predict()` の戻り値の意味はアルゴリズムの Booster/sklearn API によって異なる。LightGBM Booster API では `objective="binary"` の `predict()` は確率値を返す（sigmoid 適用済み）。実装時はアルゴリズムの predict API に合わせ、CVTrainer 側の `get_fold_pred()` / `get_fold_raw()` と整合性を確認すること。

### 2.2 オプショナルオーバーライド（4 個）

| メソッド | デフォルト | 説明 |
|---------|----------|------|
| `set_categorical_features(cols)` | no-op | CVTrainer が `fit()` 前に呼ぶ |
| `update_params(params)` | no-op | per-fold ratio param 解決用 |
| `best_iteration` (property) | `None` | Early stopping 使用時に設定 |
| `eval_results` (property) | `{}` | 学習曲線用の評価履歴 |

### 2.3 実装上の注意

- `fit()` は必ず `self` を返す
- `self._feature_names = list(X_train.columns)` を `fit()` 冒頭で保存する
- `eval_results` は `{"valid_0": {"metric_name": [v0, v1, ...]}, "train": {...}}` 形式

---

## 3. 必須: EstimatorProvider の実装

`lizyml/estimators/provider.py` の Protocol に従う。10 メソッド全てを実装する。

### 3.1 Config 抽出

| メソッド | 説明 |
|---------|------|
| `extract_model_params(model_cfg)` → `dict` | Config からネイティブパラメータを抽出 |
| `extract_smart_params(model_cfg)` → `dict` | Config から Smart parameter を抽出 |

### 3.2 Smart parameter 解決

| メソッド | 説明 |
|---------|------|
| `resolve_smart_params(smart, effective_params, n_rows, feature_names, y, task)` → `(dict, weight?)` | データサイズ依存パラメータの解決。`sample_weight` を返す場合あり |
| `build_ratio_resolver(smart)` → `Callable[[int], dict] \| None` | per-fold で n_rows に応じてパラメータを再解決する関数。不要なら `None` |

### 3.3 ファクトリ

| メソッド | 説明 |
|---------|------|
| `build_estimator_factory(task, params, n_classes, early_stopping_rounds, seed)` → `Callable[[], Adapter]` | fold ごとに新規インスタンスを生成する 0 引数ファクトリ |
| `build_pipeline_factory()` → `Callable[[], Pipeline]` | FeaturePipeline のファクトリ |

### 3.4 Tuning

| メソッド | 説明 |
|---------|------|
| `default_space(task)` → `list[SearchDim]` | デフォルト探索空間。`SearchDim` の `category` は `"model"` / `"smart"` / `"training"` |
| `default_fixed_params(task)` → `dict` | Tuning 時に全試行で固定するパラメータ |

> **SearchDim の型**:
> - `FloatDim(name, low, high, log=False, category="model")` — frozen dataclass
> - `IntDim(name, low, high, log=False, category="model")` — frozen dataclass
> - `CategoricalDim(name, choices, category="model")` — `choices` は **`tuple`** 型（`list` ではない）

### 3.5 メタ情報

| メソッド | 説明 |
|---------|------|
| `runtime_deps()` → `dict[str, str]` | 依存パッケージのバージョン（`RunMeta.deps_versions` 用） |
| `params_summary(model, model_cfg)` → `list[dict]` | `params_table()` 表示用の `[{"parameter": str, "value": Any}, ...]` |

---

## 4. 必須: Config スキーマ統合

### 4.1 Config クラス追加 (`config/schema.py`)

```python
class YourConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["your_name"]
    params: dict[str, Any] = {}
    # Smart parameter fields (任意)
```

### 4.2 ModelConfig Union 更新

現在の `ModelConfig` は単一型で定義されている:

```python
# 現状 (LightGBM のみ)
ModelConfig = Annotated[LGBMConfig, Field(discriminator="name")]
```

新規 estimator 追加時に **Union 化が必要**:

```python
# 変更後
ModelConfig = Annotated[
    LGBMConfig | YourConfig,  # ← Union に拡張
    Field(discriminator="name"),
]
```

> **注意**: この変更は Config スキーマの変更に該当するため、HISTORY.md に Proposal が必要。

### 4.3 Provider ディスパッチ登録 (`core/_model_factories.py`)

`get_provider()` に `elif` 分岐を 1 行追加。

---

## 5. 条件付き必須: Metric Bridge

アルゴリズム固有のメトリック名体系がある場合に実装する。

### 5.1 対象

- LizyML メトリック名 → アルゴリズム固有名のマッピング（例: `logloss` → `binary_logloss`）
- タスク別ホワイトリスト検証（アルゴリズムがネイティブでサポートする metric の一覧）
- feval カスタム関数生成（ネイティブ metric として存在しない指標の対応）

### 5.2 feval metric の分類

どの metric が feval（カスタム関数）扱いになるかは **アルゴリズムごとに異なる**。LightGBM の場合:

```python
# lizyml/estimators/lgbm/metric_bridge.py
_FEVAL_METRICS: dict[str, frozenset[str]] = {
    "regression": frozenset(["rmsle", "r2"]),
    "binary": frozenset(["f1", "brier", "ece", "precision_at_k", "accuracy"]),
    "multiclass": frozenset(["f1", "brier", "accuracy"]),
}
```

新アルゴリズムでは、ネイティブ metric のサポート範囲に応じて異なる `_FEVAL_METRICS` を定義する。例えば XGBoost が `rmsle` をネイティブでサポートする場合、`_FEVAL_METRICS["regression"]` から除外する。

### 5.3 `resolve_metrics()` の実装

```python
def resolve_metrics(metrics, task, num_class=None):
    """Returns: (native_metrics, feval_callables, feval_display_names)"""
```

---

## 6. 条件付き必須: Codegen 対応 (H-0059, H-0066)

`model.export_code()` で LizyML 非依存コードを生成するには、追加の実装が必要。

> **現状の制約**: codegen は現在 LightGBM 専用。以下のファイルが `LGBMAdapter` を直接 import している:
> - `_model_persistence.py` — `isinstance(adapter, LGBMAdapter)` でガード
> - `codegen/generator.py` — 関数シグネチャに `model_adapter: LGBMAdapter`
> - `codegen/artifact_writer.py` — 関数シグネチャに `model_adapter: LGBMAdapter`、`save_model_text()` を呼び出し

### 6.1 Adapter への追加メソッド

| メソッド | 説明 |
|---------|------|
| `save_model_text(path)` → `Path` | モデルをテキスト形式で保存 |
| `_build_params()` → `(params, n_rounds, feval_list, feval_names)` | パラメータ抽出（codegen が config.json に書き込む）。現在はプライベートメソッド。将来的には EstimatorProvider protocol に公開メソッドとして追加予定（TODO: H-0059 コメント参照） |

### 6.2 テンプレート (`codegen/templates.py`)

現在の `_TRAIN_PY` / `_PREDICT_PY` は LightGBM 固有。新しいアルゴリズムを追加する場合:

1. **アルゴリズム別テンプレートの分離** — `_TRAIN_PY_LGBM`, `_TRAIN_PY_XGBOOST` 等
2. **`render_train_py(algorithm)` へのシグネチャ変更** — アルゴリズムに応じたテンプレートを返す
3. **共通部分の抽出** — Feature Pipeline, Calibration, feval metric 関数は共通。学習部分のみ差し替え

### 6.3 feval metric 対応 (H-0066)

`config.json` に `feval_metrics` フィールドが含まれる。テンプレートの `build_feval_from_config()` は feval metric を純粋な numpy/sklearn で再実装している。

新しいアルゴリズムでは:
- feval callable のシグネチャがアルゴリズムごとに異なる可能性がある（LightGBM: `(y_pred, Dataset) → (name, value, is_higher_better)`、XGBoost: `(y_pred, DMatrix) → (name, value)` 等）
- `build_feval_from_config()` の feval wrapper をアルゴリズム別に実装する必要がある
- feval 内部の予測変換（sigmoid/softmax）と metric 計算関数（`_feval_f1` 等）は共通利用可能

### 6.4 `_model_persistence.py` の拡張

現在は `isinstance(adapter, LGBMAdapter)` でガードされている。新アルゴリズム対応時:
- `export_code()` のガード条件を拡張
- `_extract_feval_metadata()` は `adapter.params` と `adapter.task` のみ参照するが、**`_FEVAL_METRICS` のインポート元が `lizyml.estimators.lgbm.metric_bridge` に固定されている**。新アルゴリズムの metric_bridge から取得するようにディスパッチの追加が必要

### 6.5 `codegen/generator.py` と `codegen/artifact_writer.py` の拡張

両ファイルとも関数シグネチャが `model_adapter: LGBMAdapter` で型制約されている。新アルゴリズム対応時:
- `LGBMAdapter` → `BaseEstimatorAdapter` への型拡張
- `save_model_text()` の呼び出しをアルゴリズム別にディスパッチ
- `artifact_writer.py` のモデル保存ロジックをアルゴリズム別に分離

---

## 7. テスト要件

### 7.1 必須テスト

| カテゴリ | 内容 |
|---------|------|
| **契約テスト** | `predict()` / `predict_proba()` / `predict_raw()` の shape と dtype |
| **再現性テスト** | seed 固定で同一結果 |
| **Early stopping** | `best_iteration` が設定される、`eval_results` に履歴がある |
| **Provider 不変条件** | `check_estimator` スタイルのプロトコル準拠テスト |
| **Config バリデーション** | Smart param 競合、不正パラメータの拒否 |

### 7.2 推奨テスト

| カテゴリ | 内容 |
|---------|------|
| **Tuning 統合** | `default_space()` で Optuna 試行が動作する |
| **Codegen 等価** | `export_code()` 生成コードの予測が本体と一致 |
| **Importance** | feature_names と一致するキーを返す |

### 7.3 テストファイル構成

```
tests/test_estimators/
├── test_<name>_adapter.py     # Adapter 単体テスト
├── test_<name>_provider.py    # Provider 単体テスト
├── test_<name>_smart_params.py  # Smart params (任意)
└── test_check_provider.py     # 既存の Provider 不変条件テストに追加
```

---

## 8. 変更ゲート

以下は HISTORY.md に Proposal が必要（CLAUDE.md §2 準拠）:

- Config スキーマ (`ModelConfig` Union) の変更
- EstimatorProvider protocol の変更（新メソッド追加等）
- codegen テンプレートの構造変更
- 外部依存の追加（例: `xgboost`, `catboost`）

Proposal 不要:
- 既存 protocol / interface に従った新 estimator パッケージの追加

---

## 9. アーキテクチャ制約

| ルール | 説明 |
|--------|------|
| **Layer 2 からの具象 import 禁止** | `training/`, `evaluation/`, `tuning/` は `BaseEstimatorAdapter` / `EstimatorProvider` のみ参照。`LGBMAdapter` 等の具象クラスを直接 import しない |
| **Provider ディスパッチは Facade のみ** | `get_provider()` は `core/_model_factories.py` でのみ実行。Layer 2 以下でハードコードしない |
| **model.py は組み立てのみ** | 学習・OOF・metric・plot ロジックを `model.py` に書かない |
| **Optional dependency** | `torch` 等の重い依存は `try: import` + `LizyMLError(OPTIONAL_DEP_MISSING)` パターン |
| **Codegen は現在 LightGBM 専用** | `generator.py`, `artifact_writer.py`, `_model_persistence.py` が `LGBMAdapter` を直接参照。新アルゴリズムの codegen 対応には抽象化の拡張が必要 |

---

## 10. 実装チェックリスト

### Phase 1: 基盤

- [ ] `estimators/<name>/` ディレクトリ作成
- [ ] `adapter.py`: 6 抽象メソッド + オプショナルオーバーライド実装
- [ ] `provider.py`: 10 Protocol メソッド実装
- [ ] `defaults.py`: `_TASK_OBJECTIVE`, `_TASK_METRIC`, `_COMMON_DEFAULTS`, `default_space()`, `default_fixed_params()`

### Phase 2: Config 統合

- [ ] `config/schema.py`: Config クラス追加 + `ModelConfig` を Union に拡張（Proposal 必要）
- [ ] `core/_model_factories.py`: `get_provider()` に分岐追加
- [ ] Optional dependency ガード（`try: import` パターン）

### Phase 3: Smart Parameters（該当する場合）

- [ ] `smart_params.py`: `resolve_smart_params()`, `resolve_ratio_params()`
- [ ] Config バリデータ（Smart param と直接指定の競合検出）

### Phase 4: Metric Bridge（該当する場合）

- [ ] `metric_bridge.py`: 名前マッピング + ホワイトリスト + feval 生成
- [ ] `_FEVAL_METRICS` 定義（タスク別、アルゴリズムのネイティブ metric に応じて内容が異なる）

### Phase 5: Codegen 対応（任意）

- [ ] `save_model_text()` 実装
- [ ] `_build_params()` または同等のパラメータ抽出メソッド実装
- [ ] テンプレート分離（アルゴリズム別 `_TRAIN_PY_*`）
- [ ] feval wrapper のアルゴリズム別実装
- [ ] `_model_persistence.py`: ガード条件拡張 + `_extract_feval_metadata()` のディスパッチ追加
- [ ] `codegen/generator.py`: `model_adapter` 型の拡張
- [ ] `codegen/artifact_writer.py`: `model_adapter` 型の拡張 + アルゴリズム別保存ロジック

### Phase 6: テスト

- [ ] 契約テスト（shape / dtype / 戻り値の形）
- [ ] 再現性テスト（seed 固定）
- [ ] Early stopping テスト
- [ ] Provider 不変条件テスト（`test_check_provider.py` に追加）
- [ ] Config バリデーションテスト
- [ ] Tuning 統合テスト
- [ ] Codegen 等価テスト（Phase 5 実施時）

### Phase 7: ドキュメント

- [ ] HISTORY.md に Proposal（Config Union 変更 / 依存追加がある場合）
- [ ] BLUEPRINT.md の該当セクション更新
- [ ] README.md の対応アルゴリズム一覧更新
