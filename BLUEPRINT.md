# 0. ステータスとスコープ

## 0.1 ステータス

- 本ドキュメントは実装の単一の正とする（仕様変更は `HISTORY.md` の提案プロセスを経る）。
- 「仕様未確定は仮実装しない」を厳守する。
- `ARCHITECTURE.md` は本ドキュメントと実装から書き起こした**派生文書**であり、規範性を持たない。
  矛盾した場合は常に本ドキュメントが正で、`ARCHITECTURE.md` の側を直す（H-0092）。
  文書が述べるバージョン定数は `tests/test_docs/test_declared_versions.py` が
  コードの定義と突き合わせる。

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

- 再現性を最優先する。bit 一致保証（同一 `config + seed` で同一結果）は**固定 `(num_threads, CPU)` 環境**を前提とする。LightGBM の histogram 構築はスレッド数に依存するため、CPU / スレッド数が異なる環境間での bit 一致は保証範囲外とする（クロス環境再現性は将来 opt-in で提供しうるが、現状の defaults はスコープしない。詳細は H-0081）。
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
  - `model.importance(kind="split|gain|shap")`（特徴量重要度。全特徴量をキーに持つ `dict[str, float]` を返す。`shap` は optional dependency。`Model.importance(kind="shap")` の計算は §13.3）
  - `model.importance_plot(kind="split|gain|shap", top_n=20)`（特徴量重要度の可視化、Plotly）
  - `model.residuals()`（回帰専用。OOF 残差 `y - oof_pred` を `np.ndarray` で返す）
  - `model.residuals_plot(kind="scatter|histogram|qq|all")`（回帰専用。残差可視化、Plotly。IS/OOS 比較対応。デフォルト `kind="all"` で scatter + histogram + QQ の 3 パネル。scatter は Actual vs Predicted（x=predicted, y=actual）。IS サンプルは OOS 数に合わせてダウンサンプリング）
  - `model.evaluate_table()`（評価結果を `pd.DataFrame` で返す）
  - `model.roc_curve_plot()`（binary/multiclass。binary は IS/OOS の ROC Curve を重ね描き、multiclass はクラスごとの OvR ROC Curve を IS/OOS の subplot に並べる（§13.3、H-0019）、Plotly）
  - `model.confusion_matrix(threshold=0.5)`（binary/multiclass。IS/OOS の Confusion Matrix を `{"is": DataFrame, "oos": DataFrame}` で返す）
  - `model.calibration_plot()`（binary + calibration 有効時。Raw/Calibrated の Reliability Diagram、Plotly）
  - `model.probability_histogram_plot()`（binary + calibration 有効時。Raw/Calibrated の確率分布ヒストグラム、Plotly）
  - `model.tuning_plot()`（`tune()` 後。trial ごとのスコア推移と最良スコア推移を重ね描き、Plotly。完了/枝刈り/失敗を色分け）
  - `model.split_summary()`（fold ごとの分割情報を `pd.DataFrame` で返す。時系列分割時は期間情報を含む）
  - `model.params_table()`（解決済みパラメーターテーブル。Config smart params + resolved booster params + fold ごとの `best_iteration` を単一 `pd.DataFrame` で返す。`fit()` 未実行時は `MODEL_NOT_FIT`）
    - 形: index は `parameter`、単一列は `value`（H-0035）。
    - 行の順: Config の smart params（`auto_num_leaves` / `num_leaves_ratio` / ratio 等）、fold 0 の学習済み booster から読んだ解決済みネイティブパラメーター（`objective` / `metric` / `learning_rate` / `max_depth` / `num_leaves` / `min_data_in_leaf` / `min_data_in_bin` / `max_bin` / `feature_fraction` / `bagging_fraction` / `bagging_freq` / `lambda_l1` / `lambda_l2` / `num_iterations`。booster に無い名前は行を作らない。あれば `scale_pos_weight` / `num_class` / `feval_metrics` も）、training 設定（`early_stopping_rounds` / `validation_ratio`。その fit が使った値で、§14.4 の `build_export_params` と `applied_training_params` から読む）、fold ごとの `best_iteration_{i}`。
    - ratio の smart params と解決された絶対値は名前が違うので同じ表に並び、「指定した ratio」と「解決された絶対値」を対比確認できる。
  - `model.fit_result`（read-only プロパティ。`fit()` 後の `FitResult` の選択的 deep copy を返す（§7.1、H-0082）。`fit()` 未実行時は `MODEL_NOT_FIT`）
- 前提を満たさない呼び出しは `LizyMLError` で失敗する:
  - `Model.evaluate_table()` を `fit()` の前に呼ぶと `MODEL_NOT_FIT`（H-0005）。
  - `Model.residuals()` / `Model.residuals_plot()` は binary / multiclass で `UNSUPPORTED_TASK`（H-0006）。
  - `model.confusion_matrix()` は regression で `UNSUPPORTED_TASK`（H-0016）。`roc_curve_plot()` も regression で `UNSUPPORTED_TASK`（H-0019）。
  - `calibration_plot()` / `probability_histogram_plot()` は binary 以外の task で `UNSUPPORTED_TASK`、Calibration 未有効時に呼び出した場合は `CALIBRATION_NOT_SUPPORTED`（H-0017）。
  - `Model.tuning_plot()` を `tune()` の前に呼ぶと `MODEL_NOT_FIT`（H-0028）。
- `evaluate(metrics=None)` は内部の metrics dict の deep copy（`copy.deepcopy`）を返し、内部状態への参照を返さない。呼び出し側が戻り値を変更しても内部状態と後の `export()` は変わらない（H-0082）。
- `residuals()` / `residuals_plot()` / `importance(kind="shap")` / `roc_curve_plot()` / `confusion_matrix()` / `calibration_plot()` / `probability_histogram_plot()` は、`fit()` 後と `Model.load()` 後の両方で利用可能とする。
- `Model.load()` 後の上記 API は、Artifact に含める `analysis_context`（`y_true`, `X_for_explain`）を参照して動作させる。

## 4.2 `Model.load()`（Artifact 読込）

`export` で生成される `Model Artifact` をロードし、推論だけでなく学習時の評価情報や設定も参照できるようにする。

```python
loaded_model = Model.load("export_dir")
eval_result = loaded_model.evaluate()
pred_result = loaded_model.predict(X_new)
```

- `load()` は内部の metrics を `fit_result.metrics` の deep copy として持ち、`fit_result` プロパティが返す `FitResult` と可変オブジェクトを共有しない（`fit()` / `fit_result` と同じ隔離、H-0086）。
- tune 済みモデルの artifact では、`metadata.json` の `tuning` ブロックから trial の無い最小の `TuningResult` を復元し、load 後の再 `fit()` が tuned params を再現する。ブロックの無い artifact は tuning result 無しで読む（§15.1、H-0086）。

# 5. Config 設計

## 5.1 方針

- `pydantic`（`extra="forbid"`）で typo を確実にエラー化する。
- `config_version / schema_version` を必須にする。
- Config loader で以下を統一する。
  - 読込: `dict / JSON / YAML`
  - override: CLI / 環境変数（例: `LIZYML__model__lgbm__params__learning_rate=0.05`）
  - 正規化: 表記揺れの吸収（例: `k-fold` と `kfold`）、deprecated key の警告 / 拒否方針
    - `split.method` の別名は `config/loader.py` の表で正規化する（大文字小文字は区別しない）: `k-fold` / `stratified-kfold` / `stratifiedkfold` / `group-kfold` / `groupkfold` / `stratified-group-kfold` / `stratifiedgroupkfold`（→ `stratified_group_kfold`、H-0055）/ `time-series` / `timeseries` / `purged-time-series` / `purgedtimeseries`（→ `purged_time_series`）/ `group-time-series` / `grouptimeseries`（→ `group_time_series`、H-0032）。
    - deprecated key は受理して警告を出す（`calibration.n_splits` は `UserWarning`、他の Config キーは `DeprecationWarning`。文面は削除目標の版を明記する）。非推奨の面（`validation_ratio` の入力、`calibration.n_splits`、`purged_time_series` の `purge_window` / `embargo` / `embargo_pct` / `gap`、`PurgedTimeSeriesSplitter(embargo=...)` の引数、`build_calibration_splitter`）はいずれも v1.0 で削除する。削除目標の登録簿は `docs/DEPRECATIONS.md`（§15.2、H-0076）。
- 不正な Config は pydantic の `ValidationError` を包んだ `LizyMLError(CONFIG_INVALID)` になる（`cause` に元の例外、context `validation_errors`。`config/loader.py`）。どの入口（dict / ファイル / `Model(dict)`）でも同じである。
  - smart ratio の範囲バリデーション（`min_data_in_leaf_ratio` / `min_data_in_bin_ratio` は `(0,1)` の開区間、§5.3）に外れた値もこの経路で `CONFIG_INVALID` になる（早期エラー、H-0025）。

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

**この競合ルールが適用される入口（H-0094 / 実測）。** スマート解決はパラメーター dict のマージより後段で走り、その結果が勝つため、ルールが適用されない入口では「受理して黙って置換」になる。

| 入口 | 状態 |
|---|---|
| `model.params`（config） | `LGBMConfig._validate_smart_params` が parse 時に拒否。ただし **(a) `auto_num_leaves` / 2 つの ratio の 3 件のみ**（`balanced`→`scale_pos_weight` と `feature_weights`→`feature_contri` / `feature_pre_filter` は対象外）、かつ **(b) 文字列一致のみ**でエイリアスを見ない。**面の全体を実行して数えた（H-0094 決定 8 / 18 通り = スマートパラメーター × 書き込む native 名 × 受理綴り）: 拒否 3 / 2 綴りが `lgb.train` に届く 12 / 黙って上書き 3。** `max_leaves` / `min_child_samples` は通過して置換され、`scale_pos_weight: 10.0` は `balanced` により `0.951` で学習する（[#280](https://github.com/nbx-liz/LizyML/issues/280)）。`config/` は層規約上 `estimators/` を import できず学習器の別名表に届かないため、修正は「どこで拒否するか」の設計判断になる |
| `fit(params=...)` | **H-0094 で拒否する（5 件すべて）。** 有効なスマートパラメーターが書くネイティブ名は `CONFIG_INVALID` とし、どのスマートパラメーターが管理しているかを名指しする |
| `tuning.optuna.space` (`category: model`) | H-0099: reject names claimed by active smart parameters, including aliases, before study creation. Validate the resolved explicit/default/resumed space, including smart dimensions that can activate a conflicting owner. To tune native leaves directly, disable `model.auto_num_leaves`; sampled values then reach training. |
| `tuning.optuna.space` (`category: training`) | H-0099: accept only `early_stopping_rounds` and `validation_ratio`, the two training override consumers. Reject other names with `CONFIG_INVALID` before study creation. |

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
- 学習データの特徴量順に並び替えたリストに変換し、**LightGBM の `feature_contri` として**渡す（H-0093）。Config のフィールド名 `feature_weights` は利用者向けの名前であり、LightGBM 側の名前とは意図的に分けている。
- 副作用: `feature_pre_filter = False` を強制する。
- 制約: 重み `> 0` 必須。学習データに存在しない未知の特徴量名は `CONFIG_INVALID`。

### balanced（クラス重み自動均衡化）

- `balanced: bool | None = None`: 学習データのクラス比率から自動的に重みを算出する。
  - `None`（デフォルト）: タスク依存で自動解決（regression→`False`, binary/multiclass→`True`）。
  - `True`: binary は `scale_pos_weight = neg_count / pos_count` を設定。multiclass は `sample_weight` でクラス逆頻度重み付け。
    **重みは CV の各 fold と最終 refit の両方に同じ規則で掛かる**（H-0103）: 学習する行（inner valid があれば inner-train 行）だけが重みを持ち、inner-valid 行は重みなしの eval set である。
  - `False`: 重み均衡化を無効にする。
  - regression で `True` を指定した場合は `UNSUPPORTED_TASK`。

## 5.4 Config Reference（全キー一覧）

`config_version=1` で利用可能な全 Config キーの型・デフォルト・制約を以下にまとめる。

### トップレベル

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `config_version` | `int` | Yes | - | `1` のみサポート（`lizyml.config.loader.SUPPORTED_CONFIG_VERSIONS`、定義は `lizyml/config/version.py`）。dict / ファイル / `LizyMLConfig` インスタンス / 環境変数の上書きのどの経路でも検査し、サポート外は `CONFIG_VERSION_UNSUPPORTED`（H-0106）。`False` は `0` として拒否、`True` は `1` |
| `task` | `"regression" \| "binary" \| "multiclass"` | Yes | - | |
| `data` | `object` | Yes | - | |
| `features` | `object` | No | `{}` | |
| `split` | `object` | No | タスク依存 | binary/multiclass→stratified_kfold, regression→kfold |
| `model` | `object` | Yes | - | LightGBM のみ。`name` を判別キーとする discriminated union（`Field(discriminator="name")`、`LGBMConfig` は `name: Literal["lgbm"]` と `params` と smart params を持つ）。§5.2 の `{"lgbm": {...}}` 形は loader が `name` 形に変換し、`model_dump()` / `config_normalized` は `name` 形を書く（H-0001） |
| `training` | `object` | No | `{}` | seed=42, early stopping 有効 |
| `tuning` | `object \| null` | No | `null` | `tune()` 呼び出し時のみ必要 |
| `evaluation` | `object` | No | `{}` | |
| `calibration` | `object \| null` | No | `null` | binary 専用 |
| `output_dir` | `str \| null` | No | `null` | run ごとの出力先の基底ディレクトリ（§17）。コンストラクタ引数 `Model(..., output_dir=...)` が優先する（H-0039）。H-0034 は「`Config` の `output` セクション」と書いたが、実装は最上位のキー `output_dir` である（H-0034） |

### data

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `path` | `str \| null` | No | `null` | CSV/Parquet パス |
| `target` | `str` | Yes | - | 目的変数列名 |
| `time_col` | `str \| null` | No | `null` | 時系列列名（`time_series` / `purged_time_series` / `group_time_series` では必須） |
| `group_col` | `str \| null` | No | `null` | グループ列名（`group_kfold` / `stratified_group_kfold` / `group_time_series` では必須。未指定だと splitter が groups 無しで `ValueError` を送出する、H-0001） |

### features

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `exclude` | `list[str]` | No | `[]` | 除外列 |
| `auto_categorical` | `bool` | No | `True` | 自動カテゴリ検出 |
| `categorical` | `list[str]` | No | `[]` | 明示カテゴリ指定 |
| `unseen_policy` | `"mode" \| "nan" \| "error"` | No | `"mode"` | fit 時に無かったカテゴリの扱い（H-0104）。`mode` = 学習時の最頻値に置換、`nan` = 欠損に置換し、どちらも推論時は `PredictionResult.warnings` に報告する。`error` = `DATA_SCHEMA_INVALID`。データを変換するすべての場所に効く。§9.2 を参照 |

### split

`split.method` は以下のいずれか: `kfold` / `stratified_kfold` / `group_kfold` / `stratified_group_kfold` / `time_series` / `purged_time_series` / `group_time_series` / `blocked_group_kfold`。

| method | 固有キー |
|---|---|
| `kfold` | `n_splits=5`, `random_state=null`, `shuffle=True` |
| `stratified_kfold` | `n_splits=5`, `random_state=null` |
| `group_kfold` | `n_splits=5` |
| `stratified_group_kfold` | `n_splits=5`, `random_state=null`, `shuffle=True` |
| `time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |
| `purged_time_series` | `n_splits=5`, `purge_gap=0`, `train_size_max=null`, `test_size_max=null` |
| `group_time_series` | `n_splits=5`, `gap=0`, `train_size_max=null`, `test_size_max=null` |
| `blocked_group_kfold` | `blocks={col, cutoffs, mode, train_window}`, `groups={col, n_splits, stratify, shuffle}`, `min_train_rows=10`, `min_valid_rows=5` |

注記:
- `time_series` / `purged_time_series` / `group_time_series` は共通で `data.time_col` 必須。
- 3 メソッドは共通で `train_size_max` / `test_size_max` を受け取り、学習窓・検証窓の上限を制御する。
- `purged_time_series` の 2 つ目の gap の綴り `embargo` / `embargo_pct` / `gap` は、移行期間のみ旧キーとして受理し、値を `purge_gap` に**加算**する（H-0115）。`embargo` は `purge_gap` と同じ位置（学習の末尾）を引いていたため、つまみは `purge_gap` 1 つである。3 つの綴りは同時に 1 つまで（2 つ以上は `CONFIG_INVALID`）。値はキーがあれば `0` でも `DeprecationWarning`（文面は `purge_gap` を名指す）。値の読み方は綴りごとに以前と同じで、`embargo` は pydantic の int（bool は拒否）、`embargo_pct` / `gap` は観測数（下記）。`purge_gap` と 2 つ目の gap はそれぞれ 0 以上でなければならず（負は `CONFIG_INVALID`）、検査の後に加算する。`model_dump()` は `embargo` を書かず、`purge_gap` が合計を持つ。
- `purged_time_series` の旧キーは移行期間のみ警告付きで受理し（`DeprecationWarning`、v1.0 で削除、H-0076）、正規化する: `purge_window` → `purge_gap`（H-0038）。`embargo` / `embargo_pct` / `gap` は上記のとおり `purge_gap` に加算する（H-0115）。それ以前の正規化は `embargo_pct` → `embargo`（H-0040）、`gap` → `embargo`（H-0038 は `embargo_pct` に写すと決めたが、H-0040 で `embargo` が観測数の単位になってからは `embargo` に写した）だった。
- `random_state` の型は `int | None`（`KFoldConfig` / `StratifiedKFoldConfig` / `StratifiedGroupKFoldConfig`）。`null`（既定）なら splitter を構築するときに `training.seed` を継承し、明示した値はそれに勝つ。継承した値は Config に書き戻さない（`model_dump()` は `null` のまま、H-0080）。
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
| `optuna.params.direction` | `"minimize" \| "maximize" \| null` | No | `null` | Automatic orientation from the first effective evaluation metric; contradictory explicit values are CONFIG_INVALID (H-0099). |
| `optuna.params.timeout` | `float \| null` | No | `null` | |
| `optuna.space` | `dict[str, Any]` | No | `{}` | Per-dimension overrides in merge mode |
| `optuna.space_mode` | `merge \| replace` | No | `merge` | Merge provider defaults or use only explicit dimensions |

### evaluation

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `metrics` | `list[str]` | No | `[]` | ランタイムデフォルトあり |

### calibration

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `method` | `"platt" \| "isotonic" \| "beta"` | No | `"platt"` | |
| `n_splits` | `int` | No | `5` | **deprecated (H-0058)**: 無視される。calibration cross-fit は outer CV splits を再利用する。指定時は `UserWarning` を出力。 |

## 5.5 Config のキーが設定しない構成値（H-0108）

公開クラスの `__init__` の既定値付き引数（74 個、AST で数えた母集団）を、値の出どころで分類した。**Config のキーの値がそのまま渡る 59 個**は、どのキーから来るかを経路ごとに `tests/test_config/_knob_registry.py` の台帳に書く（§17 の `output_dir` と、§5.4 の表に行の無い `calibration.params` も含む）。`tests/test_config/test_knob_reachability.py` は、経路ごとに本物の `fit` / `tune` を実行してコンストラクタが受け取る値を確かめ、どの行にも既定でない値を設定したセルが少なくとも 1 つある。残りの 15 個をこの表に書く。表と `tests/test_config/_knob_registry.py` の台帳は同じテストが照合する。

分類は、そのクラスを構築するすべての本番の経路で読み、最初に当てはまるものを採る:

- **api**: Config のキーは設定せず、公開の呼び出しの引数が設定する。
- **derived**: ライブラリが、データから、または設定から 1 つの値を渡すのではない規則で決める。規則を書く。
- **policy**: ライブラリが値を固定している（定数を渡すか既定値のままにする）。Config のキーも公開の引数も変えられない。理由を書く。
- **internal**: 振る舞いの設定ではない（エラーの中身、部品どうしの配線）。

| 構成値 | 種類 | 値の出どころ / 理由 |
|---|---|---|
| `Model.data` | api | `Model(config, data=...)` |
| `Tuner.progress_callback` | api | `Model.tune(progress_callback=...)` |
| `Tuner.storage` | api | `Model.tune(storage=...)` |
| `Tuner.study_name` | api | `Model.tune(study_name=...)` |
| `PurgedTimeSeriesSplitter.embargo` | api | `PurgedTimeSeriesSplitter(embargo=...)` を直接構築したときだけ。非推奨で、`purge_gap` に加算され、v1.0 で削除する。Config のどのキーも渡さない（H-0115） |
| `LGBMAdapter.num_class` | derived | 目的変数のクラス数。multiclass のときだけ渡し、それ以外は `None` |
| `CVTrainer.n_classes` | derived | 目的変数のクラス数。multiclass のときだけ渡し、それ以外は `None` |
| `CVTrainer.collect_raw_scores` | derived | `fit` では `calibration` が設定されているか（較正は生のスコアで学習する、H-0030）。`tune` の trial では常に `False`（trial の評価は較正しない） |
| `LGBMAdapter.verbose_eval` | policy | `-1`: LightGBM の反復ごとの評価ログを出さない。学習曲線は `FitResult.history` に記録されるので、ログは情報を増やさず出力を埋める |
| `StratifiedKFoldSplitter.shuffle` | policy | `True`（構築箇所が定数で渡す）: `stratified_kfold` は常に行をシャッフルしてから層化する（順序は `split.random_state`、それが無ければ `training.seed` が決める）。行の順序に意味があるデータには時系列の分割を使う |
| `LizyMLError.debug_message` | internal | エラーの中身。各 raise 箇所が設定する |
| `LizyMLError.cause` | internal | エラーの中身。各 raise 箇所が設定する |
| `LizyMLError.context` | internal | エラーの中身。各 raise 箇所が設定する |
| `CVTrainer.ratio_param_resolver` | internal | 配線: 比率で書いたスマートパラメーターを fold ごとに解決する |
| `RefitTrainer.ratio_param_resolver` | internal | 配線: 比率で書いたスマートパラメーターを解決する |

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
   - `TrainComponents.sample_weight` も両方に渡す（H-0103）。`RefitTrainer.fit` が `CVTrainer.fit` から受け取らない入力は次の 3 つで、いずれも方針である: `time_values`（時間順の split では両 trainer の前に全行が並べ替え済みで、CV はこれを fold ごとの時間範囲の記録にしか使わない）、`data_fingerprint` と `run_meta`（fit 1 回につき 1 つ、同じデータから `FitResult` に記録する）。入力差は `tests/test_training/test_cv_refit_parity.py` が両方の signature から検査する。
9. `FitResult` を返し、Artifacts を保持する。
   - 返すのは内部の `FitResult` ではなく、その選択的 deep copy（`FitResult.__deepcopy__`、§7.1）である。戻り値を変更しても内部状態と後の `export()` は変わらない（H-0086）。

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
- **train.py**: Feature pipeline fit → LightGBM refit（全データ学習）→ OOF 生成（軽量 CV。fold は `config.json` の `split` ブロックから outer CV を再現する、§15.4）→ Calibrator fit
- **predict.py**: Feature transform → LightGBM predict → Calibration apply
- **config.json**: ハイパーパラメータ・特徴量定義・校正設定を集約。コード編集なしでパラメータ変更可能
- 生成コードは `import lizyml` を含まない。依存は `lightgbm` / `numpy` / `pandas` / `scikit-learn`（学習時のみ）
- `test_equivalence.py` で `Model.predict()` と codegen 出力の一致を `rtol=1e-7` で検証
- 初期実装は LightGBM のみ対応。将来の EstimatorProvider 拡張で他アルゴリズムにも対応可能
- Calibrator 保存形式: Platt → JSON (a, b)、Beta → JSON (a, b, c)、Isotonic → Booster テキスト
- **feval metric 対応（H-0066）**: ユーザー指定の feval metric（f1, brier, ece, precision_at_k, accuracy, rmsle, r2, smape, wape。`smape` / `wape` は H-0071）を `config.json` の `feval_metrics` フィールドに記録し、`train.py` 内に pure numpy で再実装する。feval metric 未使用時は `feval_metrics: []` で後方互換を維持

# 7. Artifacts（戻り値と保存対象の固定）

## 7.1 FitResult（固定スキーマ）

`FitResult` / `PredictionResult` / `SplitIndices` / `RunMeta` は `dataclass` であり、契約を固定する golden test は `dataclasses.fields` でフィールドの集合を照合する（H-0002）。

- `oof_pred`（`np.ndarray`。shape は regression / binary で `(n_samples,)`、multiclass で `(n_samples, n_classes)`）
- `if_pred_per_fold`（`list[np.ndarray]`。長さは `n_splits`、各要素はその fold の学習行（`train_idx`）全体に対する予測）
- `metrics`（階層固定）
  - 例: `{"raw": {"oof": {...}, "oof_per_fold": [...], "if_mean": {...}, "if_per_fold": [...], "oof_coverage": 1.0}, "calibrated": {...}}`
  - `oof_per_fold` / `if_per_fold` の長さは `n_splits`（fold の順）。
  - `oof_coverage`（float, 0.0–1.0）: validation fold に覆われた行の割合（H-0057）。KFold では常に `1.0`、TimeSeriesCV では `< 1.0` になりうる。`calibrated` は raw と構造一致のため別途含めない（H-0058）。
- `models`
  - fold ごとのモデル
  - 任意: refit モデル（全データ学習）
- `history`
  - fold ごとの eval history / best_iteration
- `feature_names / dtypes / categorical_features`
  - `feature_names`: 学習に使った特徴量名の順序付き `list[str]`。`dtypes`: 特徴量名 → dtype 文字列の `dict[str, str]`。`categorical_features`: カテゴリとして扱った特徴量名の `list[str]`。
- `splits`
  - 外側 CV indices（必須、元データ基準の absolute index）
  - inner valid indices（有効時必須。各 outer fold train に対する 0-based 相対 index）
  - calibration CV indices（有効時必須）
  - `time_range`（時系列分割時。fold ごとの train/valid の期間情報 `list[dict] | None`）
- `data_fingerprint`
  - `row_count / column_hash / optional: file_hash` 等
- `pipeline_state`（最後の CV fold の `FeaturePipeline` の状態、必須）
- `pipeline_state_per_fold`（`list | None`。各 CV fold の `FeaturePipeline` の状態を fold の順に持つ。`CVTrainer` は常に埋め、長さは外側 CV の fold 数、最後の要素は `pipeline_state` と同じ。`None` は fold ごとの状態が無いこと（H-0114 より前の artifact、またはこのフィールドを省いて構築した `FitResult`）を表す。H-0114）
- `calibrator`（有効時）
- `run_meta`
  - `lizyml_version / python_version / deps_versions / config_normalized / config_version / run_id / timestamp`
  - 型: `lizyml_version`（`str`）、`python_version`（`str`）、`deps_versions`（`dict[str, str]`）、`config_normalized`（`dict`）、`config_version`（`int`）、`run_id`（`str`、UUID 文字列）、`timestamp`（`str`、ISO 8601、UTC）。
- `target_encoder`（H-0070, format_version=2）
  - `TargetEncoder(classes_: tuple[Any, ...], needs_encoding: bool, original_dtype: str)`
  - 数値 y / regression では `needs_encoding=False` の no-op
  - 非数値 classification y では `classes_` に **lexicographically** sorted な元ラベルを保持
    （`sorted(unique, key=str)`、numeric-string では自然順とは一致しない: `["1","10","2"]` → `("1","10","2")`）
  - `predict()` / codegen / persistence migration で利用

公開の戻り値（`Model.fit()` と `Model.fit_result`）は選択的 deep copy（`FitResult.__deepcopy__`）である（H-0082 / H-0086）:

- data フィールド（`metrics` / `history` / `splits` / 配列など）は deep copy する。呼び出し側が変更しても内部状態と後の `export()` は変わらない。
- 学習済みの `models` / `calibrator` / `pipeline_state` は参照を共有し、慣例として read-only とする（LightGBM Booster の deep copy はモデル文字列を往復して `params` を失うため）。`models` はリストの容器だけを新しくする。
- `pipeline_state_per_fold` も同じく各 fold の状態を参照で共有し、慣例として read-only とする。リストの容器だけを新しくする（H-0114）。
- 呼び出すたびに別のオブジェクトを返す（同一性は保たない）。
- `FitResult` は frozen でない `dataclass` のままにする。内部状態は戻り値を copy することで守り、型を frozen にはしない（H-0082）。

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

- `pred`（`np.ndarray`、shape `(n_samples,)`。回帰: 予測値、分類: クラスラベル。binary は `proba >= 0.5`、multiclass は `proba` の argmax を元のラベルに戻したもの）
- `proba`（binary: 正例の確率、shape `(n_samples,)`。校正が有効なら `C_final` を通した値。multiclass: shape `(n_samples, n_classes)`。regression: `None`）
  - multiclass の `(n_samples, n_classes)` は実装が返していた公開の振る舞いで、提案なしに広がっていたものを H-0110 で決定として記録した。
- `shap_values`（要求時のみ（`return_shap=True`）、それ以外は `None`。shape は task を問わず `(n_samples, n_features)` に統一し、multiclass はクラス方向の mean(|SHAP|) で縮約する）
- `used_features`（予測に使った特徴量名。refit が学習した `feature_names`。列ズレ検知用）
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
- `applied_training_params`（`metadata.json` の最上位キー。artifact のモデルを作った fit が適用した training overlay。overlay を使わなかった fit は `{}`。H-0109）

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

- 時系列: ソート、未来情報混入疑い、**outer split の** shuffle 禁止（inner valid に時間順を崩す `method: holdout` を明示指定した場合は、禁止せず `UserWarning` を発して明示指定を尊重する。§10.3.1 参照）
- group: group 跨ぎ、分割条件の不整合
- leakage: target リーク疑い（例: target と完全一致の列、時間逆転など）
- target dtype: regression × 非数値 y は `TARGET_NOT_NUMERIC`（H-0070）
- target の欠損: 数値 target に NaN があれば `Model.fit` の入口で `LizyMLError(DATA_SCHEMA_INVALID)`（context `nan_count`）を送出する（H-0085）。分類 target の NaN も同じ code で拒否する。
- 公開 API: `lizyml.data` が `validate_time_series_order` / `validate_no_target_leakage` / `validate_group_split` を公開する（H-0087）。**`Model.fit` には自動配線しない**。利用者が明示的に呼ぶ（自動配線は通過中の config に新たに警告や例外を出す挙動変更なので、別の提案で扱う）。
- `validate_no_target_leakage(df, target, *, raise_on_violation=True)` の規則（H-0107）:
  - 列を順に target と比べる。`raise_on_violation=True` では最初に見つかった漏洩列で `LEAKAGE_SUSPECTED` を即座に送出する。戻り値（`[]` か警告のリスト）が返るのは、全列を比べ終えたときだけである。
  - 比較が例外を出した列は、`raise_on_violation` に関わらず `LizyMLError(DATA_SCHEMA_INVALID)`（context `column` / `target`、`cause` は元の例外）。検査できなかった列を検査済みとして扱わない。
  - 捕まえる範囲は比較の呼び出し 1 つだけで、例外の型は問わない（拡張配列の `OverflowError` / `AttributeError` も `DATA_SCHEMA_INVALID` になる）。`LEAKAGE_SUSPECTED` はその範囲の外で送出する。
  - target 列が `df` に無ければ（有無は完全なラベルで調べる。`MultiIndex` の部分キーは無い列として扱う）、どの列も比べずに、`raise_on_violation` に関わらず `LizyMLError(DATA_SCHEMA_INVALID)`（context `target` / `missing_columns` / `available_columns`）を送出する。したがって戻り値が返るのは、target 列があり、全列を比べ終えたときだけである（H-0112）。
- `validate_time_series_order(df, time_col, *, raise_on_violation=True)` も、`time_col` が `df` に無ければ並びを調べずに、`raise_on_violation` に関わらず `DATA_SCHEMA_INVALID`（context `time_col` / `missing_columns` / `available_columns`）を送出する。並びが非減少でなければ `LEAKAGE_SUSPECTED`（`raise_on_violation=False` なら警告のリスト）、非減少なら `[]`（H-0112）。

## 8.3 Data fingerprint（必須）

- `row_count`
- `column_hash`（列名 + dtype + 順序から作る）
- optional: `file_hash`（読み込んだファイルのハッシュ）

# 9. FeaturePipeline（`features/`）

## 9.1 必須要件

- `fit(X, y) / transform(X) / fit_transform(X, y)` の IF を固定する。
- 状態（state）の永続化を必須にする。
- OneHot のカテゴリ辞書、欠損補完統計量、target transform パラメータ等を保持する。
- `BaseFeaturePipeline.transform_with_warnings(X)` は具体メソッドで、既定の実装は `(self.transform(X), [])` である（H-0104）。推論経路と SHAP はこれを呼び、返した警告は `PredictionResult.warnings` に届く。入力を補正する pipeline（列を落とす、値を置換する）は上書きして報告する。抽象にしないのは既存の外部サブクラスを壊さないためである。
- `get_state()` の `"categorical_cols"` キーは任意で、estimator にカテゴリとして渡す出力列の宣言である。trainer はこのキーだけを読み、無ければカテゴリ列なしとして学習する（H-0104）。

## 9.2 列ズレ方針（仕様として固定）

- 余剰列: デフォルト無視（警告） or エラー（オプション）
- 不足列: デフォルトエラー（安全側）
- unseen category:
  - OneHot: unknown 用カテゴリ or all-zero（ポリシー選択）
  - LGBM native categorical: `features.unseen_policy`（既定 `"mode"`、H-0104）で選ぶ。
    - `mode` / `nan` の置換は**補正**であり、推論時は列・値・置換先を `PredictionResult.warnings` に報告する（§7.3）。欠損値は未知カテゴリではなく、どの方針でも欠損のまま残す。
    - 方針は fit が適用したものを pipeline 状態に保存し、推論時と生成コード（`predict.py`、再学習後の `train.py` も）はその値に従う。
    - **CV の検証 fold にも同じ方針が効く。** ただし既定の `auto_categorical: true`（または `categorical` 指定）の列はデータ構築時に全行の値で `category` 型になるため、どの fold でも未知にならない。fold ごとに未知カテゴリが生じるのは、`auto_categorical: false` で pipeline がカテゴリとして扱う文字列列だけである。その場合 `error` では fit が止まり、`mode` / `nan` の置換は fit 中は報告されない（`FitResult` に警告の通り道が無い）。
    - SHAP 重要度は、fold k の検証行を fold k 自身の pipeline 状態（`FitResult.pipeline_state_per_fold`）で変換して fold k のモデルを説明する（H-0114）。上記の文字列列でも、SHAP 重要度が見る符号化と置換は OOF 予測が見たものと同じで、fit 中と同じく報告しない。`error` では未知の値があれば fit が先に止まるので、fit が通ったモデルの SHAP 重要度はこの理由で送出しない。
- 推論時の列検査（不足列は `DATA_SCHEMA_INVALID`、余剰列は警告して除外）は pipeline に渡す**前**に facade（`Model.predict`）で行い、どの pipeline 実装でも省略されない（H-0104）。
- 同じ検査で、学習時に数値だった列（`FitResult.dtypes` の記録）が予測時に数値でない dtype で届いたら `INCOMPATIBLE_COLUMNS` を送出する（context `columns` の各要素は `column` / `fit_dtype` / `predict_dtype`。H-0106）。不足列の `DATA_SCHEMA_INVALID` が先に効く。学習時に `category` だった列はこの dtype の規則で検査しない。

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
- `purged_time_series` は前向き連鎖（expanding window）で、どの fold でも学習の最大 index は検証の最小 index より小さい（H-0115、`tests/test_splitters/test_purged_embargo_merge.py` が固定する）。検証の後ろに学習行が無いので、文献の embargo（検証ブロックの**後ろ**にある学習行の先頭を除く）を置く場所は無い。以前の `embargo` は `purge_gap` と同じ位置を引く 2 つ目のつまみだったので、`purge_gap` に統合した（§5 の旧キーの規則）。後ろ向きの embargo が必要になれば、検証ブロックの後ろに学習行を持つ別の splitter が要る。`PurgedTimeSeriesSplitter(embargo=...)` の引数も非推奨で、指定されれば `0` でも警告して `purge_gap` に加算する（負は `ValueError`、v1.0 で削除）。旧キー `embargo_pct` / `gap` の値は観測数として読む: 整数値（`3` / `3.0`）は受理し、端数のある値（`0.05`）と bool は `CONFIG_INVALID` で拒否する（`int()` で切り捨てると漏洩防止の gap が `0` に潰れるため、#210。H-0040 は `int()` で変換すると書いた。H-0111）。
- 3 メソッドは共通で `train_size_max` / `test_size_max` を持つ。
- `GroupTimeSeries` は group 列の出現順と `time_col` 順を整合させて時系列的にグループを分割する。

## 10.3 InnerValidStrategy（early stopping 用）

- CV fold 内でさらに `train / valid` を作る概念を分離する。
- `InnerValidStrategy` は `Model._build_train_components()` で解決し、`CVTrainer` と `RefitTrainer` に同一インスタンスを渡す。
- `early_stopping.enabled=False` の場合は `NoInnerValid` を使い、inner split を生成しない。

### 10.3.1 設定の解決規則

- `training.early_stopping.inner_valid` を明示指定した場合は、その method / ratio / random_state をそのまま使う。外側 `split.method` は参照しない。ただし、shuffle を伴う `method: holdout` を time-ordered な outer split（`time_series` / `purged_time_series`）と組み合わせた場合は、early stopping split が時間順を守らず temporally leak し得るため `UserWarning` を発する（挙動は変えず明示指定を尊重する。H-0085 / #210）。
- `training.early_stopping.validation_ratio` は legacy 入力ショートハンドであり、method 指定ではない。`inner_valid` を明示指定していない場合、ratio は `validation_ratio` から取り、method は外側 `split.method` から自動解決する。H-0069 以降、`validation_ratio` は `inner_valid.ratio` から派生する read-only の computed field。出力 (`model_dump()`) には常に同値の `validation_ratio` が含まれる。
- `validation_ratio` と `inner_valid` を同時に明示指定した場合、ratio が一致しなければ `CONFIG_INVALID`、一致すれば許容する（round-trip 互換）。一致しない場合の検知は維持される。
- `inner_valid` を明示したかどうかは dump / reload と export → load → fit を越えて保たれる（H-0086）。`LizyMLConfig.model_dump()` は computed field の `inner_valid_explicit` を書き、検証はこのキーを（`validation_ratio` と同じく）取り除いて、明示性の正とする。キーが無い入力では上記の推定（`inner_valid` があり `validation_ratio` が無ければ明示）に戻る。設定できるフィールドは増やさない。明示した time / group の inner valid が再読込で自動解決に化けないための規則である。
- 自動解決時に inner valid が継承する outer CV 設定は `split.method` と、look-ahead 防止のための境界 gap である。すなわち `purged_time_series` では `purge_gap`（H-0085 の時点では `purge_gap + embargo` だったが、H-0115 で `embargo` を `purge_gap` に加算する旧キーにしたので、量は同じ）、`time_series` では `gap` を inner valid（`TimeHoldoutInnerValid`）の inner-train と inner-valid の間に purge する（H-0085 / #212）。`n_splits` / `shuffle` / `random_state` / `train_size_max` / `test_size_max` は inner valid に伝搬しない。
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
  - `sample_weight` がある場合、estimator に渡すのは inner-train 行の重みだけで、inner-valid 行は重みなしのeval set とする（CVTrainer と同じ規則、H-0103）。inner valid が無い場合は全行の重みを渡す。
  - **pipeline は全データで 1 回のみ fit する**（H-0085）。CVTrainer が outer fold の `train` 全体で pipeline を fit するのと同じ境界であり、Refit における「outer train 全体」は全データに相当する。inner-train には狭めない。
  - estimator は inner valid がある場合、変換後データから slice した `inner_train` で学習し、`inner_valid` を eval set として early stopping を行う。
  - 最終的な `pipeline_state`（推論用）および `categorical_features` は、この全データ fit 済み pipeline から取得する（二重 fit は行わない）。
  - inner valid が無い場合（`NoInnerValid`）も、pipeline は全データで 1 回 fit し、estimator を全データで学習する。
  - `time_series` / `purged_time_series` / `group_time_series` では、`Model._prepare_training_data()` により時系列昇順へ並べ替えた後の全データに対して inner valid を切る。

### 10.3.3 各 strategy の分割規則

- `HoldoutInnerValid(ratio=0.1, random_state=42, stratify=False)`:
  - `stratify=False`: outer fold train 行を乱択し、`ceil(n_rows * ratio)` 行を validation に割り当てる。
  - `stratify=True`: `y` に基づく stratified holdout を行う。
  - `n_valid >= n_samples` の場合は `ValueError` を発出する（空の train set 防止）。
- `GroupHoldoutInnerValid(ratio=0.1, random_state=42)`:
  - group overlap を禁止する。
  - validation には、入力順（group の first appearance 順）の末尾 `max(1, floor(n_unique_groups * ratio))` 個の group を割り当てる。
  - shuffle は行わないため、`group_time_series` では time-sort 後の入力順に従って末尾 group が validation になる。
- `TimeHoldoutInnerValid(ratio=0.1, gap=0)`:
  - 行順を保持したまま、末尾 `max(1, floor(n_rows * ratio))` 行を validation に割り当てる。
  - `gap > 0` のとき、inner-train と inner-valid の境界で `gap` 行を purge する。purge された行は
    inner-train にも inner-valid にも属さない。
  - `gap` の値は §10.3.1 の解決規則が決める。**自動解決時**は outer split の境界 gap を継承する
    （H-0085）。**`training.early_stopping.inner_valid` を明示指定した場合は継承せず、常に
    `gap=0`** である。§10.3.1 の「明示指定した場合は外側 `split.method` を参照しない」に従う。
  - **`gap` は Config のフィールドではない（H-0101）。** `TimeHoldoutInnerValidConfig` は
    `method` と `ratio` だけを持ち（`extra="forbid"`）、`gap` を書けば検証で拒否される。`gap` は
    自動解決だけが設定する構築子引数であり、利用者が inner valid の境界 gap を直接指定する手段は
    意図的に設けていない。境界 gap が必要な場合は、outer split に `purge_gap` / `gap` を
    設定し、inner valid を自動解決に任せる。
  - `n_valid + gap >= n_samples` の場合は `ValueError` を発出する（空の train set 防止）。
- `BlockedGroupInnerValid(ratio=0.1, task="regression")`:
  - `blocked_group_kfold` 専用。グループ分離 + 時間順序 + 層化（分類時）を同時に満たす。
  - `task` は層化の要否を決める（`binary` / `multiclass` で層化、`regression` では行わない）。
  - グループ数 < 4 のときはフォールバックする。フォールバック先も `task` が決める：
    `binary` / `multiclass` は `StratifiedTimeHoldoutInnerValid`、**`regression` は
    `TimeHoldoutInnerValid`**。回帰で層化フォールバックを選ぶと、連続値の `y` は
    1 行 1 クラスとなり全行が validation に入って inner-train が空になる。
  - 詳細は §10.6.2 を参照。
- `StratifiedTimeHoldoutInnerValid(ratio=0.1)`:
  - `BlockedGroupInnerValid` の分類タスク時のフォールバック（グループ数 < 4 時）。
  - 各クラス内で時間順序を保持し、末尾 `ratio` 分を validation に割り当てる。全クラス最低1行を保証する。
  - **層化できない `y`（クラスあたり 1 行しかない連続値など）で train が空になる場合は
    `ValueError` を発出する。** 回帰の inner valid には `TimeHoldoutInnerValid` を使う。

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
- validation 行のスコアに NaN が混じる fold では、有限の行だけを校正し、NaN の行には fallback（未校正の OOF 確率）を書く。validation 行がすべて NaN の fold では全行に fallback を書く（`calibration/cross_fit.py`、H-0067）。outer splits を再利用する現行の構成では validation 行はすべて covered なのでこの経路は通らず、calibration split が outer split と異なる場合の防御である。fallback の行数は §12.1 の報告に含まれる。

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

**フォールバック**: `n_unique_groups < 4` の場合、警告を出した上で切り替える。切り替え先は `task` が決める。

- `binary` / `multiclass`: `StratifiedTimeHoldoutInnerValid`（各クラスの末尾行から `ratio` 分）。
- `regression`: `TimeHoldoutInnerValid`（末尾 `ratio` 分の行）。連続値の `y` は層化すると 1 行 1 クラスとなり、全行が inner valid に入って inner train が空になるため、層化フォールバックは選べない。

**明示指定**: `training.early_stopping.inner_valid` を明示指定した場合は auto 解決を上書きする。

# 11. Tuning（`tuning/`）

## 11.1 SearchSpace 表現の統一

- Optuna に依存しない space 表現（離散・連続・対数・カテゴリ）を使う。
- `parse_space` は `float` / `int` 次元の範囲を parse 時に検査する（H-0078）: `low >= high` は `CONFIG_INVALID`、`log=True` で `low <= 0` も `CONFIG_INVALID`（context `param` / `low` / `high` / `log`）。Optuna が trial の中で出す汎用のエラーより前に、設定の誤りとして名指す。

## 11.2 SearchDim カテゴリ

SearchDim にカテゴリ属性を持たせ、Tuner がパラメーターの適用先を区別する。

- `model`: `LGBMAdapter.params` に直接渡す（既存 SearchDim の挙動）
- `smart`: スマートパラメーター（`num_leaves_ratio` 等）として `resolve_smart_params()` に渡す。fit / tune で同一の dict ベース `resolve_smart_params()` を使用する（H-0050）
- `training`: trial ごとに `EarlyStoppingConfig` / `InnerValidStrategy` を再構築する

`FloatDim` / `IntDim` は任意の `min_allowed` / `max_allowed`（既定 `None`）を持つ。パラメーターとして意味のある範囲で、境界拡張（§11.5）がこれを超えないようにする。`Model.tune` は既定の空間にも利用者の空間にも `provider.parameter_bounds(task)`（§14.4）を付ける（H-0078）。

## 11.3 デフォルト Tuning Space

H-0102: `tuning.optuna.space_mode` defaults to `merge`. Start with task-specific
provider dimensions, replace matching dimensions with complete user definitions,
and retain unspecified defaults. Model dimensions match by provider parameter
identity, so `eta` replaces `learning_rate`; smart/training dimensions match by
category and name. Additional user dimensions are appended. User aliases for
one parameter in the same space remain invalid.

Set `space_mode: replace` for an explicit-only space, including an empty space.
This is the migration path for pre-H-0102 nonempty spaces. Merge can increase
training cost by adding default dimensions and retains the existing smart/native
conflict refusals. Start a fresh study when changing space semantics.

Merge applies provider fixed defaults with precedence base < fixed < sampled;
replace applies no fixed defaults. Resume retains the resolved space, bounds and
fixed-default policy of the previous successful round. Fresh rounds use their
new fixed policy without inheriting the previous round's fixed defaults.
Fresh studies also start from Config model/smart parameters rather than previous
best parameters. Resume retains its successful tuning overlay and resolved space.
Fresh-study training conflict checks likewise exclude prior training overrides.
Admission also checks training ownership introduced by the current resolved
space against effective native parameters and model dimensions before study
creation, including inherited defaults and aliases.
Export/load retains the effective policy independently of subsequent Config
changes; an explicit empty policy differs from absent legacy metadata.
Automatic boundary
expansion remains enabled only for merge with an empty/omitted user space;
partial spaces require explicit opt-in. Older saved artifacts without the mode
retain replacement for nonempty spaces when loaded, preserving re-fit behavior.

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

### `categorical` の `choices` の型（H-0095 の決定、[#287](https://github.com/nbx-liz/LizyML/issues/287)）

`choices` の各要素は**素の Python スカラー**（`None` / `bool` / `int` / `float` / `str`）で
なければならず、**判定は型の同一性で行う**（`type(v) is t`。`isinstance` でも
`type(v) in (...)` でもない —— 前者は `np.float64` / `np.str_` を通し、後者は
**メタクラスが書ける `__eq__`** に依存する。**タプルの `in` はハッシュを引かない**ので、
§14.4 が `set` について述べている `__hash__` の話はここには当たらない
（round 30 の指摘、実測で確認）。残る理由 —— 呼び出し元が等価を書ける —— は同じである）。
**numpy スカラーは拒否する** —— `choices` は 4 surface の正規化を通らないため、
通せば adapter の出口表明が study の内側で拒否し、利用者には `TUNING_FAILED` しか見えない。

`float` / `int` 次元の `low` / `high` は parse 時に `float()` / `int()` で強制変換するので
この規則の対象外である。**なお 4 surface が numpy を受理するのに探索空間が拒否する
という差は残っている** —— それを埋めるかどうかは #287 の判断。

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
- 拡張後の `new_low` / `new_high` は次元の `min_allowed` / `max_allowed`（§11.2）でクランプする。クランプが効いたら `BoundaryDimStatus.clamped_to_bound` を `True` にする（H-0078）
- 範囲を変えない拡張は拡張ではない: クランプ（`min_allowed` / `max_allowed`、linear の `0.0` 下限、`IntDim` の `max(1, ...)`）と `IntDim` の丸めのあとで `(new_low, new_high)` が元の `(low, high)` と等しい次元は、端に近くても `expanded=False`・`new_low` / `new_high` は `None` とし、`expanded_names` にも `RoundSummary.expanded_dims` にも入れない。`clamped_to_bound` は `min_allowed` / `max_allowed` のクランプが効いたときだけ `True` で、再判定のあとも `True` のまま残る（linear の `0.0` 下限と `IntDim` の `max(1, ...)` は `clamped_to_bound` を立てないので `False` のまま）。`IntDim` の拡張と比較は整数で計算する（`2**53` を超える値でも隣の整数と混ざらない）。端の検出（位置の計算）は H-0068 のとおり float で行うので、`2**53` を超える `IntDim` の範囲では端を見落としうる（H-0111 の bound）。毎ラウンド同じ空の拡張を報告し続けないため（H-0078 項目 4、H-0111）
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
    clamped_to_bound: bool = False   # H-0078: 拡張が min_allowed / max_allowed に当たった

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
- Tuner は study オブジェクトの受け取り・返却に対応する。study の永続化（disk-backed storage）は §11.5.1 で扱う。

## 11.5.1 Persistent Storage（H-0072）

長時間 tuning ジョブが process kill / 再起動 / ネットワーク断で中断した際に、journal / RDB に永続化された trial 状態から **完了済 trial を再実行せずに resume** できる仕組みを提供する。Optuna 標準の `JournalStorage` / `RDBStorage` を薄く pass-through する設計とする。

### Tuner / Model.tune() の追加パラメーター

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `storage` | `None` | Optuna storage URL（`sqlite:///path/to.db` 等）または `BaseStorage` インスタンス。`None` で in-memory（従来挙動） |
| `study_name` | `None` | Study 識別子。`storage` 指定時は必須（fail fast） |

`storage=None` で完全後方互換（disk IO ゼロ）。`storage` 指定時は `optuna.create_study(..., storage=..., study_name=..., load_if_exists=True)` で idempotent に再アタッチする。

### resume との関係

- `Model.tune(resume=False, storage=...)`: 同一 process 内で初回呼び出し時、`load_if_exists=True` により journal に既存 study があれば再アタッチ（crash recovery）。journal に該当 study が無ければ新規作成
- `Model.tune(resume=True, ...)`: 既存の `Model._study` を再利用（in-memory / disk-backed どちらでも動作）
- `storage` を指定し続ける限り、各 trial 完了直後に backend に追記される

### 制約

- `storage is not None and study_name is None` → `LizyMLError(CONFIG_INVALID)`
- `RoundSummary` / `BoundaryReport` 等のラウンドメタは journal には保存されない（trial 単位 resume のみが対象）。round 履歴の永続化は別 Proposal で扱う。
- H-0099: supplied and persisted studies must match the resolved objective direction. Reject mismatches with `CONFIG_INVALID` before enqueue or optimization; use a new study name instead of relabeling an existing study.

### Objective direction (H-0099)

The first configured evaluation metric, or the first task default when omitted,
is the objective. Omitted/null `tuning.optuna.params.direction` derives from that
metric's `greater_is_better`, including parameterized metric entries. Explicit
directions must agree or fail with `CONFIG_INVALID` before creating a study.
Null remains automatic through Config dump/load; exported Result directions
are always the resolved `minimize` or `maximize`. Historical configs containing
contradictory explicit directions require migration; old explicitness is not guessed.

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
- 推論時は保存された `C_final` を使用する。入力は `predict_raw(X)` の生スコアで、`BaseCalibratorAdapter.predict()` は生スコアを受け取り校正済み確率を返す（H-0030）。`oof_raw_scores=None` の旧 artifact では確率を入力にする（§18.1.4）。
- cross-fit で校正できない行は未校正の OOF 確率を fallback として受け取る: その fold の学習行に covered なスコアが無い、学習行が 1 クラスだけ、validation のスコアが NaN（§10.5）。fallback の行は `calibrated_oof` と calibrated metrics に含まれたまま残り（値は H-0089 で変わらない）、その数を `CalibrationResult.n_fallback_rows` と `metrics["calibrated"]["fallback_row_count"]` が報告する（H-0089）。
- Calibration が未指定の場合は従来どおり `predict_proba`（確率値）を OOF/IF 予測に使用する。Calibration 有効時のみ生スコアベースの校正パスに入る。

## 12.2 方法

- Platt Scaling
- Beta Calibration
- Isotonic Regression（LGBM の単調制約利用）

**`calibration.params` は 3 手法すべてで、それを消費する calibrator に届くか、学習開始前に拒否される（H-0100）。** 各 calibrator が `validate_params` で受理契約を宣言し、Facade が `fit()` / `tune()` のマージ直後、Booster も study も学習する前に呼ぶ。値は 3 手法すべてで入口正規化される（H-0095）。LightGBM の別名の正規名化は isotonic のみ。生成コード（§15.4）も同じ設定で calibrator を再構築する。

**最適化が収束しなかった校正は使わない（H-0113）。** Platt / Beta の `minimize` が `success` を偽で返したら（反復や関数評価の上限、線探索の失敗。理由は区別しない）、その係数を校正器に残さずに（refit なら前の fit の係数も消して）`LizyMLError(CALIBRATION_FAILED)` を送出する（context `calibrator` / `method` / `message` / `status` / `nit`）。cross-fit は fold の失敗に `stage="cross_fit"` と `fold`、C_final の失敗に `stage="c_final"` を加える。生成される `train.py` の校正器は同じ場合に `RuntimeError` を送出する。

### Platt Scaling 詳細（H-0100）

Platt (1999) の方法で推定する: `P(y=1|f) = 1/(1 + exp(A·f + B))`。**slope（A）と intercept（B）を同時に最尤推定**し、目標値は平滑化（`t+ = (N+ + 1)/(N+ + 2)`, `t− = 1/(N− + 2)`）、正則化項は置かない。**intercept はモデルの定義に含まれ、外せない** —— スコア 0 が確率 0.5 に対応しないずれを補正するのが役割である。係数は export 形式 `sigmoid(a·s + b)`（`a = −A`, `b = −B`）で持つ。

H-0100 までは `LogisticRegression(C=1.0)`（L2・目標値 0/1）で、原典から逸脱していた。

| パラメーター | デフォルト | 備考 |
|---|---|---|
| `target_smoothing` | `true` | `false` で目標値 0/1 |
| `method` | `L-BFGS-B` | 下表の手法のいずれか |
| `x0` | `[0, −log((N−+1)/(N++1))]` | `[a, b]`。Platt / scikit-learn と同じ初期値 |
| `bounds` | なし | `[[a_low, a_high], [b_low, b_high]]`。各値は数値か `null` |
| `tol` | なし | |
| `options` | L-BFGS-B のとき `gtol=1e-6`, `ftol=64·eps` | scikit-learn の Platt と同じ |

- `max(|s|) ≥ 30` のときはスコアを `k = max(|s|)` で割って最適化する（scikit-learn と同じ）。`x0` と `bounds` の slope 成分は `k` 倍してから解き、結果の slope を `k` で割って戻す。書いた座標で効く。

### Beta Calibration 詳細（H-0031, H-0100）

`sigmoid(a·log s + b·log(1 − s) + c)`（`s` はスコアの sigmoid）を負の対数尤度の最小化で推定する。尤度と 3 係数の形は固定。スコアは確率にしてから対数を取るので縮尺しない。

| パラメーター | デフォルト | 備考 |
|---|---|---|
| `method` | `L-BFGS-B` | 下表の手法のいずれか |
| `x0` | `[1, 1, 0]` | `[a, b, c]` |
| `bounds` | なし | 3 つの `[low, high]`。各値は数値か `null` |
| `tol` | なし | |
| `options` | なし（scipy の既定） | |

### Platt / Beta 共通の最適化の契約（H-0100）

| `method` | `bounds` |
|---|---|
| `L-BFGS-B` / `TNC` / `SLSQP` / `trust-constr` / `Powell` / `Nelder-Mead` | 使える |
| `BFGS` / `CG` | **使えない**（併記は拒否。scipy は bounds を警告つきで無視する） |

- ヘッセ行列を要する手法は受理しない。表は scipy 1.10 で使える手法に限る。
- **優先順位**: 書いた `options` のキー ＞ 書いた `tol` ＞ calibrator の手法ごとの既定。scipy は `tol` を `options.setdefault` で渡すので、`tol` を書いたときは既定の options のうち `tol` が設定するキーを入れない。
- `options` のキーは検査時に実物の scipy で確かめる（小さな問題に `minimize` を 1 回かけ、`Unknown solver options` を拒否に変える）。
- `bounds` の各組は**リスト**（H-0095 の受理集合で列の member は list であり、tuple は拒否）。
- 上表以外の名前（例: LogisticRegression の `C`）は `CONFIG_INVALID`。旧 artifact の platt calibrator（`LogisticRegression` を保持）は読み込み時に `(a, b)` へ移行され、predict は変わらない。`format_version` は 2 のまま。

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
- validation データ: calibration 学習データから 10% をランダムサンプリング（`validation_ratio=0.1`）。`seed` は `calibration.params` に無ければ `training.seed` を継承する（既定の `training.seed` が 42 なので、既定どおりなら 42。H-0080）。
- calibration データが少数（< 20 行）の場合は Early Stopping を無効化し、全データで学習する。

#### ユーザー上書き

- `calibration.params` で上記デフォルト（`monotone_constraints` 以外）を上書き可能。
- `validation_ratio` と `seed` も `calibration.params` 経由で指定可能。
- **名前は fit 開始前に検査される（H-0093）。** `calibration.params` の中身は `lgb.train` にほぼそのまま渡るため、LightGBM が知らない名前は黙って捨てられる。Facade は LightGBM 自身の登録表に照らして不明な名前を `CONFIG_INVALID` で拒否する。calibrator 自身が消費する `num_boost_round` / `validation_ratio` / `min_data_in_leaf_ratio` は受理される（`seed` は LightGBM のネイティブ名なので登録表側で受理される）。この検査は LightGBM を使う calibrator（現在は `isotonic` のみ）に対してのみ働く。**発火は外側 CV が始まる前**であり、拒否される config で Booster が 1 本でも学習されることはない。
- **上記の LightGBM 固有の名前検査は isotonic 限定のまま。** `platt` / `beta` の `calibration.params` は、それぞれの calibrator が宣言する受理契約で検査される（上の Platt / Beta の節、H-0100）。H-0100 までは両者とも受理して無視していた（[#277](https://github.com/nbx-liz/LizyML/issues/277)）。

#### Booster API 固有の注意

- `objective="binary"` の `Booster.predict()` は**確率を返す**（LightGBM が内部で sigmoid を適用する）ので、predict 時は `np.clip(0, 1)` だけを掛ける。**訂正（H-0100 決定 6）**: この項は H-0047 に倣って「raw score を返すため sigmoid を適用する」と書いていたが誤りで、実装（`lizyml/calibration/isotonic.py` の `predict`）は sigmoid を適用しておらず、それが正しい。

## 12.3 評価（推奨）

- `LogLoss`（必須推奨）
- `Brier score`（必須推奨）
- `ECE`（equal-width binning, M=10。各 bin の accuracy = `mean(y_true[mask])`（正例割合）、confidence = `mean(y_pred[mask])`。ECE = Σ (|bin| / N) × |accuracy − confidence|）
- `ROC-AUC / PR-AUC`（ランキング監視）
- fallback の記録（H-0089）: `CalibrationResult` は `fallback_fold_flags: list[bool]`（fold ごと、split の順。その fold の validation 行全体が未校正の OOF に fallback したとき `True`）と `n_fallback_rows: int = 0`（未校正の fallback を受けた validation 行の総数。校正した fold の中の NaN 行を含む）を持つ。`fit_result.calibrator` が保持するので `FitResult` の契約の一部である。

## 12.4 MUST NOT

- 同一行を学習に含む予測で校正器学習する（リーク）。
- `C_final` で `s_oof` を変換した値を評価に使う（楽観評価）。
- 校正器が `X` を利用する。

# 13. Metrics / Evaluation / Plots

## 13.1 Metrics

- Metric IF
- `needs_proba / greater_is_better`。指標がどのタスクを扱うかは指標のプロパティではなく、レジストリが持つ（`lizyml/metrics/registry.py`。扱わないタスクで引くと `UNSUPPORTED_METRIC`）。H-0014 は Metric IF の属性として `supports_task` を挙げたが、`BaseMetric` にその属性は無い（H-0111）
- `BaseMetric.needs_simplex` は既定 `False` の具体プロパティで、multiclass の予測が確率分布（行和 1）でなければならない指標が `True` に上書きする: `auc` と `logloss`。`auc_pr` / `brier` はクラスごとの値を使うので `False` のまま（H-0049）。
- 回帰: `rmse / mae / r2 ...`
  - `mape`: `y_true` に 0 を含むと `UNSUPPORTED_METRIC`（H-0004）。
  - `huber`: 誤差 `e` について `|e| <= delta` で `0.5 e^2`、それより大きいと `delta(|e| - delta/2)`。`HuberLoss` の `delta` は既定 `1.0` でコンストラクタ引数（MetricEntry `{huber: {delta: ...}}`、§13.1.1）。文字列 `"huber"` は `delta=1.0` を意味する（H-0004）。
  - `smape`: `mean(2|y - ŷ| / (|y| + |ŷ|)) × 100`、範囲 `[0, 200]`。`y = ŷ = 0` の行は 0 として数える（H-0071）。
  - `wape`: `sum|y - ŷ| / sum|y| × 100`。`UNSUPPORTED_METRIC` になるのは `sum(|y_true|) == 0` のときだけ（H-0071）。
  - `smape` / `wape` はどちらも `greater_is_better=False`、`needs_proba=False`。
- 分類（binary）: `logloss / auc / auc_pr / f1 / accuracy / brier / ece / precision_at_k ...`
  - `precision_at_k`: 予測確率の上位 `k`% の行の precision。`needs_proba=True`（確率で上位を選ぶ）、`greater_is_better=True`。既定は `k=10`（上位 10%、`1 <= k <= 100`）で、MetricEntry ごとに指定できる（§13.1.1、H-0014）。
- 分類（multiclass）: `logloss / auc(OvR) / auc_pr(OvR) / f1(macro) / accuracy / brier(OvR) ...`
- multiclass の `auc / auc_pr / brier` は One-vs-Rest 展開 + macro 平均で計算する。メトリクス名は binary と共通（`__call__` 内で `y_pred.ndim` により分岐）。`auc` は `roc_auc_score(..., multi_class='ovr', average='macro')`、`auc_pr` / `brier` はクラスごとの `average_precision_score` / `brier_score_loss` の macro 平均である（H-0018）。
- **task との整合**: 指標名は `metrics/registry.py` の `_TASK_METRICS` が task ごとに持つ集合で検査し、集合に無い名前は `UNSUPPORTED_METRIC`（`Model.evaluate(metrics=...)` と Config の `evaluation.metrics` の両方）。
  - `_TASK_METRICS["regression"]` = `rmse` / `mae` / `r2` / `rmsle` / `mape` / `huber` / `smape` / `wape`（H-0071）。
  - binary = `logloss` / `auc` / `auc_pr` / `f1` / `accuracy` / `brier` / `ece` / `precision_at_k`。multiclass = `logloss` / `f1` / `accuracy` / `auc` / `auc_pr` / `brier`。
  - したがって、`mape` / `huber` を回帰以外で（H-0004）、`precision_at_k` を回帰か multiclass で（H-0014）、multiclass 対応の `auc` / `auc_pr` / `brier` を回帰で（H-0018）指定すると `UNSUPPORTED_METRIC` になる。
- **確率の検査**: `needs_proba` が真の組み込み指標は、確率でない値（数値でない、有限でない、`[0, 1]` の外、3 クラス以上なのに 1 次元）を受け取ると `METRIC_REQUIRES_PROBA` を送出する（H-0106）。0/1 のハードラベルは正当な確率として通る。`objective: cross_entropy_lambda` の出力が 1 を超えた場合もこれになる（#307）。
- **evaluator が指標に渡す値**（`evaluation/evaluator.py` の `_pred_for_metric()`、H-0049）: `needs_proba` かつ `needs_simplex` の指標には、multiclass の 2 次元予測を行和で正規化して渡す（全 0 の行は割らない）。それ以外の確率指標には予測をそのまま渡す。ラベル指標には binary で 0.5 閾値、multiclass で argmax のラベルを渡す。正規化は `predict_proba()` ではなく evaluator の責務である（§14.1）。

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
    - covered 行の OOF に NaN があれば、構造的な未カバーではなくパイプラインの欠陥なので `Evaluator.evaluate()` は `LizyMLError(EVALUATION_FAILED)`（context `nan_count` / `nan_indices` / `task`）を送出する（H-0057。H-0057 の決定は `ValueError` と書くが、実装は `EVALUATION_FAILED` を送出する）。
  - `fold_0`...`fold_N-1`: 各 outer fold の OOF（valid_idx）値。
  - `if_mean`: IF（train_idx）指標の fold 平均（参考値として保持）。
  - calibrated がある場合は `cal_oof` 列だけを追加する（fold 別の calibrated 列は無い、H-0005）。
  - 列の順は `if_mean, oof, fold_0...fold_N-1, cal_oof` で固定する（H-0011）。index 名は `metric`。
  - calibrated ブランチの metrics 構造は `{"oof": {...}, "oof_per_fold": [...], "fallback_row_count": int}`。`fallback_row_count`（= `CalibrationResult.n_fallback_rows`、fallback が無ければ `0`）は未校正の fallback で採点した OOF 行の数である（§12.1、H-0089）。IF metrics は leakage リスクのため含めない。`oof_coverage` は raw と構造的に一致する（H-0058: outer splits 再利用）ため、`calibrated` に別途含めない。
  - `df.attrs["oof_coverage"]`: float (0.0–1.0)。covered 行の割合。KFold では常に `1.0`。TimeSeriesCV では `< 1.0` になりうる。

## 13.3 可視化

全プロットを Plotly ベースに統一する。Plotly は optional dependency（`pip install 'lizyml[plots]'`）。未インストール時は `OPTIONAL_DEP_MISSING` を返す。公開の plot メソッドはすべて `plotly.graph_objects.Figure` を返す（H-0008）。

実装済み:
- `importance_plot(kind="split|gain")`: fold 平均の特徴量重要度（横棒グラフ）
- `importance_plot(kind="shap")`: fold 平均の mean(|SHAP|)（横棒グラフ）。shap optional dependency も必要。
  - `Model.importance(kind="shap")` とこの plot は、fold ごとにその fold の validation 行（`valid_idx`）で SHAP を計算し、特徴量ごとの mean(|SHAP|) を fold で平均する（`compute_shap_importance()`、H-0007）。fold k の validation 行は fold k の pipeline 状態（`pipeline_state_per_fold` の k 番目）で変換する（H-0114）。shap が未導入なら `OPTIONAL_DEP_MISSING`。
- `plot_learning_curve(*, metrics=None)`: fold ごとの train/valid loss 推移（折れ線グラフ）。`metrics: list[str] | None` で表示 metric をフィルタ可能（H-0062）。`None` で全 metric、指定時は `/` 以降の metric 名で一致するもののみ表示。一致なしで `LizyMLError`。
- `plot_oof_distribution()`: OOF 予測値の分布（ヒストグラム）
- `residuals_plot(kind="scatter|histogram|qq|all")`: 回帰専用。IS/OOS 比較対応。`kind` で表示プロットを選択。デフォルト `kind="all"` で scatter + histogram + QQ の 3 パネル。scatter は Actual vs Predicted（x=predicted, y=actual, y=x 参照線）。IS サンプルは OOS 数に合わせてダウンサンプリング（`_downsample_is()`、seed=0 で再現可能）。
  - QQ パネルは OOS 残差だけを使う（IS は使わない、H-0009）。実体は `plots/residuals.py` の `plot_residuals()`。
  - 未知の `kind` は `CONFIG_INVALID`（H-0009 は `INVALID_CONFIG` と書いたが、その code は存在せず、実装は `CONFIG_INVALID` を送出する）。

追加で用意したい可視化（一部実装済み）:
- binary/multiclass: `roc_curve_plot()`（binary: IS/OOS の 2 本の ROC Curve 重ね描き。multiclass: IS/OOS を subplot 横並びにし、クラスごとの OvR ROC Curve を描画。各クラスの AUC 値を凡例に表示、macro 平均 AUC も表示。実体は `plots/classification.py` の `plot_roc_curve`。regression では `UNSUPPORTED_TASK`、H-0019）
- binary/multiclass: `confusion_matrix(threshold=0.5)`（IS/OOS の Confusion Matrix テーブル。`{"is": DataFrame, "oos": DataFrame}` を返す。binary は threshold、multiclass は argmax でクラスラベル変換。OOS は `compute_oof_valid_mask()` でカバー済み行のみを対象とする — NaN の構造的未カバー行は除外。各 DataFrame は scikit-learn の `confusion_matrix` の形で、行 = 真のラベル、列 = 予測ラベル、index / columns は整数（H-0016））
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

- `predict_proba()` は学習器の確率をそのまま返す。`objective: multiclassova` ではクラスごとの独立 sigmoid の出力なので、行和は 1 とは限らない。指標のための正規化は evaluator が `needs_simplex` の指標についてだけ行う（§13.1、H-0049）。
- `set_categorical_features()` は `fit()` 呼び出し前に CVTrainer が呼ぶ。categorical feature の扱いはアダプタの責務であり、`cv_trainer.py` に estimator 固有の kwarg を漏洩させない。

## 14.2 LGBM adapter の責務

- `objective / metric` 整合
  - 利用者または trial が書いた `objective`（エイリアスの綴りを含む）は、task の `TASK_COMPATIBLE_OBJECTIVES`（`estimators/lgbm/defaults.py`。LightGBM の canonical 名で regression 9 / binary 3 / multiclass 2）に含まれればそのまま `lgb.train` に渡し、含まれなければ `CONFIG_INVALID` とする（H-0079）。H-0079 より前は黙ってタスク既定に置き換えていた。`None` 以外の文字列でない値（dict / list など）も、包含を調べる前に `CONFIG_INVALID` とする（`check_objective_compatible`、H-0116。それ以前は dict / list が未加工の `TypeError` で落ちていた）。包含の検査とメッセージは値を書式化もハッシュもしないが、値の型やその属性の参照が例外を出すもの（`__class__`、メタクラスなど）と、送出後のエラーの表示（`context` は書いた値を保つ）は範囲外（H-0116 決定 2）。明示の `None` は「上書きなし」で、task の既定の objective で学習する。
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
   - multiclass のネイティブ whitelist に `auc` は無い（LightGBM 4 は multiclass の objective と `auc` の組を拒否する）。学習時の指標には `auc_mu` を使い、multiclass の AUC は事後評価の `evaluate(metrics=["auc"])`（scikit-learn の OvR、§13.1）で得る（H-0079）。

3. **feval カスタム関数**: LightGBM ネイティブ未対応の metric は `lgb.train(feval=...)` 経由でカスタム評価関数として注入する。

| feval Metric | Regression | Binary | Multiclass | metric に渡す値 |
|-------------|:---:|:---:|:---:|------------|
| `rmsle` | ✅ | | | 予測値そのまま |
| `r2` | ✅ | | | 予測値そのまま |
| `smape` | ✅ | | | 予測値そのまま（H-0071） |
| `wape` | ✅ | | | 予測値そのまま（H-0071） |
| `f1` | | ✅ | ✅ | 閾値 0.5 / argmax のラベル |
| `brier` | | ✅ | ✅ | 確率そのまま |
| `ece` | | ✅ | | 確率そのまま |
| `precision_at_k` | | ✅ | | 確率そのまま |
| `accuracy` | | ✅ | ✅ | 閾値 0.5 / argmax のラベル |

- `smape` / `wape` は回帰の feval 指標なので、`params.metric` に書けば early stopping と学習曲線を駆動する（H-0071）。
- **feval は LightGBM が渡す予測を再変換しない**（H-0105）。LightGBM 4 は組み込み objective の変換後の値を feval に渡す: binary は 1 次元の確率、multiclass / multiclassova は 2 次元 `(n, num_class)` の確率、regression は予測値。sigmoid / softmax をもう一度掛けると binary のラベル指標が定数になり確率指標が歪むので、掛けない。
- feval は evaluator と同じ規則（`evaluation.evaluator._pred_for_metric`、§13.1）で metric に値を渡す: ラベル指標には binary で 0.5 閾値、multiclass で argmax、multiclass の `needs_simplex` 指標には行和で正規化した確率、それ以外は確率をそのまま。学習曲線と `FitResult.metrics` で同じ名前の指標が同じ意味になる（H-0105）。
- multiclass の feval 入力が 2 次元 `(n, num_class)` でなければ reshape せず `EVALUATION_FAILED` を送出する（H-0105）。
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

H-0098: Parameter normalization and its boundary predicates derive from one
structural walk. Each visited value produces its plain representation, unchanged
status and mapping presence together. The surface predicate consumes unchanged
status; the training predicate additionally excludes mappings. Scalar and element
formatting retain their existing position-dependent byte-preservation rules.
The training mode rejects a mapping at the shared dispatch before visiting its
members, including cyclic mappings, so the assertion returns CONFIG_INVALID.
Search-space categorical choices retain H-0095's deliberate plain-scalar-only
restriction; numeric range bounds are converted before sampling.

Fit-only boundary clarification: `Model.fit()` requests value validation after
its final overlay. `Model.tune()` retains adapter validation after trial overlays;
rejecting the base value before a valid sampled replacement would be a regression.
`test_tuning_validates_after_sampled_overlay` pins this compatibility case, which
was reproduced as passing at the base and failing in the first local candidate.


H-0097 Revision 2: objective and metric validation runs on merged model parameters,
after precedence is resolved and while each written key still has an input origin.
The facade and direct adapter use the same validation rules. The provider Protocol
and adapter constructor do not acquire provenance arguments. Direct adapter calls
must supply seed and verbosity under one spelling each; duplicates are rejected
regardless of values, replacing the historical seed-priority behavior.


各 estimator モジュールが実装する protocol。`model.py`（Facade）が estimator 固有の知識なしに TrainComponents を構築するための統一 IF。

```python
class EstimatorProvider(Protocol):
    def extract_model_params(self, model_cfg: Any) -> dict[str, Any]: ...
    def extract_smart_params(self, model_cfg: Any) -> dict[str, Any]: ...
    def accepted_model_param_names(self) -> frozenset[str]: ...   # H-0093
    def smart_param_names(self) -> frozenset[str]: ...            # H-0093
    def canonical_param_names(                                    # H-0094
        self, names: Iterable[str],
    ) -> dict[str, str]: ...               # name -> the parameter it identifies
    def smart_managed_param_names(                                # H-0094
        self, smart: dict[str, Any], task: str,
    ) -> dict[str, tuple[str, str]]: ...   # spelling -> (canonical, smart)
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
    def build_pipeline_factory(
        self, unseen_policy: UnseenPolicy = "mode",
    ) -> Callable[[], BaseFeaturePipeline]: ...  # H-0104
    def default_space(self, task: str) -> list[SearchDim]: ...
    def default_fixed_params(self, task: str) -> dict[str, Any]: ...
    def runtime_deps(self) -> dict[str, str]: ...
    def params_summary(
        self, model: BaseEstimatorAdapter, model_cfg: Any,
    ) -> list[dict[str, Any]]: ...
    def build_export_params(
        self, adapter: BaseEstimatorAdapter,
    ) -> ExportParams: ...
    def parameter_bounds(                                         # H-0078
        self, task: TaskType,
    ) -> dict[str, dict[str, float | int]]: ...  # name -> {"min": ..., "max": ...}
    def objective_choices(self, task: TaskType) -> tuple[str, ...]: ...  # H-0079
    def metric_choices(self, task: TaskType) -> MetricChoices: ...      # H-0079: {"native": (...), "feval": (...)}
```

制約:
- `EstimatorProvider` は `config/` の具象型（`LGBMConfig` 等）を参照してよい（provider は Facade 層から呼ばれるため、Leaf → Leaf の依存にはならない）。
- `model_cfg` 引数は `Any` 型で受け取るが、各 provider 内部で `isinstance` チェックして具象型にキャストする。
- `runtime_deps()` はアルゴリズム固有の依存パッケージ名とバージョンを返す（例: `{"lightgbm": "4.5.0"}`）。`RunMeta.deps_versions` に使用。
- `parameter_bounds(task)` は境界拡張（§11.5）をクランプするための、パラメーターとして意味のある範囲 `{name: {"min": ..., "max": ...}}` を返す（H-0078）。表に無いパラメーターは無制限、空の dict は宣言なし。`LGBMProvider` は task によらない 15 パラメーターの表（`learning_rate` / `feature_fraction` / `bagging_fraction` / `num_leaves_ratio` / `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` / `validation_ratio` / `lambda_l1` / `lambda_l2` / `n_estimators` / `max_depth` / `max_bin` / `bagging_freq` / `early_stopping_rounds` / `seed`）を返す。`Model.tune` は既定の空間にも利用者の空間にもこれを付ける（§11.2）。
- `objective_choices(task)` は task で有効な objective の canonical 名を、決まった順の tuple で返す（エイリアスは含めない。H-0079）。`default_space` の `objective` 次元と UI の選択肢の元であり、集合は adapter が受理する `TASK_COMPATIBLE_OBJECTIVES`（§14.2）と同じ。未知の task は空の tuple。
- `metric_choices(task)` は `{"native": tuple, "feval": tuple}` を返す（H-0079）。`native` は学習器が評価する指標、`feval` は LizyML が feval として注入する指標（§14.3）。どちらも canonical 名で、順は決定的、2 つの tuple の間に重複は無い。
- `params_summary()` は `params_table()` 用のパラメータ行を返す。smart params + native model params（`metric` を含む、H-0061）の両方を含む。
- `build_pipeline_factory` は estimator 固有の FeaturePipeline が必要な場合（例: EntityEmbedding のカテゴリ埋め込み）に対応する。デフォルトは `NativeFeaturePipeline` を返す。
- `build_export_params` は codegen 経路（`Model.export_code()`）が必要とする native params / num_boost_round / early_stopping_rounds / feval metadata を `ExportParams` frozen dataclass で返す（H-0073）。`_model_persistence.py` から estimator 具象型（`LGBMAdapter` 等）への直接参照を排除するための入口。**「その fit が何を使ったか」を答える値は、config と現在の tuning result から再計算してはならない**（H-0094 決定 13）: `tune()` は tuning result を置き換えるが fit 済み adapter は置き換えないので、再計算する読み手は `fit → tune` の後に**存在しないモデルについて報告する**。実測: adapter が patience 7 で学習し、`params_table` / `export_code` はどちらも 2 と答えた。学習済み adapter は joblib で保存されるため、この経路は `load()` 後も正しい唯一の経路である（artifact が持つ tuning result は、どの fit も消費していないことがありうる）。`ExportParams.early_stopping_rounds` に **default を置かないこと** — 「provider が設定しなかった」と「early stopping が無効だった」が同じ値になるのは DC1 の形である。adapter に記録が無い値（`validation_ratio`）は `FitState.applied_training_params` から読む。この overlay は `metadata.json` の `applied_training_params` として記録され、`load()` が検査して復元する（H-0109）。記録の無い（H-0109 以前の）artifact では不明（`None`）となり config に落ち、再 export しても記録を書かない。
- `accepted_model_param_names()` / `smart_param_names()` は「その学習器が受理する名前」を宣言する（H-0093）。前者は**学習器自身から導出すること**（列挙しない）。学習器の更新で名前が増減したときに黙って古びる実装は、この IF が検出しようとしている欠陥をそれ自体が持つことになる。後者は `extract_smart_params` が返すキーと必ず一致させ、両者を単一の宣言から導くこと。
- `canonical_param_names(names)` は「その名前がどのパラメーターを指すか」を返す（H-0094）。**パラメーター層のマージは綴りではなく同一性で行うこと**: 学習器がエイリアスを解決する以上、`{**base, **override}` は 1 つのパラメーターを 2 つの綴りで残し、どちらが効くかは学習器の規則次第になる。実測: config の `learning_rate` が `fit(params={"eta": ...})` に勝っていた。**学習器に 2 つの綴りを渡さない**こと。学習器が知らない名前は自分自身に写すこと（不明名の拒否は `accepted_model_param_names` の仕事であり、この写像が二重の門になってはならない）。**マージの継ぎ目は 4 か所あり、4 か所すべてを同一性で行うこと**: config / provider の既定 fixed / `best_model_params` / `fit(params=)` に加えて、**tuning の trial マージ**（`_model_tuning.py` の objective）がある。round 11 まで trial マージだけが綴りベースで、config の `learning_rate` と `eta` という探索次元がある場合、**trial は config の値で学習し、study には trial の値が best として記録され、その後の fit は記録された値で学習していた** — tuning が一度も評価していないモデルを選んでいた（DC1）。
- **学習器 adapter が特別扱いするパラメーター**（検証する / 改名する / 呼び出し引数に変換する）は、**全綴りをまとめて取り出す**こと（H-0094 決定 6）。1 綴りだけを見ると、エイリアスで書かれた値はその特別扱いを迂回する。実測: `application`（`objective` のエイリアス）に task 非互換な値を書くと互換性検査を通らずに学習されていた。同じ層で 1 パラメーターが複数綴りで指定されたら、**値によらず** `CONFIG_INVALID` とすること（H-0096 で改訂。**それ以前は「異なる値のときだけ拒否し、同値は通す」だった**）。同値を通す規則は「2 つの綴りが同じ値か」という問いを生み、その問いは任意の Python 値の上で全域でなければならないので**入力領域が開く** —— H-0094 rounds 18-26 の 9 連続はその領域の中で起きた。**LightGBM 自身も値を比較せず、等しくても重複そのものを警告する**（実測）。拒否の根拠は「どちらが効くか不可視だから」ではなく（優先順位は決定的である）、**利用者が 1 つの設定を 2 度書いており、どちらを意図したか LizyML には決められないから**である。**この規則は宣言した層すべてに配線すること**: レビュー round 11 まで `fit(params=)` にしか配線されておらず、`model.params` に `learning_rate` と `eta` を両方書いた config は両方が `lgb.train` に届き、LightGBM が黙って canonical 側を採った（DC4 — 宣言はあるが呼び出し側が無い）。検査は facade（`_merge_params`）に置く。`config/` は層規約上 `estimators/` を import できず別名表に届かないためである。**配線先は 4 層**: `model.params` / `fit(params=)` / `calibration.params` / `tuning.optuna.space`。3 つ目は rounds 10-11 monitor が「どの層に配線したのか」を問うて見つかった（名前検査だけがあり同一性検査が無く、両綴りが calibrator の `lgbm.train` に届いていた）。4 つ目は rounds 11-12 monitor が名指しし、実行して確かめた: `sample_params` は次元ごとに `params[dim.name] = ...` を書くので、**互いにエイリアスである 2 次元は同じ trial dict に両綴りを入れる** — LightGBM が canonical を採るため、もう一方は sample され最適化されながらどの trial にも影響しない（`learning_rate` と `eta` を 2 次元にした study で実測）。**空間には同値による免除が無い**: 2 次元は独立に sample するので、1 パラメーターを 2 回名指しすることは境界に関わらず曖昧である（`check_duplicate_space_dimensions`、study 開始前）。スマート層は対象外で、その理由は**スマートパラメーター名に学習器のエイリアスが 1 つも無い**ことである（実測 0 件、テストで固定）。**`calibration.params` は同一性検査だけでは足りず、canonical 化も要る**（H-0094 決定 8）: calibrator は自分の既定値と**綴りで**マージするため、呼び出し元が**1 度しか書いていない**エイリアスが既定値と並んで `lgbm.train` に届き、LightGBM が既定値を採っていた（`{"eta": 0.5}` → `learning_rate: 0.03` で学習）。`lizyml/calibration/` は `lizyml/estimators/` を import できないので、書き換えは facade（`canonicalise_calibration_params`）で行う。**calibrator 自身が pop するキーは除外すること**（`num_boost_round` は `num_iterations` のエイリアスであり、canonical 化すると pop 先が消える）。**calibrator が強制する値は canonical 綴りで書くこと**: `verbose = -1` はエイリアス側の強制だったので `verbosity` を書いた呼び出しに負けていた。
- **「あるパラメーター dict が別のパラメーター dict に出会う」場所を列挙し、1 行ずつ実行すること**（H-0094 決定 8）。この類の欠陥は round ごとに 1 つずつ出続けたので、当たりを付けて探すより列挙するほうが安い。走査は `instruments/parameter_merge_seams.py` として**出荷し**、表を再生成できるようにすること（散文に写した数は古びる）。候補 58 式。**走査の構文集合には `d[k] = v` を必ず含めること**: 最初の版はそれを宣言しておらず、その版が報告した 3 件の欠陥のうち 2 件がその構文に住んでいた（rounds 11-12 monitor）。**そして走査を「閉じた母集団」と呼ばないこと**: この主張は 2 回なされ、2 回とも 1 ラウンド以内に反証された（2 回目は round 13 のレビュアーが `config/loader.py:167` を名指しした — カーソル変数名が `node` で hint 語に当たらなかった）。hint 語による絞り込みは識別子テキストのヒューリスティックであって型解析ではない。**表が主張するのは実行した分だけである。** 開いた空間の走査を閉包と呼ぶことは、本 run が他人の宣言に見つけ続けている DC5 そのものである。
- **「利用者が綴った dict を 1 つの綴りで読む」箇所も同じ類であり、継ぎ目走査は扱わない**（H-0094 決定 9）。実測: `_extract_feval_metadata` が `adapter.params.get("metric")` とリテラルで読んでいたため、`fit(params={"metrics": ...})` は正しく学習しながら `export_code` が評価関数を落とし、**生成コードが動かなかった**。学習側が `_pop_by_identity` で読む以上、読み手も同一性で読むこと。母集団は 4 件（`estimators/` / `persistence/` / `codegen/` / `core/` / `training/` を走査）。
- **LizyML 自身が握っているネイティブパラメーターは、`model.params` / `fit(params=)` から重ねて指定させないこと**（H-0094 決定 9、`check_training_managed_overrides`）。`training.early_stopping.rounds` は `early_stopping_round` の、`training.seed` は `seed` の LizyML 側の綴りである。実測では**両方向に**壊れていた: 前者は上書きが `lgb.train` に届いても config 由来の callback が停止を決め（早期停止を切ると今度は LightGBM 自身が honour して検証セット不在で落ちる）、後者は上書きが `training.seed` に黙って勝っていた。**2 方向が食い違うからこそ、どちらかを選ぶのではなく拒否する。** 判定は canonical 名を全綴りに展開して行い、展開は provider の `canonical_param_names` を受理名の上で反転して得る（この層は学習器を知らない）。
- `smart_managed_param_names(smart, task)` は「**有効なスマートパラメーターが上書きしてしまうネイティブ名**」を返す（H-0094）。スマート解決はパラメーター dict のマージより後段で走り、その結果が勝つため、ここに挙がる名前を手で指定しても黙って置き換えられる。呼び出し側はそれを**拒否**に使う（`CONFIG_INVALID`）。task を取るのは、同じスマートパラメーターでも task によって書くものが変わるためである（`balanced` は binary では `scale_pos_weight` を書くが、multiclass では sample weight を作る＝パラメーター名ではないので衝突しない）。宣言は**コードから閉じる**こと: 解決関数群を**実際に実行**し、宣言されていない名前が返ってきたら落ちるテストを持つ。以前はソースの `resolved[...] = ...` 代入を走査していたが、それは代入の**綴り方**についての仮説であり、`resolved.update({...})` で書かれた 4 つ目の名前は見えなかった（H-0094 レビュー round 9 の monitor）。実行には推測すべき綴りが無い。ただしこれは**有限個の実行**であって閉じた入力領域ではない — パラメーター**名**を列挙しても、解決関数が受け付ける**組み合わせ**は列挙できない（round 10）。各 activation が解決結果を変えることを個別に主張することで、activation が無効化されて何も観測しなくなる事態は防ぐ。
- **エイリアスを展開すること（H-0094 レビュー round 2 の指摘）。** 学習器がエイリアスを同一パラメーターとして解決する場合、文字列一致の検査は同じパラメーターを別綴りで通してしまう。実測: `max_leaves`（LightGBM では `num_leaves` のエイリアス）は検査を通過し、スマート解決が入れた `num_leaves` を LightGBM が優先したため、上書きはまた黙って無視された（`[(12, 32), (12, 32), (12, 32)]`、booster は `[num_leaves: 32]`）。戻り値のキーは**学習器が受理する全綴り**とし、綴りの集合は学習器の登録表から導くこと（列挙しない）。
- **パラメーター値の受理集合を入口で閉じること**（H-0095、**H-0096 で存在理由を再定義**）。この閉包はもともと `values_differ`（「2 つの綴りが同じ値か」）の入力を有界にするために入った — round 16-20 が **5 連続で「直前の修正が書いたコードの欠陥」**を出し、`__format__` / `__class__` / `tolist` / `__eq__` を任意に定義できる以上その領域は構成上開いていたためである。**H-0096 でその比較そのものが無くなった**ので、閉包が今仕えているのは残る消費者、すなわち**学習サイトの出口表明**（`assert_plain_params`）と **`export_code` の `json.dump` / UTF-8** である。後者は本 PR とは独立の既存欠陥を直している（`origin/develop` の `ccae32b` で `model.params={"feature_contri": np.array([1.0,1.0])}` が `TypeError: Object of type ndarray is not JSON serializable` を出すことを実測）。**4 surface の入口で 1 度だけ正規化し、受理集合の外は学習前に `CONFIG_INVALID` で拒否する**（`core/param_domain.py`）。受理集合は LightGBM の `_param_dict_to_str` から**導出**する（写さない）。**正規化は素の型へ行い、文字列化しないこと** — smart params の解決と boundary 展開が数値演算をする。**中心的な不変条件は wire 保存**: `_param_dict_to_str` が正規化の前後で同じ bytes を書くこと。シリアライザは**位置によってフォーマッタが違う**（スカラーは `__format__`、列の要素は `str`）ので、変換も位置ごとに分けること。実測: `np.array([0.1], dtype=float32)` は `0.1` と書かれるが `.tolist()` 後は `0.10000000149011612` になる。**素の代替が存在しない値（`str(np.float16(1e3))` = `1e+03`）は丸めずに拒否する** — 呼び出し元が書いていない bytes で学習するほうが悪い。型は**厳密一致**で見ること（サブクラスは `__format__` を上書きできる）。そして**4 surface で正規化することは配線についての主張にすぎない**ので、`lgb.train` の全サイトに `assert_plain_params` を置き、**学習サイトの母集団をソースから導出して**固定すること（DC4 の形）。**入口と出口で受理集合は異なる**: metric entry は `{"precision_at_k": {"k": 15}}` という LizyML の形（H-0065）を持ち adapter が消費するので、入口は mapping を受理し `lgb.train` の表明は拒否する。**`set` は拒否する** — シリアライザは join するが列パラメーターは位置依存であり、`list(set)` がリテラルのリストと一致するかはハッシュ順の偶然である。**リストの入れ子は深さ 2 まで** — 3 段目は Python の list repr で書かれ、正規化が wire を変えてしまう。 **numpy は厳密な型一致で受理し、型集合は numpy 自身の階層から導出すること**（レビュー round 21）。継承で受理すると `np.float64` のサブクラスが自前の `__format__` で通り、**`np.timedelta64` は `np.integer` のサブクラスなので**型集合の中に入る。実測: 前者は `0.9` と書いて `0.1` で学習し、後者は `1 nanoseconds` と書いて `1` で学習した。厳密型一致が買うのは「**正規化中に呼び出し元のコードが 1 行も走らない**」ことであり、それとは別に **`format(plain, "") == format(value, "")` を値ごとに検査すること** — `timedelta64` を捕まえるのは後者である。 **型集合は `vars(numpy)` から読むこと**（round 22）: `__subclasses__()` の走査は `__module__` という**呼び出し元が書ける属性**を信じることになり、しかも走査が import 時なので**そのクラスが定義された順序で答えが変わる**。「numpy がその名前で export しているか」は同一性の問いであり、どちらの穴も無い。**そして `format(value, "")` は `.item()` の前に読むこと** — 値は同じ問いに 2 度同じ答えを返す義務を負わない。 **型の判定は `is` で行うこと**（round 23）: `type(x) in <set/tuple>` は `__hash__` / `__eq__` による探索であり、クラスのそれらは**メタクラス**から来る＝呼び出し元が書ける。実測、メタクラスだけで numpy 継承なしに門を通過した。**そして採用の決め手は名前空間ではなく `np.dtype(kind).type is kind` の往復にすること** — `vars(numpy)` は書き込み可能で、import 前に 1 行書けば入る。**宣言する bound は「値に対して閉じる」であって「プロセス内で numpy を差し替えた呼び出し元に対する sandbox」ではない**（達成不能な宣言は DC7 であり、この run で 3 度書き直している）。 **そして受理した「型」が書ける「値」とは限らない**（round 24）: シリアライザは `isinstance(val, Path)` を見るので **pure path は通らず**、Python の `int` は十進変換上限を超えると `str()` が raise する。**型から推定せず、文字列を実際に要求すること** — scalar 位置は `format`、element 位置は `str`、書けなければ入口で拒否。
  - 検査の発火点は**学習器に渡す直前**（`_merge_params` の merge 後、および tuning study 開始前）であって構築時ではない。config は呼び出し側が参照を保持したまま変更でき、`best_model_params` は artifact から `__init__` 後に復元されるため、構築時の検査ではどちらも素通りする。`Model.load()` 自体は検査しない（artifact は起きた fit の記録であり、読めなくする理由がない）。
  - **受理集合の正は生成物である**（H-0095 の「契約の確定」節）。散文に写した表は 19 回の補正のあいだに 2 度古びたので、位置（スカラー / 要素 / 列の member / mapping）× 厳密型の表は `docs/audits/2026-09-defect-discovery/instruments/param_domain_contract.py` が `param_domain.py` から生成し、`--check` が HISTORY.md との乖離で非零終了する（DC3）。**表の各行は、それを固定しているテストを名指すこと** — テストの無い行は次のラウンドが見つける行である。
  - **消費者を全部名指すこと。** 学習器は唯一の消費者ではない: `lgb.train` 2 サイトに加えて **`export_code` が同じ値を `config.json` へ `json.dump` する**。提案がこれを名指していなかったため、受理集合の定義が実装のほうから動いた（path を型のまま通していて `TypeError: Object of type PosixPath is not JSON serializable` になり、path をテキストにする補正が入った）。**要件は消費者ごとに 1 つ立て、受理母集団全体の上で実行するオラクルを持たせること**: wire 保存 / 冪等 / `is_plain`（出口） / `json.dump` 可能 / UTF-8 encode 可能。**「比較が全域」は H-0096 で消えた** — 消費者だった `values_differ` ごと削除したためであり、要件が緩んだのではなく消費者が居なくなった。**閉じられるのは要件のリストであって消費者のリストではない** — sink 走査は候補生成であり、実際にこの走査も初版で calibrator の `lgbm.train` を別名ゆえに落とした。
  - **消費者の要件は 1 つとは限らない**（レビュー round 25）。学習器も `export_code` も、シリアライズ／json 化の**あとで UTF-8 に encode する**。孤立サロゲートは受理型の `str` で、正規化・出口の表明・`json.dumps` オラクルを通ってから両消費者で `UnicodeEncodeError` になっていた。**書く文字が encode できることまでを入口で検査すること。**
  - **「変わっていない」を表示テキストで判定しないこと**（レビュー round 26）。`repr` は numpy の `printoptions(legacy="1.25")` で変えられる＝**呼び出し元が参加できる比較**であり、その下では `np.int64(1)` と `1` が同じに印字されて変換されていない numpy 値が述語と出口の表明を通った。判定は**型の再帰的一致とスカラーの同一性（`is`）**で行うこと。**入口の門を `is` にした理由は、値の比較側にもそのまま効く。**
  - **1 つの境界に対する宣言を 2 つ持たないこと**（レビュー round 25）。`is_accepted` / `is_plain` が受理集合を自分の言葉で言い直していたため、正規化だけを「実際に文字を書ける値」へ狭めた修正で置き去りになり、`10**5000` が両述語と出口の表明を通って正規化にだけ拒否された。**述語は正規化関数を呼ぶこと。** 一致テストは**両向き**で持つこと —— 受理母集団だけを走査するテストは、緩すぎる述語を構成上見られない。
  - **母集団は領域ではない**（レビュー round 25）。値領域は無限（文字列・整数・コンテナの中身に上限が無い）であり、テストが主張できるのは**有限標本の上での網羅**である。標本は **(1) 型軸を受理型集合から導出**し（ベタ書きの版は 5 型を落としていた）、**(2) 拒否理由を閉じた列挙にして両向きに突き合わせる**ことで閉じる。理由は**位置ごとに**評価すること（同じ型が scalar 位置で拒否され element 位置で受理されることがある。実例 `longdouble`）。
  - **スコープ外を事実として書くこと**（達成不能な宣言は DC7）。(1) プロセス内で numpy を差し替え済みの呼び出し元に対する sandbox ではない。(2) 閉じているのは「このプロセスで 4 surface を通って入った値」であり、**このバージョンより前に書かれた artifact** は受理集合の外の値を持つ adapter を復元しうる（実測: 復元後に `export_code` が `TypeError`。H-0095 の前と同じ振る舞い）。(3) 入口（mapping を受理）と出口（拒否）で受理集合が違うのは意図である。

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
  - `lizyml_version`
  - `python_version`
  - 依存 versions（`lgbm / sklearn / optuna ...`）
  - `config_normalized`
  - `schema`（`feature_names / dtypes / categorical policy`）
  - split indices
  - `data_fingerprint`
  - `pipeline_state`、`pipeline_state_per_fold`（`fit_result.pkl` の中。H-0114）
  - `models, calibrator`
  - fit が適用した training overlay（`applied_training_params`。`tuning` ブロックはモデルの現在の tuning result で、どの fit も消費していないことがあるので別に記録する。H-0109）
- `metadata.json` のキー（`persistence/exporter.py`）:
  - 常に書く: `format_version` / `lizyml_version` / `python_version` / `timestamp` / `run_id` / `config` / `metrics` / `feature_names` / `task` / `checksums`。
  - `checksums` は `{"algorithm": "sha256", "files": {<ファイル名>: <16 進の digest>}}` で、`files` は `fit_result.pkl` / `refit_model.pkl` と、あれば `analysis_context.pkl` の SHA-256 を持つ。アルゴリズム名は `persistence/exporter.py` の `CHECKSUM_ALGORITHM`（H-0083）。
  - tune 済みのモデルだけ: `tuning` ブロック（`best_model_params` / `best_smart_params` / `best_training_params` / `best_score` / `metric_name` / `direction`、成功した tuning round の `fixed_params`）。`load()` はこれを tuning result として復元し、load 後の再 `fit()` が tuned params を再現する（H-0086）。trial の履歴は保存しない。
  - fit が overlay を記録したモデルだけ: `applied_training_params`（上記、H-0109）。

## 15.2 互換性ポリシー（必須）

- `format_version` が読めない場合は明示的に拒否する（黙って壊れた復元をしない）。
- 将来 migration を実装できる前提で serializer に拡張点を残す。
- 現行 `FORMAT_VERSION = 2`（H-0070）。`{1, 2}` の両方を loader が受理し、v1 artifact には no-op `TargetEncoder` を in-memory で注入して contract を整合させる（INV-5）。
- **フィールドの追加は後方互換の変更で、`format_version` を上げない。** フィールドの削除、型や意味の変更は破壊的変更で、`format_version` を上げる（H-0003）。`checksums`（H-0083）、`tuning`（H-0086）、`applied_training_params`（H-0109）はいずれも追加として `FORMAT_VERSION = 2` のまま入った。旧 loader は知らないキーを無視する。
- `load()` は毎回 `metadata.json` を検査する。必須キー（`_REQUIRED_METADATA_KEYS` = `format_version` / `task` / `feature_names` / `config` / `run_id`）が欠けていれば `DESERIALIZATION_FAILED`。
- **完全性の検査（H-0083）**: `load()` は各 `.pkl` のバイト列を 1 回だけ読み、`checksums` に記録された digest と照合してから、そのバイト列を `joblib.load(io.BytesIO(...))` で復元する（ファイルを再 open しないので、検査と復元の間の TOCTOU が無い）。`algorithm` が `sha256` でない、または digest が一致しないときは pickle を実行する前に `DESERIALIZATION_FAILED`（context: `file` / `expected` / `actual`、アルゴリズム違いでは `file` / `algorithm`）。`checksums` を持たない artifact（H-0083 以前）と、`files` に載っていないファイルは検査せずに読む。
- **脅威モデル**: `metadata.json` 自体は署名しない。書き込み権限を持つ者は `checksums` を書き換えたり消したりできる。`checksums` が検出するのは破損と改竄であり、悪意ある作成者に対して pickle を安全にするものではない。artifact は信頼できる出どころからだけ読む。
- `analysis_context.pkl` を持たない artifact（H-0026 以前）でも `predict()` と `evaluate()` は使える。load 後の診断 API（`residuals()` など）は、必要なデータが無いと `MODEL_NOT_FIT` で明示的に失敗し、最新版での再 export を促す（H-0026）。
- `pipeline_state_per_fold` を持たない `FitResult`（H-0114 より前の artifact。`format_version` は 2 のまま）は、そのまま読める（既定値 `None` がクラス属性として残るので、属性は `None` を返す）。`importance(kind="shap")` / `importance_plot(kind="shap")` だけが `MODEL_NOT_FIT`（context `task` / `kind` / `method` / `missing="pipeline_state_per_fold"`）で失敗し、`fit()` し直すことを促す。再 export では状態は作られない。この検査は `analysis_context` が無い場合の検査より先に行う。split / gain の重要度と他の API は変わらない（H-0114）。
- 非推奨の面とその削除目標（v1.0）の唯一の登録簿は `docs/DEPRECATIONS.md` である。非推奨の警告文は削除目標の版を明記する（H-0076）。

## 15.3 `export`（`Model Artifact`）

- `Model Artifact` を 1 ディレクトリにまとめる。中身は `metadata.json`、`fit_result.pkl`（`FitResult`）、`refit_model.pkl`（`RefitResult`）と、任意の `analysis_context.pkl`（load 後の診断 API が使う `y_true` と `X_for_explain`。無ければ load 時に `None`、H-0026）。`.pkl` は joblib（圧縮）で書く（H-0003）。
- `.pkl` は pickle なので、読み込むと任意の Python を実行しうる。`Model.load()` は信頼できる出どころの artifact だけに使う（H-0003、上記の脅威モデル）。
- `Model.load()` で復元し、推論と評価情報参照に加えて診断 API（残差/SHAP/分類・校正可視化）も利用可能にする。

## 15.4 `export_code`（Codegen Export, H-0059）

- LizyML 非依存の学習・推論コードを生成する。`export`（§15.3）とは独立した出力形式。
- `format_version` とは無関係（pickle を使用せず、テキスト/JSON のみ）。
- 出力ディレクトリ構造:

```
{path}/
├── config.json             # 全設定（ハイパーパラメータ / 特徴量 / 校正 / split）
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
- **校正用 OOF の fold の再現（H-0090）**: `config.json` は `split` ブロック（`config.json["split"]`）を持ち、outer split の method 固有のパラメーターを解決済みの値で書く（`_build_split_metadata(cfg)`。`stratify="auto"` は bool に畳み、`random_state` が無ければ `training.seed` を書く）。生成 `train.py` は校正用 OOF の CV fold をこのブロックから作り、`split.method` を再現する: `kfold` / `stratified_kfold` / `time_series` / `group_kfold` / `stratified_group_kfold` は scikit-learn の splitter、`purged_time_series` / `group_time_series` / `blocked_group_kfold` は LizyML のロジックを numpy に移したもの。
  - 時間の method では `time_col`、`blocked_group_kfold` では `blocks.col` で pandas の `argsort()` により並べてから分割し、fold を元の行順に戻す。
  - 生成 calibrator は covered な（OOF が NaN でない）行だけで学習する（LizyML の cross-fit の `C_final` と同じ）。
  - `split` ブロックを持たない export（H-0090 以前）は、従来の task 別のシャッフル K-fold（binary は `StratifiedKFold`、それ以外は `KFold`）に戻る。
- 生成 `train.py` の feval も LightGBM が渡す予測を再変換しない（§14.3、H-0105）。multiclass の入力が 2 次元でなければ `ValueError` を送出する。
- **calibration の再現（H-0100）**: `config.json` に `calibration_params`（fit が使ったのと同じ前処理を通した実効値）を持たせ、`train.py` の `_fit_platt` / `_fit_beta` / `_fit_isotonic` は LizyML の calibrator と同じモデル・既定値・上書き・最適化の契約で calibrator を再構築する。H-0100 までは 3 手法とも既定値を直書きしており、`isotonic` でも再学習で params が失われていた。`requirements.txt` は生成コードが scipy を import する `platt` / `beta` のとき scipy を載せる
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
LizyMLError(code, user_message, *, debug_message=None, cause=None, context=None)
```

## 16.2 例外コード

`ErrorCode` の全メンバー。どのメンバーも本番コードのどこかが、到達できる条件で発生させる（H-0106。`tests/test_core/test_error_code_population.py` が `raise` の存在を、`test_error_code_raising.py` が条件を実行して確かめる。この一覧と `docs/api.md` の表は `tests/test_docs/test_error_code_docs.py` が enum と照合する）。`DATA_FINGERPRINT_MISMATCH` は予測時に照合できる条件が無いため H-0106 で削除した（`DataFingerprint` は来歴の記録として残る）。

- `CONFIG_INVALID`
- `CONFIG_VERSION_UNSUPPORTED`
- `DATA_SCHEMA_INVALID`
- `LEAKAGE_SUSPECTED`
- `LEAKAGE_CONFIRMED`
- `OPTIONAL_DEP_MISSING`
- `MODEL_NOT_FIT`
- `INCOMPATIBLE_COLUMNS`
- `UNSUPPORTED_TASK`
- `UNSUPPORTED_METRIC`
- `METRIC_REQUIRES_PROBA`
- `TUNING_FAILED`
- `EVALUATION_FAILED`
- `CALIBRATION_NOT_SUPPORTED`
- `CALIBRATION_NOT_FITTED`
- `CALIBRATION_FAILED`
- `SERIALIZATION_FAILED`
- `DESERIALIZATION_FAILED`
- `TARGET_NOT_NUMERIC`
- `TARGET_UNSEEN_LABEL`

# 17. Logging / Run 管理（`core/logging.py`）

- `run_id` を生成し、出力先（`logs / artifacts`）を統一する。
- 重要イベントを構造化ログで出す（config hash, data fingerprint, split hash 等）。
- エラー時は `code` を必ずログに残す。
- `output_dir` オプション（Config or コンストラクタ引数）指定時、`{output_dir}/{run_id}/` をログの保存先にする。plot はファイルに保存しない: plot API は plotly の Figure を返すだけで、`output_dir` の有無で変わらない（H-0034 は plot の保存先もここにすると書いたが、plot を書く経路は無い。H-0111）。
- `output_dir` 未指定時はログを標準出力に出す。
- `output_dir` の優先順位は constructor > config > 未指定（`Model(..., output_dir=...)` が Config の `output_dir` に勝つ。H-0039）。解決は `or` なので、偽値のコンストラクタ引数は Config に落ちる。
- `output_dir` があれば `fit()` / `tune()` はそれぞれ新しい `run_id` で `{output_dir}/{run_id}/` を作り、`run.log` を書く（ログファイルが出力先に保存される、H-0034）。
- `export()` を path 無しで呼ぶと、直前の run のディレクトリがあれば `{run_dir}/export` に、無ければ `output_dir` の下に新しい run ディレクトリを作ってその `export` に書く。どちらも無ければ `SERIALIZATION_FAILED`（H-0039）。

# 18. テスト / CI（必須）

## 18.1 テスト戦略

テストは以下の 10 カテゴリで構成する。各カテゴリは独立したテスト目的を持ち、組み合わせで回帰耐性を確保する。

### 18.1.1 基本テストカテゴリ

- **Golden test（契約固定）**: `FitResult / PredictionResult / RunMeta / SplitIndices` のフィールド名・型・構造を固定し、意図しない破壊的変更を検知する。
- **再現性テスト**: 同一 config + seed で `oof_pred`, `predict`, `metrics`, `split indices` が bit 一致する。`tune()` も同一 seed で `best_params` / `best_score` / trial 順序が一致する。bit 一致は**固定 `(num_threads, CPU)` 環境**（同一スレッド数の同一プロセス / マシン）を前提とする。LightGBM の histogram 構築がスレッド数依存のため、クロス環境 bit 一致は保証範囲外（H-0081）。
- **リーク防止テスト**: OOF が held-out データのみから生成されること、calibration が cross-fit で分離されていること、feature pipeline が train fold のみで fit されることを検証する。
- **列ズレテスト**: 余剰 / 不足 / unseen category のポリシー通り動く。カテゴリ順序ずれ（学習時と推論時で同一カテゴリだが出現順が異なる）もカバーする。
- **例外テスト**: 全 `ErrorCode` に対して少なくとも 1 テストが存在し、`context` dict の必須キーを検証する。
- **optional dependency テスト**: 未導入時の例外コード / メッセージが崩れない。全 optional dependency（optuna, shap, plotly, scipy）について "missing" パスを検証する。
- **Public API surface テスト**: `from lizyml import Model` 等のトップレベル公開面が壊れていないことを検証する。
- **提案の処分テスト**: `docs/proposal_dispositions.toml` が HISTORY の全提案について BLUEPRINT での処分（`specified` + anchors / `no_obligation` / `superseded` / `pending`）を 1 行ずつ持ち、`tests/test_docs/test_proposal_blueprint_coverage.py` が anchor を BLUEPRINT と提案自身の HISTORY entry の両方で全単語一致として検査する（H-0110）。新しい提案を足す PR は、その行を同じ PR で `proposal_dispositions.toml` に足す。
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
- Config → task-locked: `objective` は task 互換なら利用者 / trial の値がそのまま Booster に届き、task 非互換なら `CONFIG_INVALID`（H-0079。それ以前はタスクから固定し黙って置換していた）。`num_class` が multiclass で自動注入。`verbosity` が `-1` 固定。

**実効性テスト**（値が動作に反映されること）:

- **2 値比較パターン**: 各 Booster パラメータについて、異なる値で fit → 予測が変わることを検証する。対象: `learning_rate`, `max_depth`, `n_estimators`, `max_bin`, `lambda_l1`, `lambda_l2`, `bagging_fraction`, `feature_fraction`, `boosting`, `metric`, `num_leaves`, `min_data_in_leaf`。
- **Smart Params 動作反映**: `feature_weights` → importance 順序変化、`balanced` → 不均衡データの予測分布変化、`scale_pos_weight` → 予測分布変化。
  - この宣言は H-0093 まで実装に対して偽だった（発出キーが LightGBM の定義に無く、重みは捨てられていた）。「importance 順序変化」は**重み無しの fit との差分**で検査すること。importance dict に列名が含まれることの確認は、重みが効いていなくても成立するため検査になっていない。
- **学習器に渡す名前の検査**: `model.params` と `tuning.optuna.space` の `category: model` 次元は、学習器が受理する名前のみを通す（H-0093）。受理集合は学習器自身から導出し、列挙しない。LightGBM は未知の名前を**エラーなく捨てる**（既定の `verbose=-1` では警告も出ない）ため、検査が無いと綴り違いは「成功したが何も起きていない run」になる。
- **Training 実効性**: `early_stopping.random_state` → 同一 seed で同一 inner split、`validation_ratio` → inner valid サイズ比例。
- **Feature 実効性**: `auto_categorical` → string 列の自動検出。
- **Calibration 実効性**: `calibration.params` → calibrator パラメータ到達。

### 18.1.3 Facade オーケストレーションテスト（H-0056 カテゴリ A 関連）

`Model.fit()` が各コンポーネントを正しい順序・正しい引数で呼ぶことを検証する。

- CVTrainer と RefitTrainer が同一の `pipeline_factory` / `estimator_factory` / `ratio_resolver` を受け取る。
- Evaluator が Config 指定のメトリクスリストを受け取る。
- Calibration が `cfg.calibration is not None` かつ `task="binary"` の場合のみ実行される。non-binary で `CALIBRATION_NOT_SUPPORTED` を返す。
- `get_provider()` が model name で正しい provider を返す。未知の name で `CONFIG_INVALID`。
- `_merge_params` の優先順位: Config defaults < tune best < fit() args。**この 3 段目は宣言だけで実際には届いていなかった（H-0094 / #264）**ため、`fit(params=...)` を渡した fit と渡さない fit で**学習済み Booster が異なること**を主張する。マージ後の dict を突き合わせるだけでは、欠陥のあるコードでも成立した。
- 不明な名前の拒否は**出所（`model.params` / `tuning best_model_params` / `fit(params=)`）を名指しする**（H-0094）。3 つの入口が 1 つの dict にマージされてから検査されるため、出所を持たないと 3 つのうち 2 つは誤った宛先を指す。
- `fit(params=)` に**有効なスマートパラメーターが管理するネイティブ名**を渡した場合は拒否する（H-0094 決定 4、§5.3 の表）。テストは 2 方向で主張すること: 有効なら拒否かつ Booster 0 本、**無効化すれば同じ値が `lgb.train` に届く**。後者が無いと、管理表に何を書いても拒否テストは通る。

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
- 依存の上限方向（forward-compat）を検証する non-blocking lane を CI に含める（`uv sync --upgrade`）。上流の破壊的リリース（pandas / numpy / lightgbm 等）を早期検知する。ランタイム依存は下限のみ宣言し上限 cap は付けない方針とする（詳細は CONTRIBUTING.md）。
- OS portability smoke（ubuntu / windows / macos）を CI に含め、path / newline 等の移植性問題を検知する（最低限の file I/O 系サブセット、単一 Python で可）。
- `develop` および `main` ブランチへの PR で CI を実行する。`develop` PR では slow テストを除外し、`main` PR では全テストを実行する（H-0043）。

# 19. ディレクトリ構成

5 層カテゴリアーキテクチャ（§2.1）に基づく。各ディレクトリの所属 Layer を明示する。

```text
lizyml/
│
├── __init__.py                     公開面 (`__all__`: Model, FitResult, PredictionResult, TuningResult,
│                                   LizyMLError, ErrorCode, load_config, TaskType, RoundSummary,
│                                   BoundaryReport, BoundaryDimStatus, TuneProgressInfo,
│                                   TuneProgressCallback, __version__, __version_tuple__。H-0086)
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
│       ├── task.py                 TaskType (canonical, H-0075)
│       ├── target_encoder.py       TargetEncoder (H-0070)
│       └── search_dim.py           SearchDim, FloatDim, IntDim, CategoricalDim, DimCategory
│
│                                   ── Layer 0/4: Facade (core/ 内の特殊位置) ──
│   ├── model.py                    Model facade (組み立てと委譲のみ)
│   ├── _model_factories.py         splitter / inner_valid / estimator provider 構築
│   ├── _model_predict.py           predict / predict_proba 実体 (facade 委譲先)
│   ├── _model_plots.py             ModelPlotsMixin
│   ├── _model_tables.py            ModelTablesMixin (EstimatorProvider 経由)
│   ├── _model_metrics.py           _has_metric_content, _filter_metrics
│   ├── _model_persistence.py       ModelPersistenceMixin
│   ├── _model_tuning.py            ModelTuningMixin (tune orchestration、唯一の writer mixin、H-0091)
│   ├── _model_state.py             FitState / TuningState frozen snapshot (H-0074/H-0084)
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
│   ├── fingerprint.py              DataFingerprint 計算 (compute 関数)
│   └── validators.py               leakage validators (public API, H-0087)
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
├── persistence/                    ── Layer 3: Persistence ──
│   ├── exporter.py                 export() + AnalysisContext + FORMAT_VERSION
│   └── loader.py                   load() + format_version validation
│
└── codegen/                        ── Layer 3: Codegen (optional, H-0059/H-0073) ──
    ├── generator.py                generate_code() エントリ (Model.export_code から)
    ├── artifact_writer.py          pipeline_state.json / model.txt 等の書き出し
    ├── config_writer.py            config.json 書き出し
    └── templates.py                train.py / predict.py / test_equivalence.py テンプレ
```

> **codegen の seam に関する既知の逸脱（H-0088, #211）**: `generate_code()` は現状 `lgbm_params=` 等 estimator 固有引数を受け取り、H-0073 の estimator-agnostic 目標と不整合。`ExportParams` / provider metadata 経由への一本化と `templates.py`（800 行超）の分割は follow-up（[#228](https://github.com/nbx-liz/LizyML/issues/228)）。

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
- `Model` クラスは mixin で構成する（H-0042）。plot 系は `_model_plots.py`、table/accessor 系は `_model_tables.py`、persistence 系は `_model_persistence.py`、tune の orchestration は `_model_tuning.py`（`ModelTuningMixin`、H-0091）に分割し、`model.py` には core lifecycle（`__init__`, `fit`, `predict`, `evaluate`）とプライベートヘルパーのみを残す。`model.py` は `tune` を持たない（H-0091）。
- mixin は 2 種類に分ける（H-0091）: **診断用の read-only mixin**（`_model_plots` / `_model_tables` / `_model_persistence`）と、**orchestrator の writer mixin**（`_model_tuning`。変更を伴う `tune()` の lifecycle を走らせるので `self._*` に書く）。writer mixin はちょうど 1 つで、`tests/test_core/test_mixin_state_isolation.py` の read-only の静的検査（`_MIXIN_FILES`）から除外される（INV-2）。
- mixin は `_` プレフィックスの非公開モジュールとし、`Model` の import パス（`lizyml.core.model.Model`）は変更しない。
- 依存関係の切り離しが必要な箇所では Lazy Import を許容する。
- `Model._get_fit_state()` が返す `FitState` frozen dataclass を mixin の唯一の入口とする（H-0074 型定義 + factory + テスト、H-0077 で全 mixin 移行完了、H-0084 で `core/_model_state.py` へ移設）。`FitState` は `cfg / fit_result / refit_result / tuning_result / provider / metrics / y / X / run_dir / output_dir` の post-fit snapshot を持ち、mixin 本体からの `self._*` 直接アクセスを置換済み（plots / tables / persistence の 3 mixin は `_get_fit_state()` / `_get_tuning_state()` 経由）。
