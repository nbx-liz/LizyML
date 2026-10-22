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

| Layer | 名称 | 依存先 | 含まれるモジュール | 下の層から参照してよいモジュール（IF） |
|---|---|---|---|---|
| 0 | Foundation | なし | `core/exceptions.py`, `core/logging.py`, `core/registries.py`, `core/types/`, `core/param_domain.py`, `core/category_values.py`, `core/category_temporal.py`, `core/magnitude.py`, `core/tuning_codec.py`, `core/specs/`, `_extras.py`, `_version.py` | 全部 |
| 1 | Leaf | Foundation + 同じ層の中で許す参照 | `config/`, `data/`, `splitters/`, `features/`, `estimators/`, `metrics/`, `calibration/` | `splitters/base.py`, `features/pipeline_base.py`, `estimators/base.py`, `estimators/provider.py`, `metrics/base.py`, `metrics/registry.py`, `calibration/base.py` |
| 2 | Composition | Foundation + Layer 1 の IF | `training/`, `evaluation/`, `tuning/` | `training/results.py` |
| 3 | Optional | Foundation + Layer 1/2 の IF | `explain/`, `plots/`, `persistence/`, `codegen/` | なし |
| 4 | Facade | 全 Layer | `core/`, `__init__.py` | なし |

依存ルール:
- 各カテゴリは、自分より**下の Layer** と、下の表で自分のカテゴリに許した同じ層のカテゴリにだけ依存する。Layer 1 は Foundation、自分のカテゴリ、許したカテゴリだけに依存する（H-0130）。
- Layer 2 は Layer 1 の **IF（表の右の列）のみ**を参照する（具象クラスを import しない）。Layer 3 はそれに加えて Layer 2 の IF（`training/results.py`）を参照してよい。同じ層の別のカテゴリは参照しない。パッケージの `__init__.py` は IF に入れない（`estimators/__init__.py` は `LGBMAdapter` を再公開する）。
- 具象の**組み立て**と**具象型ディスパッチ**は **Layer 4（Facade）** が行う（H-0130 改訂 2）。
  - 組み立て: 宣言的な入力（Config、公開 API の引数、保存された識別子など）から、Layer 1 の境界 IF の具象実装を選び、その instance か factory を使う側へつなぐこと。境界 IF には、Protocol を構造的に満たす拡張点（§2.2 の `EstimatorProvider`）を含む。
    - 「選ぶ」には、宣言的な入力の種類か値（Config の型、discriminator の `name`、それから導いた値）で実装を選び分けることを含む。Config 自身の union による検証は、実装を選ぶことではない。
    - 次のものは組み立てに入らない: 同じカテゴリの中で、境界 IF を実装しない補助のクラスや値を作ること。保存したオブジェクトを読み戻すこと（pickle は fit の時に組み立てたものを戻すだけで、選ばない）。
  - 具象型ディスパッチ: 具象の実装型か、カテゴリをまたぐ具象の結果型が何であるかで、処理を分けること。組み立てとは別の規則である。
  - 例外は二つだけ。
    - §2.2 の provider が、自分の factory で adapter と pipeline を組み、自分の adapter の具象型を扱うこと（LightGBM の provider の `params_summary`・`build_export_params` の `LGBMAdapter` の判定）。
    - 各 Layer 1 カテゴリの registry の helper（`metrics/registry.py` の `get_metric`・`get_metrics_for_task`、`calibration/registry.py` の `get_calibrator`）が、宣言的な名前から登録された境界 IF の実装を選び、instance を作って IF として返すこと。呼ぶ側は helper に名前と引数を渡し、IF だけを受け取る。
  - `core/registries.py`（Layer 0）は、登録を蓄える共有の仕組みである。Layer 0 にあることは、ほかの層に、registry から直接引いて型を作る権限も、登録を書き換える権限も与えない。Layer 1〜3 では、実装のモジュールが自分の型を登録し、helper だけが引く。新しい registry、登録と引く役割の変更、helper を迂回して引くことには、Proposal が要る。
  - Layer 1〜3 の使う側は、境界 IF、Protocol、目印のクラス（`BaseCalibrationResult`）を通して扱う。
  - 得られるのは、新しいアルゴリズムを選ぶ場所が Facade の中にまとまり、`model.py` を変えずに済むことである。Config の union と Facade の provider の選択は変わる（§2.2 の手順 2・3）。
  - この規則は規範だが、テストでは確かめない。R0〜R5 が確かめるのは、静的な import の辺だけである。H-0131 の実装と、それに対応する architecture のテストが merge されるまで、次の 4 つはレビューだけで守る: (1) 組み立て（Config の型、discriminator の `name`、それから導いた値での選択を含む）、(2) 具象型ディスパッチ、(3) registry の helper を迂回して registry から引くこと、(4) 登録の書き換え（実装のモジュールが自分の型を登録することは許す）。Layer 1〜3 のモジュールを変える PR は、PR のテンプレートの確認項目で確かめる。
- カテゴリ間の**循環依存は禁止**する（カテゴリは `core/` の中はモジュールごと、ほかは最上位のパッケージごと）。

同じ層の中で許す参照（H-0130）:

| 参照する側 | 参照される側 | 根拠 |
|---|---|---|
| `estimators/` | `features/` | §2.2 の provider が pipeline の factory を組む（H-0053）。`UnseenPolicy` を受け取り、具象の pipeline を作る |
| `estimators/` | `metrics/` | LightGBM の feval への橋渡し（`estimators/lgbm/metric_bridge.py`）と、provider の指標の解決 |

表の読み方と検査（H-0129、H-0130）:
- `lizyml/` のどのモジュールも、表の「含まれるモジュール」のどれかの行に当たる。複数の行が当たるときは、最も長いパスの行が決める（`core/exceptions.py` は `core/` より長いので Layer 0）。`_version.py` は Git の管理外で、hatch-vcs が `pyproject.toml` の `version-file` に従って生成する。
- `tests/test_architecture/test_layer_rules.py` が、この表を読んで全ての import の文（関数の中と `if TYPE_CHECKING:` の中を含む）を確かめる。層の規則は import の文と、名前で読める動的な import で確かめる: 識別子 `import_module` と `__import__` は、`lizyml` 以外のパッケージを絶対の名前の定数で import する呼び出しにだけ使ってよい。実行時に親パッケージの `__init__.py` が読み込むもの、`eval` / `exec` で組み立てた import、文字列の中の import（`codegen/` のテンプレート）は数えない。
- 同じ層の中で許す参照の表は、行ごとの完全一致で読む: 前の行（`同じ層の中で許す参照（H-0130）:`）、空行ちょうど一つ、見出しの行、区切りの行（`|---|---|---|`）、1 行以上の中身の行。参照する側と参照される側は、バッククォートで囲んだ Layer 1 のカテゴリ `pkg/` を一つずつ書き、二つは異なり、同じ組は一度だけ書く。根拠は空でなく `|` を含まない。§2.1 の中で `|` で始まる行は、層の表かこの表のどちらかに属する。許した組は、少なくとも 1 本の import が使っていなければならない（使われない組は古びとして失敗する）。
- 具象の組み立てと具象型ディスパッチ（上の依存ルールの 3 つ目）は、このテストでは確かめない（H-0130 改訂 2 で、改訂 1 の検査を取り下げた）。
- 既存の違反は無い（H-0130 で #362 の 17 本を閉じた）。新しく見つかった違反は、Proposal で表（IF の列か、同じ層の中で許す参照）を変えるか、テストの例外の一覧に件数つきで載せる。一覧の import が消えても増えても、テストは失敗する。

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
1. `estimators/<name>/` に adapter と provider を作成し、config のクラスは `config/schema.py` に置く（`config/` は `estimators/` を import しない。§2.1）
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
- `tuning_table()` は `TuningResult.trials` を `pd.DataFrame` に変換して返す（列: `trial`, メトリクス名, 探索パラメーター名）。`tune()` 未実行時は `MODEL_NOT_FIT`。load したモデルは export 前と同じ表を返す。試行の履歴を持たない成果物（format 1〜3）から load したモデルでは、`tuning_table()`・`tuning_plot()`・`boundary_table()` は `MODEL_NOT_FIT`（context の `missing` が `"tuning_history"`）を出し、空の表や空の図は返さない（H-0134）。
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
- tune 済みモデルの artifact では、`metadata.json` の `tuning` ブロックと `tuning_history.json`（試行・round・境界の報告。H-0134）から `TuningResult` を復元し、load 後の再 `fit()` が tuned params を再現する。履歴を持たない成果物（format 1〜3）では 試行の無い `TuningResult` を復元し、履歴が無いことを覚えておく（上の 3 つの面がそれを言う）。ブロックの無い artifact は tuning result 無しで読む（§15.1、H-0086）。
- 復元する `best_smart_params` の各値は、smart の探索次元と同じ値の規則（§11.3、`provider.smart_value_rules()`）で、その artifact の config のタスクについて検査する（H-0121 改訂 2、[#347](https://github.com/nbx-liz/LizyML/issues/347)）。外れる値（v0.18.0 で文字列の真偽の次元で tune した artifact の `"false"` など。booster は真で訓練されており、記録と中身が食い違う）があれば `DESERIALIZATION_FAILED` で拒否し、context に `name`・`value`・`reason`（`smart_value_invalid` / `smart_value_unsupported_task`）を入れ、メッセージで tune し直して export し直すことを案内する。`format_version` 1 と 2 の両方に当てる。`best_smart_params` が空、規則どおり、または tune の結果の無い artifact は今までどおり読み込み、予測する。`best_smart_params` が mapping でない記録、provider が宣言しない名前を持つ記録、`feature_weights` を持つ記録も `DESERIALIZATION_FAILED` で拒否する（LizyML はそうした記録を書かない）。tune の結果のブロックが object でないか、LizyML が書く六つのキーを欠く artifact も `DESERIALIZATION_FAILED`。
- 保存した全ての booster（`fit_result.pkl` の fold のモデルと `refit_model.pkl` のモデル）について、LightGBM が booster の中に記録した目的関数（モデル文字列の `objective=` の行の最初の語。別名 `xentlambda` で学習しても `cross_entropy_lambda` と記録される）を読み、出力が確率でない目的関数（binary の `cross_entropy_lambda`）なら `DESERIALIZATION_FAILED` で拒否する（H-0125、[#307](https://github.com/nbx-liz/LizyML/issues/307)）。booster の `params` は読まない（復元した booster が `params` を保つかは LightGBM の版による。4.6.0 はモデル文字列からの復元で失い、4.7.0 は保つ。また学習の後に書き換えられ、記録とは別の値になりうる）。記録は学習に実際に使った目的関数なので、config・tune の結果・`fit(params=)` のどれで指定したかによらない。context に `path`・`booster`（`fold k` か `refit`）・`objective`。format_version 1 と 2 のどちらにも当て、format_version は変えない。v0.18.0 で `cross_entropy_lambda` の短い fit から作った artifact だけが対象で、他の artifact は今どおり読み込める。
- `model.params["metric"]` に native 指標のパラメータ（`{"huber": {"delta": 9.0}}` など、§14 の Metric Bridge）を記録した古い artifact は、読み込みと `predict` ができる（学習ではパラメータが使われていなかった）。再 `fit` と `export_code` は `CONFIG_INVALID`（H-0125、[#313](https://github.com/nbx-liz/LizyML/issues/313)）。

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

**この競合ルールが適用される入口（H-0094 / H-0121）。** スマート解決はパラメーター dict のマージより後段で走り、その結果が勝つため、ルールが適用されない入口では「受理して黙って置換」になる。

**H-0121 で、モデルパラメーターの 4 つの層の検査を `fit` / `tune` の学習前の 1 か所（`Model._merge_params`、`fit` と `tune` の両方が通る）に一本化した。** 各層を、有効なスマートパラメーター（config の値に、tune の結果の `best_smart_params` を重ねたもの）に対して `check_smart_managed_overrides` で検査する。検査は alias 対応で、綴りは LightGBM の `LGBM_DumpParamAliases` から導く。ネイティブの名前を書く 5 つのスマートパラメーター（`SMART_PARAM_TARGETS`）の 6 つの正規名と、その 18 の綴りのすべてを覆う: `auto_num_leaves` → `num_leaves`、`min_data_in_leaf_ratio` → `min_data_in_leaf`、`min_data_in_bin_ratio` → `min_data_in_bin`、`feature_weights` → `feature_contri` と `feature_pre_filter`、`balanced` → `scale_pos_weight`（binary のみ）。`num_leaves_ratio` は `auto_num_leaves` の修飾子で、自分では名前を書かない。拒否は `CONFIG_INVALID` で、綴り・管理しているスマートパラメーター・名前の来た層を挙げる。H-0121 以前は `model.params` に `LGBMConfig._validate_smart_params` の文字列一致の検査しかなく、18 の綴りのうち拒否されたのは文字どおりの 3 つ（`num_leaves`、`min_data_in_leaf`、`min_data_in_bin`）だけだった（12 は別名とスマートの値の両方が `lgb.train` に届き、3 は黙って置き換わった。例: `scale_pos_weight: 10` が `0.951` で学習した、[#280](https://github.com/nbx-liz/LizyML/issues/280)）。その文字列一致の 3 件の拒否は削除した（覆う範囲の違う 2 か所の検査はずれる、DC3）。そのため、この 3 件の拒否の時期は config の構築時から `fit` / `tune` の時に移った。

| 入口 | 状態 |
|---|---|
| `model.params`（config） | **H-0121**: `_merge_params` で alias 対応、6 つの正規名・18 の綴りすべてを `fit` / `tune` の学習前に拒否する（層名 `model.params`）。config の構築時には拒否しない。tune では、smart の探索次元が試行の中で有効にしうるスマートパラメーターとの衝突も study の前に拒否する（H-0121 改訂 1、§11.3） |
| 既定の固定パラメーター（`first_metric_only`、`metric`） | **H-0121**: 同じ検査に入る（層名 `provider default fixed params`）。新しい study の `tune` では `_merge_params` はこの層を重ねない（objective が trial ごとに重ねる）が、検査はする。smart の探索次元が有効にしうる衝突も拒否する（H-0121 改訂 1、§11.3） |
| tune の結果の `best_model_params` | **H-0121**: 同じ検査に入る（層名 `tuning best_model_params`）。復元した tune の結果にも当たる。有効なスマートパラメーターは `best_smart_params` を重ねた後の値なので、config が `auto_num_leaves: false` でも `best_smart_params` が `auto_num_leaves: true` なら `max_leaves` は拒否される |
| `fit(params=...)` | **H-0094 で拒否する（5 件すべて）。** 有効なスマートパラメーターが書くネイティブ名は `CONFIG_INVALID` とし、どのスマートパラメーターが管理しているかを名指しする。H-0121 で他の層と同じ 1 か所の検査になった（挙動は変わらない） |
| `tuning.optuna.space` (`category: model`) | H-0099: reject names claimed by active smart parameters, including aliases, before study creation. Validate the resolved explicit/default/resumed space, including smart dimensions that can activate a conflicting owner. To tune native leaves directly, disable `model.auto_num_leaves`; sampled values then reach training. **H-0121 改訂 1**: smart の次元は一つずつではなく、取りうる値の組み合わせのすべて（状態ごとの代表の値の直積）に対して検査する（§11.3） |
| `tuning.optuna.space` (`category: smart`) | **H-0121**: smart パラメーターとして宣言されていない名前は study の前に `CONFIG_INVALID`（ネイティブ名ならメッセージで `category: model` を案内する）。所有者が真になりえない smart の次元（`auto_num_leaves` が真になりえないときの `num_leaves_ratio`）は、利用者が書いたものなら拒否し、既定の空間から来たものなら落とす（§11.3）。**H-0121 改訂 2**: 次元が取りうる値のすべてを、provider がスマートパラメーターごとに宣言する値の規則（型は厳密、比率はこの節の範囲）で検査し、外れれば study の前に拒否する（§11.3） |
| `tuning.optuna.space` (`category: training`) | H-0099: accept only `early_stopping_rounds` and `validation_ratio`, the two training override consumers. Reject other names with `CONFIG_INVALID` before study creation. |

### auto_num_leaves（葉の数の自動算出）

- `auto_num_leaves: bool = True`: 有効時、`max_depth` から `num_leaves` を自動算出する。
- `num_leaves_ratio: float = 1.0`（`0 < ratio ≤ 1`）: 基準値に対する割合。
- 算出ロジック:
  - `params.max_depth` が未指定、または 0 以下（制限なし。LightGBM の「`<= 0` means no limit」と同じ。H-0140 より前は 0 が 8 葉を導いた）→ 基準値 = `131072`
  - `params.max_depth` が指定されている → 基準値 = `2 ^ max_depth`
  - `num_leaves = clamp(ceil(基準値 × num_leaves_ratio), 8, 131072)`
- **メモリの警告（H-0140）**: LightGBM は、木が実際に育つ葉の数ではなく、指定した `num_leaves` の大きさでヒストグラムの領域と木の配列を確保する（150 行でも 131072 葉で約 12 GB）。`Model.fit` と `Model.tune` は、学習の前に、LightGBM が受け取る `num_leaves` が 16384 以上か、増える分のメモリの見積もりが 1 GiB 以上なら `UserWarning` を出す。値は変えない（運用のための知らせである）。
  - 判定の値は adapter が `lgb.train` に渡す確定後の params（`num_leaves` とその別名、何も無ければ LightGBM の既定の 31）と `num_boost_round`・`num_class` から読む。原因は「`auto_num_leaves` が `max_depth` から導いた」「明示した」「LightGBM の既定」のどれかで、対処の文も原因ごとに違う。
  - 見積もりは、ヒストグラム（約 16 バイト × 特徴量の bin の合計 × `num_leaves`）1 回分と、同時にメモリに残る booster の数だけの木（約 80 バイト × `num_leaves` × round 数 × class 数）の和。LizyML の adapter は `keep_training_booster=True` で学習するので、学習の済んだ booster も指定の大きさの木を持ち続ける。残る数は `Model.fit` で outer の fold の数 + refit、tune の 1 つの trial で outer の fold の数。近似で、早期終了では上限、isotonic の校正の booster は数えない。
  - 回数: `Model.fit` は全ての fold と refit を合わせて 1 回。`Model.tune` は呼び出しごとに最初に条件に当たった trial で 1 回だけ警告し、以降の trial は logger に記録する。警告を例外にする設定では学習の前に止まる（tune では `TUNING_FAILED` に包まれる）。
  - 判定は provider の任意の capability `ResourceWarningProvider`（§14.4）で、LightGBM の provider だけが持つ（`estimators/lgbm/memory_check.py`）。
- 制約: `auto_num_leaves` が有効なとき、`num_leaves` を別名（`max_leaves` など）を含めてどの層に書いても `fit` / `tune` の学習前に `CONFIG_INVALID`（H-0121。config の構築時には拒否しない）。`num_leaves_ratio` は `auto_num_leaves` が真のときだけ読まれる。

### データサイズ相対比率パラメーター

学習データの行数に対する割合で指定し、CV の各 fold 内で inner validation 分割後の実学習データ行数（`n_rows_inner_train`）を基準に絶対値に変換する（H-0036）。

- `min_data_in_leaf_ratio: float | None = 0.01`（`0 < ratio < 1`）→ `min_data_in_leaf = max(1, ceil(n_rows_inner_train × ratio))`
- `min_data_in_bin_ratio: float | None = 0.01`（`0 < ratio < 1`）→ `min_data_in_bin = max(1, ceil(n_rows_inner_train × ratio))`
- `n_rows_inner_train` の定義: outer fold の学習データから inner validation（early stopping 用）を分割した後の行数。early stopping が無効（inner validation 分割なし）の場合は outer fold の学習データ行数を使用する。
- fold ごとに `n_rows_inner_train` が異なる場合、各 fold で個別に解決する。
- 制約: ratio 指定と対応する絶対値パラメーター（`params.min_data_in_leaf` 等、別名を含む）の同時指定は `fit` / `tune` の学習前に `CONFIG_INVALID`（H-0121。config の構築時には拒否しない）。

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

**regression は目的変数で層化しない（H-0120 改訂 2）**: 層化はクラスの比率を保つための手法で、回帰の目的変数にはクラスが無い。`task: regression` で `split.method: stratified_kfold` / `stratified_group_kfold`、明示した `split.groups.stratify: true`（`blocked_group_kfold`）、明示した `training.early_stopping.inner_valid` の `stratify: true` のどれかを指定すると、`fit` と `tune` は学習の前に `CONFIG_INVALID` で拒否する（`build_splitter` の `check_regression_stratification`）。目的変数の値（連続値か整数値か）によらない。

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

公開クラスの `__init__` の既定値付き引数（76 個、AST で数えた母集団）を、値の出どころで分類した。**Config のキーの値がそのまま渡る 59 個**は、どのキーから来るかを経路ごとに `tests/test_config/_knob_registry.py` の台帳に書く（§17 の `output_dir` と、§5.4 の表に行の無い `calibration.params` も含む）。`tests/test_config/test_knob_reachability.py` は、経路ごとに本物の `fit` / `tune` を実行してコンストラクタが受け取る値を確かめ、どの行にも既定でない値を設定したセルが少なくとも 1 つある。残りの 17 個をこの表に書く。表と `tests/test_config/_knob_registry.py` の台帳は同じテストが照合する。

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
| `CategoricalEncoder.needs_stand_in` | internal | 配線: estimator が保存できない値の判定（LightGBM の provider が渡す、H-0133） |
| `NativeFeaturePipeline.needs_stand_in` | internal | 配線: estimator が保存できない値の判定（LightGBM の provider が渡す、H-0133） |

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
6. `Tuner` の責務は Optuna study の管理のみ。objective クロージャは `Model` 側で構築する（公開の `Tuner.tune` の `step` の扱いは §11.5 の制約。H-0124）

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
- **train.py**: Feature pipeline fit → LightGBM refit（全データ学習。LizyML の refit と同じ inner valid の分割・行の並び・重み・カテゴリの符号で学習し、同じデータなら校正前の予測が一致する、§15.4 の H-0120）→ OOF 生成（軽量 CV。fold は `config.json` の `split` ブロックから outer CV を再現する、§15.4）→ Calibrator fit
- **predict.py**: Feature transform → LightGBM predict → Calibration apply
- **config.json**: ハイパーパラメータ・特徴量定義・校正設定を集約。コード編集なしでパラメータ変更可能
- 生成コードは `import lizyml` を含まない。依存は `lightgbm` / `numpy` / `pandas` / `scikit-learn`（学習時のみ）/ `pyarrow`（parquet の読み込み、H-0120）
- `test_equivalence.py` で `Model.predict()` と codegen 出力の一致を `rtol=1e-7` で検証
- 初期実装は LightGBM のみ対応。将来の EstimatorProvider 拡張で他アルゴリズムにも対応可能
- Calibrator 保存形式: Platt → JSON (a, b)、Beta → JSON (a, b, c)、Isotonic → Booster テキスト
- **feval metric 対応（H-0066）**: ユーザー指定の feval metric（f1, brier, ece, precision_at_k, accuracy, rmsle, r2, smape, wape, mase。`smape` / `wape` は H-0071、`mase` は H-0132）を `config.json` の `feval_metrics` フィールドに記録し、`train.py` 内に pure numpy で再実装する。`mase` の scale は、生成 `train.py` が `build_feval_from_config(y_train)` に渡す、booster が学習する行（inner valid があれば `y.iloc[tr]`）の目的変数から作る（H-0132）。feval metric 未使用時は `feval_metrics: []` で後方互換を維持

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
  - `eval_history`（学習曲線）の指標は LightGBM が計算した値で、LightGBM の名前と定義を持つ（native 指標。feval の指標は LizyML の定義、§14 の Metric Bridge）。そのため `FitResult.metrics` の同じ名前の指標と値が違うことがある。今そうなるのは `huber`（LightGBM の閾値は `alpha`、既定 0.9。LizyML は `delta`、既定 1.0）と `mape`（LightGBM は `|y-p| / max(1,|y|)` の比率。LizyML は `|y-p| / |y|` の百分率）である（H-0125、#354）。データの契約（`history` の形とキー）は変えない。
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
- `oof_raw_scores`（`np.ndarray | None`。校正が有効なとき、OOF の生スコア（`predict_raw` の出力、ロジット）を `oof_pred` と同じ形・同じ行の順で持つ。validation fold に覆われない行は `oof_pred` と同じく NaN。校正の cross-fit と `C_final` はこれを入力にし、`predict()` も生スコアを `C_final` に渡す（H-0030）。校正が無効なら `None`。校正が有効でもこれが `None` の旧形式 artifact は、`predict()` で確率を `C_final` に渡す（§18.1 の Legacy calibration path）。校正の規則は §12.1）
- `run_meta`
  - `lizyml_version / python_version / deps_versions / config_normalized / config_version / run_id / timestamp`
  - 型: `lizyml_version`（`str`）、`python_version`（`str`）、`deps_versions`（`dict[str, str]`）、`config_normalized`（`dict`）、`config_version`（`int`）、`run_id`（`str`、UUID 文字列）、`timestamp`（`str`、ISO 8601、UTC）。
- `target_encoder`（H-0070 で v2 に追加。現行は format_version=5）
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
- `applied_sample_weight`（`metadata.json` の最上位キー。その fit の refit が使った行ごとの重みの規則、`"balanced"` か `"none"`。H-0120 改訂 1）
- `declared_categories`（`metadata.json` の最上位キー。その fit の入力の DataFrame で `category` dtype だった特徴量の列と、そのカテゴリ。H-0120 改訂 4）

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
    - LightGBM が書けない値を持つ列（§15.3、H-0133）でも、未知の値の検出と置換は元の値で行い、そのあとでカテゴリを位置の番号に置き換える。
    - **CV の検証 fold にも同じ方針が効く。** ただし既定の `auto_categorical: true`（または `categorical` 指定）の列はデータ構築時に全行の値で `category` 型になるため、どの fold でも未知にならない。fold ごとに未知カテゴリが生じるのは、`auto_categorical: false` で pipeline がカテゴリとして扱う文字列列だけである。その場合 `error` では fit が止まり、`mode` / `nan` の置換は fit 中は報告されない（`FitResult` に警告の通り道が無い）。
    - SHAP 重要度は、fold k の検証行を fold k 自身の pipeline 状態（`FitResult.pipeline_state_per_fold`）で変換して fold k のモデルを説明する（H-0114）。上記の文字列列でも、SHAP 重要度が見る符号化と置換は OOF 予測が見たものと同じで、fit 中と同じく報告しない。`error` では未知の値があれば fit が先に止まるので、fit が通ったモデルの SHAP 重要度はこの理由で送出しない。
- 推論時の列検査（不足列は `DATA_SCHEMA_INVALID`、余剰列は警告して除外）は pipeline に渡す**前**に facade（`Model.predict`）で行い、どの pipeline 実装でも省略されない（H-0104）。
- 同じ検査で、学習時に数値だった列（`FitResult.dtypes` の記録）が予測時に数値でない dtype で届いたら `INCOMPATIBLE_COLUMNS` を送出する（context `columns` の各要素は `column` / `fit_dtype` / `predict_dtype`。H-0106）。不足列の `DATA_SCHEMA_INVALID` が先に効く。学習時に `category` だった列はこの dtype の規則で検査しない。
- 学習時に `category` だった列に、pandas がカテゴリの値にできない浮動小数（`float16`、float64 より広い `longdouble`）で値が届いたとき（列の dtype でも、`object` 列の中の numpy のスカラーでも）、エンコーダは値を float64 にして照合する（`comparable_values`）。float64 にして値が変わる要素はどのカテゴリとも一致させず、未知として `unseen_policy` に従い、届いた元の値の文字列で報告する。欠損値は比較せず、どの方針でも欠損のまま。生成コードの `encode_column`（`train.py` / `predict.py`）も同じ規則を写す（H-0127、#309）。

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
- `parse_space` は `float` / `int` 次元の範囲を parse 時に検査する（H-0078）: `low >= high` は `CONFIG_INVALID`、`log=True` で `low <= 0` も `CONFIG_INVALID`（context `param` / `low` / `high` / `log`。H-0123 から書式の拒否と同じ `dim`・`key`（`low`）・`reason`・`surface` も入る）。Optuna が trial の中で出す汎用のエラーより前に、設定の誤りとして名指す。
- **探索空間の書式（H-0123）**: `tuning.optuna.space` と `tune(space=...)` は同じ書式で、どちらも `parse_space` が検査する（`tune` が空間を解決するとき。config の空間は `tune` の引数の検査の段で、study を作る前・書く前）。書式は YAML と JSON が作る型（数、真偽、文字列、list）で書く前提で、書式の外の指定は別の探索に読み替えず `CONFIG_INVALID` で拒否する（context に次元の名前 `dim`、キー `key`、理由 `reason`、空間 `surface` = `tuning.optuna.space` か `tune(space=)` が必ずある。`dim` は空間が mapping でないときだけ `None`、`key` は空間全体か次元全体についての拒否では `None`）。空間は `str` の名前から次元の指定（mapping）への mapping で、次元ごとに:
  - `type`: `float` / `int` / `categorical`。
  - キーは、`float` と `int` では `type`・`low`・`high`・`log`・`step`・`category`、`categorical` では `type`・`choices`・`category` だけ。綴りの誤り（`lo` など）も拒否する。
  - `low` / `high`（`float` / `int` で必須）: `int` か `float`（`bool` は除く）、numpy の整数・浮動小数のスカラー（`numpy.bool_` は除く。H-0095 のとおり素の値に直す）、または数の表記に完全一致する文字列 `[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?`（ASCII の数字だけ。PyYAML は YAML 1.1 で `1e-4` を文字列として読む）。`"nan"`・`"inf"`・`"0x10"`・`"1,000"` は数の表記ではない。値は有限であること。`int` の次元では整数の値であること（`16.0`・`"16.0"`・`"1e2"` は 16・16・100 に直し、`1.7` と `"1.7"` は切り捨てずに拒否する）。
  - `log`（任意、既定 `false`）: `bool`（YAML の `true` / `false` / `yes` / `no`）。文字列の `"false"` は拒否する。
  - `step`（`float` / `int` で任意、既定は刻み無し。H-0124）: 刻み。値の型は `low` / `high` と同じで、有限で 0 より大きいこと（`int` の次元では整数の値で 1 以上）。読めない値は `low` / `high` と同じ理由（`bound_not_a_number`・`bound_not_finite`・`bound_not_integral`）を `key` `step` で、0 以下は理由 `step_not_positive` で拒否する。
    - **格子は原点 0**: 次元の格子は `step` の整数倍で、探索するのは `low` から `high` までの倍数。`low` と `high` はどちらも `step` の倍数でなければならない。判定は正確な有理数で行う（`float` の値 `x` は `Fraction(Decimal(str(x)))`、`int` はそのまま。`str` は最短の表記なので `[0.1, 0.3]` step 0.1 や `[0.5, 1.0]` step 0.05 を 2 進の誤差で拒否しない。`decimal` の context の精度に依らない）。
    - 倍数でない境界は `CONFIG_INVALID`（理由 `bound_off_grid`、`key` は `low` か `high`。両方なら `low`）。メッセージは格子の上の範囲を一つ示す: 内側の範囲 `[ceil(low/step)*step, floor(high/step)*step]` が二つ以上の格子の点を持てばそれ（`[0.15, 1.0]` step 0.1 → `[0.2, 1.0]`）、持たなければ外側の範囲 `[floor(low/step)*step, ceil(high/step)*step]`（`[0.15, 0.16]` step 0.1 → `[0.1, 0.2]`）。`log` の次元で外側の下端が 0 になるときは `step` にし、それで一点になれば上端を一つ外の格子の点にする（`[0.0005, 0.0008]` step 0.001 → `[0.001, 0.002]`）。Optuna のように上端を黙って切り詰めることはしない。
    - **抽出と訓練に使う値**: `log` でない次元は Optuna の `step`（`suggest_float(step=)` / `suggest_int(step=)`）で格子の点を抽出する。`log` の `float` の次元は刻み無しに対数で抽出し、一度だけ格子に丸める。`log` の `int` の次元は、`step` が 1 なら刻みが無いのと同じ（`IntDistribution(log=True)`）、`step > 1` なら float の対数（`FloatDistribution(log=True)`）で抽出して一度だけ整数の格子に丸める（先に整数に丸めない）。丸めは生の値 `v` から `k = round(v / step)`（正確な有理数で、偶数への丸め）を `[low/step, high/step]` に収め、`k * step`（`float` の次元は最も近い `float`、`int` の次元は `int`）とする。刻みのある次元はすべてこの値で訓練し、この値を試行の値として報告する（§11.5。Optuna の刻みの抽出値の 2 進の誤差 `0.30000000000000004` は `0.3` になる）。同じ格子の点に丸まる試行も、それぞれ評価する。`log` の次元の格子のいちばん小さい点は `step`。
    - `step` の無い次元は今までと同じ抽出で、Optuna に `step=` を渡さない。
  - **大きさ（H-0124 改訂 1）**: Optuna は数を `float` で抽出するので、抽出できる大きさに限る。`float` の次元は、幅 `high - low` を正確に計算した値（`float` の値そのもので計算する）が `float` の最大値（`sys.float_info.max`）以下であること（各境界が有限でも幅が有限でなければ Optuna は `OverflowError: Range exceeds valid bounds` で抽出できない）。外れれば理由 `range_not_finite`（`key` は `high`）。`int` の次元は、`low`・`high`・`step` の絶対値が `2**52`（4503599627370496）以下であること（Optuna は `int` の次元を半刻み広げた `float` で抽出し、半整数が正確なのは `2**52` まで。`2**53` からは黙って粗く探索し、さらに大きいと失敗する）。外れれば理由 `bound_too_large`（`key` は `low`・`high`・`step` の順で最初に外れたもの）。`[-8e307, 8e307]`、`[2**52 - 10, 2**52]`、`[-2**52, -2**52 + 10]` は受け付け、`[-1e308, 1e308]`、`int` の `[0, 2**52 + 1]` は拒否する。同じ規則を、公開の `Tuner.tune` は直接作った数の次元すべてに（刻みの有無によらず。§11.5）、round の受け付けの検査は round の実効空間（拡張した次元と provider の既定の空間の次元を含む）の数の次元すべてに、round の記録の前に当てる。study の系譜の decode も、この規則に外れる次元と、有限の `float` でない `min_allowed` / `max_allowed` を読まない（§11.5.3）。
  - `choices`（`categorical` で必須）: 空でない `list` か `tuple`。文字列・`set`・dict・`range`・numpy の配列・generator は拒否する。各値は素のスカラー（`str` / `int` / `float` / `bool` / `None`。numpy のスカラーは拒否、#287）。
  - `category`（任意、既定 `model`）: `model` / `smart` / `training`（§11.2）。
- config の `tuning.optuna.space` は、schema の検証で空間と各次元の指定（mapping であればよい）を plain な `dict` に写す。名前の検査は `dict` からだけ `category` を読むため（H-0123）。`config/` は `tuning/` を import しないので、書式の検査は `tune` で行う。

## 11.2 SearchDim カテゴリ

SearchDim にカテゴリ属性を持たせ、Tuner がパラメーターの適用先を区別する。

- `model`: `LGBMAdapter.params` に直接渡す（既存 SearchDim の挙動）
- `smart`: スマートパラメーター（`num_leaves_ratio` 等）として `resolve_smart_params()` に渡す。fit / tune で同一の dict ベース `resolve_smart_params()` を使用する（H-0050）
- `training`: trial ごとに `EarlyStoppingConfig` / `InnerValidStrategy` を再構築する

次元の指定の `category` を省くと `model`。三つ以外の値は `CONFIG_INVALID`（§11.1、H-0123）。

`FloatDim` / `IntDim` は任意の `min_allowed` / `max_allowed`（既定 `None`）を持つ。パラメーターとして意味のある範囲で、境界拡張（§11.5）がこれを超えないようにする。`Model.tune` は既定の空間にも利用者の空間にも `provider.parameter_bounds(task)`（§14.4）を付ける（H-0078）。

`FloatDim` / `IntDim` は最後の欄に任意の `step`（`FloatDim` は `float | None`、`IntDim` は `int | None`、既定 `None` = 刻み無し）を持つ（H-0124、§11.1）。位置引数で作る今までの呼び出しはそのまま動く。境界を付ける（`attach_bounds`）、拡張する（`expand_dims`）、系譜から読む（§11.5.3）のどれも `step` を引き継ぐ。

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
conflict refusals.

Merge applies provider fixed defaults with precedence base < fixed < sampled;
replace applies no fixed defaults. Fresh studies start from Config model/smart
parameters rather than previous best parameters, and their training conflict
checks exclude prior training overrides.
Admission also checks training ownership introduced by the current resolved
space against effective native parameters and model dimensions before study
creation, including inherited defaults and aliases.
Export/load retains the effective policy independently of subsequent Config
changes; an explicit empty policy differs from absent legacy metadata.
Automatic boundary
expansion remains enabled only for merge with an empty/omitted user space;
partial spaces require explicit opt-in. Older saved artifacts without the mode
retain replacement for nonempty spaces when loaded, preserving re-fit behavior.

**resume の試行の基と探索空間（H-0122）。** どの round の試行も `config ⊕ fixed ⊕ その試行のサンプル値` で学習する。`fixed`（`space_mode` で決まる固定パラメーター）は、study を作ったときに study の記録（`lizyml:study`、§11.5.3）に書いたものを使い、Model のフィールドと食い違えば study の記録を正とする。resume の round は前回の最良値を試行の基に重ねない。round の探索空間は、`tune(space=...)`（§11.5.2）があればそれを study の `space_mode` で解決したもの、無ければ直前の round の記録（宣言空間と、拡張を含む実効空間）から決める。config の `tuning.optuna.space` と `space_mode` は study を作るとき（round 1）だけ読み、resume でそれらが study の記録と違えば `CONFIG_INVALID` とする。空間の変更は古い試行を消さない: 目的関数が同じなので、古い試行は観測として残り、最良値の候補であり続ける。新しい study が要るのは、§11.5.2 で受け付けない変更のときだけ。`fit` が成功した tuning の結果を使う点は変わらない。

**smart の探索次元の受け付け（H-0121、[#299](https://github.com/nbx-liz/LizyML/issues/299)）。** `category: smart` の次元は smart 解決を通してしか学習に届かない。何にも読まれない次元は、サンプルされ `best_params` に記録されて、何もしない。

- **宣言されていない名前**: `provider.smart_param_names()` に無い名前の `category: smart` の次元は、解決済みの空間に対して study の前に `CONFIG_INVALID` で拒否する（`validate_tuning_dimensions`）。ネイティブの LightGBM の名前なら、メッセージで `category: model` を案内する。
- **消費者が無効な次元**: 所有者のある smart パラメーター（`provider.smart_param_owners()`、§14.4。LightGBM では `num_leaves_ratio` → `auto_num_leaves`）の次元は、所有者が「真になりえない」とき無効とする。同じ空間に所有者の名前の smart の次元があれば、その次元が取りうる値で決める（categorical の選択肢に真が 1 つも無い、または数値の範囲が 0 だけなら真になりえない）。無ければ有効な smart の値（新しい tune が `_merge_params` から受け取る config の値）で決める（偽か `None` なら真になりえない）。
  - 利用者の空間から来た無効な次元（`replace` の空間、または `merge` で利用者が書いた次元）は、study の前に `CONFIG_INVALID` で拒否する。利用者は自分の指定が効いていないことに気づけないため。
  - 既定の空間から来た無効な次元（`merge` で利用者が上書きしていない既定の次元。`lizyml/core/_tuning_plan.py` の `resolve_config_space` が出どころを返す）は空間から外し、外した名前を INFO のログ（`event='tune.drop_dims'`）に残す。study の探索次元にも `best_params` にも入らない。`auto_num_leaves: false` でネイティブの `num_leaves` を直接チューニングする H-0099 の案内どおりの設定は、これまでどおり動く（既定の空間の `num_leaves_ratio` が外れる）。
- **範囲**: 消費者が無効な次元の規則（上の 2 項）は、round の宣言空間を解決するときに当てる。round 1 を始める tune（`resume=False`、storage に無い名前への `resume=True`、round の記録が無い study（状態 `"uninitialized"` / `"no_round"`、§11.5.3）への `resume=True`）では config の空間に、`space=` を渡した round（§11.5.2）ではその宣言に当てる。`space=` の無い resume の round は直前の round の宣言を引き継ぐので、当て直さない（H-0122 で、resume のたびに今の config の smart の値を読む H-0121 の条項を外した。smart の値の変更は目的関数の照合で拒否される）。`tune` は空間を解決する前に、storage があれば `optuna.get_all_study_names(storage)` で名前があるかを調べて決める（調べてから study を作るまでの間に別のプロセスが同じ名前の study を作ることは考えない。一つの study に書くプロセスは一つという前提で、ライブラリは強制しない）。H-0072 の、`resume=False` で storage の既存の study に `load_if_exists=True` で再接続する経路は H-0122 で無くなった（`resume=False` で既存の study を指すと `CONFIG_INVALID`、再接続は `resume=True`。§11.5.1）。宣言されていない名前の拒否は、範囲によらず当てる。

**smart の探索次元が有効にしうる衝突（H-0121 改訂 1、[#341](https://github.com/nbx-liz/LizyML/issues/341)）。** `category: smart` の次元は試行ごとにスマートパラメーターの値を変えるので、config では無効なスマートパラメーターが試行の中で有効になり、層の名前を管理しうる（§5.3 の競合ルール）。そのまま学習すると、試行は層の名前とスマートパラメーターが書く名前の両方で学習し、LightGBM は正規名の値を採るので、利用者の値は黙って捨てられる。

- **検査**: tune の各 round の受け付けの検査（`check_round_space`。新しい study でも resume の round でも、round の実効空間に対して、study に何かを書く前に走る。H-0122 の INV-R4）で、試行に重なるパラメーターの層（`model.params` と、その round の固定パラメーター `fixed`、層名 `provider default fixed params`）を、round の smart の次元が取りうる値の組み合わせのすべてに対して、`check_smart_managed_overrides` で検査する。`category: model` の次元（`validate_tuning_dimensions`）も同じ組み合わせに対して検査する（層と model の次元で規則は一つ）。有効なスマートパラメーターの値に対する検査はどちらもそのまま残る（層は `_merge_params`、model の次元は `validate_tuning_dimensions` の最初の検査）。
- **組み合わせ**: smart の各次元の候補は、categorical なら選択肢のすべて、数値なら下限と上限に、範囲が 0 をまたぐ（下限 ≤ 0 ≤ 上限）ときは 0 を足したもの。それを状態（`None` か、偽か、真か）ごとに代表の値一つに絞り、次元の間で直積を取る（次元の数を k として 3^k 通り以下。LightGBM の既定の空間では 1 通り）。次元の無いスマートパラメーターは有効な値のまま。直積にするのは、provider の有効化の規則が複数のスマートパラメーターの組で決まる場合も覆うため。候補を絞る関数は一つで、両方の検査が使う（`lizyml/core/_smart_combinations.py`）。
- **provider の約束**: `smart_managed_param_names(smart, task)` の結果は、各スマートパラメーターの値に、その真偽と `None` かどうかを通してだけ依る（§14.4）。代表の値一つで状態を代表できるのはこの約束による。LightGBM の provider は 3 つのタスクで実行して確かめる（`tests/test_estimators/test_smart_managed_contract.py`）。
- **拒否**: `CONFIG_INVALID`。メッセージは名前、それを管理するスマートパラメーター、名前の来た層、そのスマートパラメーターを有効にする探索次元とその値、その空間（`tuning.optuna.space` か `tune(space=)`）を挙げる。context の `managed` の各要素に `smart_dimensions` と `space_surface` を足す。
- **例**: config が `auto_num_leaves: false` で `model.params.max_leaves: 12`、空間に `auto_num_leaves`（categorical `[True]`、`category: smart`）→ 拒否。config が `min_data_in_leaf_ratio: null` で `model.params.min_child_samples: 3`、既定の空間（`min_data_in_leaf_ratio` の次元を持つ）→ 拒否。選択肢が `[False]` だけの `auto_num_leaves` の次元は何も有効にしないので受け付ける。

**smart の探索次元が取る値（H-0121 改訂 2、[#347](https://github.com/nbx-liz/LizyML/issues/347)）。** `category: smart` の次元の値は、そのまま試行のスマートパラメーターになり、スマート解決（と上の受け付けの検査）はそれを真偽と `None` で読む。config の項目（`LGBMConfig`）は型と範囲を検査するが、探索次元の値はそれまでどこでも検査されず、`auto_num_leaves` の選択肢 `["false"]` は真として訓練され、0 以下の比率は黙って別の値になっていた。

- **値の規則**（provider がスマートパラメーターごとに一度だけ宣言する。`provider.smart_value_rules()`、§14.4）。次元の取りうる値のすべて（数の次元は `low` と `high`、categorical は選択肢のすべて）が満たすこと。型は厳密に判定する: 真偽は `type(v) is bool`、数は `type(v) is int` か `type(v) is float`（`bool`、numpy のスカラー、`Decimal`、文字列は数でない）。NaN と無限大は範囲の比較で外れる。

  | スマートパラメーター | 次元 | 値 |
  |---|---|---|
  | `auto_num_leaves` | categorical のみ | `True` / `False` |
  | `balanced` | categorical のみ | `True` / `False` / `None`。regression では `True` を含めない（理由 `smart_value_unsupported_task`。`None` は regression で無効の意味なので受け付ける） |
  | `num_leaves_ratio` | 数、categorical | 数で `0 < v <= 1`。`None` は拒否 |
  | `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` | 数、categorical | 数で `0 < v < 1`。categorical はそれに加えて `None`（その母数を解決しない） |
  | `feature_weights` | なし | 探索次元にできない（値が dict で、選択肢は素の値に限られる） |

  比率の範囲は config の検査（`LGBMConfig._validate_smart_params`）と同じ宣言（`lizyml/core/types/smart_ratio.py` の `SMART_RATIO_RANGES`。`config/schema.py` は再公開する。H-0130）から読み、二重にしない（DC3）。
- **拒否**: `CONFIG_INVALID`、context に `surface`（`tuning.optuna.space` か `tune(space=)`）・`dim`・`key`（数の次元は `low` / `high`、categorical は `choices`）・`value`（規則に外れた最初の値）・`reason`（`smart_value_invalid`、regression の `balanced=True` は `smart_value_unsupported_task`）。真偽のスマートパラメーターの数の次元と `feature_weights` の数の次元は、`key` を `low`、`value` をその `low` として報告する。
- **位置**: round の受け付けの検査（`check_round_space` が呼ぶ `validate_tuning_dimensions`）で、round の実効空間のすべての `category: smart` の次元に当てる。config の空間、`tune(space=...)`、provider の既定の空間、拡張した次元のどれにも、study と round を書く前に当たる（H-0122 の INV-R4）。round の宣言空間を解決するとき（`admit_smart`、上の H-0121 の所有者の規則）にも、所有者の規則が値を真偽で読む前に当てる（`[0]` は「真になりえない所有者」ではなく型で拒否される）。
- **拡張の上限**: provider の `num_leaves_ratio` の拡張の上限は 1.0（config の範囲の上端。下限 0.1）。既定の空間の `num_leaves_ratio`（`0.5..1.0`）が上端で最良になっても拡張は 1.0 で止まり、その次元の報告は `expanded=False`・`clamped_to_bound=True`（§11.5）。v0.18.0 までは 2.0 まで広がり、1.0 を越える値が最良値、`fit`、export に流れた。
- **tune の結果**: `restore_tuning`（§11.5.5）と `Model.load`（§4.2）は、記録された `best_smart_params` を同じ規則で検査する。

**目的関数と指標の値の検査（H-0125、[#307](https://github.com/nbx-liz/LizyML/issues/307)、[#313](https://github.com/nbx-liz/LizyML/issues/313)）。** tune は、round の受け付けの検査（`check_round_space`。新しい study でも resume の round でも、study を作る・再接続して何かを書く前。H-0122 の INV-R4）で、試行が受け取る目的関数と指標の値を fit と同じ規則（`check_objective_compatible`、`resolve_user_metric`。LightGBM のどの綴りでも）で検査し、外れれば `CONFIG_INVALID` で拒否する（storage を渡しても study は作られない）: (a) 固定の層は `model.params` に round の固定パラメーター `fixed` を重ねた値（fit が merge した値を検査するのと同じ規則）から、round の `category: model` の次元が設定するパラメーターを除いたもの（その次元のサンプル値が試行ごとに上書きするので、層の値はどの booster にも届かない。merge の空間で既定の `objective` の次元が上書きする `model.params` の目的関数は拒否しない）。(b) round の実効空間の `category: model` の次元で正規名が `objective` / `metric` のものは、カテゴリの次元でなければ（`float` / `int`）それだけで拒否し（理由 `not_categorical`、context に `kind`。値は名前なので数の範囲から抽出した値は受理されない）、カテゴリの次元なら全ての選択肢を検査する（context に `surface`・`dim`・`key` `choices`・`value` と規則自身の context。binary の `cross_entropy_lambda` は `not_a_probability`）。config の空間、`tune(space=...)`、provider の既定の空間、resume の round の空間のどれから来た次元にも当たる。それまでは不正な値が各試行の adapter で `CONFIG_INVALID` になり、tuner が試行の失敗として記録していた（全試行なら `TUNING_FAILED`、一部の選択肢なら黙って落ちる）。保存した study の試行の値は検査しない（範囲外）。

### 探索次元

| パラメーター | 型 | 範囲 | カテゴリ |
|---|---|---|---|
| `objective` | categorical | regression: `[huber, fair]`, binary: `[binary]`, multiclass: `[multiclass, multiclassova]` | model |
| `n_estimators` | int | `[600, 2500]` | model |
| `learning_rate` | float (log) | `[0.0001, 0.1]` | model |
| `max_depth` | int | `[3, 12]` | model |
| `feature_fraction` | float | `[0.5, 1.0]` | model |
| `bagging_fraction` | float | `[0.5, 1.0]` | model |
| `num_leaves_ratio` | float | `[0.5, 1.0]` | smart（`auto_num_leaves` が真になりえない宣言空間では外す、H-0121） |
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
| `current_trial` | `int` | この round の何番目の試行か（1-indexed）。新しい round は 1 から数える。途中で落ちた round を再開したときは続きから数える（その round で既に K 件終わっていれば K+1 から。storage の有無に依らない。H-0122） |
| `total_trials` | `int` | この round で要求した `n_trials`（途中の round の再開では記録した値） |
| `elapsed_seconds` | `float` | 経過時間（秒） |
| `best_score` | `float \| None` | これまでの最良スコア（complete trial なしの場合 `None`） |
| `latest_score` | `float \| None` | 直近 trial のスコア（fail/pruned の場合 `None`） |
| `latest_state` | `str` | `"complete"` / `"pruned"` / `"fail"` |
| `round` | `int` | 現在のラウンド番号（1-indexed）。H-0068 で追加 |
| `cumulative_trials` | `int` | study 全体の試行数（`trial.number + 1`）。H-0068 で追加 |
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

## 11.5 Re-tune: Study Resume + 境界検知拡張（H-0068、H-0122）

初回 tuning 後に追加探索を行い、さらなる精度向上を目指す。一回の `tune()` 呼び出しで足す試行のまとまりを round と呼ぶ。

`tune(resume=True)` の目的は二つで、どちらも目的関数を変えない（H-0122）:

- 探索範囲を変えて、さらに良い解を探す。範囲を変えるのは利用者か外部のアルゴリズム（`space=`、§11.5.2）と、自動の境界拡張（下の規則）。
- 障害のあとで再開する。永続 storage の study に `resume=True` で再接続し、落ちた時点から続ける（§11.5.1、§11.5.3）。

不変条件: 一つの study の試行は、すべて同じ目的関数の観測である。目的関数の同一性を照合できない study には試行を足さない（§11.5.3）。採用する最良値は study 全体の最良の試行で、その試行のパラメーターで `fit` すると、その試行と同じ学習になる（`config ⊕ fixed ⊕ その試行のサンプル値`、§11.3）。拒否は試行の追加と study の属性の書き込みより前に行い、study と Model の状態を変えない。

resume の round は前回の最良値を評価し直さない（前回の最良値を study に予約して round の最初に学習し直す H-0068 の規則をやめ、enqueue しない）。最良の試行は study に COMPLETE の観測として残り、TPE は既にそれを使っているため。resume の各 round は、要求した `n_trials` の数だけ TPE でサンプルした点を評価する。

### Model.tune() の拡張パラメーター

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `resume` | `False` | `True`: study に round を足す。Model が持つ study、持たなければ `storage` / `study_name` の study（再接続。無ければ作る）。途中で落ちた round はその round を終える（§11.5.3） |
| `n_trials` | `None` | 追加試行数（`None` → config 値）。途中の round の再開では記録した値だけを受け付ける |
| `expand_boundary` | `None` | 境界拡張。`None`: round の宣言空間に H-0102 の規則を当てる（`merge` で利用者の次元が無いとき→`True`、それ以外→`False`）。途中の round の再開では記録した値だけを受け付ける |
| `boundary_threshold` | `0.05` | 端判定閾値。開区間 (0.0, 0.5)。範囲の端からこの割合の内側に最良値があると広げる。外の値は study の前に `CONFIG_INVALID`（`core/_tuning_admission.py`） |
| `space` | `None` | この round の探索空間（`tuning.optuna.space` と同じ形式）。round 2 以降だけ。`None`: 直前の round の空間（§11.5.2） |

### 境界検知ルール

- **linear 空間**: `(best - low) / (high - low) < threshold` → 下限近傍
- **log 空間**: 対数空間で同一計算
- **categorical**: 拡張不可（ログ通知のみ）

### 非対称拡張ルール

- linear: 端方向に `(high - low)` を追加（range 2 倍）
- log: 端方向に対数空間で 3 倍に拡張
- `IntDim`: `max(1, new_low)` で下限ガード
- 拡張後の `new_low` / `new_high` は次元の `min_allowed` / `max_allowed`（§11.2）でクランプする。クランプが効いたら `BoundaryDimStatus.clamped_to_bound` を `True` にする（H-0078）
- 範囲を変えない拡張は拡張ではない: クランプ（`min_allowed` / `max_allowed`、linear の `0.0` 下限、`IntDim` の `max(1, ...)`）と `IntDim` の丸めのあとで `(new_low, new_high)` が元の `(low, high)` と等しい次元は、端に近くても `expanded=False`・`new_low` / `new_high` は `None` とし、`expanded_names` にも `RoundSummary.expanded_dims` にも入れない。`clamped_to_bound` は `min_allowed` / `max_allowed` のクランプが効いたときだけ `True` で、再判定のあとも `True` のまま残る（linear の `0.0` 下限と `IntDim` の `max(1, ...)` は `clamped_to_bound` を立てないので `False` のまま）。`IntDim` の拡張と比較は整数で計算する。端の検出（位置の計算）は H-0068 のとおり float で行う（§11.1 の大きさの規則で `IntDim` の値は `2**52` 以下なので、float でも正確）。毎ラウンド同じ空の拡張を報告し続けないため（H-0078 項目 4、H-0111）
- 反対側の端は据え置き
- 拡張は範囲を狭めも反転もしない（[#342](https://github.com/nbx-liz/LizyML/issues/342)）: クランプ（`min_allowed` / `max_allowed`、linear の `0.0` 下限、`IntDim` の `max(1, ...)`）は、拡張で外へ動いた端だけに、元の端まで当てる。端を元の範囲の内側へは動かさない。利用者が provider の境界の外に宣言した範囲（§11.5.2、たとえば `min_allowed` が 10 の `n_estimators` に `[3, 5]`）で、最良値が外側の端にあっても、範囲は変わらず、上の規則で `expanded=False` と報告する（境界が外への移動を止めたので `clamped_to_bound=True`）。`IntDim` の `max(1, ...)` も、元の下限が 1 未満の次元（`max_depth` の `-1`、`bagging_freq` の `0`）の下限を動かさない
- 刻みのある次元（H-0124）: 上の計算を正確な有理数で行ったあと、新しい端を外向きに格子へ寄せる（下端は `step` の倍数への切り下げ、上端は切り上げ）。`min_allowed` / `max_allowed` に当てた端が格子の外なら内向きに寄せる。どちらも元の端より内側にはしない（#342）。寄せた端が元の端と同じなら拡張しなかったものとして扱う。`log` の下端は `step` より下に広げない（今の下端が `step` なら下へは拡張しない）。例: `float` の `log` `[0.02, 0.1]` step 0.01 の下端は 0.02/3 → 切り下げて 0 → `step` で止めて 0.01。上限 1.0 の線形 `[0.15, 0.6]` step 0.15 の上端は 1.05 → 1.0 → 内向きに 0.9。拡張した次元は `step` を持ったまま round の記録に書き、記録の前に §11.1 の格子と大きさの検査を通す（外れれば書き込みの前に `CONFIG_INVALID`、理由 `bound_off_grid` か `range_not_finite` / `bound_too_large`）
- **大きさの上限を越える拡張はしない（H-0124 改訂 1）**: 刻みの有無によらず、動かす端ごとに、上の規則（正確な計算、格子への寄せ、下限、`min_allowed` / `max_allowed` のクランプと内向きの寄せ）で出した新しい端を次元の型の値 `x`（`float` の次元は最も近い `float`、`int` の次元はその整数）にし、次をすべて満たすときだけ `x` に広げる: `float` の次元では `x` が有限で、動かない側の端との幅（正確な計算）が `float` の最大値以下、`int` の次元では `x` の絶対値が `2**52` 以下。刻みがあれば `x` が §11.1 の読み方（`Fraction(Decimal(str(x)))`）で格子の点（`float` の間隔が刻みより粗い大きさでは、格子の点を `float` にすると格子から外れうる）。`log` の次元では `x > 0`（`[5e-324, 1e-323]` の下端は 3 で割ると 0.0 になる）。上下限に当ててから大きさを確かめる（上限 5e307 の線形 `[-1e308, 0]` の上端は 5e307 に広がり、上限 9e307 なら幅 1.9e308 が上限を越えるので広げない）。上限まで寄せて止めることはしない。満たさない端は広げず、その次元の `BoundaryDimStatus` は `expanded=False`、`new_low` / `new_high` は `None`、`clamped_to_bound=False`（上下限に当てた候補でも、広げなければ「上下限に達した」とはしない）、`expanded_names` に入らず、`expand_dims` は次元を元のまま返す（`edge` と `position_pct` は検出のとおり）。新しい端が元の端と同じになる場合は、この規則ではなく上の「範囲を変えない拡張」の規則で報告する（`min_allowed` / `max_allowed` が止めたなら `clamped_to_bound=True` のまま）。こうして広げた次元は §11.1 の格子と大きさの規則を満たし、次の round の端の検出は常に有限になる
- 拡張しない次元（H-0122）: この round で宣言が変わった次元と新しい次元（§11.5.2。利用者の宣言を優先する）、前回の最良値が今の範囲の外にある次元。どちらも `boundary_table` に `expanded=False` で出る。「前回の最良値」は study の全試行のうち最良の COMPLETE の試行の値で、Model が持つ tuning の結果ではない（再接続した Model には無く、`Model.load` のあとでは別の study の結果でありうる）。study に COMPLETE の試行が無ければ（直前の round が `failed`）、拡張しない

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
rounds: tuple[RoundSummary, ...]        # completed で閉じた round ごとに 1 要素（round の順）
boundary_report: BoundaryReport | None  # この round の境界の分析（下）
```

- `rounds`: study の round のうち `completed` で閉じたものの要約。`failed` で閉じた round（その round の試行に COMPLETE が一つも無いまま終わった round。H-0128）は入らない（`RoundSummary` は最良値を数値で持つため）。round の番号はそのまま続くので、`rounds` が round 2 から始まり、あるいは途中の番号が抜けることがある。`failed` の round の次の round の `best_score_before` は study の最良値で、study にまだ COMPLETE の試行が無いときだけ `None`。`RoundSummary.n_trials` はその round の試行の実際の数（`timeout` で要求数より少なくなりうる）。途中から再開した round の要約と `boundary_report` は、その round の記録（始めたときの判断）から作る。
- `boundary_report`: この round が境界を分析したとき（round 2 以降で `expand_boundary` が真に解決され、study に COMPLETE の試行があるとき）に設定し、それ以外は `None`。`restore_tuning` は最後の round のものを返す。

### TrialResult 拡張

```python
round: int  # どの round の試行か (1-indexed)。試行の番号と study の round の記録から決まる
```

round k の試行は、round k の記録の「開始時の試行数」以上で、round k+1 の記録の「開始時の試行数」未満の番号を持つもの（最後の round は上限なし）。`failed` の round の試行も、その round の番号で `trials` に残る。

### 刻みのある次元の試行の値（H-0124）

刻みのある次元の試行の値は、Optuna が記録した生の値ではなく、その試行が訓練に使った格子の点（§11.1 の丸め）。丸めた値は保存せず、一つの純粋な関数（`recorded_params`）が、試行の生の値と、その試行が属する round（試行の番号で決まる。上の `TrialResult.round`）の実効空間の次元から決める。刻みの無い次元の値は生の値のまま。

- この値を読む位置: 目的関数の中で訓練に渡す値、tune の最良値（`best_*_params`。`fit` の上書きと export される値もここから）、`restore_tuning`、次の round の計画（`outside_range` と `BoundaryDimStatus.best_value`）、`TrialResult.params` と `tuning_table`（`int` の `log` の次元も `int` で見える）。
- study の最良の試行は前の round の試行のことがある。その値は、その試行の round の刻みで決まる（今の round の刻みではない）。round 1 が step 0.2、round 2 が step 0.25 の study で、round 1 の生の値 0.41 の最良は 0.4 のまま。
- 生の値を読むのは、名前だけを見る位置（系譜の経路の検査）、Optuna 自身、公開の `Tuner.tune`（下の制約）。
- 系譜の検査は、各試行の値の名前がその試行の round の実効空間の次元にあることを確かめ、外れる study を `"inconsistent"` とする（§11.5.3）。こうして値を決める関数は、試行のどの名前にもその round の次元を見つける。

### tuning_table() 拡張

`round` 列と `state` 列を追加。`round` 列は各試行の `TrialResult.round`。

### boundary_table() 新設

`BoundaryReport` を DataFrame に変換。列: `dim`, `best`, `low`, `high`, `position`, `edge`, `expanded`, `new_low`, `new_high`。

### plot_tuning_history() 拡張

- ラウンド境界に縦の破線。境界は試行の round（`TrialResult.round`）の変わり目から求める（`RoundSummary.n_trials` の積み上げではない。`failed` の round は `rounds` に入らず、`timeout` で区切った round と途中から再開した round は要求数と実際の数が違うため。H-0122）
- ラウンドごとのアノテーション（拡張次元名）。要約の無い round（`failed`）は番号だけ
- best score 累積線はラウンドをまたいで連続

### Widget / Studio 連携

LizyML Core は callback + 結果型でデータを提供し、Widget/Studio が消費する設計。

| 消費者 | 情報源 | 用途 |
|---|---|---|
| Widget（リアルタイム） | `TuneProgressInfo.round`, `.cumulative_trials`, `.expanded_dims` | 進捗バー、拡張パネル |
| Studio（ダッシュボード） | `TuningResult.rounds`, `.boundary_report` | Round History、Search Space Evolution、収束判定 |

収束判定（`expanded_dims` 空 + 改善微小 → fit 推奨）は Widget/Studio 側の責務。Core は判断材料のみ提供する。

### 制約

- `resume=False` は新しい study を作り、round を 1 から数える（前の study の round を積み増さない）。storage に同じ名前の study が既にあれば `CONFIG_INVALID`（§11.5.1）。
- `resume=True` で、Model が study を持たず `storage` も無い → `LizyMLError(TUNING_FAILED)`（`Model.load` のあとも同じ。study の中身は artifact に保存しない。H-0134 が保存する試行の履歴は報告のための写しで、再開には使わない）。
- `Tuner` は study の作成・読み込み・最適化と、`lizyml:` 属性の書き込みだけを行う。`resume` の意味、目的関数の同一性、系譜の中身は `Model.tune` 側が決める（§5 の責務分離）。
- 公開の `Tuner.tune`（利用者の目的関数を受け取る。H-0124）は、`log` の次元に `step` があれば（`int` の `step` 1 も含めて）study を作る前に `CONFIG_INVALID` で拒否する。利用者が直接作った `FloatDim` / `IntDim` の `step` も、study を作る前に §11.1 の `step` と格子の検査（`parse_space` と同じ関数）を通す。直接作った数の次元はすべて（刻みの無い次元も）、study を作る前に §11.1 の大きさの規則を通す（H-0124 改訂 1。数の表記の文字列の `low` / `high` / `step` は刻みの有無によらず `bound_not_a_number` で拒否する — 数として読むのは config と `space=` の mapping だけ。無限大の境界は `bound_not_finite`。それ以外の数でない値と NaN は今までどおり Optuna に渡す）。`suggest_*` を呼ぶのは利用者の目的関数で LizyML は丸めを挟めず、`Tuner.tune` は系譜を書かないので過去の試行の round の次元も分からないため。結果は今までどおり試行が記録した生の値で、目的関数が評価した値そのもの。拒否は `Tuner.__init__` ではなく `Tuner.tune` に置く（`Model.tune` は同じ次元で `Tuner` を作り、`Tuner.run` だけを呼ぶ）。
- 直接の `Tuner`（`lizyml.tuning.Tuner`）は母数の値を検査しない（H-0125）。task も provider も受け取らない汎用の探索器で、試行の値の意味は呼び出し側の目的関数が決めるため、どの次元が LightGBM の目的関数・指標かを知らない。値の検査は呼び出し側の目的関数の責務で、その中の `CONFIG_INVALID` は試行の失敗として記録される。目的関数と指標の値を study の前に検査するのは `Model.tune` の計画の段（§11.3）だけである。task を受け取る検査を `Tuner` に足すことは公開 API の追加なので行っていない。
- artifact（`Model.export` / `Model.load`）の形式と `format_version` は H-0122 で変えない。

## 11.5.1 Persistent Storage（H-0072）

長時間 tuning ジョブが process kill / 再起動 / ネットワーク断で中断した際に、journal / RDB に永続化された trial 状態から **完了済 trial を再実行せずに resume** できる仕組みを提供する。Optuna 標準の `JournalStorage` / `RDBStorage` を薄く pass-through する設計とする。

### Tuner / Model.tune() の追加パラメーター

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `storage` | `None` | Optuna storage URL（`sqlite:///path/to.db` 等）または `BaseStorage` インスタンス。`None` で in-memory（従来挙動） |
| `study_name` | `None` | Study 識別子。`storage` と一緒にだけ使う（片方だけは fail fast） |

`storage=None` は in-memory（disk IO ゼロ）。`storage` を指定すると、`Model.tune` は storage にその名前の study があるかを先に調べ、無ければ新しく作り（既存の study に黙って再接続しない）、あれば読む。試行は完了するたびに backend に追記され、LizyML の記録（§11.5.3）も study の属性として同じ backend に書かれる。

### resume との関係（H-0122）

まず、`resume` に依らず次の二つを検査する（どちらも `CONFIG_INVALID`）: `storage` があって `study_name` が無い。`study_name` があって `storage` が無い。そのうえで:

| `resume` | Model の study | `storage` / `study_name` | storage にその study | 挙動 |
|---|---|---|---|---|
| `False` | 問わない | なし | — | 新しい study（メモリ）。round 1 から |
| `False` | 問わない | あり | 無い | 新しい study（storage）。round 1 から |
| `False` | 問わない | あり | ある | `CONFIG_INVALID`（続けるなら `resume=True`、新しく始めるなら別の `study_name`） |
| `True` | ある | なし、またはその study と同じもの | — | 継続 |
| `True` | ある | その study と同じと示せないもの | — | `CONFIG_INVALID` |
| `True` | 無い | あり | ある | 再接続。保存した系譜と実効空間を復元し、study の状態（§11.5.3）に従って続ける |
| `True` | 無い | あり | 無い | 新しい study（storage）を作り、round 1 から始める。作ったことを INFO のログ（`event='tune.study_created'`）に出す。`space=` を渡したら `CONFIG_INVALID` |
| `True` | 無い | なし | — | `TUNING_FAILED` |

- 「その study と同じ」: Model は、study を作った・読んだときの `storage` と `study_name` を覚えておく。`study_name` が等しく、かつ `storage` が等しい（URL の文字列なら文字列として等しい、`BaseStorage` なら同じオブジェクト）ときだけ同じとする。同じ backend を指す別の表現は同じと示せないので拒否する。
- 落ちたあとの再開は、同じスクリプトを `resume=True` で流し直せばよい: study が storage に無ければ作り、あれば続ける。
- 同じプロセスで round が例外で止まったあとの `resume=True` も、その study が Model に載っているか storage にあれば、同じ規則（§11.5.3 の状態）で扱う。storage を使わない最初の tune が失敗したときは、作りかけのメモリ上の study を捨てる（Model の状態を変えないため）。
- 一つの study に試行を足すプロセスは一つとする。利用者に求める前提で、ライブラリは強制しない。

### 制約

- `storage is not None and study_name is None` → `LizyMLError(CONFIG_INVALID)`。`study_name` だけを渡しても `CONFIG_INVALID`（黙って無視しない）
- round の系譜（`RoundSummary`、`BoundaryReport`、各 round の空間と予算）は study の属性に保存し、再接続で続ける（§11.5.3）。再接続のあとも round の番号は続く。
- H-0099: supplied and persisted studies must match the resolved objective direction. Reject mismatches with `CONFIG_INVALID` before optimization; use a new study name instead of relabeling an existing study.

### Objective direction (H-0099)

The first configured evaluation metric, or the first task default when omitted,
is the objective. Omitted/null `tuning.optuna.params.direction` derives from that
metric's `greater_is_better`, including parameterized metric entries. Explicit
directions must agree or fail with `CONFIG_INVALID` before creating a study.
Null remains automatic through Config dump/load; exported Result directions
are always the resolved `minimize` or `maximize`. Historical configs containing
contradictory explicit directions require migration; old explicitness is not guessed.

## 11.5.2 `space=` で探索空間を変える（H-0122）

範囲を変える経路は `tune(..., space=...)` の引数とする。config は「study を作ったときの宣言」に固定され、round ごとの変更は引数と study の系譜に残る（再接続で config から `Model` を作り直しても、変えた範囲が失われない）。

- `space=` は `tuning.optuna.space` と同じ形式の dict で、その round の宣言として study の `space_mode` で解決する。書式の外の指定（§11.1 の探索空間の書式。dict でない、名前が `str` でない、未知のカテゴリー、`low` / `high` の欠けなど）は `CONFIG_INVALID`。
- round 2 以降だけに渡せる。round 1 を始める呼び出し（`resume=False`、storage に無い study への `resume=True`、状態 `"uninitialized"` / `"no_round"` の study）に渡すと `CONFIG_INVALID` で、study は作らない。study の最初の空間は config の `tuning.optuna.space`。
- resume で config の `tuning.optuna.space` が study を作ったときの宣言と違えば `CONFIG_INVALID`（「範囲を変えるなら `space=`」。黙って無視しない）。`space_mode` は resume で変えられない（固定パラメーターが変わり、目的関数が変わるため）。

**宣言空間**（その round で宣言した空間を解決したもの。provider の境界は付けない）:

- `space=` があれば、それを解決したもの。無ければ直前の round の宣言空間。
- `merge`: provider の既定の次元に `space=` の次元を重ねる。`space=` に無い既定の次元は既定の宣言になる（config で利用者が上書きした既定の次元でも、`space=` に無ければ既定の宣言に戻る）。config で利用者だけが宣言した次元（provider の既定に無いもの）は、`space=` に無ければ探索から外れ、以後の試行は config の値で学習する。
- `replace`: `space=` の次元だけ。`space=` に無い次元は外れる。
- model の次元の alias は正規名と同じ次元として扱う（経路は一つで、拡張を引き継ぐ）。

**次元ごとの比較**（直前の round の宣言と比べる）:

- 宣言が同じ次元: 直前の round の実効の範囲（拡張を含む）を引き継ぐ。
- 宣言が変わった次元、新しい次元: 新しい宣言の範囲を使い、その次元のこれまでの拡張は捨てる。この round では拡張しない。
- 宣言から消えた次元: 探索から外す。以後の試行は config の値で学習する。

**同じ study で受け付けない変更**（study を変える前に `CONFIG_INVALID`。「新しい study で」）:

- 同じ名前の次元の、分布の種類（`float` / `int` / `categorical`）、`log` の有無、categorical の選択肢。選択肢は並びと、各値の型と値で比べる（NaN どうしは等しい。`True` と `1`、`1` と `1.0` は違う）。比べる相手は、これまでのすべての round の次元（外した次元を戻す場合も）。Optuna の拒否は試行の途中で FAIL の試行を書いてしまうので、LizyML が先に比べる。
- 次元のカテゴリー（`model` / `smart` / `training`）の変更。次元の経路（正規名とカテゴリーの対応）は足すだけで、外した次元の経路も残る（古い試行の値を回すため）。
- `int` の `log` の次元の「`step > 1` かどうか」（H-0124。`step` が無いか 1 なら Optuna の `IntDistribution(log=True)`、1 より大きければ `FloatDistribution(log=True)` で、分布の種類が変わるため）。context の `changed` は `step`。
- 範囲（`low` / `high`）の変更は受け付ける。それ以外の `step` の変更（`log` でない次元の刻み、`float` の `log` の次元の刻み、`step` を足すことと外すこと。どれも上の `int` の `log` の例外を除く）も受け付ける。格子の原点は 0 で、各試行の値はその試行の round の刻みで決まる（§11.5 の試行の値）。

**受け付けの検査**: 新しい tune と同じ検査（H-0093、H-0099、H-0102、H-0121）を、この round の宣言空間に当てる。`space=` の次元は利用者の宣言として扱う（未知の名前、同じパラメーターの二重の宣言、training の所有の衝突は `CONFIG_INVALID`）。provider の境界は拡張だけを縛り、`space=` の範囲は検査もクランプもしない（H-0111）。

**最良値の経路**: 最良の試行のパラメーターは、今の次元ではなく系譜の経路で model / smart / training に分ける。外した次元を持つ試行が最良なら、`fit` はその試行の値を使う（外した smart の次元は smart として渡り、LightGBM の params には入らない）。

## 11.5.3 study の記録と状態（H-0122）

### 目的関数の同一性と照合

study を作るとき、目的関数の同一性を study の属性に書く（メモリ上の study にも書く）。`resume=True` と `restore_tuning` のたびに、今の呼び出しから同一性を計算して照合し、違えば `CONFIG_INVALID`（context に違う要素の名前）。照合は保守的な互換性とし、目的関数を変えうる入力を広めに含める（無害な変更も拒否するが、黙って間違うことはない）。

- 含める: データ（tune が実際に使う DataFrame の列名と順序、dtype、categorical の categories とその順序と `ordered`、値と行の順序。index、型の違う同じ表記の値も区別する。`data.path` は文字列ではなく読んだ中身を要約する。除外する列も含める）、config の探索空間以外（task、data と特徴量の設定、`model` の全体（探索している次元の値も含む）、実効の固定パラメーター `fixed`、provider の既定パラメーター、training の全体、split の全体）、最初の評価指標とそこから導いた最適化の向き。
- 含めない: 探索空間（`space_mode` が変える `fixed` は含める）、`n_trials`、timeout、sampler、`progress_callback`、`expand_boundary`、`boundary_threshold`、二つ目以降の評価指標、calibration、export・出力の設定、`storage`、`study_name`、時刻。
- ライブラリの版（LizyML、LightGBM、Optuna、pandas、scikit-learn）は同一性と別に記録し、違えば警告だけを出して続ける。版の更新で同一性に入る値そのもの（provider が決める `fixed` や既定パラメーター）が変われば、目的関数の変更として `CONFIG_INVALID`。
- 向きの照合（H-0099）はそのまま残り、目的関数の照合と同じく試行を足す前に行う。

### study の属性

キーは `lizyml:` で始める。一緒に書くべきものは一つのキーの値にまとめ、書き込みの途中で落ちても読み手が判断できるようにする。

- `lizyml:study`: study を作ったときに一度だけ書く。形式の版（`schema`。読めない版なら `CONFIG_INVALID`）、同一性の要約と要素ごとの要約、ライブラリの版、`space_mode`、`fixed`、study を作ったときの config の探索空間。
- `lizyml:round:<k>`: round k の準備がすべて済んでから、試行を足す前に一度だけ書く。宣言の元の形（`space=`、無ければ「直前の宣言のまま」）と次元ごとの出どころ（利用者 / 既定）、宣言空間、実効空間、拡張と `BoundaryReport` の全体、解決した `expand_boundary`、予算（`n_trials`、`timeout`）、開始時の試行数と最良値、その round で増えた次元の経路。次元の記録は `step` を持つ（無ければ `null`。H-0124、形式の版は 1 のまま）。`step` の無い次元の記録は刻み無しと読み、知らないキーを持つ次元の記録（数の次元でもカテゴリの次元でも）は読めず、その study は `"inconsistent"`（下の表）。記録された次元が §11.1 の大きさの規則に外れるか、記録された `min_allowed` / `max_allowed` が有限の `float` でない（巨大な `int`、NaN、無限大）ときも、その記録は読めない（`lizyml:study` なら `"unreadable"`、round の記録なら `"inconsistent"`。H-0124 改訂 1）。resume と `restore_tuning` は計画（端の検出、試行の値の決定）より前にこれを拒否する。
- `lizyml:round:<k>:end`: round を閉じたときに書く。終わり方（`completed` / `failed`）、終了時の最良値（COMPLETE の試行が無ければ空）、時刻。**その round の試行**（番号が round の開始時の試行数以上のもの）に COMPLETE が一つも無いまま round が終われば `failed` と書いたうえで `TUNING_FAILED` を送出する（Model の状態は変えない。H-0128 で「study に」から「その round に」に改めた: 前の round に COMPLETE があっても、試行が全て失敗した round は `completed` にならない）。`TUNING_FAILED` の context は `round`・`n_trials`・`status` に加えて `trial_states`（その round の試行の `{"trials", "FAIL", "PRUNED", "interrupted"}` の数。`interrupted` は再接続で RUNNING から FAIL にした数で、`FAIL` にも数える）と、この呼び出しの中で目的関数が出した例外があれば最初のものの `first_error`（型の名前と文。それぞれ 500 文字まで）を持つ。理由が必ず分かることは約束しない（例外を出さずに FAIL になった試行、別のプロセスの試行）。閉じた round は再開の対象にならず、次の `resume=True` は新しい round を始める。
- 書く順序: round の準備を決める → `lizyml:round:<k>` → 試行を足す → `lizyml:round:<k>:end`。試行の round は試行の番号で決まり、試行ごとの属性は置かない。

### study の状態（閉じた表）

上から順に判定し、最初に当てはまった行を使う。メモリ上の study も storage から読んだ study も同じ表で扱う。`resume=False` で storage に同じ名前の study があるときは、状態に依らず `CONFIG_INVALID`。

| 状態 | 判定の材料 | `resume=True` での扱い |
|---|---|---|
| `"legacy"` | `lizyml:study` が無く、試行が 1 件以上（この変更より前に作った study、または LizyML の外で作った study） | `CONFIG_INVALID` |
| `"uninitialized"` | `lizyml:study` が無く、試行が 0 件（作りかけ） | `"no_round"` と同じ。ただしその検査をすべて済ませてから `lizyml:study` を書く |
| `"unreadable"` | `lizyml:study` が読めない（より新しい LizyML の版、または書き換えたもの） | `CONFIG_INVALID` |
| `"no_round"` | round の記録が一つも無く、試行が 0 件 | 照合の後、round 1 を新しい study と同じように始める（config の探索空間を使い、`space=` は `CONFIG_INVALID`） |
| `"inconsistent"` | round の記録と試行が食い違う（記録が無いのに試行がある、round の番号が 1 から連続しない、`:end` の無い round が最後以外にある、round 1 の開始より小さい番号の試行がある、試行の値の名前がその試行の round の実効空間に無い（H-0124）など） | `CONFIG_INVALID`（「新しい study で」） |
| `"open"` | 最後の round に `:end` が無い（途中で落ちた） | 照合の後、同じ round を残りの予算で終える（下） |
| `"completed"` | 最後の round の `:end` が `completed` | 照合の後、新しい round を始める（§11.5.2） |
| `"failed"` | 最後の round の `:end` が `failed`（その round に COMPLETE の試行が無かった） | 照合の後、新しい round を始める。宣言の元は `failed` の round の宣言（`space=` を渡せばその空間）。study に前の round の COMPLETE の試行があればそれを前回の最良値として境界の拡張の規則を当て、無ければ拡張しない（H-0128） |

公開の関数 `read_study_lineage`（§11.5.4）は、これに storage にその study が無い `"absent"` を加えた値を返す。

### 途中で落ちた round の再開（状態 `"open"`）

同じプロセスで続けた場合と同じ空間・系譜・round 番号になるように、同じ round を残りの予算で終える。

- round 番号は変えない。その round の記録にある実効空間をそのまま使い、境界の分析と拡張はやり直さない。
- 残りの試行数は、記録した `n_trials` から、その round で終わった試行（COMPLETE、FAIL、PRUNED）の数を引いたもの。記録に `timeout` があれば、残りの時間は `timeout` から、その round の COMPLETE と PRUNED の試行それぞれの開始から終了までの時間の合計を引いたもの（落ちていた間の時間は入らない）。残りが 0 以下なら試行を足さずに round を閉じる（`:end` の規則に従う）。時間で区切る round では、続けた場合と同じ試行の数になることは保証しない。
- 落ちたときに RUNNING のまま残った試行は、再接続のときに FAIL として記録し、予算に数える。
- 引数: `n_trials`、`expand_boundary`、`space=` は、`None` 以外を渡したときだけ、その round の記録の解決した値と比べる。同じなら通し、違えば `CONFIG_INVALID`（「この round は記録の値で始まっている」）。`space=` は宣言空間に解決して次元の集合として比べるので、省略の書き方や dict の並びの違いは同じとみなす（`step` は比べる。`step` だけが違う拒否のメッセージでも二つの空間は `step` で違って見える。H-0124）。config の `tuning.optuna.params.n_trials` は使わず、記録と違えば警告を出す。終わった round のあとの新しい round では、これらの引数は自由に変えられる。
- **成功の無い開いた round の差し替え（H-0128）**: 渡した `space=` が記録した宣言空間と違い、その round の試行に COMPLETE が一つも無い（試行が 0 件を含む）ときは、上の比べ合わせで拒否せず、その round を `failed` で閉じて、渡した `space=` で次の round を始める（新しい round として `n_trials` と `expand_boundary` も自由）。書く順序は、RUNNING の試行を FAIL にする → 開いた round の `:end` を `failed` で書く（`UserWarning` で知らせる。`TUNING_FAILED` は送出しない）→ 次の round の記録 → 試行 → 閉じる。どの段の間で落ちても、状態は `"open"`・`"failed"`・`"completed"` のどれかとして読める。COMPLETE のある round と違う `space=`、記録と違う `n_trials` / `expand_boundary` は、今のとおり `CONFIG_INVALID`。
- 検査の順序: 照合と引数の比べ合わせを、RUNNING の試行を FAIL にすることを含むすべての書き込みより先に行う。拒否したときは、試行の状態も study の属性も変わらない。

## 11.5.4 系譜を読む関数 `read_study_lineage`（H-0122）

利用側が、試行を足さずに study の状態と系譜を知るための、読み取りだけの関数。`lizyml.tuning` に置き、`lizyml.tuning.__all__` に入れる（トップレベルの `lizyml.__all__` には入れない）。

```python
def read_study_lineage(storage: str | BaseStorage, study_name: str) -> StudyLineage

@dataclass(frozen=True)
class StudyLineage:
    state: str                          # 下の値のどれか
    rounds: tuple[RoundRecord, ...] = ()
    open_round: int | None = None       # 状態 "open" のとき、開いた round の番号

@dataclass(frozen=True)
class RoundRecord:
    round: int
    status: str                         # "completed" / "failed" / "open"
    n_trials: int                       # 要求した試行数
    n_finished: int                     # 終わった試行（COMPLETE、FAIL、PRUNED）の数
    best_score_before: float | None
    best_score_after: float | None      # "failed" と "open" では None
```

- `storage` と `study_name` の形と事前の検査は `Model.tune` と同じ（片方だけは `CONFIG_INVALID`）。両方を省いても `CONFIG_INVALID`（読む study が無い）。
- `state` は `"absent"`（storage にその study が無い）、`"legacy"`、`"uninitialized"`、`"unreadable"`、`"no_round"`、`"inconsistent"`、`"open"`、`"completed"`、`"failed"` のどれか。判定は §11.5.3 の表と同じ順序で行う。
- `rounds` は記録のある round ごとの要約で、`failed` の round も含む（`TuningResult.rounds` と違い、系譜をそのまま見せる）。状態が `"absent"`・`"legacy"`・`"uninitialized"`・`"unreadable"`・`"inconsistent"` のときは空。
- 読み取りだけ: 属性を書かず、RUNNING の試行も FAIL にしない（それは `resume=True` の呼び出しだけが行う）。目的関数の照合もしない（データも config も受け取らないため）。Optuna の都合で、SQLite の storage の study の一覧を読むと、ファイルとそのスキーマが無ければ作られる（study は作られない）。

## 11.5.5 tune の結果を study から取り戻す `restore_tuning`（H-0122）

round を閉じた（`:end` を書いた）あと、呼び出し側が tune の結果を受け取る前に落ちると、study は状態 `"completed"` になり、`resume=True` は新しい round を始めてしまう。試行を足さずに結果だけを取り戻す公開のメソッドを置く。

```python
def restore_tuning(
    self,
    storage: str | BaseStorage,
    study_name: str,
    data: pd.DataFrame | None = None,
) -> TuningResult
```

- `storage` と `study_name` の形と事前の検査は `Model.tune` と同じ（両方を省いても `CONFIG_INVALID`）。`data` は `tune` と同じ優先順位で決める。optuna の有無はデータを読む前、storage に触れる前に確かめる。
- 状態ごとの扱い（この順で判定する。取り戻す結果が無いことを、照合の食い違いより先に報告する）: storage にその study が無い → `TUNING_FAILED`。`"legacy"`・`"unreadable"`・`"inconsistent"` → `resume=True` と同じく `CONFIG_INVALID`。`"open"` → `CONFIG_INVALID`（「round k が途中。`tune(resume=True)` で終える」。途中の結果は取り戻さない）。`"uninitialized"`・`"no_round"`・`"failed"` → `TUNING_FAILED`（取り戻す結果が無い）。`"completed"` → 向き、目的関数の同一性、config の探索空間を `resume=True` と同じく照合し（版の違いは警告）、結果を取り戻す。
- 取り戻す結果は、その round で終わった tune が返したものと同じ: 最良の試行の値を系譜の経路で分けた `best_model_params` / `best_smart_params` / `best_training_params`、`best_score`、`metric_name`、`direction`、試行（番号から決まる round 付き）、`rounds`（`failed` の round を除く）、最後の round の `boundary_report`（記録したもの。使った `boundary_threshold` で判定したまま）。
- 取り戻す `best_smart_params` は smart の探索次元の値の規則（§11.3、H-0121 改訂 2）で検査し、外れれば `CONFIG_INVALID`（context に `surface: "tuning.study"`・`name`・`value`・`reason`）で、Model には何も載せない。改訂 2 より前の develop で文字列の真偽の次元で tune した study だけが当たる。その study への `tune(resume=True)` は、study を書く前に、保存された各試行の smart の値（記録する値を routes で分けたもの）を同じ規則で検査して拒否する（round の受け付けの検査は新しい round の次元しか見ないので、`tune(space=...)` でその次元を消した resume も、最良でない試行に残った値も、ここで拒否される）。`restore_tuning` も、結果を組み立てる前に各試行を同じく検査する。
- Model に載せるもの（成功したときだけ、まとめて。H-0094 決定 14）: tune の結果、読んだ study、最後の round の実効空間、round 番号と `rounds`、`lizyml:study` の `fixed`、覚えておく `storage` と `study_name`、データ。fit 済みのモデルと fit の記録は変えない（H-0094 決定 13）。
- study には何も書かない（属性も試行も変えず、RUNNING の試行も FAIL にしない）。
- 取り戻したあとは、`fit()` が最良の試行と同じ学習になる（`fixed` を Model が持つため）。続けて `tune(resume=True)` を呼べば、その study で新しい round を始める。

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
- `BaseMetric.needs_train_target` は既定 `False` の具体プロパティで、学習側の目的変数が要る指標（`mase`）が `True` に上書きする。そういう指標は `bind_train(y_train)` で学習側の目的変数に束ねた新しい instance を作ってから呼ぶ。呼ぶ側は `needs_train_target` が真のときだけ束ねる。既定の `bind_train` は `UNSUPPORTED_METRIC` を送出し、束ねていない `mase` を呼ぶと `UNSUPPORTED_METRIC` になる（黙った既定値は無い）。束ねていない instance は、名前・向き・`needs_*` を読むだけの経路（config の検査、tune の向きと探索次元の検査、`params_table`、export の parameters）で使える（H-0132）。
- 回帰: `rmse / mae / r2 ...`
  - `mape`: `y_true` に 0 を含むと `UNSUPPORTED_METRIC`（H-0004）。値は `mean(|y - ŷ| / |y|) × 100`（百分率）。学習曲線（`eval_history`）の `mape` は LightGBM の定義 `|y-p| / max(1,|y|)` の比率で、この値と違う（H-0125）。
  - `huber`: 誤差 `e` について `|e| <= delta` で `0.5 e^2`、それより大きいと `delta(|e| - delta/2)`。`HuberLoss` の `delta` は既定 `1.0` でコンストラクタ引数（MetricEntry `{huber: {delta: ...}}`、§13.1.1）。文字列 `"huber"` は `delta=1.0` を意味する（H-0004）。学習曲線（`eval_history`）の `huber` は LightGBM の定義で、閾値は LightGBM の `alpha`（既定 0.9、huber / quantile の目的関数と共有）なので、この値と違う（H-0125）。
  - 学習曲線は LightGBM の名前と定義で計算する（§7.1、§14 の Metric Bridge）。`FitResult.metrics` と同じ名前で定義が違うのは `huber` と `mape` だけで、`plot_learning_curve` はその subplot の題に LightGBM の定義を添える（§13.3）。どちらも LizyML の定義を保つ（一般的な定義に近い。H-0125 の代替案）。
  - `smape`: `mean(2|y - ŷ| / (|y| + |ŷ|)) × 100`、範囲 `[0, 200]`。`y = ŷ = 0` の行は 0 として数える（H-0071）。
  - `wape`: `sum|y - ŷ| / sum|y| × 100`。`UNSUPPORTED_METRIC` になるのは `sum(|y_true|) == 0` のときだけ（H-0071）。
  - `smape` / `wape` はどちらも `greater_is_better=False`、`needs_proba=False`。
  - `mase`: `mean|y - ŷ| / scale`、`scale = mean|y_train - mean(y_train)|`（学習側の目的変数の、平均からの平均絶対偏差。H-0132）。`y = 0` でも定義される。`greater_is_better=False`、`needs_proba=False`、`needs_train_target=True`。
    - scale を取る行: OOF と IF は fold の outer-train の行、feval は booster が学習する行（inner valid があれば inner-train の行）。評価する行の目的変数は scale に入らない。
    - 式は同じで、学習曲線は booster が学習する行（inner-train）の scale、`FitResult.metrics` は fold の outer-train の scale を使うので、値が違いうる。
    - OOF 全体は、各行をその行を検証した fold の scale で割った値の平均（fold の検証の行数で重みをつけた fold ごとの値の平均）。覆われない行は入らない。
    - 1 行が 2 つ以上の fold の検証に入る分割と、scale が 0 か有限でない場合は `UNSUPPORTED_METRIC`。tune では各試行が失敗し、`TUNING_FAILED` になる。時系列の naive・季節・系列ごとの scale は扱わない。
- 分類（binary）: `logloss / auc / auc_pr / f1 / accuracy / brier / ece / precision_at_k ...`
  - `precision_at_k`: 予測確率の上位 `k`% の行の precision。`needs_proba=True`（確率で上位を選ぶ）、`greater_is_better=True`。既定は `k=10`（上位 10%、`1 <= k <= 100`）で、MetricEntry ごとに指定できる（§13.1.1、H-0014）。
- 分類（multiclass）: `logloss / auc(OvR) / auc_pr(OvR) / f1(macro) / accuracy / brier(OvR) ...`
- multiclass の `auc / auc_pr / brier` は One-vs-Rest 展開 + macro 平均で計算する。メトリクス名は binary と共通（`__call__` 内で `y_pred.ndim` により分岐）。`auc` は `roc_auc_score(..., multi_class='ovr', average='macro')`、`auc_pr` / `brier` はクラスごとの `average_precision_score` / `brier_score_loss` の macro 平均である（H-0018）。
- **task との整合**: 指標名は `metrics/registry.py` の `_TASK_METRICS` が task ごとに持つ集合で検査し、集合に無い名前は `UNSUPPORTED_METRIC`（`Model.evaluate(metrics=...)` と Config の `evaluation.metrics` の両方）。
  - `_TASK_METRICS["regression"]` = `rmse` / `mae` / `r2` / `rmsle` / `mape` / `huber` / `smape` / `wape`（H-0071）/ `mase`（H-0132）。
  - binary = `logloss` / `auc` / `auc_pr` / `f1` / `accuracy` / `brier` / `ece` / `precision_at_k`。multiclass = `logloss` / `f1` / `accuracy` / `auc` / `auc_pr` / `brier`。
  - したがって、`mape` / `huber` を回帰以外で（H-0004）、`precision_at_k` を回帰か multiclass で（H-0014）、multiclass 対応の `auc` / `auc_pr` / `brier` を回帰で（H-0018）指定すると `UNSUPPORTED_METRIC` になる。
- **確率の検査**: `needs_proba` が真の組み込み指標は、確率でない値（数値でない、有限でない、`[0, 1]` の外、3 クラス以上なのに 1 次元）を受け取ると `METRIC_REQUIRES_PROBA` を送出する（H-0106）。0/1 のハードラベルは正当な確率として通る。binary の目的関数 `cross_entropy_lambda` は出力が確率でない（LightGBM は λ を出し、確率は `1 - exp(-λ)`）ので、目的関数としては学習の前に `CONFIG_INVALID`（`reason: "not_a_probability"`）で拒否し、それで学習した booster を持つ artifact は `Model.load` が `DESERIALIZATION_FAILED` で拒否する（§14.3、§4.2。H-0125、#307）。指標としての `cross_entropy_lambda` は今どおり受け付ける。
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
- `plot_learning_curve(*, metrics=None)`: fold ごとの train/valid loss 推移（折れ線グラフ）。`metrics: list[str] | None` で表示 metric をフィルタ可能（H-0062）。`None` で全 metric、指定時は `/` 以降の metric 名で一致するもののみ表示。一致なしで `LizyMLError`。曲線は `FitResult.history` の LightGBM の評価なので、`FitResult.metrics` と同じ名前で定義が違う指標の subplot の題に LightGBM の定義を添える（H-0125、#354）: `huber` → `valid_0/huber (LightGBM: threshold = alpha)`、`mape` → `valid_0/mape (LightGBM: |y-p| / max(1,|y|))`。添える文の表は `lizyml/plots/learning_curve.py` の `LIGHTGBM_CURVE_DEFINITIONS` の一か所に置き、表のキーが実際に値の食い違う組と等しいことをテストが本物の LightGBM で照合する。`alpha` の値は FitResult に無いので書かない（plot は FitResult から推測しない）。凡例（fold）、y 軸、`metrics=` の絞り込み（名前で照合する）は変えない。
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

> Layer 1（Leaf）に属する。Foundation と、§2.1 の表で許した `features/`・`metrics/` にだけ依存し、`config/` には依存しない（H-0130）。

```python
fit(X_train, y_train, X_valid=None, y_valid=None, **kwargs)
predict(X)
predict_proba(X)  # 分類（sigmoid/softmax 適用後の確率値）
predict_raw(X)    # 分類（sigmoid/softmax 適用前の生スコア / logits。Calibration 用）
importance(kind="split|gain|shap")
get_native_model()  # export用途
set_categorical_features(cols: list[str] | None) -> None  # デフォルト no-op (H-0054)
unsaveable_categories(pipeline_state) -> list[UnsaveableCategory]  # デフォルト [] (H-0129)
```

- `predict_proba()` は学習器の確率をそのまま返す。`objective: multiclassova` ではクラスごとの独立 sigmoid の出力なので、行和は 1 とは限らない。指標のための正規化は evaluator が `needs_simplex` の指標についてだけ行う（§13.1、H-0049）。
- `set_categorical_features()` は `fit()` 呼び出し前に CVTrainer が呼ぶ。categorical feature の扱いはアダプタの責務であり、`cv_trainer.py` に estimator 固有の kwarg を漏洩させない。
- `unsaveable_categories(pipeline_state)` は、その estimator が学習済みモデルを保存するときに書けないカテゴリ値を、列ごとに最初の一つ、列の順に返す（`UnsaveableCategory(column, shown, type_name)`。`shown` と `type_name` は例外を出さない表示で作った文字列で、値そのものは持たない）。既定は `[]`（書けない値は無い）。`LGBMAdapter` は、LightGBM がカテゴリ値を JSON で書けないもの（`estimators/lgbm/category_json.py`、H-0127）を、位置の番号に置き換えない列（状態の `stand_in_columns` の外）についてだけ返す。置き換えた列は位置の番号なので必ず書ける（H-0133）。`Model.export` と公開の `lizyml.persistence.export` は、何かを書く前に、保存する pipeline の状態ごとに、その状態と一緒に保存される adapter に尋ねる（refit の状態は `refit_result.model`、fold k の状態は `fit_result.models[k-1]`、fold ごとの状態が無ければ最後の fold の状態は `fit_result.models[-1]`）。fold ごとの状態の数と `fit_result.models` の数が合わない結果、fold ごとの状態も model も無い結果は、尋ねる相手が無いので `SERIALIZATION_FAILED`（context に `fold_states` と `models`）。`persistence/` は具体的な estimator を import しない（§2.1、H-0129）。

## 14.2 LGBM adapter の責務

- `objective / metric` 整合
  - 利用者または trial が書いた `objective`（エイリアスの綴りを含む）は、task の `TASK_COMPATIBLE_OBJECTIVES`（`estimators/lgbm/defaults.py`。LightGBM の canonical 名で regression 9 / binary 3 / multiclass 2）に含まれればそのまま `lgb.train` に渡し、含まれなければ `CONFIG_INVALID` とする（H-0079）。H-0079 より前は黙ってタスク既定に置き換えていた。`None` 以外の文字列でない値（dict / list など）も、包含を調べる前に `CONFIG_INVALID` とする（`check_objective_compatible`、H-0116。それ以前は dict / list が未加工の `TypeError` で落ちていた）。包含の検査とメッセージは値を書式化もハッシュもしないが、値の型やその属性の参照が例外を出すもの（`__class__`、メタクラスなど）と、送出後のエラーの表示（`context` は書いた値を保つ）は範囲外（H-0116 決定 2）。明示の `None` は「上書きなし」で、task の既定の objective で学習する。
- categorical の扱い統一
  - カテゴリの列の宣言は、学習に使う frame の列の順での位置の一覧（`categorical_positions`）にして、学習と検証の `lgb.Dataset` に渡す。名前を渡すと、LightGBM は `int` の名前を位置として読み、別の列をカテゴリにするからである（H-0142、#393）。名前と列は、型が完全に同じで値が等しいときだけ一致とし（`1`・`True`・`1.0` を取り違えない）、一致がちょうど 1 つでない宣言は `DATA_SCHEMA_INVALID`（context `{"column", "reason": "categorical_feature_not_unique"}`）。文字列の名前では、位置で渡しても学習した booster は名前で渡したときとバイトとして同じである。
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
- 利用者が指定できる目的関数は task ごとの表（`TASK_COMPATIBLE_OBJECTIVES`、`LGBMProvider.objective_choices(task)`）にあるもの。binary は `binary` と `cross_entropy` で、`cross_entropy_lambda` は出力が確率でないので受理しない（理由 `not_a_probability`、§13.1 の確率の検査。H-0125）。別名の値 `xentlambda` は表に無いので、今どおり理由の無い不一致として拒否する。
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
   - **native 指標のパラメータは拒否する**（H-0125、[#313](https://github.com/nbx-liz/LizyML/issues/313)）: LightGBM の native 指標は LightGBM 自身の母数で計算されるので、MetricEntry `{name: {param: value}}`（§13.1.1）のパラメータは使われない。名前を検査した後（名前の誤りは今の whitelist のメッセージを先に出す）、native の名前に空でないパラメータがあれば `CONFIG_INVALID`（context に `metric` と `params`（キー名の一覧））。`{"huber": {"delta": 9.0}}` も `{"rmse": {"bogus": 1}}` も拒否し、`huber` の閾値は LightGBM の `alpha`（目的関数と共有）であることをメッセージで案内する。パラメータ付きの LizyML の指標は `evaluation.metrics` に書く。空の `{}` は名前だけと同じ。feval の指標のパラメータ（`{"precision_at_k": {"k": 5}}` など）は今どおり。`resolve_metrics` は fit、adapter（直接の利用、tune の各試行、`export_code`）、tune の計画の段から呼ばれるので、どの入口でも同じ規則になる。

3. **feval カスタム関数**: LightGBM ネイティブ未対応の metric は `lgb.train(feval=...)` 経由でカスタム評価関数として注入する。

| feval Metric | Regression | Binary | Multiclass | metric に渡す値 |
|-------------|:---:|:---:|:---:|------------|
| `rmsle` | ✅ | | | 予測値そのまま |
| `r2` | ✅ | | | 予測値そのまま |
| `smape` | ✅ | | | 予測値そのまま（H-0071） |
| `wape` | ✅ | | | 予測値そのまま（H-0071） |
| `mase` | ✅ | | | 予測値そのまま。scale は booster が学習する行の目的変数から（H-0132） |
| `f1` | | ✅ | ✅ | 閾値 0.5 / argmax のラベル |
| `brier` | | ✅ | ✅ | 確率そのまま |
| `ece` | | ✅ | | 確率そのまま |
| `precision_at_k` | | ✅ | | 確率そのまま |
| `accuracy` | | ✅ | ✅ | 閾値 0.5 / argmax のラベル |

- `smape` / `wape` は回帰の feval 指標なので、`params.metric` に書けば early stopping と学習曲線を駆動する（H-0071）。
- `mase` も回帰の feval 指標である。`LGBMAdapter.fit` が受け取った学習の目的変数（inner valid があれば inner-train の行）を `resolve_metrics(..., y_train=...)` に渡し、束ねてから feval を作る。eval set のラベル（`dataset.get_label()`、評価する行）から scale を作らない。学習の目的変数を渡さない経路（config の検査、`params_table`、export の parameters）は束ねずに作り、呼ばない（H-0132）。
- **feval は LightGBM が渡す予測を再変換しない**（H-0105）。LightGBM 4 は組み込み objective の変換後の値を feval に渡す: binary は 1 次元の確率、multiclass / multiclassova は 2 次元 `(n, num_class)` の確率、regression は予測値。sigmoid / softmax をもう一度掛けると binary のラベル指標が定数になり確率指標が歪むので、掛けない。
- feval は evaluator と同じ規則（`evaluation.evaluator._pred_for_metric`、§13.1）で metric に値を渡す: ラベル指標には binary で 0.5 閾値、multiclass で argmax、multiclass の `needs_simplex` 指標には行和で正規化した確率、それ以外は確率をそのまま。feval の指標は、学習曲線と `FitResult.metrics` で同じ名前が同じ意味になる（H-0105）。ただし `mase` は式が同じで scale を取る行が違う（§13.1、H-0132）。
- 学習曲線（`FitResult.history` の `eval_history`）の native 指標は LightGBM の名前と定義で計算され、`FitResult.metrics` の同じ名前の指標と値が違うことがある。今そうなるのは `huber`（LightGBM の閾値は `alpha`）と `mape`（LightGBM は `|y-p| / max(1,|y|)` の比率）で、`plot_learning_curve` はその subplot の題に LightGBM の定義を添える（§13.3、H-0125、[#354](https://github.com/nbx-liz/LizyML/issues/354)）。
  - 記録される名前は LightGBM の正規名（`mae` と書いても `l1`、`logloss` は `binary_logloss` / `multi_logloss` として記録される）。各 task の受理する native 指標の全てを記録し、`MetricRegistry` の同じ task の名前との共通部分を同じ予測で比べると、値が違うのは上の二つだけで、他の共通の名前は一致する（`tests/test_plots/test_learning_curve_definitions.py`）。
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
    def smart_param_owners(self) -> dict[str, str]: ...           # H-0121
    def smart_value_rules(                                        # H-0121 am. 2
        self,
    ) -> Mapping[str, SmartValueRule]: ... # smart parameter -> its value rule
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


@runtime_checkable
class TrialErrorProvider(Protocol):  # 任意の機能（H-0128）
    def trial_errors(self) -> tuple[type[Exception], ...]: ...


@runtime_checkable
class ResourceWarningProvider(Protocol):  # 任意の機能（H-0140）
    def resource_warning(
        self, *, estimator_factory, X, smart_params, n_boosters
    ) -> str | None: ...
```

`ResourceWarningProvider` も `EstimatorProvider` とは別の任意の機能である。`Model.fit` と `Model.tune` は `isinstance(provider, ResourceWarningProvider)` のときだけ `resource_warning(...)` を呼び、文が返れば学習の前に `UserWarning` を出す（§5.3 の `auto_num_leaves` のメモリの警告）。持たない provider では何もしない。

`TrialErrorProvider` は `EstimatorProvider` とは別の任意の機能で、`EstimatorProvider` の成員には入れない（入れると、それを持たない既存の provider が静的に `EstimatorProvider` を満たさなくなる）。`Model.tune`（Facade）は `isinstance(provider, TrialErrorProvider)` のときだけ `trial_errors()` を呼び、`Tuner.run` / `Tuner.tune` の `extra_catch` に渡す。その型の例外は試行の FAIL になり、round は続く。`tuning/` は estimator のライブラリを知らない。LightGBM の provider は `(lightgbm.basic.LightGBMError,)` を返す。

制約:
- `EstimatorProvider` は `config/` を import しない（§2.1 の R2。import の文で判定し、呼ぶ側は問わない。H-0130）。
- `model_cfg` 引数は `Any` 型で受け取り、各 provider は設定を属性として読む（具象型の `isinstance` もキャストもしない）。
- `runtime_deps()` はアルゴリズム固有の依存パッケージ名とバージョンを返す（例: `{"lightgbm": "4.5.0"}`）。`RunMeta.deps_versions` に使用。
- `parameter_bounds(task)` は境界拡張（§11.5）をクランプするための、パラメーターとして意味のある範囲 `{name: {"min": ..., "max": ...}}` を返す（H-0078）。片側だけ（`min` だけ、または `max` だけ）でもよく、無い側は無制限で、両方あるときは `min < max`（H-0140。`LGBMProvider` の `max_depth` は `{"min": -1}` だけで、木の深さの上限は置かない）。表に無いパラメーターは無制限、空の dict は宣言なし。`LGBMProvider` は task によらない 15 パラメーターの表（`learning_rate` / `feature_fraction` / `bagging_fraction` / `num_leaves_ratio` / `min_data_in_leaf_ratio` / `min_data_in_bin_ratio` / `validation_ratio` / `lambda_l1` / `lambda_l2` / `n_estimators` / `max_depth` / `max_bin` / `bagging_freq` / `early_stopping_rounds` / `seed`）を返す。`Model.tune` は既定の空間にも利用者の空間にもこれを付ける（§11.2）。smart パラメーターの上下限は `smart_value_rules()` の範囲の内側に置くこと（拡張した次元も値の規則で検査されるため）。`num_leaves_ratio` の上限は 1.0（H-0121 改訂 2。それまでは 2.0 で、config の範囲 `(0, 1]` を越えていた）。
- `objective_choices(task)` は task で有効な objective の canonical 名を、決まった順の tuple で返す（エイリアスは含めない。H-0079）。`default_space` の `objective` 次元と UI の選択肢の元であり、集合は adapter が受理する `TASK_COMPATIBLE_OBJECTIVES`（§14.2）と同じ。未知の task は空の tuple。
- `metric_choices(task)` は `{"native": tuple, "feval": tuple}` を返す（H-0079）。`native` は学習器が評価する指標、`feval` は LizyML が feval として注入する指標（§14.3）。どちらも canonical 名で、順は決定的、2 つの tuple の間に重複は無い。
- `params_summary()` は `params_table()` 用のパラメータ行を返す。smart params + native model params（`metric` を含む、H-0061）の両方を含む。
- `build_pipeline_factory` は estimator 固有の FeaturePipeline が必要な場合（例: EntityEmbedding のカテゴリ埋め込み）に対応する。デフォルトは `NativeFeaturePipeline` を返す。
- `build_export_params` は codegen 経路（`Model.export_code()`）が必要とする native params / num_boost_round / early_stopping_rounds / feval metadata を `ExportParams` frozen dataclass で返す（H-0073）。`_model_persistence.py` から estimator 具象型（`LGBMAdapter` 等）への直接参照を排除するための入口。**「その fit が何を使ったか」を答える値は、config と現在の tuning result から再計算してはならない**（H-0094 決定 13）: `tune()` は tuning result を置き換えるが fit 済み adapter は置き換えないので、再計算する読み手は `fit → tune` の後に**存在しないモデルについて報告する**。実測: adapter が patience 7 で学習し、`params_table` / `export_code` はどちらも 2 と答えた。学習済み adapter は joblib で保存されるため、この経路は `load()` 後も正しい唯一の経路である（artifact が持つ tuning result は、どの fit も消費していないことがありうる）。`ExportParams.early_stopping_rounds` に **default を置かないこと** — 「provider が設定しなかった」と「early stopping が無効だった」が同じ値になるのは DC1 の形である。adapter に記録が無い値（`validation_ratio`）は `FitState.applied_training_params` から読む。この overlay は `metadata.json` の `applied_training_params` として記録され、`load()` が検査して復元する（H-0109）。記録の無い（H-0109 以前の）artifact では不明（`None`）となり config に落ち、再 export しても記録を書かない。
- `accepted_model_param_names()` / `smart_param_names()` は「その学習器が受理する名前」を宣言する（H-0093）。前者は**学習器自身から導出すること**（列挙しない）。学習器の更新で名前が増減したときに黙って古びる実装は、この IF が検出しようとしている欠陥をそれ自体が持つことになる。後者は `extract_smart_params` が返すキーと必ず一致させ、両者を単一の宣言から導くこと。
- `smart_param_owners()` は「ある smart パラメーターが読まれるために真でなければならない smart パラメーター（所有者）」を `{smart パラメーター: 所有者}` で返す（H-0121）。所有者の無い smart パラメーターは含めない。LightGBM は `{"num_leaves_ratio": "auto_num_leaves"}` を返す（`resolve_smart_params` は `auto_num_leaves` が真のときだけ `num_leaves_ratio` を読む）。tuning はこの宣言で、所有者が真になりえない smart の探索次元を拒否するか落とす（§11.3）。宣言は解決関数を実行するテストで固定する。smart パラメーターを持たない provider は `{}` を返す。H-0093 の分類では Protocol の変更であり、外部で Protocol を実装しているコードはこのメソッドを足す必要がある。
- `smart_value_rules()` は、smart パラメーターごとに、それが取りうる値の規則（`lizyml/estimators/provider.py` の `SmartValueRule`: 種類 `bool` / `number` / `untunable`、`None` を受け付けるか、数の区間（下端は開、上端は `high_inclusive`）、`True` を拒否するタスク）を返す（H-0121 改訂 2、§11.3）。`smart_param_names()` のすべての名前に規則があること（テストで固定）。型は厳密に判定する（`type(v) is bool`、`type(v) is int` / `type(v) is float`）。config も検査する範囲は、config の検査と同じ宣言から読むこと（LightGBM は `lizyml/core/types/smart_ratio.py` の `SMART_RATIO_RANGES` を呼び出しのたびに読む）。tune の round の受け付けの検査、`restore_tuning`、`Model.load` がこの規則を使う。Protocol の変更であり、外部で Protocol を実装しているコードはこのメソッドを足す必要がある。
- `canonical_param_names(names)` は「その名前がどのパラメーターを指すか」を返す（H-0094）。**パラメーター層のマージは綴りではなく同一性で行うこと**: 学習器がエイリアスを解決する以上、`{**base, **override}` は 1 つのパラメーターを 2 つの綴りで残し、どちらが効くかは学習器の規則次第になる。実測: config の `learning_rate` が `fit(params={"eta": ...})` に勝っていた。**学習器に 2 つの綴りを渡さない**こと。学習器が知らない名前は自分自身に写すこと（不明名の拒否は `accepted_model_param_names` の仕事であり、この写像が二重の門になってはならない）。**マージの継ぎ目は 4 か所あり、4 か所すべてを同一性で行うこと**: config / provider の既定 fixed / `best_model_params` / `fit(params=)` に加えて、**tuning の trial マージ**（`_model_tuning.py` の objective）がある。round 11 まで trial マージだけが綴りベースで、config の `learning_rate` と `eta` という探索次元がある場合、**trial は config の値で学習し、study には trial の値が best として記録され、その後の fit は記録された値で学習していた** — tuning が一度も評価していないモデルを選んでいた（DC1）。
- **学習器 adapter が特別扱いするパラメーター**（検証する / 改名する / 呼び出し引数に変換する）は、**全綴りをまとめて取り出す**こと（H-0094 決定 6）。1 綴りだけを見ると、エイリアスで書かれた値はその特別扱いを迂回する。実測: `application`（`objective` のエイリアス）に task 非互換な値を書くと互換性検査を通らずに学習されていた。同じ層で 1 パラメーターが複数綴りで指定されたら、**値によらず** `CONFIG_INVALID` とすること（H-0096 で改訂。**それ以前は「異なる値のときだけ拒否し、同値は通す」だった**）。同値を通す規則は「2 つの綴りが同じ値か」という問いを生み、その問いは任意の Python 値の上で全域でなければならないので**入力領域が開く** —— H-0094 rounds 18-26 の 9 連続はその領域の中で起きた。**LightGBM 自身も値を比較せず、等しくても重複そのものを警告する**（実測）。拒否の根拠は「どちらが効くか不可視だから」ではなく（優先順位は決定的である）、**利用者が 1 つの設定を 2 度書いており、どちらを意図したか LizyML には決められないから**である。**この規則は宣言した層すべてに配線すること**: レビュー round 11 まで `fit(params=)` にしか配線されておらず、`model.params` に `learning_rate` と `eta` を両方書いた config は両方が `lgb.train` に届き、LightGBM が黙って canonical 側を採った（DC4 — 宣言はあるが呼び出し側が無い）。検査は facade（`_merge_params`）に置く。`config/` は層規約上 `estimators/` を import できず別名表に届かないためである。**配線先は 4 層**: `model.params` / `fit(params=)` / `calibration.params` / `tuning.optuna.space`。3 つ目は rounds 10-11 monitor が「どの層に配線したのか」を問うて見つかった（名前検査だけがあり同一性検査が無く、両綴りが calibrator の `lgbm.train` に届いていた）。4 つ目は rounds 11-12 monitor が名指しし、実行して確かめた: `sample_params` は次元ごとに `params[dim.name] = ...` を書くので、**互いにエイリアスである 2 次元は同じ trial dict に両綴りを入れる** — LightGBM が canonical を採るため、もう一方は sample され最適化されながらどの trial にも影響しない（`learning_rate` と `eta` を 2 次元にした study で実測）。**空間には同値による免除が無い**: 2 次元は独立に sample するので、1 パラメーターを 2 回名指しすることは境界に関わらず曖昧である（`check_duplicate_space_dimensions`、study 開始前）。スマート層は対象外で、その理由は**スマートパラメーター名に学習器のエイリアスが 1 つも無い**ことである（実測 0 件、テストで固定）。**`calibration.params` は同一性検査だけでは足りず、canonical 化も要る**（H-0094 決定 8）: calibrator は自分の既定値と**綴りで**マージするため、呼び出し元が**1 度しか書いていない**エイリアスが既定値と並んで `lgbm.train` に届き、LightGBM が既定値を採っていた（`{"eta": 0.5}` → `learning_rate: 0.03` で学習）。`lizyml/calibration/` は `lizyml/estimators/` を import できないので、書き換えは facade（`canonicalise_calibration_params`）で行う。**calibrator 自身が pop するキーは除外すること**（`num_boost_round` は `num_iterations` のエイリアスであり、canonical 化すると pop 先が消える）。**calibrator が強制する値は canonical 綴りで書くこと**: `verbose = -1` はエイリアス側の強制だったので `verbosity` を書いた呼び出しに負けていた。
- **「あるパラメーター dict が別のパラメーター dict に出会う」場所を列挙し、1 行ずつ実行すること**（H-0094 決定 8）。この類の欠陥は round ごとに 1 つずつ出続けたので、当たりを付けて探すより列挙するほうが安い。走査は `instruments/parameter_merge_seams.py` として**出荷し**、表を再生成できるようにすること（散文に写した数は古びる）。候補 58 式。**走査の構文集合には `d[k] = v` を必ず含めること**: 最初の版はそれを宣言しておらず、その版が報告した 3 件の欠陥のうち 2 件がその構文に住んでいた（rounds 11-12 monitor）。**そして走査を「閉じた母集団」と呼ばないこと**: この主張は 2 回なされ、2 回とも 1 ラウンド以内に反証された（2 回目は round 13 のレビュアーが `config/loader.py:167` を名指しした — カーソル変数名が `node` で hint 語に当たらなかった）。hint 語による絞り込みは識別子テキストのヒューリスティックであって型解析ではない。**表が主張するのは実行した分だけである。** 開いた空間の走査を閉包と呼ぶことは、本 run が他人の宣言に見つけ続けている DC5 そのものである。
- **「利用者が綴った dict を 1 つの綴りで読む」箇所も同じ類であり、継ぎ目走査は扱わない**（H-0094 決定 9）。実測: `_extract_feval_metadata` が `adapter.params.get("metric")` とリテラルで読んでいたため、`fit(params={"metrics": ...})` は正しく学習しながら `export_code` が評価関数を落とし、**生成コードが動かなかった**。学習側が `_pop_by_identity` で読む以上、読み手も同一性で読むこと。母集団は 4 件（`estimators/` / `persistence/` / `codegen/` / `core/` / `training/` を走査）。
- **LizyML 自身が握っているネイティブパラメーターは、`model.params` / `fit(params=)` から重ねて指定させないこと**（H-0094 決定 9、`check_training_managed_overrides`）。`training.early_stopping.rounds` は `early_stopping_round` の、`training.seed` は `seed` の LizyML 側の綴りである。実測では**両方向に**壊れていた: 前者は上書きが `lgb.train` に届いても config 由来の callback が停止を決め（早期停止を切ると今度は LightGBM 自身が honour して検証セット不在で落ちる）、後者は上書きが `training.seed` に黙って勝っていた。**2 方向が食い違うからこそ、どちらかを選ぶのではなく拒否する。** 判定は canonical 名を全綴りに展開して行い、展開は provider の `canonical_param_names` を受理名の上で反転して得る（この層は学習器を知らない）。
- `smart_managed_param_names(smart, task)` は「**有効なスマートパラメーターが上書きしてしまうネイティブ名**」を返す（H-0094）。スマート解決はパラメーター dict のマージより後段で走り、その結果が勝つため、ここに挙がる名前を手で指定しても黙って置き換えられる。呼び出し側はそれを**拒否**に使う（`CONFIG_INVALID`）。task を取るのは、同じスマートパラメーターでも task によって書くものが変わるためである（`balanced` は binary では `scale_pos_weight` を書くが、multiclass では sample weight を作る＝パラメーター名ではないので衝突しない）。宣言は**コードから閉じる**こと: 解決関数群を**実際に実行**し、宣言されていない名前が返ってきたら落ちるテストを持つ。以前はソースの `resolved[...] = ...` 代入を走査していたが、それは代入の**綴り方**についての仮説であり、`resolved.update({...})` で書かれた 4 つ目の名前は見えなかった（H-0094 レビュー round 9 の monitor）。実行には推測すべき綴りが無い。ただしこれは**有限個の実行**であって閉じた入力領域ではない — パラメーター**名**を列挙しても、解決関数が受け付ける**組み合わせ**は列挙できない（round 10）。各 activation が解決結果を変えることを個別に主張することで、activation が無効化されて何も観測しなくなる事態は防ぐ。**約束（H-0121 改訂 1）**: 結果は各スマートパラメーターの値に、その真偽と `None` かどうかを通してだけ依る（値の大きさには依らない。複数のスマートパラメーターの組で決まってよい）。tune の受け付けの検査は、smart の探索次元の値を状態ごとの代表一つに絞ってこの関数に渡す（§11.3）。LightGBM の provider がこの約束を満たすことは、3 つのタスクで、各スマートパラメーターに同じ状態の別の値を入れて結果が変わらないことを実行して確かめる（`tests/test_estimators/test_smart_managed_contract.py`）。
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
│   ├── memory_check.py  num_leaves のメモリの見積もりと警告 (H-0140)
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
  - `checksums` はちょうど `{"algorithm": "sha256", "files": {<ファイル名>: <64 桁の小文字の 16 進の digest>}}` で、`files` は `fit_result.pkl` / `refit_model.pkl` と、あれば `analysis_context.pkl` と `tuning_history.json`（H-0134）の SHA-256 を持つ。アルゴリズム名は `persistence/exporter.py` の `CHECKSUM_ALGORITHM`（H-0083）。この形の検査は `persistence/_manifest.py` の `declared_files` が行う（H-0135）。
  - tune 済みのモデルだけ: `tuning` ブロック（`best_model_params` / `best_smart_params` / `best_training_params` / `best_score` / `metric_name` / `direction`、成功した tuning round の `fixed_params`）。`load()` はこれを tuning result として復元し、load 後の再 `fit()` が tuned params を再現する（H-0086）。`tuning.history` は `null` か、ちょうど `{"file": "tuning_history.json", "schema": "lizyml-tuning-history/v1"}` で、後者のとき試行・round・境界の報告を `tuning_history.json` に書く（H-0134。H-0086 の「trial の履歴は保存しない」を置き換えた）。履歴を持たないモデル（format 1〜3 から load したもの）を export すると `null` を書き、ファイルは書かない。宣言だけを正とし、ディレクトリにある宣言されていない `tuning_history.json` は読まない。宣言の形が違う、ファイルが無い・`checksums` に無い・digest が合わない、正準の形でない履歴は、どれも `DESERIALIZATION_FAILED`。符号化は `core/tuning_codec.py`（Layer 0）の `history_bytes` / `parse_history`。
  - fit が overlay を記録したモデルだけ: `applied_training_params`（上記、H-0109）。
  - fit が記録したモデルだけ: `applied_sample_weight`（`"balanced"` / `"none"`）と `declared_categories`（列名 → 型付きの値の配列。JSON が型のまま戻せない値を含む記録は書かない）。`load()` はどちらも検査し、規則に合わない値は `DESERIALIZATION_FAILED`、キーが無ければ「分からない」（`None`）として読み、再 export でもキーを書かない（H-0120 改訂 1・4）。

## 15.2 互換性ポリシー（必須）

- `format_version` が読めない場合は明示的に拒否する（黙って壊れた復元をしない）。`format_version` は JSON の整数でなければならず、`true` や `2.0` のように等しく比べられる値も拒否する。`metadata.json` が JSON の object でないときも拒否する（H-0135）。
- 将来 migration を実装できる前提で serializer に拡張点を残す。
- 現行 `FORMAT_VERSION = 5`（H-0138）。`{1, 2, 3, 4, 5}` を loader が受理し、v1 artifact には no-op `TargetEncoder` を in-memory で注入して contract を整合させる（INV-5、H-0070）。v1・v2 の pipeline 状態には `stand_in_columns` が無く、どの列も元の値のカテゴリのまま読み、今と同じに予測する。3 に上げたのは意味の変更のためである: 位置の番号に置き換えた列（§15.3）を持つ状態を古い LizyML が読むと、元の値のカテゴリとして扱い、エラー無しに予測を間違える（H-0133 の実測）。そのため古い LizyML は v3 を拒否しなければならない。4 に上げたのは、`tuning_history.json` を知らない LizyML が v4 を読むと、そのファイルを無視して試行の履歴を空の表として見せるためである（#315 の欠陥。H-0134）。v1〜v3 は履歴を持たない成果物として読む。5 に上げたのは、エンコーダの状態がカテゴリを構造のキーで引く印（`category_identity_version`、§15.3）を持つためである。古い LizyML が v5 を読むと pandas の引き方で照合し、H-0138 が直した列に古い間違った符号を付ける。v1〜v4 の状態は印を持たないので、load の後も今までの pandas の引き方で予測する（予測は変わらない。古い状態を版 1 の同一性に書き直さない、一方向の互換。H-0138）。
- **フィールドの追加は後方互換の変更で、`format_version` を上げない。** フィールドの削除、型や意味の変更は破壊的変更で、`format_version` を上げる（H-0003）。`checksums`（H-0083）、`tuning`（H-0086）、`applied_training_params`（H-0109）、`applied_sample_weight` / `declared_categories`（H-0120）はいずれも追加で、版を上げずに入った（H-0133 が意味の変更で版を 3 に上げるまで v2。H-0134 が 4 に上げ、現行は H-0138 の `FORMAT_VERSION = 5`）。旧 loader は知らないキーを無視する。**例外（H-0134）**: 追加のフィールドでも、それが無いと、受理する古い loader が報告を黙って間違えるもの（`tuning.history` と `tuning_history.json`。古い loader は空の表を返す）は意味の変更として扱い、版を上げる。
- `load()` は毎回 `metadata.json` を検査する。必須キー（`_REQUIRED_METADATA_KEYS` = `format_version` / `task` / `feature_names` / `config` / `run_id`）が欠けていれば `DESERIALIZATION_FAILED`。
- **完全性の検査（H-0083）**: `load()` は各 `.pkl` のバイト列を 1 回だけ読み、`checksums` に記録された digest と照合してから、そのバイト列を `joblib.load(io.BytesIO(...))` で復元する（ファイルを再 open しないので、検査と復元の間の TOCTOU が無い）。digest が一致しないときは pickle を実行する前に `DESERIALIZATION_FAILED`（context: `file` / `expected` / `actual`）。
- **読むファイルは `checksums` の宣言で決まる（H-0135）**: 次の拒否はどれも `DESERIALIZATION_FAILED` で、context の `reason` が理由を示し、どの pickle も読む前に起きる。
  - `checksums` が上の形でない（`reason="checksums_malformed"`。配列・`null`・キーの過不足・digest の形の違いを含む）、`algorithm` が `sha256` でない（`"checksums_algorithm"`）。どの版でも拒否する。
  - `checksums` が無い: format 1・2（H-0083 以前に書かれた成果物）だけは、今までどおり検査せずに読む。format 3・4 では拒否する（`"checksums_missing"`）。
  - `fit_result.pkl` か `refit_model.pkl` が `files` に無い（`"required_file_unlisted"`）。
  - 決まった名前（`fit_result.pkl` / `refit_model.pkl` / 読む `analysis_context.pkl`）にあるのが通常のファイル（またはそれへの symlink）でない。ディレクトリなど（`"not_a_file"`）。
  - 宣言された `analysis_context.pkl` が無い（`"declared_file_missing"`）。宣言されていない `analysis_context.pkl` がディレクトリにあっても読まず（前の export が残したもの、#373）、`lizyml.persistence.loader` の logger の INFO に記録する。`checksums` の無い format 1・2 では、あれば読む。
  - 宣言の名前は、決まった名前（`fit_result.pkl` / `refit_model.pkl` / `analysis_context.pkl` / `tuning_history.json`）との完全一致で、大文字と小文字を区別する。宣言の名前をパスとして組み立てることはせず、知らない名前（`../x` のようなパスの形も含む）は無視する。決まった名前のファイルが symlink でも読む。読んだバイト列を digest で確かめてから復元するからである。
- **脅威モデル**: `metadata.json` 自体は署名しない。書き込み権限を持つ者は `checksums` を書き換えたり消したりできる（format 3・4 で消すと、読めなくなる）。`checksums` が検出するのは破損と改竄であり、悪意ある作成者に対して pickle を安全にするものではない。artifact は信頼できる出どころからだけ読む。
- `analysis_context.pkl` を持たない artifact（H-0026 以前）、または宣言していない artifact でも `predict()` と `evaluate()` は使える。load 後の診断 API（`residuals()` など）は、必要なデータが無いと `MODEL_NOT_FIT` で明示的に失敗し、最新版での再 export を促す（H-0026）。
- `pipeline_state_per_fold` を持たない `FitResult`（H-0114 より前の artifact。`format_version` は 2 のまま）は、そのまま読める（既定値 `None` がクラス属性として残るので、属性は `None` を返す）。`importance(kind="shap")` / `importance_plot(kind="shap")` だけが `MODEL_NOT_FIT`（context `task` / `kind` / `method` / `missing="pipeline_state_per_fold"`）で失敗し、`fit()` し直すことを促す。再 export では状態は作られない。この検査は `analysis_context` が無い場合の検査より先に行う。split / gain の重要度と他の API は変わらない（H-0114）。
- 非推奨の面とその削除目標（v1.0）の唯一の登録簿は `docs/DEPRECATIONS.md` である。非推奨の警告文は削除目標の版を明記する（H-0076）。

## 15.3 `export`（`Model Artifact`）

- `Model Artifact` を 1 ディレクトリにまとめる。中身は `metadata.json`、`fit_result.pkl`（`FitResult`）、`refit_model.pkl`（`RefitResult`）と、任意の `analysis_context.pkl`（load 後の診断 API が使う `y_true` と `X_for_explain`。無ければ load 時に `None`、H-0026）、tune 済みのモデルの `tuning_history.json`（strict JSON。H-0134）。`.pkl` は joblib（圧縮）で書く（H-0003）。前の export があるディレクトリに書くときは、書く前に任意のファイル（`analysis_context.pkl` と `tuning_history.json`）を消し、今回の export が持つものだけを書き直す。ほかのファイルには触らない（H-0135）。
- `.pkl` は pickle なので、読み込むと任意の Python を実行しうる。`Model.load()` は信頼できる出どころの artifact だけに使う（H-0003、上記の脅威モデル）。
- `Model.load()` で復元し、推論と評価情報参照に加えて診断 API（残差/SHAP/分類・校正可視化）も利用可能にする。
- **保存できるカテゴリ値（H-0127、#338、H-0133、#357）**:
  - **値の集合 V**（`core/category_values.py`、Layer 0）: `Model.export`、公開の `lizyml.persistence.export`、`export_code` は、特徴量のカテゴリについて同じ閉じた集合を受け付ける。型は `ADMITTED_TYPES` のどれかに厳密に一致すること（`str`・`int`・`float`・`bool`・`bytes`・`Decimal`・`date`・`datetime`・`timedelta`・`complex`・`pd.Timestamp`・`pd.Timedelta`、それらの numpy のスカラー、`np.datetime64`・`np.timedelta64`、要素が V の `tuple`・`frozenset`。入れ物は 8 段まで）。自分自身と等しくならない値（NaN、`Decimal` の NaN、NaN を含む `complex`、NaT）は入れ子の中でも外である。タイムゾーンつきの `Timestamp` は固定の時差・`ZoneInfo`・pytz の zone、タイムゾーンつきの `datetime` は固定の時差とキャッシュされた `ZoneInfo` だけを受け付ける（H-0133 の符号化で、夏時間の重なりでも等しさと hash を保てるもの。joblib はこれを保たないことがあり、それは下の保存の後の振る舞いの検査が拒否する。H-0136）。判定は値の型を最初に見るので、V の外の型の値のコードは走らない。
  - **拒否**: `Model.export`、公開の `lizyml.persistence.export`、`export_code` は、何かを書く前に同じ検査（`persistence/exporter.py` の `check_category_values`）を行う（H-0136。3 つの出口は同じモデルを受け付ける。公開の `export` の引数 `categories_checked` は H-0133 で非推奨で、渡しても検査を飛ばさず `DeprecationWarning` を出す。v1.0 で削除）。検査は次の順である。
    1. 保存する全ての pipeline 状態（refit、各 fold、`pipeline_state_per_fold` が無いモデルでは `pipeline_state` を `last fold` と呼ぶ）のエンコーダのカテゴリと最頻値、それから `declared_categories` を V で判定する。外の値があれば `SERIALIZATION_FAILED` を送出し（context `column` / `value` / `type` / `state`。値は例外を出さない表示で書く）、出力先に何も書かない。
    2. **保存の後の振る舞い（H-0136）**: 各状態のエンコーダを、保存したモデルが使う 2 つの保存処理で写す。`joblib`（`compress=3` で書いて読み戻す。成果物の pickle と同じ）と `json`（H-0133 の符号化と JSON。codegen の `pipeline_state.json` と同じ）である。学習したカテゴリを状態の順に 1 つのバッチにし、`unseen_policy="error"` のエンコーダで、元の状態と写した状態のそれぞれで変換する。次のどれかなら、その列を `SERIALIZATION_FAILED` で拒否する: 学習したカテゴリが写した状態で unseen になる（`kind="unseen"`）、写した状態の変換が例外を出す（`kind="raises"`、context `exception` に型の名前）、学習したカテゴリの符号が変わる（`kind="codes"`）、unseen の行を `unseen_policy="mode"` で最頻値に置き換えたときの符号が変わる（`kind="mode"`。同じバッチに、どのカテゴリとも等しくない値の行を 1 つ足し、各状態が自分の最頻値で置き換えた符号を比べる）。context は `column` / `state` / `serializer` / `kind`（値があれば `value` / `type`）。`declared_categories` は `json` だけで写し、写した値の `CategoricalDtype` で元の値の符号を比べる。ただし refit のエンコーダの状態が版 1 でその列を `identity_columns` に持ち、宣言が時刻の値を持つときは、写した値のキーの並びを元のキーの並びと比べる（生成コードは宣言をキーで戻すので。H-0139）。状態のカテゴリの `json` の経路の読み手は生成 `predict.py` なので、状態が版 1 でその列が `identity_columns` にあるときは、写した状態の符号（と最頻値で置き換えた符号）を、実行時のエンコーダの変換でなく、生成コードの `_keyed_codes` が写す `keyed_codes` だけで求める（実行時のエンコーダは写したカテゴリから `pd.Categorical.from_codes` を作り、JSON が `np.uint64` を戻した `int` と時刻の値を pandas が型にできないことがあるが、生成コードはその型を作らない。H-0143）。`joblib` の経路と identity で引かない列は、実行時のエンコーダで判定する。比較の実装は `features/encoders/state_changes.py` にあり、Layer 3 からは IF の `features/pipeline_base.py` を通して使う（§2.1）。
       - 約束の範囲: 学習したカテゴリと最頻値（と、それと同じ型で等しい値）の行が、保存の後もメモリ上と同じ符号を受け取ること。基準はメモリ上の状態である。メモリ上のモデルが unseen とする値の行（型をまたいで等しい値を含む）、メモリ上のモデル自身がバッチによって符号を変える場合は、この約束の外である（pandas 2 で pytz の値の fold が fit の中で変わる #379 は H-0139 で、NumPy の 1 に等しいスカラーと 1 を包む tuple を pandas が取り違える #380 は H-0138 で、キーで引くようになり起きない）。
       - 等しさが推移的でない値（`np.uint64(4) == 4 == np.timedelta64(4, 'M')`、`np.uint64(4) != np.timedelta64(4, 'M')`）では、結果は状態のカテゴリの順（その状態の行の現れる順）で変わる。
    3. 状態ごとに adapter の `unsaveable_categories` にも尋ねる（下の IF）。
  - **fit の警告**: `fit()` は CV の学習の前に、全ての学習行で fit した pipeline の状態（refit の pipeline と同じ行）を V で判定し、外の値を持つ列ごとに `UserWarning` を出す。V の外の値が無ければ、fit を終えた後で、export が判定するのと同じ保存する状態と `declared_categories` に上の 2 の検査を当て、拒否される列ごとに `UserWarning` を出す（H-0136、`exporter.warn_saved_state_changes`。結果は状態のカテゴリの順で変わるので、保存する状態そのもので判定する）。fit は止めない。
  - **位置の番号への置き換え**: LightGBM は booster を pickle するとき、各カテゴリ列のカテゴリをモデル文字列に JSON で書くので、書けない値（`category_json.is_writable`）を持つ列はそのままでは保存できない。LightGBM の provider はこの判定を pipeline に渡し、エンコーダは fit の時点で、そのような値を一つでも持つ列を `stand_in_columns` として状態に記録する（CV は transform より前に状態を保存するので、fit で決める）。transform は元の値で照合した後（未知の値、`unseen_policy`、欠損は今と同じ）、その列だけカテゴリを位置の番号 0..k−1 に置き換える。符号は変わらないので、予測は変わらない。それ以外の列と、`get_native_model()` が返す Booster の挙動は変わらない。メモリの中のモデルで、置き換えた列を持つ Booster に元の値の列を直接渡すと、予測を黙って間違える。
  - **カテゴリの値の同一性（H-0138、時刻の型は H-0139）**: カテゴリの列の値が全て V の中にある列では、「どの値が同じカテゴリか」を `core/category_values.py` の `category_identity_key` が決める。数として等しい値（`-0.0`・`0`・`False`・`Decimal("-0")`、`True` と `np.uint8(1)`、`Decimal("1")` と `np.int8(1)`）は 1 つのカテゴリ、文字列と `np.str_`、bytes と `np.bytes_` も等しければ 1 つ、`tuple` と `frozenset` は要素ごとにこの規則で比べる。キーは値自身の `==` と `hash` を走らせない。V の外の値を持つ列は、キーを使わず今までの pandas の経路を通る。
    - **時刻の型（H-0139、#379）**: 時刻の値のキーは `core/category_temporal.py` の `temporal_key` が決める。座標はアト秒の整数（Python の `int`）である。タイムゾーンつきの `datetime`・`Timestamp` は、同じ UTC の瞬間なら時区の種類（UTC・IANA の `ZoneInfo`・pytz・固定の時差）や fold によらず 1 つのカテゴリ。naive の `datetime`・`Timestamp`・`np.datetime64` は壁時計の座標が（ns より細かい単位も含め）正確に等しければ 1 つ。naive とタイムゾーンつきは別、`date` と 0 時の `datetime` は別。固定の長さの期間（`pd.Timedelta`・`datetime.timedelta`・固定の単位の `np.timedelta64`）は長さが正確に等しければ 1 つ、単位の無い `np.timedelta64` はどの固定の期間とも別。時区の名前と fold はキーに入らないので、保存（joblib・H-0133 の符号化）でキーは変わらない。
    - **年・月の単位の `np.timedelta64`**: 長さを持たないのでキーが無い。fit（組み立て、エンコーダの fit、生成 `train.py`）は、行の値・入れ子の値・宣言したカテゴリ（使われていないものも）にこの値を持つ列を、学習の前に `DATA_SCHEMA_INVALID`（context `{"column", "units", "reason": "calendar_timedelta64"}`。生成コードは同じ文の `ValueError`）で拒否する。predict に来たこの値は unseen として `unseen_policy` に従う。V と符号化は変えないので、この値を持つ保存したモデルは今までどおり読み込める。
    - **単位の無い `np.datetime64`**: NumPy 自身が文字列にできない値で、キーが無い。年・月の `timedelta64` と同じ位置で fit が拒否する（context `{"column", "reason": "unitless_datetime64"}`）。predict に来たものは unseen で、警告の文では `<unprintable datetime64>` と書く。符号化はこの値も往復する（H-0139）。
    - **今正しく動く列は変えない**: データの組み立て（`dataframe_builder._apply_categorical`）とエンコーダの fit は、pandas の cast（`astype("category")`）と今までのカテゴリ・最頻値を実際に作り、それがキーに忠実か（カテゴリのキーが一意、欠損でない行の符号のカテゴリのキーが行のキーと同じ、欠損は -1、宣言したカテゴリは同じ順）を確かめる。忠実なら今の結果をそのまま使う。cast が例外を出すか忠実でないときだけ、キーで作り直す（代表の値はキーごとに最初に現れた値、順は推定した列で `sorted(..., key=str)`、宣言した列で宣言の順、最頻値はキーで数えて同数ならカテゴリの順で先のもの）。作り直した列は必ず位置の番号への置き換えの列（`stand_in_columns`）になる。
    - **データ層**: 組み立てが cast できない列・忠実でない列は元の object の列のまま残し、`DataFrame.attrs` の非公開の項目（`IDENTITY_OBJECT_COLUMNS`）に列の名前だけを記録する。`NativeFeaturePipeline.fit` はこの列もカテゴリの列として選ぶ。カテゴリは各 fold と refit の `pipeline.fit` の中で学習する（リークの境界は変わらない）。
    - **状態と照合**: 新しく fit したエンコーダの状態は `category_identity_version`（現行 1）と、キーで扱う列の一覧 `identity_columns` を持つ。transform はその列を、H-0127 の `comparable_values` を通した後、行のキーから位置の番号を直接引き（pandas の object の Index・`set`・`isin` を使わない）、unseen の方針を番号の上で当てる。印の無い状態（v1〜v4）は今までの pandas の引き方で動く。知らない印（`category_identity_version` が 1 以外、整数でない、`identity_columns` の形が違う）は `load_state`（と `Model.load`）で `DESERIALIZATION_FAILED`。`export_state_json` も印と一覧を書く。
  - **`metadata.json` の `declared_categories`**: 型の印つきの JSON（下の §15.4）で書く。`lizyml.persistence.load` が一度だけ復号し、`Model.load` はその結果を使う。形の外れた印は両方で `DESERIALIZATION_FAILED`。

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
- **refit の再現の約束（H-0120、H-0059 の約束を戻す）**: `Model.fit(df)` の後の `export_code(path)` で生成したプロジェクトで、同じデータを parquet で保存して `python train.py <data>` を実行すると、`artifacts/` による校正前の予測（regression の予測値、binary の陽性クラスの確率、multiclass の各クラスの確率）が、LizyML の refit モデルの校正前の予測とどの入力行でも `rtol=1e-7` で一致する。
  - **前提**: 同じ計算機。同じ版の LightGBM / numpy / pandas / scikit-learn（export 時の版を `config.json` の `_versions` に記録し、生成 `train.py` は違う版ごとに名前を挙げて警告し、学習は続ける）。LightGBM の決定性は `model.params` の `deterministic: true` / `force_col_wise: true` / `num_threads: 1` で固定でき、これらは `lgbm_params` に書かれて両側に効く。
  - **CSV**: 生成 `train.py` が `pd.read_csv` で読み、`declared_categories` を当てた後の、学習に使うすべての列（目的変数・特徴量・時間・グループ）の値と dtype が fit 時と等しい場合に限って約束する（型の混じった列、文字列として読まれる時刻、`"01"` のような文字列の目的変数は満たさない）。
  - **約束しないもの**: 校正器（fold のモデルは再現しない）、異なるデータでの再学習、H-0120 より前に生成したプロジェクト、`applied_training_params` / `applied_sample_weight` / `declared_categories` の記録が無い artifact を load したモデルからの export の該当部分、型の混じった列を持つ fit の再学習（export 直後の `predict.py` は LizyML と同じ符号を使う）。
  - **`config.json` が持つ値**: `inner_valid`（refit が使った strategy。`export_code` が fit と同じ `resolve_inner_valid` を、config と `applied_training_params` から呼び直して作る。検証集合が無ければ `null`）と `early_stopping_rounds`（adapter の patience、無ければ `null`）を独立に必ず書き、生成 `train.py` は既定値なしで読む。callback は両方があるときだけ付く。`sample_weight`（`"balanced"` か `null`、fit の記録から）、`declared_categories`（入力で `category` だった列）、`categorical_rule`（データの組み立てが `category` に変える列の規則、`{"explicit": [...], "auto": bool}`）、`_versions`。
  - **生成 `train.py` の学習**: 時間順の split と `blocked_group_kfold` は `Series.argsort()` で行を並べてから学習する。inner valid は 4 つの strategy を同じ numpy / scikit-learn の呼び出しで移したもので分割する。multiclass の `balanced` の重みは並べた後の全行で `compute_sample_weight("balanced", y)` と同じ値を作り、inner-train の行にだけ付ける。`LGBMAdapter.fit` と同じ Dataset と callback の組み立てで `lgb.train` を呼ぶ。
  - **カテゴリの符号（#304）**: 生成 `train.py` は `categorical_rule` で LizyML と同じ `astype("category")` を当ててから、`CategoricalEncoder.fit` と同じ 2 つの分岐でカテゴリと最頻値を決める。`pipeline_state.json` は列ごとに `{"categories": [型付きの値, ...], "mode": 値}` を持ち、配列の位置が符号である（`str` をキーにしない）。`predict.py` も同じ値で符号を引く。
  - **カテゴリの同一性（H-0138）**: `config.json` は元の pipeline の `category_identity_version` を必ず書く（H-0138 より前の状態から load したモデルでは `null`。古いカテゴリを黙って新しい同一性に上げない）。版 1 のとき、`pipeline_state.json` は `category_identity_version` と `identity_columns` を持ち、生成 `train.py` は §15.3 と同じ「cast・忠実の判定・キーでの作り直し」で組み立てと fit を行い（宣言したカテゴリの復元も符号をキーで引き、pandas が宣言の一覧をそのまま持てなければ fit が宣言の順にキーで作る）、`train.py` と `predict.py` の符号の照合は印のある列をキーで引く。キーと照合の写しは `codegen/identity_template.py` にあり、正と同じ答えを返すことをテストで確かめる。`null` のときは H-0138 より前と同じ pandas の引き方のコードとして動き、`pipeline_state.json` に印を書かない。`predict.py` は `pipeline_state.json` の知らない印（1 以外、整数でない、`null`、形の違う `identity_columns`）を値を読む前に `ValueError` で拒否し、`train.py` も `config.json` の知らない版を拒否する。
  - **値の書き方（H-0133）**: 特徴量のカテゴリ・最頻値・`declared_categories` は、素の値（`str`・`int`・`float`・`bool` と、それらの numpy のスカラー）を JSON の値で、それ以外の V の値を `{"$lizyml": 印, ...}` の形の JSON の object で書く。印と欄の表は `core/category_values.py`（H-0133 P2）が正で、生成 `train.py` / `predict.py` は同じ表の写しで符号化・復号する（写しと正が同じ答えを返すことをテストで確かめる）。復号は決まった位置だけで行い、形の外れた object は `ValueError` である。
  - **拒否**: 特徴量のカテゴリ・最頻値・`declared_categories` が V（§15.3）の外にあるとき、目的変数のラベルが `str` / `int` / `float` / `bool` と、`.item()` がそれらになり元の値と等しい numpy のスカラーの外にあるとき（ただし `np.datetime64` と `np.timedelta64` は外す。Y・M 単位の `np.timedelta64` は `.item()` が int で、numpy が元の値と等しいと判定するため。H-0133）、`export_code` は何も書く前に `SERIALIZATION_FAILED` で拒否する（黙って `str` にしない）。ラベルの集合は H-0133 で広げていない。特徴量のカテゴリについては、`export_code` も `Model.export` と同じ `check_category_values`（§15.3 の 3 つの検査）を、何も書く前に行う（H-0136）。
- **列名の別名（H-0141、#382）**: 生成プロジェクトは全ての列を文字列の別名 `str(name)` で呼ぶ（JSON の key・CSV の見出し・parquet の列名が、どれも同じ文字列で戻るように）。fit したモデル・Config・`Model.export` の保存形式・実行時の fit / predict の名前は変えず、書き出す写しだけを別名にする。
  - **受け付ける名前**: 型が完全に `str`・`numpy.str_`・`int`・`float`・`bool`・`NoneType`・`np.float64` のどれかで、別名が空でなく、NUL と UTF-8 にできない文字を含まないもの（`7` → `"7"`、`7.0` → `"7.0"`、NaN → `"nan"`、`True` → `"True"`、`None` → `"None"`）。数の正規化はしない。規則は `codegen/column_names.py` の `column_alias` だけにあり、生成 `train.py` / `predict.py` はそのソースの写しを持つ（同じ答えを返すことをテストで確かめる）。
  - **書く前の拒否**: `export_code` は、fit の状態を取った後・カテゴリの値の検査の前に、`generate_code` は入口で、特徴量の名前と Config の役割の名前（`data.target`・`data.time_col`・`data.group_col`・`features.exclude`・`features.categorical`・`split.blocks.col`・`split.groups.col`・`inner_valid.group_col`）を調べ、何も書く前に `SERIALIZATION_FAILED` で拒否する。context は `surface: "export_code"`・`reason`（`unsupported_column_name` か `column_name_collision`）・`kind`（`type`・`empty_alias`・`invalid_alias_text`・`unknown_reference`・`ambiguous_reference`・`duplicate_alias`）・`alias`・`columns`（各項目に `where`・`position`・`type`（`builtins.int` の形）・`repr`・`alias`）である。調べる順は、受け付けない名前（特徴量の順、次に役割）、宣言したカテゴリの key の対応づけ、別名の重なり（最初の組）。重なりは別名の文字列で調べ、2 つの特徴量の別名が同じとき、または文字列でない特徴量の別名が役割の名前と同じときである（文字列の特徴量と同じ文字列の役割の名前は 1 つの列）。
  - **宣言したカテゴリの key**: `Model.load` したモデルでは JSON の key の形で戻る（`7` → `"7"`、`inf` → `"Infinity"`、`None` → `"null"`）。書き出す写しの中で、型と値が同じ特徴量を先に、無ければ `str` の key を `numpy.str_` の特徴量の `str(name)` か、それ以外の特徴量の JSON の key の形に対応づけ、その特徴量の別名を key にする。どれにも当たらない key は `unknown_reference`、両方の規則か 2 つ以上の特徴量に当たる key は `ambiguous_reference`。
  - **印 `column_name_policy`**: `config.json` と `pipeline_state.json` の両方に `"str_alias_v1"` を書く。生成スクリプトは、予測と学習し直しの最初に（カテゴリの値の復号・identity の検査・書き込みより前に）、両方に無ければ H-0141 より前のプロジェクトとして今までの引き方で動き（学習し直しても印を足さない）、値が `"str_alias_v1"` でなければ `unsupported_column_name_policy`、片方だけなら `column_name_policy_mismatch`、両方にあれば名前の形（`feature_names` は文字列の一意な一覧で順まで同じ、カテゴリの列の一覧は同じで特徴量に含まれる、カテゴリの key はカテゴリの列と同じ集合、`identity_columns` と宣言の key はカテゴリの列に含まれる、役割の名前は文字列）を確かめ、外れれば `invalid_column_name_state` で止める。
  - **入力の境界**: 印のあるプロジェクトでは、生成スクリプトは型つきの frame の列名を別名にしてから引く（重なりは `column_name_collision`、受け付けない名前と MultiIndex は `unsupported_column_name`、足りない別名は `missing_column_alias`）。CSV は最初の行を `csv.reader(strict=True)` で読み、空の見出しと pandas が読んだ見出しとの違いを `invalid_column_header`、重なった見出しを `column_name_collision` で拒否する。parquet は列名が文字列・平ら・一意であることを確かめる。`train.py`・`predict.py`・`test_equivalence.py` の読み込みは全てこの検査を通る。例外は `ValueError` の子で `reason` と問題の名前を持つ。CSV と parquet の見出しは別名で書く（`"Unnamed: 0"` を `None` の列と推し量らない）。
  - 生成 `train.py` は LightGBM にカテゴリの列を位置で渡す（H-0142 と同じ）。すでに書き出したプロジェクトは印を持たず、今までの振る舞いのまま。直すには元のモデルから書き直す。
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
- **format_version rejection**: 未知の version（`0`, `5`, `99`, `-1`, `None`）で `DESERIALIZATION_FAILED`。`{1, 2}` だけを受理する loader（模擬）は v3 を拒否する。
- **v1 と v2 の形（H-0133 P8）**: 旧版で保存した成果物はリポジトリに無いので、テストは今の成果物を書き直して作る（`tests/test_persistence/_artifact_versions.py` の `to_v2_shape`: `stand_in_columns` を除き、`format_version` を 2 にして checksums を書き直す。v1 はさらに `target_encoder` を除く）。v4 の形は `to_v4_shape` で、エンコーダの `category_identity_version` と `identity_columns` を除いて `format_version` を 4 にする（H-0138）。v3 の形はそれに続けて作る。今の版を「v2」と呼ぶテストは置かない。
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
- 依存の下限バージョンでのテストを CI に含める（H-0126）。`Quality (lowest-direct deps)` は lock を使わずに、本体と全 extras を `uv pip install --resolution lowest-direct` で宣言した下限の版で入れ、`scripts/check_dependency_floors.py` で入った版が全て下限と等しいことを確かめて（版の表を表示する）から、slow を除くスイートを流す。`quality` の各レーンは依存の導入の前に `uv lock --check` で lock と `pyproject.toml` の整合を確かめる。`[dependency-groups].dev` は extras の下限を写さず `lizyml[tuning,explain,plots,calibration]` で参照する。
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
├── _extras.py                     optional 依存の扱い (Layer 0)
├── _version.py                    hatch-vcs が生成 (Layer 0、Git 管理外)
│
├── core/                           ── Layer 0: Foundation ──
│   ├── exceptions.py               LizyMLError + ErrorCode
│   ├── logging.py                  logger + run_id + output_dir
│   ├── registries.py               MetricRegistry, CalibratorRegistry
│   ├── types/
│   │   ├── fit_result.py           FitResult
│   │   ├── predict_result.py       PredictionResult
│   │   ├── tuning_result.py        TuningResult, TrialResult
│   │   ├── artifacts.py            RunMeta, SplitIndices, DataFingerprint
│   │   ├── task.py                 TaskType (canonical, H-0075)
│   │   ├── target_encoder.py       TargetEncoder (H-0070)
│   │   ├── search_dim.py           SearchDim, FloatDim, IntDim, CategoricalDim, DimCategory
│   │   ├── oof_coverage.py         compute_oof_valid_mask (H-0130)
│   │   └── smart_ratio.py          RatioRange, SMART_RATIO_RANGES (H-0130)
│   ├── param_domain.py             パラメーターの値の閉じた集合 (Layer 0, H-0095)
│   ├── category_values.py          保存できるカテゴリ値の閉じた集合と符号化 (Layer 0, H-0133)
│   ├── category_temporal.py        時刻の値の同一性のキー temporal_key (Layer 0, H-0139)
│   └── specs/                      ProblemSpec, FeatureSpec (Layer 0、data/ が使用)
│
│                                   ── Layer 4: Facade (core/ の残り) ──
│   ├── model.py                    Model facade (組み立てと委譲のみ)
│   ├── _model_factories.py         splitter / inner_valid / estimator provider 構築
│   ├── _model_predict.py           predict / predict_proba 実体 (facade 委譲先)
│   ├── _model_plots.py             ModelPlotsMixin
│   ├── _model_tables.py            ModelTablesMixin (EstimatorProvider 経由)
│   ├── _model_metrics.py           _has_metric_content, _filter_metrics
│   ├── _model_persistence.py       ModelPersistenceMixin
│   ├── _model_tuning.py            ModelTuningMixin (tune orchestration、唯一の writer mixin、H-0091)
│   ├── _resource_warning.py        fit / tune の学習前のメモリの警告 (H-0140)
│   ├── _model_state.py             FitState / TuningState frozen snapshot (H-0074/H-0084)
│   ├── train_components.py         TrainComponents (frozen dataclass)
│   └── seed.py                     seed 固定ユーティリティ
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
│   │   ├── categorical_encoder.py  カテゴリ処理部品
│   │   ├── category_identity.py    構造のキーでのカテゴリの学習と照合 (H-0138)
│   │   └── state_changes.py        保存処理で写した状態と元の状態の比較 (H-0136)
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
│       ├── memory_check.py         num_leaves のメモリの見積もりと警告 (H-0140)
│       ├── metric_bridge.py        BaseMetric → LightGBM feval (§2.1 の estimators → metrics)
│       ├── category_json.py        保存できるカテゴリ値の判定 (H-0127)
│       └── defaults.py             _COMMON_DEFAULTS / default_space / default_fixed_params
│
├── metrics/                        ── Layer 1: Metrics ──
│   ├── base.py                     BaseMetric
│   ├── registry.py                 MetricRegistry helpers + task validation
│   ├── regression.py               RMSE, MAE, R2, RMSLE, MAPE, Huber
│   └── classification.py           LogLoss, AUC, AUCPR, F1, Accuracy, Brier, ECE, PrecisionAtK
│
├── calibration/                    ── Layer 1: Calibration ──
│   ├── base.py                     BaseCalibratorAdapter + BaseCalibrationResult (marker, H-0130)
│   ├── cross_fit.py                cross_fit_calibrate + CalibrationResult
│   ├── registry.py                 get_calibrator
│   ├── platt.py                    PlattCalibrator
│   ├── isotonic.py                 IsotonicCalibrator
│   └── beta.py                     BetaCalibrator
│
├── training/                       ── Layer 2: Training ──
│   ├── cv_trainer.py               CVTrainer (outer CV loop)
│   ├── refit_trainer.py            RefitTrainer + RefitResult
│   ├── results.py                  RefitResultView (Protocol, Layer 2 IF, H-0130)
│   ├── inner_valid.py              BaseInnerValidStrategy + 6 concrete
│   └── oof_assembly.py             fill_oof / get_fold_pred / init_oof (+ compute_oof_valid_mask を再公開)
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
│   ├── loader.py                   load() + format_version validation
│   ├── _manifest.py                checksums の形の検査 declared_files (H-0135)
│   └── _category_preflight.py      書く前のカテゴリの検査 (H-0136)
│
└── codegen/                        ── Layer 3: Codegen (optional, H-0059/H-0073) ──
    ├── generator.py                generate_code() エントリ (Model.export_code から)
    ├── artifact_writer.py          pipeline_state.json / model.txt 等の書き出し
    ├── config_writer.py            config.json 書き出し
    ├── identity_template.py        生成コードに写すカテゴリの同一性 (H-0138)
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
- `Splitter / Tuner / Calibrator` を組み立てる（Calibrator は `calibration/registry.py` の `get_calibrator` で名前から得る）。`EstimatorAdapter / FeaturePipeline` は §2.2 の provider の factory から得る。
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
