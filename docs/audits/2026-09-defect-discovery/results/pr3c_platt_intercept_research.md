# Platt calibration の intercept はどうあるべきか（2026-09-15）

管理者の依頼: 「Platt の Calibration という目的に合わせた場合、Intercept はどうあるべきか調査してください」。
設計レビュー round 1 は `fit_intercept=True` の強制を「元資料に無い新しい方針」として推奨していた。

## 結論

**intercept は Platt の定義に含まれる必須のパラメーターであり、slope と同時に、正則化せずに
最尤推定するものである。** したがって `fit_intercept=True` の強制は新しい方針ではなく、
**BLUEPRINT §12.2 が名指す「Platt Scaling」という手法の定義から導かれる制約**である。

## 1. 原典の定式化

- **Platt (1999), "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods"**:
  `P(y=1 | f) = 1 / (1 + exp(A·f + B))`。**A と B の 2 パラメーター**を、学習データの負の対数尤度
  `−Σ t_k log r_k + (1 − t_k) log(1 − r_k)` の最小化で同時に求める。
  目標値は 0/1 ではなく**平滑化した値** `t+ = (N+ + 1)/(N+ + 2)`, `t− = 1/(N− + 2)`（ベイズ事前分布、
  過学習の抑制）。明示的な正則化項は無い。
- **scikit-learn の文書**（`~/local-docs/packed/scikit-learn-docs.md` 7607 行付近）: sigmoid 回帰器は
  「Platt の logistic model `p = 1/(1 + exp(A f + B))`。A と B は最尤推定で決める実数」。
- **scikit-learn の実装** `sklearn/calibration.py::_sigmoid_calibration`（sklearn 1.8.0、ローカルで読んだ）:
  `(A, B)` の 2 パラメーターを **Platt の目標値平滑化**つきで、**正則化なしの L-BFGS-B** により推定する。
  初期値は `A=0, B=log((N− + 1)/(N+ + 1))`。docstring は「slope」と「intercept」と明記している。

## 2. intercept が担う役割

base モデルのスコアが「0 なら確率 0.5」という位置からずれていること（**offset のずれ**）を補正する。
`B = 0` に固定すると、スコア 0 が常に確率 0.5 に写像され、**offset のずれを直せない**。
slope しか動かせないので、slope がずれを埋めようとして歪む。

## 3. 実測（`/tmp/claude-1000/platt_intercept_experiment.py`）

真の対数オッズが `slope·s + offset` になる、意図的に較正のずれたスコアを生成し、
held-out で log loss と ECE（15 bin）を比較した。seed 0。

```
true slope=0.5 offset=1.0 n_train=2000
  fit                     slope    icpt  logloss    ECE
  A intercept, no L2      0.547   0.951   0.5399  0.022
  B intercept, C=1.0      0.547   0.950   0.5399  0.021   <- LizyML の現行既定
  C no-icpt,  no L2       0.446   0.000   0.6315  0.197
  D no-icpt,  C=1.0       0.446   0.000   0.6315  0.197
  sklearn _sigmoid        0.545   0.950   0.5399  0.021

true slope=0.5 offset=1.0 n_train=100
  A intercept, no L2      0.461   0.756   0.5058  0.107
  B intercept, C=1.0      0.452   0.753   0.5063  0.114
  C no-icpt,  no L2       0.432   0.000   0.6099  0.223
  sklearn _sigmoid        0.435   0.747   0.5077  0.107

true slope=1.0 offset=0.0 n_train=2000   （offset のずれが無い場合）
  A intercept, no L2      1.001   0.023   0.4584  0.025
  C no-icpt,  no L2       1.000   0.000   0.4583  0.019

true slope=2.0 offset=-1.5 n_train=2000
  A intercept, no L2      1.999  -1.386   0.2775  0.025
  B intercept, C=1.0      1.981  -1.376   0.2774  0.024
  C no-icpt,  no L2       1.552   0.000   0.3625  0.133
  sklearn _sigmoid        1.973  -1.371   0.2774  0.023
```

- **intercept なし**は、offset のずれがあると **ECE が 0.02 → 0.13〜0.20**、log loss も大きく悪化し、
  **slope まで歪む**（真の 2.0 に対して 1.55）。
- offset のずれが無いときは intercept の有無で差が無い（ノイズの範囲）。**intercept を推定して損をする場面は無い。**

## 4. 調査で出てきた関連事項（intercept の扱いに直結する 2 点）

### 4.1 intercept は正則化してはいけない —— solver によっては正則化される

- `lbfgs`（LizyML の既定）は **intercept を正則化しない**。実測でも `C=1.0` と正則化なしで intercept が一致した。
- **`liblinear` は intercept も正則化する**（scikit-learn 文書: LIBLINEAR 実装は「intercept も正則化する」、
  影響を減らすには `intercept_scaling` を調整する）。つまり solver を `liblinear` に上書きできると、
  **intercept が 0 に向かって縮み、intercept なしの失敗に部分的に近づく**。
- 設計 round 1 は「sklearn が受け付ける solver / penalty の組み合わせは受理する」としていたが、
  intercept の役割から見ると **`liblinear` は Platt の定義に反する方向に働く**。

### 4.2 LizyML の現行既定は、原典の Platt から少しずれている

| | 原典 Platt / sklearn `CalibratedClassifierCV(method="sigmoid")` | LizyML 現行 |
|---|---|---|
| 正則化 | なし | **`C=1.0` の L2**（slope のみ。lbfgs なので intercept には掛からない） |
| 目標値 | 平滑化（`t+ = (N+ + 1)/(N+ + 2)` 等） | 0/1 のまま |

実測の影響は小さい: n=2000 では差なし、**n=100 で slope 0.452 vs 0.461、ECE 0.114 vs 0.107**。
これは**本 PR 以前からの逸脱**で、`2ac4331`（Phase 13）の実装が `LogisticRegression(C=1.0)` を選んだことに由来する。
既定を原典に寄せると、**既存の platt 利用者全員の calibrated 結果が変わる**。

## 5. 設計への帰結

1. **`fit_intercept=True` を強制する。** 分類を「新しい方針」から「Platt の定義に基づく制約」へ改める。
2. **intercept を正則化する solver（`liblinear`）の扱いを決める必要がある**（§4.1）。
3. **既定を原典の Platt に寄せるかどうかを決める必要がある**（§4.2）。本 PR の範囲に入れるかも含めて。

## 6. 管理者の決定（2026-09-15）と、それが生んだ分岐

- **`liblinear` は拒否する。**
- **platt の既定値を本 PR で原典の Platt（正則化なし・目標値平滑化）に寄せる。**

### 6.1 目標値平滑化を LogisticRegression で再現できるか（実測）

各サンプルを「正例として重み `t_k`」「負例として重み `1 − t_k`」の 2 行に複製し、
`sample_weight` つきの正則化なし LogisticRegression で fit すると、Platt の目的関数と一致する。
`/tmp/claude-1000/platt_smoothing_equivalence.py` で sklearn の `_sigmoid_calibration` と比較した:

```
slope=0.5 offset=1.0 n=2000 |s|~2   dup=(0.55724, 1.11264) ref=(0.55724, 1.11265) diff=(4.6e-06, 1.4e-05)
slope=0.5 offset=1.0 n=100  |s|~2   dup=(0.53051, 0.83424) ref=(0.53054, 0.83466) diff=(3.5e-05, 4.2e-04)
slope=2.0 offset=-1.5 n=2000 |s|~2  dup=(2.18957,-1.49716) ref=(2.18985,-1.49797) diff=(2.8e-04, 8.0e-04)
slope=0.3 offset=0.5 n=2000 |s|~40  dup=(0.28363, 0.45664) ref=(0.28364, 0.45657) diff=(2.2e-06, 6.8e-05)
```

**最適化の許容誤差の範囲（≤ 1e-3）で一致する。**

### 6.2 ただし sklearn の API の変化が「正則化なし」に掛かる（実測）

- sklearn 1.8.0 では **`penalty` が非推奨**（1.10 で削除）: `penalty=None` は
  `FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10`。
- 1.8 の推奨の書き方 **`C=np.inf` でも** `UserWarning: Setting penalty=None will ignore the C and l1_ratio parameters` が出る。
- つまり LogisticRegression 経由で既定を原典に寄せると、**platt を使う fit のたびに警告が出る**。
- 依存の下限は `scikit-learn>=1.3`（`pyproject.toml`）で、CI の lowest-direct レーンがそれを使う。
  **受理名を sklearn のシグネチャから導出すると、受理される config が sklearn のバージョンで変わる**
  （1.8 で `penalty` / `n_jobs` が非推奨、1.10 で削除予定）。

### 6.3 ~~環境のずれ~~ —— **誤りだった（2026-09-15、設計レビュー round 2 の指摘で訂正）**

初版は「`.venv`（sklearn 1.8.0 / scipy 1.17.1）が `uv.lock`（1.7.2 / 1.15.3）とずれている」と書いた。**誤り。**
`uv.lock` には scikit-learn が **2 エントリ**あり、Python のバージョン marker で分岐する —— `< 3.11` は
1.7.2 / scipy 1.15.3、`>= 3.11` は 1.8.0 / scipy 1.17.1。Python 3.11 の `.venv` は lock どおりである。
`grep ... | head -4` で最初のエントリだけを見て全体を判定した（この run で繰り返している「部分を見て全体を判定する」誤り）。

**別の事実として残るもの**: CI の lowest-direct レーンは `uv sync --frozen --dev --resolution lowest-direct` で、
`--frozen` は lock をそのまま使うため、**このレーンで sklearn 1.3 / scipy 1.10 が実際に入っている証拠にはならない**。

## 出典

- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. <https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods>
- Lin, Lin, Weng (2007). *A Note on Platt's Probabilistic Outputs for Support Vector Machines*. <https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf>
- Niculescu-Mizil, Caruana (2005). *Predicting Good Probabilities With Supervised Learning*. <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>
- scikit-learn 文書（ローカル packed）: Probability calibration — Sigmoid。LinearSVC / liblinear の intercept 正則化。
- scikit-learn 1.8.0 ソース: `sklearn/calibration.py::_sigmoid_calibration`。
