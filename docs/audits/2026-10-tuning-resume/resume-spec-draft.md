# tune の resume と storage への再接続: 仕様案（2026-10-11）

HISTORY に入れる Proposal の草案（HISTORY に合わせて日本語）。根拠の資料は同じディレクトリの `design-vs-implementation.md`。対象のコードは `develop` `6273d39` と同じ `lizyml/`。

## 0. 目的（管理者の決定、2026-10-11）

`tune(resume=True)` には二つの目的があり、どちらも**目的関数を変えない**:

- **(P1) 探索範囲を変えて、さらに良い解を探す。** 範囲を変えるのは、外部のアルゴリズムか利用者。自動の境界拡張（H-0068）もこれに含まれる。
- **(P2) 障害のあとで再開する。** 永続 storage の study に `resume=True` で再接続し、落ちた時点の探索空間から続ける。

決定:

- (D1) 範囲を変えるのは外部のアルゴリズムか利用者。
- (D2) 範囲を変えたあとでも、全試行の最良値を採用する（当初の範囲の値が最良なら、その値でよい）。
- (D3) 探索空間は study と一緒に保存し、再開で復元する。
- (D4) ライブラリの版の違いは警告だけ。照合の記録が無い過去の study は拒否する。
- (D5) 再接続は `resume=True` で行う。

## 1. 用語

- **目的関数**: 探索するパラメーターの値 θ を受け取り、OOF の最初の評価指標の値を返す関数。データ、config の探索空間以外の部分、分割、評価指標で決まる。
- **study**: Optuna の study。一つの study の試行は、すべて同じ目的関数の観測である（INV-R1）。
- **round**: 一回の `tune()` 呼び出しで足した試行のまとまり。
- **宣言空間**: その round で利用者（または外部のアルゴリズム）が宣言した探索空間を、`space_mode` に従って解決したもの（provider の境界は付けない）。
- **実効空間**: その round で実際にサンプルした空間（宣言空間に境界の拡張を重ね、provider の境界を付けたもの）。
- **系譜**: study の round ごとの記録（宣言空間、実効空間、拡張した次元、試行の範囲、前後の最良値）。

## 2. 不変条件

- **INV-R1（同じ目的関数）**: 一つの study の試行は、すべて同じ目的関数の観測である。目的関数の同一性を照合できない study には、試行を足さない。
- **INV-R2（最良値の再現）**: 採用する最良値は study 全体の最良の試行である。その試行のパラメーターで `fit` すると、その試行と同じ学習になる（`config ⊕ fixed ⊕ その試行のサンプル値`）。
- **INV-R3（空間の変更は観測を消さない）**: 探索空間を変えても、古い試行は同じ目的関数の観測として残り、最良値の候補であり続ける。
- **INV-R4（拒否は study を変える前）**: 拒否は、enqueue・試行の追加・study の属性の書き込みより前に行い、study と Model の状態を変えない。
- **INV-R5（再接続は継続と同じ）**: 落ちたあとに再接続した結果は、同じプロセスで続けた場合と同じ空間・系譜・round 番号になる。

## 3. 呼び出しの種類

| `resume` | メモリ上の study | `storage` / `study_name` | storage にその study | 挙動 |
|---|---|---|---|---|
| False | 問わない | なし | — | 新しい study（メモリ）。系譜は round 1 から |
| False | 問わない | あり | 無い | 新しい study（storage）。系譜は round 1 から |
| False | 問わない | あり | **ある** | **`CONFIG_INVALID`**（「study が既にある。続けるなら `resume=True`、新しく始めるなら別の `study_name`」） |
| True | ある | なし、またはその study と同じもの | — | 継続（P1） |
| True | ある | **別の study を指す** | — | **`CONFIG_INVALID`**（今は storage を黙って無視している） |
| True | 無い | あり | ある | 再接続（P2）。保存した系譜と実効空間を復元して継続 |
| True | 無い | あり | 無い | `TUNING_FAILED`（続ける study が無い） |
| True | 無い | なし | — | `TUNING_FAILED`（今と同じ。`Model.load` のあとも含む） |

`storage` があって `study_name` が無ければ `CONFIG_INVALID`（今と同じ）。

## 4. 目的関数の同一性

### 4.1 含めるもの

- **データ**: tune に渡したデータの値と行の順序のハッシュ（`pd.util.hash_pandas_object` で全列。target、group、time の列を含む）。`data.fingerprint.compute` は使わない（`file_path` が無いと行数・列名・dtype しか見ないため）。
- **config**: 次のものの正規化した値。
  - task、data の設定（target、除外する列など）、特徴量の pipeline
  - `model` の全体: `name`、`params`（探索する次元の名前も含む）、smart の値
  - 実効の固定パラメーター（`space_mode` で決まる `fixed`）
  - training の全体: seed、early stopping、inner validation
  - split の全体
  - 最初の評価指標の指定（名前と引数）と、そこから導いた最適化の向き
- **次元の経路**: 系譜に現れたすべての次元の、名前とカテゴリー（model / smart / training）の対応（6.4 の経路に使う）。

### 4.2 含めないもの

- 探索空間（`tuning.optuna.space` / `space_mode` / 範囲・分布）。`space_mode` が変える固定パラメーターは 4.1 に入る。
- `n_trials`、timeout、sampler、`progress_callback`、`expand_boundary`、`boundary_threshold`
- 二つ目以降の評価指標、calibration（tune のスコアは calibration 前の OOF）、export・出力の設定
- storage、study_name、run_id、時刻

### 4.3 照合

- 新しい study を作るとき、同一性を study の属性（7.1）に書く。メモリ上の study にも書く（Widget が `_study` を pickle して別プロセスへ運ぶため、study と一緒に運ばれる必要がある）。
- `resume=True` のたびに、今の呼び出しから同一性を計算し、study のものと比べる。違えば `CONFIG_INVALID`。メッセージには、違う要素（データ、config のどの部分、評価指標）を書く。
- 照合の記録が無い study（この変更より前に作った study）は `CONFIG_INVALID`（D4）。
- 照合は、向きの照合（H-0099 決定 2）と同じ位置で行う（enqueue と最適化の前）。向きは最初の評価指標から導くので、同一性に含まれる。H-0099 決定 2 の検査はそのまま残す。
- **ライブラリの版**（LizyML、LightGBM、Optuna、pandas、scikit-learn）は同一性とは別に記録し、違えば警告だけを出して続ける（D4）。

## 5. 探索空間

### 5.1 範囲を変える方法（**要決定 Q-A**）

今は `Model` の config を公開の方法で変えられない（テストは `model._cfg` を直接書いている）。範囲を変える経路として、次の二つがある。

- **(A-1、推奨) `tune(..., space=...)` の引数**: `tuning.optuna.space` と同じ形式の dict。この round の宣言として使い、study の `space_mode` で解決する。config の `tuning.optuna.space` は、新しい study を作るときだけ読む。
- **(A-2) config を宣言とする**: resume のたびに config の空間を読む。同じ `Model` で範囲を変えるには、config を差し替える公開の方法が新たに要る。storage を使う場合は、編集した config で新しい `Model` を作り、`resume=True` で再接続する。

A-1 を推奨する理由:

- 再接続（P2）では、config をそのまま使って `Model` を作り直す。config が宣言なら、二回目以降の round で変えた範囲が再接続のときに失われる。D3 と矛盾する。
- A-1 なら、config は「study を作ったときの宣言」に固定される。round ごとの変更は引数と系譜に残る。

A-1 での config の扱い: resume のとき、config の探索空間が study を作ったときの宣言と違えば `CONFIG_INVALID`（「resume で範囲を変えるには `space=` を使う」）。黙って無視しない（今の P3）。

### 5.2 resume での空間の決め方

1. **宣言空間**: `space=` があれば、それを study の `space_mode` で解決したもの。無ければ、直前の round の宣言空間。
   - `merge`: provider の既定の次元に `space=` の次元を重ねる。`space=` に無い既定の次元は既定の宣言になる。
   - `replace`: `space=` の次元だけ。`space=` に無い次元は外れる。
   - `space_mode` は resume では変えられない（固定パラメーターが変わり、目的関数が変わるため。4.1）。
2. **次元ごとの比較**: 宣言空間の各次元を、直前の round の宣言と比べる。
   - 宣言が同じ次元: 直前の round の実効の範囲（拡張を含む）を引き継ぐ。
   - 宣言が変わった次元、新しい次元: 新しい宣言の範囲を使う。その次元のこれまでの拡張は捨てる。
   - 宣言から消えた次元: 探索から外す。以後の試行では config の値で学習する（6.1）。
3. **Optuna が受け付けない変更**: 同じ名前の次元で、分布の種類（float と int）、`log` の有無、categorical の選択肢を変えることは、Optuna 4.7.0 が `ValueError` で拒否する（実行で確認）。study を変える前に `CONFIG_INVALID`（「新しい study で」）。範囲の拡大・縮小・移動は受け付ける。
4. **受け付けの検査**: この round の宣言空間に、新しい tune と同じ受け付けの検査を当てる（H-0093、H-0099、H-0102、H-0121）。`space=` の次元は利用者の宣言として扱う。
5. **provider の境界**: 拡張だけを縛る（H-0078、H-0111）。利用者が宣言した範囲は検査もクランプもしない（H-0111 の bound のまま）。

### 5.3 境界の拡張

H-0068、H-0078、H-0111 の規則のまま。変える点:

- この round で宣言が変わった次元、新しい次元は拡張しない（利用者の宣言を優先する）。
- 前回の最良値が今の範囲の外にある次元は拡張しない（範囲を動かしたあと、元の範囲の値が最良である場合。D2 によりその値は最良値として残る）。
- `expand_boundary=None` は、この round の宣言空間に H-0102 の規則を当てて決める（`merge` で利用者の次元が無いときだけ有効）。

## 6. 実行

### 6.1 試行の基のパラメーター

- 新しい study でも resume でも、試行の基は `config ⊕ fixed` とする。前回の最良値は重ねない。
- これにより、どの試行も `config ⊕ fixed ⊕ その試行のサンプル値` で学習したことになる（INV-R2）。外した次元は config の値になる。
- 空間を変えない resume では、今の挙動（前回の最良値を基に重ねる）と結果は同じになる。サンプルした値が基を上書きするためである。

### 6.2 enqueue

- 前回の最良値は、今の実効空間の点であるとき（今の各次元の範囲に入り、分布に合うとき）だけ enqueue する。
- 範囲の外なら enqueue しない。その値は既に study の観測である。Optuna 4.7.0 は範囲外の enqueue 値を、警告を出したうえでそのまま評価する（実行で確認）。
- 今の空間に無い次元の値は enqueue に含めない（Optuna はサンプルしない次元の固定値を黙って捨てる。実行で確認）。

### 6.3 最良値

- study 全体の最良の試行を採用する（今と同じ。D2）。範囲の外の試行や、外した次元を持つ試行も候補になる。
- `best_score` は、その試行のスコア。round をまたいで比べられる（INV-R1 があるため）。

### 6.4 最良値の経路

- 最良の試行のパラメーターを model / smart / training に分けるとき、今の次元ではなく、系譜に記録した次元のカテゴリーを使う。
- 今は、今の空間に無い名前を `model` に回している（`search_space.py:218`）。外した smart の次元（例: `num_leaves_ratio`）が LightGBM に未知のパラメーターとして渡る。これを直す。
- 外した次元を持つ試行が最良なら、`fit` はその試行の値を使う（INV-R2。config の値ではない）。

### 6.5 round と進捗

- round の番号は、系譜の最後の round の次。再接続のあとも続く（INV-R5）。
- 各試行に round 番号を記録する（7.1）。`TuningResult.trials` の round は、その記録から読む。
- 進捗の `current_trial` は、この round で何番目の試行か。再接続でも 1 から数える（P4b の表示の問題が直る）。
- `resume=False` で新しい study を作ったら、系譜は round 1 から始める（今は前の study の round を積み増している）。

## 7. 保存と再接続

### 7.1 study の属性に書くもの

キーは `lizyml:` で始める。メモリ上の study にも同じものを書く。

- `lizyml:schema`: この形式の版
- `lizyml:objective`: 同一性の要約（4.1）と、要素ごとの要約（照合の失敗で違う要素を示すため）
- `lizyml:versions`: ライブラリの版（4.3）
- `lizyml:space_mode`、`lizyml:fixed`: study を作ったときの値
- `lizyml:origin_space`: study を作ったときの config の探索空間（5.1 の比較用）
- `lizyml:round:<k>`: round k の系譜（宣言空間、実効空間、拡張した次元とその結果、`expand_boundary` の値、要求した `n_trials`、開始時の試行数、開始時の最良値）。round を始めるときに書く。
- `lizyml:round:<k>:end`: 終了時の最良値と、終わった時刻。正常に終わったときだけ書く。
- 各試行の `lizyml:round`（試行の user attr）

### 7.2 再接続（P2）

- `resume=True`、メモリ上の study が無い、storage にその study がある → study を読み、4.3 の照合を行う。そのうえで、系譜の最後の round の実効空間を直前の round の空間として 5.2 から続ける。
- 終了の記録が無い round（途中で落ちた round）は、系譜に「中断」として残る。その試行は記録した round 番号のまま。
- 再接続した Model は、系譜から `rounds`、round 番号、実効空間を復元する。

### 7.3 変えないもの

- artifact（`Model.export` / `Model.load`）の形式は変えない。`format_version` も上げない。
- `Model.load` のあとの `resume=True` は、storage が無ければ `TUNING_FAILED` のまま（study の中身を artifact に保存するかどうかは #315 と、まだ起票していない follow-up が扱う）。

## 8. エラーの一覧（新規と変更）

| 条件 | コード |
|---|---|
| `resume=False`、storage に同じ名前の study がある | `CONFIG_INVALID`（新規） |
| `resume=True`、メモリ上の study と別の storage / study_name | `CONFIG_INVALID`（新規） |
| `resume=True`、メモリ上の study が無い、storage にその study が無い | `TUNING_FAILED`（新規） |
| 目的関数の同一性が違う | `CONFIG_INVALID`（新規） |
| study に照合の記録が無い | `CONFIG_INVALID`（新規） |
| resume で `space_mode` を変えた | `CONFIG_INVALID`（新規） |
| resume で config の探索空間を変えた（A-1 の場合） | `CONFIG_INVALID`（新規） |
| 分布の種類・`log`・categorical の選択肢の変更 | `CONFIG_INVALID`（新規。今は Optuna の `ValueError` が試行の中で起きる） |
| ライブラリの版の違い | 警告（新規） |

## 9. 過去の記録との照合

### 9.1 この仕様が置き換えるもの（Proposal で明示する）

| 過去の記録 | 内容 | この仕様 |
|---|---|---|
| H-0102（HISTORY.md:9963-9964）、BLUEPRINT.md:882-886 | resume は保存した空間・境界・固定の方針を使い、config の変更は空間に入らない。resume は前回の tuning の overlay を使う | 空間は `space=` で変えられる（5.1）。config の変更は黙って無視せず拒否する。固定の方針を引き継ぐ点はそのまま。試行の基に前回の最良値は重ねない（6.1） |
| H-0102（HISTORY.md:9981-9982）、BLUEPRINT.md:879、CHANGELOG.md:61 | 空間の意味を変えるなら新しい study。古い試行は新しい空間の観測ではない | 目的関数が同じなら、古い試行は観測である（INV-R3）。新しい study が要るのは、Optuna が受け付けない変更（5.2-3）と `space_mode` の変更だけ |
| H-0072（HISTORY.md:5804、INV-3 :5769）、BLUEPRINT.md:1101、CHANGELOG.md:346、docs/api.md:100-101 | `resume=False` と storage で、既存の study に再接続する | 再接続は `resume=True`（D5）。`resume=False` で既存の study があれば拒否 |
| H-0072（HISTORY.md:5736） | `resume=True`、study が無い、storage あり → 自動で再接続 | 採用する（D5）。BLUEPRINT.md:1083・1101-1102 との矛盾がこれで解ける |
| BLUEPRINT.md:1083、docs/api.md:95 | `resume=True` で tune 前なら `TUNING_FAILED` | storage で既存の study を指すときは再接続。それ以外は `TUNING_FAILED` のまま |
| H-0072（HISTORY.md:5740）、BLUEPRINT.md:1108 | 再接続後の round は 1 から。round の記録は journal に保存しない（別の Proposal で扱う） | round の系譜を study の属性に保存し、再接続で続ける（7.1、7.2）。この Proposal が「別の Proposal」にあたる |
| H-0099 決定 2（HISTORY.md:9879-9880）、BLUEPRINT.md:1109 | study について照合するのは向きだけ | 向きの照合は残し、目的関数の同一性の照合を加える（4.3） |
| H-0121（Proposed）の resume の条項（HISTORY.md:11973、11977-11978、受け入れ基準 6 :12045） | resume で今の config の smart の値を読み、無効な既定の次元を外す | H-0121 から外す（A1）。smart の値の変更は目的関数の変更として 4.3 で拒否される |

### 9.2 前提を明示するもの（矛盾しない）

| 過去の記録 | 関係 |
|---|---|
| H-0068 受け入れ基準 2（HISTORY.md:5326）「best_score が悪化しない」、BLUEPRINT.md:1065-1067 の累積の最良線 | INV-R1 で前提が保証される |
| H-0068（HISTORY.md:5082、5085）study を保持して追加試行、「まだ見ていない有望領域を探す」 | P1 そのもの。拡張の規則は 5.3 の 2 点を足すだけ |
| H-0078、H-0111 の境界と no-op の拡張 | 変えない（5.2-5、5.3） |
| H-0072 INV-5（別の storage の試行と混ざらない）、INV-6（進捗と round は storage の有無に依らない） | INV-R5 で強める |
| H-0086、H-0109、H-0094 決定 13・14、H-0120 改訂 1 | artifact と報告面は変えない（7.3）。tune の失敗は状態を変えない（INV-R4） |
| H-0099 決定 1・3・4、H-0093、H-0102 の受け付けの検査 | 各 round の宣言空間に当てる（5.2-4） |
| BLUEPRINT.md:448（`Tuner` は study の管理だけ） | 属性の読み書きは Tuner、同一性と系譜の中身は Model 側で作る（実装で守る） |
| skills/tuning SKILL.md:10、16（分割と seed をそろえ、比べられるようにする） | 同一性に分割と seed を含めることで守る |

### 9.3 この仕様と関係しない、既にある記録どうしのずれ（別の docs の issue にする）

- C1 一回の round の `rounds`（BLUEPRINT.md:1045、HISTORY.md:5202 と docs/api.md:401）
- C2 `boundary_threshold` の範囲（BLUEPRINT.md:993 と docs/api.md:98）
- C3 `direction` の既定（docs/config-reference.md:413、BLUEPRINT.md:239）
- C4 `expand_boundary=None`（BLUEPRINT.md:992 と :893-895）
- C7 `boundary_report` を設定する条件（BLUEPRINT.md:1046 と docs/api.md:402）
- C9 docs/config-reference.md に `space_mode` が無い
- C10 load のあとの `boundary_table` のメッセージが、使えない `tune(resume=True)` を案内する（HISTORY.md:10684）
- C11 `current_trial` の意味（BLUEPRINT.md:948 と docs/api.md:456）。6.5 で「この round の何番目」と決める
- C12 docs/api.md:395-396 の例の名前

C11 だけはこの仕様が決める。

## 10. 利用側への影響

- **LizyStudio**: 障害からの再開（`services/training.py:440-449`）は、新しい `Model` で `resume` を付けずに `storage` と `study_name` を渡している。この仕様では `CONFIG_INVALID` になる。`resume=True` を渡すよう直す必要がある。v0.18 以前に作った study は照合の記録が無いので、再接続できない（D4）。
- **LizyStudio の retune**（`training_retune.py:86-105`）: Model 全体を cloudpickle したものを読み、`resume=True` を呼ぶ。新しい版で作った checkpoint なら study の属性ごと運ばれる。古い版の checkpoint は照合の記録が無いので拒否される。
- **LizyML-Widget**（`adapter.py:767-800`）: `_study`、`_space`、`_used_default_space`、`_round_number`、`_rounds` を pickle して別プロセスで戻している。同一性と系譜を study の属性に置くので、`_study` が運ばれれば足りる。`_study` を落とした場合（RDB）は、再接続（7.2）の経路を使う必要がある。
- 公開 API: `tune()` に `space=` を足す（A-1 の場合）。エラーの条件が増える（8）。CHANGELOG と migration に書く。

## 11. 決める必要があること

- **Q-A**: 範囲を変える方法（5.1）。A-1（`space=` の引数）を推奨。
- **Q-B**: 外した次元の扱い（6.1、6.4）。最良の試行がその次元を持てば、その値で `fit` する。以後の試行は config の値で学習する。前回の会話で示した扱いのとおり。
- **Q-C**: `resume=False` で既存の study を指したときの拒否（3）。LizyStudio の修正が要る。

## 12. 受け入れ基準（テストの観点）

1. 目的関数の要素（データの値、データの行の順序、config の各部分、最初の評価指標）を一つずつ変えて `resume=True` → `CONFIG_INVALID`。study の試行数と Model の状態が変わらない。含めないもの（4.2）を変えても通る。
2. 照合の記録が無い study（属性を消した study）に `resume=True` → `CONFIG_INVALID`。
3. ライブラリの版の記録を書き換えて `resume=True` → 警告が出て続く。
4. `space=` で範囲を動かして resume → 新しい試行は新しい範囲。最良が古い範囲の試行なら、その値が `best_params` になり、`fit` がそれを使う。その次元は拡張されない。
5. 宣言を変えない resume → 拡張した範囲が引き継がれる（今のテストと同じ）。
6. 次元を外して resume → 以後の試行はその次元を config の値で学習する。外した smart の次元が最良の試行にあれば、smart として `fit` に渡る（LightGBM の params に入らない）。
7. 分布の種類・`log`・選択肢の変更、`space_mode` の変更 → study を変える前に `CONFIG_INVALID`。
8. storage で round 2 の途中に落とした（プロセスを止めた）あと、新しい `Model` で `resume=True` → round 2 の実効空間から round 3 として続き、`rounds` と試行の round が同じプロセスで続けた場合と同じ。`current_trial` は 1 から。
9. `resume=False` で既存の study → `CONFIG_INVALID`、試行が増えない。
10. `resume=True` で別の storage → `CONFIG_INVALID`。
11. config の探索空間を変えて `resume=True`（`space=` なし） → `CONFIG_INVALID`。
12. 既存のテストのうち今の挙動を固定しているもの（`test_space_merge.py:183, 262, 305, 375`、`test_tuner_persistence.py:252`、`test_retune.py:807`、`test_error_code_raising.py:57`）は、新しい規則を固定するように書き直す（消さない）。

Firing rate: Proposal を書く段で測る（テストのうち resume と再接続を呼ぶものと、`design-vs-implementation.md` の probe P1-P7 を再生し、拒否の数を数える）。
