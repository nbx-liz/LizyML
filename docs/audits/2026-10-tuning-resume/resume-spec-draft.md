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
- **INV-R4（拒否は study を変える前）**: 拒否は、試行の追加と study の属性の書き込みより前に行い、study と Model の状態を変えない。
- **INV-R5（再接続は継続と同じ）**: 落ちたあとに再接続した結果は、同じプロセスで続けた場合と同じ空間・系譜・round 番号になる。`timeout` で区切る round の試行の数だけは、この保証の外（7.2）。

## 3. 呼び出しの種類

| `resume` | メモリ上の study | `storage` / `study_name` | storage にその study | 挙動 |
|---|---|---|---|---|
| False | 問わない | なし | — | 新しい study（メモリ）。系譜は round 1 から |
| False | 問わない | あり | 無い | 新しい study（storage）。系譜は round 1 から |
| False | 問わない | あり | **ある** | **`CONFIG_INVALID`**（「study が既にある。続けるなら `resume=True`、新しく始めるなら別の `study_name`」） |
| True | ある | なし、またはその study と同じもの | — | 継続（P1） |
| True | ある | **その study と同じと示せないもの** | — | **`CONFIG_INVALID`**（今は storage を黙って無視している） |
| True | 無い | あり | ある | 再接続（P2）。保存した系譜と実効空間を復元して継続 |
| True | 無い | あり | 無い | 新しい study（storage）を作り、round 1 から始める（U3）。作ったことを INFO のログに出す。`resume=False` で作った場合と同じ（config の探索空間を使い、この呼び出しで `space=` を渡したら `CONFIG_INVALID`） |
| True | 無い | なし | — | `TUNING_FAILED`（今と同じ。`Model.load` のあとも含む） |

表の前に、次の二つを先に検査する（どちらも `CONFIG_INVALID`、`resume` の値に依らない）:

- `storage` があって `study_name` が無い（今は `resume=True` で study が無いとき、この検査より先に `TUNING_FAILED` になる。順序を変える）。
- `study_name` があって `storage` が無い（今は黙って無視している）。

**「その study と同じ」の判定**: Model は、メモリ上の study を作った・読んだときの `storage` と `study_name` を覚えておく。`resume=True` で渡した値が、`study_name` が等しく、かつ `storage` が等しい（URL の文字列なら文字列として等しい、`BaseStorage` なら同じオブジェクト）ときだけ同じとする。同じ backend を指す別の表現は同じと示せないので拒否する（安全側）。

**同時実行**: 一つの study に試行を足すプロセスは一つとする（今の暗黙の前提を明文化する）。

## 4. 目的関数の同一性

**照合の厳しさ（Q-E、決定済み）**: 目的関数を変えない変更まで正確に見分ける「厳密な同一性」と、目的関数を変えうる入力を広めに含め、無害な変更も拒否する「保守的な互換性」がある。保守的な互換性を採る。

- 拒否しすぎても、結果は「新しい study で」という明示のエラーで、黙って間違うことはない。
- 厳密にするには、たとえば「探索している次元の config の値は効かない」と判定する必要がある。ところが、次元を外したあとの試行はその値で学習する（6.1）。正規化を誤ると黙って間違う。

以下は保守的な互換性の内容。

### 4.1 含めるもの

- **データ**: tune が実際に使う DataFrame の正規化した要約。DataFrame は今の優先順位（`tune(data=...)` > コンストラクターの `data` > config の `data.path` を読んだもの、`model.py:712-719`）で決めたもので、どの経路でも読んだ後の中身を要約する（`data.path` の文字列だけでは、同じ場所のファイルの中身の変更を見落とすため）。次の全部を含める。
  - 列名と列の順序
  - 各列の dtype
  - categorical の列の categories とその順序、ordered かどうか
  - 値と行の順序: `pd.util.hash_pandas_object(df, index=True)`。これは列名も categories も見ず、型の違う同じ表記の値（`1` と `"1"`）を区別しない（実行で確認）ので、次の二つを足す。
    - categorical の列: 型付きの categories（各値の型の名前と値）と、各行の codes のハッシュ（`hash_pandas_object(series.cat.codes)`）。categorical の `[1, "1"]` と `["1", 1]` は codes で区別される。
    - object の列: 各行の値の型の名前の列のハッシュ。
    - どれも列ごとのベクトル演算か、object の列だけの型の取り出しで、セルごとの符号化はしない（大きなデータで resume を遅くしないため）。
  - `data.fingerprint.compute` は使わない（`file_path` が無いと行数・列名・dtype しか見ないため）。
  - 除外する列（`features.exclude`）も含める（保守的）。
- **config**: 次のものの正規化した値。
  - task、data の設定（target、除外する列など）、特徴量の pipeline
  - `model` の全体: `name`、`params`（探索している次元の値も含む。保守的）、smart の値
  - 実効の固定パラメーター（`space_mode` で決まる `fixed`）
  - training の全体: seed（TPE の seed も兼ねる）、early stopping、inner validation
  - split の全体
  - 最初の評価指標の指定（名前と引数）と、そこから導いた最適化の向き
- 重み（`balanced` など）は model と smart の設定に含まれる。

### 4.1b 次元の経路（同一性とは別の規則）

- 系譜は、次元の正規化した名前（model の次元は alias を正規名に寄せたもの、H-0094 決定 7）と、そのカテゴリー（model / smart / training）の対応を持つ。
- この対応は**足すだけ**: 新しい名前は、宣言されたときに経路を得る。外した次元の経路も残る（古い試行の値を 6.4 で回すため）。
- 既にある名前のカテゴリーを変える宣言は `CONFIG_INVALID`（消費する側が変わり、目的関数が変わるため）。
- 同一性の照合（4.3）にはこの対応を入れない。次元の追加と削除は 5.2 で受け付ける。

### 4.2 含めないもの

- 探索空間（`tuning.optuna.space` / `space_mode` / 範囲・分布）。`space_mode` が変える固定パラメーターは 4.1 に入る。
- `n_trials`、timeout、sampler、`progress_callback`、`expand_boundary`、`boundary_threshold`
- 二つ目以降の評価指標、calibration（tune のスコアは calibration 前の OOF）、export・出力の設定
- storage、study_name、run_id、時刻

### 4.3 照合

- 新しい study を作るとき、同一性を study の属性（7.1）に書く。メモリ上の study にも書く（Widget が `_study` を pickle して別プロセスへ運ぶため、study と一緒に運ばれる必要がある）。
- `resume=True` のたびに、今の呼び出しから同一性を計算し、study のものと比べる。違えば `CONFIG_INVALID`。メッセージには、違う要素（データ、config のどの部分、評価指標）を書く。
- 照合の記録が無い study（この変更より前に作った study）は `CONFIG_INVALID`（D4）。ただし試行が 0 件で記録の無い study は、作りかけとして扱う（7.4 の状態 E）。
- 照合は、向きの照合（H-0099 決定 2）と同じ位置で行う（試行を足す前）。向きは最初の評価指標から導くので、同一性に含まれる。H-0099 決定 2 の検査はそのまま残す。
- **ライブラリの版**（LizyML、LightGBM、Optuna、pandas、scikit-learn）は同一性とは別に記録し、違えば警告だけを出して続ける（D4）。ただし、版の更新で同一性に入る値そのもの（たとえば provider が決める固定パラメーター `fixed` の既定値）が変われば、それは目的関数の変更なので `CONFIG_INVALID` になる。D4 の「警告だけ」は、版の番号の違いだけの場合を指す。

## 5. 探索空間

### 5.1 範囲を変える方法（Q-A、決定済み: A-1）

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
3. **同じ study で受け付けない変更**: 同じ名前の次元で、分布の種類（float と int）、`log` の有無、categorical の選択肢を変えること。LizyML が study を変える前に自分で比べ、`CONFIG_INVALID`（「新しい study で」）にする。
   - categorical の選択肢は、並びと、各値の型と値で比べる。値の等しさは Optuna と同じ（`optuna/distributions.py` の `_categorical_choice_equal`: NaN どうしは等しい）で、それに型の一致を足す。Optuna は型を見ないので `True` と `1`、`1` と `1.0` の入れ替えを通す（実行で確認）。LizyML は bool・int・float の選択肢を受け付ける（`search_space.py:37, 105-119`）。
   - LizyML が先に拒否するのは、Optuna の拒否が試行の途中で起き、FAIL の試行を書いてしまう（INV-R4 に反する）ため。
   - 範囲（`low` / `high`）の変更は受け付ける。LizyML の探索空間には `step` が無い（`search_dim.py:20-52`、`parse_space` は読まない）ので、この仕様も `step` を扱わない。
4. **受け付けの検査**: この round の宣言空間に、新しい tune と同じ受け付けの検査を当てる（H-0093、H-0099、H-0102、H-0121）。`space=` の次元は利用者の宣言として扱う。今は名前・重複・training の所有の検査が config の空間を読んでいる（`_model_tuning.py:216, 221, 228`）ので、round の宣言空間を読むように変える（そうしないと `space=` の次元が検査を素通りする）。H-0121 はこの仕様より先に実装し、その「既定の次元で無効なものを外す」規則は、宣言空間を解決するときに当てる。
5. **provider の境界**: 拡張だけを縛る（H-0078、H-0111）。利用者が宣言した範囲は検査もクランプもしない（H-0111 の bound のまま）。

### 5.3 境界の拡張

H-0068、H-0078、H-0111 の規則のまま。変える点:

- この round で宣言が変わった次元、新しい次元は拡張しない（利用者の宣言を優先する）。
- 前回の最良値が今の範囲の外にある次元は拡張しない（範囲を動かしたあと、元の範囲の値が最良である場合。D2 によりその値は最良値として残る）。
- 「前回の最良値」（境界の拡張の判断に使う）は、study の全試行のうち最良の COMPLETE の試行の値を、4.1b の経路で分けたもの。Model が持つ `_tuning_result` は使わない（再接続した Model では無く、`Model.load` のあとでは別の study の結果でありうる）。固定パラメーター（今の `_tuning_fixed_params`）も `lizyml:study` の `fixed` から読む。study の属性と Model のフィールドが食い違えば、study の属性を正とする。
- `expand_boundary=None` は、この round の宣言空間に H-0102 の規則を当てて決める（`merge` で利用者の次元が無いときだけ有効）。

## 6. 実行

### 6.1 試行の基のパラメーター

- 新しい study でも resume でも、試行の基は `config ⊕ fixed` とする。前回の最良値は重ねない。
- これにより、どの試行も `config ⊕ fixed ⊕ その試行のサンプル値` で学習したことになる（INV-R2）。外した次元は config の値になる。
- 空間を変えない resume では、今の挙動（前回の最良値を基に重ねる）と結果は同じになる。サンプルした値が基を上書きするためである。

### 6.2 前回の最良値を評価し直さない（enqueue をやめる）

今は resume の round の最初に、study に保存済みの最良の試行のパラメーターを `study.enqueue_trial` で予約し、同じ点をもう一度学習・評価している（`_model_tuning.py:606-608`、`tuner.py:174-175`。H-0068）。この案ではこれをやめる（Q-F、2026-10-11 に管理者が変更）。

- 最良の試行は study に COMPLETE の観測として残っており、TPE は既にそれを学習に使っている。目的関数は同じで（INV-R1）、seed も固定なので、評価し直しても同じ点の測り直しになるだけで、round の試行が 1 件減る。
- 最良値の採用（6.3）と `best_score` は変わらない。
- resume の各 round は、要求した `n_trials` の数だけ TPE でサンプルした点を評価する。

### 6.3 最良値

- study 全体の最良の試行を採用する（今と同じ。D2）。範囲の外の試行や、外した次元を持つ試行も候補になる。
- `best_score` は、その試行のスコア。round をまたいで比べられる（INV-R1 があるため）。

### 6.4 最良値の経路

- 最良の試行のパラメーターを model / smart / training に分けるとき、今の次元ではなく、系譜の経路（4.1b）を使う。
- 今は、今の空間に無い名前を `model` に回している（`search_space.py:218`）。外した smart の次元（例: `num_leaves_ratio`）が LightGBM に未知のパラメーターとして渡る。これを直す。
- 外した次元を持つ試行が最良なら、`fit` はその試行の値を使う（INV-R2。config の値ではない）。

### 6.5 round と進捗

- round の番号は、系譜の最後の round の次。再接続のあとも続く（INV-R5）。
- 試行の round は、試行の番号と round の記録から決める（7.1）。`TuningResult.trials` の round はそれを使う。
- 報告面の round の区切り（`rounds.py:60-80` の round の付け直し、`plots/tuning.py:90-118` の区切り線）は、今は `RoundSummary.n_trials` を積み上げて求めている。`failed` の round が `rounds` に入らないこと、時間で区切る round と途中から再開した round では要求数と実際の数が違うことから、試行の round（上）から求めるように変える。
- `RoundSummary.n_trials` は、その round の試行の実際の数。途中から再開した round の `RoundSummary` と `boundary_report` は、その round の記録（始めたときの判断）から作る。宣言を変えた次元と、最良値が範囲の外の次元は、`boundary_table` に「拡張しない」（`expanded=False`）として出る。
- 進捗の `cumulative_trials` は今と同じく study 全体の試行の数（`trial.number + 1`）、`round` はその試行の round。
- 進捗の `current_trial` は、この round で何番目の試行か（`total_trials` はこの round の `n_trials`）。新しい round は 1 から数える。途中で落ちた round を再開したときは、その round で既に終わった試行が K 件なら K+1 から数える（storage の有無に依らない。H-0072 INV-6 と、HISTORY.md:5805-5806 の `current_trial` の一致）。今の再接続で `current_trial` が `n_trials` を超える表示（P4b）も直る。
- `resume=False` で新しい study を作ったら、系譜は round 1 から始める（今は前の study の round を積み増している）。

## 7. 保存と再接続

### 7.1 study の属性に書くもの

キーは `lizyml:` で始める。メモリ上の study にも同じものを書く。Optuna の `set_user_attr` は一つのキーを一度に書くだけなので、一緒に書くべきものは一つのキーの値にまとめ、書き込みの途中で落ちても読み手が判断できるようにする。

- `lizyml:study`: study を作ったときに一度だけ書く。中身:
  - `schema`: この形式の版。読めない版（より新しい LizyML が書いたもの）なら `CONFIG_INVALID`。
  - `objective`: 同一性の要約（4.1）と、要素ごとの要約（照合の失敗で違う要素を示すため）
  - `versions`: ライブラリの版（4.3）
  - `space_mode`、`fixed`: study を作ったときの値
  - `origin_space`: study を作ったときの config の探索空間（5.1 の比較用）
- 次元の経路（4.1b）は独立したキーに置かず、各 round の記録（`lizyml:round:<k>`）に、その round で増えた経路として入れる。経路の全体は、記録のある round の経路を合わせたもの。round の記録と一緒に書かれるので、落ちたときに経路だけが残ったり欠けたりしない。
- `lizyml:round:<k>`: round k の系譜。round の準備（5.2・5.3・6.2 の判断）がすべて済んでから、試行を足す前に一度だけ書く。中身:
  - その round の宣言の元の形（`space=` の dict、無ければ「直前の宣言のまま」）と、次元ごとの出どころ（利用者 / 既定）。`expand_boundary=None` の判断（5.3）と H-0121 の出どころの規則を、再接続のあとも同じに決めるため。
  - 宣言空間、実効空間、拡張した次元とその結果、解決した `expand_boundary` の値
  - 予算: 要求した `n_trials` と `timeout`
  - 開始時の試行数（その round の試行は、これ以上の番号）、開始時の最良値
  - その round で増えた次元の経路（4.1b）
- `lizyml:round:<k>:end`: round を閉じたときに書く。中身は、終わり方（`completed` / `failed`）、終了時の最良値（study に COMPLETE の試行が一つも無ければ空）、終わった時刻。
  - round の試行を足し終えて、study に COMPLETE の試行が一つ以上あれば `completed`。
  - study に COMPLETE の試行が一つも無いまま round が終われば `failed` と書いたうえで、今と同じく `TUNING_FAILED` を送出する（Model の状態は変えない）。閉じた round は再開の対象にならず、次の `resume=True` は新しい round を始める。前回の最良値が無いので、境界の拡張は行わない（7.4 の状態 F）。
- **試行がどの round に属するかは、試行の番号で決める**: round k の試行は、round k の記録の「開始時の試行数」以上で、round k+1 の記録の「開始時の試行数」未満の番号を持つもの（最後の round は上限なし）。試行ごとに round を書く属性は置かない（番号から決まるので、写しを持つと食い違いの扱いが要るため）。

**study を作るとき**: (a) study を作る、(b) `lizyml:study` を書く。(a) と (b) の間で落ちると、試行が 0 件で `lizyml:study` の無い study が残る。再接続でこれを見つけたら、まだ何の観測も無いので、作りかけの study として (b) からやり直す（新しい study を作ったのと同じ）。試行が 1 件以上あって `lizyml:study` が無い study は、この変更より前に作った study として拒否する（4.3）。

**書く順序**: (1) round の準備をすべて決める、(2) `lizyml:round:<k>` を書く、(3) 試行を足す、(4) `lizyml:round:<k>:end` を書く。(2) より前に落ちたら、study には round k の跡が何も無い。再開は、round k を最初から始める普通の resume になる（何も足されていないので、続けた場合と同じ）。

### 7.2 再接続（P2）

`resume=True`、メモリ上の study が無い、storage にその study がある → study を読み、7.4 の表で状態を判定する。以下は状態 C・F・O の詳細。

- **最後の round が終わっている**（`:end` がある）: 同じプロセスで `resume=True` を呼んだ場合と同じ。最後の round の実効空間を直前の空間として、5.2 から新しい round を始める。
- **最後の round が途中で落ちている**（`:end` が無い）: INV-R5 に従い、**同じ round を残りの予算で終える**（Q-D、決定済み）。
  - round 番号は変えない。その round の記録にある実効空間をそのまま使う。境界の分析と拡張はやり直さない（記録に決まっている）。
  - 残りの試行数は、記録した `n_trials` から、その round で終わった試行（COMPLETE、FAIL、PRUNED）の数を引いたもの。
  - 記録に `timeout` があるとき（意味は Optuna と同じで、新しい試行を始めない期限。走っている試行は止めない）、残りの時間は `timeout` から、その round の COMPLETE と PRUNED の試行それぞれの「開始から終了まで」（Optuna が試行ごとに記録する `datetime_start` / `datetime_complete`）の時間の合計を引いたもの。FAIL の試行は数えない: 再接続で RUNNING から FAIL にした試行は、Optuna が終了の時刻を FAIL にした時点で入れる（`datetime_complete` が落ちていた間を含む）ため。目的関数の中で失敗した試行の時間も数えないことになるが、時間の予算は近似で、再開した round は続けた場合より長く走りうる（下の限界）。こうして、何度落ちても、落ちていた間の時間は入らない。
  - 残りの試行数か残りの時間が 0 以下なら、試行を足さずに round を閉じる（最適化は試行数か時間で終わったが、`:end` を書く前に落ちた場合を含む）。閉じ方は 7.1 の `:end` の規則に従う。たとえば `n_trials=1` の round 1 で、ただ一つの試行が RUNNING のまま落ちた場合は、その試行を FAIL にして `failed` で閉じ、`TUNING_FAILED` を送出する。
  - 途中の round の予算は記録が決める。config の `tuning.optuna.params.n_trials` は、途中の round の再開では使わない（新しい round にだけ使う）。config の値が記録と違えば、無視したことを警告に出す。
  - 時間による予算の限界: 残りの時間は試行の記録から求めるので、落ちる直前に走っていた試行の時間と、試行と試行の間の時間（サンプリングなど）は数えない。そのぶん、再開した round は続けた場合より長く走りうる。時間で区切る round では、続けた場合と同じ試行の数になることは保証しない（時間による打ち切りは、同じプロセスでも試行の数を決めないため）。試行の数による予算では INV-R5 を満たす。
  - 落ちたときに RUNNING のまま残った試行は、再接続のときに FAIL として記録し、予算に数える。一つの study に試行を足すプロセスは一つという前提（3）による。
  - 検査の順序: 4.3 の照合と、下の引数（U2）の比べ合わせを、RUNNING の試行を FAIL にすることを含むすべての書き込みより先に行う（INV-R4）。拒否したときは、試行の状態も study の属性も変わらない。
  - 引数（U2）: 途中の round の空間と予算は記録が決めている。次の引数は、値が `None` 以外のときだけ、その round の記録の**解決した値**と比べる。同じなら通し、違えば `CONFIG_INVALID`（「この round は記録の値で始まっている」）。`None`（三つとも既定値）は「渡していない」とみなして比べない（今のシグネチャでは省略と `None` を区別できないため、区別しない）。
    - `n_trials`（int）: 記録した、その round で要求した試行数と比べる。
    - `expand_boundary`（bool）: 記録した、その round で解決した `expand_boundary` の値と比べる。
    - `space=`（dict）: 5.2-1 と同じ手順で、study の `space_mode` に従って宣言空間に解決し、記録したその round の宣言空間と比べる。比べ方は、解決した次元（`parse_space` が返す `SearchDim`）の集合として等しいこと。次元はカテゴリーと正規名（model の次元は alias を寄せた名前）で対応させ、分布の種類・`low`・`high`・`log` は解決した値で、categorical の選択肢は 5.2-3 の比べ方（並び、型、NaN どうしは等しい）で比べる。解決の段階で既定値が入るので、省略の書き方や dict の並びの違いは同じとみなす。同じスクリプトを流し直す場合は同じ値が渡るので通る。この規則は途中の round の再開だけに当てる。終わった round のあとの `resume=True`（新しい round、再探索）では、これらの引数は自由に変えられる。
  - 代わりの案: 中断した round を閉じ、次の round として始める（今の LizyStudio は残りの試行数を自分で数えて渡しており、こちらに近い）。INV-R5 を満たさないため推奨しない。
- 再接続した Model は、系譜から `rounds`、round 番号、実効空間を復元する。
- `failed` で閉じた round は、系譜には残るが `TuningResult.rounds` には入れない（`RoundSummary` は最良値を数値で持つ型で、失敗した round を正しく表せないため。公開の型は変えない）。round の番号はそのまま続くので、`rounds` が round 2 から始まることがある。その round の試行は `TuningResult.trials` に、記録した round 番号で残る。次の round の `best_score_before` は `None`。これを docs/api.md の `rounds` の説明に書く。
- 同じプロセスで round が例外で止まったあとの `resume=True` にも、同じ規則を当てる。ただし、その study が Model に既に載っている（前に成功した tune の `_study`）か、storage にある場合に限る。失敗した tune は Model の状態を変えない（H-0094 決定 14）が、載っている study のオブジェクトや storage の study には、途中の試行と `:end` の無い round が残るため。
- storage を使わない最初の tune（Model に `_study` がまだ無い）が失敗した場合、作りかけのメモリ上の study は捨てる（Model の状態を変えないため、`_study` に載せない）。続ける study は無いので、次は `resume=False` で新しく始める（`resume=True` は 3 の表のとおり `TUNING_FAILED`）。

### 7.4 study の状態と、`resume=True` での扱い（閉じた表）

study の状態の分け方と、`resume=True` での扱いは、この表を正とする。7.1・7.2 はそれぞれの扱いの詳細を定める。メモリ上の study（同じプロセスで round が例外で止まった場合）も、storage から読んだ study も、同じ表で扱う。

**落ちる位置と、study に残る状態**（7.1 の書く順序の各段の間）:

| 落ちる位置 | 残る状態 |
|---|---|
| study を作った後、`lizyml:study` を書く前 | E |
| `lizyml:study` を書いた後、round 1 の記録を書く前 | N |
| round k（k ≥ 2）の準備中、`lizyml:round:<k>` を書く前 | 直前の round k−1 の状態（C または F）のまま |
| `lizyml:round:<k>` を書いた後、試行を足している途中 | O |
| 試行を足し終えた後、`:end` を書く前 | O（残りの予算が 0） |

**状態と扱い**（上から順に判定し、最初に当てはまった行を使う）:

| 状態 | 判定の材料 | `resume=True` での扱い |
|---|---|---|
| L（旧 study） | `lizyml:study` が無く、試行が 1 件以上 | `CONFIG_INVALID`（D4、4.3） |
| E（作りかけ） | `lizyml:study` が無く、試行が 0 件 | N と同じ扱い。ただし N の行の検査（`space=` の拒否、受け付けの検査）をすべて済ませてから `lizyml:study` を書く（INV-R4: 拒否は書き込みの前） |
| S（読めない） | `lizyml:study` の `schema` が読めない | `CONFIG_INVALID`（7.1） |
| N（round が無い） | round の記録が一つも無く、試行が 0 件 | 4.3 の照合の後、round 1 を新しい study と同じように始める（config の探索空間を使い、`space=` を渡したら `CONFIG_INVALID`） |
| X（壊れている） | round の記録が無いのに試行がある。round の番号が 1 から連続しない。`:end` の無い round が最後以外にある。試行の番号が round 1 の「開始時の試行数」より小さい | `CONFIG_INVALID`（「LizyML の記録と試行が食い違っている。新しい study で」） |
| O（開いた round） | 最後の round に `:end` が無い | 4.3 の照合の後、7.2 のとおり同じ round を終える。残りの予算が 0 なら試行を足さずに閉じる（7.1 の `:end` の規則: COMPLETE の試行があれば `completed`、無ければ `failed` と `TUNING_FAILED`） |
| C（完了） | 最後の round の `:end` が `completed` | 4.3 の照合の後、新しい round を始める（5.2 から） |
| F（失敗） | 最後の round の `:end` が `failed` | 4.3 の照合の後、新しい round を始める。study に COMPLETE の試行が無いので、境界の拡張は行わない |

`resume=False` で storage に同じ名前の study があるときは、状態に依らず `CONFIG_INVALID`（3）。

### 7.6 系譜を読む公開の関数（Q-G）

利用側が、試行を足さずに study の状態と系譜を知るための、読み取りだけの関数を足す。

- `lizyml.tuning.read_study_lineage(storage, study_name) -> StudyLineage`。`storage` と `study_name` の形は `Model.tune` と同じ（3 の事前検査も同じ）。
- 返す値 `StudyLineage`（frozen dataclass）:
  - `state`: 7.4 の状態を表す文字列。`"absent"`（storage にその study が無い）、`"legacy"`（L）、`"uninitialized"`（E）、`"unreadable"`（S）、`"no_round"`（N）、`"inconsistent"`（X）、`"open"`（O）、`"completed"`（C）、`"failed"`（F）のどれか。判定は 7.4 の表と同じ順序で行う。
  - `rounds`: 記録のある round ごとの要約の並び（`RoundRecord`: `round`、`status`（`"completed"` / `"failed"` / `"open"`）、要求した `n_trials`、終わった試行の数、`best_score_before`、`best_score_after`（`failed` と `open` では `None`））。`failed` の round も含む（`TuningResult.rounds` と違い、系譜をそのまま見せるため）。状態が `absent`・`legacy`・`uninitialized`・`unreadable`・`inconsistent` のときは空。
  - `open_round`: 開いた round の番号（状態 O のとき）、それ以外は `None`。
- 何も書かない: 属性を書かず、RUNNING の試行も FAIL にしない（その扱いは `resume=True` の呼び出しだけが行う、7.2）。目的関数の同一性の照合もしない（データも config も受け取らないため。照合は `resume=True` で行う）。
- 置き場所は `lizyml.tuning`（`Tuner` と同じ層）。`lizyml.tuning.__all__` に足し、BLUEPRINT §11 と docs/api.md に書く。トップレベルの `lizyml.__all__`（`test_public_exports.py` が固定）には入れない。

### 7.5 変えないもの

- artifact（`Model.export` / `Model.load`）の形式は変えない。`format_version` も上げない。
- `Model.load` のあとの `resume=True` は、storage が無ければ `TUNING_FAILED` のまま（study の中身を artifact に保存するかどうかは #315 と、まだ起票していない follow-up が扱う）。

## 8. エラーの一覧（新規と変更）

この表は規則の索引で、規則の正は「規則」の列の節にある。各行に、違反すると落ちる受け入れ基準（12）を対応させる。

| 条件 | コード | 規則 | 受け入れ基準 |
|---|---|---|---|
| `storage` があって `study_name` が無い | `CONFIG_INVALID`（`resume` に依らず先に検査する。今は `resume=True` で study が無いと `TUNING_FAILED`） | 3 | 12 |
| `study_name` があって `storage` が無い | `CONFIG_INVALID`（新規。今は黙って無視） | 3 | 12 |
| `resume=False`、storage に同じ名前の study がある | `CONFIG_INVALID`（新規） | 3、7.4 | 9 |
| `resume=True`、メモリ上の study と同じと示せない storage / study_name | `CONFIG_INVALID`（新規） | 3 | 10、14 |
| 新しい study を作る呼び出し（`resume=False`、U3 の `resume=True`、状態 N）で `space=` を渡した | `CONFIG_INVALID`（新規。初回の空間は config で宣言する） | 3、7.4 | 14、19 |
| `resume=True`、メモリ上の study も `storage` も無い | `TUNING_FAILED`（今と同じ） | 3 | 14 |
| 目的関数の同一性が違う | `CONFIG_INVALID`（新規） | 4.3 | 1、13 |
| 照合の記録が無く、試行がある study（状態 L） | `CONFIG_INVALID`（新規） | 4.3、7.4 | 2、19 |
| 読めない `lizyml:schema`（状態 S） | `CONFIG_INVALID`（新規） | 7.1、7.4 | 12 |
| 記録と試行が食い違う study（状態 X） | `CONFIG_INVALID`（新規） | 7.4 | 19 |
| resume で `space_mode` を変えた | `CONFIG_INVALID`（新規） | 5.2-1 | 7 |
| resume で config の探索空間を変えた | `CONFIG_INVALID`（新規） | 5.1 | 11 |
| 分布の種類・`log`・categorical の選択肢の変更 | `CONFIG_INVALID`（新規） | 5.2-3 | 7 |
| 既にある次元の名前のカテゴリーを変えた | `CONFIG_INVALID`（新規） | 4.1b | 12 |
| 途中の round の再開で、記録と違う `n_trials` / `expand_boundary` / `space=` を渡した | `CONFIG_INVALID`（新規、U2） | 7.2 | 8b |
| round を閉じるとき、study に COMPLETE の試行が無い | `TUNING_FAILED`（今と同じ。round は `failed` で閉じる） | 7.1、7.4 | 8a |
| ライブラリの版の違い | 警告（新規） | 4.3 | 3 |
| 途中の round の再開で、config の `n_trials` が記録と違う | 警告（新規。記録の予算を使う） | 7.2 | 8b |

## 9. 過去の記録との照合

### 9.1 この仕様が置き換えるもの（Proposal で明示する）

| 過去の記録 | 内容 | この仕様 |
|---|---|---|
| H-0102（HISTORY.md:9963-9964）、BLUEPRINT.md:882-883 | resume は保存した次元・境界・拡張・固定の方針を使い、config の変更は保存した空間に入らない | 空間は `space=` で変えられる（5.1）。config の探索空間の変更は黙って無視せず拒否する。固定の方針を引き継ぐ点はそのまま |
| H-0102 の決定（HISTORY.md:10015-10018）、BLUEPRINT.md:885-886 | fit と resume は成功した tuning の結果（overlay）を使い続ける | resume の試行の基に前回の最良値は重ねない（6.1）。fit が tuning の結果を使う点はそのまま |
| H-0102（HISTORY.md:9981-9982）、BLUEPRINT.md:879、CHANGELOG.md:61 | 空間の意味を変えるなら新しい study。古い試行は新しい空間の観測ではない | 目的関数が同じなら、古い試行は観測である（INV-R3）。新しい study が要るのは、Optuna が受け付けない変更（5.2-3）、カテゴリーの変更（4.1b）、`space_mode` の変更だけ |
| H-0068（HISTORY.md:5119、5310）、BLUEPRINT.md:1082 | `resume=False` は今と同じ動作（完全な後方互換） | storage に既にある study を `resume=False` で指すと拒否。新しい study の系譜は round 1 から（今は前の study の round を積み増す） |
| H-0068（HISTORY.md:5120、受け入れ基準 13 :5337）、BLUEPRINT.md:1083、docs/api.md:95、108 | `resume=True` で tune 前なら `TUNING_FAILED` | storage を指すときは、study があれば再接続（D5）、無ければ新しく作る（U3）。storage も メモリ上の study も無いときは `TUNING_FAILED` のまま |
| H-0068（HISTORY.md:5161、ログ :5269）、CHANGELOG.md:383、docs/api.md:86-90 | 前回の最良値を必ず enqueue する（resume の round の最初に評価し直す） | enqueue しない（6.2）。ログの `tune.resume: enqueued previous best` も無くなる |
| H-0072（HISTORY.md:5804、INV-3 :5769）、BLUEPRINT.md:1097・1101、CHANGELOG.md:346、docs/api.md:100-101 | `resume=False` と storage で、既にある study に再接続する | 再接続は `resume=True`（D5）。`resume=False` で既にある study は拒否 |
| H-0072（HISTORY.md:5736） | `resume=True` で `_study` があれば、storage を渡しても in-memory の study を続ける | 渡した storage / study_name がその study と同じと示せなければ拒否（3） |
| H-0072（HISTORY.md:5736） | `resume=True`、study が無い、storage あり → 自動で再接続 | 採用する（D5）。BLUEPRINT.md:1083・1101-1102 との矛盾がこれで解ける |
| H-0072（HISTORY.md:5740）、BLUEPRINT.md:1108 | 再接続後の round は 1 から。round の記録は journal に保存しない（別の Proposal で扱う） | round の系譜を study の属性に保存し、再接続で続ける（7.1、7.2）。この Proposal が「別の Proposal」にあたる |
| H-0068（HISTORY.md:5202）、BLUEPRINT.md:1045 | `rounds` の既定は `(RoundSummary(round=1, ...),)`（初回の tune でも 1 要素） | `failed` で閉じた round は `rounds` に入らないので、`rounds` が round 2 以降から始まることがある（7.2） |
| H-0121（Proposed）の resume の条項（HISTORY.md:11973、11977-11978、受け入れ基準 6 :12045） | resume で今の config の smart の値を読み、無効な既定の次元を外す | H-0121 から外す（A1）。smart の値の変更は目的関数の変更として 4.3 で拒否される |

### 9.2 矛盾しないもの（拡張、または前提の明示）

| 過去の記録 | 関係 |
|---|---|
| H-0099 決定 2（HISTORY.md:9879-9880）、BLUEPRINT.md:1109 | 向きの照合を求めている。この案はそれを残し、目的関数の照合を足す（拡張） |
| H-0068 受け入れ基準 2（HISTORY.md:5326）「best_score が悪化しない」、BLUEPRINT.md:1065-1067 の累積の最良線 | INV-R1 で前提が保証される |
| H-0068（HISTORY.md:5082、5085）study を保持して追加試行、「まだ見ていない有望領域を探す」 | P1 そのもの。拡張の規則は 5.3 の 2 点を足すだけ（enqueue をやめることは 9.1） |
| H-0078、H-0111 の境界と no-op の拡張 | 変えない（5.2-5、5.3） |
| H-0072 INV-5（別の storage の試行と混ざらない）、INV-6（進捗・enqueue・拡張・round は storage の有無に依らない）、HISTORY.md:5805-5806（`current_trial` の一致） | 変えない。enqueue をやめることも、6.5 の規則も、storage の有無に依らない。途中の round の再開でも `current_trial` は続きから数える |
| H-0086、H-0109、H-0094 決定 13・14、H-0120 改訂 1 | artifact と報告面は変えない（7.5）。tune の失敗は Model の状態を変えない（INV-R4） |
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
- C11 `current_trial` の意味（BLUEPRINT.md:948 と docs/api.md:456）。6.5 で「この round の何番目（途中の round の再開では続きから）」と決める
- C12 docs/api.md:395-396 の例の名前

C11 だけはこの仕様が決める。

## 10. 利用側への影響

- **LizyStudio**: 障害からの再開（`services/training.py:440-449`）は、新しい `Model` で `resume` を付けずに `storage` と `study_name` を渡している。この仕様では `CONFIG_INVALID` になる。`resume=True` を渡すよう直す必要がある。v0.18 以前に作った study は照合の記録が無いので、再接続できない（D4）。また LizyStudio は、残りの試行数を自分で数えて config の `tuning.optuna.params.n_trials` に書き込み（`training.py:383-425`）、tune には `n_trials` の引数を渡さない（`lifecycle_mixin.py:173-181`）。この仕様では、途中の round の再開で config の値は使われず（警告が出る）、記録した予算で round を終える。この計算は不要になるので、やめる必要がある。
- **LizyStudio の retune**（`training_retune.py:86-105`）: Model 全体を cloudpickle したものを読み、`resume=True` を呼ぶ。新しい版で作った checkpoint なら study の属性ごと運ばれる。古い版の checkpoint は照合の記録が無いので拒否される。
- **LizyML-Widget の進捗**（`job_runner.py:170-178`）: resume の呼び出しの前に、round を 2 と決めて進捗を出している。この仕様では、resume は途中の round k の続き、または k+1 番目の round になりうるので、最初の進捗の round が系譜と食い違う（後に続く `TuneProgressInfo` の round は正しい）。版の上限を上げるとき、最初の round を出さないか、study の系譜から求めるように直し、途中の round の続きと、二つ以上の round のあとの resume の両方を Widget のテストで確かめる。
- **LizyML-Widget**（`adapter.py:767-800, 845-851`）: `_study`、`_space`、`_used_default_space`、`_round_number`、`_rounds` を pickle して別プロセスで戻している（`_tuning_fixed_params` は運ばない）。同一性・系譜・固定パラメーターは study の属性から読み、Model のフィールドより優先する（5.3）ので、`_study` が運ばれれば足りる。メモリ上の study は pickle できる（実行で確認）。
- **LizyStudio の tune のあとの fit**（`services/training.py:458`、`services/training_retune.py:109` → `_training_core.py:337-347`）: `TuningSummary.best_params`（model・smart・training を平らにしたもの、`backends/lizyml/serialization.py:26-44`）を、すべて config の `model.params` に入れて fit している。smart と training の値は `model.params` では効かないか拒否されるので、INV-R2（最良の試行と同じ学習）が LizyStudio では成り立たない。今の版でも smart の次元で起きる既存の問題で、この仕様では外した次元の値が最良として残る（D2、Q-B）ため起きやすくなる。版の上限を上げるとき、tune した Model でそのまま `fit()` する（その Model は三つの区別と、study の固定パラメーター `fixed` の両方を持つ）。三つの区別を分けて運ぶだけでは足りない: INV-R2 の学習は `config ⊕ fixed ⊕ 試行の値` で、`merge` の study では provider が `fixed` として `first_metric_only=True` と `metric` を決める（`defaults.py:138-153`。通常の既定は `first_metric_only=False`）。config に書き写して別の Model で fit する経路では、`fixed` を写す公開の手段が今は無いので、early stopping の判断が変わり、別のモデルを学習しうる。
- **LizyStudio の round のループ**（`lifecycle_mixin.py:186-197`）: 再チューニングの `n_trials` / `expand_boundary` を最初の呼び出しに付け、`range(2, n_rounds + 1)` で round 2 から数える。途中で落ちた round を再開する呼び出しに違う `n_trials` が付くと U2 で `CONFIG_INVALID` になり、round が 2 から始まる前提も、途中の round の再開では成り立たない。試行ごとに Model を pickle する checkpoint（`_bridge`）は開いた round の study を保存しうるので、そこからの retune（`training_retune.py:93-105` は `n_trials` を必ず渡す）も同じく U2 に当たる。版の上限を上げるとき、round の数え方を study の系譜（`TuningResult.rounds` と試行の round）に合わせて直す。
- 公開 API: `tune()` に `space=` を足す（A-1）。`lizyml.tuning.read_study_lineage` と `StudyLineage` を足す（Q-G、7.6）。エラーの条件が増える（8）。CHANGELOG と migration に書く。

### 10.1 利用者への影響の判定（2026-10-11 の調査）

- 呼び出しを変えずに動く使い方: チュートリアル（`notebooks/tutorial_regression_tuning_lgbm.ipynb` の tune → fit → `tune(resume=True, n_trials=3, expand_boundary=True)` → `boundary_table()`）、docs/api.md のメモリ上の resume。storage を使わない tune は、round の間でデータと config（探索空間を含む）を変えず、`study_name` だけを渡すこともしない限り、移行は要らない。ただし結果は変わりうる: resume の round は前回の最良値を評価し直さず（6.2）、要求した試行がすべて TPE の新しい点になるので、同じ seed でも round の試行の中身と最良値が今と違いうる。CHANGELOG に挙動の変更として書く。codegen は tune を出力しない。
- 移行が要る（storage を使わない場合）: 同じ Model で、データ・config の探索空間以外の部分・最初の評価指標を変えてから `resume=True` を呼ぶと、今は黙って続き別の目的関数のスコアが混ざる（`design-vs-implementation.md` の P1・P2）が、`CONFIG_INVALID` になる（4.3）。config の探索空間を書き換えてから `resume=True` を呼ぶと、今は黙って無視されるが、`CONFIG_INVALID` になる（5.1、範囲を変えるなら `space=`）。`storage` 無しで `study_name` だけを渡すと、今は黙って無視されるが、`CONFIG_INVALID` になる（3）。どれも、今は黙って間違うか黙って無視される呼び出しを、明示のエラーに変えるもので、直し方（新しい study、`space=`、引数を外す）がエラーのメッセージで決まる。
- 致命的（続ける方法が無い）: v0.18 以前に作って storage に保存した study（docs/api.md:101、CHANGELOG 0.12.0 の使い方）と、アップグレードした時点で一時停止中の LizyStudio のジョブ。D4 による。CHANGELOG と migration に「旧 study は続けられない。一時停止中のジョブはアップグレードの前に終える」と書く（U1）。
- 移行が要る（明示のエラーと、決まった直し方がある）: LizyStudio の unpause と failed ジョブの resume、Widget の別プロセスの Retune。LizyStudio は `lizyml<0.18.0`、Widget は `<0.17` に固定しているので、版の上限を上げるときに直す。
- 致命的に近いもの: 「落ちたら同じスクリプトを流し直す」書き方（P2 の基本の使い方）が、1 回の呼び出しで書けない（初回は `resume=False`、流し直しは `resume=True`。途中の round では `n_trials` を渡せない）。U2・U3 で解く。

### 10.3 利用側の使用箇所の一覧（2026-10-11 の網羅調査）

LizyStudio（`src/lizystudio`）と LizyML-Widget（`src/lizyml_widget`）で、tuning の状態（`tune(resume=…)`、storage、`best_params`、`rounds`、進捗、`_study` などの private な属性）を読み書きする箇所を grep で網羅した。上の箇条に無いもの、または一部だけのものを挙げる。どれも、版の上限を上げるときに利用側で直す（この仕様の実装の範囲ではない）。tuning の状態を読み書きしない箇所（設定の既定値、文言、経路の受け渡しだけ）は除いた。

| 利用側 | 箇所 | この仕様での影響 | 直し方 |
|---|---|---|---|
| Studio | `services/training.py:287-337, 400-417`（残りが 0 なら tune を呼ばず、Optuna から結果を組み立てる） | round に `:end` が書かれず状態 O のまま。照合（4.3）を通らない。平らな `best_params`（外した次元を含みうる）が fit に渡る | tune を呼ばずに組み立てるのをやめ、`resume=True` に任せる（開いた round は 7.2 で閉じる） |
| Studio | `api/jobs.py:324-373`、`_training_core.py:66-90, 194-230`（一時停止と unpause） | 一時停止の位置により、study は状態 O（round の途中）にも C（round の間、tune の後、fit の途中）にもなる。C で `resume=True` を呼ぶと、余分な round が始まる | `read_study_lineage`（7.6）で状態と系譜を読み、開いていれば `resume=True` で終え、閉じていれば残りの round の数を系譜から決める |
| Studio | `backends/lizyml/lifecycle_mixin.py:91-114`（`model._tuning_result.trials` を読む） | 再接続した Model には無いので、それまでの試行が進捗の一覧から消える | 試行と round は tune の結果（`TuningResult.trials` と `rounds`）から読む |
| Studio | `lifecycle_mixin.py:116-168`（`_bridge` の round と試行の番号） | 最初の呼び出しを round 1 と表示する。試行の番号が局所的 | 進捗の `round`・`current_trial`・`cumulative_trials` を使う |
| Studio | `backends/lizyml/serialization.py:16-69`、`backends/types.py:176-196`、`api/models.py:229-241` | `rounds` が round 2 から始まり、`best_score_before` が `None` になりうる。`n_trials` は実際の数。平らな `best_params` に model・smart・training の区別の置き場が無い | 型に三つの区別を足し、`None` を扱う |
| Studio | フロントエンド `TuneTrialsSection.tsx:30-73`（Apply to Fit） | 平らな `best_params` を `model.params` に入れる。INV-R2 が成り立たない（バックエンドの `_prepare_autofit_config` と同じ） | tune した Model で fit する（三つの区別と `fixed` の両方が要る。上の箇条） |
| Studio | `services/training_retune.py:43-51, 85-105`（子のジョブが親の `optuna.db` の study を共有する） | 兄弟の retune が一つの study に書く。一つの study に書くのは一つのプロセスという前提（3）に反しうる。落ちた子の開いた round を、次の兄弟が引き継ぐ | 子のジョブごとに study を分ける（新しい `study_name`）か、同じ study への書き込みを直列にする |
| Studio | `api/retune.py:224-310`、フロントエンド `job-config.ts:137-147`（失敗したジョブの再開で `n_trials` を計算して渡す） | 計算した数は、開いた round の記録の `n_trials` とほとんど一致せず、U2 で `CONFIG_INVALID` | 開いた round の再開では `n_trials` を渡さない |
| Studio | フロントエンド `RoundHistoryTable.tsx`、`ConvergenceSignalPanel.tsx`、`SearchSpaceEvolutionPanel.tsx` | `rounds` が飛ぶ・round 2 から始まる・`best_score_before` が null になりうる。`rounds.length >= 2` の表示条件の意味が変わる | 表示を round の番号と null に合わせる |
| Widget | `adapter.py:600-689`（retune の画面は `n_trials` と `expand_boundary` を必ず渡す） | 開いた round で記録と違う値なら U2 で `CONFIG_INVALID` | 開いた round の再開では渡さない |
| Widget | `_subprocess_entry.py:115-143`、`widget.py:530-645`（別プロセスの retune は、その時点の画面の config で Model を作る） | tune と retune の間に config・評価指標・探索空間を編集すると `CONFIG_INVALID`（4.3、5.1）。同じプロセスの経路は元の config を使うので通り、二つの経路で結果が分かれる | tune した時点の config を保存して使い、範囲の変更は `space=` で渡す |
| Widget | `service.py:570-605`、`widget_actions.py:279-335`（取り消しで round の途中に止まる） | メモリ上の study に開いた round が残り、次の retune はその round を終える（新しい round ではない）。違う `n_trials` は U2 で拒否 | 取り消しのあとは、開いた round を終えるか、新しい study で始める |
| Widget | `adapter_params.py:67-103`、`service.py:748-866`（`apply_best_params` が固定の名前の集合で三つに分ける） | 今の六つの名前では正しいが、LizyML の系譜の経路（4.1b）と別の規則。新しい Model を作って fit する（`service.py:520-531`）ので、`fixed` も失われる | tune した Model（`_tune_model`）で fit する。config に写すなら三つの区別を使い、`fixed` が失われることを画面で示す |
| Widget | `adapter_views.py:66-225`、`adapter_internals.py:88-122`、`js/src/tabs/ResultsTab.tsx:293-330` | 試行の round の既定を 1 にする。`rounds` の数で「resume の回数」を表示するので、`failed` の round が入らないと少なく出る | 表示を round の番号に合わせる |

影響の無いもの: 両方の `tuning_plot` の保存と表示（区切りは LizyML 側で試行の round から求める、6.5）、Widget の INV-E / INV-F の検査（今の表示の範囲では成り立つ）。

### 10.2 利用者への影響から決めたこと（2026-10-11、管理者が決定）

- **U1**: v0.18 以前の study は拒否のまま（D4）。CHANGELOG と migration に、続けられないこと、一時停止中のジョブはアップグレードの前に終えることを書く。

- **U2（途中の round の引数）**: 途中で落ちた round を再開するとき、`n_trials`・`expand_boundary`・`space=` は、記録した値と同じなら通し、違えば `CONFIG_INVALID` にする（Q-D の「渡したら拒否」を緩める）。省いた引数は比べず、config の `n_trials` は警告だけ（7.2）。途中の round の再開だけに当て、新しい round（再探索）には当てない。規則は 7.2。
- **U3（作るか続けるか）**: `resume=True` で storage の study を指し、その study がまだ無ければ、新しい study を作って round 1 から始める（作ったことを INFO のログに出す）。3 の表に反映した。メモリ上の study も storage も無い `resume=True` は `TUNING_FAILED` のまま。名前の打ち間違いは新しい study になるが、目的関数の違う試行は混ざらない。

## 11. 決定（2026-10-11、管理者がすべて推奨どおりに決定）

- **Q-A**: 範囲を変える方法（5.1）。A-1（`space=` の引数）。
- **Q-B**: 外した次元の扱い（6.1、6.4）。最良の試行がその次元を持てば、その値で `fit` する。以後の試行は config の値で学習する。
- **Q-C**: `resume=False` で既にある study を指したときの拒否（3）。LizyStudio の修正が要る。
- **Q-D**: 途中で落ちた round の再開（7.2）。同じ round を残りの予算で終える。RUNNING のまま残った試行は FAIL として数える。
- **Q-E**: 照合の厳しさ（4 の冒頭）。保守的な互換性。
- **Q-F**: 前回の最良値の enqueue（評価し直し）をやめる（6.2）。当初は条件付きにすると決めたが、2026-10-11 に管理者がやめることに変更した。

- **Q-G**: 利用側が、試行を足さずに study の状態と系譜を読む公開の関数を足す（7.6）。LizyStudio の一時停止と再開（10.3）で使う。2026-10-11 に管理者が決定。

あわせて採る技術的な規則:

- 次元の経路は足すだけで、カテゴリーは変えられない（4.1b）。
- 「同じ study」の判定は文字列またはオブジェクトの一致（3）。
- `study_name` だけを渡したら拒否する（3）。
- 読めない `lizyml:schema` は拒否する（7.1）。
- 一つの study に試行を足すプロセスは一つ（3）。

## 12. 受け入れ基準（テストの観点）

1. 目的関数の要素（データの値、データの行の順序、config の各部分、最初の評価指標）を一つずつ変えて `resume=True` → `CONFIG_INVALID`。study の試行数と Model の状態が変わらない。4.2 のうち `n_trials`・timeout・sampler・`progress_callback`・`expand_boundary`・`boundary_threshold`・二つ目以降の評価指標・calibration・export の設定を変えても通る（探索空間の変更は 5.1 の経路（`space=`）で扱い、config の探索空間の変更は 11、storage・study_name の変更は 10 で拒否される）。
2. 照合の記録が無い study（属性を消した study）に `resume=True` → `CONFIG_INVALID`。
3. ライブラリの版の記録を書き換えて `resume=True` → 警告が出て続く。
4. `space=` で範囲を動かして resume → 新しい試行は新しい範囲。最良が古い範囲の試行なら、その値が `best_params` になり、`fit` がそれを使う。その次元は拡張されない。
5. 宣言を変えない resume → 拡張した範囲が引き継がれる（今のテストと同じ）。
6. 次元を外して resume → 以後の試行はその次元を config の値で学習する。外した smart の次元が最良の試行にあれば、smart として `fit` に渡る（LightGBM の params に入らない）。
6b. training の次元（`early_stopping_rounds`）を外して resume し、外す前の試行が最良のまま → `best_training_params` にその値が入り（`best_model_params` には入らない）、続く `fit()` はその値で学習する。
7. 分布の種類・`log`・選択肢の変更（`True` を `1` に、`1` を `1.0` に変えるものを含む）、`space_mode` の変更 → study を変える前に `CONFIG_INVALID`。
8. storage で round 2 の途中に落とした（プロセスを止めた）あと、新しい `Model` で `resume=True` → round 2 の残りの予算を、round 2 の実効空間で走らせて閉じる。round 2 の試行の総数、`rounds`、試行の round が、落とさずに走らせた場合と同じ。RUNNING のまま残った試行は FAIL になり、予算に数えられる。境界の拡張は起きない。`:end` を書く直前に落とした場合は、試行を足さずに round 2 を閉じる。同じプロセスで round が例外で止まった場合も同じ。
8a. `n_trials=1` の round 1 で、ただ一つの試行が RUNNING のまま落ちた study に再接続 → その試行が FAIL になり、round 1 が `failed` で閉じ、`TUNING_FAILED`。もう一度 `resume=True` → round 2 が始まり、拡張は無い。round 2 が終わったあとの `TuningResult.rounds` は round 2 だけで、その `best_score_before` は `None`。`trials` には round 1 の FAIL の試行が round 1 として残る。
8b. 途中の round の再開（U2）。RUNNING の試行が残った開いた round で確かめる: round を始めたときと同じ `n_trials`、その round で解決した値と同じ `expand_boundary`、同じ宣言空間に解決される `space=`（同じ dict、および `log` を省いた書き方や dict の並びを変えた書き方）→ それぞれ通り、残りの試行を走らせる。違う `n_trials`、違う `expand_boundary`、違う `space=`（範囲が違う、categorical の選択肢の型だけが違う）→ それぞれ `CONFIG_INVALID`。試行の数も、RUNNING の試行の状態も、study の属性も変わらない。引数を省く、または `None` を渡す → 比べずに通る。categorical の選択肢に `float('nan')` を含む `space=` で始めた round を、同じ `space=` で再接続して再開 → 通る。config の `n_trials` だけが記録と違う → 警告が出て、記録の予算で終える。終わった round のあとの `resume=True` で違う `n_trials` を渡す → 新しい round がその値で始まる。
9. `resume=False` で既存の study → `CONFIG_INVALID`、試行が増えない。
10. `resume=True` で別の storage → `CONFIG_INVALID`。
11. config の探索空間を変えて `resume=True`（`space=` なし） → `CONFIG_INVALID`。
12. `study_name` だけ、`storage` だけを渡す → `resume` に依らず `CONFIG_INVALID`。読めない `lizyml:schema` → `CONFIG_INVALID`。既にある次元のカテゴリーを変える宣言 → `CONFIG_INVALID`。
13. データの列名を変える、categorical の categories の順序を変える、categorical の列の値を `[1, "1"]` から `["1", 1]` に変える（型の違う同じ表記の値の入れ替え）→ 同一性の違いとして `CONFIG_INVALID`。データをコンストラクターの `data` で渡した場合と、`data.path` の場所のファイルの中身だけを変えた場合も、同じく `CONFIG_INVALID`。
14. 呼び出しの表（3）の各行: 
    - 同じ `storage`（URL の文字列が等しい、または同じ `BaseStorage` のオブジェクト）と同じ `study_name` を渡した `resume=True` → 継続する（試行が増え、round が進む）。`storage` と `study_name` を省いた `resume=True` → 継続する。
    - `study_name` だけが違う、URL の文字列が違う（同じファイルを指す別の書き方を含む）、同じ backend を指す別の `BaseStorage` のオブジェクト → `CONFIG_INVALID`、試行が増えない。
    - 新しい `Model` で、storage に既にある study を `resume=True` で指す → 再接続して続く（P2 の成功）。
    - 新しい `Model` で、正しい `storage` と、storage に無い `study_name` を `resume=True` で指す → 新しい study ができ、`lizyml:study` が書かれ、round 1 から始まる（U3）。INFO のログが出る。同じ呼び出しで `space=` を渡す → `CONFIG_INVALID`、study は作られない。
    - メモリ上の study も `storage` も無い `resume=True` → `TUNING_FAILED`。
    - `resume=False` で `space=` を渡す → storage 無し（メモリ上の新しい study）でも、storage あり（storage に無い名前）でも `CONFIG_INVALID`。study は作られず、storage に何も書かれない。
15. resume の round（空間を変えないもの、`space=` で変えたもの）→ 前回の最良値のパラメーターを予約した試行は無く、その round の試行はすべて TPE がサンプルした点で、数は要求した `n_trials` と同じ。`best_params` と `best_score` は study 全体の最良の試行のまま。round の準備の途中、`lizyml:round:<k>` を書く前に止めたもの → 普通の resume として round k が始まる。書いた後に止めたもの → 状態 O として round k を残りの予算で終える。
16a. 報告面の round の区切り: `failed` で閉じた round 1 のあとに round 2 を終えた study と、`timeout` で要求数より少ない試行で終わった round を含む study で、`tuning_plot` の区切り線と round の注記、`tuning_table` の round の列が、試行の round（番号から決まるもの）の変わり目と一致する。`RoundSummary.n_trials` を積み上げて区切る実装では落ちる。
16. `timeout` のある round を途中で落とした場合 → 再開では、記録から求めた残りの時間を Optuna の `timeout` に渡す（Optuna の timeout は新しい試行を始めないだけで、走っている試行は止めない。今の tune と同じ意味）。残りの時間を使い切っていれば、試行を一つも始めずに round を閉じる。時間を使い切ってから `:end` の前に落とした場合も同じ。同じ round で二度落とした場合も、残りの時間は COMPLETE と PRUNED の試行の時間の合計から求まり、落ちていた間の時間を含まない。一度目の再接続で RUNNING から FAIL にした試行が残った状態で二度目に再接続しても、その試行の時間（落ちていた間を含む）は引かれない。
17. `space=` で `merge` の宣言に利用者の次元を足した round のあと、再接続して次の round を `expand_boundary=None` で始める → 同じプロセスで続けた場合と同じ拡張の判断になる。
18. 途中で落ちた round（その round で K 件終わっている）を再開 → 進捗の `current_trial` は K+1 から始まり、storage の有無で変わらない。
19. study を作った直後、`lizyml:study` を書く前に止めた場合 → 再接続で作りかけの study として初期化され、round 1 から始まる。試行のある study から `lizyml:study` を消したもの → `CONFIG_INVALID`。次元を足した round で、`lizyml:round:<k>` を書く前に止めた場合 → その次元の経路は残らない。`lizyml:study` を書いた後、round 1 の記録の前に止めた場合（状態 N）→ `resume=True` で round 1 が始まり、`space=` を渡すと `CONFIG_INVALID`。試行はあるが round の記録が無い study、round の番号が飛んでいる study、最後以外の round に `:end` が無い study、round 1 の開始より小さい番号の試行がある study（状態 X）→ それぞれ `CONFIG_INVALID`、study は変わらない。
    状態 E（`lizyml:study` も試行も無い study）に、受け付けの検査で落ちる config（たとえば smart と衝突する探索次元）で `resume=True` → `CONFIG_INVALID`、`lizyml:study` は書かれない（INV-R4）。状態 N に同じ config で `resume=True` → `CONFIG_INVALID`、round の記録も試行も増えない。
    目的関数の途中で止めた場合（Optuna は試行を RUNNING として storage に作ってから目的関数を呼ぶ）→ 再接続で状態 O と判定され、その試行は番号から決まる round の FAIL として残りの予算に数えられる。
20. 移行の文書（U1）: CHANGELOG の該当の版の節と `docs/migration.md` に、この変更より前に作った study は続けられないこと（新しい `study_name` で始める）と、一時停止中のジョブはアップグレードの前に終えることが書かれている。`tests/test_docs/` の文書の検査で、その記述があることを確かめる。
21. 既存のテストのうち今の挙動を固定しているもの（`test_space_merge.py:183, 262, 305, 375, 395`、`test_tuner_persistence.py:252`、`test_retune.py:807`、`test_error_code_raising.py:57`）は、新しい規則を固定するように書き直す（消さない）。`test_retune.py:823-838` は試行の総数（3 + 2 = 5）が変わらないのでそのまま通るが、enqueue が数に入るという注釈（:834）を直す。`Tuner` を直接使うテスト（`test_tuner_persistence.py:116-127, 185-199`、`test_direction_reconciliation.py:100-119`）は変えない（次の段落）。

22. 拒否で何も変わらない（INV-R4）: 8 の表の `CONFIG_INVALID` の行のすべて（受け入れ基準 2・7・9・10・11・12・13・14・19・29 の各拒否と、RUNNING の試行が残った状態 O、状態 N の study でデータを変えた `resume=True`）を一つずつ表で回し、それぞれで次の全部が呼ぶ前と同じことを確かめる:
    - study: 試行の数、各試行の状態と値、`lizyml:` で始まる属性のすべて（storage の study は storage から読み直して比べる）
    - Model: tune が書く状態のすべて（`_study`、`_tuning_result`、`_space`、`_round_number`、`_rounds`、`_used_default_space`、`_tuning_fixed_params`、覚えている storage と study_name、`_X`・`_y`、ほかに H-0094 決定 14 が「tune の成功のときにまとめて書く」とするもの）
23. 新しい study と系譜: round を二つ終えた Model で `resume=False` の tune（storage なし、storage に無い名前）→ 新しい study の `TuningResult.rounds` は round 1 だけ、試行の round は 1、`lizyml:round:1` が書かれる。
24. load のあと: tune した Model を export して `Model.load` → storage なしの `resume=True` は `TUNING_FAILED`。export → load の既存のゴールデンテスト（artifact の形、`format_version`）は変わらずに通る。
25. 再接続した study の「同じ study」: 新しい Model で storage の study に再接続したあと、同じ storage・study_name で `resume=True` → 続く。`study_name` だけを変える → `CONFIG_INVALID`。
26. データの同一性の残りの要素: 行の値と並びはそのままで index のラベルだけを変える、categorical の列に使われていない category を一つ足す（どの行の値と codes も変えない）、 列の並びだけを入れ替える、表記が同じで dtype だけが違う列（int64 と float64、int64 と Int64）、categorical の `ordered` だけを変える、categorical でない object の列の値を `1` から `"1"` に変える、`features.exclude` の列の値だけを変える → それぞれ `CONFIG_INVALID`。
27. 照合の失敗のメッセージ: データだけ、config の一部だけ、評価指標だけを変えた各 resume → `CONFIG_INVALID` の context に、違う要素の名前が入る。
28. provider の既定の `fixed` が変わった場合: `fixed` の既定を monkeypatch で変えて `resume=True` → 版の記録を変えても変えなくても、警告で続かずに `CONFIG_INVALID`。
29. `space=` と受け付けの検査（5.2-1、5.2-4、5.2-5）:
    - `merge` の study で既定の次元の一つだけを `space=` で変える → 他の既定の次元は宣言空間に残り、新しい試行でもサンプルされる。
    - `space=` に未知の名前、同じ次元の重複（alias と正規名）、training の次元を別のカテゴリーに置いた宣言 → 試行を足す前に `CONFIG_INVALID`、study は変わらない。
    - H-0121 で外れる既定の次元を持つ設定で、`space=` を付けた resume と、再接続したあとの同じ resume → どちらもその既定の次元が外れ、利用者が `space=` で宣言した同じ名前の次元は外れない。
    - `space=` で provider の境界より外の範囲を宣言 → 受け付けられ、試行の値はクランプされない。境界の拡張は provider の境界を超えない。
    - model の次元を alias で宣言した round のあと正規名で同じ次元を宣言 → 同じ次元として扱われ（経路は一つ、拡張を引き継ぐ）、開いた round の再開では alias と正規名の `space=` が同じとして通る。
30. 拡張と前回の最良値（5.3）:
    - `space=` で範囲を変えた次元で、前回の最良値が新しい範囲の中かつ境界の近く → その round でその次元は拡張されず、`boundary_table` に `expanded=False` で出る。最良値が範囲の外の次元も `expanded=False` で出る。
    - 別の study の tuning の結果を持つ Model（`Model.load` のあと）で storage の study に `resume=True` → 拡張の判断はその study の最良の COMPLETE の試行から決まる。
    - `_tuning_fixed_params` を運ばない pickle の Model（Widget と同じ）、および Model のフィールドを study の `fixed` と違う値に書き換えた Model で `resume=True` → 試行・照合・拡張の判断が `lizyml:study` の `fixed` に従う。
31. round の要約と進捗（6.5）: `timeout` で要求数より少ない試行で終わった round → `RoundSummary.n_trials` が実際の試行の数。途中で落ちた round を再接続で終える → その round の `boundary_report` が落とさずに走らせた場合と同じ。開いた round の再開 → 進捗の `total_trials` は記録の `n_trials`、`cumulative_trials` は `trial.number + 1`、`round` は番号から決まる round。
32. round の終わりの記録: `completed` の round の `:end` に終了時の最良値と時刻がある。`failed` の round では最良値が空。
33. 状態の判定の順序: `schema` が読めない `lizyml:study` を持ち、round の記録も試行も無い study（S と N の両方に当たる）→ `CONFIG_INVALID`（N として round 1 を始めない）。
34. 文書: `tests/test_docs/` で次を確かめる。
    - `docs/api.md`: `Model.tune` の引数に `space=` があり、resume の説明から前回の最良値の enqueue（warm-start）の記述が無く、storage への再接続が `resume=True`（無ければ新しく作る、U3）と書かれ、エラーの一覧に 8 の新しい条件があり、`rounds` の説明に「`failed` で閉じた round は入らず、round 2 から始まりうる」とある。
    - `docs/api.md` と BLUEPRINT §11: `read_study_lineage` と `StudyLineage`・`RoundRecord`（引数、`state` の値の一覧、読み取りだけであること）が書かれている。
    - BLUEPRINT §11.5・§11.5.1: `resume` の表（今の 990-993 行）と resume・storage の関係（今の 1082-1108 行）が、この仕様の規則（enqueue なし、`resume=True` での再接続、`space=`、照合）に置き換わっている。
    - CHANGELOG の該当の版の節と `docs/migration.md`: `tune(space=...)` の追加、8 の新しいエラー条件、resume の結果が変わりうること（10.1）。

35. 系譜を読む関数（7.6）: 7.4 の各状態の study（storage に無いもの、旧 study、作りかけ、読めない、round が無い、壊れている、開いた round、完了、失敗）に `read_study_lineage` → `state` がそれぞれの値になり、`rounds` と `open_round` が系譜どおり（`failed` の round も入る）。呼ぶ前と後で、study の試行の数と状態（RUNNING の試行を含む）と属性が同じ。`storage` だけ、`study_name` だけを渡すと `CONFIG_INVALID`。`lizyml.tuning.__all__` に `read_study_lineage` と `StudyLineage` があり、トップレベルの `lizyml.__all__` は変わらない。`StudyLineage` と `RoundRecord` の欄に代入すると `FrozenInstanceError` になり、`rounds` は tuple である。

一つの study に試行を足すプロセスは一つという前提（3）は、利用者に求める前提で、ライブラリは強制しない（lease などの仕組みは入れない）。テストは、再接続のときに残った RUNNING の試行が FAIL になること（8）だけを確かめる。

この仕様の規則は `Model.tune` に置く。`Tuner`（`lizyml.tuning` にあるが、トップレベルにも docs/api.md にも無い）は今と同じく study を作る・読むだけの層とし、`resume` の意味や同一性の照合は持たない（BLUEPRINT.md:448 の責務分離）。

Firing rate: Proposal を書く段で測る（テストのうち resume と再接続を呼ぶものと、`design-vs-implementation.md` の probe P1-P7 を再生し、拒否の数を数える）。
