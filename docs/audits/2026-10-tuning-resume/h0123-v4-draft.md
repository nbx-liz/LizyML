
## H-0123: 探索空間の次元の指定を、文書の書式どおりに検査する（#344）

- **ステータス**: Proposed
- **起票日**: 2026-10-12
- **スコープ**: `lizyml/tuning/search_space.py`（`parse_space`）、`lizyml/core/_tuning_admission.py`（`parse_space_arg`）、`lizyml/config/schema.py`（`tuning.optuna.space` の入れ物の正規化）、BLUEPRINT の探索空間の節、docs/config-reference.md、CHANGELOG
- **関連**: [Issue #344](https://github.com/nbx-liz/LizyML/issues/344)、H-0068、H-0078、H-0093、H-0095、H-0122（`tune(space=...)`）、H-0124（`step`、この Proposal の後に入れる）

### 目的（課題）

`tuning.optuna.space` の書式は、docs/config-reference.md と BLUEPRINT §11.1・§11.2 が定める。次元ごとに `type`（`float` / `int` / `categorical`）、`low` と `high`（数）、`log`（真偽、任意）、`choices`（値の list）、`category`（`model` / `smart` / `training`、任意）。LizyML は config 駆動で、YAML と JSON が作る型（数、真偽、文字列、list）で書くことを前提にしている。

ところが、config の `tuning.optuna.space` は schema で `dict[str, Any]` で、書式の検査はどこにも無い（H-0122 の `tune(space=...)` だけが名前、mapping、`category` を検査する）。`parse_space` は `float()` と `int()` と `bool()` で値を変換するので、書式の外の値を黙って別のものにする（develop で実行して確かめた、#344 とそのコメント）:

- `log: "false"`（文字列）が log の探索になる。
- 書式に無いキー（`step`、`lo` のような綴りの誤り）を捨てる。LizyStudio は画面の入力欄の刻みを `step` として送っており、刻みの探索になっていると利用者に見せているが、実際は刻み無しで探索している。
- `int` の次元の `low: 1.7` を 1 に切り捨てる。
- 未知の `category`（例: `"booster"`）を受け付ける。
- `choices` の文字列（`"abc"` が `a`・`b`・`c` の三つの選択肢になる）、generator（読み尽くされて選択肢が空になる）、`set`（並びが実行ごとに変わりうる）、dict（キーだけが残る）を受け付ける。
- `low` / `high` の NaN と無限大を受け付ける。

### 対応方針（提案）

1. **検査を `parse_space` に集める。** config の空間と `tune(space=...)` が同じ規則を通る。`parse_space_arg` の形の検査は `parse_space` に移し、`parse_space_arg` は拒否の場所を `tune(space=)` と名指すだけにする。
2. **入れ物を正規化する。** 空間と各次元の指定は mapping であればよい（`dict` でなくても）。config の `tuning.optuna.space` は、schema の検証で空間と各次元を plain な `dict` に写す（名前の検査 `model_space_names` などは `dict` からだけ `category` を読むため）。mapping でない指定（今の config の経路では素の `AttributeError`）は `CONFIG_INVALID` にする。
3. **文書の書式だけを受け付ける。** YAML と JSON が作る型で書かれた書式を正とし、それ以外は「書式の外」として拒否する。次元ごとに:
   - 名前は `str`。`type` は `float`・`int`・`categorical` のどれか。
   - キーは、`float` と `int` では `type`・`low`・`high`・`log`・`category`、`categorical` では `type`・`choices`・`category` だけ。`step` は H-0124 が入るまで書式の外として拒否する（「LizyML の次元にはまだ刻みが無い」と案内する）。
   - `low` と `high` は数: `int` か `float`（`bool` は除く）。加えて二つを受け付ける。
     - **numpy の数のスカラー**: H-0095 が「数の境界は parse のときに素の値に直すので受け付ける」と定めている（`tests/test_tuning/test_literal_domain_contract.py` が固定）。
     - **数の表記にちょうど一致する文字列**: PyYAML（YAML 1.1）は小数点の無い指数表記 `1e-4` を文字列として読む（実行して確かめた。LizyML の `load_config` も同じ）。表記は `[+-]?(数字列(.数字列?)?|.数字列)([eE][+-]?数字列)?` の完全一致（ASCII の数字だけ）。`"nan"`、`"inf"`、`"0x10"`、`"1,000"` などは拒否する。
     - どの形でも、値は有限であること。`int` の次元では整数の値（`16`・`16.0`・`"16"`・numpy の整数は受け付け、`1.7` は拒否）。
   - `log` は書くなら `bool`（YAML の `true` / `false` / `yes` / `no` は `bool` になる）。
   - `choices` は空でない list か tuple（YAML と JSON は list を作る。Python で書く利用者の tuple も受け付ける）。各値の検査は今の `_validate_categorical_choices` のまま（素のスカラー、numpy のスカラーは拒否。H-0095）。
   - `category` は書くなら `model`・`smart`・`training` のどれか。
   - 拒否は `LizyMLError`（`CONFIG_INVALID`）で、次元の名前、キー、理由、空間（`tuning.optuna.space` か `tune(space=)`）を context に入れる。
4. **時期**: 方針 3 は、今と同じく `tune` が空間を解決するとき（study を作る前、H-0122 の INV-R4 の前の段）。config の構築時には方針 2 の写しだけを行う（`config/` は `tuning/` を import できない層の規則のため）。

### 規則が縛る位置（ソースから導出、実装前）

- `lizyml/tuning/search_space.py` の `parse_space`（`tuning.optuna.space` を読むすべての経路: `lizyml/core/_tuning_plan.py` の `resolve_config_space`、`lizyml/core/_tuning_resume.py` の config の空間の比べ合わせと `origin_space`、`lizyml/core/_tuning_admission.py` の名前の検査）。
- `lizyml/core/_tuning_admission.py` の `parse_space_arg`（`tune(space=...)`）。
- `lizyml/config/schema.py` の `tuning.optuna.space` の欄（方針 2 の写し）。

### 互換性

- 今は受け付けていて、この変更の後に拒否されるのは「書式の外」の指定だけ。目的（課題）の各形に加えて、`bool` や `bytes` の境界、数の表記でない文字列の境界、書式に無いキーを持つ次元、list と tuple でない `choices`（`range`、numpy の配列、`set`、dict、generator、文字列）がこれに当たる。
- 書式どおりの指定は、今と同じ次元に解決される。numpy の数の境界（H-0095）と、数の表記の文字列の境界（YAML の `1e-4`）も今と同じ値になる。
- Firing rate: 下の測定のあと、この行を `Firing rate: n/N ...` に書き換える（`docs/audits/2026-10-tuning-resume/probes/firing_rate_344.py` を方針 3 と同じ判定に直して測り直す）。
- 下流の製品（LizyStudio、LizyML-Widget）。どちらも空間を config の `tuning.optuna.space` で渡す。版の上限を上げるときに直す必要があるもの:
  - LizyStudio: Range にした次元に付ける `step`。H-0124 で受け付けるようになるが、`log` の次元の `step` と、`step` の倍数でない範囲は H-0124 で拒否される。
  - LizyML-Widget: `int` の次元に小数を入れる入力欄（`parseFloat`）。
  - 両製品: 文字列化した選択肢（`"15"`、`"true"`）は、書式上は正しい値なので H-0123 では拒否しない。smart の次元の文字列の真偽は #347 で別に扱う。
- docs/、notebooks/、README の探索空間の例に、書式の外の書き方は無い（grep で確かめた）。
- 公開 API と Result の形は変わらない。CHANGELOG に「書式の外の指定を拒否する」ことと、その例を書く。

### 代替案（検討して棄却）

- **警告だけを出す**: 探索は黙って変わったまま進む。
- **今受け付けている値を、値を変えずに読める限りすべて受け付け続ける**: Proposal の審査（round 1〜3）で、数の型の端（numpy の `longdouble`、`float` で無限大になる `Decimal`、独自のクラス、numpy の配列の選択肢）を一つずつ扱うことになり、閉じなかった。書式はそれらを意図していない（2026-10-12、管理者の判断で書式を正とした）。
- **YAML の読み方を YAML 1.2 に変えて `1e-4` を数にする**: config 全体の値の型が変わる（列名などに `1e5` と書いた設定の意味が変わる）。探索空間の境界に限って数の表記の文字列を受け付けるほうが、影響が小さい。

### 受け入れ基準（テスト観点）

今のコードで失敗し、変更で通るもの:

1. 目的（課題）の各形と、互換性の節に挙げた書式の外の各形が、config の `tuning.optuna.space` でも `tune(space=...)` でも、study を作る前に `CONFIG_INVALID` で拒否される（`lgb.train` が呼ばれない）。context に次元の名前、キー、理由、空間がある。
2. config の `tuning.optuna.space` に `dict` でない mapping（`UserDict`）で書いた `category: smart` の次元が、名前の検査で `model` の次元と誤読されずに受け付けられる。
3. config の `tuning.optuna.space` の mapping でない指定が、素の `AttributeError` ではなく `CONFIG_INVALID` で拒否される。
4. YAML の config ファイルに `low: 1e-4` と書いた探索次元が、`load_config` を通して `low=0.0001` の次元に解決される。`low: nan` や `low: "abc"` は拒否される。

実装の PR で確かめるもの:

5. 文書: BLUEPRINT の探索空間の節と docs/config-reference.md に、受け付ける書式（`choices` を含む）を書き、CHANGELOG に拒否されるようになった書き方を書く。`tests/test_docs/` で確かめる。
6. 負の対照: 方針 3 の各検査、方針 2 の写し、mapping でない指定の検査を、それぞれ一つずつ外すと、基準 1〜4 のうち対応するケースが失敗する。

今も通り、変更の後も通り続けるべきもの（回帰の保証）:

7. 書式どおりの指定（`log` と `category` の省略、`int` の次元の `16.0`、numpy の数の境界、list と tuple の `choices`）は、今と同じ次元に解決される。
8. `tune(space=...)` の H-0122 の拒否（形、category、名前）は、理由を変えずに引き続き拒否される（H-0122 の受け入れ基準 29 のテストが通る）。
