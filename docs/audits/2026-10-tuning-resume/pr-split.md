# H-0122 の実装の分け方（2026-10-11、管理者が二つの PR に分けることを承認）

H-0122（HISTORY.md）を二つの PR で実装する。どちらも v0.19.0 の前に入れる。PR 1 は `Refs #340`、PR 2 が `Fixes #340`。H-0122 は PR 2 で Accepted にする。

## PR 1（中核）

節: 3（呼び出しの表と事前の検査。`space=` の行を除く）、4.1・4.1b・4.3（次元の経路は round の記録に書く。カテゴリーの変更の拒否は `space=` が無いと起きないので PR 2）、5.1 の config の探索空間の変更の拒否（`space_mode` を含む）、5.3 の 3 点目（前回の最良値と `fixed` を study から読む）、6.1〜6.4、6.5（報告面の round の区切りを除く）、7.1、7.2（U2 は `n_trials` と `expand_boundary` だけ）、7.4、8（下の PR 2 の行を除く）。

U2 の `n_trials` と `expand_boundary` を PR 1 に入れる理由: PR 1 は途中の round の再開（7.2）を入れる。比べずに記録の値を使うと、渡した引数が黙って無視される（DC1）。`space=` との比べ合わせは、`space=` を足す PR 2 で入れる。

受け入れ基準: 1、2、3、5、8、8a、8b（`n_trials`・`expand_boundary`・config の `n_trials` の警告）、9、10、11、12（カテゴリーの変更を除く）、13、14（`space=` の行を除く）、15（空間を変えない resume）、16、18、19（`space=` を渡す部分を除く）、21、22（PR 1 にある拒否の行）、23、24、25、26、27、28、30（2・3 点目）、31、32、33。

H-0121 の一時的な固定（`tests/test_tuning/test_smart_dimension_admission.py` の再接続の経路）は、基準 9 の拒否のテストに書き直す（消さない）。

## PR 2

節: 5（`space=`、次元ごとの規則、拡張の変更）、U2 の `space=`、7.6 `read_study_lineage`、7.7 `restore_tuning`、6.5 の報告面の round の区切り、文書（BLUEPRINT §11、docs/api.md、CHANGELOG、docs/migration.md）。

受け入れ基準: 4、6、6b、7、8b（`space=`）、12（カテゴリーの変更）、14（`space=` の行）、15（`space=` で変えた resume）、16a、17、19（`space=` を渡す部分）、20、22（PR 2 で足す拒否の行）、29、30（1 点目）、34、35、36。

基準 6・6b の経路（6.4）の実装は PR 1 に入る。次元を外せるのは `space=` だけなので、外した次元での確かめは PR 2。PR 1 では系譜を直接組んだ単体のテストで確かめる。

## PR 1 と PR 2 の間の状態

PR 1 だけが入った develop では、報告面の round の区切り（16a）は `RoundSummary.n_trials` の積み上げのままで、`failed` の round や時間で区切った round があると区切りがずれる。文書は PR 2 で直すので、その間は enqueue の記述などが古い。どちらもリリースの前に PR 2 で閉じる。

## Optuna の前提（実行で確認、`probes/optuna_facts.py`、Optuna 4.7.0）

- 死んだプロセスが残した RUNNING の試行は、別のプロセスで `load_study` しても RUNNING のまま（読むだけでは変わらない）。`study.tell(number, state=FAIL)` で FAIL にでき、そのとき `datetime_complete` はその時点の時刻になる。
- PRUNED の試行にも `datetime_start` と `datetime_complete` が入る。
- メモリ上の study の `user_attrs` は pickle を通る。
- `optimize(timeout=0)` は試行を一つも始めない。
