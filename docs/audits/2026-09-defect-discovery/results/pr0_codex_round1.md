# PR 0 — Codex review, round 1 (2026-09-06)

**Gate**: independent Codex review of PR 0 (`fix/phase3-pr0-spec-contradictions`,
base `origin/develop` = `3abb6c4`), the first Phase 3 pull request. The user's
merge gate for this run is *Codex APPROVE + CI green*.

**Provenance**: the run's scratchpad under `/tmp/claude-1000/.../scratchpad/` was
lost before this record was written (the third occurrence of that class). The
verdict below is **recovered verbatim** from the session transcript by
`instruments/recover_from_transcript.py --grep`, not reconstructed from memory.
The review *prompt* was not recovered and is not reproduced here.

---

## Verdict

```
VERDICT: REQUEST_CHANGES
```

### Findings

- **blocking** — §10.3.3 の「回帰では `TimeHoldoutInnerValid` と同等」は実装上偽であり、4未満の group への production fallback は回帰 `y` をクラスとして扱うため、連続値が一意なら全行を validation にして train を空にする（実測: 6行すべて valid。一方 `TimeHoldoutInnerValid(ratio=.3)` は末尾1行のみ valid）；仕様を真にするには source change と回帰テストが必要である。`BLUEPRINT.md:664`, `lizyml/training/inner_valid.py:323`, `lizyml/training/inner_valid.py:247`

- **major** — version test の DC1 ガードは現在7サイトあるのに `MIN_SITES=4` と誤って基準化されており、今回の中心である `ARCHITECTURE.md` の3サイトがすべて走査不能になっても残り4サイトで通るため、「クラスを閉じる」「scan stopped looking を検知する」という主張を満たさない。`tests/test_docs/test_declared_versions.py:73`

### Checked and clean

- §10.3.3 の5 constructor signature/default、holdout/group/time の計数規則、time holdout の `n_valid + gap >= n_samples` 条件は実装と一致。
- gap の scoping は正しい。auto 経路だけが `_auto_inner_gap` を渡し、明示 `time_holdout` は constructor default の `gap=0` を使う。
- §8.2 は §10.3.1 と整合し、明示的な shuffled holdout に実装が実際に `UserWarning` を発して設定を尊重する。
- H-0092 は目的・影響範囲・互換性・代替案・受け入れ基準を備え、追加された2テスト群への対応も明記している。
- RED claim は再現可能。変更前の `ARCHITECTURE.md` の3行はいずれも scanner に一致する `format_version=1` で、コード値2に対して失敗する。
- `HISTORY.md`、`CHANGELOG.md`、`PLAN.md`、監査アーカイブの除外理由自体は妥当。regex の単語境界も DC2 上問題なし。
- `git blame` の commit/date claims を確認済み。
- focused tests は **34 passed**。ただし上記 regression production-path ケースは既存テストにない。
- DC1–DC7 を確認し、上記 DC1/DC5 相当以外に DC2、DC3、DC4、DC6、DC7 の追加問題は認めなかった。

---

## Disposition (main context)

Both findings were **upheld**, the blocking one by execution rather than by
reading:

```
before the fix
  TimeHoldoutInnerValid(ratio=.3)            train=[0, 1, 2, 3, 4] valid=[5]
  StratifiedTimeHoldoutInnerValid(ratio=.3)  train=[]              valid=[0..5]
  BlockedGroupInnerValid(task='regression')  train=[]              valid=[0..5]
  fallback leaves an EMPTY inner-train?      True

after the fix
  StratifiedTimeHoldoutInnerValid(ratio=.3)  refused: ValueError
  BlockedGroupInnerValid(task='regression')  train=[0, 1, 2, 3, 4] valid=[5]
  direct stratified call on continuous y refuses?   True
  production fallback == TimeHoldout for regression? True
  fallback leaves an EMPTY inner-train?              False
```

| Finding | Remedy |
|---|---|
| blocking | `BlockedGroupInnerValid` dispatches its <4-group fallback on `task` (regression → `TimeHoldoutInnerValid`); `StratifiedTimeHoldoutInnerValid` raises `ValueError` when per-class tail selection would empty inner-train. BLUEPRINT §10.3.3 **and** §10.6.2 both state the `task`-dependent fallback; H-0092 gains decision 6 and drops its 「実装コードの変更なし」 claim. Three regression tests in `TestRegressionFallbackIsTimeOrdered`. |
| major | `MUST_CONTRIBUTE = ("ARCHITECTURE.md", "BLUEPRINT.md")` — a per-document guard, not a count — plus `MIN_SITES` raised 4 → 7. |

Note the shape of the blocking finding: PR 0 set out to make documents agree
with the implementation, and this one clause could not be repaired that way,
because the implementation was the wrong side. Writing the specification down
precisely is what exposed it.
