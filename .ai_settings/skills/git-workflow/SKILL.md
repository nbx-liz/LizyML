---
name: git-workflow
description: >
  ブランチ管理・コミット規約・Push/PRルールを統一するスキル。「ブランチをどう切るか」「コミットメッセージの形式」「PRの作り方」「developとmainの使い分け」といった要求が出たときに使用する。トリガー例: "ブランチ", "コミット", "PR", "push", "develop", "merge", "Conventional Commits"
---

# スキルの具体的な指示 (Instructions)

## ブランチ構成

```
main          ← リリース可能な安定版。直接コミット禁止。
  └── develop ← 統合ブランチ。PRのマージ先。直接コミット禁止。
        ├── feat/<topic>      ← 機能追加
        ├── fix/<topic>       ← バグ修正
        ├── refactor/<topic>  ← リファクタリング
        └── docs/<topic>      ← ドキュメント変更
```

- Phase 単位の実装は `feat/phase-N-<名称>` を使う（例: `feat/phase-2-config-specs`）。
- 全ての作業ブランチは `develop` から切り、`develop` へ PR を出す。
- `main` へのマージは `develop`（またはリリースブランチ）からのみ行い、Squash merge を使う。

## コミットメッセージ形式（Conventional Commits）

```
<type>(<scope>): <summary>

[body: 任意]
```

**type 一覧:**

| type | 用途 |
|------|------|
| `feat` | 新機能 |
| `fix` | バグ修正 |
| `test` | テスト追加・修正 |
| `refactor` | 動作変更なしの構造変更 |
| `docs` | ドキュメント（BLUEPRINT/HISTORY/PLAN等） |
| `chore` | ビルド・CI・依存管理 |
| `perf` | パフォーマンス改善 |

**scope 例（このプロジェクト固有）:**
`config`, `splitters`, `training`, `evaluation`, `calibration`, `persistence`, `estimators`, `features`, `metrics`, `plots`, `explain`, `exceptions`, `ci`

**コミット粒度:**
- 1コミット = 1つの意図（機能追加とそのテストは同一コミットに含めてよい）。
- WIP コミットは `develop` へのマージ前に `git rebase -i` で整理する。

## PR ルール

**タイトル:** コミットメッセージと同形式 `<type>(<scope>): <summary>`

**本文の必須項目:**
- 概要（何をしたか1〜3行）
- 該当 SKILL（例: `skills/training-cv-and-inner-valid/SKILL.md`）
- DoD 達成確認チェックリスト（下記参照）

**DoD チェックリスト（全PRで確認）:**
- `uv run ruff check .` クリーン
- `uv run mypy lizyml/` クリーン
- `uv run pytest` 通過
- Contract 変更時: ゴールデンテストの更新または追加
- split/leakage/calibration 変更時: 「落ちるべき例」テストの追加
- 公開 API / Contract / format_version 変更時: HISTORY.md への Proposal 先行記録

### 一括変更（sed / find-replace）後の確認

型・命名・パターンの一括置換後は、grep で残存がないことを必ず確認してからコミットする。
コミット後に CI で初めて発見するのは手戻りコストが高い。

```bash
# 例: bare np.ndarray の残存確認
grep -rn "np\.ndarray" lizyml/
# → 0件であること

# 例: 特定パターンの網羅確認（一般形）
grep -rn "<置換前パターン>" lizyml/
# → 0件であること（意図的に残したもの以外）
```

**マージ戦略:**
- `develop` へ: Squash merge（WIP コミットをまとめ履歴を整理する）
- `main` へ: Squash merge（リリース単位を1コミットで記録する。Merge commit を使うと main に独自の commit が生まれ、develop との分岐・CHANGELOG コンフリクトの原因になる）

## HISTORY.md Proposal との連動

以下の変更は、コミット前に HISTORY.md へ Proposal を追加してから実装する。
その場合、コミット順序を Proposal → 実装 → テストとする。

```
docs(history): add proposal P-XXXX for <変更対象>
feat(<scope>): implement <内容> (P-XXXX)
test(<scope>): add golden/leakage test for <内容> (P-XXXX)
```

## 完了手順（feature → develop → main）

ユーザーから「コミットして」「PRまで進めて」「ship して」等の指示があった場合、以下の手順を順番に実行する。

### Step 1: 品質ゲート確認

**ruff check / format は pre-commit hook で自動実行される（コミット時にブロック）。**

残りの mypy / pytest は手動で確認する:
```bash
uv run mypy lizyml/
uv run pytest tests/ --ignore=tests/test_notebooks/test_notebook_execution.py -q
```
- 全てクリーンであることを確認してから次へ進む。

### Step 2: feature ブランチでコミット
```bash
git add <対象ファイル>
git commit -m "<type>(<scope>): <summary>"
```
- 1コミット = 1つの意図。機能追加とそのテストは同一コミットでよい。
- コミットメッセージは英語、Conventional Commits 形式。
- pre-commit hook が `ruff check` + `ruff format --check` を自動実行する。失敗時はコミットがブロックされる。

### Step 3: feature ブランチを push
```bash
git push origin <feature-branch>
```

### Step 4: develop へ PR を作成
```bash
gh pr create --base develop --title "<type>(<scope>): <summary>" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points>

## Skills
<該当 SKILL 一覧>

## DoD
- [ ] `uv run ruff check .` clean
- [ ] `uv run mypy lizyml/` clean
- [ ] `uv run pytest` passing
- <追加の DoD 項目>

## Test plan
<テスト方法・結果>
EOF
)"
```
- PR 本文は英語で記載する。
- GitHub 上で **Squash merge** する。ローカル squash merge + 直 push はしない。

### Step 5: develop → main PR（リリース時のみ）
- ユーザーから明示的にリリース指示があった場合のみ実行する。
- リリースブランチを経由する（develop から直接 PR を作らない）:
  ```bash
  git checkout -b release/vX.Y.Z develop
  git push -u origin release/vX.Y.Z
  gh pr create --base main --head release/vX.Y.Z --title "release: vX.Y.Z"
  ```
- GitHub 上で **Squash merge** する。
  - Squash merge により main に独自の merge commit が生まれず、develop との分岐を防止する。
  - main の各 commit = リリース単位（1リリース1コミット）。個別 commit 履歴は develop に残る。
  - リリース後の main → develop 同期 PR は不要。

### 注意: やってはいけない操作
```bash
# NG: ローカルで develop に squash merge して push
git checkout develop && git merge --squash feat/... && git push origin develop

# OK: feature ブランチを push して PR を作る
git push origin feat/... && gh pr create --base develop
```

## Examples (使用例)

- ユーザー: 「Phase 2 の作業を始めたい」
  - 返答: `git switch develop && git pull && git switch -c feat/phase-2-config-specs` を提案
- ユーザー: 「FitResult の構造を変えたい」
  - 返答: まず HISTORY.md に Proposal を `docs(history): add proposal ...` でコミットし、その後実装へ進む手順を提示
- ユーザー: 「コミット・Push して PR まで進めて」
  - 返答: Step 1〜4 を順番に実行する。develop → main PR はリリース指示がない限り作成しない。
- ユーザー: 「PR を出したい」
  - 返答: DoD チェックリストを確認し、`gh pr create --base develop` で PR を作成する手順を提示

## Guidelines / Rules (厳守事項)

- `main` / `develop` への直接コミット・直接 push を行わない。
- ローカルでの squash merge + 直 push を行わない。統合は必ず GitHub PR 経由で行う。
- `--no-verify` でフックをスキップしない。
- PR 説明の「該当 SKILL」と「DoD 達成」の記載を省略しない。
- Contract / format_version 変更は Proposal なしに実装しない。
