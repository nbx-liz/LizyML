# Proposal: PyPI 配布要件の明文化

## 目的

- PyPI 公開時に最低限必要な build 定義、配布メタデータ、README、optional dependency、CI 検証の要件を仕様として固定する。
- 「インストールできるが使い始められない」「README の import が壊れている」「extras が配布契約になっていない」といった初回公開時の事故を防ぐ。

## 影響範囲

- `BLUEPRINT.md` の要件定義
- `pyproject.toml` の build / project metadata / optional dependency 設計
- `README.md` の公開 API と利用例
- 配布前 CI

## 互換性

- 公開 API の shape は変更しない。
- 既存コードの実行挙動は変えず、配布契約とドキュメント契約を追加するのみ。
- 将来の初回公開および継続リリース時のチェック項目を増やす。

## 代替案

- 実装時に都度判断し、仕様には書かない。
- `README` や `pyproject.toml` を個別に直すが、配布要件として固定しない。

却下理由:

- 担当者ごとの判断に依存し、リリース品質がぶれる。
- PyPI 公開面は利用者との契約なので、都度判断ではなく仕様化が必要。

## 受け入れ基準

- `BLUEPRINT.md` に PyPI 配布要件（build-system、project metadata、optional dependencies、README 整合、CI 検証）が追加されている。
- 配布利用者向け依存と開発依存の分離方針が仕様として読める。
- README と公開 API の整合性を検証対象に含めることが明記されている。
