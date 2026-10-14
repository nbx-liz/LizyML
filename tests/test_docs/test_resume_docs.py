"""The documents state the resume and reattach rules (H-0122, #340).

Acceptance criteria 20 (the migration note for studies created before the
change) and 34 (the API reference, BLUEPRINT §11, CHANGELOG and the migration
guide). Each check reads the shipped document and pins the statement H-0122
made it carry, so a revert to the old enqueue / ``load_if_exists`` wording
fails here. Lists the code owns (the ``StudyState`` values, the parameters of
the public signatures) are read from the code, not copied (DC3).
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lizyml import Model
from lizyml.tuning import read_study_lineage
from lizyml.tuning.study_lineage import StudyState

_ROOT = Path(__file__).resolve().parents[2]
_STATES: tuple[str, ...] = typing.get_args(StudyState)


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin + len(start))]


def _api_tune() -> str:
    return _section(_read("docs/api.md"), "#### `tune`", "#### `restore_tuning`")


def _api_restore() -> str:
    return _section(_read("docs/api.md"), "#### `restore_tuning`", "\n---\n")


def _api_lineage() -> str:
    return _section(_read("docs/api.md"), "## Study lineage", "## Error Codes")


def _blueprint_resume() -> str:
    """BLUEPRINT §11.5 up to §11.6: re-tune, storage and the H-0122 sections."""
    return _section(_read("BLUEPRINT.md"), "## 11.5 ", "## 11.6 ")


def _changelog_unreleased() -> str:
    return _section(_read("CHANGELOG.md"), "## [0.19.0]", "## [0.18.0]")


def _migration_v019() -> str:
    return _section(_read("docs/migration.md"), "## v0.19.0", "## v0.18.0")


def test_states_were_read() -> None:
    assert len(_STATES) == 9, (
        "StudyState lost its Literal form; the checks read nothing"
    )


# ---------------------------------------------------------------------------
# docs/api.md (criterion 34, bullets 1 and 2; criterion 36)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("func", "section"),
    [
        pytest.param(Model.tune, _api_tune, id="tune"),
        pytest.param(Model.restore_tuning, _api_restore, id="restore_tuning"),
        pytest.param(read_study_lineage, _api_lineage, id="read_study_lineage"),
    ],
)
def test_api_tables_list_every_parameter(
    func: Callable[..., Any], section: Callable[[], str]
) -> None:
    """Each parameter of the signature has a row: a new one is documented."""
    text = section()
    params = [p for p in inspect.signature(func).parameters if p != "self"]
    assert params
    missing = [p for p in params if f"| `{p}` |" not in text]
    assert not missing, f"undocumented parameters: {missing}"


def test_api_tune_signature_has_space() -> None:
    """The documented signature carries ``space`` with the code's annotation."""
    annotation = inspect.signature(Model.tune).parameters["space"].annotation
    text = _api_tune()
    assert f"space: {annotation} = None" in text
    assert "| `space` |" in text


@pytest.mark.parametrize(
    "stale", ["enqueue", "warm-start", "re-attaches to the persisted study"]
)
def test_api_tune_has_no_enqueue_or_old_reattach(stale: str) -> None:
    assert stale not in _api_tune()


def test_api_tune_reattaches_with_resume_true_and_creates_when_absent() -> None:
    text = _api_tune()
    assert "`resume=True`" in text
    assert "creates the study" in text


#: H-0122 §8, one anchor per new or changed condition of ``Model.tune``.
_TUNE_RAISES = [
    ("storage_without_name", "`storage` without `study_name`"),
    ("name_without_storage", "`study_name` without `storage`"),
    ("resume_false_onto_stored_study", "already holds"),
    ("target_not_the_held_study", "not the study this Model holds"),
    ("space_on_a_creating_call", "`space=` on a call that starts round 1"),
    ("resume_without_study_or_storage", "neither a study nor `storage`"),
    ("objective_identity_differs", "objective"),
    ("state_legacy", "created before"),
    ("state_unreadable", "cannot be read"),
    ("state_inconsistent", "disagree"),
    ("space_mode_changed", "`tuning.optuna.space_mode`"),
    ("config_space_changed", "`tuning.optuna.space`"),
    ("kind_log_choices_changed", "distribution"),
    ("category_changed", "category"),
    ("open_round_argument_differs", "recorded"),
    ("round_without_complete_trial", "no COMPLETE trial"),
    ("library_versions_differ", "library versions"),
    ("config_n_trials_on_open_round", "`tuning.optuna.params.n_trials`"),
    ("tuning_failed", "LizyMLError(TUNING_FAILED)"),
]


@pytest.mark.parametrize(
    "anchor", [a for _, a in _TUNE_RAISES], ids=[i for i, _ in _TUNE_RAISES]
)
def test_api_tune_raises_lists_each_new_condition(anchor: str) -> None:
    raises = _section(_api_tune(), "**Raises:**", "\n---\n")
    assert anchor in raises


def test_api_rounds_row_states_failed_rounds_are_omitted() -> None:
    table = _section(_read("docs/api.md"), "### TuningResult", "### TrialResult")
    row = next(line for line in table.splitlines() if line.startswith("| `rounds` |"))
    assert "failed" in row
    assert "round 2" in row
    assert "Empty tuple" not in row


def test_api_current_trial_counts_on_after_reattach() -> None:
    table = _section(_read("docs/api.md"), "### TuneProgressInfo", "\n---\n")
    row = next(
        line for line in table.splitlines() if line.startswith("| `current_trial` |")
    )
    assert "K+1" in row


@pytest.mark.parametrize("name", ["read_study_lineage", "StudyLineage", "RoundRecord"])
def test_api_documents_the_lineage_reader(name: str) -> None:
    assert name in _api_lineage()


@pytest.mark.parametrize("state", _STATES)
def test_api_lists_every_study_state(state: str) -> None:
    assert f'`"{state}"`' in _api_lineage()


def test_api_lineage_reader_is_read_only() -> None:
    assert "read-only" in _api_lineage()


def test_api_restore_tuning_states_its_refusals() -> None:
    text = _api_restore()
    assert "Model.restore_tuning" in text or "def restore_tuning" in text
    raises = text.split("**Raises:**", 1)[1]
    assert "LizyMLError(CONFIG_INVALID)" in raises
    assert "open" in raises
    assert "LizyMLError(TUNING_FAILED)" in raises
    assert "writes nothing" in text


# ---------------------------------------------------------------------------
# BLUEPRINT §11 (criterion 34, bullets 2 and 3; criterion 36)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "anchor",
    [
        "space=",
        "resume=True",
        "照合",
        "read_study_lineage",
        "StudyLineage",
        "RoundRecord",
        "restore_tuning",
        "読み取りだけ",
    ],
)
def test_blueprint_resume_states_h0122(anchor: str) -> None:
    assert anchor in _blueprint_resume()


@pytest.mark.parametrize("state", _STATES)
def test_blueprint_lists_every_study_state(state: str) -> None:
    assert f'`"{state}"`' in _blueprint_resume()


def test_blueprint_parameter_table_has_space() -> None:
    table = _section(_blueprint_resume(), "| パラメーター |", "\n\n")
    assert "| `space` |" in table


@pytest.mark.parametrize(
    "stale",
    [
        "load_if_exists=True",
        "完全後方互換",
        "`resume=True` で未 tune",
        "journal には保存されない",
    ],
)
def test_blueprint_resume_drops_the_replaced_rules(stale: str) -> None:
    assert stale not in _blueprint_resume()


def test_blueprint_resume_mentions_enqueue_only_to_say_it_is_gone() -> None:
    text = _blueprint_resume()
    assert text.count("enqueue") == text.count("enqueue しない")


def test_blueprint_current_trial_row_counts_on_after_reattach() -> None:
    section = _section(_read("BLUEPRINT.md"), "## 11.4 ", "## 11.5 ")
    row = next(
        line for line in section.splitlines() if line.startswith("| `current_trial` |")
    )
    assert "この round の何番目" in row
    assert "続きから" in row


# ---------------------------------------------------------------------------
# CHANGELOG [0.19.0] and docs/migration.md (criteria 20 and 34, bullet 4)
# ---------------------------------------------------------------------------


_RELEASE_NOTES = [
    pytest.param(_changelog_unreleased, id="CHANGELOG"),
    pytest.param(_migration_v019, id="migration"),
]


@pytest.mark.parametrize("section", _RELEASE_NOTES)
@pytest.mark.parametrize(
    "anchor",
    [
        # Criterion 20 (U1): old studies cannot be continued; paused jobs.
        "created before v0.19.0",
        "new `study_name`",
        "before upgrading",
        # Criterion 34, bullet 4.
        "`tune(space=...)`",
        "`resume=True`",
        "CONFIG_INVALID",
        "previous best",
    ],
)
def test_release_notes_state_h0122(section: Callable[[], str], anchor: str) -> None:
    assert anchor in section()


def test_changelog_fixes_342() -> None:
    fixed = _section(_changelog_unreleased(), "### Fixed", "\n### ")
    assert "#342" in fixed
