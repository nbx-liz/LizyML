"""The documents state the H-0125 decisions (criterion 11; #307, #313, #354).

Each check reads the shipped document's section by its heading and pins the
statement H-0125 made it carry, so a revert to the old wording fails here:

* #354: the learning curve is LightGBM's names and definitions, and the names
  whose values differ from ``FitResult.metrics`` are stated. Those names are
  not copied into this test: each statement's backticked metric names must
  equal the keys of the plot's annotation table
  (``lizyml.plots.learning_curve.LIGHTGBM_CURVE_DEFINITIONS``), which
  ``tests/test_plots/test_learning_curve_definitions.py`` pins to LightGBM's
  measured values (DC3).
* #307: ``cross_entropy_lambda`` is refused as a binary objective
  (``not_a_probability``) and an artifact trained with it is refused at load;
  the objective lists are read from ``LGBMProvider.objective_choices`` and the
  example message from the refusal itself.
* #313: parameters on a LightGBM native metric are refused.
* Policy 3: ``tune`` checks objective and metric values before the study
  (``not_categorical``), and the direct ``Tuner`` does not check them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.estimators.lgbm.param_validation import (
    NOT_A_PROBABILITY_OBJECTIVES,
    check_objective_compatible,
)
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.metrics.registry import _TASK_METRICS
from lizyml.plots.learning_curve import LIGHTGBM_CURVE_DEFINITIONS

_ROOT = Path(__file__).resolve().parents[2]
_TABLE_NAMES = frozenset(LIGHTGBM_CURVE_DEFINITIONS)
_METRIC_NAMES = frozenset().union(*_TASK_METRICS.values())
_REFUSED = sorted(frozenset().union(*NOT_A_PROBABILITY_OBJECTIVES.values()))
_BACKTICKED = re.compile(r"`([^`]+)`")


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin + len(start))]


def _lines_with(section: str, *needles: str) -> list[str]:
    """The lines of *section* that contain every needle."""
    return [line for line in section.splitlines() if all(n in line for n in needles)]


def _one_line(section: str, *needles: str) -> str:
    lines = _lines_with(section, *needles)
    assert len(lines) == 1, (
        f"expected one line with {needles}, found {len(lines)}: {lines}"
    )
    return lines[0]


def _metric_names(line: str) -> set[str]:
    """The LizyML metric names a line writes as code spans (full token, DC2)."""
    return {token for token in _BACKTICKED.findall(line) if token in _METRIC_NAMES}


def _has_token(text: str, token: str) -> bool:
    return (
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text)
        is not None
    )


def test_the_table_and_the_refused_objectives_were_read() -> None:
    assert _TABLE_NAMES, "the annotation table is empty; the checks read nothing"
    assert _REFUSED == ["cross_entropy_lambda"]


# ---------------------------------------------------------------------------
# BLUEPRINT.md
# ---------------------------------------------------------------------------


def _blueprint(start: str, end: str) -> str:
    return _section(_read("BLUEPRINT.md"), start, end)


def test_blueprint_history_states_lightgbm_definitions() -> None:
    """§7.1 ``history``: LightGBM's names and definitions, and which names differ."""
    line = _one_line(_blueprint("## 7.1 ", "## 7.2 "), "`eval_history`", "LightGBM")
    assert "`FitResult.metrics`" in line
    assert _metric_names(line) == _TABLE_NAMES


def test_blueprint_metric_items_state_the_learning_curve_definition() -> None:
    """§13.1: each differing metric's item names LightGBM's definition."""
    section = _blueprint("## 13.1 Metrics", "### 13.1.1")
    for name in sorted(_TABLE_NAMES):
        item = _one_line(section, f"  - `{name}`:")
        assert "LightGBM" in item and "学習曲線" in item, name
    statement = _one_line(section, "学習曲線", "LightGBM の名前と定義")
    assert _metric_names(statement) == _TABLE_NAMES


def test_blueprint_probability_item_is_the_objective_and_load_refusal() -> None:
    """§13.1 「確率の検査」: rewritten from the evaluator outcome to the refusals."""
    item = _one_line(_blueprint("## 13.1 Metrics", "### 13.1.1"), "**確率の検査**")
    for token in [*_REFUSED, "not_a_probability", "DESERIALIZATION_FAILED"]:
        assert _has_token(item, token), token
    assert "Model.load" in item
    # The old statement: the objective's output reaching METRIC_REQUIRES_PROBA.
    for sentence in item.split("。"):
        assert not (
            "cross_entropy_lambda" in sentence and "METRIC_REQUIRES_PROBA" in sentence
        ), sentence


def test_blueprint_plot_learning_curve_annotates_titles() -> None:
    """§13.3: the subplot title carries LightGBM's definition for the table's names."""
    line = _one_line(_blueprint("## 13.3 ", "## 13.4 "), "- `plot_learning_curve(")
    assert "subplot" in line and "LightGBM" in line
    assert _metric_names(line) == _TABLE_NAMES
    for definition in LIGHTGBM_CURVE_DEFINITIONS.values():
        assert definition in line, definition


def test_blueprint_metric_bridge_states_curve_definitions_and_param_refusal() -> None:
    """§14 Metric Bridge: learning curve = LightGBM; native metric params refused."""
    section = _blueprint("#### Metric Bridge", "### 共通デフォルト")
    statement = _one_line(section, "学習曲線", "LightGBM の名前と定義")
    assert "`FitResult.metrics`" in statement
    assert _metric_names(statement) == _TABLE_NAMES
    refusal = _one_line(section, "native", "`CONFIG_INVALID`", "`evaluation.metrics`")
    assert "`alpha`" in refusal


def test_blueprint_tuning_checks_values_and_tuner_does_not() -> None:
    """§11: the pre-study value check, and the direct ``Tuner`` out of scope."""
    section = _blueprint("# 11. Tuning", "# 12. Calibration")
    check = _one_line(section, "`not_categorical`")
    for token in ("check_round_space", "not_a_probability", "CONFIG_INVALID"):
        assert _has_token(check, token), token
    tuner = _one_line(section, "`Tuner`", "母数の値を検査しない")
    assert "`CONFIG_INVALID`" in tuner and "試行の失敗" in tuner


# ---------------------------------------------------------------------------
# docs/config-reference.md
# ---------------------------------------------------------------------------


def _config_reference(start: str, end: str) -> str:
    return _section(_read("docs/config-reference.md"), start, end)


def _table_row(section: str, first_cell: str) -> list[str]:
    for line in section.splitlines():
        # A cell may hold an escaped pipe (``\|``), which is not a separator.
        cells = [
            cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        ]
        if cells and cells[0] == first_cell:
            return cells
    raise AssertionError(f"no row {first_cell} in the section")


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_config_reference_objective_table_is_the_provider_list(task: str) -> None:
    section = _config_reference("| task | valid objectives |", "Some objectives")
    cell = _table_row(section, f"`{task}`")[1]
    assert _BACKTICKED.findall(cell) == list(LGBMProvider().objective_choices(task))


def test_config_reference_keeps_the_metric_cross_entropy_lambda() -> None:
    section = _config_reference("### Training metric reference", "**LizyML name")
    assert _has_token(_table_row(section, "`binary`")[1], "cross_entropy_lambda")


def test_config_reference_states_the_objective_refusal() -> None:
    section = _config_reference("| task | valid objectives |", "### Training metric vs")
    line = _one_line(section, "`cross_entropy_lambda`", "`not_a_probability`")
    assert "`DESERIALIZATION_FAILED`" in line


def test_config_reference_training_vs_evaluation_metric() -> None:
    """Native metric params are refused; the learning curve is LightGBM's."""
    section = _config_reference(
        "### Training metric vs evaluation metric", "### Training metric reference"
    )
    refusal = _one_line(section, "native", "`CONFIG_INVALID`", "`evaluation.metrics`")
    assert "`alpha`" in refusal
    curve = _one_line(section, "learning curve", "LightGBM's names and definitions")
    assert "`FitResult.metrics`" in curve
    assert _metric_names(curve) == _TABLE_NAMES


def test_config_reference_formula_rows_name_the_lightgbm_definition() -> None:
    section = _config_reference("### Metric formula reference", "**MAPE vs sMAPE")
    for name in sorted(_TABLE_NAMES):
        notes = _table_row(section, f"`{name}`")[-1]
        assert "LightGBM" in notes and "learning curve" in notes, name


# ---------------------------------------------------------------------------
# docs/api.md
# ---------------------------------------------------------------------------


def test_api_history_and_plot_learning_curve() -> None:
    text = _read("docs/api.md")
    history = _table_row(
        _section(text, "### FitResult", "### PredictionResult"), "`history`"
    )[-1]
    plot = _table_row(
        _section(text, "### Diagnostic Methods", "### Properties"),
        "`plot_learning_curve()`",
    )[-1]
    for cell in (history, plot):
        assert "LightGBM" in cell
        assert _metric_names(cell) == _TABLE_NAMES
    assert "`FitResult.metrics`" in history


def test_api_tune_checks_values_and_tuner_does_not() -> None:
    section = _section(_read("docs/api.md"), "#### `tune`", "#### `restore_tuning`")
    check = _one_line(section, "`not_categorical`")
    assert "objective" in check and "metric" in check
    tuner = _one_line(section, "`lizyml.tuning.Tuner`", "does not validate")
    assert "`CONFIG_INVALID`" in tuner and "failed trial" in tuner


def test_api_load_refuses_the_objective() -> None:
    section = _section(
        _read("docs/api.md"), "#### `load` (classmethod)", "### Diagnostic"
    )
    raises = section[
        section.index("**Raises:** `LizyMLError(DESERIALIZATION_FAILED)`") :
    ]
    line = _one_line(raises, "`cross_entropy_lambda`", "booster")
    assert "not a probability" in line


# ---------------------------------------------------------------------------
# docs/DEPRECATIONS.md
# ---------------------------------------------------------------------------


def test_deprecations_example_message_is_the_refusal() -> None:
    """The H-0079 example prints what the refusal says today."""
    section = _section(
        _read("docs/DEPRECATIONS.md"),
        '### `LGBMConfig.params["objective"]` cross-task injection',
        "\n### ",
    )
    lines = section.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("# LizyMLError: ")
    )
    shown = " ".join(
        line.removeprefix("# LizyMLError: ").removeprefix("# ")
        for line in lines[start : start + 2]
    )
    with pytest.raises(LizyMLError) as exc_info:
        check_objective_compatible("binary", "regression")
    assert shown == exc_info.value.user_message
    assert not _has_token(shown, "cross_entropy_lambda")


def test_deprecations_record_the_objective_removal() -> None:
    text = _read("docs/DEPRECATIONS.md")
    schedule = _section(text, "## Schedule", "## Migration notes")
    row = _one_line(schedule, "`cross_entropy_lambda`")
    assert "v0.19.0" in row and "H-0125" in row
    note = _section(text, "### `cross_entropy_lambda` binary objective", "\n### ")
    for needed in (
        "v0.19.0",
        "probability",
        "`binary`",
        "`cross_entropy`",
        "`DESERIALIZATION_FAILED`",
        "Model.load",
    ):
        assert needed in note, needed


# ---------------------------------------------------------------------------
# CHANGELOG.md
# ---------------------------------------------------------------------------


def _unreleased(subsection: str) -> list[str]:
    unreleased = _section(_read("CHANGELOG.md"), "## [Unreleased]", "\n## [")
    body = _section(unreleased + "\n### ", f"### {subsection}\n", "\n### ")
    return [line for line in body.splitlines() if line.startswith("- ")]


def _entry(entries: list[str], *needles: str) -> str:
    found = [entry for entry in entries if all(n in entry for n in needles)]
    assert len(found) == 1, f"expected one entry with {needles}, found {len(found)}"
    return found[0]


_ISSUE = "https://github.com/nbx-liz/LizyML/issues/{}"


def test_changelog_changed_entries() -> None:
    entries = _unreleased("Changed -- action may be required")
    objective = _entry(entries, _ISSUE.format(307), "`cross_entropy_lambda`")
    assert (
        "`DESERIALIZATION_FAILED`" in objective and "`not_a_probability`" in objective
    )
    params = _entry(entries, _ISSUE.format(313), "native")
    assert "`export_code`" in params and "predict" in params
    tune = _entry(entries, "`not_categorical`")
    assert _ISSUE.format(307) in tune or _ISSUE.format(313) in tune
    assert "merge" in tune


def test_changelog_documentation_entry() -> None:
    entry = _entry(_unreleased("Documentation"), _ISSUE.format(354))
    assert "learning curve" in entry and "title" in entry
    assert _metric_names(entry) == _TABLE_NAMES
