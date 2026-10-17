"""H-0132 (criterion 9): the documents state where MASE's scale comes from.

H-0125's sentence about learning-curve definitions concerns LightGBM's native
metrics and is pinned by ``test_h0125_docs.py``. MASE is a feval with the same
formula in both places, but a different scale population, so its own item
says so; this test pins that item and the reference documents.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    head = text.index(start)
    return text[head : text.index(end, head)]


def test_blueprint_mase_item_states_both_scale_populations() -> None:
    section = _section(_read("BLUEPRINT.md"), "## 13.1 Metrics", "### 13.1.1")
    lines = [line for line in section.splitlines() if "式は同じで" in line]
    assert len(lines) == 1
    line = lines[0]
    assert "学習曲線は booster が学習する行（inner-train）の scale" in line
    assert "`FitResult.metrics` は fold の outer-train の scale" in line
    assert "  - `mase`:" in section and "needs_train_target=True" in section


def test_blueprint_feval_section_names_the_binding_route() -> None:
    text = _read("BLUEPRINT.md")
    assert "| `mase` | ✅ | | |" in text
    assert (
        "eval set のラベル（`dataset.get_label()`、評価する行）から scale を作らない"
        in text
    )


def test_config_reference_lists_mase_with_its_scale() -> None:
    text = _read("docs/config-reference.md")
    assert "| `mase` | ✅ | | |" in text
    assert (
        "`rmse`, `mae`, `r2`, `rmsle`, `mape`, `huber`, `smape`, `wape`, `mase`" in text
    )
    row = next(
        line for line in text.splitlines() if line.startswith("| `mase` (H-0132)")
    )
    assert "the two can differ" in row and "two validation folds" in row


def test_architecture_states_the_binding_contract() -> None:
    text = _read("ARCHITECTURE.md")
    assert "+bind_train(y_train) BaseMetric" in text
    assert "+needs_train_target" in text
