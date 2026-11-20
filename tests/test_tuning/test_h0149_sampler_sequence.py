"""H-0149 sampler wiring, warnings and seeded sequences; no model training."""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Any

import optuna
import pytest

from lizyml.config.schema import LizyMLConfig
from lizyml.core._tuning_identity import build_identity
from lizyml.core._tuning_resume import check_against_record
from lizyml.tuning import tuner
from tests._helpers import make_config
from tests.test_tuning._lineage_fixtures import stored_study

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_PATH = ROOT / "docs/audits/2026-11-i419/capture_sequences.py"
_spec = importlib.util.spec_from_file_location("h0149_capture", CAPTURE_PATH)
assert _spec is not None and _spec.loader is not None
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)

MAJOR_SENTENCE = (
    "Optuna major versions differ, so TPE sampler behavior changes and, even "
    "with the same seed, subsequent trials may differ from continuing with "
    "the recorded version."
)


@pytest.mark.parametrize("site", ["load_stored_study", "create_study", "Tuner.tune"])
def test_three_sites_construct_explicit_tpe(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    storage = optuna.storages.InMemoryStorage()
    optuna.create_study(storage=storage, study_name="stored")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original = optuna.samplers.TPESampler

    class RecordingSampler(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(optuna.samplers, "TPESampler", RecordingSampler)
    if site == "load_stored_study":
        study = tuner.load_stored_study(storage, "stored", seed=19)
    elif site == "create_study":
        study = tuner.create_study("minimize", seed=19)
    else:
        _, study = tuner.Tuner([], n_trials=1, seed=19).tune(lambda trial: 0.0)
    assert calls == [((), {"seed": 19, "multivariate": True, "constant_liar": True})]
    assert isinstance(study.sampler, RecordingSampler)


@pytest.mark.parametrize(
    "case",
    [
        "different-major",
        "same-major",
        "unreadable-stored",
        "unreadable-current",
        "both-unreadable",
        "missing-stored",
        "missing-current",
        "leading-integer",
        "equal-leading-integer",
        "other-library",
        "identical",
    ],
)
def test_resume_version_warning_sentence(case: str) -> None:
    """Exercise the production message builder directly with matching identities."""
    current = optuna.__version__
    major = int(current.split(".")[0])
    stored_versions = {"optuna": current}
    current_versions = {"optuna": current}
    if case == "different-major":
        stored_versions["optuna"] = f"{major + 1}.0.0"
    elif case == "same-major":
        stored_versions["optuna"] = f"{major}.999.0"
    elif case == "unreadable-stored":
        stored_versions["optuna"] = "unknown"
    elif case == "unreadable-current":
        current_versions["optuna"] = "v5.0.0"
    elif case == "both-unreadable":
        stored_versions["optuna"] = ""
        current_versions["optuna"] = "unknown"
    elif case == "missing-stored":
        del stored_versions["optuna"]
    elif case == "missing-current":
        del current_versions["optuna"]
    elif case == "leading-integer":
        stored_versions["optuna"] = f"0{major + 1}rc1"
    elif case == "equal-leading-integer":
        stored_versions["optuna"] = f"0{major}rc1"
    elif case == "other-library":
        stored_versions["numpy"] = "old"
        current_versions["numpy"] = "new"

    config = make_config("binary", tuning_n_trials=1)
    config["tuning"]["optuna"].update(space_mode="replace", space={})
    cfg = LizyMLConfig.model_validate(config)
    identity = build_identity({}, {})
    record = stored_study(
        objective_elements=identity.elements,
        objective_digest=identity.digest,
        versions=stored_versions,
        origin_space=(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_against_record(cfg, record, identity, current_versions, "example")
    if case == "identical":
        assert caught == []
        return
    changes = ", ".join(
        f"{key} {stored_versions.get(key)} -> {current_versions.get(key)}"
        for key in sorted(stored_versions.keys() | current_versions.keys())
        if stored_versions.get(key) != current_versions.get(key)
    )
    expected = (
        f"study 'example' was created with other library versions ({changes}). "
        "The objective identity matches, so the study continues."
    )
    if case in {"different-major", "leading-integer"}:
        expected += " " + MAJOR_SENTENCE
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert str(caught[0].message) == expected


def test_make_tpe_sampler_emits_no_experimental_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler = tuner.make_tpe_sampler(19)
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert caught == []


@pytest.mark.parametrize("category", [UserWarning, RuntimeWarning, DeprecationWarning])
def test_factory_filters_only_experimental_during_construction(
    monkeypatch: pytest.MonkeyPatch, category: type[Warning]
) -> None:
    original = optuna.samplers.TPESampler

    class WarningSampler(original):
        def __init__(self, **kwargs: Any) -> None:
            warnings.warn(
                "experimental construction",
                optuna.exceptions.ExperimentalWarning,
                stacklevel=2,
            )
            warnings.warn("other construction", category, stacklevel=2)
            super().__init__(**kwargs)

    monkeypatch.setattr(optuna.samplers, "TPESampler", WarningSampler)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuner.make_tpe_sampler(19)
        warnings.warn(
            "after construction", optuna.exceptions.ExperimentalWarning, stacklevel=2
        )
    assert [(warning.category, str(warning.message)) for warning in caught] == [
        (category, "other construction"),
        (optuna.exceptions.ExperimentalWarning, "after construction"),
    ]


@pytest.fixture(scope="module")
def golden_data() -> dict[str, Any]:
    return json.loads(capture.GOLDEN.read_text(encoding="utf-8"))


def test_seeded_default_space_sequence_matches_band(
    golden_data: dict[str, Any],
) -> None:
    band = capture.select_band(golden_data, optuna.__version__)
    actual = capture.build_sequence()
    assert golden_data["seed"] == capture.SEED
    assert golden_data["n_trials"] == len(actual) == 40
    assert golden_data["float_significant_digits"] == 12
    assert actual == band["sequence"]
    for params, expected in zip(actual, band["sequence"], strict=True):
        assert {key: type(value) for key, value in params.items()} == {
            key: type(value) for key, value in expected.items()
        }


@pytest.mark.parametrize(
    "version", ["3.0.0", "3.7.0", "4.8.0", "5.2.0", "6.0.0", "unknown"]
)
def test_unknown_optuna_minor_fails(golden_data: dict[str, Any], version: str) -> None:
    with pytest.raises(AssertionError, match=r"capture_sequences\.py VERSION=PYTHON"):
        capture.select_band(golden_data, version)


def test_univariate_negative_control_matches_no_band(
    golden_data: dict[str, Any],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=capture.SEED, multivariate=False, constant_liar=True
        )
    actual = capture.build_sequence(sampler)
    assert len(actual) == 40
    assert all(actual != band["sequence"] for band in golden_data["bands"])
