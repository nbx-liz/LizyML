"""The ``tune()`` call matrix and the checks against a study's record (H-0122).

* §3 -- which study a call targets (:func:`precheck_study_args`, which lives
  in ``lizyml.tuning.tuner`` and is re-exported here, :func:`resolve_call_row`,
  :func:`locate_study`), including the rule for
  "the same study" (:func:`same_study`).
* §7.4 -- the states that are refused outright (:func:`refuse_state`).
* §5.1 / §4.3 -- what a stored study's record must still agree with
  (:func:`check_against_record`).
* §7.1 -- the ``lizyml:study`` record a new study gets
  (:func:`study_record_payload`) and the encoded round record
  (:func:`round_record_payload`).

Every function here reads only. Loading a stored study writes nothing to it,
so every refusal raised here comes before the first write (INV-R4). The
attribute writes themselves happen in ``Model.tune`` phase E.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from lizyml.core._tuning_identity import (
    ObjectiveIdentity,
    mismatched_elements,
    version_differences,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.tuning import tuner as _tuner
from lizyml.tuning.lineage import (
    SCHEMA_VERSION,
    Lineage,
    StoredStudy,
    encode_round,
    encode_study,
)
from lizyml.tuning.lineage_codec import LineageCodecError, dim_signature
from lizyml.tuning.search_space import parse_space
from lizyml.tuning.tuner import precheck_study_args as precheck_study_args

if TYPE_CHECKING:
    from optuna.storages import BaseStorage

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core._tuning_plan import RoundPlan
    from lizyml.core.types.search_dim import SearchDim

TargetKind = Literal["continue", "reattach", "create_storage", "create_memory"]

_IN_MEMORY = "<in-memory>"
_NEW_STUDY = "start a new study (another study_name)"


def _shown(study_name: str | None) -> str:
    return study_name if study_name is not None else _IN_MEMORY


# ---------------------------------------------------------------------------
# §3 -- the call matrix
# ---------------------------------------------------------------------------


def same_study(
    storage: str | BaseStorage | None,
    study_name: str | None,
    remembered_storage: str | BaseStorage | None,
    remembered_name: str | None,
) -> bool:
    """Whether ``(storage, study_name)`` provably names the remembered study.

    Equal names, and either two equal URL strings or the very same
    ``BaseStorage`` object. Another spelling of the same backend is not proved
    the same, so it is not (§3, conservative). An in-memory study remembers
    ``(None, None)``, which no storage matches.
    """
    if storage is None or remembered_storage is None:
        return False
    if study_name != remembered_name:
        return False
    if isinstance(remembered_storage, str):
        return isinstance(storage, str) and storage == remembered_storage
    return storage is remembered_storage


def resolve_call_row(
    *,
    resume: bool,
    has_study: bool,
    storage: str | BaseStorage | None,
    study_name: str | None,
    remembered_storage: str | BaseStorage | None,
    remembered_name: str | None,
    round_number: int,
    remembered_known: bool,
) -> None:
    """The call-matrix rows decided without I/O (§3 rows 5 and 8).

    ``remembered_known`` is False for a Model pickled before H-0122, which
    never remembered its study's ``(storage, study_name)``: its ``None`` pair
    means "unknown", not "in-memory", and the row-5 refusal says so.
    """
    if resume and not has_study and storage is None:
        raise LizyMLError(
            code=ErrorCode.TUNING_FAILED,
            user_message=(
                "Cannot resume tuning: this Model holds no study and no storage "
                "was given. Run tune() first, or pass storage and study_name to "
                "resume a stored study."
            ),
            context={
                "resume": True,
                "round_number": round_number,
                "storage_given": False,
            },
        )
    if (
        resume
        and has_study
        and storage is not None
        and not same_study(storage, study_name, remembered_storage, remembered_name)
    ):
        held = (
            f"('{_shown(remembered_name)}'), and the storage / study_name given "
            "cannot be shown to name it (the same URL string or the same storage "
            "object, and the same name)"
            if remembered_known
            else "; this Model predates the study record (H-0122), so which "
            "storage and name its study has is unknown, and the storage / "
            "study_name given cannot be shown to name it"
        )
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"resume=True continues the study this Model holds {held}. Omit "
                "them to continue, or use a new Model to reattach to another "
                "study."
            ),
            context={
                "surface": "tuning.study",
                "study_name": study_name,
                "held_study_name": (
                    _shown(remembered_name) if remembered_known else None
                ),
                "held_study_known": remembered_known,
            },
        )


@dataclass(frozen=True)
class StudyTarget:
    """The study a call adds its round to (§3).

    ``study`` is ``None`` for a study this call still has to create.
    """

    kind: TargetKind
    study: Any | None = None
    announce_create: bool = False


def locate_study(
    *,
    resume: bool,
    current_study: Any | None,
    storage: str | BaseStorage | None,
    study_name: str | None,
    seed: int,
) -> StudyTarget:
    """Find the call's study, reading the storage only (§3 rows 1-4, 6, 7).

    Raises:
        LizyMLError: ``CONFIG_INVALID`` for ``resume=False`` onto a study the
            storage already holds (row 3), whatever its state.
    """
    if resume and current_study is not None:
        return StudyTarget("continue", current_study)
    if storage is None:
        return StudyTarget("create_memory")
    assert study_name is not None  # noqa: S101 -- precheck_study_args
    study = _tuner.load_stored_study(storage, study_name, seed)
    if study is None:
        return StudyTarget("create_storage", announce_create=resume)
    if not resume:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"study '{study_name}' already exists in this storage. Use "
                f"resume=True to continue it, or another study_name to start a "
                "new one."
            ),
            context={"surface": "tuning.study", "study_name": study_name},
        )
    return StudyTarget("reattach", study)


# ---------------------------------------------------------------------------
# §7.4 -- refused states
# ---------------------------------------------------------------------------


def refuse_state(lineage: Lineage, study_name: str | None, n_trials: int) -> None:
    """Refuse a study in state L (legacy), S (unreadable) or X (inconsistent)."""
    name = _shown(study_name)
    if lineage.state == "legacy":
        message = (
            f"study '{name}' has {n_trials} trials but no LizyML record "
            "(lizyml:study): it was created before this LizyML version or "
            "outside LizyML, so whether its trials observe this objective "
            f"cannot be checked. {_NEW_STUDY.capitalize()}."
        )
    elif lineage.state == "unreadable":
        message = (
            f"study '{name}' carries a LizyML record this LizyML cannot read "
            f"({lineage.reason}). It may come from a newer LizyML, or have "
            f"been edited; {_NEW_STUDY}."
        )
    elif lineage.state == "inconsistent":
        message = (
            f"LizyML's record and the trials of study '{name}' disagree "
            f"({lineage.reason}); {_NEW_STUDY}."
        )
    else:
        return
    raise LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=message,
        context={
            "surface": "tuning.study",
            "state": lineage.state,
            "study_name": name,
            "n_trials": n_trials,
        },
    )


# ---------------------------------------------------------------------------
# §5.1, §4.3 -- agreement with the record
# ---------------------------------------------------------------------------


def _space_by_key(dims: tuple[SearchDim, ...] | list[SearchDim]) -> dict[Any, str]:
    """Dims keyed by ``(category, name)``, compared through their stored form.

    The encoded form keeps ``True`` / ``1`` / ``1.0`` apart and makes NaN equal
    to NaN, which dataclass equality would not.
    """
    return {(d.category, d.name): dim_signature(d) for d in dims}


def _check_config_space(cfg: LizyMLConfig, record: StoredStudy, name: str) -> None:
    assert cfg.tuning is not None  # noqa: S101 -- validated upstream
    optuna_cfg = cfg.tuning.optuna
    unchangeable = (
        "The space a study was created with cannot be changed through the config; "
        f"{_NEW_STUDY}."
    )
    if optuna_cfg.space_mode != record.space_mode:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"tuning.optuna.space_mode is '{optuna_cfg.space_mode}', but study "
                f"'{name}' was created with '{record.space_mode}'. {unchangeable}"
            ),
            context={
                "surface": "tuning.optuna.space_mode",
                "study_name": name,
                "recorded": record.space_mode,
                "given": optuna_cfg.space_mode,
            },
        )
    try:
        current = _space_by_key(parse_space(optuna_cfg.space))
        recorded = _space_by_key(record.origin_space)
    except LineageCodecError as exc:  # pragma: no cover -- parse_space refuses first
        raise _codec_refusal(exc, "tuning.optuna.space") from exc
    if current != recorded:
        changed = sorted(
            f"{key[0]}:{key[1]}"
            for key in set(current) | set(recorded)
            if current.get(key) != recorded.get(key)
        )
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"tuning.optuna.space differs from the space study '{name}' was "
                f"created with (changed: {', '.join(changed)}). {unchangeable}"
            ),
            context={
                "surface": "tuning.optuna.space",
                "study_name": name,
                "changed": changed,
            },
        )


def check_against_record(
    cfg: LizyMLConfig,
    record: StoredStudy,
    identity: ObjectiveIdentity,
    versions: Mapping[str, str],
    study_name: str | None,
) -> None:
    """Check a stored study's record against this call (§5.1, §4.3).

    In order: the config's ``space_mode`` and space against the study's
    origin (checked first so a ``space_mode`` edit, which also changes
    ``config.fixed``, gets its own message); the objective identity; then
    the library versions, which only warn (D4).
    """
    name = _shown(study_name)
    _check_config_space(cfg, record, name)
    mismatched = mismatched_elements(record.objective_elements, identity)
    if mismatched:
        raise LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message=(
                f"this call's objective differs from the one study '{name}' "
                f"observes (changed: {', '.join(mismatched)}). The trials of one "
                "study must all observe the same objective: data, the config "
                "outside the search space, the split and the first metric. "
                f"{_NEW_STUDY.capitalize()}."
            ),
            context={
                "surface": "tuning.study",
                "mismatched": list(mismatched),
                "study_name": name,
            },
        )
    differences = version_differences(record.versions, versions)
    if differences:
        changes = ", ".join(
            f"{d.package} {d.stored} -> {d.current}" for d in differences
        )
        warnings.warn(
            f"study '{name}' was created with other library versions ({changes}). "
            "The objective identity matches, so the study continues.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# §7.1 -- the records a call writes, encoded while planning
# ---------------------------------------------------------------------------


def _codec_refusal(exc: LineageCodecError, surface: str) -> LizyMLError:
    return LizyMLError(
        code=ErrorCode.CONFIG_INVALID,
        user_message=(
            f"{surface}: a value cannot be recorded in the study ({exc}). Only "
            "plain bool, int, float, str and None values can be stored."
        ),
        context={"surface": surface, "reason": str(exc)},
        cause=exc,
    )


def study_record_payload(
    cfg: LizyMLConfig,
    *,
    identity: ObjectiveIdentity,
    direction: Literal["minimize", "maximize"],
    versions: Mapping[str, str],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    """The encoded ``lizyml:study`` record for a new or uninitialized study."""
    assert cfg.tuning is not None  # noqa: S101 -- validated upstream
    try:
        return encode_study(
            StoredStudy(
                schema=SCHEMA_VERSION,
                objective_digest=identity.digest,
                objective_elements=dict(identity.elements),
                direction=direction,
                versions=dict(versions),
                space_mode=cfg.tuning.optuna.space_mode,
                fixed=dict(fixed),
                origin_space=tuple(parse_space(cfg.tuning.optuna.space)),
                created_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
    except LineageCodecError as exc:
        raise _codec_refusal(exc, "tuning.study") from exc


def round_record_payload(plan: RoundPlan) -> dict[str, Any]:
    """The encoded ``lizyml:round:<k>`` record of *plan*."""
    try:
        return encode_round(plan.round_record())
    except LineageCodecError as exc:
        raise _codec_refusal(exc, "tuning.optuna.space") from exc
