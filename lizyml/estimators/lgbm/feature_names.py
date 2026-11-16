"""feature_names -- the feature names LightGBM's model text holds intact (H-0147).

LightGBM saves a booster (``model_to_string``, which pickling uses) as text
and writes the names as one line, ``"feature_names=" + " ".join(names)``. Its
reader (LightGBM 4.6.0 ``src/boosting/gbdt_model_text.cpp``, the header loop
and ``feature_names_ = Common::Split(..., ' ')``; ``Common::GetLine`` and
``Common::Split`` in ``include/LightGBM/utils/common.h``):

1. ends a line at ``\\n``, ``\\r`` or NUL;
2. splits the line at ``=``, dropping empty pieces. With exactly two pieces
   the value is the second piece; with more, the value is everything after
   ``feature_names=``;
3. splits the value at spaces, dropping empty pieces, and refuses the model
   when the count differs from the number of features.

So three kinds of name do not survive a save (the read-back booster is
refused, or holds other names):

- an empty name (step 3 drops it);
- a name holding ``\\n`` or ``\\r`` (step 1 cuts the line; what follows is read
  as more model text, which can change predictions);
- a ``=`` at the start of the first name or the end of the last name, when the
  joined names split at ``=`` into a single piece (step 2 drops it).

LightGBM rewrites spaces to ``_`` and cuts a name at NUL before it reaches the
booster, so the names judged here are the booster's own
(``Booster.feature_name()``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lizyml.estimators.base import UnsaveableFeatureName
from lizyml.estimators.provider import RefusedFeatureName

#: ``context["reason"]`` of the refusals.
EMPTY_FEATURE_NAME = "empty_feature_name"
LINE_BREAK_IN_FEATURE_NAME = "line_break_in_feature_name"
EQUALS_AT_FEATURE_NAMES_EDGE = "equals_sign_at_feature_names_edge"


def _equals_is_stripped(names: Sequence[str]) -> bool:
    """Step 2: the joined names hold ``=`` and split at it into one piece."""
    joined = " ".join(names)
    return "=" in joined and len([p for p in joined.split("=") if p]) <= 1


def unsaveable_feature_names(names: Sequence[str]) -> list[UnsaveableFeatureName]:
    """Each name in *names* the model text cannot hold intact, in order."""
    reasons: dict[int, str] = {}
    for position, name in enumerate(names):
        if name == "":
            reasons[position] = EMPTY_FEATURE_NAME
        elif "\n" in name or "\r" in name:
            reasons[position] = LINE_BREAK_IN_FEATURE_NAME
    if names and _equals_is_stripped(names):
        last = len(names) - 1
        if names[0].startswith("="):
            reasons.setdefault(0, EQUALS_AT_FEATURE_NAMES_EDGE)
        if names[last].endswith("="):
            reasons.setdefault(last, EQUALS_AT_FEATURE_NAMES_EDGE)
    return [
        UnsaveableFeatureName(position=p, name=names[p], reason=reasons[p])
        for p in sorted(reasons)
    ]


#: ``Common::CheckAllowedJSON`` (LightGBM 4.6.0 include/LightGBM/utils/common.h).
_JSON_SPECIAL = frozenset('",:[]{}')

#: ``context["reason"]`` of the training refusals (H-0148).
JSON_SPECIAL_CHARACTER_IN_FEATURE_NAME = "json_special_character_in_feature_name"
DUPLICATE_FEATURE_NAME = "duplicate_feature_name"


def _seen_by_lightgbm(name: Any) -> str:
    """The name LightGBM's Dataset receives: the Python package passes
    ``str(name)`` as a C string, so a NUL ends it."""
    return str(name).split("\x00", 1)[0]


def lightgbm_refused_feature_name(names: Sequence[Any]) -> RefusedFeatureName | None:
    """The first of *names* LightGBM's Dataset refuses, or ``None`` (H-0148).

    ``Dataset::set_feature_names`` (LightGBM 4.6.0 include/LightGBM/dataset.h)
    looks at each name in order: a name holding a JSON special character is
    refused, then spaces are rewritten to ``_`` and a name equal to an
    earlier rewritten one is refused.
    """
    seen: dict[str, int] = {}
    for position, name in enumerate(names):
        given = _seen_by_lightgbm(name)
        if any(char in _JSON_SPECIAL for char in given):
            return RefusedFeatureName(
                position=position,
                name=name,
                estimator_name=given,
                reason=JSON_SPECIAL_CHARACTER_IN_FEATURE_NAME,
                first_position=None,
            )
        rewritten = given.replace(" ", "_")
        if rewritten in seen:
            return RefusedFeatureName(
                position=position,
                name=name,
                estimator_name=rewritten,
                reason=DUPLICATE_FEATURE_NAME,
                first_position=seen[rewritten],
            )
        seen[rewritten] = position
    return None
