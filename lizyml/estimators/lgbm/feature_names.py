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

from lizyml.estimators.base import UnsaveableFeatureName

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
