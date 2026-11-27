"""_trial_inputs -- the parameters one tune trial trains on (H-0094, H-0150).

The tune objective and the memory estimates build a trial's parameters through
this one function, so an estimate sees the parameters a trial would train on
(DC3). It is callable per trial: the budget check of a later part reuses it.
"""

from __future__ import annotations

from typing import Any

from lizyml.core._model_factories import overlay_params
from lizyml.core.types.search_dim import SearchDim
from lizyml.tuning.search_space import split_by_category


def trial_inputs(
    provider: Any,
    space: list[SearchDim],
    trial_params: dict[str, Any],
    *,
    base_model_params: dict[str, Any],
    base_smart_params: dict[str, Any],
    fixed: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The ``(model, smart, training)`` parameters of a trial that sampled
    *trial_params* from *space*.

    Overlaid by identity, the same way ``_merge_params`` overlays its three
    layers. A plain dict merge keeps both spellings: a config ``learning_rate``
    and a search dimension named ``eta`` are one parameter to LightGBM, which
    then prefers the canonical name -- so the trials trained at the config's
    value while the study recorded the trial's, and the fit afterwards used the
    recorded one. Tuning selected a model it had never evaluated (H-0094,
    review round 11). This is the fourth seam; the other three were made
    identity-aware in round 3.
    """
    model_p, smart_p, training_p = split_by_category(trial_params, space)
    merged_model = overlay_params(provider, base_model_params, fixed)
    merged_model = overlay_params(provider, merged_model, model_p)
    merged_smart = {**base_smart_params, **smart_p}
    return merged_model, merged_smart, training_p
