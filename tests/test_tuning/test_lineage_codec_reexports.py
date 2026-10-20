"""H-0134 P1: every name ``tuning/lineage_codec.py`` defined still imports from it.

The value codecs moved to ``lizyml.core.tuning_codec`` (Layer 0). The old
module must keep handing out the same objects, so an existing import, public or
private, keeps working. The list is the module's top-level definitions at
develop ``f746814``, before the move.
"""

from __future__ import annotations

import pytest

import lizyml.core.tuning_codec as core_codec
import lizyml.tuning.lineage_codec as lineage_codec

#: Defined in lineage_codec before H-0134 and moved to core.tuning_codec.
MOVED = (
    "DIM_CATEGORIES",
    "_SPECIAL_FLOATS",
    "_EDGES",
    "LineageCodecError",
    "encode_scalar",
    "decode_scalar",
    "_decode_number",
    "_decode_optional_number",
    "_check_number",
    "_expect_dict",
    "_expect_list",
    "_expect_str",
    "_expect_bool",
    "_field",
    "_category",
    "_NUMERIC_DIM_KEYS",
    "_CATEGORICAL_DIM_KEYS",
    "encode_dim",
    "decode_dim",
    "_check_magnitude",
    "encode_dims",
    "decode_dims",
    "dim_signature",
    "_encode_status",
    "_decode_status",
    "encode_report",
    "decode_report",
)
#: Defined in lineage_codec before H-0134 and still defined there.
STAYED = ("SCHEMA_VERSION", "STUDY_KEY", "_check_keys", "to_attr_value")


@pytest.mark.parametrize("name", MOVED)
def test_a_moved_name_is_the_same_object_from_the_old_module(name: str) -> None:
    assert getattr(lineage_codec, name) is getattr(core_codec, name)


@pytest.mark.parametrize("name", STAYED)
def test_a_name_that_stayed_is_still_defined(name: str) -> None:
    assert hasattr(lineage_codec, name)
    assert not hasattr(core_codec, name)


def test_the_public_names_are_declared() -> None:
    public = {n for n in (*MOVED, *STAYED) if not n.startswith("_")}
    assert public <= set(lineage_codec.__all__)
