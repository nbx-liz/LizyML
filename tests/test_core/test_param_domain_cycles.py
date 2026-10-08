"""Mappings at the training boundary must fail without traversing their values."""

import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.param_domain import assert_plain_params, is_plain


@pytest.mark.parametrize("nested", [False, True])
def test_cyclic_mapping_is_a_controlled_training_refusal(nested: bool) -> None:
    mapping: dict[str, object] = {}
    mapping["self"] = mapping
    value = [mapping] if nested else mapping
    assert not is_plain(value)
    with pytest.raises(LizyMLError) as caught:
        assert_plain_params({"metric": value}, where="test training boundary")
    assert caught.value.code is ErrorCode.CONFIG_INVALID
