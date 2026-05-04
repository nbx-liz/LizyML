"""Unit tests for the Foundation-layer TargetEncoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.target_encoder import TargetEncoder


class TestNoOp:
    def test_no_op_constructor(self) -> None:
        enc = TargetEncoder.no_op()
        assert enc.needs_encoding is False
        assert enc.classes_ == ()
        assert enc.original_dtype == ""

    def test_no_op_transform_returns_input(self) -> None:
        enc = TargetEncoder.no_op()
        y = pd.Series([1, 2, 3])
        out = enc.transform(y)
        pd.testing.assert_series_equal(out, y)

    def test_no_op_inverse_returns_input(self) -> None:
        enc = TargetEncoder.no_op()
        codes = np.array([0, 1, 2])
        np.testing.assert_array_equal(enc.inverse_transform(codes), codes)


class TestFitNumeric:
    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    @pytest.mark.parametrize(
        "y",
        [
            pd.Series([0, 1, 0, 1], dtype="int64"),
            pd.Series([0.0, 1.0, 0.0, 1.0], dtype="float64"),
            pd.Series([0, 1, 2, 0, 1], dtype="int32"),
        ],
    )
    def test_numeric_classification_is_no_op(self, task: str, y: pd.Series) -> None:
        enc = TargetEncoder.fit(y, task)  # type: ignore[arg-type]
        assert enc.needs_encoding is False
        assert enc.classes_ == ()

    def test_regression_numeric_is_no_op(self) -> None:
        y = pd.Series([1.5, 2.7, 3.1])
        enc = TargetEncoder.fit(y, "regression")
        assert enc.needs_encoding is False


class TestFitNonNumeric:
    def test_string_binary_captures_classes_sorted(self) -> None:
        y = pd.Series(["yes", "no", "yes", "no", "yes"])
        enc = TargetEncoder.fit(y, "binary")
        assert enc.needs_encoding is True
        assert enc.classes_ == ("no", "yes")

    def test_string_multiclass_captures_three_classes(self) -> None:
        y = pd.Series(["Adelie", "Gentoo", "Chinstrap", "Adelie", "Gentoo"])
        enc = TargetEncoder.fit(y, "multiclass")
        assert enc.classes_ == ("Adelie", "Chinstrap", "Gentoo")

    def test_pandas_string_dtype(self) -> None:
        y = pd.Series(["a", "b", "c"], dtype="string")
        enc = TargetEncoder.fit(y, "multiclass")
        assert enc.needs_encoding is True
        assert enc.classes_ == ("a", "b", "c")

    def test_categorical_string_categories(self) -> None:
        y = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"]))
        enc = TargetEncoder.fit(y, "binary")
        assert enc.needs_encoding is True
        assert enc.classes_ == ("a", "b")

    def test_bool_target_treated_as_classification(self) -> None:
        y = pd.Series([True, False, True, False])
        enc = TargetEncoder.fit(y, "binary")
        assert enc.needs_encoding is True
        assert enc.classes_ == (False, True)

    def test_drops_nan_when_capturing_classes(self) -> None:
        y = pd.Series(["a", None, "b", "a", None])
        enc = TargetEncoder.fit(y, "binary")
        assert enc.classes_ == ("a", "b")

    def test_classes_are_sorted_by_str(self) -> None:
        y = pd.Series(["zebra", "apple", "mango"])
        enc = TargetEncoder.fit(y, "multiclass")
        assert enc.classes_ == ("apple", "mango", "zebra")


class TestFitRegressionRejection:
    @pytest.mark.parametrize(
        "y",
        [
            pd.Series(["1.5", "2.0"]),
            pd.Series(["a", "b"]),
            pd.Series(pd.Categorical(["a", "b"])),
            pd.Series([True, False]),
        ],
    )
    def test_regression_rejects_non_numeric(self, y: pd.Series) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            TargetEncoder.fit(y, "regression")
        assert exc_info.value.code == ErrorCode.TARGET_NOT_NUMERIC
        assert "task" in exc_info.value.context
        assert "dtype" in exc_info.value.context


class TestTransform:
    def test_string_round_trip(self) -> None:
        y = pd.Series(["yes", "no", "yes", "no"])
        enc = TargetEncoder.fit(y, "binary")
        encoded = enc.transform(y)
        assert encoded.dtype == np.int64
        # ('no', 'yes') => no->0, yes->1
        np.testing.assert_array_equal(encoded.to_numpy(), [1, 0, 1, 0])

    def test_inverse_recovers_original_values(self) -> None:
        y = pd.Series(["a", "b", "c", "a"])
        enc = TargetEncoder.fit(y, "multiclass")
        encoded = enc.transform(y).to_numpy()
        decoded = enc.inverse_transform(encoded)
        assert list(decoded) == list(y)

    def test_unseen_label_raises(self) -> None:
        enc = TargetEncoder.fit(pd.Series(["a", "b"]), "binary")
        with pytest.raises(LizyMLError) as exc_info:
            enc.transform(pd.Series(["a", "b", "c"]))
        assert exc_info.value.code == ErrorCode.TARGET_UNSEEN_LABEL
        assert "c" in exc_info.value.context["unseen"]

    def test_transform_preserves_index_and_name(self) -> None:
        y = pd.Series(["a", "b"], index=[10, 20], name="species")
        enc = TargetEncoder.fit(y, "binary")
        encoded = enc.transform(y)
        assert list(encoded.index) == [10, 20]
        assert encoded.name == "species"


class TestInverseDtype:
    def test_object_input_returns_object_array(self) -> None:
        y = pd.Series(["a", "b", "a"])
        enc = TargetEncoder.fit(y, "binary")
        out = enc.inverse_transform(np.array([0, 1, 0]))
        assert out.dtype == object
        assert list(out) == ["a", "b", "a"]

    def test_string_dtype_input_restores_string(self) -> None:
        y = pd.Series(["a", "b"], dtype="string")
        enc = TargetEncoder.fit(y, "binary")
        out = enc.inverse_transform(np.array([0, 1]))
        # pd.array("string") -> ndarray; values match
        assert list(out) == ["a", "b"]

    def test_2d_codes_for_multiclass_argmax_shape(self) -> None:
        # inverse_transform should accept 1-D code arrays, but multi-class
        # predict feeds 1-D argmax results, not 2-D probas. Verify 1-D works.
        enc = TargetEncoder.fit(pd.Series(["a", "b", "c"]), "multiclass")
        out = enc.inverse_transform(np.array([2, 0, 1]))
        assert list(out) == ["c", "a", "b"]


class TestImmutability:
    def test_frozen_dataclass_blocks_mutation(self) -> None:
        from dataclasses import FrozenInstanceError

        enc = TargetEncoder.fit(pd.Series(["a", "b"]), "binary")
        with pytest.raises(FrozenInstanceError):
            enc.classes_ = ("x", "y")  # type: ignore[misc]

    def test_classes_is_tuple_not_list(self) -> None:
        enc = TargetEncoder.fit(pd.Series(["a", "b"]), "binary")
        assert isinstance(enc.classes_, tuple)


class TestEdgeCases:
    def test_transform_with_nan_raises_clear_error(self) -> None:
        enc = TargetEncoder.fit(pd.Series(["a", "b"]), "binary")
        with pytest.raises(LizyMLError) as exc_info:
            enc.transform(pd.Series(["a", None, "b"]))
        assert exc_info.value.code == ErrorCode.DATA_SCHEMA_INVALID
        assert "NaN" in exc_info.value.user_message

    def test_inverse_transform_out_of_range_code_raises(self) -> None:
        enc = TargetEncoder.fit(pd.Series(["a", "b"]), "binary")
        with pytest.raises(LizyMLError) as exc_info:
            enc.inverse_transform(np.array([0, 1, 2]))
        assert exc_info.value.code == ErrorCode.TARGET_UNSEEN_LABEL
        assert exc_info.value.context["n_classes"] == 2

    def test_single_class_classification_encoded(self) -> None:
        # Degenerate case: only one unique label. Encoder should still
        # succeed (downstream LightGBM may reject, but encoder is dtype-only).
        enc = TargetEncoder.fit(pd.Series(["only"]), "binary")
        assert enc.classes_ == ("only",)
        assert enc.transform(pd.Series(["only", "only"])).tolist() == [0, 0]
