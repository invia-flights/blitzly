import joblib
import numpy as np
import pytest
from helper import fig_to_array

from blitzly.plots.matrix import binary_confusion_matrix

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name


@pytest.fixture(scope="session")
def valid_binary_confusion_matrix():
    return joblib.load(
        "tests/expected_figs/matrix/binary_confusion_matrix/valid_confusion_matrix.joblib"
    )


@pytest.fixture()
def valid_binary_classification_data():
    return np.array([[1, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1]])


@pytest.fixture()
def valid_multi_class_classification_data():
    return np.array([[1, 0, 2, 1, 0, 2], [2, 2, 1, 1, 0, 1], [2, 0, 1, 2, 0, 1]])


class TestBinaryConfusionMatrix:
    @staticmethod
    def test_matrix_with_valid_data(
        valid_binary_classification_data, valid_binary_confusion_matrix
    ):
        fig = binary_confusion_matrix(
            valid_binary_classification_data,
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(valid_binary_confusion_matrix)
        )

    @staticmethod
    def test_changed_matrix_with_valid_data_not_equal(
        valid_binary_classification_data, valid_binary_confusion_matrix
    ):
        fig = binary_confusion_matrix(valid_binary_classification_data, show=False)
        fig["data"][0]["showscale"] = True
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_equal,
            fig_to_array(fig),
            fig_to_array(valid_binary_confusion_matrix),
        )

    @staticmethod
    def test_matrix_with_invalid_data(valid_multi_class_classification_data):
        with pytest.raises(ValueError) as error:
            _ = binary_confusion_matrix(
                valid_multi_class_classification_data, show=False
            )
        assert str(error.value) == "The data must have a maximum of 2 row(s)!"
