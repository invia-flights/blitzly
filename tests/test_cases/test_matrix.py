import joblib
import numpy as np
import pandas as pd
import pytest

from blitzly.plots.matrix import (
    binary_confusion_matrix,
    cramers_v_corr_matrix,
    pearson_corr_matrix,
)
from tests.helper import fig_to_array

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name


@pytest.fixture(scope="session")
def valid_binary_confusion_matrix():
    return joblib.load(
        "tests/expected_figs/matrix/binary_confusion_matrix/valid_confusion_matrix.joblib"
    )


@pytest.fixture(scope="session")
def valid_pearson_corr_matrix_from_pandas():
    return joblib.load(
        "tests/expected_figs/matrix/pearson_corr_matrix/valid_pearson_corr_matrix_from_pandas.joblib"
    )


@pytest.fixture(scope="session")
def valid_pearson_corr_matrix_without_label_with_size():
    return joblib.load(
        "tests/expected_figs/matrix/pearson_corr_matrix/valid_pearson_corr_matrix_without_label_with_size.joblib"
    )


@pytest.fixture()
def valid_pearson_corr_matrix_data():
    return np.array(
        [
            [0.77395605, 0.43887844, 0.85859792],
            [0.69736803, 0.09417735, 0.97562235],
            [0.7611397, 0.78606431, 0.12811363],
        ]
    )


@pytest.fixture()
def valid_binary_classification_data():
    return np.array([[1, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1]])


@pytest.fixture()
def valid_multi_class_classification_data():
    return np.array([[1, 0, 2, 1, 0, 2], [2, 2, 1, 1, 0, 1], [2, 0, 1, 2, 0, 1]])


@pytest.fixture()
def cramers_v_corr_matrix_test_data():
    return pd.DataFrame(
        {
            "foo": [
                "1",
                "1",
                "1",
                "2",
                "2",
                "2",
            ],
            "bar": ["3", "2", "3", "7", "5", "7"],
            "blitzly": ["9", "3", "4", "6", "7", "9"],
            "licht": ["1", "1", "1", "2", "2", "2"],
        }
    )


@pytest.fixture()
def cramers_v_corr_matrix_test_numerical_data():
    return pd.DataFrame(
        {
            "foo": [
                1,
                1,
                1,
                2,
                2,
                2,
            ],
            "bar": [3, 2, 3, 7, 5, 7],
            "blitzly": [9, 3, 4, 6, 7, 9],
            "licht": [1, 1, 1, 2, 2, 2],
        }
    )


@pytest.fixture()
def valid_cramers_v_corr_matrix_from_pandas():
    return joblib.load(
        "tests/expected_figs/matrix/cramers_v_corr_matrix/valid_cramers_v_corr_matrix_from_pandas.joblib"
    )


class TestCramersVCorrelationMatrix:
    @staticmethod
    def test_matrix_with_valid_pandas_data(
        cramers_v_corr_matrix_test_data, valid_cramers_v_corr_matrix_from_pandas
    ):
        fig = cramers_v_corr_matrix(cramers_v_corr_matrix_test_data, show=False)
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(valid_cramers_v_corr_matrix_from_pandas)
        )

    @staticmethod
    def test_matrix_with_numerical_data(cramers_v_corr_matrix_test_numerical_data):
        with pytest.raises(Warning) as warning:
            _ = cramers_v_corr_matrix(
                cramers_v_corr_matrix_test_numerical_data, show=False
            )
        assert (
            str(warning.value)
            == """All columns should be from type `object` since the encoding is done internally.
        But don't worry. It should work anyway."""
        )


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


class TestPearsonCorrMatrix:
    @staticmethod
    def test_matrix_with_valid_pandas_data(
        valid_pearson_corr_matrix_data, valid_pearson_corr_matrix_from_pandas
    ):
        df = pd.DataFrame(
            valid_pearson_corr_matrix_data, columns=["foo", "bar", "blitzly"]
        )

        fig = pearson_corr_matrix(df, show=False)
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(valid_pearson_corr_matrix_from_pandas)
        )

    @staticmethod
    def test_matrix_with_valid_numpy_data_with_label(
        valid_pearson_corr_matrix_data, valid_pearson_corr_matrix_from_pandas
    ):
        fig = pearson_corr_matrix(
            valid_pearson_corr_matrix_data, labels=["foo", "bar", "blitzly"], show=False
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(valid_pearson_corr_matrix_from_pandas)
        )

    @staticmethod
    def test_matrix_with_valid_numpy_data_without_label(
        valid_pearson_corr_matrix_data,
        valid_pearson_corr_matrix_without_label_with_size,
    ):
        fig = pearson_corr_matrix(
            valid_pearson_corr_matrix_data, size=(500, 500), show=False
        )
        np.testing.assert_equal(
            fig_to_array(fig),
            fig_to_array(valid_pearson_corr_matrix_without_label_with_size),
        )
