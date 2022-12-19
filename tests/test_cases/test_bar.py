import joblib
import numpy as np
import pytest
from helper import fig_to_array

from blitzly.plots.bar import multi_chart

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name


@pytest.fixture(scope="session")
def expected_1d_numpy():
    return joblib.load("tests/expected_figs/bar/multi_chart/1d_numpy_with_error.joblib")


@pytest.fixture(scope="session")
def expected_2d_numpy():
    return joblib.load("tests/expected_figs/bar/multi_chart/2d_numpy_with_error.joblib")


class TestBinaryConfusionMatrix:
    @staticmethod
    def test_multi_bar_with_1_row_and_errors(expected_1d_numpy):
        fig = multi_chart(
            data=np.array([[4, 5, 6]]),
            errors=np.array([[0.4, 0.5, 0.6]]),
            x_labels=["X1", "X2", "X3"],
            group_labels=["Z2"],
            show=False,
        )
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_1d_numpy))

    @staticmethod
    def test_multi_bar_with_2_row_and_errors(expected_2d_numpy):
        fig = multi_chart(
            data=np.array([[1, 2, 3], [4, 5, 6]]),
            errors=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            x_labels=["X1", "X2", "X3"],
            group_labels=["Z1", "Z2"],
            show=False,
        )
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_2d_numpy))
