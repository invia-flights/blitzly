import joblib
import numpy as np
import pytest

from blitzly.plots.slope import simple_slope
from tests.helper import fig_to_array

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name


@pytest.fixture(scope="session")
def expected_pandas():
    return joblib.load("tests/expected_figs/slope/simple_slope/expected_pandas.joblib")


@pytest.fixture(scope="session")
def expected_2d_numpy():
    return joblib.load(
        "tests/expected_figs/slope/simple_slope/expected_2d_numpy.joblib"
    )


class TestSimpleSlope:
    @staticmethod
    def test_simple_slope_with_pandas(X_numbers_two_column, expected_pandas):
        fig = simple_slope(X_numbers_two_column, size=(500, 500), show=False)
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_pandas))

    @staticmethod
    def test_simple_slope_with_2d_numpy(expected_2d_numpy):
        np.random.seed(42)
        fig = simple_slope(np.random.randn(10, 2), size=(500, 500), show=False)
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_2d_numpy))
