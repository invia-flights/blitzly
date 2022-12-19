import joblib
import numpy as np
import pandas as pd
import pytest

from blitzly.plots.histogram import simple_histogram
from tests.helper import fig_to_array

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name


@pytest.fixture(scope="session")
def expected_1d_numpy():
    return joblib.load("tests/expected_figs/histogram/simple_histogram/1d_numpy.joblib")


@pytest.fixture(scope="session")
def expected_2d_numpy():
    return joblib.load("tests/expected_figs/histogram/simple_histogram/2d_numpy.joblib")


class TestSimpleHistogram:
    @staticmethod
    def test_simple_histogram_with_1d_numpy(expected_1d_numpy):
        np.random.seed(42)
        fig = simple_histogram(np.random.randn(100), show=False)
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_1d_numpy))

    @staticmethod
    def test_simple_histogram_with_1d_pandas(expected_1d_numpy):
        np.random.seed(42)
        fig = simple_histogram(
            pd.DataFrame(np.random.randn(100), columns=["a"]), show=False
        )
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_1d_numpy))

    @staticmethod
    def test_simple_histogram_with_1d_pandas_series(expected_1d_numpy):
        np.random.seed(42)
        fig = simple_histogram(pd.Series(np.random.randn(100), name="a"), show=False)
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_1d_numpy))

    @staticmethod
    def test_simple_histogram_with_2d_numpy(expected_2d_numpy):
        np.random.seed(42)
        fig = simple_histogram(np.random.randn(100, 2), show=False)
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_2d_numpy))

    @staticmethod
    def test_simple_histogram_with_pandas(expected_2d_numpy):
        np.random.seed(42)
        fig = simple_histogram(
            pd.DataFrame(np.random.randn(100, 2), columns=["a", "b"]), show=False
        )
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_2d_numpy))
