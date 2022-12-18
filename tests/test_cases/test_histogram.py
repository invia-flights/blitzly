import numpy as np
from helper import fig_to_array

from blitzly.plots.histogram import simple_histogram

# pylint: disable=missing-function-docstring, missing-class-docstring


class TestSimpleHistogram:
    @staticmethod
    def test_simple_histogram_with_1d_numpy(numpy_1d_data, expected_1d_numpy_pandas):
        fig = simple_histogram(numpy_1d_data, show=False)
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_1d_numpy_pandas)
        )

    @staticmethod
    def test_simple_histogram_with_1d_pandas(pandas_1d_data, expected_1d_numpy_pandas):
        fig = simple_histogram(pandas_1d_data, show=False)
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_1d_numpy_pandas)
        )

    @staticmethod
    def test_simple_histogram_with_1d_pandas_series(
        pandas_series_1d_data, expected_1d_numpy_pandas
    ):
        fig = simple_histogram(pandas_series_1d_data, show=False)
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_1d_numpy_pandas)
        )

    @staticmethod
    def test_simple_histogram_with_2d_numpy(numpy_2d_data, expected_2d_numpy_pandas):
        fig = simple_histogram(numpy_2d_data, show=False)
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_2d_numpy_pandas)
        )

    @staticmethod
    def test_simple_histogram_with_pandas(pandas_2d_data, expected_2d_numpy_pandas):
        fig = simple_histogram(pandas_2d_data, show=False)
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_2d_numpy_pandas)
        )
