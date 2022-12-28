import joblib
import numpy as np
import pandas as pd
import pytest

from blitzly.plots.histogram import simple_histogram
from blitzly.subplots import make_subplots
from tests.helper import fig_to_array

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, too-few-public-methods


@pytest.fixture(scope="session")
def expected_histogram_grid():
    return joblib.load(
        "tests/expected_figs/subplots/make_subplots/expected_histogram_grid.joblib"
    )


@pytest.fixture(scope="session")
def expected_histogram_grid_unequal_columns():
    return joblib.load(
        "tests/expected_figs/subplots/make_subplots/expected_histogram_grid_unequal_columns.joblib"
    )


@pytest.fixture(scope="session")
def expected_histogram_grid_filled_row():
    return joblib.load(
        "tests/expected_figs/subplots/make_subplots/expected_histogram_grid_filled_row.joblib"
    )


class TestSubplots:
    @staticmethod
    def test_subplots(expected_histogram_grid):
        np.random.seed(42)
        subfig = simple_histogram(
            pd.Series(np.random.randn(100), name="a"), show=False, title="Histogram"
        )
        fig = make_subplots(
            [subfig, subfig, subfig, subfig],
            shape=(2, 2),
            title="A figure with subplots",
            size=(800, 800),
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_histogram_grid)
        )

    @staticmethod
    def test_subplots_with_invalid_shape():
        np.random.seed(42)
        subfig = simple_histogram(pd.Series(np.random.randn(100), name="a"), show=False)
        with pytest.raises(ValueError) as error:
            _ = make_subplots(
                [subfig, subfig, subfig, subfig],
                (2, 1),
                size=(800, 800),
                show=False,
            )
        assert (
            str(error.value)
            == "The number of subfigures (4) is too large for the provided `shape` (2, 1)."
        )

    @staticmethod
    def test_subplots_unequal_columns(expected_histogram_grid_unequal_columns):
        np.random.seed(42)
        subfig = simple_histogram(
            pd.Series(np.random.randn(100), name="a"), show=False, title="Histogram"
        )
        fig = make_subplots(
            [subfig, subfig, subfig, subfig],
            shape=(2, 2),
            column_widths=[0.7, 0.3],
            title="A figure with unequal column widths",
            size=(800, 800),
            show=True,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_histogram_grid_unequal_columns)
        )

    @staticmethod
    def test_subplots_filled_row(expected_histogram_grid_filled_row):
        np.random.seed(42)
        subfig = simple_histogram(
            pd.Series(np.random.randn(100), name="a"), show=False, title="Histogram"
        )
        fig = make_subplots(
            [subfig, subfig, subfig, subfig],
            shape=(2, 3),
            title="A figure with filled last row of subplot grid",
            fill_row=True,
            size=(800, 800),
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_histogram_grid_filled_row)
        )
