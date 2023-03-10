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


@pytest.fixture(scope="session")
def expected_histogram_grid_axes_labels():
    return joblib.load(
        "tests/expected_figs/subplots/make_subplots/expected_histogram_grid_axes_labels.joblib"
    )


@pytest.fixture(scope="session")
def expected_histogram_grid_shared_axes():
    return joblib.load(
        "tests/expected_figs/subplots/make_subplots/expected_histogram_grid_shared_axes.joblib"
    )


class TestSubplots:
    @staticmethod
    def test_subplots(expected_histogram_grid):
        np.random.seed(42)
        subfig = simple_histogram(
            pd.Series(np.random.randn(100), name="a"),
            show=False,
            title="Histogram",
            x_label="",
            y_label="",
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
            pd.Series(np.random.randn(100), name="a"),
            show=False,
            title="Histogram",
            x_label="",
            y_label="",
        )
        fig = make_subplots(
            [subfig, subfig, subfig, subfig],
            shape=(2, 2),
            column_widths=[0.7, 0.3],
            title="A figure with unequal column widths",
            size=(800, 800),
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_histogram_grid_unequal_columns)
        )

    @staticmethod
    def test_subplots_filled_row(expected_histogram_grid_filled_row):
        np.random.seed(42)
        subfig = simple_histogram(
            pd.Series(np.random.randn(100), name="a"),
            show=False,
            title="Histogram",
            x_label="",
            y_label="",
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

    @staticmethod
    def test_subplots_axes_labels(expected_histogram_grid_axes_labels):
        np.random.seed(42)
        subfig_list = [
            simple_histogram(
                pd.Series(np.random.randn(100), name="a"),
                show=False,
                title=f"Histogram-{i}",
                x_label=f"x-axis label-{i}",
                y_label=f"y-axis label-{i}",
            )
            for i in range(1, 5)
        ]
        fig = make_subplots(
            subfig_list,
            shape=(2, 2),
            title="Subplots with axes labels",
            size=(800, 800),
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_histogram_grid_axes_labels)
        )

    @staticmethod
    def test_subplots_shared_axes(expected_histogram_grid_shared_axes):
        np.random.seed(42)
        subfig_list = [
            simple_histogram(
                pd.Series(np.random.randn(100), name="a"),
                show=False,
                title="",
                x_label=f"x-axis label-{i}",
                y_label=f"y-axis label-{i}",
            )
            for i in range(1, 5)
        ]
        fig = make_subplots(
            subfig_list,
            shape=(2, 2),
            title="Subplots with shared axes",
            size=(800, 800),
            show=False,
            shared_xaxes=True,
            shared_yaxes=True,
            plotly_kwargs={
                "vertical_spacing": 0.02,
                "horizontal_spacing": 0.02,
            },
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_histogram_grid_shared_axes)
        )
