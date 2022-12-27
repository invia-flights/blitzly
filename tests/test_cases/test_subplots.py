import joblib
import numpy as np
import pandas as pd
import pytest

from blitzly.plots.histogram import simple_histogram
from blitzly.subplots.subplots import make_subplots
from tests.helper import fig_to_array

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, too-few-public-methods


@pytest.fixture(scope="session")
def expected_histogram_grid():
    return joblib.load(
        "tests/expected_figs/subplots/make_subplots/expected_histogram_grid.joblib"
    )


class TestSubplots:
    @staticmethod
    def test_subplots(expected_histogram_grid):
        np.random.seed(42)
        subfig = simple_histogram(pd.Series(np.random.randn(100), name="a"), show=False)
        fig = make_subplots(
            [subfig, subfig, subfig, subfig],
            (2, 2),
            size=(800, 800),
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_histogram_grid)
        )
