import joblib
import numpy as np
import pandas as pd
import pytest

from blitzly.plots.scatter import scatter_matrix
from tests.helper import fig_to_array

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name, disallowed-name


@pytest.fixture(scope="session")
def expected_pandas_with_color():
    return joblib.load(
        "tests/expected_figs/scatter/scatter_matrix/expected_pandas_with_color.joblib"
    )


@pytest.fixture(scope="session")
def expected_pandas_without_color():
    return joblib.load(
        "tests/expected_figs/scatter/scatter_matrix/expected_pandas_without_color.joblib"
    )


class TestScatterMatrix:
    @staticmethod
    def test_scatter_matrix_with_valid_values(expected_pandas_with_color):
        np.random.seed(42)
        foo = np.random.randn(100)
        bar = np.random.randn(100) + 1
        blitz = np.random.randint(2, size=100)
        licht = np.random.randint(2, size=100)
        data = np.array([foo, bar, blitz, licht])
        df = pd.DataFrame(data.T, columns=["A", "B", "C", "D"])

        fig = scatter_matrix(
            df,
            dimensions=["A", "B", "C"],
            color_dim=df["D"],
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_pandas_with_color)
        )

    @staticmethod
    def test_scatter_matrix_with_valid_values_without_colors(
        expected_pandas_without_color,
    ):
        np.random.seed(42)
        foo = np.random.randn(100)
        bar = np.random.randn(100) + 1
        blitz = np.random.randint(2, size=100)
        licht = np.random.randint(2, size=100)
        data = np.array([foo, bar, blitz, licht])
        df = pd.DataFrame(data.T, columns=["A", "B", "C", "D"])

        fig = scatter_matrix(
            df,
            dimensions=["A", "B", "C"],
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_pandas_without_color)
        )

    @staticmethod
    def test_scatter_matrix_color_dims_exception():
        np.random.seed(42)
        foo = np.random.randn(100)
        bar = np.random.randn(100) + 1
        blitz = np.random.randint(2, size=100)
        licht = np.random.randint(2, size=100)
        data = np.array([foo, bar, blitz, licht])
        df = pd.DataFrame(data.T, columns=["A", "B", "C", "D"])

        with pytest.raises(ValueError) as error:
            _ = scatter_matrix(
                df,
                dimensions=["A", "B", "C"],
                color_dim=np.array([["C"], ["D"]]),
                show=False,
            )
        assert str(error.value) == "`color_dim` must be a 1-dimensional array!"
