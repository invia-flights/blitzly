import joblib
import numpy as np
import pandas as pd
import pytest

from blitzly.plots.scatter import multi_scatter, scatter_matrix
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


@pytest.fixture(scope="session")
def expected_pandas_without_dims_with_size():
    return joblib.load(
        "tests/expected_figs/scatter/scatter_matrix/expected_pandas_without_dims_with_size.joblib"
    )


@pytest.fixture(scope="session")
def expected_multi_scatter_with_valid_values():
    return joblib.load(
        "tests/expected_figs/scatter/multi_scatter/expected_multi_scatter_with_valid_values.joblib"
    )


@pytest.fixture(scope="session")
def expected_multi_scatter_with_valid_values_size():
    return joblib.load(
        "tests/expected_figs/scatter/multi_scatter/expected_multi_scatter_with_valid_values_size.joblib"
    )


class TestScatterMatrix:
    @staticmethod
    def test_scatter_matrix_with_valid_values_without_dims_with_size(
        expected_pandas_without_dims_with_size,
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
            size=500,
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_pandas_without_dims_with_size)
        )

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


class TestMultiScatter:
    @staticmethod
    def test_multi_scatter_with_valid_values(expected_multi_scatter_with_valid_values):
        np.random.seed(42)
        random_a = np.linspace(0, 1, 100)
        random_b = np.random.randn(100) + 5
        random_c = np.random.randn(100)
        random_d = np.random.randn(100) - 5

        data = np.array([random_a, random_b, random_c, random_d]).T
        df = pd.DataFrame(data, columns=["a", "b", "c", "d"])

        fig = multi_scatter(
            df,
            x_y_columns=[("a", "b"), ("a", "c"), ("a", "d")],
            modes=["lines", "markers", "lines+markers"],
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_multi_scatter_with_valid_values)
        )

    @staticmethod
    def test_multi_scatter_with_valid_values_and_size(
        expected_multi_scatter_with_valid_values_size,
    ):
        np.random.seed(42)
        random_a = np.linspace(0, 1, 100)
        random_b = np.random.randn(100) + 5
        random_c = np.random.randn(100)
        random_d = np.random.randn(100) - 5

        data = np.array([random_a, random_b, random_c, random_d]).T
        df = pd.DataFrame(data, columns=["a", "b", "c", "d"])

        fig = multi_scatter(
            df,
            x_y_columns=[("a", "b"), ("a", "c"), ("a", "d")],
            modes=["lines", "markers", "lines+markers"],
            size=(500, 500),
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig),
            fig_to_array(expected_multi_scatter_with_valid_values_size),
        )

    @staticmethod
    def test_multi_scatter_no_dataframe_exception():
        np.random.seed(42)
        random_a = np.linspace(0, 1, 100)
        random_b = np.random.randn(100) + 5
        random_c = np.random.randn(100)
        random_d = np.random.randn(100) - 5

        with pytest.raises(TypeError) as error:
            _ = multi_scatter(
                np.array([random_a, random_b, random_c, random_d]).T,
                x_y_columns=[("a", "b"), ("a", "c"), ("a", "d")],
                modes=["lines", "markers", "lines+markers"],
                show=False,
            )
        assert str(error.value) == "`data` must be a Pandas DataFrame!"

    @staticmethod
    def test_multi_scatter_incompatible_columns_exception():
        np.random.seed(42)
        random_a = np.linspace(0, 1, 100)
        random_b = np.random.randn(100) + 5
        random_c = np.random.randn(100)
        random_d = np.random.randn(100) - 5
        data = np.array([random_a, random_b, random_c, random_d]).T
        df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
        with pytest.raises(ValueError) as error:
            _ = multi_scatter(
                df,
                x_y_columns=[("a", "b"), ("a", "c"), ("a", "x")],
                modes=["lines", "markers", "lines+markers"],
                show=False,
            )
        assert (
            str(error.value)
            == """
            Columns ['x'] not in `data`!
            All columns passed in `x_y_columns` must be in `data`.
            """
        )

    @staticmethod
    def test_multi_scatter_incompatible_modes_exception():
        np.random.seed(42)
        random_a = np.linspace(0, 1, 100)
        random_b = np.random.randn(100) + 5
        random_c = np.random.randn(100)
        random_d = np.random.randn(100) - 5
        data = np.array([random_a, random_b, random_c, random_d]).T
        df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
        with pytest.raises(ValueError) as error:
            _ = multi_scatter(
                df,
                x_y_columns=[("a", "b"), ("a", "c"), ("a", "c")],
                modes=["lines"],
                show=False,
            )
        assert (
            str(error.value)
            == """
            Length of `modes` (1) must be equal to length of `x_y_columns` (3)!
            Or `modes` must be `None`.
            """
        )
