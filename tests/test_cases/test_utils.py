import numpy as np
import pandas as pd
import pytest

from blitzly.etc.utils import check_data

# pylint: disable=missing-function-docstring, missing-class-docstring


class TestCheckData:
    @staticmethod
    def test_check_data_for_non_square_matrix():
        with pytest.raises(ValueError) as error:
            check_data(np.array([[1, 2], [3, 4], [5, 6]]), only_square_matrix=True)
        assert (
            str(error.value)
            == "Data must be a square matrix! But it's shape is: `(3, 2)`."
        )

    @staticmethod
    def test_check_data_for_3d_numpy():
        np.random.seed(42)
        with pytest.raises(ValueError) as error:
            check_data(np.random.randn(100, 2, 2), only_numerical_values=False)
        assert str(error.value) == "NumPy array must be 1- or 2-dimensional!"

    @staticmethod
    def test_check_data_for_non_numerical_data():
        with pytest.raises(TypeError) as error:
            check_data(np.array(["1", "2", "3"]))
        assert str(error.value) == "Data must be numerical (`np.number`)!"

    @staticmethod
    def test_check_data_for_invalid_data():
        with pytest.raises(TypeError) as error:
            check_data("invalid")
        assert (
            str(error.value)
            == """
            Invalid data type! Type `str` is not supported.
            Please choose between a DataFrame, numpy array, or list of values.
        """
        )

    @staticmethod
    def test_check_data_for_min_rows():
        with pytest.raises(ValueError) as error:
            check_data(np.array([[1]]), min_rows=2)
        assert str(error.value) == "The data must have at least 2 row(s)!"

    @staticmethod
    def test_check_data_for_max_rows():
        with pytest.raises(ValueError) as error:
            check_data(np.array([[1], [2]]), max_rows=1)
        assert str(error.value) == "The data must have a maximum of 1 row(s)!"

    @staticmethod
    def test_check_data_for_min_columns():
        with pytest.raises(ValueError) as error:
            check_data(np.array([[1, 2]]), min_columns=3)
        assert str(error.value) == "The data must have at least 3 column(s)!"

    @staticmethod
    def test_check_data_for_max_columns():
        with pytest.raises(ValueError) as error:
            check_data(np.array([[1, 2]]), max_columns=1)
        assert str(error.value) == "The data must have a maximum of 1 column(s)!"

    @staticmethod
    def test_check_data_with_keep_as_pandas():
        data = check_data(pd.DataFrame(np.array([[1, 2]])), keep_as_pandas=True)
        assert isinstance(data, pd.DataFrame)

    @staticmethod
    def test_check_data_with_convert_to_pandas():
        data = check_data(np.array([[1, 2]]), keep_as_pandas=True)
        assert isinstance(data, pd.DataFrame)
