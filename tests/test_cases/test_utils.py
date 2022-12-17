import pytest

from blitzly.etc.utils import check_data

# pylint: disable=missing-function-docstring, missing-class-docstring


class TestCheckData:
    @staticmethod
    def test_check_data_for_3d_numpy(numpy_3d_data):
        with pytest.raises(ValueError) as error:
            check_data(numpy_3d_data)
        assert str(error.value) == "NumPy array must be 1- or 2-dimensional!"

    @staticmethod
    def test_check_data_for_invalid_data(invalid_data):
        with pytest.raises(TypeError) as error:
            check_data(invalid_data)
        assert (
            str(error.value)
            == "Data must be a DataFrame, numpy array, or list of values! Type `str` is not supported.`"
        )
