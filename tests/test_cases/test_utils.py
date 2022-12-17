import pytest

from blitzly.plots.histogram import simple_histogram

# pylint: disable=missing-function-docstring, missing-class-docstring


class TestCheckData:
    @staticmethod
    def test_simple_histogram_with_3d_numpy(numpy_3d_data):
        with pytest.raises(ValueError) as error:
            simple_histogram(numpy_3d_data, show=False)
        assert str(error.value) == "NumPy array must be 1- or 2-dimensional!"

    @staticmethod
    def test_simple_histogram_with_invalid_data(invalid_data):
        with pytest.raises(TypeError) as error:
            simple_histogram(invalid_data, show=False)
        assert (
            str(error.value)
            == "Data must be a DataFrame, numpy array, or list of values! Type `str` is not supported.`"
        )
