import joblib
import numpy as np
import pytest

from blitzly.plots.bar import multi_bar
from tests.helper import fig_to_array

# pylint: disable=missing-function-docstring, missing-class-docstring, redefined-outer-name


@pytest.fixture(scope="session")
def expected_1d_numpy():
    return joblib.load("tests/expected_figs/bar/multi_chart/1d_numpy_with_error.joblib")


@pytest.fixture(scope="session")
def expected_2d_numpy():
    return joblib.load("tests/expected_figs/bar/multi_chart/2d_numpy_with_error.joblib")


@pytest.fixture(scope="session")
def expected_2d_numpy_highlighted():
    return joblib.load(
        "tests/expected_figs/bar/multi_chart/2d_numpy_with_error_highlighted.joblib"
    )


class TestMultiChart:
    @staticmethod
    def test_multi_bar_with_1_row_and_errors(expected_1d_numpy):
        fig = multi_bar(
            data=np.array([[4, 5, 6]]),
            errors=np.array([[0.4, 0.5, 0.6]]),
            x_labels=["X1", "X2", "X3"],
            group_labels=["Z2"],
            show=False,
        )
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_1d_numpy))

    @staticmethod
    def test_multi_bar_with_2_row_and_errors(expected_2d_numpy):
        fig = multi_bar(
            data=np.array([[1, 2, 3], [4, 5, 6]]),
            errors=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            x_labels=["X1", "X2", "X3"],
            group_labels=["Z1", "Z2"],
            show=False,
        )
        np.testing.assert_equal(fig_to_array(fig), fig_to_array(expected_2d_numpy))

    @staticmethod
    def test_multi_bar_with_2_row_highlighted_and_text_position(
        expected_2d_numpy_highlighted,
    ):
        fig = multi_bar(
            data=np.array([[1, 2, 3], [4, 5, 6]]),
            x_labels=["X1", "X2", "X3"],
            group_labels=["Z1", "Z2"],
            mark_x_labels=["X2"],
            text_position="outside",
            show=False,
        )
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_2d_numpy_highlighted)
        )

    @staticmethod
    def test_multi_bar_1_dim_exception():
        with pytest.raises(ValueError) as error:
            _ = multi_bar(
                data=np.array([1, 2, 3]),
                x_labels=["X1", "X2", "X3"],
                group_labels=["Z1", "Z2"],
                show=False,
            )
        assert (
            str(error.value)
            == "The `data` must be at least two-dimensional! Got 1 dimension."
        )

    @staticmethod
    def test_multi_bar_group_labels_exception():
        with pytest.raises(ValueError) as error:
            _ = multi_bar(
                data=np.array([[1, 2, 3], [4, 5, 6]]),
                x_labels=["X1", "X2", "X3"],
                group_labels=["Z1"],
                show=False,
            )
        assert (
            str(error.value)
            == "The number of `group_labels` (1) does not match the number of rows in the `data` (2)!"
        )

    @staticmethod
    def test_multi_bar_x_labels_exception():
        with pytest.raises(ValueError) as error:
            _ = multi_bar(
                data=np.array([[1, 2, 3], [4, 5, 6]]),
                x_labels=["X1"],
                group_labels=["Z1", "Z2"],
                show=False,
            )
        assert (
            str(error.value)
            == "The number of `x_labels` (1) does not match the number of columns in the `data` (3)!"
        )

    @staticmethod
    def test_multi_bar_errors_exception():
        with pytest.raises(ValueError) as error:
            _ = multi_bar(
                data=np.array([[1, 2, 3], [4, 5, 6]]),
                x_labels=["X1", "X2", "X3"],
                group_labels=["Z1", "Z2"],
                errors=np.array([[0.1, 0.2, 0.3]]),
                show=False,
            )
        assert (
            str(error.value)
            == "The shape of the `errors` ((1, 3)) does not match the shape of the `data` ((2, 3))!"
        )

    @staticmethod
    def test_multi_bar_hover_text_exception():
        with pytest.raises(ValueError) as error:
            _ = multi_bar(
                data=np.array([[1, 2, 3], [4, 5, 6]]),
                x_labels=["X1", "X2", "X3"],
                group_labels=["Z1", "Z2"],
                hover_texts=["foo", "bar"],
                show=False,
            )
        assert (
            str(error.value)
            == "The number of `hover_texts` (2) does not match the number of columns in the `data` (3)!"
        )
