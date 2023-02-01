import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from blitzly.plots.bar import model_feature_importances, multi_bar
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


@pytest.fixture(scope="session")
def expected_model_feature_importances_horizontal():
    return joblib.load(
        "tests/expected_figs/bar/model_feature_importances/expected_model_feature_importances_horizontal.joblib"
    )


@pytest.fixture(scope="session")
def expected_model_feature_importances_vertical():
    return joblib.load(
        "tests/expected_figs/bar/model_feature_importances/expected_model_feature_importances_vertical.joblib"
    )


class TestFeatureImportancesChart:
    @staticmethod
    def test_feature_importances_chart_horizontal(
        expected_model_feature_importances_horizontal,
    ):
        X, y = make_classification(
            n_samples=10,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            random_state=42,
            shuffle=False,
        )

        X = pd.DataFrame(X, columns=["foo", "bar", "blitz", "licht"])
        y = pd.Series(y)

        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=42, shuffle=False
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        fig = model_feature_importances(X_test, model, show=False)
        np.testing.assert_equal(
            fig_to_array(fig),
            fig_to_array(expected_model_feature_importances_horizontal),
        )

    @staticmethod
    def test_feature_importances_chart_vertical(
        expected_model_feature_importances_vertical,
    ):
        X, y = make_classification(
            n_samples=10,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            random_state=42,
            shuffle=False,
        )

        X = pd.DataFrame(X, columns=["foo", "bar", "blitz", "licht"])
        y = pd.Series(y)

        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=42, shuffle=False
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        fig = model_feature_importances(X_test, model, horizontal=False, show=False)
        np.testing.assert_equal(
            fig_to_array(fig), fig_to_array(expected_model_feature_importances_vertical)
        )

    @staticmethod
    def test_feature_importances_chart_exception():
        with pytest.raises(AttributeError) as error:
            X, y = make_classification(
                n_samples=10,
                n_features=4,
                n_informative=2,
                n_redundant=0,
                random_state=42,
                shuffle=False,
            )

            X = pd.DataFrame(X, columns=["foo", "bar", "blitz", "licht"])
            y = pd.Series(y)

            X_train, X_test, y_train, _ = train_test_split(
                X, y, random_state=42, shuffle=False
            )

            model = HistGradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)

            _ = model_feature_importances(X_test, model, show=False)

        assert (
            str(error.value)
            == "The model does not have a `feature_importances_` attribute!"
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
