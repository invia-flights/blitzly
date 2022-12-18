import joblib
import numpy as np
import pandas as pd
import pytest

# pylint: disable=missing-function-docstring, missing-class-docstring


@pytest.fixture()
def invalid_data():
    return "invalid"


@pytest.fixture()
def object_array():
    return np.array(["1", "2", "3"])


@pytest.fixture()
def numpy_1d_data():
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture()
def pandas_1d_data():
    np.random.seed(42)
    return pd.DataFrame(np.random.randn(100), columns=["a"])


@pytest.fixture()
def pandas_series_1d_data():
    np.random.seed(42)
    return pd.Series(np.random.randn(100), name="a")


@pytest.fixture()
def numpy_2d_data():
    np.random.seed(42)
    return np.random.randn(100, 2)


@pytest.fixture()
def pandas_2d_data():
    np.random.seed(42)
    return pd.DataFrame(np.random.randn(100, 2), columns=["a", "b"])


@pytest.fixture()
def numpy_3d_data():
    np.random.seed(42)
    return np.random.randn(100, 2, 2)


@pytest.fixture()
def valid_binary_classification_data():
    return np.array([[1, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1]])


@pytest.fixture()
def valid_multi_class_classification_data():
    return np.array([[1, 0, 2, 1, 0, 2], [2, 2, 1, 1, 0, 1], [2, 0, 1, 2, 0, 1]])


@pytest.fixture(scope="session")
def expected_1d_numpy_pandas():
    return joblib.load(
        "tests/expected_figs/simple_histogram/1d_numpy_pandas_fig.joblib"
    )


@pytest.fixture(scope="session")
def expected_2d_numpy_pandas():
    return joblib.load(
        "tests/expected_figs/simple_histogram/2d_numpy_pandas_fig.joblib"
    )


@pytest.fixture(scope="session")
def valid_binary_confusion_matrix():
    return joblib.load(
        "tests/expected_figs/binary_confusion_matrix/valid_confusion_matrix.joblib"
    )
