import joblib
import numpy as np
import pandas as pd
import pytest

# pylint: disable=missing-function-docstring, missing-class-docstring


@pytest.fixture()
def list_data():
    np.random.seed(42)
    return np.random.randn(100).tolist()


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
