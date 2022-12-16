import numpy as np
import pandas as pd
import pytest
from expected_figs.utils import expected_json_string

# pylint: disable=missing-function-docstring, missing-class-docstring

np.random.seed(42)


@pytest.fixture(scope="session")
def list_data():
    return np.random.randn(100).tolist()


@pytest.fixture(scope="session")
def numpy_1d_data():
    return np.random.randn(100)


@pytest.fixture(scope="session")
def pandas_1d_data():
    return pd.DataFrame(np.random.randn(100), columns=["a"])


@pytest.fixture(scope="session")
def pandas_series_1d_data():
    return pd.Series(np.random.randn(100), name="a")


@pytest.fixture(scope="session")
def numpy_2d_data():
    return np.random.randn(100, 2)


@pytest.fixture()
def pandas_2d_data():
    return pd.DataFrame(np.random.randn(100, 2), columns=["a", "b"])


@pytest.fixture(scope="session")
def expected_1d_numpy_pandas():
    return expected_json_string("simple_histogram_with_1d_numpy_pandas")


@pytest.fixture(scope="session")
def expected_2d_numpy_pandas():
    return expected_json_string("simple_histogram_with_2d_numpy_pandas")
