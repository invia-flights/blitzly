import numpy as np
import pandas as pd
import pytest

# pylint: disable=missing-function-docstring, missing-class-docstring


def expected_json_string(filename: str) -> str:

    with open(f"tests/expected_figs/json_strings/{filename}", encoding="utf-8") as file:
        json_sting = file.read().rstrip().replace("'", "")
    return json_sting


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
    return expected_json_string("simple_histogram_with_1d_numpy_pandas")


@pytest.fixture(scope="session")
def expected_2d_numpy_pandas():
    return expected_json_string("simple_histogram_with_2d_numpy_pandas")
