import numpy as np
import pandas as pd
import pytest

# pylint: disable=missing-function-docstring


@pytest.fixture()
def X_numbers_two_column() -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "foo": np.random.rand(10),
        "bar": np.random.rand(10),
    }
    index = [f"category_{i+1}" for i in range(10)]
    return pd.DataFrame(data, index=index)
