from typing import List, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def check_data(
    data: Union[
        pd.DataFrame, NDArray, List[Union[pd.Series, NDArray, List[Union[int, float]]]]
    ]
) -> None:
    """
    Checks if the data is valid for plotting. The function checks for:

    - The data is a DataFrame, numpy array, or list of values.

    - If the data is a numpy array, it must be 1- or 2-dimensional.

    Args:
        data (Union[pd.DataFrame, NDArray, List[Union[pd.Series, NDArray, List[Union[int, float]]]]]): The data which should be plotted.
            Either one or multiple columns of data.

    Raises:
        TypeError: If the data is not a DataFrame, numpy array, or list of values.
        ValueError: If the data is a numpy array with more than 2 dimensions.
    """
    if isinstance(data, (pd.DataFrame, pd.Series, np.ndarray, list)) is False:
        raise TypeError(
            f"Data must be a DataFrame, numpy array, or list of values! Type `{data.__class__.__name__}` is not supported.`"
        )

    if isinstance(data, np.ndarray) and data.ndim > 2:
        raise ValueError("NumPy array must be 1- or 2-dimensional!")
