from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure


def save_show_return(
    fig: BaseFigure, write_html_path: Optional[str] = None, show: bool = True
) -> BaseFigure:

    """
    Saves the figure if needed, shows the figure if needed, and returns a it.

    Args:
        fig (BaseFigure): The Plotly figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.
        show (bool): Whether to show the figure.

    Returns:
        BaseFigure: The Plotly figure.
    """

    if write_html_path:
        fig.write_html(write_html_path)
    if show:
        fig.show()
    return fig


def check_data(
    data: Union[pd.DataFrame, pd.Series, NDArray],
    force_numerical: bool = True,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    min_columns: Optional[int] = None,
    max_columns: Optional[int] = None,
) -> NDArray[Any]:
    """
    Checks if the data is valid for plotting. The function checks for:

    - The data is a DataFrame or numpy array of values.

    - If the data is a numpy array, it must be 1- or 2-dimensional.

    - *(Optional)* if the data is numerical.

    - *(Optional)* If the data is a numpy array, it must have at least `min_rows` rows.

    - *(Optional)* If the data is a numpy array, it must have at most `max_rows` rows.

    - *(Optional)* If the data is a numpy array, it must have at least `min_columns` columns.

    - *(Optional)* If the data is a numpy array, it must have at most `max_columns` columns.

    Args:
        data (Union[pd.DataFrame, pd.Series, NDArray]): The data which should be plotted.
            Either one or multiple columns of data.
        force_numerical (Optional[bool]): Whether to force the data to be numerical.
        min_rows (Optional[int]): The minimum number of rows the data must have.
        max_rows (Optional[int]): The maximum number of rows the data must have.
        min_columns (Optional[int]): The minimum number of columns the data must have.
        max_columns (Optional[int]): The maximum number of columns the data must have.

    Raises:
        TypeError: If the data is not a DataFrame, numpy array, or list of values.
        TypeError: If the data is a numpy array with a non-numerical `dtype`.
        ValueError: If the data is a numpy array with more than 2 dimensions.
    """

    if isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)) is False:
        raise TypeError(
            f"""
            Invalid data type! Type `{data.__class__.__name__}` is not supported.
            Please choose between a DataFrame, numpy array, or list of values.
        """
        )

    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()

    if force_numerical and data.dtype not in [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ]:
        raise TypeError("Data must be numerical (`np.number`)!")

    if data.ndim > 2:
        raise ValueError("NumPy array must be 1- or 2-dimensional!")

    if min_rows and data.shape[0] < min_rows:
        raise ValueError(f"The data must have at least {min_rows} row(s)!")

    if max_rows and data.shape[0] > max_rows:
        raise ValueError(f"The data must have a maximum of {max_rows} row(s)!")

    if min_columns and data.shape[1] < min_columns:
        raise ValueError(f"The data must have at least {min_columns} column(s)!")

    if max_columns and data.shape[1] > max_columns:
        raise ValueError(f"The data must have a maximum of {max_columns} column(s)!")

    return data.copy()