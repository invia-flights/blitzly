from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure

from blitzly.etc.utils import check_data


def simple_histogram(
    data: Union[
        pd.DataFrame, NDArray, List[Union[pd.Series, NDArray, List[Union[int, float]]]]
    ],
    showlegend: bool = True,
    opacity: float = 0.75,
    title: str = "Histogram",
    x_label: str = "x",
    y_label: str = "y",
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Create a simple histogram from a pandas DataFrame, numpy array, or list of values.

    Example:
    ```python
    import numpy as np
    from blitzly import simple_histogram

    foo = np.random.randn(500)
    bar = np.random.randn(500) + 1

    simple_histogram(
        [a, b],
        title="Histogram of foo and bar",
        x_label="Value",
        y_label="Count",
        write_html_path="the_blitz.html"
    )
    ```

    Args:
        data (Union[pd.DataFrame, NDArray, List[Union[pd.Series, NDArray, List[Union[int, float]]]]]): The data which should be plotted.
            Either one or multiple columns of data.
        showlegend (Optional[bool]): Whether to show the legend.
        opacity (Optional[float]): The opacity of the histogram.
        title (Optional[str]): The title of the histogram.
        x_label (Optional[str]): The label of the x-axis.
        y_label (Optional[str]): The label of the y-axis.
        show (Optional[bool]): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file. If None, the histogram will not be saved.
    """

    check_data(data)

    if (isinstance(data, np.ndarray) and data.ndim == 1) or isinstance(
        data, (list, pd.Series)
    ):
        data = [data]

    elif isinstance(data, np.ndarray):
        data = data.T

    elif isinstance(data, pd.DataFrame):
        data = [data[col] for col in data.columns]

    fig = go.Figure()
    for d in data:
        fig.add_trace(go.Histogram(x=d))

    fig.update_layout(barmode="overlay")
    fig.update_layout(title_text=title)
    fig.update_layout(showlegend=showlegend)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_traces(opacity=opacity)

    if write_html_path:
        fig.write_html(write_html_path)
    if show:
        fig.show()
    return fig
