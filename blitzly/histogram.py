from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray


def simple_histogram(
    data: Union[
        pd.DataFrame, NDArray, List[Union[pd.Series, NDArray, List[Union[int, float]]]]
    ],
    showlegend: bool = True,
    opacity: float = 0.75,
    title: str = "Histogram",
    title_x: str = "x",
    title_y: str = "y",
    write_html_path: Optional[str] = None,
) -> None:

    """
    Create a simple histogram from a pandas DataFrame, numpy array, or list of values.

    Example:
    ```python
    import numpy as np
    import blitzly.histogram as blitzhist

    foo = np.random.randn(500)
    bar = np.random.randn(500) + 1

    blitzhist.simple_histogram(
        [a, b],
        title="Histogram of foo and bar",
        title_x="Value",
        title_y="Count",
        write_html_path="the_blitz.html"
    )
    ```

    Args:
        data (Union[pd.DataFrame, NDArray, List[Union[pd.Series, NDArray, List[Union[int, float]]]]]): The data which should be plotted.
            Either one or multiple columns of data.
        showlegend (Optional[bool]): Whether to show the legend.
        opacity (Optional[float]): The opacity of the histogram.
        title (Optional[str]): The title of the histogram.
        title_x (Optional[str]): The title of the x-axis.
        title_y (Optional[str]): The title of the y-axis.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file. If None, the histogram will not be saved.
    """

    if isinstance(data, list) and isinstance(data[0], np.ndarray) and data[0].ndim != 1:
        raise ValueError("Data must be 1-dimensional!")

    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = [data]

    if isinstance(data, pd.DataFrame):
        data = [data[col] for col in data.columns]

    fig = go.Figure()
    for d in data:
        fig.add_trace(go.Histogram(x=d))

    fig.update_layout(barmode="overlay")
    fig.update_layout(title_text=title)
    fig.update_layout(showlegend=showlegend)
    fig.update_xaxes(title_text=title_x)
    fig.update_yaxes(title_text=title_y)
    fig.update_traces(opacity=opacity)

    if write_html_path:
        fig.write_html(write_html_path)
    fig.show()
