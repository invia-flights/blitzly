from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure

from blitzly.etc.utils import check_data, save_show_return


def simple_histogram(
    data: Union[pd.DataFrame, pd.Series, NDArray],
    show_legend: bool = True,
    opacity: float = 0.75,
    title: str = "Histogram",
    x_label: str = "x",
    y_label: str = "y",
    plotly_kwargs: Optional[dict] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Create a simple histogram from a pandas DataFrame, numpy array, or list of values.

    Example:
    ```python
    import numpy as np
    from blitzly.plots.histogram import simple_histogram

    foo = np.random.randn(500)
    bar = np.random.randn(500) + 1
    data = np.array([foo, bar])

    simple_histogram(
        data,
        title="Histogram of foo and bar",
        x_label="Value",
        y_label="Count",
        write_html_path="the_blitz.html"
    )
    ```

    Args:
        data (Union[pd.DataFrame, pd.Series, NDArray]): The data which should be plotted.
            Either one or multiple columns of data.
        show_legend (Optional[bool]): Whether to show the legend.
        opacity (Optional[float]): The opacity of the histogram.
        title (Optional[str]): The title of the histogram.
        x_label (Optional[str]): The label of the x-axis.
        y_label (Optional[str]): The label of the y-axis.
        plotly_kwargs (Optional[dict]): Additional keyword arguments to pass to Plotly `Histogram`.
        show (Optional[bool]): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.
    """

    data = check_data(data)

    if data.ndim == 1:
        data = [data]
    else:
        data = data.T

    fig = go.Figure()
    for d in data:
        fig.add_trace(go.Histogram(x=d, **plotly_kwargs if plotly_kwargs else {}))

    fig.update_layout(barmode="overlay")
    fig.update_layout(title_text=title)
    fig.update_layout(showlegend=show_legend)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_traces(opacity=opacity)

    return save_show_return(fig, write_html_path, show)
