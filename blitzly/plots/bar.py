# pylint: disable=disallowed-name

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure

from blitzly.etc.utils import check_data, save_show_return


def multi_bar(
    data: Union[pd.DataFrame, pd.Series, NDArray],
    group_labels: List[str],
    x_labels: List[str],
    mark_x_labels: Optional[List[str]] = None,
    mark_x_label_color: str = "crimson",
    title: str = "Bar chart",
    stack: bool = False,
    text_position: str = "none",
    hover_texts: Optional[List[str]] = None,
    errors: Optional[Union[pd.DataFrame, pd.Series, NDArray]] = None,
    show_legend: bool = True,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Creates a bar chart with multiple groups. Each group is represented by a bar. The bars are grouped by the x-axis.
    The number of `group_labels` must be equal to the number of rows in the data.
    The number of `x_labels` must be equal to the number of columns in the data.

    Example:
    ```python
    from blitzly.plots.bar import multi_bar
    import numpy as np

    data=np.array([[1, 2, 3], [4, 5, 6]])
    errors=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),

    multi_bar(
        data,
        x_labels=["X1", "X2", "X3"],
        group_labels=["Z1", "Z2"],
        hover_texts=["foo", "bar", "blitzly"],
        errors=errors
    )
    ```

    Args:
        data (Union[pd.DataFrame, pd.Series, NDArray]): The data to plot.
        group_labels (List[str]): The labels for the groups.
        x_labels (List[str]): The labels for the x-axis.
        mark_x_labels (Optional[List[str]]): The bars of `x_label` which should be marked.
        mark_x_label_color (str): The color of the marked bars.
        title (Optional[str]): The title of the bar chart.
        stack (Optional[bool]): Whether to stack the bars. Values are summed up by columns.
            By default, the bars are grouped. Stacked bars don't support errors. If provided, they will be ignored.
        text_position (Optional[str]): The position of the text. Can be "auto", "inside", "outside", "none".
        hover_texts (Optional[List[str]]): The hover texts for the data.
        errors (Optional[Union[pd.DataFrame, pd.Series, NDArray]]): The errors for the data.
        show_legend (Optional[bool]): Whether to show the legend.
        show (Optional[bool]): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.

    Returns:
        BaseFigure: The figure.
    """

    data = check_data(data)

    if isinstance(errors, np.ndarray):
        errors = check_data(errors)

    _check_data_ready_for_bar(data, group_labels, x_labels, hover_texts, errors)

    colors = None
    if mark_x_labels:
        colors = [
            "lightslategray",
        ] * len(x_labels)
        for mark in mark_x_labels:
            colors[x_labels.index(mark)] = mark_x_label_color

    fig = go.Figure()
    for idx, dt in enumerate(data):
        error_dict = None
        if isinstance(errors, np.ndarray) and stack is False:
            error = errors[
                idx,
            ]
            error_dict = dict(type="data", array=error, visible=True)
        fig.add_trace(
            go.Bar(
                name=group_labels[idx],
                x=x_labels,
                y=dt,
                error_y=error_dict,
                hovertext=hover_texts,
                text=dt,
                textposition=text_position,
                marker_color=colors,
            )
        )

    fig.update_layout(barmode="stack" if stack else "group", xaxis_tickangle=-45)
    fig.update_layout(showlegend=show_legend and mark_x_labels is None)
    fig.update_layout(title_text=f"<i><b>{title}</b></i>")

    return save_show_return(fig, write_html_path, show)


def _check_data_ready_for_bar(
    data: Union[pd.DataFrame, pd.Series, NDArray],
    group_labels: List[str],
    x_labels: List[str],
    hover_texts: Optional[List[str]] = None,
    errors: Optional[Union[pd.DataFrame, pd.Series, NDArray]] = None,
) -> None:
    """
    Checks whether the data is ready for plotting.

    Args:
        data (Union[pd.DataFrame, pd.Series, NDArray]): The data to plot.
        group_labels (List[str]): The labels for the groups.
        x_labels (List[str]): The labels for the x-axis.
        hover_texts (Optional[List[str]]): The hover texts for the data.
        errors (Optional[Union[pd.DataFrame, pd.Series, NDArray]]): The errors for the data.

    Raises:
        ValueError: If the `data` is not at least two-dimensional.
        ValueError: If the number of `group_labels` does not match the number of rows in the `data`.
        ValueError: If the number of `x_labels` does not match the number of columns in the `data`.
        ValueError: If the shape of the `errors` does not match the shape of the `data`.
        ValueError: If the number of `hover_texts` does not match the number of columns in the `data`.
    """

    if data.ndim < 2:
        raise ValueError(
            f"The `data` must be at least two-dimensional! Got {data.ndim} dimension."
        )

    if data.shape[0] != len(group_labels):
        raise ValueError(
            f"The number of `group_labels` ({len(group_labels)}) does not match the number of rows in the `data` ({data.shape[0]})!"
        )

    if data.shape[1] != len(x_labels):
        raise ValueError(
            f"The number of `x_labels` ({len(x_labels)}) does not match the number of columns in the `data` ({data.shape[1]})!"
        )

    if isinstance(errors, np.ndarray) and data.shape != errors.shape:
        raise ValueError(
            f"The shape of the `errors` ({errors.shape}) does not match the shape of the `data` ({data.shape})!"
        )

    if hover_texts and len(hover_texts) != data.shape[1]:
        raise ValueError(
            f"The number of `hover_texts` ({len(hover_texts)}) does not match the number of columns in the `data` ({data.shape[1]})!"
        )
