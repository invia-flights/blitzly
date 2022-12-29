from typing import List, Optional, Tuple

import numpy as np
import plotly.subplots as sp
from plotly.basedatatypes import BaseFigure, BaseTraceType

from blitzly.etc.utils import save_show_return, update_figure_layout


def make_subplots(
    subfig_list: List[BaseFigure],
    shape: Tuple[int, int],
    title: Optional[str] = None,
    column_widths: Optional[List[float]] = None,
    fill_row: bool = False,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    plotly_kwargs: Optional[dict] = None,
    size: Optional[Tuple[int, int]] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Creates subplots using a provided list of figure objects.
    `plotly.subplots.make_subplots` requires the use of traces. This function is an
    alternative implementation that directly uses previously-created figure objects.

    Example:
    ```python
    from blitzly.subplots import make_subplots
    from blitzly.plots.histogram import simple_histogram
    import numpy as np

    fig1 = simple_histogram(np.random.randn(100), show=False)
    fig2 = simple_histogram(np.random.randn(100), show=False)

    make_subplots([fig1, fig2], (1, 2))
    ```

    Args:
        subfig_list (List[BaseFigure]): A list of figure objects.
        shape (Tuple[int, int]): The grid shape of the subplots.
        title (str): Title of the plot.
        column_width (Optional[List[float]]): The width of each column in the subplot grid.
        fill_row (bool): If True, resize the last subplot in the grid to fill the row.
        shared_xaxes (bool): Share the x-axis labels along each column.
        shared_yaxes (bool): Share the y-axis labels along each row.
        plotly_kwargs (Optional[dict]): Additional keyword arguments to pass to Plotly `subplots.make_subplots`.
        size (Optional[Tuple[int, int]): Size of the plot.
        show (bool): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.

    Returns:
        BaseFigure: The provided list figures as subplots in a single figure object.
    """

    _check_shape_for_subplots(subfig_list, shape)

    subfig_traces: List[List[BaseTraceType]] = [[] for _ in subfig_list]

    for idx, subfig in enumerate(subfig_list):
        for trace in range(len(subfig["data"])):
            subfig_traces[idx].append(subfig["data"][trace])

    subplot_titles = [subfig.layout.title.text for subfig in subfig_list]

    subplot_axes_labels = [
        [subfig.layout.xaxis.title.text for subfig in subfig_list],
        [subfig.layout.yaxis.title.text for subfig in subfig_list],
    ]

    specs: List[List[dict]] = [[{} for _ in range(shape[1])] for _ in range(shape[0])]
    n_missing_slots = int(np.prod(shape) - len(subfig_list))
    if n_missing_slots in range(1, shape[1]) and fill_row:
        specs[-1][-1 - n_missing_slots]["colspan"] = 1 + n_missing_slots

    fig = sp.make_subplots(
        rows=shape[0],
        cols=shape[1],
        subplot_titles=subplot_titles,
        column_widths=column_widths,
        specs=specs,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        **plotly_kwargs if plotly_kwargs else {},
    )

    for idx, traces in enumerate(subfig_traces):
        row = idx // shape[1]
        col = idx % shape[1]
        for trace in traces:
            fig.append_trace(trace, row=row + 1, col=col + 1)

        if row != shape[0] - 1 and shared_xaxes:
            subplot_axes_labels[0][idx] = ""
        if col != 0 and shared_yaxes:
            subplot_axes_labels[1][idx] = ""
        fig.update_xaxes(
            title_text=subplot_axes_labels[0][idx], row=row + 1, col=col + 1
        )
        fig.update_yaxes(
            title_text=subplot_axes_labels[1][idx], row=row + 1, col=col + 1
        )

    fig.update_layout(showlegend=False)
    fig = update_figure_layout(fig, title, size)
    return save_show_return(fig, write_html_path, show)


def _check_shape_for_subplots(
    subfig_list: List[BaseFigure], shape: Tuple[int, int]
) -> None:
    """
    Checks whether the `shape` is compatible for making subplots.

    Args:
        subfig_list (List[BaseFigure]): A list of figure objects.
        shape (Tuple[int, int]): The grid shape of the subplots.

    Raises:
        ValueError: If the provided `shape` is too small for the list of subfigures.
    """

    if len(subfig_list) > np.prod(shape):
        raise ValueError(
            f"The number of subfigures ({len(subfig_list)}) is too large for the provided `shape` {shape}."
        )
