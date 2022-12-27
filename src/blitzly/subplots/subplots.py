from typing import List, Optional, Tuple

import numpy as np
import plotly.subplots as sp
from plotly.basedatatypes import BaseFigure

from blitzly.etc.utils import save_show_return


def make_subplots(
    subfig_list: List[BaseFigure],
    shape: Tuple[int, int],
    title: Optional[str] = None,
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
        size (Optional[Tuple[int, int]): Size of the plot.
        show (bool): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.

    Returns:
        BaseFigure: The provided list figures as subplots in a single figure object.
    """

    _check_shape_for_subplots(subfig_list, shape)

    subfig_traces: List = [[] for _ in subfig_list]

    for idx, subfig in enumerate(subfig_list):
        for trace in range(len(subfig["data"])):
            subfig_traces[idx].append(subfig["data"][trace])

    fig = sp.make_subplots(rows=shape[0], cols=shape[1])

    for idx, traces in enumerate(subfig_traces):
        row = idx // shape[1]
        col = idx % shape[1]
        for trace in traces:
            fig.append_trace(trace, row=row + 1, col=col + 1)

    if title:
        fig.update_layout(
            title=f"<i><b>{title}</b></i>",
        )
    if size:
        fig.update_layout(
            width=size[0],
            height=size[1],
        )
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

    if len(subfig_list) > np.sum(shape):
        raise ValueError(
            f"The number of subfigures ({len(subfig_list)}) is too large for the provided `shape` {shape}."
        )
