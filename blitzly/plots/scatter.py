from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure

from blitzly.etc.utils import check_data, save_show_return


def scatter_matrix(
    data: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
    color_dim: Optional[Union[pd.Series, List[str], NDArray]] = None,
    show_upper_half: bool = False,
    diagonal_visible: bool = False,
    show_scale: bool = False,
    title: str = "Scatter matrix",
    marker_line_color: str = "white",
    marker_line_width: float = 0.5,
    marker_color_scale: str = "Plasma",
    size: Optional[int] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Create a scatter matrix plot. It can be used to visualize the relationship between multiple variables.
    The scatter matrix is a grid of scatter plots, one for each pair of variables in the data. The diagonal
    plots are histograms of the corresponding variables. It is also useful for visualizing the distribution of each
    variable.

    Example:
    ```python
    from blitzly.scatter import scatter_matrix
    import numpy as np
    import pandas as pd

    foo = np.random.randn(1000)
    bar = np.random.randn(1000) + 1
    blitz = np.random.randint(2, size=1000)
    licht = np.random.randint(2, size=1000)
    data = np.array([foo, bar, blitz, licht])
    df = pd.DataFrame(data.T, columns=["foo", "bar", "blitz", "licht"])

    scatter_matrix(
        df,
        dimensions=["foo", "bar", "blitz"],
        color_dim=df["licht"],
        title="My first scatter matrix 🙃",
        show_upper_half=True,
        diagonal_visible=False,
        marker_color_scale="Rainbow",
        marker_line_color="blue",
        size=500,
    )
    ```

    Args:
        data (pd.DataFrame): Data to plot.
        dimensions (Optional[List[str]], optional): List of columns to plot. If `None` all columns from the Pandas DataFrame are used.
        color_dim (Optional[Union[pd.Series, List[str], NDArray]]): Color dimension. If `None` no color is used.
        show_upper_half (bool): Show upper half of the scatter matrix.
        diagonal_visible (bool): Show diagonal part of the matrix.
        show_scale (bool): Show color scale.
        title (str): Title of the plot.
        marker_line_color (str): Color of the marker line.
        marker_line_width (float): Width of the marker line.
        marker_color_scale (str): Color scale of the markers.
        size (Optional[int]): Size of the plot.
        show (Optional[bool]): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.

    Returns:
        BaseFigure: The scatter matrix plot.
    """

    if isinstance(color_dim, np.ndarray) and color_dim.ndim != 1:
        raise ValueError("`color_dim` must be a 1-dimensional array!")

    if dimensions is None:
        dimensions = list(data.columns)

    dims = [dict(label=dim, values=data[dim]) for dim in dimensions]

    _ = check_data(data, min_rows=2, min_columns=2)

    fig = go.Figure(
        data=go.Splom(
            dimensions=dims,
            showupperhalf=show_upper_half,
            marker=dict(
                colorscale=marker_color_scale,
                color=color_dim,
                showscale=show_scale,
                line_color=marker_line_color,
                line_width=marker_line_width,
            ),
            diagonal=dict(visible=diagonal_visible),
        )
    )

    fig.update_layout(
        title=f"<i><b>{title}</b></i>",
    )
    if size:
        fig.update_layout(
            width=size,
            height=size,
        )
    return save_show_return(fig, write_html_path, show)
