from typing import List, Optional, Tuple, Union

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
    from blitzly.plots.scatter import scatter_matrix
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


def multi_scatter(
    data: pd.DataFrame,
    x_y_columns: List[Tuple[str, str]],
    modes: Optional[List[str]] = None,
    title: str = "Scatter plot",
    size: Optional[Tuple[int, int]] = None,
    show_legend: bool = True,
    plotly_kwargs: Optional[dict] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Create a multi scatter plot. It can be used to visualize the relationship between
    multiple variables from the same Pandas DataFrame.

    Example:
    ```python
    from blitzly.plots.scatter import multi_scatter
    import numpy as np
    import pandas as pd

    random_a = np.linspace(0, 1, 100)
    random_b = np.random.randn(100) + 5
    random_c = np.random.randn(100)
    random_d = np.random.randn(100) - 5
    data = np.array([random_a, random_b, random_c, random_d]).T

    multi_scatter(
        data=pd.DataFrame(data, columns=["foo", "bar", "blitz", "licht"]),
        x_y_columns=[("foo", "bar"), ("foo", "blitz"), ("foo", "licht")],
        modes=["lines", "markers", "lines+markers"],
        plotly_kwargs={"line": {"color": "black"}},
    ```

    Args:
        data (pd.DataFrame): Data to plot. Must be a Pandas DataFrame.
        x_y_columns (List[Tuple[str, str]]): List of tuples containing the x and y columns.
            Those columns will be used for `x` and `y` in the scatter plot.
            Since it is a multi scatter plot, multiple columns can be used by passing a list of tuples.
        modes (Optional[List[str]]): List of modes for the scatter plot. If `None` the `"markers"` mode is used.
        title (str): Title of the plot.
        size (OptionalTuple[int, int]): Size of the plot - height and width.
        show_legend (bool): Whether to show the legend.
        plotly_kwargs (Optional[dict]): Additional plotly kwargs.
        show (Optional[bool]): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.


    Returns:
        BaseFigure: The multi scatter plot.
    """

    df = check_data(data, min_rows=1, min_columns=1, keep_as_pandas=True)

    if isinstance(df, pd.DataFrame) is False:
        raise TypeError("`data` must be a Pandas DataFrame!")

    if len([i for i in list(sum(x_y_columns, ())) if i not in df.columns]) > 0:
        raise ValueError(
            f"""
            Columns {list(set(list(sum(x_y_columns, ()))) - set(df.columns))} not in `data`!
            All columns passed in `x_y_columns` must be in `data`.
            """
        )

    if modes and len(modes) != len(x_y_columns):
        raise ValueError(
            f"""
            Length of `modes` ({len(modes)}) must be equal to length of `x_y_columns` ({len(x_y_columns)})!
            Or `modes` must be `None`.
            """
        )

    fig = go.Figure()
    for idx, item in enumerate(x_y_columns):
        fig.add_trace(
            go.Scatter(
                x=df[item[0]],
                y=df[item[1]],
                mode=modes[idx] if modes else "markers",
                name=list(df.columns)[idx],
                showlegend=show_legend,
                **plotly_kwargs or {},
            )
        )

    fig.update_layout(
        title=f"<i><b>{title}</b></i>",
    )
    if size:
        fig.update_layout(
            width=size[0],
            height=size[1],
        )

    return save_show_return(fig, write_html_path, show)
