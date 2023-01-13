from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.decomposition as sk_decomp
import sklearn.preprocessing as sk_pre
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure
from sklearn.manifold import TSNE

from blitzly.etc.utils import check_data, save_show_return, update_figure_layout
from blitzly.subplots import make_subplots


def dimensionality_reduction(
    data: pd.DataFrame,
    reduction_funcs: Union[str, List[str]],
    n_components: int,
    target_column: str,
    title: str = "Dimensionality reduction plot",
    scaler_func: str = "StandardScaler",
    size: Optional[Tuple[int, int]] = None,
    show_legend: Optional[bool] = None,
    reduction_func_kwargs: Optional[dict] = None,
    plotly_kwargs: Optional[dict] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """Create a scatter plot of the dimensionality reduction representation of
    the data provided. Multiple dimensionality reduction functions can be used.
    The data is scaled using the `scaler_func`. If multiple functions are used,
    the plots are arranged in a grid using.

    [`make_subplots`](https://invia-flights.github.io/blitzly/plots/subplots/#blitzly.subplots.make_subplots) from blitzly ⚡️.

    Example:
    ```python
    from blitzly.plots.scatter import dimensionality_reduction
    import plotly.express as px

    df = px.data.iris()
    fig = dimensionality_reduction(
        df,
        n_components=2,
        target_column="species",
        reduction_funcs=["PCA", "TNSE"],
    )
    ```

    Args:
        data (pd.DataFrame): Data to plot.
        reduction_funcs (Union[str, List[str]]): Dimensionality reduction function(s) to use. The following functions are supported:
            NMF, PCA, IncrementalPCA, KernelPCA, MiniBatchSparsePCA, SparsePCA, TruncatedSVD, TSNE.
        n_components (int): Number of components to use. This parameter is passed to the dimensionality reduction function.
        target_column (str): Column to use as the color dimension.
        title (Optional[str]): Title of the plot. Defaults to "Dimensionality reduction plot".
        scaler_func (Optional[str]): Scaler function to use. Defaults to "StandardScaler". The following functions are supported:
            StandardScaler, MinMaxScaler.
        size (Optional[Tuple[int, int]): Size of the full plot.
        show_legend (Optional[bool]): Whether to show the legend.
        show (bool): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.
    """

    func_list = [
        "NMF",
        "PCA",
        "IncrementalPCA",
        "KernelPCA",
        "MiniBatchSparsePCA",
        "SparsePCA",
        "TruncatedSVD",
        "TSNE",
    ]

    if isinstance(reduction_funcs, str):
        reduction_funcs = [reduction_funcs]

    if scaler_func not in ["StandardScaler", "MinMaxScaler"]:
        raise ValueError(
            f"scaler_func must be one of ['StandardScaler', 'MinMaxScaler]! `{scaler_func}` not supported."
        )

    if n_components == 3 and len(reduction_funcs) > 1:
        raise Warning(
            "Cannot plot more than one plot in 3D! Please either set `n_components` to 2 or `reduction_funcs` to one function."
        )

    df = check_data(
        data, min_rows=1, min_columns=1, as_pandas=True, only_numerical_values=False
    )

    numerical_df = data.select_dtypes(include=[int, float])
    scaler = getattr(sk_pre, scaler_func)()
    numerical_df = scaler.fit_transform(numerical_df)

    plots = []
    for func in reduction_funcs:

        if func not in func_list:
            raise ValueError(
                f"reduction_func must be one of {func_list}! `{func}` not supported."
            )

        if func == "TSNE":
            method: Callable = TSNE
        else:
            method: Callable = getattr(sk_decomp, func)  # type: ignore

        red_func = method(n_components=n_components, **reduction_func_kwargs or {})
        projections = red_func.fit_transform(numerical_df)

        if n_components == 2:
            fig = px.scatter(
                projections,
                x=0,
                y=1,
                color=df[target_column],
                labels={"color": target_column},
                **plotly_kwargs or {},
            )
        elif n_components == 3:
            fig = px.scatter_3d(
                projections,
                x=0,
                y=1,
                z=2,
                color=df[target_column],
                labels={"color": target_column},
                **plotly_kwargs or {},
            )
        else:
            raise ValueError("`n_components` must be 2 or 3!")

        plots.append(update_figure_layout(fig, title + f" ({func})", size=None))

    len_plots = len(plots)
    shape = (int(np.ceil(len_plots / 2)), min(len_plots, 2))

    show_legend = show_legend if show_legend and len(plots) == 1 else False

    fig = (
        make_subplots(
            plots,
            shape,
            size=size,
            fill_row=True,
            show_legend=show_legend,
            show=False,
        )
        if n_components == 2
        else plots[0]
    )
    return save_show_return(fig, write_html_path, show)


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
    size: Optional[Tuple[int, int]] = None,
    show_legend: Optional[bool] = False,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """Create a scatter matrix plot. It can be used to visualize the
    relationship between multiple variables. The scatter matrix is a grid of
    scatter plots, one for each pair of variables in the data. The diagonal
    plots are histograms of the corresponding variables. It is also useful for
    visualizing the distribution of each variable.

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
        size=(500, 500),
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
        size (Optional[Tuple[int, int]): Size of the plot.
        show_legend (Optional[bool]): Whether to show the legend.
        show (bool): Whether to show the figure.
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

    fig = update_figure_layout(fig, title, size, show_legend)
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

    """Create a multi scatter plot. It can be used to visualize the
    relationship between multiple variables from the same Pandas DataFrame.

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
        show (bool): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.


    Returns:
        BaseFigure: The multi scatter plot.
    """

    df = check_data(data, min_rows=1, min_columns=1, as_pandas=True)

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
                **plotly_kwargs or {},
            )
        )

    fig = update_figure_layout(fig, title, size, show_legend)
    return save_show_return(fig, write_html_path, show)
