from typing import Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure

from blitzly.etc.utils import check_data, save_show_return, update_figure_layout


def simple_slope(
    data: Union[pd.DataFrame, NDArray],
    title: str = "Slope plot",
    marker_size: int = 16,
    marker_line_width: int = 4,
    margin_size: int = 250,
    plotly_kwargs: Optional[dict] = None,
    size: Optional[Tuple[int, int]] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """Creates a slope plot. These are useful to show the difference between
    two sets of data which have the same categories. For instance, it can be
    used to compare two binary classifiers by plotting the various
    classification metrics.

    Example:
    ```python
    from blitzly.plots.slope import simple_slope
    import numpy as np
    import pandas as pd

    data = {
        "foo": np.random.randn(10),
        "bar": np.random.randn(10),
    }
    index = [f"category_{i+1}" for i in range(10)]
    df = pd.DataFrame(data, index=index)

    simple_slope(df)
    ```

    Args:
        data (Union[pd.DataFrame, NDArray]): Data to plot.
        title (str): Title of the plot.
        marker_size (int): Size of the circular marker.
        marker_line_width (int): Thickness of the line joining the markers.
        margin_size (int): Margin for displaying text labels in pixels.
        plotly_kwargs (Optional[dict]): Additional keyword arguments to pass to Plotly `go.Scatter`.
        size (Optional[Tuple[int, int]): Size of the plot.
        show (bool): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.

    Returns:
        BaseFigure: The slope plot.
    """

    data = check_data(data, min_rows=1, min_columns=2, max_columns=2, as_pandas=True)

    data_max = data.to_numpy().max()
    data_min = data.to_numpy().min()
    data_range = data_max - data_min

    y_range_max = data_max + 0.05 * data_range
    y_range_min = data_min - 0.05 * data_range

    fig = go.Figure()

    for column_idx in range(2):
        fig.add_trace(
            go.Scatter(
                x=[column_idx, column_idx],
                y=[y_range_max, y_range_min],
                mode="lines",
                line={
                    "color": "black",
                    "width": 2,
                },
                showlegend=False,
            )
        )

    for index, row in data.iterrows():
        if row.iloc[0] > row.iloc[1]:
            line_color = "red"
        else:
            line_color = "green"

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[row.iloc[0], row.iloc[1]],
                mode="markers+lines+text",
                marker={
                    "size": marker_size,
                },
                line={
                    "width": marker_line_width,
                    "color": line_color,
                },
                text=index,
                textposition=["middle left", "middle right"],
                showlegend=False,
                **plotly_kwargs if plotly_kwargs else {},
            )
        )

    xaxis_offset = margin_size / size[0] if size is not None else 1
    fig.update_layout(
        xaxis={
            "tickvals": [0, 1],
            "ticktext": data.columns,
            "range": [-xaxis_offset, 1 + xaxis_offset],
        }
    )

    fig = update_figure_layout(fig, title, size)
    return save_show_return(fig, write_html_path, show)
