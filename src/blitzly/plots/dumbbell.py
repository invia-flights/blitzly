from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure

from blitzly.etc.utils import check_data, save_show_return


def simple_dumbbell(
    data: Union[pd.DataFrame, NDArray],
    title: str = "Dumbbell plot",
    marker_size: int = 16,
    marker_line_width: int = 8,
    plotly_kwargs: Optional[dict] = None,
    size: Optional[Tuple[int, int]] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Creates a dumbbell plot. These are useful to show the difference between
    two sets of data which have the same categories. For instance, it can be
    used to compare two binary classifiers by plotting the various classification
    metrics.

    Example:
    ```python
    from blitzly.plots.dumbbell import simple_dumbbell
    import numpy as np
    import pandas as pd

    data = {
        "foo": np.random.randn(10),
        "bar": np.random.randn(10),
    }
    index = [f"category_{i+1}" for i in range(10)]
    df = pd.DataFrame(data, index=index)

    simple_dumbbell(df)
    ```

    Args:
        data (Union[pd.DataFrame, NDArray]): Data to plot.
        title (str): Title of the plot.
        marker_size (int): Size of the circular marker of the dumbbells.
        marker_line_width (int): Thickness of the line joining the markers.
        plotly_kwargs (Optional[dict]): Additional keyword arguments to pass to Plotly `go.Scatter`.
        size (Optional[Tuple[int, int]): Size of the plot.
        show (Optional[bool]): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.

    Returns:
        BaseFigure: The dumbbell plot.
    """

    data = check_data(
        data, min_rows=1, min_columns=2, max_columns=2, keep_as_pandas=True
    )

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=data.iloc[:, 0],
                y=data.index,
                mode="markers",
                name=data.columns[0],
                **plotly_kwargs if plotly_kwargs else {},
            ),
            go.Scatter(
                x=data.iloc[:, 1],
                y=data.index,
                mode="markers",
                name=data.columns[1],
                **plotly_kwargs if plotly_kwargs else {},
            ),
        ]
    )

    for index, row in data.iterrows():
        fig.add_shape(
            type="line",
            layer="below",
            x0=row.iloc[0],
            x1=row.iloc[1],
            y0=index,
            y1=index,
            line=dict(width=marker_line_width),
        )

    fig.update_traces(
        marker=dict(size=marker_size),
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
