# pylint: disable=disallowed-name

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure

from blitzly.etc.utils import check_data, save_show_return, update_figure_layout


def _check_data_ready_for_bar(
    data: Union[pd.DataFrame, pd.Series, NDArray],
    group_labels: List[str],
    x_labels: List[str],
    hover_texts: Optional[List[str]] = None,
    errors: Optional[Union[pd.DataFrame, pd.Series, NDArray]] = None,
) -> None:
    """Checks whether the data is ready for plotting.

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


def model_feature_importances(
    X_test: pd.DataFrame,
    model: Any,
    title: str = "Feature importance",
    horizontal: bool = True,
    size: Optional[Tuple[int, int]] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:
    """Creates a bar chart with the feature importance of a model.

    Example:
    ```python
    from blitzly.plots.bar import model_feature_importance
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
        shuffle=False,
    )

    X = pd.DataFrame(X, columns=["foo", "bar", "blitz", "licht"])
    y = pd.Series(y)

    X_train, X_test, y_train, _ = train_test_split(X, y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    model_feature_importances(X_test, model)

    Args:
        X_test (pd.DataFrame): The test data for the model. You can also use the `train` data but it is recommend to use `test`.
        model (Any): The model to get the feature importance from. The model must have a `feature_importances_` attribute!
        title (Optional[str]): The title of the plot. Defaults to "Feature importance".
        horizontal (bool): Whether to plot the bar chart horizontally or vertically.
        size (Optional[Tuple[int, int]]): The size of the plot.
        show (bool): Whether to show the plot.
        write_html_path (Optional[str]): The path to write the plot as HTML.

    Raises:
        AttributeError: If the model does not have a `feature_importances_` attribute.

    Returns:
        BaseFigure: The plotly figure.
    """

    if hasattr(model, "feature_importances_"):
        df = pd.DataFrame(
            {"feature": X_test.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=True)
    else:
        raise AttributeError(
            "The model does not have a `feature_importances_` attribute!"
        )

    if horizontal:
        fig = px.bar(df, x="importance", y="feature")
    else:
        fig = px.bar(df, x="feature", y="importance")

    fig = update_figure_layout(fig, title, size)
    return save_show_return(fig, write_html_path, show)


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
    size: Optional[Tuple[int, int]] = None,
    show_legend: bool = True,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:
    """Creates a bar chart with multiple groups. Each group is represented by a
    bar. The bars are grouped by the x-axis. The number of `group_labels` must
    be equal to the number of rows in the data. The number of `x_labels` must
    be equal to the number of columns in the data.

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
        size (Optional[Tuple[int, int]): Size of the plot.
        show_legend (Optional[bool]): Whether to show the legend.
        show (bool): Whether to show the figure.
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
            error = errors[idx,]
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

    fig = update_figure_layout(
        fig, title, size, (show_legend and mark_x_labels is None)
    )
    return save_show_return(fig, write_html_path, show)
