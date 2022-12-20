from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.basedatatypes import BaseFigure
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from blitzly.etc.utils import check_data, save_show_return


def binary_confusion_matrix(
    data: Union[pd.DataFrame, NDArray],
    positive_class_label: str = "positive class",
    negative_class_label: str = "negative class",
    title: str = "Confusion matrix",
    normalize: Optional[str] = None,
    show_scale: bool = False,
    color_scale: str = "Plasma",
    plotly_kwargs: Optional[dict] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Creates a confusion matrix for binary classification.

    Example:
    ```python
    from blitzly.plots.matrix import binary_confusion_matrix
    import numpy as np

    data = np.array([[1, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1]])
    binary_confusion_matrix(data, write_html_path="the_blitz.html")
    ```

    Args:
        data (Union[pd.DataFrame, NDArray]): The data which should be plotted.
        positive_class_label (Optional[str]): The label of the positive class.
        negative_class_label (Optional[str]): The label of the negative class.
        title (Optional[str]): The title of the confusion matrix.
        normalize (Optional[str]): Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
        show_scale (Optional[bool]): Whether to show the color scale.
        color_scale (Optional[str]): The color scale of the confusion matrix.
        plotly_kwargs (Optional[dict]): Additional keyword arguments for Plotly.
        show (Optional[bool]): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.

    Returns:
        BaseFigure: The confusion matrix.
    """

    check_data(data, min_columns=2, max_rows=2)

    X = sk_confusion_matrix(
        data[0], data[1], normalize=normalize
    )  # [[tn, fp],[fn, tp]]

    classes = [negative_class_label, positive_class_label]
    fig = ff.create_annotated_heatmap(
        X,
        x=classes,
        y=classes,
        annotation_text=[[str(y) for y in x] for x in X],
        colorscale=color_scale,
        hovertext=[
            ["True negative", "False positive"],
            ["False negative", "True positive"],
        ],
        **(plotly_kwargs or {}),
    )

    fig.update_layout(
        title_text=f"<i><b>{title}</b></i>",
        xaxis=dict(title="Predicted value"),
        yaxis=dict(title="Real value"),
    )

    fig.update_layout(margin=dict(t=100, l=180))
    fig["data"][0]["showscale"] = show_scale

    return save_show_return(fig, write_html_path, show)


def pearson_corr_matrix(
    data: Union[pd.DataFrame, NDArray],
    title: str = "Pearson correlation matrix",
    show_scale: bool = False,
    size: Optional[int] = None,
    decimal_places: int = 4,
    labels: Optional[List[str]] = None,
    row_var: bool = True,
    plotly_kwargs: Optional[dict] = None,
    show: bool = True,
    write_html_path: Optional[str] = None,
) -> BaseFigure:

    """
    Creates Pearson product-moment correlation coefficients matrix
    using NumPy's [`corrcoef`](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html#numpy-corrcoef) function.
    Please refer to the NumPy documentation for [`cov`](https://numpy.org/doc/stable/reference/generated/numpy.cov.html#numpy.cov)
    for more detail. The relationship between the correlation coefficient matrix, R, and the covariance matrix, C, is:

    $$
    R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }
    $$

    The values of R are between -1 and 1, inclusive.

    Example:
    ```python
    from blitzly.matrix import pearson_corr_matrix
    import numpy as np
    import pandas as pd

    data = np.array(
        [
            [0.77395605, 0.43887844, 0.85859792],
            [0.69736803, 0.09417735, 0.97562235],
            [0.7611397, 0.78606431, 0.12811363],
        ]
    )
    df = pd.DataFrame(data, columns=["foo", "bar", "blitzly"])
    pearson_corr_matrix(df, write_html_path="the_blitz.html")
    ```

    Args:
        data (Union[pd.DataFrame, NDArray]): The data which should be plotted.
        title (Optional[str]): The title of the correlation matrix.
        show_scale (Optional[bool]): Whether to show the color scale.
        decimal_places (Optional[int]): The number of decimal places to round the values to. This only applies to the values shown on the plot.
        size (Optional[int]): The size of the correlation matrix.
        labels (Optional[List[str]]): The labels of the columns. If a Pandas DataFrame is passed, the column names will be used.
        row_var (Optional[bool]): If rowvar is True (default), then each row represents a variable, with observations in the columns.
            Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.
        plotly_kwargs (Optional[dict]): Additional keyword arguments for Plotly.
        show (Optional[bool]): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.
    """

    if isinstance(data, pd.DataFrame):
        labels = data.columns

    data = check_data(data, only_square_matrix=True, min_columns=2, min_rows=2)

    correlation = np.corrcoef(data, rowvar=row_var)

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation,
            text=np.round(correlation, decimals=decimal_places),
            texttemplate="%{text}",
            x=labels,
            y=labels,
        ),
        **(plotly_kwargs or {}),
    )

    if size:
        fig.update_layout(height=size, width=size)
    fig.update_layout(title_text=f"<i><b>{title}</b></i>")
    fig["data"][0]["showscale"] = show_scale

    return save_show_return(fig, write_html_path, show)
