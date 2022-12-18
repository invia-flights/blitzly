from typing import Optional, Union

import pandas as pd
import plotly.figure_factory as ff
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
    from blitzly.matrix import binary_confusion_matrix
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
        plotly_kwargs (Optional[dict]): Additional keyword arguments for plotly.
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
