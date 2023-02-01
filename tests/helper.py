import io
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from plotly.graph_objects import Figure


def fig_to_array(fig: Figure) -> NDArray[Any]:
    """Convert a plotly figure to a numpy array.

    Args:
        fig (plotly.graph_objects.Figure): The plotly figure which should be converted.

    Returns:
        NDArray[Any]: The numpy array which represents the Plotly figure.
    """
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.array(img)
