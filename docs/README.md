<img src="https://github.com/invia-flights/blitzly/raw/main/docs/assets/images/icon.png" alt="blitzly logo" width="200" height="200"/><br>
# blitzly ⚡️
***Lightning-fast way to get plots with Plotly***

[![DeployPackage](https://github.com/invia-flights/blitzly/actions/workflows/deploy-package.yml/badge.svg)](https://github.com/invia-flights/blitzly/actions/workflows/deploy-package.yml)
[![Testing](https://github.com/invia-flights/blitzly/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/invia-flights/blitzly/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/invia-flights/blitzly/branch/develop/graph/badge.svg?token=ROCDJJV8JV)](https://codecov.io/gh/invia-flights/blitzly)
[![pypi](https://img.shields.io/pypi/v/blitzly)](https://pypi.org/project/blitzly/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/blitzly)](https://pypistats.org/packages/blitzly)
[![python version](https://img.shields.io/pypi/pyversions/blitzly?logo=python&logoColor=yellow)](https://www.python.org/downloads/)
[![docs](https://img.shields.io/badge/docs-mkdoks%20material-blue)](https://invia-flights.github.io/blitzly/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![license](https://img.shields.io/github/license/invia-flights/blitzly)](https://github.com/invia-flights/blitzly/blob/main/LICENSE)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://github.com/PyCQA/isort)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
## Introduction 🎉
Plotly is great and powerful. But with great power comes great responsibility 🕸. And sometimes you just want to get a plot up and running as fast as possible. That's where blitzly ⚡️ comes in. It provides a set of functions that allow you to create plots with Plotly in a lightning-fast way. It's not meant to replace Plotly, but rather to complement it.

## Install the package 📦
If you are using [pip](https://pip.pypa.io/en/stable/), you can install the package with the following command:
```bash
pip install blitzly
```

If you are using [Poetry](https://python-poetry.org/), you can install the package with the following command:
```bash
poetry add blitzly
```
## installing dependencies 🧑‍🔧
With [pip](https://pip.pypa.io/en/stable/):
```bash
pip install -r requirements.txt
```

With [Poetry](https://python-poetry.org/):
```bash
poetry install
```
## Available plots (so far 🚀)
| Module | Method | Description |
| ------ | ------ | ----------- |
| [`bar`](https://invia-flights.github.io/blitzly/plots/bar/) | [`multi_chart`](https://invia-flights.github.io/blitzly/plots/bar/#blitzly.plots.bar.multi_chart) | Creates a bar chart with multiple groups. |
| [`dumbbell`](https://invia-flights.github.io/blitzly/plots/dumbbell/) | [`simple_dumbbell`](https://invia-flights.github.io/blitzly/plots/dumbbell/#blitzly.plots.dumbbell.simple_dumbbell) | Plots a dumbbell plot. This can be used to compare two columns of data to visualize changes. |
| [`histogram`](https://invia-flights.github.io/blitzly/plots/histogram/) | [`simple_histogram`](https://invia-flights.github.io/blitzly/plots/histogram/#blitzly.plots.histogram.simple_histogram) | Plots a histogram with one ore more distributions. |
| [`matrix`](https://invia-flights.github.io/blitzly/plots/matrix/) | [`binary_confusion_matrix`](https://invia-flights.github.io/blitzly/plots/matrix/#blitzly.plots.matrix.binary_confusion_matrix) | Plots a confusion matrix for binary classification data. |
| [`matrix`](https://invia-flights.github.io/blitzly/plots/matrix/) | [`pearson_corr_matrix`](https://invia-flights.github.io/blitzly/plots/matrix/#blitzly.plots.matrix.pearson_corr_matrix) | Plots a Pearson product-moment correlation coefficients matrix. |
| [`scatter`](https://invia-flights.github.io/blitzly/plots/scatter/) | [`scatter_matrix`](https://invia-flights.github.io/blitzly/plots/scatter/#blitzly.plots.scatter.scatter_matrix) | Plots a scatter matrix. |
| [`scatter`](https://invia-flights.github.io/blitzly/plots/scatter/) | [`multi_scatter`](https://invia-flights.github.io/blitzly/plots/scatter/#blitzly.plots.scatter.multi_scatter) | Create a multi scatter plot. It can be used to visualize the relationship between multiple variables from the same Pandas DataFrame. |

## Usage 🤌
Here are some examples. You can also check out the [playground notebook](https://github.com/invia-flights/blitzly/blob/main/examples/playground.ipynb) 📒.

**[`multi_bar`](https://invia-flights.github.io/blitzly/plots/bar/#blitzly.plots.bar.multi_bar):**
```python
from blitzly.plots.bar import multi_bar
import numpy as np

data = np.array([[8, 3, 6], [9, 7, 5]])
error_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

multi_bar(
    data,
    x_labels=["Vienna", "Berlin", "Lisbon"],
    group_labels=["Personal rating", "Global rating"],
    errors=error_array,
    title="City ratings 🏙",
    mark_x_labels=["Lisbon"],
    write_html_path="see_the_blitz.html",
)
```
Gives you this:

<img src="https://github.com/invia-flights/blitzly/raw/main/docs/assets/images/example_plots/multi_bars.png" alt="multi bars plot" width="1000" height="555"/>

**[`scatter matrix`](https://invia-flights.github.io/blitzly/plots/scatter/#blitzly.plots.scatter.scatter_matrix):**
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
Gives you:

<img src="https://github.com/invia-flights/blitzly/raw/main/docs/assets/images/example_plots/scatter_matrix.png" alt="scatter-matrix plot" width="500" height="500"/>

## Contributing 👩‍💻

Please check out the [guide](https://invia-flights.github.io/blitzly/CONTRIBUTING/) on how to contribute to this project.
