<img src="https://github.com/invia-flights/blitzly/raw/main/docs/assets/images/icon.png" alt="blitzly logo" width="200" height="200"/><br>
# blitzly ⚡️
***Lightning-fast way to get plots with Plotly***

[![Testing](https://github.com/invia-flights/blitzly/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/invia-flights/blitzly/actions/workflows/testing.yml)
[![docs](https://img.shields.io/badge/docs-mkdoks%20material-blue)](https://invia-flights.github.io/blitzly/)
[![license](https://img.shields.io/github/license/invia-flights/blitzly)](https://github.com/invia-flights/blitzly/blob/main/LICENSE)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://github.com/PyCQA/isort)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
## Introduction 🎉

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
| [`histogram`](https://invia-flights.github.io/blitzly/plots/histogram/) | [`simple_histogram`](https://invia-flights.github.io/blitzly/plots/histogram/#blitzly.plots.histogram.simple_histogram) | Plots a histogram with one ore more distributions. |
| [`matrix`](https://invia-flights.github.io/blitzly/plots/matrix/) | [`binary_confusion_matrix`](https://invia-flights.github.io/blitzly/plots/histogram/#blitzly.plots.histogram.simple_histogram) | Plots a confusion matrix for binary classification data. |

## Usage 🤌
Here are some examples:
[`multi_chart`](https://invia-flights.github.io/blitzly/plots/bar/#blitzly.plots.bar.multi_chart):
```python
    from blitzly.bar import multi_chart
    import numpy as np

    data = np.array([[8, 3, 6], [9, 7, 5]])
    error_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    multi_chart(
        data,
        x_labels=["Vienna", "Berlin", "Lisbon"],
        group_labels=["Personal rating", "Global rating"],
        hover_texts=["foo", "bar", "blitzly"],
        errors=error_array,
        title="City ratings 🏙",
        mark_x_labels=["Lisbon"],
        write_html_path="see_the_blitz.html",
    )
```
Gives you this:

<img src="https://github.com/invia-flights/blitzly/raw/main/docs/assets/images/example_plots/city_rating.png" alt="city rating plot" width="1000" height="555"/>
