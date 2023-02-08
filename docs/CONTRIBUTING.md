# How to contribute
Thank 🙏 you for your interest in contributing to this project! Please read this document to get started.

## Poetry
We are using [Poetry](https://python-poetry.org/) to manage the dependencies, for deployment, and the virtual environment. If you have not used it before please check out the [documentation](https://python-poetry.org/docs/) to get started.

If you want to start working on the project. The first thing you have to do is:
```bash
poetry install --with dev --with test
```
This installs all needed dependencies for development and testing.

## Pre-commit hooks
We are using [pre-commit](https://pre-commit.com/) to ensure a consistent code style and to avoid common mistakes. Please install the [pre-commit](https://pre-commit.com/#installation) and install the hook with:
```bash
pre-commit install --hook-type commit-msg
```

## Homebrew
We are using [Homebrew](https://brew.sh/) to manage the dependencies for the development environment. Please install Homebrew and run:
```bash
 brew bundle
```
to install the dependencies. If you don't want/can't use Homebrew, you can also install the dependencies manually.

## Conventional Commits
We are using [Conventional Commits](https://www.conventionalcommits.org) to ensure a consistent commit message style. Please use the following commit message format:
```bash
<type>[optional scope]: <description>
```
E.g.:
```bash
feat: new fantastic plot 📈
```

## How to contribute
The following steps will give a short guide on how to contribute to this project:

- Create a personal [fork](https://github.com/invia-flights/blitzly/fork) of the project on [GitHub](https://github.com/).
- Clone the fork on your local machine. Your remote repo on [GitHub](https://github.com/) is called `origin`.
- Add the original repository as a remote called `upstream`.
- If you created your fork a while ago be sure to pull upstream changes into your local repository.
- Create a new branch to work on! Start from `develop` if it exists, else from `main`.
- Implement/fix your feature, comment your code, and add some examples.
- Follow the code style of the project, including indentation. [Black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort), [Pylint](https://github.com/PyCQA/pylint), [mypy](https://github.com/python/mypy), and [ssort](https://github.com/bwhmather/ssort) can help you with it.
- Run all tests.
- Write or adapt tests as needed.
- Add or change the documentation as needed. Please follow the "[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)".
- Squash your commits into a single commit with git's [interactive rebase](https://help.github.com/articles/interactive-rebase). Create a new branch if necessary.
- Push your branch to your fork on [GitHub](https://github.com/), the remote `origin`.
- From your fork open a pull request in the correct branch. Target the project's `develop` branch!
- Once the pull request is approved and merged you can pull the changes from `upstream` to your local repo and delete
your extra branch(es).

## Example
```python title="Example of a new plot function"
def simple_dumbbell(
    data: Union[pd.DataFrame, NDArray],
    title: str = "Dumbbell plot",
    marker_size: int = 16,
    marker_line_width: int = 8,
    plotly_kwargs: Optional[dict] = None,
    size: Optional[Tuple[int, int]] = None,
    show_legend: Optional[bool] = None,
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
        show_legend (Optional[bool]): Whether to show the legend.
        show (bool): Whether to show the figure.
        write_html_path (Optional[str]): The path to which the histogram should be written as an HTML file.
            If None, the histogram will not be saved.

    Returns:
        BaseFigure: The dumbbell plot.
    """

    data = check_data(data, min_rows=1, min_columns=2, max_columns=2, as_pandas=True)

    fig = go.Figure()

    for index, row in data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row.iloc[0], row.iloc[1]],
                y=[index, index],
                mode="lines",
                showlegend=False,
                line={
                    "color": "black",
                    "width": marker_line_width,
                },
            )
        )

    for column_idx, column_name in enumerate(data.columns):
        fig.add_trace(
            go.Scatter(
                x=data.iloc[:, column_idx],
                y=data.index,
                mode="markers",
                name=column_name,
                **plotly_kwargs if plotly_kwargs else {},
            )
        )

    fig.update_traces(
        marker=dict(size=marker_size),
    )

    fig = update_figure_layout(fig, title, size, show_legend)
    return save_show_return(fig, write_html_path, show)
```

Let's go through the code step by step:
1. To make blitzly ⚡️ simple to use try to provide as many default arguments as possible. Also don't forget to add type hints:

```python title="Yea! 🥳"
title: str = "Dumbbell plot"
```

```python title="Nay! 😢"
title
```


2. Every function in blitzly ⚡️ requires the following arguments:
```python
plotly_kwargs: Optional[dict] = None,
show: bool = True,
write_html_path: Optional[str] = None
```
More about this in the next steps.

3. Each function should return `BaseFigure`. This gives the user the possibility to use the returned Plotly figure in every possible way.

4. The function should have a docstring. The docstring should follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). The docstring should contain a short description of the function, a short example, a list of all arguments, and a description of the return value.

5. [`check_data`](https://invia-flights.github.io/blitzly/utils/#blitzly.etc.utils.check_data) is used for validating and preparing the data for the plot. Please don't forget to add it to your implementation.

6. So no code is repeated we use [`update_figure_layout`](https://invia-flights.github.io/blitzly/utils/#blitzly.etc.utils.update_figure_layout) to update the layout of the figure.

7. [`save_show_return`](https://invia-flights.github.io/blitzly/utils/#blitzly.etc.utils.save_show_return) is used to save the figure as an HTML file - if needed, shows the figure, and returns the figure. The use of this function is also mandatory.

8. When you work on the implementation try to make the implementation as generic as possible. In other words, try to wrap a powerful Plotly implementation in your blitzly function and pass as many arguments as possible to the user while giving default arguments. In addition to this please also provide a `plotly_kwargs` argument which allows the user to pass additional keyword arguments to the Plotly function. This way the user can customize the plot as much as possible.

9. Don't forget to write some meaningful unit tests. The tests should cover all possible use cases of the function and should be located in the `tests/test_cases` folder and should be named `test_<function_name_with_use_case>.py`.

10. Documentation: Please add your new blitly ⚡️ plot to the list in the [`README.md`](https://github.com/invia-flights/blitzly#available-plots-so-far-) file. If you think your plot implementation is worth to be documented even more, feel free to add it together with a short description to the end of the [`playground.ipynb`](https://github.com/invia-flights/blitzly/blob/main/examples/playground.ipynb) notebook.

!!! tip "Get in touch"
    That's it! 🎉 Happy ploding (plotting + coding)! 📊👩‍💻 If you have any questions feel free to contact us!
