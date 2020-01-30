"""
Gridded plots
"""

import matplotlib.pyplot as plt
from .plots import plot_binned_stat

__all__ = ['plot_binned_stat_grid']

def plot_binned_stat_grid(data, x_col, n_cols=4, **binned_stat_args):
    """
    Make a figure with a grid of binned stat plots. All variable in `data` are
    plotted versus the `x_col` variable.

    Parameters
    ----------
    data: `pandas.dataframe`
    x_col: str
        name of the column in the data to consider as X variable.
    n_col: int
        number of columns in the plot grid. The number of rows in determined automatically.
    binned_stat_args: args for `ctaplot.plot.plot_binned_stat`

    Returns
    -------
    `matplotlib.figure.Figure`
    """

    n = len(data.columns)
    n_rows = n // n_cols + 1 * (n % n_cols > 0)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 20 * 0.66 * (n_rows / n_cols)), sharex=False)

    raxes = axes.ravel()
    cols = list(data.columns)
    cols.remove(x_col)  # no need to plot x_col versus itself

    if 'statistic' not in binned_stat_args:
        binned_stat_args['statistic'] = 'mean'

    for ii, k in enumerate(cols):
        ax = raxes[ii]

        plot_binned_stat(data[x_col], data[k],
                         ax=ax,
                         **binned_stat_args,
                         )

        ax.set_title(f'{k}', fontsize=15)
        ax.grid('on')

    for ii in range(len(cols), len(axes.ravel())):
        raxes[ii].remove()

    fig.suptitle(rf"{binned_stat_args['statistic']} as a function of {x_col}", fontsize=20, y=1.02)
    fig.tight_layout()

    return fig
