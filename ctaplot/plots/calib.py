import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from ..plots import plot_binned_stat

__all__ = ['plot_photoelectron_true_reco']

def plot_photoelectron_true_reco(true_pe, reco_pe, bins=200, stat='median', errorbar=True, percentile=68.27,
                                 ax=None, hist_args={}, stat_args={}, xy_args={}):
    """
    Plot the number of reconstructed photo-electrons as a function of the number of true photo-electron

    Parameters
    ----------
    true_pe: `numpy.ndarray`
        shape: (n_pixels, )
    reco_pe: `numpy.ndarray`
        shape: (n_pixels, )
    bins: int or `numpy.ndarray`
    stat: str or None
        'mean', 'median', 'min', 'max'. if None, not plotted.
    errorbar: bool
        plot the errorbar corresponding to the percentile as colored area around the stat line
    percentile: float
        between 0 and 100
        percentile for the errorbars
    ax: `matplotlib.pyplot.axis` or None
    hist_args: args for `pyplot.hist2d`
    stat_args: args for `ctaplot.plots.plot_binned_stat`
    xy_args: args for `pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    mask = (true_pe > 0) & (reco_pe > 0)
    x = np.log10(true_pe[mask])
    y = np.log10(reco_pe[mask])

    if 'bins' in hist_args:
        hist_args.pop('bins')
    h, xedges, yedges, im = ax.hist2d(x, y, bins=bins, norm=LogNorm())

    if stat is not None:
        if 'color' not in stat_args:
            stat_args['color'] = 'red'
        if 'linewidth' not in stat_args:
            stat_args['linewidth'] = 2
        plot_binned_stat(x, y, errorbar=errorbar, bins=bins, ax=ax, statistic=stat, percentile=percentile, label=stat,
                         line=True, **stat_args)

    if 'color' not in xy_args:
        xy_args['color'] = 'black'
    if 'label' not in xy_args:
        xy_args['label'] = 'y=x'
    ax.plot([x.min(), x.max()], [x.min(), x.max()], **xy_args)

    ylim = list(ax.get_ylim())
    ylim[1] *= 1.2
    ax.set_ylim(ylim)

    plt.colorbar(im, ax=ax)
    ax.set_xlabel('log10(# true p.e)', fontsize=18)
    ax.set_ylabel('log10(# reconstructed p.e)', fontsize=18)
    ax.grid()
    ax.legend(fontsize=16)
    return ax

