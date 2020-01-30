import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from ..plots import plot_binned_stat
from ..ana import bias

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


def plot_pixels_pe_spectrum(true_pe, reco_pe, ax=None, **kwargs):
    """
    Plot the pixels spectrum (reconstructed photo-electrons as a function of true photo-electrons)

    Parameters
    ----------
    true_pe: `numpy.ndarray`
        true photo-electron values
        shape: (n,)
    reco_pe: `numpy.ndarray`
        reconstructed photo-electron values
        shape: (n, )
    ax: `matplotlib.pyplot.axis` or None
    kwargs: args for `matplotlib.pyplot.hist`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    mask = (reco_pe > 0)
    y = np.log10(reco_pe[mask])
    x = true_pe[mask]

    kwargs['cumulative'] = -1
    kwargs['histtype'] = 'step'
    kwargs['density'] = False
    kwargs['log'] = True
    if 'bins' not in kwargs:
        kwargs['bins'] = 300
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 3
    if 'label' in kwargs:
        kwargs.pop('label')

    ax.hist(y, **kwargs, label='all pixels')
    ax.hist(y[x > 0], **kwargs, label='pixels with signal')
    ax.hist(y[x == 0], **kwargs, label='pixels with no signal')
    ax.hist(np.log10(true_pe[true_pe > 0]), **kwargs, label='true signal pixels', alpha=0.4)
    ax.set_xlim(-1, 5)
    ax.set_xlabel('log10(# true pe)')
    ax.set_ylabel('# reco pe')
    ax.legend(fontsize=16)
    ax.grid()
    return ax


def plot_charge_resolution(true_pe, reco_pe, xlim_bias=(50, 500), bias_correction=True, bins=400, ax=None, hist_args={},
                           bin_stat_args={}):
    """
    Plot the charge resolution

    Parameters
    ----------
    true_pe: `numpy.ndarray`
        shape: (n,)
    reco_pe: `numpy.ndarray`
        shape: (n,)
    xlim_bias: tuple (xmin, xmax)
        unused if `bias_correction=False`
    bias_correction: bool
    bins: int
    ax: `matplotlib.pyplot.axis` or None
    hist_args: dict
        args for `matplotlib.pyplot.hist2d`
    bin_stat_args: dict
        args for `ctaplot.plots.plot_binned_stat`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    mask = (true_pe > 0)
    x = np.log10(true_pe[mask])
    y = reco_pe[mask] / true_pe[mask]

    if bias_correction:
        mask_bias = (x > np.log10(xlim_bias[0])) & (x < np.log10(xlim_bias[1]))
        b = bias(np.ones_like(y[mask_bias]), y[mask_bias])
        ylabel = "(reco # pe / true # pe) - ({:.3f})".format(b)
    else:
        b = 0
        ylabel = "(reco # pe / true # pe)"

    hist_args['bins'] = bins
    hist_args['norm'] = LogNorm()

    h, xedges, yedges, im = ax.hist2d(x, y - b, label='reco pe', **hist_args)
    plt.colorbar(im, ax=ax)

    ax.hlines(1, x.min(), x.max(), color='black')

    if bias_correction:
        ax.axvspan(np.log10(xlim_bias[0]), np.log10(xlim_bias[1]), alpha=0.05, color='black',
                   label='bias computed there')

    if 'errorbar' not in bin_stat_args:
        bin_stat_args['errorbar'] = True
    if 'color' not in bin_stat_args:
        bin_stat_args['color'] = 'red'
    bin_stat_args['label'] = 'median'
    bin_stat_args['statistic'] = 'median'

    plot_binned_stat(x, y - b, **bin_stat_args)

    ax.set_xlabel('log10(# true pe)', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid()
    ax.legend(fontsize=18)
    return ax
