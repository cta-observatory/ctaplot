"""
plots.py
========
Functions to make IRF and other reconstruction quality-check plots
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde, binned_statistic
import ctaplot.ana as ana
from astropy.utils import deprecated

# plt.style.use('seaborn-colorblind')
plt.style.use('seaborn-paper')

# "#3F5D7D" is the nice dark blue color.


SizeTitleArticle = 20
SizeLabelArticle = 18
SizeTickArticle = 16
SizeTitleSlides = 28
SizeLabelSlides = 24
SizeTickSlides = 20

SizeLabel = SizeLabelArticle
SizeTick = SizeTickArticle

mpl.rc('xtick', labelsize=SizeTick)
mpl.rc('ytick', labelsize=SizeTick)
mpl.rc('axes', labelsize=SizeLabel)

# sets of colors from color-brewer
BrewReds = ['#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
BrewBlues = ['#eff3ff', '#bdd7e7', '#6baed6', '#2171b5']
BrewGreens = ['#edf8e9', '#bae4b3', '#74c476', '#238b45']
BrewGreys = ['#f7f7f7', '#cccccc', '#969696', '#525252']
BrewViolets = ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#6a51a3']
BrewOranges = ['#feedde', '#fdbe85', '#fd8d3c', '#d94701']




def plot_energy_distribution(mc_energy, reco_energy, ax=None, outfile=None, mask_mc_detected=True):
    """
    Plot the energy distribution of the simulated particles, detected particles and reconstructed particles
    The plot might be saved automatically if `outfile` is provided.

    Parameters
    ----------
    mc_energy: Numpy 1d array of simulated energies
    reco_energy: Numpy 1d array of reconstructed energies
    ax: `matplotlib.pyplot.axes`
    outfile: string - output file path
    mask_mc_detected: Numpy 1d array - mask of detected particles for the SimuE array
    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel('Count')

    ax.set_xscale('log')
    count_S, bin_S, o = ax.hist(mc_energy, log=True, bins=np.logspace(-3, 3, 30), label="Simulated")
    count_D, bin_D, o = ax.hist(mc_energy[mask_mc_detected], log=True, bins=np.logspace(-3, 3, 30), label="Detected")
    count_R, bin_R, o = ax.hist(reco_energy, log=True, bins=np.logspace(-3, 3, 30), label="Reconstructed")
    plt.legend(fontsize=SizeLabel)
    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)
        plt.close()

    return ax


def plot_multiplicity_per_energy(multiplicity, energies, ax=None, outfile=None):
    """
    Plot the telescope multiplicity as a function of the energy
    The plot might be saved automatically if `outfile` is provided.

    Parameters
    ----------
    multiplicity: `numpy.ndarray`
    energies: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    outfile: string
    """

    assert len(multiplicity) == len(energies), "arrays should have same length"
    assert len(multiplicity) > 0, "arrays are empty"

    E, m_mean, m_min, m_max, m_per = ana.multiplicity_stat_per_energy(multiplicity, energies)

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel('Multiplicity')

    ax.fill_between(E, m_min, m_max, alpha=0.5)
    ax.plot(E, m_min, '--')
    ax.plot(E, m_max, '--')
    ax.plot(E, m_mean, color='red')

    ax.set_xscale('log')
    ax.set_title("Multiplicity")

    if type(outfile) is str:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


def plot_field_of_view_map(reco_alt, reco_az, source_alt, source_az, energies=None, ax=None, outfile=None):
    """
    Plot a map in angles [in degrees] of the photons seen by the telescope (after reconstruction)

    Parameters
    ----------
    reco_alt: `numpy.ndarray`
    reco_az: `numpy.ndarray`
    source_alt: float, source Altitude
    source_az: float, source Azimuth
    energies: `numpy.ndarray` - if given, set the colorbar
    ax: `matplotlib.pyplot.axes`
    outfile: string - if None, the plot is not saved

    Returns
    -------
    ax: `matplitlib.pyplot.axes`
    """
    dx = 0.05

    ax = plt.gca() if ax is None else ax
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlim(source_az - dx, source_az + dx)
    ax.set_ylim(source_alt - dx, source_alt + dx)

    ax.set_xlabel("Az [deg]")
    ax.set_ylabel("Alt [deg]")

    ax.set_axis('equal')

    if energies is not None:
        c = np.log10(energies)
        plt.colorbar()
    else:
        c = BrewBlues[-1]

    ax.scatter(reco_az, reco_alt, c=c)
    ax.scatter(source_az, source_alt, marker='+', linewidths=3, s=200, c='orange', label="Source position")

    plt.legend()
    if type(outfile) is str:
        plt.savefig(outfile + ".png", bbox_inches="tight", format='png', dpi=200)

    return ax


def plot_angles_distribution(reco_alt, reco_az, source_alt, source_az, outfile=None):
    """
    Plot the distribution of reconstructed angles in two axes.
    Save figure to `outfile` in png format.

    Parameters
    ----------
    reco_alt: `numpy.ndarray`
    reco_az: `numpy.ndarray`
    source_alt: `float`
    source_az: `float`
    outfile: `string`

    Returns
    -------
    `matplotlib.pyplot.figure`
    """

    dx = 1

    fig = plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(211)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    ax1.set_xlabel('Az [deg]')
    ax1.set_ylabel('Count')
    ax1.set_xlim(source_az - dx, source_az + dx)

    ax1.hist(reco_az, bins=60, range=(source_az - dx, source_az + dx))

    ax2 = plt.subplot(212)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()

    ax2.set_xlabel('Alt [deg]', fontsize=SizeLabel)
    ax2.set_ylabel('Count', fontsize=SizeLabel)
    ax2.set_xlim(source_alt - dx, source_alt + dx)

    ax2.hist(reco_alt, bins=60, range=(source_alt - dx, source_alt + dx))

    if type(outfile) is str:
        fig.savefig(outfile + ".png", bbox_inches="tight", format='png', dpi=200)

    return fig


def plot_theta2(reco_alt, reco_az, mc_alt, mc_az, ax=None, **kwargs):
    """
    Plot the theta2 distribution and display the corresponding angular resolution in degrees.
    The input must be given in radians.

    Parameters
    ----------
    reco_alt: `numpy.ndarray` - reconstructed altitude angle in radians
    reco_az: `numpy.ndarray` - reconstructed azimuth angle in radians
    mc_alt: `numpy.ndarray` - true altitude angle in radians
    mc_az: `numpy.ndarray` - true azimuth angle in radians
    ax: `matplotlib.pyplot.axes`
    **kwargs: options for `matplotlib.pyplot.hist`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    theta2 = np.rad2deg(np.sqrt(ana.theta2(reco_alt, reco_az, mc_alt, mc_az))) ** 2
    ang_res = np.rad2deg(ana.angular_resolution(reco_alt, reco_az, mc_alt, mc_az))

    ax.set_xlabel(r'$\theta^2 [deg^2]$')
    ax.set_ylabel('Count')

    ax.hist(theta2, **kwargs)
    ax.set_title(r'angular resolution: {:.3f}(+{:.2f}/-{:.2f}) deg'.format(ang_res[0], ang_res[2], ang_res[1]))

    return ax


def plot_angles_map_distri(reco_alt, reco_az, source_alt, source_az, energies, outfile=None):
    """
    Plot the angles map distribution

    Parameters
    ----------
    reco_alt: `numpy.ndarray`
    reco_az: `numpy.ndarray`
    source_alt: float
    source_az: float
    energies: `numpy.ndarray`
    outfile: str

    Returns
    -------
    fig: `matplotlib.pyplot.figure`
    """

    dx = 0.5

    fig = plt.figure(figsize=(12, 12))

    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)

    plt.xticks(fontsize=SizeTick)
    plt.yticks(fontsize=SizeTick)

    ax1.set_xlim(source_az - dx, source_az + dx)
    ax1.set_ylim(source_alt - dx, source_alt + dx)

    plt.xlabel("Az [deg]")
    plt.ylabel("Alt [deg]")

    if len(reco_alt) > 1000:
        ax1.hist2d(a.RecoAlt, a.RecoAz, bins=60,
                   range=([source_alt - dx, source_alt + dx], [source_az - dx, source_az + dx]))
    else:
        ax1.scatter(reco_az, reco_alt, c=np.log10(energies))
    ax1.scatter(source_az, source_alt, marker='+', linewidths=3, s=200, c='black')

    ax1.xaxis.set_label_position('top')
    ax1.yaxis.set_tick_params(labelsize=SizeTick)
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_tick_params(labelsize=SizeTick)
    plt.legend('', 'Source position')

    ax2 = plt.subplot2grid((4, 4), (3, 0), colspan=3)
    ax2.set_xlim(source_az - dx, source_az + dx)
    ax2.yaxis.tick_right()
    ax2.xaxis.set_tick_params(labelsize=SizeTick)
    ax2.xaxis.set_ticklabels([])
    ax2.xaxis.tick_bottom()
    ax2.hist(reco_az, bins=60, range=(source_az - dx, source_az + dx))
    plt.locator_params(nbins=4)

    ax3 = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
    ax3.set_ylim(source_alt - dx, source_alt + dx)
    ax3.yaxis.set_ticklabels([])
    ax3.yaxis.set_tick_params(labelsize=SizeTick)
    plt.locator_params(nbins=4)

    ax3.spines["left"].set_visible(False)
    ax3.hist(reco_alt, bins=60, range=(source_alt - dx, source_alt + dx), orientation=u'horizontal')

    if type(outfile) is str:
        fig.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return fig


def plot_impact_point_map_distri(reco_x, reco_y, tel_x, tel_y, fit=False, outfile=None, **kwargs):
    """
    Map and distributions of the reconstructed impact points.

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    tel_x: `numpy.ndarray`
        X positions of the telescopes
    tel_y: `numpy.ndarray`
        Y positions of the telescopes
    kde: bool - if True, makes a gaussian fit of the point density
    outfile: 'str' - save a png image of the plot under 'string.png'

    Returns
    -------
    fig: `matplotlib.pyplot.figure`
    """
    fig = plt.figure(figsize=(12, 12))

    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)

    plt.xticks(fontsize=SizeTick)
    plt.yticks(fontsize=SizeTick)

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    if fit:
        kde = gaussian_kde([reco_x, reco_y])
        density = kde([reco_x, reco_y])
        ax1.scatter(reco_x, reco_y, c=density, s=2)
    else:
        ax1.hist2d(reco_x, reco_y, bins=80, norm=LogNorm())

    ax1.xaxis.set_label_position('top')
    ax1.yaxis.set_tick_params(labelsize=SizeTick)
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_tick_params(labelsize=SizeTick)
    plt.legend('', 'Source position')

    ax1.scatter(tel_x, tel_y, c='tomato', marker='+', s=90, linewidths=10)

    ax2 = plt.subplot2grid((4, 4), (3, 0), colspan=3)
    ax2.yaxis.tick_right()
    ax2.xaxis.set_tick_params(labelsize=SizeTick)
    ax2.xaxis.set_ticklabels([])
    ax2.xaxis.tick_bottom()
    ax2.hist(reco_x, bins=60)
    plt.locator_params(nbins=4)

    ax3 = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
    ax3.yaxis.set_ticklabels([])
    ax3.yaxis.set_tick_params(labelsize=SizeTick)
    plt.locator_params(nbins=4)

    ax3.spines["left"].set_visible(False)
    ax3.hist(reco_y, bins=60, orientation=u'horizontal')

    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return fig


def plot_impact_point_heatmap(reco_x, reco_y, ax=None, outfile=None):
    """
    Plot the heatmap of the impact points on the site ground and save it under Outfile

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    outfile: string
    """

    ax = plt.gca() if ax is None else ax

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=SizeTick)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=SizeTick)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis('equal')

    cm = plt.cm.get_cmap('OrRd')
    h = ax.hist2d(reco_x, reco_y, bins=75, norm=LogNorm(), cmap=cm)
    cb = plt.colorbar(h[3], ax=ax)
    cb.set_label('Event count')

    if type(outfile) is str:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


def plot_multiplicity_hist(multiplicity, ax=None, outfile=None, quartils=False, **kwargs):
    """
    Histogram of the telescopes multiplicity

    Parameters
    ----------
    multiplicity: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    outfile: string
    **kwargs: args for `matplotlib.pyplot.bar`
    """
    from matplotlib.ticker import MaxNLocator

    ax = plt.gca() if ax is None else ax
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    m = np.sort(multiplicity)
    xmin = multiplicity.min()
    xmax = multiplicity.max()


    if 'label' not in kwargs:
        kwargs['label'] = 'Telescope multiplicity'

    n, bins, patches = ax.hist(multiplicity, bins=(xmax-xmin), range=(xmin, xmax), rwidth=0.7, align='left', **kwargs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    x50 = m[int(np.floor(0.5 * len(m)))] + 0.5
    x90 = m[int(np.floor(0.9 * len(m)))] + 0.5
    if quartils and (xmin < x50 < xmax):
        ax.vlines(x50, 0, n[int(m[int(np.floor(0.5 * len(m)))])], color=BrewOranges[-2], label='50%')
    if quartils and (xmin < x90 < xmax):
        ax.vlines(x90, 0, n[int(m[int(np.floor(0.9 * len(m)))])], color=BrewOranges[-1], label='90%')

    ax.legend(fontsize=SizeLabel)
    ax.set_title("Telescope multiplicity")
    if type(outfile) is str:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


def plot_multiplicity_per_telescope_type(multiplicity, telescope_type, ax=None, outfile=None, quartils=False, **kwargs):
    """
    Plot the multiplicity for each telescope type

    Parameters
    ----------
    multiplicity: `numpy.ndarray`
    telescope_type: `numpy.ndarray`
        same shape as `multiplicity`
    ax: `matplotlib.pyplot.axes`
    outfile: path
    quartils: bool - True to plot 50% and 90% quartil mark
    kwargs: args for `matplotlib.pyplot.hist`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    from matplotlib.ticker import MaxNLocator

    ax = plt.gca() if ax is None else ax
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    m = np.sort(multiplicity)
    xmin = multiplicity.min()
    xmax = multiplicity.max()


    if 'label' not in kwargs:
        kwargs['label'] = [str(type) for type in set(telescope_type)]
    if 'stacked' not in kwargs:
        kwargs['stacked'] = True
    kwargs['rwidth'] = 0.7 if 'rwidth' not in kwargs else kwargs['rwidth']
    kwargs['align'] = 'left'

    mult_by_type = np.array([multiplicity[telescope_type==type] for type in set(telescope_type)])
    ax.hist(mult_by_type, bins=(xmax-xmin), range=(xmin, xmax), **kwargs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    x50 = m[int(np.floor(0.5 * len(m)))]
    x90 = m[int(np.floor(0.9 * len(m)))]
    if quartils and (xmin < x50 < xmax):
        ax.vlines(x50+0.5, 0, len(multiplicity[multiplicity==x50]), label='50%')
    if quartils and (xmin < x90 < xmax):
        ax.vlines(x90+0.5, 0, len(multiplicity[multiplicity==x90]), color=BrewOranges[-1], label='90%')

    ax.legend(fontsize=SizeLabel)
    ax.set_title("Telescope multiplicity")
    if type(outfile) is str:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


def plot_resolution(bins, res, log=False, ax=None, **kwargs):
    """
    Plot the passed resolution.

    Parameters
    ----------
    bins: 1D `numpy.ndarray`
    res: 2D `numpy.ndarray` - output from `ctpalot.ana.resolution`
        res[:,0]: resolution
        res[:,1]: lower confidence limit
        res[:,2]: upper confidence limit
    log: bool
        if true, x is logscaled
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylabel(r'res')

    if not log:
        x = (bins[:-1] + bins[1:]) / 2.
    else:
        x = ana.logbin_mean(bins)
        ax.set_xscale('log')

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(x, res[:, 0], xerr=(bins[1:] - bins[:-1]) / 2.,
                yerr=(res[:, 0] - res[:, 1], res[:, 2] - res[:, 0]), **kwargs)

    ax.set_title('Resolution')
    return ax


def plot_effective_area_per_energy(simu_energy, reco_energy, simulated_area, ax=None, **kwargs):
    """
    Plot the effective area as a function of the energy

    Parameters
    ----------
    simu_energy: `numpy.ndarray` - all simulated event energies
    reco_energy: `numpy.ndarray` - all reconstructed event energies
    simulated_area: float
    ax: `matplotlib.pyplot.axes`
    kwargs: options for `maplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`

    Example
    -------
    >>> import numpy as np
    >>> import ctaplot
    >>> irf = ctaplot.ana.irf_cta()
    >>> simu_e = 10**(-2 + 4*np.random.rand(1000))
    >>> reco_e = 10**(-2 + 4*np.random.rand(100))
    >>> ax = ctaplot.plots.plot_effective_area_per_energy(simu_e, reco_e, irf.LaPalmaArea_prod3)
    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    E_bin, Seff = ana.effective_area_per_energy(simu_energy, reco_energy, simulated_area)
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(E, Seff, xerr=(E_bin[1:] - E_bin[:-1]) / 2., **kwargs)

    return ax


def plot_effective_area_cta_requirement(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the effective area

    Parameters
    ----------
    cta_site: string
        see `ctaplot.ana.cta_requirement`
    ax: `matplotlib.pyplot.axes`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_requirement(cta_site)
    e_cta, ef_cta = cta_req.get_effective_area()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirement {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)

    return ax


def plot_effective_area_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the CTA performances for the effective area

    Parameters
    ----------
    cta_site: string
        see `ctaplot.ana.cta_requirement`
    ax: `matplotlib.pyplot.axes`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_performance(cta_site)
    e_cta, ef_cta = cta_req.get_effective_area()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performance {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)

    return ax



def plot_sensitivity_cta_requirement(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the sensitivity
    Parameters
    ----------
    cta_site: string - see `ctaplot.ana.cta_requirement`
    ax: `matplotlib.pyplot.axes`, optional

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_requirement(cta_site)
    e_cta, ef_cta = cta_req.get_sensitivity()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirement {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)

    return ax


def plot_sensitivity_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the CTA performances for the sensitivity

    Parameters
    ----------
    cta_site: string - see `ctaplot.ana.cta_requirement`
    ax: `matplotlib.pyplot.axes`, optional

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_performance(cta_site)
    e_cta, ef_cta = cta_req.get_sensitivity()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performance {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)

    return ax


def plot_layout_map(tel_x, tel_y, tel_type=None, ax=None, **kwargs):
    """
    Plot the layout map of telescopes positions

    Parameters
    ----------
    tel_x: `numpy.ndarray`
    tel_y: `numpy.ndarray`
    TelId: `numpy.ndarray`
    tel_type: `numpy.ndarray`
    LayoutId: `numpy.ndarray`
    Outfile: string

    Returns
    -------

    """

    ax = plt.gca() if ax is None else ax
    ax.axis('equal')

    if tel_type is not None and 'c' not in kwargs and 'color' not in kwargs:
        values = np.arange(len(set(tel_type)))
        kwargs['c'] = [values[list(set(tel_type)).index(type)] for type in tel_type]
    ax.scatter(tel_x, tel_y, **kwargs)

    return ax


def plot_resolution_per_energy(reco, simu, energy, ax=None, **kwargs):
    """
    Plot a variable resolution as a function of the energy

    Parameters
    ----------
    reco: `numpy.ndarray`
    simu: `numpy.ndarray`
    energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylabel(r'res')
    ax.set_xlabel('Energy [TeV]')
    ax.set_xscale('log')

    energy_bin, resolution = ana.resolution_per_energy(simu, reco, energy)

    E = ana.logbin_mean(energy_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(E, resolution[:, 0], xerr=(energy_bin[1:] - energy_bin[:-1]) / 2.,
                yerr=(resolution[:, 0] - resolution[:, 1], resolution[:, 2] - resolution[:, 0]), **kwargs)

    ax.set_title('Resolution')
    return ax


def plot_angular_resolution_per_energy(reco_alt, reco_az, mc_alt, mc_az, energy,
                                       percentile=68.27, confidence_level=0.95, bias_correction=False,
                                       ax=None, **kwargs):
    """
    Plot the angular resolution as a function of the energy

    Parameters
    ----------
    reco_alt: `numpy.ndarray`
    reco_az: `numpy.ndarray`
    mc_alt: `numpy.ndarray`
    mc_az: `numpy.ndarray`
    energy: `numpy.ndarray`
        energies in TeV
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylabel(r'$\theta [deg]$')
    ax.set_xlabel('Energy [TeV]')
    ax.set_xscale('log')

    e_bin, RES = ana.angular_resolution_per_energy(reco_alt, reco_az, mc_alt, mc_az, energy,
                                                   percentile=percentile,
                                                   confidence_level=confidence_level,
                                                   bias_correction=bias_correction
                                                   )

    # Angular resolution is traditionally presented in degrees
    RES = np.degrees(RES)

    E = ana.logbin_mean(e_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(E, RES[:, 0], xerr=(e_bin[1:] - e_bin[:-1]) / 2.,
                yerr=(RES[:, 0] - RES[:, 1], RES[:, 2] - RES[:, 0]), **kwargs)

    ax.set_title('Angular resolution')
    return ax


def plot_angular_resolution_cta_requirement(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the angular resolution
    Parameters
    ----------
    cta_site: string
        see `ctaplot.ana.cta_requirement`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    cta_req = ana.cta_requirement(cta_site)
    e_cta, ar_cta = cta_req.get_angular_resolution()

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirement {}".format(cta_site)

    ax.plot(e_cta, ar_cta, **kwargs)

    ax.set_ylabel(r'$\theta [deg]$')
    ax.set_xlabel('Energy [TeV]')

    ax.set_xscale('log')
    ax.set_title('Angular resolution')
    return ax


def plot_angular_resolution_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the official CTA performances (June 2018) for the angular resolution

    Parameters
    ----------
    cta_site: string, see `ana.cta_performance`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    cta_req = ana.cta_performance(cta_site)
    e_cta, ar_cta = cta_req.get_angular_resolution()

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performance {}".format(cta_site)

    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    ax.set_ylabel(r'$\theta [deg]$')
    ax.set_xlabel('Energy [TeV]')
    ax.set_title('Angular resolution')
    return ax


def hist_impact_parameter_error(reco_x, reco_y, simu_x, simu_y, ax=None, **kwargs):
    """
    plot impact parameter error distribution and save it under Outfile
    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    simu_x: `numpy.ndarray`
    simu_y: `numpy.ndarray`
    Outfile: string
    """
    d = ana.impact_parameter_error(reco_x, reco_y, simu_x, simu_y)

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Error on impact parameter [m]')
    ax.set_ylabel('Count')
    ax.set_title('Impact parameter resolution')

    kwargs['bins'] = 40 if 'bins' not in kwargs else kwargs['bins']

    ax.hist(d, **kwargs)
    return ax

@deprecated('18/08/2019', message='`plot_impact_parameter_error_per_energy` will be removed in a future release.'
                                  'Use `plot_impact_parameter_resolution_per_energy` instead')
def plot_impact_parameter_error_per_energy(reco_x, reco_y, simu_x, simu_y, energy, ax=None, **kwargs):
    """
    plot the impact parameter error distance as a function of energy and save the plot as Outfile
    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    simu_x: `numpy.ndarray`
    simu_y: `numpy.ndarray`
    energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    energy, err_mean : numpy arrays
    """
    irf = ana.irf_cta()
    E_bin = irf.E_bin
    E = []
    err_mean = []
    err_min = []
    err_max = []
    err_std = []
    for i, eb in enumerate(E_bin[:-1]):
        mask = (energy > E_bin[i]) & (energy < E_bin[i + 1])
        E.append(np.mean([E_bin[i], E_bin[i + 1]]))
        if True in mask:
            d = ana.impact_parameter_error(reco_x[mask], reco_y[mask], simu_x[mask], simu_y[mask])
            err_mean.append(d.mean())
            err_min.append(d.min())
            err_max.append(d.max())
            err_std.append(np.std(d))
        else:
            err_mean.append(0)
            err_min.append(0)
            err_max.append(0)
            err_std.append(0)

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylabel('Error on Impact Parameter [m]')
    ax.set_xlabel('Energy [TeV]')

    ax.set_xscale('log')

    ax.fill_between(E, err_min, err_max)
    ax.errorbar(E, err_mean, err_std, color="red", label="mean+std")

    plt.legend(fontsize=SizeLabel)
    ax.set_title('Impact parameter resolution')

    return E, np.array(err_mean)


def plot_impact_parameter_resolution_per_energy(reco_x, reco_y, simu_x, simu_y, energy, ax=None, **kwargs):
    """

    Parameters
    ----------
    reco_x
    reco_y
    simu_x
    simu_y
    energy
    ax
    kwargs

    Returns
    -------

    """
    bin, res = ana.impact_resolution_per_energy(reco_x, reco_y, simu_x, simu_y, energy)
    ax = plot_resolution(bin, res, log=True, ax=ax, **kwargs)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Impact parameter resolution")
    ax.set_title("Impact parameter resolution as a function of the energy")

    return ax


def plot_impact_parameter_error_per_multiplicity(reco_x, reco_y, simu_x, simu_y, multiplicity,
                                                 max_mult=None, ax=None, **kwargs):
    """
    Plot the impact parameter error as a function of multiplicity
    TODO: refactor and clean code

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    simu_x: `numpy.ndarray`
    simu_y: `numpy.ndarray`
    multiplicity: `numpy.ndarray`
    max_mult: optional, max multiplicity - float
    ax: `matplotlib.pyplot.axes`

    Returns
    -------

    """
    max_mult = multiplicity.max() + 1 if max_mult is None else max_mult

    M = np.arange(multiplicity.min(), max_mult)
    e_mean = []
    e_min = []
    e_max = []
    e_std = []
    for m in M:
        mask = (multiplicity == m)
        if True in mask:
            d = ana.impact_parameter_error(reco_x[mask], reco_y[mask], simu_x[mask], simu_y[mask])
            e_mean.append(d.mean())
            e_min.append(d.min())
            e_max.append(d.max())
            e_std.append(np.std(d))
        else:
            e_mean.append(0)
            e_min.append(0)
            e_max.append(0)
            e_std.append(0)

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylabel('Error on Impact Parameter [m]')
    ax.set_xlabel('Multiplicity')

    ax.fill_between(M, e_min, e_max)
    ax.errorbar(M, e_mean, e_std, color="red", label="mean+std")

    plt.legend(fontsize=SizeLabel)

    return M, np.array(e_mean)


def plot_impact_map(impact_x, impact_y, tel_x, tel_y, tel_types=None,
                    ax=None,
                    Outfile="ImpactMap.png",
                    hist_kwargs={},
                    scatter_kwargs={},
                    ):
    """
    Map of the site with telescopes positions and impact points heatmap

    Parameters
    ----------
    impact_x: `numpy.ndarray`
    impact_y: `numpy.ndarray`
    tel_x: `numpy.ndarray`
    tel_y: `numpy.ndarray`
    tel_types: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    hist_kwargs: `kwargs` for `matplotlib.pyplot.hist`
    scatter_kwargs: `kwargs` for `matplotlib.pyplot.scatter`
    Outfile: string - name of the output file
    """
    ax = plt.gca() if ax is None else ax

    hist_kwargs['bins'] = 40 if 'bins' not in hist_kwargs else hist_kwargs['bins']
    ax.hist2d(impact_x, impact_y, **hist_kwargs)
    pcm = ax.get_children()[0]
    plt.colorbar(pcm, ax=ax)

    assert (len(tel_x) == len(tel_y)), "tel_x and tel_y should have the same length"

    scatter_kwargs['s'] = 50 if 's' not in scatter_kwargs else scatter_kwargs['s']

    if tel_types and 'color' not in scatter_kwargs and 'c' not in scatter_kwargs:
        scatter_kwargs['color'] = tel_types
        assert (len(tel_types) == len(tel_x)), "tel_types and tel_x should have the same length"
        ax.scatter(tel_x, tel_y, **scatter_kwargs)
    else:
        if 'color' not in scatter_kwargs and 'c' not in scatter_kwargs:
            scatter_kwargs['color'] = 'black'
        scatter_kwargs['marker'] = '+' if 'marker' not in scatter_kwargs else scatter_kwargs['marker']
        ax.scatter(tel_x, tel_y, **scatter_kwargs)

    ax.axis('equal')
    plt.savefig(Outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


def plot_energy_bias(simu_energy, reco_energy, ax=None, **kwargs):
    """
    Plot the energy bias

    Parameters
    ----------
    simu_energy: `numpy.ndarray`
    reco_energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    assert len(simu_energy) == len(reco_energy), "simulated and reconstructured energy arrrays should have the same length"

    ax = plt.gca() if ax is None else ax

    E_bin, biasE = ana.energy_bias(simu_energy, reco_energy)
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel("bias (median($E_{reco}/E_{simu}$ - 1)")
    ax.set_xlabel("log(E/TeV)")
    ax.set_xscale('log')
    ax.set_title('Energy bias')

    ax.errorbar(E, biasE, xerr=(E - E_bin[:-1], E_bin[1:] - E), **kwargs)

    return ax


def plot_energy_resolution(simu_energy, reco_energy,
                           percentile=68.27, confidence_level=0.95, bias_correction=False,
                           ax=None, **kwargs):
    """
    Plot the enregy resolution as a function of the energy

    Parameters
    ----------
    simu_energy: `numpy.ndarray`
    reco_energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    bias_correction: `bool`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    assert len(simu_energy) == len(reco_energy), "simulated and reconstructured energy arrrays should have the same length"

    ax = plt.gca() if ax is None else ax

    E_bin, Eres = ana.energy_resolution_per_energy(simu_energy, reco_energy,
                                                   percentile=percentile,
                                                   confidence_level=confidence_level,
                                                   bias_correction=bias_correction,
                                                   )
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel(r"$(\Delta E/E)_{68}$")
    ax.set_xlabel("Energy [TeV]")
    ax.set_xscale('log')
    ax.set_title('Energy resolution')

    ax.errorbar(E, Eres[:, 0], xerr=(E - E_bin[:-1], E_bin[1:] - E),
                yerr=(Eres[:, 0] - Eres[:, 1], Eres[:, 2] - Eres[:, 0]), **kwargs)

    return ax


def plot_energy_resolution_cta_requirement(cta_site, ax=None, **kwargs):
    """
    Plot the cta requirement for the energy resolution

    Parameters
    ----------
    cta_site: string
        see `ana.cta_requirement`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    cta_req = ana.cta_requirement(cta_site)
    e_cta, ar_cta = cta_req.get_energy_resolution()

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirement {}".format(cta_site)

    ax.set_ylabel(r"$(\Delta E/E)_{68}$")
    ax.set_xlabel("Energy [TeV]")
    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    return ax


def plot_energy_resolution_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the cta performances (June 2018) for the energy resolution

    Parameters
    ----------
    cta_site: string
        see `ctaplot.ana.cta_performance`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    cta_req = ana.cta_performance(cta_site)
    e_cta, ar_cta = cta_req.get_energy_resolution()

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performance {}".format(cta_site)

    ax.set_ylabel(r"$(\Delta E/E)_{68}$")
    ax.set_xlabel("Energy [TeV]")
    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    return ax


def plot_impact_parameter_error_site_center(reco_x, reco_y, simu_x, simu_y, ax=None, **kwargs):
    """
    Plot the impact parameter error as a function of the distance to the site center.

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    simu_x: `numpy.ndarray`
    simu_y: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplotlib.pyplot.hist2d`

    Returns
    -------
    ax
    """

    ax = plt.gca() if ax is None else ax

    imp_err = ana.impact_parameter_error(reco_x, reco_y, simu_x, simu_y)
    distance_center = np.sqrt(simu_x ** 2 + simu_y ** 2)

    ax.hist2d(distance_center, imp_err, **kwargs)
    ax.set_xlabel("Distance to site center")
    ax.set_ylabel("Impact point error")
    return ax


def plot_impact_resolution_per_energy(reco_x, reco_y, simu_x, simu_y, simu_energy,
                                      percentile=68.27, confidence_level=0.95, bias_correction=False,
                                      ax=None, **kwargs):
    """
    Plot the angular resolution as a function of the energy

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    simu_x: float
    simu_y: float
    simu_energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylabel('Impact Resolution [m]')
    ax.set_xlabel('Energy [TeV]')
    ax.set_xscale('log')

    E_bin, RES = ana.impact_resolution_per_energy(reco_x, reco_y, simu_x, simu_y, simu_energy,
                                                  percentile=percentile,
                                                  confidence_level=confidence_level,
                                                  bias_correction=bias_correction,
                                                  )
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(
        E, RES[:, 0],
        xerr=(E - E_bin[:-1], E_bin[1:] - E),
        yerr=(RES[:, 0] - RES[:, 1], RES[:, 2] - RES[:, 0]),
        **kwargs,
    )

    return ax



def plot_migration_matrix(x, y, ax=None, colorbar=False, xy_line=False, hist2d_args={}, line_args={}):
    """
    Make a simple plot of a migration matrix

    Parameters
    ----------
    x: list or `numpy.ndarray`
    y: list or `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    colorbar: `matplotlib.colorbar`
    hist2d_args: dict, args for `matplotlib.pyplot.hist2d`
    line_args: dict, args for `matplotlib.pyplot.plot`

    Returns
    -------
    `matplotlib.pyplot.axes`

    Examples
    --------
    >>> from ctaplot.plots import plot_migration_matrix
    >>> import matplotlib
    >>> x = np.random.rand(10000)
    >>> y = x**2
    >>> plot_migration_matrix(x, y, colorbar=True, hist2d_args=dict(norm=matplotlib.colors.LogNorm()))
    In this example, the colorbar will be log normed
    """

    if 'bins' not in hist2d_args:
        hist2d_args['bins'] = 50
    if 'color' not in line_args:
        line_args['color'] = 'black'
    if 'lw' not in line_args:
        line_args['lw'] = 0.4

    ax = plt.gca() if ax is None else ax
    h = ax.hist2d(x, y, **hist2d_args)
    if colorbar:
        plt.colorbar(h[3], ax=ax)

    if xy_line:
        ax.plot(x, x, **line_args)
    return ax


def plot_dispersion(simu_x, reco_x, x_log=False, ax=None, **kwargs):
    """
    Plot the dispersion around an expected value X_true

    Parameters
    ----------
    simu_x: `numpy.ndarray`
    reco_x: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.hist2d`

    Returns
    -------
    `maptlotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    if not 'bins' in kwargs:
        kwargs['bins'] = 50

    x = np.log10(simu_x) if x_log else simu_x

    ax.hist2d(x, simu_x - reco_x, **kwargs)
    return ax


def plot_feature_importance(feature_keys, feature_importances, ax=None):
    """
    Plot features importance after model training (typically from scikit-learn)

    Parameters
    ----------
    feature_keys: list of string
    feature_importances: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`

    Returns
    -------
    ax
    """
    ax = plt.gca() if ax is None else ax

    ax.bar(feature_keys, feature_importances, ax=ax)
    ax.set_xticks(rotation='vertical')
    ax.title("Feature importances")

    return ax


def plot_binned_stat(x, y, statistic='mean', bins=20, errorbar=False, percentile=68.27, ax=None, **kwargs):
    """
    Plot statistics on the quantity y binned following the quantity x.
    The statistic can be given by a string (￿'mean￿', ￿'sum', ￿'max￿'...) or a function. See `scipy.stats.binned_statistic`.
    Errorbars may be added and represents the dispersion (given by the percentile option) of the y distribution
    around the measured value in a bin. These error bars might not make sense for some statistics,
    it is left to the user to use the function responsibly.

    Parameters
    ----------
    x: `numpy.ndarray`
    y: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    errorbar: bool
    statistic: string or callable - see `scipy.stats.binned_statistic`
    bins: bins for `scipy.stats.binned_statistic`
    kwargs: if errorbar: kwargs for `matplotlib.pyplot.hlines` else: kwargs for `matplotlib.pyplot.plot`

    Returns
    -------
    `matplotlib.pyplot.axes`

    Examples
    --------
    >>> from ctaplot.plots import plot_binned_stat
    >>> import numpy as np
    >>> x = np.random.rand(1000)
    >>> y = x**2
    >>> plot_binned_stat(x, y, statistic='median', bins=40, percentile=95, marker='o', linestyle='')
    """

    ax = plt.gca() if ax is None else ax

    bin_stat, bin_edges, binnumber = binned_statistic(x, y, statistic=statistic, bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    bin_with_data = np.unique(binnumber) - 1
    bin_r68 = np.array([np.percentile(np.abs(y[binnumber == i] - bin_stat[i - 1]), percentile)
                        for i in set(binnumber)])

    if errorbar:
        ax.hlines(bin_stat, bin_edges[:-1], bin_edges[1:], **kwargs)

        # poping label from kwargs so it does not appear twice
        if 'label' in kwargs:
            kwargs.pop('label')

        ax.vlines(bin_centers[bin_with_data],
                  bin_stat[bin_with_data] - bin_r68,
                  bin_stat[bin_with_data] + bin_r68,
                  **kwargs,
                  )
    else:
        ax.plot(bin_centers[bin_with_data],
                   bin_stat[bin_with_data],
                   **kwargs,
                   )
    return ax


def plot_effective_area_per_energy_power_law(emin, emax, total_number_events, spectral_index,
                                             reco_energy, simu_area, ax=None, **kwargs):
    """
    Plot the effective area as a function of the energy.
    The effective area is computed using the `ctaplot.ana.effective_area_per_energy_power_law`.

    Parameters
    ----------
    emin: float
        min simulated energy
    emax: float
        max simulated energy
    total_number_events: int
        total number of simulated events
    spectral_index: float
        spectral index of the simulated power-law
    reco_energy: `numpy.ndarray`
        reconstructed energies
    simu_area: float
        simulated core area
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """


    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ebin, seff = ana.effective_area_per_energy_power_law(emin, emax, total_number_events,
                                                         spectral_index, reco_energy, simu_area)

    energy_nodes = ana.logbin_mean(ebin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'
    ax.errorbar(energy_nodes, seff, xerr=(ebin[1:] - ebin[:-1]) / 2., **kwargs)

    return ax


def plot_angular_resolution_per_off_pointing_angle(simu_alt, simu_az, reco_alt, reco_az,
                                                   alt_pointing, az_pointing, res_degree=False, bins=10, ax=None,
                                                   **kwargs):
    """
    Plot the angular resolution as a function of the angular separation between events true position and the
    pointing direction. Angles must be given in radians.


    Parameters
    ----------
    simu_alt: `numpy.ndarray`
    simu_az: `numpy.ndarray`
    reco_alt: `numpy.ndarray`
    reco_az: `numpy.ndarray`
    alt_pointing: `numpy.ndarray`
    az_pointing: `numpy.ndarray`
    res_degree: bool
        if True, the angular resolution is computed in degrees.
    bins: int or `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    res_bins, res = ana.angular_resolution_per_off_pointing_angle(simu_alt, simu_az, reco_alt, reco_az,
                                                              alt_pointing, az_pointing, bins=bins)
    res_unit='rad'
    if res_degree:
        res = np.rad2deg(res)
        res_unit='deg'

    ax = plot_resolution(res_bins, res, ax=ax, **kwargs)
    ax.set_xlabel("Angular separation to pointing direction [rad]")
    ax.set_ylabel("Angular resolution [{}]".format(res_unit))

    return ax


def plot_impact_parameter_error_per_bin(x, reco_x, reco_y, simu_x, simu_y, bins=10, ax=None, **kwargs):
    """
    Plot the impact parameter error per bin

    Parameters
    ----------
    x: `numpy.ndarray`
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    simu_x: `numpy.ndarray`
    simu_y: `numpy.ndarray`
    bins: arg for `np.histogram`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `plot_resolution`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    bin, res = ana.distance_per_bin(x, reco_x, reco_y, simu_x, simu_y)
    ax = plot_resolution(bin, res, bins=bins, ax=ax, **kwargs)

    return ax


def plot_binned_bias(simu, reco, x, relative_scaling_method=None, ax=None, bins=10, log=False, **kwargs):
    """
    Plot the bias between `simu` and `reco` as a function of bins of `x`

    Parameters
    ----------
    simu: `numpy.ndarray`
    reco: `numpy.ndarray`
    x: `numpy.ndarray`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`
    ax: `matplotlib.pyplot.axis`
    bins: bins for `numpy.histogram`
    log: bool
        if True, logscale is applied to the x axis
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    assert len(simu) == len(reco), \
        "simu and reco arrays should have the same length"
    assert len(simu) == len(x), \
        "simu and energy arrays should have the same length"

    ax = plt.gca() if ax is None else ax

    bins, bias = ana.bias_per_bin(simu, reco, x,
                                  relative_scaling_method=relative_scaling_method,
                                  bins=bins
                                  )

    if log:
        mean_bins = ana.logbin_mean(bins)
        ax.set_xscale('log')
    else:
        mean_bins = (bins[:-1] + bins[1:]) / 2.

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel("bias")

    ax.errorbar(mean_bins, bias, xerr=(mean_bins - bins[:-1], bins[1:] - mean_bins), **kwargs)

    return ax



def plot_bias_per_energy(simu, reco, energy, relative_scaling_method=None, ax=None, **kwargs):
    """
    Plot the bias per bins of energy

    Parameters
    ----------
    simu: `numpy.ndarray`
    reco: `numpy.ndarray`
    energy: `numpy.ndarray`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`
    ax: `matplotlib.pyplot.axis`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    assert len(simu) == len(reco), \
        "simu and reco arrays should have the same length"
    assert len(simu) == len(energy), \
        "simu and energy arrays should have the same length"

    ax = plt.gca() if ax is None else ax

    bins, bias = ana.bias_per_energy(simu, reco, energy, relative_scaling_method=relative_scaling_method)
    mean_bins = ana.logbin_mean(bins)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel("bias")
    ax.set_xlabel("log(E/TeV)")
    ax.set_xscale('log')

    ax.errorbar(mean_bins, bias, xerr=(mean_bins - bins[:-1], bins[1:] - mean_bins), **kwargs)

    return ax


def plot_resolution_difference(bins, reference_resolution, new_resolution, ax=None, **kwargs):
    """
    Plot the algebric difference between a new resolution and reference resolution.

    Parameters
    ----------
    bins: `numpy.ndarray`
    reference_resolution: `numpy.ndarray`
        output from `ctaplot.ana.resolution`
    new_resolution: `numpy.ndarray`
        output from `ctaplot.ana.resolution`
    ax: `matplotlib.pyplot.axis`
    kwargs: args for `ctaplot.plots.plot_resolution`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """

    ax = plt.gca() if ax is None else ax
    delta_res = new_resolution - reference_resolution
    delta_res[:, 1:] = 0    # the condidence intervals have no meaning here
    plot_resolution(bins, delta_res, ax=ax, **kwargs)
    ax.set_ylabel(r"$\Delta$ res")
    ax.set_title("Resolution difference")

    return ax
