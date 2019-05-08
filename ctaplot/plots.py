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




def plot_energy_distribution(SimuE, RecoE, ax=None, outfile=None, maskSimuDetected=True):
    """
    Plot the energy distribution of the simulated particles, detected particles and reconstructed particles
    The plot might be saved automatically if `outfile` is provided.

    Parameters
    ----------
    SimuE: Numpy 1d array of simulated energies
    RecoE: Numpy 1d array of reconstructed energies
    ax: `matplotlib.pyplot.axes`
    outfile: string - output file path
    maskSimuDetected - Numpy 1d array - mask of detected particles for the SimuE array
    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel('Count')

    ax.set_xscale('log')
    count_S, bin_S, o = ax.hist(SimuE, log=True, bins=np.logspace(-3, 3, 30), label="Simulated")
    count_D, bin_D, o = ax.hist(SimuE[maskSimuDetected], log=True, bins=np.logspace(-3, 3, 30), label="Detected")
    count_R, bin_R, o = ax.hist(RecoE, log=True, bins=np.logspace(-3, 3, 30), label="Reconstructed")
    plt.legend(fontsize=SizeLabel)
    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200);
        plt.close()

    return ax


def plot_multiplicity_per_energy(Multiplicity, Energies, ax=None, outfile=None):
    """
    Plot the telescope multiplicity as a function of the energy
    The plot might be saved automatically if `outfile` is provided.

    Parameters
    ----------
    Multiplicity: `numpy.ndarray`
    Energies: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    outfile: string
    """

    assert len(Multiplicity) == len(Energies), "arrays should have same length"
    assert len(Multiplicity) > 0, "arrays are empty"

    E, m_mean, m_min, m_max, m_per = ana.multiplicity_stat_per_energy(Multiplicity, Energies)

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


def plot_field_of_view_map(RecoAlt, RecoAz, AltSource, AzSource, E=None, ax=None, Outfile=None):
    """
    Plot a map in angles [in degrees] of the photons seen by the telescope (after reconstruction)

    Parameters
    ----------
    RecoAlt: `numpy.ndarray`
    RecoAz: `numpy.ndarray`
    AltSource: float, source Altitude
    AzSource: float, source Azimuth
    E: `numpy.ndarray`
    Outfile: string
    """
    dx = 0.05

    ax = plt.gca() if ax is None else ax
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlim(AzSource - dx, AzSource + dx)
    ax.set_ylim(AltSource - dx, AltSource + dx)

    ax.set_xlabel("Az [deg]")
    ax.set_ylabel("Alt [deg]")

    ax.set_axis('equal')

    if E is not None:
        c = np.log10(E)
        plt.colorbar()
    else:
        c = BrewBlues[-1]

    ax.scatter(RecoAz, RecoAlt, c=c)
    ax.scatter(AzSource, AltSource, marker='+', linewidths=3, s=200, c='orange', label="Source position")

    plt.legend()
    if type(Outfile) is str:
        plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);

    return ax


def plot_angles_distribution(RecoAlt, RecoAz, AltSource, AzSource, Outfile=None):
    """
    Plot the distribution of reconstructed angles. Save figure to Outfile in png format.

    Parameters
    ----------
    RecoAlt: `numpy.ndarray`
    RecoAz: `numpy.ndarray`
    AltSource: `float`
    AzSource: `float`
    Outfile: `string`
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
    ax1.set_xlim(AzSource - dx, AzSource + dx)

    ax1.hist(RecoAz, bins=60, range=(AzSource - dx, AzSource + dx))

    ax2 = plt.subplot(212)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()

    ax2.set_xlabel('Alt [deg]', fontsize=SizeLabel)
    ax2.set_ylabel('Count', fontsize=SizeLabel)
    ax2.set_xlim(AltSource - dx, AltSource + dx)

    ax2.hist(RecoAlt, bins=60, range=(AltSource - dx, AltSource + dx))

    if type(Outfile) is str:
        fig.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200)

    return fig


def plot_theta2(RecoAlt, RecoAz, AltSource, AzSource, ax=None, **kwargs):
    """
    Plot the theta2 distribution and display the corresponding angular resolution in degrees.
    The input must be given in radians.

    Parameters
    ----------
    RecoAlt: `numpy.ndarray` - reconstructed altitude angle in radians
    RecoAz: `numpy.ndarray` - reconstructed azimuth angle in radians
    AltSource: `numpy.ndarray` - true altitude angle in radians
    AzSource: `numpy.ndarray` - true azimuth angle in radians
    ax: `matplotlib.pyplot.axes`
    **kwargs: options for `matplotlib.pyplot.hist`
    """

    ax = plt.gca() if ax is None else ax

    theta2 = np.rad2deg(np.sqrt(ana.theta2(RecoAlt, RecoAz, AltSource, AzSource)))**2
    AngRes = np.rad2deg(ana.angular_resolution(RecoAlt, RecoAz, AltSource, AzSource))

    ax.set_xlabel(r'$\theta^2 [deg^2]$')
    ax.set_ylabel('Count')

    ax.hist(theta2, **kwargs)
    ax.set_title(r'angular resolution: {:.3f}(+{:.2f}/-{:.2f}) deg'.format(AngRes[0], AngRes[2], AngRes[1]))

    return ax


def plot_angles_map_distri(RecoAlt, RecoAz, AltSource, AzSource, E, Outfile=None):
    """
    Plot the angles map distribution

    Parameters
    ----------
    RecoAlt: `numpy.ndarray`
    RecoAz: `numpy.ndarray`
    AltSource: float
    AzSource: float
    E: `numpy.ndarray`
    Outfile: str

    Returns
    -------
    fig: `matplotlib.pyplot.figure`
    """

    dx = 0.5

    fig = plt.figure(figsize=(12, 12))

    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)

    plt.xticks(fontsize=SizeTick)
    plt.yticks(fontsize=SizeTick)

    ax1.set_xlim(AzSource - dx, AzSource + dx)
    ax1.set_ylim(AltSource - dx, AltSource + dx)

    plt.xlabel("Az [deg]")
    plt.ylabel("Alt [deg]")

    if len(RecoAlt) > 1000:
        ax1.hist2d(a.RecoAlt, a.RecoAz, bins=60,
                   range=([AltSource - dx, AltSource + dx], [AzSource - dx, AzSource + dx]))
    else:
        ax1.scatter(RecoAz, RecoAlt, c=np.log10(E))
    ax1.scatter(AzSource, AltSource, marker='+', linewidths=3, s=200, c='black')

    ax1.xaxis.set_label_position('top')
    ax1.yaxis.set_tick_params(labelsize=SizeTick)
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_tick_params(labelsize=SizeTick)
    plt.legend('', 'Source position')

    ax2 = plt.subplot2grid((4, 4), (3, 0), colspan=3)
    ax2.set_xlim(AzSource - dx, AzSource + dx)
    ax2.yaxis.tick_right()
    ax2.xaxis.set_tick_params(labelsize=SizeTick)
    ax2.xaxis.set_ticklabels([])
    ax2.xaxis.tick_bottom()
    ax2.hist(RecoAz, bins=60, range=(AzSource - dx, AzSource + dx))
    plt.locator_params(nbins=4)

    ax3 = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
    ax3.set_ylim(AltSource - dx, AltSource + dx)
    ax3.yaxis.set_ticklabels([])
    ax3.yaxis.set_tick_params(labelsize=SizeTick)
    plt.locator_params(nbins=4)

    ax3.spines["left"].set_visible(False)
    ax3.hist(RecoAlt, bins=60, range=(AltSource - dx, AltSource + dx), orientation=u'horizontal')

    if type(Outfile) is str:
        fig.savefig(Outfile, bbox_inches="tight", format='png', dpi=200)

    return fig


def plot_impact_point_map_distri(RecoX, RecoY, telX, telY, **options):
    """
    Map and distributions of the reconstructed impact points.

    Parameters
    ----------
    RecoX: `numpy.ndarray`
    RecoY: `numpy.ndarray`
    telX: `numpy.ndarray`
    telY: `numpy.ndarray`
    options:
        kde=True : make a gaussian fit of the point density
        Outfile='string' : save a png image of the plot under 'string.png'

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

    if options.get("kde") == True:
        kde = gaussian_kde([RecoX, RecoY])
        density = kde([RecoX, RecoY])
        ax1.scatter(RecoX, RecoY, c=density, s=2)
    else:
        ax1.hist2d(RecoX, RecoY, bins=80, norm=LogNorm())

    ax1.xaxis.set_label_position('top')
    ax1.yaxis.set_tick_params(labelsize=SizeTick)
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_tick_params(labelsize=SizeTick)
    plt.legend('', 'Source position')

    ax1.scatter(telX, telY, c='tomato', marker='+', s=90, linewidths=10)

    ax2 = plt.subplot2grid((4, 4), (3, 0), colspan=3)
    ax2.yaxis.tick_right()
    ax2.xaxis.set_tick_params(labelsize=SizeTick)
    ax2.xaxis.set_ticklabels([])
    ax2.xaxis.tick_bottom()
    ax2.hist(RecoX, bins=60)
    plt.locator_params(nbins=4)

    ax3 = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
    ax3.yaxis.set_ticklabels([])
    ax3.yaxis.set_tick_params(labelsize=SizeTick)
    plt.locator_params(nbins=4)

    ax3.spines["left"].set_visible(False)
    ax3.hist(RecoY, bins=60, orientation=u'horizontal')

    if options.get("Outfile"):
        Outfile = options.get("Outfile")
        fig.savefig(Outfile, bbox_inches="tight", format='png', dpi=200)

    return fig


def plot_impact_point_heatmap(RecoX, RecoY, ax=None, Outfile=None):
    """
    Plot the heatmap of the impact points on the site ground and save it under Outfile

    Parameters
    ----------
    RecoX: `numpy.ndarray`
    RecoY: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    Outfile: string
    """

    ax = plt.gca() if ax is None else ax

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=SizeTick)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=SizeTick)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis('equal')

    cm = plt.cm.get_cmap('OrRd')
    h = ax.hist2d(RecoX, RecoY, bins=75, norm=LogNorm(), cmap=cm)
    cb = plt.colorbar(h[3], ax=ax)
    cb.set_label('Event count')

    if type(Outfile) is str:
        plt.savefig(Outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


def plot_multiplicity_hist(multiplicity, ax=None, Outfile=None, xmin=0, xmax=100):
    """
    Histogram of the telescopes multiplicity

    Parameters
    ----------
    multiplicity: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    Outfile: string
    xmin: float
    xmax: float
    """
    m = np.sort(multiplicity)

    ax = plt.gca() if ax is None else ax
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=SizeTick)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=SizeTick)

    if xmax <= xmin:
        xmax = np.floor(m.max()) + 1

    if len(m) > 0:
        ax.set_xlim(xmin, xmax + 1)
        n, b, s = ax.hist(m, bins=(xmax - xmin), align='left', range=(xmin, xmax), label='Telescope multiplicity')
        ax.xaxis.set_ticks(np.append(np.linspace(xmin, xmax, 10, dtype=int), [1, 2, 3, 4, 5]))
        x50 = m[int(np.floor(0.5 * len(m)))] + 0.5
        x90 = m[int(np.floor(0.9 * len(m)))] + 0.5
        if xmin < x50 < xmax:
            ax.vlines(x50, 0, n[int(m[int(np.floor(0.5 * len(m)))])], color=BrewOranges[-2], label='50%')
        if xmin < x90 < xmax:
            ax.vlines(x90, 0, n[int(m[int(np.floor(0.9 * len(m)))])], color=BrewOranges[-1], label='90%')

    plt.legend(fontsize=SizeLabel)
    ax.set_title("Telescope multiplicity")
    if type(Outfile) is str:
        plt.savefig(Outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


def plot_multiplicity_per_telescope_type(EventTup, Outfile=None):
    """
    Plot the multiplicity for each telescope type

    Parameters
    ----------
    EventTup
    Outfile

    Returns
    -------

    """
    LST = []
    SST = []
    MST = []
    for tups in EventTup:
        lst = 0
        mst = 0
        sst = 0
        for t in tups:
            if t[2] == 0:
                lst += 1
            elif 1 <= t[2] <= 3:
                mst += 1
            elif t[2] >= 4:
                sst += 1
        LST.append(lst)
        MST.append(mst)
        SST.append(sst)

    fig = plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(311)
    if max(LST) > 0:
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.get_xaxis().tick_bottom()
        ax1.get_yaxis().tick_left()
        ax1.set_ylabel('Count')
        ax1.hist(LST, bins=max(LST), align='right')
        ax1.text(.8, .8, 'LST', horizontalalignment='center', transform=ax1.transAxes, fontsize=SizeLabel)
    else:
        ax1.text(0.5, .5, "No LST triggered", horizontalalignment='center', fontsize=SizeLabel)

    ax2 = plt.subplot(312)
    if max(MST) > 0:
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.get_xaxis().tick_bottom()
        ax2.get_yaxis().tick_left()
        ax2.set_ylabel('Count')

        ax2.hist(MST, bins=max(MST), align='right')
        ax2.text(.8, .8, 'MST', horizontalalignment='center', transform=ax2.transAxes, fontsize=SizeLabel)
    else:
        ax2.text(0.5, .5, "No MST triggered", horizontalalignment='center', fontsize=SizeLabel)

    ax3 = plt.subplot(313)
    if max(SST) > 0:
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.get_xaxis().tick_bottom()
        ax3.get_yaxis().tick_left()
        ax3.set_xlabel('Number of Telescopes triggered')
        ax3.set_ylabel('Count')
        ax3.hist(SST, bins=max(SST), align='right')
        ax3.text(.8, .8, 'SST', horizontalalignment='center', transform=ax3.transAxes, fontsize=SizeLabel)
    else:
        ax3.text(0.5, .5, "No SST triggered", horizontalalignment='center', fontsize=SizeLabel)

    if type(Outfile) is str:
        plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200)

    return LST, MST, SST


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

    ax.errorbar(x, res[:, 0], xerr=(bins[1:] - bins[:-1]) / 2.,
                yerr=(res[:, 0] - res[:, 1], res[:, 2] - res[:, 0]), fmt='o', **kwargs)

    ax.set_title('Resolution')
    return ax


def plot_effective_area_per_energy(SimuE, RecoE, simuArea, ax=None, **kwargs):
    """
    Plot the effective area as a function of the energy

    Parameters
    ----------
    SimuE: `numpy.ndarray` - all simulated event energies
    RecoE: `numpy.ndarray` - all reconstructed event energies
    simuArea: float
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
    >>> simue = 10**(-2 + 4*np.random.rand(1000))
    >>> recoe = 10**(-2 + 4*np.random.rand(100))
    >>> ax = ctaplot.plots.plot_effective_area_per_energy(simue, recoe, irf.LaPalmaArea_prod3)
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

    E_bin, Seff = ana.effective_area_per_energy(SimuE, RecoE, simuArea)
    E = ana.logbin_mean(E_bin)
    ax.errorbar(E, Seff, xerr=(E_bin[1:] - E_bin[:-1]) / 2., fmt='o', **kwargs)

    return ax


def plot_effective_area_cta_requirements(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the effective area

    Parameters
    ----------
    cta_site: string - see `hipectaold.ana.cta_requirements`
    ax: `matplotlib.pyplot.axes`, optional

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_requirements()
    cta_req.site = cta_site
    e_cta, ef_cta = cta_req.get_effective_area()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirements {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)

    return ax


def plot_effective_area_cta_performances(cta_site, ax=None, **kwargs):
    """
    Plot the CTA performances for the effective area

    Parameters
    ----------
    cta_site: string - see `hipectaold.ana.cta_requirements`
    ax: `matplotlib.pyplot.axes`, optional

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_performances()
    cta_req.site = cta_site
    e_cta, ef_cta = cta_req.get_effective_area()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performances {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)

    return ax


def saveplot_effective_area_per_energy(SimuE, RecoE, simuArea, ax=None, Outfile="AngRes", cta_site=None, **kwargs):
    """
    Plot the angular resolution as a function of the energy and save the plot in png format

    Parameters
    ----------
    RecoAlt: `numpy.ndarray`
    RecoAz: `numpy.ndarray`
    AltSource: float
    AzSource: float
    SimuE: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    Outfile: string
    cta_site: string
    kwargs: args for `ctaplot.plots.plot_angular_res_per_energy`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    ax = plot_effective_area_per_energy(SimuE, RecoE, simuArea, ax=ax, **kwargs)

    if cta_site:
        ax = plot_effective_area_cta_requirements(cta_site, ax=ax, color='black')

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200)
    plt.close()

    return ax


def plot_sensitivity_cta_requirements(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the sensitivity
    Parameters
    ----------
    cta_site: string - see `ctaplot.ana.cta_requirements`
    ax: `matplotlib.pyplot.axes`, optional

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_requirements()
    cta_req.site = cta_site
    e_cta, ef_cta = cta_req.get_sensitivity()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirements {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)

    return ax


def plot_sensitivity_cta_performances(cta_site, ax=None, **kwargs):
    """
    Plot the CTA performances for the sensitivity
    Parameters
    ----------
    cta_site: string - see `ctaplot.ana.cta_requirements`
    ax: `matplotlib.pyplot.axes`, optional

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_performances()
    cta_req.site = cta_site
    e_cta, ef_cta = cta_req.get_sensitivity()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel(r'Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performances {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)

    return ax


def plot_layout_map(TelX, TelY, TelId, TelType, LayoutId, Outfile="LayoutMap"):
    """
    Plot the layout map of telescopes positions - depreciated

    Parameters
    ----------
    TelX: `numpy.ndarray`
    TelY: `numpy.ndarray`
    TelId: `numpy.ndarray`
    TelType: `numpy.ndarray`
    LayoutId: `numpy.ndarray`
    Outfile: string

    Returns
    -------

    """

    # 0 LST, (1,2,3) MST, (4,5,6) SST => 0 for LST, 1 for MST, 2 fot SST
    type = np.floor((2 + TelType) / 3)
    plt.figure(figsize=(12, 12))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    mask_layout = np.in1d(TelId, LayoutId)
    mask_lst = mask_layout & np.in1d(TelType, [0])
    mask_mst = mask_layout & np.in1d(TelType, [1, 2, 3])
    mask_sst = mask_layout & np.in1d(TelType, [4, 5, 6])

    plt.axis('equal')
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)

    ax.scatter(TelX[mask_lst], TelY[mask_lst],
               s=100 / (1 + type[mask_lst]), c=BrewBlues[-1], cmap='Paired', label="LST")
    ax.scatter(TelX[mask_mst], TelY[mask_mst],
               s=100 / (1 + type[mask_mst]), c=BrewReds[-1], cmap='Paired', label="MST")
    ax.scatter(TelX[mask_sst], TelY[mask_sst],
               s=100 / (1 + type[mask_sst]), c=BrewGreens[-1], cmap='Paired',
               label="SST")

    plt.title("CTA site with layout %s" % Outfile.split('.')[-1], fontsize=SizeTitleArticle)
    plt.legend(fontsize=SizeLabel)
    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()


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

    E_bin, RES = ana.resolution_per_energy(simu, reco, energy)

    E = ana.logbin_mean(E_bin)

    ax.errorbar(E, RES[:, 0], xerr=(E_bin[1:] - E_bin[:-1]) / 2.,
                yerr=(RES[:, 0] - RES[:, 1], RES[:, 2] - RES[:, 0]), fmt='o', **kwargs)

    ax.set_title('Resolution')
    return ax


def plot_angular_res_per_energy(RecoAlt, RecoAz, AltSource, AzSource, SimuE,
                                percentile=68.27, confidence_level=0.95, bias_correction=False,
                                ax=None, **kwargs):
    """
    Plot the angular resolution as a function of the energy

    Parameters
    ----------
    RecoAlt: `numpy.ndarray`
    RecoAz: `numpy.ndarray`
    AltSource: float
    AzSource: float
    SimuE: `numpy.ndarray`
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

    e_bin, RES = ana.angular_resolution_per_energy(RecoAlt, RecoAz, AltSource, AzSource, SimuE,
                                                   percentile=percentile,
                                                   confidence_level=confidence_level,
                                                   bias_correction=bias_correction
                                                   )

    # Angular resolution is traditionally presented in degrees
    RES = np.degrees(RES)

    E = ana.logbin_mean(e_bin)

    ax.errorbar(E, RES[:, 0], xerr=(e_bin[1:] - e_bin[:-1]) / 2.,
                yerr=(RES[:, 0] - RES[:, 1], RES[:, 2] - RES[:, 0]), fmt='o', **kwargs)

    ax.set_title('Angular resolution')
    return ax


def saveplot_angular_res_per_energy(RecoAlt, RecoAz, AltSource, AzSource, SimuE, ax=None, Outfile="AngRes",
        cta_site=None, **kwargs):
    """
    Plot the angular resolution as a function of the energy and save the plot in png format

    Parameters
    ----------
    RecoAlt: `numpy.ndarray`
    RecoAz: `numpy.ndarray`
    AltSource: float
    AzSource: float
    SimuE: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    Outfile: string
    cta_site: string
    kwargs: args for `hipectaold.plots.plot_angular_res_per_energy`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    ax = plot_angular_res_per_energy(RecoAlt, RecoAz, AltSource, AzSource, SimuE, ax=ax, **kwargs)

    if cta_site:
        ax = plot_angular_res_requirements(cta_site, ax=ax)

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()

    return ax


def plot_angular_res_cta_requirements(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the angular resolution
    Parameters
    ----------
    cta_site: string, see `ana.cta_requirements`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    cta_req = ana.cta_requirements()
    cta_req.site = cta_site
    e_cta, ar_cta = cta_req.get_angular_resolution()

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirements {}".format(cta_site)

    ax.plot(e_cta, ar_cta, **kwargs)

    ax.set_ylabel(r'$\theta [deg]$')
    ax.set_xlabel('Energy [TeV]')

    ax.set_xscale('log')
    ax.set_title('Angular resolution')
    return ax


def plot_angular_res_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the official CTA performances (June 2018) for the angular resolution

    Parameters
    ----------
    cta_site: string, see `ana.cta_performances`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    cta_req = ana.cta_performances()
    cta_req.site = cta_site
    e_cta, ar_cta = cta_req.get_angular_resolution()

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performances {}".format(cta_site)

    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    ax.set_ylabel(r'$\theta [deg]$')
    ax.set_xlabel('Energy [TeV]')
    ax.set_title('Angular resolution')
    return ax


def plot_impact_parameter_error(RecoX, RecoY, SimuX, SimuY, ax=None, **kwargs):
    """
    plot impact parameter error distribution and save it under Outfile
    Parameters
    ----------
    RecoX: `numpy.ndarray`
    RecoY: `numpy.ndarray`
    SimuX: `numpy.ndarray`
    SimuY: `numpy.ndarray`
    Outfile: string
    """
    d = np.sqrt((RecoX - SimuX) ** 2 + (RecoY - SimuY) ** 2)

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

def plot_impact_parameter_error_per_energy(RecoX, RecoY, SimuX, SimuY, SimuE, ax=None, **kwargs):
    """
    plot the impact parameter error distance as a function of energy and save the plot as Outfile
    Parameters
    ----------
    RecoX: `numpy.ndarray`
    RecoY: `numpy.ndarray`
    SimuX: `numpy.ndarray`
    SimuY: `numpy.ndarray`
    SimuE: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    E, err_mean : numpy arrays
    """
    irf = ana.irf_cta()
    E_bin = irf.E_bin
    E = []
    err_mean = []
    err_min = []
    err_max = []
    err_std = []
    for i, eb in enumerate(E_bin[:-1]):
        mask = (SimuE > E_bin[i]) & (SimuE < E_bin[i + 1])
        E.append(np.mean([E_bin[i], E_bin[i + 1]]))
        if True in mask:
            d = ana.impact_parameter_error(RecoX[mask], RecoY[mask], SimuX[mask], SimuY[mask])
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


def plot_impact_parameter_error_per_multiplicity(RecoX, RecoY, SimuX, SimuY, Multiplicity,
        max_mult=None, ax=None, **kwargs):
    """
    Plot the impact parameter error as a function of multiplicity
    TODO: refactor and clean code

    Parameters
    ----------
    RecoX: `numpy.ndarray`
    RecoY: `numpy.ndarray`
    SimuX: `numpy.ndarray`
    SimuY: `numpy.ndarray`
    Multiplicity: `numpy.ndarray`
    max_mult: optional, max multiplicity - float
    ax: `matplotlib.pyplot.axes`

    Returns
    -------

    """
    max_mult = Multiplicity.max() + 1 if max_mult is None else max_mult

    M = np.arange(Multiplicity.min(), max_mult)
    e_mean = []
    e_min = []
    e_max = []
    e_std = []
    for m in M:
        mask = (Multiplicity == m)
        if True in mask:
            d = ana.impact_parameter_error(RecoX[mask], RecoY[mask], SimuX[mask], SimuY[mask])
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


def plot_site_map(telX, telY, telTypes=None, Outfile="SiteMap.png"):
    """
    Map of the site with telescopes positions

    Parameters
    ----------
    telX: `numpy.ndarray`
    telY: `numpy.ndarray`
    telTypes: `numpy.ndarray`
    Outfile: string - name of the output file
    """
    plt.figure(figsize=(12, 12))

    assert (len(telX) == len(telY)), "telX and telY should have the same length"
    if telTypes:
        assert (len(telTypes) == len(telX)), "telTypes and telX should have the same length"
        plt.scatter(telX, telY, color=telTypes, s=30)
    else:
        plt.scatter(telX, telY, color='black', s=30)

    plt.axis('equal')
    plt.savefig(Outfile, bbox_inches="tight", format='png', dpi=200)
    plt.close()


def plot_impact_map(impactX, impactY, telX, telY, telTypes=None, Outfile="ImpactMap.png"):
    """
    Map of the site with telescopes positions and impact points heatmap

    Parameters
    ----------
    impactX: `numpy.ndarray`
    impactY: `numpy.ndarray`
    telX: `numpy.ndarray`
    telY: `numpy.ndarray`
    telTypes: `numpy.ndarray`
    Outfile: string - name of the output file
    """
    plt.figure(figsize=(12, 12))

    plt.hist2d(impactX, impactY, bins=40)
    plt.colorbar()

    assert (len(telX) == len(telY)), "telX and telY should have the same length"
    if telTypes:
        assert (len(telTypes) == len(telX)), "telTypes and telX should have the same length"
        plt.scatter(telX, telY, color=telTypes, s=30)
    else:
        plt.scatter(telX, telY, color='black', s=30)

    plt.axis('equal')
    plt.savefig(Outfile, bbox_inches="tight", format='png', dpi=200)
    plt.close()


def plot_energy_bias(SimuE, RecoE, ax=None, **kwargs):
    """
    Plot the energy bias

    Parameters
    ----------
    SimuE: `numpy.ndarray`
    RecoE: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    assert len(SimuE) == len(RecoE), "simulated and reconstructured energy arrrays should have the same length"

    ax = plt.gca() if ax is None else ax

    E_bin, biasE = ana.energy_bias(SimuE, RecoE)
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel("bias (median($E_{reco}/E_{simu}$ - 1)")
    ax.set_xlabel("log(E/TeV)")
    ax.set_xscale('log')
    plt.legend()
    ax.set_title('Energy bias')

    ax.errorbar(E, biasE, xerr=(E - E_bin[:-1], E_bin[1:] - E), **kwargs)

    return ax


def plot_energy_resolution(SimuE, RecoE,
                           percentile=68.27, confidence_level=0.95, bias_correction=False,
                           ax=None,  **kwargs):
    """
    Plot the enregy resolution as a function of the energy

    Parameters
    ----------
    SimuE: `numpy.ndarray`
    RecoE: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    bias_correction: `bool`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    assert len(SimuE) == len(RecoE), "simulated and reconstructured energy arrrays should have the same length"

    ax = plt.gca() if ax is None else ax

    E_bin, Eres = ana.energy_resolution_per_energy(SimuE, RecoE,
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


def plot_energy_resolution_cta_requirements(cta_site, ax=None, **kwargs):
    """
    Plot the cta requirement for the energy resolution

    Parameters
    ----------
    cta_site: string, see `ana.cta_requirements`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    cta_req = ana.cta_requirements()
    cta_req.site = cta_site
    e_cta, ar_cta = cta_req.get_energy_resolution()

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirements {}".format(cta_site)

    ax.set_ylabel(r"$(\Delta E/E)_{68}$")
    ax.set_xlabel("Energy [TeV]")
    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    return ax


def plot_energy_resolution_cta_performances(cta_site, ax=None, **kwargs):
    """
    Plot the cta performances (June 2018) for the energy resolution

    Parameters
    ----------
    cta_site: string, see `ana.cta_performances`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    cta_req = ana.cta_performances()
    cta_req.site = cta_site
    e_cta, ar_cta = cta_req.get_energy_resolution()

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performances {}".format(cta_site)

    ax.set_ylabel(r"$(\Delta E/E)_{68}$")
    ax.set_xlabel("Energy [TeV]")
    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    return ax


def saveplot_energy_resolution(SimuE, RecoE, Outfile="EnergyResolution.png", cta_site=None):
    """
    plot the energy resolution of the reconstruction
    Parameters
    ----------
    SimuE: `numpy.ndarray`
    RecoE: `numpy.ndarray`
    cta_goal: boolean - If True CTA energy resolution requirement is plotted

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plot_energy_resolution(SimuE, RecoE)

    if cta_site != None:
        ax = plot_energy_resolution_cta_requirements(cta_site, ax=ax)

    plt.savefig(Outfile, bbox_inches="tight", format='png', dpi=200)
    plt.close()
    return ax


def plot_reco_histo(y_true, y_reco):
    """
    plot the histogram of a reconstructed feature after prediction from a machine learning algorithm
    plt.show() to display
    Parameters
    ----------
    y_true: real values of the feature to predict
    y_reco: predicted values by the ML algo
    """
    plt.figure(figsize=(7, 6))

    plt.hist2d(y_true, y_reco, bins=50)
    plt.colorbar()
    plt.plot(y_true, y_true, color='black', label="perfect prediction line")
    plt.axis('equal')
    plt.xlabel("To predict")
    plt.ylabel("Predicted")
    plt.legend()

    plt.title("Histogram of the predicted feature")


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


def plot_site(tel_x, tel_y, ax=None, **kwargs):
    """
    Plot the telescopes positions
    Parameters
    ----------
    tel_x: 1D numpy array
    tel_y: 1D numpy array
    ax: `~matplotlib.axes.Axes` or None
    **kwargs : Extra keyword arguments are passed to `matplotlib.pyplot.scatter`

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
    """
    ax = plt.gca() if ax is None else ax

    ax.scatter(tel_x, tel_y, **kwargs)
    ax.axis('equal')

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

    ax.errorbar(
        E, RES[:, 0],
        xerr=(E - E_bin[:-1], E_bin[1:] - E),
        yerr=(RES[:, 0] - RES[:, 1], RES[:, 2] - RES[:, 0]),
        fmt='o',
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


def plot_dispersion(X_true, X_exp, x_log=False, ax=None, **kwargs):
    """
    Plot the dispersion around an expected value X_true

    Parameters
    ----------
    X_true: `numpy.ndarray`
    X_exp: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.hist2d`

    Returns
    -------
    `maptlotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    if not 'bins' in kwargs:
        kwargs['bins'] = 50

    x = np.log10(X_true) if x_log else X_true

    ax.hist2d(x, X_true - X_exp, **kwargs)
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
    ax.errorbar(energy_nodes, seff, xerr=(ebin[1:] - ebin[:-1]) / 2., fmt='o', **kwargs)

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