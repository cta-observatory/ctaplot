"""
plots.py
========
Functions to make IRF and other reconstruction quality-check plots
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
from ..ana import ana
from sklearn import metrics
from sklearn.multiclass import LabelBinarizer
from ..io.dataset import load_any_resource
import astropy.units as u


__all__ = ['plot_resolution',
           'plot_resolution_difference',
           'plot_energy_resolution',
           'plot_binned_bias',
           'plot_energy_bias',
           'plot_impact_parameter_error_per_bin',
           'plot_layout_map',
           'plot_multiplicity_per_telescope_type',
           'plot_multiplicity_hist',
           'plot_angular_resolution_cta_performance',
           'plot_angular_resolution_cta_requirement',
           'plot_angular_resolution_per_energy',
           'plot_angular_resolution_per_off_pointing_angle',
           'plot_bias_per_energy',
           'plot_binned_stat',
           'plot_dispersion',
           'plot_effective_area_cta_performance',
           'plot_effective_area_cta_requirement',
           'plot_effective_area_per_energy',
           'plot_effective_area_per_energy_power_law',
           'plot_energy_distribution',
           'plot_energy_resolution_cta_performance',
           'plot_energy_resolution_cta_requirement',
           'plot_feature_importance',
           'scatter_events_field_of_view',
           'plot_impact_map',
           'plot_impact_parameter_error_site_center',
           'plot_impact_parameter_resolution_per_energy',
           'plot_impact_point_heatmap',
           'plot_impact_resolution_per_energy',
           'plot_migration_matrix',
           'plot_multiplicity_per_energy',
           'plot_resolution_per_energy',
           'plot_sensitivity_cta_performance',
           'plot_sensitivity_cta_requirement',
           'plot_theta2',
           'plot_roc_curve',
           'plot_roc_curve_gammaness',
           'plot_roc_curve_multiclass',
           'plot_roc_curve_gammaness_per_energy',
           'plot_gammaness_distribution',
           'plot_sensitivity_magic_performance',
           'plot_rate',
           'plot_gamma_rate',
           'plot_background_rate',
           'plot_background_rate_magic',
           'plot_gamma_rate_magic',
           ]


def plot_energy_distribution(true_energy, reco_energy, ax=None, outfile=None, mask_mc_detected=True):
    """
    Plot the true_energy distribution of the simulated particles, detected particles and reconstructed particles
    The plot might be saved automatically if `outfile` is provided.

    Parameters
    ----------
    true_energy: `numpy.ndarray`
        simulated energies
    reco_energy: `numpy.ndarray`
        reconstructed energies
    ax: `matplotlib.pyplot.axes`
    outfile: string
        output file path
    mask_mc_detected: `numpy.ndarray`
        mask of detected particles for the SimuE array
    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel('Count')

    ax.set_xscale('log')
    count_S, bin_S, o = ax.hist(true_energy, log=True, bins=np.logspace(-3, 3, 30), label="Simulated")
    count_D, bin_D, o = ax.hist(true_energy[mask_mc_detected], log=True, bins=np.logspace(-3, 3, 30), label="Detected")
    count_R, bin_R, o = ax.hist(reco_energy, log=True, bins=np.logspace(-3, 3, 30), label="Reconstructed")
    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)
        plt.close()

    return ax


def plot_multiplicity_per_energy(multiplicity, energies, ax=None, outfile=None):
    """
    Plot the telescope multiplicity as a function of the true_energy
    The plot might be saved automatically if `outfile` is provided.

    Parameters
    ----------
    multiplicity: `numpy.ndarray`
        telescope multiplcity
    energies: `numpy.ndarray`
        event energies
    ax: `matplotlib.pyplot.axes`
    outfile: string
        path to the output file to save the figure
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


def scatter_events_field_of_view(reco_alt, reco_az, source_alt, source_az, color_scale=None, ax=None):
    """
    Plot a map in angles [in degrees] of the photons seen by the telescope (after reconstruction)

    Parameters
    ----------
    reco_alt: `numpy.ndarray`
        reconstructed altitudes
    reco_az: `numpy.ndarray`
        reconstructed azimuths
    source_alt: float, source Altitude
        altitude of the source
    source_az: float, source Azimuth
        azimuth of the source
    color_scale: `numpy.ndarray`
        if given, set the colorbar
    ax: `matplotlib.pyplot.axes`
    outfile: string
        path to the output figure file. if None, the plot is not saved

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

    ax.axis('equal')

    if color_scale is not None:
        c = color_scale
        plt.colorbar()
    else:
        c = 'blue'

    ax.scatter(reco_az, reco_alt, c=c)
    ax.scatter(source_az, source_alt, marker='+', linewidths=3, s=200, c='orange', label="Source position")

    ax.legend()

    return ax




def plot_theta2(reco_alt, reco_az, true_alt, true_az, bias_correction=False, ax=None, **kwargs):
    """
    Plot the theta2 distribution and display the corresponding angular resolution in degrees.
    The input must be given in radians.

    Parameters
    ----------
    reco_alt: `numpy.ndarray`
        reconstructed altitude angle in radians
    reco_az: `numpy.ndarray`
        reconstructed azimuth angle in radians
    true_alt: `numpy.ndarray`
        true altitude angle in radians
    true_az: `numpy.ndarray`
        true azimuth angle in radians
    ax: `matplotlib.pyplot.axes`
    **kwargs:
        options for `matplotlib.pyplot.hist`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    if bias_correction:
        bias_alt = ana.bias(true_alt, reco_alt)
        bias_az = ana.bias(true_az, reco_az)
        reco_alt = reco_alt - bias_alt
        reco_az = reco_az - bias_az

    theta2 = np.rad2deg(np.sqrt(ana.theta2(reco_alt, reco_az, true_alt, true_az)))**2
    ang_res = np.rad2deg(ana.angular_resolution(reco_alt, reco_az, true_alt, true_az))

    ax.set_xlabel(r'$\theta^2 [deg^2]$')
    ax.set_ylabel('Count')

    ax.hist(theta2, **kwargs)

    err_max = (ang_res[2] - ang_res[0])
    err_min = (ang_res[0] - ang_res[1])
    ax.set_title(rf'angular resolution: {ang_res[0]:.3f}(+{err_max:.1e}/-{err_min:.1e})deg')

    return ax


def plot_impact_point_heatmap(reco_x, reco_y, ax=None, outfile=None, **kwargs):
    """
    Plot the heatmap of the impact points on the site ground and save it under Outfile

    Parameters
    ----------
    reco_x: `numpy.ndarray`
        reconstructed x positions
    reco_y: `numpy.ndarray`
        reconstructed y positions
    ax: `matplotlib.pyplot.axes`
    outfile: string
        path to the output file. If None, the figure is not saved.
    """

    ax = plt.gca() if ax is None else ax

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis('equal')

    kwargs.setdefault('norm', LogNorm())
    kwargs.setdefault('cmap', plt.cm.get_cmap('PuBu'))
    kwargs.setdefault('bins', 50)
    h = ax.hist2d(reco_x, reco_y, **kwargs)
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
        ax.vlines(x50, 0, n[int(m[int(np.floor(0.5 * len(m)))])], label='50%')
    if quartils and (xmin < x90 < xmax):
        ax.vlines(x90, 0, n[int(m[int(np.floor(0.9 * len(m)))])], label='90%')

    ax.set_title("Telescope multiplicity")
    ax.grid(True)

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
    outfile: str
        path to the output figure. If None, the figure is not saved.
    quartils: bool - True to plot 50% and 90% quartil mark
    kwargs: args for `matplotlib.pyplot.hist`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    from matplotlib.ticker import MaxNLocator

    ax = plt.gca() if ax is None else ax

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
        ax.vlines(x90+0.5, 0, len(multiplicity[multiplicity==x90]), label='90%')

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

    ax.errorbar(x, res[:, 0], xerr=[x - bins[:-1], bins[1:] - x],
                yerr=(res[:, 0] - res[:, 1], res[:, 2] - res[:, 0]), **kwargs)

    ax.set_title('Resolution')
    return ax


def plot_effective_area_per_energy(true_energy, reco_energy, simulated_area, ax=None, **kwargs):
    """
    Plot the effective area as a function of the true energy

    Parameters
    ----------
    true_energy: `numpy.ndarray`
        all simulated event energies
    reco_energy: `numpy.ndarray`
        all reconstructed event energies
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
    >>> true_e = 10**(-2 + 4*np.random.rand(1000))
    >>> reco_e = 10**(-2 + 4*np.random.rand(100))
    >>> ax = ctaplot.plots.plot_effective_area_per_energy(true_e, reco_e, irf.LaPalmaArea_prod3)
    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel(r'$E_T$ [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    E_bin, Seff = ana.effective_area_per_energy(true_energy, reco_energy, simulated_area)
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(E, Seff, xerr=(E_bin[1:] - E_bin[:-1]) / 2., **kwargs)
    ax.grid('on', which='both')
    return ax


def plot_effective_area_cta_requirement(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the effective area as a function of the true energy

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
    ax.set_xlabel(r'$E_T$ [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirement {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)
    ax.grid('on', which='both')
    ax.legend()
    return ax


def plot_effective_area_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the CTA performances for the effective area as a function of the true energy

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
    ax.set_xlabel(r'$E_T$ [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performance {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)
    ax.grid('on', which='both')
    ax.legend()
    return ax


def plot_sensitivity_cta_requirement(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the sensitivity
    Parameters
    ----------
    cta_site: string
        see `ctaplot.ana.cta_requirement`
    ax: `matplotlib.pyplot.axes`
        optional

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_req = ana.cta_requirement(cta_site)
    e_cta, ef_cta = cta_req.get_sensitivity()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E_R$ [TeV]')
    ax.set_ylabel(r'Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA requirement {}".format(cta_site)

    ax.plot(e_cta, ef_cta, **kwargs)
    ax.grid('on', which='both')
    ax.legend()
    return ax


def plot_sensitivity_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the CTA performances for the sensitivity

    Parameters
    ----------
    cta_site: string
        see `ctaplot.ana.cta_requirement`
    ax: `matplotlib.pyplot.axes`
        optional

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    cta_perf = ana.cta_performance(cta_site)
    e_cta, ef_cta = cta_perf.get_sensitivity()
    e_bin = cta_perf.E_bin

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E_R$ [TeV]')
    ax.set_ylabel(r'Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')

    if not 'label' in kwargs:
        kwargs['label'] = "CTA performance {}".format(cta_site)

    ax.errorbar(e_cta, ef_cta, xerr=np.array([e_cta-e_bin[:-1], e_bin[1:]-e_cta]), **kwargs)
    ax.grid('on', which='both')
    ax.legend()
    return ax


def plot_layout_map(tel_x, tel_y, tel_type=None, ax=None, **kwargs):
    """
    Plot the layout map of telescopes positions

    Parameters
    ----------
    tel_x: `numpy.ndarray`
        telescopes x positions
    tel_y: `numpy.ndarray`
        telescopes y positions
    tel_type: `numpy.ndarray`
        telescopes types
    ax: `matplotlib.pyplot.axes`
        optional
    kwargs:
        options for `matplotlib.pyplot.scatter`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    ax = plt.gca() if ax is None else ax
    ax.axis('equal')

    if tel_type is not None and 'c' not in kwargs and 'color' not in kwargs:
        values = np.arange(len(set(tel_type)))
        kwargs['c'] = [values[list(set(tel_type)).index(type)] for type in tel_type]
    ax.scatter(tel_x, tel_y, **kwargs)

    return ax


def plot_resolution_per_energy(true, reco, energy, ax=None, **kwargs):
    """
    Plot a variable resolution as a function of the true_energy

    Parameters
    ----------
    reco: `numpy.ndarray`
        reconstructed values of a variable
    true: `numpy.ndarray`
        true values of the variable
    energy: `numpy.ndarray`
        event energies in TeV
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

    energy_bin, resolution = ana.resolution_per_energy(true, reco, energy)

    E = ana.logbin_mean(energy_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.errorbar(E, resolution[:, 0], xerr=(energy_bin[1:] - energy_bin[:-1]) / 2.,
                yerr=(resolution[:, 0] - resolution[:, 1], resolution[:, 2] - resolution[:, 0]), **kwargs)
    ax.grid('on', which='both')
    ax.set_title('Resolution')
    return ax


def plot_angular_resolution_per_energy(reco_alt, reco_az, true_alt, true_az, reco_energy,
                                       percentile=68.27, confidence_level=0.95, bias_correction=False,
                                       ax=None, **kwargs):
    """
    Plot the angular resolution as a function of the reconstructed energy

    Parameters
    ----------
    reco_alt: `numpy.ndarray`
        reconstructed altitudes in radians
    reco_az: `numpy.ndarray`
        reconstructed azimuths in radians
    true_alt: `numpy.ndarray`
        true altitudes in radians
    true_az: `numpy.ndarray`
        true azimuths in radians
    reco_energy: `numpy.ndarray`
        energies in TeV
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    try:
        e_bin, RES = ana.angular_resolution_per_energy(reco_alt, reco_az, true_alt, true_az, reco_energy,
                                                       percentile=percentile,
                                                       confidence_level=confidence_level,
                                                       bias_correction=bias_correction
                                                       )
    except Exception as e:
        print('Angular resolution ', e)
    else:
        # Angular resolution is traditionally presented in degrees
        RES = np.degrees(RES)

        E = ana.logbin_mean(e_bin)

        if 'fmt' not in kwargs:
            kwargs['fmt'] = 'o'

        ax.set_ylabel('Angular Resolution [deg]')
        ax.set_xlabel(r'$E_R$ [TeV]')
        ax.set_xscale('log')
        ax.set_title('Angular resolution')

        ax.errorbar(E, RES[:, 0], xerr=(e_bin[1:] - e_bin[:-1]) / 2.,
                    yerr=(RES[:, 0] - RES[:, 1], RES[:, 2] - RES[:, 0]), **kwargs)
        ax.grid('on', which='both')
    finally:
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

    ax.set_ylabel(r'Angular Resolution [deg]')
    ax.set_xlabel(r'$E_R$ [TeV]')

    ax.set_xscale('log')
    ax.set_title('Angular resolution')
    ax.grid('on', which='both')
    ax.legend()
    return ax


def plot_angular_resolution_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the official CTA performances (June 2018) for the angular resolution

    Parameters
    ----------
    cta_site: string
        see `ana.cta_performance`
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
    ax.set_ylabel('Angular resolution [deg]')
    ax.set_xlabel(r'$E_R$ [TeV]')
    ax.set_title('Angular resolution')
    ax.grid('on', which='both')
    ax.legend()
    return ax


def hist_impact_parameter_error(reco_x, reco_y, true_x, true_y, ax=None, **kwargs):
    """
    plot impact parameter error distribution and save it under Outfile
    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    true_x: `numpy.ndarray`
    true_y: `numpy.ndarray`
    Outfile: string
    """
    d = ana.impact_parameter_error(reco_x, reco_y, true_x, true_y)

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


def plot_impact_parameter_resolution_per_energy(reco_x, reco_y, true_x, true_y, energy, ax=None, **kwargs):
    """

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    true_x: `numpy.ndarray`
    true_y: `numpy.ndarray`
    energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `ctaplot.plots.plot_resolution`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """
    bin, res = ana.impact_resolution_per_energy(reco_x, reco_y, true_x, true_y, energy)
    ax = plot_resolution(bin, res, log=True, ax=ax, **kwargs)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Impact parameter resolution")
    ax.set_title("Impact parameter resolution as a function of the true_energy")
    ax.grid('on', which='both')
    return ax


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


def plot_energy_bias(true_energy, reco_energy, ax=None, **kwargs):
    """
    Plot the true_energy bias

    Parameters
    ----------
    true_energy: `numpy.ndarray`
    reco_energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    assert len(true_energy) == len(reco_energy), "simulated and reconstructured true_energy arrrays should have the same length"

    ax = plt.gca() if ax is None else ax

    E_bin, biasE = ana.energy_bias(true_energy, reco_energy)
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel("bias (median($E_{reco}/E_{true}$ - 1)")
    ax.set_xlabel(r'$E_R$ [TeV]')
    ax.set_xscale('log')
    ax.set_title('Energy bias')

    ax.errorbar(E, biasE, xerr=(E - E_bin[:-1], E_bin[1:] - E), **kwargs)
    ax.grid('on', which='both')

    return ax


def plot_energy_resolution(true_energy, reco_energy,
                           percentile=68.27, confidence_level=0.95, bias_correction=False,
                           ax=None, **kwargs):
    """
    Plot the enregy resolution as a function of the true_energy

    Parameters
    ----------
    true_energy: `numpy.ndarray`
    reco_energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    bias_correction: `bool`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    assert len(true_energy) == len(reco_energy), "simulated and reconstructured true_energy arrrays should have the same length"

    ax = plt.gca() if ax is None else ax

    try:
        E_bin, Eres = ana.energy_resolution_per_energy(true_energy, reco_energy,
                                                       percentile=percentile,
                                                       confidence_level=confidence_level,
                                                       bias_correction=bias_correction,
                                                       )
    except Exception as e:
        print('Energy resolution ', e)
    else:
        E = ana.logbin_mean(E_bin)

        if 'fmt' not in kwargs:
            kwargs['fmt'] = 'o'

        ax.set_ylabel(r"$(\Delta E/E)_{68}$")
        ax.set_xlabel(r'$E_R$ [TeV]')
        ax.set_xscale('log')
        ax.set_title('Energy resolution')

        ax.errorbar(E, Eres[:, 0], xerr=(E - E_bin[:-1], E_bin[1:] - E),
                    yerr=(Eres[:, 0] - Eres[:, 1], Eres[:, 2] - Eres[:, 0]), **kwargs)

        ax.grid('on', which='both')
    finally:
        return ax


def plot_energy_resolution_cta_requirement(cta_site, ax=None, **kwargs):
    """
    Plot the cta requirement for the true_energy resolution

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
    ax.set_xlabel(r'$E_R$ [TeV]')
    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    ax.grid('on', which='both')
    ax.legend()
    return ax


def plot_energy_resolution_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the cta performances (June 2018) for the true_energy resolution

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
    ax.set_xlabel(r'$E_R$ [TeV]')
    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    ax.grid('on', which='both')
    ax.legend()
    return ax


def plot_impact_parameter_error_site_center(reco_x, reco_y, true_x, true_y, ax=None, **kwargs):
    """
    Plot the impact parameter error as a function of the distance to the site center.

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    true_x: `numpy.ndarray`
    true_y: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplotlib.pyplot.hist2d`

    Returns
    -------
    ax
    """

    ax = plt.gca() if ax is None else ax

    imp_err = ana.impact_parameter_error(reco_x, reco_y, true_x, true_y)
    distance_center = np.sqrt(true_x ** 2 + true_y ** 2)

    ax.hist2d(distance_center, imp_err, **kwargs)
    ax.set_xlabel("Distance to site center")
    ax.set_ylabel("Impact point error")
    ax.grid('on', which='both')
    return ax


def plot_impact_resolution_per_energy(reco_x, reco_y, true_x, true_y, true_energy,
                                      percentile=68.27, confidence_level=0.95, bias_correction=False,
                                      ax=None, **kwargs):
    """
    Plot the angular resolution as a function of the true_energy

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    true_x: float
    true_y: float
    true_energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    try:
        E_bin, RES = ana.impact_resolution_per_energy(reco_x, reco_y, true_x, true_y, true_energy,
                                                      percentile=percentile,
                                                      confidence_level=confidence_level,
                                                      bias_correction=bias_correction,
                                                      )
    except Exception as e:
        print('Impact resolution ', e)
    else:
        E = ana.logbin_mean(E_bin)

        if 'fmt' not in kwargs:
            kwargs['fmt'] = 'o'
        ax.set_ylabel('Impact Resolution [m]')
        ax.set_xlabel('Energy [TeV]')
        ax.set_xscale('log')
        ax.set_title('Impact resolution')

        ax.errorbar(
            E, RES[:, 0],
            xerr=(E - E_bin[:-1], E_bin[1:] - E),
            yerr=(RES[:, 0] - RES[:, 1], RES[:, 2] - RES[:, 0]),
            **kwargs,
            )
        ax.grid('on', which='both')
    finally:
        return ax


def plot_migration_matrix(x, y, ax=None, colorbar=False, xy_line=False, hist2d_args={}, line_args={}):
    """
    Make a simple plot of a migration matrix

    Parameters
    ----------
    x: list or `numpy.ndarray`
    y: list or `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    colorbar: `bool`
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


def plot_dispersion(true_x, reco_x, x_log=False, ax=None, **kwargs):
    """
    Plot the dispersion around an expected value X_true: `(true_x-reco_x)` as a function of `true_x`

    Parameters
    ----------
    true_x: `numpy.ndarray`
        true value of a variable x
    reco_x: `numpy.ndarray`
        reconstructed value of a variable x
    x_log: bool
        if True, the dispersion is plotted as a function of `log10(true_x)`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.hist2d`

    Returns
    -------
    ax: `maptlotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    if not 'bins' in kwargs:
        kwargs['bins'] = 50

    x = np.log10(true_x) if x_log else true_x

    ax.hist2d(x, true_x - reco_x, **kwargs)
    return ax


def plot_feature_importance(feature_keys, feature_importances, ax=None, **kwargs):
    """
    Plot features importance after model training (typically from scikit-learn)

    Parameters
    ----------
    feature_keys: list of string
    feature_importances: `numpy.ndarray` or list
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplotlib.pyplot.bar`

    Returns
    -------
    ax
    """
    ax = plt.gca() if ax is None else ax

    sort_mask = np.argsort(feature_importances)[::-1]
    ax.bar(np.array(feature_keys)[sort_mask], np.array(feature_importances)[sort_mask], **kwargs)
    for t in ax.get_xticklabels():
        t.set_rotation(45)
    ax.set_title("Features importance")

    return ax


def plot_binned_stat(x, y, statistic='mean', bins=20, errorbar=False, percentile=68.27, line=True,
                     ax=None, **kwargs):
    """
    Plot statistics on the quantity y binned following the quantity x.
    The statistic can be given by a string ('mean', 'sum', 'max'...) or a function. See `scipy.stats.binned_statistic`.
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
    >>> plot_binned_stat(x, y, statistic='median', bins=40, percentile=95, line=False, color='red', errorbar=True, s=0)
    """

    ax = plt.gca() if ax is None else ax

    bin_stat, bin_edges, binnumber = binned_statistic(x, y, statistic=statistic, bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    bin_with_data = np.setdiff1d(binnumber, len(bin_edges)) - 1

    xx = bin_centers[bin_with_data]
    yy = bin_stat[bin_with_data]

    if line:
        sc = ax.plot(xx, yy, **kwargs)

        if errorbar:
            err = np.array([np.percentile(np.abs(y[binnumber == i + 1] - bin_stat[i]), percentile)
                            for i in bin_with_data])
            yy_h = yy + err
            yy_l = yy - err

            err_kwargs = dict(alpha=0.2, color=sc[0].get_color())
            ax.fill_between(xx, yy_l, yy_h, **err_kwargs)

    else:
        sc = ax.scatter(xx, yy, **kwargs)

        if errorbar:
            err = np.array([np.percentile(np.abs(y[binnumber == i + 1] - bin_stat[i]), percentile)
                            for i in bin_with_data])
            yy_h = yy + err
            yy_l = yy - err

            err_kwargs = dict(color=sc.get_facecolors()[0].tolist())

            ax.hlines(bin_stat, bin_edges[:-1], bin_edges[1:], **err_kwargs)
            ax.vlines(xx, yy_l, yy_h, **err_kwargs)

    return ax


def plot_effective_area_per_energy_power_law(emin, emax, total_number_events, spectral_index,
                                             true_energy, simu_area, ax=None, **kwargs):
    """
    Plot the effective area as a function of the true energy.
    The effective area is computed using the `ctaplot.ana.effective_area_per_energy_power_law`.

    Parameters
    ----------
    emin: float
        min simulated reco_energy
    emax: float
        max simulated reco_energy
    total_number_events: int
        total number of simulated events
    spectral_index: float
        spectral index of the simulated power-law
    true_energy: `numpy.ndarray`
        true energies of the reconstructed events
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
    ax.set_xlabel(r'$E_T$ [TeV]')
    ax.set_ylabel(r'Effective Area $[m^2]$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ebin, seff = ana.effective_area_per_energy_power_law(emin, emax, total_number_events,
                                                         spectral_index, true_energy, simu_area)

    energy_nodes = ana.logbin_mean(ebin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'
    ax.errorbar(energy_nodes, seff, xerr=(ebin[1:] - ebin[:-1]) / 2., **kwargs)
    ax.grid('on', which='both')
    return ax


def plot_angular_resolution_per_off_pointing_angle(true_alt, true_az, reco_alt, reco_az,
                                                   alt_pointing, az_pointing, res_degree=False, bins=10, ax=None,
                                                   **kwargs):
    """
    Plot the angular resolution as a function of the angular separation between events true position and the
    pointing direction. Angles must be given in radians.


    Parameters
    ----------
    true_alt: `numpy.ndarray`
    true_az: `numpy.ndarray`
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
    res_bins, res = ana.angular_resolution_per_off_pointing_angle(true_alt, true_az, reco_alt, reco_az,
                                                                  alt_pointing, az_pointing, bins=bins)
    res_unit = 'rad'
    if res_degree:
        res = np.rad2deg(res)
        res_unit = 'deg'

    ax = plot_resolution(res_bins, res, ax=ax, **kwargs)
    ax.set_xlabel("Angular separation to pointing direction [rad]")
    ax.set_ylabel("Angular resolution [{}]".format(res_unit))
    ax.grid('on', which='both')
    return ax


def plot_impact_parameter_error_per_bin(x, reco_x, reco_y, true_x, true_y, bins=10, ax=None, **kwargs):
    """
    Plot the impact parameter error per bin

    Parameters
    ----------
    x: `numpy.ndarray`
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    true_x: `numpy.ndarray`
    true_y: `numpy.ndarray`
    bins: arg for `np.histogram`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `plot_resolution`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    bin, res = ana.distance_per_bin(x, reco_x, reco_y, true_x, true_y)
    ax = plot_resolution(bin, res, bins=bins, ax=ax, **kwargs)

    return ax


def plot_binned_bias(simu, reco, x, relative_scaling_method=None, ax=None, bins=10, log=False, **kwargs):
    """
    Plot the bias between `true` and `reco` as a function of bins of `x`

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
        "true and reco arrays should have the same length"
    assert len(simu) == len(x), \
        "true and true_energy arrays should have the same length"

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
    Plot the bias per bins of true_energy

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
        "true and reco arrays should have the same length"
    assert len(simu) == len(energy), \
        "true and true_energy arrays should have the same length"

    ax = plt.gca() if ax is None else ax

    bins, bias = ana.bias_per_energy(simu, reco, energy, relative_scaling_method=relative_scaling_method)
    mean_bins = ana.logbin_mean(bins)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel("bias")
    ax.set_xlabel("log(E/TeV)")
    ax.set_xscale('log')

    ax.errorbar(mean_bins, bias, xerr=(mean_bins - bins[:-1], bins[1:] - mean_bins), **kwargs)
    ax.grid('on', which='both')
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


def plot_roc_curve(true_type, reco_proba,
                   pos_label=None, sample_weight=None, drop_intermediate=True,
                   ax=None, **kwargs):
    """

    Parameters
    ----------
    true_type: `numpy.ndarray`
        true labels: must contain only two labels of type int, float or str
    reco_proba: `numpy.ndarray`
        reconstruction probability, values must be between 0 and 1
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
    ax: `matplotlib.pyplot.axis`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    auc_score = metrics.roc_auc_score(true_type, reco_proba)
    if auc_score < 0.5:
        auc_score = 1 - auc_score

    if 'label' not in kwargs:
        kwargs['label'] = "auc score = {:.3f}".format(auc_score)

    fpr, tpr, thresholds = metrics.roc_curve(true_type,
                                             reco_proba,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight,
                                             drop_intermediate=drop_intermediate,
                                             )

    ax.plot(fpr, tpr, **kwargs)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    ax.plot([0, 1], [0, 1], '--', color='black')
    ax.axis('equal')
    ax.legend(loc=4)
    ax.grid('on')
    return ax


def plot_roc_curve_multiclass(true_type, reco_proba,
                              pos_label=None,
                              sample_weight=None, drop_intermediate=True,
                              ax=None, **kwargs):
    """
    Plot a ROC curve for a multiclass classification.

    Parameters
    ----------
    true_type: `numpy.ndarray`
        true labels: int, float or str
    reco_proba: `dict` of `numpy.ndarray` of shape `(len(true_type), len(set(true_type))`
        reconstruction probability for each class in `true_type`, values must be between 0 and 1
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, the ROC curve of each class is ploted.
        If `pos_label` is not None, only the ROC curve of this class is ploted.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
    ax: `matplotlib.pyplot.axis`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    label_binarizer = LabelBinarizer()
    binarized_classes = label_binarizer.fit_transform(true_type)

    if pos_label is not None:
        if pos_label not in set(true_type) or pos_label not in reco_proba:
            raise ValueError(f"true_type and reco_proba must contain pos_label {pos_label}")
        ii = np.where(label_binarizer.classes_ == pos_label)[0][0]

        auc_score = metrics.roc_auc_score(binarized_classes[:, ii], reco_proba[pos_label])
        kwargs['label'] = "class {} - auc = {:.3f}".format(pos_label, auc_score)
        ax = plot_roc_curve(binarized_classes[:, ii],
                            reco_proba[pos_label],
                            pos_label=1,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate,
                            ax=ax,
                            **kwargs,
                            )

    else:
        for st in set(true_type):
            if st not in reco_proba:
                raise ValueError("the class {} is not in reco_proba".format(st))

        for ii, cls in enumerate(label_binarizer.classes_):
            rp = reco_proba[cls]
            auc_score = metrics.roc_auc_score(binarized_classes[:, ii], rp)

            kwargs['label'] = "class {} - auc = {:.3f}".format(cls, auc_score)
            ax = plot_roc_curve(binarized_classes[:, ii],
                                rp,
                                sample_weight=sample_weight,
                                drop_intermediate=drop_intermediate,
                                ax=ax,
                                **kwargs,
                                )

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')

    ax.plot([0, 1], [0, 1], '--', color='black')
    ax.legend(loc=4)
    ax.axis('equal')
    ax.grid('on')
    return ax


def plot_roc_curve_gammaness(true_type, gammaness,
                             gamma_label=0,
                             sample_weight=None,
                             drop_intermediate=True,
                             ax=None, **kwargs):
    """

    Parameters
    ----------
    true_type: `numpy.ndarray`
        true labels: int, float or str
    gammaness: `numpy.ndarray`
        probability of each event to be a gamma, values must be between 0 and 1
    gamma_label: the label of the gamma class in `true_type`.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
    ax: `matplotlib.pyplot.axis`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """

    ax = plt.gca() if ax is None else ax

    if len(set(true_type)) == 2:
        ax = plot_roc_curve(true_type, gammaness,
                            pos_label=gamma_label,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate,
                            ax=ax,
                            **kwargs,
                            )
    else:
        ax = plot_roc_curve_multiclass(true_type, {gamma_label: gammaness},
                                       pos_label=gamma_label,
                                       sample_weight=sample_weight,
                                       drop_intermediate=drop_intermediate,
                                       ax=ax,
                                       **kwargs
                                       )

    ax.set_title("gamma ROC curve")
    ax.set_xlabel("gamma false positive rate")
    ax.set_ylabel("gamma true positive rate")
    ax.grid('on')
    return ax


def plot_roc_curve_gammaness_per_energy(true_type, gammaness, true_energy, gamma_label=0, energy_bins=None,
                                        ax=None,
                                        **kwargs):
    """
    Plot a gamma ROC curve per gamma true_energy bin.

    Parameters
    ----------
    true_type: `numpy.ndarray`
        true labels: int, float or str
    gammaness: `numpy.ndarray`
        probability of each event to be a gamma, values must be between 0 and 1
    true_energy: `numpy.ndarray`
        true_energy of the gamma events in TeV
        true_energy.shape == true_type.shape (but energies for events that are not gammas are not considered)
    gamma_label: the label of the gamma class in `true_type`.
    energy_bins: None or int or `numpy.ndarray`
        bins in true_energy.
        If `bins` is None, the default binning given by `ctaplot.ana.irf_cta().E_bin` if used.
        If `bins` is an int, it defines the number of equal-width
        bins in the given range.
        If `bins` is a sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
    ax: `matplotlib.pyplot.axis`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    gammas = true_type == gamma_label
    non_gammas = true_type != gamma_label
    gamma_energy = true_energy[gammas]
    binarized_label = (true_type == gamma_label).astype(int)  # binarize in a gamma vs all fashion

    if energy_bins is None:
        irf = ana.irf_cta()
        energy_bins = irf.E_bin
    elif type(energy_bins) is int:
        energy_bins = np.logspace(np.log10(gamma_energy).min(), np.log10(gamma_energy).max(), energy_bins + 1)

    bin_index = np.digitize(gamma_energy, energy_bins)

    counter = 0
    if 'label' in kwargs:
        kwargs.remove('label')

    for ii in np.arange(1, len(energy_bins)):

        mask = bin_index == ii
        e = gamma_energy[mask]

        if len(e) > 0:
            masked_types = np.concatenate([binarized_label[non_gammas], binarized_label[gammas][mask]])
            masked_gammaness = np.concatenate([gammaness[non_gammas], gammaness[gammas][mask]])

            ax = plot_roc_curve_gammaness(masked_types, masked_gammaness, gamma_label=1, ax=ax, **kwargs)

            children = ax.get_children()[counter]
            label = "[{:.2f}:{:.2f}]TeV - ".format(energy_bins[ii - 1], energy_bins[ii]) + children.get_label()
            children.set_label(label)
            counter += 2

    ax.legend(loc=4)
    ax.grid('on')
    return ax


def plot_any_resource(filename, columns_xy=[0, 1], ax=None, **kwargs):
    """
    Naive plot of any resource text file that present data organised in a table after n lines of comments

    Parameters
    ----------
    filename: path
    columns_xy: list [x,y] : index of the data columns to plot
    ax: `matplotlib.pyplot.axis` or None
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`

    """

    ax = plt.gca() if ax is None else ax

    data = load_any_resource(filename)

    if 'label' not in kwargs:
        kwargs['label'] = filename
    ax.plot(data[columns_xy[0]], data[columns_xy[1]], **kwargs)

    return ax


def plot_gammaness_distribution(mc_type, gammaness, ax=None, **kwargs):
    """
    Plot the distribution of gammaness based on `mc_type`

    Parameters
    ----------
    mc_type: `numpy.ndarray`
        true labeling
    gammaness: `numpy.ndarray`
        reconstructed gammaness
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.hist`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    ax = plt.gca() if ax is None else ax

    if 'histtype' not in kwargs:
        kwargs['histtype'] = 'step'
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 3

    is_label = 'label' in kwargs

    for particle in set(mc_type):
        if not is_label:
            kwargs['label'] = particle
        ax.hist(gammaness[mc_type == particle], **kwargs)

    ax.set_title('Gammaness distribution per particle type')
    ax.set_xlabel('gammaness')
    ax.legend()
    return ax


def plot_sensitivity_magic_performance(key='lima_5off', ax=None, **kwargs):
    """
    Plot the  MAGIC sensitivity from Aleksić, Jelena, et al. 2016, DOI: 10.1016/j.astropartphys.2015.02.005

    Parameters
    ----------
    key: string
        'lima_1off': LiMa 1 off position
        'lima_3off': LiMa 3 off positions
        'lima_5off': LiMa 5 off positions
        'snr': Nex/sqrt(Nbkg)
    ax: `matplotlib.pyplot.axis`
    kwargs: kwargs for `matplotlib.pyplot.errorbar`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """

    ax = plt.gca() if ax is None else ax

    magic_table = ana.get_magic_sensitivity()

    magic_table['e_err_lo'] = magic_table['e_center'] - magic_table['e_min']
    magic_table['e_err_hi'] = magic_table['e_max'] - magic_table['e_center']

    if 'ls' not in kwargs and 'linestyle' not in kwargs:
        kwargs['ls'] = ''
    kwargs.setdefault('label', f'MAGIC {key} (Aleksić et al, 2016)')

    k = 'sensitivity_' + key
    ax.errorbar(
        magic_table['e_center'].to_value(u.TeV),
        y=(magic_table['e_center'] ** 2 * magic_table[k]).to_value(u.Unit('erg cm-2 s-1')),
        xerr=[magic_table['e_err_lo'].to_value(u.TeV), magic_table['e_err_hi'].to_value(u.TeV)],
        yerr=(magic_table['e_center'] ** 2 * magic_table[f'{k}_err']).to_value(u.Unit('erg cm-2 s-1')),
        **kwargs
    )

    ax.set_xlabel(r'$E_R$ [TeV]')
    ax.set_ylabel(r'Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.legend()

    return ax


def plot_rate(e_min, e_max, rate, rate_err=None, ax=None, **kwargs):
    """
    Plot the background rate [Hz] as a function of the energy [TeV]

    Parameters
    ----------
    e_min: `numpy.ndarray`
        Reconstructed energy in TeV
    e_max: `numpy.ndarray`
        Reconstructed energy in TeV
    background_rate: `numpy.ndarray`
        Background rate in Hz
    ax: `matplotlib.pyplot.axis`
    kwargs: kwargs for  `matplotlib.pyplot.errobar`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    e_center = np.sqrt(e_min * e_max)

    ax.errorbar(e_center, rate, xerr=[e_center-e_min, e_max-e_center], yerr=rate_err, **kwargs)

    ax.set_xlabel(r"$E_\mathrm{Reco} [\mathrm{TeV}]$")
    ax.set_ylabel("Event rate [Hz]")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.legend()

    return ax


def plot_background_rate(e_min, e_max, background_rate, background_rate_err=None, ax=None, **kwargs):
    """
    Plot the background rate [Hz] as a function of the energy [TeV]

    Parameters
    ----------
    e_min: `numpy.ndarray`
        Reconstructed energy in TeV
    e_max: `numpy.ndarray`
        Reconstructed energy in TeV
    background_rate: `numpy.ndarray`
        Background rate in Hz
    ax: `matplotlib.pyplot.axis`
    kwargs: kwargs for  `matplotlib.pyplot.errobar`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """

    ax = plot_rate(e_min, e_max, background_rate, rate_err=background_rate_err, ax=ax, **kwargs)
    ax.set_ylabel("Background rate [Hz]")

    return ax


def plot_gamma_rate(e_min, e_max, gamma_rate, gamma_rate_err=None, ax=None, **kwargs):
    """
    Plot the gamma rate [Hz] as a function of the energy [TeV]

    Parameters
    ----------
    e_min: `numpy.ndarray`
        Reconstructed energy in TeV
    e_max: `numpy.ndarray`
        Reconstructed energy in TeV
    gamma_rate: `numpy.ndarray`
        gamma rate in Hz
    ax: `matplotlib.pyplot.axis`
    kwargs: kwargs for  `matplotlib.pyplot.errobar`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """

    ax = plot_rate(e_min, e_max, gamma_rate, rate_err=gamma_rate_err, ax=ax, **kwargs)
    ax.set_ylabel("Gamma rate [Hz]")

    return ax



def plot_background_rate_magic(ax=None, **kwargs):
    """
    Plot the  MAGIC sensitivity from Aleksić, Jelena, et al. 2016, DOI: 10.1016/j.astropartphys.2015.02.005

    Returns
    -------

    """

    magic_table = ana.get_magic_sensitivity()

    kwargs.setdefault('label', 'MAGIC (Aleksić et al, 2016)')

    ax = plot_background_rate(magic_table['e_min'].to_value(u.TeV),
                              magic_table['e_max'].to_value(u.TeV),
                              magic_table['background_rate'].to_value(u.Hz),
                              magic_table['background_rate_err'].to_value(u.Hz),
                              ax=ax,
                              **kwargs
                              )

    return ax


def plot_gamma_rate_magic(ax=None, **kwargs):
    """
    Plot the  MAGIC sensitivity from Aleksić, Jelena, et al. 2016, DOI: 10.1016/j.astropartphys.2015.02.005

    Returns
    -------

    """

    magic_table = ana.get_magic_sensitivity()

    kwargs.setdefault('label', 'MAGIC (Aleksić et al, 2016)')

    ax = plot_gamma_rate(magic_table['e_min'].to_value(u.TeV),
                         magic_table['e_max'].to_value(u.TeV),
                         magic_table['gamma_rate'].to_value(u.Hz),
                         magic_table['gamma_rate_err'].to_value(u.Hz),
                         ax=ax,
                         **kwargs
                         )

    return ax
