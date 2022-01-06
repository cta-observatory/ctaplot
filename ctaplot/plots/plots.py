"""
plots.py
========
Functions to make IRF and other reconstruction quality-check plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
from sklearn import metrics
from sklearn.multiclass import LabelBinarizer
import astropy.units as u
from astropy.visualization import quantity_support
from matplotlib.ticker import FormatStrFormatter
from ..ana import ana
from ..io.dataset import load_any_resource
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import recall_score, precision_score

__all__ = ['plot_resolution',
           'plot_resolution_difference',
           'plot_energy_resolution',
           'plot_binned_bias',
           'plot_energy_bias',
           'plot_impact_parameter_resolution_per_bin',
           'plot_layout_map',
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
           'plot_gammaness_threshold_efficiency',
           'plot_precision_recall',
           'plot_roc_auc_per_energy',
           ]


@u.quantity_input(true_energy=u.TeV, reco_energy=u.TeV)
def plot_energy_distribution(true_energy, reco_energy, bins=10, ax=None, outfile=None, mask_mc_detected=True):
    """
    Plot the true_energy distribution of the simulated particles, detected particles and reconstructed particles
    The plot might be saved automatically if `outfile` is provided.

    Parameters
    ----------
    true_energy: `astropy.Quantity`
        array of simulated energy
    reco_energy: `astropy.Quantity`
        array of reconstructed energy
    bins: int or `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    outfile: string
        output file path
    mask_mc_detected: `numpy.ndarray`
        mask of detected particles for the SimuE array
        if True (default), no mask is applied
    """

    ax = plt.gca() if ax is None else ax

    ax.set_xlabel(f'Energy {true_energy.unit.to_string("latex")}')
    ax.set_ylabel('Count')

    ax.set_xscale('log')

    if isinstance(bins, u.Quantity):
        bins = bins.to_value(true_energy.unit)

    with quantity_support():
        _, bins, _ = ax.hist(true_energy, log=True, bins=bins, label="Simulated")
        ax.hist(true_energy[mask_mc_detected], log=True, bins=bins, label="Detected")
        ax.hist(reco_energy, log=True, bins=bins, label="Reconstructed")

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


@u.quantity_input(energy=u.TeV)
def plot_multiplicity_per_energy(energy, multiplicity, bins=10, ax=None, outfile=None, **kwargs):
    """
    Plot the telescope multiplicity as a function of the true_energy
    The plot might be saved automatically if `outfile` is provided.

    Parameters
    ----------
    multiplicity: `numpy.ndarray`
        telescope multiplcity
    energy: `numpy.ndarray`
        event energies
    ax: `matplotlib.pyplot.axes`
    outfile: string
        path to the output file to save the figure
    """

    ax = plt.gca() if ax is None else ax

    if not len(multiplicity) == len(energy) > 0:
        raise ValueError("arrays should have same length > 0")

    if isinstance(bins, int):
        bins = np.geomspace(energy.min(), energy.max(), bins)

    if isinstance(bins, u.Quantity):
        bins = bins.to_value(energy.unit)

    kwargs.setdefault('marker', 'o')

    with quantity_support():
        ax = plot_binned_stat(energy.value, multiplicity, bins=bins, errorbar=True, ax=ax, **kwargs)

        if 'ls' not in kwargs and 'linestyle' not in kwargs:
            kwargs.setdefault('ls', '--')
        kwargs.setdefault('alpha', 0.5)
        kwargs['label'] = 'min multiplicity'
        plot_binned_stat(energy.value, multiplicity, bins=bins, statistic='min', ax=ax, **kwargs)
        kwargs['label'] = 'max multiplicity'
        plot_binned_stat(energy.value, multiplicity, bins=bins, statistic='max', ax=ax, **kwargs)
        ax.set_xscale('log')

    ax.set_xlabel(f'Energy [{energy.unit}]')
    ax.set_ylabel('Multiplicity')

    if isinstance(outfile, str):
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


@u.quantity_input(reco_alt=u.rad, reco_az=u.rad, source_alt=u.rad, source_az=u.rad)
def scatter_events_field_of_view(reco_alt, reco_az, source_alt, source_az, color_scale=None, ax=None):
    """
    Plot a map in angles [in degrees] of the photons seen by the telescope (after reconstruction)

    Parameters
    ----------
    reco_alt: `astropy.Quantity`
        array of reconstructed altitudes
    reco_az: `astropy.Quantity`
        array of reconstructed azimuths
    source_alt: `astropy.Quantity`
        single altitude of the source
    source_az: `astropy.Quantity`
        single azimuth of the source
    color_scale: `numpy.ndarray`
        if given, set the colorbar
    ax: `matplotlib.pyplot.axes`
    outfile: string
        path to the output figure file. if None, the plot is not saved

    Returns
    -------
    ax: `matplitlib.pyplot.axes`
    """
    dx = 1 * u.deg

    ax = plt.gca() if ax is None else ax

    ax.set_xlim(source_az.to(u.deg) - dx, source_az.to(u.deg) + dx)
    ax.set_ylim(source_alt.to(u.deg) - dx, source_alt.to(u.deg) + dx)

    ax.set_xlabel("Az [deg]")
    ax.set_ylabel("Alt [deg]")

    ax.axis('equal')

    if color_scale is not None:
        c = color_scale
        plt.colorbar()
    else:
        c = 'blue'

    with quantity_support():
        ax.scatter(reco_az, reco_alt, c=c)
        ax.scatter(source_az, source_alt, marker='+', linewidths=3, s=200, c='orange', label="Source position")

    ax.legend()

    return ax


@u.quantity_input(true_alt=u.rad, reco_alt=u.rad, true_az=u.rad, reco_az=u.rad)
def plot_theta2(true_alt, reco_alt, true_az, reco_az, bias_correction=False, ax=None, **kwargs):
    """
    Plot the theta2 distribution and display the corresponding angular resolution in degrees.
    The input must be given in radians.

    Parameters
    ----------
    reco_alt: `astropy.Quantity`
        reconstructed altitude angle in radians
    reco_az: `astropy.Quantity`
        reconstructed azimuth angle in radians
    true_alt: `astropy.Quantity`
        true altitude angle in radians
    true_az: `astropy.Quantity`
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

    theta2 = ana.theta2(true_alt, reco_alt, true_az, reco_az).to(u.deg ** 2)
    ang_res = ana.angular_resolution(true_alt, reco_alt, true_az, reco_az).to(u.deg)

    ax.set_xlabel(r'$\theta^2 [deg^2]$')
    ax.set_ylabel('Count')

    with quantity_support():
        ax.hist(theta2, **kwargs)

    err_max = (ang_res[2] - ang_res[0])
    err_min = (ang_res[0] - ang_res[1])
    ax.set_title(rf'angular resolution: {ang_res[0].value:.3f}(+{err_max.value:.1e}/-{err_min.value:.1e})deg')

    return ax


@u.quantity_input(reco_x=u.m, reco_y=u.m)
def plot_impact_point_heatmap(reco_x, reco_y, ax=None, outfile=None, **kwargs):
    """
    Plot the heatmap of the impact points on the site ground and save it under outfile

    Parameters
    ----------
    reco_x: `astropy.Quantity`
        reconstructed x positions
    reco_y: `astropy.Quantity`
        reconstructed y positions
    ax: `matplotlib.pyplot.axes`
    outfile: string
        path to the output file. If None, the figure is not saved.
    """

    ax = plt.gca() if ax is None else ax

    unit = reco_x.unit

    kwargs.setdefault('norm', LogNorm())
    kwargs.setdefault('cmap', plt.cm.get_cmap('PuBu'))
    kwargs.setdefault('bins', 50)
    h = ax.hist2d(reco_x.to_value(unit), reco_y.to_value(unit), **kwargs)
    cb = plt.colorbar(h[3], ax=ax)
    cb.set_label('Event count')

    ax.set_xlabel(f"X [{unit.to_string('latex')}]")
    ax.set_ylabel(f"Y [{unit.to_string('latex')}]")
    ax.axis('equal')

    if isinstance(outfile, str):
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

    kwargs.setdefault('label', 'Telescope multiplicity')

    n, bins, patches = ax.hist(multiplicity, bins=(xmax - xmin), range=(xmin, xmax), rwidth=0.7, align='left', **kwargs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    x50 = m[int(np.floor(0.5 * len(m)))] + 0.5
    x90 = m[int(np.floor(0.9 * len(m)))] + 0.5
    if quartils and (xmin < x50 < xmax):
        ax.vlines(x50, 0, n[int(m[int(np.floor(0.5 * len(m)))])], label='50%')
    if quartils and (xmin < x90 < xmax):
        ax.vlines(x90, 0, n[int(m[int(np.floor(0.9 * len(m)))])], label='90%')

    ax.set_title("Telescope multiplicity")
    ax.grid(True)

    if isinstance(outfile, str):
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

    ax.set_ylabel(r'res')

    if not log:
        x = (bins[:-1] + bins[1:]) / 2.
    else:
        x = ana.logbin_mean(bins)
        ax.set_xscale('log')

    kwargs.setdefault('fmt', 'o')

    ax.errorbar(x, res[:, 0], xerr=[x - bins[:-1], bins[1:] - x],
                yerr=(res[:, 0] - res[:, 1], res[:, 2] - res[:, 0]), **kwargs)

    ax.set_title('Resolution')
    return ax


@u.quantity_input(true_energy=u.eV, reco_energy=u.eV, simulated_area=u.m ** 2)
def plot_effective_area_per_energy(true_energy, reco_energy, simulated_area, ax=None, bins=None, **kwargs):
    """
    Plot the effective area as a function of the true true_energy

    Parameters
    ----------
    true_energy: `astropy.Quantity`
        all simulated event energy
    reco_energy: `astropy.Quantity`
        all reconstructed event energy
    simulated_area: `astropy.Quantity`
    ax: `matplotlib.pyplot.axes`
    bins: `numpy.ndarray`
    kwargs: options for `maplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`

    Example
    -------
    >>> import numpy as np
    >>> import ctaplot
    >>> irf = ctaplot.ana.irf_cta()
    >>> true_e = 10**(-2 + 4*np.random.rand(1000)) * u.TeV
    >>> reco_e = 10**(-2 + 4*np.random.rand(100)) * u.TeV
    >>> ax = ctaplot.plots.plot_effective_area_per_energy(true_e, reco_e, irf.LaPalmaArea_prod3)
    """

    ax = plt.gca() if ax is None else ax

    e_bin, seff = ana.effective_area_per_energy(true_energy, reco_energy, simulated_area, bins=bins)
    E = ana.logbin_mean(e_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    with quantity_support():
        ax.errorbar(E, seff, xerr=(e_bin[1:] - e_bin[:-1]) / 2., **kwargs)

    ax.set_xlabel(rf'$E_T$ [{E.unit}]')
    ax.set_ylabel(f'Effective Area [{seff.unit}]')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(True, which='both')
    return ax


def plot_effective_area_cta_requirement(cta_site, ax=None, **kwargs):
    """
    Plot the CTA requirement for the effective area as a function of the true true_energy

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

    kwargs.setdefault('label', "CTA requirement {}".format(cta_site))

    with quantity_support():
        ax.plot(e_cta, ef_cta, **kwargs)
    ax.grid(True, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(rf'$E_T$ [{e_cta.unit}]')
    ax.set_ylabel(f'Effective Area [{ef_cta.unit}]')
    ax.legend()
    return ax


def plot_effective_area_cta_performance(cta_site, ax=None, **kwargs):
    """
    Plot the CTA performances for the effective area as a function of the true true_energy

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

    cta_perf = ana.cta_performance(cta_site)
    e_cta, ef_cta = cta_perf.get_effective_area()

    kwargs.setdefault('label', f'CTA performance {cta_site}')

    with quantity_support():
        ax.plot(e_cta, ef_cta, **kwargs)

    ax.grid(True, which='both')
    ax.set_xlabel(rf"$E_T$ [{e_cta.unit.to_string('latex')}]")
    ax.set_ylabel(f"Effective Area [{ef_cta.unit.to_string('latex')}]")
    ax.set_xscale('log')
    ax.set_yscale('log')
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

    kwargs.setdefault('label', "CTA requirement {}".format(cta_site))

    with quantity_support():
        ax.plot(e_cta, ef_cta, **kwargs)
    ax.grid(True, which='both')
    ax.set_xlabel(rf"$E_R$ [{e_cta.unit.to_string('latex')}]")
    ax.set_ylabel(fr"$energy^2 \cdot$ Flux Sensitivity [{ef_cta.unit.to_string('latex')}]")
    ax.set_xscale('log')
    ax.set_yscale('log')
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
    e_bin = cta_perf.energy_bins

    kwargs.setdefault('label', "CTA performance {}".format(cta_site))

    with quantity_support():
        ax.errorbar(e_cta, ef_cta, xerr=u.Quantity([e_cta - e_bin[:-1], e_bin[1:] - e_cta]), **kwargs)

    ax.grid(True, which='both')
    ax.set_xlabel(rf"$E_R$ [{e_cta.unit.to_string('latex')}]")
    ax.set_ylabel(fr"$energy^2 \cdot$ Flux Sensitivity [{ef_cta.unit.to_string('latex')}]")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    return ax


@u.quantity_input(tel_x=u.m, tel_y=u.m)
def plot_layout_map(tel_x, tel_y, tel_type=None, ax=None, **kwargs):
    """
    Plot the layout map of telescopes positions

    Parameters
    ----------
    tel_x: `astropy.Quantity`
        telescopes x positions
    tel_y: `astropy.Quantity`
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


@u.quantity_input(energy=u.eV)
def plot_resolution_per_energy(true, reco, energy, ax=None, bins=None, **kwargs):
    """
    Plot a variable resolution as a function of the true_energy

    Parameters
    ----------
    reco: `numpy.ndarray`
        reconstructed values of a variable
    true: `numpy.ndarray`
        true values of the variable
    energy: `astropy.Quantity`
        event energy in TeV
    ax: `matplotlib.pyplot.axes`
    bins: `numpy.ndarray`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    ax.set_ylabel(r'res')
    ax.set_xlabel(f'Energy [{energy.unit.to_string("latex")}]')
    ax.set_xscale('log')

    energy_bins, resolution = ana.resolution_per_energy(true, reco, energy, bins=bins)

    E = ana.logbin_mean(energy_bins)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    with quantity_support():
        ax.errorbar(E, resolution[:, 0],
                    xerr=(energy_bins[1:] - energy_bins[:-1]) / 2.,
                    yerr=(resolution[:, 0] - resolution[:, 1], resolution[:, 2] - resolution[:, 0]),
                    **kwargs,
                    )
    ax.grid(True, which='both')
    ax.set_title('Resolution')
    return ax


@u.quantity_input(true_alt=u.rad, reco_alt=u.rad, true_az=u.rad, reco_az=u.rad, true_energy=u.eV)
def plot_angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy,
                                       percentile=68.27, confidence_level=0.95, bias_correction=False,
                                       ax=None, bins=None, **kwargs):
    """
    Plot the angular resolution as a function of the reconstructed true_energy

    Parameters
    ----------
    reco_alt: `astropy.Quantity`
        array of reconstructed altitudes in radians
    reco_az: `astropy.Quantity`
        array of reconstructed azimuths in radians
    true_alt: `astropy.Quantity`
        array of true altitudes in radians
    true_az: `astropy.Quantity`
        array of true azimuths in radians
    reco_energy: `astropy.Quantity`
        array of energy in TeV
    ax: `matplotlib.pyplot.axes`
    bins: `numpy.ndarray`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    try:
        e_bin, res = ana.angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy,
                                                       percentile=percentile,
                                                       confidence_level=confidence_level,
                                                       bias_correction=bias_correction,
                                                       bins=bins
                                                       )
    except Exception as e:
        print('Angular resolution could not be computed', e)
    else:
        # Angular resolution is traditionally presented in degrees
        res = res.to(u.deg)

        energy = ana.logbin_mean(e_bin)

        with quantity_support():
            ax = plot_resolution(e_bin, res, ax=ax, **kwargs)

        ax.set_ylabel('Angular Resolution [deg]')
        ax.set_xlabel(rf'$E_R$ [{energy.unit.to_string("latex")}]')
        ax.set_xscale('log')
        ax.set_title('Angular resolution')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax.grid(True, which='both')
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

    kwargs.setdefault('label', "CTA requirement {}".format(cta_site))

    with quantity_support():
        ax.plot(e_cta, ar_cta, **kwargs)

    ax.set_ylabel(rf'Angular Resolution [{ar_cta.unit.to_string("latex")}]')
    ax.set_xlabel(rf'$E_R$ [{e_cta.unit.to_string("latex")}]')

    ax.set_xscale('log')
    ax.set_title('Angular resolution')
    ax.grid(True, which='both')
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

    kwargs.setdefault('label', "CTA performance {}".format(cta_site))

    ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    ax.set_ylabel(f'Angular resolution [{ar_cta.unit.to_string("latex")}]')
    ax.set_xlabel(rf'$E_R$ [{e_cta.unit.to_string("latex")}]')
    ax.set_title('Angular resolution')
    ax.grid(True, which='both')
    ax.legend()
    return ax


@u.quantity_input(true_x=u.m, reco_x=u.m, true_y=u.m, reco_y=u.m, true_energy=u.TeV)
def plot_impact_parameter_resolution_per_energy(true_x, reco_x, true_y, reco_y, true_energy, ax=None, bins=None,
                                                **kwargs):
    """
    Parameters
    ----------
    true_x: `astropy.Quantity`
    reco_x: `astropy.Quantity`
    true_y: `astropy.Quantity`
    reco_y: `astropy.Quantity`
    true_energy: `astropy.Quantity`
    ax: `matplotlib.pyplot.axes`
    bins: `astropy.Quantity`
    kwargs: args for `ctaplot.plots.plot_resolution`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """

    bin, res = ana.impact_resolution_per_energy(reco_x, reco_y, true_x, true_y, true_energy, bins=bins)
    ax = plot_resolution(bin, res, log=True, ax=ax, **kwargs)
    ax.set_xlabel(fr"$E_T$ [{true_energy.unit.to_string('latex')}]")
    ax.set_ylabel(fr"Impact parameter resolution [{reco_x.unit.to_string('latex')}]")
    ax.set_title("Impact parameter resolution as a function of the true_energy")
    ax.grid('on', which='both')
    return ax


@u.quantity_input(impact_x=u.m, impact_y=u.m, tel_x=u.m, tel_y=u.m)
def plot_impact_map(impact_x, impact_y, tel_x, tel_y, tel_types=None,
                    ax=None,
                    outfile=None,
                    hist_kwargs=None,
                    scatter_kwargs=None,
                    ):
    """
    Map of the site with telescopes positions and impact points heatmap

    Parameters
    ----------
    impact_x: `astropy.Quantity`
    impact_y: `astropy.Quantity`
    tel_x: `astropy.Quantity`
    tel_y: `astropy.Quantity`
    tel_types: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    hist_kwargs: `kwargs` for `matplotlib.pyplot.hist`
    scatter_kwargs: `kwargs` for `matplotlib.pyplot.scatter`
    outfile (optional): string - name of the output file
    """
    ax = plt.gca() if ax is None else ax

    hist_kwargs = {} if hist_kwargs is None else hist_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs

    hist_kwargs.setdefault('bins', 40)
    unit = impact_x.value
    ax.hist2d(impact_x.to_value(unit), impact_y.to_value(unit), **hist_kwargs)
    pcm = ax.get_children()[0]
    plt.colorbar(pcm, ax=ax)

    if len(tel_x) != len(tel_y):
        raise ValueError("tel_x and tel_y should have the same length")

    scatter_kwargs.setdefault('s', 50)

    if tel_types and 'color' not in scatter_kwargs and 'c' not in scatter_kwargs:
        scatter_kwargs['color'] = tel_types
        assert (len(tel_types) == len(tel_x)), "tel_types and tel_x should have the same length"
    else:
        if 'color' not in scatter_kwargs and 'c' not in scatter_kwargs:
            scatter_kwargs['color'] = 'black'
        scatter_kwargs['marker'] = '+' if 'marker' not in scatter_kwargs else scatter_kwargs['marker']
    with quantity_support():
        ax.scatter(tel_x, tel_y, **scatter_kwargs)
    ax.axis('equal')
    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", format='png', dpi=200)

    return ax


@u.quantity_input(true_energy=u.eV, reco_energy=u.eV)
def plot_energy_bias(true_energy, reco_energy, ax=None, bins=None, **kwargs):
    """
    Plot the true_energy bias

    Parameters
    ----------
    true_energy: `astropy.Quantity`
    reco_energy: `astropy.Quantity`
    ax: `matplotlib.pyplot.axes`
    bins: `numpy.ndarray`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    if len(true_energy) != len(reco_energy):
        raise ValueError("simulated and reconstructured true_energy arrrays should have the same length")

    ax = plt.gca() if ax is None else ax

    e_bin, bias_e = ana.energy_bias(true_energy, reco_energy, bins=bins)
    energy_center = ana.logbin_mean(e_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel(r"bias (median($E_{reco}/E_{true}$ - 1)")
    ax.set_xlabel(rf'$E_R$ [{energy_center.unit.to_string("latex")}]')
    ax.set_xscale('log')
    ax.set_title('Energy bias')

    with quantity_support():
        ax.errorbar(energy_center, bias_e, xerr=(energy_center - e_bin[:-1], e_bin[1:] - energy_center), **kwargs)
    ax.grid(True, which='both')

    return ax


@u.quantity_input(true_energy=u.eV, reco_energy=u.eV)
def plot_energy_resolution(true_energy, reco_energy,
                           percentile=68.27, confidence_level=0.95, bias_correction=False,
                           ax=None, bins=None, **kwargs):
    """
    Plot the enregy resolution as a function of the true_energy

    Parameters
    ----------
    true_energy: `astropy.Quantity`
    reco_energy: `astropy.Quantity`
    ax: `matplotlib.pyplot.axes`
    bias_correction: `bool`
    bins: `numpy.ndarray`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    if not len(true_energy) == len(reco_energy) > 0:
        raise ValueError("simulated and reconstructured true_energy arrrays should have the same length > 0")

    ax = plt.gca() if ax is None else ax

    try:
        e_bin, e_res = ana.energy_resolution_per_energy(true_energy, reco_energy,
                                                        percentile=percentile,
                                                        confidence_level=confidence_level,
                                                        bias_correction=bias_correction,
                                                        bins=bins,
                                                        )
    except Exception as e:
        print('Energy resolution ', e)
    else:
        energy_center = ana.logbin_mean(e_bin)

        if 'fmt' not in kwargs:
            kwargs['fmt'] = 'o'

        ax.set_ylabel(r"$(\Delta energy/energy)_{68}$")
        ax.set_xlabel(rf'$E_R$ [{energy_center.unit.to_string("latex")}]')
        ax.set_xscale('log')
        ax.set_title('Energy resolution')

        with quantity_support():
            ax.errorbar(energy_center, e_res[:, 0],
                        xerr=(energy_center - e_bin[:-1], e_bin[1:] - energy_center),
                        yerr=(e_res[:, 0] - e_res[:, 1], e_res[:, 2] - e_res[:, 0]),
                        **kwargs,
                        )

        ax.grid(True, which='both')
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

    kwargs.setdefault('label', "CTA requirement {}".format(cta_site))

    ax.set_ylabel(r"$(\Delta energy/energy)_{68}$")
    ax.set_xlabel(rf'$E_R$ [{e_cta.unit.to_string("latex")}]')

    with quantity_support():
        ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    ax.grid(True, which='both')
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

    kwargs.setdefault('label', "CTA performance {}".format(cta_site))

    ax.set_ylabel(r"$(\Delta energy/energy)_{68}$")
    ax.set_xlabel(rf'$E_R$ [{e_cta.unit.to_string("latex")}]')

    with quantity_support():
        ax.plot(e_cta, ar_cta, **kwargs)
    ax.set_xscale('log')
    ax.grid(True, which='both')
    ax.legend()
    return ax


@u.quantity_input(true_x=u.m, reco_x=u.m, true_y=u.m, reco_y=u.m)
def plot_impact_parameter_error_site_center(true_x, reco_x, true_y, reco_y, ax=None, **kwargs):
    """
    Plot the impact parameter error as a function of the distance to the site center.

    Parameters
    ----------
    reco_x: `astropy.Quantity`
    reco_y: `astropy.Quantity`
    true_x: `astropy.Quantity`
    true_y: `astropy.Quantity`
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplotlib.pyplot.hist2d`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    imp_err = ana.impact_parameter_error(true_x, reco_x, true_y, reco_y)
    distance_center = np.sqrt(true_x ** 2 + true_y ** 2)

    ax.hist2d(distance_center.value, imp_err.value, **kwargs)
    ax.set_xlabel(f"Distance to site center [{distance_center.unit.to_string('latex')}]")
    ax.set_ylabel(f"Impact point error [{imp_err.unit.to_string('latex')}]")
    ax.grid(True, which='both')
    return ax


@u.quantity_input(true_x=u.m, reco_x=u.m, true_y=u.m, reco_y=u.m, true_energy=u.eV)
def plot_impact_resolution_per_energy(true_x, reco_x, true_y, reco_y, true_energy,
                                      percentile=68.27, confidence_level=0.95, bias_correction=False,
                                      ax=None, bins=None, **kwargs):
    """
    Plot the impact resolution as a function of the true_energy

    Parameters
    ----------
    reco_x: `astropy.Quantity`
    reco_y: `astropy.Quantity`
    true_x: `astropy.Quantity`
    true_y: `astropy.Quantity`
    true_energy: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    bins: `numpy.ndarray`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    try:
        e_bin, res = ana.impact_resolution_per_energy(true_x, reco_x, true_y, reco_y, true_energy,
                                                      percentile=percentile,
                                                      confidence_level=confidence_level,
                                                      bias_correction=bias_correction,
                                                      bins=bins
                                                      )
    except Exception as e:
        print('Impact resolution ', e)
    else:
        energy_center = ana.logbin_mean(e_bin)

        if 'fmt' not in kwargs:
            kwargs['fmt'] = 'o'

        with quantity_support():
            plot_resolution(e_bin, res, ax=ax, **kwargs)

        ax.set_ylabel(f'Impact Resolution [{res.unit.to_string("latex")}]')
        ax.set_xlabel(f'Energy [{energy_center.unit.to_string("latex")}]')
        ax.set_xscale('log')
        ax.set_title('Impact resolution')

        ax.grid(True, which='both')
    finally:
        return ax


def plot_migration_matrix(x, y, ax=None, colorbar=False, xy_line=False, hist2d_args=None, line_args=None):
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

    hist2d_args = {} if hist2d_args is None else hist2d_args
    line_args = {} if line_args is None else line_args

    if 'bins_x' not in hist2d_args:
        hist2d_args['bins'] = 50
    if 'color' not in line_args:
        line_args['color'] = 'black'
    if 'lw' not in line_args:
        line_args['lw'] = 0.4

    ax = plt.gca() if ax is None else ax

    with quantity_support():
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

    if isinstance(true_x, u.Quantity) or isinstance(reco_x, u.Quantity):
        raise TypeError("astropy quantities are not supported for that plot yet")

    ax = plt.gca() if ax is None else ax

    kwargs.setdefault('bins', 40)

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

    if isinstance(x, u.Quantity) or isinstance(y, u.Quantity):
        raise TypeError("astropy quantities not supported for this function at the moment")

    ax = plt.gca() if ax is None else ax

    bin_stat, bin_edges, binnumber = binned_statistic(x, y, statistic=statistic, bins=bins)
    bin_width = np.diff(bin_edges)
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


@u.quantity_input(emin=u.eV, emax=u.eV, true_energy=u.eV, simu_area=u.m ** 2)
def plot_effective_area_per_energy_power_law(emin, emax, total_number_events, spectral_index,
                                             true_energy, simu_area, ax=None, bins=None, **kwargs):
    """
    Plot the effective area as a function of the true true_energy.
    The effective area is computed using the `ctaplot.ana.effective_area_per_energy_power_law`.

    Parameters
    ----------
    emin: `astropy.Quantity`
        min simulated true_energy
    emax: `astropy.Quantity`
        max simulated true_energy
    total_number_events: int
        total number of simulated events
    spectral_index: float
        spectral index of the simulated power-law
    true_energy: `astropy.Quantity`
        array of reconstructed events' true energy
    simu_area: `astropy.Quantity`
        simulated core area
    ax: `matplotlib.pyplot.axes`
    bins: `numpy.ndarray`
    kwargs: args for `matplotlib.pyplot.errorbar`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax

    ebin, seff = ana.effective_area_per_energy_power_law(emin, emax, total_number_events,
                                                         spectral_index, true_energy, simu_area, bins=bins)

    energy_nodes = ana.logbin_mean(ebin)

    kwargs.setdefault('fmt', 'o')
    with quantity_support():
        ax.errorbar(energy_nodes, seff, xerr=(ebin[1:] - ebin[:-1]) / 2., **kwargs)

    ax.set_xlabel(rf'$E_T$ [{energy_nodes.unit.to_string("latex")}]')
    ax.set_ylabel(f'Effective Area [{seff.unit.to_string("latex")}]')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(True, which='both')
    return ax


@u.quantity_input(true_alt=u.rad, reco_alt=u.rad, true_az=u.rad, reco_az=u.rad, alt_pointing=u.rad, az_pointing=u.rad)
def plot_angular_resolution_per_off_pointing_angle(true_alt, reco_alt, true_az, reco_az,
                                                   alt_pointing, az_pointing, res_unit=u.deg, bins=10, ax=None,
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
    res_bins, res = ana.angular_resolution_per_off_pointing_angle(true_alt, reco_alt, true_az, reco_az,
                                                                  alt_pointing, az_pointing, bins=bins)
    res = res.to(res_unit)

    with quantity_support():
        ax = plot_resolution(res_bins, res, ax=ax, **kwargs)
    ax.set_xlabel(f"Angular separation to pointing direction [{res_bins.unit.to_string('latex')}]")
    ax.set_ylabel(f"Angular resolution [{res.unit.to_string('latex')}]")
    ax.grid(True, which='both')
    return ax


@u.quantity_input(true_x=u.m, reco_x=u.m, true_y=u.m, reco_y=u.m)
def plot_impact_parameter_resolution_per_bin(x, true_x, reco_x, true_y, reco_y, bins=10, ax=None, **kwargs):
    """
    Plot the impact parameter error per bin

    Parameters
    ----------
    x: `numpy.ndarray`
    reco_x: `astropy.Quantity`
    reco_y: `astropy.Quantity`
    true_x: `astropy.Quantity`
    true_y: `astropy.Quantity`
    bins: arg for `np.histogram`
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `plot_resolution`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    bin, res = ana.distance_per_bin(x, true_x, reco_x, true_y, reco_y)
    with quantity_support():
        ax = plot_resolution(bin, res, bins=bins, ax=ax, **kwargs)

    ax.set_ylabel(f'Impact Resolution [{res.unit.to_string("latex")}]')
    ax.set_title('Impact resolution')

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


@u.quantity_input(energy=u.eV, bins=u.eV)
def plot_bias_per_energy(simu, reco, energy, relative_scaling_method=None, ax=None, bins=None, **kwargs):
    """
    Plot the bias per bins of true_energy

    Parameters
    ----------
    simu: `numpy.ndarray`
    reco: `numpy.ndarray`
    energy: `astropy.Quantity`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`
    ax: `matplotlib.pyplot.axis`
    bins: `astropy.Quantity`
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

    bins, bias = ana.bias_per_energy(simu, reco, energy, relative_scaling_method=relative_scaling_method, bins=bins)
    mean_bins = ana.logbin_mean(bins)

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    ax.set_ylabel("bias")
    ax.set_xlabel(fr"Energy [{mean_bins.unit.to_string('latex')}]")
    ax.set_xscale('log')

    with quantity_support():
        ax.errorbar(mean_bins, bias, xerr=(mean_bins - bins[:-1], bins[1:] - mean_bins), **kwargs)
    ax.grid(True, which='both')
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
    delta_res[:, 1:] = 0  # the condidence intervals have no meaning here
    with quantity_support():
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

    kwargs.setdefault('label', "auc score = {:.3f}".format(auc_score))

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


@u.quantity_input(true_energy=u.eV)
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
    true_energy: `astropy.Quantity`
        true_energy of the gamma events in TeV
        true_energy.shape == true_type.shape (but energy for events that are not gammas are not considered)
    gamma_label: the label of the gamma class in `true_type`.
    energy_bins: None or int or `numpy.ndarray`
        bins in true_energy.
        If `energy_bins` is None, the default binning given by `ctaplot.ana.irf_cta().energy_bin` if used.
        If `energy_bins` is an int, it defines the number of equal-width energy_bins in the given range.
        If `energy_bins` is a sequence, it defines a monotonically increasing array of bin edges,
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
        energy_bins = irf.energy_bin
    elif isinstance(energy_bins, int):
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


def plot_any_resource(filename, columns_xy=None, ax=None, **kwargs):
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

    if columns_xy is None:
        columns_xy = [0, 1]
    ax = plt.gca() if ax is None else ax

    data = load_any_resource(filename)

    kwargs.setdefault('label', filename)
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
    with quantity_support():
        ax.errorbar(
            magic_table['e_center'].to(u.TeV),
            y=(magic_table['e_center'] ** 2 * magic_table[k]).to(u.Unit('erg cm-2 s-1')),
            xerr=[magic_table['e_err_lo'].to(u.TeV), magic_table['e_err_hi'].to(u.TeV)],
            yerr=(magic_table['e_center'] ** 2 * magic_table[f'{k}_err']).to(u.Unit('erg cm-2 s-1')),
            **kwargs
        )

    ax.set_xlabel(r'$E_R$ [TeV]')
    ax.set_ylabel(r'$energy^2 \cdot$ Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.legend()

    return ax


@u.quantity_input(e_min=u.eV, e_max=u.eV, rate=u.Hz, rate_err=u.Hz)
def plot_rate(e_min, e_max, rate, rate_err=None, ax=None, **kwargs):
    """
    Plot the background rate [Hz] as a function of the true_energy [TeV]

    Parameters
    ----------
    e_min: `astropy.Quantity`
        Reconstructed true_energy in TeV
    e_max: `astropy.Quantity`
        Reconstructed true_energy in TeV
    rate: `astropy.Quantity`
        rate in Hz
    rate_err: `astropy.Quantity`
        error bar on the rate, either 1D (symmetrical) or 2D
    ax: `matplotlib.pyplot.axis`
    kwargs: kwargs for  `matplotlib.pyplot.errobar`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    e_center = np.sqrt(e_min * e_max)

    with quantity_support():
        ax.errorbar(e_center, rate, xerr=[e_center - e_min, e_max - e_center], yerr=rate_err, **kwargs)

    ax.set_xlabel(fr"$E_R$ [{e_center.unit.to_string('latex')}]")
    ax.set_ylabel(fr"Event rate [{rate.unit.to_string('latex')}]")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.legend()

    return ax


@u.quantity_input(e_min=u.eV, e_max=u.eV, background_rate=u.Hz, background_rate_err=u.Hz)
def plot_background_rate(e_min, e_max, background_rate, background_rate_err=None, ax=None, **kwargs):
    """
    Plot the background rate [Hz] as a function of the true_energy [TeV]

    Parameters
    ----------
    e_min: `numpy.ndarray`
        Reconstructed true_energy in TeV
    e_max: `numpy.ndarray`
        Reconstructed true_energy in TeV
    background_rate: `astropy.Quantity`
        Background rate in Hz
    background_rate_err: `astropy.Quantity`
        error bar on the rate, either either 1D (symmetrical) or 2D
    ax: `matplotlib.pyplot.axis`
    kwargs: kwargs for  `matplotlib.pyplot.errobar`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """

    with quantity_support():
        ax = plot_rate(e_min, e_max, background_rate, rate_err=background_rate_err, ax=ax, **kwargs)
    ax.set_ylabel(f"Background rate [{background_rate.unit.to_string('latex')}]")

    return ax


@u.quantity_input(e_min=u.eV, e_max=u.eV, gamma_rate=u.Hz, gamma_rate_err=u.Hz)
def plot_gamma_rate(e_min, e_max, gamma_rate, gamma_rate_err=None, ax=None, **kwargs):
    """
    Plot the gamma rate [Hz] as a function of the true_energy [TeV]

    Parameters
    ----------
    e_min: `numpy.ndarray`
        Reconstructed true_energy in TeV
    e_max: `numpy.ndarray`
        Reconstructed true_energy in TeV
    gamma_rate: `astropy.Quantity`
        gamma rate in Hz
    gamma_rate_err: `astropy.Quantity`
        error bar on the rate, either either 1D (symmetrical) or 2D
    ax: `matplotlib.pyplot.axis`
    kwargs: kwargs for  `matplotlib.pyplot.errobar`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """

    with quantity_support():
        ax = plot_rate(e_min, e_max, gamma_rate, rate_err=gamma_rate_err, ax=ax, **kwargs)
    ax.set_ylabel(fr"Gamma rate [{gamma_rate.unit.to_string('latex')}]")

    return ax


def plot_background_rate_magic(ax=None, **kwargs):
    """
    Plot the  MAGIC background rate from Aleksić, Jelena, et al. 2016, DOI: 10.1016/j.astropartphys.2015.02.005

    Parameters
    ----------
    ax: `matplotlib.pyplot.axis` or None
    kwargs: kwargs for `ctaplot.plots.plot_background_rate`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """

    magic_table = ana.get_magic_sensitivity()

    kwargs.setdefault('label', 'MAGIC (Aleksić et al, 2016)')

    with quantity_support():
        ax = plot_background_rate(magic_table['e_min'].to(u.TeV),
                                  magic_table['e_max'].to(u.TeV),
                                  magic_table['background_rate'].to(u.Hz),
                                  magic_table['background_rate_err'].to(u.Hz),
                                  ax=ax,
                                  **kwargs
                                  )

    return ax


def plot_gamma_rate_magic(ax=None, **kwargs):
    """
    Plot the  MAGIC gamma rate from Aleksić, Jelena, et al. 2016, DOI: 10.1016/j.astropartphys.2015.02.005

    Parameters
    ----------
    ax: `matplotlib.pyplot.axis` or None
    kwargs: kwargs for `ctaplot.plots.plot_gamma_rate`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """

    magic_table = ana.get_magic_sensitivity()

    kwargs.setdefault('label', 'MAGIC (Aleksić et al, 2016)')

    with quantity_support():
        ax = plot_gamma_rate(magic_table['e_min'].to(u.TeV),
                             magic_table['e_max'].to(u.TeV),
                             magic_table['gamma_rate'].to(u.Hz),
                             magic_table['gamma_rate_err'].to(u.Hz),
                             ax=ax,
                             **kwargs
                             )

    return ax


def plot_gammaness_threshold_efficiency(gammaness, efficiency, ax=None, **kwargs):
    """
    Plot the cumulative histogram of the gammaness with the threshold to obtain a give efficiency.
    See also `ctaplot.ana.gammaness_threshold_efficiency`.

    Parameters
    ----------
    gammaness: `numpy.ndarray`
         gammaness of true events (e.g. gammas)
    efficiency: `float`
        between 0 and 1
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplotlib.pyplot.hist`

    Returns
    -------
    ax, threshold
        ax: `matplotlib.pyplot.axes`
        threshold: `float`
    """
    ax = plt.gca() if ax is None else ax

    kwargs.setdefault('bins', 100)
    kwargs.setdefault('range', (0, 1))
    kwargs['cumulative'] = -1
    kwargs['density'] = True
    kwargs.setdefault('color', 'darkblue')
    n, bins, _ = ax.hist(gammaness, **kwargs)
    threshold = bins[:-1][n > efficiency][-1]
    ax.vlines(threshold, 0, 1, color='red')
    ax.hlines(efficiency, 0, 1, color='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('efficiency')
    ax.set_xlabel('gammaness')

    ax2 = ax.twiny()
    ax2.set_xticks([threshold])
    ay2 = ax.twinx()
    ay2.set_yticks([efficiency])

    ax.set_title('Cumulative gammaness distribution')
    ax.grid(True)
    return ax, threshold
  

def plot_precision_recall(y_true, proba_pred, pos_label=0, sample_weigth=None, threshold=None, ax=None, **kwargs):
    """
    Precision as a function of recall.

    Parameters
    ----------
    y_true: ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    proba_pred: ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.
    pos_label: int or str, default=0
        The label of the positive class. The default is 0 for gammas'.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weigth: array-like of shape (n_samples,), default=None
        Sample weights.
    threshold: `float`
        between 0 and 1. Add a point on the curve corresponding to the given threshold.
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `sklearn.metrics.PrecisionRecallDisplay`

    Returns
    -------
    display: `sklearn.metrics.PrecisionRecallDisplay`
    """
    ax = plt.gca() if ax is None else ax

    prec, recall, thresholds = precision_recall_curve(y_true, proba_pred, pos_label=pos_label,
                                                      sample_weight=sample_weigth)

    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, pos_label=pos_label).plot(ax=ax, **kwargs)

    if threshold is not None:
        pred = (proba_pred > threshold).astype(int)
        neg_label = list(set(y_true))
        neg_label.remove(pos_label)
        if len(neg_label) != 1:
            raise ValueError("`y_true` should contain only two labels")
        neg_label = neg_label[0]
        pred_labels = np.where(pred == 1, np.ones_like(pred) * pos_label, np.ones_like(pred) * neg_label)
        r = recall_score(y_true, pred_labels, pos_label=pos_label)
        p = precision_score(y_true, pred_labels, pos_label=pos_label)
        pr_display.ax_.scatter(r, p)

    return pr_display


def plot_roc_auc_per_energy(energy_bins, auc_scores, ax=None, **kwargs):
    """
    Plot AUC scores as a function of the energy.
    These can be computed thanks to `ctaplot.ana.auc_per_energy`

    Parameters
    ----------
    energy_bins: `numpy.ndarray`
    auc_scores: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes` or None
    kwargs: options for `matplotlib.pyplot.errorbar`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """
    ax = plt.gca() if ax is None else ax

    energy_means = np.sqrt(energy_bins[:-1] * energy_bins[1:])
    xerr = (energy_means - energy_bins[:-1], energy_bins[1:] - energy_means)


    with quantity_support():
        ax.errorbar(energy_means, auc_scores, xerr=xerr, **kwargs)

    ax.set_xscale('log')
    ax.set_xlabel(f'Gammas true energy {energy_bins.unit}')
    ax.set_ylabel('AUC')
    ax.grid(True, which='both')

    return ax
