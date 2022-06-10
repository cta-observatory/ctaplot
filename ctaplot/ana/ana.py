"""
ana.py
======
Contain mathematical functions to make results analysis
(compute angular resolution, effective surface, true_energy resolution... )
"""
import numpy as np
from scipy.stats import binned_statistic, norm
from astropy.io.ascii import read
import astropy.units as u
from sklearn import metrics
from ..io import dataset as ds

_relative_scaling_method = 's1'

_north_site_names = ['north', 'lapalma']
_south_site_names = ['south', 'paranal']

__all__ = ['irf_cta',
           'cta_performance',
           'cta_requirement',
           'stat_per_energy',
           'bias',
           'relative_bias',
           'relative_scaling',
           'angular_resolution',
           'angular_separation_altaz',
           'angular_resolution_per_bin',
           'angular_resolution_per_energy',
           'angular_resolution_per_off_pointing_angle',
           'energy_resolution',
           'energy_bias',
           'energy_resolution_per_energy',
           'bias_per_energy',
           'resolution_per_bin',
           'resolution',
           'resolution_per_energy',
           'impact_resolution_per_energy',
           'impact_parameter_error',
           'impact_resolution',
           'distance2d_resolution',
           'distance2d_resolution_per_bin',
           'power_law_integrated_distribution',
           'effective_area',
           'effective_area_per_energy',
           'effective_area_per_energy_power_law',
           'bias_per_bin',
           'percentile_confidence_interval',
           'logbin_mean',
           'get_magic_sensitivity',
           'logspace_decades_nbin',
           'roc_auc_per_energy',
           ]


class irf_cta:
    """
    Class to handle Instrument Response Function data
    """

    def __init__(self):
        self.site = ''
        self.energy_bins = np.logspace(np.log10(2.51e-02), 2, 19) * u.TeV
        self.energy = logbin_mean(self.energy_bins)

        # Area of CTA sites in meters
        self.ParanalArea_prod3 = 19.63e6 * u.m**2
        self.LaPalmaArea_prod3 = 11341149 * u.m**2  # 6.61e6

    @u.quantity_input(energy_bins=u.TeV)
    def set_ebin(self, energy_bins):
        self.energy_bins = energy_bins
        self.energy = logbin_mean(self.energy_bins)


class cta_performance:
    def __init__(self, site):
        self.site = site
        self.energy = np.empty(0) * u.TeV
        self.energy_bins = np.empty(0) * u.TeV
        self.effective_area = np.empty(0) * u.m**2
        self.angular_resolution = np.empty(0) * u.deg
        self.energy_resolution = np.empty(0)
        self.sensitivity = np.empty(0) * u.erg / (u.cm ** 2 * u.s)

    @u.quantity_input(observation_time=u.h)
    def get_effective_area(self, observation_time=50 * u.h):
        """
        Return the effective area at the given observation time in hours.
        NB: Only 50h supported
        Returns the true_energy array and the effective area array
        Parameters
        ----------
        observation_time: optional

        Returns
        -------
        `numpy.ndarray`, `numpy.ndarray`
        """
        if self.site in _south_site_names:
            if observation_time == 50 * u.h:
                energy, effective_area = np.loadtxt(
                    ds.get('CTA-Performance-prod3b-v2-South-20deg-50h-EffArea.txt'),
                    skiprows=11, unpack=True)
                self.energy = energy * u.TeV
                self.effective_area = effective_area * u.m**2
            elif observation_time == 0.5 * u.h:
                energy, effective_area = np.loadtxt(
                    ds.get('CTA-Performance-prod3b-v2-North-20deg-30m-EffArea.txt'),
                    skiprows=11, unpack=True)
                self.energy = energy * u.TeV
                self.effective_area = effective_area * u.m**2
            else:
                raise ValueError("no effective area for this observation time")

        elif self.site in _north_site_names:
            if observation_time == 50 * u.h:
                energy, effective_area = np.loadtxt(
                    ds.get('CTA-Performance-prod3b-v2-North-20deg-50h-EffArea.txt'),
                    skiprows=11, unpack=True)
                self.energy = energy * u.TeV
                self.effective_area = effective_area * u.m**2
            elif observation_time == 0.5 * u.h:
                energy, effective_area = np.loadtxt(
                    ds.get('CTA-Performance-prod3b-v2-North-20deg-30m-EffArea.txt'),
                    skiprows=11, unpack=True)
                self.energy = energy * u.TeV
                self.effective_area = effective_area * u.m**2
            else:
                raise ValueError("no effective area for this observation time")

        else:
            raise ValueError(f'incorrect site specified, \
            accepted values are {_north_site_names} or {_south_site_names}')
        return self.energy, self.effective_area

    def get_angular_resolution(self):
        if self.site in _south_site_names:
            energy, angular_resolution = np.loadtxt(
                ds.get('CTA-Performance-prod3b-v2-South-20deg-50h-Angres.txt'),
                skiprows=11, unpack=True)
        elif self.site in _north_site_names:
            energy, angular_resolution = np.loadtxt(
                ds.get('CTA-Performance-prod3b-v2-North-20deg-50h-Angres.txt'),
                skiprows=11, unpack=True)

        else:
            raise ValueError(f'incorrect site specified, \
                    accepted values are {_north_site_names} or {_south_site_names}')

        self.energy = energy * u.TeV
        self.angular_resolution = angular_resolution * u.deg

        return self.energy, self.angular_resolution

    def get_energy_resolution(self):
        if self.site in _south_site_names:
            energy, energy_resolution = np.loadtxt(ds.get('CTA-Performance-prod3b-v2-South-20deg-50h-Eres.txt'),
                                                   skiprows=11, unpack=True)
        elif self.site in _north_site_names:
            energy, energy_resolution = np.loadtxt(ds.get('CTA-Performance-prod3b-v2-North-20deg-50h-Eres.txt'),
                                                   skiprows=11, unpack=True)
        else:
            raise ValueError(
                f'incorrect site specified, accepted values are {_north_site_names} or {_south_site_names}')
        self.energy = energy * u.TeV
        self.energy_resolution = energy_resolution

        return self.energy, self.energy_resolution

    @u.quantity_input(observation_time=u.h)
    def get_sensitivity(self, observation_time=50 * u.h):
        if self.site in _south_site_names:
            observation_times = {50 * u.h: 'CTA-Performance-prod3b-v2-South-20deg-50h-DiffSens.txt',
                                 0.5 * u.h: 'CTA-Performance-prod3b-v2-South-20deg-05h-DiffSens.txt',
                                 5 * u.h: 'CTA-Performance-prod3b-v2-South-20deg-05h-DiffSens.txt'
                                 }
            Emin, Emax, sensitivity = np.loadtxt(ds.get(observation_times[observation_time]),
                                                 skiprows=10, unpack=True)
            self.energy_bins = np.append(Emin, Emax[-1]) * u.TeV
            self.energy = logbin_mean(self.energy_bins)
            self.sensitivity = sensitivity * u.erg / (u.cm ** 2 * u.s)

        elif self.site in _north_site_names:
            observation_times = {50 * u.h: 'CTA-Performance-prod3b-v2-North-20deg-50h-DiffSens.txt',
                                 0.5 * u.h: 'CTA-Performance-prod3b-v2-North-20deg-05h-DiffSens.txt',
                                 5 * u.h: 'CTA-Performance-prod3b-v2-North-20deg-05h-DiffSens.txt'
                                 }
            Emin, Emax, sensitivity = np.loadtxt(ds.get(observation_times[observation_time]),
                                                 skiprows=10, unpack=True)
            self.energy_bins = np.append(Emin, Emax[-1]) * u.TeV
            self.energy = logbin_mean(self.energy_bins)
            self.sensitivity = sensitivity * u.erg / (u.cm ** 2 * u.s)

        else:
            raise ValueError(
                f'incorrect site specified, accepted values are {_north_site_names} or {_south_site_names}')

        return self.energy, self.sensitivity


class cta_requirement:
    def __init__(self, site):
        self.site = site
        self.energy = np.empty(0) * u.TeV
        self.effective_area = np.empty(0) * u.m**2
        self.angular_resolution = np.empty(0) * u.deg
        self.energy_resolution = np.empty(0)
        self.sensitivity = np.empty(0) * u.erg / (u.cm ** 2 * u.s)

    @u.quantity_input(observation_time=u.h)
    def get_effective_area(self, observation_time=50 * u.h):
        """
        Return the effective area at the given observation time in hours.
        NB: Only 0.5h supported
        Returns the true_energy array and the effective area array
        Parameters
        ----------
        observation_time: optional

        Returns
        -------
        `numpy.ndarray`, `numpy.ndarray`
        """
        if observation_time != 50 * u.h:
            raise ValueError(f"no effective area for an observation time of {observation_time}")

        if self.site in _south_site_names:
            energy, effective_area = np.loadtxt(ds.get('cta_requirements_South-30m-EffectiveArea.dat'),
                                                unpack=True)
        elif self.site in _north_site_names:
            energy, effective_area = np.loadtxt(ds.get('cta_requirements_North-30m-EffectiveArea.dat'),
                                                unpack=True)
        else:
            raise ValueError(
                f'incorrect site specified, accepted values are {_north_site_names} or {_south_site_names}')
        self.energy = energy * u.TeV
        self.effective_area = effective_area * u.m**2
        return self.energy, self.effective_area

    def get_angular_resolution(self):
        if self.site in _south_site_names:
            energy, angular_resolution = np.loadtxt(ds.get('cta_requirements_South-50h-AngRes.dat'), unpack=True)
        elif self.site in _north_site_names:
            energy, angular_resolution = np.loadtxt(ds.get('cta_requirements_North-50h-AngRes.dat'), unpack=True)
        else:
            raise ValueError(
                f'incorrect site specified, accepted values are {_north_site_names} or {_south_site_names}')
        self.energy = energy * u.TeV
        self.angular_resolution = angular_resolution * u.deg
        return self.energy, self.angular_resolution

    def get_energy_resolution(self):
        if self.site in _south_site_names:
            energy, self.energy_resolution = np.loadtxt(ds.get('cta_requirements_South-50h-ERes.dat'), unpack=True)
        elif self.site in _north_site_names:
            energy, self.energy_resolution = np.loadtxt(ds.get('cta_requirements_North-50h-ERes.dat'), unpack=True)
        else:
            raise ValueError(
                f'incorrect site specified, accepted values are {_north_site_names} or {_south_site_names}')
        self.energy = energy * u.TeV
        return self.energy, self.energy_resolution

    @u.quantity_input(observation_time=u.h)
    def get_sensitivity(self, observation_time=50 * u.h):
        if observation_time != 50 * u.h:
            raise ValueError(f"no sensitivity for an observation time of {observation_time}")
        if self.site in _south_site_names:
            energy, sensitivity = np.loadtxt(ds.get('cta_requirements_South-50h.dat'), unpack=True)
        elif self.site in _north_site_names:
            energy, sensitivity = np.loadtxt(ds.get('cta_requirements_North-50h.dat'), unpack=True)
        else:
            raise ValueError(
                f'incorrect site specified, accepted values are {_north_site_names} or {_south_site_names}')
        self.energy = energy * u.TeV
        self.sensitivity = sensitivity * u.erg / (u.cm ** 2 * u.s)
        return self.energy, self.sensitivity


def assert_unit_equivalency(x, y):
    """
    Assert that two quantities are equivalent.
    Raises an error if not.

    Parameters
    ----------
    x: `astropy.Quantity`
    y: `astropy.Quantity`
    """

    if not x.unit.is_equivalent(y.unit):
        raise ValueError(f"Units {x.unit} and {y.unit} are not equivalent")


def logspace_decades_nbin(x_min, x_max, n=5):
    """
    return an array with logspace and n bins / decade

    Parameters
    ----------
    x_min: float
    x_max: float
    n: int - number of bins per decade

    Returns
    -------
    bins: 1D Numpy array
    """
    eps = 1e-10
    if isinstance(x_min, u.Quantity) or isinstance(x_max, u.Quantity):
        assert_unit_equivalency(x_min, x_max)

        unit = x_min.unit
        bins = 10 ** np.arange(np.log10(x_min.to_value(unit)),
                               np.log10(x_max.to_value(unit) + eps),
                               1 / n,
                               )
        return u.Quantity(bins, x_min.unit, copy=False)

    else:
        bins = 10 ** np.arange(np.log10(x_min),
                               np.log10(x_max + eps),
                               1 / n,
                               )
        return bins


@u.quantity_input(energy=u.TeV, bins=u.TeV)
def stat_per_energy(energy, y, statistic='mean', bins=None):
    """
    Return statistic for the given quantity per energy bins.
    The binning is given by irf_cta

    Parameters
    ----------
    energy: `astropy.Quantity` (1d array)
        event energy
    y: `astropy.Quantity` or `numpy.ndarray` (1d array)
        len(y) == len(energy)
    statistic: string
        see `scipy.stat.binned_statistic`
    bins: `astropy.Quantity` (1d array)

    Returns
    -------
    `astropy.Quantity` or `numpy.ndarray`, `astropy.Quantity`, `numpy.ndarray`
        bin_stat, bin_edges, binnumber
    """

    if bins is None:
        irf = irf_cta()
        bins = irf.energy_bins

    bin_stat, bin_edges, binnumber = binned_statistic(energy.to_value(u.TeV),
                                                      y,
                                                      statistic=statistic,
                                                      bins=bins.to_value(u.TeV))
    if isinstance(y, u.Quantity):
        bin_stat = u.Quantity(bin_stat, y.unit)

    return bin_stat, u.Quantity(bin_edges, u.TeV), binnumber


def bias(true, reco):
    """
    Compute the bias of a reconstructed variable as `median(reco-true)`

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`

    Returns
    -------
    float
    """
    if len(true) != len(reco):
        raise ValueError("both arrays should have the same size")
    if len(true) == 0:
        return 0
    return np.median(reco - true)


def relative_bias(true, reco, relative_scaling_method='s1'):
    """
    Compute the relative bias of a reconstructed variable as
    `median(reco-true)/relative_scaling(true, reco)`

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------

    """
    assert len(reco) == len(true)
    if len(true) == 0:
        return 0
    return np.median((reco - true) / relative_scaling(true, reco, method=relative_scaling_method))


def relative_scaling(true, reco, method='s0'):
    """
    Define the relative scaling for the relative error calculation.
    There are different ways to calculate this scaling factor.
    The easiest and most spread one is simply `np.abs(true)`. However this is possible only when `true != 0`.
    Possible methods:
    - None or 's0': scale = 1
    - 's1': `scale = np.abs(true)`
    - 's2': `scale = np.abs(reco)`
    - 's3': `scale = (np.abs(true) + np.abs(reco))/2.`
    - 's4': `scale = np.max([np.abs(reco), np.abs(true)], axis=0)`

    This method is not exposed but kept for tests and future reference.
    The `s1` method is used in all `ctaplot` functions.

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    method = 's0' if method is None else method
    scaling_methods = {
        's0': lambda true, reco: np.ones(len(true)),
        's1': lambda true, reco: np.abs(true),
        's2': lambda true, reco: np.abs(reco),
        's3': lambda true, reco: (np.abs(true) + np.abs(reco)) / 2.,
        's4': lambda true, reco: np.max([np.abs(reco), np.abs(true)], axis=0)
    }

    return scaling_methods[method](true, reco)


def resolution(true, reco,
               percentile=68.27, confidence_level=0.95, bias_correction=False, relative_scaling_method='s1'):
    """
    Compute the resolution of reco as the Qth (68.27 as standard = 1 sigma) containment radius of
    `(true-reco)/relative_scaling` with the lower and upper confidence limits defined the values inside
    the error_percentile

    Parameters
    ----------
    true: `numpy.ndarray` (1d)
        simulated quantity
    reco: `numpy.ndarray` (1d)
        reconstructed quantity
    percentile: float
        percentile for the resolution containment radius
    error_percentile: float
        percentile for the confidence limits
    bias_correction: bool
        if True, the resolution is corrected with the bias computed on true and reco
    relative_scaling: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------
    `numpy.ndarray` - [resolution, lower_confidence_limit, upper_confidence_limit]
    """
    assert len(true) == len(reco), "both arrays should have the same size"

    b = bias(true, reco) if bias_correction else 0

    with np.errstate(divide='ignore', invalid='ignore'):
        reco_corr = reco - b
        res = np.nan_to_num(np.abs((reco_corr - true) /
                                   relative_scaling(true, reco_corr, method=relative_scaling_method)))

    return np.append(_percentile(res, percentile), percentile_confidence_interval(res, percentile=percentile,
                                                                                  confidence_level=confidence_level))


def resolution_per_bin(x, y_true, y_reco,
                       percentile=68.27,
                       confidence_level=0.95,
                       bias_correction=False,
                       relative_scaling_method=None,
                       bins=10):
    """
    Resolution of y as a function of binned x.

    Parameters
    ----------
    x: `numpy.ndarray`
    y_true: `numpy.ndarray`
    y_reco: `numpy.ndarray`
    percentile: float
    confidence_level: float
    bias_correction: bool
    relative_scaling_method: see `ctaplot.ana.relative_scaling`
    bins: int or `numpy.ndarray` (see `numpy.histogram`)

    Returns
    -------
    (x_bins, res): (`numpy.ndarray`, `numpy.ndarray`)
        x_bins: bins for x
        res: resolutions with confidence level intervals for each bin
    """
    _, x_bins = np.histogram(x, bins=bins)
    bin_index = np.digitize(x, x_bins)
    res = []
    for ii in np.arange(1, len(x_bins)):
        mask = bin_index == ii
        res.append(resolution(y_true[mask], y_reco[mask],
                              percentile=percentile,
                              confidence_level=confidence_level,
                              relative_scaling_method=relative_scaling_method,
                              bias_correction=bias_correction,
                              )
                   )

    res = u.Quantity(res) if isinstance(res[0], u.Quantity) else np.array(res)
    return x_bins, res


@u.quantity_input(true_energy=u.TeV, bins=u.TeV)
def resolution_per_energy(true, reco, true_energy, percentile=68.27, confidence_level=0.95, bias_correction=False,
                          bins=None):
    """
    Parameters
    ----------
    true: 1d `numpy.ndarray` of simulated quantity
    reco: 1d `numpy.ndarray` of reconstructed quantity
    true_energy: `astropy.Quantity` (1d array)
        len(true_energy) == len(true) == len(reco)
    bins: `astropy.Quantity` (1d array)

    Returns
    -------
    (energy_bins, resolution):
        energy_bins - 1D `numpy.ndarray`
        resolution: - 3D `numpy.ndarray` see `ctaplot.ana.resolution`
    """

    if bins is None:
        irf = irf_cta()
        bins = irf.energy_bins

    return resolution_per_bin(true_energy, true, reco,
                              percentile=percentile,
                              confidence_level=confidence_level,
                              bias_correction=bias_correction,
                              bins=bins)


@u.quantity_input(true_energy=u.TeV, reco_energy=u.TeV)
def energy_resolution(true_energy, reco_energy, percentile=68.27, confidence_level=0.95, bias_correction=False):
    """
    Compute the true_energy resolution of true_energy as the percentile (68 as standard) containment radius of
    `true_energy-true_energy)/true_energy`
    with the lower and upper confidence limits defined by the given confidence level

    Parameters
    ----------
    true_energy: 1d numpy array of simulated energy
    reco_energy: 1d numpy array of reconstructed energy
    percentile: float
        <= 100

    Returns
    -------
    `numpy.array` - [energy_resolution, lower_confidence_limit, upper_confidence_limit]
    """
    return resolution(true_energy, reco_energy, percentile=percentile,
                      confidence_level=confidence_level,
                      bias_correction=bias_correction,
                      relative_scaling_method='s1',
                      )


@u.quantity_input(true_energy=u.TeV, reco_energy=u.TeV)
def energy_resolution_per_energy(true_energy, reco_energy,
                                 percentile=68.27, confidence_level=0.95, bias_correction=False, bins=None):
    """
    The true_energy resolution ΔE / energy is obtained from the distribution of (ER – ET) / ET, where R and T refer
    to the reconstructed and true energy of gamma-ray events.
    ΔE/energy is the half-width of the interval around 0 which contains given percentile of the distribution.

    Parameters
    ----------
    true_energy: `astropy.Quantity`
        1d array of simulated energy
    reco_energy: `astropy.Quantity`
        1d array of reconstructed energy
    percentile: float
        between 0 and 100
    confidence_level: float
        between 0 and 1
    bias_correction: bool
    bins: int | `astropy.Quantity`

    Returns
    -------
    (e, e_res): (astropy.Quantity, numpy.array)
        true_energy, resolution in true_energy
    """
    assert len(reco_energy) > 0, "Empty arrays"

    res_e = []
    irf = irf_cta()

    if bins is None:
        bins = irf.energy_bins

    for i in range(len(bins)-1):
        mask = (reco_energy > bins[i]) & (reco_energy < bins[i + 1])

        res_e.append(energy_resolution(true_energy[mask], reco_energy[mask],
                                       percentile=percentile,
                                       confidence_level=confidence_level,
                                       bias_correction=bias_correction))

    return bins, np.array(res_e)


@u.quantity_input(true_energy=u.TeV, reco_energy=u.TeV, bins=u.TeV)
def energy_bias(true_energy, reco_energy, bins=None):
    """
    Compute the true_energy relative bias per true_energy bin.

    Parameters
    ----------
    true_energy: `astropy.Quantity` (1d array)
        simulated energies
    reco_energy: `astropy.Quantity` (1d array)
        reconstructed energies
    bins: astropy.Quantity (1d array)
        energy bins - if None, standard CTA binning is used

    Returns
    -------
    (energy_bins, bias): (astropy.Quantity, numpy.array)
        true_energy, true_energy bias
    """
    bias_e = []

    irf = irf_cta()
    if bins is None:
        bins = irf.energy_bins
    for i, e in enumerate(irf.energy):
        mask = (reco_energy > irf.energy_bins[i]) & (reco_energy < irf.energy_bins[i + 1])
        bias_e.append(relative_bias(true_energy[mask], reco_energy[mask], relative_scaling_method='s1'))

    return bins, np.array(bias_e)


def get_angles_pipi(angles):
    """
    return angles modulo between -pi and +pi

    Parameters
    ----------
    angles: `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    return np.mod(angles + np.pi, 2 * np.pi) - np.pi


def get_angles_02pi(angles):
    """
    return angles modulo between 0 and +pi

    Parameters
    ----------
    angles: `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    return np.mod(angles, 2 * np.pi)


@u.quantity_input(true_alt=u.rad, reco_alt=u.rad, true_az=u.rad, reco_az=u.rad)
def theta2(true_alt, reco_alt, true_az, reco_az, bias_correction=False):
    """
    Compute the theta2 in radians

    Parameters
    ----------
    reco_alt: 1d `astropy.Quantity` - reconstructed Altitude in radians
    reco_az: 1d `astropy.Quantity` - reconstructed Azimuth in radians
    true_alt: 1d `astropy.Quantity`- true Altitude in radians
    true_az: 1d `astropy.Quantity` -  true Azimuth in radians

    Returns
    -------
    theta2: `astropy.Quantity` (~deg2)
    """
    assert (len(reco_az) == len(reco_alt))
    assert (len(reco_alt) == len(true_alt))
    if len(reco_alt) == 0:
        return np.empty(0) * u.rad ** 2
    if bias_correction:
        bias_alt = bias(true_alt, reco_alt)
        bias_az = bias(true_az, reco_az)
    else:
        bias_alt = 0 * u.rad
        bias_az = 0 * u.rad
    return angular_separation_altaz(reco_alt - bias_alt, reco_az - bias_az, true_alt, true_az) ** 2


@u.quantity_input(true_alt=u.rad, reco_alt=u.rad, true_az=u.rad, reco_az=u.rad)
def angular_resolution(true_alt, reco_alt, true_az, reco_az,
                       percentile=68.27, confidence_level=0.95, bias_correction=False):
    """
    Compute the angular resolution as the Qth (standard being 68)
    containment radius of theta2 with lower and upper limits on this value
    corresponding to the confidence value required (1.645 for 95% confidence)

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
    percentile: float - percentile, 68 corresponds to one sigma
    confidence_level: float

    Returns
    -------
    `numpy.array` [angular_resolution, lower limit, upper limit]
    """
    if bias_correction:
        b_alt = bias(true_alt, reco_alt)
        b_az = bias(true_az, reco_az)
    else:
        b_alt = 0 * u.rad
        b_az = 0 * u.rad

    reco_alt_corr = reco_alt - b_alt
    reco_az_corr = reco_az - b_az

    t2 = np.sort(theta2(true_alt, reco_alt_corr, true_az, reco_az_corr))
    ang_res = _percentile(t2, percentile)

    return np.sqrt(np.append(ang_res, percentile_confidence_interval(t2, percentile, confidence_level)))


@u.quantity_input(true_alt=u.rad, reco_alt=u.rad, true_az=u.rad, reco_az=u.rad)
def angular_resolution_per_bin(true_alt, reco_alt, true_az, reco_az, x,
                               percentile=68.27, confidence_level=0.95, bias_correction=False, bins=10):
    """
    Compute the angular resolution per binning of x

    Parameters
    ----------
    true_alt: `astropy.Quantity`
    true_az: `astropy.Quantity`
    reco_alt: `astropy.Quantity`
    reco_az: `astropy.Quantity`
    x: `numpy.ndarray`
    percentile: float
        0 < percentile < 100
    confidence_level: float
        0 < confidence_level < 1
    bias_correction: bool
    bins: int or `numpy.ndarray`

    Returns
    -------
    bins, ang_res: (numpy.ndarray, numpy.ndarray)
    """
    _, x_bins = np.histogram(x, bins=bins)
    bin_index = np.digitize(x, x_bins)

    ang_res = []
    for ii in np.arange(1, len(x_bins)):
        mask = bin_index == ii
        ang_res.append(angular_resolution(true_alt[mask], reco_alt[mask],
                                          true_az[mask], reco_az[mask],
                                          percentile=percentile,
                                          confidence_level=confidence_level,
                                          bias_correction=bias_correction,
                                          )
                       )

    return x_bins, u.Quantity(ang_res)

@u.quantity_input(true_alt=u.rad, reco_alt=u.rad, true_az=u.rad, reco_az=u.rad, energy=u.TeV, bins=u.TeV)
def angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, energy,
                                  percentile=68.27, confidence_level=0.95, bias_correction=False, bins=None):
    """
    Plot the angular resolution as a function of the event simulated true_energy

    Parameters
    ----------
    reco_alt: `astropy.Quantity`
    reco_az: `astropy.Quantity`
    true_alt: `astropy.Quantity`
    true_az: `astropy.Quantity`
    energy: `astropy.Quantity`
    bins: `astropy.Quantity`
    **kwargs: args for `angular_resolution`

    Returns
    -------
    (energy, RES) : (astropy.Quantity, numpy.array)
    """
    if not len(reco_alt) == len(reco_az) == len(energy) > 0:
        raise ValueError("reco_alt, reco_az and true_energy must have the same length")

    if bins is None:
        irf = irf_cta()
        bins = irf.energy_bins

    res = []

    for i, e in enumerate(bins[:-1]):
        mask = (energy > bins[i]) & (energy <= bins[i + 1])
        res.append(angular_resolution(true_alt[mask], reco_alt[mask], true_az[mask], reco_az[mask],
                                      percentile=percentile,
                                      confidence_level=confidence_level,
                                      bias_correction=bias_correction,
                                      )
                   )

    res_q = u.Quantity(res)
    return bins, res_q.to(u.deg)



@u.quantity_input(true_alt=u.rad, reco_alt=u.rad, true_az=u.rad, reco_az=u.rad, alt_pointing=u.rad, az_pointing=u.rad)
def angular_resolution_per_off_pointing_angle(true_alt, reco_alt, true_az, reco_az, alt_pointing, az_pointing, bins=10):
    """
    Compute the angular resolution as a function of separation angle for the pointing direction

    Parameters
    ----------
    true_alt: `astropy.Quantity`
    true_az: `astropy.Quantity`
    reco_alt: `astropy.Quantity`
    reco_az: `astropy.Quantity`
    alt_pointing: `astropy.Quantity`
    az_pointing: `astropy.Quantity`
    bins: int or `astropy.Quantity`

    Returns
    -------
    (bins, res):
        bins: 1D `astropy.Quantity`
        res: 2D `numpy.ndarray` - resolutions with confidence intervals (output from `ctaplot.ana.resolution`)
    """
    ang_sep_to_pointing = angular_separation_altaz(true_alt, true_az, alt_pointing, az_pointing)

    return angular_resolution_per_bin(true_alt, reco_alt, true_az, reco_az, ang_sep_to_pointing, bins=bins)


@u.quantity_input(true_energy=u.TeV, reco_energy=u.TeV, simu_area=u.m ** 2)
def effective_area(true_energy, reco_energy, simu_area):
    """
    Compute the effective area from a list of simulated energy and reconstructed energy
    Parameters
    ----------
    true_energy: 1d numpy array
    reco_energy: 1d numpy array
    simu_area: float - area on which events are simulated
    Returns
    -------
    float = effective area
    """
    return simu_area * len(reco_energy) / len(true_energy)



@u.quantity_input(true_energy=u.TeV, reco_energy=u.TeV, simu_area=u.m ** 2, bins=u.TeV)
def effective_area_per_energy(true_energy, reco_energy, simu_area, bins=None):
    """
    Compute the effective area per true_energy bins from a list of simulated energy and reconstructed energy

    Parameters
    ----------
    true_energy: `astropy.Quantity`
    reco_energy: `astropy.Quantity`
    simu_area: `astropy.Quantity`
        area on which events are simulated
    bins:  `astropy.Quantity`

    Returns
    -------
    (energy, Seff) : (1d numpy array, 1d numpy array)
    """

    if bins is None:
        irf = irf_cta()
        bins = irf.energy_bins

    count_R, bin_R = np.histogram(reco_energy, bins=bins)
    count_S, bin_S = np.histogram(true_energy, bins=bins)

    np.seterr(divide='ignore', invalid='ignore')
    return bins, np.nan_to_num(simu_area * count_R / count_S)


@u.quantity_input(true_x=u.m, reco_x=u.m, true_y=u.m, reco_y=u.m)
def impact_parameter_error(true_x, reco_x, true_y, reco_y):
    """
    compute the error distance between true and reconstructed impact parameters
    Parameters
    ----------
    reco_x: `astropy.Quantity`
    reco_y: `astropy.Quantity`
    true_x: `astropy.Quantity`
    true_y: `astropy.Quantity`

    Returns
    -------
    1d numpy array: distances
    """
    return np.sqrt((reco_x - true_x) ** 2 + (reco_y - true_y) ** 2)


def _percentile(x, percentile=68.27):
    """
    Compute the value of the Qth containment radius
    Return 0 if the list is empty
    Parameters
    ----------
    x: numpy array or list

    Returns
    -------
    float
    """
    if len(x) != 0:
        return np.percentile(x, percentile)
    if isinstance(x, u.Quantity):
        return 0 * x.unit
    else:
        return 0


@u.quantity_input(alt1=u.rad, az1=u.rad, alt2=u.rad, az2=u.rad)
def angular_separation_altaz(alt1, az1, alt2, az2):
    """
    Compute the angular separation in radians or degrees
    between two pointing direction given with alt-az

    Parameters
    ----------
    alt1: 1d `astropy.Quantity`, altitude of the first pointing direction
    az1: 1d `astropy.Quantity` azimuth of the first pointing direction
    alt2: 1d `astropy.Quantity`, altitude of the second pointing direction
    az2: 1d `astropy.Quantity`, azimuth of the second pointing direction

    Returns
    -------
    1d `numpy.ndarray` or float, angular separation
    """

    cosdelta = np.cos(alt1.to_value(u.rad)) * np.cos(alt2.to_value(u.rad)) * np.cos(
        (az1 - az2).to_value(u.rad)) + np.sin(alt1.to_value(u.rad)) * np.sin(alt2.to_value(u.rad))

    cosdelta[cosdelta > 1] = 1.
    cosdelta[cosdelta < -1] = -1.

    return np.arccos(cosdelta) * u.rad


def logbin_mean(x_bin):
    """
    Function that gives back the mean of each bin in logscale

    Parameters
    ----------
    x_bin: `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    if not isinstance(x_bin, u.Quantity):
        return 10 ** ((np.log10(x_bin[:-1]) + np.log10(x_bin[1:])) / 2.)
    unit = x_bin.unit
    return (10 ** ((np.log10(x_bin[:-1].to_value(unit)) + np.log10(x_bin[1:].to_value(unit))) / 2.)) * unit


@u.quantity_input(true_x=u.m, reco_x=u.m, true_y=u.m, reco_y=u.m)
def impact_resolution(true_x, reco_x, true_y, reco_y,
                      percentile=68.27, confidence_level=0.95, bias_correction=False, relative_scaling_method=None):
    """
    Compute the shower impact parameter resolution as the Qth (68 as standard) containment radius of the square distance
    to the simulated one with the lower and upper limits corresponding to the required confidence level

    Parameters
    ----------
    reco_x: `astropy.Quantity`
    reco_y: `astropy.Quantity`
    true_x: `astropy.Quantity`
    true_y: `astropy.Quantity`
    percentile: float
        see `ctaplot.ana.resolution`
    confidence_level: float
        see `ctaplot.ana.resolution`
    bias_correction: bool
        see `ctaplot.ana.resolution`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------
    (impact_resolution, lower_confidence_level, upper_confidence_level): (`numpy.array`, `numpy.array`, `numpy.array`)
    """

    return distance2d_resolution(true_x, reco_x, true_y, reco_y,
                                 percentile=percentile,
                                 confidence_level=confidence_level,
                                 bias_correction=bias_correction,
                                 relative_scaling_method=relative_scaling_method
                                 )


@u.quantity_input(true_x=u.m, reco_x=u.m, true_y=u.m, reco_y=u.m, true_energy=u.TeV, bins=u.TeV)
def impact_resolution_per_energy(true_x, reco_x, true_y, reco_y, true_energy,
                                 percentile=68.27,
                                 confidence_level=0.95,
                                 bias_correction=False,
                                 relative_scaling_method=None,
                                 bins=None):
    """
    Plot the angular resolution as a function of the event simulated true_energy

    Parameters
    ----------
    reco_x: `astropy.Quantity`
    reco_y: `astropy.Quantity`
    true_x: `astropy.Quantity`
    true_y: `astropy.Quantity`
    true_energy: `astropy.Quantity`
    percentile: float
        see `ctaplot.ana.resolution`
    confidence_level: float
        see `ctaplot.ana.resolution`
    bias_correction: bool
        see `ctaplot.ana.resolution`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`
    bins: `astropy.Quantity`

    Returns
    -------
    (true_energy, resolution) : (`astropy.Quantity`, 1d numpy array)
    """
    assert len(reco_x) == len(true_energy)
    assert len(true_energy) > 0, "Empty arrays"

    if bins is None:
        irf = irf_cta()
        bins = irf.energy_bins

    return distance2d_resolution_per_bin(true_energy, true_x, reco_x, true_y, reco_y,
                                         bins=bins,
                                         percentile=percentile,
                                         confidence_level=confidence_level,
                                         bias_correction=bias_correction,
                                         relative_scaling_method=relative_scaling_method,
                                         )


def percentile_confidence_interval(x, percentile=68, confidence_level=0.95):
    """
    Return the confidence interval for the qth percentile of x for a given confidence level

    REF:
    http://people.stat.sfu.ca/~cschwarz/Stat-650/Notes/PDF/ChapterPercentiles.pdf
    S. Chakraborti and J. Li, Confidence Interval Estimation of a Normal Percentile, doi:10.1198/000313007X244457

    Parameters
    ----------
    x: `numpy.ndarray`
    percentile: `float`
        0 < percentile < 100
    confidence_level: `float`
        0 < confidence level (by default 95%) < 1

    Returns
    -------

    """
    sorted_x = np.sort(x)
    z = norm.ppf(confidence_level)
    if len(x) == 0:
        return 0, 0
    q = percentile / 100.

    j = np.max([0, int(len(x) * q - z * np.sqrt(len(x) * q * (1 - q)))])
    k = np.min([int(len(x) * q + z * np.sqrt(len(x) * q * (1 - q))), len(x) - 1])
    return sorted_x[j], sorted_x[k]


def power_law_integrated_distribution(xmin, xmax, total_number_events, spectral_index, bins):
    """
    For each bin, return the expected number of events for a power-law distribution.
    bins: `numpy.ndarray`, e.g. `np.logspace(np.log10(emin), np.logspace(xmax))`

    Parameters
    ----------
    xmin: `float`, min of the simulated power-law
    xmax: `float`, max of the simulated power-law
    total_number_events: `int`
    spectral_index: `float`
    bins: `numpy.ndarray`

    Returns
    -------
    y: `numpy.ndarray`, len(y) = len(bins) - 1
    """
    if spectral_index == -1:
        y0 = total_number_events / np.log(xmax / xmin)
        y = y0 * np.log(bins[1:] / bins[:-1])
    else:
        y0 = total_number_events / (xmax ** (spectral_index + 1) - xmin ** (spectral_index + 1)) * (spectral_index + 1)
        y = y0 * (bins[1:] ** (spectral_index + 1) - bins[:-1] ** (spectral_index + 1)) / (spectral_index + 1)
    return y


@u.quantity_input(emin=u.eV, emax=u.eV, true_energy=u.eV, simu_area=u.m ** 2, bins=u.eV)
def effective_area_per_energy_power_law(emin, emax, total_number_events, spectral_index, true_energy, simu_area,
                                        bins=None):
    """
    Compute the effective area per true_energy bins from a list of simulated energy and reconstructed energy

    Parameters
    ----------
    emin: `astropy.Quantity`
    emax: `astropy.Quantity`
    total_number_events: int
    spectral_index: float
    true_energy: 1d `astropy.Quantity`
    simu_area: `astropy.Quantity`
        area on which events are simulated
    bins: `astropy.Quantity`

    Returns
    -------
    (true_energy bins, effective_area) : (`astropy.Quantity` array, 1d numpy array)
    """

    if bins is None:
        irf = irf_cta()
        bins = irf.energy_bins

    simu_per_bin = power_law_integrated_distribution(emin, emax, total_number_events, spectral_index, bins)
    count_R, bin_R = np.histogram(true_energy, bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        return bins, np.nan_to_num(simu_area * count_R / simu_per_bin)


def distance2d_resolution(true_x, reco_x, true_y, reco_y,
                          percentile=68.27, confidence_level=0.95, bias_correction=False, relative_scaling_method=None):
    """
    Compute the 2D distance resolution as the Qth (standard being 68)
    containment radius of the relative distance with lower and upper limits on this value
    corresponding to the confidence value required (1.645 for 95% confidence)

    Parameters
    ----------
    true_x: `numpy.ndarray` or `astropy.units.Quantity`
    reco_x: `numpy.ndarray` or `astropy.units.Quantity`
    true_y: `numpy.ndarray` or `astropy.units.Quantity`
    reco_y: `numpy.ndarray`or `astropy.units.Quantity`
    percentile: float - percentile, 68.27 corresponds to one sigma
    confidence_level: float
    bias_correction: bool
    relative_scaling_method: str
        - see `ctaplot.ana.relative_scaling`

    Returns
    -------
    `numpy.array` [resolution, lower limit, upper limit]
    """
    if bias_correction:
        b_x = bias(true_x, reco_x)
        b_y = bias(true_y, reco_y)
    else:
        b_x = 0
        b_y = 0

    reco_x_corr = reco_x - b_x
    reco_y_corr = reco_y - b_y

    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.sort(((reco_x_corr - true_x) / relative_scaling(true_x, reco_x_corr, relative_scaling_method)) ** 2
                    + ((reco_y_corr - true_y) / relative_scaling(true_y, reco_y_corr, relative_scaling_method)) ** 2)
        res = np.nan_to_num(d)

    return np.sqrt(np.append(_percentile(res, percentile),
                             percentile_confidence_interval(res, percentile, confidence_level)))


def distance2d_resolution_per_bin(x, true_x, reco_x, true_y, reco_y,
                                  bins=10,
                                  percentile=68.27,
                                  confidence_level=0.95,
                                  bias_correction=False,
                                  relative_scaling_method=None,
                                  ):
    """
    Compute the 2D distance per bin of x

    Parameters
    ----------
    x: `numpy.ndarray`
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    true_x: `numpy.ndarray`
    true_y: `numpy.ndarray`
    bins: bins args of `np.histogram`
    percentile: float - percentile, 68.27 corresponds to one sigma
    confidence_level: float
    bias_correction: bool
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------
    x_bins, distance_res
    """

    # issue with numpy.digitize and astropy.Quantity: pass only values
    if isinstance(x, u.Quantity):
        if isinstance(bins, u.Quantity):
            bins = bins.to_value(x.unit)
        _, x_bins = np.histogram(x.value, bins=bins)
        bin_index = np.digitize(x.value, x_bins)
        x_bins *= x.unit
    else:
        _, x_bins = np.histogram(x, bins=bins)
        bin_index = np.digitize(x, x_bins)

    dist_res = []
    for ii in np.arange(1, len(x_bins)):
        mask = bin_index == ii
        dist_res.append(distance2d_resolution(true_x[mask], reco_x[mask],
                                              true_y[mask], reco_y[mask],
                                              percentile=percentile,
                                              confidence_level=confidence_level,
                                              bias_correction=bias_correction,
                                              relative_scaling_method=relative_scaling_method,
                                              )
                        )
    if isinstance(dist_res[0], u.Quantity):
        dist_res = u.Quantity(dist_res)
    else:
        dist_res = np.array(dist_res)

    return x_bins, dist_res


def bias_per_bin(true, reco, x, relative_scaling_method=None, bins=10):
    """
    Bias between `true` and `reco` per bin of `x`.

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`
    x: : `numpy.ndarray`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`
    bins: bins for `numpy.histogram`

    Returns
    -------
    bins, bias: `numpy.ndarray, numpy.ndarray`
    """
    _, x_bins = np.histogram(x, bins=bins)
    bin_index = np.digitize(x, x_bins)
    b = []
    for ii in np.arange(1, len(x_bins)):
        mask = bin_index == ii
        b.append(relative_bias(true[mask], reco[mask], relative_scaling_method=relative_scaling_method))

    b = u.Quantity(b) if isinstance(b[0], u.Quantity) else np.array(b)
    return x_bins, b



@u.quantity_input(energy=u.eV, bins=u.eV)
def bias_per_energy(true, reco, energy, relative_scaling_method=None, energy_bins=None):
    """
    Bias between `true` and `reco` per bins of true_energy

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`
    energy: : `astropy.Quantity`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`
    bins: `astropy.Quantity`

    Returns
    -------
    bins, bias: `astropy.Quantity`, `numpy.ndarray`
    """

    if energy_bins is None:
        irf = irf_cta()
        energy_bins = irf.energy_bins

    return bias_per_bin(true, reco, energy, relative_scaling_method=relative_scaling_method, bins=energy_bins)


def get_magic_sensitivity():
    """
    Load MAGIC differential sensitivity data from file `magic_sensitivity_2014.ecsv`.
    Extracted from table A.7 in Aleksić, Jelena, et al. "The major upgrade of the MAGIC telescopes,
    Part II: A performance study using observations of the Crab Nebula." Astroparticle Physics 72 (2016): 76-94.,
    DOI: 10.1016/j.astropartphys.2015.02.005'

    Returns
    -------
    `astropy.table.table.QTable`
    """
    return read(ds.get('magic_sensitivity_2014.ecsv'))


def gammaness_threshold_efficiency(gammaness, efficiency):
    """
    Compute the gammaness threshold required to get a given efficiency on a single category.
    The efficiency, or recall, is the number of correctly classified particle among true ones.

    Parameters
    ----------
    gammaness: `numpy.ndarray`
        gammaness of true events (e.g. gammas)
    efficiency: `float`
        between 0 and 1

    Returns
    -------
    treshold: `float`
        between 0 and 1
    """
    hist, edges = np.histogram(gammaness, bins=len(gammaness), range=(0, 1))
    relative_cum_hist = np.cumsum(hist[::-1])[::-1] / len(gammaness)
    threshold = edges[:-1][relative_cum_hist >= efficiency][-1]
    return threshold


def roc_auc_per_energy(true_type, gammaness, true_energy, energy_bins=None, gamma_label=0, **roc_auc_score_opt):
    """
    Compute AUC score as a function of the true gamma energy.
    The AUC score is calculated in a gamma versus all fashion.

    Parameters
    ----------
    true_type: `numpy.ndarray`
        labels
    gammaness: `numpy.ndarray`
        likeliness of a particle to be a gamma
    true_energy: `numpy.ndarray`
        particles true energy
    energy_bins: `astropy.Quantity`
    gamma_label: label of gammas in `true_type` array
    roc_auc_score_opt: see `sklearn.metrics.roc_auc_score` options

    Returns
    -------
    energy_bins, auc_scores: `numpy.ndarray, numpy.ndarray`
    """
    energy_bins = np.logspace(-2, 2, 10) * u.TeV if energy_bins is None else energy_bins

    binarized_label = (true_type == gamma_label).astype(int)

    auc_scores = []
    for i in range(len(energy_bins) - 1):
        gamma_mask = (true_type == gamma_label) & (true_energy >= energy_bins[i]) & (true_energy < energy_bins[i + 1])
        cosmic_mask = (true_type != gamma_label)
        mask = gamma_mask | cosmic_mask

        if np.count_nonzero(mask) > 0:
            auc_score = metrics.roc_auc_score(binarized_label[mask], gammaness[mask], **roc_auc_score_opt)
            if auc_score < 0.5:
                auc_score = 1 - auc_score
            auc_scores.append(auc_score)
        else:
            auc_scores.append(np.nan)

    return energy_bins, np.array(auc_scores)
