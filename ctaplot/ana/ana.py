"""
ana.py
======
Contain mathematical functions to make results analysis
(compute angular resolution, effective surface, true_energy resolution... )
"""
import numpy as np
from ..io import dataset as ds
from scipy.stats import binned_statistic, norm
from astropy.io.ascii import read

_relative_scaling_method = 's1'


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
           ]


class irf_cta:
    """
    Class to handle Instrument Response Function data
    """
    def __init__(self):
        self.site = ''
        self.E_bin = np.logspace(np.log10(2.51e-02), 2, 19)
        self.E = logbin_mean(self.E_bin)

        # Area of CTA sites in meters
        self.ParanalArea_prod3 = 19.63e6
        self.LaPalmaArea_prod3 = 11341149 #6.61e6

    def set_E_bin(self, E_bin):
        self.E_bin = E_bin
        self.E = logbin_mean(self.E_bin)


class cta_performance:
    def __init__(self, site):
        self.site = site
        self.E = np.empty(0)
        self.E_bin = np.empty(0)
        self.effective_area = np.empty(0)
        self.angular_resolution = np.empty(0)
        self.energy_resolution = np.empty(0)
        self.sensitivity = np.empty(0)

    def get_effective_area(self, observation_time=50):
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
        if self.site == 'south':
            if observation_time == 50:
                self.E, self.effective_area = np.loadtxt(
                    ds.get('CTA-Performance-prod3b-v2-South-20deg-50h-EffArea.txt'),
                    skiprows=11, unpack=True)
            if observation_time == 0.5:
                self.E, self.effective_area = np.loadtxt(
                    ds.get('CTA-Performance-prod3b-v2-North-20deg-30m-EffArea.txt'),
                    skiprows=11, unpack=True)

        if self.site == 'north':
            if observation_time == 50:
                self.E, self.effective_area = np.loadtxt(
                    ds.get('CTA-Performance-prod3b-v2-North-20deg-50h-EffArea.txt'),
                    skiprows=11, unpack=True)
            if observation_time == 0.5:
                self.E, self.effective_area = np.loadtxt(
                    ds.get('CTA-Performance-prod3b-v2-North-20deg-30m-EffArea.txt'),
                    skiprows=11, unpack=True)
        return self.E, self.effective_area

    def get_angular_resolution(self):
        if self.site == 'south':
            self.E, self.angular_resolution = np.loadtxt(
                ds.get('CTA-Performance-prod3b-v2-South-20deg-50h-Angres.txt'),
                skiprows=11, unpack=True)
        if self.site == 'north':
            self.E, self.angular_resolution = np.loadtxt(
                ds.get('CTA-Performance-prod3b-v2-North-20deg-50h-Angres.txt'),
                skiprows=11, unpack=True)

        return self.E, self.angular_resolution

    def get_energy_resolution(self):
        if self.site in ['south', 'paranal']:
            self.E, self.energy_resolution = np.loadtxt(ds.get('CTA-Performance-prod3b-v2-South-20deg-50h-Eres.txt'),
                                                        skiprows=11, unpack=True)
        if self.site in ['north', 'lapalma']:
            self.E, self.energy_resolution = np.loadtxt(ds.get('CTA-Performance-prod3b-v2-North-20deg-50h-Eres.txt'),
                                                        skiprows=11, unpack=True)

        return self.E, self.energy_resolution

    def get_sensitivity(self, observation_time=50):
        if self.site in ['south', 'paranal']:
            observation_times = {50: 'CTA-Performance-prod3b-v2-South-20deg-50h-DiffSens.txt',
                                 0.5: 'CTA-Performance-prod3b-v2-South-20deg-05h-DiffSens.txt',
                                 5: 'CTA-Performance-prod3b-v2-South-20deg-05h-DiffSens.txt'
            }
            Emin, Emax, self.sensitivity = np.loadtxt(ds.get(observation_times[observation_time]),
                                                  skiprows=10, unpack=True)
            self.E_bin = np.append(Emin, Emax[-1])
            self.E = logbin_mean(self.E_bin)

        if self.site in ['north', 'lapalma']:
            observation_times = {50: 'CTA-Performance-prod3b-v2-North-20deg-50h-DiffSens.txt',
                                 0.5: 'CTA-Performance-prod3b-v2-North-20deg-05h-DiffSens.txt',
                                 5: 'CTA-Performance-prod3b-v2-North-20deg-05h-DiffSens.txt'
            }
            Emin, Emax, self.sensitivity = np.loadtxt(ds.get(observation_times[observation_time]),
                                                  skiprows=10, unpack=True)
            self.E_bin = np.append(Emin, Emax[-1])
            self.E = logbin_mean(self.E_bin)

        return self.E, self.sensitivity



class cta_requirement:
    def __init__(self, site):
        self.site = site
        self.E = np.empty(0)
        self.effective_area = np.empty(0)
        self.angular_resolution = np.empty(0)
        self.energy_resolution = np.empty(0)
        self.sensitivity = np.empty(0)

    def get_effective_area(self, observation_time=50):
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
        if self.site == 'south':
            self.E, self.effective_area = np.loadtxt(ds.get('cta_requirements_South-30m-EffectiveArea.dat'),
                                                     unpack=True)
        if self.site == 'north':
            self.E, self.effective_area = np.loadtxt(ds.get('cta_requirements_North-30m-EffectiveArea.dat'),
                                                     unpack=True)

        return self.E, self.effective_area

    def get_angular_resolution(self):
        if self.site == 'south':
            self.E, self.angular_resolution = np.loadtxt(ds.get('cta_requirements_South-50h-AngRes.dat'), unpack=True)
        if self.site == 'north':
            self.E, self.angular_resolution = np.loadtxt(ds.get('cta_requirements_North-50h-AngRes.dat'), unpack=True)

        return self.E, self.angular_resolution

    def get_energy_resolution(self):
        if self.site in ['south', 'paranal']:
            self.E, self.energy_resolution = np.loadtxt(ds.get('cta_requirements_South-50h-ERes.dat'), unpack=True)
        if self.site in ['north', 'lapalma']:
            self.E, self.energy_resolution = np.loadtxt(ds.get('cta_requirements_North-50h-ERes.dat'), unpack=True)

        return self.E, self.energy_resolution

    def get_sensitivity(self, observation_time=50):
        if self.site in ['south', 'paranal']:
            self.E, self.sensitivity = np.loadtxt(ds.get('cta_requirements_South-50h.dat'), unpack=True)
        if self.site in ['north', 'lapalma']:
            self.E, self.sensitivity = np.loadtxt(ds.get('cta_requirements_North-50h.dat'), unpack=True)

        return self.E, self.sensitivity



def logspace_decades_nbin(Xmin, Xmax, n=5):
    """
    return an array with logspace and n bins / decade
    Parameters
    ----------
    Xmin: float
    Xmax: float
    n: int - number of bins per decade

    Returns
    -------
    1D Numpy array
    """
    ei = np.int(np.log10(Xmin))
    ea = np.int(np.floor(np.log10(Xmax)) + 1*(np.log10(Xmax) > np.floor(np.log10(Xmax))))
    return np.logspace(ei, ea, n * (ea-ei)+1)



def stat_per_energy(energy, y, statistic='mean'):
    """
    Return statistic for the given quantity per true_energy bins.
    The binning is given by irf_cta

    Parameters
    ----------
    energy: `numpy.ndarray` (1d)
        event energies
    y: `numpy.ndarray` (1d)
    statistic: string
        see `scipy.stat.binned_statistic`

    Returns
    -------
    `numpy.ndarray, numpy.ndarray, numpy.ndarray`
        bin_stat, bin_edges, binnumber
    """

    irf = irf_cta()

    bin_stat, bin_edges, binnumber = binned_statistic(energy, y, statistic=statistic, bins=irf.E)

    return bin_stat, bin_edges, binnumber


def bias(simu, reco):
    """
    Compute the bias of a reconstructed variable as `median(reco-simu)`

    Parameters
    ----------
    simu: `numpy.ndarray`
    reco: `numpy.ndarray`

    Returns
    -------
    float
    """
    assert len(simu) == len(reco), "both arrays should have the same size"
    if len(simu) == 0:
        return 0
    return np.median(reco - simu)


def relative_bias(simu, reco, relative_scaling_method='s1'):
    """
    Compute the relative bias of a reconstructed variable as
    `median(reco-simu)/relative_scaling(simu, reco)`

    Parameters
    ----------
    simu: `numpy.ndarray`
    reco: `numpy.ndarray`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------

    """
    assert len(reco) == len(simu)
    if len(simu) == 0:
        return 0
    return np.median((reco - simu) / relative_scaling(simu, reco, method=relative_scaling_method))


def relative_scaling(simu, reco, method='s0'):
    """
    Define the relative scaling for the relative error calculation.
    There are different ways to calculate this scaling factor.
    The easiest and most spread one is simply `np.abs(simu)`. However this is possible only when `simu != 0`.
    Possible methods:
        - None or 's0': scale = 1
        - 's1': `scale = np.abs(simu)`
        - 's2': `scale = np.abs(reco)`
        - 's3': `scale = (np.abs(simu) + np.abs(reco))/2.`
        - 's4': `scale = np.max([np.abs(reco), np.abs(simu)], axis=0)`

    This method is not exposed but kept for tests and future reference.
    The `s1` method is used in all `ctaplot` functions.

    Parameters
    ----------
    simu: `numpy.ndarray`
    reco: `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    method = 's0' if method is None else method
    scaling_methods = {
        's0': lambda simu, reco: np.ones(len(simu)),
        's1': lambda simu, reco: np.abs(simu),
        's2': lambda simu, reco: np.abs(reco),
        's3': lambda simu, reco: (np.abs(simu) + np.abs(reco))/2.,
        's4': lambda simu, reco: np.max([np.abs(reco), np.abs(simu)], axis=0)
    }

    return scaling_methods[method](simu, reco)


def resolution(simu, reco,
               percentile=68.27, confidence_level=0.95, bias_correction=False, relative_scaling_method='s1'):
    """
    Compute the resolution of reco as the Qth (68.27 as standard = 1 sigma) containment radius of
    `(simu-reco)/relative_scaling` with the lower and upper confidence limits defined the values inside
     the error_percentile

    Parameters
    ----------
    simu: `numpy.ndarray` (1d)
        simulated quantity
    reco: `numpy.ndarray` (1d)
        reconstructed quantity
    percentile: float
        percentile for the resolution containment radius
    error_percentile: float
        percentile for the confidence limits
    bias_correction: bool
        if True, the resolution is corrected with the bias computed on simu and reco
    relative_scaling: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------
    `numpy.ndarray` - [resolution, lower_confidence_limit, upper_confidence_limit]
    """
    assert len(simu) == len(reco), "both arrays should have the same size"

    b = bias(simu, reco) if bias_correction else 0

    with np.errstate(divide='ignore', invalid='ignore'):
        reco_corr = reco - b
        res = np.nan_to_num(np.abs((reco_corr - simu) /
                                   relative_scaling(simu, reco_corr, method=relative_scaling_method)))

    return np.append(_percentile(res, percentile), percentile_confidence_interval(res,
                                                                                  percentile=percentile,
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

    return x_bins, np.array(res)


def resolution_per_energy(simu, reco, simu_energy, percentile=68.27, confidence_level=0.95, bias_correction=False):
    """
    Parameters
    ----------
    simu: 1d `numpy.ndarray` of simulated energies
    reco: 1d `numpy.ndarray` of reconstructed energies

    Returns
    -------
    (energy_bins, resolution):
        energy_bins - 1D `numpy.ndarray`
        resolution: - 3D `numpy.ndarray` see `ctaplot.ana.resolution`
    """

    irf = irf_cta()
    return resolution_per_bin(simu_energy, simu, reco,
                              percentile=percentile,
                              confidence_level=confidence_level,
                              bias_correction=bias_correction,
                              bins=irf.E_bin)


def energy_resolution(true_energy, reco_energy, percentile=68.27, confidence_level=0.95, bias_correction=False):
    """
    Compute the true_energy resolution of true_energy as the percentile (68 as standard) containment radius of
    `true_energy-true_energy)/simu_energy
    with the lower and upper confidence limits defined by the given confidence level

    Parameters
    ----------
    true_energy: 1d numpy array of simulated energies
    reco_energy: 1d numpy array of reconstructed energies
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


def energy_resolution_per_energy(simu_energy, reco_energy,
                                 percentile=68.27, confidence_level=0.95, bias_correction=False):
    """
    The energy resolution ΔE / E is obtained from the distribution of (ER – ET) / ET, where R and T refer
    to the reconstructed and true energies of gamma-ray events.
     ΔE/E is the half-width of the interval around 0 which contains given percentile of the distribution.

    Parameters
    ----------
    simu_energy: 1d numpy array of simulated energies
    reco_energy: 1d numpy array of reconstructed energies
    percentile: float
        between 0 and 100
    confidence_level: float
        between 0 and 1
    bias_correction: bool

    Returns
    -------
    (e, e_res) : tuple of 1d numpy arrays - true_energy, resolution in true_energy
    """
    assert len(reco_energy) > 0, "Empty arrays"

    res_e = []
    irf = irf_cta()
    for i, e in enumerate(irf.E):
        mask = (reco_energy > irf.E_bin[i]) & (reco_energy < irf.E_bin[i + 1])
        res_e.append(energy_resolution(simu_energy[mask], reco_energy[mask],
                                       percentile=percentile,
                                       confidence_level=confidence_level,
                                       bias_correction=bias_correction))

    return irf.E_bin, np.array(res_e)


def energy_bias(simu_energy, reco_energy):
    """
    Compute the true_energy relative bias per true_energy bin.

    Parameters
    ----------
    simu_energy: 1d numpy array of simulated energies
    reco_energy: 1d numpy array of reconstructed energies

    Returns
    -------
    (energy_bins, bias) : tuple of 1d numpy arrays - true_energy, true_energy bias
    """
    bias_e = []
    irf = irf_cta()
    for i, e in enumerate(irf.E):
        mask = (reco_energy > irf.E_bin[i]) & (reco_energy < irf.E_bin[i + 1])
        bias_e.append(relative_bias(simu_energy[mask], reco_energy[mask], relative_scaling_method='s1'))

    return irf.E_bin, np.array(bias_e)


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


def get_angles_0pi(angles):
    """
    return angles modulo between 0 and +pi

    Parameters
    ----------
    angles: `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    return np.mod(angles, np.pi)


def theta2(reco_alt, reco_az, simu_alt, simu_az, bias_correction=False):
    """
    Compute the theta2 in radians

    Parameters
    ----------
    reco_alt: 1d `numpy.ndarray` - reconstructed Altitude in radians
    reco_az: 1d `numpy.ndarray` - reconstructed Azimuth in radians
    simu_alt: 1d `numpy.ndarray` - true Altitude in radians
    simu_az: 1d `numpy.ndarray` -  true Azimuth in radians

    Returns
    -------
    1d `numpy.ndarray`
    """
    assert (len(reco_az) == len(reco_alt))
    assert (len(reco_alt) == len(simu_alt))
    if len(reco_alt) == 0:
        return np.empty(0)
    if bias_correction:
        bias_alt = bias(simu_alt, reco_alt)
        bias_az = bias(simu_az, reco_az)
    else:
        bias_alt = 0
        bias_az = 0
    return angular_separation_altaz(reco_alt-bias_alt, reco_az-bias_az, simu_alt, simu_az) ** 2


def angular_resolution(reco_alt, reco_az, simu_alt, simu_az,
                       percentile=68.27, confidence_level=0.95, bias_correction=False):
    """
    Compute the angular resolution as the Qth (standard being 68)
    containment radius of theta2 with lower and upper limits on this value
    corresponding to the confidence value required (1.645 for 95% confidence)

    Parameters
    ----------
    reco_alt: `numpy.ndarray` - reconstructed altitude angle in radians
    reco_az: `numpy.ndarray` - reconstructed azimuth angle in radians
    simu_alt: `numpy.ndarray` - true altitude angle in radians
    simu_az: `numpy.ndarray` - true azimuth angle in radians
    percentile: float - percentile, 68 corresponds to one sigma
    confidence_level: float

    Returns
    -------
    `numpy.array` [angular_resolution, lower limit, upper limit]
    """
    if bias_correction:
        b_alt = bias(simu_alt, reco_alt)
        b_az = bias(simu_az, reco_az)
    else:
        b_alt = 0
        b_az = 0

    reco_alt_corr = reco_alt - b_alt
    reco_az_corr = reco_az - b_az

    t2 = np.sort(theta2(reco_alt_corr, reco_az_corr, simu_alt, simu_az))

    ang_res = _percentile(t2, percentile)
    return np.sqrt(np.append(ang_res, percentile_confidence_interval(t2, percentile, confidence_level)))


def angular_resolution_per_bin(simu_alt, simu_az, reco_alt, reco_az, x,
                               percentile=68.27, confidence_level=0.95, bias_correction=False, bins=10):
    """
    Compute the angular resolution per binning of x

    Parameters
    ----------
    simu_alt: `numpy.ndarray`
    simu_az: `numpy.ndarray`
    reco_alt: `numpy.ndarray`
    reco_az: `numpy.ndarray`
    x: `numpy.ndarray`
    percentile: float
        0 < percentile < 100
    confidence_level: float
        0 < confidence_level < 1
    bias_correction: bool
    bins: int or `numpy.ndarray`

    Returns
    -------
    bins, ang_res:
        bins: 1D `numpy.ndarray`
        ang_res: 2D `numpy.ndarray`
    """
    _, x_bins = np.histogram(x, bins=bins)
    bin_index = np.digitize(x, x_bins)

    ang_res = []
    for ii in np.arange(1, len(x_bins)):
        mask = bin_index == ii
        ang_res.append(angular_resolution(reco_alt[mask], reco_az[mask],
                                          simu_alt[mask], simu_az[mask],
                                          percentile=percentile,
                                          confidence_level=confidence_level,
                                          bias_correction=bias_correction,
                                          )
                       )

    return x_bins, np.array(ang_res)


def angular_resolution_per_energy(reco_alt, reco_az, simu_alt, simu_az, energy,
                                  percentile=68.27, confidence_level=0.95, bias_correction=False):
    """
    Plot the angular resolution as a function of the event simulated true_energy

    Parameters
    ----------
    reco_alt: `numpy.ndarray`
    reco_az: `numpy.ndarray`
    simu_alt: `numpy.ndarray`
    simu_az: `numpy.ndarray`
    energy: `numpy.ndarray`
    **kwargs: args for `angular_resolution`

    Returns
    -------
    (E, RES) : (1d numpy array, 1d numpy array) = Energies, Resolution
    """
    assert len(reco_alt) == len(energy)
    assert len(energy) > 0, "Empty arrays"

    irf = irf_cta()

    E_bin = irf.E_bin
    RES = []

    for i, e in enumerate(E_bin[:-1]):
        mask = (energy > E_bin[i]) & (energy <= E_bin[i + 1])
        RES.append(angular_resolution(reco_alt[mask], reco_az[mask], simu_alt[mask], simu_az[mask],
                                      percentile=percentile,
                                      confidence_level=confidence_level,
                                      bias_correction=bias_correction,
                                      )
                   )

    return E_bin, np.array(RES)


def angular_resolution_per_off_pointing_angle(simu_alt, simu_az, reco_alt, reco_az, alt_pointing, az_pointing, bins=10):
    """
    Compute the angular resolution as a function of separation angle for the pointing direction

    Parameters
    ----------
    simu_alt: `numpy.ndarray`
    simu_az: `numpy.ndarray`
    reco_alt: `numpy.ndarray`
    reco_az: `numpy.ndarray`
    alt_pointing: `numpy.ndarray`
    az_pointing: `numpy.ndarray`
    bins: float or `numpy.ndarray`

    Returns
    -------
    (bins, res):
        bins: 1D `numpy.ndarray`
        res: 2D `numpy.ndarray` - resolutions with confidence intervals (output from `ctaplot.ana.resolution`)
    """
    ang_sep_to_pointing = angular_separation_altaz(simu_alt, simu_az, alt_pointing, az_pointing)

    return angular_resolution_per_bin(simu_alt, simu_az, reco_alt, reco_az, ang_sep_to_pointing, bins=bins)


def effective_area(simu_energy, reco_energy, simu_area):
    """
    Compute the effective area from a list of simulated energies and reconstructed energies
    Parameters
    ----------
    simu_energy: 1d numpy array
    reco_energy: 1d numpy array
    simu_area: float - area on which events are simulated
    Returns
    -------
    float = effective area
    """
    return simu_area * len(reco_energy) / len(simu_energy)


def effective_area_per_energy(simu_energy, reco_energy, simu_area):
    """
    Compute the effective area per true_energy bins from a list of simulated energies and reconstructed energies

    Parameters
    ----------
    simu_energy: 1d numpy array
    reco_energy: 1d numpy array
    simu_area: float - area on which events are simulated

    Returns
    -------
    (E, Seff) : (1d numpy array, 1d numpy array)
    """

    irf = irf_cta()

    count_R, bin_R = np.histogram(reco_energy, bins=irf.E_bin)
    count_S, bin_S = np.histogram(simu_energy, bins=irf.E_bin)

    np.seterr(divide='ignore', invalid='ignore')
    return irf.E_bin, np.nan_to_num(simu_area * count_R / count_S)


def impact_parameter_error(reco_x, reco_y, simu_x, simu_y):
    """
    compute the error distance between simulated and reconstructed impact parameters
    Parameters
    ----------
    reco_x: 1d numpy array
    reco_y: 1d numpy array
    simu_x: 1d numpy array
    simu_y: 1d numpy array

    Returns
    -------
    1d numpy array: distances
    """
    return np.sqrt((reco_x - simu_x) ** 2 + (reco_y - simu_y) ** 2)


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
    if len(x) == 0:
        return 0
    else:
        return np.percentile(x, percentile)


def angular_separation_altaz(alt1, az1, alt2, az2, unit='rad'):
    """
    Compute the angular separation in radians or degrees
    between two pointing direction given with alt-az

    Parameters
    ----------
    alt1: 1d `numpy.ndarray`, altitude of the first pointing direction
    az1: 1d `numpy.ndarray` azimuth of the first pointing direction
    alt2: 1d `numpy.ndarray`, altitude of the second pointing direction
    az2: 1d `numpy.ndarray`, azimuth of the second pointing direction
    unit: 'deg' or 'rad'

    Returns
    -------
    1d `numpy.ndarray` or float, angular separation
    """
    if unit == 'deg':
        alt1 = np.radians(alt1)
        az1 = np.radians(az1)
        alt2 = np.radians(alt2)
        az2 = np.radians(az2)

    cosdelta = np.cos(alt1) * np.cos(alt2) * np.cos(az1-az2) + np.sin(alt1) * np.sin(alt2)
    cosdelta[cosdelta > 1] = 1.
    cosdelta[cosdelta < -1] = -1.

    ang_sep = np.degrees(np.arccos(cosdelta)) if unit == 'deg' else np.arccos(cosdelta)

    return ang_sep


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
    return 10 ** ((np.log10(x_bin[:-1]) + np.log10(x_bin[1:])) / 2.)


def impact_resolution(reco_x, reco_y, simu_x, simu_y,
                      percentile=68.27, confidence_level=0.95, bias_correction=False, relative_scaling_method=None):
    """
    Compute the shower impact parameter resolution as the Qth (68 as standard) containment radius of the square distance
    to the simulated one with the lower and upper limits corresponding to the required confidence level

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    simu_x: `numpy.ndarray`
    simu_y: `numpy.ndarray`
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

    return distance2d_resolution(reco_x, reco_y, simu_x, simu_y,
                                 percentile=percentile,
                                 confidence_level=confidence_level,
                                 bias_correction=bias_correction,
                                 relative_scaling_method=relative_scaling_method
                                 )


def impact_resolution_per_energy(reco_x, reco_y, simu_x, simu_y, energy,
                                 percentile=68.27,
                                 confidence_level=0.95,
                                 bias_correction=False,
                                 relative_scaling_method=None):
    """
    Plot the angular resolution as a function of the event simulated true_energy

    Parameters
    ----------
    reco_x: `numpy.ndarray`
    reco_y: `numpy.ndarray`
    simu_x: `numpy.ndarray`
    simu_y: `numpy.ndarray`
    energy: `numpy.ndarray`
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
    (true_energy, resolution) : (1d numpy array, 1d numpy array)
    """
    assert len(reco_x) == len(energy)
    assert len(energy) > 0, "Empty arrays"

    irf = irf_cta()

    return distance2d_resolution_per_bin(energy, reco_x, reco_y, simu_x, simu_y,
                                         bins=irf.E_bin,
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
    j = np.max([0, np.int(len(x) * q - z * np.sqrt(len(x) * q * (1 - q)))])
    k = np.min([np.int(len(x) * q + z * np.sqrt(len(x) * q * (1 - q))), len(x) - 1])
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


def effective_area_per_energy_power_law(emin, emax, total_number_events, spectral_index, reco_energy, simu_area):
    """
    Compute the effective area per true_energy bins from a list of simulated energies and reconstructed energies

    Parameters
    ----------
    emin: float
    emax: float
    total_number_events: int
    spectral_index: float
    reco_energy: 1d numpy array
    simu_area: float - area on which events are simulated

    Returns
    -------
    (true_energy, effective_area) : (1d numpy array, 1d numpy array)
    """

    irf = irf_cta()
    bins = irf.E_bin
    simu_per_bin = power_law_integrated_distribution(emin, emax, total_number_events, spectral_index, bins)
    count_R, bin_R = np.histogram(reco_energy, bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        return bins, np.nan_to_num(simu_area * count_R / simu_per_bin)


def distance2d_resolution(reco_x, reco_y, simu_x, simu_y,
                          percentile=68.27, confidence_level=0.95, bias_correction=False, relative_scaling_method=None):
    """
    Compute the 2D distance resolution as the Qth (standard being 68)
    containment radius of the relative distance with lower and upper limits on this value
    corresponding to the confidence value required (1.645 for 95% confidence)

    Parameters
    ----------
    reco_x: `numpy.ndarray` - reconstructed x position
    reco_y: `numpy.ndarray` - reconstructed y position
    simu_x: `numpy.ndarray` - true x position
    simu_y: `numpy.ndarray` - true y position
    percentile: float - percentile, 68.27 corresponds to one sigma
    confidence_level: float
    bias_correction: bool
    relative_scaling_method: str
        - see `ctaplot.ana.relative_scaling`

    Returns
    -------
    `numpy.array` [angular_resolution, lower limit, upper limit]
    """
    if bias_correction:
        b_x = bias(simu_x, reco_x)
        b_y = bias(simu_y, reco_y)
    else:
        b_x = 0
        b_y = 0

    reco_x_corr = reco_x - b_x
    reco_y_corr = reco_y - b_y

    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.sort(((reco_x_corr - simu_x)/relative_scaling(simu_x, reco_x_corr, relative_scaling_method)) ** 2
                    + ((reco_y_corr - simu_y)/relative_scaling(simu_y, reco_y_corr, relative_scaling_method)) ** 2)
        res = np.nan_to_num(d)

    return np.sqrt(np.append(_percentile(res, percentile),
                             percentile_confidence_interval(res, percentile, confidence_level)))


def distance2d_resolution_per_bin(x, reco_x, reco_y, simu_x, simu_y,
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
    simu_x: `numpy.ndarray`
    simu_y: `numpy.ndarray`
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

    _, x_bins = np.histogram(x, bins=bins)
    bin_index = np.digitize(x, x_bins)

    dist_res = []
    for ii in np.arange(1, len(x_bins)):
        mask = bin_index == ii
        dist_res.append(distance2d_resolution(reco_x[mask], reco_y[mask],
                                              simu_x[mask], simu_y[mask],
                                              percentile=percentile,
                                              confidence_level=confidence_level,
                                              bias_correction=bias_correction,
                                              relative_scaling_method=relative_scaling_method,
                                              )
                       )

    return x_bins, np.array(dist_res)


def bias_per_bin(simu, reco, x, relative_scaling_method=None, bins=10):
    """
    Bias between `simu` and `reco` per bin of `x`.

    Parameters
    ----------
    simu: `numpy.ndarray`
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
        b.append(relative_bias(simu[mask], reco[mask], relative_scaling_method=relative_scaling_method))

    return x_bins, np.array(b)


def bias_per_energy(simu, reco, energy, relative_scaling_method=None):
    """
    Bias between `simu` and `reco` per bins of true_energy

    Parameters
    ----------
    simu: `numpy.ndarray`
    reco: `numpy.ndarray`
    energy: : `numpy.ndarray`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------
    bins, bias: `numpy.ndarray, numpy.ndarray`
    """

    irf = irf_cta()
    energy_bin = irf.E_bin

    return bias_per_bin(simu, reco, energy, relative_scaling_method=relative_scaling_method, bins=energy_bin)


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