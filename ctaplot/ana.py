"""
ana.py
======
Contain mathematical functions to make results analysis (compute angular resolution, effective surface, energy resolution... )
"""


import numpy as np
import ctaplot.dataset as ds


class irf_cta:
    """
    Class to handle Instrument Response Function data
    """
    def __init__(self):
        self.site = ''
        self.E_bin = np.logspace(np.log10(2.51e-02), 2, 19)
        self.E = logbin_mean(self.E_bin)

        # Area of CTA sites in meters
        self.ParanalArea = 19.63e6
        self.LaPalmaArea = 6.61e6

    def set_E_bin(self, E_bin):
        self.E_bin = E_bin
        self.E = logbin_mean(self.E_bin)


class cta_performances:
    def __init__(self):
        self.site = ''
        self.E = np.empty(0)
        self.effective_area = np.empty(0)
        self.angular_resolution = np.empty(0)
        self.energy_resolution = np.empty(0)
        self.sensitivity = np.empty(0)

    def get_effective_area(self, observation_time=50):
        """
        Return the effective area at the given observation time in hours.
        NB: Only 50h supported
        Returns the energy array and the effective area array
        Parameters
        ----------
        observation_time: optional

        Returns
        -------
        `numpy.ndarray`, `numpy.ndarray`
        """
        if self.site == 'south':
            if observation_time == 50:
                self.E, self.effective_area = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-South-20deg-50h-EffArea.txt'),
                                                     skiprows=11, unpack=True)
            if observation_time == 0.5:
                self.E, self.effective_area = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-North-20deg-30m-EffArea.txt'),
                                                     skiprows=11, unpack=True)

        if self.site == 'north':
            if observation_time == 50:
                self.E, self.effective_area = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-North-20deg-50h-EffArea.txt'),
                                                     skiprows=11, unpack=True)
            if observation_time == 0.5:
                self.E, self.effective_area = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-North-20deg-30m-EffArea.txt'),
                                                     skiprows=11, unpack=True)
        return self.E, self.effective_area

    def get_angular_resolution(self):
        if self.site == 'south':
            self.E, self.angular_resolution = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-South-20deg-50h-Angres.txt'),
                                                         skiprows=11, unpack=True)
        if self.site == 'north':
            self.E, self.angular_resolution = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-North-20deg-50h-Angres.txt'),
                                                         skiprows=11, unpack=True)

        return self.E, self.angular_resolution

    def get_energy_resolution(self):
        if self.site in ['south', 'paranal']:
            self.E, self.energy_resolution = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-South-20deg-50h-Eres.txt'),
                                                        skiprows=11, unpack=True)
        if self.site in ['north', 'lapalma']:
            self.E, self.energy_resolution = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-North-20deg-50h-Eres.txt'),
                                                        skiprows=11, unpack=True)

        return self.E, self.energy_resolution

    def get_sensitivity(self, observation_time=50):
        if self.site in ['south', 'paranal']:
            Emin, Emax, self.sensitivity = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-South-20deg-50h-DiffSens.txt'),
                                                  skiprows=10, unpack=True)
            self.E = logbin_mean(np.append(Emin, Emax[-1]))

        if self.site in ['north', 'lapalma']:
            Emin, Emax, self.sensitivity = np.loadtxt(ds.get('CTA-Performance-prod3b-v1-North-20deg-50h-DiffSens.txt'),
                                                  skiprows=10, unpack=True)
            self.E = logbin_mean(np.append(Emin, Emax[-1]))

        return self.E, self.sensitivity



class cta_requirements:
    def __init__(self):
        self.site = ''
        self.E = np.empty(0)
        self.effective_area = np.empty(0)
        self.angular_resolution = np.empty(0)
        self.energy_resolution = np.empty(0)
        self.sensitivity = np.empty(0)

    def get_effective_area(self, observation_time=50):
        """
        Return the effective area at the given observation time in hours.
        NB: Only 0.5h supported
        Returns the energy array and the effective area array
        Parameters
        ----------
        observation_time: optional

        Returns
        -------
        `numpy.ndarray`, `numpy.ndarray`
        """
        if self.site == 'south':
            self.E, self.effective_area = np.loadtxt(ds.get('cta_requirements_South-30m-EffectiveArea.dat'), unpack=True)
        if self.site == 'north':
            self.E, self.effective_area = np.loadtxt(ds.get('cta_requirements_North-30m-EffectiveArea.dat'), unpack=True)

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
    ei = np.floor(np.log10(Xmin))
    ea = np.floor(np.log10(Xmax)) + 1*(np.log10(Xmax) > np.floor(np.log10(Xmax)))
    return np.logspace(ei, ea, n * (ea-ei)+1)



def multiplicity_stat_per_energy(Multiplicity, Energies, p = 50):
    """
    Return the mean, min, max, percentile telescope multiplicity per energy.
    Per default, percentile is 50 (= median)

    Parameters
    ----------
    Multiplicity: 1D Numpy array of integers
    Energy: 1D Numpy array of floats, event energies corresponding to the multiplicities

    Returns
    -------
    (E, mean, min, max, percentile): tuple of 1D Numpy arrays, Energy, Mean Multiplicity, Min Multiplicity, Max Multplicity, Percentile Multiplicity
    """

    m_mean = []
    m_min = []
    m_max = []
    m_per = []

    irf = irf_cta()

    for i, e in enumerate(irf.E):
        mask = (Energies > irf.E_bin[i]) & (Energies < irf.E_bin[i+1])
        if len(Multiplicity[mask]) > 0:
            m_mean.append(Multiplicity[mask].mean())
            m_min.append(Multiplicity[mask].min())
            m_max.append(Multiplicity[mask].max())
            m_per.append(np.percentile[mask])
        else:
            m_mean.append(0)
            m_max.append(0)
            m_min.append(0)
            m_per.append(0)

    return irf.E, m_mean, m_min, m_max, m_per


def bias(simu, reco):
    """
    Compute the bias of a reconstructed variable.

    Parameters
    ----------
    simu: `numpy.ndarray`
    reco: `numpy.ndarray`

    Returns
    -------
    float
    """
    assert len(simu) == len(reco), "both arrays should have the same size"
    res = (reco - simu) / reco
    return np.median(res)



def energy_res(SimuE, RecoE, Q=68, bias_correction=False):
    """
    Compute the energy resolution of RecoE as the Qth (68 as standard) containment radius of DeltaE/E
    with the lower and upper confidence limits

    Parameters
    ----------
    SimuE: 1d numpy array of simulated energies
    RecoE: 1d numpy array of reconstructed energies

    Returns
    -------
    `numpy.array` - [energy_resolution, lower_confidence_limit, upper_confidence_limit]
    """
    assert len(SimuE) == len(RecoE), "both arrays should have the same size"

    biasE = 0
    if bias_correction:
        biasE = bias(SimuE, RecoE)

    resE = np.abs((RecoE - SimuE) / RecoE - biasE)
    return np.append(RQ(resE, Q), percentile_confidence_interval(resE, Q=Q))


def energy_res_per_energy(SimuE, RecoE, bias_correction=False):
    """

    Parameters
    ----------
    SimuE: 1d numpy array of simulated energies
    RecoE: 1d numpy array of reconstructed energies

    Returns
    -------
    (e, e_res) : tuple of 1d numpy arrays - energy, resolution in energy
    """
    resE = []
    irf = irf_cta()
    for i, e in enumerate(irf.E):
        mask = (SimuE > irf.E_bin[i]) & (SimuE < irf.E_bin[i+1])
        resE.append(energy_res(SimuE[mask], RecoE[mask], bias_correction=bias_correction))


    return irf.E_bin, np.array(resE)


def energy_bias(SimuE, RecoE):
    """
    Compute the energy bias per energy bin.
    Parameters
    ----------
    SimuE: 1d numpy array of simulated energies
    RecoE: 1d numpy array of reconstructed energies

    Returns
    -------
    (e, biasE) : tuple of 1d numpy arrays - energy, energy bias
    """
    biasE = []
    irf = irf_cta()
    for i, e in enumerate(irf.E):
        mask = (SimuE > irf.E_bin[i]) & (SimuE < irf.E_bin[i+1])
        biasE.append(bias(SimuE[mask], RecoE[mask]))

    return irf.E_bin, np.array(biasE)



def get_angles_pipi(angles):
    return np.mod(angles+pi, 2*pi) - pi


def get_angles_0pi(angles):
    return np.mod(angles, pi)


def theta2(RecoAlt, RecoAz, AltSource, AzSource):
    """
    Compute the theta2 in radians

    Parameters
    ----------
    RecoAlt: 1d `numpy.ndarray` - reconstructed Altitude in radians
    RecoAz: 1d `numpy.ndarray` - reconstructed Azimuth in radians
    AltSource: 1d `numpy.ndarray` - true Altitude in radians
    AzSource: 1d `numpy.ndarray` -  true Azimuth in radians

    Returns
    -------
    1d `numpy.ndarray`
    """
    assert (len(RecoAz) == len(RecoAlt))
    assert (len(RecoAlt) == len(AltSource))
    if len(RecoAlt) == 0:
        return np.empty(0)
    else:
        return angular_separation_altaz(RecoAlt, RecoAz, AltSource, AzSource)**2


def angular_resolution(RecoAlt, RecoAz, SimuAlt, SimuAz, Q = 68, conf=1.645):
    """
    Compute the angular resolution as the Qth (standard being 68)
    containment radius of theta2 with lower and upper limits on this value
    corresponding to the confidence value required (1.645 for 95% confidence)

    Parameters
    ----------
    RecoAlt: `numpy.ndarray`
    RecoAz: `numpy.ndarray`
    SimuAlt: `numpy.ndarray`
    SimuAz: `numpy.ndarray`
    Q: percentile - 68 corresponds to one sigma
    conf:

    Returns
    -------
    `numpy.array` [angular_resolution, lower limit, upper limit]
    """
    t2 = np.sort(theta2(RecoAlt, RecoAz, SimuAlt, SimuAz))

    ang_res = RQ(t2, Q)
    percentile_confidence_interval(t2, Q, conf)
    return np.sqrt(np.append(ang_res, percentile_confidence_interval(t2, Q, conf)))


def angular_resolution_per_energy(RecoAlt, RecoAz, SimuAlt, SimuAz, Energy, **kwargs):
    """
    Plot the angular resolution as a function of the event simulated energy

    Parameters
    ----------
    RecoAlt: `numpy.ndarray`
    RecoAz: `numpy.ndarray`
    SimuAlt: `numpy.ndarray`
    SimuAz: `numpy.ndarray`
    Energy: `numpy.ndarray`
    **kwargs: args for `angular_resolution`

    Returns
    -------
    (E, RES) : (1d numpy array, 1d numpy array) = Energies, Resolution
    """
    assert len(RecoAlt) == len(Energy)
    assert len(Energy) > 0, "Empty arrays"

    irf = irf_cta()

    E_bin = irf.E_bin
    RES = []

    for i, e in enumerate(E_bin[:-1]):
        mask = (Energy > E_bin[i]) & (Energy <= E_bin[i+1])
        RES.append(angular_resolution(RecoAlt[mask], RecoAz[mask], SimuAlt[mask], SimuAz[mask], **kwargs))

    return E_bin, np.array(RES)


def effective_area(SimuE, RecoE, simuArea):
    """
    Compute the effective area from a list of simulated energies and reconstructed energies
    Parameters
    ----------
    SimuE: 1d numpy array
    RecoE: 1d numpy array
    simuArea: float - area on which events are simulated
    Returns
    -------
    float = effective area
    """
    return simuArea * len(RecoE)/len(SimuE)


def effective_area_per_energy(SimuE, RecoE, simuArea):
    """
    Compute the effective area per energy bins from a list of simulated energies and reconstructed energies

    Parameters
    ----------
    SimuE: 1d numpy array
    RecoE: 1d numpy array
    simuArea: float - area on which events are simulated

    Returns
    -------
    (E, Seff) : (1d numpy array, 1d numpy array)
    """

    irf = irf_cta()

    count_R, bin_R = np.histogram(RecoE, bins=irf.E_bin)
    count_S, bin_S = np.histogram(SimuE, bins=irf.E_bin)

    np.seterr(divide='ignore', invalid='ignore')
    return irf.E_bin, np.nan_to_num(simuArea * count_R/count_S)


def mask_range(X, Xmin=0, Xmax=np.inf):
    """
    create a mask for X to get values between Xmin and Xmax
    Parameters
    ----------
    X: 1d numpy array
    Xmin: float
    Xmax: float

    Returns
    -------
    1d numpy array of boolean
    """
    mask = (X > Xmin) & (X < Xmax)
    return mask


def angles_modulo_degrees(RecoAlt, RecoAz, SimuAlt, SimuAz):
    RecoAlt2 = np.degrees(get_angles_0pi(RecoAlt))
    RecoAz2 = np.degrees(get_angles_pipi(RecoAz))
    AltSource = np.degrees(get_angles_0pi(SimuAlt[0]))
    AzSource = np.degrees(get_angles_pipi(SimuAz[0]))
    return RecoAlt2, RecoAz2, AltSource, AzSource


def impact_parameter_error(RecoX, RecoY, SimuX, SimuY):
    """
    compute the error distance between simulated and reconstructed impact parameters
    Parameters
    ----------
    RecoX: 1d numpy array
    RecoY
    SimuX
    SimuY

    Returns
    -------
    1d numpy array: distances
    """
    return np.sqrt((RecoX-SimuX)**2 + (RecoY-SimuY)**2)


def RQ(x, Q=68):
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
        return np.percentile(x, Q)


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
    if unit=='deg':
        alt1 = np.radians(alt1)
        az1 = np.radians(az1)
        alt2 = np.radians(alt2)
        az2 = np.radians(az2)

    cosdelta = np.cos(alt1) * np.cos(alt2) * np.cos(az1-az2) \
                + np.sin(alt1) * np.sin(alt2)
    cosdelta[cosdelta > 1] = 1.
    cosdelta[cosdelta < -1] = -1.

    ang_sep = np.degrees(np.arccos(cosdelta)) if unit=='deg' \
                else np.arccos(cosdelta)

    return ang_sep


def logbin_mean(E_bin):
    """
    Function that gives back the mean of each bin in logscale

    Parameters
    ----------
    E_bin: `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    return 10 ** ((np.log10(E_bin[:-1]) + np.log10(E_bin[1:])) / 2.)


def impact_resolution(RecoX, RecoY, SimuX, SimuY, Q=68, conf=1.645):
    """
    Compute the shower impact parameter resolution as the Qth (68 as standard) containment radius of the square distance
    to the simulated one
    with the lower and upper limits corresponding to the required confidence level (1.645 for 95%)

    Parameters
    ----------
    RecoX: `numpy.ndarray`
    RecoY: `numpy.ndarray`
    SimuX: `numpy.ndarray`
    SimuY: `numpy.ndarray`
    conf: `float`

    Returns
    -------
    `numpy.array` - [impact_resolution, lower_limit, upper_limit]
    """
    d2 = impact_parameter_error(RecoX, RecoY, SimuX, SimuY)**2
    return np.sqrt(np.append(RQ(d2, 68), percentile_confidence_interval(d2, Q=68, conf=conf)))


def impact_resolution_per_energy(RecoX, RecoY, SimuX, SimuY, Energy, Q=68, conf=1.645):
    """
    Plot the angular resolution as a function of the event simulated energy

    Parameters
    ----------
    RecoX: `numpy.ndarray`
    RecoY: `numpy.ndarray`
    SimuX: `numpy.ndarray`
    SimuY: `numpy.ndarray`
    Energy: `numpy.ndarray`

    Returns
    -------
    (E, RES) : (1d numpy array, 1d numpy array) = Energies, Resolution
    """
    assert len(RecoX) == len(Energy)
    assert len(Energy) > 0, "Empty arrays"

    irf = irf_cta()

    E_bin = irf.E_bin
    RES = []

    for i, e in enumerate(E_bin[:-1]):
        mask = (Energy > E_bin[i]) & (Energy <= E_bin[i+1])
        RES.append(impact_resolution(RecoX[mask], RecoY[mask], SimuX[mask], SimuY[mask], Q=Q, conf=conf))

    return E_bin, np.array(RES)


def percentile_confidence_interval(X, Q=68, conf=1.645):
    """
    Return the confidence interval for the qth percentile of X
    conf=1.96 corresponds to a 95% confidence interval for a normal distribution
    One can obtain another confidence coefficient thanks to `scipy.stats.norm.ppf`

    REF:
    http://people.stat.sfu.ca/~cschwarz/Stat-650/Notes/PDF/ChapterPercentiles.pdf
    S. Chakraborti and J. Li, Confidence Interval Estimation of a Normal Percentile, doi:10.1198/000313007X244457

    Parameters
    ----------
    X: `numpy.array`
    Q: `float` - percentile (between 0 and 100)
    conf: `float` - confidence

    Returns
    -------

    """
    sort_X = np.sort(X)
    if len(X)==0:
        return (0, 0)
    q = Q / 100.
    j = np.max([0, np.int(len(X) * q - conf * np.sqrt(len(X) * q * (1 - q)))])
    k = np.min([np.int(len(X) * q + conf * np.sqrt(len(X) * q * (1 - q))), len(X) - 1])
    return sort_X[j], sort_X[k]
