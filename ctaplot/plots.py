import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
from scipy.stats import gaussian_kde
from numpy import floor
from math import inf

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


class plot_from_anadata:
    def __init__(self, anadata):
        """
        init class with a AnaData class object
        Parameters
        ----------
        anadata: AnaData class object
        """
        self.a = anadata
        self.SimuArea = 1

        self.outdir = 'anaplots/'
        self.format = 'png'
        self.dpi = 200
        if not os.path.isdir(self.outdir): os.mkdir(self.outdir)

        if len(anadata.TelTypes) > 0:
            if len(set(np.concatenate(anadata.TelTypes))) > 2:
                self.set_site('south')
            else:
                self.set_site('north')
        else:
            print("Can't guess the site from its layout, please precise with p.set_site()")

    def set_outdir(self, dir_name):
        """
        set new directory for plots and create it
        Parameters
        ----------
        dir_name: string
        """
        self.outdir = dir_name + '/'
        if not os.path.isdir(dir_name): os.mkdir(dir_name)

    def set_site(self, site):
        """
        set the observation site: 'south' or 'north'
        Parameters
        ----------
        site: string
        """
        self.site = site
        if site == 'north' or site == 'lapalma':
            self.site = site
            irf = ana.irf_cta()
            self.SimuArea = irf.LaPalmaArea
        if site == 'south' or site == 'paranal':
            self.site = site
            irf = ana.irf_cta()
            self.SimuArea = irf.ParanalArea

    def plot_all(self, MultiplicityMin=[2,4]):
        self.multiplicity_hist()
        self.impact_heatmap()
        self.energy_hist()
        self.fov_map()
        self.theta2()
        plt.close('all')
        self.angular_res_per_energy(MultiplicityMin=MultiplicityMin)
        plt.close('all')
        self.impact_point_map_distri()
        self.impact_parameter_error()
        self.impact_parameter_error_per_energy()
        self.impact_parameter_error_per_multiplicity()
        self.effective_area_per_energy(MultiplicityMin=MultiplicityMin)
        self.distributions_parameters()
        plt.close('all')
        self.energy_resolution(MultiplicityMin=MultiplicityMin)
        plt.close('all')
        self.impact_resolution_per_energy(MultiplicityMin=MultiplicityMin)
        plt.close('all')

    def plot_results(self):
        self.effective_area_per_energy()

        plt.close('all')
        self.angular_res_per_energy()
        plt.close('all')
        self.energy_resolution()
        self.energy_resolution(bias_correction=True)
        plt.close('all')
        self.energy_bias()
        plt.close('all')



    def multiplicity_hist(self, ER=(-inf, inf), xmin=0, xmax=100):
        """
        historgram of the telescopes multiplicity.
        One can select an energy range.
        Parameters
        ----------
        ER: tuple of (energy_min, energy_max) to select an energy range. Use inf for no limit.
        OutFolder:

        Returns
        -------

        """

        OutFolder = self.outdir
        if not os.path.isdir(OutFolder): os.mkdir(OutFolder)
        # mask = (self.a.RecoE > ER[0]) & (self.a.RecoE < ER[1])
        # multiplicity_hist(self.a.MultiplicityReco[mask], OutFolder + "Multiplicity-[%s-%s]" % (ER[0], ER[1]),
        #                         xmax=50)
        plot_multiplicity_hist(self.a.MultiplicityReco, Outfile=self.outdir + '/' + "Multplicity")

    def impact_heatmap(self):
        """
        plot_impact_point_heatmap
        """
        assert (len(self.a.RecoX) > 0) & (len(self.a.RecoX) > 0), "empty arrays"

        plot_impact_point_heatmap(self.a.RecoX, self.a.RecoY, Outfile=self.outdir + "ImpactHeatmap")

    def energy_hist(self):
        saveplot_energy_distribution(self.a.SimuE, self.a.SimuE[self.a.maskSimuRecoed],
                                 Outfile=self.outdir + "EnergyDistribution", maskSimuDetected=self.a.maskSimuDetected)

    def fov_map(self):
        plot_field_of_view_map(self.a.RecoAlt, self.a.RecoAz, self.a.AltSource, self.a.AzSource, E=False,
                               Outfile=self.outdir + "FovMap")

    def theta2(self):
        plot_theta2(self.a.RecoAlt, self.a.RecoAz, self.a.AltSource, self.a.AzSource, Outfile=self.outdir + "theta2")

    def angular_res_per_energy(self, MultiplicityMin=[2], cta_goal=True, ax=None, **kwargs):
        """
        Plot the angular resolution as a function of the energy

        Parameters
        ----------
        MultiplicityMin: list of `int`
        cta_goal: `bool`
        ax: `matplotlib.pyplot.axes`
        kwargs: args for `hipecta.plots.plot_angular_res_per_energy`

        Returns
        -------
        `matplotlib.pyplot.axes`
        """

        ax = plt.gca() if ax is None else ax

        if cta_goal:
            ax = plot_angular_res_requirement(self.site, ax=ax, color='black')

        if len(MultiplicityMin) > 1 and not 'alpha' in kwargs:
            kwargs['alpha'] = 0.8
        elif not 'alpha' in kwargs:
            kwargs['alpha'] = 1.0

        if len(self.a.MultiplicityReco) > 0:
            for mult in MultiplicityMin:

                mask = self.a.MultiplicityReco >= mult

                if not 'label' in kwargs:
                    kwargs['label'] = "Minimal multiplicity = {0}".format(mult)

                ax = plot_angular_res_per_energy(
                    self.a.RecoAlt[mask], self.a.RecoAz[mask],
                    self.a.SimuAlt[self.a.maskSimuRecoed][mask], self.a.SimuAz[self.a.maskSimuRecoed][mask],
                    self.a.SimuE[self.a.maskSimuRecoed][mask],
                    ax=ax,
                    **kwargs,
                )
        else:

            if not 'label' in kwargs:
                kwargs['label'] = "No cut on multiplicity"

            ax = plot_angular_res_per_energy(self.a.RecoAlt, self.a.RecoAz,
                                             self.a.AltSource, self.a.AzSource,
                                             self.a.SimuE[self.a.maskSimuRecoed],
                                             ax=ax,
                                             **kwargs)

        ax.legend()
        plt.savefig(self.outdir + '/' + "AngRes.png", bbox_inches="tight", format='png', dpi=200)
        # plt.close('all')

        return ax


    def impact_point_map_distri(self):
        plot_impact_point_map_distri(self.a.RecoX, self.a.RecoY, self.a.telX, self.a.telY, kde=True,
                                     Outfile=self.outdir + "ImpactMapDistri")

    def impact_parameter_error(self):
        plot_impact_parameter_error(self.a.RecoX, self.a.RecoY,
                                    self.a.SimuX[self.a.maskSimuRecoed], self.a.SimuY[self.a.maskSimuRecoed],
                                    Outfile=self.outdir + "ImpactParamterError")

    def effective_area_per_energy(self, MultiplicityMin=[2], cta_goal=True, ax=None, **kwargs):
        """
        Plot the effective area as a function of the energy

        Parameters
        ----------
        MultiplicityMin: list of `int` - minimal multiplicity
        cta_goal: `bool` - True to plot cta goal
        ax: `matplotlib.pyplot.axes`
        **kwargs: args for `hipecta.plots.plot_effective_area_per_energy`

        Returns
        -------
        `matplotlib.pyplot.axes`
        """

        ax = plt.gca() if ax is None else ax

        if cta_goal:
            ax = plot_effective_area_requirement(self.site, ax=ax, color='black')

        if len(MultiplicityMin) > 1 and not 'alpha' in kwargs:
            kwargs['alpha'] = 0.8
        elif not 'alpha' in kwargs:
            kwargs['alpha'] = 1.0


        if len(self.a.MultiplicityReco) >= 0:
            for i, mult in enumerate(MultiplicityMin):

                mask = self.a.MultiplicityReco > mult

                if not 'label' in kwargs:
                    kwargs['label'] = "Minimal multiplicity = {0}".format(mult)

                ax = plot_effective_area_per_energy(
                    self.a.SimuE,
                    self.a.SimuE[self.a.maskSimuRecoed][mask],
                    self.SimuArea,
                    ax=ax,
                    **kwargs,
                )
        else:
            if not 'label' in kwargs:
                kwargs['label'] = "No cut on multiplicity"

            ax = plot_effective_area_per_energy(self.a.SimuE,
                                                self.a.SimuE[self.a.maskSimuRecoed],
                                                self.SimuArea,
                                                ax=ax,
                                                **kwargs,
                                                )

        ax.legend()
        plt.savefig(self.outdir + '/' + "EffectiveArea.png", bbox_inches="tight", format='png', dpi=200)
        # plt.close('all')

        return ax

    def impact_parameter_error_per_energy(self):
        plot_impact_parameter_error_per_energy(self.a.RecoX, self.a.RecoY,
                                               self.a.SimuX[self.a.maskSimuRecoed], self.a.SimuY[self.a.maskSimuRecoed],
                                               self.a.SimuE[self.a.maskSimuRecoed],
                                               Outfile=self.outdir + "ImpactParameterErrorPerEnergy")

    def impact_parameter_error_per_multiplicity(self):
        plot_impact_parameter_error_per_multiplicity(self.a.RecoX, self.a.RecoY,
                                                     self.a.SimuX[self.a.maskSimuRecoed],
                                                     self.a.SimuY[self.a.maskSimuRecoed],
                                                     self.a.MultiplicityReco, max=None,
                                                     Outfile=self.outdir + "ImpactParameterErrorPerMultiplicity")

    def impact_parameter_error_site_center(self):
        plot_impact_parameter_error_site_center(self.a.RecoX, self.a.RecoY,
                                                self.a.SimuX[self.a.maskSimuRecoed],
                                                self.a.SimuY[self.a.maskSimuRecoed],
                                                Outfile=self.outdir + "ImpactParameterErrorSiteCenter")

    def recoQuality_hist(self):
        plot_recoQuality_hist(self.recoQuality, Outfile=self.outdir + "recoQuality")

    def distributions_parameters(self, pdfname="distributions_check.pdf"):
        """
        Plot some dirty parameters distributions for fast check.
        Save them into a multipage pdf.
        Returns
        -------

        """
        from matplotlib.backends.backend_pdf import PdfPages
        import datetime
        pp = PdfPages(self.outdir + '/' + pdfname)
        now = datetime.datetime.now()
        txt = "Some distributions of the reconstructed parameters \n " \
              "Only the events with a good reconstruction quality are kept \n {0}".format(
            now.strftime("%Y-%m-%d %H:%M"))
        fig = plt.figure()
        fig.text(.1, .5, txt)
        pp.savefig(fig)
        plt.close()

        for k in self.a.__dict__:
            fig = plt.figure()
            value = getattr(self.a, k)
            mask = np.concatenate(self.a.ImageQuality) == 0
            if type(value) == np.ndarray and len(value) > 0:
                if type(value[0]) == np.ndarray:
                    ar = np.nan_to_num(np.concatenate(value))[mask]
                else:
                    ar = value
                plt.hist(ar, label=str(k), bins=40);
                plt.legend()
                pp.savefig(fig)
            plt.close()
        pp.close()

    def energy_resolution(self, cta_goal=True, MultiplicityMin=[2], bias_correction=False):

        ax = plt.gca()

        if cta_goal:
            ax = plot_energy_resolution_requirements(self.site, ax=ax, color='black')

        alpha=1.0
        if len(MultiplicityMin) > 1:
            alpha=0.8

        if len(self.a.MultiplicityReco) > 0:
            for mult in MultiplicityMin:
                mask = self.a.MultiplicityReco >= mult
                ax = plot_energy_resolution (
                    self.a.SimuE[self.a.maskSimuRecoed][mask],
                    self.a.RecoE[mask],
                    bias_correction=bias_correction,
                    ax=ax,
                    label="Minimal multiplicity = {0}".format(mult),
                    alpha=alpha
                )
        else:
            ax = plot_energy_resolution(self.a.SimuE[self.a.maskSimuRecoed],
                                        self.a.RecoE,
                                        bias_correction=bias_correction,
                                        ax=ax,
                                        label="No cut on multiplicity")
        ax.legend()
        plt.savefig(self.outdir + '/' + 'EnergyResolution.png', bbox_inches="tight", format='png', dpi=200)
        #plt.close('all')

    def energy_bias(self, cta_goal=True, MultiplicityMin=[2]):

        ax = plt.gca()

        alpha=1.0
        if len(MultiplicityMin) > 1:
            alpha=0.8

        if len(self.a.MultiplicityReco) > 0:
            for mult in MultiplicityMin:
                mask = self.a.MultiplicityReco >= mult
                ax = plot_energy_bias (
                    self.a.SimuE[self.a.maskSimuRecoed][mask],
                    self.a.RecoE[mask],
                    ax=ax,
                    label="Minimal multiplicity = {0}".format(mult),
                    alpha=alpha
                )
        else:
            ax = plot_energy_bias(self.a.SimuE[self.a.maskSimuRecoed],
                                        self.a.RecoE,
                                        ax=ax,
                                        label="No cut on multiplicity")
        ax.legend()
        plt.savefig(self.outdir + '/' + 'EnergyBias.png', bbox_inches="tight", format='png', dpi=200)
        plt.close('all')


    def impact_resolution_per_energy(self, MultiplicityMin=[2], ax=None, **kwargs):
        """
        Plot the impact resolution as a function of the energy

        Parameters
        ----------
        MultiplicityMin: list of `int`
        ax: `matplotlib.pyplot.axes`
        kwargs: args for `hipecta.plots.plot_impact_resolution_per_energy`

        Returns
        -------
        `matplotlib.pyplot.axes`
        """

        ax = plt.gca() if ax is None else ax

        if len(MultiplicityMin)>1:
            if 'alpha' not in kwargs:
                kwargs['alpha'] = 0.8

        if len(self.a.MultiplicityReco) >= 0:
            for mult in MultiplicityMin:

                if 'label' not in kwargs:
                    kwargs['label'] = "Minimal multiplicity = {0}".format(mult)

                mask = self.a.MultiplicityReco > mult
                ax = plot_impact_resolution_per_energy(
                    self.a.RecoX[mask],
                    self.a.RecoY[mask],
                    self.a.SimuX[self.a.maskSimuRecoed][mask],
                    self.a.SimuY[self.a.maskSimuRecoed][mask],
                    self.a.SimuE[self.a.maskSimuRecoed][mask],
                    ax=ax
                )

        else:
            if 'label' not in kwargs:
                kwargs['label'] = "No cut on multiplicity"

            ax = plot_impact_resolution_per_energy(self.a.RecoX, self.a.RecoY,
                                                   self.a.SimuX[self.a.maskSimuRecoed],
                                                   self.a.SimuY[self.a.maskSimuRecoed],
                                                   self.a.SimuE[self.a.maskSimuRecoed],
                                                   ax=ax,
                                                   label="No cut on multiplicity")

        ax.legend()
        plt.savefig(self.outdir + '/' + "ImpactResolution.png", bbox_inches="tight", format='png', dpi=200)
        # plt.close('all')
        return ax


def plot_recoQuality_hist(recoQ, Outfile="recoQuality"):
    """
    Plot an histogram of the reconstruction quality flags
    Parameters
    ----------
    recoQ: 1D array
    Outfile: string
    mask
    """
    fig = plt.figure(figsize=(12, 9))

    mi = int(recoQ.min())
    ma = int(recoQ.max())

    plt.hist(recoQ, align='left', bins=int(ma), range=(0, ma));
    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()


def saveplot_energy_distribution(SimuE, RecoE, Outfile="EnergyDistribution", maskSimuDetected=True):
    """
    Plot the energy distribution of the simulated particles, detected particles and reconstructed particles

    Parameters
    ----------
    SimuE: Numpy 1d array of simulated energies
    RecoE: Numpy 1d array of reconstructed energies
    Outfile: string - filename output
    maskSimuDetected - Numpy 1d array - mask of detected particles for the SimuE array
    """

    fig = plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(111)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_xlabel('Energy [TeV]')
    ax1.set_ylabel('Count')

    ax1.set_xscale('log')
    count_S, bin_S, o = ax1.hist(SimuE, log=True, bins=np.logspace(-3, 3, 30), label="Simulated")
    count_D, bin_D, o = ax1.hist(SimuE[maskSimuDetected], log=True, bins=np.logspace(-3, 3, 30), label="Detected")
    count_R, bin_R, o = ax1.hist(RecoE, log=True, bins=np.logspace(-3, 3, 30), label="Reconstructed")
    plt.legend(fontsize=SizeLabel)
    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()


def plot_multiplicty_per_energy(Multiplicity, Energies, Outfile="MultiplicityE"):
    """
    Plot the telescope multiplicity as a function of the energy and save it under Outfile.png
    Parameters
    ----------
    Multiplicity: 1d numpy array
    Energies: 1d numpy array
    Outfile: string
    """

    assert len(Multiplicity) == len(Energies), "arrays should have same length"
    assert len(Multiplicity) > 0, "arrays are empty"

    E, m_mean, m_min, m_max, m_per = multiplicty_stat_per_energy(Multiplicity, Energies)

    fig = plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(111)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_xlabel('Energy [TeV]')
    ax1.set_ylabel('Multiplicity')

    ax1.fill_between(E, m_min, m_max, alpha=0.5)
    ax1.plot(E, m_min, '--')
    ax1.plot(E, m_max, '--')
    ax1.plot(E, m_mean, color='red')

    ax1.set_xscale('log')

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()


def plot_field_of_view_map(RecoAlt, RecoAz, AltSource, AzSource, E=False, Outfile="FovMap"):
    """
    Plot a map in angles [in degrees] of the photons seen by the telescope (after reconstruction)
    Parameters
    ----------
    RecoAlt: 1d numpy array
    RecoAz: 1d numpy array
    AltSource: float, source Altitude
    AzSource: float, source Azimuth
    E: 1d numpy array
    Outfile: string
    """
    plt.figure(figsize=(12, 12))

    dx = 0.05

    ax = plt.subplot(111)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlim(AzSource - dx, AzSource + dx)
    ax.set_ylim(AltSource - dx, AltSource + dx)

    plt.xlabel("Az [deg]")
    plt.ylabel("Alt [deg]")

    plt.axis('equal')

    c = BrewBlues[-1]
    if E:
        c = np.log10(E)
        plt.colorbar()

    plt.scatter(RecoAz, RecoAlt, c=c)
    plt.scatter(AzSource, AltSource, marker='+', linewidths=3, s=200, c='orange', label="Source position")

    plt.legend()

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()


def plot_angles_distribution(RecoAlt, RecoAz, AltSource, AzSource, Outfile="AnglesDistribution"):
    """
    Plot the disitribution of reconstructed angles. Save figure to Outfile in png format.
    Parameters
    ----------
    RecoAlt: 1d numpy array
    RecoAz: 1d numpy array
    AltSource: float
    AzSource: float
    Outfile: string
    """

    dx = 1

    plt.figure(figsize=(12, 9))

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

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()


def plot_theta2(RecoAlt, RecoAz, AltSource, AzSource, Outfile="theta2"):
    """
    Plot the theta2 distribution and save it under Outfile
    Parameters
    ----------
    RecoAlt: 1d numpy array
    RecoAz: 1d numpy array
    AltSource: float
    AzSource: float
    Outfile: string
    """

    theta2 = ana.theta2(RecoAlt, RecoAz, AltSource, AzSource)
    AngRes = ana.angular_resolution(RecoAlt, RecoAz, AltSource, AzSource)

    plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(111)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_xlabel(r'$\theta^2 [deg^2]$')
    ax1.set_ylabel('Count')

    plt.figtext(0.6, 0.7, "Angular Resolution\n = {0:.3f} deg".format(AngRes[0]), fontsize=SizeLabel)
    n, bins, patches = ax1.hist(theta2, range=(0, 0.1), bins=60)

    # ax1.vlines(theta2[i],0,n[0]/3)

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()
    plt.close('all')


def plot_angles_map_distri(RecoAlt, RecoAz, AltSource, AzSource, E, Outfile="AnglesMapDistri"):
    """

    Parameters
    ----------
    RecoAlt
    RecoAz
    AltSource
    AzSource
    E
    Outfile

    Returns
    -------

    """

    dx = 0.5

    plt.figure(figsize=(12, 12))

    # ax1 = plt.subplot(221)
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
    # ax1.spines["top"].set_visible(False)
    # ax1.spines["right"].set_visible(False)

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
    # ax1.yaxis.tick_right()
    ax1.yaxis.set_tick_params(labelsize=SizeTick)
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_tick_params(labelsize=SizeTick)
    plt.legend('', 'Source position')

    # ax2 = plt.subplot(223, sharex=ax1)
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

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()
    return 0


def plot_impact_point_map_distri(RecoX, RecoY, telX, telY, **options):
    """
    Map and distributions of the reconstructed impact points.

    Parameters
    ----------
    RecoX: 1D numpy array
    RecoY: 1D numpy array
    telX: 1D numpy array
    telY: 1D numpy array
    options:
        kde=True : make a gaussian fit of the point density
        Outfile='string' : save a png image of the plot under 'string.png'
    """
    plt.figure(figsize=(12, 12))

    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
    # ax1.spines["top"].set_visible(False)
    # ax1.spines["right"].set_visible(False)

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
    # ax1.yaxis.tick_right()
    ax1.yaxis.set_tick_params(labelsize=SizeTick)
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_tick_params(labelsize=SizeTick)
    plt.legend('', 'Source position')

    ax1.scatter(telX, telY, c='tomato', marker='+', s=90, linewidths=10)

    # ax2 = plt.subplot(223, sharex=ax1)
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
        plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
        plt.close()


def plot_impact_point_heatmap(RecoX, RecoY, Outfile="ImpactHeatmap"):
    """
    Plot the heatmap of the impact points on the site ground and save it under Outfile
    Parameters
    ----------
    RecoX: 1d numpy array
    RecoY: 1d numpy array
    Outfile: string
    """

    fig = plt.figure()  # figsize=(12, 12))

    plt.xticks(fontsize=SizeTick)
    plt.yticks(fontsize=SizeTick)

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis('equal')

    cm = plt.cm.get_cmap('OrRd')
    plt.hist2d(RecoX, RecoY, bins=75, norm=LogNorm(), cmap=cm)
    cb = plt.colorbar()
    cb.set_label('Event count')

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()


def plot_multiplicity_hist(multiplicity, Outfile="Multiplicity", xmin=0, xmax=100):
    """
    Histogram of the telescopes multiplicity
    Parameters
    ----------
    multiplicity: numpy array
    Outfile: string
    xmin: float
    xmax: float
    """
    m = np.sort(multiplicity)
    plt.figure(figsize=(12, 9))
    ax1 = plt.subplot(111)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.xticks(fontsize=SizeTick)
    plt.yticks(fontsize=SizeTick)

    if xmax <= xmin:
        xmax = floor(m.max()) + 1

    if len(m) > 0:
        ax1.set_xlim(xmin, xmax + 1)
        n, b, s = ax1.hist(m, bins=(xmax - xmin), align='left', range=(xmin, xmax), label='Telescope multiplicity')
        ax1.xaxis.set_ticks(np.append(np.linspace(xmin, xmax, 10, dtype=int), [1, 2, 3, 4, 5]))
        x50 = m[int(floor(0.5 * len(m)))] + 0.5
        x90 = m[int(floor(0.9 * len(m)))] + 0.5
        if (x50 > xmin and x50 < xmax):
            ax1.vlines(x50, 0, n[int(m[int(floor(0.5 * len(m)))])], color=BrewOranges[-2], label='50%')
        if (x90 > xmin and x90 < xmax):
            ax1.vlines(x90, 0, n[int(m[int(floor(0.9 * len(m)))])], color=BrewOranges[-1], label='90%')

    plt.legend(fontsize=SizeLabel)
    plt.title("Telescope multiplicity")
    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()
    return 0


def plot_multiplicity_per_telescope_type(EventTup, Outfile="MultiplicityPerTelescopeType"):
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
        lst = 0;
        mst = 0;
        sst = 0;
        for t in tups:
            if t[2] == 0:
                lst += 1
            elif t[2] >= 1 and t[2] <= 3:
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

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()

    return LST, MST, SST




def plot_effective_area_per_energy(SimuE, RecoE, simuArea, ax=None, **kwargs):
    """

    Parameters
    ----------
    SimuE
    RecoE
    simuArea
    ax
    kwargs

    Returns
    -------

    """

    ax = plt.gca() if ax is None else ax

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel('Energy [TeV]')
    ax.set_ylabel('Seff [m2]')
    ax.set_xscale('log')
    ax.set_yscale('log')

    E_bin, Seff = ana.effective_area_per_energy(SimuE, RecoE, simuArea)
    E = ana.logbin_mean(E_bin)
    ax.errorbar(E, Seff, xerr=(E_bin[1:] - E_bin[:-1]) / 2., fmt='o', **kwargs)

    return ax


def plot_effective_area_requirement(cta_site, ax=None, **kwargs):
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
    ax.set_ylabel('Seff [m2]')
    ax.plot(e_cta, ef_cta, label="CTA requirements {}".format(cta_site), **kwargs)

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
    kwargs: args for `hipectaold.plots.plot_angular_res_per_energy`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    ax = plot_effective_area_per_energy(SimuE, RecoE, simuArea, ax=ax, **kwargs)

    if cta_site:
        ax = plot_effective_area_requirement(cta_site, ax=ax, color='black')

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200)
    plt.close()

    return ax




def plot_layout_map(TelX, TelY, TelId, TelType, LayoutId, Outfile="LayoutMap"):
    """
    Plot the layout map
    Parameters
    ----------
    TelX
    TelY
    TelId
    TelType
    LayoutId
    Outfile

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
    # ax.scatter(TelX[mask_layout],TelY[mask_layout], s=100/(1+type[mask_layout]), c=type[mask_layout], cmap='Paired',label="label this")#, c=TelType[mask_layout])
    ax.scatter(TelX[mask_lst], TelY[mask_lst], s=100 / (1 + type[mask_lst]), c=BrewBlues[-1], cmap='Paired',
               label="LST")
    ax.scatter(TelX[mask_mst], TelY[mask_mst], s=100 / (1 + type[mask_mst]), c=BrewReds[-1], cmap='Paired', label="MST")
    ax.scatter(TelX[mask_sst], TelY[mask_sst], s=100 / (1 + type[mask_sst]), c=BrewGreens[-1], cmap='Paired',
               label="SST")

    plt.title("CTA site with layout %s" % Outfile.split('.')[-1], fontsize=SizeTitleArticle)
    plt.legend(fontsize=SizeLabel)
    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()




def plot_angular_res_per_energy(RecoAlt, RecoAz, AltSource, AzSource, SimuE, ax=None, **kwargs):
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

    E_bin, RES = ana.angular_resolution_per_energy(RecoAlt, RecoAz, AltSource, AzSource, SimuE)
    # Angular resolution is traditionnaly presented in degrees
    RES = np.degrees(RES)

    E = ana.logbin_mean(E_bin)

    ax.errorbar(E, RES[:,0], xerr=(E_bin[1:] - E_bin[:-1]) / 2.,
                yerr=(RES[:,0]-RES[:,1], RES[:,2]-RES[:,0]), fmt='o', **kwargs)

    return ax


def saveplot_angular_res_per_energy(RecoAlt, RecoAz, AltSource, AzSource, SimuE, ax=None, Outfile="AngRes", cta_site=None, **kwargs):
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
        ax = plot_angular_res_requirement(cta_site, ax=ax)

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()

    return ax


def plot_angular_res_requirement(cta_site ,ax=None, **kwargs):
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
    ax.plot(e_cta, ar_cta, label="CTA requirements {}".format(cta_site), **kwargs)
    ax.set_xscale('log')
    return ax


def plot_impact_parameter_error(RecoX, RecoY, SimuX, SimuY, Outfile="ImpactParamterError"):
    """
    plot impact parameter error distribution and save it under Outfile
    Parameters
    ----------
    RecoX: 1d numpy array
    RecoY
    SimuX
    SimuY
    Outfile: string
    """
    d = np.sqrt((RecoX - SimuX) ** 2 + (RecoY - SimuY) ** 2)

    plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(111)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_xlabel('Error on impact parameter [m]')
    ax1.set_ylabel('Count')

    ax1.hist(d, bins=40)

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()


def plot_impact_parameter_error_per_energy(RecoX, RecoY, SimuX, SimuY, SimuE, Outfile="ImpactParameterErrorPerEnergy"):
    """
    plot the impact parameter error distance as a function of energy and save the plot as Outfile
    Parameters
    ----------
    RecoX: 1d numpy array
    RecoY
    SimuX
    SimuY
    SimuE
    Outfile: string

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

    plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(111)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_ylabel('Error on Impact Parameter [m]')
    ax1.set_xlabel('Energy [TeV]')

    ax1.set_xscale('log')

    ax1.fill_between(E, err_min, err_max)
    ax1.errorbar(E, err_mean, err_std, color="red", label="mean+std")

    plt.legend(fontsize=SizeLabel)

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()

    return E, np.array(err_mean)



def plot_impact_parameter_error_per_multiplicity(RecoX, RecoY, SimuX, SimuY, Multiplicity, max=None,
        Outfile="ImpactParameterErrorPerMultiplicity"):
    """

    Parameters
    ----------
    RecoX
    RecoY
    SimuX
    SimuY
    Multiplicity
    max
    Outfile

    Returns
    -------

    """
    if max == None: max = Multiplicity.max() + 1
    M = np.arange(Multiplicity.min(), max)
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

    plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(111)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_ylabel('Error on Impact Parameter [m]')
    ax1.set_xlabel('Multiplicity')

    ax1.fill_between(M, e_min, e_max)
    ax1.errorbar(M, e_mean, e_std, color="red", label="mean+std")

    plt.legend(fontsize=SizeLabel)

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()

    return M, np.array(e_mean)






def plot_site_map(telX, telY, telTypes=None, Outfile="SiteMap.png"):
    """
    Map of the site with telescopes positions
    Parameters
    ----------
    telX: numpy array
    telY: numpy array
    telTypes: numpy array
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
    impactX: numpy array
    impactY: numpy array
    telX: numpy array
    telY: numpy array
    telTypes: numpy array
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
    assert len(SimuE)==len(RecoE), "simulated and reconstructured energy arrrays should have the same length"

    ax = plt.gca() if ax is None else ax

    E_bin, biasE = ana.energy_bias(SimuE, RecoE)
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt']='o'

    ax.set_ylabel("bias (median($E_{reco}/E_{simu}$ - 1)")
    ax.set_xlabel("log(E/TeV)")
    ax.set_xscale('log')
    plt.legend()
    plt.title('Energy bias')

    ax.errorbar(E, biasE, xerr=(E-E_bin[:-1], E_bin[1:]-E), **kwargs)

    return ax


def plot_energy_resolution(SimuE, RecoE, ax=None, bias_correction=False, **kwargs):
    """

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

    E_bin, Eres = ana.energy_res_per_energy(SimuE, RecoE, bias_correction=bias_correction)
    E = ana.logbin_mean(E_bin)

    if 'fmt' not in kwargs:
        kwargs['fmt']='o'

    ax.set_ylabel("R_68")
    ax.set_xlabel("log(E/TeV)")
    ax.set_xscale('log')
    plt.title('Energy resolution')

    ax.errorbar(E, Eres[:,0], xerr=(E-E_bin[:-1], E_bin[1:]-E), yerr=(Eres[:,0]-Eres[:,1], Eres[:,2] - Eres[:,0]), **kwargs)

    return ax


def plot_energy_resolution_requirements(cta_site, ax=None, **kwargs):
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

    if not ('color' in kwargs or 'c' in kwargs):
        kwargs['color'] = 'black'

    ax.plot(e_cta, ar_cta, label="CTA requirements {}".format(cta_site), **kwargs)
    ax.set_xscale('log')
    return ax



def saveplot_energy_resolution(SimuE, RecoE, Outfile="EnergyResolution.png", cta_site=None):
    """
    plot the energy resolution of the reconstruction
    Parameters
    ----------
    SimuE: `numpy.ndarray`
    RecoE: `numpy.ndarray`
    cta_goal: boolean - If True CTA energy resolution goal is plotted

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = plot_energy_resolution(SimuE, RecoE)

    if cta_site!=None:
        ax = plot_energy_resolution_requirements(cta_site, ax=ax)

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


def plot_impact_parameter_error_site_center(RecoX, RecoY, SimuX, SimuY, Outfile="ImpactParameterErrorSiteCenter"):
    """
    Plot the impact parameter error as a function of the distance to the site center.

    Parameters
    ----------
    RecoX: 1d array
    RecoY: 1d array
    SimuX: 1d array
    SimuY: 1d array
    Outfile: string
    """

    import seaborn as sns
    sns.set(style="white", color_codes=True)

    imp_err = ana.impact_parameter_error(a.RecoX, a.RecoY, a.SimuX[a.maskSimuRecoed], a.SimuY[a.maskSimuRecoed])
    dCenter = np.sqrt(a.SimuX[a.maskSimuRecoed] ** 2 + a.SimuY[a.maskSimuRecoed] ** 2)

    sns.jointplot(dCenter, imp_err, kind='reg')

    plt.legend(fontsize=SizeLabel)

    plt.savefig(Outfile + ".png", bbox_inches="tight", format='png', dpi=200);
    plt.close()



def plot_site(telX, telY, ax=None, **kwargs):
    """
    Plot the telescopes positions
    Parameters
    ----------
    telX: 1D numpy array
    telY: 1D numpy array
    ax: `~matplotlib.axes.Axes` or None
    **kwargs : Extra keyword arguments are passed to `matplotlib.pyplot.scatter`

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
    """
    ax = plt.gca() if ax is None else ax

    ax.scatter(telX, telY, **kwargs)
    ax.axis('equal')

    return ax



def plot_impact_resolution_per_energy(RecoX, RecoY, SimuX, SimuY, SimuE, ax=None, **kwargs):
    """
    Plot the angular resolution as a function of the energy

    Parameters
    ----------
    RecoX: `numpy.ndarray`
    RecoY: `numpy.ndarray`
    SimuX: float
    SimuY: float
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
    ax.set_ylabel('Impact Resolution [m]')
    ax.set_xlabel('Energy [TeV]')
    ax.set_xscale('log')

    E_bin, RES = ana.impact_resolution_per_energy(RecoX, RecoY, SimuX, SimuY, SimuE)
    E = ana.logbin_mean(E_bin)

    ax.errorbar(
        E, RES[:, 0],
        xerr=(E-E_bin[:-1], E_bin[1:]-E),
        yerr=(RES[:, 0]-RES[:, 1], RES[:, 2]-RES[:, 0]),
        fmt='o',
        **kwargs,
    )

    return ax


def saveplot_impact_resolution_per_energy(RecoX, RecoY, SimuX, SimuY, SimuE, ax=None,
        Outfile="impact_resolution.png", **kwargs):
    """

    Parameters
    ----------
    RecoX
    RecoY
    SimuX
    SimuY
    SimuE
    ax
    Outfile
    kwargs

    Returns
    -------

    """
    ax = plot_impact_resolution_per_energy(RecoX, RecoY, SimuX, SimuY, SimuE, ax=ax, **kwargs)
    plt.savefig(Outfile, fmt='png', dpi=200)
    plt.close()


def plot_migration_matrix(X, Y, ax=None, **kwargs):
    """
    Make a simple plot of a migration matrix

    Parameters
    ----------
    X: list or `numpy.ndarray`
    Y: list or `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    **kwargs: args for `matplotlib.pyplot.hist2d`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """

    if not 'bins' in kwargs:
        kwargs['bins'] = 50

    ax = plt.gca() if ax is None else ax
    ax.hist2d(X, Y, **kwargs)
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


def plot_features_importance(learn_class, ax=None):

    ax = plt.gca() if ax is None else ax

    sns.barplot(learn_class.input_features_keys, learn_class.model.feature_importances_, ax=ax)
    plt.xticks(rotation='vertical')
    plt.title("Features importance".format(id))

    return ax
