import os
import json
import tables
import tempfile
from shutil import copyfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
from ipywidgets import HBox, Tab, Output
from sklearn.metrics import roc_curve, roc_auc_score
from .. import plots
from .. import ana
from .. dataset import get

__all__ = ['open_dashboard',
           'load_data_from_h5',
           'GammaBoard',
           'plot_migration_matrices'
           ]


def load_data_from_h5(experiment, experiments_directory):
    """
    Load an hdf5 file containing results from an experiment

    Args
        experiment (str): the name of the experiment
        experiments_directory (str): the path to the folder containing the experiment folders

    Returns
        `pandas.DataFrame`
    """
    assert experiment in os.listdir(experiments_directory)

    try:
        data = pd.read_hdf(experiments_directory + '/' + experiment + '/' + experiment + '.h5',
                           key='data',
                           )
    except Exception as e:
        print(e)
        return None
    return data


# TODO Find a more suitable naming
def load_trig_events(experiment, experiments_directory):
    assert experiment in os.listdir(experiments_directory)

    try:
        trig_events = pd.read_hdf(experiments_directory + '/' + experiment + '/' + experiment + '.h5',
                                  key='triggered_events',
                                  )
    except:
        print("Cannot load the number of triggered events for experiment {} file".format(experiment))
        return None
    return trig_events


def load_run_config(experiment, experiments_directory):
    assert experiment in os.listdir(experiments_directory)
    file = experiments_directory + '/' + experiment + '/' + experiment + '.h5'
    num_events = 0
    spectral_index = []
    max_scatter_range = []
    energy_range_max = []
    energy_range_min = []
    min_alt = []
    max_alt = []

    result_file = None

    try:
        # result_file = pd.HDFStore(file)
        result_file = tables.open_file(file)
        run_config = result_file.root.simulation.run_config
        for row in run_config:
            num_events += row['num_showers'] * row['shower_reuse']
            spectral_index.extend([row['spectral_index']])
            max_scatter_range.extend([row['max_scatter_range']])
            energy_range_max.extend([row['energy_range_max']])
            energy_range_min.extend([row['energy_range_min']])
            min_alt.extend([row['min_alt']])
            max_alt.extend([row['max_alt']])
        assert np.alltrue(np.array(spectral_index) == spectral_index[0]), \
            'Cannot deal with different spectral index for the experiment ({})'.format(experiment)
        assert np.alltrue(np.array(max_scatter_range) == max_scatter_range[0]), \
            'Cannot deal with different max_scatter_range for the experiment ({})'.format(experiment)
        assert np.alltrue(np.array(energy_range_min) == energy_range_min[0]), \
            'Cannot deal with different energy_range_min for the experiment ({})'.format(experiment)
        assert np.alltrue(np.array(energy_range_max) == energy_range_max[0]), \
            'Cannot deal with different energy_range_max for the experiment ({})'.format(experiment)
        assert np.alltrue(np.array(min_alt) == min_alt[0]), \
            'Cannot deal with different min_alt for the experiment ({})'.format(experiment)
        assert np.alltrue(np.array(max_alt) == max_alt[0]), \
            'Cannot deal with different max_alt for the experiment ({})'.format(experiment)
        assert min_alt[0] == max_alt[0], 'Cant deal with different shower altitude for the experiment ({})'.format(
            experiment)
        scattering_surface = max_scatter_range[0] ** 2 * np.pi * np.sin(max_alt[0])
        result_file.close()
    except Exception as e:
        print("Cannot load the configuration of the simulation for experiment {} file".format(experiment))
        if result_file is not None:
            result_file.close()
        return None
    return {
        'num_events': num_events,
        'spectral_index': spectral_index[0],
        'energy_range_min': energy_range_min[0],
        'energy_range_max': energy_range_max[0],
        'scattering_surface': scattering_surface
    }


def load_info(experiment, experiments_directory):
    """
        Load a json file containing infos from an experiment

        Args
            experiment (str): the name of the experiment
            experiments_directory (str): the path to the folder containing the experiment folders

        Returns
            `pandas.DataFrame`
        """
    assert experiment in os.listdir(experiments_directory)

    try:
        info = json.load(open(experiments_directory + '/' + experiment + '/' + experiment + '_settings.json'),
                         object_pairs_hook=OrderedDict)
    except:
        return None

    return info


def print_dict(dictionary, indent=''):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(indent, key)
            print_dict(value, indent + '    ')
        else:
            print(indent, key, ' : ', value)


def change_errorbar_visibility(err_container, visible: bool):
    """
    Change the visibility of all lines of an errorbar container
    Args
        err_container (`matplotlib.container.ErrorbarContainer`)
        visible (bool)
    """
    aa, bb, cc = err_container.lines
    aa.set_visible(visible)
    try:
        cc[0].set_visible(visible)
        cc[1].set_visible(visible)
    except:
        pass


post_classification_opt = dict(ms=0,
                               elinewidth=0.001,
                               linestyle=':',
                               fmt='v',
                               )


class Experiment(object):
    r"""Class to deal with an experiment

    Args
        experiment_name (str)
        experiment_directory (str)
        ax_imp_res

    """

    def __init__(self, experiment_name, experiments_directory, bias_correction, classif_resolution):

        self.name = experiment_name
        self.experiments_directory = experiments_directory
        self.info = load_info(self.name, self.experiments_directory)
        self.bias_correction = bias_correction
        self.data = None
        self.gamma_data = None
        self.reco_gamma_data = None
        self.mc_trig_events = None
        self.run_config = None
        self.loaded = False
        self.plotted = False
        self.color = None
        self.precision = None
        self.recall = None
        self.accuracy = None
        self.auc = None
        self.rocness = None
        self.classif_resolution = classif_resolution

        self.cm = plt.cm.jet
        self.cm.set_under('w', 1)

    def load_data(self):
        self.data = load_data_from_h5(self.name, self.experiments_directory)
        if self.data is not None:
            self.set_loaded(True)
            if 'mc_particle' in self.data:
                if 'reco_hadroness' in self.data:
                    self.rocness = 'Hadroness'
                    self.gamma_data = self.data[self.data.mc_particle == 0]
                    self.reco_gamma_data = self.gamma_data[self.gamma_data.reco_particle == 0]
                elif 'reco_gammaness' in self.data:
                    self.rocness = 'Gammaness'
                    self.gamma_data = self.data[self.data.mc_particle == 1]
                    self.reco_gamma_data = self.gamma_data[self.gamma_data.reco_particle == 1]
            else:
                self.gamma_data = self.data
        self.mc_trig_events = load_trig_events(self.name, self.experiments_directory)
        self.run_config = load_run_config(self.name, self.experiments_directory)

    def get_data(self):
        return self.data

    def get_plotted(self):
        return self.plotted

    def set_plotted(self, plotted: bool):
        self.plotted = plotted

    def set_loaded(self, loaded: bool):
        self.loaded = loaded

    def get_loaded(self):
        return self.loaded

    def plot_angular_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_ang_res = plots.plot_angular_resolution_per_energy(self.gamma_data.reco_altitude,
                                                                       self.gamma_data.reco_azimuth,
                                                                       self.gamma_data.mc_altitude,
                                                                       self.gamma_data.mc_azimuth,
                                                                       self.gamma_data.mc_energy,
                                                                       bias_correction=self.bias_correction,
                                                                       ax=ax,
                                                                       label=self.name,
                                                                       color=self.color)

            if self.reco_gamma_data is not None and self.classif_resolution:
                self.ax_ang_res = plots.plot_angular_resolution_per_energy(self.reco_gamma_data.reco_altitude,
                                                                           self.reco_gamma_data.reco_azimuth,
                                                                           self.reco_gamma_data.mc_altitude,
                                                                           self.reco_gamma_data.mc_azimuth,
                                                                           self.reco_gamma_data.mc_energy,
                                                                           bias_correction=self.bias_correction,
                                                                           ax=ax,
                                                                           label=self.name + '_reco',
                                                                           color=self.color,
                                                                           **post_classification_opt,
                                                                           )

            self.set_plotted(True)

    def plot_energy_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_ene_res = plots.plot_energy_resolution(self.gamma_data.mc_energy,
                                                           self.gamma_data.reco_energy,
                                                           bias_correction=self.bias_correction,
                                                           ax=ax,
                                                           label=self.name,
                                                           color=self.color)
            if self.reco_gamma_data is not None and self.classif_resolution:
                self.ax_ene_res = plots.plot_energy_resolution(self.reco_gamma_data.mc_energy,
                                                               self.reco_gamma_data.reco_energy,
                                                               bias_correction=self.bias_correction,
                                                               ax=ax,
                                                               label=self.name + '_reco',
                                                               color=self.color,
                                                               **post_classification_opt
                                                               )

            self.set_plotted(True)

    def plot_impact_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_imp_res = plots.plot_impact_resolution_per_energy(self.gamma_data.reco_impact_x,
                                                                      self.gamma_data.reco_impact_y,
                                                                      self.gamma_data.mc_impact_x,
                                                                      self.gamma_data.mc_impact_y,
                                                                      self.gamma_data.mc_energy,
                                                                      bias_correction=self.bias_correction,
                                                                      ax=ax,
                                                                      label=self.name,
                                                                      color=self.color
                                                                      )
            if self.reco_gamma_data is not None and self.classif_resolution:
                self.ax_imp_res = plots.plot_impact_resolution_per_energy(self.reco_gamma_data.reco_impact_x,
                                                                          self.reco_gamma_data.reco_impact_y,
                                                                          self.reco_gamma_data.mc_impact_x,
                                                                          self.reco_gamma_data.mc_impact_y,
                                                                          self.reco_gamma_data.mc_energy,
                                                                          bias_correction=self.bias_correction,
                                                                          ax=ax,
                                                                          label=self.name + '_reco',
                                                                          color=self.color,
                                                                          **post_classification_opt
                                                                          )
            self.ax_imp_res.set_xscale('log')
            self.ax_imp_res.set_xlabel('Energy [TeV]')
            self.ax_imp_res.set_ylabel('Impact resolution [km]')
            self.set_plotted(True)

    def plot_effective_area(self, ax=None):
        if self.get_loaded():
            self.ax_eff_area = ax if ax is not None else plt.gca()

            if self.run_config is not None:
                if self.mc_trig_events is not None:
                    E_trig, S_trig = ana.effective_area_per_energy_power_law(self.run_config['energy_range_min'],
                                                                             self.run_config['energy_range_max'],
                                                                             self.run_config['num_events'],
                                                                             self.run_config['spectral_index'],
                                                                             self.mc_trig_events.mc_trig_energies,
                                                                             self.run_config['scattering_surface'])
                    self.ax_eff_area.plot(E_trig[:-1], S_trig, label=self.name + '_triggered', color=self.color,
                                          linestyle='-.')

                E, S = ana.effective_area_per_energy_power_law(self.run_config['energy_range_min'],
                                                               self.run_config['energy_range_max'],
                                                               self.run_config['num_events'],
                                                               self.run_config['spectral_index'],
                                                               self.gamma_data.mc_energy,
                                                               self.run_config['scattering_surface'])
                self.ax_eff_area.plot(E[:-1], S, label=self.name, color=self.color)

                if self.reco_gamma_data is not None:
                    E_reco, S_reco = ana.effective_area_per_energy_power_law(self.run_config['energy_range_min'],
                                                                             self.run_config['energy_range_max'],
                                                                             self.run_config['num_events'],
                                                                             self.run_config['spectral_index'],
                                                                             self.reco_gamma_data.mc_energy,
                                                                             self.run_config['scattering_surface']
                                                                             )
                    self.ax_eff_area.plot(E_reco[:-1], S_reco,
                                          label=self.name + '_reco',
                                          color=self.color,
                                          linestyle=':')
            else:
                print('Cannot evaluate the effective area for experiment {}'.format(self.name))

    def plot_roc_curve(self, ax=None):
        if self.get_loaded():
            self.ax_roc = plt.gca() if ax is None else ax
            if 'reco_hadroness' in self.data:
                fpr, tpr, _ = roc_curve(self.data.mc_particle,
                                        self.data.reco_hadroness, pos_label=1)
                self.auc = roc_auc_score(self.data.mc_particle,
                                         self.data.reco_hadroness)
                true_positive = self.gamma_data[self.gamma_data.reco_particle == 0]
                proton = self.data[self.data.mc_particle == 1]
                false_positive = proton[self.data.reco_particle == 0]
            elif 'reco_gammaness' in self.data:
                fpr, tpr, _ = roc_curve(self.data.mc_particle,
                                        self.data.reco_gammaness, pos_label=1)
                self.auc = roc_auc_score(self.data.mc_particle,
                                         self.data.reco_gammaness)
                true_positive = self.gamma_data[self.gamma_data.reco_particle == 1]
                proton = self.data[self.data.mc_particle == 0]
                false_positive = proton[self.data.reco_particle == 1]
            else:
                raise ValueError

            self.precision = len(true_positive) / (len(true_positive) + len(false_positive))
            self.recall = len(true_positive) / len(self.gamma_data)
            correct = len(self.data[self.data.mc_particle == self.data.reco_particle])
            self.accuracy = correct / len(self.data)
            self.ax_roc.plot(fpr, tpr, label=self.name, color=self.color)
            self.set_plotted(True)

    def visibility_angular_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_ang_res.containers:
                if self.name == c.get_label() or self.name + '_reco' == c.get_label():
                    change_errorbar_visibility(c, visible)

    def visibility_energy_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_ene_res.containers:
                if self.name == c.get_label() or self.name + '_reco' == c.get_label():
                    change_errorbar_visibility(c, visible)

    def visibility_impact_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_imp_res.containers:
                if self.name == c.get_label() or self.name + '_reco' == c.get_label():
                    change_errorbar_visibility(c, visible)

    def visibility_effective_area_plot(self, visible: bool):
        if self.get_plotted():
            for l in self.ax_eff_area.lines:
                if l.get_label() in [self.name, self.name + '_reco', self.name + '_triggered']:
                    l.set_visible(visible)

    def visibility_roc_curve_plot(self, visible: bool):
        if self.get_plotted():
            for l in self.ax_roc.lines:
                if l.get_label() == self.name:
                    l.set_visible(visible)

    def visibility_all_plot(self, visible: bool):
        if 'reco_altitude' in self.data and 'reco_azimuth' in self.data:
            self.visibility_angular_resolution_plot(visible)
        if 'reco_energy' in self.data:
            self.visibility_energy_resolution_plot(visible)
        if 'reco_impact_x' in self.data and 'reco_impact_y' in self.data:
            self.visibility_impact_resolution_plot(visible)
        if 'reco_hadroness' in self.data or 'reco_gammaness' in self.data:
            self.visibility_roc_curve_plot(visible)
        if 'mc_energy' in self.data:
            self.visibility_effective_area_plot(visible)

    def plot_energy_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = np.log10(self.gamma_data.mc_energy)
            reco = np.log10(self.gamma_data.reco_energy)
            ax = plots.plot_migration_matrix(mc, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel('True energy [log(E/TeV)]')
            ax.set_ylabel('Reco energy [log(E/TeV)]')
            ax.set_title(self.name)
        return ax

    def plot_altitude_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = self.gamma_data.mc_altitude
            reco = self.gamma_data.reco_altitude
            ax = plots.plot_migration_matrix(mc, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel('True altitude')
            ax.set_ylabel('Reco altitude')
            ax.set_title(self.name)
        return ax

    def plot_azimuth_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
       Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = self.gamma_data.mc_azimuth
            reco = self.gamma_data.reco_azimuth
            ax = plots.plot_migration_matrix(mc, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel('True azimuth')
            ax.set_ylabel('Reco azimuth')
            ax.set_title(self.name)
        return ax

    def plot_impact_x_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = self.gamma_data.mc_impact_x
            reco = self.gamma_data.reco_impact_x
            ax = plots.plot_migration_matrix(mc, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel('True impact X')
            ax.set_ylabel('Reco impact X')
            ax.set_title(self.name)
        return ax

    def plot_impact_y_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = self.gamma_data.mc_impact_y
            reco = self.gamma_data.reco_impact_y
            ax = plots.plot_migration_matrix(mc, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel('True impact Y')
            ax.set_ylabel('Reco impact Y')
            ax.set_title(self.name)
        return ax


def plot_migration_matrices(exp, colorbar=True, **kwargs):
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (25, 5)
    fig, axes = plt.subplots(1, 5, **kwargs)
    if 'reco_energy' in exp.data:
        axes[0] = exp.plot_energy_matrix(ax=axes[0], colorbar=colorbar)
    if 'reco_altitude' in exp.data:
        axes[1] = exp.plot_altitude_matrix(ax=axes[1], colorbar=colorbar)
    if 'reco_azimuth' in exp.data:
        axes[2] = exp.plot_azimuth_matrix(ax=axes[2], colorbar=colorbar)
    if 'reco_impact_x' in exp.data:
        axes[3] = exp.plot_impact_x_matrix(ax=axes[3], colorbar=colorbar)
    if 'reco_impact_y' in exp.data:
        axes[4] = exp.plot_impact_y_matrix(ax=axes[4], colorbar=colorbar)
    fig.tight_layout()
    return fig


def create_resolution_fig(site='south', ref=None):
    """
    Create the figure holding the resolution plots for the dashboard
    axes = [[ax_ang_res, ax_ene_res],[ax_imp_res, None]]
    Args
        site (string)

    Returns
        fig, axes
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    ax_ang_res = axes[0][0]
    ax_ene_res = axes[0][1]
    ax_imp_res = axes[1][0]
    ax_eff_area = axes[1][1]
    ax_roc = axes[2][0]
    ax_legend = axes[2][1]

    if ref == 'performances':
        plots.plot_angular_resolution_cta_performance(site, ax=ax_ang_res, color='black')
        plots.plot_energy_resolution_cta_performance(site, ax=ax_ene_res, color='black')
        plots.plot_effective_area_cta_performance(site, ax=ax_eff_area, color='black')
    elif ref == 'requirements':
        plots.plot_angular_resolution_cta_requirement(site, ax=ax_ang_res, color='black')
        plots.plot_energy_resolution_cta_requirement(site, ax=ax_ene_res, color='black')
        plots.plot_effective_area_cta_requirement(site, ax=ax_eff_area, color='black')
    else:
        ax_eff_area.set_xscale('log')
        ax_eff_area.set_yscale('log')
        ax_eff_area.set_xlabel('Energy [TeV]')
    if ref is not None:
        ax_ang_res.legend()
        ax_ene_res.legend()
        ax_eff_area.legend()

    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=.5)
    ax_roc.set_xlim([-0.05, 1.05])
    ax_roc.set_ylim([-0.05, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    # ax_roc.axis('equal')

    ax_legend.set_axis_off()

    fig.tight_layout()

    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(10)

    return fig, axes


def plot_exp_on_fig(exp, fig):
    """
    Plot an experiment results on a figure create with `create_fig`
    Args
        exp (experiment class)
        fig (`matplotlib.pyplot.fig`)
    """
    axes = fig.get_axes()
    ax_ang_res = axes[0]
    ax_ene_res = axes[1]
    ax_imp_res = axes[2]
    ax_eff_area = axes[3]
    ax_roc = axes[4]

    if 'reco_altitude' in exp.data and 'reco_azimuth' in exp.data:
        exp.plot_angular_resolution(ax=ax_ang_res)
    if 'reco_energy' in exp.data:
        exp.plot_energy_resolution(ax=ax_ene_res)
    if 'reco_impact_x' in exp.data and 'reco_impact_y' in exp.data:
        exp.plot_impact_resolution(ax=ax_imp_res)
    if 'reco_hadroness' in exp.data or 'reco_gammaness' in exp.data:
        exp.plot_roc_curve(ax=ax_roc)
    if 'mc_energy' in exp.data:
        exp.plot_effective_area(ax=ax_eff_area)


def update_legend(visible_experiments, ax):
    experiments = {exp.name: exp for exp in visible_experiments}
    legend_elements = [Line2D([0], [0], marker='o', color=exp.color, label=name)
                       for (name, exp) in sorted(experiments.items())]
    ax.legend(handles=legend_elements, loc='best', ncol=2)


def update_auc_legend(visible_experiments, ax):
    experiments = {exp.name: exp for exp in visible_experiments}
    legend_elements = [Line2D([0], [0], color=exp.color,
                              label='AUC = {:.4f}, Pr = {:.4f}, R = {:.4f}, Acc = {:.4f}'.format(exp.auc,
                                                                                                 exp.precision,
                                                                                                 exp.recall,
                                                                                                 exp.accuracy
                                                                                                 ))
                       for (name, exp) in sorted(experiments.items()) if exp.auc is not None]
    ax.legend(handles=legend_elements, loc='lower right')


def create_plot_on_click(experiments_dict, experiment_info_box, tabs,
                         fig_resolution, visible_experiments, ax_exp, ax_auc):
    def plot_on_click(sender):
        """
        Function to be called when a `ipywidgets.Button` is clicked

        Args
            sender: the object received by `ipywidgets.Button().on_click()`
        """
        exp_name = sender.description
        if exp_name not in experiments_dict:
            pass

        exp = experiments_dict[exp_name]

        if sender.button_style == 'warning':
            sender.button_style = 'success'
            visible = True
            if not exp.get_loaded():
                exp.load_data()
            if exp_name not in tabs.keys():
                tabs[exp_name] = Output()

            experiment_info_box.children = [value for _, value in tabs.items()]
            for i, key, in enumerate(tabs.keys()):
                experiment_info_box.set_title(i, key)
                if key == exp_name:
                    experiment_info_box.selected_index = i

            with tabs[exp_name]:
                try:
                    print_dict(exp.info)
                except:
                    print('Sorry, I have no info on the experiment {}'.format(exp_name))
            visible_experiments.add(exp)
        else:
            sender.button_style = 'warning'
            visible = False
            tabs[exp_name].close()
            tabs.pop(exp_name)
            visible_experiments.remove(exp)

        if not exp.get_plotted() and visible and exp.data is not None:
            plot_exp_on_fig(exp, fig_resolution)

        exp.visibility_all_plot(visible)
        update_legend(visible_experiments, ax_exp)
        update_auc_legend(visible_experiments, ax_auc)

    return plot_on_click


def make_experiments_carousel(experiments_dic, experiment_info_box, tabs, fig_resolution,
                              visible_experiments, ax_legend, ax_auc):
    """
    Make an ipywidget carousel holding a series of `ipywidget.Button` corresponding to
    the list of experiments in experiments_dic
    Args
        experiments_dic (dict): dictionary of experiment class
        experiment_info_box (Tab): the tab container
        tabs (dict): dictionary of active tabs
        fig_resolution
        visible_experiments
        ax_legend
        ax_auc

    Returns
        `ipywidgets.VBox()`
    """
    from ipywidgets import Layout, Button, VBox

    item_layout = Layout(min_height='30px', width='auto')
    items = [Button(layout=item_layout, description=exp_name, button_style='warning')
             for exp_name in np.sort(list(experiments_dic))[::-1]]

    for b in items:
        b.on_click(create_plot_on_click(experiments_dic, experiment_info_box, tabs,
                                        fig_resolution, visible_experiments, ax_legend, ax_auc))

    box_layout = Layout(overflow_y='scroll',
                        border='3px solid black',
                        width='300px',
                        height='600px',
                        flex_flow='columns',
                        display='flex')

    return VBox(children=items, layout=box_layout)


class GammaBoard(object):
    '''
    Args
        experiments_directory (string)
        site (string): 'south' for Paranal and 'north' for LaPalma
        ref (None or string): whether to plot the 'performances' or 'requirements' corresponding to the chosen site
    '''

    def __init__(self, experiments_directory, site='south', ref=None, bias_correction=False, classif_resolution=True):
        self._fig_resolution, self._axes_resolution = create_resolution_fig(site, ref)
        ax_eff_area = self._axes_resolution[1][1]
        ax_legend = self._axes_resolution[2][1]
        ax_roc = self._axes_resolution[2][0]

        ax_eff_area.set_ylim(ax_eff_area.get_ylim())
        self._fig_resolution.subplots_adjust(bottom=0.2)

        self.experiments_dict = {exp_name: Experiment(exp_name, experiments_directory,
                                                      bias_correction, classif_resolution)
                                 for exp_name in os.listdir(experiments_directory)
                                 if os.path.isdir(experiments_directory + '/' + exp_name) and
                                 exp_name + '.h5' in os.listdir(experiments_directory + '/' + exp_name)}

        colors = np.arange(0, 1, 1 / len(self.experiments_dict.keys()), dtype=np.float32)
        np.random.seed(1)
        np.random.shuffle(colors)
        cmap = plt.cm.tab20
        for (key, color) in zip(self.experiments_dict.keys(), colors):
            self.experiments_dict[key].color = cmap(color)

        visible_experiments = set()

        experiment_info_box = Tab()
        tabs = {}

        carousel = make_experiments_carousel(self.experiments_dict, experiment_info_box, tabs,
                                             self._fig_resolution, visible_experiments, ax_legend, ax_roc)

        self.exp_box = HBox([carousel, experiment_info_box])



def open_dashboard():
    """
    Open a temporary copy of the dashboard.
    All changes made in the dashboard by the user will be discarded when closed.

    Returns
    -------

    """
    original_dashboard_path = get('dashboard.ipynb')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dashboard = os.path.join(tmpdir, 'dashboard.ipynb')
        copyfile(original_dashboard_path, tmp_dashboard)
        command = 'jupyter notebook {}'.format(tmp_dashboard)
        os.system(command)
