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
from ipywidgets import HBox, Tab, Output, VBox, FloatSlider, Layout, Button, Dropdown, Text, Label
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from .. import plots
from .. import ana
from ..io.dataset import get
from ..io import read_lst_dl2_data


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

    filename = experiments_directory + '/' + experiment + '/' + experiment + '.h5'
    try:
        data = pd.read_hdf(filename, key='data')
    except KeyError:
        try:
            data = read_lst_dl2_data(filename)
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


class Experiment(object):
    r"""Class to deal with an experiment

    Args
        experiment_name (str)
        experiment_directory (str)
        ax_imp_res

    """

    def __init__(self, experiment_name, experiments_directory, bias_correction):

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
        self.gammaness_cut = None
        self.ax_ang_res = None
        self.ax_ene_res = None
        self.ax_imp_res = None
        self.ax_eff_area = None
        self.ax_roc = None
        self.ax_pr = None

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
                    self.gammaness_cut = 0.5
                elif 'reco_gammaness' in self.data:
                    self.rocness = 'Gammaness'
                    self.gamma_data = self.data[self.data.mc_particle == 1]
                    self.reco_gamma_data = self.gamma_data[self.gamma_data.reco_particle == 1]
                    self.gammaness_cut = 0.5
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

            self.set_plotted(True)

    def plot_angular_resolution_reco(self, ax=None):
        if self.get_loaded():
            if self.reco_gamma_data is not None:
                self.ax_ang_res = plots.plot_angular_resolution_per_energy(self.reco_gamma_data.reco_altitude,
                                                                           self.reco_gamma_data.reco_azimuth,
                                                                           self.reco_gamma_data.mc_altitude,
                                                                           self.reco_gamma_data.mc_azimuth,
                                                                           self.reco_gamma_data.mc_energy,
                                                                           bias_correction=self.bias_correction,
                                                                           ax=ax,
                                                                           label=self.name + '_reco',
                                                                           color=self.color,
                                                                           )

    def plot_energy_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_ene_res = plots.plot_energy_resolution(self.gamma_data.mc_energy,
                                                           self.gamma_data.reco_energy,
                                                           bias_correction=self.bias_correction,
                                                           ax=ax,
                                                           label=self.name,
                                                           color=self.color)
            self.set_plotted(True)

    def plot_energy_resolution_reco(self, ax=None):
        if self.get_loaded():
            if self.reco_gamma_data is not None:
                self.ax_ene_res = plots.plot_energy_resolution(self.reco_gamma_data.mc_energy,
                                                               self.reco_gamma_data.reco_energy,
                                                               bias_correction=self.bias_correction,
                                                               ax=ax,
                                                               label=self.name + '_reco',
                                                               color=self.color,
                                                               )

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
            self.ax_imp_res.set_xscale('log')
            self.ax_imp_res.set_xlabel('Energy [TeV]')
            self.ax_imp_res.set_ylabel('Impact resolution [km]')
            self.set_plotted(True)

    def plot_impact_resolution_reco(self, ax=None):
        if self.get_loaded():
            if self.reco_gamma_data is not None:
                self.ax_imp_res = plots.plot_impact_resolution_per_energy(self.reco_gamma_data.reco_impact_x,
                                                                          self.reco_gamma_data.reco_impact_y,
                                                                          self.reco_gamma_data.mc_impact_x,
                                                                          self.reco_gamma_data.mc_impact_y,
                                                                          self.reco_gamma_data.mc_energy,
                                                                          bias_correction=self.bias_correction,
                                                                          ax=ax,
                                                                          label=self.name + '_reco',
                                                                          color=self.color,
                                                                          )

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

            else:
                print('Cannot evaluate the effective area for experiment {}'.format(self.name))

    def plot_effective_area_reco(self, ax=None):
        if self.get_loaded():
            self.ax_eff_area = ax if ax is not None else plt.gca()

            if self.run_config is not None:
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

    def plot_roc_curve(self, ax=None):
        if self.get_loaded():
            self.ax_roc = plt.gca() if ax is None else ax
            if 'reco_hadroness' in self.data:
                fpr, tpr, _ = roc_curve(self.data.mc_particle,
                                        self.data.reco_hadroness, pos_label=1)
                self.auc = roc_auc_score(self.data.mc_particle,
                                         self.data.reco_hadroness)
            elif 'reco_gammaness' in self.data:
                fpr, tpr, _ = roc_curve(self.data.mc_particle,
                                        self.data.reco_gammaness, pos_label=1)
                self.auc = roc_auc_score(self.data.mc_particle,
                                         self.data.reco_gammaness)
            else:
                raise ValueError

            self.ax_roc.plot(fpr, tpr, label=self.name, color=self.color)
            self.set_plotted(True)

    def plot_pr_curve(self, ax=None):
        if self.get_loaded():
            self.ax_pr = plt.gca() if ax is None else ax
            if 'reco_hadroness' in self.data:
                precision, recall, gammaness_cut = precision_recall_curve(self.data.mc_particle,
                                                                      self.data.reco_hadroness)
            elif 'reco_gammaness' in self.data:
                precision, recall, gammaness_cut = precision_recall_curve(self.data.mc_particle,
                                                                      self.data.reco_gammaness)
            else:
                raise ValueError
            self.ax_pr.plot(recall, precision, label=self.name, color=self.color)
            self.set_plotted(True)

    def plot_gammaness_cut(self):
        if self.get_loaded() and self.gammaness_cut is not None and self.ax_pr is not None:
            if 'reco_hadroness' in self.data:
                true_positive = self.gamma_data[self.gamma_data.reco_hadroness < self.gammaness_cut]
                proton = self.data[self.data.mc_particle == 1]
                false_positive = proton[proton.reco_hadroness < self.gammaness_cut]
            elif 'reco_gammaness' in self.data:
                true_positive = self.gamma_data[self.gamma_data.reco_gammaness >= self.gammaness_cut]
                proton = self.data[self.data.mc_particle == 0]
                false_positive = proton[proton.reco_gammaness >= self.gammaness_cut]
            else:
                raise ValueError

            self.precision = len(true_positive) / (len(true_positive) + len(false_positive))
            self.recall = len(true_positive) / len(self.gamma_data)
            self.ax_pr.scatter(self.recall, self.precision, c=[self.color], label=self.name)

    def visibility_angular_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_ang_res.containers:
                if self.name == c.get_label():
                    change_errorbar_visibility(c, visible)

    def visibility_energy_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_ene_res.containers:
                if self.name == c.get_label():
                    change_errorbar_visibility(c, visible)

    def visibility_impact_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_imp_res.containers:
                if self.name == c.get_label():
                    change_errorbar_visibility(c, visible)

    def visibility_angular_resolution_reco_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_ang_res.containers:
                if self.name + '_reco' == c.get_label():
                    change_errorbar_visibility(c, visible)

    def visibility_energy_resolution_reco_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_ene_res.containers:
                if self.name + '_reco' == c.get_label():
                    change_errorbar_visibility(c, visible)

    def visibility_impact_resolution_reco_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_imp_res.containers:
                if self.name + '_reco' == c.get_label():
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

    def visibility_pr_curve_plot(self, visible: bool):
        if self.get_plotted():
            for l in self.ax_pr.lines:
                if l.get_label() == self.name:
                    l.set_visible(visible)

    def visibility_gammaness_cut(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_pr.collections:
                if c.get_label() == self.name:
                    c.set_visible(visible)

    def visibility_all_plot(self, visible: bool):
        if 'reco_altitude' in self.data and 'reco_azimuth' in self.data:
            self.visibility_angular_resolution_plot(visible)
            self.visibility_angular_resolution_reco_plot(visible)
        if 'reco_energy' in self.data:
            self.visibility_energy_resolution_plot(visible)
            self.visibility_energy_resolution_reco_plot(visible)
        if 'reco_impact_x' in self.data and 'reco_impact_y' in self.data:
            self.visibility_impact_resolution_plot(visible)
            self.visibility_impact_resolution_reco_plot(visible)
        if 'reco_hadroness' in self.data or 'reco_gammaness' in self.data:
            self.visibility_roc_curve_plot(visible)
            self.visibility_pr_curve_plot(visible)
            self.visibility_gammaness_cut(visible)
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


def create_resolution_fig(figsize=(12, 16)):
    """
    Create the figure holding the resolution plots for the dashboard
    axes = [[ax_ang_res, ax_ene_res],[ax_imp_res, None]]
    Args

    Returns
        fig, axes
    """

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    ax_ang_res = axes[0][0]
    ax_ene_res = axes[0][1]
    ax_imp_res = axes[1][0]
    ax_eff_area = axes[1][1]
    ax_roc = axes[2][0]
    ax_pr = axes[2][1]

    ax_eff_area.set_xscale('log')
    ax_eff_area.set_yscale('log')
    ax_eff_area.set_xlabel('Energy [TeV]')
    ax_eff_area.set_ylim([100, 1e7])

    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=.5)
    ax_roc.set_xlim([-0.05, 1.05])
    ax_roc.set_ylim([-0.05, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')

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
    ax_pr = axes[5]

    if 'reco_altitude' in exp.data and 'reco_azimuth' in exp.data:
        exp.plot_angular_resolution(ax=ax_ang_res)
        exp.plot_angular_resolution_reco(ax=ax_ang_res)
    if 'reco_energy' in exp.data:
        exp.plot_energy_resolution(ax=ax_ene_res)
        exp.plot_energy_resolution_reco(ax=ax_ene_res)
    if 'reco_impact_x' in exp.data and 'reco_impact_y' in exp.data:
        exp.plot_impact_resolution(ax=ax_imp_res)
        exp.plot_impact_resolution_reco(ax=ax_imp_res)
    if 'reco_hadroness' in exp.data or 'reco_gammaness' in exp.data:
        exp.plot_roc_curve(ax=ax_roc)
        exp.plot_pr_curve(ax=ax_pr)
        exp.plot_gammaness_cut()
    if 'mc_energy' in exp.data:
        exp.plot_effective_area(ax=ax_eff_area)
        exp.plot_effective_area_reco(ax=ax_eff_area)


def update_legend(visible_experiments, fig):
    for l in fig.legends:
        l.remove()
    experiments = {exp.name: exp for exp in visible_experiments}
    legend_elements = [Line2D([0], [0], marker='o', color=exp.color, label=name)
                       for (name, exp) in sorted(experiments.items())]
    fig.legend(handles=legend_elements, loc='best', bbox_to_anchor=(0.9, 0.1), ncol=6)


def update_auc_legend(visible_experiments, ax):
    experiments = {exp.name: exp for exp in visible_experiments}
    auc_legend_elements = [Line2D([0], [0], color=exp.color,
                                  label='AUC = {:.4f}'.format(exp.auc))
                           for (name, exp) in sorted(experiments.items()) if exp.auc is not None]
    ax.legend(handles=auc_legend_elements, loc='lower right')


def update_pr_legend(visible_experiments, ax):
    experiments = {exp.name: exp for exp in visible_experiments}
    pr_legend_elements = [Line2D([0], [0], color=exp.color,
                                 label='Pr = {:.4f}, R = {:.4f}'.format(exp.precision,
                                                                        exp.recall,
                                                                        ))
                          for (name, exp) in sorted(experiments.items()) if exp.gammaness_cut is not None]
    ax.legend(handles=pr_legend_elements, loc='lower left')


def create_plot_on_click(experiments_dict, experiment_info_box, tabs,
                         fig_resolution, visible_experiments):
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
                color = Text(description=exp.name, continuous_update=False)
                color.observe(create_update_color(experiments_dict, fig_resolution, visible_experiments), names='value')
                if exp.gammaness_cut is not None:
                    slider = FloatSlider(value=exp.gammaness_cut, min=0, max=1, step=0.01, description=exp_name)
                    slider.observe(create_update_gammaness_cut(experiments_dict, fig_resolution, visible_experiments),
                                   names='value')
                    item_layout = Layout(min_height='30px', width='200px')
                    b_res = Button(layout=item_layout, description='gamma_resolution', button_style='success')
                    b_res.on_click(create_display_res(exp))
                    b_reco_res = Button(layout=item_layout, description='reco_resolution', button_style='success')
                    b_reco_res.on_click(create_display_res(exp))

                    tabs[exp_name] = VBox([HBox([slider, VBox([b_res, b_reco_res]), HBox([Label('Color'), color])]),
                                           Output()])
                else:
                    tabs[exp_name] = VBox([HBox([Label('Color'), color]), Output()])

            experiment_info_box.children = [value for _, value in tabs.items()]
            for i, key, in enumerate(tabs.keys()):
                experiment_info_box.set_title(i, key)
                if key == exp_name:
                    experiment_info_box.selected_index = i

            for widget in tabs[exp_name].children:
                if isinstance(widget, Output):
                    with widget:
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

        axes = fig_resolution.get_axes()
        ax_auc = axes[4]
        ax_pr = axes[5]

        exp.visibility_all_plot(visible)
        update_auc_legend(visible_experiments, ax_auc)
        update_pr_legend(visible_experiments, ax_pr)
        update_legend(visible_experiments, fig_resolution)

    return plot_on_click


def create_update_gammaness_cut(experiments_dict, fig_resolution, visible_experiments):
    def update_gammaness_cut(change):
        """
        Function to be called when a `ipywidgets.Button` is clicked

        Args
            sender: the object received by `ipywidgets.Button().on_click()`
        """
        exp_name = change['owner'].description
        if exp_name not in experiments_dict:
            pass

        exp = experiments_dict[exp_name]
        exp.gammaness_cut = change['new']

        if 'reco_hadroness' in exp.data:
            exp.reco_gamma_data = exp.gamma_data[exp.gamma_data.reco_hadroness < exp.gammaness_cut]
        elif 'reco_gammaness' in exp.data:
            exp.reco_gamma_data = exp.gamma_data[exp.gamma_data.reco_gammaness >= exp.gammaness_cut]

        axes = fig_resolution.get_axes()
        ax_ang_res = axes[0]
        ax_ene_res = axes[1]
        ax_imp_res = axes[2]
        ax_eff_area = axes[3]
        ax_pr = axes[5]

        for c in ax_ang_res.containers:
            if exp.name + '_reco' == c.get_label():
                c.remove()
                ax_ang_res.containers.remove(c)

        for c in ax_ene_res.containers:
            if exp.name + '_reco' == c.get_label():
                c.remove()
                ax_ene_res.containers.remove(c)

        for c in ax_imp_res.containers:
            if exp.name + '_reco' == c.get_label():
                c.remove()
                ax_imp_res.containers.remove(c)

        for c in ax_eff_area.lines:
            if exp.name + '_reco' == c.get_label():
                c.remove()

        for c in ax_pr.collections:
            if exp.name == c.get_label():
                c.remove()

        if 'reco_altitude' in exp.data and 'reco_azimuth' in exp.data:
            exp.plot_angular_resolution_reco(ax=ax_ang_res)
        if 'reco_energy' in exp.data:
            exp.plot_energy_resolution_reco(ax=ax_ene_res)
        if 'reco_impact_x' in exp.data and 'reco_impact_y' in exp.data:
            exp.plot_impact_resolution_reco(ax=ax_imp_res)
        if 'mc_energy' in exp.data:
            exp.plot_effective_area_reco(ax=ax_eff_area)
        if 'reco_hadroness' in exp.data or 'reco_gammaness' in exp.data:
            exp.plot_gammaness_cut()
        update_pr_legend(visible_experiments, ax_pr)

    return update_gammaness_cut


def create_update_color(experiments_dict, fig_resolution, visible_experiments):
    """
    Function that creates the callback to update curve color.
    Args
        experiments_dict: the dictionary of loaded experiments
        fig_resolution: the figure containing the plots
        visible_experiments: the set of plotted experiments
    """
    def update_color(change):
        """
        Function to be called when a `ipywidgets.Button` is clicked

        Args
            sender: the object received by `ipywidgets.Button().on_click()`
        """
        exp_name = change['owner'].description
        if exp_name not in experiments_dict:
            pass

        exp = experiments_dict[exp_name]
        exp.color = change['new']

        axes = fig_resolution.get_axes()

        for ax in axes:
            for c in ax.containers:
                if exp.name in c.get_label():
                    c.remove()
                    ax.containers.remove(c)

            for l in ax.lines:
                if exp.name in l.get_label():
                    l.remove()

            for c in ax.collections:
                if exp.name in c.get_label():
                    c.remove()

        plot_exp_on_fig(exp, fig_resolution)
        update_legend(visible_experiments, fig_resolution)
        update_auc_legend(visible_experiments, axes[4])

    return update_color


def create_display_res(experiment):
    def display_on_click(sender):
        """
        Function to be called when a `ipywidgets.Button` is clicked

        Args
            sender: the object received by `ipywidgets.Button().on_click()`
        """
        res_type = sender.description

        if sender.button_style == 'warning':
            sender.button_style = 'success'
            visible = True
        elif sender.button_style == 'success':
            sender.button_style = 'warning'
            visible = False
        else:
            raise ValueError
        if 'reco' in res_type:
            if experiment.ax_ang_res is not None:
                experiment.visibility_angular_resolution_reco_plot(visible)
            if experiment.ax_ene_res is not None:
                experiment.visibility_energy_resolution_reco_plot(visible)
            if experiment.ax_imp_res is not None:
                experiment.visibility_impact_resolution_reco_plot(visible)
        else:
            if experiment.ax_ang_res is not None:
                experiment.visibility_angular_resolution_plot(visible)
            if experiment.ax_ene_res is not None:
                experiment.visibility_energy_resolution_plot(visible)
            if experiment.ax_imp_res is not None:
                experiment.visibility_impact_resolution_plot(visible)
    return display_on_click


def create_update_site(gb):
    def update_site(change):
        gb.site = change['new'].lower()
        update_reference_plot(gb)
    return update_site


def create_update_reference(gb):
    def update_reference(change):
        gb.ref = change['new'].lower()
        update_reference_plot(gb)
    return update_reference


def update_reference_plot(gb):
    axes = gb._fig_resolution.get_axes()
    ax_ang_res = axes[0]
    ax_ene_res = axes[1]
    ax_eff_area = axes[3]

    for l in ax_ang_res.lines:
        if 'CTA' in l.get_label():
            l.remove()
    for l in ax_ene_res.lines:
        if 'CTA' in l.get_label():
            l.remove()
    for l in ax_eff_area.lines:
        if 'CTA' in l.get_label():
            l.remove()

    if gb.ref == 'performances':
        plots.plot_angular_resolution_cta_performance(gb.site, ax=ax_ang_res, color='black')
        plots.plot_energy_resolution_cta_performance(gb.site, ax=ax_ene_res, color='black')
        plots.plot_effective_area_cta_performance(gb.site, ax=ax_eff_area, color='black')
    elif gb.ref == 'requirements':
        plots.plot_angular_resolution_cta_requirement(gb.site, ax=ax_ang_res, color='black')
        plots.plot_energy_resolution_cta_requirement(gb.site, ax=ax_ene_res, color='black')
        plots.plot_effective_area_cta_requirement(gb.site, ax=ax_eff_area, color='black')


def make_experiments_carousel(experiments_dic, experiment_info_box, tabs, fig_resolution,
                              visible_experiments):
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
    item_layout = Layout(min_height='30px', width='auto')
    items = [Button(layout=item_layout, description=exp_name, button_style='warning')
             for exp_name in np.sort(list(experiments_dic))[::-1]]

    for b in items:
        b.on_click(create_plot_on_click(experiments_dic, experiment_info_box, tabs,
                                        fig_resolution, visible_experiments))

    box_layout = Layout(overflow_y='scroll',
                        border='3px solid black',
                        width='300px',
                        height='600px',
                        flex_flow='columns',
                        display='flex')

    return VBox(children=items, layout=box_layout)


class GammaBoard(object):
    """
        Args
            experiments_directory (string)
            site (string): 'south' for Paranal and 'north' for LaPalma
            ref (None or string): whether to plot the 'performances' or 'requirements' corresponding to the chosen site
    """
    def __init__(self, experiments_directory, bias_correction=False, figsize=(12,16)):

        self.experiments_dict = {exp_name: Experiment(exp_name, experiments_directory,
                                                      bias_correction)
                                 for exp_name in os.listdir(experiments_directory)
                                 if os.path.isdir(experiments_directory + '/' + exp_name) and
                                 exp_name + '.h5' in os.listdir(experiments_directory + '/' + exp_name)}
        self.site = 'north'
        self.ref = 'none'

        colors = np.arange(0, 1, 1 / len(self.experiments_dict.keys()), dtype=np.float32)
        np.random.seed(1)
        np.random.shuffle(colors)
        cmap = plt.cm.tab20
        for (key, color) in zip(self.experiments_dict.keys(), colors):
            self.experiments_dict[key].color = cmap(color)

        visible_experiments = set()

        experiment_info_box = Tab()
        tabs = {}

        self._fig_resolution, self._axes_resolution = create_resolution_fig(figsize=figsize)
        ax_eff_area = self._axes_resolution[1][1]
        ax_eff_area.set_ylim(ax_eff_area.get_ylim())
        self._fig_resolution.subplots_adjust(bottom=0.2)

        carousel = make_experiments_carousel(self.experiments_dict, experiment_info_box, tabs,
                                             self._fig_resolution, visible_experiments)
        site_selector = Dropdown(options=['North', 'South'], value='North', description='Site')
        site_selector.observe(create_update_site(self), names='value')

        reference_selector = Dropdown(options=['None', 'performances', 'requirements'],
                                      value='None', description='Reference')
        reference_selector.observe(create_update_reference(self), names='value')

        self.exp_box = VBox([HBox([site_selector, reference_selector]), HBox([carousel, experiment_info_box])])


def open_dashboard(name='dashboard.ipynb'):
    """
    Open a temporary copy of the dashboard.
    All changes made in the dashboard by the user will be discarded when closed.

    Returns
    -------

    """
    original_dashboard_path = get(name)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dashboard = os.path.join(tmpdir, name)
        copyfile(original_dashboard_path, tmp_dashboard)
        command = 'jupyter notebook {}'.format(tmp_dashboard)
        os.system(command)
