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
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.multiclass import LabelBinarizer
from astropy.table import Table
import astropy.units as u
from tqdm.auto import tqdm
from .. import plots
from .. import ana
from ..io.dataset import get

__all__ = ['open_dashboard',
           'load_data_from_h5',
           'GammaBoard',
           'plot_migration_matrices'
           ]

GAMMA_ID = 0
ELECTRON_ID = 1
PROTON_ID = 101


def find_data_files(experiment, experiments_directory):
    """
    Find in the experiment folder all the hdf5 files containing results

    Args
        experiment (str): the name of the experiment
        experiments_directory (str): the path to the folder containing the experiment folders

    Returns
        List of files
    """
    data_folder = experiments_directory + '/' + experiment
    file_set = set()
    for dirname, dirnames, filenames in os.walk(data_folder):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            filepath = os.path.join(dirname, file)
            if ext == '.h5' and guess_particle_type_from_file(filepath) in [GAMMA_ID, PROTON_ID]:
                file_set.add(dirname + '/' + file)
    return tuple(file_set)


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

    from .reader import read_params
    from astropy.table import vstack
    result_files = find_data_files(experiment, experiments_directory)
    result_data = [read_params(r_file) for r_file in tqdm(result_files)]
    return vstack(result_data)


# TODO Find a more suitable naming
def load_trig_events(experiment, experiments_directory):
    assert experiment in os.listdir(experiments_directory)

    result_files = find_data_files(experiment, experiments_directory)
    trig_energies = []
    for file in result_files:
        try:
            result_file = tables.open_file(file)
        except Exception as e:
            print('Could not open data file {}'.format(file))
            print(e)
        else:
            try:
                if guess_particle_type_from_file(file) == GAMMA_ID:
                    result_file.close()
                    trig_energies.append(pd.read_hdf(file, key='triggered_events'))
            except:
                print("Cannot load the number of triggered events for experiment {} file".format(experiment))
                return None
    energies = Table.from_pandas(pd.concat(trig_energies))
    energies['mc_trig_energies'] *= u.TeV
    return energies


def guess_particle_type_from_file(filename):
    """
    Ever changing data formats...
    Try guessing the simulated particle type from the data format.

    Parameters
    ----------
    filename: path, str

    Returns
    -------
    int: particle id
        if None, no identified particle type
    """
    def get_id_from_array(array):
        if len(set(array)) == 1:
            return set(array)[0]
        else:
            return None

    with tables.open_file(filename) as tab:
        try:
            array = tab.root.simulation.event.subarray.shower.col('true_shower_primary_id')
            return get_id_from_array(array)
        except:
            pass
        try:
            array = tab.root.dl1.event.telescope.parameters.LST_LSTCam.col('mc_type')
            return get_id_from_array(array)
        except:
            pass
        try:
            array = tab.root.dl2.event.telescope.parameters.LST_LSTCam.col('mc_type')
            return get_id_from_array(array)
        except:
            pass
        try:
            array = tab.root.data.col('mc_type')
            return get_id_from_array(array)
        except:
            pass
        try:
            mc_type = tab.root.simulation._v_attrs['mc_type']
            return mc_type
        except:
            pass

    if 'gamma' in filename:
        return GAMMA_ID
    elif 'proton' in filename:
        return PROTON_ID
    elif 'electron' in filename:
        return ELECTRON_ID
    else:
        return None


def load_run_configs(experiment, experiments_directory):
    assert experiment in os.listdir(experiments_directory)
    result_files = find_data_files(experiment, experiments_directory)

    run_configs = {}

    for file in result_files:

        num_showers = 0
        try:
            result_file = tables.open_file(file)
        except Exception as e:
            print('Could not open data file {}'.format(file))
            print(e)
        else:
            try:
                mc_type = guess_particle_type_from_file(file)
                if mc_type is None:
                    raise ValueError(f"No particle type found for file {file}")
                run_config = result_file.root.simulation.run_config
                r_config = {}
                for row in run_config:
                    num_showers += row['num_showers'] * row['shower_reuse']
                r_config['num_showers'] = num_showers
                for col in run_config.colnames:
                    if col not in ['num_showers', 'detector_prog_start', 'shower_prog_start']:
                        try:
                            assert len(np.unique(run_config[:][col])) == (1 if col != 'run_array_direction' else 2)
                        except AssertionError:
                            print('Cannot deal with different {} for particle {} '
                                  'in the experiment ({})'.format(col, mc_type, experiment))
                            r_config[col] = None
                        else:
                            r_config[col] = np.unique(run_config[:][col])

                r_config['min_alt'] *= u.rad
                r_config['max_alt'] *= u.rad
                r_config['min_az'] *= u.rad
                r_config['max_az'] *= u.rad
                r_config['energy_range_min'] *= u.TeV
                r_config['energy_range_max'] *= u.TeV
                r_config['min_scatter_range'] *= u.m
                r_config['max_scatter_range'] *= u.m
                r_config['min_viewcone_radius'] *= u.rad
                r_config['max_viewcone_radius'] *= u.rad
                r_config['scattering_surface'] = r_config['max_scatter_range'] ** 2 * np.pi * np.sin(
                    r_config['max_alt'])

            except:
                print('Could not load run config info from file {}'.format(file))
            else:
                run_configs[mc_type] = r_config
    return run_configs


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
        self.noise_reco_gamma = None
        self.mc_trig_events = None
        self.run_configs = None
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
        self.ax_eff_area_ratio = None
        self.ax_roc = None
        self.ax_pr = None

        self.cm = plt.cm.jet
        self.cm.set_under('w', 1)

    def load_data(self):
        self.data = load_data_from_h5(self.name, self.experiments_directory)
        if self.data is not None:
            self.set_loaded(True)
            if 'true_particle' in self.data.columns:
                self.gamma_data = self.data[self.data['true_particle'] == GAMMA_ID]
                self.reco_gamma_data = self.gamma_data[self.gamma_data['reco_particle'] == GAMMA_ID]
                noise_mask = (self.data['true_particle'] != GAMMA_ID) & (self.data['reco_particle'] == GAMMA_ID)
                self.noise_reco_gamma = self.data[noise_mask]
                self.gammaness_cut = 1 / len(np.unique(self.data['true_particle']))\
                    if 'reco_gammaness' in self.data.columns else None
            else:
                self.gamma_data = self.data
        self.mc_trig_events = load_trig_events(self.name, self.experiments_directory)
        self.run_configs = load_run_configs(self.name, self.experiments_directory)

    def update_gammaness_cut(self, new_cut):

        self.gammaness_cut = new_cut
        self.reco_gamma_data = self.gamma_data[self.gamma_data['reco_particle'] >= self.gammaness_cut]
        noise_mask = (self.data['true_particle'] != GAMMA_ID) & (self.data['reco_particle'] >= self.gammaness_cut)
        self.noise_reco_gamma = self.data[noise_mask]

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
            self.ax_ang_res = plots.plot_angular_resolution_per_energy(self.gamma_data['true_altitude'].quantity,
                                                                       self.gamma_data['reco_altitude'].quantity,
                                                                       self.gamma_data['true_azimuth'].quantity,
                                                                       self.gamma_data['reco_azimuth'].quantity,
                                                                       self.gamma_data['true_energy'].quantity,
                                                                       bias_correction=self.bias_correction,
                                                                       ax=ax,
                                                                       label=self.name,
                                                                       color=self.color)

            self.set_plotted(True)

    def plot_angular_resolution_reco(self, ax=None):
        if self.get_loaded() and self.reco_gamma_data is not None:
            self.ax_ang_res = plots.plot_angular_resolution_per_energy(self.reco_gamma_data['true_altitude'].quantity,
                                                                       self.reco_gamma_data['reco_altitude'].quantity,
                                                                       self.reco_gamma_data['true_azimuth'].quantity,
                                                                       self.reco_gamma_data['reco_azimuth'].quantity,
                                                                       self.reco_gamma_data['true_energy'].quantity,
                                                                       bias_correction=self.bias_correction,
                                                                       ax=ax,
                                                                       label=self.name + '_reco',
                                                                       color=self.color,
                                                                       )

    def update_angular_resolution_reco(self, ax):
        for c in ax.containers:
            if self.name + '_reco' == c.get_label():
                c.remove()
                ax.containers.remove(c)
        self.plot_angular_resolution_reco(ax)

    def plot_energy_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_ene_res = plots.plot_energy_resolution(self.gamma_data['true_energy'].quantity,
                                                           self.gamma_data['reco_energy'].quantity,
                                                           bias_correction=self.bias_correction,
                                                           ax=ax,
                                                           label=self.name,
                                                           color=self.color)
            self.set_plotted(True)

    def plot_energy_resolution_reco(self, ax=None):
        if self.get_loaded() and self.reco_gamma_data is not None:
            self.ax_ene_res = plots.plot_energy_resolution(self.reco_gamma_data['true_energy'].quantity,
                                                           self.reco_gamma_data['reco_energy'].quantity,
                                                           bias_correction=self.bias_correction,
                                                           ax=ax,
                                                           label=self.name + '_reco',
                                                           color=self.color,
                                                           )

    def update_energy_resolution_reco(self, ax):
        for c in ax.containers:
            if self.name + '_reco' == c.get_label():
                c.remove()
                ax.containers.remove(c)
        self.plot_energy_resolution_reco(ax)

    def plot_impact_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_imp_res = plots.plot_impact_resolution_per_energy(self.gamma_data['true_core_x'].quantity,
                                                                      self.gamma_data['reco_core_x'].quantity,
                                                                      self.gamma_data['true_core_y'].quantity,
                                                                      self.gamma_data['reco_core_y'].quantity,
                                                                      self.gamma_data['true_energy'].quantity,
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
        if self.get_loaded() and self.reco_gamma_data is not None:
            self.ax_imp_res = plots.plot_impact_resolution_per_energy(self.gamma_data['true_core_x'].quantity,
                                                                      self.gamma_data['reco_core_x'].quantity,
                                                                      self.gamma_data['true_core_y'].quantity,
                                                                      self.gamma_data['reco_core_y'].quantity,
                                                                      self.gamma_data['true_energy'].quantity,
                                                                      bias_correction=self.bias_correction,
                                                                      ax=ax,
                                                                      label=self.name + '_reco',
                                                                      color=self.color,
                                                                      )

    def update_impact_resolution_reco(self, ax):
        for c in ax.containers:
            if self.name + '_reco' == c.get_label():
                c.remove()
                ax.containers.remove(c)
        self.plot_impact_resolution_reco(ax)

    def plot_effective_area(self, ax=None):
        if self.get_loaded():
            self.ax_eff_area = ax if ax is not None else plt.gca()

            if self.run_configs is not None and GAMMA_ID in self.run_configs:
                if self.mc_trig_events is not None:
                    E_trig, S_trig = ana.effective_area_per_energy_power_law(self.run_configs[0]['energy_range_min'],
                                                                             self.run_configs[0]['energy_range_max'],
                                                                             self.run_configs[0]['num_showers'],
                                                                             self.run_configs[0]['spectral_index'],
                                                                             self.mc_trig_events['mc_trig_energies'],
                                                                             self.run_configs[0]['scattering_surface'])
                    self.ax_eff_area.plot(E_trig[:-1], S_trig, label=self.name + '_triggered', color=self.color,
                                          linestyle='-.')

                E, S = ana.effective_area_per_energy_power_law(self.run_configs[0]['energy_range_min'],
                                                               self.run_configs[0]['energy_range_max'],
                                                               self.run_configs[0]['num_showers'],
                                                               self.run_configs[0]['spectral_index'],
                                                               self.gamma_data['reco_energy'],
                                                               self.run_configs[0]['scattering_surface'])
                self.ax_eff_area.plot(E[:-1], S, label=self.name, color=self.color)

            else:
                print('Cannot evaluate the effective area for experiment {}'.format(self.name))

    def plot_effective_area_reco(self, ax=None):
        if self.get_loaded():
            self.ax_eff_area = ax if ax is not None else plt.gca()

            if (
                self.run_configs is not None
                and GAMMA_ID in self.run_configs
                and self.reco_gamma_data is not None
            ):
                E_reco, S_reco = ana.effective_area_per_energy_power_law(self.run_configs[0]['energy_range_min'],
                                                                         self.run_configs[0]['energy_range_max'],
                                                                         self.run_configs[0]['num_showers'],
                                                                         self.run_configs[0]['spectral_index'],
                                                                         self.reco_gamma_data['reco_energy'],
                                                                         self.run_configs[0]['scattering_surface']
                                                                         )
                E_reco_prot, S_reco_prot = ana.effective_area_per_energy_power_law(
                    self.run_configs[0]['energy_range_min'],
                    self.run_configs[0]['energy_range_max'],
                    self.run_configs[0]['num_showers'],
                    self.run_configs[0]['spectral_index'],
                    self.noise_reco_gamma['reco_energy'],
                    self.run_configs[0]['scattering_surface']
                )
                self.ax_eff_area.plot(E_reco[:-1], S_reco,
                                      label=self.name + '_reco',
                                      color=self.color,
                                      linestyle=':')
                self.ax_eff_area.plot(E_reco_prot[:-1], S_reco_prot,
                                      label=self.name + '_reco_noise',
                                      color=self.color,
                                      linestyle='--')

    def update_effective_area_reco(self, ax):
        to_remove = []
        for l in ax.lines:
            if l.get_label() in [self.name + '_reco', self.name + '_reco_noise']:
                to_remove.append(l)
        while len(to_remove) > 0:
            to_remove.pop().remove()
        self.plot_effective_area_reco(ax)

    def plot_effective_area_ratio(self, ax=None):
        if not self.get_loaded():
            return
        self.ax_eff_area_ratio = ax if ax is not None else plt.gca()

        if (
            self.run_configs is not None
            and GAMMA_ID in self.run_configs
            and self.mc_trig_events is not None
        ):
            E_max, S_max = ana.effective_area_per_energy_power_law(self.run_configs[0]['energy_range_min'],
                                                                   self.run_configs[0]['energy_range_max'],
                                                                   self.run_configs[0]['num_showers'],
                                                                   self.run_configs[0]['spectral_index'],
                                                                   self.mc_trig_events['mc_trig_energies'],
                                                                   self.run_configs[0]['scattering_surface'])
            E_reco, S_reco = ana.effective_area_per_energy_power_law(self.run_configs[0]['energy_range_min'],
                                                                     self.run_configs[0]['energy_range_max'],
                                                                     self.run_configs[0]['num_showers'],
                                                                     self.run_configs[0]['spectral_index'],
                                                                     self.reco_gamma_data['reco_energy'],
                                                                     self.run_configs[0]['scattering_surface']
                                                                     )
            E_reco_prot, S_reco_prot = ana.effective_area_per_energy_power_law(
                self.run_configs[0]['energy_range_min'],
                self.run_configs[0]['energy_range_max'],
                self.run_configs[0]['num_showers'],
                self.run_configs[0]['spectral_index'],
                self.noise_reco_gamma['reco_energy'],
                self.run_configs[0]['scattering_surface']
            )
            assert np.all(E_reco_prot == E_max) and np.all(E_reco == E_max), \
                'To compute effective area ratio, the true_energy bins must be the same'

            self.ax_eff_area_ratio.plot(E_reco[:-1], S_reco / S_max,
                                        label=self.name + '_ratio_gamma',
                                        color=self.color,
                                        linestyle=':')
            self.ax_eff_area_ratio.plot(E_reco_prot[:-1], S_reco_prot / S_max,
                                        label=self.name + '_ratio_noise',
                                        color=self.color,
                                        linestyle='--')

    def update_effective_area_ratio(self, ax):
        to_remove = []
        for l in ax.lines:
            if l.get_label() in [self.name + '_ratio_gamma', self.name + '_ratio_noise']:
                to_remove.append(l)
        while len(to_remove) > 0:
            to_remove.pop().remove()
        self.plot_effective_area_ratio(ax)

    def plot_roc_curve(self, ax=None):
        if self.get_loaded() and 'reco_gammaness' in self.data.columns:
            self.ax_roc = plots.plot_roc_curve_gammaness(self.data['true_particle'],
                                                         self.data['reco_gammaness'],
                                                         label=self.name,
                                                         ax=ax,
                                                         color=self.color)
            binarized_class = np.ones_like(self.data['true_particle'])
            binarized_class[self.data['true_particle'] != GAMMA_ID] = 0
            self.auc = roc_auc_score(binarized_class, self.data['reco_gammaness'], )
            self.set_plotted(True)

    def plot_pr_curve(self, ax=None):
        if not self.get_loaded():
            return
        self.ax_pr = plt.gca() if ax is None else ax
        if 'reco_gammaness' not in self.data.columns:
            raise ValueError
        label_binarizer = LabelBinarizer()
        binarized_classes = label_binarizer.fit_transform(self.data['true_particle'])
        ii = np.where(label_binarizer.classes_ == GAMMA_ID)[0][0]
        precision, recall, _ = precision_recall_curve(binarized_classes[:, ii],
                                                      self.data['reco_gammaness'],
                                                      pos_label=GAMMA_ID)
        self.ax_pr.plot(recall, precision, label=self.name, color=self.color)
        self.set_plotted(True)

    def plot_gammaness_cut(self):
        if (
            not self.get_loaded()
            or self.gammaness_cut is None
            or self.ax_pr is None
        ):
            return
        if 'reco_gammaness' not in self.data.columns:
            raise ValueError
        true_positive = self.gamma_data[self.gamma_data['reco_gammaness'] >= self.gammaness_cut]
        noise = self.data[self.data['true_particle'] != GAMMA_ID]
        false_positive = noise[noise['reco_gammaness'] >= self.gammaness_cut]
        try:
            self.precision = len(true_positive) / (len(true_positive) + len(false_positive))
            self.recall = len(true_positive) / len(self.gamma_data)
        except Exception as e:
            print('Plot gammaness cut ', e)
            self.precision = 0
            self.recall = 0
        self.ax_pr.scatter(self.recall, self.precision, c=[self.color], label=self.name)

    def update_pr_cut(self, ax):
        for c in ax.collections:
            if self.name == c.get_label():
                c.remove()
        self.plot_gammaness_cut()

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
                if l.get_label() in [self.name, self.name + '_reco', self.name + '_triggered',
                                     self.name + '_reco_noise']:
                    l.set_visible(visible)

    def visibility_effective_area_ratio_plot(self, visible: bool):
        if self.get_plotted():
            for l in self.ax_eff_area_ratio.lines:
                if l.get_label() in [self.name + '_ratio_gamma',
                                     self.name + '_ratio_noise']:
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
        if 'reco_altitude' in self.data.columns and 'reco_azimuth' in self.data.columns:
            self.visibility_angular_resolution_plot(visible)
            self.visibility_angular_resolution_reco_plot(visible)
        if 'true_energy' in self.data.columns:
            self.visibility_energy_resolution_plot(visible)
            self.visibility_energy_resolution_reco_plot(visible)
        if 'reco_core_x' in self.data.columns and 'reco_core_y' in self.data.columns:
            self.visibility_impact_resolution_plot(visible)
            self.visibility_impact_resolution_reco_plot(visible)
        if 'reco_gammaness' in self.data.columns:
            self.visibility_roc_curve_plot(visible)
            self.visibility_pr_curve_plot(visible)
            self.visibility_gammaness_cut(visible)
        if 'true_energy' in self.data.columns:
            self.visibility_effective_area_plot(visible)
            self.visibility_effective_area_ratio_plot(visible)

    def plot_energy_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs true) for the log of the energy of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            simu = np.log10(self.gamma_data['true_energy'])
            reco = np.log10(self.gamma_data['reco_energy'])
            ax = plots.plot_migration_matrix(simu, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm,
                                                 'cmin': 1},
                                             )
            ax.plot(simu, simu, color='teal')
            ax.axis('equal')
            ax.set_xlim(simu.min(), simu.max())
            ax.set_ylim(simu.min(), simu.max())
            ax.set_xlabel('True energy [log(energy/TeV)]')
            ax.set_ylabel('Reco energy [log(energy/TeV)]')
            ax.set_title(self.name)
        return ax

    def plot_altitude_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs true) for the log of the energy of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            simu = self.gamma_data['true_altitude']
            reco = self.gamma_data['reco_altitude']
            ax = plots.plot_migration_matrix(simu, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm,
                                                 'cmin': 1},
                                             )
            ax.plot(simu, simu, color='teal')
            ax.axis('equal')
            ax.set_xlim(simu.min(), simu.max())
            ax.set_ylim(simu.min(), simu.max())
            ax.set_xlabel('True altitude')
            ax.set_ylabel('Reco altitude')
            ax.set_title(self.name)
        return ax

    def plot_azimuth_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs true) for the log of the energy of the experiment
       Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            simu = self.gamma_data['true_azimuth']
            reco = self.gamma_data['reco_azimuth']
            ax = plots.plot_migration_matrix(simu, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm,
                                                 'cmin': 1},
                                             )
            ax.plot(simu, simu, color='teal')
            ax.axis('equal')
            ax.set_xlim(simu.min(), simu.max())
            ax.set_ylim(simu.min(), simu.max())
            ax.set_xlabel('True azimuth')
            ax.set_ylabel('Reco azimuth')
            ax.set_title(self.name)
        return ax

    def plot_impact_x_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs true) for the log of the energy of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            simu = self.gamma_data['true_core_x']
            reco = self.gamma_data['reco_core_x']
            ax = plots.plot_migration_matrix(simu, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm,
                                                 'cmin': 1},
                                             )
            ax.plot(simu, simu, color='teal')
            ax.axis('equal')
            ax.set_xlim(simu.min(), simu.max())
            ax.set_ylim(simu.min(), simu.max())
            ax.set_xlabel('True core X')
            ax.set_ylabel('Reco core X')
            ax.set_title(self.name)
        return ax

    def plot_impact_y_matrix(self, ax=None, colorbar=True):
        """
        Plot the diffusion matrix (reco vs true) for the log of the energy of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            simu = self.gamma_data['true_core_y']
            reco = self.gamma_data['reco_core_y']
            ax = plots.plot_migration_matrix(simu, reco,
                                             ax=ax,
                                             colorbar=colorbar,
                                             hist2d_args={
                                                 'bins': 100,
                                                 'cmap': self.cm,
                                                 'cmin': 1},
                                             )
            ax.plot(simu, simu, color='teal')
            ax.axis('equal')
            ax.set_xlim(simu.min(), simu.max())
            ax.set_ylim(simu.min(), simu.max())
            ax.set_xlabel('True core Y')
            ax.set_ylabel('Reco core Y')
            ax.set_title(self.name)
        return ax


def plot_migration_matrices(exp, colorbar=True, **kwargs):
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (25, 5)
    fig, axes = plt.subplots(1, 5, **kwargs)
    if 'true_energy' in exp.data.columns:
        axes[0] = exp.plot_energy_matrix(ax=axes[0], colorbar=colorbar)
    if 'reco_altitude' in exp.data.columns:
        axes[1] = exp.plot_altitude_matrix(ax=axes[1], colorbar=colorbar)
    if 'reco_azimuth' in exp.data.columns:
        axes[2] = exp.plot_azimuth_matrix(ax=axes[2], colorbar=colorbar)
    if 'reco_core_x' in exp.data.columns:
        axes[3] = exp.plot_impact_x_matrix(ax=axes[3], colorbar=colorbar)
    if 'reco_core_y' in exp.data.columns:
        axes[4] = exp.plot_impact_y_matrix(ax=axes[4], colorbar=colorbar)
    fig.tight_layout()
    return fig


def create_resolution_fig(figsize=(12, 20)):
    """
    Create the figure holding the resolution plots for the dashboard
    axes = [[ax_ang_res, ax_ene_res],[ax_imp_res, None]]
    Args

    Returns
        fig, axes
    """

    fig, axes = plt.subplots(4, 2, figsize=figsize)

    ax_eff_area = axes[1][1]
    ax_eff_area_ratio = axes[2][1]
    ax_roc = axes[2][0]
    ax_pr = axes[3][0]
    legend_ax = axes[3][1]

    ax_eff_area.set_xscale('log')
    ax_eff_area.set_yscale('log')
    ax_eff_area.set_xlabel('Reco Energy [TeV]')
    ax_eff_area.set_ylabel(r'Effective Area $[m^2]$')
    ax_eff_area.set_title('Effective area')
    ax_eff_area.set_ylim([100, 1e7])
    ax_eff_area_legend_elements = [
        Line2D([0], [0], color='black', label='max', ls='-.'),
        Line2D([0], [0], color='black', label='gamma'),
        Line2D([0], [0], color='black', label='gamma reco gamma', ls=':'),
        Line2D([0], [0], color='black', label='noise reco gamma', ls='--')
    ]
    ax_eff_area.legend(handles=ax_eff_area_legend_elements, loc='upper left')

    ax_eff_area_ratio.set_xscale('log')
    ax_eff_area_ratio.set_xlabel('Reco Energy [TeV]')
    ax_eff_area_ratio.set_ylabel(r'Effective Area / Effective Area Max $[m^2]$')
    ax_eff_area_ratio.set_title('Effective area ratio over max')
    ax_eff_area_ratio_legend_elements = [
        Line2D([0], [0], color='black', label='gamma reco gamma', ls=':'),
        Line2D([0], [0], color='black', label='noise reco gamma', ls='--'),
    ]
    ax_eff_area_ratio.legend(handles=ax_eff_area_ratio_legend_elements, loc='upper right')

    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=.5)
    ax_roc.set_xlim([-0.05, 1.05])
    ax_roc.set_ylim([-0.05, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')

    legend_ax.axis('off')

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
    ax_eff_area_ratio = axes[5]
    ax_pr = axes[6]

    if 'reco_altitude' in exp.data.columns and 'reco_azimuth' in exp.data.columns:
        exp.plot_angular_resolution(ax=ax_ang_res)
        exp.plot_angular_resolution_reco(ax=ax_ang_res)
    if 'true_energy' in exp.data.columns:
        exp.plot_energy_resolution(ax=ax_ene_res)
        exp.plot_energy_resolution_reco(ax=ax_ene_res)
    if 'reco_core_x' in exp.data.columns and 'reco_core_y' in exp.data.columns:
        exp.plot_impact_resolution(ax=ax_imp_res)
        exp.plot_impact_resolution_reco(ax=ax_imp_res)
    if 'reco_gammaness' in exp.data.columns:
        exp.plot_roc_curve(ax=ax_roc)
        exp.plot_pr_curve(ax=ax_pr)
        exp.plot_gammaness_cut()
    if 'true_energy' in exp.data.columns:
        exp.plot_effective_area(ax=ax_eff_area)
        exp.plot_effective_area_reco(ax=ax_eff_area)
        exp.plot_effective_area_ratio(ax=ax_eff_area_ratio)


def update_legend(visible_experiments, fig):
    legend_ax = fig.get_axes()[-1]
    experiments = {exp.name: exp for exp in visible_experiments}
    legend_elements = [Line2D([0], [0], marker='o', color=exp.color, label=name)
                       for (name, exp) in sorted(experiments.items())]
    legend_ax.legend(handles=legend_elements, loc='best',
                     ncol=5)


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
        ax_pr = axes[6]

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
        exp.update_gammaness_cut(change['new'])

        axes = fig_resolution.get_axes()
        ax_ang_res = axes[0]
        ax_ene_res = axes[1]
        ax_imp_res = axes[2]
        ax_eff_area = axes[3]
        ax_eff_area_ratio = axes[5]
        ax_pr = axes[6]

        if 'reco_altitude' in exp.data.columns and 'reco_azimuth' in exp.data.columns:
            exp.update_angular_resolution_reco(ax_ang_res)
        if 'true_energy' in exp.data.columns:
            exp.update_energy_resolution_reco(ax_ene_res)
        if 'reco_core_x' in exp.data.columns and 'reco_core_y' in exp.data.columns:
            exp.update_impact_resolution_reco(ax_imp_res)
        if 'true_energy' in exp.data.columns:
            exp.update_effective_area_reco(ax_eff_area)
            exp.update_effective_area_ratio(ax_eff_area_ratio)
        if 'reco_gammaness' in exp.data.columns:
            exp.update_pr_cut(ax_pr)
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
        update_pr_legend(visible_experiments, axes[6])
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


def has_hdf5_file(folder):
    for dirname, dirnames, filenames in os.walk(folder):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            if ext == '.h5':
                return True
    return False


class GammaBoard(object):
    """
        Args
            experiments_directory (string)
            site (string): 'south' for Paranal and 'north' for LaPalma
            ref (None or string): whether to plot the 'performances' or 'requirements' corresponding to the chosen site
    """

    def __init__(self, experiments_directory, bias_correction=False, figsize=(12, 20)):
        self.experiments_dict = {exp_name: Experiment(exp_name, experiments_directory,
                                                      bias_correction)
                                 for exp_name in os.listdir(experiments_directory)
                                 if os.path.isdir(experiments_directory + '/' + exp_name) and
                                 has_hdf5_file(experiments_directory + '/' + exp_name)}
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
