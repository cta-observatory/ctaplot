import os
import json
from collections import OrderedDict

import ctaplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from ipywidgets import HBox, Tab, Output
from sklearn.metrics import roc_curve, roc_auc_score


def load_data(experiment, experiments_directory):
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
        # print("The hdf5 file for the experiment {} does not exist".format(experiment))
        return None
    return data


def load_number_run(experiment, experiments_directory):
    assert experiment in os.listdir(experiments_directory)

    try:
        num_run = int(pd.read_hdf(experiments_directory + '/' + experiment + '/' + experiment + '.h5',
                           key='runs',
                           )['num'][0])
    except:
        print("Cannot load the number of run for experiment {} file".format(experiment))
        return 0
    return num_run


def dummy_number_of_simulated_events(experiment, experiments_directory, prod=3, particle='gamma'):
    assert experiment in os.listdir(experiments_directory)

    num_run = load_number_run(experiment, experiments_directory)
    if prod == 3:
        if particle == 'proton':
            number_event_per_run = 2000000
        elif particle == 'gamma':
            number_event_per_run = 500000
        else:
            number_event_per_run = 0
    else:
        number_event_per_run = 0

    return num_run * number_event_per_run


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
        # info = "The json file for the experiment {} does not exist".format(experiment)
        return None

    return info


def print_dict(dictionary, indent=''):

    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(indent, key)
            print_dict(value, indent + '    ')
        else:
            print(indent, key, ' : ', value)


def change_errorbar_visibility(err_container, visible:bool):
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

    def __init__(self, experiment_name, experiments_directory):

        self.name = experiment_name
        self.experiments_directory = experiments_directory
        self.info = load_info(self.name, self.experiments_directory)
        self.data = None
        self.gamma_data = None
        self.loaded = False
        self.plotted = False
        self.color = None

        self.cm = plt.cm.jet
        self.cm.set_under('w', 1)

    def load_data(self):
        self.data = load_data(self.name, self.experiments_directory)
        if self.data is not None:
            self.set_loaded(True)
            if 'mc_particle' in self.data:
                self.gamma_data = self.data[self.data.mc_particle == 0]
            else:
                self.gamma_data = self.data

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
            self.ax_ang_res = ctaplot.plot_angular_res_per_energy(self.gamma_data.reco_altitude,
                                                                  self.gamma_data.reco_azimuth,
                                                                  self.gamma_data.mc_altitude,
                                                                  self.gamma_data.mc_azimuth,
                                                                  self.gamma_data.mc_energy,
                                                                  ax=ax,
                                                                  label=self.name,
                                                                  color=self.color)

            self.set_plotted(True)

    def plot_energy_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_ene_res = ctaplot.plot_energy_resolution(self.gamma_data.mc_energy,
                                                             self.gamma_data.reco_energy,
                                                             ax=ax,
                                                             label=self.name,
                                                             color=self.color)

            self.set_plotted(True)

    def plot_impact_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_imp_res = ctaplot.plot_impact_resolution_per_energy(self.gamma_data.reco_impact_x,
                                                                        self.gamma_data.reco_impact_y,
                                                                        self.gamma_data.mc_impact_x,
                                                                        self.gamma_data.mc_impact_y,
                                                                        self.gamma_data.mc_energy,
                                                                        ax=ax,
                                                                        label=self.name,
                                                                        color=self.color
                                                                        )
            self.ax_imp_res.set_xscale('log')
            self.ax_imp_res.set_xlabel('Energy [TeV]')
            self.ax_imp_res.set_ylabel('Impact resolution [km]')
            self.set_plotted(True)

    def dummy_plot_effective_area(self, ax=None, site='north', prod=3):
        if self.get_loaded():
            # number_simu_file = dummy_number_of_simulated_events(self.name,
            #                                                     self.experiments_directory,
            #                                                     prod=prod,
            #                                                     )
            self.ax_eff_area = ax if ax is not None else plt.gca()
            number_simu_file = load_number_run(self.name, self.experiments_directory)

            try:
                e = np.load('energy_gamma_diffuse_psimu.npy')
            except IOError:
                print("No simu energy file")

            if number_simu_file > 0:
                # simuE = np.concatenate([e for i in range(number_simu_file)])
                # irf = ctaplot.irf_cta()
                # site_area = irf.LaPalmaArea if site == 'north' else irf.ParanalArea
                # self.ax_eff_area = ctaplot.plot_effective_area_per_energy(simuE,
                #                                                           self.gamma_data.mc_energy,
                #                                                           site_area,
                #                                                           ax=ax,
                #                                                           label=self.name,
                #                                                           color=self.color)
                # Rough computation of effective area based on the number of simtel files
                # divided by 5 (the runlist in the data is false)
                E, S = ctaplot.ana.effective_area_per_energy_power_law(3e-3, 3.3e2,
                                                                       len(e)*number_simu_file/5, -2,
                                                                       self.gamma_data.mc_energy,
                                                                       18.45e6)
                self.ax_eff_area.plot(E[:-1], S, label=self.name, color=self.color)
            else:
                print("Cannot evaluate the effective area for this experiment")
                self.ax_eff_area = ctaplot.plot_effective_area_per_energy(np.ones(10),
                                                                          np.empty(0),
                                                                          1,
                                                                          )

    def plot_roc_curve(self, ax=None):
        if self.get_loaded():
            self.ax_roc = plt.gca() if ax is None else ax
            fpr, tpr, _ = roc_curve(self.data.mc_particle,
                                    self.data.reco_hadroness, pos_label=1)
            self.auc = roc_auc_score(self.data.mc_particle,
                                     self.data.reco_hadroness)
            self.ax_roc.plot(fpr, tpr, label=self.name, color=self.color)
            self.set_plotted(True)

    def visibility_angular_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_ang_res.containers:
                if c.get_label() == self.name:
                    change_errorbar_visibility(c, visible)

    def visibility_energy_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_ene_res.containers:
                if c.get_label() == self.name:
                    change_errorbar_visibility(c, visible)

    def visibility_impact_resolution_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_imp_res.containers:
                if c.get_label() == self.name:
                    change_errorbar_visibility(c, visible)

    def visibility_effective_area_plot(self, visible: bool):
        if self.get_plotted():
            for c in self.ax_eff_area.containers:
                if c.get_label() == self.name:
                    change_errorbar_visibility(c, visible)

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
        if 'reco_hadroness' in self.data:
            self.visibility_roc_curve_plot(visible)
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
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               colorbar=colorbar,
                                               hist2d_args={'bins': 100,
                                               'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True energy [log(E/TeV)]")
            ax.set_ylabel("Reco energy [log(E/TeV)]")
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
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               colorbar=colorbar,
                                               hist2d_args={'bins': 100,
                                                            'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True altitude")
            ax.set_ylabel("Reco altitude")
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
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               colorbar=colorbar,
                                               hist2d_args={'bins': 100,
                                                            'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True azimuth")
            ax.set_ylabel("Reco azimuth")
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
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               colorbar=colorbar,
                                               hist2d_args={'bins': 100,
                                                            'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True impact X")
            ax.set_ylabel("Reco impact X")
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
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               colorbar=colorbar,
                                               hist2d_args={'bins': 100,
                                                            'cmap': self.cm, 'cmin': 1})
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True impact Y")
            ax.set_ylabel("Reco impact Y")
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
        ctaplot.plot_angular_res_cta_performance(site, ax=ax_ang_res, color='black')
        ctaplot.plot_energy_resolution_cta_performances(site, ax=ax_ene_res, color='black')
        ctaplot.plot_effective_area_performances(site, ax=ax_eff_area, color='black')
    elif ref == 'requirements':
        ctaplot.plot_angular_res_requirement(site, ax=ax_ang_res, color='black')
        ctaplot.plot_energy_resolution_requirements(site, ax=ax_ene_res, color='black')
        ctaplot.plot_effective_area_requirement(site, ax=ax_eff_area, color='black')
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

    ax_legend.set_axis_off()

    fig.tight_layout()

    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(10)

    return fig, axes


def plot_exp_on_fig(exp, fig, site='south'):
    """
    Plot an experiment results on a figure create with `create_fig`
    Args
        exp (experiment class)
        fig (`matplotlib.pyplot.fig`)
        site (string)
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
    if 'reco_hadroness' in exp.data:
        exp.plot_roc_curve(ax=ax_roc)
    exp.dummy_plot_effective_area(ax=ax_eff_area, site=site)


def update_legend(visible_experiments, ax):

    experiments = {exp.name: exp for exp in visible_experiments}
    legend_elements = [Line2D([0], [0], marker='o', color=exp.color, label=name)
                       for (name, exp) in sorted(experiments.items())]
    ax.legend(handles=legend_elements, loc='best', ncol=4)


def update_auc_legend(visible_experiments, ax):
    experiments = {exp.name: exp for exp in visible_experiments}
    legend_elements = [Line2D([0], [0], color=exp.color, label='AUC = {:.4f}'.format(exp.auc))
                       for (name, exp) in sorted(experiments.items()) if hasattr(exp, 'auc')]
    ax.legend(handles=legend_elements, loc='best')


def create_plot_on_click(experiments_dict, experiment_info_box, tabs,
                         fig_resolution, visible_experiments, ax_exp, ax_auc, site='south'):

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
                    print("Sorry, I have no info on the experiment {}".format(exp_name))
            visible_experiments.add(exp)
        else:
            sender.button_style = 'warning'
            visible = False
            tabs[exp_name].close()
            tabs.pop(exp_name)
            visible_experiments.remove(exp)

        if not exp.get_plotted() and visible and exp.data is not None:
            plot_exp_on_fig(exp, fig_resolution, site)

        exp.visibility_all_plot(visible)
        update_legend(visible_experiments, ax_exp)
        update_auc_legend(visible_experiments, ax_auc)

    return plot_on_click


def make_experiments_carousel(experiments_dic, experiment_info_box, tabs, fig_resolution,
                              visible_experiments, ax_legend, ax_auc, site):
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
                                        fig_resolution, visible_experiments, ax_legend, ax_auc, site))

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
    def __init__(self, experiments_directory, site='south', ref=None):

        self._fig_resolution, self._axes_resolution = create_resolution_fig(site, ref)
        ax_eff_area = self._axes_resolution[1][1]
        ax_legend = self._axes_resolution[2][1]
        ax_roc = self._axes_resolution[2][0]

        ax_eff_area.set_ylim(ax_eff_area.get_ylim())
        self._fig_resolution.subplots_adjust(bottom=0.2)

        self.experiments_dict = {exp_name: Experiment(exp_name, experiments_directory)
                                 for exp_name in os.listdir(experiments_directory)
                                 if os.path.isdir(experiments_directory + '/' + exp_name) and
                                 exp_name + '.h5' in os.listdir(experiments_directory + '/' + exp_name)}
        colors = np.arange(0, 1, 1/len(self.experiments_dict.keys()), dtype=np.float32)
        np.random.shuffle(colors)
        cmap = plt.cm.tab20
        for (key, color) in zip(self.experiments_dict.keys(), colors):
            self.experiments_dict[key].color = cmap(color)

        visible_experiments = set()

        experiment_info_box = Tab()
        tabs = {}

        carousel = make_experiments_carousel(self.experiments_dict, experiment_info_box, tabs,
                                             self._fig_resolution, visible_experiments, ax_legend, ax_roc, site)

        self.exp_box = HBox([carousel, experiment_info_box])

