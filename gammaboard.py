import os
import json
from collections import OrderedDict

import ctaplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from ipywidgets import HBox


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
        data = pd.read_hdf(experiments_directory + '/' + experiment + '/' + experiment + '.h5')
    except:
        print("The hdf5 file for the experiment {} does not exist".format(experiment))
        return None

    return data


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

    def __init__(self, experiment_name, experiments_directory, ax_imp_res):

        self.name = experiment_name
        self.experiments_directory = experiments_directory
        self.ax_imp_res = ax_imp_res
        self.info = load_info(self.name, self.experiments_directory)
        self.data = None
        self.loaded = False
        self.plotted = False
        self.color = None

        self.cm = plt.cm.jet
        self.cm.set_under('w', 1)

    def load_data(self):
        self.data = load_data(self.name, self.experiments_directory)
        if not self.data is None:
            self.set_loaded(True)

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
            self.ax_ang_res = ctaplot.plot_angular_res_per_energy(self.data.reco_altitude,
                                                                  self.data.reco_azimuth,
                                                                  self.data.mc_altitude,
                                                                  self.data.mc_azimuth,
                                                                  self.data.mc_energy,
                                                                  ax=ax,
                                                                  label=self.name,
                                                                  color=self.color)

            self.set_plotted(True)

    def plot_energy_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_ene_res = ctaplot.plot_energy_resolution(self.data.mc_energy,
                                                             self.data.reco_energy,
                                                             ax=ax,
                                                             label=self.name,
                                                             color=self.color)

            self.set_plotted(True)

    def plot_impact_resolution(self, ax=None):
        if self.get_loaded():
            self.ax_imp_res = ctaplot.plot_impact_resolution_per_energy(self.data.reco_impact_x,
                                                                        self.data.reco_impact_y,
                                                                        self.data.mc_impact_x,
                                                                        self.data.mc_impact_y,
                                                                        self.data.mc_energy,
                                                                        ax=ax,
                                                                        label=self.name,
                                                                        color=self.color
                                                                        )
            self.ax_imp_res.set_xscale('log')
            self.ax_imp_res.set_xlabel('Energy [TeV]')
            self.ax_imp_res.set_ylabel('Impact resolution [km]')
            self.set_plotted(True)

    def plot_effective_area(self, ax=None, number_simu_file=32, site='north'):
        if self.get_loaded():
            try:
                e = np.load('energy_gamma_diffuse_psimu.npy')
                simuE = np.concatenate([e for i in range(number_simu_file)])
            except:
                print("No simu energy file")
                pass

            irf = ctaplot.irf_cta()
            site_area = irf.LaPalmaArea if site == 'north' else irf.ParanalArea
            self.ax_eff_area = ctaplot.plot_effective_area_per_energy(simuE,
                                                                      self.data.mc_energy,
                                                                      site_area,
                                                                      ax=ax,
                                                                      label=self.name,
                                                                      color=self.color)

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

    def visibility_all_plot(self, visible: bool):
        if 'reco_altitude' in self.data and 'reco_azimuth' in self.data:
            self.visibility_angular_resolution_plot(visible)
        if 'reco_energy' in self.data:
            self.visibility_energy_resolution_plot(visible)
        if 'reco_impact_x' in self.data and 'reco_impact_y' in self.data:
            self.visibility_impact_resolution_plot(visible)
        self.visibility_effective_area_plot(visible)

    def plot_energy_matrix(self, ax=None):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = np.log10(self.data.mc_energy)
            reco = np.log10(self.data.reco_energy)
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               bins=50,
                                               cmap=self.cm, cmin=1)
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True energy [log(E/TeV)]")
            ax.set_ylabel("Reco energy [log(E/TeV)]")
            ax.set_title(self.name)
        return ax

    def plot_altitude_matrix(self, ax=None):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = self.data.mc_altitude
            reco = self.data.reco_altitude
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               bins=50,
                                               cmap=self.cm, cmin=1)
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True altitude")
            ax.set_ylabel("Reco altitude")
            ax.set_title(self.name)
        return ax

    def plot_azimuth_matrix(self, ax=None):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
       Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = self.data.mc_azimuth
            reco = self.data.reco_azimuth
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               bins=50,
                                               cmap=self.cm, cmin=1)
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True azimuth")
            ax.set_ylabel("Reco azimuth")
            ax.set_title(self.name)
        return ax

    def plot_impact_x_matrix(self, ax=None):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = self.data.mc_impact_x
            reco = self.data.reco_impact_x
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               bins=50,
                                               cmap=self.cm, cmin=1)
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True impact X")
            ax.set_ylabel("Reco impact X")
            ax.set_title(self.name)
        return ax

    def plot_impact_y_matrix(self, ax=None):
        """
        Plot the diffusion matrix (reco vs simu) for the log of the energies of the experiment
        Args
            ax (`matplotlib.pyplot.Axes`)

        Returns
            `matplotlib.pyplot.Axes`
        """

        ax = plt.gca() if ax is None else ax
        if self.get_loaded():
            mc = self.data.mc_impact_y
            reco = self.data.reco_impact_y
            ax = ctaplot.plot_migration_matrix(mc, reco,
                                               ax=ax,
                                               bins=50,
                                               cmap=self.cm, cmin=1)
            ax.plot(mc, mc, color='teal')
            ax.axis('equal')
            ax.set_xlim(mc.min(), mc.max())
            ax.set_ylim(mc.min(), mc.max())
            ax.set_xlabel("True impact Y")
            ax.set_ylabel("Reco impact Y")
            ax.set_title(self.name)
        return ax


def plot_migration_matrices(exp, **kwargs):

    if 'figsize' not in kwargs:
        kwargs['figsize'] = (25, 5)
    fig, axes = plt.subplots(1, 5, **kwargs)
    if 'reco_energy' in exp.data:
        axes[0] = exp.plot_energy_matrix(ax=axes[0])
    if 'reco_altitude' in exp.data:
        axes[1] = exp.plot_altitude_matrix(ax=axes[1])
    if 'reco_azimuth' in exp.data:
        axes[2] = exp.plot_azimuth_matrix(ax=axes[2])
    if 'reco_impact_x' in exp.data:
        axes[3] = exp.plot_impact_x_matrix(ax=axes[3])
    if 'reco_impact_y' in exp.data:
        axes[4] = exp.plot_impact_y_matrix(ax=axes[4])

    fig.tight_layout()
    return fig


def create_resolution_fig():
    """
    Create the figure holding the resolution plots for the dashboard
    axes = [[ax_ang_res, ax_ene_res],[ax_imp_res, None]]

    Returns
        fig, axes
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_ang_res = axes[0][0]
    ax_ene_res = axes[0][1]
    ax_imp_res = axes[1][0]
    ax_eff_area = axes[1][1]

    ctaplot.plot_angular_res_cta_performance('north', ax=ax_ang_res, color='black')
    ctaplot.plot_energy_resolution_cta_performances('north', ax=ax_ene_res, color='black')
    ctaplot.plot_effective_area_performances('north', ax=ax_eff_area, color='black')

    fig.tight_layout()

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
    if 'reco_altitude' in exp.data and 'reco_azimuth' in exp.data:
        exp.plot_angular_resolution(ax=ax_ang_res)
    if 'reco_energy' in exp.data:
        exp.plot_energy_resolution(ax=ax_ene_res)
    if 'reco_impact_x' in exp.data and 'reco_impact_y' in exp.data:
        exp.plot_impact_resolution(ax=ax_imp_res)
    exp.plot_effective_area(ax=ax_eff_area)


def update_legend(visible_experiments, ax_imp_res):

    experiments = {exp.name: exp for exp in visible_experiments}
    legend_elements = [Line2D([0], [0], marker='o', color=exp.color, label=name)
                       for (name, exp) in sorted(experiments.items())]
    ax_imp_res.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1, -0.3), ncol=4)


def create_plot_on_click(experiments_dict, experiment_info_box,
                         fig_resolution, visible_experiments, ax_imp_res):

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

            experiment_info_box.clear_output()
            with experiment_info_box:
                try:
                    # exp_info = experiments_info[experiments_info.experiment == exp_name]
                    # for col in exp_info:
                    #     print(col, exp_info[col].values)
                    print_dict(exp.info)
                except:
                    print("Sorry, I have no info on the experiment {}".format(exp_name))
            visible_experiments.add(exp)
        else:
            sender.button_style = 'warning'
            visible = False
            visible_experiments.remove(exp)

        if not exp.get_plotted() and visible:
            plot_exp_on_fig(exp, fig_resolution)

        exp.visibility_all_plot(visible)
        update_legend(visible_experiments, ax_imp_res)

    return plot_on_click


def make_experiments_carousel(experiments_dic, experiment_info_box, fig_resolution,
                              visible_experiments, ax_imp_res):
    """
    Make an ipywidget carousel holding a series of `ipywidget.Button` corresponding to
    the list of experiments in experiments_dic
    Args
        experiments_dic (dict): dictionary of experiment class

    Returns
        `ipywidgets.VBox()`
    """
    from ipywidgets import Layout, Button, VBox

    item_layout = Layout(min_height='30px', width='auto')
    items = [Button(layout=item_layout, description=exp_name, button_style='warning')
             for exp_name in np.sort(list(experiments_dic))[::-1]]

    for b in items:
        b.on_click(create_plot_on_click(experiments_dic, experiment_info_box,
                                        fig_resolution, visible_experiments, ax_imp_res))

    box_layout = Layout(overflow_y='scroll',
                        border='3px solid black',
                        width='300px',
                        height='300px',
                        flex_flow='columns',
                        display='flex')

    return VBox(children=items, layout=box_layout)


def make_exp_info_box():
    from ipywidgets import Layout, Output
    layout = Layout(border='1px solid black')

    out = Output(layout=layout, description="Experiment info")

    return out


class GammaBoard(object):

    def __init__(self, experiments_directory):
        fig_resolution, axes_resolution = create_resolution_fig()
        ax_imp_res = axes_resolution[1][0]
        ax_eff_area = axes_resolution[1][1]

        ax_eff_area.set_ylim(ax_eff_area.get_ylim())
        fig_resolution.subplots_adjust(bottom=0.2)

        self.experiments_dict = {exp_name: Experiment(exp_name, experiments_directory, ax_imp_res)
                                 for exp_name in os.listdir(experiments_directory)
                                 if os.path.isdir(experiments_directory + '/' + exp_name) and
                                 exp_name + '.h5' in os.listdir(experiments_directory + '/' + exp_name)}
        colors = np.arange(0, 1, 1/len(self.experiments_dict.keys()), dtype=np.float32)
        np.random.shuffle(colors)
        cmap = plt.cm.tab20

        for (key, color) in zip(self.experiments_dict.keys(), colors):
            self.experiments_dict[key].color = cmap(color)

        visible_experiments = set()

        # try:
        #     experiments_info = pd.read_csv(experiments_directory + '/experiments.csv')
        # except:
        #     experiments_info = None
        #     print("The file 'experiments.csv' cannot be found in the experiments directory")

        experiment_info_box = make_exp_info_box()
        # carousel = make_experiments_carousel(self.experiments_dict, experiment_info_box,
        #                                      experiments_info, fig_resolution, visible_experiments, ax_imp_res)
        carousel = make_experiments_carousel(self.experiments_dict, experiment_info_box,
                                             fig_resolution, visible_experiments, ax_imp_res)

        self.exp_box = HBox([carousel, experiment_info_box])

