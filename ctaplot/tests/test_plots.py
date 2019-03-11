from ctaplot import plots
import matplotlib.pyplot as plt
import numpy as np


def plot_energy_distribution():
    SimuE = np.random.rand(100)
    RecoE = np.random.rand(10)
    maskSimuDetected = np.ones(100, dtype=bool)
    maskSimuDetected[50:] = False
    plots.plot_energy_distribution(SimuE, RecoE, maskSimuDetected=maskSimuDetected)


def test_plot_energy_resolution():
    plt.close('all')
    E = np.logspace(-2, 2, 10)
    plots.plot_energy_resolution(E, E**2, color='red')


def test_plot_energy_resolution_requirements():
    plt.close('all')
    plots.plot_energy_resolution_requirements('north', color='green')


def test_saveplot_energy_resolution():
    import os
    plt.close('all')
    E = np.logspace(-2, 2, 10)
    plots.saveplot_energy_resolution(E, E**2, Outfile="eres.png")
    assert os.path.isfile("eres.png")
    os.remove("eres.png")


def test_plot_energy_resolution_cta_performances():
    plots.plot_energy_resolution_cta_performances('north', color='green')


def test_plot_angular_resolution_cta_performances():
    plots.plot_angular_res_cta_performance('north', color='green')


def test_plot_angular_resolution_requirements():
    plots.plot_angular_res_requirement('north', color='green')


def test_plot_effective_area_performances():
    plots.plot_effective_area_performances('north', color='green')


def test_plot_effective_area_requirement():
    plots.plot_effective_area_requirement('north', color='green')


def test_plot_sensitivity_performances():
    plots.plot_sensitivity_performances('north', color='green')


def test_plot_sensitivity_requirement():
    plots.plot_sensitivity_requirement('north', color='green')

def test_plot_theta2():
    n = 10
    RecoAlt = 1 + np.random.rand(n)
    RecoAz = 1.5 + np.random.rand(n)
    SimuAlt = np.ones(n)
    SimuAz = 1.5 * np.ones(n)
    plots.plot_theta2(RecoAlt, RecoAz, SimuAlt, SimuAz)

def test_plot_angles_map_distri():
    n = 10
    RecoAlt = 1 + np.random.rand(n)
    RecoAz = 1.5 + np.random.rand(n)
    SimuAlt = 1
    SimuAz = 1.5
    E = 10**(np.random.rand(n)*6 - 3)
    plots.plot_angles_map_distri(RecoAlt, RecoAz, SimuAlt, SimuAz, E)


def test_plot_impact_point_map_distri():
    n = 10
    RecoX = 1000 * np.random.rand(n) - 500
    RecoY = 1000 * np.random.rand(n) - 500
    telX = np.array([10, 100])
    telY = np.array([100, -10])
    plots.plot_impact_point_map_distri(RecoX, RecoY, telX, telY)


def test_plot_impact_point_heatmap():
    n = 10
    RecoX = np.random.rand(n)
    RecoY = np.random.rand(n)
    plots.plot_impact_point_heatmap(RecoX, RecoY)

def test_plot_multiplicity_hist():
    multiplicity = np.random.randint(low=2, high=30, size=100)
    plots.plot_multiplicity_hist(multiplicity)


def test_plot_effective_area_per_energy():
    SimuE = 10**(6 * np.random.rand(100) - 3)
    RecoE = 10 ** (6 * np.random.rand(10) - 3)
    simuArea = 1000
    plots.plot_effective_area_per_energy(SimuE, RecoE, simuArea)


def test_plot_resolution_per_energy():
    simu = np.ones(100)
    reco = np.random.normal(loc=1, scale=1, size=100)
    energy = 10**(-3 + 6 * np.random.rand(100))
    plots.plot_resolution_per_energy(reco, simu, energy)


def test_plot_binned_stat():
    x = np.random.rand(100)
    y = np.random.rand(100)
    for stat in ['min', 'max', 'mean']:
        plots.plot_binned_stat(x, y, statistic=stat)