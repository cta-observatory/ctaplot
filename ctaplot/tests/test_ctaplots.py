from ctaplot import plots
import matplotlib.pyplot as plt




def test_plot_energy_resolution():
    plt.close('all')
    import numpy as np
    E = np.logspace(-2, 2, 10)
    plots.plot_energy_resolution(E, E**2, color='red')


def test_plot_energy_resolution_requirements():
    plt.close('all')
    plots.plot_energy_resolution_requirements('north', color='green')


def test_saveplot_energy_resolution():
    import numpy as np
    import os
    plt.close('all')
    E = np.logspace(-2, 2, 10)
    plots.saveplot_energy_resolution(E, E**2, Outfile="eres.png")
    assert os.path.isfile("eres.png")
    os.remove("eres.png")

