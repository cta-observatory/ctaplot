from ctaplot.plots import plots
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def plot_energy_distribution():
    SimuE = np.random.rand(100)
    RecoE = np.random.rand(10)
    maskSimuDetected = np.ones(100, dtype=bool)
    maskSimuDetected[50:] = False
    plots.plot_energy_distribution(SimuE, RecoE, mask_mc_detected=maskSimuDetected)


def test_plot_energy_resolution():
    plt.close('all')
    E = np.logspace(-2, 2, 10)
    plots.plot_energy_resolution(E, E ** 2, color='red')


def test_plot_energy_resolution_cta_requirements():
    plt.close('all')
    plots.plot_energy_resolution_cta_requirement('north', color='green')


def test_plot_energy_resolution_cta_performances():
    plots.plot_energy_resolution_cta_performance('north', color='green')


def test_plot_angular_resolution_cta_performances():
    plots.plot_angular_resolution_cta_performance('north', color='green')


def test_plot_angular_resolution_cta_requirements():
    plots.plot_angular_resolution_cta_requirement('north', color='green')


def test_plot_effective_area_cta_performances():
    plots.plot_effective_area_cta_performance('north', color='green')


def test_plot_effective_area_cta_requirements():
    plots.plot_effective_area_cta_requirement('north', color='green')


def test_plot_sensitivity_cta_performances():
    plots.plot_sensitivity_cta_performance('north', color='green')


def test_plot_sensitivity_cta_requirements():
    plots.plot_sensitivity_cta_requirement('north', color='green')


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
    E = 10 ** (np.random.rand(n) * 6 - 3)
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
    SimuE = 10 ** (6 * np.random.rand(100) - 3)
    RecoE = 10 ** (6 * np.random.rand(10) - 3)
    simuArea = 1000
    plots.plot_effective_area_per_energy(SimuE, RecoE, simuArea)


def test_plot_resolution_per_energy():
    simu = np.ones(100)
    reco = np.random.normal(loc=1, scale=1, size=100)
    energy = 10 ** (-3 + 6 * np.random.rand(100))
    plots.plot_resolution_per_energy(reco, simu, energy)


def test_plot_binned_stat():
    x = np.random.rand(100)
    y = np.random.rand(100)
    for stat in ['min', 'median', 'mean']:
        for errorbar in [False, True]:
            plots.plot_binned_stat(x, y, statistic=stat, percentile=95, errorbar=errorbar, color='red', line=True,
                                   linestyle='dashed')
            plots.plot_binned_stat(x, y, statistic=stat, errorbar=errorbar, line=False, color='blue', marker='o', lw=3)


def test_plot_migration_matrix():
    x = np.random.rand(100)
    y = np.random.rand(100)
    plt.clf()
    plots.plot_migration_matrix(x, y, colorbar=True, xy_line=True,
                                hist2d_args=dict(range=[[0, 1], [0, 0.5]], density=True),
                                line_args=dict(color='red', lw=0.4)
                                )


def test_plot_impact_parameter_error_site_center():
    simu_x = np.random.rand(100)
    simu_y = np.random.rand(100)
    reco_x = 2 * simu_x
    reco_y = 2 * simu_y
    fig, ax = plt.subplots()
    plots.plot_impact_parameter_error_site_center(reco_x, reco_y, simu_x, simu_y, ax=ax, bins=30)


def test_plot_effective_area_per_energy_power_law():
    emin = 1e-3
    emax = 1e3
    total_number_events = 100000
    spectral_index = 2.4
    reco_energy = 10 ** (6 * np.random.rand(1000) - 3)
    simu_area = 1e7

    plots.plot_effective_area_per_energy_power_law(emin, emax, total_number_events, spectral_index,
                                                   reco_energy, simu_area, color='black')


def test_plot_resolution():
    x = np.linspace(0, 10, 1000)
    y_true = 0.5 + np.random.rand(1000)
    y_reco = np.random.normal(loc=1, size=1000)
    from ctaplot.ana.ana import resolution_per_bin
    bins, res = resolution_per_bin(x, y_true, y_reco)
    plots.plot_resolution(bins, res, color='black')


def test_plot_angular_resolution_per_off_pointing_angle():
    n = 1000
    simu_alt = 0.5 + np.random.rand(n)
    simu_az = 0.5 + np.random.rand(n)
    reco_alt = 0.5 + np.random.rand(n)
    reco_az = 0.5 + np.random.rand(n)
    alt_p = np.ones(n)
    az_p = np.ones(n)

    plots.plot_angular_resolution_per_off_pointing_angle(simu_alt, simu_az, reco_alt, reco_az, alt_p, az_p,
                                                         bins=4, color='red', alpha=0.5
                                                         )


def test_plot_multiplicity_hist():
    multiplicity = np.random.randint(24, size=100)
    plots.plot_multiplicity_hist(multiplicity, ax=None, outfile=None, quartils=True, alpha=0.5)


def test_plot_multiplicity_per_telescope_type():
    multiplicity = np.random.randint(10, size=50)
    telescope_type = np.random.choice(['a', 'b', 'c'], size=len(multiplicity))
    plots.plot_multiplicity_per_telescope_type(multiplicity, telescope_type, quartils=True, alpha=0.6)


def test_plot_angular_res_per_energy():
    reco_alt = np.random.rand(10)
    reco_az = np.random.rand(10)
    mc_alt = np.ones(10)
    mc_az = np.zeros(10)
    energy = 10 ** np.random.rand(10)
    plots.plot_angular_resolution_per_energy(reco_alt, reco_az, mc_alt, mc_az, energy, bias_correction=True, alpha=0.4)


def test_plot_resolution_difference():
    from ctaplot.ana.ana import resolution_per_bin, irf_cta
    size = 1000
    simu = np.logspace(-2, 2, size)
    reco = 2 * simu
    reco2 = 3 * simu
    irf = irf_cta()
    bin = irf.E_bin
    bins, res1 = resolution_per_bin(simu, simu, reco, bins=bin, relative_scaling_method='s1')
    bins, res2 = resolution_per_bin(simu, simu, reco2, bins=bin, relative_scaling_method='s1')
    plots.plot_resolution_difference(bins, res1, res2, ax=None, color='red', alpha=0.8, label='nice diff')


def test_plot_roc_curve():
    size = 1000
    simu_type = np.random.choice(['g', 'p'], size=size)
    reco_proba = np.random.rand(size)
    plots.plot_roc_curve(simu_type, reco_proba,
                         pos_label='p',
                         ax=None, c='green')


def test_plot_roc_curve_multiclass():
    size = 1000
    simu_classes = np.random.choice(['gamma', 'proton', 'electron', 'positron'], size=size)

    reco_proba = {
        'gamma': np.random.rand(size),
        'proton': np.random.rand(size),
        'electron': np.random.rand(size),
        'positron': np.array(simu_classes == 'positron', dtype=int),
    }

    plots.plot_roc_curve_multiclass(simu_classes, reco_proba, alpha=0.6, lw=3)

    plots.plot_roc_curve_multiclass(simu_classes, reco_proba, pos_label='gamma')


def test_plot_roc_curve_gammaness():
    size = 1000
    simu_classes = np.random.choice(['gamma', 'proton', 'electron', 'positron'], size=size)
    gamma_reco_proba = np.random.rand(size)

    plots.plot_roc_curve_gammaness(simu_classes, gamma_reco_proba, gamma_label='gamma', alpha=0.6, lw=3)


def test_plot_roc_curve_gammaness_per_energy():
    size = 1000
    simu_classes = np.random.choice(['gamma', 'proton', 'electron', 'positron'], size=size)
    gamma_reco_proba = np.random.rand(size)
    simu_energy = 10 ** (np.random.rand(size) * 4 - 2)

    plots.plot_roc_curve_gammaness_per_energy(simu_classes, gamma_reco_proba, simu_energy,
                                              gamma_label='gamma',
                                              energy_bins=np.array([1e-2, 1e-1, 1, 10, 100]),
                                              alpha=0.6, lw=3,
                                              )

    size = 1000
    simu_classes = np.random.choice(['gamma', 'proton'], size=size)
    gamma_reco_proba = np.random.rand(size)
    simu_energy = 10 ** (np.random.rand(size) * 4 - 2)

    plots.plot_roc_curve_gammaness_per_energy(simu_classes, gamma_reco_proba, simu_energy,
                                              gamma_label='gamma',
                                              energy_bins=np.array([1e-2, 1e-1, 1, 10, 100]),
                                              alpha=0.6, lw=3,
                                              )


def test_plot_any_resource():
    from ctaplot.io.dataset import resources_list
    for filename in resources_list:
        plots.plot_any_resource(filename)


def test_plot_gammaness_distribution():
    nb_events = 1000
    mc_type = np.random.choice([0, 1, 2, 3], size=nb_events)
    gammaness = np.random.rand(nb_events)
    plots.plot_gammaness_distribution(mc_type, gammaness)


def test_plot_sensitivity_magic_performance():
    ax = plots.plot_sensitivity_magic_performance(key='lima_5off')
    plots.plot_sensitivity_magic_performance(key='lima_3off', ax=ax, color='black', ls='--')
