import ctaplot
from ctaplot.plots import plots
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

np.random.seed(42)


def test_plot_energy_distribution():
    true_e = np.random.rand(100) * u.TeV
    reco_e = np.random.rand(10) * u.TeV
    mask_simu_detected = np.ones(100, dtype=bool)
    mask_simu_detected[50:] = False
    bins = np.logspace(-2, 2, 10) * u.TeV
    plots.plot_energy_distribution(true_e, reco_e, bins=bins, mask_mc_detected=mask_simu_detected)


def test_plot_multiplicity_per_energy():
    size = 100
    energy = 10**np.random.rand(size) * u.TeV
    mult = np.floor(np.log10(energy.value) * 10)
    plots.plot_multiplicity_per_energy(energy, mult)
    plots.plot_multiplicity_per_energy(energy, mult, bins=np.logspace(-1, 1, 10)*u.TeV)


def test_plot_energy_resolution():
    plt.close('all')
    E = np.logspace(-2, 2, 10) * u.TeV
    plots.plot_energy_resolution(E, E.value ** 2 * u.TeV, color='red')


def test_plot_energy_resolution_cta_requirement():
    plt.close('all')
    plots.plot_energy_resolution_cta_requirement('north', color='green')
    plots.plot_energy_resolution(np.random.rand(2)*u.erg, np.random.rand(2)*u.J)


def test_plot_energy_resolution_cta_performance():
    plt.close('all')
    plots.plot_energy_resolution_cta_performance('north', color='green')
    plots.plot_energy_resolution(np.random.rand(2) * u.erg, np.random.rand(2) * u.J)


def test_plot_angular_resolution_cta_performance():
    plt.close('all')
    plots.plot_angular_resolution_cta_performance('north', color='green')
    a = np.random.rand(3) * u.rad
    e = np.random.rand(3) * u.erg
    plots.plot_angular_resolution_per_energy(a, a, a, a, e)


def test_plot_angular_resolution_cta_requirement():
    plt.close('all')
    plots.plot_angular_resolution_cta_requirement('north', color='green')
    a = np.random.rand(3) * u.rad
    e = np.random.rand(3) * u.erg
    plots.plot_angular_resolution_per_energy(a, a, a, a, e)


def test_plot_effective_area_cta_performance():
    plt.close('all')
    plots.plot_effective_area_cta_performance('north', color='green')
    e = np.random.rand(3)*u.erg
    ctaplot.plot_effective_area_per_energy(e, e, 10*u.m**2)


def test_plot_effective_area_cta_requirement():
    plt.close('all')
    plots.plot_effective_area_cta_requirement('north', color='green')
    e = np.random.rand(3) * u.erg
    ctaplot.plot_effective_area_per_energy(e, e, 10 * u.m ** 2)


def test_plot_sensitivity_cta_performance():
    plt.close('all')
    plots.plot_sensitivity_cta_performance('north', color='green')


def test_plot_sensitivity_cta_requirement():
    plt.close('all')
    plots.plot_sensitivity_cta_requirement('north', color='green')


def test_plot_theta2():
    plt.close('all')
    n = 10
    reco_alt = (1 + np.random.rand(n)) * u.rad
    reco_az = (1.5 + np.random.rand(n)) * u.rad
    true_alt = np.ones(n) * u.rad
    true_az = (1.5 * np.ones(n)) * u.rad
    plots.plot_theta2(true_alt, reco_alt, true_az, reco_az)


def test_plot_impact_point_heatmap():
    plt.close('all')
    n = 10
    reco_x = np.random.rand(n) * u.m
    reco_y = np.random.rand(n) * u.m
    plots.plot_impact_point_heatmap(reco_x, reco_y)


def test_plot_multiplicity_hist():
    plt.close('all')
    multiplicity = np.random.randint(low=2, high=30, size=100)
    plots.plot_multiplicity_hist(multiplicity)


def test_plot_effective_area_per_energy():
    plt.close('all')
    true_e = 10 ** (6 * np.random.rand(100) - 3) * u.TeV
    reco_e = 10 ** (6 * np.random.rand(10) - 3) * u.TeV
    simu_area = (1000 * u.m)**2
    plots.plot_effective_area_per_energy(true_e, reco_e, simu_area)


def test_plot_resolution_per_energy():
    plt.close('all')
    true = np.ones(100)
    reco = np.random.normal(loc=1, scale=1, size=100)
    energy = 10 ** (-3 + 6 * np.random.rand(100)) * u.TeV
    plots.plot_resolution_per_energy(true, reco, energy)


def test_plot_binned_stat():
    plt.close('all')
    x = np.random.rand(100)
    y = np.random.rand(100)
    for stat in ['min', 'median', 'mean']:
        for errorbar in [False, True]:
            plots.plot_binned_stat(x, y, statistic=stat, percentile=95, errorbar=errorbar, color='red', line=True,
                                   linestyle='dashed')
            plots.plot_binned_stat(x, y, statistic=stat, errorbar=errorbar, line=False, color='blue', marker='o', lw=3)


def test_plot_migration_matrix():
    plt.close('all')
    x = np.random.rand(100)
    y = np.random.rand(100)
    plt.clf()
    plots.plot_migration_matrix(x, y, colorbar=True, xy_line=True,
                                hist2d_args=dict(range=[[0, 1], [0, 0.5]], density=True),
                                line_args=dict(color='red', lw=0.4)
                                )


def test_plot_impact_parameter_error_site_center():
    plt.close('all')
    simu_x = np.random.rand(100) * u.m
    simu_y = np.random.rand(100) * u.m
    reco_x = 2 * simu_x
    reco_y = 2 * simu_y
    fig, ax = plt.subplots()
    plots.plot_impact_parameter_error_site_center(simu_x, reco_x, simu_y, reco_y, ax=ax, bins=30)


def test_plot_effective_area_per_energy_power_law():
    plt.close('all')
    emin = 1e-3 * u.TeV
    emax = 1e3 * u.TeV
    total_number_events = 100000
    spectral_index = 2.4
    reco_energy = 10 ** (6 * np.random.rand(1000) - 3) * u.TeV
    simu_area = 1e7 * u.m**2

    plots.plot_effective_area_per_energy_power_law(emin, emax, total_number_events, spectral_index,
                                                   reco_energy, simu_area, color='black')


def test_plot_resolution():
    plt.close('all')
    x = np.linspace(0, 10, 1000)
    y_true = 0.5 + np.random.rand(1000)
    y_reco = np.random.normal(loc=1, size=1000)
    from ctaplot.ana.ana import resolution_per_bin
    bins, res = resolution_per_bin(x, y_true, y_reco)
    plots.plot_resolution(bins, res, color='black')


def test_plot_angular_resolution_per_off_pointing_angle():
    plt.close('all')
    n = 1000
    simu_alt = (0.5 + np.random.rand(n)) * u.rad
    simu_az = (0.5 + np.random.rand(n)) * u.rad
    reco_alt = (0.5 + np.random.rand(n)) * u.rad
    reco_az = (0.5 + np.random.rand(n)) * u.rad
    alt_p = np.ones(n) * u.rad
    az_p = np.ones(n) * u.rad

    plots.plot_angular_resolution_per_off_pointing_angle(simu_alt, simu_az, reco_alt, reco_az, alt_p, az_p,
                                                         bins=4, color='red', alpha=0.5
                                                         )


def test_plot_multiplicity_hist():
    plt.close('all')
    multiplicity = np.random.randint(24, size=100)
    plots.plot_multiplicity_hist(multiplicity, ax=None, outfile=None, quartils=True, alpha=0.5)


def test_plot_angular_res_per_energy():
    plt.close('all')
    reco_alt = np.random.rand(10) * u.rad
    reco_az = np.random.rand(10) * u.rad
    mc_alt = np.ones(10) * u.rad
    mc_az = np.zeros(10) * u.rad
    energy = 10 ** np.random.rand(10) * u.TeV
    plots.plot_angular_resolution_per_energy(reco_alt, reco_az, mc_alt, mc_az, energy, bias_correction=True, alpha=0.4)


def test_plot_resolution_difference():
    plt.close('all')
    from ctaplot.ana.ana import resolution_per_bin, irf_cta
    size = 1000
    simu = np.logspace(-2, 2, size) * u.TeV
    reco = 2 * simu
    reco2 = 3 * simu
    irf = irf_cta()
    bin = irf.energy_bins
    bins, res1 = resolution_per_bin(simu, simu, reco, bins=bin, relative_scaling_method='s1')
    bins, res2 = resolution_per_bin(simu, simu, reco2, bins=bin, relative_scaling_method='s1')
    plots.plot_resolution_difference(bins, res1, res2, ax=None, color='red', alpha=0.8, label='nice diff')


def test_plot_roc_curve():
    plt.close('all')
    size = 1000
    simu_type = np.random.choice(['g', 'p'], size=size)
    reco_proba = np.random.rand(size)
    plots.plot_roc_curve(simu_type, reco_proba,
                         pos_label='p',
                         ax=None, c='green')


def test_plot_roc_curve_multiclass():
    plt.close('all')
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
    plt.close('all')
    size = 1000
    simu_classes = np.random.choice(['gamma', 'proton', 'electron', 'positron'], size=size)
    gamma_reco_proba = np.random.rand(size)

    plots.plot_roc_curve_gammaness(simu_classes, gamma_reco_proba, gamma_label='gamma', alpha=0.6, lw=3)


def test_plot_roc_curve_gammaness_per_energy():
    plt.close('all')
    size = 1000
    simu_classes = np.random.choice(['gamma', 'proton', 'electron', 'positron'], size=size)
    gamma_reco_proba = np.random.rand(size)
    simu_energy = 10 ** (np.random.rand(size) * 4 - 2) * u.TeV

    plots.plot_roc_curve_gammaness_per_energy(simu_classes, gamma_reco_proba, simu_energy,
                                              gamma_label='gamma',
                                              energy_bins=u.Quantity([1e-2, 1e-1, 1, 10, 100], u.TeV),
                                              alpha=0.6, lw=3,
                                              )

    size = 1000
    simu_classes = np.random.choice(['gamma', 'proton'], size=size)
    gamma_reco_proba = np.random.rand(size)
    simu_energy = 10 ** (np.random.rand(size) * 4 - 2) * u.TeV

    plots.plot_roc_curve_gammaness_per_energy(simu_classes, gamma_reco_proba, simu_energy,
                                              gamma_label='gamma',
                                              energy_bins=u.Quantity([1e-2, 1e-1, 1, 10, 100], u.TeV),
                                              alpha=0.6, lw=3,
                                              )


def test_plot_any_resource():
    from ctaplot.io.dataset import resources_list
    for filename in resources_list:
        plt.close('all')
        plots.plot_any_resource(filename)


def test_plot_gammaness_distribution():
    plt.close('all')
    nb_events = 1000
    mc_type = np.random.choice([0, 1, 2, 3], size=nb_events)
    gammaness = np.random.rand(nb_events)
    plots.plot_gammaness_distribution(mc_type, gammaness)


def test_plot_sensitivity_magic_performance():
    plt.close('all')
    ax = plots.plot_sensitivity_magic_performance(key='lima_5off')
    plots.plot_sensitivity_magic_performance(key='lima_3off', ax=ax, color='black', ls='--')


def test_plot_rate():
    plt.close('all')
    e_bins = np.logspace(-2, 2) * u.TeV
    e_min = e_bins[:-1]
    e_max = e_bins[1:]
    plots.plot_rate(e_min, e_max, (1e-12 / e_min.value**2)/u.s, rate_err=None, color='green', ls='--')
    plots.plot_rate(e_min, e_max, (1e-12 / e_min.value**2)/u.s, rate_err=1e-15 * e_min.value/u.s)


def test_plot_background_rate_magic():
    plt.close('all')
    plots.plot_background_rate_magic(color='grey')


def test_plot_gamma_rate_magic():
    plt.close('all')
    plots.plot_background_rate_magic(color='red')


def test_plot_dispersion():
    plt.close('all')
    x = 10**np.random.rand(1000)
    y = x**2
    plots.plot_dispersion(x, y, x_log=False)
    plots.plot_dispersion(x, y, x_log=True, bins=20)
    plots.plot_dispersion(x, y, x_log=False, bins=(np.logspace(0, 1, 10), np.linspace(-1, 1, 10)))


def test_plot_gammaness_threshold_efficiency():
    plots.plot_gammaness_threshold_efficiency(np.random.rand(1000), np.random.rand(), color='black', alpha=0.4)


def test_plot_precision_recall():
    size = 1000
    plots.plot_precision_recall(np.random.choice([0, 101], size=size),
                                np.random.rand(size),
                                threshold=np.random.rand(),
                                color='green')


def test_plot_roc_auc_per_energy():
    energy_bins = np.array([0.05, 0.5, 5, 50]) * u.TeV
    roc_auc_scores = np.array([0.6, 0.7, 0.8])
    plots.plot_roc_auc_per_energy(energy_bins, roc_auc_scores, color='black', label='label', alpha=0.3, ls='--')