import ctaplot.ana.ana as ana
import numpy as np

np.random.seed(42)


def test_logspace_decades_nbin():
    ca = ana.logspace_decades_nbin(0.1, 10, n=9)
    assert len(ca) == 19
    assert ca[0] == 0.1
    assert ca[-1] == 10


def test_class_cta_requirement():
    for site in ['north', 'south']:
        ctaq = ana.cta_requirement(site)
        ctaq.get_effective_area()
        ctaq.get_angular_resolution()
        ctaq.get_energy_resolution()
        ctaq.get_sensitivity()


def test_class_cta_performance():
    for site in ['north', 'south']:
        ctaq = ana.cta_performance(site)
        ctaq.get_effective_area()
        ctaq.get_angular_resolution()
        ctaq.get_energy_resolution()
        ctaq.get_sensitivity()


def test_impact_resolution_per_energy():
    simu_x = np.random.rand(100) * 1000
    simu_y = np.random.rand(100) * 1000
    reco_x = simu_x + 1
    reco_y = simu_y + 1
    energy = np.logspace(-2, 2, 100)
    E, R = ana.impact_resolution_per_energy(reco_x, reco_y, simu_x, simu_y, energy)
    assert (np.isclose(R, np.sqrt(2))).all()


def test_resolution():
    x = np.linspace(0, 10, 100)
    assert (ana.resolution(x, x) == np.zeros(3)).all()

    # For a normal distribution, the resolution at `percentile=68.27` is equal to 1 sigma
    loc = np.random.rand() * 100
    scale = np.random.rand() * 10
    size = 1000000
    y_true = loc * np.ones(size)
    y_reco = np.random.normal(loc=loc, scale=scale, size=size)
    relative_scaling_method = 's1'
    res = ana.resolution(y_true, y_reco, relative_scaling_method=relative_scaling_method)
    assert np.isclose(res[0],
                      scale / ana.relative_scaling(y_true, y_reco, relative_scaling_method).mean(),
                      rtol=1e-1)

    # Test bias
    bias = np.random.rand() * 100
    y_reco_bias = np.random.normal(loc=loc + bias, scale=scale, size=size)

    assert np.isclose(ana.resolution(y_true, y_reco_bias,
                                     bias_correction=True,
                                     relative_scaling_method=relative_scaling_method)[0],
                      scale / ana.relative_scaling(y_true, y_reco, relative_scaling_method).mean(),
                      rtol=1e-1)

    assert np.isclose(ana.resolution(y_true, y_reco)[0],
                      ana.resolution(y_true, y_reco, bias_correction=True)[0],
                      rtol=1e-1,
                      )

    # Test relative scaling
    for relative_scaling_method in ['s0', 's1', 's2', 's3', 's4']:
        assert np.isclose(ana.resolution(y_true, y_reco_bias,
                                         bias_correction=True,
                                         relative_scaling_method=relative_scaling_method)[0],
                          scale / ana.relative_scaling(y_true, y_reco, relative_scaling_method).mean(),
                          rtol=1e-1)


def test_resolution_per_bin():
    # For a normal distribution, the resolution at `percentile=68.27` is equal to 1 sigma
    size = 1000000
    loc = np.random.rand() * 100
    scale = np.random.rand() * 10
    x = np.linspace(0, 10, size)
    y_true = loc * np.ones(size)
    y_reco = np.random.normal(loc=loc, scale=scale, size=size)
    for scaling in ['s0', 's1', 's2', 's3', 's4']:
        bins, res = ana.resolution_per_bin(x, y_true, y_reco, bins=6, relative_scaling_method=scaling)
        np.testing.assert_allclose(res[:, 0], scale / ana.relative_scaling(y_true, y_reco, scaling).mean(), rtol=1e-1)

    bias = 50
    y_reco = np.random.normal(loc=loc + bias, scale=scale, size=size)
    bins, res = ana.resolution_per_bin(x, y_true, y_reco, bias_correction=True)
    np.testing.assert_allclose(res[:, 0], scale / ana.relative_scaling(y_true, y_reco).mean(), rtol=1e-1)


def test_resolution_per_bin_empty():
    '''
    testing for empty bins
    '''
    # For a normal distribution, the resolution at `percentile=68.27` is equal to 1 sigma
    size = 1000000
    loc = np.random.rand() * 100
    scale = np.random.rand() * 10
    x = np.linspace(0, 10, size)
    bins = np.array([1, 2, 3, 5, 11, 15])
    y_true = loc * np.ones(size)
    y_reco = np.random.normal(loc=loc, scale=scale, size=size)
    for scaling in ['s0', 's1', 's2', 's3', 's4']:
        bins, res = ana.resolution_per_bin(x, y_true, y_reco,
                                           bins=bins,
                                           relative_scaling_method=scaling,
                                           bias_correction=True)
        v = scale / ana.relative_scaling(y_true, y_reco, scaling).mean()
        expected_res = np.array([v, v, v, v, 0])
        np.testing.assert_allclose(res[:, 0], expected_res, rtol=1e-1)


def test_resolution_per_energy():
    x = np.random.normal(size=100000, scale=1, loc=10)
    y = 10 * np.ones(x.shape[0])
    E = 10 ** (-3 + 6 * np.random.rand(x.shape[0]))
    e_bin, res_e = ana.resolution_per_energy(y, x, E)
    assert np.isclose(res_e, 1. / ana.relative_scaling(y, x).mean(), rtol=1e-1).all()


def test_power_law_integrated_distribution():
    from ctaplot.ana.ana import power_law_integrated_distribution
    emin = 50.  # u.GeV
    emax = 500.e3  # u.GeV
    Nevents = 1e6
    spectral_index = -2.1
    bins = np.logspace(np.log10(emin),
                       np.log10(emax),
                       40)

    y = power_law_integrated_distribution(emin, emax, Nevents, spectral_index, bins)

    np.testing.assert_allclose(Nevents, np.sum(y), rtol=1.e-2)


def test_distance2d_resolution():
    size = 10000
    simu_x = np.ones(size)
    simu_y = np.ones(size)
    # reconstructed positions on a circle around true position
    t = np.random.rand(size) * np.pi * 2
    reco_x = 1 + 3 * np.cos(t)
    reco_y = 1 + 3 * np.sin(t)
    res, err_min, err_max = ana.distance2d_resolution(reco_x, reco_y, simu_x, simu_y,
                                                      percentile=68.27, confidence_level=0.95, bias_correction=False)

    np.testing.assert_equal(res, 3)

    # with different bias on X and Y:
    reco_x = 5 + 2 * np.cos(t)
    reco_y = 7 + 2 * np.sin(t)
    res, err_min, err_max = ana.distance2d_resolution(reco_x, reco_y, simu_x, simu_y,
                                                      percentile=68.27, confidence_level=0.95, bias_correction=True)

    assert np.isclose(res, 2, rtol=1e-1)


def test_distance2d_resolution_per_bin():
    size = 1000000
    x = np.random.rand(size)
    simu_x = np.ones(size)
    simu_y = np.ones(size)
    t = np.random.rand(size) * np.pi * 2
    reco_x = 3 * np.cos(t)
    reco_y = 3 * np.sin(t)

    bin, res = ana.distance2d_resolution_per_bin(x, reco_x, reco_y, simu_x, simu_y, bins=10, bias_correction=True)

    np.testing.assert_allclose(res, 3, rtol=1e-1)


def test_angular_resolution():
    size = 10000
    simu_alt = np.random.rand(size)
    simu_az = np.random.rand(size)
    scale = 0.01
    bias = 3

    # test alt
    reco_alt = simu_alt + np.random.normal(loc=bias, scale=scale, size=size)
    reco_az = simu_az

    assert np.isclose(ana.angular_resolution(reco_alt, reco_az, simu_alt, simu_az, bias_correction=True)[0],
                      scale,
                      rtol=1e-1)

    # test az
    simu_alt = np.zeros(size)  # angular separation evolves linearly with az if alt=0
    reco_alt = simu_alt
    reco_az = simu_az + np.random.normal(loc=-1, scale=scale, size=size)
    assert np.isclose(ana.angular_resolution(reco_alt, reco_az, simu_alt, simu_az, bias_correction=True)[0],
                      scale,
                      rtol=1e-1)


def test_angular_resolution_small_angles():
    """
    At small angles, the angular resolution should be equal to the distance2d resolution
    """
    size = 1000
    simu_az = np.ones(size)
    simu_alt = np.random.rand(size)
    reco_alt = simu_alt + np.random.normal(1, 0.05, size)
    reco_az = 2 * simu_az
    np.testing.assert_allclose(ana.angular_resolution(reco_alt, reco_az, simu_alt, simu_az, bias_correction=True),
                               ana.distance2d_resolution(reco_alt, reco_az, simu_alt, simu_az, bias_correction=True),
                               rtol=1e-1,
                               )

    simu_az = np.random.rand(size)
    simu_alt = np.zeros(size)
    reco_alt = simu_alt
    reco_az = simu_az + np.random.normal(-3, 2, size)
    np.testing.assert_allclose(ana.angular_resolution(reco_alt, reco_az, simu_alt, simu_az, bias_correction=True),
                               ana.distance2d_resolution(reco_alt, reco_az, simu_alt, simu_az, bias_correction=True),
                               rtol=1e-1,
                               )


def test_bias_empty():
    x = np.empty(0)
    assert ana.bias(x, x) == 0


def test_bias_per_bin():
    size = 100000
    simu = np.ones(size)
    reco = np.random.normal(loc=2, scale=0.5, size=size)
    x = np.linspace(0, 10, size)
    bins, bias = ana.bias_per_bin(simu, reco, x)
    np.testing.assert_allclose(bias, 1, rtol=1e-1)


def test_bias_per_energy():
    size = 100000
    simu = np.ones(size)
    reco = np.random.normal(loc=2, scale=0.5, size=size)
    energy = np.logspace(-2, 2, size)
    bins, bias = ana.bias_per_energy(simu, reco, energy)
    np.testing.assert_allclose(bias, 1, rtol=1e-1)


def test_get_magic_sensitivity():
    from astropy.table.table import QTable
    import astropy.units as u
    table = ana.get_magic_sensitivity()
    assert type(table) is QTable
    assert table['e_min'][0] == 63 * u.GeV
