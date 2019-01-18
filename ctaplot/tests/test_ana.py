import ctaplot.ana as ana


def tests_logspace_decades_nbin():
    ca = ana.logspace_decades_nbin(0.1, 10, n=9)
    assert len(ca)==19
    assert ca[0]==0.1
    assert ca[-1]==10


def test_class_cta_requirements():
    ctaq = ana.cta_requirements()
    ctaq.get_effective_area()
    ctaq.get_angular_resolution()
    ctaq.get_energy_resolution()
    ctaq.get_sensitivity()


def test_class_cta_performances():
    ctaq = ana.cta_performances()
    ctaq.get_effective_area()
    ctaq.get_angular_resolution()
    ctaq.get_energy_resolution()
    ctaq.get_sensitivity()


def test_impact_resolution_per_energy():
    import numpy as np
    SimuX = np.random.rand(100) * 1000
    SimuY = np.random.rand(100) * 1000
    RecoX = SimuX + 1
    RecoY = SimuY + 1
    Energy = np.logspace(-2, 2, 100)
    E, R = ana.impact_resolution_per_energy(RecoX, RecoY, SimuX, SimuY, Energy)
    assert (np.isclose(R, np.sqrt(2))).all()


def test_resolution():
    import numpy as np
    x = np.random.rand(100)
    assert (ana.resolution(x, x) == np.zeros(3)).all()
    x = np.random.normal(size=100000, scale=1, loc=10)
    y = 10 * np.ones(x.shape[0])
    assert np.isclose(ana.resolution(y, x), 0.099 * np.ones(3), rtol=1e-1).all()


def test_resolution_per_energy():
    import numpy as np
    x = np.random.normal(size=100000, scale=1, loc=10)
    y = 10 * np.ones(x.shape[0])
    E = 10 ** (-3 + 6 * np.random.rand(x.shape[0]))
    e_bin, res_e = ana.resolution_per_energy(y, x, E)
    assert np.isclose(res_e, 0.099 * np.ones(res_e.shape[1]), rtol=1e-1).all()
