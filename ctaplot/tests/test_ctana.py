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

