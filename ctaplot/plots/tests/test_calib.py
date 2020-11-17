from ctaplot.plots import calib
import numpy as np

np.random.seed(42)

def test_plot_photoelectron_true_reco():
    true_pe = 100 * np.random.rand(1000)
    reco_pe = true_pe * np.random.rand(len(true_pe))
    calib.plot_photoelectron_true_reco(true_pe, reco_pe, bins=40, stat='median', errorbar=True, percentile=68.27,
                                       hist_args=dict(cmax=50), stat_args=dict(alpha=0), xy_args=dict(color='blue'))

    calib.plot_photoelectron_true_reco(true_pe, reco_pe, bins=np.logspace(-2, 4, 50))


def test_plot_pixels_pe_spectrum():
    true_pe = 100 * np.random.rand(1000)
    reco_pe = -50 + 100 * np.random.rand(len(true_pe))

    calib.plot_pixels_pe_spectrum(true_pe, reco_pe, bins=100)
    calib.plot_pixels_pe_spectrum(true_pe, reco_pe, bins=np.logspace(-2, 4, 50))


def test_plot_charge_resolution():
    true_pe = 100 * np.random.rand(1000)
    reco_pe = true_pe * np.random.rand(len(true_pe))

    for bc in [False, True]:
        calib.plot_charge_resolution(true_pe, reco_pe, xlim_bias=(20, 80), bias_correction=bc,
                                     bins=50,
                                     ax=None,
                                     hist_args=dict(alpha=0.5),
                                     bin_stat_args=dict(color='green')
                                     )