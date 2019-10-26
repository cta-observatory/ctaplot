from ctaplot.plots import calib
import numpy as np

def test_plot_photoelectron_true_reco():
    true_pe = 100 * np.random.rand(1000)
    reco_pe = true_pe * np.random.rand(len(true_pe))
    calib.plot_photoelectron_true_reco(true_pe, reco_pe, bins=40, stat='median', errorbar=True, percentile=68.27,
                                       hist_args={}, stat_args=dict(alpha=0), xy_args=dict(color='blue'))