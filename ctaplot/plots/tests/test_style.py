from ctaplot.plots import style
import matplotlib.pyplot as plt

def test_set_style():
    style.set_style('slides')
    assert plt.rcParams['figure.dpi'] == 200.
    style.set_style('paper')
    assert plt.rcParams['figure.dpi'] == 200.
    style.set_style('notebook')
    assert plt.rcParams['figure.dpi'] == 200.

def test_context():
    with style.context('slides'):
        plt.plot([1, 2, 6])
