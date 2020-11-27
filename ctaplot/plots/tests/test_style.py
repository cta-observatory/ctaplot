from ctaplot.plots import style
import matplotlib.pyplot as plt

def test_set_style():
    style.set_style('slides')
    style.set_style('paper')
    style.set_style('notebook')

def test_context():
    with style.context('slides'):
        plt.plot([1, 2, 6])
