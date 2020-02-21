from ctaplot.plots import grid
import pandas as pd
import numpy as np

np.random.seed(42)

def test_plot_binned_stat_grid():
    df = pd.DataFrame(data=np.random.rand(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
    grid.plot_binned_stat_grid(df, 'a', statistic='mean', errorbar=True)