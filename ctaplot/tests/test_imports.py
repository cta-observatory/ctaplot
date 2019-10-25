def test_import_plots():
    try:
        from ctaplot.plots import plots
    except:
        raise()

def test_import_ana():
    try:
        from ctaplot.ana import ana
    except:
        raise ()

def test_import_dataset():
    try:
        from ctaplot.io import dataset
    except:
        raise ()

def test_import_gammaboard():
    try:
        from ctaplot import gammaboard
    except:
        raise()