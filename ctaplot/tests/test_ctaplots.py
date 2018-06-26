from hipectaold import plots
import matplotlib.pyplot as plt


'''
def test_plot_from_anadata():
    from hipectaold import data as cd
    import hipectaold.dataset as ds
    plt.close('all')
    a = cd.AnaData()
    a.append_from_parchive(ds.get('anadata.p'))
    plots.plot_from_anadata(a)
'''


def test_plot_ptabhillas_distributions():
    import hipectaold.dataset as ds
    plt.close('all')
    pt = ds.get('pcal_split.ptabhillas')
    plots.plot_ptabhillas_distributions(pt)

def test_plot_ptabrecoevent_distributions():
    import hipectaold.dataset as ds
    plt.close('all')
    pt = ds.get('pcal.ptabrecoevent')
    plots.plot_ptabrecoevent_distributions(pt)

def test_plot_ptimehillas_distributions():
    import hipectaold.dataset as ds
    plt.close('all')
    pt = ds.get('pcal.ptimehillas')
    plots.plot_ptimehillas_distributions(pt)

### TODO: use a @requires_dependency on matplotlib backend
### TODO: see http://docs.gammapy.org/en/latest/api/gammapy.utils.testing.requires_dependency.html
# def test_plot_ptimehillas_distributions_filecreation():
#     import hipectaold.dataset as ds
#     import os
#     pt = ds.get('Split.ptimehillas')
#     pdfname = 'ptimehillasdistri.pdf'
#     plots.plot_ptimehillas_distributions(pt, pdfname=pdfname)
#     assert os.path.exists(pdfname)
#     os.remove(pdfname)


def test_plot_site_ctafile():
    import hipectaold.dataset as ds
    import hipectaold.data as cd
    plt.close('all')
    pt = cd.load_ctafile(ds.get('pcal_split.ptabhillas'))
    plots.plot_site_ctafile(pt, layout=None, ax=None, color='black', alpha=0.5)

def test_plot_triggered_telescopes_ptabhillas():
    import hipectaold.dataset as ds
    import hipectaold.data as cd
    plt.close('all')
    pt = cd.load_ctafile(ds.get('pcal_split.ptabhillas'))
    eventId = pt.tabTel[0].tabEvent[0].eventId
    plots.plot_triggered_telescopes_ptabhillas(pt, eventId)


def test_plot_energy_resolution():
    plt.close('all')
    import numpy as np
    E = np.logspace(-2, 2, 10)
    plots.plot_energy_resolution(E, E**2, color='red')


def test_plot_energy_resolution_requirements():
    plt.close('all')
    plots.plot_energy_resolution_requirements('north', color='green')


def test_saveplot_energy_resolution():
    import numpy as np
    plt.close('all')
    E = np.logspace(-2, 2, 10)
    plots.saveplot_energy_resolution(E, E**2, Outfile="eres.png")

