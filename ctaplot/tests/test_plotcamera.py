from hipectaold.plots.plotcamera import *


def test_hillas_on_calib_events():
    from hipectaold import data as cd
    import hipectaold.dataset as ds
    pcal = ds.get('pcal.pcalibrun')
    ptabhillas = ds.get('pcal.ptabhillas')

    hillas_on_calib_events(cd.load_ctafile(pcal), cd.load_ctafile(ptabhillas), nevent=2)


def test_plot_camera_image():
    import hipectaold.dataset as ds
    import hipectaold.data as cd
    pcal = cd.load_ctafile(ds.get('pcal.pcalibrun'))
    plot_camera_image(pcal, 0, 0,)


def test_plot_timehillas_direction_mono():
    import hipectaold.dataset as ds
    import hipectaold.data as cd

    ptime = cd.load_ctafile(ds.get('pcal.ptimehillas'))
    plot_timehillas_direction_mono(ptime, 0, 0)


def test_plot_timehillas_direction_stereo():
    import hipectaold.dataset as ds
    import hipectaold.data as cd

    ptime = cd.load_ctafile(ds.get('pcal.ptimehillas'))
    plot_timehillas_direction_stereo(ptime, 0, recoquality=[0])

def test_plot_timehillas_source_camera():
    import hipectaold.dataset as ds
    import hipectaold.data as cd
    ptime = cd.load_ctafile(ds.get('pcal.ptimehillas'))
    psimu = cd.load_ctafile(ds.get('pcal.psimu'))

    eventId = ptime.tabEvent[0].eventId

    plot_timehillas_source_camera(ptime, psimu, eventId, recoquality=[0])


def test_plot_tabhillas_signal_camera_grid():
    import hipectaold.dataset as ds
    import hipectaold.data as cd

    ptab = cd.load_ctafile(ds.get('pcal.ptabhillas'))
    pcal = cd.load_ctafile(ds.get('pcal.pcalibrun'))

    eventId = ptab.tabTel[3].tabEvent[1].eventId

    plot_tabhillas_signal_camera_grid(ptab, pcal, eventId, recoquality=[0])


def test_plot_stacked_images_event():
    import hipectaold.dataset as ds
    import hipectaold.data as cd

    pcal = cd.load_ctafile(ds.get('pcal.pcalibrun'))

    eventId = pcal.tabTelescope[0].tabTelEvent[0].eventId
    layout = np.array([0, 1])
    plot_stacked_images_event(pcal, eventId, layout=layout)


def test_plot_stacked_event():
    import hipectaold.dataset as ds
    import hipectaold.data as cd

    ptab = cd.load_ctafile(ds.get('pcal.ptabhillas'))
    pcal = cd.load_ctafile(ds.get('pcal.pcalibrun'))

    eventId = ptab.tabTel[0].tabEvent[0].eventId

    plot_stacked_event(ptab, pcal, eventId)


def test_plot_simu_source():
    import hipectaold.dataset as ds
    import hipectaold.data as cd

    psimu = cd.load_ctafile(ds.get('pcal.psimu'))
    eventId = psimu.tabSimuEvent[0].id

    plot_simu_source(psimu, eventId)
