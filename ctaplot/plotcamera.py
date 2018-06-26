import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
import numpy as np




camera_types = ['LST', 'Nectar', 'Flash', 'SCT', 'ASTRI', 'DC', 'GCT']


def calib_event(tel, eventidx, ax=None, Outfile=None):
    """
    Given a telescope and an event index, display this event
    If a string is specified for Outfile, the event is dumped as a png image
    Parameters
    ----------
    tel: core.PCalibTel
    eventidx: int
    Outfile: string
    """
    ax = plt.gca() if ax is None else ax

    assert len(tel.tabTelEvent) > eventidx, "event index too big"

    pix_pos = tel.tabPos.tabPixelPosXY
    ev = tel.tabTelEvent[eventidx]

    plt.figure(figsize=(12, 12))
    plt.scatter(pix_pos[:, 0], pix_pos[:, 1], c=ev.tabPixel, s=130)
    plt.title("Telescope {0} ({1}): event {2}".format(tel.telescopeId, camera_types[tel.telescopeType], ev.eventId))
    plt.axis('equal')
    if type(Outfile) == str:
        plt.savefig(Outfile + ".png", format='png', dpi=200)
    else:
        plt.show()


def calib_events_allteltypes(pcal, nb_events, dump=False):
    if dump:
        if not os.path.isdir("images"): os.mkdir("images")
    teltypes = np.array([tel.telescopeType for tel in pcal.tabTelescope])
    tidx = [np.where(teltypes==t)[0][0] for t in set(teltypes)]
    for idx in tidx:
        tel = pcal.tabTelescope[idx]
        for evidx in range(min(nb_events, len(tel.tabTelEvent))):
            if dump:
                Outfile = "images/tel{0}_ev{1}". format(tel.telescopeId, evidx)
            else:
                Outfile = None
            plot_calib_event(tel, evidx, Outfile=Outfile)


def hillas_on_calib_events(pcal, ptab, telidx=0, nevent=5, dump=False):
    """
    Plot camera images of the events from pcal with Hillas ellipses from ptab
    The ptab object should have been computed from the pcal one.

    Parameters
    ----------
    pcalibrun: hipectaold.data.PCalibRun object
    ptablehillas: hipectaold.data.PTableHillas object
    telidx: index of the telescope you want to plot the events of
    nevent: number of events to plot
    dump: if True, save the images in the folder 'images/'. Else display the images.

    """
    from matplotlib.patches import Ellipse

    figures = []
    tel = pcal.tabTelescope[telidx]
    focal = pcal.header.tabFocalTel[telidx]

    for iev, (ev, evcal) in enumerate(zip(ptab.tabTel[telidx].tabEvent[:nevent],
                                          pcal.tabTelescope[telidx].tabTelEvent[:nevent])):

        if ev.recoQuality == 0:
            assert ev.eventId == evcal.eventId
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

            # plot camera image
            pix_pos = tel.tabPos.tabPixelPosXY

            plt.scatter(pix_pos[:, 0], pix_pos[:, 1], c=evcal.tabPixel, s=20)  # s=120/(12/fs)**2)
            plt.colorbar()
            plt.axis('equal')

            # barycenter
            plt.scatter(focal * ev.gx, focal * ev.gy, marker='+', s=140, color='red')

            # Hillas ellipse
            x = np.linspace(-0.5, 0.5, 10)
            e = Ellipse([focal * ev.gx, focal * ev.gy], ev.width, ev.length, angle=np.degrees(ev.direction - np.pi / 2))
            print(ev.width, ev.length)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.6)
            e.set_facecolor('white')
            plt.plot(focal * ev.gx + x * np.cos(ev.direction), focal * ev.gy + x * np.sin(ev.direction), color='white')
            plt.title("Telescope {0} ({1}): event {2}".format(pcal.tabTelescope[telidx].telescopeId,
                                                              camera_types[pcal.tabTelescope[telidx].telescopeType],
                                                              ev.eventId))

            if dump:
                Outfile = "images/hillas_tel{0}_ev{1}".format(pcal.tabTelescope[telidx].telescopeId, ev.eventId)
                plt.savefig(Outfile, fmt='png', dpi=150)

            figures.append(fig)

    return figures


def plot_camera_image(pcal, telescopeIndex, eventIndex, ax=None, **kwargs):
    """
    Plot the camera image of an event given the telescope index and event index

    Parameters
    ----------
    pcal: `hipectaold.data.PCalibRun`
    telescopeIndex: int
    eventIndex: int
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `matplotlib.pyplot.scatter`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """
    ax = plt.gca() if ax is None else ax

    tel = pcal.tabTelescope[telescopeIndex]
    event = tel.tabTelEvent[eventIndex]

    cf = ax.scatter(tel.tabPos.tabPixelPosXY[:, 0], tel.tabPos.tabPixelPosXY[:, 1], c=event.tabPixel, **kwargs)
    ax.axis('equal')
    cb = plt.colorbar(cf, ax=ax)
    cb.ax.yaxis.set_tick_params(labelsize=12)

    return ax


def plot_timehillas_direction_mono(ptimehillas, telIndex, eventIndex, ax=None, **kwargs):
    """
    Plot the Hillas ellipse and direction for a single camera.
    This method is intended to be applied in other functions.

    Parameters
    ----------
    ptimehillas: `hipectaold.data.PTimeHillas`
    telIndex: int
    eventIndex: int
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """
    from matplotlib.patches import Ellipse

    ax = plt.gca() if ax is None else ax

    event = ptimehillas.tabEvent[eventIndex]
    tel = event.tabTel[telIndex]
    focal = ptimehillas.header.tabFocalTel[telIndex]
    l = np.linspace(-0.5, 0.5)

    if not 'color' in kwargs.keys():
        kwargs['color'] = 'white'
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.6

    ax.plot(focal * tel.gx + l * np.cos(tel.direction), focal * tel.gy + l * np.sin(tel.direction), **kwargs)
    ax.scatter(focal * tel.gx, focal * tel.gy, marker='+', color='black')

    e = Ellipse([focal * tel.gx, focal * tel.gy], tel.width, tel.length,
                angle=np.degrees(tel.direction - np.pi / 2.))
    ax.add_artist(e);
    e.set_clip_box(ax.bbox)
    e.set_alpha(kwargs['alpha'])
    e.set_facecolor(kwargs['color'])
    ax.axis('equal')

    return ax


def plot_timehillas_direction_stereo(ptimehillas, eventIndex, recoquality=[0], ax=None, **kwargs):
    """
    Plot the Hillas ellipses and directions for a complete event

    Parameters
    ----------
    ptimehillas: `hipectaold.data.PTimeHillas`
    eventIndex: int
    recoquality: list to filter the recoquality of the events to plot
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """

    ax = plt.gca() if ax is None else ax

    event = ptimehillas.tabEvent[eventIndex]
    l = np.linspace(-0.5, 0.5)

    for tel_idx, tel in enumerate(event.tabTel):
        if tel.recoQuality in recoquality:
            ax = plot_timehillas_direction_mono(ptimehillas, tel_idx, eventIndex, ax=ax, **kwargs)
    ax.axis('equal')

    return ax


def plot_timehillas_source_camera(ptimehillas, psimu, eventId, recoquality=[0], ax=None, **kwargs):
    """
    Plot the Hillas ellipse and directions for a complete event
    as well as the simulated source position

    Parameters
    ----------
    ptimehillas: `hipectaold.data.PTimeHillas`
    psimu: `hipectaold.data.PSimu`
    eventId: int
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `matplotlib.pyplot.scatter`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """

    ax = plt.gca() if ax is None else ax

    w = np.where(np.array([ev.eventId for ev in ptimehillas.tabEvent]) == eventId)
    assert len(w[0]) > 0, "There is no event with this eventId"

    event_index = w[0][0]
    event = psimu.tabSimuEvent[np.where(np.array([ev.id for ev in psimu.tabSimuEvent]) == eventId)[0][0]]
    shower = psimu.tabSimuShower[np.where(np.array([sh.id for sh in psimu.tabSimuShower]) == event.showerNum)[0][0]]

    ax = plot_timehillas_direction_stereo(ptimehillas, event_index, recoquality=recoquality, ax=ax, **kwargs)
    ax.scatter(shower.altitude, shower.azimuth, marker='x', label='Simulated source')
    ax.legend()

    return ax


def plot_tabhillas_direction_mono(ptabhillas, telIndex, eventIndex, ax=None, **kwargs):
    """
    Plot the Hillas ellipse and direction for a single camera.
    This method is intended to be applied in other functions.

    Parameters
    ----------
    ptabhillas: `hipectaold.data.PTabHillas`
    telIndex: int
    eventIndex: int
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """

    from matplotlib.patches import Ellipse
    ax = plt.gca() if ax is None else ax

    event = ptabhillas.tabTel[telIndex].tabEvent[eventIndex]
    focal = ptabhillas.header.tabFocalTel[telIndex]

    if not 'color' in kwargs.keys():
        kwargs['color'] = 'white'
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.6

    l = np.linspace(-0.5, 0.5)
    ax.plot(focal * event.gx + l * np.cos(event.direction), focal * event.gy + l * np.sin(event.direction), **kwargs)

    ax.scatter(focal * event.gx, focal * event.gy, marker='+', color='black')
    ax.axis('equal')

    e = Ellipse([focal * event.gx, focal * event.gy], event.width, event.length,
                angle=np.degrees(event.direction - np.pi / 2.))
    ax.add_artist(e);
    e.set_clip_box(ax.bbox)
    e.set_alpha(kwargs['alpha'])
    e.set_facecolor(kwargs['color'])

    return ax


def plot_tabhillas_direction_stereo(ptabhillas, eventId, recoquality=[0], ax=None, **kwargs):
    """
    Plot the Hillas ellipses and directions for a complete event

    Parameters
    ----------
    ptabhillas: `hipectaold.data.PTabHillas`
    eventIndex: int
    recoquality: list to filter the recoquality of the events to plot
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """
    ax = plt.gca() if ax is None else ax

    l = np.linspace(-0.5, 0.5)

    for tel_idx, tel in enumerate(ptabhillas.tabTel):
        for ev_idx, ev in enumerate(tel.tabEvent):
            if ev.eventId == eventId and ev.recoQuality in recoquality:
                ax = plot_tabhillas_direction_mono(ptabhillas, tel_idx, ev_idx, ax=ax, **kwargs)

    ax.axis('equal')

    return ax


def plot_tabhillas_source_camera(ptabhillas, psimu, eventId, recoquality=[0], ax=None, **kwargs):
    """
    Plot the Hillas ellipse and directions for a complete event
    as well as the simulated source position

    Parameters
    ----------
    ptimehillas: `hipectaold.data.PTabHillas`
    psimu: `hipectaold.data.PSimu`
    eventId: int
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `plot_tabhillas_direction_stereo`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """
    ax = plt.gca() if ax is None else ax

    event = psimu.tabSimuEvent[np.where(np.array([ev.id for ev in psimu.tabSimuEvent]) == eventId)[0][0]]
    shower = psimu.tabSimuShower[np.where(np.array([sh.id for sh in psimu.tabSimuShower]) == event.showerNum)[0][0]]

    ax = plot_tabhillas_direction_stereo(ptabhillas, eventId, recoquality=recoquality, ax=ax, **kwargs)
    ax.scatter(shower.altitude, shower.azimuth, marker='x', label='Simulated source')
    ax.legend()

    return ax


def plot_tabhillas_signal_camera_mono(ptabhillas, pcal, telescopeId, eventId, recoquality=[0],
        ax=None, imgargs={}, **kwargs):
    """

    Parameters
    ----------
    ptabhillas: `hipectaold.data.PTabHillas`
    pcal: `hipectaold.data.PCalibRun`
    telescopeId: int
    eventId: int
    recoquality: list of int
    ax: `~matplotlib.axes.Axes`
    imgargs: Dictionnary of keywords arguments passed to `plot_camera_image`
    **kwargs: Extra keyword arguments are passed to `plot_tabhillas_camera_mono`

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
    """
    import hipectaold.data as cd

    ax = plt.gca() if ax is None else ax

    tidx, eidx = cd.find_event_dl0(pcal, eventId, telescopeId=telescopeId)
    tabtel_idx, tabev_idx = cd.find_event_tabhillas(ptabhillas, eventId, telescopeId=telescopeId)

    assert len(tidx) > 0
    assert len(eidx) > 0
    assert len(tabtel_idx) > 0
    assert len(tabev_idx) > 0

    ax = plot_camera_image(pcal, tidx[0], eidx[0], ax=ax, **imgargs)
    ax = plot_tabhillas_direction_mono(ptabhillas, tabtel_idx[0], tabev_idx[0], ax=ax, **kwargs)

    return ax


def plot_tabhillas_signal_camera_grid(ptabhillas, pcal, eventId, recoquality=[0], imgargs={}, **kwargs):
    """
    Plot all the camera images with the Hillas parameters from a given event in a grid

    Parameters
    ----------
    ptabhillas: `hipectaold.data.PTabHillas` object
    pcal: `hipectaold.data.PCalibRun` object
    eventId: int
    recoquality: list of int
    imgargs: Dictionnary of keywords arguments passed to `plot_camera_image`
    **kwargs: Extra keyword arguments are passed to `plot_tabhillas_direction_mono`

    Returns
    -------
    fig, axes:
        fig : `~matplotlib.figure.Figure`
        ax : `~matplotlib.axes.Axes`
    """

    tel_ids = [tel.telId for tel in ptabhillas.tabTel if
               (eventId in [ev.eventId for ev in tel.tabEvent if ev.recoQuality in recoquality])]

    m = 3
    n = len(tel_ids) // 3 + 1 * (len(tel_ids) % 3 > 0)
    if n > 0:
        fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n))

        for i, tid in enumerate(tel_ids):
            ax = axes[i // m][i % m]
            ax = plot_tabhillas_signal_camera_mono(ptabhillas, pcal, tid, eventId, ax=ax, imgargs=imgargs, **kwargs)
            ax.set_title("Event{} - Telescope {}".format(eventId, tid))
            ax.xaxis.set_tick_params(labelsize=10)
            ax.yaxis.set_tick_params(labelsize=10)

        for ax in axes.reshape(m*n, )[len(tel_ids):]:
            ax.axis('off')

    else:
        fig, axes = plt.subplots()

    return fig, axes



def stacked_images_event(pcal, eventId, layout=None, verbose=False):
    """
    Compute the stacked pixels signals of cameras in an event
    !!! ALL cameras should have the same number of pixels !!!
    A layout may be given as a list of telescope indexes

    Parameters
    ----------
    pcal: `hipectaold.data.PCalibRun`
    eventId: int
    layout: list of int, optional
    verbose: boolean, optional: print the number of stacked images

    Returns
    -------
    `numpy.ndarray`
    """
    import hipectaold.data as cd
    telIdx, eIdx = cd.find_event_dl0(pcal, eventId)
    telescopesIndex = telIdx[np.in1d(telIdx, layout)]
    eventIndex = eIdx[np.in1d(telIdx, layout)]

    stacked_signal = np.zeros(len(pcal.tabTelescope[telescopesIndex[0]].tabTelEvent[eventIndex[0]].tabPixel))

    msg = "The camera of the telescope with id {0} has {1} pixels != first camera that has {2} pixels"

    for tid, eid in zip(telescopesIndex, eventIndex):
        assert len(stacked_signal) == len(pcal.tabTelescope[tid].tabTelEvent[eid].tabPixel), \
            msg.format(tid, len(pcal.tabTelescope[tid].tabTelEvent[eid].tabPixel), len(stacked_signal))
        stacked_signal += pcal.tabTelescope[tid].tabTelEvent[eid].tabPixel

    if verbose:
        print("{} event stacked".format(len(telescopesIndex)))

    return stacked_signal


def plot_stacked_images_event(pcal, eventId, layout=None, ax=None, **kwargs):
    """
    Plot the stacked pixels signals of cameras in an event
    !!! ALL cameras should have the same number of pixels !!!
    A layout may be given as a list of telescope indexes


    Parameters
    ----------
    pcal : `hipectaold.data.PCalibRun`
    eventId : int
    layout: list of int, optional
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `matplotlib.pyplot.scatter`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """
    import hipectaold.data as cd
    ax = plt.gca() if ax is None else ax

    stack = stacked_images_event(pcal, eventId, layout=layout)
    telescopesIndex, eventIndex = cd.find_event_dl0(pcal, eventId)
    pix_pos = pcal.tabTelescope[telescopesIndex[0]].tabPos.tabPixelPosXY

    cf = ax.scatter(pix_pos[:, 0], pix_pos[:, 1], c=stack, **kwargs)
    ax.axis('equal')

    plt.colorbar(cf, ax=ax)

    return ax


def plot_stacked_event(ptabhillas, pcal, eventId, recoquality=[0], ax=None, imgargs={}, **kwargs):
    """
    Plot the stacked images and their Hillas directions

    Parameters
    ----------
    ptabhillas: `hipectaold.data.PTabHillas`
    pcal: `hipectaold.data.PCalibRun`
    eventId: int
    recoquality: list of int, optional
    ax: `~matplotlib.axes.Axes`
    imgargs: Dictionnary of keywords arguments passed to `plot_stacked_images_event`
    **kwargs: Extra keyword arguments are passed to `plot_tabhillas_direction_stereo`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`

    Example
    -------
    ```
    import hipectaold.data as cd
    import hipectaold.dataset as ds

    ptab = cd.load_ctafile(ds.get('pcal.ptabhillas'))
    pcal = cd.load_ctafile(ds.get('pcal.pcalibrun'))

    eventId = ptab.tabTel[0].tabEvent[0].eventId

    ax = plot_stacked_event(ptab, pcal, eventId, recoquality=range(6), imgargs={'s':8, 'cmap':plt.get_cmap('Greys')}, color='red')
    ```
    """
    ax = plt.gca() if ax is None else ax

    telids = [tel.telId for tel in ptabhillas.tabTel]
    layout = np.array([tel.telescopeId - 1 for idx, tel in enumerate(pcal.tabTelescope) if tel.telescopeId in telids])

    ax = plot_stacked_images_event(pcal, eventId, layout=layout, **imgargs)
    ax = plot_tabhillas_direction_stereo(ptabhillas, eventId, ax=ax, **kwargs)

    return ax


def plot_simu_source(psimu, eventId, ax=None, **kwargs):
    """
    Plot the position of the simulated source in the sky

    Parameters
    ----------
    psimu: `hipectaold.data.PSimu`
    eventId: int
    ax: `~matplotlib.axes.Axes`
    **kwargs: Extra keyword arguments are passed to `plot_tabhillas_direction_stereo`

    Returns
    -------
    ax: `~matplotlib.axes.Axes`
    """
    ax = plt.gca() if ax is None else ax

    event = [ev for ev in psimu.tabSimuEvent if ev.id == eventId][0]
    shower = [sh for sh in psimu.tabSimuShower if sh.id == event.showerNum][0]

    ax.scatter(shower.altitude, shower.azimuth, label='Simulated source', marker='x')
    ax.legend()

    return ax