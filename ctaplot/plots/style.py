import matplotlib as mpl


SizeTitlePaper = 20
SizeLabelPaper = 18
SizeTickPaper = 16
SizeTitleSlides = 28
SizeLabelSlides = 24
SizeTickSlides = 20


def set_style(output='slides'):
    """
    Set styling for plots

    Parameters
    ----------
    output: str
        'slides' or 'paper'
    """
    mpl.pyplot.style.use('seaborn-deep')
    set_figsize()
    set_font(output=output)


def set_figsize():
    """
    Set default figsize
    """
    mpl.rcParams['figure.figsize'] = (12, 8)


def set_font(output='slides'):
    """
    Set font style

    Parameters
    ----------
    output: str
        'slides' or 'paper'
    """
    if output == 'slides':
        size_label = SizeLabelSlides
        size_tick = SizeTickSlides
        size_title = SizeTitleSlides
    elif output == 'paper':
        size_label = SizeLabelPaper
        size_tick = SizeTickPaper
        size_title = SizeTitlePaper
    else:
        raise ValueError

    params = {
        'axes.labelsize': size_label,
        'axes.titlesize': size_label,
        'figure.titlesize': size_title,
        'xtick.labelsize': size_tick,
        'ytick.labelsize': size_tick,
        'legend.fontsize': size_label,
        'legend.title_fontsize': size_title,
    }
    mpl.pyplot.rcParams.update(params)
    mpl.rc('font', **{'size': size_label})
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rc('text', usetex=True)
