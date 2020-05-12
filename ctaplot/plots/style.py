import matplotlib as mpl

_SizeTitlePaper = 11
_SizeLabelPaper = 9
_SizeTickPaper = 8
_SizeTitleSlides = 28
_SizeLabelSlides = 24
_SizeTickSlides = 20

_global_style = 'notebook'  # internal - set by `set_style`


def set_style(style='notebook'):
    """
    Set styling for plots

    Parameters
    ----------
    style: str
        'notebook', 'slides' or 'paper'
    """
    mpl.pyplot.style.use('seaborn-deep')
    set_figsize()
    _global_style = style
    set_font(style=style)


def set_figsize(style='notebook'):
    """
    Set default figsize
    Parameters
    ----------
    style: str
        'notebook', 'slides' or 'paper'
    """
    if style == 'notebook' or 'slides':
        mpl.rcParams['figure.figsize'] = (12, 8)
    elif style == 'paper':
        mpl.rcParams['figure.figsize'] = (5.25, 3.5)  # column-width in inches of a 2-columns article
    else:
        raise ValueError


def set_font(style='notebook'):
    """
    Set font style

    Parameters
    ----------
    output: str
        'notebook', 'slides' or 'paper'
    """
    if style == 'slides' or 'notebook':
        size_label = _SizeLabelSlides
        size_tick = _SizeTickSlides
        size_title = _SizeTitleSlides
        if style == 'slides':
            mpl.rc('text', usetex=True)
        else:
            mpl.rc('text', usetex=False)
    elif style == 'paper':
        size_label = _SizeLabelPaper
        size_tick = _SizeTickPaper
        size_title = _SizeTitlePaper
        mpl.rc('text', usetex=True)
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

