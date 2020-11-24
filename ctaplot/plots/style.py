import matplotlib as mpl
from distutils.spawn import find_executable
from contextlib import contextmanager
from ..io.dataset import get


def check_latex():
    """
    Check if a latex distribution is installed.

    Returns
    -------
    bool: True if a LaTeX distribution could be found
    """
    return not find_executable('latex') is None


@contextmanager
def context(style='notebook'):
    style_path = get(f'ctaplot-{style}')
    with mpl.style.context(['seaborn-deep', style_path]):
        if not check_latex():
            mpl.rcParams['text.usetex'] = False
        yield


def set_style(style='notebook'):
    mpl.rcParams.update(mpl.rcParamsDefault)

    style_path = get(f'ctaplot-{style}')
    mpl.pyplot.style.use(['seaborn-deep', style_path])

    if not check_latex():
        mpl.rcParams['text.usetex'] = False


