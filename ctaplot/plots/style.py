import matplotlib as mpl
from distutils.spawn import find_executable
from contextlib import contextmanager
import logging
from ..io.dataset import get

logger = logging.getLogger(__name__)


def check_latex():
    """
    Check if a latex distribution is installed and has required packages.

    Returns
    -------
    bool: True if a LaTeX distribution with required packages could be found
    """
    if not find_executable('latex'):
        return False
    
    # Check if dvipng is available (needed for matplotlib LaTeX rendering)
    if not find_executable('dvipng'):
        logger.warning("LaTeX found but dvipng is missing. Install dvipng for full LaTeX support.")
        return False
    
    return True


@contextmanager
def context(style='notebook'):
    """
    Context manager for styling options
    Styling adapted from the `seaborn-deep` style.
    'slides' and 'paper' will use the LaTeX distribution if one is available

    Parameters
    ----------
    style: str
        'notebook', 'slides' or 'paper'

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from ctaplot.plots.style import context
    >>> with context('notebook'):
    >>>     plt.plot([1, 2, 4])
    """
    style_path = get(f'ctaplot-{style}')
    with mpl.style.context(['seaborn-v0_8-deep', style_path]):
        latex_available = check_latex()
        if not latex_available:
            mpl.rcParams['text.usetex'] = False
            if style in ['slides', 'paper']:
                logger.info(f"LaTeX not fully available. For enhanced {style} rendering, "
                           "install LaTeX with dvipng and cm-super packages.")
        yield


def set_style(style='notebook'):
    """
    Set styling for plots adapted from the `seaborn-deep` style.
    'slides' and 'paper' will use the LaTeX distribution if one is available

    Parameters
    ----------
    style: str
        'notebook', 'slides' or 'paper'
    """

    style_path = get(f'ctaplot-{style}')
    mpl.pyplot.style.use(['seaborn-v0_8-deep', style_path])

    latex_available = check_latex()
    if not latex_available:
        mpl.rcParams['text.usetex'] = False
        if style in ['slides', 'paper']:
            logger.info(f"LaTeX not fully available. For enhanced {style} rendering, "
                       "install LaTeX with dvipng and cm-super packages.")


