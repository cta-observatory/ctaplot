import pkg_resources
import os
import sys
import numpy as np

__all__ = ['get']

resources_list = ['CTA-Performance-prod3b-v1-South-20deg-50h-EffArea.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-05h-EffArea.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-50h-Angres.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-05h-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-50h-BackgroundSqdeg.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-50h-EffAreaNoDirectionCut.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-50h-EffAreaNoDirectionCut.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-30m-EffArea.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-50h-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-50h-Angres.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-30m-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-30m-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-50h-BackgroundSqdeg.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-50h-EffArea.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-05h-EffArea.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-50h-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-30m-EffArea.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-50h-Eres.txt',
                  'CTA-Performance-prod3b-v1-South-20deg-05h-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-North-20deg-50h-Eres.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-50h-Angres.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-05h-EffArea.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-50h-EffArea.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-05h-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-50h-Eres.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-50h-BackgroundSqdeg.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-50h-Angres.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-50h-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-50h-Eres.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-30m-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-30m-EffArea.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-05h-EffArea.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-50h-EffArea.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-30m-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-50h-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-50h-EffAreaNoDirectionCut.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-50h-EffAreaNoDirectionCut.txt',
                  'CTA-Performance-prod3b-v1-North-40deg-05h-DiffSens.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-50h-BackgroundSqdeg.txt',
                  'CTA-Performance-prod3b-v1-South-40deg-30m-EffArea.txt',
                  'cta_requirements_South-50h-LST.dat',
                  'cta_requirements_North-50h-LST.dat',
                  'cta_requirements_South-50h.dat',
                  'cta_requirements_South-50h-SST.dat',
                  'cta_requirements_North-50h-MST-ERes.dat',
                  'cta_requirements_North-30m-EffectiveArea.dat',
                  'cta_requirements_North-50h-MST.dat',
                  'cta_requirements_South-50h-MST.dat',
                  'cta_requirements_North-30m-LST-EffectiveArea.dat',
                  'cta_requirements_North-30m-MST-EffectiveArea.dat',
                  'cta_requirements_South-50h-MST-ERes.dat',
                  'cta_requirements_South-50h-SST-ERes.dat',
                  'cta_requirements_South-50h-ERes.dat',
                  'cta_requirements_South-50h-MST-AngRes.dat',
                  'cta_requirements_North-50h-LST-AngRes.dat',
                  'cta_requirements_South-30m-MST-EffectiveArea.dat',
                  'cta_requirements_North-50h-LST-ERes.dat',
                  'cta_requirements_North-50h.dat',
                  'cta_requirements_South-30m-LST-EffectiveArea.dat',
                  'cta_requirements_North-50h-MST-AngRes.dat',
                  'cta_requirements_South-50h-AngRes.dat',
                  'cta_requirements_South-50h-LST-AngRes.dat',
                  'cta_requirements_South-50h-SST-AngRes.dat',
                  'cta_requirements_South-50h-LST-ERes.dat',
                  'cta_requirements_South-30m-EffectiveArea.dat',
                  'cta_requirements_North-50h-AngRes.dat',
                  'cta_requirements_North-50h-ERes.dat',
                  'cta_requirements_South-30m-SST-EffectiveArea.dat',
                  'HESS_Impact_Angular_Resolution_Loose_Mono.txt',
                  'HESS_Impact_Effective_Area_Std_Mono.txt',
                  'HESS_Impact_Energy_Resolution_Loose_Mono.txt',
                  'HESS_Impact_Angular_Resolution_Safe_Mono.txt',
                  'HESS_Impact_Effective_Area_Stereo.txt',
                  'HESS_Impact_Energy_Resolution_Safe_Mono.txt',
                  'HESS_Impact_Angular_Resolution_Std_Mono.txt',
                  'HESS_Impact_Energy_Bias_Loose_Mono.txt',
                  'HESS_Impact_Energy_Resolution_Std_Mono.txt',
                  'HESS_Impact_Angular_Resolution_Stereo.txt',
                  'HESS_Impact_Energy_Bias_Safe_Mono.txt',
                  'HESS_Impact_Energy_Resolution_Stereo.txt',
                  'HESS_Impact_Effective_Area_Loose_Mono.txt',
                  'HESS_Impact_Energy_Bias_Std_Mono.txt',
                  'HESS_Impact_Effective_Area_Safe_Mono.txt',
                  'HESS_Impact_Energy_Bias_Stereo.txt',
                  'magic_sensitivity_2014.ecsv'
                  ]


def get(resource_name):
    """ get the filename for a resource """
    try:
        resource_path = find_resource(resource_name)
    except FileNotFoundError:
        if not pkg_resources.resource_exists(__name__, resource_name):
            raise FileNotFoundError("Couldn't find resource: '{}'"
                                    .format(resource_name))
        else:
            resource_path = pkg_resources.resource_filename(__name__, resource_name)
    return resource_path


def find_resource(resource_name):
    """
    Find a resource in the share directory

    Parameters
    ----------
    resource_name: str
        name of a file to find

    Returns
    -------
    str - absolute path to the resource
    """
    # If ctaplot is installed via python setup.py develop, data files stay in share
    share_dir = os.path.join(pkg_resources.resource_filename('ctaplot', ''), '../share/')
    gammaboard_dir = os.path.join(pkg_resources.resource_filename('ctaplot', ''), 'gammaboard/')
    resources_dirs = [share_dir, gammaboard_dir]
    for res_dir in resources_dirs:
        for root, dirs, files in os.walk(res_dir):
            if resource_name in files:
                return os.path.abspath(os.path.join(root, resource_name))

    # If ctaplot is installed via pip install, data files are copied in <sys.prefix>/ctaplot
    sys_dir = os.path.join(sys.prefix, 'ctaplot')
    if not os.path.exists(os.path.join(sys_dir, resource_name)):
        raise FileNotFoundError("Couldn't find resource: '{}'".format(resource_name))
    else:
        return os.path.join(sys_dir, resource_name)


def load_any_resource(filename):
    """
    Naive load of any resource text file that present data organised in a table after n lines of comments

    Parameters
    ----------
    filename: path

    Returns
    -------
    data: tuple of `numpy.ndarray`
    """
    sr = 0
    with open(get(filename)) as file:
        n_lines = len(file.readlines())

    while sr < n_lines:
        try:
            data = np.loadtxt(get(filename), skiprows=sr, unpack=True)
            break
        except:
            sr += 1

    return data
