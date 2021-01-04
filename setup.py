#!/usr/bin/env python
# Licensed under a MIT license - see LICENSE.rst

import os
from setuptools import setup, find_packages
import re


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('.', path, filename))
    return paths


def get_property(prop, project):
    with open(os.path.join(project, '__init__.py')) as f:
        result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), f.read())
    return result.group(1)


dataset = package_files('share')
dataset.append('./ctaplot/gammaboard/dashboard.ipynb')
dataset.append('README.rst')

setup(
    packages=find_packages(),
    version=get_property('__version__', 'ctaplot'),
    install_requires=[
        'numpy>1.16',
        'matplotlib>=2.0',
        'scipy>=0.19',
        'astropy',
        'tables',
        'pandas',
        'scikit-learn',
        'jupyter',
        'ipywidgets',
        'pyyaml',
    ],
    tests_require=['pytest'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    data_files=[('ctaplot', dataset)],
    entry_points={
        'console_scripts': [
            'gammaboard = ctaplot.gammaboard:open_dashboard'
        ]
    }
)
