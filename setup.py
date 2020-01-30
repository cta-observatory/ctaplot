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
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()


dataset = package_files('share')
dataset.append('./ctaplot/gammaboard/dashboard.ipynb')
dataset.append('README.rst')

print("dataset {}".format(dataset))

setup(name='ctaplot',
      version=get_property('__version__', 'ctaplot'),
      description="compute and plot cta IRF",
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
          'recommonmark',
          'sphinx>=1.4',
          'nbsphinx',
          'sphinx_rtd_theme',
      ],
      packages=find_packages(),
      tests_require=['pytest'],
      author='Thomas Vuillaume, Mikael Jacquemont',
      author_email='thomas.vuillaume@lapp.in2p3.fr',
      url='https://github.com/vuillaut/ctaplot',
      long_description=readfile('README.rst'),
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      data_files=[('ctaplot/', dataset)],
      entry_points={
          'console_scripts': [
              'gammaboard = ctaplot.gammaboard:open_dashboard'
          ]
      }
      )
