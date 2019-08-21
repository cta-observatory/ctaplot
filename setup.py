#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from setuptools import setup

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('.', path, filename))
    return paths

dataset = package_files('share')

print("dataset {}".format(dataset))

setup(name='ctaplot',
      version='0.3.1',
      description="compute and plot cta IRF",
      install_requires=[
          'numpy',
          'matplotlib>=2.0',
          'scipy>=0.19',
          'astropy',
      ],
      packages=['ctaplot'],
      tests_require=['pytest'],
      author='Thomas Vuillaume',
      author_email='thomas.vuillaume@lapp.in2p3.fr',
      license='BSD3',
      url='https://github.com/vuillaut/ctaplot',
      long_description='',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Development Status :: 3 - Alpha',
      ],
      data_files=[('ctaplot/', dataset)],
      )
