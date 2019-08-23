#!/usr/bin/env python
# Licensed under MIT license - see LICENSE

from setuptools import setup
import re


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()


setup(name='gammaboard',
      version=get_property('__version__', 'gammaboard'),
      description="A dashboard to show them all",
      install_requires=[
          'ctaplot>=0.3.0',
          'pytables',
          'pandas',
          'scikit-learn',
      ],
      packages=['gammaboard'],
      tests_require=['pytest'],
      author='Thomas Vuillaume, Mikael Jacquemont',
      author_email='thomas.vuillaume@lapp.in2p3.fr',
      license=readfile('LICENSE'),
      url='https://github.com/gammaboard/gammaboard',
      long_description=readfile('README.md'),
      data_files=[('gammaboard/', 'gammaboard/dashboard.ipynb')],
      entry_points={
          'console_scripts': [
              'gammaboard = gammaboard:open_dashboard'
          ]
      }
      )


