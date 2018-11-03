ctaplot
-------

ctaplot is a collection of functions to make IRF and reconstruction quality-checks plots for Imaging Atmospheric Cherenkov Telescopes such as CTA

Given a list of reconstructed and simulated quantities, compute and plot the Instrument Response Functions:

* angular resolution
* energy resolution
* effective surface
* impact point resolution


You may find examples in the `documentation <https://ctaplot.readthedocs.io/en/latest/>`_.

----


* Code : https://github.com/vuillaut/ctaplot
* Documentation : https://ctaplot.readthedocs.io/en/latest/
* Author contact: Thomas Vuillaume - thomas.vuillaume@lapp.in2p3.fr
* License: MIT


.. image:: https://travis-ci.org/vuillaut/ctaplot.svg?branch=master
    :target: https://travis-ci.org/vuillaut/ctaplot

.. image:: https://readthedocs.org/projects/ctaplot/badge/?version=latest
   :target: https://ctaplot.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

----



Install
-------

Required packages:


* numpy  
* scipy>=0.19    
* matplotlib>=2.0   

We recommend the use of `anaconda <https://www.anaconda.com>`_

To install, get in ctaplot main directory and type:

.. code-block:: bash

   python setup.py install

