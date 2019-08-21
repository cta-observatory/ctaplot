ctaplot
-------

ctaplot is a collection of functions to make instrument response functions (IRF) and reconstruction quality-checks plots for Imaging Atmospheric Cherenkov Telescopes such as CTA

Given a list of reconstructed and simulated quantities, compute and plot the Instrument Response Functions such as:

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

----

The CTA instrument response functions data used in ctaplot come from the CTA Consortium and Observatory and may be found on the `cta-observatory website <http://www.cta-observatory.org/science/cta-performance/>`_ .

In cases for which the CTA instrument response functions are used in a research project, we ask to add the following acknowledgement in any resulting publication:    

“This research has made use of the CTA instrument response functions provided by the CTA Consortium and Observatory, see http://www.cta-observatory.org/science/cta-performance/ (version prod3b-v2) for more details.”

----


.. image:: https://readthedocs.org/projects/ctaplot/badge/?version=latest
   :target: https://ctaplot.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
    
.. image:: https://travis-ci.org/vuillaut/ctaplot.svg?branch=master
    :target: https://travis-ci.org/vuillaut/ctaplot
    
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT


Install
-------


Requirements packages:

* python > 3.6
* numpy  
* scipy>=0.19    
* matplotlib>=2.0
* astropy

We recommend the use of `anaconda <https://www.anaconda.com>`_

The package is available through pip:

.. code-block:: bash

   pip install ctaplot

