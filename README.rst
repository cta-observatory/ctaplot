=======
ctaplot
=======

ctaplot is a collection of functions to make instrument response functions (IRF) and reconstruction quality-checks plots for Imaging Atmospheric Cherenkov Telescopes such as CTA

Given a list of reconstructed and simulated quantities, compute and plot the Instrument Response Functions such as:

* angular resolution
* energy resolution
* effective surface
* impact point resolution


You may find examples in the `documentation <https://ctaplot.readthedocs.io/en/latest/>`_.     
Or you can run a simple one here:

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/vuillaut/ctaplot/master?filepath=examples%2Fnotebooks%2Fresolution_examples.ipynb

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

.. image:: https://github.com/vuillaut/ctaplot/workflows/CI/badge.svg
   :target: https://github.com/vuillaut/ctaplot/actions?query=workflow%3ACI
   :alt: Continuous Integration Action

.. image:: https://readthedocs.org/projects/ctaplot/badge/?version=latest
   :target: https://ctaplot.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
    
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT


Install
=======


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


.. code-block:: bash

    export GAMMABOARD_DATA=path_to_the_data_directory


We recommend that you add this line to your bash source file (`$HOME/.bashrc` or `$HOME/.bash_profile`)



GammaBoard
==========

*A dashboard to show them all.*


GammaBoard is a simple jupyter dashboard thought to display metrics assessing the reconstructions performances of
Imaging Atmospheric Cherenkov Telescopes (IACTs).
Deep learning is a lot about bookkeeping and trials and errors. GammaBoard ease this bookkeeping and allows quick
comparison of the reconstruction performances of your machine learning experiments.

It is a working prototype used in CTA, especially by the [GammaLearn](https://gitlab.lapp.in2p3.fr/GammaLearn/) project.


Run GammaBoard
--------------

To launch the dashboard, you can simply try the command:

.. code-block:: bash

    gammaboard

This will run a temporary copy of the dashboard (a jupyter notebook).
Local changes that you make in the dashboard will be discarded afterwards.

GammaBoard is using data in a specific directory storing all your experiments files.
This directory is known under `$GAMMABOARD_DATA` by default.
However, you can change the path access at any time in the dashboard itself.

Demo
----

Here is a simple demo of GammaBoard:  

* On top the plots (metrics) such as angular resolution and energy resolution.
* Below, the list of experiments in the user folder.

When an experiment is selected in the list, the data is automatically loaded, the metrics computed and displayed.
A list of information provided during the training phase is also displayed.
As many experiments results can be overlaid.
When an experiment is deselected, it simply is removed from the plots.


.. image:: /share/gammaboard.gif
   :alt: gammaboard_demo

