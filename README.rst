ctaplot
-------

ctaplot is a collection of functions to make IRF and reconstruction quality-checks plots for Imaging Atmospheric Cherenkov Telescopes such as CTA

Given a list of reconstructed and simulated quantities, compute and plot the Instrument Response Functions:

* angular resolution
* energy resolution
* effective surface
* impact point resolution

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

----

Examples
--------

.. code-block:: python

    fig, ax = plt.subplots(figsize=(12,8))
    ax = ctaplot.plot_angular_res_requirement('north', ax=ax, linewidth=3)
    ax = ctaplot.plot_angular_res_cta_performance('north', ax=ax, marker='o')
    ax = ctaplot.plot_angular_res_requirement('south', ax=ax,  linewidth=3)
    ax = ctaplot.plot_angular_res_cta_performance('south', ax=ax, marker='o')
    ax.grid()
    plt.legend(prop = font)

.. image:: share/images/CTA_angular_resolution.png
   :target: share/images/CTA_angular_resolution.png
   :alt: CTA angular resolution

.. code-block:: python

    fig, ax = plt.subplots(figsize=(12,8))
    ax = ctaplot.plot_energy_resolution_requirements('north', ax=ax, linewidth=3)
    ax = ctaplot.plot_energy_resolution_cta_performances('north', ax=ax, marker='o')
    ax = ctaplot.plot_energy_resolution_requirements('south', ax=ax,  linewidth=3)
    ax = ctaplot.plot_energy_resolution_cta_performances('south', ax=ax, marker='o')
    ax.grid()
    plt.legend(prop = font)

.. image:: share/images/CTA_effective_area.png
   :target: share/images/CTA_effective_area.png
   :alt: CTA effective area

.. code-block:: python

    fig, ax = plt.subplots(figsize=(12,8))
    ax = ctaplot.plot_effective_area_requirement('north', ax=ax, linewidth=3)
    ax = ctaplot.plot_effective_area_performances('north', ax=ax, marker='o')
    ax = ctaplot.plot_effective_area_requirement('south', ax=ax,  linewidth=3)
    ax = ctaplot.plot_effective_area_performances('south', ax=ax, marker='o')
    ax.grid()
    plt.legend(prop = font)

.. image:: share/images/CTA_energy_resolution.png
   :target: share/images/CTA_energy_resolution.png
   :alt: CTA energy resolution

.. code-block:: python

    fig, ax = plt.subplots(figsize=(12,8))
    ax = ctaplot.plot_sensitivity_requirement('north', ax=ax, linewidth=3)
    ax = ctaplot.plot_sensitivity_performances('north', ax=ax, marker='o')
    ax = ctaplot.plot_sensitivity_requirement('south', ax=ax,  linewidth=3)
    ax = ctaplot.plot_sensitivity_performances('south', ax=ax, marker='o')
    ax.set_ylabel(r'Flux Sensitivity $[erg.cm^{-2}.s^{-1}]$')
    ax.grid()
    plt.legend(prop = font)

.. image:: share/images/CTA_sensitivity.png
   :target: share/images/CTA_sensitivity.png
   :alt: CTA sensitivity

