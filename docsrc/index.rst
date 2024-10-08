***********************************************************
`cfr`: a Python package for Climate Field Reconstruction
***********************************************************

.. note::
    If you use `cfr` in any way for your publications, please cite:

    + Zhu, F., Emile-Geay, J., Hakim, G. J., Guillot, D., Khider, D., Tardif, R., & Perkins, W. A. (2024). cfr (v2024.1.26): a Python package for climate field reconstruction. Geoscientific Model Development, 17(8), 3409–3431. https://doi.org/10.5194/gmd-17-3409-2024
    + Zhu, F., Emile-Geay, J., Anchukaitis, K.J., McKay, N.P., Stevenson, S., Meng, Z., 2023. A pseudoproxy emulation of the PAGES 2k database using a hierarchy of proxy system models. Sci Data 10, 624. https://doi.org/10.1038/s41597-023-02489-1


`cfr` aims to provide a universal framework for climate field reconstruction (CFR).
It provides a toolkit for

+ the processing and visualization of the proxy records, climate model simulations, and instrumental observations,
+ the calibration and running of the proxy system models (PSMs, `Evans et al., 2013 <https://doi.org/10.1016/j.quascirev.2013.05.024>`_),
+ the preparation and running of the multiple reconstruction frameworks/algorithms, such as LMR (`Hakim et al., 2016 <https://doi.org/10.1002/2016JD024751>`_; `Tardif et al., 2019 <https://doi.org/https://doi.org/10.5194/cp-15-1251-2019>`_) and GraphEM (`Guillot et al., 2015 <https://doi.org/10.1214/14-AOAS794>`_), and
+ the validation of the reconstructions, etc.

|

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card::  Installation
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/installation.png
        :link: ug-installation
        :link-type: doc

        Installation instructions.

    .. grid-item-card::  pseudoPAGES2k
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/pp2k.png
        :link: ug-pp2k
        :link-type: doc

        An illustration with the pseudoPAGES2k dataset.


    .. grid-item-card::  Climate
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/climate.png
        :link: ug-climate
        :link-type: doc

        Processing and visualization of the **gridded** climate model simulations and instrumental observations.

    .. grid-item-card::  Proxy
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/proxy.png
        :link: ug-proxy
        :link-type: doc

        Processing and visualization of the proxy records and databases.

    .. grid-item-card::  PSM
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/psm.png
        :link: ug-psm
        :link-type: doc

        Proxy System Models for multiple proxy types.

    .. grid-item-card::  LMR
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/lmr.png
        :link: ug-lmr
        :link-type: doc

        The Last Millennium Reanalysis (LMR) workflows.

    .. grid-item-card::  GraphEM
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/graphem.png
        :link: ug-graphem
        :link-type: doc

        The Graphical Expectation Maximization (GraphEM) workflows.

    .. grid-item-card::  API
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/api.png
        :link: ug-api
        :link-type: doc

        The essential API.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   ug-installation
   ug-pp2k
   ug-climate
   ug-proxy
   ug-psm
   ug-lmr
   ug-graphem
   ug-api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contributing Guide

   cg-overview
   cg-working-with-codebase
   cg-updating-docs