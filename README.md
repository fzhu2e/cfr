[![PyPI version](https://badge.fury.io/py/cfr.svg)](https://badge.fury.io/py/cfr)
[![PyPI](https://img.shields.io/badge/python-3.13-blue.svg)]()
[![license](https://img.shields.io/github/license/fzhu2e/cfr.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7855587.svg)](https://doi.org/10.5281/zenodo.7855587)
![NSF 1948822](https://img.shields.io/badge/NSF-Award%20%231948822-blue?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=)
![NSF 2202777](https://img.shields.io/badge/NSF-Award%20%232202777-blue?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=)
![NSF 2303530](https://img.shields.io/badge/NSF-Award%20%232303530-blue?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=)
![NOAA CPO](https://img.shields.io/badge/NOAA%20Climate%20Program-Award%20%23NA18OAR4310426-0077be?style=flat-square)



# `cfr`: a Python package for Climate Field Reconstruction

> [!NOTE]
> If you use `cfr` in any way for your publications, please cite:
> 
> + Zhu, F., Emile-Geay, J., Hakim, G. J., Guillot, D., Khider, D., Tardif, R., & Perkins, W. A. (2024). cfr (v2024.1.26): a Python package for climate field reconstruction. Geoscientific Model Development, 17(8), 3409–3431. https://doi.org/10.5194/gmd-17-3409-2024
> + Zhu, F., Emile-Geay, J., Anchukaitis, K.J., McKay, N.P., Stevenson, S., Meng, Z., 2023. A pseudoproxy emulation of the PAGES 2k database using a hierarchy of proxy system models. Sci Data 10, 624. https://doi.org/10.1038/s41597-023-02489-1

`cfr` aims to provide a universal framework for climate field reconstruction (CFR).
It provides a toolkit for

+ the processing and visualization of the proxy records, climate model simulations, and instrumental observations,
+ the calibration and running of the proxy system models (PSMs, [Evans et al., 2013](https://doi.org/10.1016/j.quascirev.2013.05.024)),
+ the preparation and running of the multiple reconstruction frameworks/algorithms, such as LMR ([Hakim et al., 2016](https://doi.org/10.1002/2016JD024751); [Tardif et al., 2019](https://doi.org/https://doi.org/10.5194/cp-15-1251-2019)) and GraphEM ([Guillot et al., 2015](https://doi.org/10.1214/14-AOAS794)), and
+ the validation of the reconstructions, etc.

For more details, please refer to the documentation linked below.

## Documentation

+ Homepage: https://fzhu2e.github.io/cfr
+ Installation: https://fzhu2e.github.io/cfr/ug-installation.html
