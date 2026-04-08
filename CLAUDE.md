# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**cfr** (Climate Field Reconstruction) is a Python package for paleoclimate field reconstruction. It processes proxy records, climate model simulations, and instrumental observations to reconstruct past climate fields using methods like data assimilation (LMR/EnKF) and GraphEM.

- **Documentation**: https://fzhu2e.github.io/cfr
- **Python**: 3.11, 3.12, 3.13
- **Authors**: Feng Zhu, Julien Emile-Geay
- **Conda env**: `cfr-py313`

## Build & Install

Uses `pyproject.toml` with setuptools backend.

```bash
pip install -e .                  # core install
pip install -e ".[psm]"          # with PSM extras (pathos, fbm)
pip install -e ".[ml]"           # with ML extras (sklearn, torch)
pip install -e ".[graphem]"      # with GraphEM extras (cython, cfr-graphem)
```

Publishing: `python -m build && twine upload dist/*`

## Testing

Tests are notebook-based (no `tests/` directory). Run with:

```bash
pip install pytest pytest-xdist nbmake
pytest --nbmake -n=auto --nbmake-timeout=3000 ./docsrc/notebooks/*.ipynb
```

New reconstruction methods **must** include a pseudoproxy experiment (PPE) using the pseudoPAGES2k dataset for comparability.

## Documentation

Built with MkDocs (Material theme). Source files in `docs/`, notebooks in `docsrc/notebooks/` (symlinked into `docs/notebooks`).

```bash
pip install mkdocs mkdocs-material mkdocs-jupyter mkdocstrings[python]
mkdocs serve                     # local dev server at http://127.0.0.1:8000
mkdocs gh-deploy                 # publish to GitHub Pages (gh-pages branch)
```

## Architecture

### Backend System

The package uses a backend selection system. Core modules live under versioned subpackages:

```python
import cfr
cfr.use('v2024')  # default, loaded on import
cfr.use('v2026')  # experimental
```

`cfr.use()` dynamically imports the backend and copies its public API into the `cfr` namespace.

### Core Classes (in `cfr/v2024/`)

| Module | Key Classes | Purpose |
|--------|------------|---------|
| `climate.py` | `ClimateField` | Gridded climate data (wraps xarray.DataArray) |
| `proxy.py` | `ProxyRecord`, `ProxyDatabase` | Individual and collections of proxy records |
| `ts.py` | `EnsTS` | Ensemble timeseries with uncertainty |
| `reconjob.py` | `ReconJob` | Orchestrates full reconstruction workflows |
| `reconres.py` | `ReconRes` | Loads and analyzes reconstruction results (netCDF) |
| `gcm.py` | `GCMCase`, `GCMCases` | Climate model simulation handling |
| `psm.py` | `Linear`, `Bilinear`, `VSLite`, `Coral_*`, `Ice_d18O`, etc. | Proxy System Models |
| `da/enkf.py` | `EnKF` | Ensemble Kalman Filter implementation |
| `ml.py` | `LinearNet`, `GRUNet`, `LSTMNet` | ML models (optional, requires torch) |
| `visual.py` | `PAGES2k`, `CartopySettings`, `STYLE` | Visualization utilities |

### Data Flow

1. **Input**: `ProxyDatabase` + `ClimateField` (priors from GCM simulations)
2. **Calibration**: PSM models calibrate proxy-climate relationships
3. **Reconstruction**: `ReconJob` orchestrates DA (EnKF) or GraphEM
4. **Output**: `ReconRes` loads netCDF results for validation and visualization

### Optional Dependencies

- `graphem` is an external package (`cfr-graphem`), imported at top-level with try/except
- `ml` module similarly guarded; requires torch

## Code Style

- CamelCase for classes (e.g., `ClimateField`, `ProxyDatabase`)
- `lowercase_with_underscores` for methods and variables
- Time values are assumed in increasing order throughout the codebase
- No linter/formatter configured; follow existing style
