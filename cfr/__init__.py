"""
CFR - Climate Field Reconstruction

A Python package for climate field reconstruction using various methods.
"""

import sys
from importlib import import_module

# Registry of available backends
_BACKENDS = {
    'v2024': {
        'description': 'cfr v2024',
        'module': 'cfr.v2024',
    },
    'v2026': {
        'description': 'cfr v2026 (experimental)',
        'module': 'cfr.v2026',
    },
}

_active_backend = None


def use(backend='v2024'):
    """
    Load and activate a reconstruction backend.
    
    This function dynamically imports the specified backend module and makes
    its public API available in the cfr namespace.
    
    Parameters
    ----------
    backend : str, optional
        Name of the backend to load. Options: 'v2024', 'v2026'.
        Defaults to 'v2024'.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the backend is not recognized or cannot be imported.
    
    Examples
    --------
    >>> import cfr
    >>> cfr.use('v2024')  # Load v2024 methods (default)
    >>> job = cfr.ReconJob()
    
    >>> cfr.use('v2026')  # Switch to v2026 methods
    >>> job = cfr.ReconJob()
    """
    global _active_backend
    
    if backend not in _BACKENDS:
        available = ', '.join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend '{backend}'. Available: {available}")
    
    try:
        backend_module = import_module(_BACKENDS[backend]['module'])
    except ImportError as e:
        raise ImportError(
            f"Failed to load backend '{backend}': {e}\n"
            f"Description: {_BACKENDS[backend]['description']}"
        ) from e
    
    # Copy public attributes from backend to cfr namespace
    for attr in dir(backend_module):
        if not attr.startswith('_'):
            globals()[attr] = getattr(backend_module, attr)
    
    _active_backend = backend
    
    # Print status message (optional, can be disabled with verbose=False)
    # print(f">>> Loaded cfr backend: '{backend}'")


def get_active_backend():
    """
    Get the name of the currently active backend.
    
    Returns
    -------
    str or None
        Name of the active backend, or None if no backend has been loaded.
    """
    return _active_backend


def list_backends():
    """
    List all available backends with descriptions.
    
    Returns
    -------
    dict
        Dictionary of backend names and their descriptions.
    """
    return {name: info['description'] for name, info in _BACKENDS.items()}


# Load v2024 backend by default on import
use('v2024')

# get the version
from importlib.metadata import version
__version__ = version('cfr')

import warnings
try:
    from cartopy.io import DownloadWarning
    warnings.filterwarnings('ignore', category=DownloadWarning)
except (ImportError, TypeError):
    pass

# mute future warnings from pkgs like pandas
warnings.simplefilter(action='ignore', category=FutureWarning)
        
# mute the numpy warnings
warnings.simplefilter('ignore', category=RuntimeWarning)