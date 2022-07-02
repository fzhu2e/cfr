from .climate import ClimateField, ClimateDataset
from .proxy import ProxyRecord, ProxyDatabase
from .reconjob import ReconJob
from .reconres import ReconRes
from . import graphem
try:
    from . import ml
except:
    pass

from .visual import (
    set_style,
    showfig,
    closefig,
    savefig,
)
set_style(style='journal', font_scale=1.2)

# get the version
from importlib.metadata import version
__version__ = version('cfr')


# mute future warnings from pkgs like pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)