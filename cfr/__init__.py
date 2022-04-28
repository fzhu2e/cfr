from .climate import *
from .proxy import *
from .reconjob import *
from .visual import (
    set_style,
    showfig,
    closefig,
    savefig,
)
set_style(style='journal', font_scale=1.4)

# get the version
from importlib.metadata import version
__version__ = version('cfr')