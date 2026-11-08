"""cellscope: expression and cell-state analysis on native scverse objects."""

__version__ = "1.3.0"
from . import logging

from ._device import device_info

settings = logging.settings
logging.show_banner(__version__)

from . import best_practices, datasets, get, io, r
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl

for _namespace, _prefix in ((io, "io"), (pp, "pp"), (tl, "tl"), (pl, "pl")):
    logging.instrument_namespace(_namespace, _prefix)
del _namespace, _prefix
logging.debug("READY cellscope | io / pp / tl / pl")

__all__ = [
    "best_practices",
    "datasets",
    "device_info",
    "get",
    "io",
    "logging",
    "pl",
    "pp",
    "r",
    "settings",
    "tl",
]
