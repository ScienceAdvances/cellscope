"""cellscope: expression and cell-state analysis on native scverse objects."""

__version__ = "1.3.0"
from . import best_practices, datasets, get, io, r
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl

__all__ = [
    "best_practices",
    "datasets",
    "get",
    "io",
    "pl",
    "pp",
    "r",
    "tl",
]
