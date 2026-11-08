"""Plotting namespace with explicit multi-format export and native returns."""

from ._api import cell_composition, dotplot, embedding, heatmap, qc, volcano
from ._pca import pca_variance_ratio
from ._phenotrack import phenotrack
from ._phenotrack_ternary import phenotrack_ternary

__all__ = [
    "cell_composition",
    "cell_ratio",
    "dimplot",
    "dotplot",
    "embedding",
    "heatmap",
    "pca_variance_ratio",
    "phenotrack",
    "phenotrack_ternary",
    "plot_batch_effect",
    "qc",
    "volcano",
]


def __getattr__(name):
    from .._logging import instrument_function

    if name in {"dimplot", "plot_batch_effect"}:
        from . import _emmbeding

        return instrument_function(getattr(_emmbeding, name), f"pl.{name}")
    if name == "cell_ratio":
        from ._stat import cell_ratio

        return instrument_function(cell_ratio, "pl.cell_ratio")
    raise AttributeError(name)
