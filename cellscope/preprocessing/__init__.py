"""Preprocessing namespace; legacy entry points remain available."""

from ._adata_preprocessing import cellbender, filter_qc, scDblFinder
from ._api import (
    doublets,
    filter_cells,
    highly_variable_genes,
    mad_outliers,
    neighbors,
    normalize,
    qc,
)
from ._best_practices import (
    deviance_features,
    pearson_hvg,
    pearson_residuals,
    scdblfinder,
    scran,
    sctransform,
    soupx,
)
from ._python_qc import cellbender_file, scdblfinder_py

__all__ = [
    "cellbender",
    "cellbender_file",
    "deviance_features",
    "doublets",
    "fastqc",
    "filter_cells",
    "filter_qc",
    "flag_gene_family",
    "highly_variable_genes",
    "mad_filter",
    "mad_outliers",
    "neighbors",
    "normalise",
    "normalize",
    "pearson_hvg",
    "pearson_residuals",
    "qc",
    "scDblFinder",
    "scdblfinder",
    "scdblfinder_py",
    "scran",
    "sctransform",
    "soupx",
]


def __getattr__(name):
    if name == "normalise":
        from .._logging import instrument_function
        from ._normalization import normalise

        return instrument_function(normalise, "pp.normalise")
    if name in {"flag_gene_family", "fastqc", "mad_filter"}:
        from .._logging import instrument_function
        from . import _qc

        return instrument_function(getattr(_qc, name), f"pp.{name}")
    raise AttributeError(name)
