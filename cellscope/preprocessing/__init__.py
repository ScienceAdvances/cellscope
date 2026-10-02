"""Preprocessing namespace; legacy entry points remain available."""

from ._api import (
    doublets,
    filter_cells,
    highly_variable_genes,
    mad_outliers,
    neighbors,
    normalize,
    qc,
)

__all__ = [
    "doublets",
    "fastqc",
    "filter_cells",
    "flag_gene_family",
    "highly_variable_genes",
    "mad_filter",
    "mad_outliers",
    "neighbors",
    "normalise",
    "normalize",
    "qc",
]


def __getattr__(name):
    if name == "normalise":
        from ._normalization import normalise

        return normalise
    if name in {"flag_gene_family", "fastqc", "mad_filter"}:
        from . import _qc

        return getattr(_qc, name)
    raise AttributeError(name)
