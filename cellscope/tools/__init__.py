"""Analysis tools; expression methods accept AnnData and MuData."""

from ._api import (
    activity,
    annotate,
    aucell,
    differential_expression,
    leiden,
    markers,
    pca,
    pseudobulk,
    reference_mapping,
    scanvi,
    score_genes,
    scvi,
    trajectory,
    umap,
    workflow,
)
from ._differential import de_methods, find_all_markers, find_markers, pseudobulk_de
from .compositional_analysis import cell_composition, composition_test
from .utils import Chrom_size, read_json, subset

__all__ = [
    "Chrom_size",
    "activity",
    "add_label",
    "annotate",
    "aucell",
    "cell_composition",
    "composition_test",
    "de_methods",
    "deseq",
    "differential_expression",
    "find_all_markers",
    "find_markers",
    "get_rank_array",
    "leiden",
    "markers",
    "pca",
    "pseudobulk",
    "pseudobulk_de",
    "read_json",
    "reference_mapping",
    "scanvi",
    "score_genes",
    "scvi",
    "subset",
    "trajectory",
    "umap",
    "workflow",
]


def __getattr__(name):
    if name == "add_label":
        from .cell_type_annotation import add_label

        return add_label
    if name in {"get_rank_array", "deseq"}:
        from . import gene_level_analysis

        return getattr(gene_level_analysis, name)
    raise AttributeError(name)
