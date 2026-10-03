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
from ._best_practices import (
    bbknn,
    celltypist,
    communication,
    grn,
    harmony,
    milo,
    mixscape,
    palantir,
    perturbation_distance,
    regulon_activity,
    sccoda,
    scenic_regulons,
    tsne,
    velocity,
)
from ._differential import de_methods, find_all_markers, find_markers, pseudobulk_de
from ._r_methods import (
    glmpca,
    nichenet,
    slingshot,
    tradeseq,
)
from .compositional_analysis import cell_composition, composition_test
from .utils import read_json, subset

__all__ = [
    "activity",
    "add_label",
    "annotate",
    "aucell",
    "bbknn",
    "cell_composition",
    "celltypist",
    "communication",
    "composition_test",
    "de_methods",
    "deseq",
    "differential_expression",
    "find_all_markers",
    "find_markers",
    "get_rank_array",
    "glmpca",
    "grn",
    "harmony",
    "leiden",
    "markers",
    "milo",
    "mixscape",
    "nichenet",
    "palantir",
    "pca",
    "perturbation_distance",
    "pseudobulk",
    "pseudobulk_de",
    "read_json",
    "reference_mapping",
    "regulon_activity",
    "scanvi",
    "sccoda",
    "scenic_regulons",
    "score_genes",
    "scvi",
    "slingshot",
    "subset",
    "tradeseq",
    "trajectory",
    "tsne",
    "umap",
    "velocity",
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
