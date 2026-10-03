"""Discover the public interfaces corresponding to Single-cell Best Practices.

The catalog describes routing, not algorithmic equivalence or installed software.
Optional backends are loaded only when an analysis function is called.
"""

import pandas as pd

# stage, interface, default backend, Python installation extra, R packages
_METHODS = (
    ("RNA preprocessing", "pp.qc", "scanpy", "", ""),
    ("RNA preprocessing", "pp.doublets", "scanpy/Scrublet", "", ""),
    ("RNA preprocessing", "pp.soupx", "pysoupx", "ambient", "SoupX (optional)"),
    ("RNA preprocessing", "pp.scdblfinder", "R/scDblFinder", "r", "scDblFinder"),
    ("Normalization", "pp.normalize", "scanpy", "", ""),
    ("Normalization", "pp.pearson_residuals", "scanpy", "", ""),
    ("Normalization", "pp.scran", "R/scran", "r", "scran, SingleCellExperiment"),
    ("Normalization", "pp.sctransform", "sctransform", "normalization", "sctransform (optional)"),
    ("Feature selection", "pp.pearson_hvg", "scanpy", "", ""),
    ("Feature selection", "pp.highly_variable_genes", "scanpy", "", ""),
    ("Feature selection", "pp.deviance_features", "R/scry", "r", "scry"),
    ("Dimension reduction", "tl.pca", "scanpy", "", ""),
    ("Dimension reduction", "tl.glmpca", "glmpca", "glmpca", "glmpca (optional)"),
    ("Dimension reduction", "tl.tsne", "scanpy", "", ""),
    ("Clustering", "tl.leiden", "scanpy", "clustering", ""),
    ("Clustering", "tl.umap", "scanpy", "", ""),
    ("Annotation", "tl.celltypist", "celltypist", "annotation", ""),
    ("Annotation", "tl.scanvi", "scvi-tools", "integration", ""),
    ("Integration", "tl.harmony", "harmonypy", "batch", ""),
    ("Integration", "tl.bbknn", "bbknn", "batch", ""),
    ("Integration", "tl.scvi", "scvi-tools", "integration", ""),
    ("Trajectories", "tl.trajectory", "scanpy/PAGA/DPT", "", ""),
    ("Trajectories", "tl.slingshot", "pyslingshot", "slingshot", "slingshot (optional)"),
    ("Trajectories", "tl.palantir", "palantir", "trajectory", ""),
    ("Trajectories", "tl.velocity", "scvelo", "trajectory", ""),
    ("Trajectories", "tl.tradeseq", "R/tradeSeq", "r", "tradeSeq"),
    ("Differential expression", "tl.find_all_markers", "scanpy", "", ""),
    ("Differential expression", "tl.pseudobulk_de", "pydeseq2", "de", ""),
    ("Composition", "tl.sccoda", "pertpy/scCODA/tascCODA", "composition", ""),
    ("Composition", "tl.milo", "pertpy/PyDESeq2", "composition", ""),
    ("Pathways and TFs", "tl.activity", "decoupler", "functional", ""),
    ("Pathways and TFs", "tl.aucell", "decoupler", "functional", ""),
    ("Regulatory networks", "tl.grn", "arboreto", "grn", ""),
    ("Regulatory networks", "tl.scenic_regulons", "pyscenic", "grn", ""),
    ("Regulatory networks", "tl.regulon_activity", "pyscenic", "grn", ""),
    ("Communication", "tl.communication", "liana", "communication", ""),
    ("Communication", "tl.nichenet", "nichenetpy", "communication", "nichenetr (optional)"),
    ("Perturbations", "tl.mixscape", "pertpy", "perturbation", ""),
    ("Perturbations", "tl.perturbation_distance", "pertpy", "perturbation", ""),
    ("Interoperability", "r.to_sce", "R/SingleCellExperiment", "r", "SingleCellExperiment"),
    ("Interoperability", "r.from_sce", "R/SingleCellExperiment", "r", "SingleCellExperiment"),
)


def methods(stage=None):
    """Return an independent routing table; optionally select an exact stage name."""
    table = pd.DataFrame(_METHODS, columns=["stage", "interface", "backend", "extra", "r_packages"])
    if stage is not None:
        table = table.loc[table.stage == stage]
    return table.reset_index(drop=True)
