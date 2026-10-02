"""Compatibility entry points for existing cellscope expression workflows."""

import numpy as np

from ._api import aucell, differential_expression, markers, pseudobulk

__all__ = ["aucell", "deseq", "find_all_markers", "get_rank_array"]


def get_rank_array(adata, key, rank_name="rank_genes_groups"):
    return np.asarray(adata.uns[rank_name][key].tolist()).flatten()


def find_all_markers(adata, groupby="Cluster", use_raw=True):
    table = markers(adata, groupby=groupby, use_raw=use_raw, pts=True, key_added="AllMakers")
    return table.rename(
        columns={
            "group": "Identy",
            "names": "Feature",
            "scores": "Score",
            "pvals": "Pvalue",
            "pvals_adj": "Padj",
            "logfoldchanges": "LogFC",
            "pct_nz_group": "PTS",
            "pct_nz_reference": "PTS_Rest",
        }
    ).set_index("Feature")


def deseq(
    adata,
    cluster_key="CellType",
    sample_key="Sample",
    condition_key="Condition",
    ref_level="Control",
    n_jobs=20,
    *,
    layer="counts",
    design=None,
    contrast=None,
):
    """Compatibility wrapper using raw counts and current PyDESeq2 metadata API."""
    if contrast is None:
        levels = [level for level in adata.obs[condition_key].unique() if level != ref_level]
        if len(levels) != 1:
            raise ValueError("Supply contrast explicitly when more than two conditions are present")
        contrast = [condition_key, levels[0], ref_level]
    pdata = pseudobulk(
        adata,
        sample_col=sample_key,
        groups_col=cluster_key,
        layer=layer,
        min_cells=1,
        metadata_cols=[condition_key],
    )
    table, _ = differential_expression(
        pdata,
        design=design or f"~ {condition_key}",
        contrast=contrast,
        groups_col=cluster_key,
        sample_col=sample_key,
        n_cpus=n_jobs,
    )
    table = table.rename(
        columns={"gene": "GeneID", "log2FoldChange": "LFC", "padj": "FDR", "pvalue": "Pvalue"}
    )
    return {
        key: group.drop(columns="group").reset_index(drop=True)
        for key, group in table.groupby("group", observed=True)
    }
