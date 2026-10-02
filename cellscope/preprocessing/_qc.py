"""Compatibility QC entry points; no dependency on external file utilities."""

import re

import matplotlib.pyplot as plt
import scanpy as sc

from .._legacy_save import save_images
from ..tools import read_json
from ._api import mad_outliers


def flag_gene_family(
    adata, *, species="hsa", gene_family_name=None, gene_family_pattern=None, gene_list=None
):
    genes = read_json("genes.json")
    if species:
        adata.var["Mito"] = adata.var_names.isin(genes[f"{species}_mito"])
        if genes.get(f"{species}_ribo"):
            adata.var["Ribo"] = adata.var_names.isin(genes[f"{species}_ribo"])
    if (gene_family_pattern is not None or gene_list is not None) and gene_family_name is None:
        raise ValueError("gene_family_name is required for a custom gene family")
    if gene_family_pattern is not None:
        adata.var[gene_family_name] = adata.var_names.str.contains(
            gene_family_pattern, flags=re.IGNORECASE
        )
    if gene_list is not None:
        adata.var[gene_family_name] = adata.var_names.isin(gene_list)


def fastqc(
    adata,
    *,
    qc_vars,
    sample="Sample",
    outdir=".",
    min_genes=200,
    min_cells=3,
    percent_top=(20, 50),
    log1p=True,
    dpi=300,
    inplace=True,
    formats=("pdf", "png"),
):
    data = adata if inplace else adata.copy()
    if min_genes:
        sc.pp.filter_cells(data, min_genes=min_genes)
    if min_cells:
        sc.pp.filter_genes(data, min_cells=min_cells)
    tops = tuple(top for top in percent_top if top <= data.n_vars) if percent_top else None
    sc.pp.calculate_qc_metrics(data, qc_vars=qc_vars, percent_top=tops, log1p=log1p, inplace=True)
    keys = ["total_counts", "n_genes_by_counts", *[f"pct_counts_{key}" for key in qc_vars]]
    save = save_images(outdir=outdir, dpi=dpi, formats=formats)
    figure, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for key, ax in zip(keys, axes.ravel(), strict=True):
        sc.pl.violin(data, keys=key, groupby=sample, rotation=90, show=False, ax=ax)
    figure.tight_layout()
    save("QC_Violin")
    return None if inplace else data


def mad_filter(adata, *metric_nmad, **kwargs):
    data = adata.copy()
    if not metric_nmad:
        raise ValueError("Supply at least one metric/nmad pair")
    keys = []
    for index, (metric, nmads) in enumerate(metric_nmad):
        key = f"_mad_{index}"
        mad_outliers(
            data, metrics=[metric], batch_key=kwargs.get("batch_key"), nmads=nmads, key_added=key
        )
        keys.append(key)
    keep = ~data.obs[keys].any(axis=1)
    data.obs.drop(columns=keys, inplace=True)
    return data[keep].copy()
