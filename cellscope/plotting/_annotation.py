"""Legacy annotation plots using native Scanpy/decoupler interfaces."""

from pathlib import Path

import pandas as pd
import scanpy as sc

from .._legacy_save import save_images
from ..tools._api import aucell
from ._emmbeding import dimplot


def plot_marker(
    adata,
    annotation,
    outdir=".",
    cell_type_key="CellType",
    marker_key="Marker",
    formats=("pdf", "png"),
    palette="Reds",
):
    table = pd.read_csv(annotation, sep="\t") if isinstance(annotation, (str, Path)) else annotation
    markers = {row[cell_type_key]: str(row[marker_key]).split(",") for _, row in table.iterrows()}
    save = save_images(outdir=outdir, formats=formats)
    plots = {}
    for name in ("dotplot", "stacked_violin", "matrixplot", "heatmap"):
        plots[name] = getattr(sc.pl, name)(
            adata, var_names=markers, groupby=cell_type_key, show=False, cmap=palette
        )
        save(f"Marker_{name}")
    for name, genes in markers.items():
        dimplot(adata, reduction="umap", outdir=outdir, filename=f"Marker_{name}", color=genes)
    return plots


def auc_heatmap(adata, marker, out_prefix, ref_key="Cluster", figsize=(12, 6), use_raw=True):
    import seaborn as sns

    network = marker.melt(var_name="source", value_name="target").dropna()
    scores = aucell(adata, network=network, use_raw=use_raw)
    means = scores.groupby(adata.obs[ref_key], observed=True).mean()
    plot = sns.clustermap(means.T, method="complete", z_score=0, cmap="viridis", figsize=figsize)
    destination = Path(out_prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(f"{out_prefix}.pdf")
    means.to_csv(f"{out_prefix}_score.csv.gz")
    return means


def score_heatmap(
    adata, marker_df, reference_key="Cluster", figsize=(9, 6), return_score=False, save_fig=False
):
    import seaborn as sns

    working = adata.copy()
    names = []
    for label in marker_df:
        name = f"{label}_Marker_Score"
        sc.tl.score_genes(working, gene_list=marker_df[label].dropna().tolist(), score_name=name)
        names.append(name)
    scores = working.obs[[reference_key, *names]].copy()
    if return_score:
        return scores
    means = scores.groupby(reference_key, observed=True)[names].mean()
    plot = sns.clustermap(
        means.T, method="complete", standard_scale=0, cmap="viridis", figsize=figsize
    )
    if save_fig:
        destination = Path(save_fig)
        destination.mkdir(parents=True, exist_ok=True)
        plot.savefig(destination / "annotation_heatmap.pdf")
    return plot
